#!/usr/bin/env python3
"""Compare the V10 SSL checkpoint against the locked V2B per-image control."""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = PROJECT_ROOT / "baseline_code"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(BASELINE_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402
from ocr.metrics import CLEvalMetric  # noqa: E402
from v8_scale_tta import (  # noqa: E402
    SCORE_STATE_NAMES,
    STATE_NAMES,
    bootstrap_paired,
    build_dataset,
    evaluate_boxes,
    global_scores_from_sums,
    make_loader,
    summarize,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=BASELINE_ROOT
        / "outputs/v10_ssl_moco_epoch8_eval/.hydra/config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BASELINE_ROOT
        / "outputs/v10_ssl_moco_finetune/checkpoints/epoch=8-step=1845.ckpt",
    )
    parser.add_argument(
        "--control-per-image",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/20260714-v8-scale-tta/per_image_val.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/20260714-v10-domain-ssl",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=PROJECT_ROOT / "data/datasets"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_boxes(boxes):
    invalid_point_count = 0
    non_finite = 0
    zero_area = 0
    for polygon in boxes:
        points = np.asarray(polygon, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 4 or points.shape[1] != 2:
            invalid_point_count += 1
            continue
        if not np.isfinite(points).all():
            non_finite += 1
            continue
        if abs(cv2.contourArea(points.astype(np.float32))) <= 0:
            zero_area += 1
    return invalid_point_count, non_finite, zero_area


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = OmegaConf.load(args.config)
    config.dataset_base_path = f"{args.dataset_root.resolve()}/"
    module, _ = get_pl_modules_by_cfg(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    module.to(args.device).eval()

    dataset = build_dataset(config, "val", 1024)
    loader = make_loader(dataset, config, args.batch_size, args.workers)
    metric = CLEvalMetric()
    global_metric = CLEvalMetric()
    rows = []
    invalid_totals = {
        "invalid_point_count": 0,
        "non_finite_coordinate": 0,
        "zero_area": 0,
        "images_over_500_regions": 0,
        "empty_images": 0,
    }
    start = time.monotonic()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="V10 independent per-image evaluation"):
            images = batch["images"].to(args.device, non_blocking=True)
            prediction = module.model(images=images, return_loss=False)
            postprocess_batch = {
                "images": batch["images"],
                "inverse_matrix": batch["inverse_matrix"],
            }
            boxes_batch, _ = module.model.get_polygons_from_maps(
                postprocess_batch, {"prob_maps": prediction["prob_maps"].cpu()}
            )
            for index, filename in enumerate(batch["image_filename"]):
                boxes = boxes_batch[index]
                result = evaluate_boxes(
                    boxes, dataset.anns[filename], metric, global_metric
                )
                row = {"filename": filename, "ssl_regions": len(boxes)}
                for key, value in result.items():
                    column = (
                        f"ssl_state_{key}" if key in STATE_NAMES else f"ssl_{key}"
                    )
                    row[column] = value
                rows.append(row)
                point_count, non_finite, zero_area = validate_boxes(boxes)
                invalid_totals["invalid_point_count"] += point_count
                invalid_totals["non_finite_coordinate"] += non_finite
                invalid_totals["zero_area"] += zero_area
                invalid_totals["images_over_500_regions"] += int(len(boxes) > 500)
                invalid_totals["empty_images"] += int(len(boxes) == 0)

    ssl_frame = pd.DataFrame(rows)
    control = pd.read_csv(args.control_per_image)
    control_columns = [
        column
        for column in control.columns
        if column == "filename" or column.startswith("control_")
    ]
    frame = control[control_columns].merge(ssl_frame, on="filename", how="outer")
    if len(frame) != len(dataset) or frame.isna().any().any():
        raise RuntimeError("Control and SSL per-image rows do not align exactly")
    for metric_name in ("hmean", "precision", "recall"):
        frame[f"delta_{metric_name}"] = (
            frame[f"ssl_{metric_name}"] - frame[f"control_{metric_name}"]
        )
    frame.to_csv(args.output_dir / "per_image_val.csv", index=False)

    control_states = {
        name: np.asarray([frame[f"control_state_{name}"].sum()])
        for name in SCORE_STATE_NAMES
    }
    control_global = global_scores_from_sums(control_states)
    control_summary = {
        "images": int(len(frame)),
        "regions_total": int(frame["control_regions"].sum()),
        "regions_mean": float(frame["control_regions"].mean()),
        "regions_min": int(frame["control_regions"].min()),
        "regions_max": int(frame["control_regions"].max()),
        "macro_hmean": float(frame["control_hmean"].mean()),
        "macro_precision": float(frame["control_precision"].mean()),
        "macro_recall": float(frame["control_recall"].mean()),
        "global_hmean": float(control_global["hmean"][0]),
        "global_precision": float(control_global["precision"][0]),
        "global_recall": float(control_global["recall"][0]),
    }
    ssl_summary = summarize(frame, "ssl", global_metric)
    delta = {
        key: ssl_summary[key] - control_summary[key]
        for key in (
            "macro_hmean",
            "macro_precision",
            "macro_recall",
            "global_hmean",
            "global_precision",
            "global_recall",
            "regions_mean",
        )
    }
    bootstrap = bootstrap_paired(
        frame,
        args.bootstrap_samples,
        args.seed,
        candidate_prefix="ssl",
    )
    bootstrap.to_csv(args.output_dir / "bootstrap_paired.csv", index=False)
    result = {
        "control": control_summary,
        "ssl": ssl_summary,
        "delta": delta,
        "prediction_validation": invalid_totals,
        "runtime_seconds": time.monotonic() - start,
        "checkpoint": str(args.checkpoint.resolve()),
    }
    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
