#!/usr/bin/env python3
"""Evaluate fixed-epoch V11B checkpoints and aggregate leakage-free OOF metrics."""

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

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
    build_dataset,
    evaluate_boxes,
    global_scores_from_sums,
    make_loader,
    to_builtin,
    write_prediction_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/splits/v11_5fold_seed42.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/20260714-v11b-oof",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.fold is None and not args.aggregate:
        parser.error("provide --fold or --aggregate")
    if args.fold is not None and args.aggregate:
        parser.error("--fold and --aggregate are mutually exclusive")
    return args


def fold_paths(fold):
    run_root = BASELINE_ROOT / "outputs" / f"v11b_fold{fold}"
    return {
        "config": run_root / ".hydra/config.yaml",
        "checkpoint": run_root / "checkpoints/fixed-epoch=8.ckpt",
    }


def evaluate_fold(args):
    fold = args.fold
    paths = fold_paths(fold)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = pd.read_csv(args.manifest)
    expected = manifest[manifest["fold"] == fold].copy()
    config = OmegaConf.load(paths["config"])
    module, _ = get_pl_modules_by_cfg(config)
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu")
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    module.to(args.device).eval()

    dataset = build_dataset(config, "val", 1024)
    if set(dataset.anns) != set(expected["filename"]):
        raise RuntimeError(f"Fold {fold} config does not select its held-out filenames")
    loader = make_loader(dataset, config, args.batch_size, args.workers)
    metric = CLEvalMetric()
    global_metric = CLEvalMetric()
    predictions = OrderedDict()
    rows = []
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"V11B fold {fold} OOF"):
            images = batch["images"].to(args.device, non_blocking=True)
            prediction = module.model(images=images, return_loss=False)
            postprocess_batch = {
                "images": batch["images"],
                "inverse_matrix": batch["inverse_matrix"],
            }
            boxes_batch, _ = module.model.get_polygons_from_maps(
                postprocess_batch,
                {"prob_maps": prediction["prob_maps"].cpu()},
            )
            for index, filename in enumerate(batch["image_filename"]):
                boxes = boxes_batch[index]
                predictions[filename] = boxes
                values = evaluate_boxes(
                    boxes,
                    dataset.anns[filename],
                    metric,
                    global_metric,
                )
                row = {
                    "filename": filename,
                    "fold": fold,
                    "regions": len(boxes),
                }
                row.update(values)
                rows.append(row)

    frame = pd.DataFrame(rows).sort_values("filename")
    fold_dir = args.output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(fold_dir / "per_image.csv", index=False)
    write_prediction_json(fold_dir / "predictions.json", predictions)
    global_values = global_metric.compute()
    result = {
        "fold": fold,
        "images": len(frame),
        "regions_total": int(frame["regions"].sum()),
        "regions_mean": float(frame["regions"].mean()),
        "macro_hmean": float(frame["hmean"].mean()),
        "macro_precision": float(frame["precision"].mean()),
        "macro_recall": float(frame["recall"].mean()),
        "global_hmean": float(global_values["det_h"].cpu()),
        "global_precision": float(global_values["det_p"].cpu()),
        "global_recall": float(global_values["det_r"].cpu()),
        "runtime_seconds": time.monotonic() - started,
        "peak_gpu_memory_gb": (
            torch.cuda.max_memory_allocated() / (1024**3)
            if args.device.startswith("cuda")
            else 0.0
        ),
        "checkpoint": str(paths["checkpoint"].resolve()),
    }
    with (fold_dir / "metrics.json").open("w") as handle:
        json.dump(to_builtin(result), handle, indent=2)
        handle.write("\n")
    print(json.dumps(to_builtin(result), indent=2))


def scores_from_frame(frame):
    state = {
        name: np.asarray([frame[name].sum()], dtype=np.float64)
        for name in SCORE_STATE_NAMES
    }
    global_values = global_scores_from_sums(state)
    return {
        "images": int(len(frame)),
        "macro_hmean": float(frame["hmean"].mean()),
        "macro_precision": float(frame["precision"].mean()),
        "macro_recall": float(frame["recall"].mean()),
        "global_hmean": float(global_values["hmean"][0]),
        "global_precision": float(global_values["precision"][0]),
        "global_recall": float(global_values["recall"][0]),
        "regions_total": int(frame["regions"].sum()),
        "regions_mean": float(frame["regions"].mean()),
    }


def aggregate(args):
    manifest = pd.read_csv(args.manifest)
    manifest_by_filename = manifest.set_index("filename")
    frames = []
    fold_metrics = []
    prediction_filenames = set()
    out_of_bounds_filenames = set()
    prediction_sanity = {
        "images": 0,
        "polygons": 0,
        "min_polygon_points": None,
        "polygons_with_fewer_than_4_points": 0,
        "out_of_bounds_polygons": 0,
        "out_of_bounds_images": 0,
        "images_over_500_polygons": 0,
        "max_polygons_per_image": 0,
    }
    for fold in range(5):
        fold_dir = args.output_dir / f"fold_{fold}"
        frame = pd.read_csv(fold_dir / "per_image.csv")
        if set(frame["fold"]) != {fold}:
            raise RuntimeError(f"Fold ID mismatch in {fold_dir / 'per_image.csv'}")
        with (fold_dir / "predictions.json").open() as handle:
            predictions = json.load(handle)["images"]
        if set(predictions) != set(frame["filename"]):
            raise RuntimeError(f"Prediction JSON mismatch in fold {fold}")
        if prediction_filenames.intersection(predictions):
            raise RuntimeError(f"Duplicate prediction filename in fold {fold}")
        prediction_filenames.update(predictions)
        for filename, payload in predictions.items():
            words = payload["words"]
            polygon_count = len(words)
            prediction_sanity["images"] += 1
            prediction_sanity["polygons"] += polygon_count
            prediction_sanity["images_over_500_polygons"] += polygon_count > 500
            prediction_sanity["max_polygons_per_image"] = max(
                prediction_sanity["max_polygons_per_image"], polygon_count
            )
            width = int(manifest_by_filename.loc[filename, "width"])
            height = int(manifest_by_filename.loc[filename, "height"])
            for item in words.values():
                points = item["points"]
                point_count = len(points)
                current_min = prediction_sanity["min_polygon_points"]
                prediction_sanity["min_polygon_points"] = (
                    point_count if current_min is None else min(current_min, point_count)
                )
                prediction_sanity["polygons_with_fewer_than_4_points"] += point_count < 4
                is_out_of_bounds = any(
                    x < 0 or y < 0 or x >= width or y >= height for x, y in points
                )
                prediction_sanity["out_of_bounds_polygons"] += is_out_of_bounds
                if is_out_of_bounds:
                    out_of_bounds_filenames.add(filename)
        frames.append(frame)
        fold_metrics.append({"fold": fold, **scores_from_frame(frame)})
    oof = pd.concat(frames, ignore_index=True)
    if len(oof) != 3676 or not oof["filename"].is_unique:
        raise RuntimeError("OOF rows must contain 3,676 unique filenames")
    expected_fold = manifest.set_index("filename")["fold"]
    actual_fold = oof.set_index("filename")["fold"]
    if set(actual_fold.index) != set(expected_fold.index):
        raise RuntimeError("OOF filenames do not match the V11 manifest")
    if not actual_fold.sort_index().equals(expected_fold.sort_index()):
        raise RuntimeError("At least one OOF prediction came from the wrong fold")
    if prediction_filenames != set(manifest["filename"]):
        raise RuntimeError("Prediction JSON files do not cover the full manifest")

    merge_columns = [
        "filename",
        "original_split",
        "stratum",
        "gt_region_bin",
        "small_text_bin",
        "small_text_lt8_ratio",
        "small_text_lt12_ratio",
        "short_side_1024_median",
        "aspect_long_short",
        "brightness",
        "contrast",
        "blur_laplacian",
    ]
    oof = oof.merge(manifest[merge_columns], on="filename", how="left", validate="one_to_one")
    oof = oof.sort_values("filename").reset_index(drop=True)
    oof.to_csv(args.output_dir / "oof_per_image.csv", index=False)

    fold_frame = pd.DataFrame(fold_metrics).sort_values("fold")
    fold_frame.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    overall = scores_from_frame(oof)
    if prediction_sanity["polygons"] != overall["regions_total"]:
        raise RuntimeError("Prediction JSON polygon count does not match per-image output")
    prediction_sanity["out_of_bounds_images"] = len(out_of_bounds_filenames)
    stability = {}
    for column in (
        "macro_hmean",
        "macro_precision",
        "macro_recall",
        "global_hmean",
        "global_precision",
        "global_recall",
    ):
        stability[column] = {
            "fold_mean": float(fold_frame[column].mean()),
            "fold_std": float(fold_frame[column].std(ddof=1)),
            "fold_min": float(fold_frame[column].min()),
            "fold_max": float(fold_frame[column].max()),
        }

    strata_rows = []

    def append_stratum(name, group):
        strata_rows.append({"stratum": name, **scores_from_frame(group)})

    append_stratum("all", oof)
    for column in ("original_split", "gt_region_bin", "small_text_bin"):
        for value, group in oof.groupby(column, sort=True):
            append_stratum(f"{column}={value}", group)
    for column, direction in (
        ("small_text_lt12_ratio", "high"),
        ("short_side_1024_median", "low"),
        ("contrast", "low"),
        ("blur_laplacian", "low"),
    ):
        threshold = oof[column].quantile(0.75 if direction == "high" else 0.25)
        mask = oof[column] >= threshold if direction == "high" else oof[column] <= threshold
        append_stratum(f"{column}_{direction}_quartile", oof[mask])
    pd.DataFrame(strata_rows).to_csv(args.output_dir / "strata_metrics.csv", index=False)

    gates = {
        "all_images_have_exactly_one_oof_prediction": len(oof) == 3676
        and oof["filename"].is_unique,
        "macro_hmean_fold_std_at_most_0_005": stability["macro_hmean"]["fold_std"]
        <= 0.005,
        "global_hmean_fold_std_at_most_0_005": stability["global_hmean"]["fold_std"]
        <= 0.005,
        "worst_macro_hmean_at_least_0_955": stability["macro_hmean"]["fold_min"]
        >= 0.955,
        "worst_global_hmean_at_least_0_955": stability["global_hmean"]["fold_min"]
        >= 0.955,
    }
    result = {
        "primary_checkpoint_policy": "fixed epoch 8 for every fold",
        "overall_oof": overall,
        "prediction_sanity": prediction_sanity,
        "fold_stability": stability,
        "gates": gates,
        "passed": all(gates.values()),
    }
    with (args.output_dir / "oof_metrics.json").open("w") as handle:
        json.dump(to_builtin(result), handle, indent=2)
        handle.write("\n")
    print(json.dumps(to_builtin(result), indent=2))


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate(args)
    else:
        evaluate_fold(args)


if __name__ == "__main__":
    main()
