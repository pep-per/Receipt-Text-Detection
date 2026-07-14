#!/usr/bin/env python3
"""Evaluate V2B and V5 probability-map model ensembles."""

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
    STATE_NAMES,
    bootstrap_paired,
    build_dataset,
    evaluate_boxes,
    make_loader,
    summarize,
    to_builtin,
    write_prediction_json,
)


MODEL_SPECS = {
    "v2b": {
        "config": BASELINE_ROOT
        / "outputs/v2b_resolution1024_best_eval/.hydra/config.yaml",
        "checkpoint": BASELINE_ROOT
        / "outputs/v2b_resolution1024/checkpoints/epoch=8-step=1845.ckpt",
    },
    "v5": {
        "config": BASELINE_ROOT
        / "outputs/v5_resnet34_1024_epoch7_eval/.hydra/config.yaml",
        "checkpoint": BASELINE_ROOT
        / "outputs/v5_resnet34_1024/checkpoints/epoch=7-step=1640.ckpt",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--v2b-weight", type=float, choices=(0.5, 0.75), default=0.5)
    parser.add_argument(
        "--dataset-root", type=Path, default=PROJECT_ROOT / "data/datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/20260714-v9-model-ensemble",
    )
    parser.add_argument(
        "--disagreement-csv",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/20260714-d0-data-audit/v2b_v5_val_disagreement.csv",
    )
    parser.add_argument("--predictions-json", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="disabled"
    )
    parser.add_argument("--run-name", default="v9_v2b_v5_equal_eval")
    args = parser.parse_args()
    if args.split == "test" and args.predictions_json is None:
        parser.error("--predictions-json is required for test inference")
    return args


def load_module(spec, dataset_root, device):
    config = OmegaConf.load(spec["config"])
    config.dataset_base_path = f"{dataset_root.resolve()}/"
    module, _ = get_pl_modules_by_cfg(config)
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu")
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    module.to(device)
    module.eval()
    return module, config


def summarize_disagreement_strata(frame, disagreement_path):
    disagreement = pd.read_csv(disagreement_path)
    merged = frame.merge(disagreement, on="filename", how="left")
    rows = []

    def append_stratum(name, mask):
        group = merged[mask]
        rows.append(
            {
                "stratum": name,
                "images": len(group),
                "control_hmean": group["control_hmean"].mean(),
                "ensemble_hmean": group["ensemble_hmean"].mean(),
                "delta_hmean": group["delta_hmean"].mean(),
                "delta_precision": group["delta_precision"].mean(),
                "delta_recall": group["delta_recall"].mean(),
                "improved_images": int((group["delta_hmean"] > 1e-12).sum()),
                "worse_images": int((group["delta_hmean"] < -1e-12).sum()),
            }
        )

    append_stratum("all", np.ones(len(merged), dtype=bool))
    append_stratum(
        "control_h_bottom_quartile",
        merged["control_hmean"] <= merged["control_hmean"].quantile(0.25),
    )
    append_stratum("v5_won_d0", merged["hmean_delta_v5_v2b"] > 1e-12)
    append_stratum("v2b_won_d0", merged["hmean_delta_v5_v2b"] < -1e-12)
    append_stratum(
        "high_disagreement_quartile",
        merged["prediction_jaccard_iou50"]
        <= merged["prediction_jaccard_iou50"].quantile(0.25),
    )
    return pd.DataFrame(rows)


def init_wandb(args):
    if args.wandb_mode == "disabled":
        return None
    import wandb

    return wandb.init(
        project="receipt-text-detection",
        name=args.run_name,
        job_type="evaluation" if args.split == "val" else "prediction",
        mode=args.wandb_mode,
        dir=str(BASELINE_ROOT),
        config={
            "experiment": "V9 Existing-model Probability-map Ensemble",
            "split": args.split,
            "v2b_weight": args.v2b_weight,
            "v5_weight": 1.0 - args.v2b_weight,
            "fusion": "same-scale probability-map weighted average",
            "v2b_config": str(MODEL_SPECS["v2b"]["config"]),
            "v2b_checkpoint": str(MODEL_SPECS["v2b"]["checkpoint"]),
            "v5_config": str(MODEL_SPECS["v5"]["config"]),
            "v5_checkpoint": str(MODEL_SPECS["v5"]["checkpoint"]),
            "batch_size": args.batch_size,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
    )


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weight_tag = f"w{round(args.v2b_weight * 100):03d}"

    v2b, v2b_config = load_module(
        MODEL_SPECS["v2b"], args.dataset_root, args.device
    )
    v5, v5_config = load_module(MODEL_SPECS["v5"], args.dataset_root, args.device)
    if (
        float(v2b_config.models.head.postprocess.thresh)
        != float(v5_config.models.head.postprocess.thresh)
        or float(v2b_config.models.head.postprocess.box_thresh)
        != float(v5_config.models.head.postprocess.box_thresh)
    ):
        raise ValueError("V2B and V5 post-processing thresholds differ")

    dataset = build_dataset(v2b_config, args.split, 1024)
    loader = make_loader(dataset, v2b_config, args.batch_size, args.workers)
    wandb_run = init_wandb(args)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    start_time = time.monotonic()
    rows = []
    predictions = OrderedDict()
    control_global = CLEvalMetric()
    ensemble_global = CLEvalMetric()
    control_metric = CLEvalMetric()
    ensemble_metric = CLEvalMetric()

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"V9 {args.split} {weight_tag}"):
            images = batch["images"].to(args.device, non_blocking=True)
            v2b_maps = v2b.model(images=images, return_loss=False)["prob_maps"]
            v5_maps = v5.model(images=images, return_loss=False)["prob_maps"]
            if v2b_maps.shape != v5_maps.shape:
                raise RuntimeError("V2B and V5 probability-map shapes differ")
            ensemble_maps = (
                args.v2b_weight * v2b_maps + (1.0 - args.v2b_weight) * v5_maps
            )
            postprocess_batch = {
                "images": batch["images"],
                "inverse_matrix": batch["inverse_matrix"],
            }
            control_boxes, _ = v2b.model.get_polygons_from_maps(
                postprocess_batch, {"prob_maps": v2b_maps.cpu()}
            )
            ensemble_boxes, _ = v2b.model.get_polygons_from_maps(
                postprocess_batch, {"prob_maps": ensemble_maps.cpu()}
            )

            for index, filename in enumerate(batch["image_filename"]):
                predictions[filename] = ensemble_boxes[index]
                row = {
                    "filename": filename,
                    "control_regions": len(control_boxes[index]),
                    "ensemble_regions": len(ensemble_boxes[index]),
                }
                if args.split == "val":
                    gt_words = dataset.anns[filename]
                    for prefix, boxes, metric, global_metric in (
                        ("control", control_boxes[index], control_metric, control_global),
                        (
                            "ensemble",
                            ensemble_boxes[index],
                            ensemble_metric,
                            ensemble_global,
                        ),
                    ):
                        result = evaluate_boxes(boxes, gt_words, metric, global_metric)
                        for key, value in result.items():
                            column = (
                                f"{prefix}_state_{key}"
                                if key in STATE_NAMES
                                else f"{prefix}_{key}"
                            )
                            row[column] = value
                    for metric_name in ("hmean", "precision", "recall"):
                        row[f"delta_{metric_name}"] = (
                            row[f"ensemble_{metric_name}"]
                            - row[f"control_{metric_name}"]
                        )
                rows.append(row)

    runtime_seconds = time.monotonic() - start_time
    peak_gpu_gb = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if args.device.startswith("cuda")
        else 0.0
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / f"per_image_{args.split}_{weight_tag}.csv", index=False)
    run_config = {
        "split": args.split,
        "v2b_weight": args.v2b_weight,
        "v5_weight": 1.0 - args.v2b_weight,
        "resolution": 1024,
        "fusion": "same-scale probability-map weighted average",
        "v2b_config": str(MODEL_SPECS["v2b"]["config"].resolve()),
        "v2b_checkpoint": str(MODEL_SPECS["v2b"]["checkpoint"].resolve()),
        "v5_config": str(MODEL_SPECS["v5"]["config"].resolve()),
        "v5_checkpoint": str(MODEL_SPECS["v5"]["checkpoint"].resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": args.device,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "segmentation_thresh": float(v2b_config.models.head.postprocess.thresh),
        "box_thresh": float(v2b_config.models.head.postprocess.box_thresh),
        "runtime_seconds": runtime_seconds,
        "peak_gpu_memory_gb": peak_gpu_gb,
    }
    with (args.output_dir / f"run_config_{args.split}_{weight_tag}.json").open("w") as file:
        json.dump(to_builtin(run_config), file, indent=2)

    if args.split == "val":
        control_summary = summarize(frame, "control", control_global)
        ensemble_summary = summarize(frame, "ensemble", ensemble_global)
        bootstrap = bootstrap_paired(
            frame,
            args.bootstrap_samples,
            args.seed,
            candidate_prefix="ensemble",
        )
        bootstrap.to_csv(
            args.output_dir / f"bootstrap_paired_{weight_tag}.csv", index=False
        )
        strata = summarize_disagreement_strata(frame, args.disagreement_csv)
        strata.to_csv(args.output_dir / f"strata_metrics_{weight_tag}.csv", index=False)
        metrics = {
            "control": control_summary,
            "ensemble": ensemble_summary,
            "delta": {
                key: ensemble_summary[key] - control_summary[key]
                for key in (
                    "macro_hmean",
                    "macro_precision",
                    "macro_recall",
                    "global_hmean",
                    "global_precision",
                    "global_recall",
                    "regions_mean",
                )
            },
            "runtime_seconds": runtime_seconds,
            "peak_gpu_memory_gb": peak_gpu_gb,
        }
        with (args.output_dir / f"metrics_{weight_tag}.json").open("w") as file:
            json.dump(to_builtin(metrics), file, indent=2)
        print(json.dumps(to_builtin(metrics), indent=2))

        if wandb_run is not None:
            import wandb

            log_values = {
                f"control/{key}": value for key, value in control_summary.items()
            }
            log_values.update(
                {f"ensemble/{key}": value for key, value in ensemble_summary.items()}
            )
            log_values.update(
                {f"delta/{key}": value for key, value in metrics["delta"].items()}
            )
            log_values.update(
                {
                    "runtime_seconds": runtime_seconds,
                    "peak_gpu_memory_gb": peak_gpu_gb,
                    "paired_results": wandb.Table(
                        dataframe=frame[
                            [
                                "filename",
                                "control_hmean",
                                "ensemble_hmean",
                                "delta_hmean",
                                "control_precision",
                                "ensemble_precision",
                                "control_recall",
                                "ensemble_recall",
                            ]
                        ]
                    ),
                }
            )
            for _, item in bootstrap.iterrows():
                name = item["metric"]
                log_values[f"bootstrap/{name}_ci95_low"] = item["ci95_low"]
                log_values[f"bootstrap/{name}_ci95_high"] = item["ci95_high"]
                log_values[f"bootstrap/{name}_probability_gt_0"] = item[
                    "probability_delta_gt_0"
                ]
            wandb_run.log(log_values)
    else:
        write_prediction_json(args.predictions_json, predictions)
        test_summary = {
            "images": int(len(frame)),
            "regions_total": int(frame["ensemble_regions"].sum()),
            "regions_mean": float(frame["ensemble_regions"].mean()),
            "regions_min": int(frame["ensemble_regions"].min()),
            "regions_max": int(frame["ensemble_regions"].max()),
            "empty_images": int((frame["ensemble_regions"] == 0).sum()),
            "over_500_images": int((frame["ensemble_regions"] > 500).sum()),
            "runtime_seconds": runtime_seconds,
            "peak_gpu_memory_gb": peak_gpu_gb,
            "predictions_json": str(args.predictions_json.resolve()),
        }
        with (args.output_dir / f"test_prediction_summary_{weight_tag}.json").open(
            "w"
        ) as file:
            json.dump(to_builtin(test_summary), file, indent=2)
        print(json.dumps(to_builtin(test_summary), indent=2))
        if wandb_run is not None:
            numeric_summary = {
                f"test/{key}": value
                for key, value in test_summary.items()
                if isinstance(value, (int, float))
            }
            wandb_run.log(numeric_summary)

    if wandb_run is not None:
        print(f"W&B run: {wandb_run.url}")
        wandb_run.finish()


if __name__ == "__main__":
    main()
