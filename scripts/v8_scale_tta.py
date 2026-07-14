#!/usr/bin/env python3
"""Evaluate probability-map scale TTA for the V2B DBNet checkpoint."""

import argparse
import copy
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = PROJECT_ROOT / "baseline_code"
sys.path.insert(0, str(BASELINE_ROOT))

from ocr.datasets import DBCollateFN, OCRDataset  # noqa: E402
from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402
from ocr.metrics import CLEvalMetric  # noqa: E402


DEFAULT_CONFIG = (
    BASELINE_ROOT / "outputs/v2b_resolution1024_best_eval/.hydra/config.yaml"
)
DEFAULT_CHECKPOINT = (
    BASELINE_ROOT
    / "outputs/v2b_resolution1024/checkpoints/epoch=8-step=1845.ckpt"
)
STATE_NAMES = CLEvalMetric.ACCUMULATED_STATE_NAMES
SCORE_STATE_NAMES = (
    "det_num_char_gt",
    "det_num_char_det",
    "det_gran_score_recall",
    "det_num_char_tp_recall",
    "det_gran_score_precision",
    "det_num_char_tp_precision",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--scales", type=int, nargs="+", default=(1024, 1152))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--dataset-root", type=Path, default=PROJECT_ROOT / "data/datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/20260714-v8-scale-tta",
    )
    parser.add_argument(
        "--quality-metrics",
        type=Path,
        default=PROJECT_ROOT / "experiments/20260714-d0-data-audit/image_metrics.csv",
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
    parser.add_argument("--run-name", default="v8_scale_tta_1024_1152_eval")
    args = parser.parse_args()
    if len(args.scales) < 2:
        parser.error("--scales requires a control scale and at least one TTA scale")
    if len(set(args.scales)) != len(args.scales):
        parser.error("--scales must be unique")
    if args.split == "test" and args.predictions_json is None:
        parser.error("--predictions-json is required for test inference")
    return args


def set_transform_scale(transform_config, scale):
    found_resize = False
    found_pad = False
    for transform in transform_config.transforms:
        target = str(transform.get("_target_", ""))
        if target.endswith("LongestMaxSize"):
            transform.max_size = scale
            found_resize = True
        elif target.endswith("PadIfNeeded"):
            transform.min_width = scale
            transform.min_height = scale
            found_pad = True
    if not found_resize or not found_pad:
        raise ValueError("Expected LongestMaxSize and PadIfNeeded in inference transform")


def build_dataset(config, split, scale):
    scaled_config = copy.deepcopy(config)
    transform_name = "test_transform" if split == "val" else "predict_transform"
    dataset_name = "test_dataset" if split == "val" else "predict_dataset"
    set_transform_scale(scaled_config.transforms[transform_name], scale)
    return instantiate(scaled_config.datasets[dataset_name])


def make_loader(dataset, config, batch_size, workers):
    collate_fn = instantiate(config.collate_fn)
    if not isinstance(collate_fn, DBCollateFN):
        raise TypeError("V8 expects DBCollateFN")
    collate_fn.inference_mode = True
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def move_images_to_device(batch, device):
    return batch["images"].to(device, non_blocking=True)


def transform_points(points, matrix):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, matrix.astype(np.float32)).reshape(-1, 2)


def valid_content_mask(map_shape, map_to_original, original_size):
    width, height = original_size
    original_corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    original_to_map = np.linalg.inv(np.asarray(map_to_original, dtype=np.float64))
    map_corners = transform_points(original_corners, original_to_map)
    map_corners[:, 0] = np.clip(map_corners[:, 0], 0, map_shape[1] - 1)
    map_corners[:, 1] = np.clip(map_corners[:, 1], 0, map_shape[0] - 1)
    mask = np.zeros(map_shape, dtype=np.float32)
    cv2.fillConvexPoly(mask, np.round(map_corners).astype(np.int32), 1.0)
    return mask


def align_probability_map(
    probability_map,
    source_to_original,
    base_to_original,
    base_shape,
    original_size,
):
    source_mask = valid_content_mask(
        probability_map.shape, source_to_original, original_size
    )
    source_to_base = np.linalg.inv(np.asarray(base_to_original)) @ np.asarray(
        source_to_original
    )
    output_size = (base_shape[1], base_shape[0])
    aligned_map = cv2.warpPerspective(
        probability_map,
        source_to_base,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    aligned_mask = cv2.warpPerspective(
        source_mask,
        source_to_base,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return aligned_map, aligned_mask


def fuse_probability_maps(maps, matrices, original_sizes):
    base_maps = maps[0]
    fused_maps = []
    for image_index in range(base_maps.shape[0]):
        base_shape = base_maps[image_index].shape
        numerator = np.zeros(base_shape, dtype=np.float32)
        denominator = np.zeros(base_shape, dtype=np.float32)
        for scale_index, scale_maps in enumerate(maps):
            aligned_map, aligned_mask = align_probability_map(
                scale_maps[image_index],
                matrices[scale_index][image_index],
                matrices[0][image_index],
                base_shape,
                original_sizes[image_index],
            )
            numerator += aligned_map * aligned_mask
            denominator += aligned_mask
        fused = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
        fused_maps.append(fused)
    return np.stack(fused_maps)[:, None, :, :]


def get_original_size(dataset, filename, cache):
    if filename in cache:
        return cache[filename]
    with Image.open(dataset.image_path / filename) as image:
        exif = image.getexif()
        if exif and 274 in exif:
            image = OCRDataset.rotate_image(image, exif[274])
        cache[filename] = image.size
    return cache[filename]


def flatten_polygons(boxes):
    return [[point for coordinate in polygon for point in coordinate] for polygon in boxes]


def evaluate_boxes(boxes, gt_words, metric, global_metric):
    detections = flatten_polygons(boxes)
    ground_truth = [item.squeeze().reshape(-1) for item in gt_words]
    metric(detections, ground_truth)
    values = metric.compute()
    state = {name: float(getattr(metric, name).cpu()) for name in STATE_NAMES}
    global_metric.accumulate_metric(metric)
    result = {
        "hmean": float(values["det_h"].cpu()),
        "precision": float(values["det_p"].cpu()),
        "recall": float(values["det_r"].cpu()),
        "gt_regions": len(ground_truth),
        **state,
    }
    metric.reset()
    return result


def summarize(frame, prefix, global_metric):
    global_values = global_metric.compute()
    return {
        "images": int(len(frame)),
        "regions_total": int(frame[f"{prefix}_regions"].sum()),
        "regions_mean": float(frame[f"{prefix}_regions"].mean()),
        "regions_min": int(frame[f"{prefix}_regions"].min()),
        "regions_max": int(frame[f"{prefix}_regions"].max()),
        "macro_hmean": float(frame[f"{prefix}_hmean"].mean()),
        "macro_precision": float(frame[f"{prefix}_precision"].mean()),
        "macro_recall": float(frame[f"{prefix}_recall"].mean()),
        "global_hmean": float(global_values["det_h"].cpu()),
        "global_precision": float(global_values["det_p"].cpu()),
        "global_recall": float(global_values["det_r"].cpu()),
    }


def summarize_quality_strata(frame, quality_metrics_path):
    if not quality_metrics_path.exists():
        return None
    quality = pd.read_csv(quality_metrics_path)
    quality = quality[quality["dataset"] == "val"]
    merged = frame.merge(quality, on="filename", how="left")
    strata = []

    def append_stratum(name, mask):
        group = merged[mask]
        strata.append(
            {
                "stratum": name,
                "images": len(group),
                "control_hmean": group["control_hmean"].mean(),
                "tta_hmean": group["tta_hmean"].mean(),
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
    for column, direction in (
        ("short_side_1024_median", "low"),
        ("small_text_lt8_ratio", "high"),
        ("contrast", "low"),
        ("blur_laplacian", "low"),
    ):
        quantile = merged[column].quantile(0.25 if direction == "low" else 0.75)
        mask = merged[column] <= quantile if direction == "low" else merged[column] >= quantile
        append_stratum(f"{column}_{direction}_quartile", mask)
    return pd.DataFrame(strata)


def global_scores_from_sums(state_sums):
    gt = state_sums["det_num_char_gt"]
    det = state_sums["det_num_char_det"]
    recall_numerator = np.maximum(
        0.0,
        state_sums["det_num_char_tp_recall"]
        - state_sums["det_gran_score_recall"],
    )
    precision_numerator = np.maximum(
        0.0,
        state_sums["det_num_char_tp_precision"]
        - state_sums["det_gran_score_precision"],
    )
    recall = np.divide(
        recall_numerator, gt, out=np.zeros_like(recall_numerator), where=gt > 0
    )
    precision = np.divide(
        precision_numerator,
        det,
        out=np.zeros_like(precision_numerator),
        where=det > 0,
    )
    hmean = np.divide(
        2 * recall * precision,
        recall + precision,
        out=np.zeros_like(recall),
        where=(recall + precision) > 0,
    )
    return {"hmean": hmean, "precision": precision, "recall": recall}


def bootstrap_paired(frame, samples, seed, candidate_prefix="tta"):
    rng = np.random.default_rng(seed)
    count = len(frame)
    distributions = {
        f"{aggregation}_{metric}": []
        for aggregation in ("macro", "global")
        for metric in ("hmean", "precision", "recall")
    }
    chunk_size = min(500, samples)
    for start in range(0, samples, chunk_size):
        chunk = min(chunk_size, samples - start)
        indices = rng.integers(0, count, size=(chunk, count))
        for metric in ("hmean", "precision", "recall"):
            difference = (
                frame[f"{candidate_prefix}_{metric}"].to_numpy()
                - frame[f"control_{metric}"].to_numpy()
            )
            distributions[f"macro_{metric}"].append(difference[indices].mean(axis=1))

        method_scores = {}
        for method in ("control", candidate_prefix):
            state_sums = {
                name: frame[f"{method}_state_{name}"].to_numpy()[indices].sum(axis=1)
                for name in SCORE_STATE_NAMES
            }
            method_scores[method] = global_scores_from_sums(state_sums)
        for metric in ("hmean", "precision", "recall"):
            distributions[f"global_{metric}"].append(
                method_scores[candidate_prefix][metric]
                - method_scores["control"][metric]
            )

    rows = []
    for name, chunks in distributions.items():
        values = np.concatenate(chunks)
        rows.append(
            {
                "metric": name,
                "delta_mean": float(values.mean()),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "probability_delta_gt_0": float((values > 0).mean()),
                "bootstrap_samples": samples,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def write_prediction_json(path, predictions):
    path.parent.mkdir(parents=True, exist_ok=True)
    result = OrderedDict(images=OrderedDict())
    for filename, boxes in predictions.items():
        words = OrderedDict()
        for index, box in enumerate(boxes, start=1):
            words[f"{index:04}"] = OrderedDict(points=box)
        result["images"][filename] = OrderedDict(words=words)
    with path.open("w") as file:
        json.dump(result, file, separators=(",", ":"))


def to_builtin(value):
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


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
            "experiment": "V8 Scale TTA Screen",
            "split": args.split,
            "scales": list(args.scales),
            "fusion": "valid-region probability-map equal average",
            "config_path": str(args.config),
            "checkpoint_path": str(args.checkpoint),
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

    config = OmegaConf.load(args.config)
    config.dataset_base_path = f"{args.dataset_root.resolve()}/"
    module, _ = get_pl_modules_by_cfg(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    module.to(args.device)
    module.eval()

    datasets = [build_dataset(config, args.split, scale) for scale in args.scales]
    loaders = [
        make_loader(dataset, config, args.batch_size, args.workers)
        for dataset in datasets
    ]
    expected_filenames = list(datasets[0].anns)
    for dataset in datasets[1:]:
        if list(dataset.anns) != expected_filenames:
            raise RuntimeError("Scale datasets have different image order")

    wandb_run = init_wandb(args)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start_time = time.monotonic()
    size_cache = {}
    rows = []
    tta_predictions = OrderedDict()
    control_global = CLEvalMetric()
    tta_global = CLEvalMetric()
    control_metric = CLEvalMetric()
    tta_metric = CLEvalMetric()

    iterator = zip(*loaders)
    total_batches = len(loaders[0])
    with torch.inference_mode():
        for batches in tqdm(iterator, total=total_batches, desc=f"V8 {args.split}"):
            filenames = batches[0]["image_filename"]
            for batch in batches[1:]:
                if batch["image_filename"] != filenames:
                    raise RuntimeError("Scale dataloaders are out of order")

            probability_maps = []
            inverse_matrices = []
            for batch in batches:
                images = move_images_to_device(batch, args.device)
                prediction = module.model(images=images, return_loss=False)
                probability_maps.append(
                    prediction["prob_maps"].detach().cpu().numpy()[:, 0].astype(np.float32)
                )
                inverse_matrices.append(batch["inverse_matrix"])

            original_sizes = [
                get_original_size(datasets[0], filename, size_cache)
                for filename in filenames
            ]
            fused_maps = fuse_probability_maps(
                probability_maps, inverse_matrices, original_sizes
            )
            postprocess_batch = {
                "images": batches[0]["images"],
                "inverse_matrix": batches[0]["inverse_matrix"],
            }
            control_boxes, _ = module.model.get_polygons_from_maps(
                postprocess_batch,
                {"prob_maps": torch.from_numpy(probability_maps[0][:, None])},
            )
            tta_boxes, _ = module.model.get_polygons_from_maps(
                postprocess_batch, {"prob_maps": torch.from_numpy(fused_maps)}
            )

            for index, filename in enumerate(filenames):
                tta_predictions[filename] = tta_boxes[index]
                row = {
                    "filename": filename,
                    "control_regions": len(control_boxes[index]),
                    "tta_regions": len(tta_boxes[index]),
                }
                if args.split == "val":
                    gt_words = datasets[0].anns[filename]
                    for prefix, boxes, metric, global_metric in (
                        ("control", control_boxes[index], control_metric, control_global),
                        ("tta", tta_boxes[index], tta_metric, tta_global),
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
                            row[f"tta_{metric_name}"] - row[f"control_{metric_name}"]
                        )
                rows.append(row)

    runtime_seconds = time.monotonic() - start_time
    peak_gpu_gb = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if args.device.startswith("cuda")
        else 0.0
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / f"per_image_{args.split}.csv", index=False)

    run_config = {
        "split": args.split,
        "scales": list(args.scales),
        "fusion": "valid-region probability-map equal average",
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": args.device,
        "bootstrap_samples": args.bootstrap_samples,
        "quality_metrics": str(args.quality_metrics.resolve()),
        "seed": args.seed,
        "segmentation_thresh": float(config.models.head.postprocess.thresh),
        "box_thresh": float(config.models.head.postprocess.box_thresh),
        "runtime_seconds": runtime_seconds,
        "peak_gpu_memory_gb": peak_gpu_gb,
    }
    with (args.output_dir / f"run_config_{args.split}.json").open("w") as file:
        json.dump(to_builtin(run_config), file, indent=2)

    if args.split == "val":
        control_summary = summarize(frame, "control", control_global)
        tta_summary = summarize(frame, "tta", tta_global)
        bootstrap = bootstrap_paired(frame, args.bootstrap_samples, args.seed)
        bootstrap.to_csv(args.output_dir / "bootstrap_paired.csv", index=False)
        strata = summarize_quality_strata(frame, args.quality_metrics)
        if strata is not None:
            strata.to_csv(args.output_dir / "strata_metrics.csv", index=False)
        metrics = {
            "control": control_summary,
            "tta": tta_summary,
            "delta": {
                key: tta_summary[key] - control_summary[key]
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
        with (args.output_dir / "metrics.json").open("w") as file:
            json.dump(to_builtin(metrics), file, indent=2)
        print(json.dumps(to_builtin(metrics), indent=2))

        if wandb_run is not None:
            import wandb

            log_values = {
                f"control/{key}": value for key, value in control_summary.items()
            }
            log_values.update({f"tta/{key}": value for key, value in tta_summary.items()})
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
                                "tta_hmean",
                                "delta_hmean",
                                "control_precision",
                                "tta_precision",
                                "control_recall",
                                "tta_recall",
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
        write_prediction_json(args.predictions_json, tta_predictions)
        test_summary = {
            "images": int(len(frame)),
            "regions_total": int(frame["tta_regions"].sum()),
            "regions_mean": float(frame["tta_regions"].mean()),
            "regions_min": int(frame["tta_regions"].min()),
            "regions_max": int(frame["tta_regions"].max()),
            "empty_images": int((frame["tta_regions"] == 0).sum()),
            "over_500_images": int((frame["tta_regions"] > 500).sum()),
            "runtime_seconds": runtime_seconds,
            "peak_gpu_memory_gb": peak_gpu_gb,
            "predictions_json": str(args.predictions_json.resolve()),
        }
        with (args.output_dir / "test_prediction_summary.json").open("w") as file:
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
