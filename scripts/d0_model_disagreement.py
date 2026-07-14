#!/usr/bin/env python3
"""Compare V2B and V5 detections on validation and test for D0."""

import argparse
import gc
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = PROJECT_ROOT / "baseline_code"
sys.path.insert(0, str(BASELINE_ROOT))

from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402
from ocr.metrics import CLEvalMetric  # noqa: E402


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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/20260714-d0-data-audit"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def move_batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def polygon_area(polygon):
    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim != 2 or len(points) < 3:
        return 0.0
    return abs(float(cv2.contourArea(points)))


def prediction_statistics(filename, boxes, scores, image_shape=None):
    scores = np.asarray(scores, dtype=np.float32)
    areas = np.asarray([polygon_area(box) for box in boxes], dtype=np.float64)
    row = {
        "filename": filename,
        "pred_regions": len(boxes),
        "score_mean": float(scores.mean()) if len(scores) else np.nan,
        "score_median": float(np.median(scores)) if len(scores) else np.nan,
        "score_min": float(scores.min()) if len(scores) else np.nan,
        "score_q10": float(np.quantile(scores, 0.1)) if len(scores) else np.nan,
        "area_median": float(np.median(areas)) if len(areas) else np.nan,
        "area_q10": float(np.quantile(areas, 0.1)) if len(areas) else np.nan,
    }
    if image_shape is not None:
        height, width = image_shape
        image_area = max(height * width, 1)
        row["area_ratio_median"] = (
            float(np.median(areas) / image_area) if len(areas) else np.nan
        )
    return row


def evaluate_cleval(boxes, gt_words, metric, global_metric):
    detections = [
        [point for coordinate in polygon for point in coordinate] for polygon in boxes
    ]
    ground_truth = [item.squeeze().reshape(-1) for item in gt_words]
    metric(detections, ground_truth)
    values = metric.compute()
    global_metric.accumulate_metric(metric)
    result = {
        "hmean": float(values["det_h"].cpu()),
        "precision": float(values["det_p"].cpu()),
        "recall": float(values["det_r"].cpu()),
        "gt_regions": len(ground_truth),
    }
    metric.reset()
    return result


def make_loader(module, split, batch_size, workers):
    collate_fn = instantiate(module.config.collate_fn)
    collate_fn.inference_mode = True
    return DataLoader(
        module.dataset[split],
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def load_model(spec, device):
    config = OmegaConf.load(spec["config"])
    module, _ = get_pl_modules_by_cfg(config)
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu")
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    module.to(device)
    module.eval()
    return module


def run_split(module, split, model_name, args):
    loader = make_loader(module, split, args.batch_size, args.workers)
    records = {}
    rows = []
    metric = CLEvalMetric()
    global_metric = CLEvalMetric()
    global_metric.reset()
    start_time = time.monotonic()

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{model_name} {split}"):
            device_batch = move_batch_to_device(batch, args.device)
            prediction = module.model(return_loss=False, **device_batch)
            boxes_batch, scores_batch = module.model.get_polygons_from_maps(
                device_batch, prediction
            )

            for index, filename in enumerate(batch["image_filename"]):
                boxes = boxes_batch[index]
                scores = scores_batch[index]
                records[filename] = {"boxes": boxes, "scores": scores}
                row = prediction_statistics(filename, boxes, scores)
                if split == "val":
                    row.update(
                        evaluate_cleval(
                            boxes,
                            module.dataset["val"].anns[filename],
                            metric,
                            global_metric,
                        )
                    )
                rows.append(row)

    elapsed = time.monotonic() - start_time
    global_values = global_metric.compute() if split == "val" else None
    global_metric.reset()
    frame = pd.DataFrame(rows)
    summary = {
        "model": model_name,
        "split": split,
        "images": len(frame),
        "pred_regions_mean": frame["pred_regions"].mean(),
        "pred_regions_median": frame["pred_regions"].median(),
        "pred_regions_max": frame["pred_regions"].max(),
        "score_mean": frame["score_mean"].mean(),
        "inference_seconds": elapsed,
    }
    if split == "val":
        summary.update(
            {
                "macro_hmean": frame["hmean"].mean(),
                "macro_precision": frame["precision"].mean(),
                "macro_recall": frame["recall"].mean(),
                "global_hmean": float(global_values["det_h"].cpu()),
                "global_precision": float(global_values["det_p"].cpu()),
                "global_recall": float(global_values["det_r"].cpu()),
            }
        )
    return records, frame, summary


def boxes_to_aabbs(boxes):
    if not boxes:
        return np.empty((0, 4), dtype=np.float32)
    bounds = []
    for box in boxes:
        points = np.asarray(box, dtype=np.float32)
        bounds.append(
            [
                points[:, 0].min(),
                points[:, 1].min(),
                points[:, 0].max(),
                points[:, 1].max(),
            ]
        )
    return np.asarray(bounds, dtype=np.float32)


def greedy_bbox_matches(boxes_a, boxes_b, threshold=0.5):
    a = boxes_to_aabbs(boxes_a)
    b = boxes_to_aabbs(boxes_b)
    if not len(a) or not len(b):
        return 0, np.nan

    left_top = np.maximum(a[:, None, :2], b[None, :, :2])
    right_bottom = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection_size = np.clip(right_bottom - left_top, 0, None)
    intersection = intersection_size[:, :, 0] * intersection_size[:, :, 1]
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)

    candidates = np.argwhere(iou >= threshold)
    if not len(candidates):
        return 0, 0.0
    order = np.argsort(iou[candidates[:, 0], candidates[:, 1]])[::-1]
    used_a = set()
    used_b = set()
    matched_ious = []
    for candidate_index in order:
        index_a, index_b = candidates[candidate_index]
        if int(index_a) in used_a or int(index_b) in used_b:
            continue
        used_a.add(int(index_a))
        used_b.add(int(index_b))
        matched_ious.append(float(iou[index_a, index_b]))
    return len(matched_ious), float(np.mean(matched_ious))


def compare_predictions(records_a, records_b, split):
    rows = []
    filenames = sorted(set(records_a) | set(records_b))
    for filename in filenames:
        boxes_a = records_a.get(filename, {}).get("boxes", [])
        boxes_b = records_b.get(filename, {}).get("boxes", [])
        matched, mean_iou = greedy_bbox_matches(boxes_a, boxes_b, threshold=0.5)
        union_count = len(boxes_a) + len(boxes_b) - matched
        rows.append(
            {
                "split": split,
                "filename": filename,
                "v2b_regions": len(boxes_a),
                "v5_regions": len(boxes_b),
                "region_count_delta_v5_v2b": len(boxes_b) - len(boxes_a),
                "matched_iou50": matched,
                "matched_mean_bbox_iou": mean_iou,
                "v2b_only_iou50": len(boxes_a) - matched,
                "v5_only_iou50": len(boxes_b) - matched,
                "prediction_jaccard_iou50": matched / max(union_count, 1),
            }
        )
    return pd.DataFrame(rows)


def draw_polygons(image, boxes, color, thickness=1):
    for box in boxes:
        points = np.asarray(box, dtype=np.int32).reshape((-1, 1, 2))
        if len(points) >= 3:
            cv2.polylines(image, [points], True, color, thickness, cv2.LINE_AA)


def fit_overlay_tile(image, width=420, height=330):
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    scale = min((width - 8) / image.shape[1], (height - 38) / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    x = (width - resized.shape[1]) // 2
    y = 4 + (height - 38 - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def make_overlay_sheet(
    selected,
    records_a,
    records_b,
    image_root,
    output_path,
    annotation_column,
):
    columns = 3
    tile_width, tile_height = 420, 330
    rows = int(np.ceil(len(selected) / columns))
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 255, dtype=np.uint8)

    for position, (_, item) in enumerate(selected.iterrows()):
        filename = item["filename"]
        image = cv2.imread(str(image_root / filename), cv2.IMREAD_COLOR)
        if image is None:
            continue
        draw_polygons(image, records_a[filename]["boxes"], (40, 210, 40), 2)
        draw_polygons(image, records_b[filename]["boxes"], (220, 50, 220), 1)
        tile = fit_overlay_tile(image, tile_width, tile_height)
        label = f"green=V2B magenta=V5 | {annotation_column}={item[annotation_column]:.3f}"
        cv2.putText(
            tile,
            label,
            (5, tile_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            filename[-48:],
            (5, tile_height - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )
        y = (position // columns) * tile_height
        x = (position % columns) * tile_width
        sheet[y : y + tile_height, x : x + tile_width] = tile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def quality_correlations(image_metrics, model_frame):
    quality = image_metrics[image_metrics["dataset"] == "val"].copy()
    merged = quality.merge(model_frame, on="filename", how="inner")
    quality_columns = [
        "brightness",
        "contrast",
        "blur_laplacian",
        "edge_density",
        "entropy",
        "saturation",
        "dark_fraction",
        "aspect_long_short",
        "megapixels",
        "gt_regions_x",
        "short_side_1024_median",
        "small_text_lt8_ratio",
        "small_text_lt12_ratio",
    ]
    rows = []
    for quality_column in quality_columns:
        if quality_column not in merged:
            continue
        for metric in ("hmean", "precision", "recall"):
            pair = merged[[quality_column, metric]].dropna()
            if len(pair) < 3:
                continue
            correlation, pvalue = spearmanr(pair[quality_column], pair[metric])
            rows.append(
                {
                    "quality_metric": quality_column,
                    "cleval_metric": metric,
                    "spearman_rho": correlation,
                    "pvalue": pvalue,
                    "samples": len(pair),
                }
            )
    return merged, pd.DataFrame(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")

    all_records = {}
    frames = {}
    summaries = []
    for model_name, spec in MODEL_SPECS.items():
        print(f"Loading {model_name}: {spec['checkpoint']}")
        module = load_model(spec, args.device)
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        for split in ("val", "predict"):
            display_split = "test" if split == "predict" else split
            records, frame, summary = run_split(module, split, model_name, args)
            summary["split"] = display_split
            if args.device.startswith("cuda"):
                summary["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
            frame.insert(0, "model", model_name)
            frame.insert(1, "split", display_split)
            frame.to_csv(
                args.output_dir / f"{model_name}_{display_split}_predictions.csv",
                index=False,
            )
            all_records[(model_name, display_split)] = records
            frames[(model_name, display_split)] = frame
            summaries.append(summary)

        del module
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(args.output_dir / "model_prediction_summary.csv", index=False)

    for split in ("val", "test"):
        comparison = compare_predictions(
            all_records[("v2b", split)], all_records[("v5", split)], split
        )
        if split == "val":
            v2b_metrics = frames[("v2b", "val")][
                ["filename", "hmean", "precision", "recall"]
            ].rename(
                columns={
                    "hmean": "v2b_hmean",
                    "precision": "v2b_precision",
                    "recall": "v2b_recall",
                }
            )
            v5_metrics = frames[("v5", "val")][
                ["filename", "hmean", "precision", "recall"]
            ].rename(
                columns={
                    "hmean": "v5_hmean",
                    "precision": "v5_precision",
                    "recall": "v5_recall",
                }
            )
            comparison = comparison.merge(v2b_metrics, on="filename").merge(
                v5_metrics, on="filename"
            )
            comparison["hmean_delta_v5_v2b"] = (
                comparison["v5_hmean"] - comparison["v2b_hmean"]
            )
        comparison.to_csv(
            args.output_dir / f"v2b_v5_{split}_disagreement.csv", index=False
        )

        image_root = PROJECT_ROOT / "data/datasets/images" / split
        if split == "test":
            selected = comparison.nsmallest(12, "prediction_jaccard_iou50")
            annotation_column = "prediction_jaccard_iou50"
        else:
            positive = comparison.nlargest(6, "hmean_delta_v5_v2b")
            negative = comparison.nsmallest(6, "hmean_delta_v5_v2b")
            selected = pd.concat([positive, negative], ignore_index=True)
            annotation_column = "hmean_delta_v5_v2b"
        make_overlay_sheet(
            selected,
            all_records[("v2b", split)],
            all_records[("v5", split)],
            image_root,
            args.output_dir / "visualizations" / f"{split}_model_disagreement.jpg",
            annotation_column,
        )

    image_metrics = pd.read_csv(args.output_dir / "image_metrics.csv")
    merged, correlations = quality_correlations(image_metrics, frames[("v2b", "val")])
    merged.to_csv(args.output_dir / "v2b_val_quality_metrics.csv", index=False)
    correlations.to_csv(args.output_dir / "val_quality_correlation.csv", index=False)

    print(f"Wrote D0 model comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
