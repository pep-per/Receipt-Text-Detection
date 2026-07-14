#!/usr/bin/env python3
"""Compute image-quality and annotation statistics for D0 data audit."""

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
QUALITY_METRICS = [
    "brightness",
    "contrast",
    "blur_laplacian",
    "edge_density",
    "entropy",
    "saturation",
    "dark_fraction",
    "aspect_long_short",
    "megapixels",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/20260714-d0-data-audit"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--quality-size", type=int, default=512)
    return parser.parse_args()


def load_official_records(data_root):
    dataset_root = data_root / "datasets"
    records = []
    for split in ("train", "val", "test"):
        annotation_path = dataset_root / "jsons" / f"{split}.json"
        with annotation_path.open() as fp:
            annotations = json.load(fp)["images"]

        for filename, item in annotations.items():
            records.append(
                {
                    "dataset": split,
                    "filename": filename,
                    "image_path": dataset_root / "images" / split / filename,
                    "words": item.get("words", {}),
                }
            )
    return records


def load_auxiliary_records(data_root):
    pseudo_root = data_root / "pseudo_label"
    records = []
    for source in ("sroie", "wildreceipt", "cord-v2"):
        source_root = pseudo_root / source
        for image_path in sorted(source_root.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(
                    {
                        "dataset": f"pseudo_{source}",
                        "filename": str(image_path.relative_to(source_root)),
                        "image_path": image_path,
                        "words": {},
                    }
                )
    return records


def resize_for_quality(image, max_size):
    height, width = image.shape[:2]
    scale = min(1.0, max_size / max(height, width))
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def grayscale_entropy(gray):
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probabilities = histogram / max(histogram.sum(), 1)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def normalized_rect_angle(rect):
    (_, _), (width, height), angle = rect
    if width < height:
        angle += 90.0
    angle = ((angle + 90.0) % 180.0) - 90.0
    return abs(float(angle))


def polygon_metrics(words, image_width, image_height):
    short_sides = []
    areas = []
    vertices = []
    angles = []
    scale_1024 = 1024.0 / max(image_width, image_height)

    for word in words.values():
        points = np.asarray(word.get("points", []), dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 3:
            continue
        rect = cv2.minAreaRect(points)
        side_a, side_b = rect[1]
        short_sides.append(min(side_a, side_b) * scale_1024)
        areas.append(abs(float(cv2.contourArea(points))))
        vertices.append(points.shape[0])
        angles.append(normalized_rect_angle(rect))

    image_area = max(image_width * image_height, 1)
    if not short_sides:
        return {
            "gt_regions": 0,
            "vertices_mean": np.nan,
            "vertices_max": np.nan,
            "short_side_1024_median": np.nan,
            "short_side_1024_q10": np.nan,
            "small_text_lt8_ratio": np.nan,
            "small_text_lt12_ratio": np.nan,
            "polygon_area_ratio_median": np.nan,
            "polygon_area_ratio_sum": np.nan,
            "polygon_angle_abs_median": np.nan,
        }

    short_sides = np.asarray(short_sides)
    area_ratios = np.asarray(areas) / image_area
    return {
        "gt_regions": len(short_sides),
        "vertices_mean": float(np.mean(vertices)),
        "vertices_max": int(np.max(vertices)),
        "short_side_1024_median": float(np.median(short_sides)),
        "short_side_1024_q10": float(np.quantile(short_sides, 0.1)),
        "small_text_lt8_ratio": float(np.mean(short_sides < 8)),
        "small_text_lt12_ratio": float(np.mean(short_sides < 12)),
        "polygon_area_ratio_median": float(np.median(area_ratios)),
        "polygon_area_ratio_sum": float(np.sum(area_ratios)),
        "polygon_angle_abs_median": float(np.median(angles)),
    }


def analyze_record(record, quality_size, project_root):
    image = cv2.imread(str(record["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "dataset": record["dataset"],
            "filename": record["filename"],
            "image_path": str(record["image_path"]),
            "read_error": True,
        }

    height, width = image.shape[:2]
    quality_image = resize_for_quality(image, quality_size)
    gray = cv2.cvtColor(quality_image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(quality_image, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 100, 200)

    try:
        relative_path = record["image_path"].resolve().relative_to(project_root.resolve())
    except ValueError:
        relative_path = record["image_path"]

    result = {
        "dataset": record["dataset"],
        "filename": record["filename"],
        "image_path": str(relative_path),
        "read_error": False,
        "width": width,
        "height": height,
        "megapixels": width * height / 1_000_000.0,
        "aspect_width_height": width / max(height, 1),
        "aspect_long_short": max(width, height) / max(min(width, height), 1),
        "portrait": height > width,
        "brightness": float(gray.mean() / 255.0),
        "contrast": float(gray.std() / 255.0),
        "blur_laplacian": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "edge_density": float(np.mean(edges > 0)),
        "entropy": grayscale_entropy(gray),
        "saturation": float(hsv[:, :, 1].mean() / 255.0),
        "dark_fraction": float(np.mean(gray < 64)),
        "bright_fraction": float(np.mean(gray > 224)),
    }
    result.update(polygon_metrics(record["words"], width, height))
    return result


def summarize_datasets(frame):
    metrics = QUALITY_METRICS + [
        "width",
        "height",
        "gt_regions",
        "short_side_1024_median",
        "small_text_lt8_ratio",
        "small_text_lt12_ratio",
    ]
    rows = []
    for dataset, group in frame.groupby("dataset", sort=False):
        row = {"dataset": dataset, "images": len(group)}
        for metric in metrics:
            values = group[metric].dropna()
            if values.empty:
                continue
            row[f"{metric}_median"] = values.median()
            row[f"{metric}_q10"] = values.quantile(0.1)
            row[f"{metric}_q90"] = values.quantile(0.9)
        rows.append(row)
    return pd.DataFrame(rows)


def calculate_distribution_shift(frame):
    reference = frame[frame["dataset"] == "train"]
    rows = []
    for dataset, group in frame.groupby("dataset", sort=False):
        if dataset == "train":
            continue
        for metric in QUALITY_METRICS:
            ref_values = reference[metric].dropna().to_numpy()
            target_values = group[metric].dropna().to_numpy()
            if not len(ref_values) or not len(target_values):
                continue
            iqr = np.quantile(ref_values, 0.75) - np.quantile(ref_values, 0.25)
            distance = wasserstein_distance(ref_values, target_values)
            ks_result = ks_2samp(ref_values, target_values)
            rows.append(
                {
                    "reference": "train",
                    "target": dataset,
                    "metric": metric,
                    "reference_median": np.median(ref_values),
                    "target_median": np.median(target_values),
                    "median_delta": np.median(target_values) - np.median(ref_values),
                    "ks_statistic": ks_result.statistic,
                    "ks_pvalue": ks_result.pvalue,
                    "wasserstein_distance": distance,
                    "wasserstein_over_train_iqr": distance / max(iqr, 1e-12),
                }
            )
    return pd.DataFrame(rows)


def plot_distributions(frame, output_path):
    plot_frame = frame.copy()
    plot_frame["log_blur"] = np.log1p(plot_frame["blur_laplacian"])
    metrics = [
        ("brightness", "Brightness"),
        ("contrast", "Contrast"),
        ("log_blur", "log(1 + Laplacian variance)"),
        ("edge_density", "Edge density"),
        ("aspect_long_short", "Long/short aspect"),
        ("megapixels", "Megapixels"),
    ]
    datasets = list(plot_frame["dataset"].drop_duplicates())
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    figure, axes = plt.subplots(2, 3, figsize=(18, 10))
    for axis, (metric, title) in zip(axes.ravel(), metrics):
        for dataset, color in zip(datasets, colors):
            values = plot_frame.loc[plot_frame["dataset"] == dataset, metric].dropna()
            if values.empty:
                continue
            sorted_values = np.sort(values.to_numpy())
            y = np.linspace(0, 1, len(sorted_values), endpoint=True)
            axis.plot(sorted_values, y, label=dataset, color=color, linewidth=1.5)
        axis.set_title(title)
        axis.set_ylabel("ECDF")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Train/Val/Test and auxiliary receipt image distributions", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def fit_thumbnail(image, width=240, height=190):
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    scale = min((width - 8) / image.shape[1], (height - 30) / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    x = (width - resized.shape[1]) // 2
    y = 4 + (height - 30 - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def create_quantile_sheet(group, output_path, project_root):
    metrics = [
        ("brightness", "brightness"),
        ("contrast", "contrast"),
        ("blur_laplacian", "blur"),
        ("aspect_long_short", "aspect"),
    ]
    quantiles = [0.0, 0.1, 0.5, 0.9, 1.0]
    tile_width, tile_height = 240, 190
    sheet = np.full(
        (len(metrics) * tile_height, len(quantiles) * tile_width, 3),
        255,
        dtype=np.uint8,
    )

    for row_index, (metric, short_name) in enumerate(metrics):
        ordered = group.dropna(subset=[metric]).sort_values(metric).reset_index(drop=True)
        for column_index, quantile in enumerate(quantiles):
            position = round(quantile * max(len(ordered) - 1, 0))
            item = ordered.iloc[position]
            image_path = project_root / item["image_path"]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            tile = fit_thumbnail(image, tile_width, tile_height)
            label = f"{short_name} q{quantile:.1f}={item[metric]:.3g}"
            cv2.putText(
                tile,
                label,
                (5, tile_height - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            filename = str(item["filename"])[-31:]
            cv2.putText(
                tile,
                filename,
                (5, tile_height - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (60, 60, 60),
                1,
                cv2.LINE_AA,
            )
            y = row_index * tile_height
            x = column_index * tile_width
            sheet[y : y + tile_height, x : x + tile_width] = tile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def write_summary_markdown(summary, shifts, output_path):
    def to_markdown(frame, float_digits=4):
        def format_value(value):
            if pd.isna(value):
                return ""
            if isinstance(value, (float, np.floating)):
                return f"{value:.{float_digits}f}"
            return str(value)

        headers = [str(column) for column in frame.columns]
        rows = [
            [format_value(value) for value in row]
            for row in frame.itertuples(index=False, name=None)
        ]
        widths = [
            max(len(headers[index]), *(len(row[index]) for row in rows))
            for index in range(len(headers))
        ]

        def render(row):
            return "| " + " | ".join(
                value.ljust(widths[index]) for index, value in enumerate(row)
            ) + " |"

        separator = ["-" * width for width in widths]
        return "\n".join([render(headers), render(separator), *(render(row) for row in rows)])

    official = summary[summary["dataset"].isin(["train", "val", "test"])].copy()
    columns = [
        "dataset",
        "images",
        "brightness_median",
        "contrast_median",
        "blur_laplacian_median",
        "aspect_long_short_median",
        "megapixels_median",
        "gt_regions_median",
    ]
    available_columns = [column for column in columns if column in official.columns]
    strongest = shifts.sort_values("ks_statistic", ascending=False).groupby("target").head(3)

    lines = [
        "# D0 Data Summary",
        "",
        "## Official Dataset Medians",
        "",
        to_markdown(official[available_columns]),
        "",
        "## Largest Train-to-Target Distribution Shifts",
        "",
        to_markdown(
            strongest[
                ["target", "metric", "reference_median", "target_median", "ks_statistic"]
            ]
        ),
        "",
        "KS statistics rank distribution differences but do not by themselves prove that an",
        "augmentation will improve CLEval. Adoption still requires controlled local evaluation.",
        "",
    ]
    output_path.write_text("\n".join(lines))


def main():
    args = parse_args()
    project_root = Path.cwd()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_official_records(args.data_root) + load_auxiliary_records(args.data_root)
    worker = lambda record: analyze_record(record, args.quality_size, project_root)  # noqa: E731
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        analyzed = list(
            tqdm(
                executor.map(worker, records),
                total=len(records),
                desc="Image audit",
            )
        )

    frame = pd.DataFrame(analyzed)
    read_errors = frame[frame["read_error"]]
    if not read_errors.empty:
        read_errors.to_csv(output_dir / "read_errors.csv", index=False)
    frame = frame[~frame["read_error"]].copy()
    frame.to_csv(output_dir / "image_metrics.csv", index=False)

    summary = summarize_datasets(frame)
    summary.to_csv(output_dir / "dataset_summary.csv", index=False)
    shifts = calculate_distribution_shift(frame)
    shifts.to_csv(output_dir / "distribution_shift.csv", index=False)
    plot_distributions(frame, output_dir / "distribution_ecdf.png")

    contact_dir = output_dir / "contact_sheets"
    for dataset, group in frame.groupby("dataset", sort=False):
        create_quantile_sheet(
            group,
            contact_dir / f"{dataset}_quality_quantiles.jpg",
            project_root,
        )

    write_summary_markdown(summary, shifts, output_dir / "data_summary.md")
    print(f"Wrote D0 audit for {len(frame)} images to {output_dir}")


if __name__ == "__main__":
    main()
