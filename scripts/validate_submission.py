#!/usr/bin/env python3
"""Validate receipt text detection JSON and CSV submission artifacts."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = PROJECT_ROOT / "baseline_code"
sys.path.insert(0, str(BASELINE_ROOT))

from ocr.datasets import OCRDataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--test-json",
        type=Path,
        default=PROJECT_ROOT / "data/datasets/jsons/test.json",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=PROJECT_ROOT / "data/datasets/images/test",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-polygons", type=int, default=500)
    parser.add_argument("--boundary-tolerance", type=float, default=8.0)
    return parser.parse_args()


def load_original_size(image_path):
    with Image.open(image_path) as image:
        exif = image.getexif()
        if exif and 274 in exif:
            image = OCRDataset.rotate_image(image, exif[274])
        return image.size


def boundary_overshoot(points, width, height):
    points = np.asarray(points, dtype=np.float64)
    return float(
        max(
            0.0,
            -points[:, 0].min(),
            -points[:, 1].min(),
            points[:, 0].max() - (width - 1),
            points[:, 1].max() - (height - 1),
        )
    )


def validate_json(args, expected_filenames):
    with args.json.open() as file:
        data = json.load(file)
    images = data.get("images")
    if not isinstance(images, dict):
        raise ValueError("Prediction JSON must contain an images object")

    filenames = list(images)
    expected = set(expected_filenames)
    actual = set(filenames)
    counts = []
    invalid_point_count = 0
    non_finite_coordinates = 0
    zero_area = 0
    invalid_geometry = 0
    duplicate_polygons = 0
    boundary_overshoot_polygons = 0
    hard_boundary_violations = 0
    max_boundary_overshoot = 0.0

    for filename, image_data in images.items():
        words = image_data.get("words")
        if not isinstance(words, dict):
            raise ValueError(f"Missing words object: {filename}")
        counts.append(len(words))
        size = load_original_size(args.image_dir / filename)
        seen = set()
        for word in words.values():
            points = word.get("points")
            if not isinstance(points, list) or len(points) < 4:
                invalid_point_count += 1
                continue
            try:
                array = np.asarray(points, dtype=np.float64)
            except (TypeError, ValueError):
                non_finite_coordinates += 1
                continue
            if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
                non_finite_coordinates += 1
                continue
            area = abs(float(cv2.contourArea(array.astype(np.float32))))
            if area <= 0:
                zero_area += 1
            polygon = Polygon(array)
            if not polygon.is_valid:
                invalid_geometry += 1
            canonical = tuple(map(tuple, array.tolist()))
            if canonical in seen:
                duplicate_polygons += 1
            seen.add(canonical)
            overshoot = boundary_overshoot(array, *size)
            if overshoot > 0:
                boundary_overshoot_polygons += 1
                max_boundary_overshoot = max(max_boundary_overshoot, overshoot)
            if overshoot > args.boundary_tolerance:
                hard_boundary_violations += 1

    return {
        "rows": len(filenames),
        "missing_filenames": sorted(expected - actual),
        "extra_filenames": sorted(actual - expected),
        "total_polygons": int(sum(counts)),
        "min_polygons": int(min(counts)) if counts else 0,
        "max_polygons": int(max(counts)) if counts else 0,
        "mean_polygons": float(np.mean(counts)) if counts else 0.0,
        "empty_images": int(sum(count == 0 for count in counts)),
        "over_cap_images": int(sum(count > args.max_polygons for count in counts)),
        "invalid_point_count": invalid_point_count,
        "non_finite_coordinates": non_finite_coordinates,
        "zero_area": zero_area,
        "invalid_geometry": invalid_geometry,
        "duplicate_polygons": duplicate_polygons,
        "boundary_overshoot_polygons": boundary_overshoot_polygons,
        "max_boundary_overshoot": max_boundary_overshoot,
        "hard_boundary_violations": hard_boundary_violations,
        "polygon_counts": dict(zip(filenames, counts)),
    }


def validate_csv(args, expected_filenames, json_counts):
    rows = []
    invalid_coordinate_sets = 0
    non_finite_coordinates = 0
    count_mismatches = 0
    with args.csv.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["filename", "polygons"]:
            raise ValueError(f"Unexpected CSV header: {reader.fieldnames}")
        for row in reader:
            filename = row["filename"]
            polygon_field = row["polygons"] or ""
            polygon_strings = polygon_field.split("|") if polygon_field else []
            rows.append(filename)
            if len(polygon_strings) != json_counts.get(filename, -1):
                count_mismatches += 1
            for polygon_string in polygon_strings:
                try:
                    coordinates = [float(value) for value in polygon_string.split()]
                except ValueError:
                    invalid_coordinate_sets += 1
                    continue
                if len(coordinates) < 8 or len(coordinates) % 2 != 0:
                    invalid_coordinate_sets += 1
                if not all(math.isfinite(value) for value in coordinates):
                    non_finite_coordinates += 1

    expected = set(expected_filenames)
    actual = set(rows)
    return {
        "rows": len(rows),
        "missing_filenames": sorted(expected - actual),
        "extra_filenames": sorted(actual - expected),
        "duplicate_filename_rows": len(rows) - len(actual),
        "invalid_coordinate_sets": invalid_coordinate_sets,
        "non_finite_coordinates": non_finite_coordinates,
        "json_polygon_count_mismatches": count_mismatches,
    }


def main():
    args = parse_args()
    with args.test_json.open() as file:
        expected_filenames = list(json.load(file)["images"])

    json_report = validate_json(args, expected_filenames)
    polygon_counts = json_report.pop("polygon_counts")
    csv_report = validate_csv(args, expected_filenames, polygon_counts)
    report = {
        "json_path": str(args.json.resolve()),
        "csv_path": str(args.csv.resolve()),
        "expected_images": len(expected_filenames),
        "max_polygons_per_image": args.max_polygons,
        "boundary_tolerance": args.boundary_tolerance,
        "json": json_report,
        "csv": csv_report,
    }
    fatal_values = (
        len(json_report["missing_filenames"]),
        len(json_report["extra_filenames"]),
        json_report["over_cap_images"],
        json_report["invalid_point_count"],
        json_report["non_finite_coordinates"],
        json_report["zero_area"],
        json_report["invalid_geometry"],
        json_report["duplicate_polygons"],
        json_report["hard_boundary_violations"],
        len(csv_report["missing_filenames"]),
        len(csv_report["extra_filenames"]),
        csv_report["duplicate_filename_rows"],
        csv_report["invalid_coordinate_sets"],
        csv_report["non_finite_coordinates"],
        csv_report["json_polygon_count_mismatches"],
    )
    report["passed"] = not any(fatal_values)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
