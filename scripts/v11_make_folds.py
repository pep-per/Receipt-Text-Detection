#!/usr/bin/env python3
"""Create the pre-registered V11 group-aware stratified folds."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


OFFICIAL_SPLITS = ("train", "val")
CONTINUOUS_AUDIT_COLUMNS = [
    "gt_regions",
    "small_text_lt12_ratio",
    "short_side_1024_median",
    "aspect_long_short",
    "brightness",
    "contrast",
    "blur_laplacian",
    "megapixels",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/datasets"))
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("experiments/20260714-d0-data-audit/image_metrics.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/splits/v11_5fold_seed42.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/splits/v11_5fold_seed42_metadata.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/20260714-v11a-kfold-split"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        keep, merge = sorted((left_root, right_root))
        self.parent[merge] = keep


def load_annotations(dataset_root):
    rows = []
    for split in OFFICIAL_SPLITS:
        annotation_path = dataset_root / "jsons" / f"{split}.json"
        with annotation_path.open() as fp:
            annotations = json.load(fp)["images"]
        for filename, item in annotations.items():
            rows.append(
                {
                    "filename": filename,
                    "original_split": split,
                    "annotation_regions": len(item.get("words", {})),
                }
            )
    return pd.DataFrame(rows)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(gray):
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(resized.astype(np.float32))[:8, :8]
    bits = coefficients > np.median(coefficients)
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def image_fingerprint(path):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Could not read image: {path}")
    normalized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    return {
        "sha256": sha256_file(path),
        "phash_int": perceptual_hash(gray),
        "gray_128": normalized.astype(np.float32).ravel(),
    }


def hamming_distance(left, right):
    return (left ^ right).bit_count()


def relative_difference(left, right):
    return abs(float(left) - float(right)) / max(abs(float(left)), abs(float(right)), 1e-12)


def grayscale_correlation(left, right):
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denominator == 0:
        return float(left.mean() == right.mean())
    return float(np.dot(left_centered, right_centered) / denominator)


def audit_near_duplicates(frame, dataset_root):
    fingerprints = {}
    for row in frame.itertuples(index=False):
        path = dataset_root / "images" / row.original_split / row.filename
        fingerprints[row.filename] = image_fingerprint(path)

    rows = frame.set_index("filename").to_dict("index")
    filenames = sorted(rows)
    candidates = []
    accepted_pairs = []
    union_find = UnionFind(filenames)

    for left_index, left_name in enumerate(filenames):
        left = rows[left_name]
        left_fp = fingerprints[left_name]
        for right_name in filenames[left_index + 1 :]:
            right = rows[right_name]
            right_fp = fingerprints[right_name]

            exact_sha = left_fp["sha256"] == right_fp["sha256"]
            aspect_difference = relative_difference(
                left["aspect_long_short"], right["aspect_long_short"]
            )
            distance = hamming_distance(left_fp["phash_int"], right_fp["phash_int"])
            if not exact_sha and (aspect_difference > 0.12 or distance > 8):
                continue

            word_difference = relative_difference(left["gt_regions"], right["gt_regions"])
            correlation = grayscale_correlation(
                left_fp["gray_128"], right_fp["gray_128"]
            )
            rule = ""
            if exact_sha:
                rule = "exact_sha256"
            elif distance <= 2 and word_difference <= 0.10:
                rule = "phash_le2_words_le10pct"
            elif distance <= 6 and correlation >= 0.90 and word_difference <= 0.10:
                rule = "phash_le6_corr_ge090_words_le10pct"

            accepted = bool(rule)
            candidate = {
                "left_filename": left_name,
                "right_filename": right_name,
                "left_split": left["original_split"],
                "right_split": right["original_split"],
                "phash_distance": distance,
                "aspect_relative_difference": aspect_difference,
                "grayscale_correlation": correlation,
                "left_gt_regions": int(left["gt_regions"]),
                "right_gt_regions": int(right["gt_regions"]),
                "word_count_relative_difference": word_difference,
                "exact_sha256": exact_sha,
                "accepted": accepted,
                "acceptance_rule": rule,
            }
            candidates.append(candidate)
            if accepted:
                union_find.union(left_name, right_name)
                accepted_pairs.append(candidate)

    members = {}
    for filename in filenames:
        members.setdefault(union_find.find(filename), []).append(filename)
    canonical_group = {}
    for component in members.values():
        group_id = min(component)
        for filename in component:
            canonical_group[filename] = group_id

    fingerprints_for_manifest = {
        filename: {
            "sha256": values["sha256"],
            "phash64": f"{values['phash_int']:016x}",
        }
        for filename, values in fingerprints.items()
    }
    return candidates, accepted_pairs, canonical_group, fingerprints_for_manifest


def assign_strata_and_folds(frame, folds, seed):
    frame = frame.copy()
    frame["gt_region_bin"] = pd.qcut(
        frame["gt_regions"].rank(method="first"), folds, labels=False
    ).astype(int)
    frame["small_text_bin"] = pd.qcut(
        frame["small_text_lt12_ratio"].rank(method="first"),
        2,
        labels=False,
    ).astype(int)
    frame["stratum"] = (
        frame["original_split"]
        + "_w"
        + frame["gt_region_bin"].astype(str)
        + "_s"
        + frame["small_text_bin"].astype(str)
    )

    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=False,
    )
    assignments = np.full(len(frame), -1, dtype=int)
    inputs = np.zeros((len(frame), 1), dtype=np.uint8)
    for fold, (_, validation_indices) in enumerate(
        splitter.split(inputs, frame["stratum"], frame["group_id"])
    ):
        assignments[validation_indices] = fold
    if np.any(assignments < 0):
        raise RuntimeError("At least one image was not assigned to a fold")
    frame["fold"] = assignments
    return frame


def fold_summary(frame):
    rows = []
    for fold, group in frame.groupby("fold", sort=True):
        row = {
            "fold": int(fold),
            "images": len(group),
            "train_source_images": int((group["original_split"] == "train").sum()),
            "val_source_images": int((group["original_split"] == "val").sum()),
            "groups": group["group_id"].nunique(),
            "gt_regions_total": int(group["gt_regions"].sum()),
        }
        for column in CONTINUOUS_AUDIT_COLUMNS:
            row[f"{column}_mean"] = float(group[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def stratum_summary(frame):
    overall = frame["stratum"].value_counts(normalize=True)
    rows = []
    for fold, group in frame.groupby("fold", sort=True):
        counts = group["stratum"].value_counts()
        for stratum in sorted(overall.index):
            count = int(counts.get(stratum, 0))
            proportion = count / len(group)
            rows.append(
                {
                    "fold": int(fold),
                    "stratum": stratum,
                    "count": count,
                    "fold_proportion": proportion,
                    "overall_proportion": float(overall[stratum]),
                    "proportion_delta": proportion - float(overall[stratum]),
                }
            )
    return pd.DataFrame(rows)


def validate_split(frame, accepted_pairs, folds, seed):
    group_fold_counts = frame.groupby("group_id")["fold"].nunique()
    accepted_pair_leakage = []
    fold_by_filename = frame.set_index("filename")["fold"].to_dict()
    for pair in accepted_pairs:
        if fold_by_filename[pair["left_filename"]] != fold_by_filename[pair["right_filename"]]:
            accepted_pair_leakage.append(
                [pair["left_filename"], pair["right_filename"]]
            )

    fold_sizes = frame.groupby("fold").size()
    source_val_counts = (
        frame[frame["original_split"] == "val"].groupby("fold").size().reindex(range(folds), fill_value=0)
    )
    feature_balance = {}
    max_feature_mean_z = 0.0
    for column in CONTINUOUS_AUDIT_COLUMNS:
        overall_mean = float(frame[column].mean())
        overall_std = float(frame[column].std(ddof=0))
        fold_means = frame.groupby("fold")[column].mean()
        if overall_std > 0:
            max_abs_z = float(((fold_means - overall_mean).abs() / overall_std).max())
        else:
            max_abs_z = 0.0
        max_feature_mean_z = max(max_feature_mean_z, max_abs_z)
        feature_balance[column] = {
            "overall_mean": overall_mean,
            "fold_means": {str(int(k)): float(v) for k, v in fold_means.items()},
            "max_abs_fold_mean_z": max_abs_z,
        }

    duplicate_filenames = int(frame["filename"].duplicated().sum())
    missing_fold_ids = sorted(set(range(folds)) - set(frame["fold"].unique()))
    stratum = stratum_summary(frame)
    max_stratum_delta = float(stratum["proportion_delta"].abs().max())
    checks = {
        "expected_image_count": len(frame) == 3676,
        "unique_filenames": duplicate_filenames == 0,
        "all_fold_ids_present": not missing_fold_ids,
        "fold_size_difference_at_most_2": int(fold_sizes.max() - fold_sizes.min()) <= 2,
        "no_group_leakage": int((group_fold_counts > 1).sum()) == 0,
        "no_accepted_pair_leakage": not accepted_pair_leakage,
        "source_val_count_difference_at_most_2": int(source_val_counts.max() - source_val_counts.min()) <= 2,
        "max_feature_mean_z_below_0_15": max_feature_mean_z < 0.15,
        "max_stratum_proportion_delta_below_0_01": max_stratum_delta < 0.01,
    }
    return {
        "fold_count": folds,
        "seed": seed,
        "image_count": len(frame),
        "unique_filename_count": int(frame["filename"].nunique()),
        "duplicate_filename_count": duplicate_filenames,
        "fold_sizes": {str(int(k)): int(v) for k, v in fold_sizes.items()},
        "fold_size_difference": int(fold_sizes.max() - fold_sizes.min()),
        "source_val_counts": {str(int(k)): int(v) for k, v in source_val_counts.items()},
        "group_count": int(frame["group_id"].nunique()),
        "multi_image_group_count": int((frame.groupby("group_id").size() > 1).sum()),
        "group_leakage_count": int((group_fold_counts > 1).sum()),
        "accepted_pair_leakage": accepted_pair_leakage,
        "missing_fold_ids": missing_fold_ids,
        "max_feature_mean_z": max_feature_mean_z,
        "max_stratum_proportion_delta": max_stratum_delta,
        "feature_balance": feature_balance,
        "checks": checks,
        "passed": all(checks.values()),
    }


def draw_duplicate_contact_sheet(accepted_pairs, frame, dataset_root, output_path):
    if not accepted_pairs:
        return
    split_by_filename = frame.set_index("filename")["original_split"].to_dict()
    thumb_width, thumb_height = 260, 320
    label_height = 58
    canvas = np.full(
        (len(accepted_pairs) * (thumb_height + label_height), thumb_width * 2, 3),
        255,
        dtype=np.uint8,
    )
    for row_index, pair in enumerate(accepted_pairs):
        y = row_index * (thumb_height + label_height)
        for column_index, key in enumerate(("left_filename", "right_filename")):
            filename = pair[key]
            path = dataset_root / "images" / split_by_filename[filename] / filename
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            height, width = image.shape[:2]
            scale = min(thumb_width / width, thumb_height / height)
            resized = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            x = column_index * thumb_width + (thumb_width - resized.shape[1]) // 2
            image_y = y + (thumb_height - resized.shape[0]) // 2
            canvas[image_y : image_y + resized.shape[0], x : x + resized.shape[1]] = resized
            short_name = re.search(r"(\d{6})\.jpg$", filename).group(1)
            cv2.putText(
                canvas,
                f"{short_name} ({split_by_filename[filename]})",
                (column_index * thumb_width + 8, y + thumb_height + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            f"pHash {pair['phash_distance']}  corr {pair['grayscale_correlation']:.3f}",
            (8, y + thumb_height + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), canvas)


def main():
    args = parse_args()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_annotations(args.dataset_root)
    metrics = pd.read_csv(args.metrics)
    metrics = metrics[metrics["dataset"].isin(OFFICIAL_SPLITS)].copy()
    metrics = metrics.rename(columns={"dataset": "original_split"})
    metrics = metrics.drop(columns=["image_path", "read_error"], errors="ignore")
    frame = annotations.merge(
        metrics,
        on=["filename", "original_split"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if set(frame["_merge"]) != {"both"}:
        raise RuntimeError(f"Annotation/D0 mismatch: {frame['_merge'].value_counts().to_dict()}")
    frame = frame.drop(columns="_merge")
    if not np.array_equal(frame["annotation_regions"], frame["gt_regions"]):
        raise RuntimeError("Annotation word counts differ from D0 gt_regions")
    frame = frame.drop(columns="annotation_regions")

    candidates, accepted_pairs, group_ids, fingerprints = audit_near_duplicates(
        frame, args.dataset_root
    )
    frame["group_id"] = frame["filename"].map(group_ids)
    group_sizes = frame.groupby("group_id").size()
    frame["group_size"] = frame["group_id"].map(group_sizes).astype(int)
    frame["sha256"] = frame["filename"].map(lambda name: fingerprints[name]["sha256"])
    frame["phash64"] = frame["filename"].map(lambda name: fingerprints[name]["phash64"])
    frame["numeric_id"] = frame["filename"].str.extract(r"(\d{6})\.jpg$")[0].astype(int)
    frame["image_path"] = frame.apply(
        lambda row: f"data/datasets/images/{row['original_split']}/{row['filename']}",
        axis=1,
    )
    frame = assign_strata_and_folds(frame, args.folds, args.seed)
    frame = frame.sort_values(["numeric_id", "filename"]).reset_index(drop=True)

    manifest_columns = [
        "filename",
        "image_path",
        "original_split",
        "fold",
        "group_id",
        "group_size",
        "numeric_id",
        "stratum",
        "gt_region_bin",
        "small_text_bin",
        "gt_regions",
        "width",
        "height",
        "megapixels",
        "aspect_long_short",
        "brightness",
        "contrast",
        "blur_laplacian",
        "short_side_1024_median",
        "short_side_1024_q10",
        "small_text_lt8_ratio",
        "small_text_lt12_ratio",
        "sha256",
        "phash64",
    ]
    frame[manifest_columns].to_csv(args.manifest, index=False, float_format="%.10g")

    candidate_frame = pd.DataFrame(candidates).sort_values(
        ["accepted", "phash_distance", "left_filename", "right_filename"],
        ascending=[False, True, True, True],
    )
    candidate_frame.to_csv(args.output_dir / "near_duplicate_pairs.csv", index=False)
    fold_summary(frame).to_csv(args.output_dir / "fold_summary.csv", index=False)
    stratum_summary(frame).to_csv(args.output_dir / "stratum_balance.csv", index=False)

    validation = validate_split(frame, accepted_pairs, args.folds, args.seed)
    with (args.output_dir / "split_validation.json").open("w") as fp:
        json.dump(validation, fp, indent=2, ensure_ascii=True)
        fp.write("\n")

    draw_duplicate_contact_sheet(
        accepted_pairs,
        frame,
        args.dataset_root,
        args.output_dir / "near_duplicate_groups.jpg",
    )

    manifest_sha256 = sha256_file(args.manifest)
    metadata = {
        "version": "v11_5fold_seed42",
        "image_count": len(frame),
        "folds": args.folds,
        "seed": args.seed,
        "splitter": "sklearn.model_selection.StratifiedGroupKFold",
        "shuffle": False,
        "splitter_random_state": None,
        "training_seed_reserved": args.seed,
        "stratification": "original_split x gt_regions quintile x small_text_lt12_ratio binary",
        "grouping": {
            "candidate_rule": "exact SHA or aspect relative difference <= 0.12 and pHash distance <= 8",
            "acceptance_rules": [
                "exact SHA-256",
                "pHash distance <= 2 and GT-region relative difference <= 0.10",
                "pHash distance <= 6, grayscale correlation >= 0.90, and GT-region relative difference <= 0.10",
            ],
            "candidate_pair_count": len(candidates),
            "accepted_pair_count": len(accepted_pairs),
            "multi_image_group_count": validation["multi_image_group_count"],
        },
        "manifest_sha256": manifest_sha256,
        "validation_passed": validation["passed"],
    }
    with args.metadata.open("w") as fp:
        json.dump(metadata, fp, indent=2, ensure_ascii=True)
        fp.write("\n")

    print(json.dumps(metadata, indent=2))
    if not validation["passed"]:
        raise SystemExit("Split validation failed; inspect split_validation.json")


if __name__ == "__main__":
    main()
