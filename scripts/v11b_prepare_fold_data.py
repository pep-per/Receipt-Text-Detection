#!/usr/bin/env python3
"""Prepare symlinked image views and a combined annotation for V11B folds."""

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/splits/v11_5fold_seed42.csv",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data/datasets",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data/v11_folds",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_combined_annotations(dataset_root):
    combined = OrderedDict()
    source_by_filename = {}
    for split in ("train", "val"):
        with (dataset_root / "jsons" / f"{split}.json").open() as handle:
            annotations = json.load(handle)["images"]
        for filename, item in annotations.items():
            if filename in combined:
                raise RuntimeError(f"Duplicate annotation filename: {filename}")
            combined[filename] = item
            source_by_filename[filename] = split
    return combined, source_by_filename


def replace_symlink(link_path, target_path):
    if link_path.is_symlink():
        if link_path.resolve() == target_path.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        raise RuntimeError(f"Expected a symlink, found another file: {link_path}")
    link_path.symlink_to(target_path.resolve())


def sync_image_view(directory, filenames, source_by_filename, dataset_root):
    directory.mkdir(parents=True, exist_ok=True)
    expected = set(filenames)
    for existing in directory.iterdir():
        if existing.name not in expected:
            if existing.is_symlink():
                existing.unlink()
            else:
                raise RuntimeError(f"Unexpected non-symlink in image view: {existing}")
    for filename in filenames:
        source = dataset_root / "images" / source_by_filename[filename] / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        replace_symlink(directory / filename, source)


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest).sort_values(["numeric_id", "filename"])
    if len(manifest) != 3676 or not manifest["filename"].is_unique:
        raise RuntimeError("V11 manifest must contain 3,676 unique filenames")
    if set(manifest["fold"]) != set(range(5)):
        raise RuntimeError("V11 manifest must contain fold IDs 0 through 4")

    annotations, source_by_filename = load_combined_annotations(args.dataset_root)
    if set(manifest["filename"]) != set(annotations):
        raise RuntimeError("Manifest and combined annotations do not match")

    args.output_root.mkdir(parents=True, exist_ok=True)
    annotation_path = args.output_root / "all_annotations.json"
    with annotation_path.open("w") as handle:
        json.dump(
            {"images": annotations},
            handle,
            ensure_ascii=True,
            separators=(",", ":"),
        )

    fold_records = []
    for fold in range(5):
        fold_root = args.output_root / f"fold_{fold}"
        validation = manifest[manifest["fold"] == fold]
        training = manifest[manifest["fold"] != fold]
        if set(training["filename"]) & set(validation["filename"]):
            raise RuntimeError(f"Fold {fold} has train/validation overlap")
        if len(training) + len(validation) != len(manifest):
            raise RuntimeError(f"Fold {fold} does not cover the manifest")

        sync_image_view(
            fold_root / "train_images",
            training["filename"].tolist(),
            source_by_filename,
            args.dataset_root,
        )
        sync_image_view(
            fold_root / "val_images",
            validation["filename"].tolist(),
            source_by_filename,
            args.dataset_root,
        )
        fold_records.append(
            {
                "fold": fold,
                "train_images": len(training),
                "val_images": len(validation),
                "train_original_train": int(
                    (training["original_split"] == "train").sum()
                ),
                "train_original_val": int(
                    (training["original_split"] == "val").sum()
                ),
                "val_original_train": int(
                    (validation["original_split"] == "train").sum()
                ),
                "val_original_val": int(
                    (validation["original_split"] == "val").sum()
                ),
            }
        )

    metadata = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "combined_annotation": str(annotation_path.resolve()),
        "combined_annotation_sha256": sha256_file(annotation_path),
        "images": len(manifest),
        "folds": fold_records,
        "image_views": "absolute symlinks; regenerate after moving the project",
    }
    with (args.output_root / "prepared_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
