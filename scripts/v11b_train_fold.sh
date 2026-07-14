#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ! "$1" =~ ^[0-4]$ ]]; then
  echo "Usage: $0 FOLD_ID(0-4)" >&2
  exit 2
fi

FOLD="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASELINE_ROOT="$PROJECT_ROOT/baseline_code"
FOLD_DATA="$PROJECT_ROOT/data/v11_folds/fold_$FOLD"
ANNOTATIONS="$PROJECT_ROOT/data/v11_folds/all_annotations.json"
EXP_NAME="v11b_fold$FOLD"
OUTPUT_ROOT="$BASELINE_ROOT/outputs/$EXP_NAME"
FIXED_CHECKPOINT="$OUTPUT_ROOT/checkpoints/fixed-epoch=8.ckpt"

for required in \
  "$ANNOTATIONS" \
  "$FOLD_DATA/train_images" \
  "$FOLD_DATA/val_images"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing prepared fold data: $required" >&2
    echo "Run: python scripts/v11b_prepare_fold_data.py" >&2
    exit 1
  fi
done

if [[ -s "$FIXED_CHECKPOINT" && "${V11B_FORCE:-0}" != "1" ]]; then
  echo "Fold $FOLD fixed checkpoint already exists: $FIXED_CHECKPOINT"
  exit 0
fi

mkdir -p "$OUTPUT_ROOT"
cd "$BASELINE_ROOT"

python runners/train.py \
  preset=example \
  "dataset_base_path=$PROJECT_ROOT/data/datasets/" \
  "exp_name=$EXP_NAME" \
  project_name=receipt-text-detection \
  wandb=true \
  "exp_version=v11b-fold$FOLD" \
  trainer.max_epochs=10 \
  trainer.num_sanity_val_steps=0 \
  trainer.check_val_every_n_epoch=9 \
  +trainer.accelerator=gpu \
  +trainer.devices=1 \
  +fixed_checkpoint_epoch=8 \
  +checkpoint_save_top_k=1 \
  +checkpoint_save_last=false \
  +test_after_fit=false \
  "datasets.train_dataset.image_path=$FOLD_DATA/train_images" \
  "datasets.train_dataset.annotation_path=$ANNOTATIONS" \
  "datasets.val_dataset.image_path=$FOLD_DATA/val_images" \
  "datasets.val_dataset.annotation_path=$ANNOTATIONS" \
  "datasets.test_dataset.image_path=$FOLD_DATA/val_images" \
  "datasets.test_dataset.annotation_path=$ANNOTATIONS" \
  models.head.postprocess.box_thresh=0.25 \
  transforms.train_transform.transforms.0.max_size=1024 \
  transforms.train_transform.transforms.1.min_width=1024 \
  transforms.train_transform.transforms.1.min_height=1024 \
  transforms.val_transform.transforms.0.max_size=1024 \
  transforms.val_transform.transforms.1.min_width=1024 \
  transforms.val_transform.transforms.1.min_height=1024 \
  transforms.test_transform.transforms.0.max_size=1024 \
  transforms.test_transform.transforms.1.min_width=1024 \
  transforms.test_transform.transforms.1.min_height=1024 \
  transforms.predict_transform.transforms.0.max_size=1024 \
  transforms.predict_transform.transforms.1.min_width=1024 \
  transforms.predict_transform.transforms.1.min_height=1024 \
  2>&1 | tee "$OUTPUT_ROOT/train_stdout.log"

if [[ ! -s "$FIXED_CHECKPOINT" ]]; then
  echo "Training finished without the required fixed epoch checkpoint" >&2
  exit 1
fi

echo "V11B fold $FOLD training complete: $FIXED_CHECKPOINT"
