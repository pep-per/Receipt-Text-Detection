#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECK_DATA=1

if [[ "${1:-}" == "--code-only" ]]; then
  CHECK_DATA=0
elif [[ "$#" -ne 0 ]]; then
  echo "Usage: $0 [--code-only]" >&2
  exit 2
fi

required_files=(
  "baseline_code/requirements.txt"
  "baseline_code/runners/train.py"
  "baseline_code/runners/test.py"
  "baseline_code/runners/predict.py"
  "baseline_code/outputs/v2b_resolution1024/checkpoints/epoch=8-step=1845.ckpt"
  "baseline_code/outputs/v5_resnet34_1024/checkpoints/epoch=7-step=1640.ckpt"
  "baseline_code/outputs/v10_ssl_moco/encoder_state_dict.pt"
  "baseline_code/outputs/v10_ssl_moco_finetune/checkpoints/epoch=8-step=1845.ckpt"
)

missing=0
for relative_path in "${required_files[@]}"; do
  if [[ ! -s "$PROJECT_ROOT/$relative_path" ]]; then
    echo "MISSING: $relative_path" >&2
    missing=1
  fi
done

if [[ "$CHECK_DATA" -eq 1 ]]; then
  for json_name in train val test; do
    json_path="$PROJECT_ROOT/data/datasets/jsons/$json_name.json"
    if [[ ! -s "$json_path" ]]; then
      echo "MISSING: data/datasets/jsons/$json_name.json" >&2
      missing=1
    fi
  done

  for split in train val test; do
    image_dir="$PROJECT_ROOT/data/datasets/images/$split"
    if [[ ! -d "$image_dir" ]]; then
      echo "MISSING: data/datasets/images/$split" >&2
      missing=1
      continue
    fi
    count="$(find "$image_dir" -maxdepth 1 -type f | wc -l)"
    echo "$split images: $count"
  done

  for dataset in cord-v2 sroie wildreceipt; do
    pseudo_dir="$PROJECT_ROOT/data/pseudo_label/$dataset"
    if [[ ! -d "$pseudo_dir" ]]; then
      echo "MISSING: data/pseudo_label/$dataset" >&2
      missing=1
    else
      echo "pseudo-label candidate: $dataset"
    fi
  done
else
  echo "Data checks skipped (--code-only). Restore data/ before training or inference."
fi

if [[ "$missing" -ne 0 ]]; then
  echo "Resume asset verification failed." >&2
  exit 1
fi

echo "Required resume assets are present."
