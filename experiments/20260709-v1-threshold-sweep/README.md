# V1 Threshold Sweep

## Purpose

Improve V0 recall without retraining by tuning DBNet postprocess `box_thresh`.

V0 public leaderboard showed high precision but weaker recall, so this experiment keeps the
same checkpoint and searches lower thresholds on the official validation split.

## Base Artifact

- Base checkpoint: [epoch=8-step=1845.ckpt](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v0_baseline/checkpoints/epoch=8-step=1845.ckpt)
- Base config: [config.yaml](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v0_baseline/.hydra/config.yaml)
- Dataset: `/data/ephemeral/home/receipt-text-detection/data/datasets/`
- wandb: offline mode, because no API key/account was configured in the environment

## Sweep Results

Official validation, detection-only CLEval POLY:

| box_thresh | H-Mean | Precision | Recall | wandb Offline Run | Decision |
| ---: | ---: | ---: | ---: | --- | --- |
| 0.40 | 0.8913 | 0.9633 | 0.8369 | N/A | V0 baseline |
| 0.35 | 0.9153 | 0.9571 | 0.8823 | [obdki529](/data/ephemeral/home/receipt-text-detection/baseline_code/wandb/offline-run-20260709_234441-obdki529) | Improved recall |
| 0.30 | 0.9230 | 0.9528 | 0.8997 | [0ndp9r73](/data/ephemeral/home/receipt-text-detection/baseline_code/wandb/offline-run-20260709_234617-0ndp9r73) | Strong candidate |
| 0.25 | 0.9248 | 0.9499 | 0.9057 | [orcmqhxn](/data/ephemeral/home/receipt-text-detection/baseline_code/wandb/offline-run-20260709_234751-orcmqhxn) | Selected |
| 0.20 | 0.9249 | 0.9482 | 0.9075 | [5jlik5pa](/data/ephemeral/home/receipt-text-detection/baseline_code/wandb/offline-run-20260709_234924-5jlik5pa) | Not selected |

## Selection

`box_thresh=0.25` was selected.

`box_thresh=0.20` had the highest local H-Mean by only 0.0001, but precision was lower. Since
the private split may contain more unseen clutter, backgrounds, and barcode-like false positives,
the near-tie should not be treated as strong evidence for the lower threshold.

## Commands

Validation sweep example:

```bash
cd /data/ephemeral/home/receipt-text-detection/baseline_code
WANDB_MODE=offline python runners/test.py \
  preset=example \
  dataset_base_path=/data/ephemeral/home/receipt-text-detection/data/datasets/ \
  exp_name=v1_box025_eval \
  project_name=receipt-text-detection \
  wandb=True \
  'checkpoint_path="outputs/v0_baseline/checkpoints/epoch=8-step=1845.ckpt"' \
  models.head.postprocess.box_thresh=0.25
```

Prediction:

```bash
cd /data/ephemeral/home/receipt-text-detection/baseline_code
WANDB_MODE=offline python runners/predict.py \
  preset=example \
  dataset_base_path=/data/ephemeral/home/receipt-text-detection/data/datasets/ \
  exp_name=v1_box025_epoch8 \
  minified_json=True \
  'checkpoint_path="outputs/v0_baseline/checkpoints/epoch=8-step=1845.ckpt"' \
  models.head.postprocess.box_thresh=0.25
```

CSV conversion:

```bash
python ocr/utils/convert_submission.py \
  --json_path outputs/v1_box025_epoch8/submissions/20260709_235118.json \
  --output_path /data/ephemeral/home/receipt-text-detection/submissions/v1_box025_epoch8_20260709_235118.csv
```

## Submission Sanity Check

- CSV rows: 413
- Expected test files: 413
- Missing test files: 0
- Extra files: 0
- Empty rows: 0
- Invalid polygons: 0
- Images over 500 regions: 0
- Max regions per image: 184
- Min regions per image: 38
- Avg regions per image: 104.20

## Leaderboard Result

Extracted from submitted leaderboard screenshot:

- Model name: `v1`
- Created at: 2026.07.09 23:54
- Phase: Complete
- Public H-Mean: 0.9185
- Public Precision: 0.9511
- Public Recall: 0.8932
- Final H-Mean: 0.9221
- Final Precision: 0.9554
- Final Recall: 0.8978

Compared with local official validation:

- Public H-Mean - Local H-Mean: -0.0063
- Public Precision - Local Precision: +0.0012
- Public Recall - Local Recall: -0.0125

Compared with V0 public leaderboard:

- H-Mean: +0.0367
- Precision: -0.0140
- Recall: +0.0738

Compared with V0 Final, V1 improved H by `+0.0323` and recall by `+0.0654`, while precision fell
`-0.0121`. The locally selected threshold change therefore transferred strongly to the hidden Final
split. Final evidence:
[2026-07-14 leaderboard](/data/ephemeral/home/receipt-text-detection/docs/leaderboard/20260714/README.md).

## Artifacts

- Prediction JSON: [20260709_235118.json](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v1_box025_epoch8/submissions/20260709_235118.json)
- Submission CSV: [v1_box025_epoch8_20260709_235118.csv](/data/ephemeral/home/receipt-text-detection/submissions/v1_box025_epoch8_20260709_235118.csv)
- Prediction config: [config.yaml](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v1_box025_epoch8/.hydra/config.yaml)
- Prediction overrides: [overrides.yaml](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v1_box025_epoch8/.hydra/overrides.yaml)

## Strategy Note

This experiment supports a recall-oriented postprocess adjustment. Public precision stayed near
0.95 and recall improved strongly, so `box_thresh=0.25` should be the current default for the next
experiment.

Do not keep tuning thresholds against public submissions alone. The next useful step should come
from training/data/augmentation changes or visual error analysis, with `box_thresh=0.25` applied
unless the new model creates many false positives.

To sync wandb logs after configuring an account:

```bash
cd /data/ephemeral/home/receipt-text-detection/baseline_code
wandb sync wandb/offline-run-20260709_234441-obdki529
wandb sync wandb/offline-run-20260709_234617-0ndp9r73
wandb sync wandb/offline-run-20260709_234751-orcmqhxn
wandb sync wandb/offline-run-20260709_234924-5jlik5pa
```
