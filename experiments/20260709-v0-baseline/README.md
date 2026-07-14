# V0 Baseline

## Purpose

Reproduce the provided DBNet baseline with minimal changes before applying strategy-driven improvements.

## Environment

- Python: 3.10.13
- GPU: NVIDIA GeForce RTX 3090
- PyTorch: 2.1.2+cu118
- wandb: disabled

## Command

```bash
cd /data/ephemeral/home/receipt-text-detection/baseline_code
python runners/train.py \
  preset=example \
  dataset_base_path=/data/ephemeral/home/receipt-text-detection/data/datasets/ \
  exp_name=v0_baseline \
  exp_version=v0 \
  trainer.max_epochs=10 \
  trainer.num_sanity_val_steps=0 \
  +trainer.accelerator=gpu \
  +trainer.devices=1
```

Prediction:

```bash
python runners/predict.py \
  preset=example \
  dataset_base_path=/data/ephemeral/home/receipt-text-detection/data/datasets/ \
  exp_name=v0_baseline_epoch8 \
  minified_json=True \
  'checkpoint_path="outputs/v0_baseline/checkpoints/epoch=8-step=1845.ckpt"'
```

CSV conversion:

```bash
python ocr/utils/convert_submission.py \
  --json_path outputs/v0_baseline_epoch8/submissions/20260709_012836.json \
  --output_path /data/ephemeral/home/receipt-text-detection/submissions/v0_baseline_epoch8_20260709_012836.csv
```

## Selected Checkpoint

`baseline_code/outputs/v0_baseline/checkpoints/epoch=8-step=1845.ckpt`

Reason: best validation H-Mean among logged epochs.

## Validation Metrics

| Epoch | Recall | Precision | H-Mean |
| --- | --- | --- | --- |
| 0 | 0.7124 | 0.8787 | 0.7809 |
| 1 | 0.7165 | 0.9386 | 0.8055 |
| 2 | 0.7686 | 0.9372 | 0.8381 |
| 3 | 0.7777 | 0.9543 | 0.8504 |
| 4 | 0.8001 | 0.9520 | 0.8637 |
| 5 | 0.8335 | 0.9536 | 0.8844 |
| 6 | 0.7007 | 0.9653 | 0.8039 |
| 7 | 0.8283 | 0.9560 | 0.8830 |
| 8 | 0.8371 | 0.9633 | 0.8914 |
| 9 | 0.8106 | 0.9581 | 0.8727 |

Re-tested selected checkpoint:

- H-Mean: 0.8913
- Recall: 0.8369
- Precision: 0.9633

## Leaderboard Result

Extracted from submitted leaderboard screenshot:

- Model name: `v0`
- Created at: 2026.07.09 23:21
- Phase: Complete
- Public H-Mean: 0.8818
- Public Precision: 0.9651
- Public Recall: 0.8194
- Final H-Mean: 0.8898
- Final Precision: 0.9675
- Final Recall: 0.8324

Compared with local selected-checkpoint validation:

- H-Mean gap: -0.0095
- Precision gap: +0.0018
- Recall gap: -0.0175

Interpretation: public precision matches local validation well, but public recall is lower. The next experiment should try to recover recall without making false positives explode.

Final H was `-0.0015` below local and `+0.0080` above Public. The baseline diagnosis remains the
same: precision was already high, while recall was the main weakness. Final evidence:
[2026-07-14 leaderboard](/data/ephemeral/home/receipt-text-detection/docs/leaderboard/20260714/README.md).

## Submission Sanity Check

- CSV rows: 413
- Unique filenames: 413
- Missing test files: 0
- Extra files: 0
- Empty rows: 0
- Max regions per image: 174
- Min regions per image: 38
- Avg regions per image: 101.48
- Images over 500 regions: 0
- Invalid polygons with fewer than 4 points: 0
- Odd coordinate polygons: 0
- Non-finite coordinates: 0

## Artifacts

- Prediction JSON: [20260709_012836.json](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v0_baseline_epoch8/submissions/20260709_012836.json)
- Submission CSV: [v0_baseline_epoch8_20260709_012836.csv](/data/ephemeral/home/receipt-text-detection/submissions/v0_baseline_epoch8_20260709_012836.csv)
- Checkpoint: [epoch=8-step=1845.ckpt](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v0_baseline/checkpoints/epoch=8-step=1845.ckpt)
- Config: [config.yaml](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v0_baseline/.hydra/config.yaml)

## Notes

- No code changes were made for V0.
- Dataset path and GPU trainer options were supplied through Hydra overrides.
- A smoke run with limited test batches failed because baseline `on_test_epoch_end` expects predictions for all 404 validation images. Full training/evaluation was unaffected.
