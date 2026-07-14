# V2 Resolution 896

## Status

Completed. Training, best-checkpoint validation, test prediction, CSV conversion, sanity checks,
Public submission, and Final leaderboard evaluation are complete.

The follow-up order is maintained in the
[Clean-data Experiment Roadmap](../clean_data_experiment_roadmap.md).

Before training, checkpoint retention should save the top models by `val/hmean` and the final model.
This changes which artifacts are retained, not the model's gradient updates, and prevents the best
competition-metric epoch from being discarded by the current `val/loss`-only policy.

## Interpretation Correction

V1 showed that post-processing had a large effect **for the fixed V0 checkpoint**. It did not
compare post-processing against a newly trained model. Therefore, the evidence supports the narrow
statement below:

> The V0 checkpoint was under-recalled at the baseline `box_thresh=0.40`, and changing only that
> threshold was sufficient to produce a large local and Public improvement.

It does not yet support the broader claim that post-processing matters more than model training.
Experiment 3 will provide the first training-side A/B comparison.

## What `box_thresh` Does

DBNet produces a pixel-level text probability map. The current post-processor works in two main
threshold stages:

1. `thresh=0.30` converts the pixel probability map into a binary bitmap used to find contours.
2. For each contour, the code computes the mean probability inside its candidate region.
   `box_thresh` discards the candidate when this score is below the configured value.

Lower `box_thresh` values retain lower-confidence text candidates. This tends to improve recall but
can add false positives from barcodes, lines, logos, shadows, or background texture.

## Threshold Runs Already Completed

| Experiment | Checkpoint | Validation thresholds | Test submission threshold | Retraining |
| --- | --- | --- | ---: | --- |
| V0 | `epoch=8-step=1845.ckpt` | `0.40` | `0.40` | Yes, baseline training |
| V1 | Same V0 checkpoint | `0.35`, `0.30`, `0.25`, `0.20` | `0.25` | No |

The four V1 threshold values were separate official-validation inference/evaluation runs. They were
not separate model-training runs. The `0.40` row in the sweep table came from the existing V0
validation result.

## Evidence From V1

At the same checkpoint:

| Comparison | H-Mean change | Precision change | Recall change |
| --- | ---: | ---: | ---: |
| Local, `0.40` to `0.25` | +0.0335 | -0.0134 | +0.0688 |
| Public, V0 to V1 | +0.0367 | -0.0140 | +0.0738 |

The similar local and Public direction makes `0.25` a reasonable current default. However, it is
still based on one checkpoint and one official validation split. It is not evidence about Private
performance by itself.

## Plan Change

Before V1, the next training experiment was described broadly as augmentation, training/data
changes, or error analysis. After V1, the plan is narrowed to a controlled resolution experiment.

Changed plan:

- Increase `LongestMaxSize` and padding size from `640` to `896` consistently for train,
  validation, test, and prediction transforms.
- Keep the DBNet architecture, official train/validation split, optimizer, epoch count, seed, and
  loss unchanged.
- Reduce physical batch size if required by GPU memory and use gradient accumulation to preserve
  the V0 effective batch size as closely as possible.
- Evaluate the new checkpoint first with `box_thresh=0.25`.
- Inspect `box_thresh=0.30` on local validation only if the new model's confidence calibration or
  precision changes materially.
- Do not combine new augmentation policies with the resolution change in this experiment.

## Why Resolution Is Tested First

The baseline resizes every image so its longest side is at most 640 pixels. Annotation geometry
after that scale shows:

| Split | Median region short side | Regions below 8 px | Regions below 12 px |
| --- | ---: | ---: | ---: |
| Train | 10.5 px | 23.0% | 67.1% |
| Official validation | 10.7 px | 22.4% | 65.3% |

This makes small-text information loss a plausible remaining recall bottleneck. Lowering
`box_thresh` can retain a weak proposal that already exists, but it cannot recover a word that
disappeared from the probability map after downscaling.

This is a hypothesis, not a conclusion. A 896 input may also increase false positives, memory use,
training time, and inference time. Those costs and metrics must be measured.

## Control And Adoption Criteria

Control result: V1 official validation at `box_thresh=0.25`:

- H-Mean: `0.9248`
- Precision: `0.9499`
- Recall: `0.9057`

V2 is adopted only when:

- Its official-validation H-Mean improves at the same `box_thresh=0.25`.
- Precision/recall changes are explainable and prediction validity checks pass.
- No image exceeds 500 regions and prediction counts do not show a dangerous false-positive jump.
- The resource cost is acceptable relative to the measured gain.

Only one locally selected V2 candidate should be submitted to the Public leaderboard. Repeatedly
choosing resolution or thresholds from Public scores would increase Private overfitting risk.

## Metrics To Record

- Best checkpoint and epoch
- Official-validation CLEval H-Mean, precision, and recall
- Per-image prediction-count distribution and maximum
- Invalid polygon and 500-region-cap checks
- Peak GPU memory
- Training wall time and test inference time
- wandb run and exact Hydra config
- Public result for the single selected submission

## Execution

- Date: 2026-07-12 to 2026-07-13 KST
- GPU: NVIDIA RTX 3090 24 GB
- Peak observed GPU memory: about 17.4 GB
- Input size: 896 for train/validation/test/prediction
- Physical batch size: 16; no gradient accumulation was needed
- Epochs: 10
- Seed: 42
- Architecture: pretrained ResNet18 + UNet + DBHead
- Optimizer/scheduler: Adam `lr=0.001` + unchanged StepLR
- Primary post-processing: `box_thresh=0.25`
- Training wall time recorded by wandb: 1,763 seconds, about 29 minutes 23 seconds

Checkpoint retention was changed from top-3 `val/loss` to top-3 `val/hmean` plus the final
checkpoint in [train.py](/data/ephemeral/home/receipt-text-detection/baseline_code/runners/train.py).
This affected artifact selection only, not gradient updates.

## Epoch Results

Official validation, detection-only CLEval POLY at `box_thresh=0.25`:

| Epoch | H-Mean | Precision | Recall |
| ---: | ---: | ---: | ---: |
| 0 | 0.8846 | 0.8716 | 0.9038 |
| 1 | 0.9254 | 0.9227 | 0.9322 |
| 2 | 0.9413 | 0.9392 | 0.9460 |
| 3 | 0.9475 | 0.9461 | 0.9519 |
| 4 | 0.9528 | 0.9547 | 0.9530 |
| 5 | 0.9559 | 0.9628 | 0.9515 |
| 6 | 0.9525 | 0.9527 | 0.9543 |
| 7 | 0.9524 | 0.9610 | 0.9459 |
| 8 | **0.9615** | **0.9638** | **0.9611** |
| 9 | 0.9603 | 0.9605 | 0.9618 |

Epoch 8 was selected. An independent `runners/test.py` reload produced:

- H-Mean: `0.9615069`
- Precision: `0.9638232`
- Recall: `0.9611344`

Compared with the V1 control at the same `box_thresh=0.25`:

- H-Mean: `+0.0367`
- Precision: `+0.0139`
- Recall: `+0.0555`

This is the first trained-model A/B result and supports adopting 896 as the current best clean
single-model resolution. Both precision and recall improved, so the gain is not explained by a
more aggressive precision/recall trade-off.

## Wandb Offline Runs

- Training: [dzb2awlm](/data/ephemeral/home/receipt-text-detection/baseline_code/wandb/offline-run-20260712_235204-dzb2awlm)
- Best-checkpoint evaluation: [4qdc8lei](/data/ephemeral/home/receipt-text-detection/baseline_code/wandb/offline-run-20260713_002216-4qdc8lei)

After wandb login:

```bash
cd /data/ephemeral/home/receipt-text-detection/baseline_code
wandb sync -p receipt-text-detection \
  wandb/offline-run-20260712_235204-dzb2awlm \
  wandb/offline-run-20260713_002216-4qdc8lei
```

## Artifacts

- Best checkpoint: [epoch=8-step=1845.ckpt](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v2_resolution896/checkpoints/epoch=8-step=1845.ckpt)
- Training config: [config.yaml](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v2_resolution896/.hydra/config.yaml)
- Training overrides: [overrides.yaml](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v2_resolution896/.hydra/overrides.yaml)
- Prediction config: [config.yaml](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v2_resolution896_epoch8/.hydra/config.yaml)
- Prediction JSON: [20260713_002412.json](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v2_resolution896_epoch8/submissions/20260713_002412.json)
- Submission CSV: [v2_resolution896_epoch8_20260713_002412.csv](/data/ephemeral/home/receipt-text-detection/submissions/v2_resolution896_epoch8_20260713_002412.csv)

## Submission Sanity Check

- CSV rows: 413
- Expected test files: 413
- Missing/extra files: 0 / 0
- Empty prediction images: 0
- Invalid point-count/non-finite/zero-area/self-intersecting polygons: 0
- Images over 500 regions: 0
- Maximum regions per image: 215
- Minimum regions per image: 39
- Average regions per image: 108.96
- Total predicted regions: 45,002
- Boundary overshoot after EXIF orientation handling: 25 polygons, at most 8 pixels
- Cap or coordinate clipping applied: no

The small boundary overshoot also existed in V1 and does not create invalid geometry in the local
CLEval path. Predictions were left unchanged to preserve the controlled baseline comparison.

## Per-image Error Analysis Artifacts

- Official-val prediction JSON: [20260713_003015.json](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v2_resolution896_epoch8_val_predictions/submissions/20260713_003015.json)
- Per-image CLEval table: [per_image_cleval.csv](/data/ephemeral/home/receipt-text-detection/experiments/20260712-v2-resolution-896/per_image_cleval.csv)
- Lowest-H visualizations: [visualizations](/data/ephemeral/home/receipt-text-detection/experiments/20260712-v2-resolution-896/visualizations)

Lowest five official-val images:

| Filename | H-Mean | Precision | Recall | GT / Pred regions |
| --- | ---: | ---: | ---: | ---: |
| `selectstar_001418.jpg` | 0.6740 | 0.5591 | 0.8483 | 220 / 83 |
| `selectstar_000530.jpg` | 0.7470 | 0.8483 | 0.6673 | 167 / 148 |
| `selectstar_000422.jpg` | 0.7878 | 0.6558 | 0.9864 | 54 / 63 |
| `selectstar_002135.jpg` | 0.7879 | 0.7188 | 0.8717 | 276 / 160 |
| `selectstar_000905.jpg` | 0.7907 | 0.8601 | 0.7316 | 143 / 138 |

Initial visual observations:

- Very long, narrow, dense receipts still show merged neighboring regions and granularity errors.
- Faint or low-contrast text areas account for substantial misses in several low-recall images.
- Barcodes, separators, and compact non-text patterns remain false-positive candidates.

These are hypotheses for later augmentation and post-processing experiments, not rules selected
from five images. No V2 prediction was changed from this analysis.

## Decision And Next Step

Decision: **Keep** 896 as the current best clean single-model setting.

The result passes the roadmap gate for an optional 1024 resolution sub-experiment because 896
produced a clear gain and still fit batch 16 on the RTX 3090. Do not select 896 versus 1024 from
Public scores. If the 1024 branch is skipped, proceed to Experiment 4 cosine LR scheduling at 896.

Only the locally selected V2 CSV should be submitted for the 896 milestone. Record its Public
result without changing the selected threshold from leaderboard feedback.

## Public Leaderboard Result

Screenshot values:

- Model name: `v2`
- Created at: 2026.07.13 05:31
- Phase: Complete
- H-Mean: `0.9603`
- Precision: `0.9667`
- Recall: `0.9556`
- Final H/P/R: `0.9637 / 0.9682 / 0.9606`

Local-to-Public gap:

| Metric | Local | Public | Public - Local |
| --- | ---: | ---: | ---: |
| H-Mean | 0.9615 | 0.9603 | -0.0012 |
| Precision | 0.9638 | 0.9667 | +0.0029 |
| Recall | 0.9611 | 0.9556 | -0.0055 |

Public improvement over V1 was H `+0.0418`, precision `+0.0156`, and recall `+0.0624`.

Final improvement over V1 was H `+0.0416`, precision `+0.0128`, and recall `+0.0628`. Final H was
`+0.0022` above local and `+0.0034` above Public. Resolution 896 produced the largest Final gain in
the submitted sequence. Final evidence:
[2026-07-14 leaderboard](/data/ephemeral/home/receipt-text-detection/docs/leaderboard/20260714/README.md).

The Public score exceeded the rough pre-submission estimate, and its H-Mean is very close to the
local result. Because model and threshold selection happened before seeing this Public score, this
result was evidence of transfer rather than Public-driven selection. The later Final result
confirmed the direction and slightly exceeded local H. Repeated official-val tuning remains a risk
for future experiments even though this submitted sequence transferred successfully.
