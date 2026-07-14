# V2B Resolution 1024

## Status

Completed. Resolution 1024 was adopted and achieved Final rank 1 with H/P/R
`0.9647 / 0.9739 / 0.9580`.

## Purpose

V2의 896 해상도 개선 추세가 1024에서도 이어지는지 확인한다. 이 실험은 입력 해상도만
변경하는 controlled sub-experiment이며 Public leaderboard가 아니라 official-val CLEval로
896과 1024를 선택한다.

## Control

- Control: V2 896, official-val H/P/R `0.9615 / 0.9638 / 0.9611`
- Architecture: pretrained ResNet18 + UNet + DBHead
- Optimizer/scheduler: Adam `lr=0.001` + StepLR, V2와 동일
- Epochs/seed: 10 / 42
- Primary post-processing: `box_thresh=0.25`
- Main change: train/validation/test/predict resize and padding `896 -> 1024`

Batch 16이 GPU memory에 맞지 않을 때만 batch 8과 gradient accumulation 2를 사용해
effective batch size를 유지한다.

## Adoption Rule

- 같은 `box_thresh=0.25`의 official-val CLEval을 primary comparison으로 사용한다.
- H-Mean의 미세한 변동만으로 채택하지 않고 precision/recall, epoch 안정성, GPU 비용을
  함께 본다.
- 1024가 의미 있게 개선되지 않으면 폐기하고 실험 4는 V2의 896에서 진행한다.
- Public에는 local에서 채택된 경우에만 제출 후보를 만들며, Public 결과로 해상도 결정을
  뒤집지 않는다.

## Result

Best checkpoint: epoch 8 at `box_thresh=0.25`.

| Resolution | H-Mean | Precision | Recall | Peak GPU memory |
| ---: | ---: | ---: | ---: | ---: |
| 896 control | 0.961507 | 0.963823 | 0.961134 | about 17.4 GB |
| 1024 | **0.964760** | **0.969976** | **0.961422** | about 22.5 GB |

Delta versus 896: H `+0.003253`, precision `+0.006153`, recall `+0.000288`.
The gain came mainly from precision while recall was maintained. The independent checkpoint reload
was used for the adoption decision. Its H-Mean differed from the in-training epoch-8 value
`0.964890` by only `0.000130`.

## Epoch History

| Epoch | H-Mean | Precision | Recall |
| ---: | ---: | ---: | ---: |
| 0 | 0.885184 | 0.919067 | 0.869104 |
| 1 | 0.928564 | 0.936477 | 0.926954 |
| 2 | 0.945818 | 0.952450 | 0.942941 |
| 3 | 0.949809 | 0.946747 | 0.955621 |
| 4 | 0.957084 | 0.958881 | 0.957373 |
| 5 | 0.957131 | 0.958195 | 0.958154 |
| 6 | 0.953540 | 0.956971 | 0.953362 |
| 7 | 0.958017 | 0.963490 | 0.954440 |
| 8 | **0.964890** | **0.970040** | **0.961622** |
| 9 | 0.963708 | 0.965813 | 0.963160 |

## Artifacts

- Best checkpoint: [epoch=8-step=1845.ckpt](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v2b_resolution1024/checkpoints/epoch=8-step=1845.ckpt)
- Config: [config.yaml](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v2b_resolution1024/.hydra/config.yaml)
- W&B training run: [fq4fn7jx](https://wandb.ai/pep-per/receipt-text-detection/runs/fq4fn7jx)
- W&B independent evaluation: [6pifubi5](https://wandb.ai/pep-per/receipt-text-detection/runs/6pifubi5)
- Prediction JSON: [20260714_001730.json](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v2b_resolution1024_epoch8/submissions/20260714_001730.json)
- Submission CSV: [v2b_resolution1024_epoch8_20260714_001730.csv](/data/ephemeral/home/receipt-text-detection/submissions/v2b_resolution1024_epoch8_20260714_001730.csv)

The W&B training runtime including the final test pass was 2,107 seconds, about 35 minutes 7
seconds. Batch 16 fit without gradient accumulation.

## Submission Sanity Check

- JSON/CSV/expected test rows: `413 / 413 / 413`
- Missing/extra filenames: `0 / 0`
- Total predicted regions: `45,236`
- Minimum/maximum/average regions per image: `39 / 226 / 109.53`
- Empty images and images over 500 regions: `0 / 0`
- Invalid point count, non-finite coordinate, zero area, invalid geometry: all `0`
- Invalid CSV coordinate sets: `0`
- Boundary overshoot after EXIF orientation handling: 25 polygons, at most 6 pixels
- Cap or coordinate clipping applied: no

## Public Leaderboard Result

| Evaluation | H-Mean | Precision | Recall |
| --- | ---: | ---: | ---: |
| Local official val | 0.964760 | 0.969976 | 0.961422 |
| Public leaderboard | 0.9621 | 0.9754 | 0.9520 |
| Final leaderboard | **0.9647** | **0.9739** | **0.9580** |
| Public - local | -0.002660 | +0.005424 | -0.009422 |
| Final - local | -0.000060 | +0.003924 | -0.003422 |

Public H-Mean improved by `+0.0018` over V2 Public (`0.9603`). Precision increased by `+0.0087`
while recall decreased by `-0.0036`. Therefore, the local-selected 1024 direction transferred to
Public, but its gain was smaller and primarily precision-driven.

Higher resolution may produce cleaner probability-map boundaries for small text and reduce
false-positive character areas. The weaker Public recall may reflect a different proportion of
faint, blurred, or distorted images in the roughly half-sized Public subset. Public per-image labels
are unavailable, so these are plausible explanations rather than confirmed causes.

The local evaluator separately averages per-image H-Mean, precision, and recall. For the Public
values, the harmonic mean of displayed precision and recall is `0.9636`, not the displayed H-Mean
`0.9621`. Four-decimal rounding alone is too small to explain this gap, so separate aggregation or
another server-side weighting rule is likely. The screenshot alone does not reveal the exact
server implementation.

Public submission was not used to select 896 versus 1024. The candidate was generated after the
clean local decision, and Experiment 4 proceeded from 1024 regardless of Public feedback. V2B
remains the best clean single model. The next model-side experiment targets camera degradation and
faint-text recall; threshold recalibration remains a later local-only step.

Final H improved `+0.0010` over V2 Final. Precision improved `+0.0057` while recall fell `-0.0026`,
so the 1024 gain remained precision-driven and smaller than the local gain. The local-selected
direction nevertheless transferred and V2B placed first overall, `+0.0195` H above rank 2. Final
evidence:
[2026-07-14 leaderboard](/data/ephemeral/home/receipt-text-detection/docs/leaderboard/20260714/README.md).
