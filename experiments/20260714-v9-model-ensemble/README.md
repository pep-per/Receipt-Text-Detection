# V9 Existing-model Probability-map Ensemble

## Status

Completed. Equal-weight fusion passed the pre-registered local gate and was adopted as a model
ensemble candidate. The V2B-heavy fallback was not run. The competition is closed, so the test
artifact has no leaderboard score.

## Purpose

V2B ResNet18의 높은 precision과 V5 ResNet34의 높은 recall이 probability-map 수준에서 실제로
보완되는지 확인한다. V8의 scale TTA는 섞지 않아 architecture diversity 효과만 측정한다.

## Control And Candidate

- Control: V2B epoch 8, 1024 single probability map
- Equal candidate: `0.50 * V2B + 0.50 * V5 epoch 7`
- Pre-registered fallback: `0.75 * V2B + 0.25 * V5`
- Fixed: input 1024, `thresh=0.30`, `box_thresh=0.25`, official val 404장, seed 42
- Changed: second model과 probability-map fusion만
- Excluded: V8 scale TTA, polygon union, threshold sweep, V4/V3 추가

## Pre-registered Gate

1. Equal candidate의 macro/global H가 control보다 모두 높고, paired bootstrap
   `P(delta > 0)`가 각각 0.95 이상이면 채택한다.
2. Equal candidate가 recall은 높이지만 precision 손실 때문에 H가 낮아질 때만 고정된
   V2B-heavy `0.75/0.25` fallback을 한 번 평가한다.
3. Fallback도 같은 gate를 적용한다. 동률이면 단일 V2B를 유지한다.
4. 통과한 후보만 test JSON/CSV를 `Generated offline, competition closed`로 만든다.
5. Equal candidate가 통과하면 fallback을 실행하지 않는다.

## Command

```bash
cd /data/ephemeral/home/receipt-text-detection

python scripts/v9_model_ensemble.py \
  --split val \
  --v2b-weight 0.5 \
  --batch-size 4 \
  --workers 4 \
  --wandb-mode online \
  --run-name v9_v2b_v5_equal_eval
```

## Artifacts

- Implementation: [v9_model_ensemble.py](../../scripts/v9_model_ensemble.py)
- Equal metrics: [metrics_w050.json](metrics_w050.json)
- Paired results: [per_image_val_w050.csv](per_image_val_w050.csv)
- Bootstrap: [bootstrap_paired_w050.csv](bootstrap_paired_w050.csv)
- D0 disagreement strata: [strata_metrics_w050.csv](strata_metrics_w050.csv)
- Validation config: [run_config_val_w050.json](run_config_val_w050.json)
- Test summary: [test_prediction_summary_w050.json](test_prediction_summary_w050.json)
- Submission validation: [submission_validation_w050.json](submission_validation_w050.json)
- Test JSON: `baseline_code/outputs/v9_v2b_v5_equal/submissions/20260714_170846.json`
- Offline CSV: `submissions/v9_v2b_v5_equal_20260714_170846.csv`
- W&B validation: [85wojhfj](https://wandb.ai/pep-per/receipt-text-detection/runs/85wojhfj)
- W&B test prediction: [996k08wi](https://wandb.ai/pep-per/receipt-text-detection/runs/996k08wi)

## Result

| Evaluation | H-Mean | Precision | Recall |
|---|---:|---:|---:|
| V2B control macro | 0.964785 | 0.969979 | 0.961463 |
| V2B+V5 equal macro | **0.967266** | 0.969083 | **0.967177** |
| Macro delta | **+0.002481** | -0.000896 | **+0.005714** |
| V2B control global | 0.962249 | 0.967516 | 0.957039 |
| V2B+V5 equal global | **0.965090** | 0.965880 | **0.964301** |
| Global delta | **+0.002841** | -0.001636 | **+0.007262** |

V5가 더 많은 true character area를 복구하면서 recall을 크게 높였고 precision은 소폭 낮췄다.
하지만 두 aggregation에서 H가 모두 상승해 단순히 V5의 낮은 precision을 평균낸 결과는 아니다.

## Paired Bootstrap

| Metric delta | Mean | 95% CI | P(delta > 0) |
|---|---:|---:|---:|
| Macro H | +0.002487 | [+0.000890, +0.004150] | 0.9998 |
| Macro P | -0.000891 | [-0.002366, +0.000590] | 0.1172 |
| Macro R | +0.005719 | [+0.003298, +0.008295] | 1.0000 |
| Global H | +0.002850 | [+0.000886, +0.004929] | 0.9990 |
| Global P | -0.001623 | [-0.003283, +0.000016] | 0.0259 |
| Global R | +0.007266 | [+0.004014, +0.010823] | 1.0000 |

Macro/global H 개선 확률이 사전 기준 0.95를 넘고 두 신뢰구간 하한이 양수다. Equal candidate가
H를 낮추지 않았으므로 fallback 실행 조건이 성립하지 않는다. 결과를 본 뒤 `0.75/0.25`를
추가하면 official val weight fitting이 되므로 실행하지 않았다.

## Diversity Analysis

| Validation stratum | Images | Control H | Ensemble H | Delta H |
|---|---:|---:|---:|---:|
| All | 404 | 0.964785 | 0.967266 | +0.002481 |
| Control H bottom quartile | 101 | 0.904428 | 0.914977 | +0.010550 |
| V5 won in D0 | 166 | 0.951827 | 0.962812 | +0.010985 |
| V2B won in D0 | 220 | 0.971901 | 0.968155 | -0.003746 |
| High-disagreement quartile | 101 | 0.935155 | 0.939911 | +0.004756 |

V5가 원래 이기던 이미지와 V2B 저성능 이미지에서 보완 효과가 컸다. 반대로 V2B가 이기던
이미지에서는 손실이 있어 model diversity가 모든 샘플에 일관된 이득은 아니다. Per-image H는
202장에서 상승, 168장에서 하락, 34장에서 같았다. 이 분해는 ensemble을 보존할 근거이지만
추가 weight sweep의 근거로 사용하지 않는다.

## Comparison With V8

| Candidate | Macro H | Global H |
|---|---:|---:|
| V8 scale TTA | 0.966840 | **0.965130** |
| V9 model ensemble | **0.967266** | 0.965090 |

V9가 macro H는 `+0.000426` 높고 V8이 global H는 `+0.000040` 높아 사실상 서로 다른 장점을
가진 후보다. V9가 V8을 완전히 대체했다고 보지 않고 두 fusion 축을 모두 V13 후보로 유지한다.

## Test Artifact And Sanity Check

- Images/rows: `413 / 413`
- Total/min/max/mean regions: `44,247 / 38 / 219 / 107.14`
- Empty images and images over 500 regions: `0 / 0`
- Missing/extra/duplicate filenames: all `0`
- Invalid point count, non-finite, zero-area, invalid geometry, duplicate polygon: all `0`
- Boundary overshoot: 26 polygons, maximum 7 px; violations over 8 px tolerance: `0`
- CSV coordinate/count mismatch: `0`

V2B test의 45,236개보다 영역 수가 약 2.2% 적다. Validation에서는 이 변화와 함께 recall이
상승했지만 test label은 없으므로 같은 효과라고 단정하지 않는다. CSV는
`Generated offline, competition closed` 상태의 재현 artifact다.

## Decision And Next Step

- V2B/V5 equal model-map fusion을 V9 후보로 채택한다.
- V2B-heavy fallback, V4 추가, threshold 및 weight sweep은 실행하지 않는다.
- V2B는 clean single control, V8은 scale TTA 후보, V9는 architecture ensemble 후보로 각각
  역할을 분리해 보존한다.
- 다음은 V10 domain self-supervised pilot이다. V8/V9 결합은 최종 single model이 정해진 뒤
  V13에서 평가한다.
