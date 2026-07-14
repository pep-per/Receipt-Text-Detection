# V8 Scale TTA Screen

## Status

Completed and adopted as a probability-map TTA candidate. The competition is closed, so the test
artifact was generated offline and has no leaderboard score.

## Purpose

V2B epoch 8 모델을 재학습하지 않고, 1024와 1152 입력에서 생성한 DB probability map이
small/faint text를 상호 보완하는지 확인한다. Scale별 polygon을 합치지 않고 padding을 제외한
유효 map을 1024 좌표로 정렬해 equal average한 뒤 DB post-processing을 한 번만 적용한다.

## Control And Change

- Checkpoint: V2B ResNet18 epoch 8
- Control: 1024 probability map, `thresh=0.30`, `box_thresh=0.25`
- Candidate: valid-region aligned `mean(map_1024, map_1152)`
- Fixed: model weights, post-processing, official val 404장, seed 42
- Changed: inference scale과 map fusion만
- Excluded: polygon union, flip, rotation, threshold sweep

## Pre-registered Gate

결과를 보기 전에 다음 기준을 고정한다.

1. Candidate의 official-val macro H와 global H가 control보다 모두 높아야 한다.
2. 10,000회 paired image bootstrap에서 macro/global H `P(delta > 0)`가 각각 0.95 이상이어야
   한다.
3. 위 조건을 통과하면 test 413장 JSON/CSV를 `Generated offline, competition closed`로 만든다.
4. Recall 상승과 precision 하락 때문에 H가 낮아진 경우에만 roadmap의 fallback인
   `896+1024`를 한 번 평가한다.
5. 첫 candidate가 통과하면 fallback과 threshold sweep은 실행하지 않는다.

Bootstrap은 같은 이미지의 control/candidate 결과를 함께 재표집한다. Macro는 per-image
metric 평균으로, global은 각 이미지의 CLEval raw count를 합친 후 H/P/R을 다시 계산한다.

## Command

```bash
cd /data/ephemeral/home/receipt-text-detection

python scripts/v8_scale_tta.py \
  --split val \
  --scales 1024 1152 \
  --batch-size 4 \
  --workers 4 \
  --wandb-mode online \
  --run-name v8_scale_tta_1024_1152_eval
```

## Artifacts

- Implementation: [v8_scale_tta.py](../../scripts/v8_scale_tta.py)
- Submission validation: [validate_submission.py](../../scripts/validate_submission.py)
- Metrics: [metrics.json](metrics.json)
- Paired image results: [per_image_val.csv](per_image_val.csv)
- Bootstrap: [bootstrap_paired.csv](bootstrap_paired.csv)
- Quality strata: [strata_metrics.csv](strata_metrics.csv)
- Validation config: [run_config_val.json](run_config_val.json)
- Test summary: [test_prediction_summary.json](test_prediction_summary.json)
- Submission validation: [submission_validation.json](submission_validation.json)
- Test JSON: `baseline_code/outputs/v8_scale_tta_1024_1152/submissions/20260714_154949.json`
- Offline CSV: `submissions/v8_scale_tta_1024_1152_20260714_154949.csv`
- W&B validation: [7eb4lky8](https://wandb.ai/pep-per/receipt-text-detection/runs/7eb4lky8)
- W&B test prediction: [2vwg3ymn](https://wandb.ai/pep-per/receipt-text-detection/runs/2vwg3ymn)

## Result

| Evaluation | H-Mean | Precision | Recall |
|---|---:|---:|---:|
| 1024 control macro | 0.964785 | 0.969979 | 0.961463 |
| 1024+1152 TTA macro | **0.966840** | **0.972512** | **0.962827** |
| Macro delta | **+0.002055** | **+0.002533** | **+0.001364** |
| 1024 control global | 0.962249 | 0.967516 | 0.957039 |
| 1024+1152 TTA global | **0.965130** | **0.971017** | **0.959314** |
| Global delta | **+0.002881** | **+0.003501** | **+0.002274** |

Control macro H는 이전 independent V2B 값 `0.964760`과 `0.000025` 차이로 재현됐다. 따라서
상승분은 evaluator 변경이나 checkpoint 불일치보다 map fusion 효과로 해석할 수 있다.

## Paired Bootstrap

| Metric delta | Mean | 95% CI | P(delta > 0) |
|---|---:|---:|---:|
| Macro H | +0.002065 | [+0.000548, +0.003631] | 0.9967 |
| Macro P | +0.002538 | [+0.001471, +0.003643] | 1.0000 |
| Macro R | +0.001379 | [-0.001004, +0.003839] | 0.8679 |
| Global H | +0.002885 | [+0.001036, +0.004851] | 0.9998 |
| Global P | +0.003497 | [+0.002110, +0.005033] | 1.0000 |
| Global R | +0.002285 | [-0.000794, +0.005684] | 0.9211 |

Macro/global H의 개선 확률이 모두 사전 기준 0.95를 넘고 신뢰구간 하한도 양수다. Recall
단독 신뢰구간은 0을 포함하지만 precision 상승이 안정적이고 H 개선은 양쪽 aggregation에서
유의하다. 따라서 V8은 gate를 통과했다.

## Error And Strata Analysis

| Validation stratum | Images | Control H | TTA H | Delta H |
|---|---:|---:|---:|---:|
| All | 404 | 0.964785 | 0.966840 | +0.002055 |
| Control H bottom quartile | 101 | 0.904428 | 0.912320 | +0.007892 |
| Short text side bottom quartile | 101 | 0.949935 | 0.957711 | +0.007776 |
| Small-text ratio top quartile | 101 | 0.947765 | 0.953037 | +0.005272 |
| Low-contrast quartile | 101 | 0.958204 | 0.962377 | +0.004173 |

이득이 기존 저성능 이미지와 작은 글자 stratum에서 더 컸으므로 D0에서 세운 scale TTA
가설과 방향이 일치한다. Per-image H는 199장에서 상승, 131장에서 하락, 74장에서 같았다.
즉 모든 이미지가 좋아진 것은 아니지만 paired aggregate 개선은 일부 한두 장만의 우연으로
설명되지 않는다.

## Test Artifact And Sanity Check

- Images/rows: `413 / 413`
- Total/min/max/mean regions: `45,237 / 39 / 226 / 109.53`
- Empty images and images over 500 regions: `0 / 0`
- Missing/extra/duplicate filenames: all `0`
- Invalid point count, non-finite, zero-area, invalid geometry, duplicate polygon: all `0`
- Boundary overshoot: 25 polygons, maximum 6 px; violations over the fixed 8 px tolerance: `0`
- CSV coordinate/count mismatch: `0`

기존 V2B test JSON과 비교하면 381/413장의 polygon이 완전히 같고, region count가 달라진
이미지는 1장뿐이다. V8은 test 전체를 공격적으로 바꾼 것이 아니라 일부 불안정한 경계를
조정했다. Test label이 없고 대회가 종료됐으므로 이 사실은 hidden H 개선의 증거가 아니며,
CSV 상태는 `Generated offline, competition closed`로 기록한다.

## Decision And Next Step

- `1024+1152` TTA를 채택하고 `896+1024` fallback 및 threshold sweep은 실행하지 않는다.
- V2B 1024는 여전히 clean single-model control이다. V8은 추론 단계 후보이지 새 학습 모델이
  아니다.
- 다음 V9는 TTA를 섞지 않고 V2B와 V5의 model probability-map ensemble만 평가해 model
  diversity 효과를 분리한다.
- V13에서 최종 single model이 정해진 뒤 V8 TTA와 통과한 model ensemble의 결합을 다시
  검증한다.
