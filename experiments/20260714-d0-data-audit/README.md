# D0 Train/Val/Test Data Audit

## 상태

완료. D0는 모델을 새로 학습하거나 leaderboard에 제출하는 version 실험이 아니라, 이후
augmentation, TTA, ensemble, SSL의 가설을 제한하기 위한 분석 단계다. Test 이미지는 자동
통계와 기존 모델 추론에만 사용했으며 사람이 polygon을 만들거나 수정하지 않았다.

## 질문

1. 공식 train/val/test의 촬영 품질과 크기 분포가 실제로 다른가?
2. V2B가 실패하는 조건은 blur, 밝기 같은 photometric 품질인가, 작은 글자와 밀도인가?
3. V2B와 V5는 같은 곳에서 틀리는가, 이후 model ensemble을 시험할 정도로 보완적인가?
4. `pseudo_label/`의 세 source를 기존 데이터와 구분 없이 사용해도 되는가?

## 데이터와 방법

실제 JSON과 이미지 파일을 기준으로 다음 수를 처리했다.

| Source | 이미지 수 | Label 사용 |
| --- | ---: | --- |
| Official train | 3,272 | Human polygon |
| Official val | 404 | Human polygon, 평가에만 사용 |
| Official test | 413 | 사용하지 않음 |
| Pseudo SROIE | 916 | 사용하지 않음 |
| Pseudo WildReceipt | 1,767 | 사용하지 않음 |
| Pseudo CORD-v2 | 1,000 | 사용하지 않음 |
| 합계 | 7,772 |  |

대회 설명의 train 3,273장과 달리 현재 제공된 `train.json`과 연결 이미지에서는 3,272장을
읽었다. 분석은 로컬에 실제 존재하는 데이터 수를 기준으로 한다.

이미지 품질 지표는 해상도 차이의 영향을 줄이기 위해 긴 변을 최대 512 px로 축소한 뒤
계산했다. 밝기, 대비, Laplacian variance blur proxy, edge density, entropy, saturation,
dark/bright pixel 비율을 측정했다. Polygon이 있는 train/val에서는 1024 입력으로 resize했을
때의 글자 짧은 변, 작은 글자 비율, 면적, 각도, vertex 수를 추가로 계산했다.

모델 비교에는 같은 `box_thresh=0.25`, pixel `thresh=0.30`, 1024 추론 조건을 사용했다.

- V2B: ResNet18, 현재 최고 clean single checkpoint
- V5: ResNet34 epoch 7, H-Mean은 동률이지만 recall이 높은 checkpoint
- Validation: 이미지별 CLEval H/P/R 및 global CLEval 계산
- Test: 정답 없이 예측 수, confidence, 크기와 모델 disagreement만 계산
- Disagreement: polygon의 axis-aligned bounding box를 IoU 0.5에서 greedy matching한 근사치

## 공식 데이터 분포

| Dataset | 밝기 median | 대비 median | Blur proxy median | 긴/짧은 변 median | MP median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 0.5744 | 0.1838 | 1863.6 | 1.3333 | 1.2288 |
| Val | 0.5709 | 0.1832 | 1764.1 | 1.3333 | 1.2288 |
| Test | 0.5735 | 0.1885 | 1781.7 | 1.3333 | 1.2288 |

Train과 test에서 가장 큰 단일 KS 차이도 edge density의 `0.0655`였고 p-value는
`0.0819`였다. Train과 val의 최대 KS 차이는 blur proxy의 `0.0613`이었다. ECDF와 contact
sheet를 함께 확인해도 공식 세 split은 밝기, 대비, blur, 종횡비, 해상도에서 매우 비슷하다.

따라서 D0는 blur, JPEG, illumination, rotation/perspective 중 하나를 test domain gap
대응용으로 새로 학습해야 한다는 근거를 만들지 못했다. 이것은 그런 augmentation이 항상
무효라는 뜻이 아니라, 현재 test 분포를 이유로 반복 튜닝할 근거가 없다는 뜻이다.

## V2B 오류와 작은 글자

가장 큰 관계는 photometric 품질이 아니라 작은 글자 비율과 text-region 밀도에서 나왔다.

| 관계 | Spearman rho | p-value |
| --- | ---: | ---: |
| 12 px 미만 글자 비율 vs Recall | -0.2061 | 0.000030 |
| 12 px 미만 글자 비율 vs H-Mean | -0.1945 | 0.000083 |
| Region 수 vs Recall | -0.1835 | 0.000208 |
| 1024 변환 후 median 짧은 변 vs Recall | +0.1761 | 0.000376 |

Validation을 12 px 미만 글자 비율의 사분위로 나누면 상위 25% 구간의 평균 H/P/R은
`0.9427 / 0.9522 / 0.9363`이었다. 나머지 세 구간의 평균 H는 각각 `0.9661`, `0.9759`,
`0.9744`였다. 글자가 특히 작고 많은 영수증에서 recall과 precision이 함께 어려워진다.

반면 밝기, 대비, blur proxy, edge density, entropy, saturation, dark fraction과 H-Mean의
상관 절댓값은 모두 `0.067` 이하였다. Edge density와 recall의 관계가 `rho=0.0853`으로
photometric 지표 중 가장 컸지만 p-value가 `0.0869`이고 효과도 작았다. V4의 결합
photometric policy가 local H를 낮춘 결과와도 방향이 일치한다.

## V2B와 V5 비교

| Model/split | 예측 수 평균 | 예측 수 최대 | Macro H/P/R | Global H/P/R |
| --- | ---: | ---: | --- | --- |
| V2B val | 111.55 | 216 | 0.964776 / 0.969978 / 0.961447 | 0.962238 / 0.967515 / 0.957018 |
| V5 val | 106.64 | 216 | 0.964633 / 0.964343 / 0.966860 | 0.962202 / 0.960290 / 0.964121 |
| V2B test | 109.53 | 226 | N/A | N/A |
| V5 test | 105.78 | 210 | N/A | N/A |

V5는 평균 polygon 수가 더 적은데도 CLEval recall이 높다. 단순히 약한 box를 더 많이
받아들인 모델이라기보다, V2B와 polygon의 연결, 분리, 경계 형태가 다른 모델로 보는 것이
맞다. 두 모델 모두 이미지당 최대 226개 이하여서 500개 제한과는 거리가 있다.

IoU 0.5 prediction Jaccard의 평균은 val `0.8733`, test `0.8732`로 거의 같다. Validation에서
V5가 이미지별 H가 높은 경우는 166장, V2B가 높은 경우는 220장, 동률은 18장이었다. 두 값 중
좋은 것만 고르는 실현 불가능한 oracle의 macro H는 `0.97114`로 V2B보다 `+0.00637` 높다.
또한 prediction Jaccard와 두 모델 H 차이의 절댓값 사이에는 `rho=-0.522` 관계가 있어,
실제로 예측이 다를수록 어느 한 모델이 크게 이기거나 지는 경향이 있다.

이 결과는 V9 probability-map ensemble을 시험할 충분한 후보 근거다. 그러나 oracle은 정답을
보고 모델을 선택한 상한일 뿐이고, polygon union은 false positive와 중복을 늘릴 수 있다.
따라서 ensemble 채택 여부는 V9의 macro/global local CLEval로 따로 결정한다.

## Auxiliary/Pseudo 이미지 분포

세 source는 공식 데이터와 서로도 크게 다르다.

- SROIE: 밝기 median `0.9384`, saturation `0`, entropy `3.4291`로 스캔 또는 흰 배경 영수증이
  매우 많다.
- WildReceipt: 해상도 median `0.2123 MP`로 공식 데이터의 `1.2288 MP`보다 훨씬 작다.
- CORD-v2: blur proxy median `423.4`, edge density `0.0323`으로 공식 데이터보다 낮다.

따라서 V10 SSL의 image-only pool로는 domain 다양성을 줄 수 있지만, source를 구분 없이
균등한 이미지 단위로 섞으면 1,767장인 WildReceipt가 분포를 지배한다. V10에서는 source-balanced
sampling을 사용하고, text stroke를 지울 수 있는 강한 blur/crop은 피한다. V12에서 polygon을
사용할 때는 source별 confidence와 prediction-count 분포를 따로 calibration해야 한다. D0만으로
pseudo polygon의 정확도나 supervised 성능 향상을 보장할 수는 없다.

## 후속 실험 결정

1. **V6은 계획대로 실행한다.** V5의 recall 우위가 polygon 수 증가 때문이 아니므로
   `box_thresh=0.30`에서 precision을 회복할 수 있는지 local CLEval로 확인한다.
2. **V7은 skip한다.** `skipped: no D0-supported train/test augmentation gap`으로 기록한다.
   공식 split의 photometric 분포가 같고 V2B 실패와 photometric 지표의 관계도 약하다.
3. **V8 scale TTA는 유지하되 첫 후보를 고해상도 방향으로 바꾼다.** 작은 글자 구간의 실패가
   뚜렷하므로 1024 control과 `1024+1152` map average를 먼저 비교한다. 낮은 scale인 896은
   precision 보정 fallback으로만 남긴다.
4. **V8-ROT과 V8-PHOTO는 실행하지 않는다.** Orientation 또는 촬영 품질 shift 근거가 없다.
   V8-TILE은 작은 글자가 많은 긴 영수증의 scale TTA 결과가 여전히 나쁠 때만 고려한다.
5. **V9 V2B+V5 map ensemble은 유지한다.** Val/test disagreement가 비슷하고 validation의
   보완 가능성이 확인됐지만, local fusion 이득을 보기 전에는 채택하지 않는다.
6. **V10 SSL은 source-balanced pool을 사용한다.** Auxiliary source별 분포 차이를 sampling과
   transform 설계에 반영한다.

V6, V8, V9의 선택은 Public 점수를 보지 않고 official-val macro와 global CLEval이 모두
개선되는지로 내린다. D0 자체에는 제출 파일을 만들지 않았다.

## 실행

```bash
python scripts/d0_data_audit.py
python scripts/d0_model_disagreement.py
```

CPU 이미지 감사는 약 26초 걸렸다. Batch 8 GPU 추론 시간은 V2B val/test가 약
`72.8/12.3초`, V5 val/test가 약 `70.2/13.8초`였고 peak allocated GPU memory는 약
`2.66/2.69 GiB`였다. Validation은 이미지별 CLEval 계산 때문에 test보다 오래 걸렸다.

## 한계

- Laplacian variance는 초점 blur만 측정하는 완전한 지표가 아니며 글자량과 배경 edge에도
  영향을 받는다.
- KS와 상관관계는 가설을 좁히는 통계이지 augmentation의 인과 효과가 아니다.
- Bounding-box IoU disagreement는 CLEval polygon matching과 다르다.
- Test에는 정답이 없으므로 disagreement가 큰 이미지에서 어느 모델이 맞는지 판단하지 않았다.
- Official val을 반복 사용한 model-selection 위험은 남아 있으므로 최종 선택은 V11 K-fold에서
  split 안정성을 다시 확인한다.

## Artifact

- 분석 코드: [d0_data_audit.py](/data/ephemeral/home/receipt-text-detection/scripts/d0_data_audit.py)
- 모델 비교 코드: [d0_model_disagreement.py](/data/ephemeral/home/receipt-text-detection/scripts/d0_model_disagreement.py)
- 전체 이미지 지표: [image_metrics.csv](/data/ephemeral/home/receipt-text-detection/experiments/20260714-d0-data-audit/image_metrics.csv)
- Dataset 요약: [dataset_summary.csv](/data/ephemeral/home/receipt-text-detection/experiments/20260714-d0-data-audit/dataset_summary.csv)
- 분포 차이: [distribution_shift.csv](/data/ephemeral/home/receipt-text-detection/experiments/20260714-d0-data-audit/distribution_shift.csv)
- ECDF 시각화: [distribution_ecdf.png](/data/ephemeral/home/receipt-text-detection/experiments/20260714-d0-data-audit/distribution_ecdf.png)
- 품질 contact sheet: [contact_sheets](/data/ephemeral/home/receipt-text-detection/experiments/20260714-d0-data-audit/contact_sheets)
- 모델 요약: [model_prediction_summary.csv](/data/ephemeral/home/receipt-text-detection/experiments/20260714-d0-data-audit/model_prediction_summary.csv)
- Val 품질/성능 결합: [v2b_val_quality_metrics.csv](/data/ephemeral/home/receipt-text-detection/experiments/20260714-d0-data-audit/v2b_val_quality_metrics.csv)
- 품질 상관관계: [val_quality_correlation.csv](/data/ephemeral/home/receipt-text-detection/experiments/20260714-d0-data-audit/val_quality_correlation.csv)
- Val disagreement: [v2b_v5_val_disagreement.csv](/data/ephemeral/home/receipt-text-detection/experiments/20260714-d0-data-audit/v2b_v5_val_disagreement.csv)
- Test disagreement: [v2b_v5_test_disagreement.csv](/data/ephemeral/home/receipt-text-detection/experiments/20260714-d0-data-audit/v2b_v5_test_disagreement.csv)
- Val 비교 시각화: [val_model_disagreement.jpg](/data/ephemeral/home/receipt-text-detection/experiments/20260714-d0-data-audit/visualizations/val_model_disagreement.jpg)
- Test 비교 시각화: [test_model_disagreement.jpg](/data/ephemeral/home/receipt-text-detection/experiments/20260714-d0-data-audit/visualizations/test_model_disagreement.jpg)

