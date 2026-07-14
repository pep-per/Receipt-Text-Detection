# V11A Group-aware K-fold Split

## Status

Completed and adopted for V11B OOF training.

## Objective

Official train 3,272장과 val 404장을 합친 3,676장의 human-labeled 데이터를 5개 fold로 나눈다.
각 fold는 V11B의 held-out validation이 되며, 모든 이미지는 정확히 한 번 OOF prediction을
받는다. Test 413장과 pseudo-label 이미지는 split 생성에 사용하지 않는다.

## Fixed Split Recipe

- Fold count: 5
- Splitter: scikit-learn `StratifiedGroupKFold`, shuffle disabled
- Reserved model-training seed: 42; deterministic splitter 자체는 random seed를 사용하지 않음
- Stratification label: original train/val source x GT-region quintile x small-text-ratio binary
- Group: high-confidence exact/near-duplicate connected component
- Balance audit only: aspect ratio, brightness, contrast, blur, resolution, total polygon count

Fold 수와 training seed는 model score를 보기 전에 고정한다. 여러 seed의 split balance를
비교해 가장 좋은 것을 고르지 않는다.

최초 구현에서는 `shuffle=True, random_state=42`를 사용했지만 model score를 계산하기 전의
structural validation에서 fold 크기가 `733~737`, 기존 val 출처 수가 `70~91`, 최대 stratum
비율 편차가 `0.03439`로 사전 기준을 통과하지 못했다. scikit-learn의 group shuffle은 class
분포가 비슷한 group 일부만 섞을 수 있어 이 데이터에서는 층화를 오히려 약화했다.
`shuffle=False`는 결정론적 group-stratification으로 이 구조적 실패를 수정한 것이다. 성능을
보고 split을 선택한 것이 아니며, 실패한 split에서는 어떤 학습이나 평가도 하지 않았다.

## Duplicate Group Rule

모든 train+val 이미지에 SHA-256과 64-bit perceptual hash를 계산한다. Aspect-ratio relative
difference가 12% 이하이고 pHash Hamming distance가 8 이하인 pair를 audit candidate로 만든다.
다음 자동 규칙 중 하나를 만족할 때만 같은 group으로 묶는다.

1. Exact SHA-256 match
2. pHash distance 2 이하, GT word-count relative difference 10% 이하
3. pHash distance 6 이하, 128x128 grayscale correlation 0.90 이상, word-count difference 10% 이하

보수적인 규칙이므로 모든 시각적 유사 영수증을 찾는다고 주장하지 않는다. False grouping으로
서로 다른 영수증을 한 group으로 과도하게 묶는 위험과 명백한 near-duplicate leakage 방지를
절충한다.

## Required Validation

- 3,676개 filename이 중복 없이 정확히 한 fold에 존재
- Fold 크기 차이 최대 1장 수준
- Accepted duplicate pair/group이 fold를 가로지르지 않음
- 각 fold의 기존 val 이미지 수와 GT-region strata가 균형적
- 주요 continuous quality feature의 fold 평균이 전체 평균에서 과도하게 벗어나지 않음
- Manifest 재실행 시 byte-identical fold assignment

## Result

최종 deterministic split은 모든 structural gate를 통과했다. 이 단계에서는 model prediction이나
CLEval score를 전혀 사용하지 않았다.

| Check | Result |
| --- | ---: |
| Total/unique images | `3,676 / 3,676` |
| Fold sizes | `735 / 735 / 736 / 735 / 735` |
| Original-val images per fold | `81 / 81 / 80 / 80 / 82` |
| Candidate/accepted near-duplicate pairs | `96 / 11` |
| Multi-image groups | `11` |
| Group/pair leakage | `0 / 0` |
| Maximum stratum proportion delta | `0.001159` |
| Maximum continuous-feature fold-mean z | `0.06513` |
| Re-run manifest SHA-256 match | pass |

Accepted 11쌍은 모두 기존 train 내부 쌍이었다. Contact sheet를 확인한 결과 동일 영수증의
근접 촬영본으로 보였으며, 이 판단은 fold assignment나 model score를 본 뒤 수동으로 pair를
추가/삭제하는 데 사용하지 않았다. 자동 acceptance rule의 결과를 확인하는 audit 용도였다.

Manifest SHA-256은
`d422e2b86da3b2225213a9c6159f62e8fa51283990940c821298ea0a4103ebac`이다. 같은 명령을
다시 실행해 byte-identical manifest임을 확인했다.

```bash
python scripts/v11_make_folds.py
```

### 첫 split을 폐기한 이유

처음 사전 구현한 `shuffle=True, random_state=42`는 fold 크기 `733~737`, 기존 val 출처
`70~91`, 최대 stratum 비율 편차 `0.03439`로 structural validation을 통과하지 못했다.
어떤 모델도 이 split으로 학습하거나 평가하지 않았다. Seed나 model score를 탐색하는 대신
`shuffle=False`의 deterministic SGKF로 고쳤고, fold 크기 `735~736`, 기존 val 출처
`80~82`, 최대 stratum 비율 편차 `0.001159`가 됐다. 따라서 이는 성능에 맞춘 split 선택이
아니라 잘못 동작한 층화 구현을 model training 전에 수정한 것이다.

## Decision

- V11B의 유일한 OOF manifest로 채택한다.
- Fold 수, strata, duplicate group과 assignment를 이후 결과에 맞춰 바꾸지 않는다.
- Training seed 42는 각 fold model에 공통 적용하되 split 자체는 deterministic이라 seed를
  소비하지 않는다.
- V11B는 먼저 V2B recipe의 5-fold OOF를 측정한다. V10 SSL initialization은 사용하지 않는다.

## Artifacts

- Main manifest: [v11_5fold_seed42.csv](../../data/splits/v11_5fold_seed42.csv)
- Split metadata: [v11_5fold_seed42_metadata.json](../../data/splits/v11_5fold_seed42_metadata.json)
- Fold balance: [fold_summary.csv](fold_summary.csv)
- Stratum balance: [stratum_balance.csv](stratum_balance.csv)
- Duplicate audit: [near_duplicate_pairs.csv](near_duplicate_pairs.csv)
- Validation: [split_validation.json](split_validation.json)
- Duplicate contact sheet: [near_duplicate_groups.jpg](near_duplicate_groups.jpg)
- Generator: [v11_make_folds.py](../../scripts/v11_make_folds.py)
