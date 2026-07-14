# V10 SSL Effect Analysis And Revisit Plan

## 결론

V10은 representation을 전혀 배우지 못한 실험이 아니다. 작은 글자가 많은 이미지에서 recall과
H-Mean을 높였지만, 더 큰 영역을 합쳐 검출하는 경향과 precision 손실 때문에 전체 H-Mean은
V2B와 통계적으로 동률이었다. 따라서 현재 SSL encoder는 V11 기본 초기화로 채택하지 않지만,
실패 원인과 재실험 조건은 분리해 보존한다.

분석 근거 artifact:

- [Aggregate metrics](metrics.json)
- [Per-image CLEval](per_image_val.csv)
- [Paired bootstrap](bootstrap_paired.csv)
- [V10 execution record](README.md)

## 관측된 효과

| Metric | V2B control | V10 SSL | Delta |
| --- | ---: | ---: | ---: |
| Macro H/P/R | 0.964785 / 0.969979 / 0.961463 | 0.964897 / 0.964765 / 0.966682 | +0.000112 / -0.005214 / +0.005219 |
| Global H/P/R | 0.962249 / 0.967516 / 0.957039 | 0.962613 / 0.961242 / 0.963987 | +0.000363 / -0.006274 / +0.006948 |

Paired bootstrap에서 macro/global H 개선 확률은 `0.5552 / 0.6396`이고 두 95% CI가 모두 0을
포함했다. 반면 recall delta의 CI는 macro/global 모두 양수였고 precision delta의 CI는 모두
음수였다. 즉 H 개선이 불확실한 것이지 recall 방향의 변화까지 불확실한 것은 아니다.

- Per-image 승/패/동률: `172 / 193 / 39`
- Per-image H delta 표준편차: `0.01768`, 평균 delta `+0.000112`
- 작은 글자 비율 상위 25%: H/P/R delta `+0.004266 / -0.005985 / +0.013488`
- 1024-scale text short side 하위 25%: H/P/R delta
  `+0.004191 / -0.006883 / +0.014227`
- 낮은 contrast 25%: H/P/R delta `+0.001707 / -0.004992 / +0.008029`

V10은 어려운 작은 글자 구간에서 유용했지만 쉬운 구간의 손실이 이를 상쇄했다. V10 H delta와
V5 H delta의 per-image 상관은 `0.486`이었다. V5가 이긴 이미지에서 V10 평균 H delta는
`+0.00589`, V2B가 이긴 이미지에서는 `-0.00417`이어서 V9의 V5와 겹치는 recall diversity가
상당하다.

## CLEval State 진단

| CLEval state | Control | V10 | Delta |
| --- | ---: | ---: | ---: |
| Predicted regions | 45,067 | 44,648 | -419 |
| Estimated detected characters | 234,546 | 237,084 | +2,538 |
| True-positive characters | 230,296 | 231,846 | +1,550 |
| False-positive characters | 2,313 | 2,845 | +532 |
| Precision granularity penalty | 3,369 | 3,951 | +582 |
| Recall granularity penalty | 1,777 | 1,668 | -109 |
| Merged regions | 1,891 | 2,129 | +238 |
| Split regions | 1,309 | 1,263 | -46 |
| Overlapped characters | 1,937 | 2,393 | +456 |

Polygon 수는 줄었는데 estimated character와 merge가 늘었다. 이는 V10이 약한 작은 글자를 더
포함하는 동시에 인접 text를 더 큰 영역으로 합치는 경향을 보였다는 뜻이다. CLEval recall에는
유리하지만 precision granularity penalty와 false-positive character가 늘어 H 개선을 막았다.

## 효과가 전체 H로 이어지지 않은 원인

### 1. Global SSL objective와 dense detection의 불일치

MoCo는 224 입력을 ResNet18 global average pooling과 projection head로 압축해 image-level
representation을 학습했다. 실제 detector는 1024 입력에서 pixel-level text map과 polygon
boundary를 예측한다.

Official val의 text short-side median을 224 scale로 환산하면 이미지별 median도 약 4 px다.
1024에서 12 px보다 작은 region은 224에서 2.625 px보다 작아진다. 이 크기에서는 글자 stroke와
경계 정보가 사라지기 쉽다. 따라서 V10은 receipt layout과 약한 text 존재 여부에는 민감해져
recall을 높였지만, 정밀한 분리와 boundary에는 충분한 pretext signal을 받지 못했을 수 있다.

### 2. Target domain을 과도하게 낮춘 source balancing

Image pool의 자연 비율은 official 52.61%, CORD-v2 12.87%, SROIE 11.79%, WildReceipt
22.74%였다. V10은 네 source family를 각각 25%로 sampling했다. 그 결과 official target
비율은 절반 이하로 줄고 CORD-v2와 SROIE 노출은 약 두 배가 됐다.

D0에서 세 auxiliary source의 밝기, 해상도, edge/blur 분포가 official과 다름을 확인했다.
Equal-family sampling은 source 하나의 수량 지배는 막았지만, target 분포 최적화 관점에서는
auxiliary domain을 과대표집했다. 이는 representation 다양성에는 도움이 될 수 있으나 official
precision과 boundary calibration에는 불리할 수 있다.

### 3. SSL initialization을 빠르게 지울 수 있는 fine-tuning

V10은 공정한 initialization A/B를 위해 V2B와 동일하게 encoder, decoder, head 전체에 Adam
`lr=0.001`을 적용했다. Epoch 0/1 H는 `0.9099 / 0.9398`로 V2B보다 빨랐지만 epoch 3에는
recall 0.5059의 단발성 collapse가 있었고, epoch 6~9 H는
`0.9601 / 0.9573 / 0.9649 / 0.9623`으로 흔들렸다.

초기 수렴 이득 뒤 최종 H가 동률인 패턴은 SSL weight가 유용하지 않았다는 설명과 함께,
encoder에 동일한 큰 LR을 적용해 초기 이득을 빠르게 덮어썼다는 설명도 가능하다. 둘을
구분하려면 ImageNet과 SSL 양쪽에 같은 layer-wise LR을 적용한 새 control이 필요하다.

### 4. SSL optimization은 끝났지만 downstream 최적점은 보장하지 않음

SSL loss는 epoch 14~20에 `7.8821, 7.8589, 7.8479, 7.8458, 7.8497, 7.8436,
7.8528`로 plateau했고 cosine LR은 마지막에 0이 됐다. 현재 checkpoint를 그대로 연장하면
update가 일어나지 않는다. 더 긴 schedule로 재학습할 수는 있지만 contrastive loss 감소가
detection H 상승을 보장하지 않으므로 첫 개선 축으로 삼지 않는다.

### 5. 동일 post-processing은 공정하지만 V10 calibration에는 최적이 아닐 수 있음

V2B와 V10 모두 `box_thresh=0.25`를 사용해 initialization 효과를 공정하게 비교했다. 하지만
merge와 precision penalty 변화는 probability-map calibration도 달라졌음을 보여준다. V10만
threshold를 다시 맞추면 순수 SSL A/B가 아니므로 V10 adoption에는 사용하지 않았다. 최종
모델 구성원이 정해진 뒤 V13에서만 좁게 calibration한다.

## 개선안 우선순위

### A. SSL-aware fine-tuning, 가장 직접적인 진단

- Encoder LR `1e-4`, decoder/head LR `1e-3`의 discriminative learning rate
- 1 epoch encoder warmup 또는 short freeze 후 전체 fine-tuning
- ImageNet control과 SSL candidate 양쪽에 같은 optimizer policy 적용
- 한 모델만 낮은 LR을 사용해 얻은 결과는 SSL 효과로 해석하지 않음

이 실험은 초기 이득이 supervised training에서 지워졌는지를 직접 검증한다. 다만 제공 val을
더 사용하지 않고 V11 fold protocol에서 비교해야 한다.

### B. Target-weighted image sampling

- 첫 후보: natural sampling 또는 official 50%, auxiliary 전체 50%
- Auxiliary 내부 비율은 이미지 수에 비례하거나 source별 최대 exposure 제한
- Test/val pixels 사용 여부는 계속 `transductive`로 표시

Equal-family 25%보다 official target을 보존한다. Sampling과 objective를 동시에 바꾸지 않는다.

### C. Dense/local 정보를 보존하는 pretext task

- SSL input 384 이상, conservative crop scale 0.85 이상
- DenseCL/PixPro 계열 dense contrastive 또는 masked-image modeling 후보
- Global MoCo, higher resolution, local objective를 한 번에 모두 바꾸지 않음

Receipt detection은 image category보다 local stroke와 인접 text 분리가 중요하므로 장기적으로
가장 task-aligned한 방향이다. 하지만 새로운 algorithm 탐색 비용이 크므로 pseudo-label보다
우선하지 않는다.

### D. Longer epochs는 마지막 후보

40~100 epoch schedule은 처음부터 LR schedule을 다시 정의해 학습해야 한다. 현재 plateau와
downstream 동률만으로는 비용 대비 근거가 약하다. A~C 중 하나가 OOF에서 개선 신호를 보일 때만
epoch budget을 확장한다.

## 재실험 Gate

V10을 본 직후 같은 official val에서 MoCo/SimSiam/MAE, epoch, crop, sampling을 연속 sweep하지
않는다. `V10-R1`은 다음 조건을 모두 만족할 때만 연다.

1. V11 OOF에서 작은 text stratum이 반복적으로 주요 failure mode로 확인된다.
2. V12 pseudo-label student보다 SSL 재실험의 예상 정보 가치가 높다.
3. 한 번에 한 축만 바꾸고 동일한 ImageNet control을 함께 둔다.
4. Fold별 macro/global H와 variance가 모두 좋아야 채택한다.

## Pseudo-label과의 관계

SSL rejection은 pseudo-label 사용을 막지 않는다. SSL은 image-only encoder pretraining이고,
pseudo-label learning은 teacher polygon으로 DB detection loss를 직접 학습하는 semi-supervised
경로다. V12는 V11 clean fold teacher로 auxiliary 이미지를 자동 라벨링하고 source별 confidence,
geometry, teacher agreement를 필터링해 그대로 진행한다. V10 encoder는 V12 student 기본
초기화로 사용하지 않는다.
