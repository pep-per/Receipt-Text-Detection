# K-fold, Augmentation, Ensemble, TTA Guide

이 문서는 text detection 실험에서 자주 함께 언급되는 K-fold, training augmentation,
multi-seed ensemble, architecture ensemble, model ensemble, TTA를 구분하고 이 프로젝트에서
어떤 순서로 사용하는지 설명한다.

## 한눈에 보는 차이

| 방법 | 바꾸는 것 | 새 학습 필요 | 다양성의 원천 | 주된 목적 |
| --- | --- | ---: | --- | --- |
| Training augmentation | 학습 입력 이미지 | 필요 | 촬영 조건과 기하 변형 | 일반화 향상 |
| K-fold | train/validation 분할 | Fold마다 필요 | 학습 데이터 구성 | 안정성 측정, fold teacher 생성 |
| Multi-seed | 초기화와 데이터 순서 | Seed마다 필요 | 학습의 확률적 변동 | 분산 감소 |
| Architecture ensemble | Backbone/detector 구조 | 모델마다 필요 | 서로 다른 inductive bias | 오류 보완 |
| TTA | 추론 입력의 scale/flip 등 | 불필요 | 같은 이미지의 여러 view | 예측 안정화 |
| Model ensemble | 여러 학습 모델의 출력 | 모델은 미리 학습 | Fold/seed/architecture | 예측 분산 감소 |

핵심 구분은 다음과 같다.

```text
학습 전에 이미지를 바꾸고 gradient update에 사용한다
-> training augmentation

학습이 끝난 같은 모델에 여러 변형 이미지를 넣어 예측을 합친다
-> TTA

학습 데이터, seed, architecture를 바꿔 여러 weight를 만든다
-> model ensemble 후보
```

## 전체 모델과 추론 구조

현재 detector는 다음 순서로 동작한다.

```text
image
-> encoder
-> decoder
-> DBHead
-> text probability/threshold map
-> DB post-processing
-> polygon
```

Augmentation과 K-fold는 주로 모델을 학습하는 과정에 영향을 준다. TTA와 ensemble은 학습된
모델의 probability map을 만드는 추론 과정에 영향을 준다.

## Training Augmentation

Training augmentation은 학습 이미지에 변형을 적용한 뒤 그 이미지로 loss와 gradient를
계산하는 방법이다.

```text
original train image
-> transform
-> model forward
-> human polygon으로 loss 계산
-> weight update
```

대표적인 변형은 다음과 같다.

- Photometric: brightness, contrast, gamma, blur, JPEG compression, shadow
- Geometric: resize, crop, flip, rotation, perspective

같은 augmentation recipe를 모든 fold에서 사용하는 것은 K-fold의 학습 조건을 통일하는
것이다. 이것은 TTA가 아니다.

### V4가 알려준 것

V4는 학습 이미지 일부에 brightness/contrast, gamma, Gaussian blur, motion blur, JPEG 중
하나를 적용했다. 이 결합 policy는 V2B보다 H-Mean이 낮았다.

이 결과가 의미하는 것은 다음과 같다.

- 현재 확률과 강도의 결합 training policy는 채택할 근거가 없다.
- 다섯 transform 중 어느 하나가 항상 나쁘다고 증명된 것은 아니다.
- Photometric TTA가 항상 실패한다고 증명된 것도 아니다.
- D0에서 실제 domain gap이 확인되면 한 transform family만 좁게 다시 검증할 수 있다.

학습 augmentation은 입력 정보를 일부 파괴할 수 있다. 작은 text stroke가 blur나 JPEG로
사라졌는데 polygon target은 그대로면 모델은 모호한 입력에서 정밀한 경계를 맞춰야 한다.
따라서 실제 train/val/test 분포보다 강한 변형은 precision과 recall을 함께 낮출 수 있다.

## Test-Time Augmentation

TTA는 모델 weight를 바꾸지 않고 같은 이미지를 여러 view로 추론해 결과를 합친다.

```text
test image
├-> 896 view  -> probability map
├-> 1024 view -> probability map
└-> 1152 view -> probability map

각 map을 원본 좌표로 복원
-> map 평균
-> DB post-processing 한 번
-> polygons
```

### 왜 scale TTA부터 하는가

이 프로젝트에서는 640에서 896으로 해상도를 높였을 때 성능이 크게 올랐고, 896에서
1024로 높였을 때도 local과 Public H-Mean이 올랐다. 작은 text에 scale이 중요하다는 직접
근거가 있으므로 scale TTA가 첫 후보다.

Photometric TTA, rotation TTA, tiling TTA도 가능하지만 D0와 local CLEval 근거가 있을 때만
조건부 실험으로 실행한다.

### Polygon보다 probability map을 평균하는 이유

각 view에서 polygon을 만든 뒤 모두 합치면 같은 text가 여러 번 남을 수 있다. Duplicate와
overlap은 precision을 낮추고 이미지당 500개 제한 위험을 높인다.

권장 순서는 다음과 같다.

1. 각 view의 probability map을 얻는다.
2. Resize, flip, rotation을 역변환해 원본 좌표로 복원한다.
3. Padding 영역을 제외하는 valid mask를 적용한다.
4. Map을 평균한다.
5. 평균 map에 DB post-processing을 한 번 적용한다.

기하 변환을 정확히 역변환하지 못하면 text boundary가 흐려지므로 TTA를 적용하기 전에
identity view가 원본 추론과 같은 결과를 내는지 검증해야 한다.

## K-fold Cross-validation

K-fold는 human-labeled 데이터를 K개 부분으로 나눈 뒤 K번 학습하는 검증 방법이다.

5-fold 예시:

```text
M1: fold 2~5 학습 -> fold 1 검증
M2: fold 1,3~5 학습 -> fold 2 검증
M3: fold 1,2,4,5 학습 -> fold 3 검증
M4: fold 1~3,5 학습 -> fold 4 검증
M5: fold 1~4 학습 -> fold 5 검증
```

각 모델은 자기 held-out fold의 human label을 학습에서 사용하지 않는다. Fold별 H/P/R의
평균과 표준편차를 보면 설정이 특정 split에만 우연히 잘 맞았는지 확인할 수 있다.

### Stratified와 Group-aware split

영수증 이미지를 단순히 index 구간으로 자르지 않는다. 다음 특성이 fold마다 비슷하게
분배되도록 한다.

- Source 또는 filename prefix
- 이미지 크기와 종횡비
- Word/polygon 수
- 작은 text 비율
- Blur, 밝기와 대비
- 긴 영수증과 배경이 넓은 이미지

Near-duplicate나 같은 원본에서 파생된 이미지는 같은 fold에 둔다. 유사 이미지가 train과
validation에 나뉘면 실제 일반화보다 높은 점수가 나올 수 있다.

## OOF Prediction

OOF는 Out-Of-Fold의 약자다. 각 이미지에 대해 그 이미지의 human label을 보지 않은 fold
모델의 예측만 사용한다.

```text
fold 1 이미지의 OOF prediction = M1 prediction
fold 2 이미지의 OOF prediction = M2 prediction
...
fold 5 이미지의 OOF prediction = M5 prediction
```

이 예측을 전체 데이터 순서로 합치면 OOF macro/global CLEval을 계산할 수 있다. OOF는
선택한 training recipe의 일반화 성능을 측정한다.

### 잘못된 OOF ensemble

Fold 1 이미지에 M1~M5를 모두 적용해 평균하면 안 된다.

```text
M1: fold 1을 학습에서 보지 않음
M2~M5: fold 1을 human label과 함께 학습에서 봄
```

M2~M5를 fold 1 평가에 포함하면 label leakage가 생긴다. 점수가 높아도 공정한 OOF ensemble
성능이 아니다.

### K-fold에서 TTA는 가능한가

가능하다. Fold 1 이미지에 M1 하나를 사용하되 M1으로 여러 TTA view를 추론한다.

```text
fold 1 image
├-> M1 at 896
├-> M1 at 1024
└-> M1 at 1152
-> map average
-> OOF TTA prediction
```

M1은 fold 1의 human label을 보지 않았으므로 공정한 OOF TTA다. 권장 기록은 다음과 같다.

- Primary OOF: 1024 single view, TTA 없음
- Secondary OOF: V8에서 미리 통과한 고정 TTA

K-fold 안에서 새로운 TTA scale과 threshold를 다시 대량 탐색하지 않는다.

## 두 가지 K-fold 설계

### Train+val K-fold

```text
공식 train + 제공 val
-> K-fold
-> OOF recipe 평가
-> K개 model을 test에서 ensemble
```

장점:

- 모든 human annotation을 활용한다.
- Fold 평균과 분산으로 recipe 안정성을 확인한다.
- Test와 외부 pseudo 이미지는 모든 fold model에 unseen이므로 final ensemble을 사용할 수 있다.

한계:

- 같은 train+val에서 모든 fold model을 평균한 ensemble lift는 공정하게 측정할 수 없다.
- 각 OOF 이미지는 해당 held-out model 하나로만 평가해야 한다.

### Train-only K-fold + 제공 val holdout

```text
공식 train만 K-fold 학습
제공 val 404장은 어떤 model 학습에도 넣지 않음
-> 제공 val에서 K개 model과 K-model ensemble을 모두 평가
```

장점:

- 제공 val이 모든 ensemble 구성원에게 unseen이다.
- Single model과 fold ensemble의 차이를 직접 비교할 수 있다.

한계:

- 각 fold model은 공식 train의 일부만 학습한다.
- 제공 val의 human label을 최종 training에 쓰지 못한다.
- 제공 val은 이미 여러 실험 선택에 사용돼 완전히 새로운 holdout은 아니다.

현재 로드맵은 V9에서 train-only로 학습한 기존 V2B/V5의 map ensemble 거동을 제공 val에서
확인하고, V11에서는 train+val K-fold OOF로 recipe 안정성을 확인한 뒤 test/pseudo 이미지에
fold ensemble을 사용하는 방식이다.

## Multi-seed Ensemble

Multi-seed는 architecture, 데이터와 설정은 같게 두고 random seed만 바꿔 여러 모델을
학습한다.

```text
ResNet18, seed 42
ResNet18, seed 123
ResNet18, seed 777
```

Seed는 weight initialization, 데이터 shuffle, stochastic augmentation 등에 영향을 준다.
여러 seed의 예측을 평균하면 한 번의 우연한 학습 경로에 대한 의존을 줄일 수 있다.

Multi-seed는 TTA가 아니다. 각 seed마다 새로운 weight와 checkpoint가 생긴다. Fold와 seed를
함께 사용하면 `K folds x S seeds`개의 모델을 학습해야 하므로 비용이 곱해진다.

## Architecture Ensemble

Architecture ensemble은 서로 다른 backbone이나 detector를 학습해 결합한다.

```text
ResNet18 DB detector
ResNet34 DB detector
CRAFT 또는 다른 detector
```

서로 다른 모델은 오류 성향이 달라 ensemble 가치가 있을 수 있다. 이 프로젝트에서는 V2B가
높은 precision, V5가 높은 recall 성향을 보여 첫 architecture-diverse 후보가 된다.

단일 성능이 낮다는 이유만으로 항상 ensemble 가치가 없는 것은 아니다. 다른 모델이 놓친
정답을 독립적으로 맞히면 도움이 될 수 있다. 반대로 precision과 recall이 모두 낮고 오류가
같다면 평균해도 이득이 없다. Per-image disagreement와 local map ensemble로 보완성을 먼저
확인해야 한다.

## Model Ensemble과 TTA의 결합

Fold, seed, architecture는 여러 모델 weight를 만든다. TTA는 각 weight에 여러 입력 view를
만든다.

모델 `m`, TTA view `t`의 원본 좌표 복원 probability map을 `P(m,t)`라고 하면 개념적으로
다음 순서다.

```text
view 평균:  P_m = mean_t(P(m,t))
model 평균: P   = mean_m(P_m)
polygon:    DBPostProcess(P)
```

TTA와 model 평균 순서는 모든 map이 같은 좌표와 scale로 복원되고 같은 weight를 사용한다면
수학적으로 교환 가능하다. 구현에서는 view별 valid mask와 padding 처리가 있으므로 model별
TTA map을 먼저 완성한 뒤 model average를 하는 편이 디버깅하기 쉽다.

### 계산 비용

추론 횟수는 대략 다음처럼 증가한다.

```text
K folds x S seeds x A architectures x T TTA views
```

예를 들어 `5 folds x 2 seeds x 1 architecture x 3 scales`는 이미지당 30번 forward가
필요하다. 성능이 오르더라도 inference time과 제출 생성 시간을 함께 기록해야 한다.

## 이 프로젝트의 실행 순서

현재 상세 순서는
[Clean-data Experiment Roadmap](../experiments/clean_data_experiment_roadmap.md)을 따른다.

```text
D0  데이터 분포와 오류 분석
V6  단일 모델 후처리 보정
V7  D0 근거가 있는 training augmentation 하나
V8  같은 모델의 scale TTA
V9  V2B/V5 probability-map model ensemble
V10 self-supervised encoder pilot
V11 K-fold OOF와 final clean teacher
V12 pseudo-label student
V13 최종 TTA/model/fold ensemble과 threshold
```

이 순서를 사용하는 이유는 다음과 같다.

1. Training augmentation과 single model을 먼저 결정해야 뒤 TTA와 fold 모델이 같은 recipe를
   사용할 수 있다.
2. TTA와 기존 model ensemble은 재학습이 없어 K-fold 전에 저렴하게 가능성을 선별할 수 있다.
3. SSL은 encoder 초기화를 바꾸므로 K-fold 전에 채택 여부를 정해야 fold를 두 번 학습하지
   않는다.
4. K-fold teacher가 정해진 뒤 pseudo label을 생성해야 자동 polygon noise를 줄일 수 있다.
5. 최종 student가 달라지면 map score도 달라지므로 threshold와 fusion은 마지막에 다시
   보정한다.

## CLEval 평가 시 주의점

모든 비교에서 다음을 함께 기록한다.

- Macro H/P/R
- Global H/P/R
- Fold 평균과 표준편차
- 이미지별 prediction 수
- Empty prediction 수
- Invalid polygon 수
- Duplicate/overlap 비율
- 이미지당 500개 cap 위험
- 추론 시간과 GPU memory

TTA 또는 ensemble은 약한 영역을 더 많이 살려 recall을 올릴 수 있지만 false positive와
over-extended polygon도 늘릴 수 있다. H-Mean만 보지 않고 precision과 recall 이동을 함께
해석한다.

같은 checkpoint의 fusion 효과를 비교할 때 첫 평가는 같은 post-processing threshold를
사용한다. Fusion이 개선된 뒤에만 좁은 threshold 재보정을 수행한다. Threshold, TTA view,
ensemble weight를 동시에 탐색하면 어떤 요소가 효과를 냈는지 알 수 없고 제공 val에
과적합하기 쉽다.

## Public/Private 과적합 방지

- TTA scale과 ensemble weight는 local CLEval로 선택한다.
- 모든 후보를 Public에 제출해 최고 점수를 고르지 않는다.
- Public에서 recall이 낮다는 이유만으로 test-specific threshold를 만들지 않는다.
- 파일명, 추정 Public subset, 이미지별 수동 규칙을 사용하지 않는다.
- Final 후보는 fold variance와 local macro/global 근거를 우선한다.

## 실험 기록 체크리스트

각 K-fold, augmentation, TTA, ensemble 실험은 다음을 남긴다.

- 실험 version과 목적
- Control과 변경 변수 하나
- Git commit SHA
- Canonical config와 resolved Hydra config
- Dataset/split manifest
- Fold, seed, architecture
- Training augmentation
- TTA views와 역변환 방식
- Model/view fusion 순서와 weight
- Post-processing 설정
- Macro/global CLEval 결과
- Runtime와 peak GPU memory
- W&B run URL
- Checkpoint 또는 artifact 위치
- 채택/폐기 이유와 다음 단계

이 기록이 있으면 단순히 점수가 높은 결과가 아니라 어떤 다양성이 실제로 도움이 됐는지,
같은 결과를 다시 만들 수 있는지까지 판단할 수 있다.
