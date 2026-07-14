# Clean-data Experiment Roadmap

## 목적

Pseudo label을 사용하기 전에 공식 human-labeled 데이터만으로 가능한 성능을 충분히
확보한다. 강한 clean teacher를 만든 뒤 pseudo label을 생성하면 자동 라벨의 누락과
오검출도 줄일 수 있다.

이 문서의 실험 번호는 현재 결과에 따른 잠정 순서다. 각 실험이 실패하면 해당 변경을
버리고, 지금까지 local validation으로 확인된 가장 좋은 clean 후보에서 다음 실험을
시작한다. 가장 최근 모델이 자동으로 다음 baseline이 되는 것은 아니다.

## 용어

### Clean 모델

이 프로젝트에서 clean 모델은 공식 train/val의 human annotation만 사용해 학습한 모델을
뜻한다. 이미지 정제 전용 모델이라는 뜻이 아니다.

- Clean 학습 데이터: 사람이 polygon을 붙인 공식 데이터
- Pseudo 학습 데이터: teacher 모델이 polygon을 자동 생성한 외부 이미지
- Clean validation: 사람이 붙인 정답으로 평가하는 official val 또는 clean CV fold

자동 생성 polygon을 detection supervision으로 사용한 순간부터 해당 모델은 pseudo-label
student로 구분한다. `pseudo_label/` 폴더의 이미지를 polygon 없이 SSL에만 사용하는 것은
pseudo-label 학습이 아니며, `SSL-adapted clean-supervised model`로 따로 기록한다. 이 모델은
detection fine-tuning에서 human annotation만 사용하지만 strict clean-only 모델과 달리 외부
또는 평가 이미지의 unlabeled representation을 사용한다.

### LR

LR은 learning rate의 약자다. 한 번의 gradient update에서 모델 가중치를 얼마나 크게
움직일지 정하는 값이다.

- 너무 크면 좋은 지점을 지나치거나 validation 점수가 흔들릴 수 있다.
- 너무 작으면 학습이 느리고 제한된 epoch 안에 충분히 수렴하지 못할 수 있다.

현재 초기 LR은 `0.001`이다. 현재 scheduler는 `StepLR(step_size=100)`인데 학습은 10
epoch뿐이므로 LR 감소가 한 번도 발생하지 않는다. V0 TensorBoard 기록에서도 LR은 학습
내내 `0.001`로 유지됐다.

### Cosine decay

Cosine decay는 학습 초기에 비교적 큰 LR로 이동하고, 학습 후반으로 갈수록 cosine 곡선
모양으로 LR을 부드럽게 줄이는 schedule이다. 후반에는 작은 LR로 가중치를 미세 조정할
수 있다.

Cosine decay가 항상 성능을 올리는 것은 아니다. 현재처럼 동작하지 않는 scheduler와
비교해 late-epoch 안정성과 H-Mean이 실제로 개선되는지 실험으로 판단한다.

### Checkpoint

Checkpoint는 특정 학습 시점의 모델 가중치와 학습 상태를 저장한 `.ckpt` 파일이다.
학습 중 epoch마다 성능이 다르므로 어떤 checkpoint를 보존하고 제출에 사용할지가
중요하다.

V0 코드는 `val/loss`가 낮은 checkpoint 세 개만 저장했다. V2부터는 대회 목표인 CLEval
H-Mean에 맞춰 아래 artifact를 남긴다.

- `val/hmean` 상위 checkpoint, `mode=max`
- 마지막 epoch checkpoint
- 필요하면 기존 `val/loss` 상위 checkpoint도 진단용으로 보존

Checkpoint 저장 기준 변경은 gradient나 학습 데이터에는 영향을 주지 않는다. 좋은
epoch의 파일을 놓치지 않기 위한 실험 인프라 수정이다.

### Self-Supervised Learning

이 문서와 대회 규정에서 SSL은 `Self-Supervised Learning`을 뜻한다. 이미지 자체에서
학습 목표를 만들기 때문에 polygon이나 transcription을 사용하지 않는다. Contrastive
learning은 같은 이미지의 두 변형을 가까운 representation으로 만들고, masked-image
modeling은 가린 patch의 pixel 또는 feature를 복원한다.

대회 규정은 평가 데이터의 시각화, TTA, SSL 활용을 허용하며 자동화되지 않은 인위적
labeling을 금지한다. 따라서 test 413장을 정답 없이 SSL에 사용하는 transductive learning은
허용된다. 사람이 test polygon을 그리거나 예측 polygon을 수정해 학습에 사용하는 것은
금지한다. 모든 활용은 본 대회 참여 목적으로 제한한다.

이 프로젝트의 SSL 후보 데이터는 다음과 같다.

- 공식 train/val 이미지: SSL 단계에서는 human polygon을 읽지 않음
- test 413장: label 없이 target-domain adaptation에 사용 가능
- `data/pseudo_label/`의 SROIE, WildReceipt, CORD-v2 이미지: 자동 polygon과 분리해 image-only
  SSL pool로 사용 가능

SSL이 held-out validation 이미지 자체를 label 없이 미리 보면 평가는 inductive가 아니라
transductive validation이다. 규정상 허용되지만 기존 V2B local 점수와 평가 조건이 다르므로
run과 표에 반드시 `transductive`라고 표시한다.

### Control과 Pilot

Control은 새 변경의 효과를 비교하는 기준 실험이다. V10 SSL pilot의 현재 control은 가장
좋은 ResNet18 single model인 V2B다. V7 augmentation이 ResNet18 recipe에서 채택되면 그
supervised recipe를 V10 control로 승격한다. Control은 ImageNet-pretrained ResNet18에서
시작해 human-labeled train으로 DB detector를 supervised 학습한다.

Pilot은 비용이 큰 방법을 전체 K-fold에 적용하기 전에 한 split과 고정된 budget으로 가능성과
효과를 확인하는 소규모 실험이다. SSL pilot은 algorithm, unlabeled pool, augmentation,
pretraining budget을 한 번 고정하고 V2B와 동일한 supervised fine-tuning 조건으로 비교한다.
Pilot이 실패하면 fold 전체를 다시 학습하는 비용을 쓰지 않는다.

### Encoder와 DB detector

현재 전체 text detector는
[OCRModel](../baseline_code/ocr/models/architecture.py)에서 다음처럼 조립된다.

```text
image
-> ResNet encoder
-> multi-scale feature maps
-> UNet-style decoder
-> DBHead
-> text probability/threshold maps
-> polygon post-processing
```

DBNet 또는 DB detector는 encoder만을 뜻하지 않고 이 전체 detection pipeline을 가리킨다.
Encoder는 입력 이미지를 여러 해상도의 feature map으로 바꾸는 앞부분이다. 얕은 층은 획과
경계 같은 local pattern을, 깊은 층은 글자와 영수증 layout 같은 semantic pattern을 담는다.

SSL은 먼저 ResNet18 encoder를 unlabeled 이미지로 학습해 encoder checkpoint를 만든다. 이후
동일한 ResNet18 구조로 DB detector를 생성하고 그 encoder 자리에 SSL weight를 로드한다.
Decoder와 DBHead는 기존과 같은 초기화와 seed를 사용하고, human polygon으로 encoder를
포함한 detector 전체를 fine-tuning한다. 즉 DBNet 안에 별도 모델을 삽입하는 것이 아니라,
DBNet을 구성하는 encoder의 시작 weight를 ImageNet weight에서 domain-SSL weight로 교체한다.

## 왜 Public이 아니라 local CLEval로 선택하는가

Official val은 404장 모두에 human polygon 정답이 있어 같은 조건으로 반복 평가하고
precision, recall, H-Mean 및 이미지별 오류를 분석할 수 있다. Local evaluator도 대회와
같은 detection-only CLEval POLY 조건을 사용한다.

Public leaderboard는 test 약 절반의 숨겨진 정답만 사용하며 어떤 이미지가 Public인지
알 수 없다. 여러 설정을 제출하고 가장 높은 Public 점수를 고르면 그 약 206장에 우연히
잘 맞는 threshold, augmentation, seed를 선택할 수 있다. 이 선택은 나머지 Private 약
207장에 일반화되지 않을 수 있다.

따라서 역할을 분리한다.

- Local CLEval: 실험 비교, 설정 채택/폐기, threshold 선택
- Clean K-fold: official val 반복 사용에 따른 local overfit과 split 운을 점검
- Public leaderboard: local에서 이미 선택한 소수 후보의 분포 일치 여부 확인
- Private leaderboard: 대회 종료 후 최종 평가이며 튜닝 신호로 사용할 수 없음

Local validation도 무한히 반복하면 과적합할 수 있으므로, 상위 설정은 K-fold 평균과
분산으로 다시 확인한다.

## 공통 실험 규칙

모든 실험은 다음 원칙을 따른다.

1. 한 번에 하나의 주된 가설만 바꾼다.
2. 같은 checkpoint를 비교할 때 primary 평가는 같은 post-processing 설정을 사용한다.
3. H-Mean뿐 아니라 precision, recall, 이미지별 prediction 수를 함께 기록한다.
4. invalid polygon, 누락 파일, 500개 제한을 검사한다.
5. GPU memory, 학습 시간, 추론 시간과 wandb run/config를 남긴다.
6. Local 근거 없이 Public 점수만 오른 변경은 채택하지 않는다.
7. 다음 실험은 직전 모델이 아니라 현재까지 검증된 최고 clean 후보에서 시작한다.

## 실험 3: V2 Resolution 896

상세 계획:
[V2 Resolution 896 Plan](20260712-v2-resolution-896/README.md)

주요 변경:

- train/validation/test/predict 입력 크기 `640 -> 896`
- 나머지 architecture, optimizer, epoch, seed는 고정
- primary post-processing은 `box_thresh=0.25`

결과별 다음 행동:

| 결과 | 결정 |
| --- | --- |
| H-Mean과 recall 개선, precision 유지 | 896을 clean baseline 후보로 채택 |
| Recall 개선, precision 하락으로 H 정체 | local에서 `box_thresh=0.30`을 보조 확인 |
| H가 비슷하지만 비용이 크게 증가 | 640 유지 또는 736 절충안을 별도 검토 |
| H가 명확히 하락 | 896 폐기, 640 clean baseline으로 복귀 |

1024는 896에서 명확한 개선 추세가 있을 때만 추가한다. 해상도 후보를 모두 Public에
제출해 고르지 않는다.

실행 결과:

- 상태: 완료, Public H/P/R `0.9603 / 0.9667 / 0.9556`
- Best epoch: 8
- Official-val H/P/R: `0.9615 / 0.9638 / 0.9611`
- V1 대비 H/P/R 변화: `+0.0367 / +0.0139 / +0.0555`
- Peak GPU memory: 약 17.4 GB, batch 16 유지
- 결정: 896을 현재 best clean single-model 설정으로 채택

Public H는 local보다 `0.0012`만 낮았고 V1 Public보다 `0.0418` 높았다. 896과 threshold는
Public을 보기 전에 local에서 고정했으므로 현재 결과에는 Public-driven overfit 증거가
없다. Private 결과를 보장하지는 않으며, 이 Public 결과로 threshold를 다시 조정하지
않는다.

896의 개선이 명확하므로 1024 controlled sub-experiment 조건은 충족됐다. 1024를 실행하면
실험 3B로 취급하고 다른 학습 설정은 바꾸지 않는다. 1024를 생략하면 896을 기준으로
실험 4 cosine LR schedule을 진행한다.

실험 3B 실행 결과:

- 상세 기록: [V2B Resolution 1024](20260713-v2b-resolution-1024/README.md)
- Best epoch: 8
- Independent official-val H/P/R: `0.964760 / 0.969976 / 0.961422`
- 896 대비 H/P/R: `+0.003253 / +0.006153 / +0.000288`
- Peak GPU memory: 약 22.5 GB, batch 16 유지
- 결정: 1024를 현재 best clean single-model 해상도로 채택

이 결정은 Public 결과를 보기 전에 local CLEval로 내렸다. 1024 제출 후보는 milestone
확인용으로만 사용하며, Public 결과로 896과 1024 선택을 뒤집지 않는다.

실험 3B Public 결과:

- Public H/P/R: `0.9621 / 0.9754 / 0.9520`
- V2B local 대비 H/P/R: `-0.002660 / +0.005424 / -0.009422`
- V2 Public 대비 H/P/R: `+0.0018 / +0.0087 / -0.0036`

Local과 Public에서 모두 1024의 H-Mean이 896보다 높아 해상도 선택은 유지한다. 다만
Public에서는 향상이 local보다 작고 precision 중심으로 나타났다. 이는 1024가 만드는 더
정확한 경계와 false positive 감소 효과일 수 있지만, 숨겨진 Public 이미지별 오류가 없어
원인을 확정할 수는 없다. Public의 낮은 recall만 보고 `box_thresh`를 내리지 않는다.
다음 실험 순서는 유지하되 실험 5에서는 흐림, 저대비 글자의 local recall 개선 여부를
우선 진단한다. 최종 checkpoint가 정해진 뒤 실험 7에서 local CLEval로 후처리를 보정한다.

## 실험 4: V3 Cosine LR Schedule

목적:

- 현재 10 epoch 동안 동작하지 않는 StepLR를 실제로 감소하는 schedule로 바꾼다.

통제 조건:

- 실험 3까지 선택된 해상도와 모델 구조 사용
- optimizer는 Adam 유지
- 초기 LR `0.001` 유지
- epoch 수 10 유지
- augmentation과 backbone은 변경하지 않음

후보 설정:

- `CosineAnnealingLR`
- `T_max=10`
- `eta_min=1e-6`을 시작 후보로 사용

비교:

- 기존 고정 LR 성격의 StepLR 결과 대 cosine decay 결과
- 같은 `box_thresh`에서 official-val CLEval 비교
- epoch별 H-Mean 안정성과 마지막 구간의 score 하락 여부 확인

Cosine이 개선되지 않으면 기존 Adam/StepLR 설정으로 돌아간다. 이 실험에서 AdamW나
긴 epoch를 동시에 적용하지 않는다.

실행 결과:

- 상세 기록: [V3 Cosine LR At Resolution 1024](20260713-v3-cosine-1024/README.md)
- Best epoch: 8
- Independent official-val H/P/R: `0.959166 / 0.960237 / 0.960184`
- V2B control 대비 H/P/R: `-0.005595 / -0.009739 / -0.001237`
- 결정: cosine 변경 폐기, 1024 + 기존 StepLR의 V2B checkpoint 유지

Cosine은 초반 일부 epoch에서 빠르게 수렴했지만 후반 최고점이 control에 못 미쳤다.
`T_max`나 epoch를 추가로 바꾸면 별도 가설이 되므로 지금 연속 튜닝하지 않는다. 다음
실험은 V2B checkpoint의 학습 recipe에서 photometric augmentation만 변경한다.

## 실험 5: V4 Photometric Augmentation

목적:

- 실제 촬영에서 생기는 흐림, 압축, 조도 차이에 대한 일반화를 높인다.

현재 baseline의 학습 변형은 resize, padding, horizontal flip이 중심이다. 다음 변형을
약한 확률의 하나의 camera-degradation policy로 비교한다.

실험 4가 폐기됐으므로 control은 V2B의 1024 + Adam/StepLR 설정이다.

- Brightness/contrast/gamma
- Gaussian blur 또는 motion blur
- JPEG compression
- 약한 shadow/uneven illumination 후보

Rotation, perspective, aggressive crop, MixUp/CutMix은 이 실험에 섞지 않는다. Photometric
policy가 이득일 때 geometric augmentation을 후속 소실험으로 분리할 수 있다.

채택 조건:

- Local H-Mean 또는 K-fold 후보 성능이 개선된다.
- 흐린 글자 recall이 오르면서 선, 바코드, 배경 false positive가 과도하게 늘지 않는다.
- V2B Public의 높은 precision과 상대적으로 낮은 recall은 보조 진단 신호로만 사용하고,
  augmentation 채택 여부는 official-val CLEval과 이미지별 오류로 결정한다.

실행 결과:

- 상세 기록: [V4 Photometric Augmentation](20260714-v4-photometric-1024/README.md)
- Independent epoch-9 macro H/P/R: `0.962595 / 0.968465 / 0.958750`
- Independent global H/P/R: `0.958719 / 0.963323 / 0.954158`
- V2B 대비 macro H/P/R: `-0.002165 / -0.001511 / -0.002672`
- V2B 대비 global H/P/R: `-0.003500 / -0.004187 / -0.002827`
- 결정: combined photometric policy 폐기, Public 제출 생성하지 않음

Epoch 8에서는 macro recall이 V2B보다 `+0.001714` 높았지만 precision 손실로 H가
`-0.002432` 낮았다. Recall 가능성은 확인했으나 현재 단일 모델 목표에는 불충분하다.
Official val에 다섯 augmentation을 연속 개별 튜닝하지 않고, 실험 6은 V2B의 기존
augmentation으로 돌아가 backbone만 ResNet34로 변경한다.

## 실험 6: V5 ResNet34 Backbone

목적:

- 현재 pretrained ResNet18보다 표현력이 높은 backbone의 clean-data 효과를 측정한다.

첫 후보는 ResNet34다. ResNet18과 feature channel 구성이 같아 현재 decoder 연결을
유지할 수 있어 구현 위험이 낮다.

통제 조건:

- V2B에서 채택된 1024 입력 크기, Adam/StepLR, 기존 augmentation 사용
- decoder, head, loss, epoch와 primary post-processing 고정
- pretrained weight 사용

ResNet50은 ResNet34의 이득과 비용을 확인한 뒤에만 고려한다. ResNet50은 decoder input
channel 변경과 더 큰 GPU 비용이 필요하다.

실행 결과:

- 상세 기록: [V5 ResNet34 Backbone](20260714-v5-resnet34-1024/README.md)
- Best epoch: 7, macro와 global H가 같은 checkpoint를 선택
- Independent macro H/P/R: `0.964617 / 0.964341 / 0.966831`
- Independent global H/P/R: `0.962185 / 0.960289 / 0.964088`
- V2B 대비 macro H/P/R: `-0.000143 / -0.005635 / +0.005409`
- V2B 대비 global H/P/R: `-0.000034 / -0.007221 / +0.007103`
- 비용: runtime 2,457초, peak allocated memory 23.08 GiB, RTX 3090의 96.15%
- 결정: 기본 clean single model은 V2B 유지, V5는 recall-diverse 후보로 보존

H는 사실상 동률이지만 ResNet34는 precision을 낮추고 recall을 높였다. 더 큰 checkpoint와
GPU 비용까지 고려하면 고정 threshold 단일 모델 교체 근거는 없다. ResNet50으로 바로
확장하지 않는다. 다만 이 precision-recall 위치는 이미 계획된 실험 7의 local 보정 대상으로
가치가 있으므로 V5 epoch 7을 조건부 후보로 넘긴다. Public 제출은 생성하지 않았다.

## 업데이트된 실행 순서

평가 데이터의 시각화, TTA, SSL 사용이 허용된다는 규정을 반영해 V6 이후 순서를 다음처럼
고정한다. D0는 model artifact가 없는 분석 단계이므로 version 번호를 쓰지 않는다. V6부터는
실행 순서대로 연속 번호를 사용하고 한 version에 하나의 주된 가설만 둔다.

| 순서 | 단계 | 핵심 질문 | 다음 단계로 넘기는 결과 |
| ---: | --- | --- | --- |
| 1 | D0 Data Audit | 실제 train/val/test 분포와 오류 유형은 무엇인가? | augmentation, TTA, split, SSL transform 후보 |
| 2 | V6 Post-processing | V5의 높은 recall을 precision 손실 없이 살릴 수 있는가? | 최종 clean single checkpoint |
| 3 | V7 D0-guided Augmentation | D0가 찾은 한 가지 domain gap을 학습으로 줄일 수 있는가? | augmentation 채택 또는 명시적 skip |
| 4 | V8 Scale TTA | 같은 model의 multi-scale map 평균이 안정적으로 이득인가? | 고정된 TTA 후보와 구현 |
| 5 | V9 Existing-model Ensemble | V2B와 recall-diverse model이 서로 보완되는가? | 고정된 model-fusion 후보 |
| 6 | V10 SSL Pilot | Domain self-supervised encoder가 control 초기화보다 좋은가? | V11 fold의 encoder 초기화 |
| 7 | V11 K-fold/Teacher | 선택 recipe가 split에 안정적이며 teacher로 충분한가? | OOF 근거와 final clean teacher |
| 8 | V12 Pseudo Student | 자동 polygon이 clean control보다 실제로 도움이 되는가? | pseudo 사용/미사용 final 후보 |
| 9 | V13 Final Fusion/Calibration | 최종 model에 맞는 TTA, ensemble, threshold는 무엇인가? | Stable/Aggressive 제출 후보 |

이 순서의 핵심은 D0로 학습 가설을 제한하고, 재학습 없는 V6/V8/V9를 이용해 후보를 줄인 뒤,
V10 SSL의 효과를 pilot으로 확인한 경우에만 비용이 큰 V11 K-fold를 수행하는 것이다. Pseudo
label은 가장 강한 clean teacher가 나온 후 V12에서 생성하고, confidence calibration은 model과
fusion 구성이 모두 결정된 V13에서 다시 수행한다.

## D0: Train/Val/Test Data Audit

목적:

- Label이 없는 test를 포함해 촬영 조건과 domain shift를 정량화한다.
- V4처럼 여러 augmentation을 근거 없이 묶지 않고 다음 가설의 종류와 강도를 정한다.
- V11 split과 V10 SSL augmentation이 실제 데이터 분포를 보존하도록 기준을 만든다.

분석 순서:

1. 이미지 크기, 종횡비, 밝기, 대비, blur, entropy, edge density를 train/val/test에서 계산한다.
2. Train/val polygon의 개수, 짧은 변, 면적, 각도, vertex 수와 text 밀도를 계산한다.
3. 품질 지표와 V2B per-image CLEval의 상관관계를 분석한다.
4. V2B와 V5를 test 전체에 추론해 prediction 수, confidence, 크기와 model disagreement를
   계산한다.
5. 각 지표의 분위수와 cluster별 contact sheet를 만들되 test polygon을 사람이 만들거나
   고치지 않는다.

D0의 결과는 가설이지 채택 근거가 아니다. 예를 들어 test blur가 train보다 강하면 blur
단독 augmentation 후보를 만들 수 있지만, 실제 채택은 clean validation 또는 fold CLEval
A/B 결과로 결정한다. D0는 다음에 직접 연결된다.

- Augmentation: 관측된 blur, compression, illumination, rotation 범위만 단독 후보로 만든다.
- TTA: 실제 orientation과 작은-text 분포에 따라 scale/rotation 후보를 제한한다.
- K-fold: source, word 수, 종횡비, blur와 밝기 strata를 만들고 near-duplicate는 같은 fold에
  둔다.
- SSL: text stroke를 지우지 않는 crop, blur, color transform 범위를 정한다.
- Pseudo filtering: source/품질 cluster별 confidence와 prediction-count 이상치를 찾는다.

### D0 실행 결과와 Gate 결정

상세 기록은 [D0 Train/Val/Test Data Audit](20260714-d0-data-audit/README.md)에 있다.

- 상태: `completed`, 새 model 학습과 Public 제출 없음
- 공식 train/val/test는 밝기, 대비, blur proxy, edge density, 종횡비, 해상도 분포가 거의
  같았다. Train-test 최대 KS statistic은 `0.0655`였다.
- V2B 실패의 가장 큰 관측 신호는 1024 변환 후 12 px 미만 글자 비율이었다. H와
  `rho=-0.1945`, recall과 `rho=-0.2061`이었다.
- 밝기, 대비, blur 등 photometric 지표와 H의 상관 절댓값은 모두 `0.067` 이하였다.
- V2B/V5 prediction Jaccard는 val/test 모두 약 `0.873`이었고, validation에서 V5가 이긴
  이미지는 166장, V2B가 이긴 이미지는 220장이었다.
- Auxiliary 세 source의 밝기, 해상도, edge/blur 분포가 크게 달라 V10/V12에서 source
  identity를 유지하고 source-balanced sampling을 사용한다.

이에 따라 V7은 `skipped: no D0-supported train/test augmentation gap`으로 결정한다. V6 뒤에
재학습 없이 V8로 이동한다. 작은 글자 실패가 확인됐으므로 V8의 첫 비교는 원래 계획한
`896+1024` 대신 `1024+1152`로 바꾼다. 이것은 Public 결과가 아니라 D0 validation 오류
분석으로 내린 변경이다. V9 V2B+V5 후보는 유지하되 map fusion의 local CLEval을 보기 전에는
ensemble을 채택하지 않는다.

## V6: Local Post-processing Recalibration

목적:

- V5 ResNet34의 높은 recall과 낮은 precision이 model weakness인지 threshold calibration
  차이인지 확인하고 V7의 provisional clean single checkpoint를 고른다.

이 단계는 재학습 없이 제공 val 404장을 추론한다.

1. V5 epoch 7의 `box_thresh=0.30`, `thresh=0.30`을 평가한다.
2. 0.30이 V5의 macro/global H를 모두 개선할 때만 `box_thresh=0.35`를 평가한다.
3. V5가 V2B H를 넘지 못하면 V5 branch를 종료하고 V2B를 유지한다.
4. V2B pixel `thresh` 보정은 V5 branch와 섞지 않고 필요할 때 별도로 수행한다.

모든 threshold 조합을 sweep하거나 Public에 각각 제출하지 않는다. V6의 threshold는 단일
model 비교용이며, TTA/ensemble/pseudo student는 score map calibration이 달라지므로 final
단계에서 별도로 다시 맞춘다.

실행 결과:

- 상세 기록: [V6 V5 Post-processing](20260714-v6-v5-postprocess/README.md)
- V5 `box_thresh=0.30` macro H/P/R: `0.960568 / 0.966257 / 0.957238`
- V5 `box_thresh=0.30` global H/P/R: `0.958472 / 0.963035 / 0.953953`
- V5 box 0.25 대비 macro delta: H `-0.004049`, P `+0.001916`, R `-0.009593`
- V5 box 0.25 대비 global delta: H `-0.003713`, P `+0.002746`, R `-0.010135`
- 결정: 0.30 폐기, 사전 gate에 따라 0.35 미실행, V2B를 single control로 유지

V5의 recall 우위는 box threshold 하나로 precision 쪽에 유리하게 재보정되지 않았다. V5 box
0.25 checkpoint는 V9의 recall-diverse ensemble member로만 보존한다. V7은 D0 결과로 이미
skip되었으므로 다음 실행 단계는 V8 `1024+1152` scale TTA다.

## V7: D0-guided Single Augmentation

목적:

- D0가 train과 test 사이에서 확인한 가장 큰 한 가지 촬영 조건 차이를 supervised training
  augmentation으로 줄일 수 있는지 검증한다.
- 실패한 V4처럼 brightness, gamma, blur, motion blur, JPEG를 한 번에 묶지 않는다.

실행 gate:

- D0에서 train/val/test 사이에 명확하고 해석 가능한 차이가 있어야 한다.
- 제공 val의 V2B per-image failure와 같은 품질 지표 사이에 일관된 관계가 있어야 한다.
- 위 근거가 없으면 V7은 `skipped: no D0-supported augmentation`으로 기록하고 V8로 간다.

실행 방법:

1. V6에서 선택한 single-model training recipe를 control로 둔다.
2. Blur, compression, illumination, rotation/perspective 중 D0 근거가 가장 강한 family 하나만
   고른다.
3. 실제 관측 분위수에 맞춰 transform 강도와 확률을 사전에 고정한다.
4. Resolution, backbone, optimizer, epoch, seed, post-processing은 control과 동일하게 둔다.
5. Macro/global H와 품질 strata별 H/P/R이 모두 악화되지 않을 때만 채택한다.

V7이 채택되면 이후 V8과 V10의 supervised control은 이 checkpoint/recipe로 갱신한다. V7이
실패하거나 skip되면 V6에서 선택한 V2B 또는 V5 recipe를 유지한다.

D0 실행 결과 official train/test domain gap과 photometric failure correlation이 모두 gate를
통과하지 못했다. 따라서 이 단계는 새 학습 없이 skip하고, V6에서 선택한 checkpoint를 V8
control로 넘긴다.

## V8: Scale TTA Screen

목적:

- 재학습 없이 같은 model의 서로 다른 입력 scale이 small/faint text와 false positive를
  상호 보완하는지 확인한다.
- V11 final teacher와 V13 final candidate에서 재사용할 map-fusion 코드를 검증한다.

실행 방법:

1. V7까지 선택된 single checkpoint의 1024 추론을 control로 둔다.
2. D0의 small-text failure 근거에 따라 `1024 + 1152` map을 원본 좌표로 복원해 equal
   average한다.
3. Recall은 오르지만 precision 손실로 H가 낮을 때만 `896 + 1024`를 한 번의 lower-scale
   fallback으로 확인한다. 첫 후보가 macro/global H를 모두 개선하면 fallback은 실행하지 않는다.
4. Flip과 rotation은 scale과 동시에 넣지 않고 D0 근거가 있을 때 `V8-ROT`로 분리한다.

각 augmentation의 polygon을 합치지 않는다. Padding을 제외한 valid probability map을 원본
좌표계로 복원해 평균하고 DB post-processing을 한 번만 적용해 duplicate와 500개 cap 위험을
줄인다. V8은 가능성 선별 단계이므로 최종 threshold를 광범위하게 최적화하지 않는다.

실행 결과:

- 상세 기록: [V8 Scale TTA](20260714-v8-scale-tta/README.md)
- `1024+1152` macro H/P/R: `0.966840 / 0.972512 / 0.962827`
- 1024 control 대비 macro delta: `+0.002055 / +0.002533 / +0.001364`
- `1024+1152` global H/P/R: `0.965130 / 0.971017 / 0.959314`
- 1024 control 대비 global delta: `+0.002881 / +0.003501 / +0.002274`
- Macro/global H paired bootstrap 개선 확률: `0.9967 / 0.9998`; 두 95% CI 하한 모두 양수
- 결정: `1024+1152` 채택, lower-scale fallback과 threshold sweep 미실행

V8은 clean single model을 교체한 것이 아니라 inference candidate를 하나 통과시킨 것이다.
다음 V9에서는 V8 TTA를 섞지 않고 V2B/V5 model-map fusion만 평가해 architecture diversity의
효과를 분리한다. 최종 single model이 결정된 뒤 V13에서 두 fusion 축의 결합을 재평가한다.

## V9: Existing-model Probability-map Ensemble

목적:

- V2B의 높은 precision과 V5 epoch 7의 높은 recall이 같은 이미지에서 실제로 보완되는지
  기존 checkpoint만으로 확인한다.
- 같은 model의 입력 변형을 평균하는 V8 TTA와 다른 model의 inductive bias를 평균하는 model
  ensemble을 분리해 효과를 측정한다.

기존 checkpoint 중 첫 후보는 V2B와 V5 epoch 7이다. V4 epoch 8은 D0의 per-image 분석에서
V2B/V5가 놓친 true region을 독립적으로 보완할 때만 `V9-V4`로 추가한다. Precision과 recall이
모두 낮았던 V3는 제외한다.

1. V2B single map을 control로 둔다.
2. V2B와 V5 map의 사전 고정 equal average를 한 번 평가한다.
3. Precision 손실로 H가 낮으면 한 번의 V2B-heavy fallback만 확인하고 branch를 종료한다.
4. V4를 포함한 다중 weight sweep은 하지 않는다.

V9도 선별 단계다. V12 student가 최종 single model이 되면 V8 TTA와 V9 ensemble을 V13에서
다시 검증한다.

실행 결과:

- 상세 기록: [V9 Model Ensemble](20260714-v9-model-ensemble/README.md)
- V2B/V5 equal macro H/P/R: `0.967266 / 0.969083 / 0.967177`
- V2B 대비 macro delta: `+0.002481 / -0.000896 / +0.005714`
- Equal global H/P/R: `0.965090 / 0.965880 / 0.964301`
- V2B 대비 global delta: `+0.002841 / -0.001636 / +0.007262`
- Macro/global H paired bootstrap 개선 확률: `0.9998 / 0.9990`; 두 95% CI 하한 모두 양수
- 결정: equal ensemble 채택, V2B-heavy fallback과 추가 weight/model sweep 미실행

V9는 V5가 이기던 이미지에서 H를 크게 높였지만 V2B가 이기던 이미지에서는 낮아져 diversity가
sample-dependent함을 확인했다. V8과 V9는 macro/global 우위가 갈리므로 둘 다 final fusion
후보로 보존하고 즉시 결합하지 않는다. 다음 실행 단계는 V10 domain SSL pilot이다.

## V10: Domain Self-supervised Pilot

실행 문서: [V10 Domain Self-supervised Pilot](20260714-v10-domain-ssl/README.md)

목적:

- 평가 이미지와 외부 영수증 이미지를 label 없이 본 ResNet18 encoder가 기존 ImageNet
  encoder보다 text detection fine-tuning에 좋은 시작점인지 확인한다.
- 효과가 불확실한 SSL을 모든 V11 fold에 적용하기 전에 작은 비용으로 중단 가능하게 만든다.

Control과 pilot은 다음 한 가지 차이만 갖는다.

```text
Control C0
ImageNet ResNet18
-> V7까지 선택된 ResNet18 DB detector supervised fine-tuning

Pilot P0
ImageNet ResNet18
-> receipt image-only self-supervised pretraining
-> 같은 DB detector supervised fine-tuning
```

첫 algorithm 후보는 현재 ResNet18과 작은 batch에 연결하기 쉬운 MoCo v2다. 검증된 library를
사용하고 core contrastive algorithm을 새로 구현하지 않는다. D0 결과로 text stroke를 지우지
않는 augmentation을 고정하고, MoCo/SimSiam/MAE를 같은 제공 val에 연속 탐색하지 않는다.

SSL pool은 공식 train/val/test와 `pseudo_label/` 이미지다. SSL dataloader는 polygon과
transcription을 읽지 않는다. Fine-tuning은 original train human polygon만 사용하고 제공 val
human polygon은 평가 시에만 사용한다. 제공 val 이미지가 SSL에서 label 없이 보였으므로 이
결과는 `transductive local`로 표시한다.

공정한 비교를 위해 supervised resolution, decoder, DBHead, loss, optimizer, epoch, seed,
augmentation, post-processing을 control과 동일하게 유지한다. SSL weight는 ResNet18 encoder에
로드하고 detector 전체를 fine-tuning한다.

채택 조건:

- Independent reload 기준 macro와 global H가 control보다 모두 높아야 한다.
- Precision 또는 recall 한쪽의 큰 붕괴와 invalid/excess prediction 증가가 없어야 한다.
- 차이가 사실상 동률이면 추가 복잡성과 K-fold 재학습 비용 때문에 ImageNet control을 유지한다.
- Public leaderboard 결과는 SSL algorithm이나 checkpoint 선택에 사용하지 않는다.

V10을 V11보다 먼저 두는 이유는 명확하다. Pilot이 통과한 뒤에야 같은 SSL encoder로 V11
fold들을 학습하면 되고, V11 이후 SSL을 시작해 fold 전체를 두 번 학습하는 낭비를 피할 수
있다.

실행 결과:

- 상세 기록: [V10 Domain Self-supervised Pilot](20260714-v10-domain-ssl/README.md)
- Independent macro H/P/R: `0.964897 / 0.964765 / 0.966682`
- Independent global H/P/R: `0.962613 / 0.961242 / 0.963987`
- V2B 대비 macro/global H delta: `+0.000112 / +0.000363`
- Paired bootstrap H 개선 확률: macro `0.5552`, global `0.6396`; 두 95% CI가 0 포함
- Recall은 유의하게 올랐으나 precision이 거의 같은 크기로 내려간 trade-off
- 결정: statistical tie로 V10 reject, V11은 ImageNet ResNet18 initialization 사용

MoCo pretraining과 encoder load는 기술적으로 정상 동작했다. 따라서 성능 실패 뒤 SSL
algorithm을 제공 val에 연속 탐색하지 않는 원칙에 따라 `V10-ALT`는 실행하지 않는다. 다음
실행 단계는 V11A group-aware K-fold manifest 생성이다.

V10의 detailed failure mechanism과 조건부 `V10-R1`은
[SSL Effect Analysis And Revisit Plan](20260714-v10-domain-ssl/ssl_effect_analysis.md)에
기록한다. 즉시 epoch나 algorithm을 늘리지 않고 V11/V12 근거가 생긴 뒤에만 재검토한다.

## V11: K-fold Stability And Final Clean Teacher

V11은 OOF recipe 검증과 final fold ensemble 생성을 구분한다. OOF는 측정할 수 있지만 같은
K-fold 데이터에서 모든 fold model을 평균한 ensemble lift는 공정하게 측정할 수 없다는 점을
기록한다.

### V11A Split Manifest

실행 문서: [V11A Group-aware K-fold Split](20260714-v11a-kfold-split/README.md)

Train+val human-labeled 데이터를 대상으로 group-aware stratified K-fold manifest를
`data/splits/`에 만든다. D0에서 정한 source, word 수, 종횡비, blur, 밝기 strata를 최대한
균형화하고 near-duplicate 또는 같은 receipt source group은 같은 fold에 둔다.

Fold 수는 GPU 시간과 분산 추정 사이에서 사전에 고정한다. Fold 수를 Public 또는 중간 local
점수에 따라 바꾸지 않는다.

실행 결과:

- 3,676장을 `735 / 735 / 736 / 735 / 735`의 5개 fold로 배정
- 기존 val 출처는 fold당 `81 / 81 / 80 / 80 / 82`장
- 자동 후보 96쌍 중 high-confidence near-duplicate 11쌍을 group으로 묶음
- Group leakage와 accepted-pair leakage 모두 0
- 최대 stratum proportion delta `0.001159`, 최대 continuous-feature mean z `0.06513`
- 동일 명령 재실행 시 manifest SHA-256 일치
- 결정: V11B의 고정 OOF manifest로 채택

초기 `shuffle=True` 구현은 model score를 보기 전에 structural balance gate를 실패해 폐기했다.
Seed를 바꿔 좋은 split을 찾지 않고 deterministic `shuffle=False` SGKF로 층화 동작을 수정했다.
이제 split은 잠그며 이후 OOF 결과에 따라 다시 만들지 않는다.

### V11B OOF Recipe Evaluation

각 fold model은 해당 held-out fold의 human polygon을 supervised 학습에서 사용하지 않는다.
각 이미지의 OOF prediction은 그 이미지의 human label을 보지 않은 fold model 하나에서만
가져온다. OOF prediction을 합쳐 macro/global H/P/R, fold 평균과 표준편차, 품질 strata별
성능을 기록한다.

V10이 채택되면 shared SSL encoder는 held-out 이미지 pixel을 label 없이 봤을 수 있다.
이 경우 OOF는 strict inductive가 아니라 대회 test-SSL 조건을 모사한 transductive OOF이며,
human label leakage와 구분해 기록한다.

OOF는 recipe의 안정성을 측정하지만 final K-fold ensemble의 추가 이득을 그대로 측정하지는
않는다. 예를 들어 fold A 이미지에서 A를 제외하고 학습한 model만 OOF에 사용할 수 있다.
다른 fold model들은 A의 human label을 학습에서 봤으므로 A에서 모두 평균하면 leakage다.

### V11C Final Fold Ensemble Teacher

Recipe가 fold 평균과 분산 기준을 통과하면 모든 fold model의 probability map을 equal
average해 test와 외부 pseudo 이미지에 적용한다. Test와 외부 이미지는 어떤 fold의
supervised human-label 학습에도 들어가지 않았으므로 모든 fold model을 함께 사용할 수 있다.

V8/V9는 map averaging의 local 거동을 검증하고, V11B는 각 구성 recipe의 OOF 안정성을
검증한다. 이 두 근거를 합쳐 V11C를 stable clean teacher로 사용한다. Ensemble lift 자체를 unbiased하게
재려면 각 outer fold를 제외하고 여러 seed 또는 architecture를 학습해 같은 outer fold에서
평가해야 하므로, V8/V9/V11 결과가 불확실할 때만 `V11-NEST`로 비용 높은 nested ensemble
audit을 수행한다.

### V11D Full-data Single Teacher

고정된 recipe로 train+val 전체를 한 번 재학습한 full-data single model도 별도 teacher 후보로
보존한다. Fold ensemble과 full-data model의 test 예측 차이를 기록하되 Public 점수만으로 둘을
선택하지 않는다.

## V12: Automated Pseudo-label Student

목적:

- V11 clean teacher가 외부 영수증 이미지에 만든 자동 polygon을 사용해 실제 DB detection
  supervision을 늘리고 clean control보다 나은 student를 만든다.

### V12A Pseudo Generation And Filtering

V11C teacher로 SROIE, WildReceipt, CORD-v2 이미지의 probability map과 polygon을 생성한다.
Confidence, teacher/checkpoint, source, image quality, polygon geometry metadata를 보존한다.
Filtering은 confidence, 크기, 중복, prediction count 같은 사전 정의 자동 규칙만 사용한다.
사람이 polygon을 추가, 삭제, 이동 또는 수정하지 않는다.

### V12B Student Training

Human-labeled clean data와 filtered pseudo data를 함께 학습한다. Clean:pseudo sampling ratio,
pseudo loss weight, epoch를 한 번에 모두 sweep하지 않고 첫 고정 recipe를 clean control과
비교한다. Pseudo polygon은 validation 정답으로 사용하지 않는다.

### V12C Clean Fine-tuning And Ablation

Pseudo mixed training 후 human-labeled clean data만으로 짧게 fine-tuning한 후보를 만든다.
다음 세 모델을 같은 clean/transductive fold protocol로 비교한다.

- V11 clean teacher 또는 동일 single recipe control
- Pseudo mixed student
- Pseudo mixed 후 clean-only fine-tuned student

Pseudo student가 clean control보다 macro/global H를 안정적으로 높이지 못하면 pseudo branch를
폐기한다. Public 결과만 오른 pseudo model은 채택하지 않는다.

## V13: Final Fusion And Calibration

최종 single model이 결정된 뒤 V8/V9에서 통과한 TTA와 existing/fold ensemble을 다시 평가한다.
Probability map fusion은 score calibration을 바꾸므로 `box_thresh`, pixel `thresh`, 필요하면
`unclip_ratio`를 이 단계에서 좁게 보정한다. 순서는 model 선택, TTA, ensemble, threshold다.

최종 후보는 둘을 유지한다.

- Stable candidate: fold/OOF 근거가 강하고 precision-recall 균형과 500개 cap 위험이 낮음
- Aggressive candidate: local 근거는 있으나 더 높은 recall과 fusion 복잡성을 가짐

## 순서를 이렇게 만든 이유

1. D0를 먼저 해야 실제 분포에 근거해 augmentation, SSL transform, TTA와 split을 정할 수
   있다.
2. V6은 재학습 없이 V5를 살릴 수 있는지 확인하고 뒤 단계의 provisional single control을
   고르는 가장 싼 실험이다.
3. V7은 D0 근거가 있을 때만 한 augmentation을 학습해 V4의 결합-policy 실패를 반복하지
   않는다. 근거가 없으면 명시적으로 skip해 시간과 local overfit을 줄인다.
4. V8과 V9는 재학습 없이 TTA와 model diversity를 각각 분리 측정해 fusion 후보를 좁힌다.
5. V10은 encoder 초기화를 바꾸므로 V11 전에 통과 여부를 정해야 fold 전체 재학습을 피한다.
6. V11은 반복 사용한 제공 val 한 split의 운을 줄이고 가장 안정적인 clean teacher를 만든다.
7. V12는 teacher quality에 직접 의존하므로 V11 뒤에 두되, ablation과 재학습 시간을 남기기
   위해 대회 마지막까지 미루지 않는다.
8. V13 calibration은 model/TTA/ensemble마다 score 분포가 달라지므로 모든 구성원이 정해진
   뒤 마지막에 수행한다.

## 조건부 및 보류 실험 버전

아래 실험은 앞서 제안됐지만 모든 run에 자동으로 포함하지 않는다. Main version에 suffix를
붙여 실행 시점과 gate를 고정한다. Gate가 충족되지 않으면 번호를 지우지 않고 실험 로그에
`skipped`와 이유를 남긴다.

| Version | 실험 | 실행 가능한 시점 | 실행 gate 및 현재 판단 |
| --- | --- | --- | --- |
| `V6-R50` | ResNet50 backbone | V6 뒤, V7 전 | V6 보정 후 V5가 V2B를 명확히 이기고 batch 축소 계획이 있을 때만 실행. 현재 V5 H 동률·VRAM 96.15%라 gate 미충족 |
| `V7-PHOTO` | Blur/JPEG/illumination 중 한 photometric augmentation | D0 뒤, V8 전 | D0 domain gap과 제공 val failure 상관이 같은 family를 지목할 때 V7의 유일한 후보로 실행 |
| `V7-GEO` | Mild rotation 또는 perspective augmentation | D0 뒤, V8 전 | 실제 orientation/perspective 차이가 확인될 때 V7-PHOTO 대신 실행. 둘을 한 run에 섞지 않음 |
| `V8-ROT` | Flip/90도/180도 TTA | V8 scale TTA 뒤 | D0에서 해당 orientation이 존재하고 제공 val 변환 평가가 macro/global H를 모두 높일 때만 실행 |
| `V8-TILE` | Long-receipt tiling TTA | V8 scale TTA 뒤 | 긴 이미지의 작은 text가 주 실패 원인이고 crop overlap/map stitching을 자동 복원할 수 있을 때만 실행 |
| `V8-PHOTO` | Photometric TTA | V8 scale TTA 뒤 | V7-PHOTO가 학습에서 이득이고 같은 transform의 map 평균이 local에서 안정적일 때만 실행. 현재 우선순위 낮음 |
| `V9-V4` | V4 epoch 8을 third ensemble member로 추가 | V9 V2B+V5 뒤 | D0/per-image 분석에서 V4만 맞히는 clean-val region이 확인되고 V2B+V5가 먼저 개선될 때만 실행 |
| `V10-ALT` | SimSiam 또는 masked-image SSL 대체 | V10 뒤, V11 전 | MoCo가 정상 실행됐고 H가 statistical tie여서 현재 gate 미충족, skip. 성능 실패 뒤 algorithm sweep은 하지 않음 |
| `V11-MS` | Same-recipe multi-seed ensemble | V11B 뒤, V11C 전 | Fold variance 또는 seed sensitivity가 크고 추가 학습 비용을 감당할 때 held-out fold를 보지 않은 seed끼리 평가 |
| `V11-NEST` | Nested architecture/seed ensemble audit | V11B 뒤, V11C 전 | Final fold ensemble lift의 unbiased 측정이 의사결정을 바꿀 정도로 중요할 때만 실행. 비용이 가장 큰 검증 |
| `V13-HET` | DB detector와 CRAFT 등 다른 detector의 heterogeneous ensemble | V12 model 선택 뒤, V13 calibration 전 | 기존 fold/model ensemble이 포화되고 다른 detector의 OOF 보완성이 확인될 때만 실행. 현재 구현·비용 우선순위 낮음 |

조건부 실험 중 D0가 직접 결정하는 것은 V7-PHOTO/V7-GEO, V8-ROT/V8-TILE이다. V9-V4는
기존 checkpoint의 보완성, V10-ALT는 SSL pilot의 기술적 적합성, V11-MS/V11-NEST는 fold
분산과 ensemble 검증 필요성, V13-HET는 최종 diversity 부족이 gate다. 이 구분으로 아이디어가
실행 순서에서 사라지지 않으면서도 모든 후보를 제공 val에 무작정 맞추는 것을 막는다.

## 대회 종료 후 Local 성능 근거

Leaderboard 제출 종료 후에는 새로운 모델의 hidden test 성능을 직접 측정할 수 없다. Local
CLEval은 계속 주 metric으로 사용하되, 반복 사용한 official val 404장의 단일 점수만으로
성능 향상을 확정하지 않는다.

1. V0/V1/V2/V2B의 Final leaderboard를 과거 local metric이 hidden data와 같은 방향으로
   움직였는지 확인하는 retrospective calibration에만 사용한다.
2. 재학습이 없는 V8/V9는 같은 404장에 대한 paired per-image 차이와 bootstrap confidence
   interval을 macro/global CLEval에 함께 기록한다.
3. V10 이후 학습 recipe는 V11의 고정 group-aware K-fold OOF에서 fold 평균, 표준편차,
   worst fold와 전체 OOF macro/global H를 비교한다.
4. 별도 locked audit fold를 사용할 경우 기존 checkpoint와 직접 비교하지 않는다. Audit
   이미지를 학습에서 제외한 control과 candidate를 같은 조건으로 새로 학습해야 한다.
5. 여러 candidate를 본 뒤 audit 결과로 다시 hyperparameter를 바꾸지 않는다. Audit은 한
   phase에서 사전에 고른 control/candidate를 한 번 확인하는 용도다.

Official val이 있는데도 locked audit이 필요한 이유는 데이터 부족이 아니라 반복된 model
selection 때문이다. V0부터 threshold, resolution, scheduler, augmentation, backbone과 V6를
같은 404장에서 비교했으므로 official val은 이제 완전히 untouched인 final holdout이 아니다.
다만 locked audit은 control 재학습 비용이 크므로 V8/V9 전에 즉시 만들지 않고, V10 이후
학습 recipe의 엄격한 검증이 필요할 때 V11 K-fold와 함께 도입한다.

### 2026-07-14 Final Leaderboard Calibration

상세 수치와 증빙은
[Final Leaderboard Evidence](/data/ephemeral/home/receipt-text-detection/docs/leaderboard/20260714/README.md)에
기록한다.

- V0/V1/V2/V2B H 순서가 Local, Public, Final에서 모두 동일했다.
- Local-Final H Spearman correlation은 `1.0`, Pearson correlation은 약 `0.9991`이었다.
- Local-Final H MAE는 `0.001625`, RMSE는 `0.001897`, 최대 절댓값은 `0.0027`이었다.
- V2B는 Final H/P/R `0.9647 / 0.9739 / 0.9580`으로 1위를 기록했다.
- V0 -> V1, V1 -> V2, V2 -> V2B의 H 개선 방향이 모두 Final에서 유지됐다.

따라서 local CLEval을 primary metric으로 유지한다. 다만 제출이 4개뿐이고 같은 model family의
연속 실험이므로 높은 correlation을 일반 법칙으로 과해석하지 않는다. 특히 V2 -> V2B 개선은
Local `+0.0033`, Final `+0.0010`으로 줄었으므로 앞으로 `0.001` 수준의 차이는 paired bootstrap
또는 K-fold 일관성 없이 채택하지 않는다. 기존 V8 `1024+1152` 실행 순서는 유지한다.

## Public 제출 정책

모든 local 실험을 Public에 제출하지 않는다. 제출 가치가 있는 milestone은 다음과 같다.

- 이미 제출한 V2/V2B resolution milestone
- V6/V7에서 local 개선이 명확하고 재현된 clean single 후보
- V8/V9에서 macro/global local 근거를 모두 확보한 fusion 후보
- V11의 stable fold teacher 또는 full-data clean model
- V12 clean ablation을 통과한 pseudo student
- V13에서 invalid polygon, 누락 파일과 500개 cap 검사를 통과한 final 후보

SSL algorithm, TTA scale, ensemble weight, pseudo filtering threshold를 Public 점수로 선택하지
않는다. Local에서 폐기한 설정을 Public 결과로 되살리거나 Public/Private 파일을 추정해
파일별 규칙을 만들지 않는다.

### 대회 종료 후 Prediction Artifact 정책

제출 마감 이후에는 실제 leaderboard 전송 대신 다음 규칙을 적용한다.

- Local 채택 기준을 통과한 single model, TTA, ensemble, SSL/pseudo student마다 test 413장
  prediction JSON과 제출 형식 CSV를 생성한다.
- 모든 이미지 포함, polygon 최소 4점, 좌표 유효성, 빈 행, 중복, 이미지당 500개 cap을 검사한다.
- `submissions/submission_log.md`에 local 근거와 함께
  `Generated offline, competition closed`로 기록한다.
- Local에서 폐기된 진단 실험은 test artifact를 만들지 않는다. 다만 뒤 ensemble의 구성원으로
  실제 사용되는 경우 재현을 위해 예측 artifact 또는 probability-map cache를 남긴다.
- 파일 생성은 모델 채택과 inference pipeline 재현을 증명하기 위한 것이며 leaderboard 성능
  증거로 표현하지 않는다.

이에 따라 V6은 local H가 하락해 submission artifact를 만들지 않았다. V8 `1024+1152`는
사전 local gate를 통과해 test JSON과 CSV를 `Generated offline, competition closed` 상태로
생성했다. V9 equal V2B/V5 ensemble도 gate를 통과해 offline JSON/CSV를 생성했다. V10은
point estimate가 근소하게 높았지만 bootstrap상 동률이라 폐기했다. Test JSON/CSV는 이전
요청에 따른 recall-diverse 재현 artifact로만 생성했다. V11A manifest는 structural gate를
통과해 고정했으며 다음 실험은 V11B V2B-recipe 5-fold OOF 학습이다.
