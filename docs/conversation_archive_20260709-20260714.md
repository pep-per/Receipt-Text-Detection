# Receipt Text Detection Conversation Archive

## 문서 성격

이 문서는 2026-07-09부터 2026-07-14까지 사용자와 Codex가 영수증 글자 검출 프로젝트를
진행하며 나눈 대화를 프로젝트 관점에서 시간순으로 복원한 기록이다.

채팅 서비스의 원문 transcript가 workspace 파일로 제공되지 않아 모든 문장을 그대로 복사한
verbatim export는 아니다. 현재 대화 문맥, 실행 로그, W&B metadata, experiment README,
performance/submission log를 바탕으로 질문, 답변, 결정과 결과를 빠짐없이 찾기 쉬운 형태로
요약했다. 정확한 수치와 경로는 각 링크의 원본 artifact를 우선한다.

Markdown은 plain text이므로 별도 binary 형식보다 작고 Git diff와 검색에 적합하다.

## 프로젝트 목표와 최종 결과

- Task: 영수증 이미지 text-region polygon 검출
- Metric: CLEval detection-only H-Mean
- Official data: train 3,272장, val 404장, test 413장
- 목표: Public이 아니라 Private/Final 일반화 성능 최적화
- Best submitted model: V2B, ResNet18 DBNet 계열, resolution 1024
- Final H/P/R: `0.9647 / 0.9739 / 0.9580`
- Final leaderboard: 1위
- Final evidence: [2026-07-14 leaderboard](leaderboard/20260714/README.md)

## 1. 프로젝트 시작과 초기 전략

### 사용자 요청

대회 설명을 제공하고 프로젝트 폴더 구성, Private leaderboard 최고점을 목표로 한 전략,
Public leaderboard 과적합 위험의 사전 경고를 요청했다.

### 함께 정한 원칙

- Public 점수는 확인 신호로만 사용하고 model selection은 local CLEval로 수행한다.
- 한 실험에서 한 가지 주된 변수만 변경한다.
- 모델, config, checkpoint, validation, submission과 판단을 실험별로 기록한다.
- Polygon은 최소 4점, 모든 test 이미지 포함, 이미지당 500개 제한을 검사한다.
- Pseudo label은 강한 clean teacher가 만들어진 뒤 사용한다.

### 생성된 구조

- `docs/`: CLEval, 제출 형식, Private 전략, 과적합 위험
- `experiments/`: 실험별 README, metric과 분석
- `submissions/`: CSV와 누적 제출 기록
- `data/`: official/pseudo 데이터와 split manifest 위치
- `scripts/`: 분석, 평가와 submission helper

## 2. CLEval 이해와 전략 수정

### 사용자 질문

CLEval 논문과 구현, detection-only POLY 평가, 최대 500개, 최소 4점 polygon이라는 추가 규칙을
전략에 반영해 달라고 했다.

### 정리한 내용

- CLEval은 word box IoU만 보는 대신 pseudo-character coverage와 granularity penalty를 사용한다.
- `RIVER`와 `SIDE`처럼 나뉜 검출을 일반 IoU보다 유연하게 다룰 수 있다.
- 대회에서는 recognition transcription을 사용하지 않는다.
- Local evaluator는 QUAD가 아닌 POLY와 detection-only 설정을 사용해야 한다.
- H-Mean만 보지 않고 precision, recall을 함께 진단한다.

관련 문서: [CLEval Korean Guide](cleval_korean_guide.md)

## 3. 데이터 구조와 split 논의

### 사용자 질문

실제 data 폴더 구조, 기존에 만든 빈 폴더의 필요성, train/val/test JSON 형식, Public/Private가
어떤 데이터인지 질문했다.

### 정리한 내용

- `images/train`, `images/val`, `images/test`와 `jsons/train.json`, `val.json`, `test.json`을
  official 구조로 사용한다.
- Official val 404장은 label이 있는 local 평가용이다.
- Test 413장이 hidden Public/Private 50:50으로 나뉜다.
- Official val은 Public/Private test에 포함되지 않는다.
- Split manifest는 학습만 하면 자동으로 생기는 파일이 아니다. K-fold나 새 holdout을 만들 때
  명시적으로 생성한다.
- 고정 official train/val을 사용한 실험에는 새 split 파일이 없어도 정상이다.

관련 문서: [Data And Submission Format](data_submission_format.md)

## 4. Baseline code와 실험 방식

### 사용자 질문

Baseline을 그대로 실행한 버전과 분석·개선한 버전을 비교하는 것이 좋은지, `use_polygon`을
바꿀 것인지, 실험 속도와 W&B 사용, LoRA/FlashAttention/vLLM의 필요성을 질문했다.

### 정리한 내용

- 먼저 baseline을 재현한 뒤 controlled improvement를 적용하는 비교는 타당하다.
- `use_polygon=False`는 대회 POLY 평가와 직접 같은 말이 아니다. DB postprocess가 contour에서
  어떤 polygon/box 형태를 반환하는지와 evaluator 입력을 분리해 이해해야 한다.
- LoRA는 transformer 대형 언어모델 파라미터 효율 fine-tuning 도구라 현재 CNN DB detector에
  직접적인 이득이 없다.
- FlashAttention은 attention 계산 최적화이며 ResNet+UNet DBNet의 병목과 맞지 않는다.
- vLLM은 autoregressive LLM serving engine이라 OCR detection 학습/추론과 무관하다.
- 속도 개선은 mixed precision, batch, dataloader, resolution, cache와 불필요한 반복 inference
  제거가 핵심이다.
- V2 이후 training/evaluation run은 W&B에 기록한다.

## 5. V0 Baseline

### 실행과 결과

- Official train/val 고정 split으로 모델을 실제 학습했다.
- Best checkpoint를 test에 추론해 JSON과 CSV를 만들었다.
- Local H/P/R: `0.8913 / 0.9633 / 0.8369`
- Public H/P/R: `0.8818 / 0.9651 / 0.8194`
- Final H/P/R: `0.8898 / 0.9675 / 0.8324`

### 대화에서 해소한 혼동

- `data/splits/`가 비었다고 학습하지 않은 것이 아니다.
- 학습 checkpoint로 val/test inference를 실행했기 때문에 결과 파일을 만들 수 있었다.
- 새 split은 K-fold나 다른 holdout을 선택할 때만 필요하다.

상세 기록: [V0 Baseline](../experiments/20260709-v0-baseline/README.md)

## 6. 실험 성능 기록 체계

사용자가 leaderboard screenshot의 수치를 추출하고 이후 모든 실험에서 성능과 관련 파일을
누적 기록해 달라고 요청했다.

이에 따라 다음을 유지했다.

- [Performance Log](../experiments/performance_log.md)
- [Submission Log](../submissions/submission_log.md)
- 실험별 README와 checkpoint/config/W&B/CSV 링크
- Local, Public, Final H/P/R과 gap
- 결과에 따른 다음 실험의 전략 변경 또는 branch 종료

## 7. V1 Threshold Sweep

### 사용자 질문

`box_thresh`가 무엇인지, 후처리가 학습보다 중요하다고 판단할 수 있는지, 여러 threshold를
평가했는지 질문했다.

### 정리한 내용

- `thresh`는 probability map을 contour 후보로 이진화하는 pixel threshold다.
- `box_thresh`는 contour proposal의 평균 text score가 이 값 이상일 때 유지하는 threshold다.
- V1은 V0 checkpoint를 그대로 사용하고 `0.35`, `0.30`, `0.25`, `0.20`을 local에서 비교했다.
- 네 번의 추론이지 네 번의 학습이 아니다.
- V1만으로 후처리가 모든 모델 학습보다 중요하다고 일반화할 수는 없다고 명시했다.

### 결과

- 선택: `box_thresh=0.25`
- Local H/P/R: `0.9248 / 0.9499 / 0.9057`
- Public H/P/R: `0.9185 / 0.9511 / 0.8932`
- Final H/P/R: `0.9221 / 0.9554 / 0.8978`
- Final H는 V0보다 `+0.0323`, 주된 개선은 recall 회복이었다.

상세 기록: [V1 Threshold Sweep](../experiments/20260709-v1-threshold-sweep/README.md)

## 8. Clean-data roadmap와 용어

사용자는 pseudo label 전에 기존 human-label 데이터로 최대한 성능을 높이고 싶다고 했다.
이에 따라 LR, cosine decay, checkpoint, clean model, local CLEval 선택 이유와 실험 순서를
문서화했다.

- Clean model: pseudo polygon 없이 official human polygon만 학습한 모델
- LR: optimizer가 weight를 한 update에서 얼마나 바꾸는지 정하는 learning rate
- StepLR: 일정 step마다 LR을 계단식으로 감소
- Cosine decay: cosine 곡선으로 LR을 부드럽게 감소
- Checkpoint: 특정 epoch의 weight와 optimizer/training state 저장 파일
- Local CLEval: 제공 val human label에 대해 로컬에서 계산하는 대회 계열 metric

로드맵: [Clean-data Experiment Roadmap](../experiments/clean_data_experiment_roadmap.md)

## 9. V2 Resolution 896

### 가설

640 입력에서는 작은 text가 feature map에서 지나치게 작아져 threshold로도 되살릴 수 없었다.
한 번에 augmentation까지 바꾸지 않고 resolution만 896으로 높였다.

### 결과

- Local H/P/R: `0.9615 / 0.9638 / 0.9611`
- Public H/P/R: `0.9603 / 0.9667 / 0.9556`
- Final H/P/R: `0.9637 / 0.9682 / 0.9606`
- V1 대비 Final H `+0.0416`, 제출 실험 중 가장 큰 Final 개선이었다.

상세 기록: [V2 Resolution 896](../experiments/20260712-v2-resolution-896/README.md)

## 10. V2B Resolution 1024

사용자는 V2B local 결과에 따라 다음 실험을 진행한 것인지, 왜 제출 파일이 없었는지, local이
낮아도 leaderboard가 높을 수 있는지 질문했다. Local 결정을 먼저 내리고 milestone 후보의
test CSV를 별도로 만드는 흐름을 정리했다.

### 결과

- Local H/P/R: `0.9648 / 0.9700 / 0.9614`
- Public H/P/R: `0.9621 / 0.9754 / 0.9520`
- Final H/P/R: `0.9647 / 0.9739 / 0.9580`
- V2 대비 Final H `+0.0010`, precision `+0.0057`, recall `-0.0026`
- Final leaderboard 1위

1024의 이득은 recall 증가보다 cleaner boundary와 precision 개선 쪽에 가까웠다. Local 개선
`+0.0033`이 Final에서는 `+0.0010`으로 줄었으므로 이후 `0.001` 수준의 차이는 추가 검증이
필요하다고 결정했다.

상세 기록: [V2B Resolution 1024](../experiments/20260713-v2b-resolution-1024/README.md)

## 11. Local CLEval aggregation audit

사용자는 CLEval 식의 recall, precision과 H-Mean이 local에서도 같은 방식인지 질문했다.

조사 결과 baseline Lightning wrapper는 다음 두 aggregation을 구분해야 했다.

- Macro: 이미지마다 CLEval H/P/R을 계산한 후 각각 평균
- Global: raw character counts와 granularity penalty를 전체 이미지에 누적한 후 계산

Leaderboard의 표시 H가 표시 P/R의 단순 조화평균과 정확히 같지 않아 server aggregation을
완전히 확정할 수 없었다. 이후 macro와 global을 모두 기록했고, 주요 model 선택이 두 방식에서
같은 방향인지 확인했다.

## 12. V3 Cosine LR

### 통제 조건

V2B 1024 recipe에서 StepLR만 `CosineAnnealingLR`로 바꾸고 나머지를 고정했다. Fresh training
run이며 V2B checkpoint resume가 아니었다.

### 결과

- Local macro H/P/R: `0.9592 / 0.9602 / 0.9602`
- V2B 대비 H `-0.0056`
- Cosine 후보를 폐기하고 Public 제출을 만들지 않았다.

상세 기록: [V3 Cosine 1024](../experiments/20260713-v3-cosine-1024/README.md)

## 13. V4 Photometric Augmentation

Blur, brightness/contrast/gamma, motion blur, JPEG 등 촬영 열화 변형을 하나의 약한 policy로
시험했다. 사용자는 성능이 떨어진 원인과 평가 dataset/official val 관계를 질문했다.

### 결과

- Best independent macro H/P/R: `0.9626 / 0.9685 / 0.9587`
- V2B 대비 macro H `-0.0022`, global H `-0.0035`
- Epoch 8 recall 신호는 있었지만 precision 손실로 H가 낮았다.
- 결합 photometric policy를 폐기하고 제출하지 않았다.

상세 기록: [V4 Photometric 1024](../experiments/20260714-v4-photometric-1024/README.md)

## 14. V5 ResNet34

ResNet18에서 ResNet34로 backbone만 바꿔 표현력 증가를 시험했다.

### 결과

- Macro H/P/R: `0.9646 / 0.9643 / 0.9668`
- Global H/P/R: `0.9622 / 0.9603 / 0.9641`
- V2B와 H는 사실상 동률
- V2B보다 precision이 낮고 recall이 높음
- Peak GPU memory 23.08 GiB로 RTX 3090의 약 96%
- V2B를 default single model로 유지하고 V5를 recall-diverse ensemble 후보로 보존

상세 기록: [V5 ResNet34 1024](../experiments/20260714-v5-resnet34-1024/README.md)

## 15. Pseudo label, SSL과 semi-supervised learning

### 사용자 질문

Pseudo 이미지는 기존 데이터와 같은지, 모델이 어떻게 자동 polygon을 만드는지, validation이나
CV에 포함할 수 있는지, SSL과 semi-supervised learning의 차이, contrastive learning,
masked-image modeling, encoder와 DBNet 연결을 질문했다.

### 정리한 내용

- Pseudo label은 teacher가 unlabeled 이미지에 자동 생성한 polygon이라 정답을 보장하지 않는다.
- Pseudo polygon을 validation 정답으로 사용하지 않는다.
- Pseudo 데이터는 clean human label과 source/loss weight를 구분해 학습한다.
- Self-supervised learning은 label 없이 image transformation/representation objective를 학습한다.
- Semi-supervised learning은 human label과 pseudo label을 함께 사용한다.
- Contrastive learning은 같은 이미지의 두 view representation을 가깝게, 다른 이미지를 멀게 한다.
- Masked-image modeling은 가린 patch나 feature를 복원해 표현을 학습한다.
- SSL encoder weight를 DBNet encoder에 넣고 decoder/head를 포함한 detector 전체를 supervised
  fine-tuning해야 최종 polygon task에 맞는다.
- 규정상 test 이미지의 자동 SSL/TTA/분석은 허용되지만 사람이 test polygon을 만드는 것은
  금지된다.

## 16. Transductive learning과 대회 규정

사용자는 test를 SSL에 넣어도 되는지 질문하고, 평가 데이터의 시각화/TTA/SSL을 허용하되
인위적 labeling은 금지한다는 규정을 제공했다.

- Test image pixel을 label 없이 representation learning에 쓰면 transductive learning이다.
- 이 대회 규정에서는 해당 방식이 허용된다.
- Test 수동 labeling이나 파일별 수작업 규칙은 금지된다.
- Pseudo source 이미지도 image-only SSL pool에 사용할 수 있다.
- Local 문서에는 strict inductive와 transductive local 결과를 구분해 표시한다.

## 17. D0 Data Audit

사용자는 평가 데이터 시각화, TTA, SSL을 활용하고 싶다고 했고 D0, V6, V7, V8, V10 순서와
각 단계의 의미를 논의했다.

D0는 새 model version이 아니라 다음 가설을 제한하는 분석 단계였다.

- Official/auxiliary 7,772장 품질 통계
- Train/val polygon 크기와 small-text 비율
- V2B per-image CLEval과 품질 상관
- V2B/V5 val/test disagreement
- Quality quantile contact sheet

### 핵심 결과

- Official train/val/test의 brightness, contrast, blur, aspect, resolution 분포가 매우 비슷함
- Train-test 최대 KS statistic `0.0655`
- 12px 미만 글자 비율과 V2B H `rho=-0.1945`, recall `rho=-0.2061`
- Photometric 품질과 H 관계는 약함
- V2B/V5 prediction Jaccard는 val/test 모두 약 `0.873`
- V7 photometric/geometric augmentation은 근거 부족으로 skip
- V8 첫 scale TTA를 `1024+1152`로 변경
- V9 V2B+V5 probability-map ensemble 후보 유지
- SSL auxiliary source는 분포가 달라 source-balanced sampling 필요

상세 기록: [D0 Data Audit](../experiments/20260714-d0-data-audit/README.md)

## 18. K-fold, augmentation, seed/architecture ensemble과 TTA

사용자는 K-fold에서 TTA를 함께 하는지, fold augmentation이 TTA인지, official val holdout,
seed/architecture와 TTA 관계를 질문했다.

- Training augmentation: 학습 입력을 확률적으로 바꾸는 regularization
- TTA: 이미 학습된 모델에 test/val 입력 변형을 적용해 예측을 합치는 inference 방법
- Fold마다 augmentation을 사용해도 그것은 training augmentation이지 TTA가 아니다.
- K-fold: 데이터를 K개로 나누고 각 model이 한 fold를 label 학습에서 제외
- OOF: 각 이미지를 자기 label을 학습하지 않은 model 하나로만 예측
- Seed ensemble: 같은 architecture/recipe를 다른 random seed로 학습
- Architecture ensemble: 다른 inductive bias model을 결합
- Scale TTA: 같은 model의 1024/1152 probability map을 원 좌표계로 복원해 평균
- Fold ensemble의 test 이득과 OOF 성능 측정은 구분해야 한다.

상세 가이드: [K-fold, Augmentation, Ensemble And TTA](kfold_augmentation_ensemble_tta_guide.md)

## 19. V6 V5 Post-processing Recalibration

### 실험 이유

V5가 V2B와 H 동률이면서 high-recall/low-precision이므로 `box_thresh=0.30`이 작은 recall
손실보다 큰 precision 회복을 만들 수 있는지 저비용으로 확인했다. 학습 없이 약 76초가 걸렸다.

### 결과

- V6 macro H/P/R: `0.960568 / 0.966257 / 0.957238`
- V5 box 0.25 대비 H `-0.004049`, P `+0.001916`, R `-0.009593`
- Macro/global H가 모두 하락해 0.30 폐기
- 사전 gate에 따라 0.35는 실행하지 않음
- V2B를 single control로 유지

상세 기록: [V6 V5 Post-processing](../experiments/20260714-v6-v5-postprocess/README.md)

## 20. W&B 대화와 기록 차이

### 로그인과 sync

사용자는 W&B API key 설정, `wandb status`, 이전 offline run sync와 실험 기록 여부를 질문했다.
API key는 `.netrc`에 저장될 수 있어 `wandb status` JSON의 `api_key: null`만으로 미로그인이라고
판단하면 안 된다고 설명했다. V2 이후 training/evaluation run은 online 또는 offline-sync로
대시보드에 보존했다.

### 왜 V6에는 epoch history가 없는가

V6 run `8cy2bpn0`은 이름 그대로 `v6_v5_box030_eval`인 evaluation-only run이다.

```text
V5 training run scibn6c0
trainer.fit() -> 10 epochs -> trainer.test()
epoch=10, trainer/global_step=2050, train/*, val/*, test/*, lr 기록

V6 evaluation run 8cy2bpn0
checkpoint epoch 7 로드 -> trainer.test() 한 번
epoch=0, trainer/global_step=0, test/* metric 한 행만 기록
```

Checkpoint filename의 `epoch=7`은 weight가 어느 training epoch에서 저장됐는지를 뜻한다.
새 evaluation run이 7개 epoch를 다시 실행하거나 원 training history를 복사하는 것은 아니다.
V5 학습 곡선은 training run `scibn6c0`, V6 threshold 결과는 evaluation run `8cy2bpn0`에서 본다.

실험마다 기록이 다른 이유는 실행 runner와 `self.log()` 호출이 다르기 때문이다.

| Run 종류 | Lightning 호출 | W&B에 기록되는 값 |
| --- | --- | --- |
| Training | `trainer.fit()` 후 `trainer.test()` | train/val epoch history, LR, global step, 마지막 test |
| Evaluation | `trainer.test()` | test H/P/R와 global H/P/R 한 번 |
| Prediction | `trainer.predict()` | 현재 구현에서는 metric 없이 JSON 생성 |
| D0 analysis | custom script | W&B run 없음, CSV/PNG/README artifact |

## 21. Final leaderboard와 local metric 검증

사용자가 Final/Mid leaderboard와 submission list screenshot을 제공했다.

- V0 Final H/P/R: `0.8898 / 0.9675 / 0.8324`
- V1 Final H/P/R: `0.9221 / 0.9554 / 0.8978`
- V2 Final H/P/R: `0.9637 / 0.9682 / 0.9606`
- V2B Final H/P/R: `0.9647 / 0.9739 / 0.9580`
- V2B Final rank: 1위, rank 2보다 H `+0.0195`

Local/Public/Final H 순서가 모두 `V0 < V1 < V2 < V2B`로 같았다. Local-Final Spearman은
`1.0`, Pearson은 약 `0.9991`, H MAE는 `0.001625`였다. 제출된 model sequence에서는 local
CLEval 선택 방향이 hidden Final에 강하게 전이됐다.

같은 validation을 반복 사용하는 것이 금지된 것은 아니다. 다만 여러 결과를 보고 다음 후보를
적응적으로 선택하면 그 val은 untouched final test가 아니라 development set이 된다. 이번 Final
결과에는 제출 모델의 유해한 validation/Public overfit 증거가 없지만, V2B 개선이 Local
`+0.0033`에서 Final `+0.0010`으로 줄어 작은 차이에는 추가 검증이 필요하다.

다른 팀의 Public-Final 하락 한 번만으로 과적합이라고 단정할 수 없다. 서로 다른 약 200장
표본의 sampling variance일 수 있으며, 반복적인 Public 선택과 local 불일치 같은 추가 근거가
필요하다.

## 22. Git, W&B와 대용량 artifact

### 사용자 질문

W&B sync 뒤 local `wandb/`를 Git에 올려야 하는지, 2.9GB `outputs/`가 계속 실험에 필요한지,
용량을 어떻게 관리할지 질문했다.

### 정리한 정책

Git에 포함:

- Source code
- Hydra config/overrides
- Experiment README와 작은 metric CSV/JSON
- Final submission CSV
- Split manifest와 문서

Git에서 제외:

- Dataset과 다운로드 archive
- W&B local cache
- Checkpoint, TensorBoard log, output prediction JSON

W&B 기본 sync는 metric/config/log를 올리지만 현재 logger는 checkpoint를 자동 업로드하지 않는다.
V8에는 V2B epoch 8, V9에는 V2B epoch 8과 V5 epoch 7 checkpoint가 필요하므로 이 파일들은
로컬과 별도 artifact storage에 보존해야 한다. `.gitignore`는 파일을 삭제하지 않아 기존 실험
진행에는 영향을 주지 않는다.

## 23. 대회 종료 후 실행 정책

- Local gate를 통과한 single model/TTA/ensemble은 test 413장 JSON과 CSV를 계속 생성한다.
- 상태는 `Generated offline, competition closed`로 기록한다.
- 폐기된 진단 실험은 test artifact를 만들지 않는다.
- V8/V9는 official val paired comparison과 bootstrap을 사용한다.
- V10 이후 training recipe는 고정 K-fold OOF로 안정성을 확인한다.
- Test CSV는 inference 재현 artifact이지 새로운 hidden score 증거는 아니다.

## 현재 다음 단계

1. V7은 D0 근거 부족으로 skip 상태를 유지한다.
2. V8에서 V2B epoch 8의 1024 control과 `1024+1152` probability-map TTA를 비교한다.
3. Macro/global H가 모두 개선되면 test JSON/CSV와 sanity-check를 생성한다.
4. V9에서 V2B와 V5 probability-map ensemble을 평가한다.
5. 이후 V10 SSL pilot, V11 K-fold teacher, V12 pseudo student, V13 final fusion 순서로 진행한다.

## 핵심 문서 Index

- [Project README](../README.md)
- [Performance Log](../experiments/performance_log.md)
- [Submission Log](../submissions/submission_log.md)
- [Clean-data Roadmap](../experiments/clean_data_experiment_roadmap.md)
- [Private Leaderboard Strategy](private_lb_strategy.md)
- [CLEval Korean Guide](cleval_korean_guide.md)
- [Data And Submission Format](data_submission_format.md)
- [K-fold, Augmentation, Ensemble And TTA](kfold_augmentation_ensemble_tta_guide.md)
- [Final Leaderboard Evidence](leaderboard/20260714/README.md)

