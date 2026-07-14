# CLEval Korean Guide

이 문서는 CLEval 논문과 공식 GitHub README를 읽고, 이번 Receipt Text Detection 대회에 필요한 부분을 한국어로 정리한 것이다.

접근 여부:

- arXiv 논문 페이지와 PDF 접근 가능
- `clovaai/CLEval` GitHub README 접근 가능
- 참고용으로 공식 저장소 코드를 임시 경로에 clone해서 argument와 POLY 처리도 확인함

## 한 줄 요약

CLEval은 OCR text detection/recognition 평가에서 "박스가 IoU 임계값을 넘었는가"만 보지 않고, 문자를 얼마나 빠뜨리지 않고, 중복 없이, 적절한 단위로 검출했는지를 character-level 관점에서 평가하려는 metric이다.

이번 대회에서는 CLEval의 end-to-end recognition 평가는 쓰지 않고, detection 결과만 평가한다. 즉, transcription 문자열은 제출과 평가에 사용하지 않는다.

## 왜 CLEval이 필요한가

기존 text detection 평가는 IoU나 DetEval처럼 instance-level matching에 많이 의존한다. 이 방식은 단순하고 직관적이지만 OCR 관점에서는 어색한 점이 있다.

예를 들어 Ground Truth가 `RIVERSIDE`라는 하나의 text region인데 모델이 `RIVER`, `SIDE` 두 영역으로 나눠 검출했다고 하자. IoU threshold 기반 평가에서는 둘 중 하나만 매칭되거나 둘 다 실패처럼 처리될 수 있다. 하지만 OCR pipeline 입장에서는 두 영역을 모두 읽을 수 있다면 완전한 실패라고 보기 어렵다.

반대로 큰 박스 하나로 여러 단어와 배경까지 덮거나, 같은 문자를 여러 detection이 중복해서 덮는 경우도 있다. 이런 결과는 겉보기 IoU가 나쁘지 않아도 recognition 단계에서는 혼란을 만들 수 있다. CLEval은 이런 split, merge, missing, overlapping 문제를 character-level 관점에서 더 세밀하게 반영하려고 한다.

## CLEval의 핵심 개념

논문은 크게 두 문제를 다룬다.

### Granularity

Granularity는 text를 얼마나 적절한 단위로 검출했는지에 관한 문제다.

- Split: 하나의 GT word 또는 text instance가 여러 prediction으로 나뉜 경우
- Merge: 여러 GT instance가 하나의 prediction으로 합쳐진 경우

CLEval은 split/merge를 무조건 0점 처리하지 않고, 가능한 matching 관계를 더 넓게 고려한다. 하지만 split/merge가 반복되면 penalty가 생기므로 "무조건 쪼개면 안전하다"거나 "크게 합치면 안전하다"는 해석은 위험하다.

### Correctness

Correctness는 문자를 빠뜨리거나 중복해서 덮지 않았는지에 관한 문제다.

- Missing: GT 문자 일부를 detection이 덮지 못한 경우
- Overlapping/Duplicate: 같은 문자 영역을 여러 prediction이 중복해서 덮는 경우

Detection-only 평가에서도 문자 단위 관점이 중요하다. 작은 글자를 놓치면 recall이 떨어지고, 바코드 조각이나 배경 선을 text로 잡으면 precision이 떨어진다.

## Matching Process

CLEval은 먼저 GT와 prediction 사이의 가능한 연결을 찾는다. 논문에서는 character annotation이 없는 일반적인 word-level dataset에서도 평가가 가능하도록 pseudo-character center라는 개념을 사용한다.

Pseudo-character center는 text box 안에 문자가 균등하게 놓여 있다고 가정하고 만든 가상의 문자 중심점이다. 이 점들을 prediction polygon이 포함하는지 보고, GT와 prediction이 서로 어떤 문자들을 공유하는지 판단한다.

또한 단순히 점 하나를 포함했다고 다 맞았다고 보지 않는다. prediction이 text가 아닌 영역을 너무 많이 포함하면 area precision 조건에서 걸러질 수 있다. 이 점 때문에 polygon을 크게 부풀리는 후처리는 recall에는 좋아 보여도 precision에 손해가 날 수 있다.

이번 대회에서는 transcription 정보를 사용하지 않는 detection-only 설정이므로, local evaluator는 대회 evaluator와 같은 wrapper/config로 맞추는 것이 중요하다. 공개 CLEval README의 end-to-end 예시는 transcription을 포함하지만, 이 대회 제출에는 적용하지 않는다.

## Scoring Process

Upstream CLEval은 이미지별 matching에서 나온 문자 수와 granularity penalty를 전체
데이터셋에 누적한 뒤 최종 recall과 precision을 계산한다. 공식 TorchMetric 예시도 모든
이미지에 `metric(...)`을 호출한 후 마지막에 `compute()`를 한 번 호출한다.

개념적으로는 다음과 같다.

```text
Recall    = correctly detected GT characters / total GT characters
Precision = correctly detected prediction characters / total prediction characters
H-Mean    = 2 * Recall * Precision / (Recall + Precision)
```

논문에서는 여기에 granularity penalty를 반영한다. 그래서 한두 번의 split은 기존 IoU 평가보다 덜 가혹할 수 있지만, split/merge가 많아지면 H-Mean이 떨어질 수 있다.

### 이 baseline의 macro 집계와 upstream global 집계

제공된 [ocr_pl.py](/data/ephemeral/home/receipt-text-detection/baseline_code/ocr/lightning_modules/ocr_pl.py)는
각 이미지마다 CLEval H/P/R을 계산하고 metric을 reset한 뒤, 이미지별 H/P/R을 각각
산술평균한다.

```text
Macro H = mean(H_i)
Macro P = mean(P_i)
Macro R = mean(R_i)
```

따라서 `Macro H`는 `harmonic_mean(Macro P, Macro R)`와 같을 필요가 없다. 반면 upstream
global 방식은 모든 이미지의 문자 numerator, denominator와 penalty를 누적해 Global P/R을
구한 후 한 번 조화평균한다.

이 프로젝트는 기존 비교를 유지하기 위해 macro `val/hmean`을 checkpoint 기준으로
보존하면서, 다음 global 지표도 함께 기록하도록 보강했다.

- `val/global_hmean`
- `val/global_precision`
- `val/global_recall`

Dual 재평가 결과:

| Model | Macro H/P/R | Global H/P/R |
| --- | --- | --- |
| V2 896 | `0.961507 / 0.963823 / 0.961134` | `0.958298 / 0.959732 / 0.956868` |
| V2B 1024 | `0.964760 / 0.969976 / 0.961422` | `0.962219 / 0.967510 / 0.956985` |

두 집계 모두 V2B 1024를 선택하므로 지금까지의 해상도 결정은 유지된다. 리더보드의
표시 H가 표시 P/R의 단순 조화평균과 다르다는 점은 대회 서버도 별도 집계를 사용할
가능성을 보여주지만, 정확한 서버 evaluator가 공개되지 않았으므로 macro와 global을
함께 보고 방향이 일치하는 후보를 우선한다.

## GitHub README에서 확인한 사용법

공식 구현체는 CLI와 TorchMetric 사용을 지원한다. README 기준 지원 annotation type은 다음과 같다.

- LTRB: `xmin, ymin, xmax, ymax`
- QUAD: `x1, y1, x2, y2, x3, y3, x4, y4`
- POLY: `x1, y1, x2, y2, ..., x_2n, y_2n`

Detection evaluation 예시에서는 TotalText처럼 polygon dataset을 평가할 때 `--BOX_TYPE=POLY`를 사용한다. 기본값은 QUAD이므로, 이번 대회 local evaluation에서는 POLY 설정을 명시하는 것이 안전하다.

README에는 다음 옵션들이 설명되어 있다.

- `-g`: ground truth zip 경로
- `-s`: prediction/result zip 경로
- `-o`: sample result 저장 경로
- `--BOX_TYPE`: LTRB, QUAD, POLY 중 선택
- `--TRANSCRIPTION`: result file에 transcription이 있을 때 사용
- `--CONFIDENCES`: result file에 confidence가 있을 때 사용
- `--E2E`: end-to-end recognition까지 평가할 때 사용

이번 대회에서는 detection-only 평가이므로 `--E2E`를 사용하지 않는다. 대회 설명상 Ground Truth와 Prediction 모두 transcription 정보를 사용하지 않으므로, 제출 파일에도 transcription 기반 최적화를 넣지 않는다.

## 이번 대회에 직접 적용되는 규칙

대회 설명에서 확인된 운영 규칙은 다음과 같다.

- CLEval Metric을 사용한다.
- 순위는 H-Mean으로 결정한다.
- H-Mean은 높을수록 좋다.
- H-Mean은 소수점 4번째 자리까지 계산한다.
- 동점이면 먼저 제출한 팀이 우선이다.
- Detection-only 평가이며 Recognition/End-to-End 평가는 아니다.
- Ground Truth와 Prediction 모두 transcription 정보를 사용하지 않는다.
- GT label은 polygon 기준이다.
- CLEval도 QUAD가 아니라 POLY 방식으로 평가한다.
- 제출 polygon은 4점 이상이어야 한다.
- 3점 이하 polygon은 무시된다.
- 모든 이미지에 대해 답을 제출해야 한다.
- 이미지당 최대 text region은 500개다.
- 500개를 초과한 text region은 평가 대상에서 제외된다.

## 전략적으로 바뀌는 점

### IoU만 보고 튜닝하지 않기

IoU validation script가 있다면 빠른 sanity check에는 쓸 수 있지만, 최종 모델 선택 기준으로 쓰면 안 된다. CLEval detection-only H-Mean을 local에서 재현해야 한다.

### 과도한 unclip 줄이기

DBNet 계열에서 `unclip_ratio`를 키우면 작은 글자를 덜 놓칠 수 있다. 하지만 polygon이 너무 커져 배경과 인접 글자를 많이 포함하면 precision과 granularity에서 손해가 날 수 있다.

### split과 merge를 따로 분석하기

`RIVER`와 `SIDE`처럼 나뉜 결과가 완전 실패가 아닐 수 있다는 점은 좋은 소식이다. 그러나 반복적인 과분할은 penalty가 누적될 수 있다. 반대로 여러 단어 또는 여러 줄을 하나로 합치는 것은 private에서 특히 위험하다.

### 500개 제한을 모델 선택 기준에 넣기

영수증 한 장에 평균 100개 text region이 있다면 일반적으로 500개는 넉넉하다. 하지만 threshold를 낮추거나 ensemble/TTA를 적용하면 작은 false positive가 급증할 수 있다. 이미지별 prediction count histogram을 보고, 500개에 가까운 이미지는 따로 시각화해야 한다.

### 제출 sanity check 자동화

제출 전에는 모델 점수보다 포맷 오류 방지가 먼저다.

- 모든 test image가 있는가
- polygon이 4점 이상인가
- 좌표가 원본 이미지 기준인가
- 좌표가 NaN/inf가 아닌가
- 좌표가 이미지 밖으로 과도하게 나가지 않는가
- 면적 0 polygon이 없는가
- self-intersection polygon이 심하지 않은가
- 이미지당 region 수가 500개 이하인가

## Private LB 관점의 경고

다음 방법은 public leaderboard에서는 좋아질 수 있지만 private에서 낮아질 수 있다.

- public score만 보고 threshold hill-climbing
- 작은 box를 validation 근거 없이 일괄 제거
- 바코드, 로고, 표 선 제거 규칙을 test 눈대중으로 추가
- 여러 줄을 큰 polygon으로 묶는 merge rule
- ensemble/TTA 후 duplicate polygon을 충분히 제거하지 않음
- 500개 제한에 가까운 submission을 그대로 제출
- local CLEval 없이 IoU 또는 visual impression만 보고 최종 선택

## Sources

- CLEval paper: https://arxiv.org/abs/2006.06244
- CLEval PDF: https://arxiv.org/pdf/2006.06244
- CLEval official implementation: https://github.com/clovaai/CLEval
- CLEval raw README: https://raw.githubusercontent.com/clovaai/CLEval/master/README.md
