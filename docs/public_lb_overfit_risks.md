# Public Leaderboard Overfit Risks

이 파일은 public leaderboard 점수가 올라도 private leaderboard에서 떨어질 수 있는 선택지를 사전에 표시하기 위한 체크리스트다.

## Red Flags

다음 중 하나라도 해당하면 "public overfit 가능성 있음"으로 표시한다.

- Local validation은 나빠졌는데 public leaderboard만 좋아졌다.
- 제출 1-2회 점수 차이를 보고 threshold를 바꿨다.
- public 점수가 오른 특정 post-processing rule이 특정 resolution, filename, image count에 의존한다.
- test image를 눈으로 많이 보고 수동 규칙을 만들었다.
- tiny box 제거, barcode 제거, logo 제거 같은 규칙이 validation 근거 없이 추가됐다.
- public score가 높은 single seed를 최종 모델로 고르고, fold/seed variance를 확인하지 않았다.
- pseudo-label을 test에 적용했지만 local ablation이 없다.
- TTA/ensemble 후 precision 하락이 있는데 public 점수만 보고 채택했다.
- train + val 전체 재학습 모델을 validation 대체 검증 없이 최종으로 골랐다.
- 이미지당 500개 제한에 가까운 submission인데 public 점수만 보고 유지했다.
- 3점 이하 polygon, self-intersection polygon, 좌표 복원 오류를 local sanity check 없이 제출했다.
- IoU validation은 좋아졌지만 CLEval detection-only validation을 재현하지 않았다.
- Public/Private가 random 50:50이라는 이유로 public score를 곧 private score라고 가정했다.

## 특히 위험한 방법

### Public Score Hill-climbing

Public 제출 점수만 보고 `box_thresh`, `unclip_ratio`, `min_text_size`를 계속 바꾸는 방식은 private에서 가장 위험하다. Public split이 작은 경우 특정 글자 크기나 특정 배경 패턴에 맞춘 값이 될 수 있다.

대응:

- threshold는 fold validation grid search로 고정한다.
- public score는 최종 tie-breaker로만 사용한다.
- Public/Private가 평균 words 수를 맞춘 random split이어도, public은 test의 약 절반뿐이라는 점을 기록한다.

### Aggressive False Positive Filtering

바코드, 로고, 표 선, 총액 박스 등을 제거하려는 규칙은 public에서 좋아질 수 있지만 private에 다른 형태의 영수증이 나오면 실제 text를 지울 수 있다.

대응:

- rule을 만들기 전에 validation false positive/false negative 사례를 100개 이상 본다.
- 제거 규칙은 recall 손실을 반드시 측정한다.

### CLEval을 오해한 과분할/과병합

CLEval은 split/merge detection을 IoU보다 유연하게 다루지만, 그렇다고 일부러 잘게 쪼개거나 크게 합치는 것이 항상 유리하다는 뜻은 아니다. 반복적인 split/merge는 granularity penalty와 precision/recall 손실로 이어질 수 있다.

대응:

- word-level, line-level, price column에서 split/merge 오류를 따로 집계한다.
- post-processing merge rule은 CLEval fold score와 시각화로 동시에 검증한다.
- 큰 polygon으로 여러 줄을 덮는 전략은 기본적으로 금지한다.

### 500개 제한 무시

이미지당 text region이 500개를 넘으면 초과 영역은 평가 대상에서 제외된다. 출력 순서가 좋지 않으면 실제 text polygon이 잘리고 false positive만 남는 최악의 경우가 생길 수 있다.

대응:

- 이미지별 prediction count histogram을 매 제출마다 기록한다.
- confidence가 있으면 confidence 기준으로 정렬한 뒤 cap을 적용한다.
- 500개에 가까운 이미지의 prediction visualization을 별도로 확인한다.

### Manual Test-set Adaptation

Test 이미지를 보고 특정 케이스를 수동 보정하면 public에는 맞을 수 있지만 private split에는 맞지 않을 수 있다. 대회 규정상 test label이 없더라도, test distribution에 과도하게 맞추는 것은 재현성과 일반화 관점에서 위험하다.

대응:

- test visualization은 submission sanity check에 한정한다.
- 규칙은 train/val에서 관찰된 failure mode로만 정당화한다.

### Unvalidated Pseudo-labeling

고신뢰 예측을 test pseudo-label로 추가하면 성능이 오를 수 있지만, text detection에서는 잘못된 polygon이 누적되어 false positive가 커질 수 있다.

대응:

- 규정 허용 여부를 먼저 확인한다.
- train fold에서 pseudo-label simulation을 해보고 이득이 있을 때만 사용한다.
- 최종 후보는 pseudo-label 사용/미사용 모델을 둘 다 유지한다.

### Train/Val 전체 사용의 검증 공백

제공 validation set을 학습에 사용해도 무방하지만, 너무 일찍 합치면 local holdout 신호가 사라진다. 이 상태에서 public score만 보고 모델을 고르면 private overfit 위험이 커진다.

대응:

- baseline 재현 전에는 제공 val을 학습에 섞지 않는다.
- 설정 탐색은 train+val K-fold로 대체 검증을 만든 뒤 진행한다.
- 최종 train+val 재학습 모델은 fold 모델과 함께 후보로 유지한다.

## Decision Rule

새로운 실험을 채택하려면 아래 조건을 만족해야 한다.

- CV 평균 H-Mean이 개선된다.
- fold별 점수 하락이 제한적이다.
- precision/recall 중 하나가 비정상적으로 무너지지 않는다.
- prediction visualization에서 새 failure mode가 늘지 않는다.
- public score 개선만이 유일한 근거가 아니다.
- CLEval detection-only 조건으로 평가했다.
- 모든 제출 포맷 sanity check를 통과했다.
- JSON-image 파일명 매칭과 CSV row 수 검사를 통과했다.

## Final Candidate Table

최종 제출 전 아래 표를 채운다.

| Candidate | Local CV | Official Val | Public LB | Risk | Decision |
| --- | --- | --- | --- | --- | --- |
| stable | TBD | TBD | TBD | low | TBD |
| aggressive | TBD | TBD | TBD | medium/high | TBD |
