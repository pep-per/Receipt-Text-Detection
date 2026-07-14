# 2026-07-14 Final Leaderboard Evidence

## 출처

2026-07-14 대회 종료 후 사용자가 대화에 첨부한 다음 세 화면에서 수치를 전사했다.

- Final leaderboard, last update `2026.07.14 14:21:17`
- Mid/Public leaderboard, last update `2026.07.14 14:25:39`
- Submission list with Public and Final values for V0, V1, V2, V2B

현재 작업 환경에는 대화 첨부 이미지의 원본 binary가 파일 경로로 노출되지 않았다. 따라서
`*.transcribed.png`는 원본 screenshot 자체가 아니라, 화면에서 읽은 수치와 구조를 보존한
증빙용 재구성 이미지다. 이미지 안에도 `Transcribed from supplied screenshot`라고 표시했다.
정량 분석에는 이미지 OCR 결과가 아니라 사람이 확인한 아래 CSV 값을 사용한다.

- Raw scores: [final_scores.csv](final_scores.csv)
- Final leaderboard evidence: [leaderboard_final.transcribed.png](leaderboard_final.transcribed.png)
- Mid leaderboard evidence: [leaderboard_mid.transcribed.png](leaderboard_mid.transcribed.png)
- Submission list evidence: [submission_list.transcribed.png](submission_list.transcribed.png)
- Reproduction source: [leaderboard_evidence.html](leaderboard_evidence.html)

## 제출별 결과

| Model | Local H/P/R | Public H/P/R | Final H/P/R | Final - Local H |
| --- | --- | --- | --- | ---: |
| V0 | 0.8913 / 0.9633 / 0.8369 | 0.8818 / 0.9651 / 0.8194 | 0.8898 / 0.9675 / 0.8324 | -0.0015 |
| V1 | 0.9248 / 0.9499 / 0.9057 | 0.9185 / 0.9511 / 0.8932 | 0.9221 / 0.9554 / 0.8978 | -0.0027 |
| V2 | 0.9615 / 0.9638 / 0.9611 | 0.9603 / 0.9667 / 0.9556 | 0.9637 / 0.9682 / 0.9606 | +0.0022 |
| V2B | 0.9648 / 0.9700 / 0.9614 | 0.9621 / 0.9754 / 0.9520 | **0.9647 / 0.9739 / 0.9580** | -0.0001 |

## 핵심 분석

### Local 선택은 Final에서도 유지됐다

H-Mean 순서가 Local, Public, Final에서 모두 `V0 < V1 < V2 < V2B`로 같다. 네 점에 대한
Spearman rank correlation은 `1.0`, Local-Final Pearson correlation은 약 `0.9991`이다. 표본이
4개뿐이고 모델들이 서로 독립적이지 않아 일반적인 통계적 증명으로 보기는 어렵지만, 이번
실험 계열 안에서는 local CLEval이 모델 선택 방향을 정확히 보존했다.

Local-Final H 차이의 평균 절댓값은 `0.001625`, RMSE는 `0.001897`, 최대 절댓값은
`0.0027`이었다. 기존 official val 단일 점수가 hidden Final의 절대 수준도 꽤 잘 근사했다.

### 모든 채택 변경의 H 개선 방향이 전이됐다

| Transition | Local ΔH | Public ΔH | Final ΔH | Final 해석 |
| --- | ---: | ---: | ---: | --- |
| V0 -> V1, box threshold | +0.0335 | +0.0367 | +0.0323 | 큰 recall 회복이 그대로 전이 |
| V1 -> V2, resolution 896 | +0.0367 | +0.0418 | +0.0416 | 가장 큰 Final 개선 |
| V2 -> V2B, resolution 1024 | +0.0033 | +0.0018 | +0.0010 | 방향은 유지됐지만 이득은 축소 |

V2B는 V2보다 Final precision이 `+0.0057`, recall이 `-0.0026`, H가 `+0.0010`이다. 1024의
효과가 small-text recall 향상보다는 더 정확한 경계와 false-positive 억제에 가까웠다는 기존
해석을 지지한다. 다만 향상 폭이 local 예상의 약 31%이므로 앞으로 `0.001` 안팎의 단일-split
개선은 bootstrap/K-fold 없이 확정하지 않는다.

### Public보다 Final이 전반적으로 높았다

Final-Public H 차이는 V0 `+0.0080`, V1 `+0.0036`, V2 `+0.0034`, V2B `+0.0026`이었다.
V2B에서는 Final recall이 Public보다 `+0.0060`, precision은 `-0.0015`였다. Public 절반이
상대적으로 recall이 어려운 표본이었을 가능성이 있지만, 이미지별 hidden label이 없어 원인을
확정할 수는 없다. Public의 낮은 recall을 보고 threshold를 추가 조정하지 않은 판단은 Final에서
H가 유지된 점을 고려하면 적절했다.

### 최종 순위

팀 `UP24_이지연`은 Final H/P/R `0.9647 / 0.9739 / 0.9580`으로 1위였다. 2위 H `0.9452`보다
`+0.0195`, 3위 H `0.9178`보다 `+0.0469` 높았다. 네 제출 가운데 최고 Final 모델은 V2B였다.

## 앞으로의 평가에 주는 의미

- Local CLEval은 계속 primary metric으로 사용할 충분한 실제 근거가 생겼다.
- 네 제출의 순위 일치는 강한 긍정 신호지만 표본 수가 작으므로 official val 반복 튜닝 위험이
  사라진 것은 아니다.
- V8/V9는 paired bootstrap과 macro/global CLEval을 함께 사용한다.
- V10 이후 학습 recipe는 고정 K-fold OOF로 안정성을 검증한다.
- 앞으로 생성하는 test CSV는 inference 재현 artifact이며, leaderboard가 닫혔으므로 hidden
  성능 근거로 표현하지 않는다.

