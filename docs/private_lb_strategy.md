# Private Leaderboard Strategy

## 핵심 원칙

Private leaderboard에서 높은 점수를 얻기 위한 가장 중요한 원칙은 public leaderboard를 직접 최적화하지 않는 것이다. 이 대회는 test 413장이 public/private 50:50으로 나뉘므로 public은 약 206장뿐이다. 영수증 촬영 조건과 레이아웃 편차가 커서 몇 번의 제출 점수만 보고 threshold나 후처리를 맞추면 private에서 쉽게 무너질 수 있다.

따라서 최종 의사결정 기준은 다음 순서로 둔다.

1. Internal cross-validation H-Mean
2. Official val H-Mean
3. Error analysis에서 확인한 failure mode 감소
4. Public leaderboard 점수

Public leaderboard는 "이상 징후 탐지" 용도에 가깝게 사용한다. local validation은 좋아졌는데 public만 나빠지는 경우는 split mismatch를 의심하고, public만 좋아졌는데 local validation이 나빠지는 경우는 public overfit 후보로 기록한다.

## CLEval 반영 전략

이 대회는 CLEval의 detection-only H-Mean으로 순위를 정한다. Ground Truth와 Prediction 모두 transcription 정보를 쓰지 않고, polygon label은 QUAD가 아니라 POLY 방식으로 평가된다. 따라서 validation도 반드시 official evaluator와 동일한 조건으로 맞춘다.

대회 적용 조건:

- Evaluation: CLEval detection-only
- Ranking metric: H-Mean, higher is better
- Score precision: 소수점 4번째 자리까지 계산
- Tie-break: 동점이면 먼저 제출한 팀 우선
- Box type: POLY
- Polygon: 4점 이상만 유효, 3점 이하는 무시
- Submission coverage: 모든 test image에 대해 결과 제출
- Region cap: 이미지당 최대 500개 text region, 500개 초과 영역은 평가 제외
- Transcription: 사용하지 않음
- Public/Private split: test 413장을 50:50으로 random split
- Public/Private balance: 이미지당 평균 words 수가 동일하게 분배됨

전략적으로 중요한 변화:

- IoU 0.5 같은 임계값에만 맞춘 박스 최적화는 우선순위를 낮춘다.
- Word 하나가 `RIVERSIDE`인데 `RIVER`, `SIDE`처럼 나뉘는 경우도 완전 실패가 아닐 수 있다. 다만 반복적인 split/merge는 granularity penalty로 손해를 볼 수 있으므로 일부러 과분할하는 전략은 금지한다.
- 너무 크게 부풀린 polygon은 문자 일부를 놓치는 문제는 줄일 수 있지만 non-text 영역이 커져 precision을 해칠 수 있다.
- 여러 단어 또는 여러 줄을 하나의 큰 polygon으로 합치는 후처리는 private에서 위험하다. CLEval이 split/merge를 다루더라도 merged detection은 정밀도와 granularity 측면에서 손해가 날 수 있다.
- 최종 모델 선택은 CLEval H-Mean뿐 아니라 CLEval precision, CLEval recall, 이미지당 region count 분포를 함께 본다.
- Public과 Private가 random 50:50이고 평균 words 수가 맞춰져 있어 public score는 어느 정도 참고 가치가 있다. 그래도 public 약 206장만 보고 threshold를 반복 조정하면 private 약 207장에 과적합될 수 있다.

## Data Layout

현재 로컬 데이터는 다음 구조다.

```text
data/datasets/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── jsons/
│   ├── train.json
│   ├── val.json
│   └── test.json
└── sample_submission.csv
```

로컬 확인 결과:

- `train.json`: 3272 images, `images/train`과 1:1 매칭
- `val.json`: 404 images, `images/val`과 1:1 매칭
- `test.json`: 413 images, `images/test`와 1:1 매칭

대회 안내에는 train 3273장으로 적힌 설명도 있으므로, 실험 로그에는 항상 실제 로컬 count를 기록한다. 데이터가 갱신되거나 압축 해제가 다시 이루어진 경우 `jsons/*.json`과 `images/*`의 파일명 매칭을 다시 검사한다.

JSON 구조:

- 최상위 key는 `images`
- `images` 아래 key는 경로 없는 이미지 파일명
- 각 이미지 항목에는 `words`, `img_w`, `img_h`가 있음
- `words` key는 `0001`부터 시작하는 0-padded 4자리 index
- 각 word에는 `points`가 있고, 좌표는 원본 이미지 기준 `(x, y)`
- polygon은 최소 4점 이상이어야 평가 대상이 됨
- 좌표 기준점은 이미지 좌측 상단 `(0, 0)`

`test.json`은 baseline 추론 결과와 같은 형식의 prediction JSON을 만들기 위한 기준 파일로 사용한다. 최종 제출은 JSON을 CSV로 변환한다.

## Baseline

첫 baseline은 DBNet 계열을 추천한다. 이 대회의 공개 베이스라인도 DBNet 흐름이고, 영수증은 작은 글자가 많고 줄 단위 text region이 조밀하게 배치되어 있어 segmentation-based detector가 시작점으로 좋다.

추천 순서:

1. 공식 baseline DBNet 재현
2. DBNet + stronger backbone / pretrained weight
3. DBNet++ 또는 MMOCR/PaddleOCR 구현체 비교
4. CRAFT는 보조 모델 또는 ensemble 후보로 비교

초반에는 모델을 바꾸기보다 evaluation, submission format, post-processing threshold를 먼저 안정화한다. 잘못된 좌표 scaling, polygon order, image resize 복원 버그가 모델 차이보다 큰 손실을 만들 수 있다.

특히 local evaluator는 `--BOX_TYPE=POLY`와 detection-only 설정을 먼저 검증한다. 공개 CLEval README는 end-to-end 평가와 transcription 옵션도 설명하지만, 이 대회 제출에서는 transcription을 넣지 않는 쪽으로 맞춘다.

## Validation 설계

제공된 train/val 구분은 학습 편의를 위한 것이며, 대회 설명상 다른 기준으로 재분류하거나 validation set을 학습에 사용해도 된다. 그래서 초반과 후반의 validation 운영을 나눈다.

권장 운영:

1. Baseline 단계: 제공 train으로 학습하고 제공 val 404장으로 공식 baseline 점수를 재현한다.
2. 탐색 단계: train+val 전체에서 stratified K-fold를 만들어 하이퍼파라미터, augmentation, post-processing을 비교한다.
3. 최종 단계: 설정을 고정한 뒤 train+val 전체 재학습 모델과 fold ensemble 후보를 비교한다.

권장 split:

- 5-fold stratified split
- Stratification feature: image height/width, aspect ratio, text region count, 평균 polygon 면적, blur/darkness score, receipt orientation
- 가능하면 파일명/source prefix 기준 group leakage 확인
- split manifest는 `data/splits/`에 저장한다.

실험 채택 기준:

- 평균 fold H-Mean이 오른다.
- 특정 fold 하나만 크게 좋아진 결과는 보류한다.
- Precision만 오르고 recall이 크게 떨어지는 후처리는 보류한다.
- Recall만 오르고 false positive가 늘어난 경우 OCR pipeline 관점에서 risk로 표시한다.
- 이미지당 predicted region count가 500개에 가까워지는 실험은 보류한다.
- polygon 점 개수, polygon 유효성, 좌표 복원 오류가 있는 실험은 점수가 좋아도 폐기한다.

## EDA 체크리스트

처음 하루는 모델 학습보다 EDA와 visualization에 투자한다.

- 긴 영수증, 매우 짧은 영수증 분리
- 기울어진 촬영, perspective distortion
- 어두운 배경, 책상/손/그림자 포함 이미지
- 흐림, 흔들림, 저해상도, JPEG artifact
- 바코드, QR, 로고, 표, 선, 가격 구분선
- 작은 글자와 굵은 총액 영역
- polygon label이 word-level인지 line-level인지, ignore label이 있는지
- text region이 receipt 밖 배경에도 있는지

이 분석 결과를 augmentation과 post-processing tuning의 근거로 사용한다.

## Augmentation 전략

Private leaderboard 방어를 위해 현실 촬영 변형을 넓게 커버한다.

우선순위 높은 augmentation:

- Random resize / scale jitter
- Mild rotation
- Perspective transform
- Motion blur / Gaussian blur
- JPEG compression
- Brightness / contrast / gamma
- Shadow / uneven illumination
- Random crop with text-preserving constraints

주의할 augmentation:

- 너무 강한 rotation: 실제 영수증 분포보다 과하면 precision이 흔들릴 수 있다.
- 과도한 color jitter: thermal receipt의 약한 글자를 지울 수 있다.
- aggressive crop: 긴 영수증의 context를 깨뜨릴 수 있다.
- mixup/cutmix: text polygon detection에서는 label noise를 만들기 쉽다.

## Input Resolution

영수증 text는 작고 조밀하므로 resolution이 중요하다. 단, 무작정 키우면 batch size가 줄고 학습 안정성이 낮아진다.

실험 후보:

- short side 736 / 896 / 1024
- long side cap 1280 / 1536 / 1600
- test-time multi-scale: 0.75x, 1.0x, 1.25x

입력 크기 실험은 반드시 inference time, memory, validation H-Mean을 같이 기록한다.

## Post-processing

DBNet 계열에서는 binarization threshold, box threshold, unclip ratio, min text size가 매우 중요하다. 이 값들은 public leaderboard가 아니라 fold validation에서 grid search한다.

권장 탐색:

- bin_thresh: 0.2 - 0.4
- box_thresh: 0.45 - 0.75
- unclip_ratio: 1.3 - 2.2
- min_text_size: 2 - 8

CLEval은 character-level 관점에서 split, merge, missing, overlapping detection을 더 세밀하게 반영한다. 따라서 post-processing은 다음 원칙으로 조정한다.

- Over-unclip 금지: recall이 좋아져도 non-text 영역이 커져 precision이 떨어질 수 있다.
- Over-merge 금지: 가까운 단어, 가격 열, 여러 줄을 한 polygon으로 합치면 public 일부에서는 좋아져도 private에서 불안정하다.
- Over-split 금지: 한 단어를 둘로 나누는 것은 완전 실패는 아니지만 반복되면 granularity penalty가 누적될 수 있다.
- Tiny FP 관리: 1-2픽셀 잡음, 선, 바코드 조각은 false positive character로 precision을 낮출 수 있다.
- Confidence/order 관리: 이미지당 500개 제한에 걸릴 가능성이 있으면 confidence 기준 정렬 후 상위 500개 이하로 제한한다.

최종 점수 H-Mean이 같다면 private에서는 더 안정적인 fold variance, 낮은 500개 cap risk, 균형 잡힌 precision/recall을 우선한다.

## Ensemble / TTA

최종 상위권을 노릴 때는 단일 모델보다 ensemble과 TTA가 유리할 수 있다. 하지만 detection ensemble은 false positive가 쉽게 늘기 때문에 private risk가 있다.

안전한 순서:

1. 같은 모델의 multi-seed ensemble
2. fold ensemble
3. scale TTA
4. DBNet + CRAFT처럼 다른 detector ensemble

가능하면 polygon 결과를 합치는 것보다 probability map을 평균낸 뒤 한 번만 post-processing하는 방식이 안정적이다. 구현체가 polygon만 제공한다면 IoU 기반 merge/NMS를 validation에서 충분히 검증한다.

CLEval 환경에서는 polygon ensemble이 특히 조심스럽다. 서로 약간 어긋난 polygon을 모두 남기면 duplicate/overlap 성격의 false positive가 늘 수 있다. Ensemble 채택 전에는 이미지별 prediction count와 duplicate polygon 비율을 반드시 확인한다.

## Submission 운영

모든 제출은 `submissions/submission_log.md`에 기록한다.

기록 항목:

- submission file
- git commit or code snapshot
- config
- seed
- train data 범위
- local fold score
- official val score
- public leaderboard score
- 선택 이유
- private overfit risk memo
- 이미지당 최대 predicted region count
- invalid polygon count
- 500개 cap 적용 여부
- prediction JSON path
- converted CSV path
- CSV conversion command

최종 제출 후보는 최소 2개로 유지한다.

- Stable candidate: local validation 최상위이면서 fold variance가 낮은 모델
- Aggressive candidate: public score가 높지만 local 근거가 약간 부족한 모델

최종 선택은 원칙적으로 stable candidate를 우선한다.

## Final Training Policy

하이퍼파라미터, augmentation, post-processing이 모두 고정된 뒤에만 train + val 전체 재학습을 고려한다. train + val 재학습은 private 성능을 올릴 수 있지만, validation 신호를 잃는 trade-off가 있다.

권장 방식:

1. CV와 official val에서 최종 설정 확정
2. 같은 설정으로 train-only 모델과 train+val 모델을 둘 다 생성
3. test prediction visualization 50장 이상 점검
4. public score가 크게 흔들리지 않으면 train+val 모델을 최종 후보로 둔다

## Submission Sanity Check

제출 직전 자동 검사 스크립트를 만든다.

- 모든 test image id가 존재한다.
- CSV header가 `filename,polygons`이다.
- CSV row 수가 test image 수 413개와 일치한다.
- `filename`은 경로 없이 이미지 파일명만 기록한다.
- `polygons` 열에서 word들은 `|`로 구분한다.
- polygon 내부 좌표는 공백 문자로 `X Y X Y ...` 형식으로 기록한다.
- 이미지별 predicted region 수가 500개 이하이다.
- 모든 polygon은 4점 이상이다.
- 각 polygon은 짝수 개 좌표를 갖는다.
- 모든 좌표는 원본 이미지 좌표계로 복원되었다.
- 좌표에 NaN, inf, 음수, 이미지 밖 과도한 값이 없다.
- polygon vertex order가 evaluator에서 유효하게 해석된다.
- 면적이 0이거나 self-intersection이 심한 polygon을 제거했다.
- confidence가 있다면 낮은 confidence부터 잘리지 않도록 정렬/필터링했다.

Baseline 변환 명령 예시:

```bash
python ocr/utils/convert_submission.py --json_path {json_path} --output_path {output_path}
```

CSV 예시:

```csv
filename,polygons
drp.en_ko.in_house.selectstar_003883.jpg,10 50 100 50 100 150 10 150|110 150 200 150 200 250 110 250
```

## Sources

- CLEval paper: https://arxiv.org/abs/2006.06244
- CLEval official implementation: https://github.com/clovaai/CLEval
- DBNet paper: https://arxiv.org/abs/1911.08947
- DBNet++ paper: https://arxiv.org/abs/2202.10304
- CRAFT paper: https://arxiv.org/abs/1904.01941
- MMOCR text detection docs: https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html
- PaddleOCR DB/DB++ docs: https://www.paddleocr.ai/v2.10.0/en/algorithm/text_detection/algorithm_det_db.html
- Public baseline/reference repo found for this competition family: https://github.com/UpstageAILab/upstage-ai-final-ocr1
