# Data And Submission Format

이 문서는 현재 로컬 데이터 구조와 대회 제출 포맷을 정리한다.

## 현재 데이터 구조

```text
data/
├── datasets/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── jsons/
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   └── sample_submission.csv
├── pseudo_label/
└── splits/
```

`raw/`, `processed/`는 초기에 데이터 위치를 모를 때 만든 빈 스캐폴드였고, 실제 대회 데이터 구조와 맞지 않아 삭제했다. `splits/`는 train/val 재분류와 K-fold manifest를 저장하기 위해 남긴다.

## 로컬 데이터 확인 결과

- Train: 3272 images
- Val: 404 images
- Test: 413 images
- `train.json`, `val.json`, `test.json`의 이미지 key는 각각 image directory의 파일명과 1:1로 매칭됨

대회 안내에 train 3273장으로 쓰인 설명이 있으므로, 실험 시작 시 데이터 개수 검증 결과를 로그에 남긴다.

## JSON 구조

각 JSON은 최상위에 `images` key를 갖는다.

```text
images
└── IMAGE_FILENAME
    ├── words
    │   └── 0001
    │       ├── points
    │       ├── orientation
    │       └── language
    ├── img_w
    └── img_h
```

주요 규칙:

- `IMAGE_FILENAME`은 경로 없는 이미지 파일명이다.
- `words`의 index는 `0001`부터 시작하는 0-padded 4자리 정수 문자열이다.
- `points`는 원본 이미지 좌표계 기준 polygon 좌표다.
- 좌표 기준점은 이미지 좌측 상단 `(0, 0)`이다.
- 각 point는 `[x, y]` 형태다.
- polygon은 최소 4점 이상이어야 평가 대상이다.
- 4점 미만 polygon은 평가에서 예외 처리된다.

`test.json`의 `words`는 비어 있으며, baseline 추론 결과는 이 파일과 같은 JSON 구조로 prediction을 채우는 방식이다.

## Train/Val 사용 정책

제공된 train/val 구분은 학습 편의를 위한 것이다. 대회 설명상 다른 기준으로 재분류하거나 validation set을 학습에 사용해도 된다.

추천 운영:

- Baseline 재현: 제공 train으로 학습, 제공 val로 평가
- 탐색: train+val 전체에서 stratified K-fold 생성
- 최종: 설정 고정 후 train+val 전체 재학습 또는 fold ensemble 검토

## Public/Private

평가 데이터 413장은 public/private로 나뉘지만 어떤 이미지가 private인지는 공개되지 않는다.

- Public: 50%
- Private: 50%
- Public과 Private는 random하게 섞여 있음
- 이미지당 평균 words 수가 동일하게 분배됨
- 대회 중에는 Public 점수만 표시됨
- 대회 종료 후 Private 기준 최종 점수와 순위가 공개됨

Public과 Private가 random split이고 평균 words 수가 맞춰져 있어 public score는 참고 가치가 있다. 하지만 public은 약 절반뿐이므로, 반복 제출로 threshold를 맞추면 private overfit이 생길 수 있다.

## CSV 제출 포맷

최종 제출은 CSV 파일이다.

헤더:

```csv
filename,polygons
```

데이터행 형식:

```text
IMAGE_FILENAME,X Y X Y X Y X Y|X Y X Y X Y X Y|...
```

규칙:

- `filename` 열에는 경로 없는 평가 이미지 파일명을 기록한다.
- `polygons` 열에는 각 text region polygon을 기록한다.
- polygon 내부 좌표는 공백 문자로 구분한다.
- 두 번째 이후 word polygon은 `|`로 구분한다.
- 모든 평가 이미지에 대해 행을 작성해야 한다.
- 이미지당 text region은 최대 500개까지 평가된다.
- 500개를 초과한 text region은 평가 대상에서 제외된다.

예시:

```csv
filename,polygons
drp.en_ko.in_house.selectstar_003883.jpg,10 50 100 50 100 150 10 150|110 150 200 150 200 250 110 250
drp.en_ko.in_house.selectstar_000132.jpg,10 50 100 50 100 150 10 150|110 150 200 150 200 250 110 250
```

## JSON to CSV 변환

Baseline 기준 추론 결과는 test JSON과 같은 형식의 JSON으로 나온다. 제공 변환 툴을 사용해 CSV로 변환한다.

```bash
python ocr/utils/convert_submission.py --json_path {json_path} --output_path {output_path}
```

변환 후에는 반드시 sanity check를 실행한다.

- CSV row 수가 413개인지 확인
- 모든 test filename이 포함되었는지 확인
- polygon 좌표 개수가 짝수인지 확인
- polygon이 4점 이상인지 확인
- 이미지당 polygon 수가 500개 이하인지 확인
- 좌표가 원본 이미지 기준인지 확인
