# Data

현재 대회 데이터는 `datasets/` 아래에 배치되어 있다.

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
│   ├── cord-v2/
│   ├── sroie/
│   └── wildreceipt/
└── splits/
```

원본 `datasets/` 파일은 직접 수정하지 않는다. 새 train/validation split manifest나 K-fold 파일은 `splits/`에 저장한다.

`pseudo_label/` 데이터는 외부/보조 데이터 후보로 취급한다. 사용 전에는 대회 규정 허용 여부와 local validation ablation을 반드시 확인한다.
