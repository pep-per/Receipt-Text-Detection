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

현재 고정 split:

- `splits/v11_5fold_seed42.csv`: train+val 3,676장의 group-aware 5-fold manifest
- `splits/v11_5fold_seed42_metadata.json`: 생성 규칙, checksum, validation 상태
- 생성 명령: `python scripts/v11_make_folds.py`

V11B 이후에는 fold score를 보고 이 manifest를 다시 생성하거나 seed를 바꾸지 않는다.

`pseudo_label/` 데이터는 외부/보조 데이터 후보로 취급한다. 사용 전에는 대회 규정 허용 여부와 local validation ablation을 반드시 확인한다.
