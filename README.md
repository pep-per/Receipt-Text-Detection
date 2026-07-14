# Receipt Text Detection

영수증 이미지에서 text region polygon을 예측하는 OCR text detection 대회용 프로젝트입니다.

## 목표

- Private leaderboard H-Mean을 최우선으로 최적화한다.
- Public leaderboard 점수는 보조 신호로만 사용한다.
- 모든 실험은 재현 가능한 config, seed, submission, validation score로 남긴다.

## 최종 결과

- Final leaderboard: **1위**
- Best submission: V2B ResNet18, resolution 1024
- Final H-Mean / Precision / Recall: `0.9647 / 0.9739 / 0.9580`
- 결과 증빙과 Local/Public/Final 분석:
  [2026-07-14 Final Leaderboard](docs/leaderboard/20260714/README.md)

## 대회 종료 후 실험

- Current local inference candidate: V8 `1024+1152` probability-map TTA
- Official-val macro H/P/R: `0.9668 / 0.9725 / 0.9628`
- V2B 1024 control 대비 macro/global H: `+0.0021 / +0.0029`
- 상세 결과: [V8 Scale TTA](experiments/20260714-v8-scale-tta/README.md)
- 대회가 종료되어 V8에는 leaderboard 점수가 없으며, test CSV는 offline 재현 artifact다.

## 대회 요약

- Task: receipt text detection
- Train: 3,272 images in current local data
- Val: 404 images
- Test: 413 images
- Label: text region polygon coordinates
- Main metric: CLEval detection-only H-Mean, higher is better
- Evaluation box type: POLY
- Transcription: 사용하지 않음
- Submission rule: 모든 이미지 제출, 이미지당 최대 text region 500개, polygon은 4점 이상

## 폴더 구조

```text
receipt-text-detection/
├── configs/       # model, dataset, augmentation, inference configs
├── data/          # official dataset, pseudo-label candidates, split files
├── docs/          # strategy, risk checklist, experiment policy
├── experiments/   # per-run notes, metrics, selected artifacts
├── notebooks/     # EDA and visual inspection
├── scripts/       # train/eval/infer/submission helper scripts
├── src/           # project-specific code
└── submissions/   # generated csv submissions and submission log
```

## 첫 실행 순서

1. 데이터 개수와 JSON-image 매칭을 검증한다.
2. 공식 baseline을 그대로 재현한다.
3. 공식 validation 점수와 JSON/CSV submission 포맷을 검증한다.
4. local CLEval detection-only 평가를 먼저 재현한다.
5. train/val 전체를 대상으로 stratified K-fold split을 만든다.
6. augmentation, input size, post-processing threshold를 local validation 기준으로 탐색한다.
7. 제출 전 모든 이미지, polygon 점 개수, 500개 제한, 좌표 복원 sanity check를 통과시킨다.

자세한 전략은 [docs/private_lb_strategy.md](docs/private_lb_strategy.md)를 본다.
CLEval 설명은 [docs/cleval_korean_guide.md](docs/cleval_korean_guide.md)를 본다.
데이터와 제출 포맷은 [docs/data_submission_format.md](docs/data_submission_format.md)를 본다.
K-fold, augmentation, ensemble과 TTA의 차이는
[docs/kfold_augmentation_ensemble_tta_guide.md](docs/kfold_augmentation_ensemble_tta_guide.md)를
본다.
프로젝트 진행 중 사용자와 Codex가 나눈 질문, 결정과 실험 chronology는
[docs/conversation_archive_20260709-20260714.md](docs/conversation_archive_20260709-20260714.md)에
보존한다.

다른 머신으로 프로젝트를 옮겨 실험을 계속하는 방법은
[docs/transfer_and_resume.md](docs/transfer_and_resume.md)를 본다. 필수 데이터와 체크포인트는
`scripts/verify_resume_assets.sh`로 확인하고, 이식용 압축본은
`scripts/create_portable_archive.sh`로 다시 만들 수 있다.
