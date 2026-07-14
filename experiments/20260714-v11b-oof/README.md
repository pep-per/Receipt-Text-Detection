# V11B Five-fold Training And OOF Recipe Evaluation

## Status

Completed. The fixed-epoch recipe failed the pre-registered stability gates.

## Objective

V11A에서 고정한 5-fold manifest로 V2B supervised recipe의 split 안정성을 평가한다. 각
이미지는 자신의 human polygon을 학습하지 않은 model 하나에서만 OOF prediction을 받는다.
기존 V2B checkpoint는 original train label을 이미 보았으므로 재사용하지 않는다.

## Fixed Recipe

- Data: official train+val 3,676 human-labeled images only
- Fold manifest: `data/splits/v11_5fold_seed42.csv`
- Models: five fresh ImageNet-pretrained ResNet18 DB detectors
- Train/held-out size: about 2,941 / 735 images per fold
- Resolution: 1024
- Optimizer: Adam, LR `0.001`, weight decay `0.0001`
- Scheduler: StepLR step 100, so LR remains constant during 10 epochs
- Epochs: 10
- Training seed: 42 for every fold
- Post-processing: pixel threshold `0.30`, box threshold `0.25`
- W&B: one training run per fold
- In-training validation: epoch 8 once; independent OOF evaluation follows checkpoint reload

## Checkpoint Policy

Primary OOF evaluation uses **fixed epoch 8 for every fold**. Fold별 held-out H가 가장 높은 epoch를
고르면 validation fold가 checkpoint selection에도 사용되어 OOF가 낙관적으로 편향될 수 있다.
Epoch 8은 V2B에서 이미 채택된 시점이므로 새 fold score를 보기 전에 고정한다.

Fold-best checkpoint는 학습 진단용 top-1으로만 보존하고 primary OOF 수치에는 사용하지 않는다.
Retention policy와 fit 후 중복 test 생략은 gradient, optimizer 또는 model output을 바꾸지 않는다.
마찬가지로 epoch 0~7과 9의 validation을 생략한다. Validation은 gradient update에 참여하지
않으며 primary epoch가 이미 8로 고정돼 있으므로, 735장 CLEval을 매 epoch 반복하는 CPU 비용만
줄인다. 이 실행 정책은 fold 0 첫 시도를 epoch 1에서 중단하고 어떤 checkpoint도 만들기 전에
고정했으며 모든 fold를 처음부터 같은 조건으로 실행한다.

## Pre-registered Stability Gates

- 3,676장 모두 정확히 하나의 OOF prediction을 가짐
- Fold macro H 표준편차 `<= 0.005`
- Fold global H 표준편차 `<= 0.005`
- Worst-fold macro H `>= 0.955`
- Worst-fold global H `>= 0.955`

절대 gate는 기존 official-val 점수를 복제한다는 뜻이 아니라 catastrophic split failure를 막는
하한이다. Primary 결과는 pooled OOF macro/global H와 fold 평균, 표준편차, worst fold를 함께
보고한다. Fold 결과를 본 뒤 manifest, epoch, seed 또는 gate를 바꾸지 않는다.

## Commands

```bash
python scripts/v11b_prepare_fold_data.py
```

Fold 학습 명령은 fold 번호만 바꾸며, 각 실행의 Hydra config에 모든 override가 저장된다.

```bash
cd baseline_code
python runners/train.py \
  preset=example \
  exp_name=v11b_fold0 \
  project_name=receipt-text-detection \
  wandb=true \
  exp_version=v11b-fold0 \
  dataset_base_path="/data/ephemeral/home/receipt-text-detection/data/datasets/" \
  datasets.train_dataset.image_path="/data/ephemeral/home/receipt-text-detection/data/v11_folds/fold_0/train_images" \
  datasets.train_dataset.annotation_path="/data/ephemeral/home/receipt-text-detection/data/v11_folds/all_annotations.json" \
  datasets.val_dataset.image_path="/data/ephemeral/home/receipt-text-detection/data/v11_folds/fold_0/val_images" \
  datasets.val_dataset.annotation_path="/data/ephemeral/home/receipt-text-detection/data/v11_folds/all_annotations.json" \
  datasets.test_dataset.image_path="/data/ephemeral/home/receipt-text-detection/data/v11_folds/fold_0/val_images" \
  datasets.test_dataset.annotation_path="/data/ephemeral/home/receipt-text-detection/data/v11_folds/all_annotations.json" \
  trainer.max_epochs=10 \
  trainer.num_sanity_val_steps=0 \
  trainer.check_val_every_n_epoch=9 \
  +trainer.accelerator=gpu \
  +trainer.devices=1 \
  +fixed_checkpoint_epoch=8 \
  +checkpoint_save_top_k=1 \
  +checkpoint_save_last=false \
  +test_after_fit=false \
  transforms.train_transform.transforms.0.max_size=1024 \
  transforms.train_transform.transforms.1.min_width=1024 \
  transforms.train_transform.transforms.1.min_height=1024 \
  transforms.val_transform.transforms.0.max_size=1024 \
  transforms.val_transform.transforms.1.min_width=1024 \
  transforms.val_transform.transforms.1.min_height=1024 \
  transforms.test_transform.transforms.0.max_size=1024 \
  transforms.test_transform.transforms.1.min_width=1024 \
  transforms.test_transform.transforms.1.min_height=1024 \
  models.head.postprocess.box_thresh=0.25
```

실제 순차 실행에는 동일 override를 고정한 wrapper를 사용한다.

```bash
bash scripts/v11b_train_fold.sh 0
```

각 fold 학습 후 다음을 실행한다.

```bash
python scripts/v11b_evaluate_oof.py --fold 0
python scripts/v11b_evaluate_oof.py --aggregate
```

## Expected Artifacts

- `baseline_code/outputs/v11b_fold{0..4}/checkpoints/fixed-epoch=8.ckpt`
- `experiments/20260714-v11b-oof/fold_{0..4}/per_image.csv`
- `experiments/20260714-v11b-oof/fold_{0..4}/predictions.json`
- `experiments/20260714-v11b-oof/fold_metrics.csv`
- `experiments/20260714-v11b-oof/oof_per_image.csv`
- `experiments/20260714-v11b-oof/oof_metrics.json`
- `experiments/20260714-v11b-oof/strata_metrics.csv`

## Results

All 3,676 images received exactly one prediction from a model that did not train on that image's
human annotation.

| Fold | Images | Macro H | Macro P | Macro R | Global H | Global P | Global R |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 735 | 0.952515 | 0.953083 | 0.954574 | 0.950227 | 0.948197 | 0.952265 |
| 1 | 735 | 0.956353 | 0.956109 | 0.958670 | 0.954015 | 0.952537 | 0.955498 |
| 2 | 736 | 0.949446 | 0.951732 | 0.950908 | 0.952822 | 0.955247 | 0.950409 |
| 3 | 735 | 0.932941 | 0.942488 | 0.933583 | 0.936876 | 0.942309 | 0.931505 |
| 4 | 735 | 0.943430 | 0.950280 | 0.940513 | 0.944895 | 0.951926 | 0.937967 |

Pooled OOF results:

- Macro H/P/R: `0.946938 / 0.950739 / 0.947650`
- Global H/P/R: `0.947775 / 0.950045 / 0.945515`
- Fold macro H mean/std/min/max: `0.946937 / 0.009142 / 0.932941 / 0.956353`
- Fold global H mean/std/min/max: `0.947767 / 0.007027 / 0.936876 / 0.954015`
- Total predictions: `405,120`, mean `110.21` per image
- Prediction sanity: maximum `241` regions per image, no image over 500, no polygon below four points
- Raw DB unclip output: `159` polygons in `79` images cross the source-image boundary; retained
  unchanged for the registered baseline comparison

Gate results:

| Gate | Result |
| --- | --- |
| Exactly one OOF prediction per image | Pass |
| Macro H fold standard deviation `<= 0.005` | Fail: `0.009142` |
| Global H fold standard deviation `<= 0.005` | Fail: `0.007027` |
| Worst-fold macro H `>= 0.955` | Fail: `0.932941` |
| Worst-fold global H `>= 0.955` | Fail: `0.936876` |

## Diagnosis

Fold 3 was not low only for one obvious image stratum. Its macro H was lower for both
`original_split=train` and `original_split=val`, both small-text bins, and all five GT-region-count
bins. Fold 3 also lost both precision and recall, while fold 4 mainly had lower recall. This makes a
single post-processing threshold explanation unlikely. The locked manifest had already balanced the
measured source, word-count, scale, brightness, contrast and blur features, so the observed spread is
evidence that the constant-LR V2B training recipe is sensitive to which 20% is held out, to
unmeasured receipt clusters, or to both.

Checkpoint integrity checks passed. Every primary checkpoint reports `epoch=8`,
`global_step=1656`, and its state dict is tensor-for-tensor identical to the independently
saved top checkpoint from the same epoch. Independent reload evaluation also agrees with the
in-training result within evaluator-level rounding differences.

There is one important budget confound. The original V2B epoch-8 checkpoint used 3,272 training
images and `1,845` optimizer updates. Each OOF model used about 2,941 training images, so its
epoch-8 checkpoint has only `1,656` updates, `10.2%` fewer. Therefore V11B validly rejects the
literal fixed-epoch-8 policy, but it does not yet prove that the architecture or split itself is
unstable under a matched update budget.

Hard strata across pooled OOF were high GT-region count (macro H `0.93035`), high small-text ratio
(`0.93297`), low median text short side (`0.93239`) and low blur-Laplacian quartile (`0.93959`). These
results support keeping scale-aware inference and small-text diagnostics, but do not justify a new
augmentation bundle by themselves.

## Decision And Next Step

- Do not promote these checkpoints directly to V11C stable teacher or generate a test submission.
- Do not alter the locked fold manifest, seed, threshold or gates after seeing these scores.
- Run `V11B-R1` as a matched-update-budget correction: resume every fold's fixed epoch-8 checkpoint
  for exactly one more epoch and evaluate the fixed epoch-9 state at about `1,840` updates.
- Apply the same five-fold aggregate and the same stability gates. Do not choose epoch 8 versus 9
  separately per fold.
- If V11B-R1 still fails, measure seed sensitivity before building V11C; do not hide the variance by
  averaging all five models on data whose labels some ensemble members saw.

## W&B Runs

- [Fold 0](https://wandb.ai/pep-per/receipt-text-detection/runs/yf55qqe5)
- [Fold 1](https://wandb.ai/pep-per/receipt-text-detection/runs/h76l2wtw)
- [Fold 2](https://wandb.ai/pep-per/receipt-text-detection/runs/8n4k2l9y)
- [Fold 3](https://wandb.ai/pep-per/receipt-text-detection/runs/ueh9cyrj)
- [Fold 4](https://wandb.ai/pep-per/receipt-text-detection/runs/3p3ipctk)
- [OOF aggregate](https://wandb.ai/pep-per/receipt-text-detection/runs/ca01s6al)
