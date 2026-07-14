# V10 Domain Self-supervised Pilot

## Status

Completed. Rejected as a statistical tie; keep the ImageNet V2B initialization for V11.

## Question

영수증 image-only MoCo v2 pretraining을 거친 ResNet18 encoder가 기존 ImageNet 초기화보다
동일한 DBNet supervised fine-tuning에서 더 높은 CLEval을 만드는지 확인한다.

공식 val/test 이미지를 SSL에서 label 없이 사용하므로 이 결과는 strict inductive CV가 아니라
`transductive local`이다. SSL loader는 JSON, polygon, transcription을 열지 않는다.

## Fixed Control And Pilot

- Control C0: V2B. ImageNet ResNet18, 1024 DBNet, seed 42, 10 epochs, Adam 0.001,
  StepLR, `box_thresh=0.25`.
- Pilot P0: ImageNet ResNet18 -> 아래 MoCo v2 -> C0와 동일한 DBNet fine-tuning.
- 유일한 supervised 차이는 encoder initialization이다.
- C0 independent reload macro H/P/R: `0.964760 / 0.969976 / 0.961422`.
- C0 independent reload global H/P/R은 V2B prediction을 같은 evaluator로 다시 산출해 비교한다.

## Pre-registered SSL Recipe

- Library: `lightly==1.5.25`; MoCo projection head, momentum update, NT-Xent memory bank 사용
- Encoder: timm ResNet18, ImageNet initialization, output dimension 512
- Image pool: official train/val/test + CORD-v2/SROIE/WildReceipt, 총 7,772장
- Sampling: official을 한 family로 묶고 CORD-v2/SROIE/WildReceipt와 함께 4개 family의 기대
  sampling 비율을 동일하게 맞춘 replacement sampling
- Input/batch/epochs: 224 / 128 / 20
- Optimizer: SGD, lr 0.03, momentum 0.9, weight decay 0.0001, step-wise cosine decay
- MoCo: projection 512-2048-128, temperature 0.1, memory bank 4,096, encoder momentum
  0.996에서 1.0 cosine schedule
- Text-preserving views: crop scale 0.7 이상, mild color jitter 0.5, grayscale 0.1,
  Gaussian blur 0.2, flip/rotation 없음
- SSL checkpoint selection: label metric을 보지 않고 고정한 final epoch 20만 사용

Batch 128이 기술적으로 OOM이면 algorithm이나 metric을 보지 않고 batch 64와 선형 축소한
lr 0.015로 한 번만 재시작한다. 이 변경은 결과 문서에 남긴다.

## Adoption Gate

1. 독립 checkpoint reload에서 macro와 global H가 C0보다 모두 높아야 한다.
2. Precision 또는 recall 한쪽의 큰 붕괴, invalid polygon, 500개 초과가 없어야 한다.
3. 사실상 동률이면 추가 비용 때문에 C0를 유지한다.
4. 결과를 보고 SSL algorithm, epoch, augmentation을 sweep하지 않는다.

통과하면 V11 K-fold의 shared initialization 후보로 채택한다. 실패하면 V11은 ImageNet
initialization으로 진행한다.

## Result

MoCo pretraining은 사전 고정한 설정을 변경하지 않고 20 epochs, 1,200 steps를 완료했다.
Final epoch loss는 `7.852832`, runtime은 738초, peak GPU memory는 2.29 GB였다. Batch 128이
정상 동작해 OOM fallback은 사용하지 않았다.

DBNet fine-tuning의 best는 epoch 8이었다. 독립 reload와 per-image evaluator의 작은 차이는
부동소수점 집계 차이이며 결론을 바꾸지 않는다.

| Method | Macro H | Macro P | Macro R | Global H | Global P | Global R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V2B ImageNet control | 0.964785 | **0.969979** | 0.961463 | 0.962249 | **0.967516** | 0.957039 |
| V10 MoCo SSL | **0.964897** | 0.964765 | **0.966682** | **0.962613** | 0.961242 | **0.963987** |
| Delta | +0.000112 | -0.005214 | +0.005219 | +0.000363 | -0.006274 | +0.006948 |

Prediction 수는 control 45,067개에서 SSL 44,648개로 419개 감소했다. Invalid point count,
non-finite coordinate, zero area, empty image와 이미지당 500개 초과는 모두 0이었다.

### Paired Bootstrap

| Metric | Delta mean | 95% CI | P(delta > 0) |
| --- | ---: | ---: | ---: |
| Macro H | +0.000129 | [-0.001548, +0.001852] | 0.5552 |
| Global H | +0.000382 | [-0.001709, +0.002546] | 0.6396 |
| Macro precision | -0.005194 | [-0.007046, -0.003445] | 0.0000 |
| Macro recall | +0.005231 | [+0.002860, +0.007829] | 1.0000 |
| Global precision | -0.006247 | [-0.008686, -0.004076] | 0.0000 |
| Global recall | +0.006956 | [+0.003658, +0.010568] | 1.0000 |

H의 두 confidence interval이 모두 0을 포함하고 개선 확률도 낮다. 따라서 H point estimate의
미세한 증가는 유의한 개선이 아니라 precision을 recall로 교환한 동률로 판정한다.

## Training Behavior

- Epoch 0/1은 H `0.9099 / 0.9398`로 V2B보다 빠르게 수렴했다.
- Epoch 3은 recall이 0.5059로 떨어져 H 0.5852의 단발성 instability가 발생했다.
- Epoch 4에 H 0.9573으로 회복했고 epoch 8에서 최고 H 0.9649를 기록했다.
- 빠른 초기 수렴은 확인했지만 최고 H 개선과 안정성은 확인하지 못했다.

## Decision

- V10은 adoption gate 3의 `사실상 동률이면 control 유지`에 따라 폐기한다.
- V11 K-fold는 ImageNet ResNet18 initialization과 V2B supervised recipe를 사용한다.
- MoCo는 기술적으로 정상 동작했으므로 성능 결과를 보고 SimSiam/MAE를 연속 탐색하는
  `V10-ALT` gate는 열지 않는다.
- 공식 val/test pixels를 SSL에서 보았으므로 V10 수치는 계속 `transductive local`로 표시한다.
- Test CSV는 모델 채택이 아니라, 이전의 offline artifact 요청과 recall-diverse 후보 보존을
  위해 생성했다. Hidden score 개선 증거로 사용하지 않는다.

원인별 수치, CLEval state 변화, 개선 우선순위와 재실험 gate는
[SSL Effect Analysis And Revisit Plan](ssl_effect_analysis.md)에 기록한다.

## Commands

```bash
python scripts/v10_ssl_moco.py

cd baseline_code
python runners/train.py preset=example \
  dataset_base_path=/data/ephemeral/home/receipt-text-detection/data/datasets/ \
  exp_name=v10_ssl_moco_finetune project_name=receipt-text-detection wandb=True \
  exp_version=v10 trainer.max_epochs=10 trainer.num_sanity_val_steps=0 \
  +trainer.accelerator=gpu +trainer.devices=1 \
  +encoder_init_path=/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v10_ssl_moco/encoder_state_dict.pt \
  models.head.postprocess.box_thresh=0.25 \
  transforms.train_transform.transforms.0.max_size=1024 \
  transforms.train_transform.transforms.1.min_width=1024 \
  transforms.train_transform.transforms.1.min_height=1024 \
  transforms.val_transform.transforms.0.max_size=1024 \
  transforms.val_transform.transforms.1.min_width=1024 \
  transforms.val_transform.transforms.1.min_height=1024 \
  transforms.test_transform.transforms.0.max_size=1024 \
  transforms.test_transform.transforms.1.min_width=1024 \
  transforms.test_transform.transforms.1.min_height=1024 \
  transforms.predict_transform.transforms.0.max_size=1024 \
  transforms.predict_transform.transforms.1.min_width=1024 \
  transforms.predict_transform.transforms.1.min_height=1024
```

## Artifacts

- SSL manifest: `baseline_code/outputs/v10_ssl_moco/image_manifest.csv`
- SSL recipe/result: `baseline_code/outputs/v10_ssl_moco/pretrain_config.json`,
  `baseline_code/outputs/v10_ssl_moco/pretrain_result.json`
- SSL encoder: `baseline_code/outputs/v10_ssl_moco/encoder_state_dict.pt`
- Best detector: `baseline_code/outputs/v10_ssl_moco_finetune/checkpoints/epoch=8-step=1845.ckpt`
- Independent metrics: [metrics.json](metrics.json)
- Per-image values: [per_image_val.csv](per_image_val.csv)
- Paired bootstrap: [bootstrap_paired.csv](bootstrap_paired.csv)
- Prediction validation: [submission_validation.json](submission_validation.json)
- Offline CSV: [v10_ssl_moco_epoch8_20260714_181610.csv](/data/ephemeral/home/receipt-text-detection/submissions/v10_ssl_moco_epoch8_20260714_181610.csv)
- Prediction JSON: `baseline_code/outputs/v10_ssl_moco_epoch8_predict/submissions/20260714_181610.json`
- W&B SSL pretraining: [uc692vle](https://wandb.ai/pep-per/receipt-text-detection/runs/uc692vle)
- W&B fine-tuning: [jfq6ucaf](https://wandb.ai/pep-per/receipt-text-detection/runs/jfq6ucaf)
- W&B independent reload: [f3odsy45](https://wandb.ai/pep-per/receipt-text-detection/runs/f3odsy45)
