# V3 Cosine LR At Resolution 1024

## Status

Completed and rejected by local CLEval.

## Purpose

Replace the effectively constant StepLR used by V2B with a cosine learning-rate schedule and test
whether smaller late-epoch updates improve official-val CLEval.

## Control

- Control: V2B 1024, independently evaluated H/P/R
  `0.964760 / 0.969976 / 0.961422`
- Resolution: 1024
- Architecture: pretrained ResNet18 + UNet + DBHead
- Optimizer: Adam, initial LR `0.001`, weight decay `0.0001`
- Epochs/seed/batch: 10 / 42 / 16
- Primary post-processing: `box_thresh=0.25`
- Augmentation and official train/val split: unchanged

## Main Change

```yaml
scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: 10
  eta_min: 1.0e-6
```

Adam, initial LR, epoch count, architecture, input resolution, augmentation, and post-processing
remain fixed. This is a fresh training run from the same ImageNet-pretrained initialization path,
not a resume from the V2B checkpoint.

## Adoption Rule

- Compare the best checkpoint at the same `box_thresh=0.25` using official-val CLEval.
- Adopt cosine only if H-Mean improves without a harmful precision/recall imbalance.
- If cosine is worse or effectively tied, retain the V2B StepLR checkpoint as the best clean model.
- Public leaderboard is not used to select the scheduler.

## Result

Best checkpoint: epoch 8 at `box_thresh=0.25`.

| Model | H-Mean | Precision | Recall |
| --- | ---: | ---: | ---: |
| V2B StepLR control | **0.964760** | **0.969976** | **0.961422** |
| V3 cosine | 0.959166 | 0.960237 | 0.960184 |

V3 minus V2B: H `-0.005595`, precision `-0.009739`, recall `-0.001237`.

## Epoch History

| Epoch | H-Mean | Precision | Recall |
| ---: | ---: | ---: | ---: |
| 0 | 0.885184 | 0.919067 | 0.869104 |
| 1 | 0.938891 | 0.939868 | 0.940857 |
| 2 | 0.941873 | 0.946740 | 0.941084 |
| 3 | 0.951137 | 0.951505 | 0.953692 |
| 4 | 0.957398 | 0.960077 | 0.957028 |
| 5 | 0.955813 | 0.954184 | 0.959374 |
| 6 | 0.956478 | 0.960001 | 0.955430 |
| 7 | 0.952541 | 0.956833 | 0.952587 |
| 8 | **0.959149** | **0.960220** | **0.960166** |
| 9 | 0.958410 | 0.958996 | 0.960361 |

The scheduler was active: the logged LR decreased from `0.001` to about `0.0000254` by epoch 9.
Cosine converged faster in some early epochs, but its best late-epoch H-Mean remained below the
StepLR control. The W&B runtime including the final test pass was 2,120 seconds, about 35 minutes
20 seconds.

## Artifacts

- Best checkpoint: [epoch=8-step=1845.ckpt](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v3_cosine1024/checkpoints/epoch=8-step=1845.ckpt)
- Config: [config.yaml](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v3_cosine1024/.hydra/config.yaml)
- W&B training run: [nt8ea7bv](https://wandb.ai/pep-per/receipt-text-detection/runs/nt8ea7bv)
- W&B independent evaluation: [xzm6buuf](https://wandb.ai/pep-per/receipt-text-detection/runs/xzm6buuf)

## Decision

- Reject this cosine candidate and do not generate a Public submission from it.
- Keep V2B 1024 + StepLR as the current best clean single model.
- Do not immediately sweep `T_max`, epoch count, and LR against the same official val.
- Start Experiment 5 from V2B and change only the mild photometric augmentation policy.
