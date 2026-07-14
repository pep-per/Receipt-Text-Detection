# V4 Photometric Augmentation At Resolution 1024

## Status

Completed and rejected as the next clean single-model baseline. No Public submission was generated.

## Hypothesis

Mild camera degradation during training may improve detection of faint, blurred, and compressed
receipt text without sacrificing the strong precision of V2B.

## Control

- Control: V2B 1024 + Adam/StepLR
- Architecture: pretrained ResNet18 + UNet + DBHead
- Epochs/seed/batch: 10 / 42 / 16
- Primary post-processing: `box_thresh=0.25`
- Main change: one mild photometric transform on 50% of training samples
- Unchanged validation/test transforms

The `OneOf` policy used normalized selection weights:

| Transform | Conditional weight | Effective probability |
| --- | ---: | ---: |
| Brightness/contrast, limit 0.15 | 0.30 | 0.15 |
| Gamma, 85-115 | 0.20 | 0.10 |
| Gaussian blur, kernel 3-5 | 0.15 | 0.075 |
| Motion blur, kernel 3-5 | 0.10 | 0.05 |
| JPEG compression, quality 65-95 | 0.25 | 0.125 |

Synthetic shadow was excluded so the first policy would not combine uneven illumination with the
other camera-degradation hypothesis.

## Dual CLEval Control Audit

The Lightning wrapper now records two aggregation methods:

- Macro: compute CLEval H/P/R per image and average each metric; existing `val/hmean` behavior
- Global: accumulate character counts and granularity penalties over all images, then compute H/P/R

| Model | Macro H | Macro P | Macro R | Global H | Global P | Global R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 896 | 0.961507 | 0.963823 | 0.961134 | 0.958298 | 0.959732 | 0.956868 |
| V2B 1024 | **0.964760** | **0.969976** | **0.961422** | **0.962219** | **0.967510** | **0.956985** |

Both aggregation methods select V2B over V2. Therefore, the aggregation audit did not reverse the
1024 resolution choice.

W&B dual-evaluation runs:

- [V2 dual metric](https://wandb.ai/pep-per/receipt-text-detection/runs/ztsq1jj2)
- [V2B dual metric](https://wandb.ai/pep-per/receipt-text-detection/runs/hkunaokq)

## Epoch History

| Epoch | Macro H | Macro P | Macro R | Global H | Global P | Global R |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.888038 | 0.892470 | 0.893867 | 0.879500 | 0.885212 | 0.873861 |
| 1 | 0.943617 | 0.944494 | 0.945484 | 0.937611 | 0.939520 | 0.935710 |
| 2 | 0.941409 | 0.950951 | 0.936515 | 0.938316 | 0.949272 | 0.927610 |
| 3 | 0.950203 | 0.949816 | 0.953188 | 0.945494 | 0.944946 | 0.946042 |
| 4 | 0.955448 | 0.952659 | 0.960465 | 0.951030 | 0.947778 | 0.954305 |
| 5 | 0.953996 | 0.957299 | 0.953309 | 0.950412 | 0.954682 | 0.946180 |
| 6 | 0.955090 | 0.963406 | 0.949607 | 0.951028 | 0.960065 | 0.942159 |
| 7 | 0.958985 | 0.963670 | 0.956484 | 0.955000 | 0.958251 | 0.951771 |
| 8 | 0.962328 | 0.963772 | **0.963136** | **0.959786** | 0.959623 | **0.959950** |
| 9 | **0.962485** | **0.968428** | 0.958578 | 0.958608 | **0.963299** | 0.953961 |

Macro and global selected different best epochs. Epoch 9 was retained as the primary checkpoint to
preserve the competition baseline's existing macro model-selection rule.

## Independent Result

Independent reload of epoch 9 at `box_thresh=0.25`:

| Model | Macro H | Macro P | Macro R | Global H | Global P | Global R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V2B control | **0.964760** | **0.969976** | **0.961422** | **0.962219** | **0.967510** | **0.956985** |
| V4 epoch 9 | 0.962595 | 0.968465 | 0.958750 | 0.958719 | 0.963323 | 0.954158 |
| V4 - V2B | -0.002165 | -0.001511 | -0.002672 | -0.003500 | -0.004187 | -0.002827 |

Epoch 8 produced the highest Recall in this run, including macro Recall `+0.001714` over V2B, but
its macro H was still `-0.002432` below V2B because precision fell. The primary epoch-9 checkpoint
was lower than V2B on every macro and global metric.

## Decision

- Reject this combined photometric policy as the clean single-model baseline.
- Keep V2B 1024 + StepLR with the original augmentation as the current best clean model.
- Do not generate a Public submission from V4.
- Start the ResNet34 experiment from V2B, without this photometric policy.
- Preserve V4 epoch 8 as a possible recall-diverse ensemble diagnostic, not as a selected model.
- Do not immediately tune the five transforms separately against the same official val; revisit a
  narrower blur/compression ablation only if later K-fold error analysis supports it.

## Artifacts

- Best macro checkpoint: [epoch=9-step=2050.ckpt](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v4_photometric1024/checkpoints/epoch=9-step=2050.ckpt)
- Best global/recall checkpoint: [epoch=8-step=1845.ckpt](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v4_photometric1024/checkpoints/epoch=8-step=1845.ckpt)
- Training config: [config.yaml](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v4_photometric1024/.hydra/config.yaml)
- Augmentation preset: [db_photometric.yaml](/data/ephemeral/home/receipt-text-detection/baseline_code/configs/preset/datasets/db_photometric.yaml)
- W&B training: [uzpx3za9](https://wandb.ai/pep-per/receipt-text-detection/runs/uzpx3za9)
- W&B independent evaluation: [3viepvdq](https://wandb.ai/pep-per/receipt-text-detection/runs/3viepvdq)

Training plus final test took 2,118 seconds, about 35 minutes 18 seconds. Peak W&B allocated GPU
memory was about 22.0 GiB; batch 16 fit without gradient accumulation.
