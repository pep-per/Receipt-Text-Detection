# V5 ResNet34 Backbone At Resolution 1024

## Status

Completed. ResNet34 was not adopted as the default clean single model. V2B remains the default;
V5 is retained as a recall-diverse recalibration and ensemble candidate.

## Hypothesis

A pretrained ResNet34 backbone may improve receipt text features over ResNet18 while preserving the
same feature-channel interface to the existing UNet decoder.

## Control

- Control: V2B 1024 + Adam/StepLR
- Architecture control: pretrained ResNet18 + UNet + DBHead
- Epochs/seed/batch: 10 / 42 / 16
- Training augmentation: resize, padding, horizontal flip
- Primary post-processing: `box_thresh=0.25`, `thresh=0.30`
- Main change: backbone `resnet18 -> resnet34`
- Fresh training from ImageNet pretrained weights; no V2B checkpoint resume

ResNet18 and ResNet34 both expose selected feature channels `[64, 128, 256, 512]` at strides
`[4, 8, 16, 32]`, so the decoder and head configuration remained unchanged. The timm feature-only
backbones contain about 11.18M and 21.28M parameters respectively.

## Preflight

Batch 16 passed one 1024 training batch and validation batch without CUDA OOM. The preflight then
stopped in the existing test hook because Lightning's `fast_dev_run` supplied one test batch while
the hook expects predictions for all 404 filenames. This was not a model or memory failure and did
not apply to the full run.

The full run fit at batch 16 without gradient accumulation, but W&B recorded peak allocated GPU
memory of 24,777,850,880 bytes, about 23.08 GiB or 96.15% of the RTX 3090. Reproduction should use
the GPU without another memory-consuming process.

## Epoch History

| Epoch | Macro H | Macro P | Macro R | Global H | Global P | Global R |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.917382 | 0.920023 | 0.919950 | 0.911014 | 0.913355 | 0.908685 |
| 1 | 0.883881 | 0.930131 | 0.877426 | 0.893979 | 0.923340 | 0.866428 |
| 2 | 0.950274 | 0.951931 | 0.951682 | 0.946481 | 0.946476 | 0.946486 |
| 3 | 0.922530 | 0.954232 | 0.912267 | 0.923553 | 0.950127 | 0.898424 |
| 4 | 0.950737 | 0.949114 | 0.956096 | 0.947604 | 0.943985 | 0.951252 |
| 5 | 0.950789 | **0.975093** | 0.932259 | 0.945937 | **0.975081** | 0.918485 |
| 6 | 0.961598 | 0.961255 | 0.963667 | 0.957508 | 0.955486 | 0.959540 |
| 7 | **0.964608** | 0.964326 | **0.966828** | **0.962177** | 0.960277 | **0.964084** |
| 8 | 0.961759 | 0.964130 | 0.961656 | 0.959890 | 0.960745 | 0.959037 |
| 9 | 0.961742 | 0.959544 | 0.966420 | 0.959358 | 0.955259 | 0.963493 |

Early training was unstable, including large recall drops at epochs 1 and 3. The run stabilized
after epoch 6, and both macro and global H selected epoch 7.

## Independent Result

Independent reload of epoch 7 at the unchanged primary post-processing settings:

| Model | Macro H | Macro P | Macro R | Global H | Global P | Global R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V2B ResNet18 | **0.964760** | **0.969976** | 0.961422 | **0.962219** | **0.967510** | 0.956985 |
| V5 ResNet34 | 0.964617 | 0.964341 | **0.966831** | 0.962185 | 0.960289 | **0.964088** |
| V5 - V2B | -0.000143 | -0.005635 | +0.005409 | -0.000034 | -0.007221 | +0.007103 |

The independent result reproduced the training-time checkpoint within about `1e-5`. H-Mean is
effectively tied, but V5 moves along a clear precision-recall tradeoff: it detects more character
area while adding false-positive or over-extended detections. The additional backbone capacity did
not produce a clean H-Mean improvement at the fixed threshold.

## Cost Comparison

| Model | Backbone params | Checkpoint size | Training runtime | Peak allocated GPU memory |
| --- | ---: | ---: | ---: | ---: |
| V2B ResNet18 | 11.18M | 142 MB | 2,107 s | about 22.5 GB |
| V5 ResNet34 | 21.28M | 258 MB | 2,457 s | 23.08 GiB, 96.15% |

The V5 runtime includes the final test pass and was about 40 minutes 57 seconds. Its H tie does not
justify replacing the lighter and more precise V2B as the default single model.

## Decision

- Keep V2B as the current best clean single model.
- Do not generate a V5 Public submission at `box_thresh=0.25`.
- Preserve V5 epoch 7 as a recall-diverse candidate for local post-processing recalibration and a
  later probability-map ensemble experiment.
- Do not move to ResNet50 now; ResNet34 already reaches the 24 GB memory limit and did not improve
  fixed-threshold H-Mean.
- In V6, evaluate V5 epoch 7 once at `box_thresh=0.30` with `thresh=0.30`. Test `box_thresh=0.35`
  only if 0.30 improves both V5 macro and global H. Do not use Public feedback for this choice.

This V6 branch was already planned as local post-processing recalibration. Narrowing its first step
to V5 is justified by V5's measured high-recall/low-precision profile, rather than by Public score.

## Artifacts

- Best checkpoint: [epoch=7-step=1640.ckpt](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v5_resnet34_1024/checkpoints/epoch=7-step=1640.ckpt)
- Training config: [config.yaml](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v5_resnet34_1024/.hydra/config.yaml)
- Evaluation config: [config.yaml](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v5_resnet34_1024_epoch7_eval/.hydra/config.yaml)
- W&B training: [scibn6c0](https://wandb.ai/pep-per/receipt-text-detection/runs/scibn6c0)
- W&B independent evaluation: [vjg0xs74](https://wandb.ai/pep-per/receipt-text-detection/runs/vjg0xs74)
