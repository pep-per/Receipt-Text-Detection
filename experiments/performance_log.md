# Performance Log

이 파일은 실험별 local validation, leaderboard 결과, 관련 artifact, 다음 실험에서 고려할 전략 변화를 누적 기록한다. 기존 전략 문서는 그대로 두고, 실험 결과에 따른 판단은 이 파일에 추가한다.

## Summary Table

| Experiment | Model | Local H | Local P | Local R | Public H | Public P | Public R | Final H | Final P | Final R | Public-Local H | Final-Local H | Created At | Files | Strategy Note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| [V0 Baseline](20260709-v0-baseline/README.md) | v0 | 0.8913 | 0.9633 | 0.8369 | 0.8818 | 0.9651 | 0.8194 | 0.8898 | 0.9675 | 0.8324 | -0.0095 | -0.0015 | 2026.07.09 23:21 | [CSV](/data/ephemeral/home/receipt-text-detection/submissions/v0_baseline_epoch8_20260709_012836.csv), [JSON](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v0_baseline_epoch8/submissions/20260709_012836.json), [evidence](/data/ephemeral/home/receipt-text-detection/docs/leaderboard/20260714/README.md) | Final was close to local; baseline weakness was recall. |
| [V1 Threshold Sweep](20260709-v1-threshold-sweep/README.md) | v1 | 0.9248 | 0.9499 | 0.9057 | 0.9185 | 0.9511 | 0.8932 | 0.9221 | 0.9554 | 0.8978 | -0.0063 | -0.0027 | 2026.07.09 23:54 | [CSV](/data/ephemeral/home/receipt-text-detection/submissions/v1_box025_epoch8_20260709_235118.csv), [JSON](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v1_box025_epoch8/submissions/20260709_235118.json), [evidence](/data/ephemeral/home/receipt-text-detection/docs/leaderboard/20260714/README.md) | Final H +0.0323 over V0 confirms that local threshold selection transferred. |
| [V2 Resolution 896](20260712-v2-resolution-896/README.md) | v2 | 0.9615 | 0.9638 | 0.9611 | 0.9603 | 0.9667 | 0.9556 | 0.9637 | 0.9682 | 0.9606 | -0.0012 | +0.0022 | 2026.07.13 05:31 | [CSV](/data/ephemeral/home/receipt-text-detection/submissions/v2_resolution896_epoch8_20260713_002412.csv), [JSON](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v2_resolution896_epoch8/submissions/20260713_002412.json), [evidence](/data/ephemeral/home/receipt-text-detection/docs/leaderboard/20260714/README.md) | Final H +0.0416 over V1; resolution 896 produced the largest hidden-set gain. |
| [V2B Resolution 1024](20260713-v2b-resolution-1024/README.md) | v2b | 0.9648 | 0.9700 | 0.9614 | 0.9621 | 0.9754 | 0.9520 | **0.9647** | **0.9739** | **0.9580** | -0.0027 | -0.0001 | 2026.07.14 01:17 | [CSV](/data/ephemeral/home/receipt-text-detection/submissions/v2b_resolution1024_epoch8_20260714_001730.csv), [JSON](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v2b_resolution1024_epoch8/submissions/20260714_001730.json), [evidence](/data/ephemeral/home/receipt-text-detection/docs/leaderboard/20260714/README.md) | Final rank 1. Final H +0.0010 over V2, mainly from precision; local direction transferred with a smaller gain. |
| [V3 Cosine 1024](20260713-v3-cosine-1024/README.md) | v3 | 0.9592 | 0.9602 | 0.9602 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 2026.07.14 local | [ckpt](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v3_cosine1024/checkpoints/epoch=8-step=1845.ckpt), [config](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v3_cosine1024/.hydra/config.yaml) | Rejected locally; no leaderboard score. |
| [V4 Photometric 1024](20260714-v4-photometric-1024/README.md) | v4 | 0.9626 | 0.9685 | 0.9587 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 2026.07.14 local | [ckpt](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v4_photometric1024/checkpoints/epoch=9-step=2050.ckpt), [W&B](https://wandb.ai/pep-per/receipt-text-detection/runs/uzpx3za9) | Rejected locally; no leaderboard score. |
| [V5 ResNet34 1024](20260714-v5-resnet34-1024/README.md) | v5 | 0.9646 | 0.9643 | 0.9668 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 2026.07.14 local | [ckpt](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v5_resnet34_1024/checkpoints/epoch=7-step=1640.ckpt), [W&B](https://wandb.ai/pep-per/receipt-text-detection/runs/scibn6c0) | H tied locally; retained only as a recall-diverse ensemble candidate. |
| [V6 V5 Post-processing](20260714-v6-v5-postprocess/README.md) | v6 | 0.9606 | 0.9663 | 0.9572 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 2026.07.14 local | [metric](20260714-v6-v5-postprocess/metrics.json), [W&B](https://wandb.ai/pep-per/receipt-text-detection/runs/8cy2bpn0) | Rejected locally; no leaderboard score. |
| [V8 Scale TTA](20260714-v8-scale-tta/README.md) | v8 | **0.9668** | **0.9725** | **0.9628** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 2026.07.14 offline | [metric](20260714-v8-scale-tta/metrics.json), [CSV](/data/ephemeral/home/receipt-text-detection/submissions/v8_scale_tta_1024_1152_20260714_154949.csv), [W&B](https://wandb.ai/pep-per/receipt-text-detection/runs/7eb4lky8) | Adopted local TTA candidate; macro/global H bootstrap CIs were positive. Competition closed, so no leaderboard score. |
| [V9 Model Ensemble](20260714-v9-model-ensemble/README.md) | v9 | **0.9673** | 0.9691 | **0.9672** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 2026.07.14 offline | [metric](20260714-v9-model-ensemble/metrics_w050.json), [CSV](/data/ephemeral/home/receipt-text-detection/submissions/v9_v2b_v5_equal_20260714_170846.csv), [W&B](https://wandb.ai/pep-per/receipt-text-detection/runs/85wojhfj) | Adopted model-ensemble candidate; recall and both H aggregations improved. Competition closed, so no leaderboard score. |
| [V10 Domain SSL](20260714-v10-domain-ssl/README.md) | v10 | 0.9649 | 0.9648 | 0.9667 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 2026.07.14 offline | [metric](20260714-v10-domain-ssl/metrics.json), [CSV](/data/ephemeral/home/receipt-text-detection/submissions/v10_ssl_moco_epoch8_20260714_181610.csv), [W&B](https://wandb.ai/pep-per/receipt-text-detection/runs/jfq6ucaf) | Rejected as statistical tie: macro/global H +0.0001/+0.0004, but bootstrap CIs crossed zero. Recall rose while precision fell. |

`N/A`는 해당 실험을 leaderboard에 제출하지 않아 점수가 존재하지 않는다는 뜻이다. Final
score 원시 값과 증빙은 [2026-07-14 Final Leaderboard Evidence](/data/ephemeral/home/receipt-text-detection/docs/leaderboard/20260714/README.md)에 보존한다.

분석 단계인 D0는 새 model 또는 submission이 없어 점수 표의 version row로 추가하지 않았다.
상세 결과와 artifact는 [D0 Train/Val/Test Data Audit](20260714-d0-data-audit/README.md)에 있다.

## V0 Baseline

Leaderboard values extracted from the screenshot:

- Model name: `v0`
- H-Mean: 0.8818
- Precision: 0.9651
- Recall: 0.8194
- Created at: 2026.07.09 23:21
- Phase: Complete
- Final H/P/R: `0.8898 / 0.9675 / 0.8324`

Local selected-checkpoint validation:

- H-Mean: 0.8913
- Precision: 0.9633
- Recall: 0.8369

Delta:

- Public H-Mean - Local H-Mean: -0.0095
- Public Precision - Local Precision: +0.0018
- Public Recall - Local Recall: -0.0175

## Strategy Implications For Next Experiment

Do not rewrite the main strategy yet. For the next experiment, test a narrow recall-oriented change against V0:

- Lower `box_thresh` slightly or search `box_thresh` around the baseline value `0.4`.
- Keep `max_candidates` away from the 500 cap; V0 max was only 174, so there is room.
- Watch precision carefully because V0's public precision is already strong.
- Record all runs in this file and in `submissions/submission_log.md`.
- Enable wandb from V1 onward as planned.

## V1 Threshold Sweep

Local official-validation sweep using the V0 checkpoint:

| box_thresh | H-Mean | Precision | Recall | Note |
| ---: | ---: | ---: | ---: | --- |
| 0.35 | 0.9153 | 0.9571 | 0.8823 | Large recall gain from V0 |
| 0.30 | 0.9230 | 0.9528 | 0.8997 | Strong H-Mean gain, still precise |
| 0.25 | 0.9248 | 0.9499 | 0.9057 | Selected for V1 submission |
| 0.20 | 0.9249 | 0.9482 | 0.9075 | Local H only +0.0001 vs 0.25, lower precision |

Strategy adjustment for the next experiment, without editing the original strategy:

- Use `box_thresh=0.25` as the current postprocess default. V1 public H improved from 0.8818 to 0.9185 over V0.
- Keep `box_thresh=0.30` as a safer fallback only if later experiments add false positives.
- Avoid lowering below `0.20` before visual error analysis; the local gain has flattened while precision continues to fall.
- Because the public/private test split is random 50:50 and hidden, do not keep tuning thresholds against public submissions alone.
- Next improvement should come from training/data/augmentation or error analysis, not another threshold-only public leaderboard probe.

## Interpretation Limit After V1

V1 proves that `box_thresh` was a high-impact variable for the fixed V0 checkpoint. It does **not**
prove that post-processing is generally more important than model training, because no alternative
training recipe has been tested yet.

Controlled evidence available so far:

- V0 and V1 used exactly the same `epoch=8-step=1845.ckpt` checkpoint.
- Changing only `box_thresh` from `0.40` to `0.25` raised local H-Mean by `+0.0335` and public
  H-Mean by `+0.0367`.
- This isolates a large post-processing effect for this checkpoint.
- There is no trained-model A/B result yet, so the relative effect size of input resolution,
  augmentation, backbone, optimizer, or longer training is unknown.

`box_thresh` is the minimum mean text-probability score required to keep a contour proposal as a
detected text region. Lowering it accepts weaker proposals, which usually raises recall and lowers
precision. It is different from `thresh`, which binarizes the pixel-level probability map before
contours are extracted.

Threshold evaluations performed so far:

- V0 used the baseline `box_thresh=0.40` for validation and submission.
- V1 reused the V0 checkpoint and evaluated `0.35`, `0.30`, `0.25`, and `0.20` on the official
  validation set. These were four inference/evaluation runs, not four training runs.
- Only the locally selected `0.25` result was generated for the V1 test submission.

## Experiment 3 Plan Revision

Detailed plan: [V2 Resolution 896 Plan](20260712-v2-resolution-896/README.md)

Previous tentative plan:

- Move broadly into training/data augmentation or error analysis while applying
  `box_thresh=0.25`.

Revised controlled plan:

- Test one training-side variable first: increase train/validation/test resize from `640` to `896`.
- Keep the architecture, official train/validation split, optimizer, epoch budget, seed, and other
  post-processing settings unchanged.
- Use `box_thresh=0.25` as the primary comparison so V1 is the direct control.
- Check `box_thresh=0.30` locally only if the new checkpoint shows a clear precision collapse or
  changed score calibration. Do not select it from Public leaderboard results.
- Postpone a broad augmentation bundle until the resolution effect is measured, because changing
  resolution and augmentation together would make the source of any gain or regression unclear.

Why this plan changed:

- V1 already recovered much of the V0 recall loss without retraining, so another threshold-only
  experiment has low expected value. Local H-Mean flattened between `0.25` and `0.20` while
  precision continued to fall.
- At the current 640 resize, the official validation annotations have a median text-region short
  side of about `10.7 px`; `22.4%` of regions are below `8 px` and `65.3%` are below `12 px`.
  Increasing resolution may recover small or faint text that threshold lowering cannot create in
  the probability map.
- Resolution is still an unverified hypothesis. Experiment 3 is designed to measure it, not to
  assume it is superior to augmentation or other training changes.

Adoption criteria:

- Primary: official-validation H-Mean exceeds the V1 control `0.9248` at the same
  `box_thresh=0.25`.
- Record precision and recall separately; reject a nominal H-Mean gain that depends on an unstable
  precision collapse or invalid/excessive predictions.
- Record GPU memory, training time, inference time, and prediction-count statistics.
- Use the Public leaderboard only as a confirmation for the single locally selected candidate.

## Clean-data Roadmap Before Pseudo-labeling

Detailed roadmap and glossary:
[Clean-data Experiment Roadmap](clean_data_experiment_roadmap.md)

Pseudo-labeling is postponed until the strongest reasonable teacher has been selected using only
official human-labeled data. The provisional order after Experiment 3 is:

1. Experiment 4: make the learning-rate schedule effective with cosine decay.
2. Experiment 5: add mild receipt-camera photometric augmentation.
3. Experiment 6: compare the `resnet34` backbone with the current `resnet18`.
4. Experiment 7: recalibrate post-processing on local CLEval for the selected checkpoint.
5. Experiment 8: confirm the clean configuration with K-fold and build a clean ensemble/final
   train+val teacher candidate.
6. Experiment 9 or later: generate, filter, and ablate pseudo labels.

The next experiment always starts from the best validated clean candidate, not automatically from
the most recently run model. Each stage changes one main variable and may be rejected. Public
leaderboard results are confirmation signals, not model-selection labels.

## V2 Resolution 896 Result

Best epoch 8 independent official-validation result at `box_thresh=0.25`:

- H-Mean: `0.9615`
- Precision: `0.9638`
- Recall: `0.9611`
- Delta versus V1 local: H `+0.0367`, precision `+0.0139`, recall `+0.0555`

Decision for the next experiment, without editing the original strategy:

- Keep 896 as the current best clean single-model setting.
- The gain validates the small-text resolution hypothesis for this split and model, but does not
  yet establish Private performance.
- An optional 1024 controlled resolution branch is now justified because 896 improved clearly and
  used about 17.4 GB with batch 16. Select it only by local CLEval and resource cost.
- Otherwise start Experiment 4 cosine scheduling from the 896 configuration.
- Do not tune `box_thresh` from the pending V2 Public result; both local precision and recall are
  already balanced near 0.96.

## V2 Public Leaderboard Result

Leaderboard values extracted from the screenshot:

- Model name: `v2`
- Created at: 2026.07.13 05:31
- Phase: Complete
- Public H-Mean: `0.9603`
- Public Precision: `0.9667`
- Public Recall: `0.9556`
- Final H/P/R: `0.9637 / 0.9682 / 0.9606`

Compared with local official validation:

- Public H-Mean - Local H-Mean: `-0.0012`
- Public Precision - Local Precision: `+0.0029`
- Public Recall - Local Recall: `-0.0055`

Compared with V1 Public:

- H-Mean: `+0.0418`
- Precision: `+0.0156`
- Recall: `+0.0624`

The actual Public H-Mean was `+0.0067` above the rough pre-submission center estimate of `0.9536`
and above the estimated `0.9520-0.9552` range. V2 was selected from local evidence before this
Public result and only one 896 candidate was submitted, so there is currently no direct evidence of
Public leaderboard overfitting. Private sampling variance and official-val tuning risk remain; do
not use this result to retune threshold or add test-specific rules.

## V2B Resolution 1024 Result

The 1024 branch changed only train/validation/test/predict resolution from 896 to 1024. Batch 16,
architecture, optimizer, scheduler, epoch budget, seed, augmentation, and `box_thresh=0.25` were
kept fixed.

Independent best-checkpoint official-validation result:

- H-Mean: `0.964760`
- Precision: `0.969976`
- Recall: `0.961422`
- Delta versus V2 local: H `+0.003253`, precision `+0.006153`, recall `+0.000288`

Decision and next-strategy implication:

- Adopt 1024 as the current best clean single-model resolution.
- The gain is mainly a precision improvement while recall is maintained, rather than a more
  aggressive detection trade-off.
- Peak memory increased from about 17.4 GB to 22.5 GB, but batch 16 still fit.
- A submission candidate was generated only after the local decision. Do not select 896 versus
  1024 from its future Public result.
- Experiment 4 therefore used 1024 as its fixed resolution.

### V2B Public Leaderboard Result

Leaderboard values extracted from the screenshot:

- Model name: `v2b`
- Created at: `2026.07.14 01:17`
- Phase: Complete
- Public H-Mean: `0.9621`
- Public Precision: `0.9754`
- Public Recall: `0.9520`
- Final H/P/R: `0.9647 / 0.9739 / 0.9580`

Compared with V2B local official validation:

- Public H-Mean - Local H-Mean: `-0.002660`
- Public Precision - Local Precision: `+0.005424`
- Public Recall - Local Recall: `-0.009422`

Compared with V2 Public:

- H-Mean: `+0.0018`
- Precision: `+0.0087`
- Recall: `-0.0036`

The 1024 change improved H-Mean on both local validation and Public, so the direction transferred.
The Public improvement was smaller than the local improvement (`+0.0018` versus `+0.003253`) and
was precision-driven. The most likely explanations are that higher resolution produces cleaner
text boundaries and fewer false-positive character areas, while the 404-image official validation
and roughly half-sized hidden Public subset contain different proportions of faint, blurred, small,
or distorted text. These are hypotheses because Public per-image labels and errors are unavailable.

The local implementation computes per-image H-Mean, precision, and recall and then averages each
metric separately. On the leaderboard, the harmonic mean of the displayed precision and recall is
`0.9636`, not the displayed H-Mean `0.9621`. This is consistent with separately aggregated metrics
or another server-side weighting rule; four-decimal rounding alone is too small to explain the
gap. The screenshot alone cannot confirm the exact server implementation. The Public
precision/recall gap is therefore a diagnostic signal, not a formula from which the exact
leaderboard H-Mean can be reconstructed.

Strategy implication:

- Retain V2B 1024 + StepLR as the best clean single model.
- Keep Experiment 5 photometric augmentation next, with special attention to recall on faint and
  blurred text while preserving the already high precision.
- Do not lower `box_thresh` in response to this Public recall alone. Recalibrate post-processing
  later on local CLEval after selecting the final clean checkpoint.
- Keep K-fold confirmation before pseudo-labeling because repeated use of the same official val,
  rather than current Public evidence, is now the more important model-selection risk.

## V3 Cosine 1024 Result

V3 changed only StepLR to `CosineAnnealingLR(T_max=10, eta_min=1e-6)`. It was a fresh training run
from the same ImageNet-pretrained initialization path, not a resume from V2B.

Independent best-checkpoint official-validation result:

- H-Mean: `0.959166`
- Precision: `0.960237`
- Recall: `0.960184`
- Delta versus V2B: H `-0.005595`, precision `-0.009739`, recall `-0.001237`

Decision and next-strategy implication:

- Reject this cosine candidate and retain V2B 1024 + StepLR as the best clean model.
- Do not submit V3 to Public; local evidence already rejects it.
- Do not immediately tune `T_max`, longer epochs, and lower initial LR as a bundle. That would turn
  one failed scheduler test into repeated official-val fitting.
- Experiment 5 should start from the V2B training recipe and change only a mild photometric
  augmentation policy.

## Dual CLEval Aggregation Audit

The provided Lightning wrapper computes CLEval per image, resets the metric, and averages image
H/P/R independently. Upstream CLEval instead supports accumulating raw character counts and
granularity penalties before one global H/P/R calculation. Both are now logged:

- Existing macro keys: `val/hmean`, `val/precision`, `val/recall`
- Added global keys: `val/global_hmean`, `val/global_precision`, `val/global_recall`

| Model | Macro H/P/R | Global H/P/R |
| --- | --- | --- |
| V2 896 | `0.961507 / 0.963823 / 0.961134` | `0.958298 / 0.959732 / 0.956868` |
| V2B 1024 | `0.964760 / 0.969976 / 0.961422` | `0.962219 / 0.967510 / 0.956985` |

V2B improves H under macro by `+0.003253` and under global by `+0.003921`. The 1024 selection is
therefore robust to this aggregation choice. Existing `val/hmean` remains the checkpoint monitor
because it preserves comparison with the provided baseline and its non-harmonic aggregate pattern
is consistent with the displayed leaderboard metrics. Global H is retained as a required secondary
diagnostic because the exact competition server implementation is unavailable.

## V4 Photometric 1024 Result

Detailed record: [V4 Photometric Augmentation](20260714-v4-photometric-1024/README.md)

Independent epoch-9 result at `box_thresh=0.25`:

- Macro H/P/R: `0.962595 / 0.968465 / 0.958750`
- Global H/P/R: `0.958719 / 0.963323 / 0.954158`
- Macro delta versus V2B: `-0.002165 / -0.001511 / -0.002672`
- Global delta versus V2B: `-0.003500 / -0.004187 / -0.002827`

Epoch 8 raised macro Recall to `0.963136`, `+0.001714` over V2B, but its macro H remained
`-0.002432` lower because precision fell. Reject the combined camera-degradation policy as a single
model. Do not submit it to Public and do not immediately fit individual augmentation probabilities
to the repeatedly used official val. Experiment 6 starts from V2B and changes only ResNet18 to
ResNet34.

## V5 ResNet34 1024 Result

Detailed record: [V5 ResNet34 Backbone](20260714-v5-resnet34-1024/README.md)

Independent epoch-7 result at `box_thresh=0.25`:

- Macro H/P/R: `0.964617 / 0.964341 / 0.966831`
- Global H/P/R: `0.962185 / 0.960289 / 0.964088`
- Macro delta versus V2B: `-0.000143 / -0.005635 / +0.005409`
- Global delta versus V2B: `-0.000034 / -0.007221 / +0.007103`
- Runtime/peak allocated memory: 2,457 seconds / 23.08 GiB, 96.15% of RTX 3090

V5 is effectively tied with V2B in H but exchanges precision for recall and nearly doubles the
backbone parameter count. Keep V2B as the default clean single model and do not make a Public
submission from uncalibrated V5. Preserve V5 as a recall-diverse candidate. In the already planned
local post-processing stage, test V5 once at `box_thresh=0.30`; expand to 0.35 only if both macro and
global H improve. This is a local, pre-submission decision and must not be selected from Public.

## D0 Train/Val/Test Data Audit Result

Detailed record: [D0 Train/Val/Test Data Audit](20260714-d0-data-audit/README.md)

- 7,772 official and auxiliary images were analyzed with fixed-scale image-quality proxies.
- Official train/val/test medians and ECDFs are closely aligned. The largest train-test KS statistic
  was only `0.0655`, so no photometric or geometric domain-gap augmentation is justified.
- The strongest V2B failure signal was the ratio of text regions below 12 px at 1024 scale:
  Spearman `rho=-0.1945` with H and `-0.2061` with recall.
- The highest small-text-ratio quartile had mean V2B H/P/R
  `0.9427 / 0.9522 / 0.9363`, versus H `0.9661-0.9759` in the other quartiles.
- V2B/V5 prediction Jaccard averaged `0.8733` on val and `0.8732` on test. V5 won 166 val images,
  V2B won 220, and an oracle per-image choice was `+0.00637` macro H above V2B.
- SROIE, WildReceipt, and CORD-v2 have sharply different brightness, resolution, and edge/blur
  distributions, so later SSL and pseudo training must retain source identity and balance sources.

Strategy implication after D0, without using a Public submission:

- Run V6 post-processing as planned.
- Mark V7 `skipped: no D0-supported train/test augmentation gap`.
- Keep V8 scale TTA, but screen `1024+1152` before a lower-scale fallback because the measured
  failure mode is small text.
- Keep V9 V2B+V5 probability-map ensemble as a local candidate; disagreement is evidence to test
  fusion, not evidence that fusion already works.
- Use source-balanced image-only sampling and conservative text-preserving transforms in V10 SSL.

## V6 V5 Post-processing Recalibration Result

Detailed record: [V6 V5 Post-processing](20260714-v6-v5-postprocess/README.md)

V6 reused V5 ResNet34 epoch 7 and changed only `box_thresh` from `0.25` to `0.30` at 1024
resolution and pixel `thresh=0.30`.

- Macro H/P/R: `0.960568 / 0.966257 / 0.957238`
- Global H/P/R: `0.958472 / 0.963035 / 0.953953`
- Macro delta versus V5 box 0.25: `-0.004049 / +0.001916 / -0.009593`
- Global delta versus V5 box 0.25: `-0.003713 / +0.002746 / -0.010135`

The precision gain was too small to offset the recall loss, and V6 was below V2B on all three
macro metrics. Reject V5 box 0.30. The predeclared gate required 0.30 to improve both macro and
global H before testing 0.35, so 0.35 was not run. V2B remains the clean single control while V5
box 0.25 remains only as the recall-diverse V9 ensemble member. V7 is already skipped by D0, so the
next run is V8 `1024+1152` scale probability-map TTA.

## V8 Scale TTA Result

Detailed record: [V8 Scale TTA](20260714-v8-scale-tta/README.md)

V8 reused the V2B epoch-8 checkpoint. It aligned the valid, unpadded regions of 1024 and 1152
probability maps in the 1024 coordinate system, averaged them equally, and ran DB post-processing
once at the unchanged `thresh=0.30`, `box_thresh=0.25`.

- Macro H/P/R: `0.966840 / 0.972512 / 0.962827`
- Global H/P/R: `0.965130 / 0.971017 / 0.959314`
- Macro delta versus 1024 control: `+0.002055 / +0.002533 / +0.001364`
- Global delta versus 1024 control: `+0.002881 / +0.003501 / +0.002274`
- Paired bootstrap macro H 95% CI and P(delta > 0):
  `[+0.000548, +0.003631]`, `0.9967`
- Paired bootstrap global H 95% CI and P(delta > 0):
  `[+0.001036, +0.004851]`, `0.9998`

Both H aggregation gates passed. The gain was larger on the control-H bottom quartile (`+0.00789`),
short-text-side bottom quartile (`+0.00778`), and high-small-text-ratio quartile (`+0.00527`), which
supports the D0 hypothesis. Adopt `1024+1152` as a TTA candidate and skip the lower-scale fallback.
The post-competition test JSON/CSV passed all format checks, but it is not hidden-score evidence.
Next run V9 with V2B/V5 model map fusion alone so scale TTA and architecture diversity remain
separate effects.

## V9 Existing-model Probability-map Ensemble Result

Detailed record: [V9 Model Ensemble](20260714-v9-model-ensemble/README.md)

V9 averaged the 1024 probability maps from V2B ResNet18 epoch 8 and V5 ResNet34 epoch 7 with fixed
equal weights, then applied V2B's unchanged DB post-processing once.

- Macro H/P/R: `0.967266 / 0.969083 / 0.967177`
- Global H/P/R: `0.965090 / 0.965880 / 0.964301`
- Macro delta versus V2B: `+0.002481 / -0.000896 / +0.005714`
- Global delta versus V2B: `+0.002841 / -0.001636 / +0.007262`
- Macro/global H paired bootstrap 95% CI:
  `[+0.000890, +0.004150] / [+0.000886, +0.004929]`
- Macro/global H P(delta > 0): `0.9998 / 0.9990`

The equal ensemble passed the gate, so the pre-registered V2B-heavy fallback was not run. V5-won
images improved by `+0.01098` H while V2B-won images lost `-0.00375`, confirming useful but
sample-dependent diversity. V9 has slightly higher macro H than V8 while V8 has slightly higher
global H, so retain both rather than declaring a universal winner. The validated test JSON/CSV is
an offline artifact, not hidden-score evidence. Combine V8 and V9 only in the later V13 final
fusion stage.

## V10 Domain Self-supervised Pilot Result

Detailed record: [V10 Domain SSL](20260714-v10-domain-ssl/README.md)

V10 used `lightly==1.5.25` MoCo v2 to pretrain the timm ResNet18 encoder on 7,772 receipt images
without opening annotation files. The final SSL encoder then replaced only V2B's ImageNet
initialization; supervised data, DBNet recipe, seed, epochs and post-processing stayed fixed.

- Independent macro H/P/R: `0.964897 / 0.964765 / 0.966682`
- Independent global H/P/R: `0.962613 / 0.961242 / 0.963987`
- Macro delta versus V2B: `+0.000112 / -0.005214 / +0.005219`
- Global delta versus V2B: `+0.000363 / -0.006274 / +0.006948`
- Macro/global H P(delta > 0): `0.5552 / 0.6396`
- Macro/global H 95% CI: `[-0.001548, +0.001852] / [-0.001709, +0.002546]`

Recall improved consistently but precision fell by a similar amount, leaving H statistically tied.
There was also a one-epoch recall collapse at epoch 3 before recovery. Reject V10 under the
pre-registered tie rule and use the V2B ImageNet initialization in V11. MoCo was technically valid,
so do not open V10-ALT merely to search SSL algorithms on the same val. The validated test CSV is
a recall-diverse reproducibility artifact, not a selected model or hidden-score claim. Next run
V11A group-aware K-fold manifest generation.

## Final Leaderboard Result And Local Calibration

Detailed evidence:
[2026-07-14 Final Leaderboard](/data/ephemeral/home/receipt-text-detection/docs/leaderboard/20260714/README.md)

| Model | Local H | Public H | Final H | Final - Local | Final rank within submissions |
| --- | ---: | ---: | ---: | ---: | ---: |
| V0 | 0.8913 | 0.8818 | 0.8898 | -0.0015 | 4 |
| V1 | 0.9248 | 0.9185 | 0.9221 | -0.0027 | 3 |
| V2 | 0.9615 | 0.9603 | 0.9637 | +0.0022 | 2 |
| V2B | 0.9648 | 0.9621 | **0.9647** | -0.0001 | **1** |

The ordering `V0 < V1 < V2 < V2B` was identical for Local, Public, and Final H. Across these four
submitted checkpoints, Local-Final Spearman correlation was `1.0`, Pearson correlation was about
`0.9991`, H MAE was `0.001625`, and H RMSE was `0.001897`. This is strong retrospective evidence
that local CLEval selected the right direction for this experiment sequence, though four related
models are too few to establish a universal calibration.

All adopted changes transferred to Final: V0 -> V1 `+0.0323`, V1 -> V2 `+0.0416`, and V2 -> V2B
`+0.0010` H. V2B ranked first overall with Final H/P/R `0.9647 / 0.9739 / 0.9580`, `+0.0195` H above
rank 2. The smaller V2 -> V2B Final gain versus Local (`+0.0010` versus `+0.0033`) supports using
paired bootstrap or K-fold consistency before accepting future changes near the `0.001` scale.
