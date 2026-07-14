# V6 V5 Post-processing Recalibration

## 상태

완료. V5의 `box_thresh=0.30`은 채택하지 않았고, 사전 gate에 따라 `0.35`는 실행하지 않았다.
V2B ResNet18 1024를 clean single-model control로 유지한다. 새 학습과 Public 제출은 없었다.

## 가설

V5 ResNet34 epoch 7은 V2B와 H-Mean이 사실상 같으면서 recall이 높고 precision이 낮았다.
이 차이가 score calibration 때문이라면 `box_thresh`를 `0.25`에서 `0.30`으로 높였을 때 약한
false positive가 제거되어, recall 손실보다 precision 이득이 커질 수 있다.

사전에 정한 실행 gate는 다음과 같았다.

1. V5 epoch 7에서 pixel `thresh=0.30`을 고정하고 `box_thresh=0.30`만 평가한다.
2. 0.30이 V5의 macro/global H를 모두 높일 때만 `box_thresh=0.35`를 평가한다.
3. V5가 V2B를 넘지 못하면 이 branch를 끝내고 V2B를 유지한다.

## 통제 조건

- Checkpoint: V5 ResNet34 `epoch=7-step=1640.ckpt`
- Resolution: 1024
- Dataset: official val 404장
- CLEval: detection-only POLY
- Pixel `thresh`: 0.30 고정
- `max_candidates`: 300 고정
- 변경 변수: `box_thresh 0.25 -> 0.30`
- Model weight와 입력 transform은 변경하지 않음

## 결과

| Candidate | Macro H | Macro P | Macro R | Global H | Global P | Global R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V2B control, box 0.25 | **0.964760** | **0.969976** | 0.961422 | **0.962219** | **0.967510** | 0.956985 |
| V5 control, box 0.25 | 0.964617 | 0.964341 | **0.966831** | 0.962185 | 0.960289 | **0.964088** |
| V6 V5, box 0.30 | 0.960568 | 0.966257 | 0.957238 | 0.958472 | 0.963035 | 0.953953 |
| V6 - V5 box 0.25 | -0.004049 | +0.001916 | -0.009593 | -0.003713 | +0.002746 | -0.010135 |
| V6 - V2B | -0.004192 | -0.003719 | -0.004184 | -0.003747 | -0.004475 | -0.003032 |

Precision은 조금 올랐지만 recall 손실이 약 다섯 배 커서 macro와 global H가 모두 하락했다.
0.30은 V2B보다 precision, recall, H가 모두 낮다. 따라서 0.35를 추가로 높이면 현재 관측된
recall 손실이 더 커질 가능성이 높고, 무엇보다 사전에 정한 실행 gate를 통과하지 못했다.

## 해석

V5의 높은 recall은 단순히 낮은 `box_thresh`로 약한 proposal을 많이 수용한 결과가 아니었다.
D0에서도 V5는 V2B보다 평균 polygon 수가 적으면서 recall이 높았다. 두 모델은 contour의 연결,
분리와 경계 형태가 다르며, V5의 유효한 작은 글자 또는 문자 coverage 중 일부가 0.25와 0.30
사이의 score를 가진 것으로 해석할 수 있다.

Higher threshold가 제거한 영역 가운데 false positive보다 true character coverage가 많았기
때문에 precision은 `+0.0019`만 올랐고 recall은 `-0.0096` 떨어졌다. 이는 D0에서 작은 글자
비율이 높을수록 recall이 낮아진 결과와도 일관된다. 다만 box별 score와 GT match를 직접 저장한
실험은 아니므로 제거된 각 polygon의 원인을 확정한 것은 아니다.

## 결정

- V5 `box_thresh=0.30` 폐기
- `box_thresh=0.35` 미실행: `skipped by predeclared gate`
- V2B 1024, `box_thresh=0.25`를 다음 single-model control로 유지
- V5 epoch 7, `box_thresh=0.25`는 V9 probability-map ensemble의 recall-diverse member로만 유지
- V7 augmentation은 D0 결정에 따라 skip
- 다음 실행 실험은 V8 `1024+1152` scale probability-map TTA
- Public leaderboard에 제출하지 않음

이 결과는 V5 single-model threshold branch만 종료한다. V9에서 probability map을 V2B와
평균하면 score calibration과 contour가 다시 달라지므로, final fusion threshold는 V9/V13에서
별도로 평가해야 한다.

## 실행 명령

```bash
cd /data/ephemeral/home/receipt-text-detection/baseline_code
python runners/test.py \
  preset=example \
  dataset_base_path=/data/ephemeral/home/receipt-text-detection/data/datasets/ \
  exp_name=v6_v5_box030_eval \
  project_name=receipt-text-detection \
  wandb=True \
  exp_version=v6-box030 \
  'checkpoint_path="/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v5_resnet34_1024/checkpoints/epoch=7-step=1640.ckpt"' \
  models.encoder.model_name=resnet34 \
  models.head.postprocess.box_thresh=0.30 \
  transforms.test_transform.transforms.0.max_size=1024 \
  transforms.test_transform.transforms.1.min_width=1024 \
  transforms.test_transform.transforms.1.min_height=1024
```

W&B runtime은 약 76초다. 추론 후 CLEval의 이미지별 polygon 평가가 실행 시간 대부분을
차지했다.

## Artifact

- Metric: [metrics.json](/data/ephemeral/home/receipt-text-detection/experiments/20260714-v6-v5-postprocess/metrics.json)
- Evaluation config: [config.yaml](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v6_v5_box030_eval/.hydra/config.yaml)
- Hydra overrides: [overrides.yaml](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v6_v5_box030_eval/.hydra/overrides.yaml)
- V5 checkpoint: [epoch=7-step=1640.ckpt](/data/ephemeral/home/receipt-text-detection/baseline_code/outputs/v5_resnet34_1024/checkpoints/epoch=7-step=1640.ckpt)
- W&B: [8cy2bpn0](https://wandb.ai/pep-per/receipt-text-detection/runs/8cy2bpn0)

