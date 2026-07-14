# Submission Log

대회 종료 후에도 local 채택 기준을 통과한 candidate는 test prediction JSON과 CSV를 생성해
이 표에 남긴다. 실제 제출한 파일은 `Submitted`, 마감 후 생성한 파일은
`Generated offline, competition closed`로 구분한다. Local에서 폐기된 진단 실험은 생성하지
않는다.

| Date | CSV File | Prediction JSON | Config | Local CV | Official Val | Public LB | Final LB | Max Regions/Image | Invalid Polygons | Cap Applied | Risk Memo | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-07-09 | `submissions/v0_baseline_epoch8_20260709_012836.csv` | `baseline_code/outputs/v0_baseline_epoch8/submissions/20260709_012836.json` | `baseline_code/outputs/v0_baseline/.hydra/config.yaml` | N/A | H=0.8913 / R=0.8369 / P=0.9633 | H=0.8818 / R=0.8194 / P=0.9651 | H=0.8898 / R=0.8324 / P=0.9675 | 174 | 0 | No | Final H was -0.0015 versus local; high precision and low recall diagnosis was stable. | Submitted, Final complete |
| 2026-07-09 | `submissions/v1_box025_epoch8_20260709_235118.csv` | `baseline_code/outputs/v1_box025_epoch8/submissions/20260709_235118.json` | `baseline_code/outputs/v1_box025_epoch8/.hydra/config.yaml` | N/A | H=0.9248 / R=0.9057 / P=0.9499 | H=0.9185 / R=0.8932 / P=0.9511 | H=0.9221 / R=0.8978 / P=0.9554 | 184 | 0 | No | Final H +0.0323 over V0 confirms the local threshold direction. | Submitted, Final complete |
| 2026-07-13 | `submissions/v2_resolution896_epoch8_20260713_002412.csv` | `baseline_code/outputs/v2_resolution896_epoch8/submissions/20260713_002412.json` | `baseline_code/outputs/v2_resolution896/.hydra/config.yaml` | N/A | H=0.9615 / R=0.9611 / P=0.9638 | H=0.9603 / R=0.9556 / P=0.9667 | H=0.9637 / R=0.9606 / P=0.9682 | 215 | 0 | No | Final H +0.0416 over V1 and +0.0022 over local. Resolution 896 produced the largest Final gain. | Submitted, Final complete |
| 2026-07-14 | `submissions/v2b_resolution1024_epoch8_20260714_001730.csv` | `baseline_code/outputs/v2b_resolution1024_epoch8/submissions/20260714_001730.json` | `baseline_code/outputs/v2b_resolution1024/.hydra/config.yaml` | N/A | H=0.9648 / R=0.9614 / P=0.9700 | H=0.9621 / R=0.9520 / P=0.9754 | H=0.9647 / R=0.9580 / P=0.9739 | 226 | 0 | No | Final rank 1. H +0.0010 over V2, with precision +0.0057 and recall -0.0026. | Submitted, Final rank 1 |

Final score evidence: [2026-07-14 leaderboard](/data/ephemeral/home/receipt-text-detection/docs/leaderboard/20260714/README.md).
