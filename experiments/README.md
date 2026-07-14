# Experiments

실험마다 별도 폴더를 만들고 config, score, submission, error note를 남긴다.

현재 권장 실행 순서와 용어 설명은
[Clean-data Experiment Roadmap](clean_data_experiment_roadmap.md)에 기록한다. 이 로드맵은
공식 human label 데이터로 성능을 먼저 높인 뒤 pseudo label을 사용하는 순서다.

최근 실험:

- [V10 Domain SSL](20260714-v10-domain-ssl/README.md): rejected as statistical tie; recall improved but macro/global H bootstrap CIs crossed zero
- [V9 Model Ensemble](20260714-v9-model-ensemble/README.md): adopted locally; equal V2B/V5 map fusion raised recall and macro/global H
- [V8 Scale TTA](20260714-v8-scale-tta/README.md): adopted locally; 1024+1152 probability-map fusion raised macro/global H
- [V6 V5 Post-processing](20260714-v6-v5-postprocess/README.md): rejected; box 0.30 lost recall and both macro/global H
- [D0 Train/Val/Test Data Audit](20260714-d0-data-audit/README.md): completed; V7 skip, small-text scale TTA and V2B/V5 ensemble hypotheses retained
- [V2 Resolution 896](20260712-v2-resolution-896/README.md): adopted, then superseded by 1024
- [V2B Resolution 1024](20260713-v2b-resolution-1024/README.md): Final H 0.9647, rank 1, current best clean single model
- [V3 Cosine 1024](20260713-v3-cosine-1024/README.md): rejected by local CLEval
- [V4 Photometric 1024](20260714-v4-photometric-1024/README.md): rejected; recall signal did not offset H-Mean loss
- [V5 ResNet34 1024](20260714-v5-resnet34-1024/README.md): H tied V2B; retained as a recall-diverse candidate

권장 이름:

```text
YYYYMMDD-HHMM-short-name/
```

각 실험 폴더에는 최소한 다음을 저장한다.

- config copy
- local validation score
- official validation score if available
- sample prediction visualization
- changed hypothesis
- decision: keep / reject / revisit

대회 종료 후에도 local 채택 기준을 통과한 candidate는 test 413장 prediction JSON, 제출 형식
CSV와 sanity-check 결과를 생성한다. 이 artifact는 `Generated offline, competition closed`로
표시하며 leaderboard 점수로 해석하지 않는다.
