# 프로젝트 이전과 실험 재개

## 결론

Git에서 제외되는 대용량 파일은 `baseline_code/outputs/`에만 있지 않다.

| 위치 | 현재 크기 | 내용 | 실험 재개 필요성 |
|---|---:|---|---|
| `data/datasets/` | 약 888MB | train/val/test 이미지와 JSON | 필수, 별도 보관 |
| `data/pseudo_label/` | 약 2.9GB | CORD-v2, SROIE, WildReceipt | V10 이후에 필요, 별도 보관 |
| `baseline_code/outputs/` | 약 2.9GB | 체크포인트, Hydra 설정, 예측 JSON | 필수 체크포인트 포함 |
| `baseline_code/wandb/` | 약 12MB | 로컬 W&B run metadata | 온라인 sync 후에는 선택 사항 |
| `data.tar.gz` | 약 3.1GB | 이미 풀린 `data/`의 원본 압축본 | 중복이므로 이식본에서 제외 |

Git에는 코드, 설정, 실험 문서와 작은 metric 파일을 올리고, 데이터와 체크포인트는 portable
archive 또는 별도의 artifact storage로 보존해야 한다. W&B sync만으로 checkpoint가 보존되는 것은
아니다.

## Portable archive

프로젝트 상위 디렉터리에서 다음 파일이 생성된다.

```text
receipt-text-detection-portable-no-data-20260714.tar.zst
receipt-text-detection-portable-no-data-20260714.tar.zst.sha256
```

포함 항목:

- 기존의 모든 model checkpoint와 Hydra run config
- baseline 및 프로젝트 코드
- 실험 문서, metric, 시각화와 submission
- 로컬 W&B 기록과 Git metadata
- 원본 baseline 보존본인 작은 `code.tar.gz`

제외 항목:

- 전체 `data/` 디렉터리
- 풀린 데이터와 중복되는 `data.tar.gz` 및 `data.tar.gz.*`
- `__pycache__`, `*.pyc`와 도구 cache
- `.venv` 같은 머신 종속 Python 환경

다시 생성할 때는 다음을 실행한다.

```bash
cd /path/to/receipt-text-detection
bash scripts/create_portable_archive.sh
```

## 왜 `.tar.zst`인가

`tar`는 디렉터리 구조, 파일명과 실행 권한을 하나의 파일로 묶고, `zstd`는 그 tar 파일을 빠르게
압축한다. 모델 checkpoint처럼 이미 압축 효율이 낮은 대용량 파일이 많을 때 `gzip`보다 생성과
검사가 빠른 편이라 `.tar.zst`를 사용했다. 단점은 일부 Windows 기본 압축 도구가 지원하지 않을
수 있다는 점이다. Linux ML 환경에서는 `tar --zstd` 또는 `zstd`로 바로 풀 수 있다.

Shell script는 압축을 풀거나 실험하는 데 필수 프로그램이 아니다. 긴 제외 목록, 필수 checkpoint
검사, SHA-256 생성과 tar 검증을 매번 같은 방식으로 수행하기 위한 자동화 기록이다. 아래의 `tar`,
`zstd`, `sha256sum` 명령을 직접 실행해도 결과는 같다.

## 무결성 확인과 압축 해제

```bash
sha256sum -c receipt-text-detection-portable-no-data-20260714.tar.zst.sha256
tar --zstd -xf receipt-text-detection-portable-no-data-20260714.tar.zst
cd receipt-text-detection
bash scripts/verify_resume_assets.sh --code-only
```

이후 별도로 보관한 `data/`를 프로젝트 루트에 배치하고 전체 검사를 실행한다.

```bash
bash scripts/verify_resume_assets.sh
```

`tar --zstd`를 지원하지 않는 환경에서는 먼저 `zstd`를 설치하거나 다음처럼 푼다.

```bash
zstd -dc receipt-text-detection-portable-no-data-20260714.tar.zst | tar -xf -
```

## Python 환경 재구성

Portable archive는 OS, CUDA driver와 Python virtual environment 자체를 포함하지 않는다.
현재 실험 환경의 핵심 정보는 다음과 같다.

- Python `3.10.13`
- PyTorch `2.1.2+cu118`
- torchvision `0.16.2+cu118`
- PyTorch Lightning `2.1.3`
- CUDA-enabled GPU: NVIDIA GeForce RTX 3090 24GB
- NVIDIA driver `535.86.10`
- 실제 설치된 W&B `0.22.3`

기본 환경은 다음처럼 재구성한다.

```bash
cd receipt-text-detection
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r baseline_code/requirements.txt
python -m pip install wandb==0.22.3
```

`requirements.txt`에는 원본 baseline의 `wandb==0.16.1`이 적혀 있지만 현재 실험은 `0.22.3`에서
동작했다. 마지막 명령은 현재 logging 환경에 맞추기 위한 명시적 보정이다.

## 다른 경로에서 실행할 때

기존 experiment README와 Hydra output에는 당시 절대 경로
`/data/ephemeral/home/receipt-text-detection`이 기록되어 있다. 이는 과거 실행 기록이므로 일괄
수정하지 않는다. 새 머신에서는 실행할 때 현재 경로를 override한다.

```bash
PROJECT_ROOT="$(pwd)"
cd baseline_code

python runners/test.py \
  preset=example \
  dataset_base_path="$PROJECT_ROOT/data/datasets/" \
  checkpoint_path="$PROJECT_ROOT/baseline_code/outputs/v2b_resolution1024/checkpoints/epoch=8-step=1845.ckpt"
```

Hydra 인자 이름과 preset은 실행하려는 실험 README의 명령을 따르고, 절대 경로만 현재
`PROJECT_ROOT` 기준으로 바꾼다.

## 다음 실험에 필요한 핵심 자산

- V8 TTA control: V2B `epoch=8-step=1845.ckpt`
- V9 model ensemble: 위 V2B checkpoint와 V5 `epoch=7-step=1640.ckpt`
- V10 SSL 이후: 별도 보관한 공식 데이터와 `data/pseudo_label/` 후보 데이터
- 재현성 확인: 각 output의 `.hydra/config.yaml`, `.hydra/overrides.yaml`

Archive는 checkpoint와 설정을 보존하지만 `data/`는 포함하지 않는다. 압축 해제 직후에는
`scripts/verify_resume_assets.sh --code-only`로 코드와 핵심 checkpoint를 검사하고, 별도 데이터를
배치한 뒤 옵션 없이 실행하여 이미지와 pseudo-label 후보까지 확인한다.
