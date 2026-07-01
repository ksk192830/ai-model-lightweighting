# 다른 학습 장비에서 실행하기

RF-DETR recovery fine-tuning은 저장소 경로를 기준으로 실행하며 특정 사용자
홈 디렉터리나 RTX 3080에 의존하지 않는다. 다른 NVIDIA 데스크탑에서는
코드, 학습 데이터, 실험 checkpoint만 같은 상대 경로로 준비하면 된다.

## 옮겨야 하는 항목

```text
configs/
scripts/
src/
models/
artifacts/experiments/M01/front/model.pth
artifacts/experiments/M01/front/2to4-eligibility.json
data/training/front/
requirements.txt
requirements-train.txt
```

기존 학습을 중간부터 이어가려면 해당 recovery 디렉터리의
`checkpoint_<epoch>.ckpt`도 복사한다. TensorRT engine은 GPU와 TensorRT
환경에 종속되므로 학습 장비로 복사할 필요가 없으며 최종 배포 장비에서
다시 생성한다.

## 환경 설치

Python 3.10 이상과 NVIDIA driver가 필요하다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
```

CUDA 인식 여부를 확인한다.

```bash
nvidia-smi
.venv/bin/python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 사전검증과 학습

```bash
.venv/bin/python scripts/experiments/analyze_candidate.py \
  M01 --camera front --fine-tuning-preflight \
  --dataset-dir data/training/front

.venv/bin/python scripts/experiments/train_candidate.py \
  M01 --camera front --device auto \
  --output-dir artifacts/experiments/M01/front/recovery-portable
```

`--device auto`는 CUDA GPU가 있으면 GPU를 사용하고, 없으면 CPU를 선택한다.
GPU 메모리에 따른 기본 micro-batch는 다음과 같다.

| VRAM | batch size | accumulation | effective batch |
|---:|---:|---:|---:|
| 40GB 이상 | 8 | 2 | 16 |
| 16GB 이상 | 4 | 4 | 16 |
| 8GB 이상 | 2 | 8 | 16 |
| 8GB 미만 | 1 | 16 | 16 |

메모리가 부족하거나 여유가 많으면 직접 덮어쓴다.

```bash
.venv/bin/python scripts/experiments/train_candidate.py \
  M01 --camera front --device cuda:0 \
  --batch-size 4 --grad-accum-steps 4 \
  --output-dir artifacts/experiments/M01/front/recovery-portable
```

## 중단된 학습 재개

같은 output 디렉터리와 마지막 `.ckpt`를 지정한다.

```bash
.venv/bin/python scripts/experiments/train_candidate.py \
  M01 --camera front --device auto \
  --output-dir artifacts/experiments/M01/front/recovery-portable \
  --resume artifacts/experiments/M01/front/recovery-portable/checkpoint_4.ckpt
```

`.pth`는 배포·변환용 가중치이고 `.ckpt`는 optimizer와 scheduler까지 포함한
학습 재개 파일이다. 재개에는 반드시 `.ckpt`를 사용한다.

## 실행 기록

각 output 디렉터리의 `recovery-training.json`에 다음 정보가 기록된다.

- GPU 이름과 VRAM
- 실제 batch size와 gradient accumulation
- dataset/output 경로
- source checkpoint SHA-256
- resume checkpoint
- epoch와 global step
- 2:4 mask 검증 및 regrowth 개수

서로 다른 GPU에서 생성한 결과라도 이 기록을 함께 전달해야 재현할 수 있다.
