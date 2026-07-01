# 다른 학습 장비에서 실행하기

AI에게 전체 작업을 위임할 때는
[고성능 데스크탑 Recovery 학습용 AI 프롬프트](high-performance-recovery-prompt.md)를
그대로 전달한다.

RF-DETR recovery fine-tuning은 저장소 경로를 기준으로 실행하며 특정 사용자
홈 디렉터리나 RTX 3080에 의존하지 않는다. 다른 NVIDIA 데스크탑에서는
코드, 학습 데이터, 실험 checkpoint만 같은 상대 경로로 준비하면 된다.

## Structured recovery 데스크탑 인계 체크리스트

학습 장비에서 아래 순서로 확인한다.

- [ ] 이 저장소의 `feature/lightweighting-pipeline` 최신 commit을 checkout
- [ ] Git LFS 파일을 `git lfs pull`로 내려받음
- [ ] `models/parking_front.pth`가 LFS pointer가 아닌 실제 PTH인지 확인
- [ ] `data/training/front/train/_annotations.coco.json` 존재
- [ ] `data/training/front/valid/_annotations.coco.json` 존재
- [ ] train/valid class 순서가 `configs/baseline.yaml`과 동일
- [ ] Python 3.10+, NVIDIA driver, CUDA 지원 PyTorch 설치
- [ ] `nvidia-smi`와 `torch.cuda.is_available()` 성공
- [ ] S01~S04 후보 PTH를 아래 명령으로 재생성
- [ ] 각 후보에 대해 1-batch smoke test 성공
- [ ] 본 학습 output 디렉터리가 비어 있거나 `--resume`이 지정됨
- [ ] 학습 종료 후 `checkpoint_best_total.pth`와
  `recovery-training.json`을 함께 전달

## 옮겨야 하는 항목

```text
configs/
scripts/
src/
models/
data/training/front/
requirements.txt
requirements-train.txt
```

S01~S04는 원본 front checkpoint와 registry에서 결정적으로 다시 생성할 수
있으므로 prototype PTH를 별도로 복사할 필요가 없다. M01을 다시 학습하는
경우에만 `artifacts/experiments/M01/front/model.pth`와
`2to4-eligibility.json`을 추가로 옮긴다.

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

Structured 후보는 별도 mask 없이 동일 실행기를 사용한다.

```bash
for id in S01 S02 S03 S04; do
  .venv/bin/python scripts/experiments/create_candidate.py "$id" --camera front
done

.venv/bin/python scripts/experiments/train_candidate.py \
  S02 --camera front --device auto --fast-dev-run 1
```

본 학습은 smoke output과 분리되어 experiment별 기본 recovery 경로를
사용한다.

```bash
for id in S01 S02 S03 S04; do
  .venv/bin/python scripts/experiments/train_candidate.py \
    "$id" --camera front --device cuda \
    --batch-size 2 --grad-accum-steps 8
done
```

S01~S04의 output 디렉터리는 experiment별
`artifacts/experiments/<ID>/front/recovery/`로 자동 분리된다.
후보 간 공정한 비교를 위해 본 학습은 GPU 성능과 무관하게 batch size 2,
gradient accumulation 8, effective batch size 16으로 고정한다. 이 조건에서
OOM이 발생하면 임의로 값을 변경하지 않고 blocker로 보고한다.

여러 GPU 중 특정 GPU를 고를 때 RF-DETR 호환성을 위해 `cuda:N` 대신
환경변수로 노출 장치를 제한한다.

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  scripts/experiments/train_candidate.py \
  S02 --camera front --device cuda \
  --batch-size 2 --grad-accum-steps 8
```

## 학습 결과를 변환 장비에서 반영

학습 장비에서 받은 `checkpoint_best_total.pth`를 해당 experiment의
`recovery/` 디렉터리에 둔다. 다음 한 명령이 기존 prototype을
`prototype-before-recovery/`에 보존하고, recovery PTH 승격, ONNX export,
TensorRT engine 생성, B01 정적 비교와 문서 갱신까지 수행한다.

```bash
.venv/bin/python scripts/experiments/finalize_recovery.py \
  S01 S02 S03 S04 --camera front --force
```

파일을 다른 위치에 받았다면 experiment 하나와 함께 직접 지정한다.

```bash
.venv/bin/python scripts/experiments/finalize_recovery.py \
  S02 --camera front \
  --checkpoint /path/to/checkpoint_best_total.pth --force
```

실행 계획만 확인하려면 `--dry-run`을 사용한다. TensorRT engine은 변환
장비 환경에 종속되므로 최종 배포 GPU/TensorRT 환경에서도 다시 생성한다.

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
