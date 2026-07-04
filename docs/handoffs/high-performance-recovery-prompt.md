# 고성능 데스크탑 Recovery 학습용 AI 프롬프트

아래 코드 블록 전체를 고성능 학습 데스크탑의 AI에게 전달한다.

```text
RF-DETR Segmentation Large 기반 경량화 프로젝트의 structured pruning
recovery fine-tuning을 수행하라.

목표는 S01~S04 front 후보를 고성능 GPU에서 recovery fine-tuning하고,
학습 결과를 기존 변환 데스크탑으로 전달할 수 있도록 정리하는 것이다.

인계 기준 Git branch와 학습 데이터:

- Repository: git@github.com:ksk192830/kips-ai-model-lightweighting.git
- Branch: feature/lightweighting-pipeline
- Dataset file: parking-front-v8-coco-segmentation.tar.gz
- Google Drive file ID: 1JiEhJBDuSm-KtgpJjBwDvQSd1pyP28El
- Google Drive:
  https://drive.google.com/file/d/1JiEhJBDuSm-KtgpJjBwDvQSd1pyP28El/view
- Size: 1,179,503,114 bytes
- SHA-256:
  599b231fdab3fd4c5b409e841f3217e4ad706569e5a8d63ccd597007181a1b8b

먼저 다음 문서를 읽어라.

1. docs/guides/experiment-workflow.md
2. docs/guides/training-portability.md
3. docs/handoffs/model-artifact-index.md
4. configs/experiments/registry.yaml
5. configs/experiments/defaults.yaml
6. configs/experiments/schema.yaml
7. scripts/experiments/README.md

작업 대상:

- S01: decoder layer 1개 제거
- S02: decoder layer 2개 제거
- S03: FFN dimension 약 20% 축소
- S04: FFN dimension 약 40% 축소
- Camera: front만 사용
- 최종 목적: recovery checkpoint 생성
- 이 장비에서는 TensorRT engine 생성이 필수 작업이 아니다.
- 정확도·latency 최종 평가는 수행하지 않는다.

중요 원칙:

- SSH 방식으로 Git 작업을 수행한다.
- 작업 시작 전 현재 branch와 working tree를 확인한다.
- 원격 feature/lightweighting-pipeline 브랜치의 최신 변경사항을 받는다.
- 사용자의 기존 변경사항을 삭제하거나 덮어쓰지 않는다.
- Git LFS 파일을 반드시 내려받는다.
- 기존 registry와 scripts/experiments 공통 실행 구조를 사용한다.
- 새로운 독립 학습 스크립트를 만들지 않는다.
- 결과는 artifacts/experiments/<ID>/front/recovery/에 저장한다.
- configs/experiments/defaults.yaml의 recovery_structured를 사용한다.
- S01~S04는 서로 다른 output 디렉터리를 사용한다.
- GPU 성능이나 VRAM에 따라 학습 hyperparameter를 자동 조절하지 않는다.
- S01~S04 본 학습은 batch size 2, gradient accumulation 8,
  effective batch size 16으로 고정한다.
- 장시간 학습 전에 각 후보의 1-batch smoke test를 수행한다.
- 실패 시 무조건 재시도하지 말고 원인을 기록한다.
- checkpoint_best_total.pth와 recovery-training.json을 함께 보존한다.

1. 저장소와 환경 확인

다음을 확인하고 기록하라.

- Git branch와 working tree
- Git commit
- Python, PyTorch, RF-DETR 버전
- CUDA 사용 가능 여부
- GPU 이름과 VRAM
- NVIDIA driver
- Git LFS 설치 여부

기본 명령:

저장소가 없다면 SSH로 clone한다.

git clone --branch feature/lightweighting-pipeline \
  git@github.com:ksk192830/kips-ai-model-lightweighting.git
cd kips-ai-model-lightweighting

이미 clone돼 있다면 다음을 실행한다.

git status
git branch --show-current
git pull --ff-only origin feature/lightweighting-pipeline
git lfs pull
nvidia-smi
.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

.venv가 없다면 다음을 실행한다.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt

2. 학습 데이터 검증

학습 데이터는 Git에 포함되지 않는다. 프로젝트 루트에서 다음과 같이
Google Drive 파일을 받는다.

python -m pip install gdown
gdown 1JiEhJBDuSm-KtgpJjBwDvQSd1pyP28El \
  -O /tmp/parking-front-v8-coco-segmentation.tar.gz

다운로드 파일의 SHA-256을 반드시 검증한다.

echo "599b231fdab3fd4c5b409e841f3217e4ad706569e5a8d63ccd597007181a1b8b  /tmp/parking-front-v8-coco-segmentation.tar.gz" \
  | sha256sum -c -

검증 성공 후 압축을 해제한다.

mkdir -p data/training/front
tar -xzf /tmp/parking-front-v8-coco-segmentation.tar.gz \
  -C data/training/front

다음 파일을 확인하라.

data/training/front/train/_annotations.coco.json
data/training/front/valid/_annotations.coco.json

검증 조건:

- COCO images, annotations, categories 필드 존재
- train/valid split 및 실제 이미지 존재
- 빈 데이터셋이 아님
- annotation이 참조하는 이미지가 실제로 존재
- class 순서가 configs/baseline.yaml의 front classes와 동일

데이터가 없거나 class 순서가 다르면 학습하지 말고 blocker를 보고하라.

3. 원본 checkpoint 검증

models/parking_front.pth가 Git LFS pointer가 아닌 실제 PTH인지 확인한다.
torch.load와 RF-DETR checkpoint load를 검증하고 SHA-256을 기록한다.

4. S01~S04 후보 생성

for id in S01 S02 S03 S04; do
  .venv/bin/python scripts/experiments/create_candidate.py \
    "$id" --camera front
done

기존 후보가 있다면 무조건 덮어쓰지 말고 metadata·registry와 일치하는지
확인한다. 후보별로 다음을 검증한다.

- model.pth, metadata.json, structured-pruning.json 존재
- checkpoint load 성공
- S01/S02 decoder와 segmentation block 수 일치
- S03 FFN dimension 1632
- S04 FFN dimension 1216

5. 1-batch GPU smoke test

for id in S01 S02 S03 S04; do
  .venv/bin/python scripts/experiments/train_candidate.py \
    "$id" --camera front --device auto \
    --batch-size 1 --grad-accum-steps 1 --fast-dev-run 1
done

각 smoke test에서 checkpoint와 dataloader load, forward/backward,
validation, CUDA OOM, NaN/Inf 및 구조 검증 결과를 확인한다. Smoke 결과는
본 학습 결과로 취급하지 않는다.

6. S01~S04 recovery fine-tuning

smoke test가 성공한 후보만 본 학습한다.

for id in S01 S02 S03 S04; do
  .venv/bin/python scripts/experiments/train_candidate.py \
    "$id" --camera front --device cuda \
    --batch-size 2 --grad-accum-steps 8
done

특정 GPU를 선택할 때는 --device cuda:N 대신 다음처럼 실행한다.

CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  scripts/experiments/train_candidate.py \
  S02 --camera front --device cuda \
  --batch-size 2 --grad-accum-steps 8

실행 전 다음 조건을 보고하고 기록한다.

- epoch, learning rate, encoder learning rate, weight decay
- batch size, gradient accumulation, effective batch size
- random seed와 AMP precision
- GPU 이름과 VRAM
- dataset 및 output 경로

S01~S04에 다음 조건을 동일하게 적용한다.

- epochs: 10
- learning rate: 1.0e-5
- encoder learning rate: 1.5e-5
- weight decay: 1.0e-4
- batch size: 2
- gradient accumulation: 8
- effective batch size: 16
- scheduler: cosine
- warmup epochs: 1
- EMA: enabled, decay 0.993
- early stopping: enabled, patience 3
- checkpoint interval: 1
- random seed: 42

GPU 성능이나 VRAM을 근거로 batch size, accumulation, epoch, learning rate
등을 자동 변경하지 않는다. batch size 2에서 CUDA OOM이 발생하면 micro
batch를 임의로 줄이지 말고 해당 후보 학습을 중단하여 blocker로 보고하고
사용자 승인을 기다린다.

7. 중단된 학습 재개

기존 output을 삭제하지 말고 Lightning CKPT로 재개한다.

.venv/bin/python scripts/experiments/train_candidate.py \
  S02 --camera front --device cuda \
  --batch-size 2 --grad-accum-steps 8 \
  --resume artifacts/experiments/S02/front/recovery/checkpoint_<epoch>.ckpt

PTH는 변환용이고 CKPT는 optimizer/scheduler를 포함한 재개용이다.

8. 학습 결과 검증

각 후보에서 다음 파일을 확인한다.

artifacts/experiments/<ID>/front/recovery/checkpoint_best_total.pth
artifacts/experiments/<ID>/front/recovery/recovery-training.json

검증 항목:

- training status completed
- checkpoint가 비어 있지 않고 torch.load 성공
- source checkpoint SHA-256 및 실제 학습 조건 기록
- epoch와 global step 기록
- structure_verification.valid=true
- S01/S02 decoder와 segmentation block 수 유지
- S03 FFN 1632 및 S04 FFN 1216 유지
- checkpoint tensor에 NaN/Inf 없음

recovery 결과를 기존 prototype model.pth에 임의로 덮어쓰지 않는다.

9. 인계 파일

S01~S04 각각의 아래 두 파일을 전달한다.

artifacts/experiments/<ID>/front/recovery/checkpoint_best_total.pth
artifacts/experiments/<ID>/front/recovery/recovery-training.json

각 파일의 SHA-256, 크기, Git commit, GPU/driver/CUDA/PyTorch/RF-DETR
버전, 실제 학습 조건, 완료 epoch, 실패·재개 이력과 구조 검증 결과도
정리한다. TensorRT engine은 환경 종속이므로 전달하지 않는다.

검증이 끝나면 네 후보 결과를 하나의 인계 파일로 묶는다.

mkdir -p /tmp/structured-recovery-handoff
for id in S01 S02 S03 S04; do
  mkdir -p "/tmp/structured-recovery-handoff/$id"
  cp "artifacts/experiments/$id/front/recovery/checkpoint_best_total.pth" \
    "/tmp/structured-recovery-handoff/$id/"
  cp "artifacts/experiments/$id/front/recovery/recovery-training.json" \
    "/tmp/structured-recovery-handoff/$id/"
done

tar -C /tmp -czf /tmp/structured-recovery-front-S01-S04.tar.gz \
  structured-recovery-handoff
sha256sum /tmp/structured-recovery-front-S01-S04.tar.gz

최종 보고에 `/tmp/structured-recovery-front-S01-S04.tar.gz`의 크기와
SHA-256을 포함한다. 사용자가 이 파일을 Drive로 옮길 수 있도록 정확한
경로를 보고한다.

10. Git 작업

코드나 문서를 수정한 경우에만 검토 후 커밋한다. PTH와 CKPT를 승인 없이
일반 Git에 추가하지 않는다. 공유가 필요하면 기존 Git LFS 정책과
shared-models 구조를 먼저 확인한다. 원격 작업은 SSH를 사용한다.

11. 최종 보고

- GPU와 소프트웨어 버전
- dataset 및 원본 checkpoint 검증 결과
- S01~S04 생성과 smoke test 결과
- 후보별 학습 성공 여부와 실제 조건
- 완료 epoch/global step 및 구조 유지 결과
- 생성된 checkpoint·training report 경로, 크기와 SHA-256
- 실패·재개 여부
- 변환 데스크탑으로 전달할 파일 목록
- 다음 작업

변환 데스크탑에서는 전달받은 PTH를 배치한 뒤 다음을 실행할 예정이다.

.venv/bin/python scripts/experiments/finalize_recovery.py \
  S01 S02 S03 S04 --camera front --force

이 명령은 prototype 보존, recovery PTH 승격, ONNX export, TensorRT engine
생성, B01 정적 비교와 문서 갱신을 수행한다.
```
