# 새 RF-DETR Front 모델 학습

## 현재 상태

학습 환경, 실행기, 공식 pretrained weight와 무증강 최종 데이터셋이
준비되어 있으며 본 학습과 최종 test 평가까지 완료되었다.
`front_session_split_v1`은 증강·중복·세션·COCO 참조 감사를 통과했다.

| 준비 항목 | 결과 |
|---|---|
| RTX 5060 Ti / CUDA 인식 | PASS |
| RF-DETR 1.8.1 학습 의존성 | PASS |
| 공식 Seg Large pretrained MD5 | PASS (`275f7b094909544ed2841c94a677d07e`) |
| COCO 이미지·annotation·클래스 검사 | PASS |
| Batch 2 GPU train 1-batch | PASS |
| Validation 1-batch | PASS |
| 최종 데이터 provenance | PASS - Roboflow Version 9 무증강 |
| split 간 이미지·세션 누수 | PASS - 각각 0개 |
| 본 학습 | 완료 - 18 epoch에서 조기 종료 |
| 최종 test 평가 | 완료 - Mask AP50:95 0.6010 |

## 재현 설정

이전에 기록된 실제 학습 인자를 복구해 새 baseline에 적용했다. 과거 checkpoint
파일은 신규 학습 완료 후 제거했다.

| 항목 | 값 |
|---|---:|
| Architecture | RF-DETR Segmentation Large |
| 입력 크기 | 504×504 |
| Epoch 상한 | 25 |
| Learning rate | 1.0e-4 |
| Encoder learning rate | 1.5e-4 |
| Weight decay | 1.0e-4 |
| LR scheduler | step, lr_drop=100 |
| Multi-scale | 사용 |
| EMA | 사용, decay=0.993 |
| Early stopping | patience=6, min_delta=0.001 |
| Micro-batch | 2 |
| Gradient accumulation | 8 |
| Effective batch | 16 |
| Random seed | 42 |

기존 모델은 batch 4 × accumulation 4였지만 이 장비는 8GB GPU이므로 batch 2 ×
accumulation 8로 바꿨다. Effective batch 16은 동일하다. 기존 학습의 seed는
기록되지 않아 새 실험에서는 42로 고정했다. Test는 checkpoint 선택에 사용하지
않도록 학습 중 자동 평가를 끈다.

기존 `parking_front.pth`는 설정상 최대 40 epoch였지만 checkpoint의 학습 상태를
확인하면 조기 종료로 실제 완료된 epoch는 19회다. 따라서 40을 그대로 반복할
근거가 없으며, 새 실험은 25 epoch를 안전 상한으로 두고 validation segmentation
mAP50:95가 0.001 이상 개선되지 않는 상태가 6회 연속이면 종료한다. 최종 모델은
마지막 epoch가 아니라 validation 지표가 가장 높았던 checkpoint를 사용한다.
25 epoch까지 계속 개선되면 `last.ckpt`에서 상한만 늘려 재개할 수 있다.

## 환경

저장소 루트에서 실행한다.

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install "rfdetr[train]==1.8.1" \
  "PyYAML==6.0.2" "psutil>=5.9.0"
```

학습 설정의 단일 기준은
`configs/training/front_rfdetr_seg_large.yaml`이다.

## 최종 데이터 GPU smoke test

```bash
.venv/bin/python scripts/training/train_rfdetr_front.py \
  --device cuda \
  --fast-dev-run 1 \
  --output-dir artifacts/training/front-rfdetr-seg-large-v1-final-smoke
```

Smoke test는 학습과 validation을 한 배치씩만 실행하여 실제 본 학습의
데이터 로딩·loss·CUDA 경로를 검증한다.

## 본 학습

먼저 사전검사를 실행한다. 설정은 이미
`data/training/front_session_split_v1`을 가리킨다.

```bash
.venv/bin/python scripts/training/train_rfdetr_front.py \
  --device cuda --preflight-only
```

PASS이면 본 학습을 시작한다.

```bash
.venv/bin/python scripts/training/train_rfdetr_front.py --device cuda
```

출력은 `artifacts/training/front-rfdetr-seg-large-v1/`에 저장된다. 5 epoch마다
재개용 checkpoint를 남기며 실행 환경,
데이터 경로, 실제 인자와 상태는 `preflight.json` 및 `training-run.json`에
남는다.

## 중단 후 재개

```bash
.venv/bin/python scripts/training/train_rfdetr_front.py \
  --device cuda \
  --resume artifacts/training/front-rfdetr-seg-large-v1/checkpoint_<epoch>.ckpt
```

재개에는 optimizer와 scheduler 상태가 들어 있는 `.ckpt`를 사용한다.

## 최종 평가

학습과 checkpoint 선택이 끝난 뒤 한 번만 test를 평가한다.

```bash
.venv/bin/python scripts/evaluation/evaluate_coco_rfdetr_pth.py \
  --checkpoint artifacts/training/front-rfdetr-seg-large-v1/checkpoint_best_total.pth \
  --dataset-dir data/training/front_session_split_v1/test \
  --name front-rfdetr-seg-large-v1-test \
  --device cuda
```

평가 결과에는 bbox AP, mask AP와 semantic mask mIoU가 함께 기록된다.

현재 최종 test 437장에 대한 1회 평가는 완료되었다. 결과는
`docs/reports/front-rfdetr-seg-large-v1-test.md`와
`results/coco-evaluation/front-rfdetr-seg-large-v1-test.json`에 기록한다.
