# M01/M02 2:4 Recovery Fine-tuning 계획

## 현재 상태

M01/M02 recovery fine-tuning과 최종 TensorRT 재변환을 완료했다. 학습 전
prototype은 각 실험의 `prototype-before-recovery/`에 보존했다.

사전검증 원본:

- [M01 fine-tuning preflight](../artifacts/experiments/M01/front/fine-tuning-preflight.json)

| 검사 항목 | 결과 |
|---|---|
| M01 RF-DETR checkpoint 로드 | 성공 |
| checkpoint·설정·labeled-test class 순서 | 일치 |
| 원 학습 dataset | `/home/lair/datum/rfdetr_seg`, 현재 없음 |
| 설정된 recovery dataset | `data/training/front` |
| Roboflow source | `hmobility-fvu9x/parking_front`, version 8 |
| Roboflow `train` split | 9,380 images, COCO segmentation |
| Roboflow `valid` split | 893 images, COCO segmentation |
| `pytorch_lightning` | 설치 및 학습 확인 |
| `albumentations` | 설치 및 학습 확인 |
| `pycocotools` | 설치 및 학습 확인 |
| epoch/LR/batch 설정 | 표준 recovery profile로 확정 |
| 학습 실행 결과 | 10 epochs 완료 |

Roboflow API는 version 8을 총 10,720장으로 표시하지만 내려받은
COCO segmentation export에는 10,719장(train 9,380 / valid 893 /
test 446)이 들어 있다. 이 1장 차이는 원본 version과 export의 차이로
기록하고 평가 재현 시 동일 export version을 사용한다.

`data/labeled_test/front`는 COCO 형식이며 class 순서는 맞지만 평가 전용
296장 split이다. `train/valid` 구조가 없고 평가 오염을 일으키므로 recovery
학습 데이터로 사용하지 않는다.

## RF-DETR 1.8.1 학습 API 조사

공개 entry point는 다음과 같다.

```python
model.train(dataset_dir=..., output_dir=..., epochs=..., ...)
```

내부적으로 다음 Lightning stack을 구성한다.

1. `RFDETRModelModule`
2. `RFDETRDataModule`
3. `build_trainer`
4. `trainer.fit`

Roboflow COCO dataset은 아래 구조가 필요하다.

```text
<dataset>/
├── train/
│   └── _annotations.coco.json
├── valid/
│   └── _annotations.coco.json
└── test/
    └── _annotations.coco.json
```

`RFDETR.train(callbacks=...)`의 기존 callbacks dictionary는 RF-DETR 1.8.1에서
폐기되고 실제 Trainer로 전달되지 않는다. 따라서 2:4 recovery는 내부
Lightning 구성요소를 사용해 Trainer를 만든 뒤
`TwoOfFourMaskCallback`을 callback 목록에 추가하는 방식이 가장 안전하다.
site-packages를 직접 수정하거나 monkey patch하지 않는다.

## 구현된 mask 유지 방식

구현:

- `src/kips_lightweighting/pruning/sparse_2to4.py`
- `TwoOfFourMaskController`
- `make_lightning_mask_callback`
- 다른 GPU 실행 절차: [training-portability.md](training-portability.md)

두 단계로 zero regrowth를 차단한다.

1. 각 대상 parameter에 gradient hook을 등록해 pruned 위치의 gradient를
   0으로 만든다.
2. optimizer step 이후 동일한 고정 mask를 다시 곱해 momentum과 optimizer
   state에 의한 regrowth를 제거한다.

Lightning callback은 추가로 다음 시점에 mask를 적용한다.

- `on_before_optimizer_step`: gradient mask 안전 확인
- `on_before_zero_grad`: optimizer step 직후 weight mask 재적용
- `on_train_batch_end`: accumulation/strategy 차이에 대한 보조 재적용

일반 PyTorch 루프에서는 다음 API를 사용한다.

```python
controller.bind_gradient_hooks(model)
loss.backward()
controller.step(optimizer, model)
```

단위 테스트는 gradient 차단, AdamW step 후 pattern 유지, 강제로 되살린
weight의 재마스킹을 검증한다.

```bash
.venv/bin/python -m unittest tests/test_sparse_2to4.py -v
```

실제 M01 RF-DETR module의 200개 대상 parameter와 mask 이름·shape도 일치하며
regrown weight가 0개임을 확인했다.

## 학습 재개에 필요한 입력

다음 정보가 모두 확정될 때까지 학습하지 않는다.

1. front 학습 dataset 경로
2. `train/valid` COCO annotation과 이미지
3. 학습 dataset class 순서가
   `front, out_line, parking_lot, parking_space`인지 확인
4. 학습 extras 설치

확정한 표준 recovery profile:

```yaml
epochs: 10
lr: 1.0e-5
lr_encoder: 1.5e-5
batch_size: 2
grad_accum_steps: 8
effective_batch_size: 16
weight_decay: 1.0e-4
lr_scheduler: cosine
warmup_epochs: 1
use_ema: true
ema_decay: 0.993
early_stopping: true
early_stopping_patience: 3
checkpoint_interval: 1
random_seed: 42
```

설정 위치:

```text
configs/experiments/defaults.yaml
fine_tuning.recovery_2to4
```

최종 checkpoint:

```text
artifacts/experiments/M01/front/model.pth
```

학습 기록:

```text
artifacts/experiments/M01/front/recovery/recovery-training.json
```

## 완료된 실행 결과

- recovery epochs: 10
- mask 대상 layer: 200
- regrown weight: 0
- checkpoint 2:4 준수율: 100%
- ONNX checker: 성공
- ONNX 2:4 준수율: 100%
- M01 dense FP16 engine: 71,189,508 bytes
- M02 sparse FP16 engine: 70,647,028 bytes
- M02 sparse tactic 선택: 3개 Conv layer

학습 전후 정확도와 latency 평가는 모델 생성자의 범위가 아니며 평가자가
동일한 조건에서 수행한다.
