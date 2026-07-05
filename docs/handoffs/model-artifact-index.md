# 모델 Artifact 인덱스

> 이 문서는 `scripts/experiments/update_index.py`가 생성한다.
> 직접 편집하지 말고 `configs/experiments/registry.yaml`을 수정한다.

| ID | 이름 | Camera | 방식 | Precision | 상태 | 파일 |
|---|---|---|---|---|---|---|
| B01 | baseline-fp32 | front | none | fp32 | engine-built | - |
| B01 | baseline-fp32 | rear | none | fp32 | engine-built | - |
| B02 | baseline-fp16 | front | tensorrt-fp16 | fp16 | engine-built | - |
| B02 | baseline-fp16 | rear | tensorrt-fp16 | fp16 | engine-built | - |
| B03 | baseline-int8 | front | tensorrt-int8-ptq | int8 | engine-built | - |
| B03 | baseline-int8 | rear | tensorrt-int8-ptq | int8 | engine-built | - |
| U01 | unstructured-magnitude-10 | front | global-magnitude | fp32 | onnx-exported | - |
| U02 | unstructured-magnitude-30 | front | global-magnitude | fp32 | engine-built | - |
| U03 | unstructured-magnitude-50 | front | global-magnitude | fp32 | onnx-exported | - |
| M01 | sparse-2to4-dense-control | front | nvidia-2to4 | fp16 | engine-built | - |
| M02 | sparse-2to4 | front | nvidia-2to4 | fp16 | engine-built | - |
| M03 | sparse-2to4-extended-recovery | front | nvidia-2to4 | fp16 | checkpoint-created | [PTH](../../artifacts/experiments/M03/front/model.pth), [metadata](../../artifacts/experiments/M03/front/metadata.json), [sparsity](../../artifacts/experiments/M03/front/sparsity.json), [2:4 survey](../../artifacts/experiments/M03/front/2to4-eligibility.json), [fine-tuning preflight](../../artifacts/experiments/M03/front/fine-tuning-preflight.json), [recovery training](../../artifacts/experiments/M03/front/recovery/recovery-training.json) |
| Q01 | int8-smoothquant-ptq | front | modelopt-int8-smoothquant | int8 | planned | - |
| Q02 | mixed-precision-sensitive-fp16 | front | modelopt-mixed-precision | mixed | planned | - |
| Q03 | int4-weight-only-ffn | front | modelopt-int4-weight-only | int4 | planned | - |
| S01 | structured-decoder-layer-1 | front | decoder-layer | fp32 | checkpoint-created | [PTH](../../artifacts/experiments/S01/front/model.pth), [metadata](../../artifacts/experiments/S01/front/metadata.json), [structured pruning](../../artifacts/experiments/S01/front/structured-pruning.json), [recovery training](../../artifacts/experiments/S01/front/recovery/recovery-training.json) |
| S02 | structured-decoder-layer-2 | front | decoder-layer | fp32 | checkpoint-created | [PTH](../../artifacts/experiments/S02/front/model.pth), [metadata](../../artifacts/experiments/S02/front/metadata.json), [structured pruning](../../artifacts/experiments/S02/front/structured-pruning.json), [recovery training](../../artifacts/experiments/S02/front/recovery/recovery-training.json) |
| S03 | structured-ffn-20 | front | ffn-dimension | fp32 | checkpoint-created | [PTH](../../artifacts/experiments/S03/front/model.pth), [metadata](../../artifacts/experiments/S03/front/metadata.json), [structured pruning](../../artifacts/experiments/S03/front/structured-pruning.json), [recovery training](../../artifacts/experiments/S03/front/recovery/recovery-training.json) |
| S04 | structured-ffn-40 | front | ffn-dimension | fp32 | checkpoint-created | [PTH](../../artifacts/experiments/S04/front/model.pth), [metadata](../../artifacts/experiments/S04/front/metadata.json), [structured pruning](../../artifacts/experiments/S04/front/structured-pruning.json), [recovery training](../../artifacts/experiments/S04/front/recovery/recovery-training.json) |
| C01 | selected-structured-fp16 | front | structured-selected | fp16 | engine-built | [recovery training](../../artifacts/experiments/S01/front/recovery/recovery-training.json) |
| C02 | selected-structured-int8 | front | structured-selected | int8 | engine-built | [recovery training](../../artifacts/experiments/S01/front/recovery/recovery-training.json) |
| R01 | reduced-resolution-fp16 | front | input-resolution | fp16 | engine-built | - |

## 저장 규칙

```text
artifacts/experiments/<experiment-id>/<camera>/
```

평가 대상과 진행 순서는 [경량화 모델 실험 진행 계획](../guides/experiment-workflow.md)을 따른다.
