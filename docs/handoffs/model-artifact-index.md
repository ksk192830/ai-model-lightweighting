# 모델 Artifact 인덱스

> 이 문서는 `scripts/experiments/update_index.py`가 생성한다.
> 직접 편집하지 말고 `configs/experiments/registry.yaml`을 수정한다.

| ID | 이름 | Camera | 방식 | Precision | 상태 | 파일 |
|---|---|---|---|---|---|---|
| B01 | baseline-fp32 | front | none | fp32 | engine-built | [ONNX](../../artifacts/experiments/B01/front/model.onnx), [engine](../../artifacts/experiments/B01/front/model.engine), [metadata](../../artifacts/experiments/B01/front/metadata.json) |
| B01 | baseline-fp32 | rear | none | fp32 | engine-built | - |
| B02 | baseline-fp16 | front | tensorrt-fp16 | fp16 | engine-built | [engine](../../artifacts/experiments/B02/front/model.engine), [metadata](../../artifacts/experiments/B02/front/metadata.json) |
| B02 | baseline-fp16 | rear | tensorrt-fp16 | fp16 | engine-built | - |
| B03 | baseline-int8 | front | tensorrt-int8-ptq | int8 | engine-built | [engine](../../artifacts/experiments/B03/front/model.engine), [metadata](../../artifacts/experiments/B03/front/metadata.json) |
| B03 | baseline-int8 | rear | tensorrt-int8-ptq | int8 | engine-built | - |
| U01 | unstructured-magnitude-10 | front | global-magnitude | fp32 | onnx-exported | - |
| U02 | unstructured-magnitude-30 | front | global-magnitude | fp32 | engine-built | - |
| U03 | unstructured-magnitude-50 | front | global-magnitude | fp32 | onnx-exported | - |
| M01 | sparse-2to4-dense-control | front | nvidia-2to4 | fp16 | engine-built | [ONNX](../../artifacts/experiments/M01/front/model.onnx), [engine](../../artifacts/experiments/M01/front/model.engine), [metadata](../../artifacts/experiments/M01/front/metadata.json) |
| M02 | sparse-2to4 | front | nvidia-2to4 | fp16 | engine-built | [engine](../../artifacts/experiments/M02/front/model.engine), [metadata](../../artifacts/experiments/M02/front/metadata.json), [sparse tactics](../../artifacts/experiments/M02/front/sparse-tactics.json) |
| S01 | structured-decoder-layer-1 | front | decoder-layer | fp32 | engine-built | [ONNX](../../artifacts/experiments/S01/front/model.onnx), [engine](../../artifacts/experiments/S01/front/model.engine), [metadata](../../artifacts/experiments/S01/front/metadata.json) |
| S02 | structured-decoder-layer-2 | front | decoder-layer | fp32 | engine-built | - |
| S03 | structured-ffn-20 | front | ffn-dimension | fp32 | engine-built | - |
| S04 | structured-ffn-40 | front | ffn-dimension | fp32 | engine-built | - |
| C01 | selected-structured-fp16 | front | structured-selected | fp16 | engine-built | [engine](../../artifacts/experiments/C01/front/model.engine), [metadata](../../artifacts/experiments/C01/front/metadata.json) |
| C02 | selected-structured-int8 | front | structured-selected | int8 | engine-built | - |
| R01 | reduced-resolution-fp16 | front | input-resolution | fp16 | engine-built | [ONNX](../../artifacts/experiments/R01/front/model.onnx), [engine](../../artifacts/experiments/R01/front/model.engine), [metadata](../../artifacts/experiments/R01/front/metadata.json) |

## 저장 규칙

```text
artifacts/experiments/<experiment-id>/<camera>/
```

평가 대상과 진행 순서는 [경량화 모델 실험 진행 계획](../guides/experiment-workflow.md)을 따른다.
