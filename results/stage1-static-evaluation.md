# RF-DETR 1차 정적평가 전체 후보표

등록된 26개 후보 중 26개의 1차 평가를 완료했다. 통과 22개, 불통 4개, 미수행 0개다.

1차는 산출물 유효성, 그래프 구조, 정적 효율, 양자화/희소성 적용 여부만 판정한다. 정확도, latency, FPS, GPU memory는 노트북 2차 평가에서만 판정한다.

| ID | 경량화 방식 요약 | 1차 평가 통과 여부 | 불통 사유(통과하지 못한 후보만) |
|---|---|---|---|
| B01 | none | 통과 | - |
| B02 | tensorrt-fp16 | 통과 | - |
| B03 | tensorrt-int8-ptq | 통과 | - |
| U01 | global-magnitude; sparsity=0.1 | 불통 | ONNX 크기·node·dense MACs 중 어느 항목도 사전 고정 5% 감소 기준을 충족하지 못함 |
| U02 | global-magnitude; sparsity=0.3 | 불통 | ONNX 크기·node·dense MACs 중 어느 항목도 사전 고정 5% 감소 기준을 충족하지 못함 |
| U03 | global-magnitude; sparsity=0.5 | 불통 | ONNX 크기·node·dense MACs 중 어느 항목도 사전 고정 5% 감소 기준을 충족하지 못함 |
| M01 | nvidia-2to4; pattern=2:4 | 통과 | - |
| M02 | nvidia-2to4; pattern=2:4 | 통과 | - |
| S01 | decoder-layer; remove_layers=1 | 통과 | - |
| S02 | decoder-layer; remove_layers=2 | 통과 | - |
| S03 | ffn-dimension; reduction=0.2 | 불통 | ONNX 크기·node·dense MACs 중 어느 항목도 사전 고정 5% 감소 기준을 충족하지 못함 |
| S04 | ffn-dimension; reduction=0.4 | 통과 | - |
| C01 | structured-selected | 통과 | - |
| C02 | structured-selected | 통과 | - |
| C03 | structured-resolution; input=432x432 | 통과 | - |
| C04 | structured-resolution; input=480x480 | 통과 | - |
| R01 | input-resolution; input=432x432 | 통과 | - |
| R02 | input-resolution; input=480x480 | 통과 | - |
| R03 | input-resolution; input=384x384 | 통과 | - |
| Q01 | modelopt-int8-smoothquant; INT8_SMOOTHQUANT_CFG | 통과 | - |
| Q02 | modelopt-mixed-precision; INT8_DEFAULT_CFG | 통과 | - |
| Q03 | modelopt-int4-weight-only; INT4_BLOCKWISE_WEIGHT_ONLY_CFG | 통과 | - |
| Q04 | modelopt-fp8; FP8_DEFAULT_CFG | 통과 | - |
| Q05 | modelopt-fp8-mixed; FP8_DEFAULT_CFG | 통과 | - |
| Q06 | modelopt-fp8-mixed-kld; FP8_DEFAULT_CFG | 통과 | - |
| Q07 | modelopt-fp8-mixed; FP8_DEFAULT_CFG | 통과 | - |

## 후속 절차

1. `stage2_notebook_eligible=true`인 후보를 노트북에서 TensorRT engine으로 빌드한다.
2. 고정 test 437장으로 bbox AP, mask AP, semantic mIoU를 측정한다.
3. warm-up 20회 뒤 200회 반복으로 median/p95 latency, FPS, peak GPU memory를 측정한다.
4. 모든 2차 결과가 모인 뒤 정확도-지연시간-메모리-크기의 비지배해를 Pareto 분석한다.
