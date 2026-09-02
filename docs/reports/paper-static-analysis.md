# RF-DETR 경량화 정적 분석

## 논문 기재용 분석 방법

체크포인트는 `checkpoint['model']`의 전체 tensor element와 pruning 대상 weight element를 분리해 집계했다. ONNX는 initializer element, dtype 기준 tensor 저장 바이트, graph node 수를 집계하고 ONNX checker를 통과한 graph만 비교에 포함했다.

연산량은 shape inference 이후 Conv, MatMul, Gemm에 대해 산출했다. 곱셈-누산 1회를 1 MAC 또는 2 FLOPs로 정의했다. 동적 차원은 1로 치환했으며 elementwise, normalization, activation, resize, control-flow 및 data movement는 제외했다. 따라서 아래 MAC/FLOP는 전체 실행 비용이 아니라 동일 분석기로 얻은 dense 산술의 **부분 하한 추정치**다.

## 구조 및 연산량 결과

| ID | 상태 | 입력 | ONNX initializer elements | Δ vs B01 | Nodes | Δ vs B01 | MACs (lower bound) | Δ vs B01 | FLOPs (lower bound) | Estimated/all nodes | Current |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B01 | onnx-exported | 504 | 32,537,747 | +0.000% | 2,127 | +0.000% | 54,083,668,224 | +0.000% | 108,167,336,448 | 8.60% | PASS |
| U02 | static-analysis-rejected | 504 | 32,537,747 | +0.000% | 2,127 | +0.000% | 54,083,668,224 | +0.000% | 108,167,336,448 | 8.60% | PASS |
| R01 | onnx-exported | 432 | 32,358,035 | -0.552% | 2,127 | +0.000% | 37,749,285,120 | -30.202% | 75,498,570,240 | 8.60% | PASS |
| S01 | onnx-exported | 504 | 30,997,043 | -4.735% | 1,956 | -8.039% | 53,782,953,984 | -0.556% | 107,565,907,968 | 8.69% | PASS |
| S02 | onnx-exported | 504 | 29,456,339 | -9.470% | 1,783 | -16.173% | 53,482,239,744 | -1.112% | 106,964,479,488 | 8.81% | PASS |
| S03 | static-analysis-rejected | 504 | 31,470,707 | -3.279% | 2,127 | +0.000% | 53,870,676,224 | -0.394% | 107,741,352,448 | 8.60% | PASS |
| M01 | provisional-recovery-running | 504 | 32,537,747 | +0.000% | 2,127 | +0.000% | 54,083,668,224 | +0.000% | 108,167,336,448 | 8.60% | PENDING/STALE |

`ONNX initializer elements`는 ONNX 상수 tensor element 수이며 학습 가능한 parameter 수와 동일하다고 단정하지 않는다. 실제 지연시간 감소는 이 표가 아니라 동일 GPU에서 측정한 TensorRT median/IQR/P95로 검증해야 한다.

## 연산량 추정 커버리지

| ID | Conv MACs | MatMul MACs | Gemm MACs | Estimated nodes | Unresolved compute nodes | Excluded nodes | Resolved compute coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| B01 | 2,836,024,576 | 51,182,107,648 | 65,536,000 | 183 | 7 | 1,937 | 96.316% |
| U02 | 2,836,024,576 | 51,182,107,648 | 65,536,000 | 183 | 7 | 1,937 | 96.316% |
| R01 | 2,083,630,336 | 35,600,118,784 | 65,536,000 | 183 | 7 | 1,937 | 96.316% |
| S01 | 2,836,022,272 | 50,894,502,912 | 52,428,800 | 170 | 6 | 1,780 | 96.591% |
| S02 | 2,836,019,968 | 50,606,898,176 | 39,321,600 | 157 | 5 | 1,621 | 96.914% |
| S03 | 2,836,024,576 | 50,969,115,648 | 65,536,000 | 183 | 7 | 1,937 | 96.316% |
| M01 | 2,836,024,576 | 51,182,107,648 | 65,536,000 | 183 | 7 | 1,937 | 96.316% |

`Resolved compute coverage`는 Conv/MatMul/Gemm 후보 중 shape을 해석한 비율이며 전체 graph 연산 커버리지가 아니다. `Excluded nodes`는 자료 이동·activation 등을 포함하므로 node 개수만으로 비용 비율을 추론하지 않는다.

## 체크포인트 희소도

| ID | Model-state elements | Prunable weight elements | Zero sparsity |
|---|---:|---:|---:|
| B01 | 35,564,543 | 34,759,584 | 0.000% |
| U02 | 35,564,543 | 34,759,584 | 30.000% |
| R01 | N/A | N/A | N/A |
| S01 | 34,023,839 | 33,225,376 | 0.000% |
| S02 | 32,483,135 | 31,691,168 | 0.000% |
| S03 | 34,497,503 | 33,694,624 | 0.000% |
| M01 | N/A | 34,759,584 | 48.772% |

## 2:4 구조 검증

ONNX constant-weight 연산 156개 중 shape 적격 150개, 2:4 준수 150개로 graph-level compliance는 100.000%다. 다만 이 결과는 pattern 존재만 증명하며 실제 sparse tactic 선택은 TensorRT build log와 M01 dense-control 대비 M02 latency로 별도 입증한다.

## TensorRT engine 정적 분석

현재 M01 복구 완료를 대기 중이며 engine build 후 자동 채워진다.

## 해석상 제한

- 후보 간 MAC/FLOP 비교는 동일 분석기와 동일 연산 범위에서만 해석한다.
- S01/S02 decoder 제거처럼 node와 initializer가 감소해도 attention 계열의 산술 하한 감소폭은 작을 수 있다.
- 비정형 또는 2:4 pruning은 dense ONNX node/MAC를 줄이지 않는다. 지원 kernel 선택과 실측 latency가 가속 증거다.
- 복구 학습 중인 M01 행은 최종 ONNX가 생성될 때까지 잠정치다.
