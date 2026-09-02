# RF-DETR NVIDIA 2:4 적용 가능성 조사

## 결론

RF-DETR Segmentation Large의 front 모델에서 Conv2d, Linear 및
MultiheadAttention projection weight를 조사했다.

| 항목 | 결과 |
|---|---:|
| 조사 layer | 206 |
| shape 기준 적용 가능 layer | 200 |
| shape 기준 적용 불가 layer | 6 |
| 조사 대상 parameter | 34,083,584 |
| shape 기준 적용 가능 parameter | 33,906,176 |

신규 M01 실행 시 전체 layer 목록, shape, parameter 수와 pruning 전 pattern
준수율을 `artifacts/experiments/M01/front/2to4-eligibility.json`에 기록한다.

이 결과는 weight shape가 2:4 pattern을 표현할 수 있다는 뜻이며, 실제
TensorRT sparse tactic 사용을 보장하지 않는다.

## TensorRT 요구 조건

NVIDIA TensorRT의 structured sparsity 조건은 다음과 같다.

- Ampere 이상 GPU가 필요하다. RTX 3080은 Ampere이므로 해당한다.
- Conv weight `[K, C, R, S]`는 각 `K, R, S` 위치에서 C축의 연속된
  4개마다 non-zero가 최대 2개여야 한다.
- constant weight MatrixMultiply는 reduction K축의 연속된 4개마다
  non-zero가 최대 2개여야 한다.
- 해당 Convolution 또는 MatrixMultiply 연산이 FP16이나 INT8이어야 한다.
- TensorRT builder에서 `BuilderFlag.SPARSE_WEIGHTS`를 활성화해야 한다.
- pattern을 만족해도 dense tactic이 더 빠르면 TensorRT가 dense tactic을
  선택할 수 있다.

참고:

- [NVIDIA TensorRT structured sparsity](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/io-formats-sparsity.html)
- [TensorRT 10.16 trtexec sparsity option](https://docs.nvidia.com/deeplearning/tensorrt/10.16.0/reference/command-line-programs.html)

현재 설치된 TensorRT `10.16.1.11`에는
`BuilderFlag.SPARSE_WEIGHTS`가 존재한다. Polygraphy CLI는 설치되어 있지
않아 checkpoint와 ONNX constant weight를 검사하는 동일 목적의 pattern
검증기를 프로젝트 내부에 구현했다.

## 적용 가능 weight

shape 기준 후보:

| 유형 | Layer 수 | 주요 대상 |
|---|---:|---|
| Linear | 186 | backbone attention/MLP, decoder FFN, projection, bbox/class head |
| MultiheadAttention in-projection | 5 | decoder self-attention QKV projection |
| Conv2d | 9 | projector convolution, segmentation projection |

Linear weight `[out_features, in_features]`는 `in_features % 4 == 0`인 경우
후보로 분류했다. Conv2d는 group당 input channel이 4의 배수인 경우
후보로 분류했다.

## 적용 제외 weight

다음 6개 Conv2d는 reduction channel을 4개 단위로 묶을 수 없어 제외한다.

- `segmentation_head.blocks.0.dwconv.weight`
- `segmentation_head.blocks.1.dwconv.weight`
- `segmentation_head.blocks.2.dwconv.weight`
- `segmentation_head.blocks.3.dwconv.weight`
- `segmentation_head.blocks.4.dwconv.weight`
- `backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight`

앞의 5개는 depthwise convolution이라 group당 input channel이 1이다.
patch embedding convolution은 RGB 입력이라 input channel이 3이다.

## M01/M02 구현 계획

### 공통 checkpoint

1. 적용 가능 weight를 reduction axis 기준 4개씩 묶는다.
2. 각 그룹에서 magnitude가 큰 2개만 유지한다.
3. 모든 layer의 pattern 준수율이 100%인지 검증한다.
4. fine-tuning 동안 제거된 weight가 다시 살아나지 않도록 mask를 유지한다.
5. FP16 ONNX를 생성한다.

### M01 — Dense tactic 대조군

- 동일한 2:4 checkpoint와 FP16 설정 사용
- `SPARSE_WEIGHTS` 비활성화
- sparse tactic 외의 조건을 M02와 동일하게 유지

### M02 — Sparse tactic

- M01과 동일한 checkpoint와 FP16 설정 사용
- `BuilderFlag.SPARSE_WEIGHTS` 활성화
- TensorRT logger를 VERBOSE로 설정
- build log에서 eligible layer와 실제 sparse tactic 선택 layer를 각각 기록

M01/M02는 weight와 precision이 같고 sparse tactic 허용 여부만 달라야 한다.
그래야 속도 차이를 sparse tactic 효과로 해석할 수 있다.

## 구현 전 통과 조건

- checkpoint의 대상 layer pattern 준수율 100%
- 제외 layer가 명시적으로 기록됨
- ONNX export/checker 성공
- ONNX constant weight의 2:4 pattern 재검증
- M02 build log에서 eligible sparse layer가 1개 이상 확인됨

eligible layer가 확인돼도 실제 sparse tactic 선택이 0개라면 실패가 아니라
“RF-DETR의 해당 problem size에서는 dense tactic이 선택됨”으로 기록한다.

## 신규 실험 기록 항목

과거 prototype과 engine은 제거했다. 새 checkpoint로 다음 항목을 모두 다시
측정한다.

- pruning 전후 2:4 pattern 준수율
- recovery fine-tuning epoch와 validation 정확도
- ONNX constant weight의 pattern 준수율
- TensorRT eligible layer와 실제 sparse tactic 선택 layer
- M01·M02 engine 크기와 동일 장비 latency
