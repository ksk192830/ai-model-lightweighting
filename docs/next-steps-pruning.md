# 다음 작업 — Pruning 실험

## 목적

양자화 실험에 이어 pruning 방식별 모델 크기, 추론 속도 및 정확도 변화를
비교한다. TensorRT에서 가속되지 않는 방식도 제외하지 않고, 가속 효과가
없는 원인과 실험 결과를 논문에 기록한다.

모든 실험은 동일한 RF-DETR checkpoint, 데이터, RTX 3080, TensorRT 버전,
입력 크기, warmup 횟수 및 측정 횟수를 사용한다.

## 실험군

### 1. Unstructured pruning

개별 가중치의 magnitude를 기준으로 작은 값을 0으로 만든다.

- pruning 비율: 10%, 20%, 30%, 50%
- 전체 모델 및 layer별 sparsity 기록
- pruning 전후 checkpoint 크기와 0의 비율 비교
- ONNX 및 TensorRT FP32/FP16 엔진 생성
- dense TensorRT 실행에서 모델 크기와 속도가 줄지 않는지 검증
- 정확도 저하와 재학습에 따른 회복 정도 기록

TensorRT가 임의의 0 값을 건너뛰지 않고 dense kernel을 선택하면 tensor
shape, FLOPs 및 engine 크기가 유지된다. 이 경우 비구조적 pruning이
실질적인 가속으로 연결되지 않는 원인을 실험 결과와 함께 논문에 정리한다.

### 2. Structured pruning

채널, attention head, FFN 차원 또는 layer처럼 모델 구조 단위를 제거한다.

- RF-DETR에서 제거 가능한 Conv/Linear channel과 attention head 조사
- 구조 의존성을 추적해 연결된 입력·출력 차원을 함께 변경
- pruning 비율별 실제 parameter 수와 FLOPs 감소 기록
- pruning 후 fine-tuning 수행
- ONNX export 및 TensorRT 엔진 생성 가능 여부 확인
- 모델 크기, 속도 및 정확도 비교

구조가 실제로 작아지므로 일반 dense TensorRT kernel에서도 속도와 메모리
감소를 기대할 수 있지만, 구현 난이도와 정확도 손실이 가장 크다.

### 3. M:N pruning

각 N개 가중치 그룹에서 M개만 남기는 반정형 pruning을 적용한다.

- 우선 실험: NVIDIA 2:4 sparsity
- 지원되는 Linear/Conv weight에만 2:4 pattern 적용
- pattern 준수 여부를 layer별로 검증
- pruning 후 fine-tuning 수행
- TensorRT sparse tactic 활성화 여부와 로그 기록
- sparse tactic 사용 전후 engine 속도 비교

RTX 3080의 sparse Tensor Core와 TensorRT가 지원하는 pattern을 만족해야
가속을 기대할 수 있다. pattern을 만족해도 지원되지 않는 layer 또는
입출력 변환 비용 때문에 전체 모델 속도 향상이 제한될 수 있다.

## 공통 평가

각 방식과 pruning 비율에 대해 다음 항목을 같은 조건으로 기록한다.

- checkpoint, ONNX 및 TensorRT engine 크기
- 전체 및 layer별 sparsity
- parameter 수와 FLOPs
- TensorRT engine-only latency
- 전처리와 후처리를 포함한 end-to-end latency
- mean, median, p95 및 FPS
- detection 및 segmentation 정확도
- 원본 대비 정확도 손실
- fine-tuning 전후 결과
- TensorRT tactic과 precision fallback 여부

## 권장 진행 순서

1. 공통 pruning 및 sparsity 측정 유틸리티 작성
2. Unstructured pruning 구현과 “TensorRT 가속 효과 없음” 가설 검증
3. 2:4 M:N pruning 구현과 sparse tactic 확인
4. Structured pruning 적용 대상을 선정하고 최소 단위부터 구현
5. 각 방식에 fine-tuning 추가
6. 동일 평가 스크립트로 결과 수집
7. 표와 그래프로 논문 결과 정리
