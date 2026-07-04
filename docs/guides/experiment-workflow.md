# 경량화 모델 실험 진행 계획

이 문서는 모델을 무작정 많이 만들지 않고, 비교 목적이 분명한 front 후보를
먼저 생성한 뒤 유망한 모델만 rear로 확장하기 위한 작업 순서다.

파일 위치는 [모델 Artifact 인덱스](../handoffs/model-artifact-index.md)에서 관리한다.

## 1. 기본 원칙

1. 탐색과 방식별 비교는 `front` 모델만 사용한다.
2. 최종 성능평가는 TensorRT engine을 기준으로 한다.
3. PTH와 ONNX는 재현, 정적 분석 및 engine 재생성을 위해 보관한다.
4. 정적 분석을 통과한 모델만 TensorRT engine으로 만든다.
5. front 평가에서 유망한 모델만 동일한 조건으로 `rear`에 적용한다.
6. 최종 선정은 parameter나 sparsity가 아니라 평가자의 정확도·latency
   Pareto 결과로 결정한다.
7. 모든 조합을 만드는 full factorial 실험은 하지 않는다.

## 2. 최종적으로 비교할 경량화 계열

| ID | 계열 | 대표 설정 | 검증 목적 |
|---|---|---|---|
| B01 | Baseline | TensorRT FP32 | 모든 결과의 기준 |
| B02 | Precision | TensorRT FP16 | 일반적인 저정밀도 최적화 |
| B03 | Precision | TensorRT INT8 | 강한 양자화 효과 |
| U01~U03 | Unstructured | magnitude 10/30/50% | sparsity 증가와 dense 실행의 한계 |
| M01~M02 | Semi-structured | NVIDIA 2:4, dense/sparse tactic | 하드웨어 지원 sparsity 효과 |
| S01~S02 | Structured layer | decoder layer 1/2개 제거 | graph depth 감소 효과 |
| S03~S04 | Structured FFN | FFN dimension 20/40% 축소 | Transformer 내부 연산 감소 |
| C01 | 결합 | structured 대표 + FP16 | 구조 축소와 FP16 결합 |
| C02 | 결합 | structured 대표 + INT8 | 구조 축소와 INT8 결합 |
| R01 | 입력 최적화 | 입력 해상도 504→432 + FP16 | 노트북 환경의 직접적인 연산량 감소 |

`C01`과 `C02`의 structured 설정은 `S01~S04`의 정적 분석과 변환 결과를
확인한 뒤 하나를 선택한다.

## 3. 단계별 진행 순서

### 0단계 — 실험 조건 고정

- [x] 원본 checkpoint와 SHA-256 기록
- [x] RF-DETR, PyTorch, CUDA, TensorRT 버전 기록 구조 확정
- [x] input shape, batch size, class 목록 기록
- [x] ONNX opset과 TensorRT workspace 고정
- [x] INT8 calibration dataset 경로 고정
- [x] artifact 저장 규칙과 experiment ID 확정

조건이 바뀌면 같은 실험 ID를 재사용하지 않는다.

설정과 진행 상태는 다음 파일을 단일 기준으로 사용한다.

```text
configs/experiments/defaults.yaml
configs/experiments/registry.yaml
configs/experiments/schema.yaml
```

일반 작업은 `scripts/experiments/`의 실행기를 사용하고,
`scripts/lightweighting/`은 내부 변환 구현으로만 사용한다.

Candidate 생성·분석·빌드·학습 명령은 성공 시 `metadata.json`과
`docs/handoffs/model-artifact-index.md`를 자동으로 갱신한다. 모델 생성 후 인덱스
명령을 별도로 기억할 필요가 없다. 이 문서의 연구 판단과 체크리스트는
검토를 거쳐 수정하고, 모델 경로와 artifact 상태는 자동 생성 인덱스를
기준으로 한다.

### 1단계 — Baseline 확인

- [x] front/rear 원본 PTH 확인
- [x] front/rear ONNX FP32 생성
- [x] front/rear TensorRT FP32/FP16/INT8 생성
- [ ] 각 engine의 precision fallback과 build log 정리

Baseline은 후보 탈락 대상이 아니며 모든 비교에 포함한다.

### 2단계 — Unstructured magnitude 대조군

- [x] front 10/30/50% PTH와 ONNX 생성
- [x] 전체·layer별 sparsity 및 파일 크기 기록
- [x] 중복인 front 20%와 탐색 전 rear artifact 삭제
- [x] front 30% TensorRT FP32 engine 하나 생성
- [x] engine 크기와 graph가 baseline과 동일한지 확인

Unstructured 모델의 목적은 실제 가속 후보 탐색이 아니라, 임의의 zero가
dense TensorRT에서 구조·FLOPs 감소로 이어지지 않는다는 점을 검증하는
것이다. 따라서 모든 비율의 FP32/FP16 engine을 만들지 않는다.

### 3단계 — NVIDIA 2:4 pruning

- [x] 지원 가능한 Conv/Linear weight 목록 조사
- [x] front 2:4 prototype checkpoint 생성
- [x] 모든 대상 layer의 pattern 준수율 검증
- [x] 2:4 mask 유지 recovery fine-tuning 10 epochs
- [x] ONNX export와 checker 및 ONNX pattern 검증
- [x] FP16 dense-tactic 대조 engine 생성
- [x] FP16 sparse-tactic engine 생성
- [x] build log에서 sparse tactic 활성화 여부 기록

학습 전 M01/M02 산출물은 각 실험의 `prototype-before-recovery/`에
보존했다. 최종 M01/M02는 2:4 mask를 유지한 10-epoch recovery checkpoint와
동일 ONNX를 사용해 다시 생성했다.

조사 결과와 M01/M02 조건은
[RF-DETR NVIDIA 2:4 적용 가능성 조사](../concepts/2to4-eligibility.md)에 기록한다.
학습 API, mask 구현 및 현재 blocker는
[M01/M02 2:4 Recovery Fine-tuning 계획](2to4-fine-tuning.md)에 기록한다.

### 4단계 — Structured pruning

먼저 구현 안정성이 높은 decoder layer pruning을 수행하고, 이후 FFN
dimension pruning을 검토한다.

- [x] S01: decoder layer 1개 제거 prototype·ONNX·TensorRT 검증
- [x] S02: decoder layer 2개 제거 prototype·ONNX·TensorRT 검증
- [x] S03: FFN dimension 20% 축소(실제 2048→1632, TensorRT 검증)
- [x] S04: FFN dimension 40% 축소(실제 2048→1216, TensorRT 검증)
- [x] 연결된 tensor shape와 출력 shape 검증
- [x] parameter 감소 측정(FLOPs는 operator-aware 분석 대기)
- [x] recovery fine-tuning
- [x] ONNX export와 TensorRT FP32 변환

S01 prototype 결과:

- decoder layer: 5 → 4
- segmentation block: 5 → 4
- ONNX initializer parameter: 4.74% 감소
- ONNX node: 8.04% 감소
- baseline과 입력·출력 interface 동일
- CPU detection/segmentation smoke inference 성공
- parameter 5% 기준에는 0.27%p 미달하므로 FLOPs 분석과 recovery 결과를
  확인하는 조건부 후보로 유지

S02 prototype 결과:

- decoder layer: 5 → 3
- segmentation block: 5 → 3
- ONNX initializer parameter: 9.47% 감소
- ONNX node: 16.17% 감소
- baseline과 입력·출력 interface 동일
- CPU detection/segmentation smoke inference 성공
- parameter 5% 기준을 통과해 recovery 학습 후보로 선정
- portable recovery runner의 GPU 1-batch train/validation smoke test 성공

S01~S04 TensorRT FP32 변환 결과:

| ID | 구조 | Engine 크기 | Engine B01 대비 | FLOPs lower-bound | FLOPs B01 대비 | 역직렬화 |
|---|---|---:|---:|---:|---:|---|
| B01 | 원본 baseline | 135,608,716 bytes | 기준 | 108.167 GFLOPs | 기준 | 성공 |
| S01 | decoder 5→4 | 129,252,636 bytes | -4.69% | 107.566 GFLOPs | -0.56% | 성공 |
| S02 | decoder 5→3 | 122,930,876 bytes | -9.35% | 106.964 GFLOPs | -1.11% | 성공 |
| S03 | FFN 2048→1632 | 131,273,148 bytes | -3.20% | 107.741 GFLOPs | -0.39% | 성공 |
| S04 | FFN 2048→1216 | 127,076,028 bytes | -6.29% | 107.315 GFLOPs | -0.79% | 성공 |

모든 structured engine은 TensorRT 10.16.1.11, FP32, batch 1,
입력 504×504, workspace 4096 MiB 조건에서 생성됐다. TensorRT runtime
역직렬화와 input 1개·output 3개의 총 4개 I/O tensor 확인을 통과했다.
FLOPs는 ONNX Conv/MatMul/Gemm을 대상으로 MAC 1회를 2 FLOPs로 계산한
보수적 lower-bound다. shape inference가 복원하지 못한 symbolic dimension은
1로 두며, 미지원 또는 미해결 연산과 계산 coverage를 각
`static-analysis.json`에 함께 기록한다. 따라서 절대 하드웨어 처리량이 아니라
동일 export 조건에서의 후보 간 정적 비교값으로 사용한다.

S01~S04 recovery fine-tuning은 RTX 5070에서 완료했고 전달받은 checkpoint의
SHA-256을 학습 보고서와 대조했다. 조기 종료 시점은 S01 4 epochs, S02 9
epochs, S03/S04 6 epochs다. Recovery PTH와 ONNX는 변환 저장소에
승격했으며, prototype TensorRT engine은 각 후보의
`prototype-before-recovery/`에 보존했다. 최종 FP32 engine은 RTX 3080,
TensorRT 10.16.1.11 환경에서 recovery ONNX를 기준으로 다시 생성했다.

정적 분석에서 다음을 모두 만족하지 못하면 TensorRT 평가 후보에서
제외한다.

- checkpoint 로드와 smoke inference 성공
- ONNX export와 checker 성공
- baseline과 동일한 출력 인터페이스
- 전체 parameter 5% 이상 또는 FLOPs 10% 이상 감소
- NaN/Inf 없음
- 다른 후보와 parameter 차이 3%, FLOPs 차이 5% 미만이면 중복 후보 제거

### 5단계 — 결합 모델

Structured 후보 중 변환이 안정적이고 감소량이 큰 모델 하나를 선택한다.

- [x] C01: S01 structured 모델 + FP16
- [x] C02: S01 structured 모델 + INT8
- [x] INT8 calibration과 FP16 fallback 기록
- [x] R01: 입력 해상도 504→432 + FP16

PTQ INT8의 출력이 비정상적이거나 평가 정확도 손실이 크다는 피드백이 있을
때만 QAT를 추가한다.

### 6단계 — Front 평가 인계

평가자에게 넘길 front TensorRT 후보는 약 12~15개로 제한한다.

- Baseline 3개
- Unstructured 대표 1개
- 2:4 대조/희소 engine 2개
- Structured 단일 방식 4개 이하
- Structured 결합 2개
- 입력 해상도 후보 1개

정확도, latency, FPS, GPU memory 결과를 받은 뒤 Pareto frontier와 방식별
대표성을 기준으로 5~8개를 선정한다.

### 7단계 — Rear 확장

현재 최종 범위는 front 모델로 한정한다. Rear 확장은 수행하지 않는다.

## 4. 정적 분석 탈락 기준

### 공통

- checkpoint 로드 또는 smoke inference 실패
- ONNX export/checker 실패
- TensorRT parser/build 실패
- baseline과 입력·출력 interface가 달라 비교 불가
- NaN/Inf 발생
- 생성 조건이나 source checkpoint를 재현할 수 없음

### 방식별 예외

- FP16/INT8은 parameter 감소가 없어도 탈락시키지 않는다.
- Unstructured pruning은 FLOPs 감소가 없어도 대조군으로 유지한다.
- 2:4는 sparsity 50%만으로 통과시키지 않고 pattern과 sparse tactic을 본다.
- Structured pruning은 실제 parameter·FLOPs·graph 감소가 있어야 한다.

정적 분석 통과는 우수 모델이라는 뜻이 아니라, 평가자가 실제 성능을
측정할 가치가 있다는 뜻이다.

## 5. 모델별 필수 산출물

```text
<experiment-id>/
├── model.pth
├── model.onnx
├── model.engine
├── metadata.json
├── build-command.txt
├── build.log
└── sparsity.json
```

`metadata.json`에는 다음을 기록한다.

- experiment ID와 front/rear
- source checkpoint와 checksum
- 경량화 방식과 비율
- precision과 precision fallback
- fine-tuning 여부와 epoch
- input shape, batch size, class
- preprocessing과 postprocessing
- ONNX opset
- CUDA, TensorRT, GPU 정보
- sparse tactic 활성화 여부
- PTH/ONNX/engine 크기
- parameter, FLOPs, 전체·layer별 sparsity
- 생성 날짜와 Git commit
- 생성·변환 성공 여부 및 실패 원인

## 6. 평가자가 동일하게 맞출 조건

- test dataset과 split
- 입력 해상도와 batch size
- confidence threshold
- preprocessing과 postprocessing
- warmup 및 반복 횟수
- TensorRT/CUDA 버전
- GPU power mode
- CUDA synchronization 방식

평가자는 latency, FPS, GPU memory, detection accuracy, segmentation accuracy를
채우고 모델 생성자는 정적 지표와 변환 조건을 제공한다.

## 7. 현재 바로 할 다음 작업

1. 최종 front engine 패키지를 평가자에게 전달한다.
2. 필요하면 FLOPs 감소를 operator-aware 방식으로 보완한다.

## 8. Artifact 무결성 검사

다음 명령은 registry에 등록된 전체 실험을 대상으로 필수 artifact 존재 여부,
metadata 필수 필드와 참조 경로, 원본 및 artifact checkpoint SHA-256,
실제 파일 inventory hash를 검사한다.

```bash
.venv/bin/python scripts/experiments/audit_artifacts.py
```

결과는 `artifacts/experiments/audit-report.json`에 저장된다. 2026-07-01
기준 18개 camera 실험을 검사했으며 오류는 0건이다.
