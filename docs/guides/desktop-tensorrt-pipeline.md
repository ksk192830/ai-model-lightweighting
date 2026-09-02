# 데스크탑 TensorRT 자동 파이프라인

## 목적

로컬 Linux NVIDIA 데스크탑에서 복구 학습 후보를 검증한 뒤 TensorRT
engine 생성, 속도 측정, COCO bbox/mask AP와 semantic mIoU 평가를
자동으로 실행한다.

## 고정 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti 8 GB |
| Compute capability | 12.0 |
| TensorRT | 10.16.1.11 |
| PyTorch | 2.10.0+cu128 |
| 입력 | batch 1, 기본 504×504, R01 432×432 |
| 속도 측정 | test에서 seed 42로 고정한 32장, warm-up 20회, 본 측정 200회 |
| 정확도 평가 | 고정 benchmark 437장 |
| INT8 calibration | train 3,625장 중 seed 42로 선정한 128장 |

## 실행

```bash
cd "/home/scope/ai 경량화 및 최적화 논문"
.venv/bin/python scripts/experiments/run_desktop_tensorrt_queue.py \
  --poll-interval 30 --warmup 20 --runs 200
```

큐는 S02와 M01의 `post-recovery-queue.json` 상태가 모두 `completed`가
될 때까지 기다린다. 따라서 recovery 전 prototype ONNX나 학습 중
checkpoint로 engine을 만들지 않으며, 학습과 TensorRT build가 동시에
GPU 메모리를 사용하지 않는다.

## 실행 순서

1. S02·M01 복구 후 PTH, ONNX, 동등성, PTH 전체 평가 완료 확인
2. B01, B02, B03, R01, S01, S02, M01, M02 engine 순차 생성
3. S01과 S02의 FP32 TensorRT 지연시간 측정
4. 정확도 보존과 속도 개선 기준으로 structured 후보 선정
5. 선정 structured ONNX로 C01 FP16 engine 생성
6. 모든 engine의 레이어 구조·I/O 형상·FP32/FP16/INT8 정밀도 근거 정적 분석
7. 모든 생성 engine 속도 측정과 437장 전체 평가
8. JSON/CSV 종합 결과, 논문용 그래프, 정적 분석 보고서, 평가 감사 자동 생성

S02는 bbox AP·mask AP가 S01에서 각각 0.005 이내, semantic mIoU가
0.01 이내이면서 TensorRT FP32 중앙 지연시간이 5% 이상 개선될 때만
선정한다. 조건을 만족하지 못하면 S01을 유지한다.
이 임계값을 포함한 모든 평가 기준은
`configs/experiments/defaults.yaml` 한 곳에서 정의하고 파이프라인이
직접 읽는다.

속도는 이미 메모리에 로드한 RGB 이미지를 입력으로 한 end-to-end
latency다. Resize/normalize, host-to-device 복사, TensorRT 실행,
postprocess, device-to-host 복사를 포함하지만 디스크 이미지 decode는
포함하지 않는다. 결과에는 mean, median, standard deviation, IQR,
P95, P99와 coefficient of variation을 함께 저장한다.

## 결과 위치

```text
results/desktop-tensorrt-pipeline.json  # 전체 상태와 오류
results/desktop-tensorrt-queue.log      # 실시간 진행 로그
results/desktop-pipeline-logs/          # 단계별 로그
artifacts/experiments/<ID>/front/engine-static-analysis.json
results/desktop-structured-selection.json
results/desktop-engine-summary.json
results/desktop-engine-summary.csv
results/benchmarks/desktop/<ID>/
results/coco-evaluation/<ID>-front.json
results/results_final.csv
results/paper-static-analysis.csv
docs/reports/paper-static-analysis.md
results/evaluation-automation-audit.json
docs/reports/evaluation-automation-audit.md
```

개별 engine 실패는 다른 후보 실행을 중단하지 않고 상태 JSON과
단계별 로그에 기록한다.
