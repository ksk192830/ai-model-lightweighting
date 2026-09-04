# 노트북 Stage 2 TensorRT 최종 평가

## 결론

- Stage 1 통과 후보 22개가 모두 terminal 상태에 도달했다. 21개는 평가를
  완료했고 Q03은 TensorRT INT4 parser 오류로 엔진 생성에 실패했다.
- 성공 후보 21개의 지연시간을 AC 전원 및 `performance` 정책에서 처음부터
  3회씩, 총 63회 다시 측정했다. 실행 ID는 `20260904_104755`이다.
- 63개 측정 전 부하 확인이 모두 첫 시도에 통과했다. 반복 median CV가 5%를
  넘은 후보가 없어 추가 재측정 라운드는 발생하지 않았다.
- Mask AP 최대화, median latency와 engine 크기 최소화의 3축 Pareto 후보는
  **C01과 R01**이다. 두 후보 모두 median 33.33 ms 이하의 30 FPS 배포 기준을
  만족한다.
- C01은 정확도와 크기의 균형이 가장 좋고, R01은 정확도 gate를 통과한 후보 중
  가장 빠르다.

## 평가 환경과 프로토콜

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB |
| Compute capability | 8.9 |
| CUDA / TensorRT | 12.8 / 10.16.1.11 |
| NVIDIA driver | 580.173.02 |
| CPU | AMD Ryzen 7 7735HS, 8 cores / 16 threads |
| RAM / OS | 15.97 GB / Ubuntu 22.04, kernel 6.8.0-136-generic |
| 전원 조건 | AC 연결, platform profile `performance`, AMD P-State EPP `performance` |
| 정확도 데이터 | train/calibration과 분리된 고정 비교용 test 437장, annotation 593개 |
| 지연시간 | batch 1, 32장 고정 표본, warmup 20회, 측정 200회 × 3반복 |
| 측정 범위 | 메모리 내 이미지 입력부터 후처리까지, 디스크 decode 제외 |

첫 측정 전에 30초 동안 화면과 시스템 부하가 정착하도록 기다렸다. 각 반복 전
1초 간격 3개 표본에서 CPU 30%, GPU 50%, GPU memory-controller 50%, GPU
온도 75°C 이하인지 확인했다. 세 표본의 CPU 범위는 20%p, GPU 및 memory-controller
범위는 50%p, 온도 범위는 6°C 이하여야 한다. 첫 안정 구간과 후속 구간의 평균
차이도 CPU 20%p, GPU 및 memory-controller 50%p, 온도 15°C 이내로 제한했다.

실제 63개 승인 구간 평균은 CPU 2.17~5.43%, GPU 4.33~25.67%, GPU
memory-controller 1.00~18.67%, GPU 온도 40~55°C였다. 모든 구간이 첫
시도에 통과했고 timeout 또는 강제 측정은 없었다. 자세한 기준은
[측정 부하 통제](measurement-load-control.md)에 기록했다.

## 후보별 최종 결과

| 후보 | Precision | BBox AP | Mask AP | mIoU | Median(ms) | P95(ms) | FPS | CV | Engine(MiB) | 정확도 gate | 30 FPS | Pareto |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| B01 | fp32 | 0.7379 | 0.6026 | 0.7120 | 45.820 | 48.562 | 21.73 | 0.12% | 129.3 | PASS |  |  |
| B02 | fp16 | 0.7416 | 0.6017 | 0.7138 | 25.775 | 28.130 | 38.40 | 2.88% | 67.9 | PASS | ✓ |  |
| B03 | int8 | 0.7422 | 0.6027 | 0.7124 | 25.360 | 27.903 | 38.92 | 2.26% | 68.0 | PASS | ✓ |  |
| M01 | fp16 | 0.6346 | 0.5180 | 0.6826 | 25.387 | 27.630 | 39.00 | 0.96% | 67.9 | FAIL | ✓ |  |
| M02 | fp16 | 0.6324 | 0.5148 | 0.6751 | 25.262 | 28.146 | 39.02 | 0.57% | 62.4 | FAIL | ✓ |  |
| S01 | fp32 | 0.7390 | 0.6022 | 0.7339 | 44.510 | 46.954 | 22.38 | 1.16% | 123.3 | PASS |  |  |
| S02 | fp32 | 0.7161 | 0.5850 | 0.7531 | 42.957 | 45.106 | 23.20 | 0.56% | 117.2 | FAIL |  |  |
| S04 | fp32 | 0.5181 | 0.5214 | 0.6476 | 45.769 | 48.077 | 21.79 | 0.58% | 121.2 | FAIL |  |  |
| C01 | fp16 | 0.7386 | 0.6029 | 0.7367 | 25.300 | 27.512 | 39.13 | 2.91% | 64.7 | PASS | ✓ | ✓ |
| C02 | int8 | 0.7401 | 0.6023 | 0.7412 | 26.183 | 27.425 | 38.46 | 0.85% | 64.8 | PASS | ✓ |  |
| C03 | fp32 | 0.7319 | 0.5940 | 0.7353 | 36.296 | 37.834 | 27.62 | 0.49% | 122.6 | PASS |  |  |
| C04 | fp32 | 0.7310 | 0.5996 | 0.7282 | 41.973 | 43.176 | 23.95 | 0.26% | 123.0 | PASS |  |  |
| R01 | fp16 | 0.7288 | 0.5941 | 0.7034 | 24.010 | 25.821 | 41.48 | 3.82% | 68.1 | PASS | ✓ | ✓ |
| R02 | fp32 | 0.7330 | 0.5997 | 0.7108 | 42.963 | 44.721 | 23.34 | 1.26% | 129.1 | PASS |  |  |
| R03 | fp32 | 0.7265 | 0.5865 | 0.7281 | 32.386 | 34.875 | 30.72 | 1.85% | 128.2 | FAIL | ✓ |  |
| Q01 | int8 | 0.5932 | 0.4743 | 0.6591 | 25.114 | 27.849 | 39.34 | 1.09% | 47.8 | FAIL | ✓ |  |
| Q02 | mixed | 0.5965 | 0.4832 | 0.6504 | 25.183 | 27.597 | 39.31 | 1.32% | 43.4 | FAIL | ✓ |  |
| Q04 | fp8 | 0.0036 | 0.3702 | 0.4531 | 37.994 | 40.587 | 26.10 | 0.58% | 48.1 | FAIL |  |  |
| Q05 | mixed | 0.7358 | 0.5981 | 0.7102 | 43.333 | 45.668 | 22.97 | 1.06% | 111.8 | PASS |  |  |
| Q06 | mixed | 0.0041 | 0.3541 | 0.4566 | 39.409 | 41.529 | 25.22 | 0.23% | 64.8 | FAIL |  |  |
| Q07 | mixed | 0.0282 | 0.3579 | 0.4289 | 38.299 | 40.826 | 25.91 | 0.89% | 53.0 | FAIL |  |  |

30 FPS 표시는 median latency 기준이다. P95가 33.33 ms를 넘은 12개 후보에는
진단 경고가 기록됐지만 정확도 gate나 Pareto 판정을 바꾸지 않는다.

Q03에는 수치가 없다. TensorRT 10.16.1.11 parser가 block size 128인 INT4
weight-only 그래프의 `DequantizeLinear` 입력을 Float로 판정해 `INVALID_NODE`를
반환했다. 최신 ONNX SHA-256 검증 이후 발생한 플랫폼 호환성 실패다.

## 선택 해석

| 목적 | 후보 | 근거 |
|---|---|---|
| 정확도·크기 균형 | C01 | B01 대비 Mask AP +0.00027, mIoU +0.02468, median 44.79% 감소, engine 49.95% 감소 |
| 최저 지연시간 | R01 | 정확도 gate 통과 후보 중 최저 24.010 ms, B01 대비 1.91배, CV 3.82% |
| 최고 BBox AP | B03 | BBox AP 0.7422, median 25.360 ms, engine 68.0 MiB |
| 최고 mIoU | C02 | mIoU 0.7412, median 26.183 ms, engine 64.8 MiB |

C01은 R01보다 1.29 ms 느리지만 Mask AP가 0.00880 높고 engine이 약 3.3 MiB
작다. R01은 더 빠르지만 품질과 크기에서 C01보다 열세이므로 두 후보가 서로를
지배하지 않는다. 다른 정확도 gate 통과 후보는 이 세 주 목적에서 C01 또는
R01에 지배된다.

## 남은 작업

1. 이 결과와 원시 측정 근거를 데스크톱 저장소에 동기화한다.
2. 논문 표·그림과 결과 장을 새 Pareto 후보 C01·R01 및 최종 수치로 갱신한다.
3. 22개 후보 모두의 값이 필요할 때만 Q03을 TensorRT가 지원하는 INT4 형식으로
   다시 export해 Q03만 평가한다.
4. 외적 일반화 주장이 필요하면 Pareto 선택 이후 별도 촬영 세션 holdout으로 한 번
   검증한다. 현재 437장은 고정 반복 비교용 benchmark로 해석한다.

정확도 평가와 엔진 생성까지 포함한 전체 Stage 2를 다시 실행할 필요는 없다.

## 근거 파일

- [통합 결과 JSON](../../results/stage2-notebook-summary.json)
- [통합 결과 CSV](../../results/stage2-notebook-summary.csv)
- [Stage 2 상태](../../results/stage2-notebook-state.json)
- [Stage 3 Pareto JSON](../../results/stage3-pareto.json)
- [Stage 3 Pareto CSV](../../results/stage3-pareto.csv)
- [Excel 평가 보고서](../../results/stage2-evaluation-report.xlsx)
- [측정 부하 통제 기록](measurement-load-control.md)

정적 MAC/FLOP는 Conv·MatMul·Gemm 중 shape를 해석한 연산만 포함하는 dense
하한 추정치다. TensorRT fusion, 메모리 이동, 전처리와 후처리 비용을 포함하지
않으므로 이 문서의 속도 비교는 TensorRT engine 실측값만 사용한다.
