# RF-DETR 주차 인식 모델 경량화 논문 — 한국어 개조식 구성안 (Stage 3)

> **문서 역할:** Stage 3 공식 실행 결과까지 반영한 논문 전체 개조식 기준서.
> 수치와 판정은 GitHub의 공식 실행 ID `20260904_104755`에서만 가져오며,
> 저장소에 없는 정보는 추정하지 않는다.

## 작성 원칙과 최종 상태

- 가제: **노트북 GPU 배포를 위한 RF-DETR 기반 주차 인식 모델의 경량화 및 다목적 평가**
- 연구 대상: 전방 카메라 주차 인식용 RF-DETR Segmentation Large
- 연구 성격: 새 detector 구조 제안이 아니라 동일 baseline 파생 후보의 배포 trade-off 비교
- 데이터: Roboflow `parking_front` Version 9, 총 4,466장
- 분할: train 3,625장 / validation 404장 / fixed comparative benchmark 437장
- Baseline 학습: 18 epoch 조기 종료, seed 42 단일 실행
- Stage 1: 26개 전 후보 완료, 22개 통과, 4개 불통
- Stage 2: 22개 terminal, 21개 engine·정확도·성능 측정 완료, Q03 build 실패
- 정확도 gate: B01 포함 11개 통과
- Stage 3: C01·R01 두 후보가 Mask AP–median latency–engine size 3축 Pareto 비지배해
- 최종 추천: 균형·정확도·크기 우선 C01, 속도 우선 R01
- 단일 가중합으로 절대 우승자를 만들지 않고 배포 목적별 권장안을 제시
- 공식 근거: [Stage 3 최종 분석](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/reports/stage3-final-analysis.md)
- 근거 연결: [Stage 3 논문 근거 연결표](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/paper/stage3-evidence-map.md)

## Abstract 작성 항목

- 배경
  - 제한된 노트북 GPU에서 주차 인식 모델의 정확도와 실행 비용을 함께 고려할 필요
  - 정적 parameter·ONNX 크기·dense MAC 감소만으로 실제 TensorRT 성능을 설명하기 어려움
- 방법
  - 공통 RF-DETR Segmentation Large checkpoint에서 26개 후보 구성
  - precision, 비정형·구조적 pruning, 2:4 sparsity, 해상도, Q/DQ 양자화와 결합 후보 포함
  - Stage 1 정적 선별 → Stage 2 목표 GPU 실측 → 정확도 gate → Stage 3 Pareto 분석
- 결과
  - Stage 1에서 22개 통과, U01·U02·U03·S03 불통
  - Stage 2에서 21개 완료, Q03은 TensorRT INT4 parser 오류로 build 실패
  - 정확도 gate는 B01 포함 11개 통과
  - C01: Mask AP 0.6029, median 25.300 ms, P95 27.512 ms, engine 64.7 MiB
  - R01: Mask AP 0.5941, median 24.010 ms, P95 25.821 ms, engine 68.1 MiB
  - B01 대비 C01은 median latency 44.79%, engine 49.95% 감소
  - B01 대비 R01은 median latency 47.60%, engine 47.38% 감소
- 결론
  - C01은 균형·정확도·크기 우선, R01은 속도 우선 배포 후보
  - 두 후보 모두 median 기준 30 FPS frame budget 충족
- Keywords
  - RF-DETR; TensorRT; Model Lightweighting; Autonomous Parking; Instance Segmentation

## 1. Introduction

### 1.1 연구 배경

- ROS2 기반 주차 시스템에서 perception 결과가 경로 계획과 제어의 입력으로 사용됨
- 전방 카메라에서 주차 구획과 경계의 위치·영역을 함께 인식해야 하므로 detection과 segmentation 동시 평가 필요
- 실제 배포에서는 정확도뿐 아니라 latency, 처리량, GPU memory와 engine 크기가 중요
- 전력·열 조건은 성능에 영향을 주므로 공식 실행에서 AC 전원과 performance profile을 고정

### 1.2 문제 정의와 연구 공백

- 정적 parameter 수, checkpoint 크기와 dense MAC/FLOP 감소가 실제 GPU latency 감소를 보장하지 않음
- TensorRT 성능은 graph fusion, precision fallback, kernel/tactic 선택과 memory 이동에 의존
- 여러 경량화 축을 동일 데이터·동일 GPU·동일 평가기로 비교할 필요
- 단일 최고 점수보다 정확도·latency·engine size의 다목적 trade-off 분석 필요

### 1.3 연구 질문

- RQ1: decoder layer, FFN dimension과 입력 해상도 변화가 정적 복잡도와 detection·segmentation 품질에 미치는 영향은 무엇인가?
- RQ2: FP16, INT8, INT4, FP8과 mixed precision이 정확도·latency·engine size에 만드는 trade-off는 무엇인가?
- RQ3: 비정형 sparsity와 하드웨어 지원 2:4 sparsity는 정적 희소성 및 실제 TensorRT 결과에서 어떤 차이를 보이는가?
- RQ4: 구조·해상도·정밀도 결합 후보가 단일 경량화 후보보다 우수한 Pareto 지점을 제공하는가?

### 1.4 주요 기여

- 동일 baseline에서 파생한 26개 후보의 재현 가능한 registry와 artifact 추적 체계
- 정확도를 배제한 정적 선별, 목표 GPU 실측, 정확도 gate와 Pareto 분석으로 이어지는 3단계 평가 절차
- bbox AP·mask AP·semantic mIoU와 median·P95 latency·engine size를 함께 기록하는 다목적 평가
- build 실패·미지원 precision도 누락하지 않고 terminal 결과로 남기는 자동화
- 환경·모델·결과 fingerprint와 원시 로그를 연결한 재현성 확보

### 1.5 연구 범위

- Front camera 모델과 단일 RTX 4050 Laptop GPU만 대상
- ROS2 전체 end-to-end latency와 실제 주차 성공률은 범위 밖
- 새로운 주차장·카메라·날씨에 대한 외적 일반화는 주장하지 않음
- 437장은 후보 개발에서 반복 사용된 비교용 benchmark이며 untouched confirmatory test로 표현하지 않음

## 2. Related Work

### 2.1 DETR 계열과 RF-DETR

- DETR: set prediction, object query, bipartite matching 기반 end-to-end detection
- DINO: denoising query 학습과 초기화 개선
- Mask DINO: detection과 segmentation 통합
- RF-DETR: 본 연구의 기존 주차 인식 baseline
- 다른 detector 계열에 대한 보편적 우월성은 주장하지 않음

### 2.2 Pruning과 희소 실행

- 비정형 pruning: zero sparsity와 dense graph 감소를 구분
- Structured pruning: decoder layer·FFN dimension 축소와 recovery fine-tuning
- NVIDIA 2:4 sparsity: 패턴 준수와 TensorRT sparse tactic 선택을 별도 검증

### 2.3 저정밀화와 해상도 조정

- FP16·INT8·INT4·FP8 및 mixed precision 후보 비교
- Q/DQ node 존재는 양자화 표현의 정적 증거이며 실제 저정밀 kernel 선택의 충분조건은 아님
- SmoothQuant·AWQ의 원리는 RF-DETR에서 보장되지 않으므로 실험 후보로만 사용
- 해상도 축소는 연산량을 줄이나 작은 객체와 얇은 mask 경계 손실 가능

### 2.4 다목적 평가

- 정확도 gate 후 Mask AP 최대화, median latency·engine size 최소화
- BBox AP, semantic mIoU, P95 latency와 GPU memory는 보조 해석 지표
- 임의 가중합 대신 Pareto 비지배 관계와 배포 시나리오별 추천 사용

## 3. Data Construction and Split Protocol

### 3.1 데이터 구성

- Roboflow `parking_front` Version 8 annotation 계승
- Offline preprocessing·augmentation이 없는 Version 9 생성
- 총 4,466장, COCO bbox·polygon/mask annotation
- 평가 클래스: out_line, parking_lot, parking_space
- Category 0 `front`는 placeholder, 평가 instance는 category 1–3

### 3.2 세션 기반 분할

- Timestamp·frame 번호 복원 가능 이미지 1,461장
- 세션 경계: 시간 차 2초 초과 또는 시각 변화와 frame 번호 감소
- 내부 최대 간격 1.411초, 관측 세션 경계 최소 간격 5.010초
- Timestamp 미복원 3,005장은 잘못된 세션 추정을 피하기 위해 train에만 고정
- 가능한 30개 주요 세션 배정을 전수 조사하여 비율·클래스 분포 편차 최소화

### 3.3 최종 split과 감사

- Train 3,625장(81.17%): 학습·recovery·calibration 모집단
- Validation 404장(9.05%): early stopping·checkpoint 및 민감 block 선택
- Fixed benchmark 437장(9.79%): 후보 간 반복 비교
- Split 간 동일 이미지 0, 세션 중복 0, COCO 참조 오류 0
- 노트북·데스크톱의 의미적 annotation fingerprint와 이미지 collection fingerprint 일치
- Raw annotation byte hash 차이는 JSON 직렬화 순서 차이로 확인

### 3.4 근거 부족 항목

- `[GitHub 근거 없음]` 카메라 모델, 원본 해상도, 프레임 추출 간격, 장소·조명·날씨, 데이터 사용 권한
- `[GitHub 근거 없음]` 클래스 정의, polygon 경계·가림·잘림 처리, 라벨링 도구와 검수 이력
- `[GitHub 근거 없음]` split별 instance 수, 객체 크기와 mask 면적 분포

## 4. Deployment-Oriented Lightweighting Pipeline

### 4.1 Baseline 학습

- RF-DETR Segmentation Large 1.8.1, 입력 504×504
- Epoch 상한 25, 실제 18 epoch 조기 종료
- Micro-batch 2, gradient accumulation 8, effective batch 16
- LR 1.0e-4, encoder LR 1.5e-4, weight decay 1.0e-4
- EMA 0.993, seed 42, patience 6, min delta 0.001
- Train으로 학습하고 validation으로 checkpoint 선택; 학습 중 benchmark 미사용

### 4.2 후보군

- B01–B03: FP32 baseline, FP16, INT8 PTQ
- U01–U03: global magnitude pruning 10/30/50%
- M01/M02: 2:4 dense control / sparse tactic
- S01–S04: decoder layer 또는 FFN structured pruning
- C01–C04: S01 구조와 precision 또는 resolution 결합
- R01–R03: 입력 432/480/384
- Q01–Q07: INT8·INT4·FP8·mixed precision ModelOpt
- 총 26개, W 계열은 최종 연구 범위에서 제거됨

### 4.3 단계별 판정

- Stage 1
  - 모든 후보의 artifact 유효성과 경량화 적용 증거 검사
  - 구조·해상도 후보는 ONNX 크기·node·dense MAC 중 하나 이상 5% 감소 필요
  - Q 후보는 Q/DQ node와 quantization report 일치 필요
  - M 후보는 2:4 pattern과 sparse tactic recipe 필요
  - 정확도·latency·FPS·GPU memory는 사용하지 않음
- Stage 2
  - Stage 1 통과 22개 모두 목표 GPU에서 engine build·inspection·정확도·성능 측정
  - 성공과 실패를 모두 terminal 결과로 기록
- Stage 3
  - 유효 측정과 정확도 gate를 통과한 후보만 Pareto 입력
  - Mask AP 최대화, median latency·engine size 최소화

## 5. Experimental Setup and Evaluation Protocol

### 5.1 공식 실행 환경

- Run ID: `20260904_104755`
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, VRAM 6,141 MiB
- CUDA 12.8, TensorRT 10.16.1.11, driver 580.173.02
- CPU: AMD Ryzen 7 7735HS, 8 cores / 16 threads
- RAM: 15.97 GB
- OS: Ubuntu 22.04, kernel 6.8.0-136-generic
- 조건: AC 전원, platform profile `performance`, AMD P-State EPP `performance`
- Engine은 이 환경에서 새로 build

### 5.2 정확도 평가

- 고정 benchmark 437장
- COCO bbox/mask AP@[0.50:0.05:0.95], AP50, AP75, 크기별 AP, AR100
- 클래스별 bbox/mask AP, semantic mIoU와 클래스별 IoU
- AP confidence floor 0.001, semantic confidence 0.25, max detections 100
- 공통 gate: B01 대비 bbox AP·mask AP 하락 각각 0.01 이하, mIoU 하락 0.02 이하
- 세 조건을 모두 만족해야 통과
- Engineering gate이며 통계적 유의성 기준이 아님

### 5.3 실행 성능과 안정성

- Batch 1, seed 42, 고정 32장
- 후보마다 warm-up 20회 후 200회 측정, 총 3반복
- Disk decode 제외 in-memory image end-to-end 범위
- 보고: pooled median·mean·P95·P99·IQR·FPS, 반복 median CV와 95% t 구간
- CV 5% 초과 시 동일 엔진 전체 재측정; 두 번째 결과도 초과하면 최종 권장에서 제외
- 공식 실행 21개 × 3반복 = 63개 모두 부하 gate 통과
- 재측정 후보 0, 최대 CV R01 3.8249%
- 30 FPS 기준: median latency 33.33 ms 이하
- P95 33.33 ms 초과는 경고이며 hard gate나 Pareto 제외 조건이 아님
- Median 기준 판정은 지속 처리량이나 모든 프레임의 30 FPS를 보장하지 않음

### 5.4 자동화와 재현성

- Checksum, 환경 점검, engine build, inspection, benchmark, accuracy, gate, Pareto, Excel 생성 자동화
- 중단 후 완료 후보 재사용
- 결과는 JSON·CSV·Markdown·Excel·후보별 로그로 저장
- 자동화 감사: 필수 실패 0, 미완료 필수 항목 0, advisory 9
- Advisory는 legacy PTH annotation hash, legacy graph schema, 반복 비교 benchmark와 single seed 제한

## 6. Results and Discussion

### 6.1 Stage 1 정적 결과

- 26/26 완료, 22 통과, 4 불통
- U01·U02·U03: zero sparsity는 증가했지만 dense ONNX 크기·node·MAC 5% 감소 기준 미충족
- S03: ONNX 3.27%, dense MAC 0.39%, node 0% 감소로 기준 미달
- S01/S02 node 8.04%/16.17% 감소
- S04 ONNX 크기 6.53% 감소
- C03/C04 dense MAC 30.76%/11.51% 감소
- R01/R02/R03 dense MAC 30.20%/10.96%/46.25% 감소
- M01/M02 적격 연산 150/150의 2:4 pattern 준수
- Q01–Q07 ONNX checker 통과, Q/DQ node 34–626개 확인
- 정적 MAC/FLOP는 해석 가능한 dense 연산 하한이며 실제 속도 증거가 아님

### 6.2 Stage 2 결과

- Stage 2 대상 22개 모두 terminal
- 21개 engine build·정확도·지연시간 측정 완료
- Q03: TensorRT 10.16.1.11에서 INT4 block quantization DequantizeLinear parser 오류
- Q03에는 성능 수치가 없으며 Pareto 입력에서 제외
- 이 실패는 현재 export 형식과 parser 조합의 호환성 결과이며 INT4 전체 실패로 일반화하지 않음
- 정확도 gate 통과 11개
  - B01, B02, B03, S01, C01, C02, C03, C04, R01, R02, Q05
- 정확도 gate 실패 10개
  - M01, M02, S02, S04, R03, Q01, Q02, Q04, Q06, Q07
- B03: gate 통과 후보 중 최고 BBox AP 0.7422
- C02: gate 통과 후보 중 최고 semantic mIoU 0.7412
- S02는 mIoU 0.7531이지만 bbox·mask AP 하락 때문에 정확도 gate 실패

### 6.3 Pareto와 최종 후보

| 후보 | Mask AP | Median ms | P95 ms | Engine MiB | B01 대비 latency | B01 대비 engine |
|---|---:|---:|---:|---:|---:|---:|
| B01 | 0.6026 | 45.820 | 48.562 | 129.3 | 기준 | 기준 |
| C01 | 0.6029 | 25.300 | 27.512 | 64.7 | 44.79% 감소 | 49.95% 감소 |
| R01 | 0.5941 | 24.010 | 25.821 | 68.1 | 47.60% 감소 | 47.38% 감소 |

- C01과 R01만 세 주 목적의 Pareto 비지배해
- C01은 R01보다 1.289 ms 느리지만 Mask AP가 0.00880 높고 engine이 약 3.3 MiB 작음
- 두 후보 모두 정확도 gate 및 median 30 FPS 기준 통과, P95 경고 없음
- 균형형: C01
- 정확도 우선: C01
- Engine 크기 우선: C01
- Median latency 우선: R01
- B03과 C02는 각각 BBox AP·mIoU 보조 비교 후보이나 C01에 지배되어 최종 Pareto 추천에는 미포함

### 6.4 전체 후보 판정 요약

| 구분 | 후보 |
|---|---|
| Stage 1 불통 | U01, U02, U03, S03 |
| Stage 2 build 실패 | Q03 |
| 정확도 gate 실패 | M01, M02, S02, S04, R03, Q01, Q02, Q04, Q06, Q07 |
| 정확도 통과 후 Pareto 지배됨 | B01, B02, B03, S01, C02, C03, C04, R02, Q05 |
| 최종 Pareto | C01, R01 |

### 6.5 타당성의 위협

- Single seed 학습으로 학습 분산 미추정
- Validation·benchmark가 각각 제한된 촬영 세션으로 구성
- Benchmark를 후보 개발에서 반복 사용
- 단일 RTX 4050 Laptop GPU와 단일 TensorRT 버전 결과
- Engine-only/in-memory 평가이며 ROS2 전체 pipeline latency와 다름
- 외부 장소·날씨·카메라 holdout 없음
- GPU memory 차이가 약 890–898 MiB 범위로 작아 후보 구분력이 제한적

## 7. Conclusion

- 세션 기반 분할과 누수 감사 완료
- 26개 후보의 Stage 1 전수 평가 및 22개 Stage 2 진입
- 21개 TensorRT engine 실측, Q03 호환성 실패 기록
- B01 포함 11개 정확도 보존 후보 확인
- C01·R01이 3축 Pareto 비지배해
- C01은 B01 수준 Mask AP를 유지하면서 latency·engine을 약 45%·50% 줄인 균형형 후보
- R01은 정확도 gate를 만족하면서 최저 median 24.010 ms를 기록한 속도 우선 후보
- 실제 배포 목적에 따라 두 후보 중 선택해야 하며 단일 절대 우승자는 선언하지 않음

## Acknowledgement

- `[GitHub 근거 없음]` 지원기관, 과제번호와 공식 국·영문 사사 문구 필요

## 참고문헌 및 제출 전 보완

- GitHub README의 12개 기술 근거를 KSAE 양식으로 검증하고 최초 인용 순서로 재배열
- `[GitHub 근거 없음]` 기존 ASK 2026 논문의 제목·저자·학회·페이지·DOI
- 데이터 provenance와 annotation guideline 확보
- 필요 시 별도 장소의 잠금 holdout, ROS2 end-to-end latency, 다른 GPU와 다중 seed 실험 추가

## 공식 근거 목록

- [Stage 3 최종 분석](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/reports/stage3-final-analysis.md)
- [후보별 Stage 2 결과](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/reports/notebook-stage2-results.md)
- [전체 후보 판정 CSV](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/results/stage3-paper-candidate-decisions.csv)
- [최종 추천 JSON](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/results/stage3-final-recommendations.json)
- [Stage 3 Pareto JSON](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/results/stage3-pareto.json)
- [데이터 fingerprint 비교](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/reports/metrics/dataset-fingerprint-comparison.json)
- [측정 부하 통제](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/reports/measurement-load-control.md)
- [최종 engine 검증](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/reports/metrics/notebook-final-engine-verification.json)

