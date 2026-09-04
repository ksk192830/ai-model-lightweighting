# RF-DETR 주차 인식 모델 경량화 논문 — 한국어 개조식 구성안

> **문서 역할:** 논문 전체의 주장, 수치, 표·그림, 근거와 미확정 항목을 개조식으로
> 관리하는 작성 기준서. 확정되지 않은 결과를 추정하지 않는다.
>
> **스냅샷 상태:** Stage 2 실행 전 작성본이다. 최신 확정 결과는
> [Stage 3 최종 분석](../reports/stage3-final-analysis.md)과
> [근거 연결표](stage3-evidence-map.md)에 준비했으며 다음 원고 갱신에서 본문에 반영한다.

## 작성 원칙과 현재 상태

- 가제: **노트북 GPU 배포를 위한 RF-DETR 기반 주차 인식 모델의 경량화 및 다목적 평가**
- 연구 범위: 전방 카메라 주차 인식, RF-DETR Segmentation Large, ONNX와 TensorRT 중심 배포 최적화
- 새로운 detector 구조를 제안하는 연구가 아니라 동일 baseline에서 파생한 경량화 후보의 배포 trade-off 연구
- 정적 지표를 실제 속도로 해석하지 않고 목표 GPU의 TensorRT engine 실측으로 최종 판단
- 데이터셋: Roboflow `parking_front` Version 9, 총 4,466장
- 분할: train 3,625장 / validation 404장 / 고정 비교 benchmark 437장
- 기준 모델 학습: 완료, 18 epoch에서 조기 종료
- 기준 모델 정확도: bbox AP 0.737791 / mask AP 0.601016 / semantic mIoU 0.712200
- Stage-1 정적평가: 26/26 완료, 22 통과, 4 불통, 미수행 0
- Stage-2 TensorRT 평가: 미수행
- Stage-3 Pareto 분석: 미수행
- 최종 우수 후보·latency·FPS·GPU memory·engine size: **아직 단정 금지**
- GitHub 기준 상태: [PROJECT_STATUS.md](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/PROJECT_STATUS.md)

## Abstract 작성 항목

- 배경
  - 제한된 노트북 GPU에서 주차 인식 모델의 정확도와 실행 비용을 함께 고려할 필요
  - parameter·checkpoint 크기만으로 실제 TensorRT 성능을 설명하기 어려움
- 방법
  - 공통 RF-DETR Segmentation Large checkpoint에서 26개 후보 구성
  - precision, unstructured pruning, structured pruning, 2:4 sparsity, resolution, mixed precision 및 결합 후보 포함
  - Stage-1 정적 선별 → Stage-2 TensorRT 실측 → Stage-3 Pareto 분석
- 확정 결과
  - 데이터 분할 감사 통과
  - Stage-1에서 26개 중 22개 통과, U01·U02·U03·S03 불통
- 추후 삽입
  - `[Stage-2 후 삽입]` baseline 대비 대표 latency·FPS·memory·engine size 변화
  - `[Stage-2 후 삽입]` 정확도 보존 gate 통과 후보 수
  - `[Stage-3 후 삽입]` Pareto 후보와 시나리오별 권장 후보
- Keywords
  - RF-DETR; TensorRT; Model Lightweighting; Autonomous Parking; Instance Segmentation

## 1. Introduction

### 1.1 연구 배경

- ROS2 기반 주차 시스템에서 perception 결과가 후속 경로 계획·제어의 입력으로 사용됨
- 전방 카메라에서 주차 구획·경계의 위치와 영역을 함께 인식해야 하므로 detection과 segmentation을 동시에 고려
- 실제 배포 시 정확도뿐 아니라 latency, 처리량, GPU memory와 배포 artifact 크기가 중요
- 전력·발열은 배경 제약으로만 언급하고, 직접 측정하지 않는 한 연구 결과로 주장하지 않음

### 1.2 문제 정의와 연구 공백

- 정적 parameter 수, checkpoint 크기, dense MAC/FLOP 감소가 실제 GPU latency 감소를 보장하지 않음
- TensorRT 성능은 graph fusion, precision fallback, kernel/tactic 선택과 memory 이동에 의존
- RF-DETR Segmentation Large 기반 주차 인식 모델에서 여러 경량화 축을 동일 데이터·동일 GPU·동일 평가기로 비교한 근거가 필요
- 단일 최고 점수보다 정확도·latency·memory·engine size의 다목적 trade-off 분석 필요
- 관련 연구 공백은 2장의 선행연구 비교표로 뒷받침

### 1.3 연구 목적과 연구 질문

- 목적: RF-DETR 주차 인식 모델을 실제 TensorRT 배포 artifact로 변환하고 경량화 방법별 정확도·실행 비용 trade-off를 분석
- RQ1: decoder layer, FFN dimension, 입력 해상도 변화가 정적 복잡도와 detection·segmentation 품질에 미치는 영향은 무엇인가?
- RQ2: FP16, INT8, INT4, FP8 및 민감도 기반 mixed precision이 정확도·latency·memory·engine size에 만드는 trade-off는 무엇인가?
- RQ3: 비정형 sparsity와 하드웨어 지원 2:4 sparsity는 정적 희소성과 실제 TensorRT 가속에서 어떤 차이를 보이는가?
- RQ4: 구조·해상도·정밀도 결합 후보가 단일 경량화 후보보다 우수한 Pareto 지점을 제공하는가?

### 1.4 주요 기여

- 동일 baseline에서 파생된 26개 후보의 재현 가능한 실험 registry
- 정확도를 배제한 정적 선별, 목표 장비 실측, 정확도 gate 이후 Pareto 분석으로 이어지는 3단계 평가 절차
- bbox AP·mask AP·semantic mIoU와 latency·FPS·memory·engine size를 함께 기록하는 다목적 평가
- build 실패·미지원 precision·fallback도 누락하지 않고 terminal 결과로 남기는 자동화
- 설정, 모델·결과 SHA-256, 하드웨어 정보와 원시 로그를 연결하는 추적성

### 1.5 연구 범위

- front camera 모델만 대상
- 목표 장비는 노트북 NVIDIA GPU이나 실제 모델·VRAM·TGP·driver는 Stage-2 실행 로그로 확정
- ROS2 전체 end-to-end latency와 실제 주차 성공률은 평가 범위 밖
- 새로운 주차장·카메라·날씨에 대한 외적 일반화는 주장하지 않음
- 근거: [프로젝트 README](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/README.md)

## 2. Related Work

### 2.1 자율주행 주차 인식

- 주차 공간·경계·주변 객체 인식이 경로 계획과 제어에 전달되는 구조 설명
- detection만으로 설명하기 어려운 주차 영역·경계의 mask 품질 필요성 설명
- `[GitHub 근거 없음]` 기존 ASK 2026 논문의 정확한 제목·저자·학회·페이지와 인식 모듈 성과 추가 필요

### 2.2 DETR 계열과 RF-DETR

- DETR의 set prediction, object query, bipartite matching과 일반 NMS 비의존 구조
- DINO의 denoising 기반 query 학습 개선
- Mask DINO의 detection·segmentation 통합
- RF-DETR을 본 연구의 기존 주차 인식 baseline으로 선택한 이유
- 특정 YOLO 계열보다 보편적으로 우수하다고 주장하지 않음
- RF-DETR 실행 버전은 설정상 `1.8.1`; 세부 backbone·decoder·segmentation head는 공식 구현 대조 후 확정

### 2.3 배포 경량화

- FP16·INT8·INT4·FP8 및 mixed precision
- 비정형 pruning과 dense graph 한계
- decoder/FFN 구조 축소와 recovery fine-tuning
- NVIDIA 2:4 semi-structured sparsity와 sparse tactic 조건
- 입력 해상도 축소의 연산량 이득과 small object·mask 경계 손실 가능성
- SmoothQuant·AWQ는 LLM에서 제안된 방법이므로 RF-DETR에서의 효과를 보장하지 않고 후보 가설로만 사용

### 2.4 연구 공백 정리

- 기존 연구별 detection, segmentation, TensorRT engine, latency, memory, 복합 경량화, Pareto 평가 여부 비교표 추가
- 현재 참고문헌 12개의 정확한 KSAE 형식 변환 필요
- 근거: [README 참고문헌 및 기술 근거](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/README.md#참고문헌-및-기술-근거)

## 3. Data Construction and Split Protocol

### 3.1 데이터 출처

- Roboflow `parking_front` Version 8 annotation을 계승
- preprocessing과 offline augmentation을 빈 설정으로 지정한 Version 9 생성
- Version 9 총 4,466장
- 원래 Roboflow split을 사용하지 않고 전체 이미지를 세션 단위로 재분할
- `[GitHub 근거 없음]` 카메라 모델, 원본 영상 해상도, 프레임 추출 간격, 촬영 장소·조명·날씨, 데이터 사용 권한 필요

### 3.2 클래스와 annotation

- 평가 클래스: out_line, parking_lot, parking_space
- COCO 형식의 bbox·polygon/mask annotation 사용
- category 0 `front`는 상위 placeholder이고 평가 instance는 category 1~3에 존재
- `[GitHub 근거 없음]` 클래스 의미, polygon 경계 기준, 가림·잘림·모호한 경계 처리, 라벨링 도구, 검수자와 수정·제외 이력 필요

### 3.3 무증강과 품질 감사

- 무증강은 Roboflow offline preprocessing/augmentation이 없다는 의미
- 학습 train에는 multi-scale 등 online augmentation 사용 가능; valid/benchmark에는 결정적 resize·normalize 적용
- split 간 바이트 동일 이미지 0개
- split 간 촬영 세션 중복 0개
- COCO image–annotation 참조 오류 0개
- 손상·blur·비정상 polygon에 대한 수동 품질 검수는 확인되지 않음

### 3.4 세션 기반 분할

- timestamp·frame 번호 복원 가능 이미지: 1,461장
- 세션 시작 조건: 연속 시간 차 2.0초 초과 또는 시각 변화와 함께 frame 번호 감소
- 세션 내부 최대 간격 1.411초, 관측 세션 경계 최소 간격 5.010초
- 2초 임계값은 두 분포의 비관측 구간에 위치
- timestamp 세션 크기: 1, 1, 165, 179, 225, 272, 618장
- 20장 이상 주요 세션 5개를 train/valid/benchmark에 1/2/2개 배정
- timestamp를 복원하지 못한 3,005장은 잘못된 세션 추정을 피하기 위해 train에만 고정
- 30개 가능한 배정을 전수 조사하고 split 비율 편차와 클래스 출현률 편차를 사전순 최소화

### 3.5 최종 split과 역할

- train: 3,625장, 81.17%, baseline·recovery 학습 및 calibration 모집단
- validation: 404장, 9.05%, early stopping·checkpoint 선택 및 Q05~Q07 민감도 분석
- fixed benchmark: 437장, 9.79%, 후보 간 동일 조건 비교
- benchmark는 후보 개발 과정에서 반복 사용되어 완전히 독립된 confirmatory test로 표현하지 않음
- 외적 일반화 주장에는 별도 장소·조건의 잠금 holdout session 필요

### 3.6 추가 통계 필요

- `[GitHub 근거 없음]` split별 instance 수
- `[GitHub 근거 없음]` small/medium/large 객체 수와 mask 면적 분포
- `[GitHub 근거 없음]` calibration 128장의 구체적 샘플 목록과 대표성 분석
- 근거: [데이터 분할 타당성 보고서](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/reports/dataset-split-validity.md)

## 4. Deployment-Oriented Lightweighting Pipeline

### 4.1 Baseline 학습

- 모델: RF-DETR Segmentation Large, 설정상 RF-DETR 1.8.1
- 입력: 504×504
- pretrained weight MD5: `275f7b094909544ed2841c94a677d07e`
- epoch 상한 25, 실제 18 epoch 조기 종료
- micro-batch 2, gradient accumulation 8, effective batch 16
- learning rate 1.0e-4, encoder learning rate 1.5e-4, weight decay 1.0e-4
- EMA 0.993, seed 42, early stopping patience 6·min delta 0.001
- train으로 학습, validation으로 checkpoint 선택, 학습 중 benchmark 미사용
- 근거: [학습 설정](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/configs/training/front_rfdetr_seg_large.yaml), [학습 가이드](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/guides/front-baseline-training.md)

### 4.2 후보군

- Baseline/precision recipe: B01~B03
- Unstructured magnitude pruning: U01~U03, sparsity 10/30/50%
- 2:4 semi-structured sparsity: M01 dense control, M02 sparse tactic
- Structured pruning: S01 decoder 1개 제거, S02 decoder 2개 제거, S03 FFN 20%, S04 FFN 40%
- Combined: C01 S01+FP16, C02 S01+INT8, C03 S01+432, C04 S01+480
- Resolution: R01 432, R02 480, R03 384
- ModelOpt Q/DQ: Q01 INT8 SmoothQuant, Q02 INT8 mixed, Q03 INT4 weight-only, Q04 FP8, Q05~Q07 FP8 mixed
- 전체 26개 후보, Stage-1 통과 후보가 사용하는 고유 ONNX는 17개
- 근거: [실험 registry](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/configs/experiments/registry.yaml)

### 4.3 변환과 검증

- checkpoint → ONNX opset 17 → ONNX checker → 구조·동등성 검사 → 목표 GPU에서 TensorRT engine build
- 기본 입력 shape `[1,3,504,504]`, batch 1, static shape
- precision recipe 후보는 source ONNX와 deterministic build command 보관
- Q/DQ 후보는 자체 ONNX와 quantization report 보관
- artifact·report SHA-256 및 build log 기록

### 4.4 3단계 후보 판정

- Stage-1 정적평가
  - 정확도·latency·FPS·GPU memory를 사용하지 않음
  - 공통: 정적 보고서, 유효한 ONNX 또는 source ONNX와 build recipe
  - 구조·해상도 후보: B01 대비 ONNX 크기·node·dense MAC 중 하나 이상 5% 감소
  - Q01~Q07: 후보 자체 ONNX의 Q/DQ node와 quantization report 일치
  - M01/M02: 2:4 pattern 준수와 sparse tactic recipe
- Stage-2 TensorRT 평가
  - Stage-1 통과 22개 모두에 대해 engine build 시도
  - 성공뿐 아니라 build·inspection·benchmark·accuracy 실패도 terminal 결과로 기록
- Stage-3 Pareto
  - Stage-2 대상 22개가 모두 terminal일 때만 실행
  - 측정 유효성과 B01 대비 정확도 보존 gate를 통과한 후보만 Pareto 입력으로 사용
- 근거: [3단계 평가 프로토콜](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/guides/three-stage-candidate-evaluation.md)

## 5. Experimental Setup and Evaluation Protocol

### 5.1 장비와 소프트웨어

- `[Stage-2 후 삽입]` GPU 정확한 모델명, compute capability, VRAM, TGP와 전력 모드
- `[Stage-2 후 삽입]` CPU, RAM, OS, driver, CUDA, cuDNN, TensorRT, Python·PyTorch 버전
- 전원 연결, 성능 모드, thermal 상태, 백그라운드 부하 기록
- engine은 목표 GPU·CUDA·TensorRT 환경에서 새로 build

### 5.2 정확도 평가

- 고정 benchmark 437장
- COCO bbox/mask AP@[0.50:0.05:0.95], AP50, AP75, 크기별 AP, AR100
- 클래스별 bbox/mask AP
- semantic mIoU와 클래스별 IoU
- AP confidence floor 0.001
- semantic mIoU confidence 0.25
- max detections per image 100
- RF-DETR query를 score 순으로 평가하며 현재 NMS 미사용

### 5.3 실행 성능

- batch 1, seed 42로 고정한 32장
- 반복마다 warm-up 20회 후 본 측정 200회, 총 3반복
- 범위: disk image decode를 제외한 in-memory image end-to-end
- 보고: pooled median·mean·P95·P99·IQR·FPS
- 안정성: 반복 median CV와 95% t 구간
- CV 5% 초과 시 동일 엔진의 지연시간 3회를 한 번 전체 재측정
- 재측정 시 두 번째 라운드를 공식값으로 사용하고 첫 라운드도 원시 근거로 보존
- 두 번째 CV도 5% 초과 시 Pareto에는 유지하되 최종 권장 모델에서는 제외
- CV 재측정에서는 engine과 437장 정확도 결과 재사용
- peak allocated/reserved GPU memory와 engine bytes 기록

### 5.4 정확도 및 특수 후보 gate

- 공통: B01 대비 bbox AP·mask AP 절대 하락 각각 0.01 이하, semantic mIoU 하락 0.02 이하
- S02 vs S01: bbox/mask AP 하락 0.005 이하, mIoU 하락 0.01 이하, median latency 5% 이상 감소
- B03 INT8 vs B02 FP16: B01 대비 정확도 보존 및 median latency 10% 이상 감소
- M02 sparse vs M01 dense: 정확도 보존, sparse tactic 증거, median latency 10% 이상 감소
- engineering gate이며 통계적 유의성 기준이라고 주장하지 않음

### 5.5 평가 자동화와 재현성

- 단일 명령으로 checksum, environment, engine build, inspection, benchmark, accuracy, gate, Pareto와 Excel 생성
- 중단 후 완료 후보 재사용
- 결과: JSON·CSV·Markdown·Excel·후보별 로그
- 현재 자동화 감사: 필수 실패 0, Stage-2 실행 대기 상태
- 근거: [평가 설정](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/configs/experiments/defaults.yaml), [평가 지표](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/concepts/evaluation-metrics.md), [자동화 감사](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/reports/evaluation-automation-audit.md)

## 6. Results and Discussion

### 6.1 Baseline 결과

- benchmark 437장
- bbox AP 0.737791
- mask AP 0.601016
- semantic mIoU 0.712200
- small-object mask AP 0.0550으로 기록되어 작은 객체가 기준 모델의 취약점
- 근거: [Baseline 평가 보고서](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/reports/front-rfdetr-seg-large-v1-test.md)

### 6.2 Stage-1 정적평가 결과

- 전체 26개 모두 평가
- 통과 22개, 불통 4개, 미수행 0개
- U01·U02·U03 불통: 비정형 sparsity 10/30/50%가 dense ONNX 크기·node·MAC을 줄이지 못함
- S03 불통: ONNX 크기 3.27%, dense MAC 0.39%, node 0% 감소로 5% gate 미달
- M01/M02: 적격 150/150 연산이 2:4 준수, sparse build recipe 존재
- S01/S02: node 8.04%/16.17% 감소
- S04: ONNX 크기 6.53% 감소
- C03/C04: node 8.04%, dense MAC 30.76%/11.51% 감소
- R01/R02/R03: dense MAC 30.20%/10.96%/46.25% 감소
- Q01~Q07: ONNX checker 통과, Q/DQ node 34~626개 확인
- static MAC/FLOP는 Conv·MatMul·Gemm 중 shape를 해석한 dense 산술량 하한이며 실제 속도 증거가 아님
- 근거: [Stage-1 전체 결과](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/results/stage1-static-evaluation.md)

### 6.3 Stage-2 TensorRT 결과

- `[Stage-2 후 삽입]` 22개 후보별 build 성공/실패와 사유
- `[Stage-2 후 삽입]` 실제 layer precision과 fallback 비율
- `[Stage-2 후 삽입]` M02 sparse tactic 선택 증거
- `[Stage-2 후 삽입]` bbox/mask AP와 semantic mIoU
- `[Stage-2 후 삽입]` median/P95/P99 latency, FPS, CV와 95% 구간
- `[Stage-2 후 삽입]` peak GPU memory와 engine size
- 실패 후보도 표에서 제거하지 않고 terminal failure로 표시

### 6.4 계열별 분석

- `[Stage-2 후 삽입]` FP32 vs FP16 vs INT8
- `[Stage-2 후 삽입]` INT8 SmoothQuant·mixed precision·INT4·FP8 비교
- `[Stage-2 후 삽입]` decoder/FFN 축소 및 recovery 효과
- `[Stage-2 후 삽입]` 2:4 dense control vs sparse tactic
- `[Stage-2 후 삽입]` 504/480/432/384 해상도
- `[Stage-2 후 삽입]` 단일 방법 vs combined 후보
- 정확도 차이는 클래스·객체 크기·mask 경계와 연결해 해석

### 6.5 Accuracy gate와 Pareto

- `[Stage-2 후 삽입]` 측정 유효 후보 수와 accuracy gate 통과 수
- 정확도 선행 gate: bbox AP, mask AP, semantic mIoU 보존 기준
- 주 Pareto 최대화: mask AP
- 주 Pareto 최소화: median latency, engine size
- 보조지표: bbox AP, semantic mIoU, P95 latency, peak allocated GPU memory
- 한 후보가 세 주 목적에서 같거나 우수하고 하나 이상 엄격히 우수하면 지배 관계로 판정
- 실시간 배포 후보: Pareto front 중 median latency ≤ 33.33 ms(30 FPS frame period)
- P95 정책: 33.33 ms 초과 시 tail-latency 경고, hard gate·Pareto 제외에는 미사용
- 30 FPS 기준은 median 조건이며 P95·지속 처리량 보장은 별도 해석
- 전체 재측정 후에도 반복 median CV > 5%인 후보는 Pareto에 보존하되 최종 권장에서 제외
- `[Stage-3 후 삽입]` Pareto front 후보
- `[Stage-3 후 삽입]` 균형형·정확도 우선·속도 우선·크기 우선 권장 후보와 선택 근거

### 6.6 위협 요인

- single seed 학습으로 학습 분산을 추정하지 않음
- validation/test가 각각 2개 촬영 세션
- 고정 benchmark를 후보 개발에서 반복 사용
- 일부 데스크탑 진단 정확도는 CPU/GPU가 혼재하므로 실행 속도 비교에 사용하지 않음
- 단일 목표 GPU 결과를 다른 GPU로 일반화할 수 없음
- engine-only/in-memory 평가와 실제 ROS2 전체 latency는 다름
- 별도 외부 holdout이 없어 새로운 장소·조건의 일반화는 검증하지 않음

## 7. Conclusion

- 확정 가능한 결론
  - 세션 기반 데이터 분할과 누수 감사 완료
  - 동일 baseline에서 26개 경량화 후보 구성
  - Stage-1 26/26 완료, 22개 통과, 4개 불통
  - 비정형 pruning의 zero sparsity가 dense graph 감소를 보장하지 않는 사례 확인
  - 정적 screening과 목표 GPU runtime 평가를 분리할 필요성 확인
- 아직 확정할 수 없는 결론
  - `[Stage-2 후 삽입]` 가장 빠른 후보
  - `[Stage-2 후 삽입]` 정확도 보존 후보
  - `[Stage-2 후 삽입]` memory·engine size 감소량
  - `[Stage-3 후 삽입]` Pareto 및 시나리오별 최종 권장 후보
- 향후 연구
  - 별도 장소·날씨·카메라의 잠금 holdout session
  - rear camera와 다중 시점 확장
  - 다른 NVIDIA GPU와 전력·thermal 조건 비교
  - ROS2 message 수신부터 결과 publish까지의 end-to-end latency
  - QAT 또는 layer-wise mixed precision

## Acknowledgement

- `[GitHub 근거 없음]` 지원기관, 과제번호와 공식 영문 사사 문구를 증빙하는 저장소 문서 필요
- GitHub 근거가 추가되기 전에는 최종 원고 문구로 확정하지 않음

## 참고문헌 작업

- 현재 12개 기술 근거는 GitHub README에 기록
- `[GitHub 근거 없음]` 기존 ASK 2026 논문의 정확한 서지정보 필요
- 각 문헌의 저자·제목·학회·연도·권호·페이지·DOI를 KSAE 형식으로 검증 필요
- 본문 작성 종료 후 최초 인용 순서로 번호 재배열
- 근거: [참고문헌 목록](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/README.md#참고문헌-및-기술-근거)

## 근거 공백 및 후속 입력 목록

| 상태 | 필요한 정보 | 필요한 이유 |
|---|---|---|
| GitHub 근거 없음 | 기존 ASK 논문 전체 서지정보 | 선행 연구와 후속 연구의 연결 증명 |
| GitHub 근거 없음 | 카메라·원본 영상·촬영 조건·데이터 권한 | 데이터 provenance와 재현성 |
| GitHub 근거 없음 | annotation guideline·도구·검수·수정 이력 | 라벨 신뢰도 설명 |
| GitHub 근거 없음 | split별 instance·객체 크기·mask 면적 통계 | 데이터 분포와 small-object 해석 |
| GitHub 근거 없음 | calibration 128장 표본 목록·대표성 | PTQ 재현성과 편향 점검 |
| GitHub 근거 없음 | 공식 사사 증빙 | Acknowledgement 확정 |
| Stage-2 대기 | 실제 노트북 하드웨어·소프트웨어 환경 | TensorRT 결과 재현 |
| Stage-2 대기 | 22개 engine build·정확도·latency·memory·size | 최종 성능 비교 |
| Stage-3 대기 | accuracy gate 결과·Pareto front | 최종 후보 선정 |
| 추가 실험 필요 | 별도 장소의 잠금 holdout session | 외적 일반화 주장 |
