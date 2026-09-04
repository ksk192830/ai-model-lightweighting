# 노트북 GPU 배포를 위한 RF-DETR 기반 주차 인식 모델의 경량화 및 다목적 평가

> **작성 상태:** Stage-1까지 반영한 결과 확정 전 줄글 초안이다. `[Stage-2 후 삽입]`,
> `[Stage-3 후 삽입]`, `[GitHub 근거 없음]` 표시는 필요한 근거가 확보되기 전까지
> 임의로 채우지 않는다. 각 절의 `작성 근거` 링크는 제출본 편집 시 제거하고 정식
> 참고문헌 또는 실험 표·그림으로 대체한다.

## Abstract

Autonomous parking perception requires both reliable recognition of parking-related regions and efficient inference under the latency and memory constraints of an on-board GPU. This study investigates deployment-oriented lightweighting of an RF-DETR Segmentation Large model trained for front-view parking perception. Starting from a common PyTorch checkpoint, 26 candidates were constructed using FP32 and FP16 execution, INT8 post-training quantization, INT4 weight-only quantization, FP8 and sensitivity-based mixed precision, unstructured pruning, structured decoder and feed-forward-network reduction, hardware-oriented 2:4 sparsity, input-resolution scaling, and selected combinations. The evaluation procedure separates static artifact screening from device-dependent TensorRT measurement and final Pareto analysis. The dataset contains 4,466 images divided by capture session into 3,625 training, 404 validation, and 437 fixed-benchmark images, with no exact cross-split duplicate, session overlap, or COCO reference error detected by the automated audit. Stage-1 static evaluation was completed for all 26 candidates; 22 passed and four candidates, U01, U02, U03, and S03, failed the predefined static-efficiency gate. `[Stage-2 후 삽입: accuracy, latency, throughput, GPU memory, engine size 및 accuracy-gate 결과]` `[Stage-3 후 삽입: Pareto 후보와 배포 시나리오별 권장 후보]`

**Keywords:** RF-DETR; TensorRT; Model Lightweighting; Autonomous Parking; Instance Segmentation

## 1. Introduction

### 1.1 연구 배경

자율주행 주차 시스템의 인지 모듈은 주차 가능 영역과 주차선, 주변 객체를 인식하고 그 결과를 경로 계획과 제어 단계에 전달한다. 전방 카메라 기반 주차 인지에서는 객체의 존재와 위치뿐 아니라 주차 영역과 경계의 형태도 중요하므로 bounding box 기반 객체 검출과 mask 기반 분할 품질을 함께 고려할 필요가 있다. 또한 인지 결과가 후속 모듈의 입력으로 사용되는 실제 시스템에서는 정확도뿐 아니라 추론 지연시간, 처리량, GPU 메모리와 배포 파일 크기가 탑재 가능성을 결정한다. 본 연구에서 전력과 발열은 제한된 장비 환경을 설명하는 배경 요인으로만 다루며, 별도 계측 근거가 확보되지 않는 한 정량 결과로 주장하지 않는다.

Transformer 기반 detector는 object query와 전역 문맥을 활용하는 end-to-end 예측 구조를 제공하며 detection과 segmentation으로 확장될 수 있다. 그러나 encoder–decoder 연산, 고해상도 feature 처리와 segmentation head는 실제 배포 환경에서 계산량과 메모리 부담을 만들 수 있다. 따라서 parameter 수, checkpoint 크기 또는 이론적 FLOP만으로는 목표 GPU에서의 실질적인 배포 효율을 충분히 설명하기 어렵다.

기존 연구의 연장선에서 본 연구는 ROS2 기반 주차 시스템 전체를 다시 제안하지 않고 전방 perception model의 배포 최적화에 범위를 한정한다. 다만 기존 ASK 2026 논문의 정확한 서지정보와 시스템 성과는 현재 GitHub에 근거가 없으므로, 해당 자료가 저장소에 추가되기 전까지 구체적인 제목·수치·페이지를 본문에 삽입하지 않는다.

### 1.2 문제 정의와 연구 공백

모델 경량화에서 정적 parameter 수와 연산량 감소는 중요한 분석 항목이지만 실제 GPU latency 감소와 동일하지 않다. TensorRT engine의 실행 성능은 graph fusion, precision fallback, kernel과 tactic 선택, memory 이동 및 목표 하드웨어의 지원 범위에 영향을 받는다. 예를 들어 비정형 pruning은 다수의 weight를 0으로 만들 수 있으나 tensor shape와 dense graph가 유지되면 일반적인 dense kernel의 실행량이 줄지 않을 수 있다. 반대로 decoder layer나 FFN 차원을 제거하는 구조 축소, 입력 해상도 축소, 저정밀 kernel 또는 지원되는 2:4 sparse tactic은 실행 경로 자체를 바꿀 가능성이 있다.

이에 본 연구는 RF-DETR Segmentation Large 기반 주차 인식 모델에서 여러 경량화 축을 동일한 데이터와 동일한 TensorRT 평가 절차로 비교하는 문제를 다룬다. 정적 변화가 존재한다는 사실과 실제 GPU 가속을 구분하고, detection·segmentation 품질과 latency·memory·engine size를 동시에 분석하며, 단일 가중합 점수 대신 비지배 관계를 이용하여 목적별 배포 후보를 선정하는 것을 연구의 중심 과제로 설정한다.

### 1.3 연구 목적과 연구 질문

본 연구의 목적은 전방 주차 인식용 RF-DETR Segmentation Large를 PyTorch checkpoint에서 ONNX를 거쳐 TensorRT engine으로 변환하고, 서로 다른 경량화 후보가 정확도와 실행 자원 사이에 만드는 trade-off를 실제 배포 artifact를 기준으로 분석하는 데 있다. 이를 위해 모든 후보를 먼저 정적 유효성과 경량화 적용 증거로 선별하고, 통과 후보를 동일한 노트북 GPU에서 TensorRT engine으로 생성해 정확도와 실행 성능을 측정하며, 모든 후보가 완료 또는 명시적 실패 상태가 된 뒤 정확도 보존 gate와 Pareto 분석을 수행한다.

본 연구는 네 가지 질문에 답하고자 한다. 첫째, decoder layer, FFN dimension과 입력 해상도 변화가 RF-DETR의 정적 복잡도 및 detection·segmentation 품질에 어떠한 영향을 주는가? 둘째, FP16, INT8, INT4, FP8과 민감도 기반 mixed precision이 정확도, latency, GPU memory와 engine size 사이에 어떠한 trade-off를 형성하는가? 셋째, 비정형 sparsity와 하드웨어가 지원하는 2:4 sparsity는 정적 희소성과 실제 TensorRT 가속 측면에서 어떠한 차이를 보이는가? 넷째, 구조·해상도·정밀도를 결합한 후보가 단일 경량화 후보보다 우수한 Pareto 지점을 제공하는가?

### 1.4 주요 기여와 범위

본 연구의 첫 번째 기여는 동일 baseline에서 파생된 26개 경량화 후보를 실험 registry와 artifact hash로 추적할 수 있는 재현 가능한 비교 체계를 구성한 것이다. 두 번째 기여는 정확도를 사용하지 않는 정적 선별, 목표 장비의 정확도·실행 성능 측정, 정확도 gate 이후 Pareto 분석으로 이어지는 3단계 평가 절차를 제시한 것이다. 세 번째 기여는 bbox AP, mask AP, semantic mIoU와 median·tail latency, 처리량, GPU memory, engine size를 함께 기록하여 detection과 segmentation 품질을 실제 배포 비용과 연결하는 것이다. 네 번째 기여는 지원되지 않는 precision, build 실패 또는 평가 실패를 결과에서 누락하지 않고 terminal 상태와 로그로 보존하는 자동화를 구성한 것이다.

연구 범위는 전방 카메라 주차 인식 모델과 단일 NVIDIA 노트북 GPU의 TensorRT engine 평가로 한정한다. 새로운 detector 구조, ROS2 전체 end-to-end 지연시간, 실제 주차 성공률, 다른 GPU와 새로운 주차 환경에 대한 일반화는 본 연구의 직접적인 검증 범위가 아니다.

> 작성 근거: [프로젝트 README](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/README.md), [프로젝트 진행 현황](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/PROJECT_STATUS.md)

## 2. Related Work

### 2.1 DETR 계열 객체 검출과 분할

DETR은 객체 검출을 고정된 object query 집합에 대한 set prediction 문제로 정의하고 Transformer encoder–decoder와 bipartite matching을 사용하여 예측과 정답을 일대일로 대응시켰다[2]. 이 접근은 anchor 설계와 일반적인 NMS 중심 후처리에 대한 의존성을 줄이고 detection을 end-to-end 구조로 다룰 수 있게 하였다. DINO는 denoising 학습과 query 초기화를 개선하여 DETR 계열의 학습 및 예측 구조를 발전시켰으며[3], Mask DINO는 mask prediction을 결합하여 detection과 segmentation을 통합하는 방향을 제시하였다[4].

RF-DETR은 정확도와 실시간 검출 성능의 절충을 목표로 하는 DETR 계열 모델이다[1]. 본 연구는 기존 주차 인식 과제에서 사용한 RF-DETR Segmentation Large의 bbox와 instance mask 출력을 유지한 상태에서 배포 비용을 줄이는 문제를 다룬다. 따라서 다른 detector 계열에 대한 보편적 우월성을 주장하지 않고, 동일 baseline에서 파생된 후보 사이의 차이만을 주요 비교 대상으로 삼는다. 저장소 설정에는 RF-DETR 1.8.1과 504×504 입력이 기록되어 있으나, 세부 backbone·decoder·segmentation head 설명은 사용한 공식 구현과 대조한 근거가 추가된 뒤 확정한다.

### 2.2 Pruning과 희소 실행

비정형 pruning은 중요도가 낮은 개별 weight를 제거하여 높은 zero sparsity를 만들 수 있다[5]. 그러나 불규칙한 희소 패턴은 지원 sparse kernel이 없을 경우 dense 실행 비용을 직접 줄이지 못할 수 있다. 반면 filter나 layer, hidden dimension과 같은 연산 단위를 제거하는 structured pruning은 tensor shape 또는 graph depth를 바꾸므로 기존 dense kernel에서도 실행량 감소를 기대할 수 있지만[6], 표현 용량 감소에 따른 정확도 저하를 보완하기 위해 recovery fine-tuning이 필요하다.

NVIDIA 2:4 sparsity는 연속된 네 weight 중 두 개를 0으로 제한하는 반구조적 패턴이다. 이 패턴의 존재만으로 가속이 보장되는 것은 아니며 지원 GPU, 대상 연산과 TensorRT sparse tactic이 함께 충족되어야 한다[9]. 이에 본 연구는 동일한 2:4 weight를 dense tactic으로 실행하는 M01과 sparse tactic을 허용하는 M02를 분리하여 weight 변화와 kernel 선택 효과를 구분한다.

### 2.3 저정밀 양자화와 입력 해상도

FP16, INT8, INT4와 FP8은 weight와 activation의 표현 정밀도를 낮추어 저장·메모리 대역폭과 연산 비용을 줄일 수 있지만, calibration과 activation outlier, 민감 layer 및 목표 하드웨어의 지원 여부에 따라 실제 정확도와 속도가 달라진다. TensorRT의 explicit quantization은 ONNX의 Quantize/Dequantize node로 정밀도 경계를 표현하며, Q/DQ node가 존재한다는 사실은 양자화 적용의 정적 증거이지 실제 저정밀 kernel 선택과 가속의 증거는 아니다[10].

SmoothQuant와 AWQ는 대규모 언어모델의 저정밀 추론을 위해 제안된 방법이다[7,8]. 본 연구에서는 이 방법들이 RF-DETR에서 동일한 효과를 낸다고 가정하지 않고, activation 또는 weight 민감도 원리가 다른 모델 계열로 전이될 수 있는지를 확인하는 실험 후보로만 사용한다. 입력 해상도 축소는 공간 위치와 feature map의 크기를 줄여 연산량을 직접 낮출 수 있지만 작은 객체와 얇은 주차 경계의 정보 손실을 만들 수 있으므로 독립적인 배포 trade-off로 평가한다.

### 2.4 연구의 차별점

본 연구는 모델 파일이나 이론 연산량만으로 경량화 효과를 결론 내리지 않고, 정적 screening을 통과한 후보를 동일한 목표 GPU에서 TensorRT engine으로 다시 생성하여 detection·segmentation 정확도와 실행 비용을 함께 비교한다. 이후 정확도 보존 조건을 만족한 후보에 대해 Pareto 비지배 관계를 분석한다[12]. 다만 기존 ROS2 수직주차 연구와 직접 비교하는 서술은 해당 논문의 정확한 서지정보가 GitHub에 추가된 뒤 보완한다.

> 작성 근거: [README의 이론적 배경과 참고문헌](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/README.md#이론적-배경)

## 3. Data Construction and Split Protocol

### 3.1 데이터 출처와 annotation

본 연구는 Roboflow `parking_front` Version 8의 annotation을 계승하되 preprocessing과 offline augmentation을 빈 설정으로 지정하여 생성한 Version 9를 사용한다. Version 9는 총 4,466장의 이미지로 구성되며, 기존 Roboflow split을 그대로 사용하지 않고 전체 이미지를 다시 합친 뒤 촬영 session을 분할 단위로 사용하였다. 최종 평가 클래스는 out_line, parking_lot과 parking_space이며 COCO 형식의 bbox와 polygon/mask annotation을 사용한다.

현재 GitHub에는 카메라 모델, 원본 영상 해상도, 프레임 추출 간격, 촬영 장소·조명·날씨, 데이터 사용 권한과 세부 annotation guideline이 기록되어 있지 않다. 따라서 이러한 정보는 추정하지 않으며, 수집 기록과 라벨링 자료가 저장소에 추가된 뒤 본 절에 삽입한다. 라벨링 도구, polygon 경계 기준, 가림·잘림·모호한 객체 처리, 검수자와 수정·제외 이력도 같은 방식으로 보완해야 한다.

### 3.2 세션 기반 분할

연속 영상에서 추출된 유사 프레임을 이미지 단위로 무작위 분할하면 인접 프레임이나 동일 장면이 train과 benchmark에 동시에 포함될 수 있다. 이를 줄이기 위해 파일명에서 촬영 시각과 frame 번호를 복원하고, 인접 이미지의 시간 차가 2.0초를 초과하거나 시각이 달라진 상태에서 frame 번호가 감소하는 지점을 새로운 session으로 정의하였다. 양의 세션 내부 시간 간격 1,454개의 최댓값은 1.411초이고 관측된 세션 경계 간격의 최솟값은 5.010초이므로, 2.0초 임계값은 두 관측 분포 사이의 빈 구간에 위치한다.

타임스탬프를 복원할 수 있는 이미지는 1,461장이며 session 크기는 1, 1, 165, 179, 225, 272와 618장이었다. 20장 이상의 주요 session 다섯 개를 train, validation과 benchmark에 각각 1개, 2개와 2개 배정하고, 1장짜리 session 두 개는 train에 포함하였다. 파일명만으로 촬영 시점을 복원할 수 없는 3,005장은 잘못된 session 추정으로 평가 누수가 발생하는 것을 피하기 위해 train에만 고정하였다.

세션 완전성과 최소 session 수 조건을 만족하는 30개 배정을 전수 조사하였다. 배정은 80:10:10 목표 비율과의 최대·총 절대 편차를 먼저 최소화하고, 이후 benchmark, validation, train 순으로 클래스 출현률의 제곱 편차를 최소화하는 사전순 목적함수로 결정하였다. 그 결과 train 3,625장(81.17%), validation 404장(9.05%), fixed benchmark 437장(9.79%)으로 구성되었으며 split 간 바이트 단위 동일 이미지, session 중복과 COCO 참조 오류는 모두 0건이었다.

### 3.3 데이터 역할과 한계

Train split은 baseline 및 recovery 학습과 양자화 calibration 후보 모집단으로 사용한다. Validation split은 early stopping과 checkpoint 선택, Q05~Q07의 민감도 기반 보호 block 선정에 사용한다. 437장 benchmark는 후보 간 동일 조건 비교에 사용한다. 이 benchmark는 후보 개발 과정에서 반복 참조되었으므로 완전히 손대지 않은 confirmatory test로 표현하지 않으며, 새로운 장소·카메라·날씨에 대한 외적 일반화를 주장하려면 별도 촬영 session을 잠금 holdout으로 확보해야 한다.

무증강은 Roboflow export 단계의 offline preprocessing과 augmentation이 없다는 뜻이다. 학습 시 train에 적용되는 multi-scale 등 online augmentation과는 구분한다. 현재 이미지 단위 클래스 출현률은 기록되어 있지만 split별 instance 수, small·medium·large 객체 수, mask 면적 분포와 calibration 128장의 대표성 분석은 GitHub에 근거가 없으므로 후속 집계가 필요하다.

> 작성 근거: [데이터 분할 타당성 보고서](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/reports/dataset-split-validity.md)

## 4. Deployment-Oriented Lightweighting Pipeline

### 4.1 Baseline 학습과 공통 변환

기준 모델은 RF-DETR Segmentation Large이며 입력 해상도는 504×504이다. 저장소 설정에는 RF-DETR 1.8.1, pretrained weight MD5 `275f7b094909544ed2841c94a677d07e`, epoch 상한 25, micro-batch 2, gradient accumulation 8, effective batch 16, learning rate 1.0×10⁻⁴, encoder learning rate 1.5×10⁻⁴, weight decay 1.0×10⁻⁴, EMA decay 0.993과 random seed 42가 기록되어 있다. Validation segmentation 지표가 0.001 이상 개선되지 않는 상태가 6회 이어지면 조기 종료하며 실제 baseline 학습은 18 epoch에서 종료되었다. 학습에는 train을 사용하고 validation으로 checkpoint를 선택했으며 학습 중 benchmark 평가는 수행하지 않았다.

선정 checkpoint는 ONNX opset 17, batch 1과 static input shape로 export한다. 이후 ONNX checker, graph structure, 출력 동등성과 artifact hash를 검사하고, TensorRT engine은 GPU·CUDA·TensorRT 버전 의존성을 고려하여 실제 평가 노트북에서 새로 build한다. Precision만 바뀌는 후보는 동일 source ONNX를 공유할 수 있고, 구조·해상도 또는 Q/DQ graph가 달라지는 후보는 별도 ONNX를 사용한다.

### 4.2 경량화 후보 구성

후보군은 총 26개로 구성된다. B01은 FP32 기준 graph이며 B02와 B03은 각각 TensorRT FP16과 INT8 PTQ build recipe이다. U01~U03은 global magnitude 기준 10%, 30%, 50% 비정형 pruning 후보다. M01은 2:4 weight를 dense tactic으로 실행하는 대조군이고 M02는 sparse tactic을 허용하는 후보이다. S01과 S02는 decoder와 연결 segmentation block을 각각 한 개와 두 개 제거하며, S03과 S04는 FFN dimension을 각각 20%와 40% 축소한다. C01과 C02는 S01 구조에 FP16과 INT8을 결합하고, C03과 C04는 S01 구조에 432×432와 480×480 입력을 결합한다. R01~R03은 기준 weight를 유지하면서 입력을 각각 432×432, 480×480과 384×384로 조정한다. Q01~Q07은 INT8 SmoothQuant, 선택적 INT8/FP16, INT4 weight-only, 균일 FP8과 세 종류의 민감도 기반 FP8 mixed-precision 후보로 구성된다.

### 4.3 Stage-1 정적 선별

Stage-1은 모든 후보의 배포 artifact 유효성과 경량화 적용 여부를 검사하며 정확도, latency, FPS와 GPU memory는 판정에 사용하지 않는다. 구조 또는 해상도 후보는 B01 대비 ONNX 크기, graph node 수와 dense MAC 추정치 중 하나 이상이 사전에 정한 5% 감소 기준을 만족해야 한다. 이 기준은 통계적 유의성 기준이 아니라 목표 장비에서 추가 측정할 가치가 있는 최소 정적 변화를 선별하는 engineering gate이다.

Precision recipe 후보는 유효한 source ONNX와 재현 가능한 TensorRT build command를 요구한다. Q01~Q07은 후보 자체 ONNX의 Q/DQ node와 quantization report가 일치해야 하며, M01과 M02는 적격 constant-weight 연산의 2:4 pattern 준수와 sparse tactic 설정을 확인한다. Stage-1을 통과한 후보만 Stage-2 노트북 평가 대상으로 삼되, Stage-2에서 build나 평가에 실패한 후보도 누락하지 않고 terminal failure로 기록한다.

> 작성 근거: [Baseline 학습 설정](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/configs/training/front_rfdetr_seg_large.yaml), [실험 registry](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/configs/experiments/registry.yaml), [3단계 평가 프로토콜](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/guides/three-stage-candidate-evaluation.md)

## 5. Experimental Setup and Evaluation Protocol

### 5.1 평가 환경

TensorRT engine은 평가 장비의 GPU, compute capability, CUDA와 TensorRT 버전에 종속되므로 Stage-2를 수행할 노트북에서 새로 생성한다. `[Stage-2 후 삽입: GPU 모델, VRAM, TGP·전력 모드, CPU, RAM, OS, driver, CUDA, cuDNN, TensorRT, Python과 PyTorch 버전]` 측정 중에는 전원 연결과 성능 모드, thermal 상태 및 백그라운드 부하를 기록한다. 현재 특정 노트북 환경의 수치는 실행 로그가 GitHub에 추가되기 전까지 확정값으로 작성하지 않는다.

### 5.2 정확도 평가

모든 성공 engine은 동일한 437장 fixed benchmark에서 평가한다. Detection과 instance segmentation은 COCO AP@[0.50:0.05:0.95], AP50, AP75, 객체 크기별 AP와 AR100을 사용하고 클래스별 AP를 함께 기록한다[11]. Semantic segmentation 품질은 instance mask를 클래스별 영역으로 합성한 semantic mIoU와 클래스별 IoU로 보완한다. AP 계산은 precision–recall 곡선을 충분히 구성하기 위해 confidence 0.001 이상의 예측을 수집하고, semantic mIoU에는 confidence 0.25를 적용한다. 이미지당 최대 예측 수는 100으로 고정하며 RF-DETR query는 현재 NMS를 적용하지 않고 score 순으로 평가한다.

### 5.3 실행 성능과 자원 측정

실행 성능은 batch 1과 seed 42로 고정한 32장에서 측정한다. 각 후보를 세 번 반복하고, 매 반복에서 20회 warm-up 후 200회 본 측정을 수행한다. 측정 범위는 disk image decoding을 제외한 in-memory image end-to-end 추론이다. 주 통계는 pooled median latency이고 mean, P95, P99, IQR, FPS, 반복 median의 CV와 95% t 구간을 함께 보고한다. 반복 median CV가 5%를 초과하면 후보를 자동 탈락시키지 않고 전력 모드, thermal throttling과 백그라운드 부하를 확인한 후 재측정 대상으로 표시한다. 자원 지표로 peak allocated/reserved GPU memory와 TensorRT engine byte 크기를 기록한다.

### 5.4 수용 기준과 자동화

Stage-2의 공통 정확도 gate는 B01 대비 bbox AP와 mask AP의 절대 하락을 각각 0.01, semantic mIoU 하락을 0.02까지 허용한다. S02가 S01보다 유효한 구조 후보가 되려면 bbox/mask AP 하락 0.005 이하, mIoU 하락 0.01 이하와 median latency 5% 이상 감소를 함께 만족해야 한다. B03 INT8은 B01 대비 정확도를 보존하면서 B02 FP16보다 median latency가 10% 이상 감소해야 하며, M02는 M01 대비 정확도를 보존하고 sparse tactic 선택 근거와 median latency 10% 이상 감소를 갖춰야 한다. 이 수용값은 보편적인 통계 유의성 기준이 아니라 실험 전에 고정한 프로젝트 engineering gate이다.

노트북 자동화는 checksum과 환경 점검, 22개 후보의 engine build, engine inspection, 세 반복 benchmark, 437장 정확도 평가, 정확도 gate, Pareto 분석과 통합 Excel 보고서 생성을 한 번에 수행한다. 중단된 경우 완료 후보를 재사용하고, build·inspection·benchmark·accuracy 실패를 후보별 로그와 terminal 상태로 저장한다.

> 작성 근거: [평가 기본 설정](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/configs/experiments/defaults.yaml), [평가 지표](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/concepts/evaluation-metrics.md), [자동화 감사](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/reports/evaluation-automation-audit.md)

## 6. Results and Discussion

### 6.1 Baseline과 Stage-1 결과

기준 RF-DETR Segmentation Large는 437장 benchmark에서 bbox AP 0.737791, mask AP 0.601016과 semantic mIoU 0.712200을 기록하였다. 이는 이후 TensorRT 후보의 정확도 보존을 판정하는 기준값이다. 세부 COCO 결과에서는 small-object mask AP가 0.0550으로 기록되어 작은 객체와 세밀한 mask가 baseline에서도 취약한 조건임을 확인하였다. 다만 이 정확도는 PyTorch baseline의 고정 benchmark 결과이며 TensorRT 실행 속도를 나타내지 않는다.

Stage-1에서는 registry의 26개 후보를 모두 평가했으며 22개가 통과하고 U01, U02, U03과 S03의 네 후보가 불통했다. U01~U03은 비정형 sparsity를 각각 10%, 30%, 50% 적용했으나 dense ONNX 크기, graph node와 dense MAC 추정치가 사실상 감소하지 않아 5% gate를 충족하지 못했다. 이는 weight의 zero 비율 증가와 dense 배포 graph의 감소를 구분해야 함을 보여 준다. S03은 FFN dimension을 20% 축소했지만 ONNX 크기 감소 3.27%, dense MAC 감소 0.39%, node 감소 0%에 그쳐 같은 기준을 통과하지 못했다.

반면 S01과 S02는 decoder 제거에 따라 graph node가 각각 8.04%와 16.17% 감소했고, S04는 ONNX 크기가 6.53% 감소하여 통과했다. C03과 C04는 S01 구조와 해상도 축소를 결합하여 node를 각각 8.04% 줄이고 dense MAC을 30.76%와 11.51% 줄였다. R01, R02와 R03은 입력 해상도 변경으로 dense MAC이 각각 30.20%, 10.96%와 46.25% 감소했다. M01/M02의 source graph에서는 적격 연산 150개 모두가 2:4 pattern을 준수했으며 sparse build recipe가 확인되었다. Q01~Q07은 모두 ONNX checker를 통과했고 각 graph에서 34~626개의 Q/DQ node가 확인되었다.

이 정적 결과의 MAC/FLOP는 shape를 해석할 수 있는 Conv, MatMul과 Gemm 연산만 집계한 dense 산술량 하한이다. TensorRT fusion, memory 이동, 전처리와 후처리를 포함하지 않으므로 실제 latency와 FPS 개선은 Stage-2에서 별도로 검증해야 한다.

### 6.2 Stage-2 TensorRT 결과

`[Stage-2 후 삽입]` 이 절에는 22개 후보의 engine build 성공 또는 실패 상태, 실제 layer precision과 fallback, bbox/mask AP와 semantic mIoU, median·P95·P99 latency, FPS, 반복 변동성, peak GPU memory와 engine size를 제시한다. Build가 불가능하거나 precision이 지원되지 않는 후보도 삭제하지 않고 실패 단계와 원인을 함께 보고한다. M02는 sparse tactic이 실제로 선택되었는지를 build 및 inspection log로 확인한다.

`[Stage-2 후 삽입]` Precision, pruning, 2:4 sparsity, 입력 해상도와 combined 후보의 결과는 baseline 대비 절대 변화와 상대 변화로 제시한다. 서로 다른 해상도 후보는 동일 입력 조건의 직접 비교와 speed-oriented 배포 시나리오 해석을 구분한다. 정확도 변화는 전체 평균뿐 아니라 out_line, parking_lot, parking_space의 클래스별 AP·IoU와 small·medium·large 객체별 결과로 분석한다.

### 6.3 Accuracy gate와 Pareto 분석

`[Stage-3 후 삽입]` Stage-2 대상 22개가 모두 성공 또는 명시적 실패 상태가 된 뒤, 측정이 유효하고 bbox AP·mask AP·semantic mIoU 보존 gate를 통과한 후보만 Pareto 분석에 포함한다. 주 Pareto는 segmentation의 대표 품질인 mask AP를 최대화하고 median latency와 TensorRT engine size를 최소화하는 세 목적을 사용한다. 한 후보가 다른 후보보다 세 목적에서 모두 같거나 우수하며 적어도 하나에서 엄격히 우수하면 전자가 후자를 지배한다고 정의한다. Bbox AP, semantic mIoU, P95 latency와 peak allocated GPU memory는 후보 해석과 안정성 확인을 위한 보조지표로 함께 제시한다. Pareto front 중 median latency가 33.33 ms 이하인 후보를 30 FPS 실시간 배포 후보군으로 별도 표시한다. P95가 33.33 ms를 초과하면 tail-latency 경고를 기록하지만 hard gate나 Pareto 제외 조건으로 사용하지 않는다. 따라서 이 판정은 중앙 지연시간 기준이며 P95나 지속 처리량의 30 FPS 보장을 의미하지 않는다. 비지배 후보를 대상으로 균형형, 정확도 우선, 속도 우선과 크기 우선 배포 시나리오의 권장안을 제시한다.

### 6.4 타당성의 위협

Baseline과 recovery 학습은 seed 42의 단일 실행이므로 학습 분산을 추정하지 않는다. Validation과 benchmark가 각각 두 개의 timestamp session으로 구성되고, timestamp를 복원하지 못한 3,005장이 train에만 포함되어 데이터 분포 차이가 남는다. 또한 437장 benchmark는 후보 개발에서 반복 참조되었기 때문에 완전히 독립된 최종 test로 해석할 수 없다. 일부 기존 데스크탑 진단 정확도는 CPU와 GPU 실행이 혼재하지만 속도 비교에는 사용하지 않으며, 최종 실행 성능은 동일 노트북에서 새로 측정한 TensorRT 결과만 사용한다. 단일 GPU에서 얻은 결과는 다른 GPU, 다른 TensorRT 버전, 실제 ROS2 전체 파이프라인 또는 새로운 주차 환경으로 직접 일반화하지 않는다.

> 작성 근거: [Baseline 평가](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/reports/front-rfdetr-seg-large-v1-test.md), [Stage-1 전체 결과](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/results/stage1-static-evaluation.md), [정적 분석 방법](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/reports/paper-static-analysis.md)

## 7. Conclusion

본 연구는 전방 주차 인식용 RF-DETR Segmentation Large를 대상으로 precision, pruning, 2:4 sparsity, 입력 해상도와 mixed precision을 포함하는 26개 경량화 후보를 구성하고, 정적 선별과 목표 장비 실측을 분리한 3단계 평가 절차를 마련하였다. Roboflow Version 9의 4,466장을 촬영 session 단위로 train 3,625장, validation 404장과 fixed benchmark 437장으로 나누었으며, 자동 감사에서 split 간 동일 이미지, session 중복과 COCO 참조 오류는 모두 0건이었다.

현재까지 완료된 Stage-1에서는 26개 후보 전체를 평가하여 22개를 Stage-2 대상으로 선정하고 U01, U02, U03과 S03을 불통 처리하였다. 비정형 pruning은 높은 zero sparsity 자체가 dense ONNX 크기, graph node 또는 dense MAC 감소를 보장하지 않았고, FFN 20% 축소도 전체 graph 수준의 사전 고정 5% gate에 미달했다. 이 결과는 정적 sparsity나 이론 연산량만으로 배포 성능을 결론 내리지 않고 목표 장비의 engine 실행을 별도로 측정해야 한다는 본 연구의 설계를 뒷받침한다.

`[Stage-2 후 삽입]` 최종 원고에서는 22개 후보의 TensorRT build 및 정확도·latency·FPS·memory·engine size 결과로 RQ1~RQ3에 답한다. `[Stage-3 후 삽입]` 정확도 gate를 통과한 후보의 Pareto 관계를 이용하여 RQ4에 답하고 균형형·정확도 우선·속도 우선·크기 우선 권장 후보를 제시한다. 이 결과가 확보되기 전에는 특정 precision, pruning, sparsity 또는 해상도 후보의 최종 우월성을 주장하지 않는다.

향후 연구에서는 별도 장소와 조건에서 수집한 잠금 holdout session으로 외적 일반화를 검증하고, rear camera와 다중 시점으로 범위를 확장할 필요가 있다. 또한 서로 다른 NVIDIA GPU와 전력·thermal 조건의 차이, ROS2 image message 수신부터 결과 publish까지의 end-to-end latency, QAT와 layer-wise mixed precision을 후속 과제로 고려할 수 있다.

## Acknowledgement

`[GitHub 근거 없음: 지원기관·과제번호·공식 사사 문구를 증빙하는 저장소 자료 추가 후 삽입]`

## References

아래 목록은 GitHub README에 기록된 기술 근거를 초안 순서로 옮긴 것이다. 제출 전 저자, 제목, 학회·저널, 연도, 권호, 페이지와 DOI를 KSAE 양식으로 검증하고 본문 최초 인용 순서로 재배열한다.

1. Robinson et al., “RF-DETR: Neural Architecture Search for Real-Time Detection Transformers,” 2025.
2. Carion et al., “End-to-End Object Detection with Transformers,” ECCV, 2020.
3. Zhang et al., “DINO: DETR with Improved DeNoising Anchor Boxes,” ICLR, 2023.
4. Li et al., “Mask DINO: Towards a Unified Transformer-based Framework for Object Detection and Segmentation,” 2022.
5. Han et al., “Learning both Weights and Connections for Efficient Neural Networks,” NeurIPS, 2015.
6. Li et al., “Pruning Filters for Efficient ConvNets,” ICLR, 2017.
7. Xiao et al., “SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models,” ICML, 2023.
8. Lin et al., “AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration,” MLSys, 2024.
9. NVIDIA, “Accelerating Inference with Sparsity Using Ampere and TensorRT.”
10. NVIDIA, “TensorRT: Working with Quantized Types.”
11. Lin et al., “Microsoft COCO: Common Objects in Context,” ECCV, 2014.
12. Deb et al., “A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II,” IEEE Transactions on Evolutionary Computation, 2002.
13. `[GitHub 근거 없음: 기존 ASK 2026 논문의 정확한 서지정보 추가 필요]`

> 참고문헌 근거: [GitHub README](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/README.md#참고문헌-및-기술-근거)
