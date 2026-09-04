# 노트북 GPU 배포를 위한 RF-DETR 기반 주차 인식 모델의 경량화 및 다목적 평가 — Stage 3 줄글 초안

> **작성 상태:** 공식 실행 ID `20260904_104755`의 Stage 3 결과를 반영한 한국어 줄글 초안이다.
> 수치와 판정은 GitHub 근거가 있는 범위에서만 서술한다. 데이터 수집·라벨링 세부사항,
> 기존 ASK 논문의 서지정보와 사사 문구는 근거가 없어 보완 예정으로 남긴다.

## Abstract

자율주행 주차 인지는 주차 관련 영역을 정확히 검출하고 분할하는 동시에 제한된 온보드 GPU에서 짧은 지연시간으로 실행되어야 한다. 본 연구는 전방 주차 인식용 RF-DETR Segmentation Large를 대상으로 실제 배포 환경을 고려한 경량화 방법을 비교하였다. 공통 학습 checkpoint에서 FP16 및 INT8 실행, INT4 weight-only와 FP8 양자화, 비정형·구조적 pruning, NVIDIA 2:4 sparsity, 입력 해상도 조정 및 이들의 결합을 포함한 26개 후보를 구성하였다. 평가는 정적 artifact를 검사하는 Stage 1, NVIDIA GeForce RTX 4050 Laptop GPU에서 TensorRT engine의 정확도와 실행 성능을 측정하는 Stage 2, 정확도 보존 후보의 Pareto 비지배 관계를 분석하는 Stage 3으로 구분하였다. 총 4,466장의 데이터는 촬영 세션을 기준으로 train 3,625장, validation 404장, fixed comparative benchmark 437장으로 분리하였다. Stage 1에서는 26개 중 22개가 통과하였고, Stage 2에서는 21개의 engine 측정을 완료하였으며 Q03 INT4 후보는 TensorRT parser 호환성 문제로 build에 실패하였다. B01을 포함한 11개 후보가 사전 정의한 정확도 gate를 만족하였다. Mask AP를 최대화하고 median latency와 engine 크기를 최소화한 Stage 3 분석에서 C01과 R01이 Pareto 비지배해로 확인되었다. C01은 Mask AP 0.6029, median latency 25.300 ms, engine 크기 64.7 MiB로 B01 대비 latency 44.79%와 engine 크기 49.95%를 줄였다. R01은 Mask AP 0.5941, median latency 24.010 ms, engine 크기 68.1 MiB로 가장 짧은 중앙 지연시간을 기록하였다. 이에 따라 C01을 균형·정확도·크기 우선 후보로, R01을 속도 우선 후보로 제안한다.

**Keywords:** RF-DETR; TensorRT; Model Lightweighting; Autonomous Parking; Instance Segmentation

## 1. Introduction

자율주행 주차 시스템에서 인지 모듈의 출력은 경로 계획과 제어 단계의 입력으로 사용된다. 전방 카메라를 이용한 주차 인지에서는 객체의 존재와 위치뿐 아니라 주차 구획 및 경계의 형상도 중요하므로 bounding box 기반 검출 성능과 mask 기반 분할 성능을 함께 평가할 필요가 있다. 또한 실제 시스템에 탑재되는 모델은 정확도가 높더라도 지연시간과 메모리, 배포 artifact 크기가 장비의 제약을 넘으면 사용할 수 없다. 따라서 주차 인식 모델의 배포 가능성을 판단하려면 정확도와 실행 비용을 동일한 조건에서 함께 측정해야 한다.

Transformer 기반 detector는 object query와 전역 문맥을 활용하는 end-to-end 예측 구조를 제공하지만 encoder–decoder 연산과 고해상도 feature, segmentation head는 배포 환경에서 계산 부담을 만들 수 있다. 일반적으로 parameter 수, checkpoint 크기와 이론적 FLOP는 모델 복잡도를 설명하는 데 유용하지만 TensorRT의 실제 실행시간을 직접 보장하지 않는다. 실제 성능은 graph fusion, precision fallback, kernel과 tactic 선택, memory 이동 및 하드웨어 지원 범위에 좌우되기 때문이다. 특히 비정형 pruning은 weight의 0 비율을 높여도 dense tensor shape가 유지되면 일반적인 GPU kernel의 실행량을 줄이지 못할 수 있다.

본 연구는 새로운 detector 구조를 제안하기보다 기존 전방 주차 인식용 RF-DETR Segmentation Large에서 파생한 다양한 경량화 후보를 동일 데이터와 동일 GPU 평가기로 비교한다. 연구 질문은 구조 축소와 해상도 변화가 정확도와 복잡도에 미치는 영향, FP16·INT8·INT4·FP8 및 mixed precision의 정확도–실행 비용 trade-off, 비정형 sparsity와 2:4 sparsity의 차이, 그리고 결합 후보가 제공하는 Pareto 이점으로 구성하였다. 이를 위해 26개 후보를 정적 선별, 목표 장비 실측, 정확도 gate 이후 Pareto 분석의 세 단계로 평가하였다.

본 연구의 기여는 네 가지이다. 첫째, 동일 baseline에서 파생한 26개 후보를 registry와 artifact hash로 추적할 수 있는 비교 체계를 구성하였다. 둘째, 정확도를 사용하지 않는 정적 선별과 장비 종속적인 TensorRT 측정을 분리하여 정적 경량화와 실제 가속을 혼동하지 않도록 하였다. 셋째, bbox AP, mask AP, semantic mIoU와 median·tail latency, GPU memory 및 engine 크기를 함께 기록하였다. 넷째, build 실패와 미지원 precision도 결과에서 제외하지 않고 terminal 상태와 로그로 보존하는 자동화된 평가 절차를 구축하였다.

연구 범위는 전방 카메라 모델과 단일 RTX 4050 Laptop GPU의 TensorRT 실행으로 한정한다. ROS2 전체 end-to-end latency, 실제 주차 성공률, 다른 GPU 및 새로운 장소·날씨·카메라에 대한 일반화는 직접 검증하지 않았다. 또한 기존 ASK 2026 연구의 정확한 서지정보와 연결 내용은 현재 GitHub에 근거가 없으므로 자료가 제공된 뒤 보완한다.

## 2. Related Work

DETR은 객체 검출을 고정된 object query 집합에 대한 set prediction 문제로 정의하고 Transformer encoder–decoder와 bipartite matching을 이용하여 예측과 정답을 일대일로 대응시켰다. DINO는 denoising query 학습과 초기화를 개선하였고, Mask DINO는 detection과 segmentation을 통합하는 방향을 제시하였다. RF-DETR은 실시간 검출과 정확도의 절충을 목표로 한 DETR 계열 모델이며, 본 연구는 해당 모델의 segmentation 출력을 유지한 상태에서 배포 비용을 줄이는 문제를 다룬다. 따라서 다른 detector 계열에 대한 보편적 우월성은 주장하지 않고 동일 RF-DETR baseline에서 파생된 후보만 비교한다.

비정형 pruning은 중요도가 낮은 개별 weight를 제거하여 높은 zero sparsity를 만들 수 있지만, 불규칙한 희소 패턴을 지원하는 kernel이 없으면 dense 실행 비용을 줄이지 못할 수 있다. 반면 layer 또는 hidden dimension을 제거하는 structured pruning은 graph depth와 tensor shape를 바꾸므로 일반적인 dense kernel에서도 연산 감소를 기대할 수 있지만 표현 용량 감소에 따른 정확도 손실을 동반할 수 있다. NVIDIA 2:4 sparsity는 네 개의 연속 weight 중 두 개를 0으로 제한하는 반구조적 방식으로, 실제 가속에는 pattern 준수뿐 아니라 지원 GPU와 TensorRT sparse tactic이 필요하다.

FP16, INT8, INT4와 FP8은 weight와 activation의 표현 정밀도를 낮추어 저장 공간과 메모리 대역폭, 연산 비용을 줄일 수 있다. 그러나 calibration 표본, activation outlier, 민감 layer 및 목표 하드웨어의 지원 여부에 따라 정확도와 속도가 달라진다. ONNX의 QuantizeLinear와 DequantizeLinear node는 양자화 표현의 정적 증거지만 실제 저정밀 kernel 선택과 가속의 충분조건은 아니다. SmoothQuant와 AWQ는 주로 대규모 언어모델에서 제안된 방법이므로 RF-DETR에서도 같은 효과가 발생한다고 가정하지 않고 실험 후보로만 적용하였다. 입력 해상도 축소는 feature map의 크기를 직접 줄이지만 작은 객체와 얇은 mask 경계의 정보를 잃을 수 있어 별도 후보군으로 평가하였다.

경량화 후보의 배포 적합성은 하나의 정확도나 속도 지표로 결정하기 어렵다. 본 연구는 먼저 정확도 보존 기준을 통과한 후보만 남기고, Mask AP 최대화와 median latency·engine 크기 최소화의 세 목적에서 Pareto 비지배 관계를 계산하였다. BBox AP, semantic mIoU, P95 latency와 GPU memory는 보조지표로 사용하였다. 이는 임의의 가중치로 단일 점수를 만드는 대신 실제 배포 목적에 따라 후보를 선택할 수 있게 한다.

## 3. Data Construction and Split Protocol

본 연구는 Roboflow `parking_front` Version 8의 annotation을 계승하면서 export 단계의 offline preprocessing과 augmentation을 적용하지 않은 Version 9를 생성하여 사용하였다. 데이터는 총 4,466장이며 평가 대상 클래스는 out_line, parking_lot과 parking_space이다. Annotation은 COCO 형식의 bounding box와 polygon mask로 구성된다. Category 0의 front는 상위 placeholder이고 실제 평가 instance는 category 1부터 3까지에 존재한다.

연속 영상에서 추출된 유사 프레임을 이미지 단위로 무작위 분할하면 인접 장면이 train과 benchmark에 동시에 포함되어 성능을 과대평가할 수 있다. 이를 줄이기 위해 파일명에서 촬영 시각과 frame 번호를 복원하고, 연속 이미지의 시간 차가 2초를 초과하거나 시각이 달라진 상태에서 frame 번호가 감소하는 지점을 새 세션의 시작으로 정의하였다. Timestamp를 복원할 수 있는 1,461장은 관측 세션 단위로 배정하였다. 복원하지 못한 3,005장은 잘못된 세션 추정에 따른 평가 누수를 피하기 위해 train에만 고정하였다.

세션 완전성과 최소 세션 수를 만족하는 30개 주요 배정을 전수 조사하고 목표 비율 및 클래스 출현률 편차를 최소화하였다. 최종 분할은 train 3,625장(81.17%), validation 404장(9.05%)과 fixed comparative benchmark 437장(9.79%)이다. 자동 감사에서는 split 간 바이트 동일 이미지, 촬영 세션 중복과 COCO 참조 오류가 모두 0건으로 확인되었다. 노트북과 데스크톱의 annotation raw byte hash는 달랐지만 key 정렬과 직렬화를 정규화한 의미적 fingerprint와 이미지 collection fingerprint가 일치하여 실제 평가 데이터 내용은 동일한 것으로 확인하였다.

Train split은 baseline 및 recovery 학습과 calibration 모집단으로, validation은 early stopping과 checkpoint 선택 및 민감 block 선정에 사용하였다. 437장 benchmark는 후보 간 동일 조건 비교에 사용하였다. 이 benchmark는 후보 개발 과정에서 반복 참조되었기 때문에 완전히 독립된 confirmatory test로 표현하지 않는다. 카메라 모델, 원본 영상 해상도, 프레임 추출 간격, 촬영 장소·조명·날씨, 데이터 사용 권한과 세부 annotation guideline은 GitHub에 근거가 없어 추후 자료 확보가 필요하다.

## 4. Deployment-Oriented Lightweighting Pipeline

기준 모델은 입력 504×504의 RF-DETR Segmentation Large 1.8.1이다. Epoch 상한은 25이며 validation segmentation 지표가 0.001 이상 개선되지 않는 상태가 6회 이어지면 종료하도록 설정하였다. Micro-batch 2와 gradient accumulation 8로 effective batch 16을 구성하였고, learning rate 1.0×10⁻⁴, encoder learning rate 1.5×10⁻⁴, weight decay 1.0×10⁻⁴, EMA 0.993과 seed 42를 사용하였다. 실제 학습은 18 epoch에서 조기 종료되었다. 학습에는 train만 사용하고 validation으로 checkpoint를 선택했으며 학습 중 benchmark 평가는 수행하지 않았다.

후보군은 총 26개이다. B01은 FP32 baseline이며 B02와 B03은 각각 TensorRT FP16과 INT8 PTQ recipe이다. U01부터 U03은 10%, 30%, 50% global magnitude pruning 후보이고, M01과 M02는 동일한 2:4 weight를 dense control과 sparse tactic으로 비교한다. S01과 S02는 decoder layer를 한 개와 두 개 제거하며, S03과 S04는 FFN dimension을 20%와 40% 축소한다. C01과 C02는 S01 구조에 FP16과 INT8을 결합하고, C03과 C04는 S01 구조에 432×432와 480×480 입력을 결합한다. R01부터 R03은 432×432, 480×480과 384×384 입력을 사용한다. Q01부터 Q07은 SmoothQuant INT8, mixed INT8, INT4 weight-only, FP8과 민감도 기반 mixed FP8 후보로 구성하였다.

Stage 1은 ONNX 유효성, build recipe와 정적 효율 증거만 검사하고 정확도나 실행 성능을 사용하지 않았다. 구조 또는 해상도 후보는 B01 대비 ONNX 크기, graph node와 dense MAC 중 하나 이상이 5% 감소해야 한다. Precision recipe는 유효한 source ONNX와 재현 가능한 build command를 요구하고, Q 계열은 Q/DQ node와 quantization report의 일치를, M 계열은 2:4 pattern과 sparse tactic recipe를 확인하였다. Stage 1을 통과한 후보만 Stage 2로 보내되, 이후의 build 또는 평가 실패도 terminal 결과로 기록하였다.

Stage 2에서는 통과한 22개 후보를 목표 노트북에서 TensorRT engine으로 생성하고 engine inspection, 437장 정확도 평가와 반복 지연시간 측정을 수행하였다. 모든 후보가 완료 또는 명시적 실패 상태에 도달한 뒤 B01 대비 bbox AP·mask AP 하락 0.01 이하와 semantic mIoU 하락 0.02 이하를 모두 만족하는 후보만 Stage 3 Pareto 분석에 포함하였다. 이 기준은 통계적 유의성 기준이 아니라 실험 전에 고정한 engineering gate이다.

## 5. Experimental Setup and Evaluation Protocol

공식 실행 ID `20260904_104755`는 NVIDIA GeForce RTX 4050 Laptop GPU에서 수행하였다. VRAM은 6,141 MiB이며 CUDA 12.8, TensorRT 10.16.1.11과 driver 580.173.02를 사용하였다. CPU는 AMD Ryzen 7 7735HS 8-core/16-thread, RAM은 15.97 GB, 운영체제는 Ubuntu 22.04와 kernel 6.8.0-136-generic이다. 측정은 AC 전원에서 platform profile과 AMD P-State EPP를 모두 performance로 고정하였다. TensorRT engine은 이 환경에서 새로 생성하였다.

정확도는 동일한 437장 benchmark에서 평가하였다. Detection과 instance segmentation에는 COCO AP@[0.50:0.05:0.95], AP50, AP75, 객체 크기별 AP와 AR100을 사용하고 클래스별 AP를 함께 기록하였다. Semantic segmentation은 instance mask를 클래스 영역으로 합성한 mIoU와 클래스별 IoU로 보완하였다. AP 계산은 confidence 0.001 이상의 예측을 사용하고 semantic mIoU에는 confidence 0.25를 적용하였다. 이미지당 최대 검출은 100으로 고정하였다.

실행 성능은 batch 1과 seed 42로 고정한 32개 이미지에서 측정하였다. 각 후보를 세 번 반복하고 매 반복에서 20회 warm-up 뒤 200회 본 측정을 수행하였다. 측정 범위는 disk image decoding을 제외한 in-memory image end-to-end 추론이다. Pooled median을 주 지표로 사용하고 mean, P95, P99, IQR, FPS, 반복 median CV와 95% t 구간을 함께 기록하였다. 반복 median CV가 5%를 초과하면 동일 engine의 지연시간 측정을 한 차례 전체 재실행하도록 하였다. 공식 실행의 21개 후보와 각 3반복, 총 63개 반복은 모두 부하 통제 gate를 통과하였고 재측정 후보는 없었다. 최대 CV는 R01의 3.8249%였다.

Median latency가 33.33 ms 이하이면 30 FPS frame period를 만족하는 후보로 표시하였다. P95가 33.33 ms를 초과하면 tail-latency 경고를 기록하되 hard gate나 Pareto 제외 조건으로 사용하지 않았다. 따라서 30 FPS 표시는 중앙 지연시간 기준이며 지속 처리량 또는 모든 프레임의 30 FPS를 보장한다는 뜻이 아니다. GPU memory는 peak allocated와 reserved 값을 기록했지만 성공 후보 대부분이 약 890–898 MiB 범위에 있어 Pareto의 주 목적에서는 제외하였다.

전체 절차는 checksum과 환경 점검, engine build, inspection, benchmark, accuracy, gate, Pareto 및 Excel 보고서 생성을 연속 실행하도록 자동화하였다. 중단된 경우 완료 후보를 재사용하고 실패 단계는 후보별 로그와 terminal 상태에 보존하였다. 최종 자동화 감사에서 필수 실패와 미완료 항목은 0건이었고, legacy 자료와 반복 비교 benchmark 및 single seed에 관한 advisory만 남았다.

## 6. Results and Discussion

Stage 1에서는 26개 후보 전부를 평가하여 22개를 통과시켰다. U01부터 U03까지의 비정형 pruning 후보는 weight zero sparsity가 증가했지만 dense ONNX 크기, graph node와 dense MAC가 사전 고정한 5% 이상 감소하지 않아 불통하였다. S03은 FFN dimension을 20% 줄였으나 ONNX 크기 감소 3.27%, dense MAC 감소 0.39%, node 감소 0%에 그쳐 같은 기준에 미달하였다. 반면 decoder 축소, 더 큰 FFN 축소, 해상도 조정과 Q/DQ 표현 후보는 각 방식에 맞는 정적 증거를 충족하였다. 이 결과는 zero sparsity와 dense 배포 graph 감소를 구분해야 한다는 점을 보여 준다.

Stage 2 대상 22개는 모두 terminal 상태에 도달하였다. 이 가운데 21개는 engine build, 정확도와 지연시간 측정을 완료하였다. Q03 INT4 weight-only 후보는 TensorRT 10.16.1.11 parser가 block size 128의 DequantizeLinear 입력을 Float로 판단하면서 INVALID_NODE 오류를 반환하여 build에 실패하였다. 지정 ONNX의 hash를 확인한 뒤 발생한 오류이므로 누락 모델이나 과거 engine 오사용에 따른 실패가 아니다. 다만 이 결과는 현재 export 형식과 parser 조합의 호환성 문제이며 INT4가 일반적으로 불가능하다는 뜻은 아니다.

B01을 포함한 11개 후보가 정확도 gate를 통과하였다. 통과 집합은 B01, B02, B03, S01, C01, C02, C03, C04, R01, R02와 Q05이다. B03은 bbox AP 0.7422로 통과 후보 중 가장 높았고 C02는 semantic mIoU 0.7412로 같은 집합에서 가장 높았다. S02의 mIoU는 0.7531로 더 높았지만 bbox AP와 mask AP가 허용 범위를 벗어나 gate에 실패하였다. 이는 단일 segmentation 지표만으로 detection과 instance segmentation 품질까지 보존되었다고 결론 내릴 수 없음을 보여 준다.

2:4 계열 M01과 M02는 bbox·mask AP 및 mIoU가 모두 허용 범위를 넘게 하락하였다. R03은 384×384로 dense MAC을 크게 줄였지만 bbox AP와 mask AP 하락이 gate를 초과하였다. ModelOpt 계열에서는 Q05만 정확도를 보존했으며 Q01·Q02는 큰 정확도 손실, Q04·Q06·Q07은 bbox AP 붕괴를 보였다. 따라서 정적 Q/DQ node, 높은 sparsity 또는 작은 engine 크기만으로 경량화 성공을 주장할 수 없다.

Stage 3은 정확도 gate를 통과한 후보를 대상으로 Mask AP를 최대화하고 median latency와 engine 크기를 최소화하였다. 그 결과 C01과 R01만 Pareto 비지배해로 남았다. B01의 Mask AP, median latency와 engine 크기는 각각 0.6026, 45.820 ms와 129.3 MiB이다. C01은 0.6029, 25.300 ms와 64.7 MiB로 B01보다 median latency를 44.79%, engine 크기를 49.95% 줄이면서 Mask AP를 보존하였다. R01은 0.5941, 24.010 ms와 68.1 MiB로 B01보다 latency를 47.60%, engine 크기를 47.38% 줄였다. C01은 R01보다 1.289 ms 느리지만 Mask AP가 0.00880 높고 engine은 약 3.3 MiB 작다.

두 후보 모두 median 33.33 ms 이하와 P95 33.33 ms 이하를 만족하였다. C01의 P95는 27.512 ms, R01의 P95는 25.821 ms이다. 정확도와 크기, 속도를 함께 고려한 균형형 배포에는 C01이 적합하며 Mask AP 또는 engine 크기를 우선할 때도 C01이 유리하다. 반면 중앙 지연시간을 최우선으로 하면 R01이 적합하다. 본 연구는 이 trade-off를 임의의 가중합으로 제거하지 않고 목적별 추천으로 제시한다. B03과 C02는 각각 BBox AP와 mIoU가 우수한 보조 비교 후보이지만 세 주 목적에서는 C01에 지배되므로 최종 Pareto 추천에는 포함하지 않았다.

본 결과의 해석에는 몇 가지 제한이 있다. Baseline과 recovery 학습은 seed 42의 단일 실행이므로 학습 분산을 추정하지 않는다. Validation과 benchmark는 제한된 수의 촬영 세션으로 구성되며 437장 benchmark는 후보 개발에서 반복 사용되었다. 성능 측정은 단일 RTX 4050 Laptop GPU와 TensorRT 10.16.1.11에 한정되며 engine-only/in-memory 범위이므로 실제 ROS2 message 수신부터 결과 publish까지의 end-to-end latency와 다르다. 외부 장소·카메라·날씨의 잠금 holdout도 없어 외적 일반화를 주장할 수 없다.

## 7. Conclusion

본 연구는 전방 주차 인식용 RF-DETR Segmentation Large를 대상으로 precision, pruning, 2:4 sparsity, 구조 축소, 입력 해상도와 mixed precision을 포함하는 26개 경량화 후보를 구성하고 정적 선별과 목표 장비 실측을 분리한 3단계 평가 절차를 적용하였다. 총 4,466장의 데이터는 촬영 세션 단위로 train 3,625장, validation 404장과 fixed comparative benchmark 437장으로 분리했으며 자동 감사에서 split 간 동일 이미지, 세션 중복과 COCO 참조 오류는 발견되지 않았다.

Stage 1에서는 26개 후보 중 22개가 통과하였다. Stage 2에서는 21개 TensorRT engine의 정확도와 실행 성능을 측정했고 Q03은 INT4 parser 호환성 오류로 build에 실패하였다. B01을 포함한 11개가 정확도 보존 gate를 만족하였다. Stage 3의 세 목적 Pareto 분석에서는 C01과 R01이 비지배해로 확인되었다. C01은 baseline 수준의 Mask AP를 유지하면서 median latency와 engine 크기를 각각 약 44.79%와 49.95% 줄였고, R01은 정확도 gate를 만족하면서 가장 짧은 24.010 ms의 median latency를 기록하였다.

따라서 균형·Mask AP·engine 크기를 우선하는 배포에는 C01을, latency를 우선하는 배포에는 R01을 권장한다. 두 후보는 서로 다른 목적에서 장점을 가지므로 단일 절대 우승자로 합치지 않는다. 향후 연구에서는 별도 장소와 조건의 잠금 holdout, 다중 학습 seed, 다른 GPU와 TensorRT 버전, ROS2 전체 end-to-end latency 및 전력·thermal 조건 비교를 통해 결과의 일반화 범위를 넓힐 필요가 있다.

## Acknowledgement

`[GitHub 근거 없음: 지원기관, 과제번호와 공식 국·영문 사사 문구를 증빙하는 자료 추가 후 삽입]`

## References 작업 메모

GitHub README에 정리된 RF-DETR, DETR, DINO, Mask DINO, pruning, SmoothQuant, AWQ, NVIDIA 2:4 sparsity, TensorRT quantization, COCO와 NSGA-II 자료를 제출 양식에 맞게 검증한다. 저자, 제목, 학회·저널, 연도, 권호, 페이지와 DOI를 확인하고 본문 최초 인용 순서로 재배열해야 한다. 기존 ASK 2026 논문의 정확한 서지정보는 GitHub 근거가 없어 자료 제공 뒤 추가한다.

## 작성 근거

- [Stage 3 최종 분석](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/reports/stage3-final-analysis.md)
- [Stage 3 논문 근거 연결표](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/paper/stage3-evidence-map.md)
- [Stage 2 후보별 결과](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/reports/notebook-stage2-results.md)
- [전체 후보 판정 CSV](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/results/stage3-paper-candidate-decisions.csv)
- [Stage 3 Pareto JSON](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/results/stage3-pareto.json)
- [측정 부하 통제](https://github.com/ksk192830/kips-ai-model-lightweighting/blob/main/docs/reports/measurement-load-control.md)

