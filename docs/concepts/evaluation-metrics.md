# 모델 경량화 평가 개념 정리

이 문서는 평가 결과를 읽고 비교하기 위한 개념 문서다. 실행 절차와 현재
현재 실행 기준은 [신규 Front 경량화 2차 계획](../guides/front-lightweighting-round-2.md)을
참고한다.

## 1. 평가를 나누는 세 축

모델 경량화 평가는 한 숫자로 결정하지 않는다.

1. 정확도: 객체와 mask를 얼마나 올바르게 예측하는가
2. 실행 성능: 얼마나 빠르고 안정적으로 추론하는가
3. 자원 효율: 모델 크기와 GPU 메모리가 얼마나 작은가

한 축만 좋아진 모델은 다른 축에서 큰 손실이 있을 수 있다. 따라서 같은
평가 데이터와 동일한 측정 조건에서 여러 지표를 함께 본다.

## 2. bbox 정확도

### Precision

모델이 객체라고 예측한 것 중 실제 정답인 비율이다. 낮으면 오탐이 많다.

### Recall

실제 객체 중 모델이 찾아낸 비율이다. 낮으면 미탐이 많다.

현재 자동 평가의 기본 산출물은 COCO AP/AR와 semantic mIoU다.
특정 confidence의 단일 precision/recall은 임계값 선정 실험을 별도로
수행했을 때만 보고하며, 현재 결과 JSON에 없는 값을 논문에 기재하지
않는다.

### mAP50

bbox IoU가 0.50 이상이면 정답으로 인정해 계산한 클래스 평균 AP다. 객체의
대략적인 위치를 찾는 능력을 보여준다.

### mAP50-95

IoU 0.50부터 0.95까지 0.05 간격의 AP를 평균한 COCO 지표다. 위치와 크기가
정밀해야 높은 점수를 얻으므로 모델 간 bbox 품질 비교의 기본 지표로 사용한다.

## 3. segmentation 정확도

### Mask AP

인스턴스 mask에 대해 IoU 0.50~0.95의 AP를 평균한다. 객체 검출, 클래스,
mask 경계, 인스턴스 분리를 함께 평가하는 핵심 지표다.

### Mask AP50 / AP75

- AP50: 대략적인 객체 영역을 찾는 능력
- AP75: 더 정밀한 mask 경계와 형태 품질

AP50은 높고 AP75가 크게 낮으면 객체 영역은 찾지만 경계가 부정확하다고
해석한다.

### Mask mIoU

클래스별로 전체 데이터셋의 intersection을 union으로 나눈 뒤 평균한다.
confidence 0.25 이상의 mask를 클래스별 semantic mask로 합쳐 계산한다.

mIoU는 큰 영역의 전체 겹침에 강하고, Mask AP는 개별 인스턴스 분리와
confidence 순위에도 민감하다. 따라서 mIoU가 비슷해도 Mask AP가 낮을 수
있으며, 두 지표는 서로 대체할 수 없다.

## 4. 속도와 자원 지표

### FPS와 latency

- FPS: 1초에 처리할 수 있는 이미지 수. 높을수록 좋다.
- 평균 latency: 이미지 한 장의 평균 처리 시간. 낮을수록 좋다.
- P95 latency: 전체 실행의 95%가 이 시간 이내에 끝난다는 의미다.
- P99 latency: 드문 tail 지연을 더 엄격하게 보여준다.
- 반복 median CV: 동일 32장 측정을 3회 반복했을 때 median의 상대 변동성이다.
- 반복 median 95% t 구간: 세 반복의 median 평균에 대한 불확실성 범위다.

평균값이 같아도 P95가 크면 간헐적인 지연이 발생한다. 실시간 시스템에서는
FPS뿐 아니라 P95/P99도 함께 확인한다. 반복 median CV가 5%를 넘으면 후보를
즉시 탈락시키지 않고 전력 모드, 열 throttling과 백그라운드 부하를 점검한 뒤
재측정 대상으로 표시한다.

### 모델 크기와 GPU 메모리

파일 크기는 저장·전송·로딩 비용에 영향을 주며 GPU peak 메모리는 실제
하드웨어에서 실행 가능한지를 결정한다. 파일이 작다고 항상 빠른 것은 아니고,
희소화나 INT8이 실제 지원 커널을 사용하지 못하면 기대한 속도 이득이 없다.

## 5. confidence를 두 개로 사용하는 이유

- AP 평가 confidence: 0.001
- semantic mIoU confidence: 0.25

AP는 precision-recall 곡선 전체가 필요하므로 낮은 confidence에서 예측을
충분히 수집한다. 반면 semantic mIoU는 사용할 예측만 남기는
0.25를 적용한다. AP를 0.25로 평가하면 저신뢰 예측이 사라져
곡선이 잘리고 모델 성능이 왜곡될 수 있다.

## 6. 클래스 ID 정렬

모델과 데이터셋이 같은 클래스 이름을 사용해도 숫자 ID가 다를 수 있다.
숫자 ID를 그대로 비교하면 완전히 다른 클래스로 채점될 수 있으므로 이름을
기준으로 일대일 매핑해야 한다.

이 프로젝트의 Ultralytics 모델에는 다음 매핑이 필요하다.

```text
model 0(out_line)      -> COCO 1(out_line)
model 1(parking_lot)   -> COCO 2(parking_lot)
model 2(parking_space) -> COCO 3(parking_space)
```

## 7. Pareto front

Pareto front는 한 지표를 개선하려면 다른 지표를 포기해야 하는 모델들의
집합이다. 예를 들어 정확도와 FPS가 모두 다른 모델보다 낮은 모델은
지배됐다고 하며 선택 근거가 약하다.

- mAP–FPS: 정확도와 처리량의 균형
- mAP–latency: 정확도와 응답시간의 균형
- mAP–size: 정확도와 저장·배포 비용의 균형

Pareto에 포함됐다는 사실만으로 최적 모델이 되는 것은 아니다. YOLO처럼 매우
작아서 front에 포함되지만 정확도가 크게 낮은 선택지도 있다. 실제 요구사항의
최소 정확도, 최소 FPS, 최대 메모리를 먼저 정한 뒤 front에서 선택한다. 본
실험에서는 먼저 B01 대비 bbox AP·mask AP·mIoU 보존 gate와 측정 유효성을
적용한 뒤 Mask AP를 최대화하고 median latency와 engine size를 최소화한다.
BBox AP, mIoU, P95 latency와 GPU memory는 보조지표로 보고한다.

## 8. 공정한 비교 조건

동일 계열 경량화 효과를 주장하려면 다음 조건이 같아야 한다.

- 데이터셋과 split
- 입력 해상도
- batch size
- AP 수집 confidence와 semantic mIoU confidence
- 평가 API와 클래스 매핑
- GPU, 전력 모드, warm-up과 반복 횟수

RF-DETR/DETR 평가에서는 현재 NMS를 적용하지 않고 query 예측을
score 순으로 직접 평가한다. 따라서 YOLO와 같은 NMS 기반 모델과 비교할
때는 postprocess 차이를 반드시 밝힌다.

R01은 432 입력이고 YOLO는 512 입력이며 모델 계열도 다르다. 그래프에는
참고점으로 포함할 수 있지만 RF-DETR 504 후보와 완전히 동일 조건의
경량화 실험으로 해석해서는 안 된다.

## 9. 이번 실험의 해석 원칙

과거 모델 결과는 제거했으므로 후보별 우열을 미리 가정하지 않는다. 신규
baseline에서 다시 측정한 정확도, latency, FPS, engine 크기와 메모리만 사용해
Pareto 후보를 선정한다.

## 10. 이번 실험의 자동 수용 기준

기준은 `configs/experiments/defaults.yaml`을 단일 원본으로 삼으며
자동 파이프라인이 직접 읽는다. 이 값은 보편적 통계 유의성 기준이 아니라
본 프로젝트의 사전 정의 engineering gate다.

- B01 대비 bbox AP·mask AP 절대 하락 허용치: 각 0.01
- B01 대비 semantic mIoU 절대 하락 허용치: 0.02
- S02를 S01 대신 선택하는 추가 조건: bbox/mask AP 하락 0.005 이하,
  mIoU 하락 0.01 이하, TensorRT median latency 5% 이상 감소
- INT8: FP16 대비 median latency 10% 이상 감소와 B01 대비 AP 보존
- 2:4 sparse: dense control 대비 median latency 10% 이상 감소,
  sparse tactic 선택 근거, AP 보존을 모두 요구

## 11. 고정 반복 benchmark의 한계

현재 437장 split은 후보 개발 과정에서 반복 사용했다. 따라서
같은 데이터에서의 공정한 상대 비교용 `fixed benchmark` 이지만, 더 이상
완전히 손대지 않은 confirmatory test로 표현하지 않는다. 외적
일반화를 강하게 주장하려면 별도 촬영 세션을 추가로 확보해 단 한 번
최종 평가해야 한다.
