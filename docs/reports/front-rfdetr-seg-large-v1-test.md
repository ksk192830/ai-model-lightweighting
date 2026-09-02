# RF-DETR Seg Large 무증강 세션 분리 Test 결과

## 평가 조건

- 모델: `front-rfdetr-seg-large-v1/checkpoint_best_total.pth`
- 선택 기준: validation EMA segmentation mAP50:95 최고 checkpoint
- 선택 epoch: 12번째 epoch(내부 index 11)
- test 데이터: `front_session_split_v1/test`, 437장, 독립 촬영 세션 2개
- 학습·test 간 동일 이미지 및 촬영 세션 중복: 0개
- 평가 코드: `scripts/evaluation/evaluate_coco_rfdetr_pth.py`
- COCO prediction threshold: 0.001
- semantic mIoU threshold: 0.25
- 평가 객체 범주: `out_line`, `parking_lot`, `parking_space`
- 실행 장치: NVIDIA GeForce RTX 5060 Ti
- checkpoint SHA-256:
  `769ee97e38a2c6665bd664a9e035f5649d9f0a45c2bfe20172fe92ee545e5620`

빈 상위 범주 `0: front`는 annotation이 없으므로 객체별 평가에서 제외했다.
Test는 모델·checkpoint·threshold 선택에 사용하지 않고 학습 종료 후 한 번만
실행했다.

## 결과

| 지표 | 결과 |
|---|---:|
| BBox AP50:95 | 0.7378 |
| BBox AP50 | 0.9285 |
| BBox AP75 | 0.7918 |
| Mask AP50:95 | 0.6010 |
| Mask AP50 | 0.8402 |
| Mask AP75 | 0.6470 |
| Semantic mIoU | 0.7122 |

### 객체 크기별 AP50:95

| 크기 | BBox AP | Mask AP |
|---|---:|---:|
| Small | 0.3123 | 0.0550 |
| Medium | 0.6379 | 0.4875 |
| Large | 0.8418 | 0.7124 |

### 클래스별 semantic IoU

| 클래스 | IoU |
|---|---:|
| out_line | 0.4588 |
| parking_lot | 0.7920 |
| parking_space | 0.8859 |

## 해석 범위

이 결과는 무증강 원본을 촬영 세션 단위로 분리한 내부 test 결과다. 기존
Roboflow random split 446장으로 평가한 과거 결과와 test 구성이 다르므로 수치를
직접 증감 비교하지 않는다. 현재 가장 큰 약점은 small-object mask AP(0.0550)와
`out_line` semantic IoU(0.4588)다. 또한 test가 동일 수집 환경의 두 촬영
세션으로 구성되므로 새로운 주차장·카메라·날씨에 대한 외부 일반화 결과로
해석하지 않는다.

원시 결과는
`results/coco-evaluation/front-rfdetr-seg-large-v1-test.json`에 저장한다.
