# 4. Experimental Results (초안)

> 원본 데이터: `results/paper_metrics.csv` → 정리본: `results/results_final.csv`
> 재현: `.venv/bin/python scripts/paper_results.py` (표의 수치와 `figures/` 그래프를 재생성)
>
> 평가 조건: RTX 4050 Laptop GPU(6 GB), batch 1, 순차 실행, warm-up 10회 /
> timed 100회 × 3 blocks, 운영 confidence 0.25, AP 평가 confidence 0.001,
> NMS IoU 0.7, pycocotools COCOeval, 테스트셋 296장(정답 객체 423개).
> 기준 모델(baseline)은 B01(TensorRT FP32, 504×504)이다.

## Table 1. 성능 비교표

| Model | Input | Precision | Recall | mAP50 | mAP50-95 | FPS | Latency(ms) | P95(ms) | Size(MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline FP32 (B01) | 504 | 0.9308 | 0.9858 | 0.9836 | 0.8310 | 18.91 | 52.47 | 56.86 | 129.28 |
| FP16 (B02) | 504 | 0.9414 | 0.9882 | 0.9857 | 0.8333 | 32.65 | 30.64 | 31.60 | 67.90 |
| INT8 PTQ (B03) | 504 | 0.9371 | 0.9858 | 0.9853 | 0.8352 | 31.86 | 31.55 | 32.88 | 67.98 |
| Structured+FP16 (C01) | 504 | 0.9455 | 0.9835 | 0.9886 | 0.8378 | 32.33 | 31.00 | 32.14 | 64.70 |
| 2:4 Control (M01) | 504 | 0.9318 | 0.9693 | 0.9700 | 0.7961 | 31.64 | 31.72 | 33.01 | 67.93 |
| 2:4 Sparse (M02) | 504 | 0.9297 | 0.9693 | 0.9734 | 0.8005 | 31.91 | 31.47 | 32.87 | 62.40 |
| Decoder-Pruned (S01) | 504 | 0.9432 | 0.9811 | 0.9877 | 0.8390 | 19.78 | 50.70 | 52.11 | 123.28 |
| Low-Res FP16 (R01) | 432 | 0.9263 | 0.9811 | 0.9792 | 0.8239 | 36.10 | 27.89 | 29.52 | 68.05 |

메인 비교표는 동일 RF-DETR Segmentation Large에서 파생된 TensorRT 후보로
한정한다. 기존 Ultralytics `parking_front.pt`는 아키텍처와 입력 크기가
다르므로 정량 경량화 비교에서 제외하고 참고 모델로만 관리한다.

## Table 2. 경량화 효과표 (Baseline B01 대비)

| Model | Size Reduction | FPS Increase | Latency Reduction | ΔmAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| FP16 (B02) | 47.5% | +72.7% | 41.6% | +0.0024 |
| INT8 PTQ (B03) | 47.4% | +68.5% | 39.9% | +0.0042 |
| Structured+FP16 (C01) | 50.0% | +71.0% | 40.9% | +0.0068 |
| 2:4 Control (M01) | 47.5% | +67.3% | 39.6% | −0.0349 |
| 2:4 Sparse (M02) | 51.7% | +68.8% | 40.0% | −0.0305 |
| Decoder-Pruned (S01) | 4.6% | +4.6% | 3.4% | +0.0081 |
| Low-Res FP16 (R01)‡ | 47.4% | +90.9% | 46.9% | −0.0071 |

‡ 입력 432×432 조건. 해상도 감소 효과가 포함되어 있어 504 후보와 동일
조건 비교가 아니다.

## Figures

| 파일 | 내용 |
| --- | --- |
| `figures/fps_comparison.png` | 모델별 FPS |
| `figures/latency_comparison.png` | 모델별 평균 지연시간 |
| `figures/size_comparison.png` | 모델별 아티팩트 크기 |
| `figures/map_comparison.png` | 모델별 mAP50 / mAP50-95 |
| `figures/mask_comparison.png` | 모델별 Mask AP / AP50 / AP75 |
| `figures/mask_miou_comparison.png` | 모델별 Mask mIoU |
| `figures/pareto_map_fps.png` | mAP50-95 vs FPS Pareto (front: S01, C01, B02, R01) |
| `figures/pareto_map_size.png` | mAP50-95 vs Size Pareto (front: M02, C01, S01) |
| `figures/pareto_map_latency.png` | mAP50-95 vs Latency Pareto (front: R01, B02, C01, S01) |

---

## 4.1 Baseline Performance

기준 모델인 TensorRT FP32 엔진(B01)은 504×504 입력에서 mAP50 0.9836,
mAP50-95 0.8310의 정확도를 보였으나, 처리 속도는 18.91 FPS(평균 지연
52.47 ms, P95 56.86 ms)로 실시간 처리 기준인 30 FPS에 미치지 못했다.
모델 크기는 129.28 MB, 추론 시 GPU 메모리 피크는 486 MB로, 6 GB급
노트북 GPU에서는 동작하지만 임베디드 환경 배포에는 부담이 되는 수준이다.
따라서 정확도를 유지하면서 처리 속도와 메모리 사용량을 개선하는 경량화가
필요하다.

## 4.2 Lightweight Model Performance

FP16 양자화(B02)는 baseline 대비 모델 크기를 47.5% 감소시키고 FPS를
72.7% 향상(18.91 → 32.65 FPS)시키면서도 mAP50-95가 0.8310에서 0.8333으로
오히려 소폭 상승하여, 정확도 손실 없이 실시간 기준(30 FPS)을 달성했다.
평균 지연시간은 52.47 ms에서 30.64 ms로 41.6% 감소했고, GPU 메모리
피크도 486 MB에서 252 MB로 절반 가까이 줄었다.

INT8 PTQ(B03)는 크기 47.4% 감소, FPS 68.5% 향상으로 FP16과 유사한 속도
이득을 보였으며, mAP50-95는 0.8352로 baseline보다 +0.0042 높았다. 다만
본 모델의 TensorRT 엔진에서는 INT8이 FP16 대비 추가적인 크기·속도 이점을
보이지 않았는데(67.98 MB vs 67.90 MB, 31.86 vs 32.65 FPS), 이는 네트워크
주요 연산이 FP16 커널로 폴백되었기 때문으로 분석된다.

구조적 pruning과 FP16을 결합한 C01은 크기 50.0% 감소, FPS 71.0% 향상과
함께 전체 후보 중 두 번째로 높은 mAP50-95(0.8378, baseline 대비 +0.0068)를
기록하여 정확도·속도·크기 세 축 모두에서 균형이 가장 좋았다. 구조적
pruning 단독(S01, FP32)은 mAP50-95 0.8390으로 가장 높은 정확도를 보였으나
FP16 변환 없이는 속도 이득이 4.6%에 그쳐, pruning의 실질적인 배포 이득은
FP16과 결합할 때 발생함을 확인했다.

2:4 구조적 희소화(M02)는 가장 큰 크기 감소(51.7%, 62.40 MB)를 달성했지만
mAP50-95가 0.8005로 baseline 대비 0.0305 하락하여 정확도 손실이 가장
컸다. 희소화 전 dense 대조군(M01, mAP50-95 0.7961)과 비교하면 정확도
하락의 대부분은 희소화 자체가 아니라 2:4 제약을 위한 fine-tuning 과정에서
발생한 것으로 나타났다.

입력 해상도를 432×432로 낮춘 R01은 TensorRT 후보 중 가장 빠른 36.10 FPS
(baseline 대비 +90.9%, 지연 46.9% 감소)를 기록했으며, mAP50-95 손실은
0.0071로 제한적이었다. 해상도 축소는 재학습 없이 속도를 확보하는 유효한
수단이지만, 504 후보와 입력 조건이 다르므로 동일 조건 비교에서는 제외한다.

## 4.3 Trade-off Analysis

mAP50-95–FPS 평면에서 Pareto front는 S01(0.8390, 19.78 FPS),
C01(0.8378, 32.33 FPS), B02(0.8333, 32.65 FPS), R01(0.8239, 36.10 FPS)로
구성된다(Fig. pareto_map_fps). Baseline(B01)은 B02가 정확도와 속도 모두에서
앞서므로 지배(dominated)되며, 2:4 계열(M01, M02)은 유사한 속도의 C01보다
정확도가 0.037 이상 낮아 front에 들지 못했다.

mAP50-95–크기 평면의 Pareto front는 M02(62.40 MB), C01(64.70 MB),
S01(123.28 MB)이다(Fig. pareto_map_size). C01은 M02보다 2.3 MB 크지만
mAP50-95가 0.0373 높아, 크기 축에서도 실질적인 선택지는 C01로 수렴한다.

mAP50-95–지연시간 평면에서는 R01(27.89 ms), B02(30.64 ms), C01(31.00 ms),
S01(50.70 ms)이 front를 구성한다(Fig. pareto_map_latency). C01은 B02 대비
0.36 ms의 지연 증가로 +0.0045의 mAP50-95를 얻는 위치에 있다.

종합하면 C01(Structured+FP16)이 세 Pareto front에 모두 포함되는 유일한
504×504 후보로, 정확도 최우선이면 S01, 지연시간 최우선이면 R01, 그 외의
일반적인 배포 조건에서는 C01이 합리적인 선택이다.

## 4.4 Discussion

첫째, FP16 변환은 본 모델에서 사실상 무손실 경량화로 작동했다. 크기 절반,
속도 1.7배의 이득에 정확도 하락이 없었으며(오히려 +0.0024), 이는 검출
모델의 활성값·가중치 분포가 FP16 표현 범위 내에 있음을 시사한다. 모든
경량화 조합의 기본 단계로 FP16을 채택하는 것이 타당하다.

둘째, INT8 PTQ는 FP16 대비 추가 이득이 없었다. 크기와 속도가 FP16과
동일 수준에 머문 것은 TensorRT가 상당수 레이어를 FP16 커널로 처리했기
때문으로, transformer 기반 검출기에서 INT8의 이득을 얻으려면 QAT
(quantization-aware training) 등 추가 조치가 필요하다.

셋째, 구조적 pruning의 이득은 정밀도 변환과 결합할 때 실현된다. S01
(pruning 단독, FP32)은 정확도는 가장 높았지만 속도 이득이 4.6%에
불과했고, 동일한 pruning에 FP16을 결합한 C01은 71.0%의 속도 이득을
보였다. 반면 2:4 희소화는 fine-tuning 과정의 정확도 손실(mAP50-95
−0.0305~−0.0349)이 크기 이득(51.7%)을 상쇄하여, 본 과제와 같이 높은
정확도가 요구되는 주차 보조 시나리오에는 부적합하다.

넷째, 기존 Ultralytics 모델(`parking_front.pt`)의 최초 mAP50 0.0084와
mAP50-95 0.0017은 모델 성능이 아니라 평가 class-ID 불일치로 발생한 무효
수치다. COCO 정답은 빈 배경 범주 `0: front`를 포함해 실제 객체 ID가
1(`out_line`), 2(`parking_lot`), 3(`parking_space`)인 반면, Ultralytics
모델 출력은 동일한 세 이름을 0, 1, 2로 사용했다. 기존 평가기는 이름을
확인하지 않고 숫자 ID를 직접 비교해 모든 예측을 한 클래스씩 어긋나게
채점했다. 수정된 평가기는 클래스 이름으로 `{0: 1, 1: 2, 2: 3}` 매핑을
검증·적용한다. 재평가 시 기존 속도 측정을 반복하지 않고 다음 명령으로
정확도 열만 갱신한다.

```bash
python scripts/evaluate.py \
  --model models/parking_front.pt \
  --backend ultralytics \
  --data data/labeled_test/front \
  --imgsz 512 --conf 0.25 --eval-conf 0.001 \
  --device cuda:0 \
  --output results/paper_metrics.csv \
  --update-existing
```

수정 후 결과도 모델 계열과 입력 크기가 RF-DETR TensorRT 후보와 다르므로
메인 성능표와 Pareto 분석에서는 제외하고 참고 결과로만 해석한다.

마지막으로 본 평가는 전방 카메라 모델과 단일 GPU(RTX 4050 Laptop)에
한정되므로, 후방 모델과 실제 배포 대상 하드웨어에서의 재평가가 향후 과제다.
