# 추가 경량화 후보 및 1차 평가 설계

> 주의: 이 문서의 4~6단계 정확도·동등성 gate를 1차에 포함한 구 설계는
> 2026-09-02에 폐기했다. 현재 1차는 정적평가만 수행하며, 권위 있는 결과는
> [`results/stage1-static-evaluation.md`](../../results/stage1-static-evaluation.md)이다.
> 이하 내용은 추가 graph 후보의 선정 배경과 기존 데스크탑 진단 기록으로만 보존한다.

## 목적

기존 대표 후보만으로는 경량화 강도에 따른 변화와 정확도-효율 Pareto 경계를 설명하기 어렵다. 따라서 동일한 B01 기준 모델과 고정 test set을 사용해 해상도, 비정형 sparsity, 구조 축소 강도, 구조-해상도 결합의 네 축을 보강한다.

## 추가 후보

| ID | 축 | 설정 | 비교 목적 |
|---|---|---|---|
| U01 | 비정형 pruning | global magnitude 10% | U02(30%), U03(50%)와 sparsity dose-response 구성 |
| U03 | 비정형 pruning | global magnitude 50% | 높은 sparsity에서 정확도 붕괴 및 dense ONNX 한계 확인 |
| R02 | 입력 해상도 | 480×480 | B01(504)과 R01(432) 사이의 보수적 지점 확인 |
| R03 | 입력 해상도 | 384×384 | R01보다 공격적인 축소의 효율-정확도 한계 확인 |
| S04 | 구조적 pruning | FFN dimension 40% 축소 | S03(20%)과 구조 축소 강도 비교 |
| C03 | 결합 | S01 decoder 1층 축소 + 432×432 | 구조 축소와 입력 축소의 누적 효과 확인 |
| C04 | 결합 | S01 decoder 1층 축소 + 480×480 | 결합 후보의 보수적인 Pareto 지점 확인 |

모든 후보는 새로운 학습 데이터 분할을 만들지 않는다. B01과 동일한 고정 test set 437장을 사용하며, C03/C04는 이미 복구 학습과 정확도 검증을 통과한 S01 checkpoint에서 입력 크기만 달리하여 export한다.

## 사전 고정한 1차 평가 절차

1. 후보 checkpoint 또는 source checkpoint에서 FP32 ONNX를 export한다.
2. ONNX checker, initializer element 수, graph node 수, dense MACs/FLOPs 하한 추정치, 파일 크기를 기록한다.
3. 아래 효율 gate 중 하나 이상을 충족하는지 판단한다.
4. 동일한 test set 437장에서 PTH 정확도를 측정한다.
5. B01 대비 정확도 gate를 통과한 후보에 대해 production-image ONNX 동등성을 확인한다.
6. 효율·정확도·동등성 gate를 모두 통과한 후보만 ONNX 전체 test set 평가로 재확인한다.

효율 gate는 다음 중 하나 이상의 B01 대비 상대 감소를 요구한다.

- ONNX 파일 크기 5% 이상 감소
- ONNX graph node 수 5% 이상 감소
- 추정 dense MACs 5% 이상 감소

정확도 gate는 B01 대비 다음 최대 절대 하락 한계를 동시에 만족해야 한다.

- bbox AP: 0.01
- mask AP: 0.01
- semantic mIoU: 0.02

ONNX 동등성은 고정 seed 42로 뽑은 production image 10장을 사용한다. 활성 query membership/class 일치율, box/score/mask 오차, binary mask 일치율의 기준은 `configs/experiments/defaults.yaml`에 사전 고정되어 있다.

RF-DETR export는 현재 모델의 patch/window 구조상 입력의 가로·세로가 24의 배수여야 한다. 따라서 해상도 지점은 임의의 중간값 468/396이 아니라, 같은 비교 목적을 만족하는 가장 가까운 유효 지점 480/384로 고정했다.

비정형 pruning은 sparse runtime 없이 dense ONNX 크기나 연산량이 줄지 않을 가능성이 높다. 그래도 U01/U02/U03의 정확도를 같은 방식으로 측정해 sparsity 강도에 따른 정확도 변화와 “0 weight 비율만으로는 dense 배포 비용이 줄지 않는다”는 대조 결과를 남긴다.

## 실행 결과

`Δ` 정확도는 B01 대비 candidate minus B01이다. 효율 수치는 B01 대비 상대 감소율이며, `—`는 앞 단계 탈락으로 실행하지 않은 항목이다.

| ID | 주요 효율 변화 | Δ bbox AP | Δ mask AP | Δ mIoU | 10장 ONNX 동등성 | 437장 PTH-ONNX parity | 1차 판정 |
|---|---:|---:|---:|---:|---|---|---|
| U01 | dense MACs 0.00% | -0.000260 | +0.008710 | +0.006963 | — | — | 탈락: dense 효율 없음 |
| U03 | dense MACs 0.00% | -0.737777 | -0.601016 | -0.712200 | — | — | 탈락: 효율 없음·정확도 붕괴 |
| S04 | ONNX size -6.53%, MACs -0.79% | -0.216339 | -0.076490 | -0.054941 | — | — | 탈락: 정확도 |
| R02 | MACs -10.96% | -0.006526 | -0.000623 | -0.008550 | 통과 | 실패: mIoU 차이 0.008998 | 탈락: 전체 parity |
| R03 | MACs -46.25% | -0.009132 | -0.014341 | +0.016842 | — | — | 탈락: mask AP |
| C03 | nodes -8.04%, MACs -30.76% | -0.007054 | -0.007219 | +0.027037 | 통과 | 통과 | **통과** |
| C04 | nodes -8.04%, MACs -11.51% | -0.004736 | -0.001568 | +0.016748 | 통과 | 통과 | **통과** |

따라서 이번 추가 후보 7개 중 C03과 C04 두 개를 TensorRT 2차 평가 대상으로 유지한다. C03은 더 큰 연산량 감소를 제공하는 공격적 결합 후보이고, C04는 AP 하락폭이 더 작은 보수적 결합 후보다. R02는 PTH 정확도만 보면 유효하지만 전체 데이터셋 변환 parity를 통과하지 못했으므로 사전 기준에 따라 제외한다.

전체 기존·신규 후보의 원수치, 절대·상대 변화량 및 평가 산출물 감사 결과는 `results/round2-candidate-summary.md`와 동일 이름의 CSV에 자동 생성된다.

## 이번 단계에서 분리한 후보

TensorRT FP16/INT8(B02/B03/C01/C02/M02)과 ModelOpt 기반 Q01-Q07/W01은 현재 FP32 ONNX 후보의 1차 선별과 다른 실행 환경 및 런타임 근거가 필요하다. 특히 ModelOpt 후보는 현재 환경에 의존성이 설치되어 있지 않아, 이번 1차 표에 결과를 섞지 않고 TensorRT 엔진 생성 단계에서 별도의 2차 실험군으로 평가한다.

## 결과 해석 원칙

- test set은 후보 선택을 위한 고정 반복 benchmark이므로 최종 일반화 성능을 독립적으로 주장하지 않는다.
- 정적 MACs/FLOPs는 일부 dense 연산의 하한 추정치이며 실제 지연시간을 대체하지 않는다.
- 1차 통과는 TensorRT에서 빠르다는 의미가 아니라, 엔진 평가에 올릴 근거가 충분하다는 의미이다.
- 후보 탈락도 음성 결과로 보존하며 임계값을 결과 확인 후 변경하지 않는다.
