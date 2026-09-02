# Front RF-DETR 경량화 후보 정량 비교

## 결론

2026-09-02 산출물 기준으로 S01(decoder 1개 제거)은
6 epoch 복구 학습과 조기 종료 후 B01 대비 bbox AP 차이를
-0.000411(-0.0557%)로 억제하고, mask AP는 +0.001735(+0.2886%),
semantic mIoU는 +0.021721(+3.0499%) 개선했다. ONNX node와 파일 크기는
각각 8.0395%, 4.7447% 감소했고 실제 test 이미지 10장의 production
동등성도 통과했다. 따라서 S01을 구조 pruning 선정 후보로 유지하고
notebook TensorRT FP16/INT8 실측 단계로 넘긴다.

R01(입력 504→432)은 MACs/FLOPs 30.2021% 감소와 비교적 작은 정확도
하락을 제공하므로 최대 연산량 절감 후보로 별도 유지한다. S01은 정확도
유지를, R01은 연산량 절감을 우선하는 서로 다른 Pareto 후보이며, 최종
선택은 노트북에서 동일 조건으로 측정한 latency·FPS·memory 결과 후 확정한다.

## 비교 기준

- 공통 source checkpoint는
  `artifacts/training/front-rfdetr-seg-large-v1/checkpoint_best_total.pth`이며
  SHA-256은
  `769ee97e38a2c6665bd664a9e035f5649d9f0a45c2bfe20172fe92ee545e5620`이다.
- 정확도는 `data/training/front_session_split_v1/test` 437장, detection threshold
  0.001, semantic mIoU threshold 0.25로 평가했다. R01만 입력 해상도가 432이고
  나머지는 504이다.
- B01, U02, S01은 RTX 5060 Ti에서, R01과 S02는 CPU에서 평가했다. 동일한
  PyTorch evaluator와 데이터셋을 사용했지만, 이 표는 단일 실행 결과이며
  장치 간 반복 측정 오차나 신뢰구간은 제공하지 않는다.
- 파라미터 수는 ONNX initializer 원소 수이다. 구조 pruning JSON의 PyTorch 전체
  파라미터 감소율은 S01 4.3321%, S02 8.6643%, S03 3.0003%이며, ONNX
  initializer 감소율과 정의가 다르므로 표에서는 섞지 않았다.
- MACs/FLOPs는 동일한 정적 분석기의 추정값이다. FLOPs는 이 분석에서
  `2 × MACs`로 정의되므로 감소율이 동일하다. 이 값은 TensorRT latency 또는
  sparse tactic 사용을 직접 증명하지 않는다.
- 정확도 괄호 안 값은 `후보 - B01`의 절대 delta이다. N/A는 해당 전체 test
  평가 또는 최종 산출물이 아직 없다는 뜻이며 추정값으로 대체하지 않았다.

## 방법과 현재 상태

| ID | 적용 방법 | 정량적 구조/희소성 변화 | registry 상태 | 판정 |
|---|---|---|---|---|
| B01 | pruning 없음, 504×504 FP32 기준 모델 | 기준 | `onnx-exported` | 기준선 유지 |
| U02 | 전역 magnitude 비정형 pruning 30% | pruning 대상 weight sparsity 29.999999% | `static-analysis-rejected` | 탈락 |
| R01 | 입력 해상도 504×504→432×432; weight와 checkpoint는 B01과 동일 | 픽셀 수 26.5306% 감소; shape-dependent ONNX 상수도 감소 | `onnx-exported` | 채택 후보 |
| S01 | decoder 및 연결 segmentation block 5→4 | PyTorch 전체 파라미터 4.3321% 감소 | `onnx-exported`; recovery `completed` | 채택 후보 |
| S02 | decoder 및 연결 segmentation block 5→3 | PyTorch 전체 파라미터 8.6643% 감소 | `onnx-exported`; recovery `completed` | 정확도 탈락·속도 대조군 |
| S03 | decoder FFN 2048→1632(actual 20.3125% 축소) | PyTorch 전체 파라미터 3.0003% 감소 | `static-analysis-rejected` | 탈락 |
| M01 | NVIDIA 2:4 semi-structured pruning | shape 적격 200/206 layer, 적격 weight 99.4795%; pruning 대상 전체 sparsity 48.7724%; ONNX 적격 constant op 150/150 준수 | recovery `running` | 하드웨어 조건부 보류 |

R01 registry의 `precision: fp16`은 목표 TensorRT precision을 뜻한다. 현재 정적
분석과 아래 PyTorch 정확도는 FP16 TensorRT 엔진 결과가 아니며, 엔진 생성과
실측은 아직 남아 있다. M01도 현재 dense ONNX 자체의 MACs가 줄어든 모델이
아니므로 TensorRT verbose build log에서 sparse tactic 사용이 확인되어야만
가속 후보로 인정할 수 있다.

## 정적 분석 비교

| ID | ONNX initializer params (Δ) | nodes (Δ) | estimated MACs (Δ) | estimated FLOPs (Δ) | ONNX bytes (Δ) |
|---|---:|---:|---:|---:|---:|
| B01 | 32,537,747 (기준) | 2,127 (기준) | 54,083,668,224 (기준) | 108,167,336,448 (기준) | 130,677,849 (기준) |
| U02 | 32,537,747 (0.0000%) | 2,127 (0.0000%) | 54,083,668,224 (0.0000%) | 108,167,336,448 (0.0000%) | 130,677,820 (-0.000022%) |
| R01 | 32,358,035 (-0.5523%) | 2,127 (0.0000%) | 37,749,285,120 (-30.2021%) | 75,498,570,240 (-30.2021%) | 129,955,207 (-0.5530%) |
| S01 | 30,997,043 (-4.7351%) | 1,956 (-8.0395%) | 53,782,953,984 (-0.5560%) | 107,565,907,968 (-0.5560%) | 124,477,541 (-4.7447%) |
| S02 | 29,456,339 (-9.4703%) | 1,783 (-16.1730%) | 53,482,239,744 (-1.1120%) | 106,964,479,488 (-1.1120%) | 118,276,976 (-9.4897%) |
| S03 | 31,470,707 (-3.2794%) | 2,127 (0.0000%) | 53,870,676,224 (-0.3938%) | 107,741,352,448 (-0.3938%) | 126,409,640 (-3.2662%) |
| M01 | 32,537,747 (0.0000%) | 2,127 (0.0000%) | 54,083,668,224 (0.0000%) | 108,167,336,448 (0.0000%) | 130,677,820 (-0.000022%) |

R01의 initializer 0.5523% 감소는 학습 weight pruning이 아니라 입력 크기에
의존하는 ONNX 상수 shape 변화이다. 반대로 U02와 M01은 weight의 0 비율이
증가해도 dense ONNX tensor shape가 같기 때문에 파라미터 수, node 수,
MACs/FLOPs가 감소하지 않았다.

## 전체 test 정확도

| ID | 평가 단계 | bbox AP (Δ) | mask AP (Δ) | semantic mIoU (Δ) | 이미지 수 |
|---|---|---:|---:|---:|---:|
| B01 | 기준 checkpoint | 0.737791 (기준) | 0.601016 (기준) | 0.712200 (기준) | 437 |
| U02 | pruning 직후 | 0.096442 (-0.641349) | 0.082022 (-0.518994) | 0.032849 (-0.679351) | 437 |
| R01 | 432 입력, 재학습 없음 | 0.730661 (-0.007130) | 0.592974 (-0.008042) | 0.696878 (-0.015321) | 437 |
| S01 | **복구 후** | 0.737380 (-0.000411) | 0.602751 (+0.001735) | 0.733921 (+0.021721) | 437 |
| S02 | **복구 후** | 0.719456 (-0.018335) | 0.587589 (-0.013427) | 0.762691 (+0.050492) | 437 |
| S03 | 전체 test 미평가 | N/A | N/A | N/A | N/A |
| M01 | 전체 test 미평가 | N/A | N/A | N/A | N/A |

S01 recovery는 10 epoch 상한 중 6 epoch에서 early stopping되었고 regular
checkpoint가 선택되었다. 복구 후 bbox AP는 baseline과 실질적으로
동등한 수준이며 mask AP와 semantic mIoU는 상승했다. S02는 8 epoch에서
early stopping되었고 mIoU는 상승했지만 bbox AP와 mask AP가 사전 정의된
허용 하락폭 0.01을 넘어 최종 정확도 후보에서 제외했다.

## ONNX 변환 검증

10개 실제 test 이미지에 대한 production-image 동등성 정책 결과는 다음과 같다.
이 검사는 후보와 B01의 정확도 비교가 아니라, 각 후보 PTH와 그 후보 ONNX 사이의
출력 보존 검사이다.

| ID | production 동등성 | 활성 query | 해석 |
|---|---|---:|---|
| B01 | 통과 | 146 | 기준 ONNX 사용 가능 |
| U02 | 통과 | 49 | 변환은 정상이지만 모델 정확도 자체가 붕괴 |
| R01 | 통과 | 135 | 432 ONNX 사용 가능 |
| S01 | 통과 | 92 | 복구 후 최종 ONNX; membership·class·binary mask agreement 1.0 |
| S02 | 통과 | 144 | 복구 후 재수출한 최종 ONNX; membership·class·binary mask agreement 1.0 |
| S03 | 실패 | 60 | 활성 label 최대 오차 0.002126이 한계 0.0005 초과 |
| M01 | 실패 | 0 | confidence 0.05 이상 query가 없어 production 동등성을 판정할 수 없음 |

S03은 box 오차와 binary mask agreement는 기준을 충족했지만 label 오차
기준을 넘었다. M01은 복구 전 원형에서 활성 query가 없어 최종 판정을 보류한다.
ONNX 2:4 형식 자체는 150/150 적격 op에서 100% 준수했으나, 이는 예측 정확도나
sparse TensorRT tactic 선택을 보증하지 않는다.

## 채택·보류·탈락 근거

1. **R01 — 채택 후보.** MACs/FLOPs 30.2021% 감소가 현재 후보 중 가장 크고,
   bbox AP와 mask AP 하락은 각각 0.007130과 0.008042이다. 재학습 없이 바로
   FP16 TensorRT 실측 단계로 넘길 수 있다.
2. **S01 — 채택 후보.** node 수를 8.0395%, ONNX 파일 크기를 4.7447%
   줄이면서 bbox AP를 baseline 대비 -0.000411로 유지했고 mask AP와 mIoU는
   각각 +0.001735, +0.021721 개선했다. 복구 PTH와 ONNX 동등성 검증까지
   완료했으므로 FP16/INT8 TensorRT 실측 입력으로 채택한다.
3. **S02 — 정확도 기준 탈락, 속도 대조군 유지.** 8 epoch 복구 후
   bbox AP 0.719456, mask AP 0.587589, semantic mIoU 0.762691을 기록했다.
   S01보다 bbox AP 0.017924, mask AP 0.015162가 낮아 최종 structured
   후보에서는 제외한다. 다만 parameter 9.47%, node 16.17% 감소의 실제
   TensorRT 지연시간 효과를 정량화하기 위한 속도 대조군으로는 유지한다.
4. **M01 — 하드웨어 조건부 보류.** 2:4 pattern은 형식 검증을 통과했지만 dense
   graph의 파라미터, node, MACs가 전혀 줄지 않았고 복구 전 활성 query가 0이다.
   target GPU와 TensorRT가 sparse tactic을 실제 선택할 수 있을 때만 S02 뒤에
   복구 학습을 배정한다.
5. **U02 — 탈락.** 30% sparsity에도 dense graph 계산량 감소가 0%이고 bbox AP,
   mask AP, mIoU가 각각 0.641349, 0.518994, 0.679351 하락했다.
6. **S03 — 탈락.** FFN 자체를 20.3125% 줄였지만 전체 ONNX MACs 감소가
   0.3938%, node 감소가 0%에 불과하며 ONNX production 동등성도 실패했다.
7. **B01 — 기준선.** 모든 TensorRT FP32/FP16/INT8 및 후보의 정확도·latency
   delta를 계산하는 고정 대조군으로 유지한다.

## GPU 실행 우선순위

1. 최종 평가와 ONNX 동등성을 통과한 S01을 TensorRT FP16/INT8 엔진 생성
   큐로 보낸다.
2. S02 recovery와 최종 평가는 완료됐다. 정확도 gate에서 탈락했으므로 최종
   후보가 아니라 구조 감소 대비 실제 latency를 확인하는 대조군으로만 측정한다.
3. 현재 M01 recovery를 단독 실행 중이다. 완료 후 같은 모델로 dense tactic
   M01과 sparse tactic M02 engine을 만들고 build log의 tactic 근거를 확인한다.

R01에는 GPU 재학습 우선순위를 배정하지 않는다. 이미 측정 가능한 PTH 후보가
완성되어 있으므로 notebook TensorRT FP16 엔진 생성·benchmark 큐로 바로 보낸다.
U02와 S03은 현 수치상 추가 복구 학습의 비용을 정당화하지 못한다.

## 재현성과 해석 제한

- 원시 정확도 결과 SHA-256:
  - B01: `742252e55c46f2c76a7e41a8b9d8382314346b61602d140a54025028a6176633`
  - U02: `6ca23323b2ce3a3e0a33a144a8383ca3c8690e5aace7c70edd93e5559b54324e`
  - R01: `e5a80d1a6931a3a8d81f28956485b49cb0a2e9bb8100921e089c08349e7b14bb`
  - S01 복구 전: `5a8c6423ebad72ddfb0c6029cc33ab75e2cc9e3edef0d6bd3a6ee2ba5a2540d6`
  - S01 복구 후: `f1fdcac30d62cedf994ade45ec7484164d8ec080d22aacb4044c5fb564ab3406`
  - S02 복구 전: `4cf0ee009bc83fc01229df8d27ac5710c29dc5ca81c4cc8f147d8a35f0ab58fa`
- S01 최종 PTH/ONNX/동등성 기록 SHA-256은 각각
  `57e45c7553ec8829a53167b11ccc4da6964019f741b5762c087d82b2a899b821`,
  `b8e436175ef25eb6885a3e8c48813e7c0ab1ea3d7c2339ab496c701338dba6e1`,
  `c19c294995b354437003abbc2be4becd78bd004deb97b85ea141a84b8d00d4e6`이다.
- registry에 기록된 B01 baseline, R01, S01, S02 평가 hash는 실제 파일 hash와
  일치한다.
- 이 표의 test 결과를 후보 선택과 GPU 우선순위 결정에 사용했으므로, 동일한
  437장을 이후 논문에서 완전히 손대지 않은 최종 test set이라고 표현하면 안
  된다. 이후 checkpoint 선택은 valid만 사용하고, 최종 엔진 비교에서는 이
  반복 사용 사실을 명시하거나 별도의 독립 holdout을 마련해야 한다.
- TensorRT engine 크기, latency, FPS, peak memory는 아직 생성·측정되지 않아
  모두 N/A이다. 최종 Pareto 선택은 해당 실측값을 추가한 뒤 수행한다.

## 근거 파일

- 실험 상태: `configs/experiments/registry.yaml`
- 후보별 정적 분석: `artifacts/experiments/<ID>/front/static-analysis.json`
- B01 대비 변화: `artifacts/experiments/<ID>/front/comparison-B01.json`
- PTH–ONNX 동등성: `artifacts/experiments/<ID>/front/onnx-equivalence.json`
- 전체 test 정확도: `results/coco-evaluation/*.json`
- S01 학습 상태: `artifacts/experiments/S01/front/recovery/recovery-training.json`
- 구조 pruning 상세: `artifacts/experiments/S01|S02|S03/front/structured-pruning.json`
- M01 2:4 상세: `artifacts/experiments/M01/front/2to4-eligibility.json`,
  `onnx-2to4.json`, `fine-tuning-preflight.json`
