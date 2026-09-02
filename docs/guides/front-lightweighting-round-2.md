# 신규 Front baseline 경량화 2차 계획

> 현재 실행 순서는 **전체 후보 생성 → 1차 정적평가 → 통과 후보 노트북
> 2차 정확도·성능 실측 → 전 후보 terminal 후 Pareto**다. 이하의 이전
> 우선순위·정확도 조기 탈락 표현은 이력으로만 보존하며, 현재 판정은
> [`results/stage1-static-evaluation.md`](../../results/stage1-static-evaluation.md)를 따른다.

## 목적과 고정 기준

이번 경량화는 세션 누수 없이 다시 학습한 RF-DETR Segmentation Large를 유일한
출발점으로 사용한다. 과거 checkpoint, ONNX, engine과 평가 결과는 제거했으며
실험 ID는 모두 `planned` 상태로 초기화했다.

| 항목 | 고정값 |
|---|---|
| Source checkpoint | `artifacts/training/front-rfdetr-seg-large-v1/checkpoint_best_total.pth` |
| Checkpoint SHA-256 | `769ee97e38a2c6665bd664a9e035f5649d9f0a45c2bfe20172fe92ee545e5620` |
| 학습 데이터 | `data/training/front_session_split_v1/train` |
| 모델 선택 데이터 | `data/training/front_session_split_v1/valid` |
| 고정 비교 benchmark | `data/training/front_session_split_v1/test` 437장 |
| 기본 입력 | 504×504, batch 1 |
| 최종 평가 코드 | `scripts/evaluation/evaluate_coco_rfdetr_pth.py` 및 TensorRT 공통 평가기 |

양자화 calibration 이미지와 recovery fine-tuning 데이터는 `train`에서만
선정한다. `valid`는 checkpoint 선택에 쓴다. 현재 `test` 437장은
경량화 후보 비교에 반복 사용되었으므로 이후 ‘고정 benchmark set’으로
표기한다. 완전히 독립된 최종 test를 주장하려면 새 촬영 세션의
holdout을 별도로 확보해야 한다.

## 적용 가능한 경량화 방식

| 계열 | 기존 실험 | 적용 내용 | 실제 가속 가능성 | 2차 실험 판단 |
|---|---|---|---|---|
| TensorRT FP16 | B02 | 연산·weight를 반정밀도로 실행 | 높음 | 필수 기준 후보 |
| TensorRT INT8 PTQ | B03 | train 표본으로 scale 보정 후 INT8 engine 생성 | 중간, FP16 fallback 확인 필요 | 필수 비교 후보 |
| 비정형 pruning | U01~U03 | magnitude 기준 weight 10/30/50%를 0으로 만듦 | dense ONNX에서는 낮음 | 30% 한 개만 대조군 |
| NVIDIA 2:4 | M01~M03 | 4개 weight마다 2개를 남기고 recovery 학습 | 지원 GPU·tactic에서 가능 | 하드웨어 지원 시 수행 |
| Decoder 구조 pruning | S01~S02 | decoder와 연결 segmentation block 1/2개 제거 | 있음 | 1개 제거 우선 |
| FFN 구조 pruning | S03~S04 | FFN hidden dimension 20/40% 축소 | 있음 | 20% 축소 우선 |
| 입력 해상도 축소 | R01 | 504×504를 432×432로 축소 | 높음 | 별도 해상도 실험으로 수행 |
| 구조 축소+FP16 | C01 | 선정 structured 모델을 FP16 engine으로 생성 | 높음 | 핵심 결합 후보 |
| 구조 축소+INT8 | C02 | 선정 structured 모델에 INT8 PTQ 적용 | 중간 | C01 통과 후 수행 |
| SmoothQuant INT8 | Q01 | activation outlier를 weight 쪽으로 이동 후 Q/DQ 양자화 | 환경 의존 | 기본 INT8 손실이 클 때 수행 |
| 민감도 기반 혼합 INT8/FP16 | Q02 | head·sampling 등 민감 block을 FP16으로 보호 | 환경 의존 | 선택 후보 |
| 균일 FP8 | Q04 | ModelOpt FP8 Q/DQ export | 지원 GPU에서 높음 | 노트북 GPU 지원 시 수행 |
| 혼합 FP8/FP16 | Q05~Q07 | 민감도 측정 후 손상이 큰 block만 FP16 유지 | 지원 GPU에서 높음 | FP8 핵심 후보 |
| W8/W4/W3/W2 weight-only | W01 | RTN·AWQ·혼합 bit-width 정적 분석 | RF-DETR engine 지원이 제한적 | 연구용, 후순위 |
| INT4 weight-only | Q03 | FFN 중심 4-bit weight-only | 기존 build 실패 | 기본 실험에서 제외 |

비정형 pruning은 weight의 0 비율은 늘리지만 dense ONNX의 tensor shape와 연산량을
줄이지 않는다. 따라서 파일 크기나 sparsity 변화가 곧 FPS 향상을 의미하지
않으며, 논문의 음성 대조군으로만 사용한다. 반대로 structured pruning과 입력
해상도 축소는 graph 또는 입력 연산량 자체를 줄이므로 실제 지연시간 개선을
기대할 수 있다.

## 권장 실행 순서

1. 신규 PTH를 FP32 ONNX로 export하고 PTH test 결과가 재현되는지 확인한다.
2. 같은 ONNX에서 TensorRT FP32와 FP16 engine을 생성해 기준 성능을 고정한다.
3. train calibration 표본으로 INT8 PTQ를 만들고 실제 INT8/FP16 layer 비율을
   build log에서 확인한다.
4. decoder 1개 제거를 recovery fine-tuning하고, 병렬 정적 분석으로
   감소폭이 더 큰 decoder 2개 제거를 후속 후보로 준비한다. FFN 20%
   축소는 전체 FLOPs 감소가 0.39%에 그쳐 탈락시킨다.
5. 두 structured 후보 중 정확도-지연시간 Pareto가 좋은 하나만 INT8과 결합한다.
6. 432 해상도 FP16은 입력 크기가 다르므로 504 계열 표와 분리해 보고한다.
7. 노트북 GPU와 TensorRT가 FP8 또는 2:4를 실제 지원할 때만 해당 engine을
   만들고, 지원 tactic 사용 여부를 log로 증명한다.
8. 모든 후보 선택을 마친 뒤 동일한 437장 test에서 최종 평가한다.

## 진행 상태

- [x] B01 신규 PTH → FP32 ONNX export
- [x] ONNX checker, shape, 정적 분석 통과
- [x] PyTorch–ONNX 출력 동등성 검증 통과
- [x] U02 비정형 magnitude pruning 30% 생성·ONNX·전체 test 평가
- [x] U02 탈락: ONNX 크기/MACs/FLOPs 감소 0%, bbox AP 0.0964,
  mask AP 0.0820, mIoU 0.0328
- [x] S01 decoder 1개 제거 recovery fine-tuning·ONNX·전체 test 완료: 6 epoch
  early stopping, bbox AP 0.7374, mask AP 0.6028, mIoU 0.7339, production
  동등성 통과
- [x] R01 432×432 ONNX 생성·검증·전체 test 평가: MACs/FLOPs 30.20%
  감소, bbox AP 0.7307, mask AP 0.5930, mIoU 0.6969
- [x] S02 decoder 2개 제거 원형 준비: ONNX 9.49%, node 16.17% 감소
- [x] S02 복구 전 CPU 전체 test 평가: bbox AP 0.6770, mask AP 0.5726,
  semantic mIoU 0.7137
- [x] S02 8 epoch 복구·ONNX 동등성·전체 평가: bbox AP 0.7195,
  mask AP 0.5876, mIoU 0.7627. S01 대비 bbox/mask AP 보존 기준 미달로
  최종 structured 후보에서 탈락
- [x] S03 FFN 20% 탈락: 전체 parameter 3.00%, FLOPs 0.39% 감소
- [x] M01 2:4 원형·사전점검: 적용 대상 99.48%, pattern 100% 준수
- [ ] M01 2:4 복구 학습·ONNX 검증·전체 평가 진행 중
- [ ] 로컬 데스크탑 TensorRT engine 생성·benchmark·437장 평가
- [ ] 노트북에서 B01 FP32 engine 생성
- [ ] 노트북에서 B02 FP16 engine 생성

## R01 입력 해상도 평가 결과

R01은 baseline best PTH의 weight를 변경하거나 재학습하지 않고 입력만
504×504에서 432×432로 축소한 후보이다. 해상도 효과만 분리하기 위해 아래
정확도 값은 TensorRT FP16이 아닌 PyTorch checkpoint를 CPU에서 평가했으며,
baseline과 R01 모두 동일한 `test` 437장, detection threshold 0.001,
semantic mIoU threshold 0.25 조건을 사용했다.

| 지표 | Baseline 504 | R01 432 | 절대 delta | 상대 delta |
|---|---:|---:|---:|---:|
| bbox AP | 0.737791 | 0.730661 | -0.007130 | -0.966% |
| mask AP | 0.601016 | 0.592974 | -0.008042 | -1.338% |
| semantic mIoU | 0.712200 | 0.696878 | -0.015321 | -2.151% |

R01의 정적 MACs와 FLOPs는 B01 대비 30.20% 감소했다. 따라서 입력 해상도
축소만으로 bbox AP 0.713점, mask AP 0.804점, mIoU 1.532점의 손실을 감수하고
직접적인 연산량 감소를 얻는 후보로 유지한다. 여기서 점(point)은 지표를
0~100 척도로 표시했을 때의 차이다. 최종 TensorRT FP16 엔진은 별도로 생성한
뒤 동일 조건에서 다시 평가한다.

- R01 결과: `results/coco-evaluation/R01-front-432-pth.json`
- R01 결과 SHA-256:
  `e5a80d1a6931a3a8d81f28956485b49cb0a2e9bb8100921e089c08349e7b14bb`
- Baseline 결과: `results/coco-evaluation/front-rfdetr-seg-large-v1-test.json`
- Baseline 결과 SHA-256:
  `742252e55c46f2c76a7e41a8b9d8382314346b61602d140a54025028a6176633`

## S02 복구 전 정량 평가

S02는 decoder와 연결된 segmentation block을 각각 5개에서 3개로
줄인 직후의 모델이다. 복구 학습 전 손실을 분리해 측정하기 위해
`front_session_split_v1/test` 437장 전체를 CPU에서 평가했다. 이 split은
후보 개발에 반복 사용되었으므로 손대지 않은 최종 test가 아니라 고정
비교 benchmark로 표현한다. 입력은
504×504, detection threshold는 0.001, semantic mIoU threshold는
0.25였으며 CPU 연산 스레드는 3개로 제한했다.

| 지표 | B01 baseline | S02 복구 전 | 절대 delta | 상대 delta |
|---|---:|---:|---:|---:|
| bbox AP | 0.7377911741 | 0.6770020982 | -0.0607890759 | -8.2393% |
| mask AP | 0.6010160064 | 0.5726346577 | -0.0283813487 | -4.7222% |
| semantic mIoU | 0.7121996005 | 0.7136643614 | +0.0014647609 | +0.2057% |

절대 delta는 `S02 - B01`, 상대 delta는 `(S02 - B01) / B01`로
계산했다. bbox AP와 mask AP의 절대 하락은 각각 6.0789 AP point와
2.8381 AP point이다. mIoU의 소폭 상승은 bbox AP 하락을 상쇄하지
못하므로 S02는 현재 복구 학습 후보로 분류한다.

- S02 평가 결과: `results/coco-evaluation/S02-front-before-recovery.json`
- S02 결과 SHA-256:
  `4cf0ee009bc83fc01229df8d27ac5710c29dc5ca81c4cc8f147d8a35f0ab58fa`
- Baseline 결과: `results/coco-evaluation/front-rfdetr-seg-large-v1-test.json`
- Baseline 결과 SHA-256:
  `742252e55c46f2c76a7e41a8b9d8382314346b61602d140a54025028a6176633`

## S01 복구 후 최종 평가

S01은 decoder와 연결 segmentation block을 각각 5개에서 4개로
줄인 후, train으로만 복구 학습하고 valid로 checkpoint를 선택했다.
최대 10 epoch 중 6 epoch에서 patience 3으로 조기 종료했으며,
선택된 checkpoint의 decoder layer와 segmentation block은 모두 4개로
구조 검증을 통과했다. 아래 결과는 baseline과 동일한 test 437장,
504×504 입력, detection threshold 0.001, semantic mIoU threshold 0.25에서
측정했다.

| 지표 | B01 baseline | S01 복구 후 | 절대 delta | 상대 delta |
|---|---:|---:|---:|---:|
| bbox AP | 0.7377911741 | 0.7373798659 | -0.0004113082 | -0.0557% |
| mask AP | 0.6010160064 | 0.6027506763 | +0.0017346699 | +0.2886% |
| semantic mIoU | 0.7121996005 | 0.7339210573 | +0.0217214569 | +3.0499% |

절대 delta는 `S01 - B01`, 상대 delta는 `(S01 - B01) / B01`로
계산했다. S01은 bbox AP를 0.0411 AP point 이내로 유지하면서
mask AP를 0.1735 AP point, semantic mIoU를 2.1721 point 높였다.
또한 실제 test 이미지 10장의 PTH–ONNX production 동등성 검증에서
confidence 0.05 이상 활성 query 92개의 membership·class agreement와
binary mask agreement가 모두 1.0으로 통과했다. 따라서 S01은 구조
경량화의 선정 후보로 유지하고, 검증된 ONNX를 notebook TensorRT
FP16/INT8 실측 단계로 넘긴다.

- 학습 기록: `artifacts/experiments/S01/front/recovery/recovery-training.json`
- 최종 PTH SHA-256:
  `57e45c7553ec8829a53167b11ccc4da6964019f741b5762c087d82b2a899b821`
- 최종 ONNX SHA-256:
  `b8e436175ef25eb6885a3e8c48813e7c0ab1ea3d7c2339ab496c701338dba6e1`
- 동등성 기록 SHA-256:
  `c19c294995b354437003abbc2be4becd78bd004deb97b85ea141a84b8d00d4e6`
- 평가 결과: `results/coco-evaluation/S01-front-after-recovery.json`
- 평가 결과 SHA-256:
  `f1fdcac30d62cedf994ade45ec7484164d8ec080d22aacb4044c5fb564ab3406`

FP16과 일반 TensorRT INT8은 같은 FP32 ONNX를 입력으로 사용할 수 있으므로
precision마다 ONNX를 중복 생성할 필요가 없다. Structured pruning, 입력 해상도
변경, ModelOpt Q/DQ 양자화는 graph가 달라지므로 각각 별도 ONNX가 필요하다.

## 산출물 배치 규칙

- 실험 정의: `configs/experiments/registry.yaml`
- 로컬 생성물: `artifacts/experiments/<새 실험 ID>/front/`
- 노트북 전달용 ONNX: 검증 후 `shared-models/` manifest에 등록
- 장비 종속 TensorRT engine: 해당 노트북에서 생성하고 로컬 artifact에 보관
- 원시 평가값: `results/coco-evaluation/` 또는 `results/benchmarks/`
- 논문용 해석: `docs/reports/`

모든 실험은 registry에 기록된 신규 checkpoint와 source checkpoint hash를
기준으로 다시 생성한다. TensorRT engine은 GPU, CUDA, TensorRT 버전에
종속되므로 ONNX를 휴대 가능한 기준 산출물로 삼는다.
