# 프로젝트 진행 현황

이 문서는 논문 초안 작성자와 다음 실험 담당자가 가장 먼저 읽는 인수인계
페이지다. 정량값은 2026-09-02에 생성된 저장소 결과를 기준으로 하며, 실행
중인 단계는 결과를 추정하지 않고 `진행 중`으로 표시한다.

## 완료된 작업

1. Roboflow Version 9를 무증강으로 내보내고, 프레임이 아닌 촬영 세션을
   분할 단위로 사용해 train 3,625장, valid 404장, test 437장으로 재구성했다.
2. 교차 split 이미지 중복, 세션 중복, COCO 참조 오류가 모두 0임을 자동
   감사했다.
3. RF-DETR Segmentation Large baseline을 다시 학습하고 437장 benchmark에서
   bbox AP 0.737791, mask AP 0.601016, semantic mIoU 0.712200을 얻었다.
4. B01, R01, S01, S02의 PTH–ONNX 변환 동등성과 전체 ONNX 정확도를 검증했다.
5. U02, R01, S01, S02, S03, M01 후보의 구조·희소성·ONNX·부분 MAC/FLOP를
   같은 분석기로 산출했다.
6. 평가 설정, artifact hash, TensorRT build, latency benchmark, 결과표 생성을
   하나의 레지스트리 기반 파이프라인으로 연결했다.

## 후보 판정

| ID | 방법 | 정확도/구조 결과 | 현재 판정 |
|---|---|---|---|
| B01 | 504×504 FP32 baseline | bbox 0.737791 / mask 0.601016 / mIoU 0.712200 | 기준선 |
| R01 | 입력 432×432 | MAC 30.20% 감소, bbox AP -0.007130 | Pareto 후보 |
| S01 | decoder 1개 제거 + 6 epoch 복구 | node 8.04% 감소, bbox AP -0.000411 | Pareto 후보 |
| S02 | decoder 2개 제거 + 8 epoch 복구 | node 16.17% 감소, bbox AP -0.018335 | 정확도 gate 탈락, 속도 대조군 |
| U02 | magnitude 30% | dense MAC 감소 0%, bbox AP -0.641349 | 탈락 |
| S03 | FFN 20.31% 축소 | 전체 MAC 0.39% 감소 | 탈락 |
| M01/M02 | NVIDIA 2:4 | eligible ONNX constant op 150/150 형식 준수 | 복구·실측 진행 중 |

정적 MAC/FLOP는 Conv·MatMul·Gemm 중 shape를 해석한 연산만 포함하는 dense
하한 추정치다. TensorRT fusion, 메모리 이동, 전처리와 후처리 비용을 포함하지
않으므로 실제 속도 주장은 engine 실측 이후에만 한다.

## 현재 실행 중인 작업

- M01 2:4 모델을 최대 10 epoch, early stopping patience 3으로 복구 학습 중이다.
- 복구가 끝나면 자동으로 최종 checkpoint 선정, ONNX 재수출, 10장 동등성
  검사, 437장 PTH/ONNX 평가, 정적 분석 갱신을 수행한다.
- 그 다음 데스크탑 큐가 TensorRT engine 생성·검사·benchmark·정확도 평가와
  논문용 표/그림 생성을 이어서 수행한다.

실행 중 파일은 `artifacts/`와 `results/`의 로컬 작업 영역에 남으며, 완료되지
않은 checkpoint·engine·로그는 이 커밋에 포함하지 않는다.

## 논문 초안에 바로 사용할 근거

| 초안 섹션 | 우선 참고 문서 |
|---|---|
| 데이터 수집·분할 | [데이터 분할 타당성](reports/dataset-split-validity.md) |
| 학습 설정 | [baseline 학습 가이드](guides/front-baseline-training.md) |
| 경량화 방법 | [2차 경량화 계획](guides/front-lightweighting-round-2.md) |
| 정적 모델 비교 | [논문용 정적 분석](reports/paper-static-analysis.md) |
| 후보 정확도 비교 | [후보 정량 비교](reports/front-lightweighting-candidate-comparison.md) |
| 평가 지표 정의 | [평가 지표와 해석](concepts/evaluation-metrics.md) |
| 평가 타당성·한계 | [평가 자동화 감사](reports/evaluation-automation-audit.md) |
| 재현용 모델 위치 | [모델 artifact 인덱스](handoffs/model-artifact-index.md) |

논문에는 다음 한계를 명시한다.

- 현재 437장은 모델 선택에 반복 사용한 고정 benchmark set이다.
- 복구 학습은 seed 42의 단일 실행이므로 학습 분산을 추정하지 않는다.
- PTH 평가는 일부 후보에서 GPU, 일부에서 CPU로 수행했으나 같은 evaluator와
  데이터셋을 사용했다. 속도 비교에는 이 측정 시간을 사용하지 않는다.
- 최종 TensorRT latency/FPS/memory와 반복 측정 분산은 아직 대기 상태다.

## 완료 조건

- M01 최종 PTH/ONNX 정확도와 동등성 기록 완료
- M01 dense와 M02 sparse build log에서 tactic 사용 여부 구분
- 모든 비교 engine의 동일 장비 latency 200회 측정 완료
- TensorRT 정확도 gate 적용 및 최종 Pareto 후보 확정
- `paper_results.py`, 정적 분석 생성기, 평가 감사 재실행
- 외부 일반화 주장이 필요하면 별도 촬영 세션 holdout 확보
