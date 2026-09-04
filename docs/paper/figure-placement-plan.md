# 논문 시각자료 배치안

| 번호 | 파일 | 삽입 위치 | 핵심 주장 |
|---|---|---|---|
| 그림 1 | `paper/01_overall_pipeline.*` | 1장 말미 또는 4장 시작 | 정적 선별과 목표 장비 평가를 분리한 연구 절차 |
| 그림 2 | `paper/02_dataset_examples.*` | 3.1 데이터 구성 | 실제 입력과 세 클래스의 GT annotation 형태 |
| 그림 3 | `dataset_split_validity.*` | 3.3 데이터 분할 | 세션 기반 분할 비율·분포·임계값·배정 |
| 그림 4 | `paper/04_training_curves.*` | 4.1 Baseline 학습 | 18 epoch 종료 및 best EMA checkpoint 근거 |
| 그림 5 | `paper/05_candidate_taxonomy.*` | 4.2 후보 구성 | 26개 후보의 8개 방법군 |
| 그림 6 | `paper/06_candidate_flow.*` | 4.3 말미 또는 6장 시작 | 26→22→21→11→2 및 단계별 제외 사유 |
| 그림 7 | `paper/07_accuracy_gate_heatmap.*` | 6.2 Stage 2 결과 | B01 대비 세 정확도 변화와 gate 판정 |
| 그림 8 | `paper/08_pareto_tradeoff.*` | 6.3 Pareto 분석 | C01·R01의 정확도–속도–크기 비지배 관계 |
| 그림 9 | `paper/09_qualitative_source_onnx.*` | 6.4 정성 분석 | 동일 benchmark 이미지의 GT와 source ONNX 출력 비교 |
| 그림 A1 | `appendix/A1_stage1_static_reductions.*` | 부록 | 구조·해상도 후보의 정적 감소율 |
| 그림 A2 | `appendix/A2_latency_stability.*` | 부록 | 63개 반복의 CV 5% 이하 확인 |
| 그림 A3 | `appendix/A3_classwise_accuracy.*` | 부록 | B01·C01·R01 클래스별 정확도 |

## 사용 주의

- 그림 9는 로컬에 최종 TensorRT engine이 없으므로 대응 source ONNX graph의 정성 출력이다. TensorRT engine 출력으로 표현하면 안 된다.
- 30 FPS 선은 median 33.33 ms 기준이며 모든 프레임 또는 지속 처리량의 30 FPS 보장을 뜻하지 않는다.
- `test`는 논문에서 `fixed comparative benchmark` 또는 `benchmark`로 표기한다.
- 본문 삽입에는 PDF를 우선 사용하고 PNG는 Notion·README 미리보기용으로 사용한다.
- GPU memory는 후보 간 차이가 작아 별도 그래프로 만들지 않는다.

## 논문용 캡션 초안

- **그림 1. 정적 경량화와 장비 종속 성능을 분리한 평가 파이프라인.** 무증강 데이터 구성과 RF-DETR baseline 학습 후 26개 경량화 후보를 생성하였다. Stage 1은 정확도를 사용하지 않는 artifact·recipe 정적 검증이며, Stage 2는 동일 benchmark에서 TensorRT build, 정확도 및 지연시간을 측정한다. 정확도 보존 gate를 통과한 후보만 정확도 최대화, median latency 최소화, engine 크기 최소화의 3목적 Pareto 분석에 포함하였다.
- **그림 2. 고정 benchmark의 대표 입력 및 ground-truth annotation.** 작은 주석 영역 비율, polygon 꼭짓점 수, 큰 주석 영역 비율을 기준으로 각 극단 사례를 결정론적으로 선택하고, 평가 객체가 없는 negative 장면을 함께 제시하였다. 색상은 `out_line`, `parking_lot`, `parking_space`를 나타낸다.
- **그림 3. 무증강 데이터의 세션 단위 분할 타당성.** 전체 4,466장을 train 3,625장(81.17%), validation 404장(9.05%), benchmark 437장(9.79%)으로 분할하였다. 2초 임계값은 관측된 세션 내부 최대 간격 1.411초와 세션 경계 최소 간격 5.010초 사이에 위치하며, 하나의 촬영 세션은 둘 이상의 분할에 포함되지 않는다.
- **그림 4. RF-DETR baseline의 18 epoch 학습 과정.** 좌측은 train·validation loss, 우측은 validation BBox AP, Mask AP 및 EMA Mask AP를 나타낸다. EMA Mask AP의 최댓값은 epoch 12에서 기록되었고 이후 6 epoch 동안 0.001을 초과하는 개선이 없어 epoch 18에서 학습을 종료하였다.
- **그림 5. 동일 RF-DETR baseline에서 파생한 26개 경량화 후보의 방법 분류.** 후보는 baseline, 정밀도 변환, 비정형 pruning, 2:4 sparsity, 구조 축소, 결합형, 입력 해상도, ModelOpt quantization/dequantization의 8개 방법군으로 구성된다. 연구 범위에서 제외된 W 계열은 포함하지 않았다.
- **그림 6. 후보 선별 결과와 단계별 제외 사유.** 최초 26개 중 Stage 1에서 22개가 통과하였고, Q03은 TensorRT parser 문제로 build에 실패하여 21개가 Stage 2 측정을 완료하였다. 이 중 11개가 정확도 보존 gate를 통과했으며, 3목적 Pareto 분석 결과 C01과 R01이 비지배 후보로 남았다.
- **그림 7. B01 대비 후보별 정확도 변화와 보존 gate 판정.** 각 셀은 B01 대비 BBox AP, Mask AP, semantic mIoU 변화량이며, 허용 하락 한계는 각각 −0.010, −0.010, −0.020이다. 하나 이상의 하한을 위반한 셀은 테두리로 표시하고 후보별 최종 PASS/FAIL을 함께 제시하였다.
- **그림 8. 정확도 보존 후보의 성능–자원 trade-off와 3목적 Pareto 해.** 오차막대는 후보별 3회 반복 median의 평균에 대한 95% t 신뢰구간이며, 점의 x 좌표는 총 600회 측정의 pooled median이다. 수직선은 30 FPS에 대응하는 33.33 ms 기준이다. 녹색으로 강조한 C01과 R01은 Mask AP 최대화, median latency 최소화, TensorRT engine 크기 최소화에서 서로 지배되지 않는다.
- **그림 9. 동일 benchmark 사례에서의 ground truth와 source ONNX 정성 비교.** B01, C01, R01에 대응하는 source ONNX graph를 confidence threshold 0.25로 추론하였다. 이 그림은 최종 TensorRT engine의 정성 결과가 아니므로 장비 배포 결과와 구분해 해석한다.
- **그림 A1. 구조·해상도 후보의 Stage 1 정적 감소율.** B01 대비 ONNX 파일 크기, graph node 및 dense MAC 감소율과 정적 검증 통과 여부를 나타낸다.
- **그림 A2. 공식 지연시간 반복 측정의 안정성.** 21개 후보에서 얻은 총 63회 측정의 후보별 반복 median 변동계수(CV)를 제시하였다. 모든 후보가 사전 정의한 재측정 기준 5%보다 낮았다.
- **그림 A3. B01, C01, R01의 클래스별 정확도.** 세 후보의 `out_line`, `parking_lot`, `parking_space`에 대한 BBox AP, Mask AP 및 semantic IoU를 동일 축에서 비교하였다.
