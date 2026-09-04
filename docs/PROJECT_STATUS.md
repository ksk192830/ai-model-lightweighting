# 프로젝트 진행 현황

이 문서는 논문 초안 작성자와 다음 실험 담당자가 가장 먼저 읽는 인수인계
페이지다. 정량값은 2026-09-04에 생성된 저장소 결과를 기준으로 하며, 실행
중인 단계는 결과를 추정하지 않고 `진행 중`으로 표시한다.

## 완료된 작업

1. Roboflow Version 9를 무증강으로 내보내고, 프레임이 아닌 촬영 세션을
   분할 단위로 사용해 train 3,625장, valid 404장, test 437장으로 재구성했다.
2. 교차 split 이미지 중복, 세션 중복, COCO 참조 오류가 모두 0임을 자동
   감사했다.
3. RF-DETR Segmentation Large baseline을 다시 학습하고 437장 benchmark에서
   bbox AP 0.737791, mask AP 0.601016, semantic mIoU 0.712200을 얻었다.
4. registry의 전체 26개 후보에 대해 1차 정적평가를 수행했다. 22개가
   통과했고 4개가 불통했으며 미수행은 0개다.
5. Q01~Q07 ModelOpt ONNX를 모두 생성하고 Q/DQ 노드 34~626개, ONNX checker,
   graph 및 빌드 recipe를 검증했다. Q05~Q07 민감도 선정은 valid만 사용했다.
6. 기존 데스크탑 정확도를 1차 gate에서 제거했다. 이에 따라 정적 기준을
   충족한 R02, R03, S02, S04, M01, M02도 노트북 2차 대상으로 복구했다.
7. 1차 통과표, 22개 engine plan, 17개 unique ONNX 패키지, 후보별 실패
   로그, 2차 통합 표, 전 후보 terminal 후 Pareto 생성을 하나의 자동화로 연결했다.
8. 노트북 원클릭 실행기를 추가해 환경 설치·checksum·engine 생성·정적검사·3회
   latency 반복·437장 정확도·정확도 gate·Pareto·Excel 보고서까지 한 번에 실행하고,
   중단 뒤 완료 후보를 재사용하도록 구성했다.
9. RTX 4050 Laptop GPU에서 22개 후보를 terminal 상태로 만들었다. 21개는 전체
   평가를 완료했고 Q03은 TensorRT INT4 block quantization parser 오류를 기록했다.
   이 최초 결과의 다목적 Pareto 집합은 이후 확정한 3축 주 Pareto의 최종 결과로
   사용하지 않고, performance 통제 재측정 후 다시 산출한다.

## 1차 정적평가 판정

| ID | 방법 | 1차 정적 근거 | 현재 판정 |
|---|---|---|---|
| B01/B02/B03 | FP32 대조·FP16·INT8 PTQ | 유효 ONNX + 고정 build recipe | 통과 |
| U01/U02/U03 | magnitude 10/30/50% | dense ONNX 크기·node·MAC 감소가 모두 5% 미만 | **불통** |
| M01/M02 | NVIDIA 2:4 dense/sparse tactic | ONNX 적격 op 150/150이 2:4 준수, sparse build recipe 존재 | 통과 |
| S01/S02 | decoder 1/2개 제거 | graph node 8.04/16.17% 감소 | 통과 |
| S03 | FFN 20% 축소 | ONNX 크기 3.27%, dense MAC 0.39%로 5% gate 미달 | **불통** |
| S04 | FFN 40% 축소 | ONNX 크기 6.53% 감소 | 통과 |
| C01/C02 | S01 + FP16/INT8 | S01 ONNX + 고정 build recipe | 통과 |
| C03/C04 | S01 + 432/480 해상도 | node 8.04%, MAC 30.76/11.51% 감소 | 통과 |
| R01/R02/R03 | 432/480/384 해상도 | dense MAC 30.20/10.96/46.25% 감소 | 통과 |
| Q01~Q07 | INT8·INT4·FP8·혼합 ModelOpt | 7개 후보 ONNX 전부 valid, Q/DQ 34~626개 | 통과 |

정적 MAC/FLOP는 Conv·MatMul·Gemm 중 shape를 해석한 연산만 포함하는 dense
하한 추정치다. TensorRT fusion, 메모리 이동, 전처리와 후처리 비용을 포함하지
않으므로 실제 속도 주장은 engine 실측 이후에만 한다.

## 현재 실행 상태와 다음 작업

- 1차는 26/26 완료됐고, 최초 2차 실행은 terminal 22/22에 도달했다.
- 엔진 성공 후보 21개의 latency를 동일 `performance` 전원 조건에서 다시 측정한
  뒤 주 Pareto를 재산출해야 한다.
- 기존 21개 후보의 정확도·latency·FPS·memory·engine 크기는 보존하지만 통제 전
  latency와 기존 7축 Pareto는 최종 논문 수치로 사용하지 않는다.
- Q03은 TensorRT 10.16.1.11이 block size 128 INT4 `DequantizeLinear` 입력을
  파싱하지 못해 build-failed로 종료됐다. 같은 환경에서 변경 없이 재실행할 이유는 없다.
- C02, B03과 R01은 현재 잠정 비교 후보이며 최종 후보는 새 3축 Pareto 이후 정한다.
- 새 Pareto 중 median latency 33.33 ms 이하인 후보만 30 FPS 실시간 배포 후보로
  별도 표시한다.
- P95는 33.33 ms 초과 시 경고하는 보조지표이며 배포 후보·Pareto hard gate로
  사용하지 않는다.
- 반복 median CV가 5%를 초과한 후보는 지연시간 3회를 한 번 자동 재측정한다.
  재측정 후에도 초과하면 두 라운드를 보존하고 Pareto에는 유지하지만 최종 권장
  모델에서는 제외한다.
- 과거 R01은 반복 중앙값 CV 5.25%였으나, 공식 성능 우선 전체 재평가에서 위 자동
  재측정 정책을 새로 적용한다.
- 전체 결과는 [노트북 Stage 2 최종 평가](reports/notebook-stage2-results.md)와
  `results/stage2-evaluation-report.xlsx`에 있다.

## 논문 초안에 바로 사용할 근거

| 초안 섹션 | 우선 참고 문서 |
|---|---|
| 데이터 수집·분할 | [데이터 분할 타당성](reports/dataset-split-validity.md) |
| 학습 설정 | [baseline 학습 가이드](guides/front-baseline-training.md) |
| 경량화 방법 | [2차 경량화 계획](guides/front-lightweighting-round-2.md) |
| 3단계 후보 평가 | [1차 정적 → 2차 노트북 → 3차 Pareto](guides/three-stage-candidate-evaluation.md) |
| 1차 전체 후보표 | [26개 전체 정적평가](../results/stage1-static-evaluation.md) |
| 추가 후보 설계·1차 판정 | [추가 후보 및 1차 평가](guides/additional-candidate-screening.md) |
| 전체 후보 자동 생성표 | [Round-2 후보 요약](../results/round2-candidate-summary.md) |
| 정적 모델 비교 | [논문용 정적 분석](reports/paper-static-analysis.md) |
| 후보 정확도 비교 | [후보 정량 비교](reports/front-lightweighting-candidate-comparison.md) |
| 평가 지표 정의 | [평가 지표와 해석](concepts/evaluation-metrics.md) |
| 평가 타당성·한계 | [평가 자동화 감사](reports/evaluation-automation-audit.md) |
| 노트북 TensorRT 최종 결과 | [Stage 2/3 최종 평가](reports/notebook-stage2-results.md) |
| 재현용 모델 위치 | [모델 artifact 인덱스](handoffs/model-artifact-index.md) |

논문에는 다음 한계를 명시한다.

- 현재 437장은 모델 선택에 반복 사용한 고정 benchmark set이다.
- 복구 학습은 seed 42의 단일 실행이므로 학습 분산을 추정하지 않는다.
- PTH 평가는 일부 후보에서 GPU, 일부에서 CPU로 수행했으나 같은 evaluator와
  데이터셋을 사용했다. 속도 비교에는 이 측정 시간을 사용하지 않는다.
- TensorRT 결과는 RTX 4050 Laptop GPU와 기록된 CUDA/TensorRT 환경에 한정된다.
- R01의 반복 latency 변동성과 Q03 INT4 parser 호환성은 후속 확인 대상이다.

## 후속 작업

- R01을 고정된 전력·온도·백그라운드 조건에서 추가 측정해 속도 순위를 확인
- 22개 전부의 수치가 필요하면 Q03을 호환 형식으로 재export한 뒤 Q03만 재평가
- `paper_results.py`가 최신 노트북 summary를 읽도록 바꾸고 논문 표·그림 갱신
- 과거 데스크탑/PTH 경로와 파일 mtime에 의존하는 평가 감사 도구를 최신 해시 기반으로 보완
- 논문에 외부 일반화 주장이 필요하면 별도 촬영 세션 holdout 확보
