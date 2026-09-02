# RF-DETR 후보 3단계 평가 프로토콜

이 문서는 논문의 후보 선정·평가·최종 비교 절차를 고정한다. 등록된
후보를 임의로 노트북 단계로 미루지 않고, 모든 후보에 대해 1차 정적평가를
먼저 수행한다. 1차 정적 gate를 통과한 후보는 동일한 노트북에서 2차
정확도·실행 성능을 측정하고, 전체 대상이 terminal 상태가 된 후에만 Pareto
분석을 수행한다.

## 후보 모집단과 데이터 역할

- 후보 모집단: `configs/experiments/registry.yaml`의 26개 ID
- train: 3,625장. 학습과 양자화 calibration 128장 선정에만 사용
- valid: 404장. Q05/Q06/Q07 민감도 정책에 사용
- test: 437장. 1차 gate에는 사용하지 않고 2차 최종 정확도 비교에 사용

기존에 test 437장으로 생성한 PTH/ONNX 정확도는 데스크탑 사전 진단으로만
보존한다. 이 수치를 1차 통과·탈락에 사용하지 않는다. 또한 해당 test는
기존 후보 비교에 반복 사용되었으므로 논문에서 `고정 비교 benchmark`로 표기하고,
완전히 독립된 외적 일반화 검증으로 표현하지 않는다.

## 1차: 전 후보 정적평가

1차에서는 정확도, latency, FPS, GPU memory를 판정하지 않는다. 다음 항목만
자동 검사한다.

1. 후보 ONNX 또는 원본 ONNX + TensorRT build recipe가 존재하는가.
2. `onnx.checker.check_model` 검증을 통과하는가.
3. 구조·해상도 후보는 B01 대비 ONNX 크기, graph node, dense MACs 중 하나 이상이
   5% 이상 감소하는가.
4. Q01~Q07은 후보 자체 ONNX에 Q/DQ 노드가 실제로 존재하고 양자화 보고서와
   일치하는가.
5. M01/M02는 상수 weight op의 2:4 패턴 준수와 sparse tactic 빌드 정책을
   명시했는가.

구조·해상도 후보에 대한 5%는 통계적 유의성 기준이 아니라, 실행 시스템에서
추가 측정할 가치가 있는 최소 변화를 걸러내는 사전 고정 engineering gate다. Q/DQ와
TensorRT precision recipe는 portable ONNX 파일 크기가 실제 engine 크기를 대변하지
못하므로 5% dense graph gate를 적용하지 않고 변환 근거 자체를 검증한다.

권위 있는 결과는 `results/stage1-static-evaluation.json|csv|md`이다. 후보별
세부 검사는 `artifacts/experiments/<ID>/front/stage1-static-evaluation.json`에 저장한다.

## 2차: 노트북 TensorRT 정확도·성능 평가

1차 `stage2_notebook_eligible=true`인 후보를 평가한다. engine은 평가 GPU,
CUDA, TensorRT 버전에 종속되므로 노트북에서 새로 생성한다.

- 정확도: test 437장, bbox AP@[.50:.95], mask AP@[.50:.95], semantic mIoU
- 임계값: AP 후보 0.001, semantic mIoU mask 0.25
- 지연시간: batch 1, 이미지 32장 고정 추출, warm-up 20회, 본 측정 200회
- 보고: median, mean, p95, IQR, CV, FPS
- 자원: peak allocated/reserved GPU memory, engine byte 크기
- 근거: ONNX/engine SHA-256, GPU·compute capability·CUDA·TensorRT, layer precision,
  M02 sparse tactic 선택 로그

engine build 불가, engine inspection 실패, benchmark 실패, 정확도 실패도 각 후보의
terminal 결과로 저장한다. 따라서 지원되지 않는 후보가 다시 `미수행`으로
숨어 버리지 않는다.

## 3차: Pareto 분석

1차 통과 대상의 2차 상태가 모두 terminal일 때만 생성한다. 정상 실행을
완료한 engine을 대상으로 bbox AP, mask AP, semantic mIoU는 극대화하고 median
latency, peak allocated GPU memory, engine size는 극소화한다. 다른 후보가 모든
목표에서 이상이고 적어도 하나에서 엄격히 우세하면 해당 후보를 dominated로
판정한다.

## 재현 명령

```bash
# 1차 전체 후보표 재생성
.venv/bin/python scripts/reporting/generate_stage1_static_evaluation.py

# 1차 통과 22개 engine plan과 17개 unique ONNX 확인
.venv/bin/python scripts/experiments/build_engine_suite.py --suite stage1 --dry-run
.venv/bin/python scripts/experiments/package_notebook_bundle.py --suite stage1 \
  --output-dir delivery/notebook-front-stage1 --dry-run

# 패키지 생성 후 train calibration 128장과 test 437장 추가
.venv/bin/python scripts/experiments/package_notebook_bundle.py --suite stage1 \
  --output-dir delivery/notebook-front-stage1 --force
.venv/bin/python scripts/data_preparation/package_notebook_data.py \
  --output-dir delivery/notebook-front-stage1

# 노트북에서 2차 전체 실행, 완료 후 3차 Pareto 자동 생성
python scripts/experiments/run_notebook_stage2.py --force-build
```

2차 자동화는 후보별 로그를 `results/stage2-notebook-logs/`, 상태를
`results/stage2-notebook-state.json`, 통합 결과를 `results/stage2-notebook-summary.*`, Pareto
결과를 `results/stage3-pareto.*`에 저장한다.
