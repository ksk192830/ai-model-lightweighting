# RF-DETR 경량화 및 최적화

무증강 주차장 전방 영상으로 재학습한 RF-DETR Segmentation Large를 대상으로
정확도뿐 아니라 실제 배포 비용까지 함께 비교하는 연구 저장소다. 구조 축소,
입력 해상도 축소, 비정형·2:4 희소화, FP16·INT8·INT4·FP8 및 혼합정밀도 후보를
동일한 데이터와 프로토콜로 평가하고, 최종적으로 정확도·지연시간·GPU 메모리·
engine 크기의 Pareto 비지배해를 선택한다.

- GitHub: [kips-ai-model-lightweighting](https://github.com/ksk192830/kips-ai-model-lightweighting)
- Notion: [AI 경량화 및 최적화 프로젝트](https://app.notion.com/p/389d4a78ceed80d28d95c37d83422a60)
- 현재 상태: [프로젝트 진행 현황](docs/PROJECT_STATUS.md)
- 전체 후보표: [1차 정적평가 26개 결과](results/stage1-static-evaluation.md)
- 노트북 최종 결과: [Stage 2 TensorRT 평가와 Stage 3 Pareto](docs/reports/notebook-stage2-results.md)
- 평가 방법: [1차 정적 → 2차 노트북 → 3차 Pareto](docs/guides/three-stage-candidate-evaluation.md)
- 노트북 전달: [TensorRT 노트북 재현 가이드](docs/guides/tensorrt-notebook-portability.md)
- 논문 개조식 구성안: [한국어 논문 구성안](docs/paper/paper-outline-ko.md)
- 논문 줄글 초안: [결과 확정 전 원고](docs/paper/manuscript-draft-ko.md)

## 현재 진행 상황

기준일은 **2026-09-04**다. 모델 준비, 1차 정적평가와 노트북 TensorRT
평가가 끝났으며 현재 실행 중인 학습이나 로컬 평가는 없다.

| 단계 | 대상 | 상태 | 결과 및 다음 조건 |
|---|---:|---|---|
| 데이터 재구성 | 4,466장 | 완료 | train 3,625 / valid 404 / benchmark 437, split 간 이미지·세션 중복 0 |
| 기준 모델 재학습 | B01 | 완료 | bbox AP 0.737791 / mask AP 0.601016 / semantic mIoU 0.712200 |
| 1차 정적평가 | 26개 | **완료** | 평가 26/26, 통과 22, 불통 4, 미수행 0 |
| 노트북 전달 묶음 | 통과 22개 | 완료 | engine plan 22개, unique ONNX 17개, calibration 128장, benchmark 437장 |
| 2차 TensorRT 평가 | 통과 22개 | **완료** | 21개 측정 완료, Q03 TensorRT INT4 parser build 실패 1개 |
| 3차 Pareto 분석 | 2차 성공 후보 | **완료** | B02·B03·C01·C02·C03·C04·R01·R02 |

현재 1차 불통 후보는 U01·U02·U03·S03이다. U01~U03은 비정형 희소화가
dense ONNX 크기·노드·MAC을 줄이지 못했고, S03은 세 정적 효율 항목 모두
사전 고정 5% 기준에 미달했다. 나머지 22개는 기존 데스크탑 정확도 판정과
무관하게 노트북 2차 평가 대상으로 유지한다.

2차 실측에서는 C02가 B01 대비 중앙 지연시간을 47.1%, engine 크기를 49.9%
줄이면서 세 정확도 gate를 모두 통과했다. R01은 gate 통과 후보 중 가장 빠른
25.82 ms였지만 반복 중앙값 CV 5.25%로 경고 기준을 소폭 넘어 재측정을 권장한다.
전체 수치와 후속 작업은 [노트북 Stage 2 최종 평가](docs/reports/notebook-stage2-results.md)에
정리했다.

진행 상태는 생성된 결과 파일에서 직접 읽는다.

```bash
.venv/bin/python scripts/reporting/show_project_status.py
```

노트북 평가 중에는 다음처럼 자동 갱신할 수 있다.

```bash
watch -n 5 '.venv/bin/python scripts/reporting/show_project_status.py'
```

기계 판독이 필요하면 `--json`을 붙인다. 상세 상태는
`results/stage2-notebook-state.json`, 후보별 로그는
`results/stage2-notebook-logs/`, 최종 Pareto 결과는
`results/stage3-pareto.json`에 기록된다.

## 논문 인트로 초안에 사용할 핵심 내용

주차장 환경의 객체 인식 시스템은 차량, 주차 구획 및 장애물의 위치와 영역을
정확히 추정해야 하는 동시에 제한된 GPU 자원에서 짧은 응답시간을 만족해야 한다.
Transformer 기반 검출기는 전역 문맥과 객체 간 관계를 직접 모델링해 높은 표현력을
제공하지만, encoder–decoder 연산, 고해상도 특징 처리 및 segmentation head로 인해
실제 배포 시 연산량과 메모리 사용량이 증가할 수 있다. 따라서 기준 모델의 정확도만
보고 배포 모델을 정하는 방식으로는 현장 장비에서의 지연시간, 처리량, 메모리 한계와
정확도 손실 사이의 절충관계를 설명하기 어렵다.

본 연구는 무증강 주차장 전방 영상으로 재학습한 RF-DETR Segmentation Large를
기준으로, 모델 구조·입력 해상도·희소성·수치 정밀도를 서로 다른 경량화 축으로
정의한다. 각 방법을 독립 후보와 결합 후보로 구성하고, 모든 후보에 먼저 정적
유효성 검사를 적용한 뒤 동일한 NVIDIA 노트북 환경에서 TensorRT engine을 다시
생성한다. 이후 동일한 437장 benchmark와 고정된 반복 측정 조건으로 bbox AP,
mask AP, semantic mIoU, latency, FPS, GPU memory 및 engine 크기를 측정한다.
마지막으로 단일 가중합 점수 대신 Pareto 비지배 관계를 이용해 정확도와 자원 효율
사이에서 다른 후보에 일방적으로 열등하지 않은 배포 대안을 제시한다.

이 설계의 핵심은 “파라미터가 0이 되었는가”와 “실제 실행이 빨라졌는가”를 구분하는
것이다. 비정형 pruning처럼 weight의 0 비율은 높지만 dense graph의 shape가 그대로인
방법은 일반적인 dense kernel에서 속도 향상으로 이어지지 않을 수 있다. 반대로
구조적 pruning, 입력 해상도 축소, 지원 하드웨어의 2:4 sparse tactic 및 저정밀
kernel은 실제 실행 비용을 바꿀 가능성이 있다. 그러므로 정적 분석은 후보의 변환이
실재하는지 확인하는 1차 선별 수단으로만 사용하고, 최종 효율 주장은 목표 장비에서의
TensorRT 실측 결과로 제한한다.

### 연구 질문

1. RF-DETR의 decoder layer, FFN 차원 및 입력 해상도 축소가 segmentation 정확도와
   정적 연산량에 각각 어떤 영향을 주는가?
2. FP16, INT8, INT4, FP8 및 민감도 기반 혼합정밀도가 동일 장비에서 정확도,
   latency, GPU memory와 engine 크기에 어떤 절충관계를 만드는가?
3. 비정형 sparsity와 하드웨어 지원 2:4 sparsity의 정적 희소율 차이가 실제
   TensorRT 가속 차이로 연결되는가?
4. 구조·해상도·정밀도를 결합한 후보가 단일 경량화 후보보다 더 나은 Pareto
   지점을 형성하는가?

### 연구 기여로 정리할 수 있는 항목

- 동일 baseline에서 파생된 26개 후보를 누락 없이 비교하는 재현 가능한 후보군
- 데이터 역할을 분리한 정적 선별, 장비 종속 실측, Pareto 분석의 3단계 평가 절차
- detection과 segmentation 품질 및 실제 시스템 효율을 함께 다루는 다목적 평가
- 미지원 kernel이나 build 실패도 누락하지 않고 terminal 실패 근거로 남기는 자동화
- 실험 설정, 모델·결과 SHA-256, 하드웨어 정보와 원시 로그를 연결한 추적 가능성

## 이론적 배경

### DETR와 RF-DETR 기반 instance segmentation

DETR는 객체 검출을 고정된 object query 집합에 대한 직접적인 set prediction
문제로 정의한다. Transformer encoder–decoder와 bipartite matching loss를 사용해
예측과 정답을 일대일로 대응시키며, anchor 설계나 일반적인 NMS 의존성을 줄인다.
DINO 계열은 denoising 학습과 query 초기화를 개선했고, Mask DINO는 검출기에 mask
prediction branch를 결합해 instance·panoptic·semantic segmentation을 통합했다.

RF-DETR는 DETR 계열의 실시간 검출·분할 모델이며 정확도–지연시간 절충을 주요 설계
목표로 둔다. 본 실험은 RF-DETR Segmentation Large의 bbox와 instance mask 출력을
동시에 평가하므로, 검출 정확도만 유지되고 mask 품질이 하락하는 후보를 동일한
성능으로 간주하지 않는다.

### 모델 경량화의 평가 관점

경량화는 단일 수치의 최소화 문제가 아니다. 모델 크기와 이론 연산량이 감소해도
kernel 지원, 연산 fusion, 메모리 이동 및 전·후처리 비용에 따라 실제 latency가
감소하지 않을 수 있다. 이 저장소는 다음을 분리해 기록한다.

| 구분 | 지표 | 해석 |
|---|---|---|
| 정적 구조 | ONNX bytes, node 수, 추정 MAC/FLOP | 그래프 수준 변화와 후속 측정 가치 확인 |
| 표현 정밀도 | Q/DQ node, layer precision, engine bytes | 저정밀 변환이 실제 graph/engine에 반영됐는지 확인 |
| 예측 품질 | bbox AP, mask AP, semantic mIoU | 검출·인스턴스 분할·클래스별 영역 품질을 분리 평가 |
| 실행 성능 | median/p95 latency, FPS | 동일 장비와 동일 반복 조건에서 비교 |
| 자원 사용 | peak GPU memory, engine bytes | 장비 수용 가능성과 배포 비용 평가 |

정적 MAC/FLOP는 shape를 해석할 수 있는 Conv·MatMul·Gemm의 dense 산술량 하한이며,
TensorRT fusion이나 메모리 병목을 포함하지 않는다. 따라서 정적 수치를 실제 FPS로
환산하지 않는다.

### Pruning과 sparsity

비정형 pruning은 중요도가 낮은 개별 weight를 0으로 만들어 저장·압축 가능성을
높이지만 불규칙한 희소 패턴 때문에 sparse kernel이 없으면 dense 실행 비용이
그대로일 수 있다. 구조적 pruning은 channel, FFN 차원, decoder layer처럼 연산
단위를 제거해 tensor shape 또는 graph depth를 직접 줄이므로 기존 dense kernel에서도
가속 가능성이 더 명확하다. 다만 구조 제거는 표현 용량도 줄이므로 복구 fine-tuning과
정확도 검증이 필요하다.

NVIDIA 2:4 sparsity는 연속된 weight 네 개 중 두 개를 0으로 제한하는 반구조적
패턴이다. 지원 GPU의 sparse Tensor Core와 TensorRT tactic이 실제 선택될 때만
연산 이득을 기대할 수 있다. 따라서 M01은 동일 2:4 weight를 dense tactic으로
실행하는 대조군, M02는 sparse tactic을 허용하는 실험군으로 두어 weight 효과와
kernel 효과를 분리한다.

### 저정밀 양자화와 혼합정밀도

균일 양자화는 실수값 `x`를 scale `s`와 zero-point `z`를 이용해
`q = clip(round(x / s) + z)`로 변환하고, 실행 시 `x_hat = s(q - z)`로 근사한다.
bit 수를 줄이면 메모리와 대역폭 요구량이 감소하지만 rounding과 clipping 오차가
증가한다. PTQ는 학습 완료 모델과 calibration 표본으로 scale을 정하므로 재학습
비용이 작지만, activation outlier와 민감 layer에서 오차가 커질 수 있다.

본 연구의 INT8·FP8 후보는 train에서 고정 추출한 128장만 calibration에 사용한다.
Q05~Q07은 valid에서 측정한 AP 또는 KLD 민감도로 보호 block을 선택하고, test는
이 선택에 사용하지 않는다. 혼합정밀도는 민감한 head나 decoder block을 높은
정밀도로 남기고 나머지를 낮은 정밀도로 변환해 정확도와 효율의 중간점을 찾는다.
SmoothQuant와 AWQ는 원래 대규모 언어모델에서 제안된 원리이므로 RF-DETR에서 같은
효과가 보장된다고 가정하지 않고, 본 연구에서는 전이 가능한 후보 가설로만 다룬다.

TensorRT의 explicit quantization은 ONNX의 Quantize/Dequantize(Q/DQ) node를 통해
정밀도 경계를 표현한다. Q/DQ node 수는 양자화 적용 증거지만 실제 속도는 지원되는
GPU 형식, tactic과 kernel 선택에 따라 달라지므로 engine build log와 실측이 필요하다.

### 입력 해상도와 결합 경량화

영상 모델의 입력 해상도를 줄이면 공간 위치 수가 감소해 backbone, attention 및
mask 연산량을 직접 줄일 수 있다. 반면 작은 객체의 경계와 세부 정보가 손실될 수
있으므로 504, 480, 432, 384 해상도를 별도 후보로 비교한다. C01~C04는 구조 축소와
FP16·INT8 또는 해상도 축소를 결합해 서로 다른 경량화 축이 만드는 추가 Pareto
지점을 탐색한다.

### 평가 지표와 Pareto 선택

bbox AP와 mask AP는 IoU 0.50~0.95 구간을 0.05 간격으로 평균해 위치 및 mask
품질을 평가한다. semantic mIoU는 instance mask를 클래스별 영역으로 합성한 뒤
교집합/합집합 비율을 평균해 픽셀 수준 품질을 보완한다. 실행 성능은 batch 1,
고정된 32장에 대해 warm-up 20회 후 200회 측정을 3번 반복하며 pooled median
latency를 1차 대표값, p95/p99와 반복 median의 CV·95% 구간을 안정성 지표로 사용한다.

최종 선택에서는 B01 대비 정확도 보존 gate를 먼저 적용한 뒤 bbox AP·mask AP·
semantic mIoU를 극대화하고 median/p95 latency·peak allocated GPU memory·engine
크기를 최소화한다. 후보 A가 후보 B보다 모든
목표에서 같거나 우수하고 적어도 하나에서 엄격히 우수하면 A가 B를 지배한다.
어느 후보에도 지배되지 않는 집합을 Pareto front로 보고, 하나의 임의 가중치로
정확도와 효율을 합산하지 않는다.

## 데이터와 평가 타당성

Roboflow Version 9 무증강 원본을 촬영 세션 단위로 다시 나눴다. 인접 프레임이
서로 다른 split에 섞이는 누수를 막기 위해 이미지가 아니라 세션을 분할 단위로
사용했으며, 파일명·내용 SHA-256·세션 중복과 COCO 참조 무결성을 자동 검사했다.

| split | 이미지 | 역할 |
|---|---:|---|
| train | 3,625 | baseline/복구 학습, 양자화 calibration 후보 모집단 |
| valid | 404 | early stopping, Q05~Q07 민감도 선정 |
| benchmark | 437 | 노트북 2차 정확도 비교만 수행 |

현재 437장은 이전 후보 진단에도 반복 사용됐으므로 논문에서 “완전히 손대지 않은
독립 test set”이 아니라 **고정 비교 benchmark set**으로 기술한다. 외적 일반화
성능을 주장하려면 별도 촬영 세션 holdout이 추가로 필요하다. 자세한 근거는
[데이터 분할 타당성 보고서](docs/reports/dataset-split-validity.md)에 있다.

## 3단계 실험 설계

### 1차: 전체 후보 정적평가 — 완료

등록된 26개 후보를 모두 평가했다. 구조·해상도 후보는 유효한 ONNX와 함께 B01
대비 ONNX 크기, graph node, dense MAC 중 적어도 하나가 5% 이상 감소해야 한다.
precision 후보는 유효 source ONNX와 재현 가능한 TensorRT recipe 또는 후보 자체
Q/DQ ONNX를 요구한다. 2:4 후보는 패턴 준수와 sparse tactic recipe를 확인한다.
정확도, latency, FPS와 GPU memory는 1차 gate에서 제외한다.

### 2차: 노트북 TensorRT 정확도·성능 평가 — 다음 단계

1차 통과 22개 전부에 대해 engine build를 시도한다. build, engine inspection,
benchmark, 정확도 평가 중 실패하면 이유와 로그를 남겨 terminal 실패로 처리한다.
성공한 후보는 동일 장비에서 다음 항목을 측정한다.

- 정확도: 고정 benchmark 437장, bbox/mask AP·AP50·AP75·크기별 AP·AR100,
  semantic mIoU와 클래스별 AP/IoU
- 성능: batch 1, 고정 32장, 반복마다 warm-up 20회 후 200회 측정을 3회 수행
- 통계: pooled median·mean·p95·p99·IQR·FPS, 반복 median의 CV와 95% t 구간
- 자원: peak allocated/reserved GPU memory, engine 크기
- 환경: GPU, compute capability, driver, CUDA, TensorRT 버전
- 재현성: ONNX·engine SHA-256, layer precision, M02 sparse tactic 근거

### 3차: Pareto 분석 — 2차 전체 terminal 후 자동 실행

22개 모두가 성공 또는 명시적 실패 상태가 되어야 시작한다. 측정이 유효하고 B01
대비 bbox/mask AP와 mIoU 보존 gate를 통과한 engine만 비지배 판정에 사용한다.
실패·gate 탈락 후보의 수치와 사유도 삭제하지 않고 Excel과 원시 결과에 보존한다.

## 데스크탑과 노트북 작업 분리

TensorRT engine은 GPU architecture, CUDA와 TensorRT 버전에 종속된다. 최종 비교
장비가 노트북이라면 portable ONNX와 실험 정의는 데스크탑에서 만들고, engine과
최종 성능값은 노트북에서 생성해야 한다.

### 이 데스크탑에서 할 일

#### 노트북 실행 전

1. **완료 상태 확인**
   `show_project_status.py`에서 1차가 `26/26`, `unperformed 0`인지 확인한다.
2. **정적 결과 고정**
   `generate_stage1_static_evaluation.py --check`로 registry, 후보별 정적 보고서와
   전체 표가 일치하는지 검사한다.
3. **자동화 회귀 검사**
   전체 테스트를 실행해 데이터 분할, evaluator, 후보 선택, 패키징과 Pareto gate의
   변경 여부를 확인한다.
4. **노트북 전달 묶음 확인**
   `delivery/notebook-front-stage1/`에 22개 engine plan과 17개 unique ONNX,
   calibration 128장 및 benchmark 437장이 있는지 manifest와 SHA-256으로 확인한다.
5. **전달**
   22개 recipe의 중복 제거 ONNX 17개는 `shared-models/`에서 Git LFS로 받는다.
   데이터셋은 GitHub에 올리지 않으므로 calibration/test 데이터를 별도로 복사한다.
   PTH와 장비 종속 TensorRT engine도 원격 저장소에 올리지 않는다.

현재 1~4번은 완료됐다. 실제 노트북이 준비되면 5번만 수행하면 된다.

```bash
.venv/bin/python scripts/reporting/show_project_status.py
.venv/bin/python scripts/reporting/generate_stage1_static_evaluation.py --check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
.venv/bin/python scripts/experiments/package_notebook_bundle.py \
  --suite stage1 --output-dir delivery/notebook-front-stage1 --dry-run
```

#### 노트북 결과 회수 후

1. `stage2-evaluation-report.xlsx`, `stage2-notebook-state.json`, summary, Pareto,
   후보별 평가 JSON과 전체 로그를 동일한 상대경로로 데스크탑 저장소에 복사한다.
2. `audit_evaluation_protocol.py`를 다시 실행해 후보 누락, 데이터·threshold 차이,
   반복 횟수와 장비 메타데이터 누락을 검사한다.
3. 자동 생성 표·그림을 갱신하고 Pareto 후보의 정량 근거를 논문 결과 장에 반영한다.
4. build 실패 후보는 삭제하지 않고 지원되지 않은 precision/kernel과 장비 정보를
   실패 원인으로 보고한다.
5. 필요하면 별도 세션 holdout을 수집해 Pareto 선택 이후 한 번만 외적 검증한다.

이 데스크탑에서 생성한 TensorRT 성능을 노트북 결과와 섞거나, benchmark 437장을
calibration 또는 후보 민감도 선정에 재사용하면 안 된다.

### 노트북에서 할 일

1. **전달 파일 배치**
   `notebook-front-stage1` 폴더를 노트북의 로컬 SSD에 복사하고 그 폴더를 연다.
2. **원클릭 실행**
   `bash run_notebook_pipeline.sh` 한 줄을 실행한다. Python 3.10 가상환경 설치,
   ONNX·데이터 무결성 검사, 환경 기록, 22개 engine 생성, 반복 benchmark, 437장
   정확도 평가, 정확도 gate, Pareto 및 Excel 보고서 생성이 순차 실행된다.
3. **재시작**
   중단 후 같은 명령을 실행하면 완료 후보를 재사용한다. engine까지 전부 다시 만들
   때만 `--force-rebuild`를 사용한다.
4. **진행 모니터링**
   별도 터미널에서 `show_project_status.py`를 주기적으로 실행한다. 실패가 발생해도
   다음 후보가 계속 진행되며 실패 로그가 남는다.
5. **완료 조건 확인**
   Stage 2가 `terminal 22/22`인지 확인한다. 성공 수와 실패 수의 합이 22여야 한다.
6. **Pareto 결과 확인**
   전체 terminal 후 `stage3-pareto.json|csv`가 생성됐는지 확인한다. 부분 실행용
   `--only` 옵션은 공식 Pareto 파일을 생성하지 않는다.
7. **결과 회수**
   최종 `results/stage2-evaluation-report.xlsx`와 `results/`, 후보별 engine metadata와
   build log를 데스크탑으로 복사한다. engine
   자체는 논문 근거 보존이 필요할 때만 별도 저장하며 다른 GPU 성능 측정에 재사용하지
   않는다.

```bash
bash run_notebook_pipeline.sh
```

별도 터미널:

```bash
watch -n 5 '.venv/bin/python scripts/reporting/show_project_status.py'
```

## 재현 설정

- 데이터 설정: [configs/dataset.yaml](configs/dataset.yaml)
- 기준 모델 설정: [configs/baseline.yaml](configs/baseline.yaml)
- 학습 설정: [configs/training/front_rfdetr_seg_large.yaml](configs/training/front_rfdetr_seg_large.yaml)
- 26개 후보 registry: [configs/experiments/registry.yaml](configs/experiments/registry.yaml)
- 평가·수용 기준: [configs/experiments/defaults.yaml](configs/experiments/defaults.yaml)
- 모델·결과 인덱스: [docs/handoffs/model-artifact-index.md](docs/handoffs/model-artifact-index.md)

Python 3.10 환경에서 실행한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

데이터셋, checkpoint/PTH, 로컬 ONNX와 TensorRT engine은 Git에서 제외한다. 논문용
설정, 생성 명령, hash, 정적 보고서와 집계 결과만 버전 관리한다.

## 저장소 구조

| 폴더 | 역할 | 안내 |
|---|---|---|
| `configs/` | 데이터·학습·실험·평가의 단일 설정 원본 | [README](configs/README.md) |
| `data/` | 로컬 데이터셋; Git 제외 | [README](data/README.md) |
| `artifacts/` | checkpoint, ONNX, engine, 후보별 근거 | [README](artifacts/README.md) |
| `shared-models/` | 검증된 portable ONNX manifest | [README](shared-models/README.md) |
| `results/` | 평가 JSON, 상태, 논문용 집계표 | [README](results/README.md) |
| `figures/` | 보고서·논문용 그림 | [README](figures/README.md) |
| `docs/` | 이론, 절차, 인수인계, 결과 보고서 | [README](docs/README.md) |
| `scripts/` | 데이터 준비부터 Pareto까지의 실행 진입점 | [README](scripts/README.md) |
| `src/` | 스크립트가 공유하는 Python 구현 | [README](src/README.md) |
| `notebooks/` | TensorRT 재현용 노트북 | [README](notebooks/README.md) |
| `tests/` | 자동화·평가 프로토콜 회귀 테스트 | [README](tests/README.md) |

## 참고문헌 및 기술 근거

1. Robinson et al., [RF-DETR: Neural Architecture Search for Real-Time Detection Transformers](https://arxiv.org/abs/2511.09554), 2025.
2. Carion et al., [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872), ECCV 2020.
3. Zhang et al., [DINO: DETR with Improved DeNoising Anchor Boxes](https://arxiv.org/abs/2203.03605), ICLR 2023.
4. Li et al., [Mask DINO: Towards a Unified Transformer-based Framework for Object Detection and Segmentation](https://arxiv.org/abs/2206.02777), 2022.
5. Han et al., [Learning both Weights and Connections for Efficient Neural Networks](https://papers.nips.cc/paper_files/paper/2015/hash/ae0eb3eed39d2bcef4622b2499a05fe6-Abstract.html), NeurIPS 2015.
6. Li et al., [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710), ICLR 2017.
7. Xiao et al., [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://proceedings.mlr.press/v202/xiao23c.html), ICML 2023.
8. Lin et al., [AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html), MLSys 2024.
9. NVIDIA, [Accelerating Inference with Sparsity Using Ampere and TensorRT](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/).
10. NVIDIA, [TensorRT: Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html).
11. Lin et al., [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312), ECCV 2014.
12. Deb et al., [A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II](https://doi.org/10.1109/4235.996017), IEEE TEC 2002.
