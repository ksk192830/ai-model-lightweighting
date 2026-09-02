# RF-DETR 경량화 및 최적화

무증강 주차장 전방 영상으로 다시 학습한 RF-DETR Segmentation Large를
기준으로 구조 pruning, 입력 해상도 축소, NVIDIA 2:4 sparsity, TensorRT
FP16/INT8을 비교하는 논문 실험 저장소다.

- GitHub: [kips-ai-model-lightweighting](https://github.com/ksk192830/kips-ai-model-lightweighting)
- Notion: [AI 경량화 및 최적화 프로젝트](https://app.notion.com/p/389d4a78ceed80d28d95c37d83422a60)
- 인수인계 시작점: [현재 진행 현황](docs/PROJECT_STATUS.md)
- 논문용 정적 분석: [paper-static-analysis.md](docs/reports/paper-static-analysis.md)
- 평가 설계 감사: [evaluation-automation-audit.md](docs/reports/evaluation-automation-audit.md)

## 5분 요약

| 항목 | 현재 상태 | 핵심 결과/다음 단계 |
|---|---|---|
| 데이터 | 완료 | 세션 단위 분리, 3,625 / 404 / 437장, 교차 split 중복 0 |
| B01 기준 모델 | 완료 | bbox AP 0.737791, mask AP 0.601016, mIoU 0.712200 |
| R01 432 입력 | 후보 유지 | MAC/FLOP 추정치 30.20% 감소, bbox AP -0.007130 |
| S01 decoder -1 | 후보 유지 | node 8.04%, ONNX 4.74% 감소, bbox AP -0.000411 |
| S02 decoder -2 | 정확도 탈락·속도 대조군 | node 16.17% 감소, bbox AP -0.018335 |
| U02 비정형 30% | 탈락 | dense graph 연산량 감소 없음, 정확도 크게 하락 |
| S03 FFN 축소 | 탈락 | 전체 MAC 0.39% 감소에 그침 |
| M01/M02 2:4 | 진행 중 | M01 복구 학습 후 dense/sparse TensorRT 비교 예정 |
| TensorRT 실측 | 대기 | 같은 장비에서 FP32/FP16/INT8 latency·FPS·memory 측정 예정 |

위 숫자는 고정된 437장 비교용 benchmark set의 단일 실행 결과다. 이 set은
후보 선택에 반복 사용됐으므로 논문에서 완전히 손대지 않은 최종 test set으로
표현하지 않는다. 외부 일반화 주장은 새로운 촬영 세션 holdout이 필요하다.

## 재현의 기준

- 데이터 설정: [configs/dataset.yaml](configs/dataset.yaml)
- 기준 모델 설정: [configs/baseline.yaml](configs/baseline.yaml)
- 학습 설정: [configs/training/front_rfdetr_seg_large.yaml](configs/training/front_rfdetr_seg_large.yaml)
- 실험 레지스트리: [configs/experiments/registry.yaml](configs/experiments/registry.yaml)
- 평가·수용 기준: [configs/experiments/defaults.yaml](configs/experiments/defaults.yaml)
- 모델/결과 인덱스: [docs/handoffs/model-artifact-index.md](docs/handoffs/model-artifact-index.md)

모든 후보는 같은 baseline checkpoint에서 파생된다. 정확도 평가는 437장,
COCO AP confidence 0.001, semantic mIoU confidence 0.25를 사용한다. latency는
batch 1, 고정 seed로 선택한 32장, warm-up 20회, 측정 200회의 in-memory
end-to-end 범위로 측정한다. 상세 정의는 위 `defaults.yaml`이 단일 기준이다.

## 저장소 구조

| 폴더 | 역할 | 안내 |
|---|---|---|
| `configs/` | 데이터·학습·실험·평가의 단일 설정 원본 | [README](configs/README.md) |
| `data/` | 로컬 데이터셋; Git에서 제외 | [README](data/README.md) |
| `artifacts/` | checkpoint, 중간 모델, 재현 메타데이터 | [README](artifacts/README.md) |
| `shared-models/` | 검증된 portable ONNX; Git LFS 사용 | [README](shared-models/README.md) |
| `results/` | 원시 평가 JSON과 논문용 집계표 | [README](results/README.md) |
| `figures/` | 보고서·논문용 그림 | [README](figures/README.md) |
| `docs/` | 개념, 절차, 인수인계, 결과 보고서 | [README](docs/README.md) |
| `scripts/` | 데이터 준비부터 보고서 생성까지의 실행 진입점 | [README](scripts/README.md) |
| `src/` | 스크립트가 공유하는 Python 구현 | [README](src/README.md) |
| `notebooks/` | 다른 NVIDIA 장비에서 TensorRT를 재현하는 노트북 | [README](notebooks/README.md) |
| `tests/` | 자동화·평가 프로토콜 회귀 테스트 | [README](tests/README.md) |

## 설치와 최소 검증

Python 3.10 환경에서 실행한다. ONNX 파일을 받으려면 Git LFS가 필요하다.

```bash
git lfs install
git lfs pull
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

데이터셋과 로컬 checkpoint/PTH/TensorRT engine은 Git에 올리지 않는다. 현재
baseline checkpoint의 SHA-256은
`769ee97e38a2c6665bd664a9e035f5649d9f0a45c2bfe20172fe92ee545e5620`이며,
공유 ONNX의 해시는 [manifest.yaml](shared-models/manifest.yaml)에서 확인한다.

## 다음 실행 순서

1. 진행 중인 M01 2:4 복구 학습과 PTH–ONNX 동등성 검증을 완료한다.
2. B01, R01, S01, S02, M01/M02의 TensorRT engine을 동일 데스크탑에서 만든다.
3. 고정 프로토콜로 latency·FPS·peak GPU memory와 전체 benchmark 정확도를 측정한다.
4. 수용 기준과 Pareto 분석으로 최종 후보를 선택하고 표·그림을 다시 생성한다.

세부 명령과 완료 조건은 [현재 진행 현황](docs/PROJECT_STATUS.md)과
[데스크탑 TensorRT 파이프라인](docs/guides/desktop-tensorrt-pipeline.md)을 따른다.
