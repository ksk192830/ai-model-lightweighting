# 실행 스크립트

| 순서 | 폴더 | 역할 |
|---:|---|---|
| 1 | [`data_preparation/`](data_preparation/README.md) | 무증강 export, 세션 분할, 누수 감사 |
| 2 | [`training/`](training/README.md) | baseline 학습과 진행도 확인 |
| 3 | [`experiments/`](experiments/README.md) | 레지스트리 기반 후보 생성·복구·queue 실행 |
| 4 | [`lightweighting/`](lightweighting/README.md) | pruning, ONNX export, TensorRT build 구현 |
| 5 | [`evaluation/`](evaluation/README.md) | PTH/ONNX/engine 정확도와 latency 평가 |
| 6 | [`reporting/`](reporting/README.md) | 논문용 CSV·표·그림·감사 보고서 생성 |

모든 명령은 저장소 루트에서 프로젝트 가상환경으로 실행한다.

```bash
.venv/bin/python scripts/<group>/<script>.py
```

일반 실험은 `experiments/`의 상위 진입점을 사용하고, `lightweighting/`의
저수준 스크립트는 디버깅이나 단일 변환이 필요할 때 사용한다. 공통 threshold,
seed, 반복 횟수는 `configs/experiments/defaults.yaml`을 기준으로 한다.
