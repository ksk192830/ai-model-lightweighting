# `kips_lightweighting`

- `registry.py`: 실험 레지스트리 조회와 검증
- `artifacts.py`, `metadata.py`: 경로, hash, provenance 기록
- `rfdetr_compat.py`: RF-DETR 버전 차이를 흡수하는 호환 계층
- `onnx_export.py`: ONNX export 공통 로직
- `static_analysis.py`: checkpoint/ONNX 구조·희소성·부분 MAC/FLOP 분석
- `tensorrt_build.py`: FP32/FP16/INT8 engine build 공통 로직
- `pruning/`: 비정형, 구조적, 2:4 pruning 구현

CLI 출력 형식을 바꾸면 `tests/`와 보고서 생성기가 소비하는 JSON schema도
함께 확인한다.
