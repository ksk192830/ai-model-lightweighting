# 실행 가이드

재현 순서는 다음과 같다.

1. `leakage-safe-dataset-split.md`: 무증강 export를 세션 단위로 분할·감사
2. `front-baseline-training.md`: RF-DETR baseline 학습과 최종 평가
3. `front-lightweighting-round-2.md`: 후보 생성·복구·선정 순서
4. `desktop-tensorrt-pipeline.md`: 데스크탑 engine 생성과 자동 평가
5. `tensorrt-notebook-portability.md`: 다른 NVIDIA 장비로 ONNX/설정을 전달

현재 완료 여부는 `../PROJECT_STATUS.md`, 관측 결과는 `../reports/`에서 확인한다.
