# 실행 가이드

재현 순서는 다음과 같다.

1. `leakage-safe-dataset-split.md`: 무증강 export를 세션 단위로 분할·감사
2. `front-baseline-training.md`: RF-DETR baseline 학습과 최종 평가
3. `three-stage-candidate-evaluation.md`: 전 후보 1차 정적 → 노트북 2차 → Pareto 절차
4. `front-lightweighting-round-2.md`: 후보 생성·복구 배경
5. `additional-candidate-screening.md`: 추가 graph 후보 설계와 기존 진단 기록
6. `desktop-tensorrt-pipeline.md`: 데스크탑 engine 생성과 자동 평가
7. `tensorrt-notebook-portability.md`: 다른 NVIDIA 장비로 ONNX/설정을 전달

현재 완료 여부는 `../PROJECT_STATUS.md`, 관측 결과는 `../reports/`에서 확인한다.
