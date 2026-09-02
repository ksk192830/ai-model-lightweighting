# 결과

평가와 보고서 생성기의 machine-readable 출력을 둔다. 실행 중 로그·lock과
대형 중간 결과는 Git에서 제외하고, 논문 수치의 근거가 되는 작은 JSON/CSV만
완료 시점에 선별해 추적한다.

- `coco-evaluation/`: PTH, ONNX, TensorRT의 bbox AP, mask AP, semantic mIoU
- `round2-candidate-summary.csv|md`: 후보별 정적/정확도 요약
- `paper-static-analysis.csv`: 논문 표에 사용한 정적 분석 집계
- `evaluation-automation-audit.json`: 프로토콜과 artifact 자동 감사 결과
- `desktop-engine-summary.json`: TensorRT 실측 완료 후 생성될 통합 원본

사람이 읽는 해석은 `docs/reports/`에 있고, 재생성 코드는
`scripts/reporting/`에 있다.
