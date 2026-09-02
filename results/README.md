# 결과

평가와 보고서 생성기의 machine-readable 출력을 둔다. 실행 중 로그·lock과
대형 중간 결과는 Git에서 제외하고, 논문 수치의 근거가 되는 작은 JSON/CSV만
완료 시점에 선별해 추적한다.

- `coco-evaluation/`: PTH, ONNX, TensorRT의 bbox AP, mask AP, semantic mIoU
- `round2-candidate-summary.csv|md`: 후보별 정적/정확도 요약
- `stage1-static-evaluation.json|csv|md`: 등록된 27개 전체 후보의 1차 정적평가
- `stage2-notebook-state.json`: 노트북 실행 중 후보별 현재/terminal 상태
- `stage2-notebook-summary.json|csv`: 노트북 정확도·성능 통합 결과
- `stage2-notebook-logs/`: 후보별 build·검사·benchmark·정확도 실패 근거
- `stage3-pareto.json|csv`: 2차 전체 terminal 후 생성되는 비지배 후보 집합
- `paper-static-analysis.csv`: 논문 표에 사용한 정적 분석 집계
- `evaluation-automation-audit.json`: 프로토콜과 artifact 자동 감사 결과
- `desktop-engine-summary.json`: 과거 데스크탑 실행용 통합 원본; 최종 노트북
  비교의 권위 있는 결과는 `stage2-notebook-summary.*`를 사용

사람이 읽는 해석은 `docs/reports/`에 있고, 재생성 코드는
`scripts/reporting/`에 있다.
