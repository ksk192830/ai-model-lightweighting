# 결과 보고서

실험에서 관측한 수치, 판정, 논문용 해석을 둔다.

- `dataset-split-validity.md`: 분할 정량화와 누수 검사
- `front-rfdetr-seg-large-v1-test.md`: 신규 baseline test 결과
- `front-lightweighting-candidate-comparison.md`: 후보 정확도·구조 비교와 판정
- `paper-static-analysis.md`: 동일 분석기로 계산한 파라미터·node·MAC/FLOP
- `evaluation-automation-audit.md`: 평가 기준, provenance, 자동화 누락 감사
- `notebook-stage2-results.md`: 노트북 TensorRT 실측, 정확도 gate, Pareto와 후속 작업
- `stage3-final-analysis.md`: 공식 결과 동결, 26개 최종 판정, Pareto와 용도별 추천
- `measurement-load-control.md`: Stage 2 지연시간 재측정의 부하 통제 기준과 기록 방식
- `metrics/`: 데이터 분할 보고서의 CSV/JSON 근거

`paper-static-analysis.md`와 평가 감사 문서는 스크립트가 생성한다. 원시 결과는
`results/`, 생성 코드는 `scripts/reporting/`에 있다.
