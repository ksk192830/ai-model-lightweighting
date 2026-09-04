# Reporting

평가가 끝난 machine-readable JSON에서 논문용 표와 그래프를
재생성하는 스크립트를 둔다.
모델 추론과 GPU benchmark는 `../evaluation/`에서 수행하며, 이 디렉터리는
기존 결과를 읽어 표현 형식과 비교 산출물만 만든다.

```bash
.venv/bin/python scripts/reporting/show_project_status.py
.venv/bin/python scripts/reporting/generate_stage1_static_evaluation.py --check
.venv/bin/python scripts/reporting/paper_results.py
.venv/bin/python scripts/reporting/generate_paper_static_analysis.py
.venv/bin/python scripts/reporting/generate_stage2_excel.py
.venv/bin/python scripts/reporting/generate_stage3_paper_assets.py
.venv/bin/python scripts/experiments/audit_evaluation_protocol.py
```

입력:

- `results/desktop-engine-summary.json`
- `results/stage1-static-evaluation.json`
- `results/stage2-notebook-state.json`
- `results/stage3-pareto.json`
- `artifacts/experiments/*/front/static-analysis.json`
- `configs/experiments/defaults.yaml`

출력:

- `results/results_final.csv`
- `figures/*.png`
- `results/paper-static-analysis.csv`
- `docs/reports/paper-static-analysis.md`
- `results/evaluation-automation-audit.json`
- `docs/reports/evaluation-automation-audit.md`
- `results/stage2-evaluation-report.xlsx`
- `results/stage3-paper-candidate-decisions.csv`
- `results/stage3-final-recommendations.json`
- `docs/reports/stage3-final-analysis.md`
- `docs/paper/stage3-evidence-map.md`
- `figures/stage3_*.png`

`show_project_status.py`는 1차의 평가/통과/불통/미수행 수와 2차의
완료/실패/실행/대기 후보를 실제 상태 JSON에서 읽는다. 3차 Pareto가 생성되기
전에는 `WAITING`, 생성된 뒤에는 비지배 후보 ID를 표시한다.

`generate_stage2_excel.py`는 1차 후보표, 2차 성공·실패, 클래스별 정확도, 반복별
latency, 평가 기준, 실험환경과 Pareto를 한 Excel 파일로 만든다. 수치의 원본은
JSON/CSV로 유지하고, 비교·gate 시트는 B01 기준 변화량을 수식으로도 추적한다.

`generate_stage3_paper_assets.py`는 공식 3축 Pareto 결과가 C01·R01인지 확인한 뒤
최초 26개 후보의 최종 판정표, 용도별 추천, 논문 근거 연결표와 출판용 그림을
동일 JSON에서 재생성한다.
