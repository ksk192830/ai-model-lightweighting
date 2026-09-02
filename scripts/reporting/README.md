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

`show_project_status.py`는 1차의 평가/통과/불통/미수행 수와 2차의
완료/실패/실행/대기 후보를 실제 상태 JSON에서 읽는다. 3차 Pareto가 생성되기
전에는 `WAITING`, 생성된 뒤에는 비지배 후보 ID를 표시한다.
