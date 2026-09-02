# Reporting

평가가 끝난 machine-readable JSON에서 논문용 표와 그래프를
재생성하는 스크립트를 둔다.
모델 추론과 GPU benchmark는 `../evaluation/`에서 수행하며, 이 디렉터리는
기존 결과를 읽어 표현 형식과 비교 산출물만 만든다.

```bash
.venv/bin/python scripts/reporting/paper_results.py
.venv/bin/python scripts/reporting/generate_paper_static_analysis.py
.venv/bin/python scripts/experiments/audit_evaluation_protocol.py
```

입력:

- `results/desktop-engine-summary.json`
- `artifacts/experiments/*/front/static-analysis.json`
- `configs/experiments/defaults.yaml`

출력:

- `results/results_final.csv`
- `figures/*.png`
- `results/paper-static-analysis.csv`
- `docs/reports/paper-static-analysis.md`
- `results/evaluation-automation-audit.json`
- `docs/reports/evaluation-automation-audit.md`
