# 실험 산출물

후보별 구조는 `artifacts/experiments/<experiment-id>/<camera>/`이다.
`metadata.json`, `static-analysis.json`, `comparison-B01.json`,
`onnx-equivalence.json`이 핵심 추적 근거다. PTH, 중간 ONNX, checkpoint,
engine과 실시간 로그는 로컬에만 둔다.

후보 의미와 상태는 `configs/experiments/registry.yaml`, 사람이 읽는 링크
목록은 `docs/handoffs/model-artifact-index.md`에서 확인한다.
