# 테스트

데이터를 다운로드하거나 전체 모델을 학습하지 않고, 평가 수식·ONNX 해석·
pruning·TensorRT 명령 구성·queue 연결·보고서 생성을 회귀 검증한다.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

TensorRT/CUDA가 없는 환경에서는 실제 engine build 대신 명령과 메타데이터
계약을 검사한다. 전체 정확도와 latency 검증은 별도 실험 실행 결과로 판단한다.
