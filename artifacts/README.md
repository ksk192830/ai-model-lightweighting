# 로컬 Artifact

학습 checkpoint, 후보 PTH/ONNX, TensorRT engine과 실행 로그가 생성되는 작업
영역이다. 대형 바이너리와 실행 중 파일은 기본적으로 Git에서 제외한다.
논문 수치의 근거가 되는 작은 JSON/CSV 메타데이터만 선별해 추적할 수 있다.

- `training/`: baseline 학습 checkpoint와 metrics
- `experiments/<ID>/<camera>/`: pruning, recovery, ONNX, 정적 분석 기록

재현 가능한 portable ONNX만 `shared-models/`에 Git LFS로 승격한다. 파일의
출처와 SHA-256은 실험 레지스트리 및 manifest에 기록한다.
