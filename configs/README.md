# 설정

실험을 재현할 때 가장 먼저 확인하는 단일 설정 원본이다. 실행 스크립트에
같은 숫자를 다시 적지 말고 여기의 값을 읽도록 유지한다.

- `baseline.yaml`: 기준 checkpoint, 모델 종류, 입력 크기, class 순서
- `dataset.yaml`: Roboflow 버전, 무증강 여부, split 경로와 이미지 수
- `training/`: baseline 학습 hyperparameter와 출력 위치
- `experiments/defaults.yaml`: ONNX/TensorRT, 평가 프로토콜, 후보 수용 기준
- `experiments/registry.yaml`: 후보 ID, 방법, 의존성, 진행 상태와 결과
- `experiments/schema.yaml`: 레지스트리 필드 형식

경로는 저장소 루트 기준 상대 경로로 기록한다. API token, 계정 정보, 로컬
절대 경로는 설정 파일에 넣지 않는다.
