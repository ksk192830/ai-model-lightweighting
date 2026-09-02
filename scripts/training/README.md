# 학습 실행기

- `train_rfdetr_front.py`: 데이터·환경 preflight 후 baseline 학습
- `watch_rfdetr_training.py`: baseline epoch/validation 진행도 표시
- `watch_lightweighting_pipeline.py`: recovery와 후속 queue를 한 화면에 표시

baseline 설정은 `configs/training/`, pruning 복구 설정은
`configs/experiments/defaults.yaml`에서 읽는다. checkpoint 선택은 valid
metric으로 수행하고 test 평가는 `scripts/evaluation/`에서 별도로 실행한다.
