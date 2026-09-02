# 노트북

TensorRT가 설치된 NVIDIA 장비에서 portable ONNX를 engine으로 변환하고 같은
평가 프로토콜을 실행하기 위한 전달용 노트북을 둔다.

`front_tensorrt_ready4.ipynb`는 4개 후보만 다루는 기존 예시 UI다. 현재 전체
평가는 `results/stage1-static-evaluation.json`의 1차 통과 후보를 읽는
`scripts/experiments/run_notebook_stage2.py`를 사용한다. 실험 정의의 단일 원본은
`configs/experiments/`이며, engine은 생성 GPU/TensorRT 환경에 종속되므로 Git에
올리지 않는다.
