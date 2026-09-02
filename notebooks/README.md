# 노트북

TensorRT가 설치된 NVIDIA 장비에서 portable ONNX를 engine으로 변환하고 같은
평가 프로토콜을 실행하기 위한 전달용 노트북을 둔다.

`front_tensorrt_ready4.ipynb`는 B01, R01, S01, S02 입력을 대상으로 하는 현재
노트북이다. 노트북은 편의용 UI이며 실험 정의의 단일 원본은
`configs/experiments/`이다. engine은 생성 GPU/TensorRT 환경에 종속되므로
Git에 올리지 않는다.
