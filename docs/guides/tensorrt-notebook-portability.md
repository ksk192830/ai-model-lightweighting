# NVIDIA 노트북에서 TensorRT engine 생성·검증

TensorRT engine은 생성한 GPU, TensorRT, CUDA 환경에 종속된다. 데스크탑에서
만든 `.engine`을 노트북에 그대로 복사하는 대신, portable ONNX bundle을
복사하고 노트북에서 engine을 다시 생성한다.

## 요구 조건

- Linux x86-64
- NVIDIA GPU와 정상 동작하는 driver
- Python 3.10+
- CUDA 지원 PyTorch
- TensorRT 10.16.1.11
- 최소 4 GiB TensorRT workspace 여유

```bash
nvidia-smi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-tensorrt.txt
```

Albumentations/Roboflow가 `opencv-python-headless`를 다시 설치하면 live GUI와
충돌할 수 있다. 실시간 비교가 필요하면 마지막에 다음을 실행한다.

```bash
python -m pip uninstall -y opencv-python-headless
python -m pip install --force-reinstall opencv-python==4.13.0.92
```

## Portable ONNX bundle 준비

변환 데스크탑에서 최종 8개가 참조하는 ONNX만 묶는다.
새 clone에서는 먼저 Git LFS 모델을 내려받는다.

```bash
git lfs pull

.venv/bin/python scripts/experiments/package_notebook_bundle.py \
  --suite final8 --force
```

생성된 `delivery/notebook-front/`의 내용을 노트북 저장소 루트에 덮어쓴다.
Git 저장소의 코드·설정과 bundle의 ONNX가 모두 필요하다.

INT8 B03을 다시 만들려면 calibration 데이터도 내려받는다.

```bash
.venv/bin/python scripts/data_preparation/download_data.py \
  --subset calibration
```

평가용 샘플과 전체 COCO 평가가 필요하면 labeled test도 받는다.

```bash
.venv/bin/python scripts/data_preparation/download_data.py \
  --subset labeled-test
```

## Final 8 일괄 생성과 smoke test

실행 계획 확인:

```bash
.venv/bin/python scripts/experiments/build_engine_suite.py \
  --suite final8 --dry-run
```

노트북 GPU용 engine을 전부 다시 생성하고 샘플 추론을 검증한다.

```bash
.venv/bin/python scripts/experiments/build_engine_suite.py \
  --suite final8 --force
```

이미 engine을 생성했다면 빌드는 생략하고 역직렬화·추론만 검사한다.

```bash
.venv/bin/python scripts/experiments/build_engine_suite.py \
  --suite final8 --skip-build
```

결과는 `results/notebook-engine-smoke.json`에 GPU, CUDA, TensorRT 버전,
engine SHA-256, 입출력 shape, 검출 수, latency와 FPS로 기록된다.

13개 실험 engine을 모두 재생성하려면 `--suite all13`을 사용한다. 이 경우
portable bundle도 같은 suite로 다시 만든다.

```bash
.venv/bin/python scripts/experiments/package_notebook_bundle.py \
  --suite all13 --output-dir delivery/notebook-front-all13 --force

.venv/bin/python scripts/experiments/build_engine_suite.py \
  --suite all13 --force
```

## 시각 비교

```bash
.venv/bin/python scripts/evaluation/visualize_inference_stream.py \
  --camera front \
  --image-dir data/labeled_test/front/images \
  --backend final8 --interval 0.1 --live
```

노트북 GPU가 RTX 3080과 다르면 engine 크기와 latency가 달라질 수 있으므로
노트북에서 생성된 smoke report와 COCO 평가 결과를 별도 보관한다.
