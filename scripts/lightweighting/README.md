# Model lightweighting

This directory contains the model conversion pipeline.

- `check_environment.py`: validate RF-DETR, PyTorch, and checkpoints
- `export_onnx.py`: export and validate ONNX models
- `build_tensorrt_fp32.py`: build TensorRT FP32 reference engines
- `build_tensorrt_fp16.py`: build TensorRT FP16 engines
- `build_tensorrt_int8.py`: calibrate and build TensorRT INT8 engines
- `run_lightweighting.py`: orchestrate front/rear conversions

Example:

```bash
.venv/bin/python scripts/lightweighting/run_lightweighting.py \
  --camera all \
  --methods onnx tensorrt-int8
```
