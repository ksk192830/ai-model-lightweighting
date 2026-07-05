# Model lightweighting

This directory contains the model conversion pipeline.

- `check_environment.py`: validate RF-DETR, PyTorch, and checkpoints
- `export_onnx.py`: export and validate ONNX models
- `build_tensorrt_fp32.py`: build TensorRT FP32 reference engines
- `build_tensorrt_fp16.py`: build TensorRT FP16 engines
- `build_tensorrt_int8.py`: calibrate and build TensorRT INT8 engines
- `quantize_modelopt.py`: nvidia-modelopt quantization (Q01 SmoothQuant INT8,
  Q02 sensitive-layer-FP16 mixed precision, Q03 INT4 weight-only FFN) with
  Q/DQ ONNX export; build the result with `build_tensorrt_fp16.py --int8-qdq`

These are low-level implementation scripts. Normal experiment work uses the
registry entry points:

```bash
.venv/bin/python scripts/experiments/build_candidate.py \
  B01 --camera front --target onnx
```

Create and analyze a registered unstructured candidate:

```bash
.venv/bin/python scripts/experiments/create_candidate.py U02 --camera front
.venv/bin/python scripts/experiments/analyze_candidate.py U02 --camera front
```

Export and build it:

```bash
.venv/bin/python scripts/experiments/build_candidate.py \
  U02 --camera front --target onnx
.venv/bin/python scripts/experiments/build_candidate.py \
  U02 --camera front --target engine
```

The experiment workflow keeps 10/30/50% front checkpoints for static analysis
and builds only the selected representative as TensorRT. See
[`docs/guides/experiment-workflow.md`](../../docs/guides/experiment-workflow.md).

Artifacts follow the experiment-centric layout:

```text
artifacts/experiments/<experiment-id>/<camera>/
```
