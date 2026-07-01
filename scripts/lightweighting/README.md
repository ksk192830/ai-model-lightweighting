# Model lightweighting

This directory contains the model conversion pipeline.

- `check_environment.py`: validate RF-DETR, PyTorch, and checkpoints
- `export_onnx.py`: export and validate ONNX models
- `build_tensorrt_fp32.py`: build TensorRT FP32 reference engines
- `build_tensorrt_fp16.py`: build TensorRT FP16 engines
- `build_tensorrt_int8.py`: calibrate and build TensorRT INT8 engines

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
[`docs/experiment-workflow.md`](../../docs/experiment-workflow.md).

Artifacts follow the experiment-centric layout:

```text
artifacts/experiments/<experiment-id>/<camera>/
```
