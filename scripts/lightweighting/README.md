# Model lightweighting

This directory contains the model conversion pipeline.

- `check_environment.py`: validate RF-DETR, PyTorch, and checkpoints
- `export_onnx.py`: export and validate ONNX models
- `build_tensorrt_fp32.py`: build TensorRT FP32 reference engines
- `build_tensorrt_fp16.py`: build TensorRT FP16 engines
- `build_tensorrt_int8.py`: calibrate and build TensorRT INT8 engines
- `quantize_modelopt.py`: nvidia-modelopt Q01~Q07 INT8/INT4/FP8/혼합 양자화,
  Q/DQ ONNX export, 보호 block 및 적용 quantizer 근거 기록
- `sensitivity_sweep.py`, `kld_sensitivity.py`: valid 분할에서 Q05~Q07
  FP16 보호 block을 선정
- `weight_bitwidth_study.py`: W01 W8/W4/W3/W2 7개 정적 연구

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

The registry keeps all candidate families for Stage-1 static analysis. Every
Stage-1-passing candidate enters the notebook Stage-2 queue. See
[`docs/guides/three-stage-candidate-evaluation.md`](../../docs/guides/three-stage-candidate-evaluation.md).

Artifacts follow the experiment-centric layout:

```text
artifacts/experiments/<experiment-id>/<camera>/
```
