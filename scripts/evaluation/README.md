# Lightweighting result evaluation

This directory contains tools for inspecting and timing original and
lightweighted models.

- `infer_baseline.py`: compare annotated TensorRT FP32, FP16, and INT8 predictions
- `benchmark_baseline.py`: benchmark PyTorch checkpoints or TensorRT engines
- `visualize_inference_stream.py`: compare sequential predictions live or as MP4
- `evaluate_coco_tensorrt.py`: measure COCO bbox/segmentation AP and latency

Example:

```bash
.venv/bin/python scripts/evaluation/infer_baseline.py \
  --backend all \
  --camera front \
  --image data/labeled_test/front/images/image000002.png \
  --device cuda
```

Benchmark an unstructured-pruned PyTorch checkpoint:

```bash
.venv/bin/python scripts/evaluation/benchmark_baseline.py \
  --camera front \
  --backend pytorch \
  --checkpoint artifacts/experiments/U01/front/model.pth \
  --image data/labeled_test/front/images/image000522.png \
  --device cuda
```

Evaluate a registered TensorRT engine on the complete labeled split:

```bash
.venv/bin/python scripts/evaluation/evaluate_coco_tensorrt.py \
  --experiment S02 --camera front
```

Show the selected eight front engines in a live 2×4 comparison:

```bash
.venv/bin/python scripts/evaluation/visualize_inference_stream.py \
  --camera front \
  --image-dir data/labeled_test/front/images \
  --backend final8 --interval 0.1 --live
```
