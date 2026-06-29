# Lightweighting result evaluation

This directory contains tools for inspecting and timing original and
lightweighted models.

- `infer_baseline.py`: compare annotated TensorRT FP32, FP16, and INT8 predictions
- `benchmark_baseline.py`: benchmark PyTorch checkpoints or TensorRT engines
- `visualize_inference_stream.py`: compare sequential predictions live or as MP4

Example:

```bash
.venv/bin/python scripts/evaluation/infer_baseline.py \
  --backend all \
  --camera front \
  --image data/labeled_test/front/images/image000002.png \
  --device cuda
```
