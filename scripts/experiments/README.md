# Experiment management

All commands use `configs/experiments/registry.yaml` as the source of truth.

```bash
# Create a registered pruning checkpoint.
.venv/bin/python scripts/experiments/create_candidate.py U02 --camera front

# Write checkpoint and ONNX static analysis.
.venv/bin/python scripts/experiments/analyze_candidate.py U02 --camera front

# Validate 2:4 recovery-training prerequisites without starting training.
.venv/bin/python scripts/experiments/analyze_candidate.py \
  M01 --camera front --fine-tuning-preflight \
  --dataset-dir data/training/front

# Run 2:4 recovery fine-tuning after the preflight reports training_allowed=true.
.venv/bin/python scripts/experiments/train_candidate.py \
  M01 --camera front

# On another workstation, auto-select a GPU-aware batch size and a new output.
.venv/bin/python scripts/experiments/train_candidate.py \
  M01 --camera front --device auto \
  --output-dir artifacts/experiments/M01/front/recovery-portable

# Resume an interrupted run with optimizer and scheduler state.
.venv/bin/python scripts/experiments/train_candidate.py \
  M01 --camera front --device auto \
  --output-dir artifacts/experiments/M01/front/recovery-portable \
  --resume artifacts/experiments/M01/front/recovery-portable/checkpoint_4.ckpt

# Export ONNX or build the registered TensorRT precision.
.venv/bin/python scripts/experiments/build_candidate.py U02 --camera front --target onnx
.venv/bin/python scripts/experiments/build_candidate.py U02 --camera front --target engine

# Regenerate the human-readable artifact index.
.venv/bin/python scripts/experiments/sync_metadata.py
.venv/bin/python scripts/experiments/update_index.py

# Assemble selected experiments for evaluator handoff.
.venv/bin/python scripts/experiments/package_delivery.py \
  --experiments B01 B02 B03 U02 --camera front
```
