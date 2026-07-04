# Experiment management

All commands use `configs/experiments/registry.yaml` as the source of truth.
Successful candidate creation, analysis, build, and training commands
automatically synchronize metadata and regenerate
`docs/handoffs/model-artifact-index.md`. The explicit synchronization commands below
are repair and verification commands, not steps that must be remembered after
every model.

```bash
# Create a registered pruning checkpoint.
.venv/bin/python scripts/experiments/create_candidate.py U02 --camera front

# Create a registered structured decoder-pruning checkpoint.
.venv/bin/python scripts/experiments/create_candidate.py S01 --camera front

# Create a registered structured FFN-pruning checkpoint.
.venv/bin/python scripts/experiments/create_candidate.py S03 --camera front

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

# Promote returned recovery PTH files and rebuild ONNX/engine/comparisons.
.venv/bin/python scripts/experiments/finalize_recovery.py \
  S01 S02 S03 S04 --camera front --force

# On a machine without CUDA/TensorRT, promote PTH and rebuild ONNX only.
.venv/bin/python scripts/experiments/finalize_recovery.py \
  S01 S02 S03 S04 --camera front --force --skip-engine

# Verify required artifacts, metadata references, and SHA-256 records.
.venv/bin/python scripts/experiments/audit_artifacts.py

# Assemble selected experiments for evaluator handoff.
.venv/bin/python scripts/experiments/package_delivery.py \
  --experiments B01 B02 B03 U02 --camera front

# Package portable ONNX inputs for another NVIDIA machine.
.venv/bin/python scripts/experiments/package_notebook_bundle.py \
  --suite final8 --force

# Rebuild the final engine suite and run one-image smoke inference.
.venv/bin/python scripts/experiments/build_engine_suite.py \
  --suite final8 --force
```
