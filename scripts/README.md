# Scripts

Scripts are grouped by workflow:

- `data_preparation/`: download, select, and materialize datasets
- `experiments/`: registry-driven candidate creation and artifact management
- `lightweighting/`: check the environment and build ONNX/TensorRT models
- `evaluation/`: run inference, benchmark models, and visualize results
- `reporting/`: regenerate paper-ready CSV files and figures

Run every command from the repository root with the project virtual
environment:

```bash
.venv/bin/python scripts/<group>/<script>.py
```
