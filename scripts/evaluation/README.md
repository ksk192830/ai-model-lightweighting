# Lightweighting result evaluation

This directory contains tools for inspecting and timing original and
lightweighted models.

- `infer_baseline.py`: compare annotated TensorRT FP32, FP16, and INT8 predictions
- `benchmark.py`: measure speed, latency, memory, and bbox accuracy
- `evaluate.py`: evaluate bbox accuracy with class-name alignment
- `benchmark_baseline.py`: benchmark PyTorch checkpoints or TensorRT engines
- `visualize_inference_stream.py`: compare sequential predictions live or as MP4
- `evaluate_coco_tensorrt.py`: add COCO mask AP and mask mIoU to an existing CSV
- `evaluate_coco_ultralytics.py`: evaluate Ultralytics segmentation masks
- `evaluate_coco_rfdetr_pth.py`: evaluate an RF-DETR `.pth` checkpoint (bbox +
  mask AP + mask mIoU) on any COCO split, before any TensorRT conversion exists

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

Add only the missing segmentation metrics to the matching row in
`results/paper_metrics.csv`. This runs mask inference and official COCO `segm`
evaluation, but does not repeat bbox evaluation or the latency benchmark:

```bash
.venv/bin/python scripts/evaluation/evaluate_coco_tensorrt.py \
  --experiment S01 --camera front
```

The script adds these columns when needed: `Mask AP`, `Mask AP50`, `Mask AP75`,
and `Mask mIoU`. Mask AP uses the low `--threshold 0.001` predictions required
for the complete COCO precision-recall curve. Mask mIoU uses predictions at
`--miou-threshold 0.25`, merges instance masks by category, and averages the
dataset-wide pixel IoU of the evaluated categories. A detailed audit JSON is
also written to `results/coco-evaluation/<experiment>-<camera>.json`.

To fill all existing front-model rows:

```bash
for experiment in B01 B02 B03 C01 M01 M02 S01 R01; do
  .venv/bin/python scripts/evaluation/evaluate_coco_tensorrt.py \
    --experiment "$experiment" --camera front
done
```

Use `--no-update-csv` for a dry run that leaves the CSV unchanged, or `--csv`
to select a different existing results file.

Evaluate a PyTorch checkpoint (for example a new baseline that has no engine
yet) on a labeled COCO split; the audit JSON goes to `results/coco-evaluation/`:

```bash
.venv/bin/python scripts/evaluation/evaluate_coco_rfdetr_pth.py \
  --checkpoint "models/general_mission(with_crosswalk).pth" \
  --dataset-dir /home/lair/datum/rfdetr_seg_general_mission/test \
  --name general_mission_with_crosswalk-test
```

Show the selected eight front engines in a live 2×4 comparison:

```bash
.venv/bin/python scripts/evaluation/visualize_inference_stream.py \
  --camera front \
  --image-dir data/labeled_test/front/images \
  --backend final8 --interval 0.1 --live
```
