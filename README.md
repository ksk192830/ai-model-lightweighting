# KIPS AI Model Lightweighting

Code and experiments for the KIPS paper project on AI model lightweighting.

## Project Links

- Notion project: https://app.notion.com/p/389d4a78ceed80d28d95c37d83422a60
- [FP16 lightweighting notes](docs/fp16.md)
- [INT8 quantization notes](docs/int8.md)

## Setup

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Verify the RF-DETR runtime and baseline checkpoints:

```bash
.venv/bin/python scripts/lightweighting/check_environment.py
```

Install `rclone` on macOS:

```bash
brew install rclone
```

Authenticate with a Google account that has viewer access to the curated
data folder:

```bash
rclone config
```

Create a remote named `kips-drive`, choose `Google Drive`, select the read-only
(`drive.readonly`) scope, and complete the Google login in the browser. Do not
commit or share the generated rclone configuration because it contains an OAuth
token.

## Download datasets

The image datasets are stored in Google Drive and are not committed to Git.
The Drive folders remain restricted. Each authorized user must be explicitly
added as a viewer and authenticate their own Google account through `rclone`.

Download both the INT8 calibration data and labeled test data:

```bash
.venv/bin/python scripts/data_preparation/download_data.py
```

The configured Drive root contains `calibration/` and `labeled_test/`; they are
downloaded to the same paths under local `data/`. After a successful download,
each subset receives a `.download_complete` marker and is skipped on later runs.

Download only one subset:

```bash
.venv/bin/python scripts/data_preparation/download_data.py --subset calibration
.venv/bin/python scripts/data_preparation/download_data.py --subset labeled-test
```

Download again even when a subset is already marked complete:

```bash
.venv/bin/python scripts/data_preparation/download_data.py --subset calibration --force
```

Preview the download paths without downloading:

```bash
.venv/bin/python scripts/data_preparation/download_data.py --dry-run
```

Dataset and baseline model mappings are defined in
[`configs/dataset.yaml`](configs/dataset.yaml) and
[`configs/baseline.yaml`](configs/baseline.yaml).

## Inference visualization

Compare TensorRT FP32, FP16, and INT8 predictions on one downloaded sample:

```bash
.venv/bin/python scripts/evaluation/infer_baseline.py \
  --backend all \
  --camera front \
  --image data/labeled_test/front/images/image000002.png \
  --device cuda
```

Run sequential inference on a directory and update the live window once per
second:

```bash
.venv/bin/python scripts/evaluation/visualize_inference_stream.py \
  --camera front \
  --image-dir data/labeled_test/front/images \
  --backend all \
  --interval 1 \
  --live
```

Omit `--live` to create a 1 FPS MP4 under `results/inference_stream/`. Use
`--max-images` to limit the number of frames. Single-image prediction JSON and
annotated segmentation images are written under `results/inference/`.

## Optional dataset regeneration

The curated Drive data is ready to use, so the following utilities are not
required for normal lightweighting work. Use them only to reproduce how the
data was selected.

Create deterministic front/rear calibration and test lists:

```bash
python3 scripts/data_preparation/create_calibration_splits.py
```

Calibration uses `numeric_id % 5 == 0`; test uses
`numeric_id % 5 == 2`. The script verifies that the lists do not overlap.
Generated lists and metadata are written under `splits/`. Images are not copied;
each text file contains paths to the original files under `data/`.

After extracting complete Roboflow COCO Segmentation exports, create labeled
test subsets:

```bash
python3 scripts/data_preparation/extract_coco_subset.py \
  --camera front \
  --source data/roboflow/front

python3 scripts/data_preparation/extract_coco_subset.py \
  --camera rear \
  --source data/roboflow/rear
```

Filtered images and COCO annotations are written under
`data/labeled_test/<camera>/`. Images with no objects are preserved as negative
test samples.

## Inference benchmark

Benchmark the TensorRT FP32 reference with warmup and repeated end-to-end
inference:

```bash
.venv/bin/python scripts/evaluation/benchmark_baseline.py \
  --backend fp32 \
  --camera front \
  --image data/labeled_test/front/images/image000002.png \
  --device cuda \
  --warmup 10 \
  --runs 100
```

Benchmark the generated TensorRT FP16 or INT8 engine with the same measurement
logic by changing `--backend`:

```bash
.venv/bin/python scripts/evaluation/benchmark_baseline.py \
  --backend int8 \
  --camera front \
  --image data/labeled_test/front/images/image000002.png \
  --device cuda \
  --warmup 10 \
  --runs 100
```

The default engine is selected from `--backend fp32|fp16|int8`. Use `--engine`
to benchmark another engine.

The measured interval includes preprocessing, model execution, and
postprocessing, but excludes image loading and model loading. Results are
appended to `results/benchmarks/summary.csv`, with raw timings saved in a
separate JSON file. Final paper measurements must be collected on the same
NVIDIA GPU with the same image, threshold, warmup, and run count for every
TensorRT precision.

## TensorRT FP16

Install the ONNX export dependencies:

```bash
python3 -m pip install -r requirements-export.txt
```

Export the front and rear checkpoints to ONNX:

```bash
.venv/bin/python scripts/lightweighting/export_onnx.py --camera front
.venv/bin/python scripts/lightweighting/export_onnx.py --camera rear
```

On the target NVIDIA machine with TensorRT installed, build FP16 engines:

```bash
.venv/bin/python scripts/lightweighting/build_tensorrt_fp16.py --camera front
.venv/bin/python scripts/lightweighting/build_tensorrt_fp16.py --camera rear
```

TensorRT engines are hardware and TensorRT-version dependent. Build and
benchmark them on the deployment GPU. ONNX models and engines are written under
`artifacts/` and are not committed to Git.

Run the complete front/rear ONNX + TensorRT FP16 pipeline with one command:

```bash
.venv/bin/python scripts/lightweighting/run_lightweighting.py
```

Run selected cameras or methods:

```bash
.venv/bin/python scripts/lightweighting/run_lightweighting.py \
  --camera front \
  --methods tensorrt-fp16
```

Preview every command without running it:

```bash
.venv/bin/python scripts/lightweighting/run_lightweighting.py --dry-run
```

The pipeline automatically creates an ONNX dependency before TensorRT and
writes an execution report under `results/lightweighting/`. INT8 is registered in the same entry point; pruning can be added as the next
independent method.

## Repository Layout

```text
.
├── artifacts/      # Generated lightweight models (not committed)
├── configs/        # Dataset and baseline model configuration
├── data/           # Downloaded datasets (not committed)
│   ├── calibration/
│   └── labeled_test/
├── models/         # Baseline model checkpoints
├── notebooks/      # Exploration and experiment notebooks
├── scripts/        # Reproducible experiment scripts
├── src/            # Reusable implementation code
└── results/        # Metrics, tables, and figures
```

## Notes

- `models/parking_front.pth` uses front-camera data.
- `models/parking_rear.pth` uses rear-camera data.
- Keep raw datasets and large generated outputs out of Git.
- Commit scripts, configuration, small result summaries, and figure-generation code.


## TensorRT INT8

Prepare the selected calibration images under `data/calibration/front/` and
`data/calibration/rear/`. On an NVIDIA machine with CUDA-enabled PyTorch and
the TensorRT Python bindings, build calibrated INT8 engines with:

```bash
.venv/bin/python scripts/lightweighting/build_tensorrt_int8.py --camera front
.venv/bin/python scripts/lightweighting/build_tensorrt_int8.py --camera rear
```

Run every currently supported export for both camera models with one command:

```bash
.venv/bin/python scripts/lightweighting/run_lightweighting.py \
  --camera all \
  --methods onnx tensorrt-fp16 tensorrt-int8
```

Use `--dry-run` on a non-NVIDIA machine to verify paths and commands. INT8
engines, calibration caches, and reproducibility metadata are written under
`artifacts/tensorrt/<camera>/`.
