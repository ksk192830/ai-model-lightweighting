#!/bin/bash
# KLD-based mixed-precision pipeline (Q06), chained after run_fp8_pipeline.sh:
# waits for the FP8 pipeline to finish, runs the KL-divergence sensitivity
# sweep (fine backbone granularity), then quantizes/builds/evaluates Q06.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
LOG=artifacts/experiments/fp8-pipeline.log
DATASET=/home/lair/datum/rfdetr_seg/test
SENS=artifacts/experiments/sensitivity/front-fp8-kld.json
IMG=$(ls "$DATASET"/*.jpg | head -1)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

while pgrep -f "run_fp8_pipeline.sh" > /dev/null || \
      pgrep -f "quantize_modelopt.py" > /dev/null || \
      pgrep -f "evaluate_coco_tensorrt.py" > /dev/null; do
  sleep 30
done
log "=== KLD pipeline: sensitivity sweep (per-layer, 48 images) ==="
$PY scripts/lightweighting/kld_sensitivity.py \
  --camera front --backbone-groups 12 --top-fraction 0.25 >> "$LOG" 2>&1 \
  || { log "KLD sweep FAILED — aborting"; exit 1; }

id=Q06
dir="artifacts/experiments/$id/front"
log "=== $id (fp8-mixed, KLD-selected): quantize ==="
$PY scripts/lightweighting/quantize_modelopt.py \
  --experiment "$id" --camera front --mode fp8-mixed \
  --sensitivity-report "$SENS" \
  --calib-count 128 --device cuda --force >> "$LOG" 2>&1 \
  || { log "$id quantize FAILED"; exit 1; }
log "=== $id: engine build ==="
$PY scripts/lightweighting/build_tensorrt_fp16.py \
  --camera front --onnx "$dir/model.onnx" \
  --output-dir "artifacts/experiments/$id" --output-name model \
  --force >> "$LOG" 2>&1 \
  || { log "$id engine build FAILED"; exit 1; }
log "=== $id: COCO evaluation ==="
$PY scripts/evaluation/evaluate_coco_tensorrt.py \
  --experiment "$id" --camera front --engine "$dir/model.engine" \
  --dataset-dir "$DATASET" --no-update-csv >> "$LOG" 2>&1 \
  || { log "$id evaluation FAILED"; exit 1; }
log "=== $id: latency benchmark ==="
$PY scripts/evaluation/benchmark_baseline.py \
  --camera front --image "$IMG" --backend fp16 \
  --engine "$dir/model.engine" --warmup 20 --runs 100 >> "$LOG" 2>&1 \
  || log "$id benchmark FAILED (continuing)"
log "KLD pipeline finished"
