#!/bin/bash
# FP8 pipeline: measured sensitivity sweep -> Q04 (uniform FP8) ->
# Q05 (measured mixed FP8/FP16) -> engines -> COCO eval -> latency benchmark.
# INT8 is intentionally absent (user decision 2026-07-06: INT8 PTQ is not
# efficient on this architecture; see 0705.md section 16).
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
LOG=artifacts/experiments/fp8-pipeline.log
DATASET=/home/lair/datum/rfdetr_seg/test
SENS=artifacts/experiments/sensitivity/front-fp8-sensitivity.json
IMG=$(ls "$DATASET"/*.jpg | head -1)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== sensitivity sweep (FP8, leave-one-in) ==="
$PY scripts/lightweighting/sensitivity_sweep.py \
  --camera front --quant-config FP8_DEFAULT_CFG \
  --calib-count 64 --eval-count 150 >> "$LOG" 2>&1 \
  || { log "sensitivity sweep FAILED — aborting"; exit 1; }

run_experiment() {
  local id="$1" mode="$2"; shift 2
  local dir="artifacts/experiments/$id/front"
  log "=== $id ($mode): quantize ==="
  $PY scripts/lightweighting/quantize_modelopt.py \
    --experiment "$id" --camera front --mode "$mode" \
    --calib-count 128 --device cuda --force "$@" >> "$LOG" 2>&1 \
    || { log "$id quantize FAILED"; return 1; }
  log "=== $id: engine build ==="
  $PY scripts/lightweighting/build_tensorrt_fp16.py \
    --camera front --onnx "$dir/model.onnx" \
    --output-dir "artifacts/experiments/$id" --output-name model \
    --force >> "$LOG" 2>&1 \
    || { log "$id engine build FAILED"; return 1; }
  log "=== $id: COCO evaluation ==="
  $PY scripts/evaluation/evaluate_coco_tensorrt.py \
    --experiment "$id" --camera front --engine "$dir/model.engine" \
    --dataset-dir "$DATASET" --no-update-csv >> "$LOG" 2>&1 \
    || { log "$id evaluation FAILED"; return 1; }
  log "=== $id: latency benchmark ==="
  $PY scripts/evaluation/benchmark_baseline.py \
    --camera front --image "$IMG" --backend fp16 \
    --engine "$dir/model.engine" --warmup 20 --runs 100 >> "$LOG" 2>&1 \
    || log "$id benchmark FAILED (continuing)"
  log "$id done"
}

run_experiment Q04 fp8
run_experiment Q05 fp8-mixed --sensitivity-report "$SENS"

log "FP8 pipeline finished"
