#!/bin/bash
# Q-series quantization experiment queue (Q01 SmoothQuant / Q02 mixed / Q03 INT4-FFN).
#
# Waits until no recovery training (train_candidate.py) has been running for
# five consecutive minutes (S01 recovery-extended, then the queued M03 run),
# then executes quantize -> engine build -> COCO evaluation for each
# registered Q experiment, plus a local B01 FP16 reference engine as the
# comparison anchor. Every step logs to artifacts/experiments/qseries-queue.log
# and failures do not stop later steps.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
LOG=artifacts/experiments/qseries-queue.log
DATASET=/home/lair/datum/rfdetr_seg/test

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "queue registered; waiting for recovery trainings to finish"
idle=0
while [ "$idle" -lt 5 ]; do
  if pgrep -f "train_candidate.py" > /dev/null; then
    idle=0
  else
    idle=$((idle + 1))
  fi
  sleep 60
done
log "GPU free; starting Q-series"

run_experiment() {
  local id="$1" mode="$2"
  local dir="artifacts/experiments/$id/front"
  log "=== $id ($mode): quantize ==="
  $PY scripts/lightweighting/quantize_modelopt.py \
    --experiment "$id" --camera front --mode "$mode" \
    --calib-count 128 --device cuda >> "$LOG" 2>&1 \
    || { log "$id quantize FAILED"; return 1; }
  log "=== $id: engine build ==="
  $PY scripts/lightweighting/build_tensorrt_fp16.py \
    --camera front --onnx "$dir/model.onnx" \
    --output-dir "artifacts/experiments/$id" --output-name model \
    --int8-qdq --force >> "$LOG" 2>&1 \
    || { log "$id engine build FAILED"; return 1; }
  log "=== $id: COCO evaluation ==="
  $PY scripts/evaluation/evaluate_coco_tensorrt.py \
    --experiment "$id" --camera front --engine "$dir/model.engine" \
    --dataset-dir "$DATASET" --no-update-csv >> "$LOG" 2>&1 \
    || { log "$id evaluation FAILED"; return 1; }
  log "$id done"
}

log "=== B01 FP16 local reference (comparison anchor) ==="
$PY scripts/lightweighting/build_tensorrt_fp16.py \
  --camera front --onnx shared-models/B01-front-baseline.onnx \
  --output-dir artifacts/experiments/B01REF --output-name model \
  --force >> "$LOG" 2>&1 \
  && $PY scripts/evaluation/evaluate_coco_tensorrt.py \
    --experiment B01REF --camera front \
    --engine artifacts/experiments/B01REF/front/model.engine \
    --dataset-dir "$DATASET" --no-update-csv >> "$LOG" 2>&1 \
  || log "B01 reference FAILED (continuing)"

run_experiment Q01 smoothquant
run_experiment Q02 mixed
run_experiment Q03 int4-ffn

log "Q-series queue finished"
