#!/bin/bash
# KLD v2: rerun the sensitivity sweep with Hungarian-matched (permutation-
# invariant) KLD — the per-index v1 map was dominated by DETR query
# permutation churn, not real damage. Archives the v1 report/results, then
# rebuilds Q06 from the corrected map.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
LOG=artifacts/experiments/fp8-pipeline.log
DATASET=/home/lair/datum/rfdetr_seg/test
SENS=artifacts/experiments/sensitivity/front-fp8-kld.json
IMG=$(ls "$DATASET"/*.jpg | head -1)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

while pgrep -f "run_kld_pipeline.sh" > /dev/null; do sleep 30; done

log "=== KLD v2: archiving per-index v1 artifacts ==="
[ -f "$SENS" ] && cp "$SENS" "${SENS%.json}-perindex-v1.json"
[ -f results/coco-evaluation/Q06-front.json ] && \
  cp results/coco-evaluation/Q06-front.json \
     results/coco-evaluation/Q06-front-perindex-v1.json

log "=== KLD v2: Hungarian-matched sensitivity sweep ==="
$PY scripts/lightweighting/kld_sensitivity.py \
  --camera front --backbone-groups 12 --top-fraction 0.25 >> "$LOG" 2>&1 \
  || { log "KLD v2 sweep FAILED — aborting"; exit 1; }

id=Q06
dir="artifacts/experiments/$id/front"
log "=== $id v2: quantize ==="
$PY scripts/lightweighting/quantize_modelopt.py \
  --experiment "$id" --camera front --mode fp8-mixed \
  --sensitivity-report "$SENS" \
  --calib-count 128 --device cuda --force >> "$LOG" 2>&1 \
  || { log "$id v2 quantize FAILED"; exit 1; }
log "=== $id v2: engine build ==="
$PY scripts/lightweighting/build_tensorrt_fp16.py \
  --camera front --onnx "$dir/model.onnx" \
  --output-dir "artifacts/experiments/$id" --output-name model \
  --force >> "$LOG" 2>&1 \
  || { log "$id v2 engine build FAILED"; exit 1; }
log "=== $id v2: COCO evaluation ==="
$PY scripts/evaluation/evaluate_coco_tensorrt.py \
  --experiment "$id" --camera front --engine "$dir/model.engine" \
  --dataset-dir "$DATASET" --no-update-csv >> "$LOG" 2>&1 \
  || { log "$id v2 evaluation FAILED"; exit 1; }
log "=== $id v2: latency benchmark ==="
$PY scripts/evaluation/benchmark_baseline.py \
  --camera front --image "$IMG" --backend fp16 \
  --engine "$dir/model.engine" --warmup 20 --runs 100 >> "$LOG" 2>&1 \
  || log "$id v2 benchmark FAILED (continuing)"
log "KLD v2 pipeline finished"
