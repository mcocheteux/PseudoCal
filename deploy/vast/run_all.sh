#!/usr/bin/env bash
#
# Unattended end-to-end research run for a 2-GPU vast.ai box.
#
# Provisions the instance, downloads KITTI raw, precomputes the (frozen) pseudo-LiDAR
# cache once, then trains the two research variants in parallel — one GPU each —
# streaming progress to stdout so it is readable over the HTTPS API via `vastai logs`:
#
#     GPU 0 : pseudopillars_scale   (LiDAR-anchored depth-scale alignment, #2)
#     GPU 1 : pseudopillars_clsreg  (classify-then-regress yaw head, #3)
#
# Both run devices=1 (the custom CalibMetrics accumulator isn't DDP-safe). The
# pseudo-LiDAR cache is shared: both variants inherit the base back-projection
# params, so one precompute pass feeds both and makes every training step cheap.
#
# Designed to be the instance onstart target. Idempotent: re-running (e.g. after a
# stop/start) skips already-finished provisioning / download / cache steps.
set -uo pipefail

CODE_DIR="${CODE_DIR:-/workspace/pseudocal}"
DATA_DIR="${DATA_DIR:-/workspace/kitti_raw}"
CACHE_DIR="${CACHE_DIR:-$DATA_DIR/pseudo_cache}"
RUN_DIR="${RUN_DIR:-/workspace/runs}"
LOG_DIR="${LOG_DIR:-/workspace/runlogs}"
PARALLEL="${PARALLEL:-16}"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$LOG_DIR" "$RUN_DIR"
cd "$CODE_DIR"

say() { echo "[run_all $(date -u +%H:%M:%S)] $*"; }

# ── 1. Provision (uv, deps incl. unical-plus, warm depth weights) ───────────────
if ! command -v uv >/dev/null 2>&1 || [ ! -d "$CODE_DIR/.venv" ]; then
  say "provisioning (uv sync + depth weights) …"
  bash deploy/vast/provision.sh 2>&1 | tee "$LOG_DIR/provision.log"
else
  say "provision: already done, skipping."
fi

# ── 2. KITTI raw (download_kitti uses unzip -n, so this is resumable/idempotent) ─
if [ ! -f "$DATA_DIR/.kitti_done" ]; then
  say "downloading KITTI raw (PARALLEL=$PARALLEL) …"
  PARALLEL="$PARALLEL" bash deploy/vast/download_kitti.sh "$DATA_DIR" 2>&1 | tee "$LOG_DIR/kitti.log"
  touch "$DATA_DIR/.kitti_done"
else
  say "KITTI: already downloaded, skipping."
fi

# ── 3. Precompute pseudo-LiDAR cache (one frozen-depth pass, shared by both) ─────
if [ ! -f "$CACHE_DIR/.cache_done" ]; then
  say "precomputing pseudo-LiDAR cache → $CACHE_DIR …"
  uv run python precompute_pseudolidar.py \
    data=kitti_pillars model=pseudopillars depth=depth_anything \
    data_dir="$DATA_DIR" data.pseudolidar_cache_dir="$CACHE_DIR" \
    2>&1 | tee "$LOG_DIR/precompute.log"
  touch "$CACHE_DIR/.cache_done"
else
  say "pseudo-LiDAR cache: already built, skipping."
fi

# ── 4. Launch both variants in parallel, one GPU each, from the cache ────────────
COMMON=(
  data=kitti_pillars
  data_dir="$DATA_DIR"
  data.pseudolidar_cache_dir="$CACHE_DIR"
  trainer.accelerator=gpu
  trainer.devices=1
  trainer.precision=bf16-mixed
  data.num_workers=8
  logger=csv
)

say "launching pseudopillars_scale on GPU 0 …"
CUDA_VISIBLE_DEVICES=0 uv run python train.py \
  model=pseudopillars_scale log_dir="$RUN_DIR/scale" "${COMMON[@]}" \
  > "$LOG_DIR/scale.log" 2>&1 &
PID_SCALE=$!

say "launching pseudopillars_clsreg on GPU 1 …"
CUDA_VISIBLE_DEVICES=1 uv run python train.py \
  model=pseudopillars_clsreg log_dir="$RUN_DIR/clsreg" "${COMMON[@]}" \
  > "$LOG_DIR/clsreg.log" 2>&1 &
PID_CLSREG=$!

say "scale PID=$PID_SCALE  clsreg PID=$PID_CLSREG"

# ── 5. Heartbeat: stream GPU + per-variant log tails to stdout (vastai logs) ─────
while kill -0 "$PID_SCALE" 2>/dev/null || kill -0 "$PID_CLSREG" 2>/dev/null; do
  say "================ heartbeat ================"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader 2>/dev/null || true
  for v in scale clsreg; do
    echo "---- $v : last 8 log lines ----"
    tail -n 8 "$LOG_DIR/$v.log" 2>/dev/null || true
    m=$(ls "$RUN_DIR/$v"/pseudocal/version_*/metrics.csv 2>/dev/null | head -1)
    if [ -n "$m" ]; then
      echo "---- $v : last 2 metrics rows ($m) ----"
      { head -1 "$m"; tail -n 2 "$m"; } 2>/dev/null || true
    fi
  done
  sleep 120
done

wait "$PID_SCALE"; SC=$?
wait "$PID_CLSREG"; CC=$?
say "DONE. scale exit=$SC  clsreg exit=$CC"
say "checkpoints: $RUN_DIR/{scale,clsreg}/checkpoints/  metrics: $RUN_DIR/{scale,clsreg}/pseudocal/version_*/metrics.csv"
