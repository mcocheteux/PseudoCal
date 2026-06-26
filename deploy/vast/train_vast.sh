#!/usr/bin/env bash
#
# Launch PseudoCal training for one stage on a (GPU) vast.ai instance.
#
#   bash deploy/vast/train_vast.sh <stage> [extra Hydra overrides...]
#
# where <stage> is one of: pseudopillars | pseudopillars_scale | pseudopillars_clsreg | unical_m | unical_s
#
# Examples:
#   # Standard training
#   bash deploy/vast/train_vast.sh pseudopillars
#
#   # Research variants
#   bash deploy/vast/train_vast.sh pseudopillars_scale  # depth-scale alignment
#   bash deploy/vast/train_vast.sh pseudopillars_clsreg  # yaw head
#
#   # Larger batch size (more VRAM = faster)
#   bash deploy/vast/train_vast.sh pseudopillars_scale data.batch_size=16
#
#   # Smoke test (fast, no W&B)
#   bash deploy/vast/train_vast.sh pseudopillars experiment=debug depth=dummy trainer.fast_dev_run=true logger=csv
#
# W&B tracking is forced on (charts + best/last checkpoint artifacts). Set
# WANDB_API_KEY (or run `wandb login`) first, or append `logger=csv` to log locally.
set -euo pipefail

STAGE="${1:-pseudopillars}"
shift || true

CODE_DIR="${CODE_DIR:-/workspace/pseudocal}"
DATA_DIR="${DATA_DIR:-/workspace/kitti_raw}"
export PATH="$HOME/.local/bin:$PATH"

# Each stage pairs with its own data config (different decalibration ranges).
case "$STAGE" in
  pseudopillars)        DATA="kitti_pillars" ;;
  pseudopillars_scale)  DATA="kitti_pillars" ;;
  pseudopillars_clsreg) DATA="kitti_pillars" ;;
  unical_m)             DATA="kitti_m" ;;
  unical_s)             DATA="kitti_s" ;;
  *) echo "Unknown stage '$STAGE' (expected pseudopillars|pseudopillars_scale|pseudopillars_clsreg|unical_m|unical_s)"; exit 1 ;;
esac

cd "$CODE_DIR"

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "[train] WARNING: WANDB_API_KEY is not set — W&B will not be able to sync."
  echo "[train]          export WANDB_API_KEY=... (or run 'wandb login') first,"
  echo "[train]          or pass 'logger=csv' to log locally only."
fi

# accelerator=auto picks CUDA on a GPU box; bf16-mixed is a safe speedup on Ampere+.
# devices=1: the custom CalibMetrics accumulator (flip_rate, tail percentiles, ...)
# isn't a torchmetrics Metric, so sync_dist=True under multi-GPU DDP would average
# each rank's per-shard metric instead of computing a true global one. Pass
# trainer.devices=N yourself only if you've verified eval correctness under DDP.
exec uv run python train.py \
  model="$STAGE" \
  data="$DATA" \
  data_dir="$DATA_DIR" \
  trainer.accelerator=auto \
  trainer.devices=1 \
  trainer.precision=bf16-mixed \
  data.num_workers=8 \
  logger=wandb \
  "$@"
