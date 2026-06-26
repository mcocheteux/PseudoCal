#!/usr/bin/env bash
#
# Local training script for PseudoCal.
#
#   bash deploy/local/train_local.sh <stage> [extra Hydra overrides...]
#
# where <stage> is one of: pseudopillars | pseudopillars_scale | pseudopillars_clsreg | unical_m | unical_s
#
# Examples:
#   # Standard PseudoPillars (CSV logging)
#   bash deploy/local/train_local.sh pseudopillars
#
#   # Research: LiDAR-anchored depth-scale alignment
#   bash deploy/local/train_local.sh pseudopillars_scale
#
#   # Research: classify-then-regress yaw head
#   bash deploy/local/train_local.sh pseudopillars_clsreg
#
#   # With W&B tracking
#   bash deploy/local/train_local.sh pseudopillars_scale logger=wandb
#
#   # With overrides
#   bash deploy/local/train_local.sh pseudopillars_scale trainer.max_epochs=100
#
#   # Smoke test (offline, fast)
#   bash deploy/local/train_local.sh pseudopillars experiment=debug depth=dummy trainer.fast_dev_run=true
#
# W&B tracking is disabled by default (uses CSV logger).
# To enable W&B, set WANDB_API_KEY (or run `wandb login`) and pass `logger=wandb`.
#
# Data directory can be set via DATA_DIR environment variable (default: ./data/kitti_raw)
# or passed as an override: data_dir=/my/path
#
set -euo pipefail

STAGE="${1:-pseudopillars}"
shift || true

# Default data directory (relative to repo root or absolute)
DATA_DIR="${DATA_DIR:-./data/kitti_raw}"

# Resolve CODE_DIR to the directory containing this script's grandparent (repo root)
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Each stage pairs with its own data config (different decalibration ranges).
case "$STAGE" in
  pseudopillars)
    DATA="kitti_pillars"
    ;;
  pseudopillars_scale)
    DATA="kitti_pillars"
    ;;
  pseudopillars_clsreg)
    DATA="kitti_pillars"
    ;;
  unical_m)
    DATA="kitti_m"
    ;;
  unical_s)
    DATA="kitti_s"
    ;;
  *)
    echo "Unknown stage '$STAGE'"
    echo "Valid stages: pseudopillars, pseudopillars_scale, pseudopillars_clsreg, unical_m, unical_s"
    exit 1
    ;;
esac

cd "$CODE_DIR"

# Check if user wants W&B but isn't authenticated
if [[ "$*" == *"logger=wandb"* ]] || [[ "$*" == *"logger wandb"* ]]; then
  if [ -z "${WANDB_API_KEY:-}" ] && ! command -v wandb &> /dev/null; then
    echo "[train] WARNING: W&B requested but WANDB_API_KEY is not set."
    echo "[train]          Run 'wandb login' or set WANDB_API_KEY."
  fi
fi

# Run training with sensible defaults for local execution
# Uses `uv run` to ensure the project's virtual environment is used.
# W&B is disabled by default (logger=csv). Pass logger=wandb to enable.
exec uv run python train.py \
  model="$STAGE" \
  data="$DATA" \
  data_dir="$DATA_DIR" \
  trainer.accelerator=auto \
  trainer.devices=1 \
  trainer.precision=bf16-mixed \
  data.num_workers=4 \
  logger=csv \
  "$@"
