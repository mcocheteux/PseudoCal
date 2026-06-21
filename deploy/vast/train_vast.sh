#!/usr/bin/env bash
#
# Launch PseudoCal training for one stage on a (GPU) vast.ai instance.
#
#   bash deploy/vast/train_vast.sh <stage> [extra Hydra overrides...]
#
# where <stage> is one of: pseudopillars | unical_m | unical_s
#
# Examples:
#   bash deploy/vast/train_vast.sh pseudopillars
#   bash deploy/vast/train_vast.sh unical_m trainer.max_epochs=200 data.batch_size=16
#   bash deploy/vast/train_vast.sh pseudopillars experiment=debug depth=dummy  # smoke
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
  pseudopillars) DATA="kitti_pillars" ;;
  unical_m)      DATA="kitti_m" ;;
  unical_s)      DATA="kitti_s" ;;
  *) echo "Unknown stage '$STAGE' (expected pseudopillars|unical_m|unical_s)"; exit 1 ;;
esac

cd "$CODE_DIR"

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "[train] WARNING: WANDB_API_KEY is not set — W&B will not be able to sync."
  echo "[train]          export WANDB_API_KEY=... (or run 'wandb login') first,"
  echo "[train]          or pass 'logger=csv' to log locally only."
fi

# accelerator=auto picks CUDA on a GPU box; bf16-mixed is a safe speedup on Ampere+.
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
