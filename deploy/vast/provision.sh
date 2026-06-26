#!/usr/bin/env bash
#
# Provision a fresh (GPU) vast.ai instance for PseudoCal training.
#
#   bash deploy/vast/provision.sh
#
# Installs uv, syncs the project (with the dev + logger extras, which pulls in the
# unical-plus git dependency), and warms the default monocular-depth weights so the
# first training step does not stall on a multi-GB download.
set -euo pipefail

CODE_DIR="${CODE_DIR:-/workspace/pseudocal}"
export PATH="$HOME/.local/bin:$PATH"

echo "[provision] installing system packages …"
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -y
  # opencv (cv2) needs libGL / glib at runtime; git for the unical-plus dependency.
  apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 ca-certificates
fi

echo "[provision] installing uv …"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

cd "$CODE_DIR"

echo "[provision] syncing dependencies (this resolves the unical-plus git dep) …"
uv sync --extra dev --extra logger

echo "[provision] warming Depth-Anything-V2 metric weights …"
uv run python - <<'PY'
from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation
mid = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
DepthAnythingForDepthEstimation.from_pretrained(mid)
AutoImageProcessor.from_pretrained(mid)
print("[provision] depth weights cached.")
PY

echo "[provision] done. Next:"
echo "  bash deploy/vast/download_kitti.sh              # fetch KITTI raw (~50GB)"
echo "  PARALLEL=12 bash deploy/vast/download_kitti.sh # faster on good connections"
echo "  export WANDB_API_KEY=...                      # enable W&B tracking"
echo "  bash deploy/vast/train_vast.sh pseudopillars  # train stage 1"
