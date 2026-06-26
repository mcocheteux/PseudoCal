# Training PseudoCal on vast.ai

Helper scripts to provision a GPU instance, fetch KITTI raw, and train the PseudoCal stages
with Weights & Biases tracking. Supports both the main branch and **research variants**
(`pseudopillars_scale`, `pseudopillars_clsreg`).

## 1. Rent an instance

Pick a CUDA image (e.g. `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`) with a
single GPU (≥ 16 GB) and a disk large enough for KITTI (~50 GB for the default
split). Clone this repo into `/workspace/pseudocal` (override with `CODE_DIR`).

## 2. Provision

```bash
bash deploy/vast/provision.sh
```

Installs `uv`, syncs dependencies (including the `unical-plus` git dependency and
the `logger` extra), and pre-downloads the Depth-Anything-V2 metric weights.

## 3. Get the data

```bash
bash deploy/vast/download_kitti.sh /workspace/kitti_raw
```

## 4. Enable W&B and train each stage

```bash
export WANDB_API_KEY=...            # or run: wandb login

bash deploy/vast/train_vast.sh pseudopillars   # stage 1 (coarse, init-free)
bash deploy/vast/train_vast.sh unical_m        # stage 2 (medium refinement)
bash deploy/vast/train_vast.sh unical_s        # stage 3 (fine refinement)
```

Each run logs `train/loss`, `val/rot/global/MAE`, `val/trans/global/MAE` (and the
per-axis breakdowns) to W&B and uploads the best + last checkpoints as artifacts.
Extra Hydra overrides are forwarded, e.g.:

```bash
bash deploy/vast/train_vast.sh pseudopillars trainer.max_epochs=200 data.batch_size=16
```

## 5. End-to-end cascade evaluation

```bash
uv run python cascade.py data_dir=/workspace/kitti_raw \
  +ckpt.pseudopillars=logs/checkpoints/pseudopillars/last.ckpt \
  +ckpt.unical_m=logs/checkpoints/unical_m/last.ckpt \
  +ckpt.unical_s=logs/checkpoints/unical_s/last.ckpt
```

## Environment variables

| Var | Default | Meaning |
|-----|---------|---------|
| `CODE_DIR` | `/workspace/pseudocal` | Repo location on the instance |
| `DATA_DIR` | `/workspace/kitti_raw` | KITTI raw root |
| `WANDB_API_KEY` | — | W&B auth (required for `logger=wandb`) |
| `PARALLEL` | `6` | Number of parallel downloads for KITTI |

---

## Research Variants

The `research/scale-yaw` branch adds two Stage 1 variants:

| Variant | Config | Focus |
|---------|--------|-------|
| `pseudopillars_scale` | LiDAR-anchored depth-scale alignment | Robust to monocular depth-scale drift |
| `pseudopillars_clsreg` | Classify-then-regress yaw head | Solves ±180° BEV ambiguity |

Train them the same way:

```bash
bash deploy/vast/train_vast.sh pseudopillars_scale
bash deploy/vast/train_vast.sh pseudopillars_clsreg
```

---

## Performance Tips

### Faster Data Downloads
On instances with good bandwidth to EU (Frankfurt), increase parallel downloads:

```bash
PARALLEL=12 bash deploy/vast/download_kitti.sh /workspace/kitti_raw
```

### Multi-GPU Training
Use all GPUs on your instance for faster training:

```bash
bash deploy/vast/train_vast.sh pseudopillars_scale \
  trainer.devices=auto \
  data.batch_size=32  # scale batch size with GPU count
```

### Larger Batch Sizes
Increase batch size to saturate GPU (more VRAM = faster):

| GPU | Recommended batch_size |
|-----|----------------------|
| RTX 4090 (24GB) | 16-32 |
| A100 (40GB) | 24-48 |
| H100 (80GB) | 32-64 |

Example:
```bash
bash deploy/vast/train_vast.sh pseudopillars_scale data.batch_size=24
```

### Custom Run Names for Comparison
When comparing variants on W&B, use custom run names:

```bash
bash deploy/vast/train_vast.sh pseudopillars +logger.name=baseline
bash deploy/vast/train_vast.sh pseudopillars_scale +logger.name=scale-variant
bash deploy/vast/train_vast.sh pseudopillars_clsreg +logger.name=yaw-variant
```

### Smoke Tests (Fast Validation)
Quick offline test without downloads:

```bash
bash deploy/vast/train_vast.sh pseudopillars \
  experiment=debug \
  depth=dummy \
  trainer.fast_dev_run=true \
  logger=csv
```
