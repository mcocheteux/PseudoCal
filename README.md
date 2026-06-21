<div align="center">

# PseudoCal

### Initialisation-Free Camera–LiDAR Self-Calibration via Pseudo-LiDAR Pillars

[![CI](https://github.com/mcocheteux/pseudocal/actions/workflows/ci.yml/badge.svg)](https://github.com/mcocheteux/pseudocal/actions/workflows/ci.yml)
[![Paper](https://img.shields.io/badge/BMVC%202023-paper-blue.svg)](https://papers.bmvc2023.org/0829.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2309.09855-b31b1b.svg)](https://arxiv.org/abs/2309.09855)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.2+-792ee5.svg)](https://lightning.ai/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)

*A reference implementation by one of the paper's authors — see [Citation](#citation).*

</div>

---

Most learned camera–LiDAR calibrators only correct *small* misalignments and need a good
initial estimate. **PseudoCal** recovers the extrinsic **one-shot, from scratch**, even under
extreme decalibration (up to **±180° yaw**). It lifts the camera image into the **3-D space the
LiDAR lives in** — a *pseudo-LiDAR* point cloud — and matches the two clouds in bird's-eye view
rather than within the camera field of view, sidestepping the initialisation problem that limits
projection-based methods.

Built on top of [**UniCal++**](https://github.com/mcocheteux/unical-plus) and following its
project style.

> **About this repository.** A from-scratch implementation of the PseudoCal method by one of the
> paper's authors, built as a well-tested, reproducible engineering reference rather than a verbatim
> reproduction of the paper's published benchmark numbers. It demonstrates the complete pipeline
> (pseudo-LiDAR generation, BEV pillar matching, and the coarse-to-fine cascade) with modern tooling
> (Hydra, Lightning, W&B, CI). For the original method's reported results, refer to the
> [paper](https://papers.bmvc2023.org/0829.pdf).

---

## Highlights

| | |
|---|---|
| 🎯 **Initialisation-free** | Recovers the extrinsic one-shot from a single image + scan, with no prior estimate — even under ±180° yaw |
| 🧊 **Pseudo-LiDAR in BEV** | Monocular metric depth is back-projected to a 3-D cloud and matched against the real LiDAR in bird's-eye view, not the image plane |
| 🪜 **3-stage cascade** | A coarse init-free stage (PseudoPillars) hands off to two UniCal refiners, each trained on the residual the stage before it leaves |
| 🧭 **Euler-angle loss through 180°** | Matrix-MSE rotation loss has a vanishing gradient near 180°; a wrap-around Euler loss keeps the coarse stage able to escape front/back yaw flips |
| 🧱 **Pure-PyTorch PointPillars** | BEV encoder with no `spconv` / custom CUDA ops — trivial to install on rented GPUs |
| 🔌 **Pluggable depth** | Depth-Anything-V2 by default, the paper's GLPN selectable, a synthetic estimator for offline tests |
| 🧩 **Hydra · Lightning · W&B** | Every hyperparameter is a CLI override; training, tracking and Vast.ai deployment scripts included |

---

## How it works — a 3-stage cascade

```
                    ┌──────────────────────── stage 1: PseudoPillars (coarse, init-free) ────────────────────────┐
  camera image ──►  monocular metric depth ──► Canny filter ──► back-project ──► pseudo-LiDAR ─┐
                                                                                               ├─► PointPillars ─► MobileViT ─► T̂_decal
  real LiDAR  ──────────────── transform by current extrinsic estimate ─────────────────────► ┘     (BEV)         fusion
                    └─────────────────────────────────────────────────────────────────────────────────────────┘
                                                   │  apply  T̂⁻¹
                                                   ▼
            stage 2: UniCal-M  (medium ±6°/±60 cm refinement)     ─── reused from unical-plus
                                                   │
                                                   ▼
            stage 3: UniCal-S  (fine ±1°/±10 cm refinement)       ─── reused from unical-plus
```

| Stage | Role | Decalibration range | Source |
|-------|------|---------------------|--------|
| **PseudoPillars** | coarse, initialisation-free | ±15°/±15°/±180°, ±100 cm | this repo (`pseudocal`) |
| **UniCal-M** | medium residual refinement | ±6°, ±60 cm | `unical.models.module.UniCal` |
| **UniCal-S** | fine residual refinement | ±1°, ±10 cm | `unical.models.module.UniCal` |

Each refiner's training range is matched to the actual residual distribution left by the stage
before it, rather than an arbitrary fixed window — a stage can only correct what it was trained to
see.

The two refinement stages **are** the UniCal++ model trained on tighter ranges, so they come
directly from the `unical-plus` dependency. Only the PseudoPillars stage and the cascade
orchestration are new here. PseudoCal also reuses UniCal's MobileViT backbone, split regression
head, regression + spatial losses, calibration metrics, `Transform` geometry, and KITTI parsing.

---

## Quick Start

### Install

```bash
# with uv (recommended) — resolves the unical-plus git dependency automatically
uv sync --extra dev --extra logger

# or with pip
pip install -e ".[dev,logger]"
```

Requires Python ≥ 3.11. The default depth estimator downloads Depth-Anything-V2 weights on first
use; tests and smoke runs use a synthetic estimator (`depth=dummy`) and stay offline.

### Prepare KITTI

KITTI raw, in the standard layout (same split as RegNet/LCCNet):

```
kitti_raw/
  2011_09_26/
    calib_cam_to_cam.txt
    calib_velo_to_cam.txt
    2011_09_26_drive_0001_sync/
      image_02/data/0000000000.png ...
      velodyne_points/data/0000000000.bin ...
  2011_09_30/ ...
```

`deploy/vast/download_kitti.sh` fetches the drives used by the default split.

### Train

Each stage is trained independently (pair each `model` with its `data` config):

```bash
# Stage 1 — PseudoPillars (coarse, initialisation-free)
python train.py data_dir=/path/to/kitti_raw model=pseudopillars data=kitti_pillars

# Stage 2 — UniCal-M (medium refinement)
python train.py data_dir=/path/to/kitti_raw model=unical_m data=kitti_m

# Stage 3 — UniCal-S (fine refinement)
python train.py data_dir=/path/to/kitti_raw model=unical_s data=kitti_s
```

Useful overrides:

```bash
logger=wandb                     # track on Weights & Biases (charts + checkpoint artifacts)
trainer.precision=bf16-mixed     # ~40% faster on Ampere+ GPUs
depth=glpn                       # paper-faithful depth estimator
experiment=debug depth=dummy trainer.fast_dev_run=true   # offline smoke test
```

### Evaluate

Per-stage test metrics:

```bash
python evaluate.py data_dir=/path/to/kitti_raw model=pseudopillars data=kitti_pillars \
  +ckpt=logs/checkpoints/pseudopillars/last.ckpt
```

End-to-end cascade evaluation:

```bash
python cascade.py data_dir=/path/to/kitti_raw \
  +ckpt.pseudopillars=logs/checkpoints/pseudopillars/last.ckpt \
  +ckpt.unical_m=logs/checkpoints/unical_m/last.ckpt \
  +ckpt.unical_s=logs/checkpoints/unical_s/last.ckpt
```

### Tests

```bash
uv run ruff check .
uv run pytest -v          # offline: pillars / pseudo-LiDAR / depth / cascade math / configs
```

Tests that require the `unical-plus` dependency `importorskip` cleanly if it is not installed;
none of them download model weights.

### Demo notebook

[`notebooks/demo.ipynb`](notebooks/demo.ipynb) is an offline, end-to-end walkthrough of the pipeline
(camera frame → monocular depth → pseudo-LiDAR → BEV pillars → cascade) using the synthetic depth
estimator, so it runs without KITTI data or trained checkpoints. Needs `jupyter` and `matplotlib`.

---

## Weights & Biases

`logger=wandb` logs `train/loss`, `val/{rot,trans}/global/MAE` (and the per-axis STD/MAE) plus the
resolved hyperparameters, and uploads the best + last checkpoints as a versioned artifact. Set
`WANDB_API_KEY` (or run `wandb login`) first. See `configs/logger/wandb.yaml`.

## Vast.ai

End-to-end GPU recipe (provision → data → train → evaluate) in
[`deploy/vast/README.md`](deploy/vast/README.md):

```bash
bash deploy/vast/provision.sh
bash deploy/vast/download_kitti.sh
export WANDB_API_KEY=...
bash deploy/vast/train_vast.sh pseudopillars   # then unical_m, then unical_s
```

---

## Project structure

```
src/pseudocal/
├── depth/         DepthEstimator interface + Depth-Anything-V2 / GLPN / Dummy backends
├── pseudolidar.py Canny edge mask + depth back-projection to camera-frame points
├── pillars.py     PillarGrid + PillarFeatureNet (PointPillars BEV encoder)
├── losses/        Euler-angle regression loss for the coarse stage
├── models/        PseudoPillars LightningModule + spatial backbone
├── data/          PseudoKittiDataset / DataModule (raw image + raw LiDAR)
└── cascade/       CascadeCalibrator + per-stage wrappers
configs/           Hydra configs (model, data, depth, experiment, logger, cascade)
deploy/vast/       provision / download / train scripts for Vast.ai
notebooks/         offline pipeline walkthrough (demo.ipynb)
tests/             pytest suite
train.py  evaluate.py  cascade.py
```

---

## Citation

If you use this code or the PseudoCal method, please cite:

```bibtex
@inproceedings{Cocheteux_2023_BMVC,
  author    = {Mathieu Cocheteux and Julien Moreau and Franck Davoine},
  title     = {PseudoCal: Towards Initialisation-Free Deep Learning-Based Camera-LiDAR Self-Calibration},
  booktitle = {34th British Machine Vision Conference 2023, {BMVC} 2023, Aberdeen, UK, November 20-24, 2023},
  publisher = {{BMVA}},
  year      = {2023},
  url       = {https://papers.bmvc2023.org/0829.pdf}
}
```

---

## Acknowledgements & license

Builds on [UniCal++](https://github.com/mcocheteux/unical-plus). Depth backbone:
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2). BEV encoder inspired by
[PointPillars](https://arxiv.org/abs/1812.05784).

Released under **CC BY-NC 4.0** (research / non-commercial use). See [LICENSE](LICENSE).
</content>
