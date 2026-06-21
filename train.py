"""
Training entry point — trains a single PseudoCal stage.

The stage is selected by the ``model`` (and matching ``data``) config group:

    # PseudoPillars (coarse, initialisation-free)
    python train.py data_dir=/path/to/kitti_raw model=pseudopillars data=kitti_pillars

    # UniCal-M / UniCal-S refiners (reuse the unical model + image/lidar-map data)
    python train.py data_dir=/path/to/kitti_raw model=unical_m data=kitti
    python train.py data_dir=/path/to/kitti_raw model=unical_s data=kitti

    # Track on Weights & Biases / quick smoke test
    python train.py data_dir=/path/to/kitti_raw logger=wandb
    python train.py data_dir=/path/to/kitti_raw experiment=debug trainer.fast_dev_run=true depth=dummy
"""

from __future__ import annotations

import os

import hydra
import pytorch_lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger

# Speeds up float32 matmuls on Ampere+ GPUs (TF32) with negligible accuracy impact.
torch.set_float32_matmul_precision("high")


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    L.seed_everything(cfg.seed, workers=True)

    # ── DataModule ────────────────────────────────────────────────────
    datamodule = instantiate(cfg.data)

    # ── Model (Hydra instantiates depth / backbone / head / loss recursively) ─
    model = instantiate(cfg.model)

    # ── Logger ────────────────────────────────────────────────────────
    if cfg.get("logger"):
        logger = instantiate(cfg.logger)
    else:
        logger = CSVLogger(save_dir=cfg.log_dir, name="pseudocal")

    # ── Callbacks ─────────────────────────────────────────────────────
    callbacks: list[L.Callback] = [
        ModelCheckpoint(
            dirpath=os.path.join(cfg.log_dir, "checkpoints", cfg.get("stage", "stage")),
            filename="pseudocal-{epoch:03d}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,  # keep the best checkpoint …
            save_last=True,  # … and the most recent one
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = L.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        callbacks=callbacks,
        logger=logger,
        default_root_dir=cfg.log_dir,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
