"""
Lightning DataModule for the PseudoPillars stage.

Mirrors :class:`unical.data.datamodule.KittiDataModule` but builds
:class:`~pseudocal.data.dataset.PseudoKittiDataset` instances (raw image + raw
LiDAR) and collates into :class:`~pseudocal.data.dataset.PseudoBatch`.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as L
from torch.utils.data import DataLoader

from pseudocal.data.dataset import PseudoKittiDataset


class PseudoKittiDataModule(L.LightningDataModule):
    """
    Manages train / val / test splits of the KITTI raw dataset for PseudoPillars.

    Args:
        data_dir:     Path to KITTI raw root.
        splits:       Dict with keys "train"/"val"/"test", each a
                      List[Tuple[date_str, List[drive_id]]].
        decalibrator: Configured decalibration generator instance.
        width/height: Working image (and depth-map) resolution.
        edge_low/high/dilate: Canny parameters forwarded to the dataset.
        batch_size:   Samples per GPU.
        num_workers:  DataLoader workers.
        pin_memory:   Pin memory for GPU transfers.
    """

    def __init__(
        self,
        data_dir: str,
        splits: dict[str, Any],
        decalibrator: Any,
        width: int = 512,
        height: int = 512,
        edge_low: int = 50,
        edge_high: int = 150,
        edge_dilate: int = 2,
        pseudolidar_cache_dir: str | None = None,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["decalibrator", "splits"])
        self._splits = splits
        self._decalibrator = decalibrator

    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        def _make(key: str) -> PseudoKittiDataset:
            # val/test use deterministic (seeded-per-index) decalibrations so their
            # metrics are stable and reproducible across epochs and runs.
            return PseudoKittiDataset(
                data_dir=self.hparams.data_dir,
                split=self._splits[key],
                decalibrator=self._decalibrator,
                width=self.hparams.width,
                height=self.hparams.height,
                edge_low=self.hparams.edge_low,
                edge_high=self.hparams.edge_high,
                edge_dilate=self.hparams.edge_dilate,
                pseudolidar_cache_dir=self.hparams.pseudolidar_cache_dir,
                deterministic=key != "train",
            )

        self.train_ds = _make("train")
        self.val_ds = _make("val")
        self.test_ds = _make("test")

    def _loader(self, ds: PseudoKittiDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            collate_fn=PseudoKittiDataset.collate,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)
