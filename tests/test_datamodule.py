"""Tests for the PseudoPillars DataModule loader wiring (offline — no KITTI data)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("unical")

from pseudocal.data.datamodule import PseudoKittiDataModule  # noqa: E402
from pseudocal.data.dataset import PseudoBatch, PseudoKittiDataset  # noqa: E402


def _sample(n: int) -> dict:
    return {
        "K": torch.eye(3),
        "trans": torch.randn(3),
        "rot_mat": torch.eye(3),
        "pcl": torch.rand(n, 4),
        "image": torch.randn(3, 8, 8),
        "edge_mask": torch.ones(1, 8, 8, dtype=torch.bool),
        "metadata": {"img_name": "0000000000"},
    }


def test_loader_batches_into_pseudobatch() -> None:
    dm = PseudoKittiDataModule(
        data_dir="/tmp/kitti",
        splits={"train": [], "val": [], "test": []},
        decalibrator=object(),  # unused by _loader; setup() is not called here
        batch_size=2,
        num_workers=0,
    )
    # An in-memory dataset of variable-length scans (no disk access).
    dataset = [_sample(5), _sample(7), _sample(3), _sample(6)]

    loader = dm._loader(dataset, shuffle=False)
    batch = next(iter(loader))

    assert isinstance(batch, PseudoBatch)
    assert batch.pcl.shape[0] == 2  # batch_size
    assert batch.pcl.shape[2] == 4
    assert batch.image.shape == (2, 3, 8, 8)
    assert loader.collate_fn is PseudoKittiDataset.collate
