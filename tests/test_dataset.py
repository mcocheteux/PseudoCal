"""Tests for the PseudoPillars dataset helpers (camera pre-processing, collation,
deterministic decalibration). All offline — no KITTI data on disk required."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("unical")

from unical.utils.transform import Transform  # noqa: E402

from pseudocal.data.dataset import (  # noqa: E402
    PseudoBatch,
    PseudoKittiDataset,
    prepare_camera_input,
)

# ----------------------------------------------------------------------------
# prepare_camera_input
# ----------------------------------------------------------------------------


def test_prepare_camera_input_shapes_and_intrinsics_scaling() -> None:
    h0, w0 = 64, 128
    img = np.zeros((h0, w0, 3), dtype=np.uint8)
    img[:, w0 // 2 :] = 255  # a strong vertical edge so the Canny mask is non-trivial
    K = np.array([[500.0, 0.0, w0 / 2], [0.0, 500.0, h0 / 2], [0.0, 0.0, 1.0]], np.float32)

    width, height = 96, 48
    image, edge_mask, K_scaled = prepare_camera_input(img, K, width, height)

    assert image.shape == (3, height, width)
    assert image.dtype == torch.float32
    assert edge_mask.shape == (1, height, width)
    assert edge_mask.dtype == torch.bool

    # Intrinsics scale with the resize ratios (fx, cx by width; fy, cy by height).
    np.testing.assert_allclose(K_scaled[0, 0], 500.0 * width / w0, rtol=1e-5)
    np.testing.assert_allclose(K_scaled[1, 1], 500.0 * height / h0, rtol=1e-5)
    np.testing.assert_allclose(K_scaled[0, 2], (w0 / 2) * width / w0, rtol=1e-5)
    np.testing.assert_allclose(K_scaled[1, 2], (h0 / 2) * height / h0, rtol=1e-5)

    # The caller's intrinsics must not be mutated in place.
    assert K[0, 0] == 500.0


# ----------------------------------------------------------------------------
# PseudoKittiDataset.collate
# ----------------------------------------------------------------------------


def _sample(n: int, *, live: bool = True, p: int = 0) -> dict:
    """A single sample dict shaped like ``PseudoKittiDataset.__getitem__``."""
    s: dict = {
        "K": torch.eye(3),
        "trans": torch.randn(3),
        "rot_mat": torch.eye(3),
        "pcl": torch.rand(n, 4),
        "metadata": {"img_name": "0000000000"},
    }
    if live:
        s["image"] = torch.randn(3, 8, 8)
        s["edge_mask"] = torch.ones(1, 8, 8, dtype=torch.bool)
    else:
        s["pseudo_pcl"] = torch.rand(p, 3)
    return s


def test_collate_live_path_pads_and_stacks() -> None:
    batch = PseudoKittiDataset.collate([_sample(5), _sample(8)])

    assert isinstance(batch, PseudoBatch)
    assert batch.pcl.shape == (2, 8, 4)  # padded to the longest scan
    assert batch.K.shape == (2, 3, 3)
    assert batch.target_reg[0].shape == (2, 3)
    assert batch.target_reg[1].shape == (2, 3, 3)
    assert batch.image.shape == (2, 3, 8, 8)
    assert batch.edge_mask.shape == (2, 1, 8, 8)
    assert batch.edge_mask.dtype == torch.bool
    assert batch.pseudo_pcl is None
    assert len(batch.metadata) == 2


def test_collate_cached_path_uses_pseudo_pcl() -> None:
    batch = PseudoKittiDataset.collate([_sample(5, live=False, p=10), _sample(8, live=False, p=7)])

    assert batch.image is None
    assert batch.edge_mask is None
    assert batch.pseudo_pcl.shape == (2, 10, 3)  # padded to the longest cached cloud
    assert batch.pcl.shape == (2, 8, 4)


# ----------------------------------------------------------------------------
# Deterministic decalibration sampling
# ----------------------------------------------------------------------------


def _bare_dataset(decalibrator: object, *, deterministic: bool = True) -> PseudoKittiDataset:
    """Build a dataset without touching disk (skips ``__init__``/``_parse``)."""
    ds = object.__new__(PseudoKittiDataset)
    ds.decalibrator = decalibrator
    ds.deterministic = deterministic
    ds.seed = 0
    return ds


class _GeneratorDecalibrator:
    """Decalibrator that honours a ``generator`` kwarg (the preferred path)."""

    def __call__(self, generator: torch.Generator | None = None) -> Transform:
        t = torch.rand(3, generator=generator).numpy().astype(np.float32)
        return Transform.from_rotation_translation(np.eye(3, dtype=np.float32), t)


class _GlobalRngDecalibrator:
    """Decalibrator that rejects ``generator`` — forces the global-seed fallback."""

    def __call__(self) -> Transform:
        t = torch.rand(3).numpy().astype(np.float32)
        return Transform.from_rotation_translation(np.eye(3, dtype=np.float32), t)


@pytest.mark.parametrize("decalibrator", [_GeneratorDecalibrator(), _GlobalRngDecalibrator()])
def test_sample_decalib_is_deterministic_per_index(decalibrator: object) -> None:
    ds = _bare_dataset(decalibrator)
    # Same index → identical decalibration on repeated calls (both code paths).
    a = ds._sample_decalib(7)
    b = ds._sample_decalib(7)
    np.testing.assert_allclose(a.translation, b.translation)
    # Different indices → (almost surely) different decalibrations.
    c = ds._sample_decalib(8)
    assert not np.allclose(a.translation, c.translation)


def test_sample_decalib_random_when_not_deterministic() -> None:
    ds = _bare_dataset(_GlobalRngDecalibrator(), deterministic=False)
    # Without determinism the same index should vary across draws.
    draws = {tuple(ds._sample_decalib(0).translation.round(4)) for _ in range(5)}
    assert len(draws) > 1
