"""Tests for pseudo-LiDAR generation (edge masking + back-projection)."""

from __future__ import annotations

import numpy as np
import torch

from pseudocal.pseudolidar import backproject, canny_edge_mask
from tests.helpers import make_intrinsics


def test_canny_edge_mask_shape_and_dtype() -> None:
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    img[:, 30:] = 255  # a strong vertical edge down the middle
    keep = canny_edge_mask(img, dilate=1)

    assert keep.shape == (40, 60)
    assert keep.dtype == bool
    # The edge column is rejected (some pixels masked out).
    assert keep.sum() < keep.size


def test_backproject_recovers_known_geometry() -> None:
    # A single pixel at the principal point with depth z must map to (0, 0, z).
    H = W = 64
    K = make_intrinsics(W, H, f=500.0).unsqueeze(0)
    depth = torch.full((1, 1, H, W), 10.0)

    cloud = backproject(
        depth, K, keep_mask=None, stride=1, min_depth=1.0, max_depth=80.0, max_points=H * W
    )
    valid = cloud[0, :, 3] > 0
    xyz = cloud[0, valid, :3]

    # Principal-point pixel → x = y = 0.
    cu, cv = int(K[0, 0, 2]), int(K[0, 1, 2])
    centre_lin = cv * W + cu
    # Find the centre point among kept points (stride=1 keeps row-major order).
    np.testing.assert_allclose(xyz[centre_lin].numpy(), [0.0, 0.0, 10.0], atol=1e-4)
    # Every point keeps the constant depth.
    np.testing.assert_allclose(xyz[:, 2].numpy(), 10.0, atol=1e-4)


def test_backproject_depth_filtering_and_padding() -> None:
    H = W = 32
    K = make_intrinsics(W, H).unsqueeze(0)
    depth = torch.full((1, 1, H, W), 200.0)  # beyond max_depth → all filtered

    cloud = backproject(depth, K, stride=1, min_depth=1.0, max_depth=80.0, max_points=128)
    assert cloud.shape == (1, 128, 4)
    assert (cloud[..., 3] == 0).all()  # nothing valid
    assert torch.count_nonzero(cloud) == 0  # padding rows are exact zeros


def test_backproject_edge_mask_drops_points() -> None:
    H = W = 16
    K = make_intrinsics(W, H).unsqueeze(0)
    depth = torch.full((1, 1, H, W), 10.0)
    keep = torch.zeros(1, 1, H, W, dtype=torch.bool)
    keep[:, :, :8, :8] = True  # keep only a quadrant

    cloud = backproject(
        depth, K, keep_mask=keep, stride=1, min_depth=1.0, max_depth=80.0, max_points=H * W
    )
    assert int((cloud[..., 3] > 0).sum()) == 8 * 8
