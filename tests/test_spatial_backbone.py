"""Tests for SpatialMobileViTBackbone (spatial-preserving pooling for BEV registration).

Offline: the backbone is built with ``pretrained=None`` so no weights are downloaded."""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

pytest.importorskip("unical")

from pseudocal.models.spatial_backbone import SpatialMobileViTBackbone  # noqa: E402


def _backbone(spatial_pool: int, ch: int = 8, image_size: int = 64) -> SpatialMobileViTBackbone:
    return SpatialMobileViTBackbone(
        img_channels=ch,
        lidar_channels=ch,
        image_size=image_size,
        pretrained=None,
        spatial_head=True,
        spatial_pool=spatial_pool,
    )


def test_spatial_pool_scales_flattened_width() -> None:
    ch, size, b = 8, 64, 2
    batch = types.SimpleNamespace(
        img=torch.randn(b, ch, size, size),
        lidar_map=torch.randn(b, ch, size, size),
    )
    bb = _backbone(spatial_pool=1).eval()

    with torch.no_grad():
        out1 = bb(batch)  # 1x1 global pool → (B, D)
        # Swap to a 4x4 grid on the same instance: width must grow exactly 16x.
        bb.spatial_pool = 4
        bb.pool = nn.AdaptiveAvgPool2d((4, 4))
        out4 = bb(batch)  # (B, D*4*4)

    assert out1.ndim == 2 and out4.ndim == 2
    assert out1.shape[0] == b and out4.shape[0] == b
    assert out4.shape[1] == out1.shape[1] * 16
    assert torch.isfinite(out4).all()
