"""Tests for the PointPillars BEV encoder."""

from __future__ import annotations

import torch

from pseudocal.pillars import PillarFeatureNet, PillarGrid


def _grid() -> PillarGrid:
    return PillarGrid(x_min=-4.0, x_max=4.0, z_min=0.0, z_max=8.0, pillar_size=0.5)


def test_grid_dimensions() -> None:
    g = _grid()
    assert g.width == 16  # (4 - -4) / 0.5
    assert g.height == 16  # (8 - 0) / 0.5


def test_pillar_output_shape() -> None:
    g = _grid()
    net = PillarFeatureNet(g, out_channels=24).eval()
    # B=2 clouds of 100 points each, all valid, inside the grid.
    pts = torch.rand(2, 100, 3)
    pts[..., 0] = pts[..., 0] * 8 - 4  # x ∈ [-4, 4)
    pts[..., 2] = pts[..., 2] * 8  # z ∈ [0, 8)
    cloud = torch.cat([pts, torch.ones(2, 100, 1)], dim=-1)

    img = net(cloud)
    assert img.shape == (2, 24, g.height, g.width)
    assert torch.isfinite(img).all()


def test_pillar_drops_out_of_range_and_padding() -> None:
    g = _grid()
    net = PillarFeatureNet(g, out_channels=8).eval()

    # One in-range point; one far out of range; one zero-padding row.
    cloud = torch.tensor(
        [
            [
                [0.0, 0.0, 4.0, 1.0],  # in range (centre-ish)
                [999.0, 0.0, 999.0, 1.0],  # out of range
                [0.0, 0.0, 0.0, 0.0],  # padding (valid flag = 0)
            ]
        ]
    )
    img = net(cloud)
    # Exactly one occupied pillar (a column with any non-zero activation).
    occupied = (img.abs().sum(dim=1) > 0).sum().item()
    assert occupied == 1


def test_pillar_deterministic() -> None:
    g = _grid()
    net = PillarFeatureNet(g, out_channels=8).eval()
    cloud = torch.cat([torch.rand(1, 50, 3) * 4, torch.ones(1, 50, 1)], dim=-1)
    with torch.no_grad():
        a = net(cloud)
        b = net(cloud)
    torch.testing.assert_close(a, b)
