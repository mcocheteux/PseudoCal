"""Tests for LiDAR-anchored depth-scale alignment (#2). Offline / deterministic."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("unical")

from unical.losses.combined import CombinedLoss  # noqa: E402
from unical.losses.regression import RegressionLoss  # noqa: E402
from unical.losses.spatial import SpatialLoss  # noqa: E402
from unical.models.backbone import MobileViTBackbone  # noqa: E402
from unical.models.head import SplitRegressionHead  # noqa: E402
from unical.utils.transform import Transform, euler_to_rotation_matrix  # noqa: E402

from pseudocal.data.dataset import PseudoBatch  # noqa: E402
from pseudocal.depth import DummyDepthEstimator  # noqa: E402
from pseudocal.models.pseudopillars import PseudoPillars  # noqa: E402
from pseudocal.pillars import PillarGrid  # noqa: E402
from pseudocal.scale import estimate_range_scale, point_range_quantiles  # noqa: E402


def _cloud(b: int = 2, n: int = 500, pad: int = 50) -> torch.Tensor:
    """A forward point cloud (valid=1) with zero-padding rows (valid=0)."""
    xyz = torch.rand(b, n, 3)
    xyz[..., 2] = xyz[..., 2] * 40 + 3  # z forward in [3, 43]
    xyz[..., 0] = (xyz[..., 0] - 0.5) * 30
    xyz[..., 1] = (xyz[..., 1] - 0.5) * 4
    cloud = torch.cat([xyz, torch.ones(b, n, 1)], dim=-1)
    padding = torch.zeros(b, pad, 4)
    return torch.cat([cloud, padding], dim=1)


def test_range_scale_is_rotation_invariant() -> None:
    cloud = _cloud()
    euler = torch.tensor([[0.1, -0.2, 2.7]])  # includes a near-180° yaw
    R = euler_to_rotation_matrix(euler, convention="XYZ")[0]
    rotated = cloud.clone()
    rotated[..., :3] = cloud[..., :3] @ R.T  # rotation about the origin preserves ranges

    s = estimate_range_scale(rotated, cloud)
    np.testing.assert_allclose(s.numpy(), 1.0, atol=1e-3)


def test_range_scale_recovers_known_factor() -> None:
    real = _cloud()
    for k in (0.5, 1.5, 2.0):
        pseudo = real.clone()
        pseudo[..., :3] = pseudo[..., :3] * k  # pseudo-LiDAR scaled by k
        # The correction multiplies the pseudo cloud, so it should be ~1/k.
        s = estimate_range_scale(pseudo, real)
        np.testing.assert_allclose(s.numpy(), 1.0 / k, rtol=0.05)


def test_quantiles_ignore_padding_and_empty_is_unit() -> None:
    cloud = _cloud(b=1, n=100, pad=900)  # mostly padding
    q = point_range_quantiles(cloud, (0.5,))
    assert q.shape == (1, 1)
    assert torch.isfinite(q).all()

    empty = torch.zeros(1, 10, 4)
    s = estimate_range_scale(empty, empty)
    np.testing.assert_allclose(s.numpy(), 1.0)


def _build_model(scale_align: str, scale_bias: float = 1.0) -> PseudoPillars:
    grid = PillarGrid(x_min=-8.0, x_max=8.0, z_min=0.0, z_max=16.0, pillar_size=0.25)
    backbone = MobileViTBackbone(
        img_channels=8, lidar_channels=8, image_size=64, pretrained=None, spatial_head=True
    )
    head = SplitRegressionHead(
        in_features=640, common_hidden=[], trans_hidden=[64], rot_hidden=[64]
    )
    return PseudoPillars(
        depth=DummyDepthEstimator(),
        backbone=backbone,
        head=head,
        loss=CombinedLoss(RegressionLoss(), SpatialLoss()),
        grid=grid,
        pillar_channels=8,
        backproject_stride=4,
        max_pseudo_points=2000,
        scale_align=scale_align,
        scale_bias=scale_bias,
    )


def _make_batch(b: int = 2, h: int = 64, w: int = 64, n: int = 400) -> PseudoBatch:
    image = torch.randn(b, 3, h, w)
    edge_mask = torch.ones(b, 1, h, w, dtype=torch.bool)
    pcl = torch.rand(b, n, 4)
    pcl[..., :3] *= 10.0
    pcl[..., 3] = 1.0
    K = torch.tensor([[200.0, 0, w / 2], [0, 200.0, h / 2], [0, 0, 1]]).expand(b, 3, 3).clone()
    trans = torch.randn(b, 3) * 0.05
    rot = torch.eye(3).expand(b, 3, 3).clone()
    meta = []
    for i in range(b):
        T_gt = Transform.from_rotation_translation(
            np.eye(3, dtype=np.float32), np.array([0.0, 0.0, 0.05], np.float32)
        )
        T_decal = Transform.from_rotation_translation(rot[i].numpy(), trans[i].numpy())
        meta.append({"T_gt": T_gt, "T_init": T_decal @ T_gt, "T_decal": T_decal, "K": K[i].numpy()})
    return PseudoBatch(
        image=image, edge_mask=edge_mask, pcl=pcl, K=K, target_reg=(trans, rot), metadata=meta
    )


def test_scale_align_forward_runs_and_default_is_noop() -> None:
    batch = _make_batch()
    out_none = _build_model("none").eval()(batch)
    out_align = _build_model("range").eval()(batch)
    assert out_none[0].shape == (2, 3) and out_align[0].shape == (2, 3)
    assert torch.isfinite(out_align[1]).all()


def test_residual_scale_head_receives_gradient() -> None:
    model = _build_model("range+residual", scale_bias=0.8).train()
    assert model.scale_residual is not None
    loss = model.training_step(_make_batch(), 0)
    loss.backward()
    grads = [p.grad for p in model.scale_residual.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_invalid_scale_align_rejected() -> None:
    with pytest.raises(ValueError):
        _build_model("bogus")
