"""Tests for the classify-then-regress yaw head, its loss, and the model variant.

Offline / deterministic; the model is built with ``pretrained=None`` (no downloads)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

pytest.importorskip("unical")

from unical.models.backbone import MobileViTBackbone  # noqa: E402
from unical.utils.transform import (  # noqa: E402
    Transform,
    euler_to_rotation_matrix,
    rotation_6d_to_matrix,
)

from pseudocal.data.dataset import PseudoBatch  # noqa: E402
from pseudocal.depth import DummyDepthEstimator  # noqa: E402
from pseudocal.losses.euler import matrix_to_euler_xyz  # noqa: E402
from pseudocal.losses.yaw_cls import YawClsRegLoss  # noqa: E402
from pseudocal.models.pseudopillars_clsreg import PseudoPillarsClsReg  # noqa: E402
from pseudocal.models.yaw_head import (  # noqa: E402
    ClassifyRegressYawHead,
    YawHeadOut,
    yaw_bin_centers,
)
from pseudocal.pillars import PillarGrid  # noqa: E402


def _head(n_bins: int = 12, in_features: int = 16) -> ClassifyRegressYawHead:
    return ClassifyRegressYawHead(
        in_features=in_features,
        common_hidden=[],
        trans_hidden=[8],
        rot_hidden=[8],
        n_yaw_bins=n_bins,
    )


# ----------------------------------------------------------------------------
# Geometry: compose round-trips
# ----------------------------------------------------------------------------


def test_compose_roundtrips_euler() -> None:
    head = _head()
    b = 4
    roll = torch.empty(b).uniform_(-0.2, 0.2)
    pitch = torch.empty(b).uniform_(-0.2, 0.2)
    yaw = torch.empty(b).uniform_(-3.0, 3.0)
    out = YawHeadOut(
        trans=torch.zeros(b, 3),
        roll_pitch=torch.stack([roll, pitch], dim=-1),
        yaw_logits=torch.zeros(b, head.n_yaw_bins),
        yaw_residual=torch.zeros(b, head.n_yaw_bins),
        bin_centers=head.bin_centers,
        half_bin=head.half_bin,
    )
    _, rot6d = head._compose(out, yaw)
    euler = matrix_to_euler_xyz(rotation_6d_to_matrix(rot6d))
    np.testing.assert_allclose(euler[:, 0].numpy(), roll.numpy(), atol=1e-4)
    np.testing.assert_allclose(euler[:, 1].numpy(), pitch.numpy(), atol=1e-4)
    np.testing.assert_allclose(euler[:, 2].numpy(), yaw.numpy(), atol=1e-4)


def test_forward_emits_standard_contract() -> None:
    head = _head().eval()
    with torch.no_grad():
        trans, rot6d = head(torch.randn(3, 16))
    assert trans.shape == (3, 3)
    assert rot6d.shape == (3, 6)
    assert torch.isfinite(rot6d).all()


# ----------------------------------------------------------------------------
# Loss: bin assignment + the gradient-through-π property
# ----------------------------------------------------------------------------


def _target_batch(yaw_vals: torch.Tensor) -> PseudoBatch:
    b = yaw_vals.shape[0]
    euler = torch.zeros(b, 3)
    euler[:, 2] = yaw_vals
    R = euler_to_rotation_matrix(euler, convention="XYZ")
    return PseudoBatch(
        pcl=torch.zeros(b, 1, 4),
        K=torch.eye(3).expand(b, 3, 3),
        target_reg=(torch.zeros(b, 3), R),
        metadata=[{} for _ in range(b)],
    )


def test_loss_runs_and_targets_correct_bin() -> None:
    n_bins = 12
    head = _head(n_bins)
    loss_fn = YawClsRegLoss()
    # A yaw of +175° should fall in the last bin; -175° in the first.
    batch = _target_batch(torch.tensor([math.radians(175.0), math.radians(-175.0)]))
    out = head.train_outputs(torch.randn(2, 16))
    losses = loss_fn(out, batch)
    assert torch.isfinite(losses["loss"])
    assert {"loss/yaw_cls", "loss/yaw_res", "loss/reg_rollpitch", "loss/reg_trans"} <= losses.keys()

    # Re-derive the target bins the loss uses and check the extremes map as expected.
    bw = 2 * math.pi / n_bins
    yaw_t = batch.target_reg[1]
    yaw_ang = matrix_to_euler_xyz(yaw_t)[:, 2]
    target_bin = torch.floor((yaw_ang + math.pi) / bw).long().clamp(0, n_bins - 1)
    assert target_bin[0].item() == n_bins - 1
    assert target_bin[1].item() == 0


def test_classification_gradient_survives_180_flip() -> None:
    # Cross-entropy keeps a non-vanishing gradient when the prediction is the flipped bin …
    n_bins = 12
    logits = torch.zeros(1, n_bins, requires_grad=True)
    target_bin = torch.tensor([n_bins // 2])  # opposite side from bin 0
    ce = torch.nn.functional.cross_entropy(logits, target_bin)
    ce.backward()
    assert logits.grad.abs().sum() > 1e-3

    # … whereas Frobenius matrix-MSE between R(π) and R(0) has a ~zero gradient at the flip
    # (∝ sin θ → 0), which is exactly why a plain rotation regressor gets stuck there.
    theta = torch.tensor(math.pi, requires_grad=True)
    euler = torch.stack([torch.zeros(()), torch.zeros(()), theta]).unsqueeze(0)
    R = euler_to_rotation_matrix(euler, convention="XYZ")
    R0 = torch.eye(3).unsqueeze(0)
    ((R - R0) ** 2).sum().backward()
    assert theta.grad.abs().item() < 1e-5


# ----------------------------------------------------------------------------
# Model variant: a training step learns through the classification path
# ----------------------------------------------------------------------------


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


def test_clsreg_model_training_step_backprops() -> None:
    grid = PillarGrid(x_min=-8.0, x_max=8.0, z_min=0.0, z_max=16.0, pillar_size=0.25)
    backbone = MobileViTBackbone(
        img_channels=8, lidar_channels=8, image_size=64, pretrained=None, spatial_head=True
    )
    head = ClassifyRegressYawHead(
        in_features=640, common_hidden=[], trans_hidden=[64], rot_hidden=[64], n_yaw_bins=8
    )
    model = PseudoPillarsClsReg(
        depth=DummyDepthEstimator(),
        backbone=backbone,
        head=head,
        loss=YawClsRegLoss(),
        grid=grid,
        pillar_channels=8,
        backproject_stride=4,
        max_pseudo_points=2000,
    ).train()

    loss = model.training_step(_make_batch(), 0)
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.head.parameters() if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in grads)


def test_bin_centers_tile_the_circle() -> None:
    c = yaw_bin_centers(12)
    assert c.shape == (12,)
    assert torch.isclose(c[0], torch.tensor(-math.pi + math.pi / 12))
    assert (c.diff() > 0).all()
