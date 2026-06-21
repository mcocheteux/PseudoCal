"""Integration test for the PseudoPillars module (Dummy depth — no downloads)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("unical")
pytest.importorskip("transformers")

from unical.losses.combined import CombinedLoss  # noqa: E402
from unical.losses.regression import RegressionLoss  # noqa: E402
from unical.losses.spatial import SpatialLoss  # noqa: E402
from unical.models.backbone import MobileViTBackbone  # noqa: E402
from unical.models.head import SplitRegressionHead  # noqa: E402
from unical.utils.transform import Transform  # noqa: E402

from pseudocal.data.dataset import PseudoBatch  # noqa: E402
from pseudocal.depth import DummyDepthEstimator  # noqa: E402
from pseudocal.models.pseudopillars import PseudoPillars  # noqa: E402
from pseudocal.pillars import PillarGrid  # noqa: E402


def _build_model(pillar_channels: int = 8) -> PseudoPillars:
    grid = PillarGrid(x_min=-8.0, x_max=8.0, z_min=0.0, z_max=16.0, pillar_size=0.25)  # 64×64
    backbone = MobileViTBackbone(
        img_channels=pillar_channels,
        lidar_channels=pillar_channels,
        image_size=64,
        pretrained=None,
        spatial_head=True,
    )
    head = SplitRegressionHead(
        in_features=640, common_hidden=[], trans_hidden=[64], rot_hidden=[64]
    )
    loss = CombinedLoss(RegressionLoss(), SpatialLoss())
    return PseudoPillars(
        depth=DummyDepthEstimator(),
        backbone=backbone,
        head=head,
        loss=loss,
        grid=grid,
        pillar_channels=pillar_channels,
        backproject_stride=4,
        max_pseudo_points=2000,
    )


def _make_batch(b: int = 2, h: int = 64, w: int = 64, n: int = 500) -> PseudoBatch:
    image = torch.randn(b, 3, h, w)
    edge_mask = torch.ones(b, 1, h, w, dtype=torch.bool)
    # Random LiDAR points a few metres in front of the sensor.
    pcl = torch.rand(b, n, 4)
    pcl[..., :3] = pcl[..., :3] * 10.0
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


def test_forward_output_shapes() -> None:
    model = _build_model().eval()
    batch = _make_batch()
    with torch.no_grad():
        trans, rot6d = model(batch)
    assert trans.shape == (2, 3)
    assert rot6d.shape == (2, 6)
    assert torch.isfinite(trans).all() and torch.isfinite(rot6d).all()


def test_training_step_backprops() -> None:
    model = _build_model().train()
    batch = _make_batch()
    loss = model.training_step(batch, 0)
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    # At least some trainable parameter received a gradient …
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert any(g.abs().sum() > 0 for g in grads)
    # … and the frozen depth estimator received none.
    assert all(p.grad is None for p in model.depth.parameters())


def _make_cached_batch(b: int = 2, n: int = 500, p: int = 400) -> PseudoBatch:
    """A batch as produced by the pseudo-LiDAR cache path: no image, pseudo_pcl set."""
    pcl = torch.rand(b, n, 4)
    pcl[..., :3] = pcl[..., :3] * 10.0
    pcl[..., 3] = 1.0
    pseudo = torch.rand(b, p, 3)
    pseudo[..., 2] = pseudo[..., 2] + 2.0  # in front of the camera
    K = torch.eye(3).expand(b, 3, 3).clone()
    trans = torch.randn(b, 3) * 0.05
    rot = torch.eye(3).expand(b, 3, 3).clone()
    meta = []
    for i in range(b):
        T_gt = Transform.from_rotation_translation(
            np.eye(3, dtype=np.float32), np.array([0.0, 0.0, 0.05], np.float32)
        )
        T_decal = Transform.from_rotation_translation(rot[i].numpy(), trans[i].numpy())
        meta.append({"T_gt": T_gt, "T_init": T_decal @ T_gt, "T_decal": T_decal, "K": K[i].numpy()})
    return PseudoBatch(pcl=pcl, K=K, target_reg=(trans, rot), metadata=meta, pseudo_pcl=pseudo)


def test_cached_pseudolidar_path_trains() -> None:
    """With cached pseudo-LiDAR the model must skip depth and still learn."""
    model = _build_model().train()
    batch = _make_cached_batch()
    assert batch.image is None and batch.pseudo_pcl is not None
    loss = model.training_step(batch, 0)
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    # The pillar encoder (which consumes the cached cloud) must receive gradients …
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.pillars_pseudo.parameters()
    )
    # … while the depth estimator is never touched (no image fed to it).
    assert all(p.grad is None for p in model.depth.parameters())


def test_checkpoint_excludes_depth_weights() -> None:
    model = _build_model()
    # The real estimators (DA-V2 / GLPN) register weights + buffers under "depth.";
    # the Dummy is stateless, so add a probe buffer to exercise the strip/reinsert hook.
    model.depth.register_buffer("_probe", torch.zeros(4))
    ckpt: dict = {"state_dict": model.state_dict()}
    assert any(k.startswith("depth.") for k in ckpt["state_dict"])  # present pre-hook
    model.on_save_checkpoint(ckpt)
    assert not any(k.startswith("depth.") for k in ckpt["state_dict"])  # stripped
    model.on_load_checkpoint(ckpt)
    assert any(k.startswith("depth.") for k in ckpt["state_dict"])  # re-inserted
