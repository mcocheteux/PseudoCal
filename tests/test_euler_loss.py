"""Tests for the Euler-angle regression loss (yaw-flip-robust rotation loss)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("unical")

from unical.utils.transform import (  # noqa: E402
    euler_to_rotation_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

from pseudocal.losses.euler import (  # noqa: E402
    EulerRegressionLoss,
    _wrap,
    matrix_to_euler_xyz,
)


def _R(angles: torch.Tensor) -> torch.Tensor:
    return torch.stack([euler_to_rotation_matrix(a) for a in angles])


def test_matrix_to_euler_xyz_is_inverse_of_euler_to_matrix() -> None:
    angles = torch.tensor(
        [
            [0.20, -0.10, 2.90],  # yaw near +π
            [0.00, 0.05, -3.05],  # yaw near -π
            [0.26, -0.26, 1.50],  # ±15° roll/pitch
            [-0.1, 0.1, 0.0],
        ]
    )
    rec = matrix_to_euler_xyz(_R(angles))
    # Wrap the difference so ±π yaw compares cleanly.
    assert torch.allclose(_wrap(rec - angles), torch.zeros_like(angles), atol=1e-4)


def test_yaw_wraparound_is_small() -> None:
    """+179° predicted vs -179° target is a 2° error, not 358°."""
    loss = EulerRegressionLoss(trans_weight=0.0, rot_weight=1.0)
    pred_R = _R(torch.tensor([[0.0, 0.0, math.radians(179.0)]]))
    tgt_R = _R(torch.tensor([[0.0, 0.0, math.radians(-179.0)]]))
    pred = (torch.zeros(1, 3), matrix_to_rotation_6d(pred_R))
    batch = SimpleNamespace(target_reg=(torch.zeros(1, 3), tgt_R))
    r = loss(pred, batch)["loss/reg_rot"].item()
    assert r < math.radians(3.0) ** 2  # ≈ (2°)², not (358°)²


def _rot_grad_norm(loss_fn, pred_yaw: float, target_yaw: float) -> float:
    """Grad norm of the rotation loss w.r.t. a predicted-yaw leaf."""
    yaw = torch.tensor([pred_yaw], requires_grad=True)
    zeros = torch.zeros(1)
    pred_R = torch.stack([euler_to_rotation_matrix(torch.cat([zeros, zeros, yaw]))])
    tgt_R = _R(torch.tensor([[0.0, 0.0, target_yaw]]))
    pred = (torch.zeros(1, 3), matrix_to_rotation_6d(pred_R))
    batch = SimpleNamespace(target_reg=(torch.zeros(1, 3), tgt_R))
    loss_fn(pred, batch)["loss/reg_rot"].backward()
    return yaw.grad.abs().item()


def test_gradient_does_not_vanish_at_180_degree_error() -> None:
    """The whole point: a ~180° yaw error still yields a strong gradient to escape it,
    whereas Frobenius matrix-MSE's gradient vanishes there (∝ sin θ → 0)."""
    euler = EulerRegressionLoss(trans_weight=0.0, rot_weight=1.0)
    g_big = _rot_grad_norm(euler, pred_yaw=math.radians(179.0), target_yaw=0.0)
    g_small = _rot_grad_norm(euler, pred_yaw=math.radians(5.0), target_yaw=0.0)
    assert g_big > 1.0  # strong push away from the flip
    assert g_small > 0.0
    # Matrix-MSE gradient at the same 179° error is ~vanishing (∝ sin(179°)).
    matrix_grad = abs(math.sin(math.radians(179.0)))
    assert g_big > 10 * matrix_grad


def test_translation_and_rotation_terms_present_and_backprop() -> None:
    loss = EulerRegressionLoss(trans_weight=5.0, rot_weight=1.0)
    r6 = matrix_to_rotation_6d(_R(torch.tensor([[0.1, 0.05, 1.0]]))).requires_grad_(True)
    t = torch.zeros(1, 3, requires_grad=True)
    tgt_R = _R(torch.tensor([[0.0, 0.0, 0.0]]))
    batch = SimpleNamespace(target_reg=(torch.ones(1, 3), tgt_R))
    out = loss((t, r6), batch)
    assert {"loss/reg_trans", "loss/reg_rot"} == set(out)
    (out["loss/reg_trans"] + out["loss/reg_rot"]).backward()
    assert r6.grad is not None and t.grad is not None
    # sanity: 6D→matrix→euler→matrix round-trips
    assert torch.allclose(
        rotation_6d_to_matrix(r6) @ rotation_6d_to_matrix(r6).transpose(-1, -2),
        torch.eye(3),
        atol=1e-4,
    )
