"""
Euler-angle regression loss for PseudoPillars.

The reused ``unical.losses.regression.RegressionLoss`` compares rotations via Frobenius
matrix MSE, whose gradient **vanishes at 180°** (∝ sin θ): once the coarse stage lands
on a ~180° yaw flip it gets almost no signal to escape, a major source of the cascade's
heavy-tailed error. This loss instead penalises the per-axis **Euler-angle** error
(wrap-around safe), which keeps a bounded, non-vanishing gradient through 180° — matching
the original PseudoCal paper, which regresses Euler angles directly.

The model still predicts a 6-D rotation (so the spatial loss, head, cascade and metrics
are unchanged); we extract Euler angles from the predicted/target rotation matrices in
the **XYZ** convention used by ``unical.utils.transform.euler_to_rotation_matrix`` (the
same convention the decalibrator builds ``T_decal`` with), so the angle error is exact.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from unical.utils.transform import rotation_6d_to_matrix


def matrix_to_euler_xyz(R: torch.Tensor) -> torch.Tensor:
    """
    Differentiable 'XYZ' Euler extraction, the inverse of ``euler_to_rotation_matrix``.

    For ``R = Rx(a) · Ry(b) · Rz(c)``:
        a = atan2(-R[1,2], R[2,2]),  b = asin(R[0,2]),  c = atan2(-R[0,1], R[0,0]).
    Stable for ``|pitch| ≪ 90°`` (our decalibration ranges keep roll/pitch ≤ ±15°).

    Args:
        R: ``(..., 3, 3)`` rotation matrices.
    Returns:
        ``(..., 3)`` Euler angles ``[roll(x), pitch(y), yaw(z)]`` in radians.
    """
    b = torch.asin(R[..., 0, 2].clamp(-1.0, 1.0))
    a = torch.atan2(-R[..., 1, 2], R[..., 2, 2])
    c = torch.atan2(-R[..., 0, 1], R[..., 0, 0])
    return torch.stack([a, b, c], dim=-1)


def _wrap(x: torch.Tensor) -> torch.Tensor:
    """Wrap angles to ``(-π, π]`` so a ±180° target has no artificial discontinuity."""
    return torch.atan2(torch.sin(x), torch.cos(x))


class EulerRegressionLoss(nn.Module):
    """
    Translation MSE + wrap-around Euler-angle MSE. Drop-in for ``RegressionLoss``: same
    ``forward(pred, batch)`` signature and ``loss/reg_{trans,rot}`` output keys, so it
    plugs into ``CombinedLoss`` / logging / metrics unchanged.

    Args:
        trans_weight / rot_weight: per-term weights (rot error is in radians²).
    """

    def __init__(self, trans_weight: float = 5.0, rot_weight: float = 1.0) -> None:
        super().__init__()
        self.trans_weight = trans_weight
        self.rot_weight = rot_weight
        self._mse = nn.MSELoss()

    def forward(
        self,
        pred: tuple[torch.Tensor, torch.Tensor],
        batch: Any,
    ) -> dict[str, torch.Tensor]:
        pred_t, pred_r6 = pred
        target_t, target_R = batch.target_reg
        pred_euler = matrix_to_euler_xyz(rotation_6d_to_matrix(pred_r6))
        target_euler = matrix_to_euler_xyz(target_R)

        t_loss = self._mse(pred_t, target_t) * self.trans_weight
        r_loss = (_wrap(pred_euler - target_euler) ** 2).mean() * self.rot_weight
        return {"loss/reg_trans": t_loss, "loss/reg_rot": r_loss}
