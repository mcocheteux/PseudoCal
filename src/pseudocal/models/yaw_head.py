"""
Classify-then-regress head for the wide-range yaw of the coarse PseudoPillars stage.

A road scene and its 180°-yaw-rotated twin look nearly identical in bird's-eye-view
pillars, so a single unimodal rotation regressor is prone to *committing* to the wrong
front/back orientation — the source of PseudoCal's heavy-tailed error. Matrix-MSE makes
this worse (its gradient vanishes at π); an Euler loss softens it but the regressor still
has to pick one mode.

This head instead models the wide-range angle (the Euler-XYZ component that spans the full
±180°, i.e. index 2) as a **classification over ``n_yaw_bins``** plus an **in-bin residual**.
Classification represents the multi-modal front/back posterior explicitly and has a
non-vanishing cross-entropy gradient through π; the residual recovers the fine offset. Roll
and pitch (small ±15° ranges) stay as direct regression.

The composed output keeps the standard ``(trans (B,3), rot6d (B,6))`` contract, so the
cascade runner, metrics and inference path are unchanged — only training uses the extra
classification logits (see :class:`pseudocal.losses.yaw_cls.YawClsRegLoss`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from unical.utils.transform import euler_to_rotation_matrix, matrix_to_rotation_6d


def yaw_bin_centers(n_bins: int) -> torch.Tensor:
    """Centres (radians) of ``n_bins`` equal bins tiling ``[-π, π)``."""
    bw = 2.0 * math.pi / n_bins
    return -math.pi + (torch.arange(n_bins, dtype=torch.float32) + 0.5) * bw


def _mlp(dims: list[int], activate_last: bool) -> nn.Sequential:
    """LeakyReLU MLP (activation *before* each linear except the first), UniCal-style."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        if i > 0:
            layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Linear(dims[i], dims[i + 1]))
    if activate_last:
        layers.append(nn.LeakyReLU(inplace=True))
    return nn.Sequential(*layers)


@dataclass
class YawHeadOut:
    """Training-time outputs of :class:`ClassifyRegressYawHead`."""

    trans: torch.Tensor  # (B, 3)
    roll_pitch: torch.Tensor  # (B, 2)  Euler x, y (radians)
    yaw_logits: torch.Tensor  # (B, n_bins)
    yaw_residual: torch.Tensor  # (B, n_bins)  in-bin offset (radians), already scaled
    bin_centers: torch.Tensor  # (n_bins,)
    half_bin: float


class ClassifyRegressYawHead(nn.Module):
    """
    Two-stage head: regress translation + roll/pitch, classify + refine yaw.

    Args:
        in_features:   Backbone feature width.
        common_hidden: Shared trunk hidden sizes (may be empty).
        trans_hidden:  Translation branch hidden sizes.
        rot_hidden:    Hidden sizes shared by the roll/pitch, yaw-logit and yaw-residual
                       branches.
        n_yaw_bins:    Number of yaw classification bins tiling ±180°.
        yaw_axis:      Euler-XYZ index of the wide-range angle (2 = the ±180° component
                       in this project's decalibration convention).
    """

    def __init__(
        self,
        in_features: int,
        common_hidden: list[int],
        trans_hidden: list[int],
        rot_hidden: list[int],
        n_yaw_bins: int = 12,
        yaw_axis: int = 2,
    ) -> None:
        super().__init__()
        self.n_yaw_bins = n_yaw_bins
        self.yaw_axis = yaw_axis
        self.half_bin = math.pi / n_yaw_bins  # half of 2π/n_bins
        self.register_buffer("bin_centers", yaw_bin_centers(n_yaw_bins))

        trunk_dims = [in_features] + list(common_hidden)
        if len(trunk_dims) > 1:
            self.trunk: nn.Module = _mlp(trunk_dims, activate_last=False)
            trunk_out = trunk_dims[-1]
        else:
            self.trunk = nn.Identity()
            trunk_out = in_features

        self.trans_head = _mlp([trunk_out] + list(trans_hidden) + [3], activate_last=True)
        self.rp_head = _mlp([trunk_out] + list(rot_hidden) + [2], activate_last=True)
        self.yaw_cls = _mlp([trunk_out] + list(rot_hidden) + [n_yaw_bins], activate_last=True)
        self.yaw_res = _mlp([trunk_out] + list(rot_hidden) + [n_yaw_bins], activate_last=True)

    # ------------------------------------------------------------------

    def train_outputs(self, x: torch.Tensor) -> YawHeadOut:
        """Full outputs (incl. yaw logits) used by the training loss."""
        h = self.trunk(x)
        # tanh keeps the residual inside its bin (±half_bin).
        residual = torch.tanh(self.yaw_res(h)) * self.half_bin
        return YawHeadOut(
            trans=self.trans_head(h),
            roll_pitch=self.rp_head(h),
            yaw_logits=self.yaw_cls(h),
            yaw_residual=residual,
            bin_centers=self.bin_centers,
            half_bin=self.half_bin,
        )

    def _compose(self, out: YawHeadOut, yaw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ``(trans, rot6d)`` from translation, roll/pitch and a chosen yaw (B,)."""
        euler = torch.zeros(out.trans.shape[0], 3, device=yaw.device, dtype=yaw.dtype)
        euler[:, 0] = out.roll_pitch[:, 0]
        euler[:, 1] = out.roll_pitch[:, 1]
        euler[:, self.yaw_axis] = yaw
        rot6d = matrix_to_rotation_6d(euler_to_rotation_matrix(euler, convention="XYZ"))
        return out.trans, rot6d

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference: hard argmax bin + its residual → standard ``(trans, rot6d)``."""
        out = self.train_outputs(x)
        k = out.yaw_logits.argmax(dim=-1)  # (B,)
        yaw = out.bin_centers[k] + out.yaw_residual.gather(1, k[:, None]).squeeze(1)
        return self._compose(out, yaw)
