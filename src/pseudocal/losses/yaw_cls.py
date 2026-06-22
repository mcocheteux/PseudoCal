"""
Loss for the classify-then-regress yaw head.

Supervises four things against the decalibration target:
  * **yaw classification** — cross-entropy over the bins (non-vanishing gradient through
    ±180°, the property a matrix-MSE/Euler regressor lacks at the flip);
  * **yaw residual** — L1 on the in-bin offset, supervised only on the *true* bin (the
    standard anchor-and-delta recipe);
  * **roll/pitch** — wrap-around MSE (small ±15° ranges, direct regression);
  * **translation** — MSE.

Target Euler angles are extracted in the same XYZ convention the decalibrator uses
(:func:`pseudocal.losses.euler.matrix_to_euler_xyz`), so binning is exact.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from pseudocal.losses.euler import _wrap, matrix_to_euler_xyz
from pseudocal.models.yaw_head import YawHeadOut


class YawClsRegLoss(nn.Module):
    """
    Args:
        yaw_axis:      Euler-XYZ index of the wide-range (classified) angle.
        cls_weight / res_weight / rollpitch_weight / trans_weight: per-term weights.
    """

    def __init__(
        self,
        yaw_axis: int = 2,
        cls_weight: float = 1.0,
        res_weight: float = 1.0,
        rollpitch_weight: float = 1.0,
        trans_weight: float = 5.0,
    ) -> None:
        super().__init__()
        self.yaw_axis = yaw_axis
        self.cls_weight = cls_weight
        self.res_weight = res_weight
        self.rollpitch_weight = rollpitch_weight
        self.trans_weight = trans_weight
        self._ce = nn.CrossEntropyLoss()
        self._mse = nn.MSELoss()
        self._l1 = nn.L1Loss()

    def forward(self, out: YawHeadOut, batch: Any) -> dict[str, torch.Tensor]:
        target_t, target_R = batch.target_reg
        target_euler = matrix_to_euler_xyz(target_R)  # (B, 3) [roll, pitch, yaw]
        yaw_t = _wrap(target_euler[:, self.yaw_axis])  # (-π, π]

        # --- yaw classification + in-bin residual --------------------------------
        n_bins = out.bin_centers.shape[0]
        bw = 2.0 * out.half_bin
        target_bin = torch.floor((yaw_t + torch.pi) / bw).long().clamp(0, n_bins - 1)
        cls_loss = self._ce(out.yaw_logits, target_bin) * self.cls_weight

        res_target = _wrap(yaw_t - out.bin_centers[target_bin])  # within ±half_bin
        res_pred = out.yaw_residual.gather(1, target_bin[:, None]).squeeze(1)
        res_loss = self._l1(res_pred, res_target) * self.res_weight

        # --- roll / pitch (wrap-around MSE) and translation ----------------------
        rp_idx = [i for i in (0, 1, 2) if i != self.yaw_axis]
        rp_target = target_euler[:, rp_idx]
        rp_loss = (_wrap(out.roll_pitch - rp_target) ** 2).mean() * self.rollpitch_weight
        t_loss = self._mse(out.trans, target_t) * self.trans_weight

        total = cls_loss + res_loss + rp_loss + t_loss
        return {
            "loss/yaw_cls": cls_loss,
            "loss/yaw_res": res_loss,
            "loss/reg_rollpitch": rp_loss,
            "loss/reg_trans": t_loss,
            "loss": total,
        }
