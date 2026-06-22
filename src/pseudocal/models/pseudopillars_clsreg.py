"""
PseudoPillars variant with the classify-then-regress yaw head.

Identical to :class:`~pseudocal.models.pseudopillars.PseudoPillars` except that the head
is a :class:`~pseudocal.models.yaw_head.ClassifyRegressYawHead` and training is supervised
by :class:`~pseudocal.losses.yaw_cls.YawClsRegLoss`, which needs the head's yaw *logits*
(not just the composed rotation). Inference is unchanged — the head still emits the
standard ``(trans, rot6d)`` via its hard-argmax path — so the cascade runner and metrics
work as-is.
"""

from __future__ import annotations

import torch
from unical.utils.transform import Transform, rotation_6d_to_matrix

from pseudocal.data.dataset import PseudoBatch
from pseudocal.models.pseudopillars import PseudoPillars
from pseudocal.models.yaw_head import ClassifyRegressYawHead


class PseudoPillarsClsReg(PseudoPillars):
    """PseudoPillars whose yaw is classified-then-regressed (see module docstring)."""

    head: ClassifyRegressYawHead

    def _step(
        self, batch: PseudoBatch
    ) -> tuple[dict[str, torch.Tensor], list[Transform], list[Transform]]:
        # One feature pass; the loss uses the yaw logits, the metrics use the composed pose.
        feat = self._features(batch)
        out = self.head.train_outputs(feat)
        losses = self.loss_fn(out, batch)

        with torch.no_grad():
            pred_t, pred_r6 = self.head(feat)  # hard-argmax compose → (trans, rot6d)
        B = pred_t.shape[0]
        pred_t_np = pred_t.detach().float().cpu().numpy()
        pred_R = rotation_6d_to_matrix(pred_r6).detach().float().cpu().numpy()
        tgt_t = batch.target_reg[0].detach().float().cpu().numpy()
        tgt_R = batch.target_reg[1].detach().float().cpu().numpy()
        pred_Ts = [Transform.from_rotation_translation(pred_R[i], pred_t_np[i]) for i in range(B)]
        target_Ts = [Transform.from_rotation_translation(tgt_R[i], tgt_t[i]) for i in range(B)]
        return losses, pred_Ts, target_Ts
