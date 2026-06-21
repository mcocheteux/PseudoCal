"""
Spatial-preserving MobileViT backbone for PseudoPillars.

UniCal's :class:`~unical.models.backbone.MobileViTBackbone` ends in a global
average pool to ``1×1`` — fine for its task (image + projected depth, tiny ±1°
errors), where misalignment shows up as a *global* photometric-mismatch signal.

PseudoPillars is different: recovering a 6-DOF transform from two bird's-eye-view
pillar images is a **spatial registration** problem, and global pooling discards the
"where" the network needs (it makes the feature roughly translation-invariant — the
opposite of what a calibration regressor wants). This subclass keeps a coarse ``k×k``
spatial grid instead of collapsing to ``1×1`` and flattens it, so the downstream head
sees spatial structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from unical.data.dataset import Batch
from unical.models.backbone import MobileViTBackbone


class SpatialMobileViTBackbone(MobileViTBackbone):
    """
    MobileViT backbone that pools to a ``spatial_pool × spatial_pool`` grid and
    flattens, preserving coarse spatial layout for BEV registration.

    Args:
        spatial_pool: Side ``k`` of the retained feature grid. The flattened output
                      width is ``neck_hidden_sizes[-1] * k * k`` (set the head's
                      ``in_features`` accordingly). ``k=1`` reproduces the original
                      global-average-pooling behaviour.
        **kwargs:     Forwarded to :class:`MobileViTBackbone`.
    """

    def __init__(self, *args, spatial_pool: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.spatial_pool = spatial_pool
        # Replace the inherited AdaptiveAvgPool2d((1, 1)) with a k×k pool.
        self.pool = nn.AdaptiveAvgPool2d((spatial_pool, spatial_pool))

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Args:
            batch: object exposing ``.img`` and ``.lidar_map`` (B, C, H, W).

        Returns:
            (B, D * k * k) flattened spatial feature vector.
        """
        x = torch.cat([batch.img, batch.lidar_map], dim=1)
        out = self.model(x, return_dict=True)["last_hidden_state"]  # (B, D, H', W')
        out = self.spatial_head(out)  # (B, D, H', W')
        out = self.pool(out)  # (B, D, k, k)
        return rearrange(out, "b c h w -> b (c h w)")  # (B, D*k*k)
