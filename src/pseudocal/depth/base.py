"""
Abstract interface for the monocular depth estimators that produce pseudo-LiDAR.

A ``DepthEstimator`` maps a batch of RGB images to a **metric** depth map (in
metres). Metric scale matters here: the depth map is back-projected to a 3-D
pseudo-LiDAR cloud that must line up — in absolute scale — with the real LiDAR
scan so that the two bird's-eye-view pillar images are comparable.

Estimators are frozen feature extractors: they are not optimised during
training, so their parameters are excluded from the optimiser and their forward
pass runs under ``torch.no_grad()`` (see :meth:`DepthEstimator.estimate`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class DepthEstimator(nn.Module, ABC):
    """
    Base class for frozen monocular metric-depth estimators.

    Subclasses implement :meth:`_forward`, which receives a normalised image
    batch and returns a metric depth map of the same spatial size. The public
    :meth:`estimate` wraps it in ``no_grad`` and guarantees a ``(B, 1, H, W)``
    output, so callers never need to worry about gradient bookkeeping.
    """

    def __init__(self) -> None:
        super().__init__()
        self._frozen = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def freeze(self) -> DepthEstimator:
        """Disable gradients and switch to eval mode (idempotent)."""
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self._frozen = True
        return self

    def train(self, mode: bool = True) -> DepthEstimator:
        # A frozen estimator must stay in eval mode even when the parent
        # LightningModule calls ``.train()`` at the start of each epoch
        # (otherwise BatchNorm/Dropout statistics would drift).
        if self._frozen:
            return super().train(False)
        return super().train(mode)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def estimate(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) float tensor, ImageNet-normalised RGB.

        Returns:
            (B, 1, H, W) float tensor of **metric** depth in metres, on the same
            device as ``images``.
        """
        depth = self._forward(images)
        if depth.dim() == 3:  # (B, H, W) -> (B, 1, H, W)
            depth = depth.unsqueeze(1)
        return depth

    # ------------------------------------------------------------------
    # To implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return metric depth for ``images`` as ``(B, 1, H, W)`` or ``(B, H, W)``."""
        raise NotImplementedError
