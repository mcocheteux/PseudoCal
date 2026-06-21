"""
Deterministic synthetic depth estimator for tests and CPU smoke runs.

Produces a plausible, fully reproducible metric depth map without downloading any
model weights, so the unit tests and ``trainer.fast_dev_run`` checks stay offline
and fast. Selected via the Hydra override ``depth=dummy``.
"""

from __future__ import annotations

import torch

from pseudocal.depth.base import DepthEstimator


class DummyDepthEstimator(DepthEstimator):
    """
    Synthetic depth: a ground-plane-like vertical gradient plus a small,
    image-content-dependent perturbation (so different images yield different
    depth, which exercises the downstream pillar encoder).

    Args:
        near:      Depth (m) at the top of the image.
        far:       Depth (m) at the bottom of the image.
        max_depth: Upper clamp, in metres.
    """

    def __init__(self, near: float = 5.0, far: float = 50.0, max_depth: float = 80.0) -> None:
        super().__init__()
        self.near = near
        self.far = far
        self.max_depth = max_depth
        # No parameters, but mark frozen so train()/eval() behave like a real estimator.
        self.freeze()

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        b, _, h, w = images.shape
        # Vertical gradient near (top) -> far (bottom): a crude flat-ground prior.
        rows = torch.linspace(self.near, self.far, h, device=images.device, dtype=images.dtype)
        depth = rows.view(1, 1, h, 1).expand(b, 1, h, w).clone()
        # Add a deterministic, image-dependent ripple so pillars are non-trivial.
        depth = depth + images.mean(dim=1, keepdim=True)
        return depth.clamp(min=0.1, max=self.max_depth)
