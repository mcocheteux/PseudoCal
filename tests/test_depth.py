"""Tests for the pluggable depth estimators (Dummy only — no weight downloads)."""

from __future__ import annotations

import torch

from pseudocal.depth import DepthEstimator, DummyDepthEstimator


def test_dummy_depth_shape_and_range() -> None:
    est = DummyDepthEstimator(near=5.0, far=50.0, max_depth=80.0)
    images = torch.randn(2, 3, 64, 48)
    depth = est.estimate(images)

    assert depth.shape == (2, 1, 64, 48)
    assert torch.isfinite(depth).all()
    assert (depth >= 0.0).all() and (depth <= 80.0).all()


def test_dummy_depth_is_frozen() -> None:
    est = DummyDepthEstimator()
    assert isinstance(est, DepthEstimator)
    # No trainable parameters leak into an optimiser.
    assert all(not p.requires_grad for p in est.parameters())
    # Stays in eval mode even when the parent calls .train().
    est.train(True)
    assert est.training is False


def test_dummy_depth_no_grad() -> None:
    est = DummyDepthEstimator()
    images = torch.randn(1, 3, 32, 32, requires_grad=True)
    depth = est.estimate(images)
    assert depth.requires_grad is False
