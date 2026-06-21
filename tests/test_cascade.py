"""Tests for the cascade composition math (no trained models required)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("unical")  # cascade runner builds on unical's Transform

from unical.utils.transform import (
    Transform,  # noqa: E402
    euler_to_transform_matrix,  # noqa: E402
)

from pseudocal.cascade.runner import CascadeCalibrator, RawSample  # noqa: E402


def _transform(tx: float, ty: float, tz: float, rz: float) -> Transform:
    T = euler_to_transform_matrix(torch.tensor([tx, ty, tz]), torch.tensor([0.0, 0.0, rz])).numpy()
    return Transform(T)


class _ConstantStage:
    """A stage that always predicts a fixed residual decalibration."""

    def __init__(self, residual: Transform) -> None:
        self._residual = residual

    def predict(self, raw: RawSample, T_current: Transform, device: torch.device) -> Transform:
        return self._residual


def test_single_stage_inverts_decalibration() -> None:
    # If a stage predicts exactly the applied decalibration, applying its inverse
    # recovers the ground truth.
    T_gt = _transform(0.5, -0.2, 1.0, 0.1)
    T_decal = _transform(0.05, 0.0, -0.03, 0.2)
    T_init = T_decal @ T_gt

    raw = RawSample(
        image_rgb=np.zeros((4, 4, 3), np.uint8),
        pcl=np.zeros((1, 4), np.float32),
        K=np.eye(3, dtype=np.float32),
        T_gt=T_gt,
        T_init=T_init,
    )
    cal = CascadeCalibrator([_ConstantStage(T_decal)], device="cpu")
    T_pred = cal.calibrate(raw)
    np.testing.assert_allclose(T_pred.matrix, T_gt.matrix, atol=1e-5)


def test_cascade_reduces_error_monotonically() -> None:
    # Two stages each removing part of the decalibration must not increase error.
    T_gt = _transform(1.0, 0.0, 2.0, 0.0)
    d1 = _transform(0.1, 0.0, 0.1, 0.15)
    d2 = _transform(0.02, 0.0, 0.0, 0.03)
    T_init = (d2 @ d1) @ T_gt  # total decalibration is d2∘d1

    raw = RawSample(
        image_rgb=np.zeros((4, 4, 3), np.uint8),
        pcl=np.zeros((1, 4), np.float32),
        K=np.eye(3, dtype=np.float32),
        T_gt=T_gt,
        T_init=T_init,
    )

    def err(T: Transform) -> float:
        return float(
            np.abs(T.translation - T_gt.translation).sum()
            + np.abs((T.inverse() @ T_gt).euler).sum()
        )

    # Stage 1 removes d2 (the outermost), stage 2 removes d1 — exact recovery.
    cal = CascadeCalibrator([_ConstantStage(d2), _ConstantStage(d1)], device="cpu")
    T_pred = cal.calibrate(raw)
    assert err(T_pred) < err(T_init)
    np.testing.assert_allclose(T_pred.matrix, T_gt.matrix, atol=1e-5)
