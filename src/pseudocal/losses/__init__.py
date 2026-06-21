"""Custom losses for PseudoCal (beyond those reused from ``unical.losses``)."""

from __future__ import annotations

from pseudocal.losses.euler import EulerRegressionLoss, matrix_to_euler_xyz

__all__ = ["EulerRegressionLoss", "matrix_to_euler_xyz"]
