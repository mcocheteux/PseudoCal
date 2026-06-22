"""Custom losses for PseudoCal (beyond those reused from ``unical.losses``)."""

from __future__ import annotations

from pseudocal.losses.euler import EulerRegressionLoss, matrix_to_euler_xyz
from pseudocal.losses.yaw_cls import YawClsRegLoss

__all__ = ["EulerRegressionLoss", "YawClsRegLoss", "matrix_to_euler_xyz"]
