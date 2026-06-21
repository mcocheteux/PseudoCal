"""Pluggable monocular depth estimators used to build the pseudo-LiDAR cloud."""

from __future__ import annotations

from pseudocal.depth.base import DepthEstimator
from pseudocal.depth.depth_anything import DepthAnythingV2MetricEstimator
from pseudocal.depth.dummy import DummyDepthEstimator
from pseudocal.depth.glpn import GLPNDepthEstimator

__all__ = [
    "DepthEstimator",
    "DepthAnythingV2MetricEstimator",
    "GLPNDepthEstimator",
    "DummyDepthEstimator",
]
