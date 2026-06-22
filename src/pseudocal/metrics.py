"""
Diagnostic metrics for the 180°-yaw-ambiguity study.

These complement UniCal's :class:`~unical.utils.metrics.CalibMetrics` (mean/STD) with the
quantities that actually characterise the *heavy tail*:

  * :func:`yaw_error_deg` / :func:`flip_rate` — how often the coarse stage lands on the
    wrong front/back orientation (|yaw error| > 90°);
  * :func:`rotation_tail_percentiles` — P95/P99 of the error, where the tail lives;
  * :func:`bev_yaw_symmetry` — a per-scene score (normalised cross-correlation between a
    BEV pillar image and its 180°-rotated copy) used to test the hypothesis that flips
    concentrate on near-symmetric scenes.
"""

from __future__ import annotations

import torch

from pseudocal.losses.euler import _wrap, matrix_to_euler_xyz

_RAD2DEG = 180.0 / torch.pi


def yaw_error_deg(pred_R: torch.Tensor, target_R: torch.Tensor, yaw_axis: int = 2) -> torch.Tensor:
    """Absolute wrap-around yaw error (degrees) between rotation matrices ``(..., 3, 3)``."""
    dy = _wrap(
        matrix_to_euler_xyz(pred_R)[..., yaw_axis] - matrix_to_euler_xyz(target_R)[..., yaw_axis]
    )
    return dy.abs() * _RAD2DEG


def flip_rate(
    pred_R: torch.Tensor, target_R: torch.Tensor, thresh_deg: float = 90.0, yaw_axis: int = 2
) -> float:
    """Fraction of samples whose yaw error exceeds ``thresh_deg`` (i.e. a front/back flip)."""
    return float((yaw_error_deg(pred_R, target_R, yaw_axis) > thresh_deg).float().mean())


def rotation_tail_percentiles(
    errors_deg: torch.Tensor, ps: tuple[float, ...] = (95.0, 99.0)
) -> dict[str, float]:
    """Percentiles of an error tensor (degrees); the tail the mean/STD hide."""
    qs = torch.tensor([p / 100.0 for p in ps], dtype=errors_deg.dtype)
    vals = torch.quantile(errors_deg.flatten(), qs)
    return {f"P{p:g}": float(v) for p, v in zip(ps, vals)}


def bev_yaw_symmetry(image: torch.Tensor) -> torch.Tensor:
    """
    Per-sample yaw-symmetry score in ``[-1, 1]``: the normalised cross-correlation between
    a BEV pillar image and its 180°-rotated copy. ~1 ⇒ the scene looks the same front/back
    (where the coarse stage is most prone to flip).

    Args:
        image: ``(B, C, H, W)`` BEV pillar pseudo-image (e.g. ``PillarFeatureNet`` output).
    Returns:
        ``(B,)`` correlation scores.
    """
    if image.dim() != 4:
        raise ValueError(f"expected (B, C, H, W), got {tuple(image.shape)}")
    b = image.shape[0]
    a = image.reshape(b, -1)
    f = torch.flip(image, dims=(-2, -1)).reshape(b, -1)
    a = a - a.mean(dim=1, keepdim=True)
    f = f - f.mean(dim=1, keepdim=True)
    num = (a * f).sum(dim=1)
    den = a.norm(dim=1) * f.norm(dim=1)
    return num / den.clamp_min(1e-8)
