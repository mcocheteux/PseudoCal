"""
LiDAR-anchored depth-scale alignment.

PseudoCal's pseudo-LiDAR inherits its *metric scale* entirely from the monocular depth
network — the method's most fragile input. The real LiDAR, however, is a free metric
anchor. Key observation: the **range** of a point, ``r = ||xyz||``, is *exactly* invariant
to rotation about the sensor origin, so the pseudo↔real scale can be recovered from the two
clouds' range distributions **independently of the unknown extrinsic rotation** — including
the ±180° yaw the network is trying to solve.

:func:`estimate_range_scale` returns, per sample, the factor by which the pseudo-LiDAR
should be multiplied so its range distribution matches the real LiDAR's. It is robust
(quantile-based) and translation-tolerant (the small sensor baseline is negligible against
scene ranges).
"""

from __future__ import annotations

import torch


def _ranges(cloud: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-point range ``||xyz||`` and validity mask for a ``(B, N, 4+)`` cloud."""
    xyz = cloud[..., :3]
    valid = cloud[..., 3] > 0
    return xyz.norm(dim=-1), valid


def point_range_quantiles(cloud: torch.Tensor, quantiles: tuple[float, ...]) -> torch.Tensor:
    """
    Robust range quantiles per sample, ignoring padding/invalid points.

    Args:
        cloud:     ``(B, N, 4+)`` ``[x, y, z, valid, ...]``.
        quantiles: quantile levels in ``[0, 1]``.
    Returns:
        ``(B, Q)`` tensor of range quantiles (NaN for an all-empty sample).
    """
    r, valid = _ranges(cloud)
    r = r.masked_fill(~valid, float("nan"))
    q = torch.tensor(quantiles, device=cloud.device, dtype=r.dtype)
    return torch.nanquantile(r, q, dim=1).transpose(0, 1)  # (Q, B) -> (B, Q)


def estimate_range_scale(
    pseudo: torch.Tensor,
    real: torch.Tensor,
    quantiles: tuple[float, ...] = (0.5, 0.7, 0.9),
    eps: float = 1e-6,
    clamp: tuple[float, float] = (0.2, 5.0),
) -> torch.Tensor:
    """
    Per-sample factor to multiply the **pseudo** cloud by so its range distribution matches
    the **real** cloud's (rotation-invariant; robust to the unknown extrinsic).

    Args:
        pseudo / real: ``(B, N, 4+)`` clouds in the camera frame.
    Returns:
        ``(B,)`` scale factors (clamped; 1.0 where a cloud is empty).
    """
    qp = point_range_quantiles(pseudo, quantiles)
    qr = point_range_quantiles(real, quantiles)
    ratio = qr / qp.clamp_min(eps)  # (B, Q)
    s = ratio.nanmedian(dim=-1).values  # combine the quantile estimates robustly
    s = torch.nan_to_num(s, nan=1.0, posinf=clamp[1], neginf=clamp[0])
    return s.clamp(*clamp)
