"""
Pseudo-LiDAR generation.

The camera image is turned into a 3-D *pseudo-LiDAR* point cloud in two steps:

  1. ``canny_edge_mask`` — a Canny edge detector flags depth-discontinuity pixels.
     Back-projecting straight through a depth edge produces long "flying pixel"
     streaks (interpolated depth between a foreground and background surface), which
     pollute the cloud. The paper reports that dropping these pixels improves the
     coarse calibration by a large margin, so we compute a *keep* mask (True where a
     pixel is **not** on/near an edge) on the CPU in the dataloader and carry it in
     the batch.

  2. ``backproject`` — each kept pixel ``(u, v)`` with metric depth ``z = D(u, v)`` is
     lifted to camera-frame coordinates using the pinhole model:

         x = (u - c_u) * z / f_u
         y = (v - c_v) * z / f_v
         z = z

     This runs on the GPU, batched, and is **not** part of the autograd graph (the
     depth network is frozen, so the pseudo-LiDAR is a fixed input to the pillars).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


def canny_edge_mask(
    image: np.ndarray,
    low: int = 50,
    high: int = 150,
    dilate: int = 2,
) -> np.ndarray:
    """
    Compute a boolean *keep* mask that is False on (dilated) image edges.

    Args:
        image:  (H, W, 3) RGB or (H, W) grayscale image. Floats are accepted and
                rescaled to uint8 internally.
        low:    Canny lower hysteresis threshold.
        high:   Canny upper hysteresis threshold.
        dilate: Edge dilation radius in pixels (widen the rejection band around
                each edge). 0 disables dilation.

    Returns:
        (H, W) boolean array — True for pixels to keep (off-edge).
    """
    img = image
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.dtype != np.uint8:
        lo, hi = float(img.min()), float(img.max())
        img = ((img - lo) / (hi - lo + 1e-8) * 255.0).astype(np.uint8)

    edges = cv2.Canny(img, low, high)
    if dilate > 0:
        kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        edges = cv2.dilate(edges, kernel)
    return edges == 0


def backproject(
    depth: torch.Tensor,
    K: torch.Tensor,
    keep_mask: torch.Tensor | None = None,
    stride: int = 2,
    min_depth: float = 1.0,
    max_depth: float = 80.0,
    max_points: int = 30000,
) -> torch.Tensor:
    """
    Back-project a metric depth map to a camera-frame pseudo-LiDAR cloud.

    Pixels are sub-sampled by ``stride`` (depth maps are dense; full resolution is
    both unnecessary and slow for the pillar encoder), filtered by the edge keep
    mask and a depth range, then padded to a fixed ``max_points`` so the batch is a
    regular tensor. Padding rows are exact zeros and carry a 0 in the returned
    validity channel.

    Args:
        depth:     (B, 1, H, W) metric depth in metres.
        K:         (B, 3, 3) camera intrinsics (matching the depth map resolution).
        keep_mask: (B, 1, H, W) or (B, H, W) boolean keep mask, or None.
        stride:    Pixel sub-sampling stride.
        min_depth: Discard points closer than this (metres).
        max_depth: Discard points farther than this (metres).
        max_points: Fixed padded length of the returned cloud.

    Returns:
        (B, max_points, 4) tensor ``[x, y, z, valid]`` in the camera frame.
    """
    b, _, h, w = depth.shape
    device, dtype = depth.device, depth.dtype

    # Pixel grid (sub-sampled).
    vs = torch.arange(0, h, stride, device=device)
    us = torch.arange(0, w, stride, device=device)
    grid_v, grid_u = torch.meshgrid(vs, us, indexing="ij")  # (h', w')
    grid_u = grid_u.reshape(-1).to(dtype)  # (P,)
    grid_v = grid_v.reshape(-1).to(dtype)
    p = grid_u.shape[0]

    z = depth[:, 0, ::stride, ::stride].reshape(b, p)  # (B, P)

    fu = K[:, 0, 0].view(b, 1)
    fv = K[:, 1, 1].view(b, 1)
    cu = K[:, 0, 2].view(b, 1)
    cv_ = K[:, 1, 2].view(b, 1)

    x = (grid_u.view(1, p) - cu) * z / fu
    y = (grid_v.view(1, p) - cv_) * z / fv
    xyz = torch.stack([x, y, z], dim=-1)  # (B, P, 3)

    valid = (z > min_depth) & (z < max_depth)  # (B, P)
    if keep_mask is not None:
        km = keep_mask
        if km.dim() == 4:
            km = km[:, 0]
        km = km[:, ::stride, ::stride].reshape(b, p).to(torch.bool)
        valid = valid & km

    xyz = xyz * valid.unsqueeze(-1)  # zero out invalid
    cloud = torch.cat([xyz, valid.to(dtype).unsqueeze(-1)], dim=-1)  # (B, P, 4)

    return _pad_or_sample(cloud, max_points)


def _pad_or_sample(cloud: torch.Tensor, max_points: int) -> torch.Tensor:
    """Pad with zero rows or randomly subsample to exactly ``max_points`` per batch."""
    b, p, c = cloud.shape
    if p == max_points:
        return cloud
    if p < max_points:
        pad = torch.zeros(b, max_points - p, c, device=cloud.device, dtype=cloud.dtype)
        return torch.cat([cloud, pad], dim=1)
    # Subsample (deterministic per call is not required; uniform without replacement).
    idx = torch.randperm(p, device=cloud.device)[:max_points]
    return cloud[:, idx]
