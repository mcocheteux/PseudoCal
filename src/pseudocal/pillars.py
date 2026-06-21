"""
PointPillars encoder (plain-PyTorch, dependency-free).

Both the pseudo-LiDAR and the real LiDAR clouds are encoded into a compact 2-D
*bird's-eye-view* pseudo-image so a standard image backbone (MobileViT) can fuse
them. We follow PointPillars (Lang et al., CVPR 2019):

  1. Discretise the BEV plane into a fixed grid of square *pillars*.
  2. Augment each point with offsets to its pillar's point-mean and geometric
     centre, then embed it with a tiny per-point PointNet (Linear → BN → ReLU).
  3. Max-pool the embeddings within each pillar and *scatter* them back to a dense
     ``(C, H, W)`` pseudo-image.

Everything runs in batched ``torch`` with no custom CUDA ops (no ``spconv``), which
keeps installation trivial on Vast.ai at the cost of some speed — acceptable at
KITTI point-cloud sizes.

Frame convention
----------------
Clouds are expressed in the **camera frame** (x → right, y → down, z → forward).
The BEV plane is therefore ``(x, z)`` and the pillar feature/height axis is ``y``.
Both the pseudo-LiDAR (already in the camera frame) and the real LiDAR (mapped to
the camera frame via the current extrinsic estimate) live in this same frame, so
their pillar images are directly comparable — the residual misalignment between
them is exactly what the network regresses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PillarGrid:
    """
    BEV pillar grid over the camera-frame ``(x, z)`` plane.

    Args:
        x_min, x_max: Lateral (right) range in metres.
        z_min, z_max: Forward range in metres.
        pillar_size:  Square pillar side in metres.
    """

    x_min: float = -40.0
    x_max: float = 40.0
    z_min: float = 0.0
    z_max: float = 80.0
    pillar_size: float = 0.32

    @property
    def width(self) -> int:
        """Number of pillars along x."""
        return int(round((self.x_max - self.x_min) / self.pillar_size))

    @property
    def height(self) -> int:
        """Number of pillars along z."""
        return int(round((self.z_max - self.z_min) / self.pillar_size))


class PillarFeatureNet(nn.Module):
    """
    Encode a point cloud into a dense BEV pseudo-image.

    Args:
        grid:         Pillar grid definition.
        out_channels: Channel count ``C`` of the produced pseudo-image.
    """

    # Per-point augmented feature size: [x, y, z, dx, dy, dz, px, pz]
    _IN_FEATURES = 8

    def __init__(self, grid: PillarGrid, out_channels: int = 32) -> None:
        super().__init__()
        self.grid = grid
        self.out_channels = out_channels
        self.linear = nn.Linear(self._IN_FEATURES, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, cloud: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cloud: (B, N, 4+) tensor ``[x, y, z, valid, ...]`` in the camera frame.
                   The 4th channel is a validity flag (> 0 means a real point);
                   padding rows must be all-zero.

        Returns:
            (B, C, H, W) pseudo-image.
        """
        b, n, _ = cloud.shape
        device = cloud.device
        g = self.grid
        W, H = g.width, g.height
        n_cells = H * W

        xyz = cloud[..., :3]  # (B, N, 3)
        valid = cloud[..., 3] > 0  # (B, N)

        # Pillar (column) indices on the (x, z) plane.
        ix = ((xyz[..., 0] - g.x_min) / g.pillar_size).floor().long()
        iz = ((xyz[..., 2] - g.z_min) / g.pillar_size).floor().long()
        in_range = valid & (ix >= 0) & (ix < W) & (iz >= 0) & (iz < H)

        ix = ix.clamp(0, W - 1)
        iz = iz.clamp(0, H - 1)
        cell = iz * W + ix  # (B, N) in [0, H*W)

        # Global (batch-flattened) destination index; out-of-range/padding points are
        # routed to a throwaway "dump" bucket at index ``B * n_cells``.
        batch_off = torch.arange(b, device=device).view(b, 1) * n_cells
        gidx = torch.where(in_range, cell + batch_off, torch.full_like(cell, b * n_cells))  # (B, N)
        gidx_flat = gidx.reshape(-1)  # (B*N,)

        # --- per-pillar point mean (for the dx,dy,dz offset features) -------------
        sums = torch.zeros(b * n_cells + 1, 3, device=device, dtype=xyz.dtype)
        counts = torch.zeros(b * n_cells + 1, 1, device=device, dtype=xyz.dtype)
        sums.index_add_(0, gidx_flat, xyz.reshape(-1, 3))
        counts.index_add_(0, gidx_flat, in_range.reshape(-1, 1).to(xyz.dtype))
        mean = sums / counts.clamp(min=1.0)  # (B*n_cells+1, 3)
        point_mean = mean[gidx_flat].reshape(b, n, 3)  # (B, N, 3)

        # --- per-pillar geometric centre (x, z) ----------------------------------
        x_centre = g.x_min + (ix.to(xyz.dtype) + 0.5) * g.pillar_size
        z_centre = g.z_min + (iz.to(xyz.dtype) + 0.5) * g.pillar_size

        feats = torch.cat(
            [
                xyz,  # absolute x, y, z
                xyz - point_mean,  # offset to pillar mean
                (xyz[..., 0] - x_centre).unsqueeze(-1),  # offset to pillar centre (x)
                (xyz[..., 2] - z_centre).unsqueeze(-1),  # offset to pillar centre (z)
            ],
            dim=-1,
        )  # (B, N, 8)
        feats = feats * in_range.unsqueeze(-1)  # zero padding/out-of-range

        # --- per-point PointNet embedding ----------------------------------------
        x = self.linear(feats.reshape(b * n, self._IN_FEATURES))
        x = self.norm(x)
        x = self.act(x)  # (B*N, C), >= 0
        x = x * in_range.reshape(b * n, 1)  # kill padding contributions

        # --- scatter-max into the dense BEV grid ---------------------------------
        # ReLU output is non-negative, so an init of 0 is the correct "empty pillar"
        # value and ``amax`` reduction yields per-pillar max-pooling.
        canvas = torch.zeros(b * n_cells + 1, self.out_channels, device=device, dtype=x.dtype)
        canvas.scatter_reduce_(
            0,
            gidx_flat.view(-1, 1).expand(-1, self.out_channels),
            x,
            reduce="amax",
            include_self=True,
        )
        canvas = canvas[: b * n_cells]  # drop dump bucket
        image = canvas.view(b, H, W, self.out_channels).permute(0, 3, 1, 2).contiguous()
        return image  # (B, C, H, W)
