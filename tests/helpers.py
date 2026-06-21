"""Small shared helpers for the test suite."""

from __future__ import annotations

import torch


def make_intrinsics(width: int = 512, height: int = 512, f: float = 500.0) -> torch.Tensor:
    """A plausible pinhole intrinsic matrix for a (width, height) image."""
    return torch.tensor(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
