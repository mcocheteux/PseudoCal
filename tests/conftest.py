"""Shared pytest fixtures / helpers for the PseudoCal test suite."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Make every test deterministic."""
    torch.manual_seed(0)
    np.random.seed(0)
