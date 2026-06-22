"""Tests for the yaw-ambiguity diagnostic metrics."""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("unical")

from unical.utils.transform import euler_to_rotation_matrix  # noqa: E402

from pseudocal.metrics import (  # noqa: E402
    bev_yaw_symmetry,
    flip_rate,
    rotation_tail_percentiles,
    yaw_error_deg,
)


def _yaw_R(yaw_deg: list[float]) -> torch.Tensor:
    euler = torch.zeros(len(yaw_deg), 3)
    euler[:, 2] = torch.tensor([math.radians(y) for y in yaw_deg])
    return euler_to_rotation_matrix(euler, convention="XYZ")


def test_yaw_error_and_flip_rate() -> None:
    target = _yaw_R([0.0, 0.0, 0.0, 0.0])
    pred = _yaw_R([5.0, 180.0, -170.0, 2.0])  # two of four are front/back flips

    err = yaw_error_deg(pred, target)
    assert err.shape == (4,)
    assert err[0] < 10.0 and err[3] < 10.0
    assert err[1] > 170.0 and err[2] > 160.0

    assert flip_rate(pred, target, thresh_deg=90.0) == pytest.approx(0.5)


def test_tail_percentiles_monotonic() -> None:
    errors = torch.linspace(0.0, 100.0, 101)
    p = rotation_tail_percentiles(errors, ps=(95.0, 99.0))
    assert p["P95"] < p["P99"]
    assert 94.0 < p["P95"] < 96.0


def test_bev_yaw_symmetry_high_for_symmetric_scene() -> None:
    # A 180°-rotation-symmetric image scores ~1; an asymmetric one scores lower.
    sym = torch.ones(1, 1, 8, 8)
    sym[0, 0, :4, :4] = 5.0
    sym[0, 0, 4:, 4:] = 5.0  # point-symmetric about the centre
    asym = torch.zeros(1, 1, 8, 8)
    asym[0, 0, :4, :] = 5.0  # top half only — not point-symmetric

    s_sym = bev_yaw_symmetry(sym)
    s_asym = bev_yaw_symmetry(asym)
    assert s_sym.shape == (1,)
    assert float(s_sym) > 0.9
    assert float(s_asym) < float(s_sym)


def test_bev_yaw_symmetry_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        bev_yaw_symmetry(torch.zeros(8, 8))
