"""Smoke tests: equation modules — reference solutions and training data."""

from __future__ import annotations

import numpy as np


def test_logistic_reference_shape() -> None:
    from pinns.equations.logistic import compute_reference_solution

    snap = compute_reference_solution()
    assert len(snap) > 0
    for t_val, (t_pts, u_pts) in snap.items():
        assert t_pts.shape == u_pts.shape
        assert u_pts.min() >= 0.0


def test_burgers_reference_shape() -> None:
    from pinns.equations.burgers import compute_reference_solution

    snap = compute_reference_solution(nx=64, snap_times=[0.25, 0.5])
    assert len(snap) == 2
    for t_val, (x_pts, u_pts) in snap.items():
        assert x_pts.shape == (64,)
        assert u_pts.shape == (64,)
        assert np.all(np.isfinite(u_pts))


def test_allen_cahn_reference_shape() -> None:
    from pinns.equations.allen_cahn import compute_reference_solution

    snap = compute_reference_solution(nx=64, snap_times=[0.25, 0.5])
    assert len(snap) == 2
    for t_val, (x_pts, u_pts) in snap.items():
        assert x_pts.shape == (64,)
        assert u_pts.shape == (64,)
        assert np.all(np.isfinite(u_pts))
