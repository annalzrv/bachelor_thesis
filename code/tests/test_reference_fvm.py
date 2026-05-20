"""Smoke test: BL FVM reference returns consistent snapshot shapes."""

from __future__ import annotations

from pinns.equations.buckley_leverett import DEFAULT_DOMAIN, compute_reference_solution


def test_reference_snapshots_shapes() -> None:
    snap = compute_reference_solution(nx=40, domain=DEFAULT_DOMAIN)
    assert 0.0 in snap
    x, s = snap[0.0]
    assert x.shape == (40,) and s.shape == (40,)
    assert s.min() >= -1e-9 and s.max() <= 1.0 + 1e-9
