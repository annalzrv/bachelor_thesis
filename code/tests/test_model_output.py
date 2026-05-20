"""Smoke tests: LossConditionalPINN output modes."""

from __future__ import annotations

import torch

from pinns.model import LossConditionalPINN


def test_sigmoid_output_in_open_unit_interval() -> None:
    torch.manual_seed(0)
    m = LossConditionalPINN(2, 4, [8, 8], output="sigmoid")
    coords = torch.rand(16, 2)
    log_lam = torch.randn(4)
    out = m(coords, log_lam)
    assert out.shape == (16, 1)
    assert (out > 0).all() and (out < 1).all()


def test_identity_forward_runs_on_cpu() -> None:
    m = LossConditionalPINN(2, 4, [8, 8], output="identity")
    coords = torch.zeros(4, 2)
    log_lam = torch.zeros(4)
    out = m(coords, log_lam)
    assert out.shape == (4, 1)


def test_invalid_output_raises() -> None:
    try:
        LossConditionalPINN(2, 4, [8], output="relu")  # type: ignore[arg-type]
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
