"""Generic inference utilities for LC-PINN (equation-agnostic)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from pinns.lambda_sampler import LambdaSampler
from pinns.model import LossConditionalPINN


def _weights_from_log_lambda(log_lambda: torch.Tensor, mode: str) -> torch.Tensor:
    # uniform mode: model was trained on raw weights in [0,1], not softmaxed.
    # simplex/logspace: weights live on the simplex — softmax is the right map.
    if mode == "uniform":
        return torch.exp(log_lambda)
    return torch.softmax(log_lambda, dim=-1)


def sweep_lambda(
    model: LossConditionalPINN,
    batch: dict[str, torch.Tensor],
    sampler: LambdaSampler,
    device: torch.device,
    loss_fn: Callable[..., dict[str, torch.Tensor]],
    n_candidates: int = 500,
    step: int | None = None,
    exclude_terms: set[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """
    Sweep lambda space to find a weight vector minimizing validation loss.

    loss_fn(model, log_lambda, batch) -> dict[str, Tensor]
    exclude_terms: loss keys to exclude from the validation metric (e.g. {"pde"}).
                   Defaults to {"pde"} — sweep on BC+IC+data only.

    ⚠ Caveat on the default: excluding PDE from the metric biases the chosen
    λ toward small λ_pde by construction (the λ that told the network to
    ignore PDE naturally minimises BC+IC+data loss). Pass exclude_terms=set()
    for an unbiased sweep over all loss terms. See
    notebooks/05_weight_audit.ipynb.
    """
    if exclude_terms is None:
        exclude_terms = {"pde"}
    dev = torch.device(device)
    batch = {k: v.to(dev) for k, v in batch.items()}
    if step is None:
        step = sampler.curriculum_steps
    log_lams, _ = sampler.sample_batch(n_candidates, step=step)
    log_lams = log_lams.to(dev)
    best_val = float("inf")
    best_ll = None
    vals = []
    for i in range(n_candidates):
        ll = log_lams[i]
        losses = loss_fn(model, ll, batch)
        val = sum(v for k, v in losses.items() if k not in exclude_terms).item()
        vals.append(val)
        if val < best_val:
            best_val = val
            best_ll = ll.clone()
    assert best_ll is not None
    p_best = _weights_from_log_lambda(best_ll, sampler.mode)
    print(f"Best log(lambda):     {best_ll.cpu().numpy().round(3)}")
    print(f"Best weights ({sampler.mode}): {p_best.cpu().numpy().round(4)}")
    print(f"Best validation loss: {best_val:.6e}")
    return best_ll, p_best, vals


@torch.no_grad()
def find_best_lambda(
    model: LossConditionalPINN,
    ref_snapshots: dict[float, tuple[np.ndarray, np.ndarray]],
    sampler: LambdaSampler,
    device: torch.device,
    predict_fn: Callable,
    n_candidates: int = 1000,
    step: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[float, float]]:
    """
    Find the lambda that minimises mean relative L2 error vs reference.

    predict_fn(model, log_lambda_or_None, x_pts, t_val, device) -> np.ndarray
    """
    if step is None:
        step = sampler.curriculum_steps
    log_lams, _ = sampler.sample_batch(n_candidates, step)

    best_mean_err = float("inf")
    best_ll = log_lams[0]
    best_errors: dict[float, float] = {}

    for i in range(n_candidates):
        ll = log_lams[i]
        errors: dict[float, float] = {}
        for t_val, (x_ref, u_ref) in sorted(ref_snapshots.items()):
            if t_val == 0.0:
                continue
            u_pred = predict_fn(model, ll, x_ref, t_val, device)
            errors[t_val] = float(np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-10))
        mean_err = float(np.mean(list(errors.values())))
        if mean_err < best_mean_err:
            best_mean_err = mean_err
            best_ll = ll.clone()
            best_errors = errors

    p_best = _weights_from_log_lambda(best_ll, sampler.mode)
    print(f"CV best log(lambda):     {best_ll.cpu().numpy().round(3)}")
    print(f"CV best weights ({sampler.mode}): {p_best.cpu().numpy().round(4)}")
    print(f"CV best mean rel-L2:     {best_mean_err:.4f}")
    return best_ll, p_best, best_errors


@torch.no_grad()
def find_worst_lambda(
    model: LossConditionalPINN,
    ref_snapshots: dict[float, tuple[np.ndarray, np.ndarray]],
    sampler: LambdaSampler,
    device: torch.device,
    predict_fn: Callable,
    n_candidates: int = 500,
    step: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[float, float]]:
    """Find the lambda that maximises mean relative L2 error (worst case)."""
    if step is None:
        step = sampler.curriculum_steps
    log_lams, _ = sampler.sample_batch(n_candidates, step)

    worst_mean_err = -1.0
    worst_ll = log_lams[0]
    worst_errors: dict[float, float] = {}

    for i in range(n_candidates):
        ll = log_lams[i]
        errors: dict[float, float] = {}
        for t_val, (x_ref, u_ref) in sorted(ref_snapshots.items()):
            if t_val == 0.0:
                continue
            u_pred = predict_fn(model, ll, x_ref, t_val, device)
            errors[t_val] = float(np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-10))
        mean_err = float(np.mean(list(errors.values())))
        if mean_err > worst_mean_err:
            worst_mean_err = mean_err
            worst_ll = ll.clone()
            worst_errors = errors

    p_worst = _weights_from_log_lambda(worst_ll, sampler.mode)
    print(f"Worst log(lambda):     {worst_ll.cpu().numpy().round(3)}")
    print(f"Worst weights ({sampler.mode}): {p_worst.cpu().numpy().round(4)}")
    print(f"Worst mean rel-L2:     {worst_mean_err:.4f}")
    return worst_ll, p_worst, worst_errors
