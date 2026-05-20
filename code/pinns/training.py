"""Generic training loop for loss-conditional PINNs (equation-agnostic)."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.auto import tqdm

from pinns.lambda_sampler import LambdaSampler
from pinns.model import LossConditionalPINN


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    warmup_frac: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup then cosine annealing. If warmup_frac=0, cosine only."""
    if warmup_frac <= 0.0:
        return CosineAnnealingLR(optimizer, T_max=n_epochs)
    warmup_steps = max(1, int(n_epochs * warmup_frac))
    cosine_steps = max(1, n_epochs - warmup_steps)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def train_lc_pinn(
    model: LossConditionalPINN,
    sampler: LambdaSampler,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    loss_fn: Callable[..., dict[str, torch.Tensor]],
    n_epochs: int = 50000,
    lr: float = 1e-3,
    n_lambda_samples: int = 4,
    log_every: int = 500,
    warmup_frac: float = 0.0,
    on_log: Callable[[dict], None] | None = None,
) -> dict[str, list]:
    """
    Training loop for the Loss-Conditional PINN.

    loss_fn(model, log_lambda, batch) -> dict[str, Tensor]
        Must return a dict of named loss terms (e.g. {"pde": ..., "bc": ..., "ic": ...}).
        The training loop computes the weighted sum using sampled lambda.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = _make_scheduler(optimizer, n_epochs, warmup_frac)
    history: dict[str, list] = {"step": [], "total": [], "hw": []}
    t0 = time.time()

    pbar = tqdm(
        range(n_epochs),
        desc="Training",
        unit="step",
        miniters=log_every,
        smoothing=0.05,
    )
    for step in pbar:
        optimizer.zero_grad()
        total_loss = torch.zeros(1, device=device).squeeze()
        last_losses = None

        for _ in range(n_lambda_samples):
            log_lam, p_lam = sampler.sample(step)
            losses = loss_fn(model, log_lam, batch)
            weighted = sum(p * L for p, L in zip(p_lam, losses.values()))
            total_loss = total_loss + weighted
            last_losses = {k: v.item() for k, v in losses.items()}

        total_loss = total_loss / n_lambda_samples
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % log_every == 0:
            elapsed = time.time() - t0
            hw = sampler.half_width(step)
            history["step"].append(step)
            history["total"].append(total_loss.item())
            history["hw"].append(hw)
            for k, v in last_losses.items():
                history.setdefault(k, []).append(v)
            pbar.set_postfix(
                L=f"{total_loss.item():.2e}",
                hw=f"{hw:.2f}",
                refresh=False,
            )
            if on_log is not None:
                on_log(history)

    elapsed = time.time() - t0
    history["elapsed_sec"] = elapsed
    print(f"\nTraining complete in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    return history


def lbfgs_finish_lc(
    model: LossConditionalPINN,
    sampler: LambdaSampler,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    loss_fn: Callable[..., dict[str, torch.Tensor]],
    n_iter: int,
    n_lambda_support: int = 16,
    log_every: int = 50,
    resample_every: int = 100,
) -> dict:
    """L-BFGS finishing for LambdaSampler-based LC-PINN training.

    Wolfe line search assumes deterministic loss. We pre-sample
    n_lambda_support (log_lam, p_lam) pairs from the (post-curriculum)
    sampler and freeze them across closure calls — the loss is then a
    deterministic average over a quasi-Monte-Carlo grid. The support is
    refreshed every resample_every iterations to avoid overfitting to a
    single grid.
    """
    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        max_eval=25,
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    state: dict = {"step": 0, "last_loss": float("nan"), "last_parts": None,
                   "support": None}

    def refresh_support():
        log_lams, p_lams = sampler.sample_batch(n_lambda_support, step=10**9)
        state["support"] = (log_lams, p_lams)

    refresh_support()

    def closure():
        opt.zero_grad()
        total = torch.zeros(1, device=device).squeeze()
        last_parts = None
        log_lams, p_lams = state["support"]
        for i in range(log_lams.shape[0]):
            log_lam = log_lams[i]
            p_lam = p_lams[i]
            losses = loss_fn(model, log_lam, batch)
            weighted = sum(p * L for p, L in zip(p_lam, losses.values()))
            total = total + weighted
            last_parts = {k: float(v.item()) for k, v in losses.items()}
        total = total / log_lams.shape[0]
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        state["last_loss"] = float(total.item())
        state["last_parts"] = last_parts
        return total

    t0 = time.time()
    for it in range(n_iter):
        if it > 0 and it % resample_every == 0:
            refresh_support()
        opt.step(closure)
        state["step"] = it
        loss = state["last_loss"]
        if not (loss == loss):
            print(f"  [lbfgs] NaN at iter {it} — aborting L-BFGS phase", flush=True)
            break
        if it % log_every == 0:
            print(f"  [lbfgs] step {it:5d}  L={loss:.4e}  parts={state['last_parts']}",
                  flush=True)
    return {"elapsed_sec": time.time() - t0, "final_loss": state["last_loss"]}
