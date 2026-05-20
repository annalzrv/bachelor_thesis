"""Causal PINN baseline on Burgers (Wang, Sankaran, Perdikaris 2022).

Time-binned exponential causal weighting of the PDE residual:
    L_r(θ) = (1/M) Σ_i w_i(θ) · L_r^(i)(θ)
    w_i(θ) = exp( - eps · Σ_{j<i} L_r^(j)(θ) )       (w_1 ≡ 1, stop-grad)

Bins are defined on the t-axis (M equal slices), L_r^(i) is the mean
residual² inside bin i. The weights enforce sequential-in-time training.
Same backbone as LC-PINN (hidden_dims = [64,64,64,64]).

Usage:
    python scripts/causal_pinn_burgers.py --seeds 0 1 2 3 --n-epochs 50000 --M 32 --eps 100

Output:
    results/causal_pinn_burgers.json
    checkpoints/causal_pinn_burgers_seed{s}.pt
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pinns.baseline import FixedWeightPINN
from pinns.device import select_device, device_info
from pinns.equations import burgers as burg


HIDDEN_DIMS = [64, 64, 64, 64]
REPO = pathlib.Path(__file__).resolve().parent.parent


def _bin_indices(t_vals: torch.Tensor, M: int, t_min: float, t_max: float) -> torch.Tensor:
    """Map t ∈ [t_min, t_max] to a bin index in {0, …, M-1}."""
    eps = 1e-6
    norm = (t_vals - t_min) / (t_max - t_min + eps)
    idx = torch.clamp((norm * M).long(), 0, M - 1)
    return idx.squeeze(-1)


def _bin_means(values: torch.Tensor, bin_idx: torch.Tensor, M: int) -> torch.Tensor:
    """Mean of `values` inside each of M bins. Empty bins default to 0."""
    sums  = torch.zeros(M, device=values.device, dtype=values.dtype)
    counts = torch.zeros(M, device=values.device, dtype=values.dtype)
    sums.scatter_add_(0, bin_idx, values.squeeze(-1))
    counts.scatter_add_(0, bin_idx, torch.ones_like(values.squeeze(-1)))
    return sums / counts.clamp(min=1.0)


def causal_residual_loss(model, batch, M: int, eps: float, t_min: float, t_max: float):
    coords = batch["coords_pde"].requires_grad_(True)
    u = model(coords)
    grads = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = grads[:, 0:1], grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, coords, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    residual = u_t + u * u_x - burg.NU * u_xx          # (N, 1)
    res_sq = residual ** 2

    bin_idx = _bin_indices(coords[:, 1:2], M, t_min, t_max)
    L_per_bin = _bin_means(res_sq, bin_idx, M)         # (M,)

    cum_prev = torch.cat([torch.zeros(1, device=L_per_bin.device),
                          torch.cumsum(L_per_bin.detach(), dim=0)[:-1]])
    weights = torch.exp(-eps * cum_prev)               # stop-grad on weights
    L_pde = (weights * L_per_bin).mean()
    return L_pde, L_per_bin.detach(), weights.detach()


def total_loss(model, batch, M, eps, t_min, t_max):
    L_pde, L_bins, w_bins = causal_residual_loss(model, batch, M, eps, t_min, t_max)
    L_bc   = torch.mean((model(batch["coords_bc"])   - batch["u_bc"])   ** 2)
    L_ic   = torch.mean((model(batch["coords_ic"])   - batch["u_ic"])   ** 2)
    L_data = torch.mean((model(batch["coords_data"]) - batch["u_data"]) ** 2)
    return L_pde + L_bc + L_ic + L_data, {
        "pde": float(L_pde.item()), "bc": float(L_bc.item()),
        "ic": float(L_ic.item()), "data": float(L_data.item()),
    }, L_bins, w_bins


def run_one_seed(seed, batch, ref, device, n_epochs, lr, M, eps):
    torch.manual_seed(seed); np.random.seed(seed)

    model = FixedWeightPINN(burg.DIM_PHYS, HIDDEN_DIMS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    t0 = time.time()
    print(f"[seed {seed}] Causal-PINN Adam: {n_epochs} steps  M={M}  eps={eps}")
    for step in range(n_epochs):
        opt.zero_grad()
        total, parts, L_bins, w_bins = total_loss(
            model, batch, M, eps, burg.T_MIN, burg.T_MAX
        )
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step(); sched.step()
        if step % 2000 == 0:
            w_min, w_max = float(w_bins.min()), float(w_bins.max())
            print(f"  step {step:6d}  L={total.item():.4e}  parts={parts}  "
                  f"w∈[{w_min:.3f},{w_max:.3f}]")

    elapsed = time.time() - t0
    errors = burg.evaluate(model, None, ref, device)
    mean_rel = float(np.mean(list(errors.values())))

    ckpt = REPO / "checkpoints" / f"causal_pinn_burgers_seed{seed}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "rel_l2_per_snapshot": {float(k): float(v) for k, v in errors.items()},
        "mean_rel_l2": mean_rel,
        "elapsed_sec": elapsed, "seed": seed,
        "n_epochs": n_epochs, "lr": lr, "M": M, "eps": eps,
    }, ckpt)

    print(f"[seed {seed}] done in {elapsed/60:.1f} min  rel-L2 = {mean_rel:.4e}")
    return {
        "seed": seed,
        "rel_l2_per_snapshot": {float(k): float(v) for k, v in errors.items()},
        "mean_rel_l2": mean_rel, "elapsed_sec": elapsed,
        "checkpoint": str(ckpt.relative_to(REPO)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--n-epochs", type=int, default=50_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--M", type=int, default=32, help="number of time bins")
    p.add_argument("--eps", type=float, default=100.0, help="causal weight steepness")
    args = p.parse_args()

    device = select_device()
    print(f"Device: {device_info(device)}")
    print(f"Config: seeds={args.seeds} n_epochs={args.n_epochs} M={args.M} eps={args.eps}\n")

    print("Building Burgers reference & training batch…")
    ref = burg.compute_reference_solution()
    batch = burg.generate_training_data(ref, device=device)

    runs = [run_one_seed(s, batch, ref, device, args.n_epochs, args.lr, args.M, args.eps)
            for s in args.seeds]
    means = [r["mean_rel_l2"] for r in runs]
    elapsed_total = [r["elapsed_sec"] for r in runs]
    summary = {
        "method": "causal-pinn", "equation": "burgers",
        "config": vars(args), "runs": runs,
        "summary": {
            "n_seeds": len(runs),
            "rel_l2_mean": float(np.mean(means)), "rel_l2_std":  float(np.std(means)),
            "rel_l2_min":  float(np.min(means)),  "rel_l2_max":  float(np.max(means)),
            "elapsed_mean_sec":  float(np.mean(elapsed_total)),
            "elapsed_total_sec": float(np.sum(elapsed_total)),
        },
    }
    out = REPO / "results" / "causal_pinn_burgers.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out.relative_to(REPO)}")
    print(f"  rel-L2: {summary['summary']['rel_l2_mean']:.4e} ± {summary['summary']['rel_l2_std']:.4e}")
    print(f"  mean wall time per seed: {summary['summary']['elapsed_mean_sec']/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
