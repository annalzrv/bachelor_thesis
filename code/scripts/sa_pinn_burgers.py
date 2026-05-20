"""SA-PINN baseline on Burgers (ν = 0.01/π).

Per-point trainable weights with polynomial mask m(λ) = λ², minimax
saddle-point optimisation: descend on θ, ascend on λ. Replicates
McClenny & Braga-Neto (2023) on the same backbone as LC-PINN
(hidden_dims = [64, 64, 64, 64]) so the comparison is identical
in capacity.

Usage:
    python scripts/sa_pinn_burgers.py --seeds 0 1 2 3 \\
        --n-adam 10000 --n-lbfgs 5000

Output:
    results/sa_pinn_burgers.json
    checkpoints/sa_pinn_burgers_seed{s}.pt
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


def _per_point_residual_pde(model, coords):
    coords = coords.requires_grad_(True)
    u = model(coords)
    grads = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = grads[:, 0:1], grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, coords, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    return u_t + u * u_x - burg.NU * u_xx  # (N_r, 1)


def _per_point_squared_error(model, coords, target):
    return (model(coords) - target) ** 2  # (N, 1)


def _sa_total_loss(model, lam_r, lam_b, lam_0, batch, q=2):
    """Σ_p Σ_i m(λ_p^i) · |residual|^2_i, mask m(λ)=λ^q."""
    res_pde_sq = _per_point_residual_pde(model, batch["coords_pde"]) ** 2  # (N_r, 1)
    res_bc_sq  = _per_point_squared_error(model, batch["coords_bc"], batch["u_bc"])
    res_ic_sq  = _per_point_squared_error(model, batch["coords_ic"], batch["u_ic"])
    # Sparse-data term uses unit weights (it's already ground truth)
    res_dat_sq = _per_point_squared_error(model, batch["coords_data"], batch["u_data"])

    L_pde = 0.5 * ((lam_r ** q).unsqueeze(1) * res_pde_sq).mean()
    L_bc  = 0.5 * ((lam_b ** q).unsqueeze(1) * res_bc_sq).mean()
    L_ic  = 0.5 * ((lam_0 ** q).unsqueeze(1) * res_ic_sq).mean()
    L_dat = res_dat_sq.mean()
    return L_pde + L_bc + L_ic + L_dat, {
        "pde": float(L_pde.item()), "bc": float(L_bc.item()),
        "ic": float(L_ic.item()), "data": float(L_dat.item()),
    }


def run_one_seed(seed: int, batch, ref, device, n_adam: int, n_lbfgs: int,
                 lr_theta: float, lr_lambda: float) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = FixedWeightPINN(burg.DIM_PHYS, HIDDEN_DIMS).to(device)

    N_r = batch["coords_pde"].shape[0]
    N_b = batch["coords_bc"].shape[0]
    N_0 = batch["coords_ic"].shape[0]

    lam_r = nn.Parameter(torch.rand(N_r, device=device))
    lam_b = nn.Parameter(torch.rand(N_b, device=device))
    lam_0 = nn.Parameter(torch.rand(N_0, device=device))

    opt_theta = torch.optim.Adam(model.parameters(), lr=lr_theta)
    opt_lambda = torch.optim.Adam([lam_r, lam_b, lam_0], lr=lr_lambda, maximize=True)

    t0 = time.time()
    last = {}
    print(f"[seed {seed}] Adam phase: {n_adam} steps")
    for step in range(n_adam):
        opt_theta.zero_grad()
        opt_lambda.zero_grad()
        total, parts = _sa_total_loss(model, lam_r, lam_b, lam_0, batch)
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt_theta.step()
        opt_lambda.step()
        if step % 1000 == 0:
            last = parts
            print(f"  step {step:6d}  L={total.item():.4e}  parts={parts}")

    if n_lbfgs > 0:
        print(f"[seed {seed}] L-BFGS phase: {n_lbfgs} steps (λ frozen)")
        lam_r.requires_grad_(False)
        lam_b.requires_grad_(False)
        lam_0.requires_grad_(False)
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            max_iter=20, tolerance_grad=1e-9, tolerance_change=1e-12,
            history_size=50, line_search_fn="strong_wolfe",
        )
        n_outer = max(1, n_lbfgs // 20)
        for outer in range(n_outer):
            def closure():
                lbfgs.zero_grad()
                total, _ = _sa_total_loss(model, lam_r, lam_b, lam_0, batch)
                total.backward()
                return total
            loss_val = lbfgs.step(closure)
            if outer % 50 == 0:
                print(f"  outer {outer:4d}  L={float(loss_val):.4e}")

    elapsed = time.time() - t0

    errors = burg.evaluate(model, None, ref, device)
    mean_rel = float(np.mean(list(errors.values())))

    ckpt = REPO / "checkpoints" / f"sa_pinn_burgers_seed{seed}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "lam_r": lam_r.detach().cpu(), "lam_b": lam_b.detach().cpu(), "lam_0": lam_0.detach().cpu(),
        "rel_l2_per_snapshot": {float(k): float(v) for k, v in errors.items()},
        "mean_rel_l2": mean_rel,
        "elapsed_sec": elapsed, "seed": seed,
        "n_adam": n_adam, "n_lbfgs": n_lbfgs,
        "lr_theta": lr_theta, "lr_lambda": lr_lambda,
    }, ckpt)

    print(f"[seed {seed}] done in {elapsed/60:.1f} min  rel-L2 = {mean_rel:.4e}")
    return {
        "seed": seed,
        "rel_l2_per_snapshot": {float(k): float(v) for k, v in errors.items()},
        "mean_rel_l2": mean_rel,
        "elapsed_sec": elapsed,
        "checkpoint": str(ckpt.relative_to(REPO)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--n-adam", type=int, default=10_000)
    p.add_argument("--n-lbfgs", type=int, default=5_000)
    p.add_argument("--lr-theta", type=float, default=1e-3)
    p.add_argument("--lr-lambda", type=float, default=5e-3)
    args = p.parse_args()

    device = select_device()
    print(f"Device: {device_info(device)}")
    print(f"Config: seeds={args.seeds} n_adam={args.n_adam} n_lbfgs={args.n_lbfgs}\n")

    print("Building Burgers reference & training batch…")
    ref = burg.compute_reference_solution()
    batch = burg.generate_training_data(ref, device=device)

    runs = []
    for s in args.seeds:
        runs.append(run_one_seed(
            s, batch, ref, device,
            args.n_adam, args.n_lbfgs,
            args.lr_theta, args.lr_lambda,
        ))

    means = [r["mean_rel_l2"] for r in runs]
    elapsed_total = [r["elapsed_sec"] for r in runs]
    summary = {
        "method": "sa-pinn",
        "config": vars(args),
        "runs": runs,
        "summary": {
            "n_seeds": len(runs),
            "rel_l2_mean": float(np.mean(means)),
            "rel_l2_std":  float(np.std(means)),
            "rel_l2_min":  float(np.min(means)),
            "rel_l2_max":  float(np.max(means)),
            "elapsed_mean_sec": float(np.mean(elapsed_total)),
            "elapsed_total_sec": float(np.sum(elapsed_total)),
        },
    }
    out = REPO / "results" / "sa_pinn_burgers.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out.relative_to(REPO)}")
    print(f"  rel-L2: {summary['summary']['rel_l2_mean']:.4e} ± {summary['summary']['rel_l2_std']:.4e}")
    print(f"  mean wall time per seed: {summary['summary']['elapsed_mean_sec']/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
