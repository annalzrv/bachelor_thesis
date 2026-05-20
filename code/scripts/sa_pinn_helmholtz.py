"""SA-PINN baseline on parametric Helmholtz.

For each (seed, k_train) we train a fresh SA-PINN at that fixed wavenumber
and evaluate against the manufactured reference. The headline comparison
to LC-PINN is at the centre wavenumber k=(K_MIN+K_MAX)/2; per-k breakdown
across a small grid lets us show how baselines perform across the family
without amortisation.

Usage:
    python scripts/sa_pinn_helmholtz.py --seeds 0 1 2 3 \\
        --k-trains 1.0 3.25 5.5 7.75 10.0 --n-adam 10000 --n-lbfgs 5000

Output:
    results/sa_pinn_helmholtz.json
    checkpoints/sa_pinn_helmholtz_seed{s}_k{k:.2f}.pt
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
from pinns.equations import helmholtz as helm


HIDDEN_DIMS = [64, 64, 64, 64]
REPO = pathlib.Path(__file__).resolve().parent.parent


def _per_point_residual_pde(model, coords, k_fixed):
    coords = coords.requires_grad_(True)
    u = model(coords)
    grads = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_xx = torch.autograd.grad(u_x, coords, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    f = helm.forcing(coords[:, 0:1], k_fixed)
    return u_xx + (k_fixed ** 2) * u - f


def _per_point_squared_error(model, coords, target):
    return (model(coords) - target) ** 2


def _sa_total_loss(model, lam_r, lam_b, batch, k_fixed, q=2):
    """SA loss adapted to elliptic Helmholtz: PDE + BC + data (no IC)."""
    res_pde_sq = _per_point_residual_pde(model, batch["coords_pde"], k_fixed) ** 2
    res_bc_sq  = _per_point_squared_error(model, batch["coords_bc"], batch["u_bc"])

    L_pde = 0.5 * ((lam_r ** q).unsqueeze(1) * res_pde_sq).mean()
    L_bc  = 0.5 * ((lam_b ** q).unsqueeze(1) * res_bc_sq).mean()

    # Data points filtered to k_fixed (matching helmholtz.compute_losses_fixed).
    mask = torch.isclose(
        batch["k_data"], torch.tensor(k_fixed, device=batch["k_data"].device)
    )
    if mask.any():
        coords_d = batch["coords_data"][mask]
        u_d = batch["u_data"][mask]
        L_dat = torch.mean((model(coords_d) - u_d) ** 2)
    else:
        L_dat = torch.zeros(1, device=L_pde.device).squeeze()

    return L_pde + L_bc + L_dat, {
        "pde": float(L_pde.item()), "bc": float(L_bc.item()), "data": float(L_dat.item()),
    }


def run_one(seed: int, k_train: float, batch, device, n_adam: int, n_lbfgs: int,
            lr_theta: float, lr_lambda: float) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)

    model = FixedWeightPINN(helm.DIM_PHYS, HIDDEN_DIMS).to(device)

    N_r = batch["coords_pde"].shape[0]
    N_b = batch["coords_bc"].shape[0]
    lam_r = nn.Parameter(torch.rand(N_r, device=device))
    lam_b = nn.Parameter(torch.rand(N_b, device=device))

    opt_theta  = torch.optim.Adam(model.parameters(), lr=lr_theta)
    opt_lambda = torch.optim.Adam([lam_r, lam_b], lr=lr_lambda, maximize=True)

    t0 = time.time()
    print(f"[seed {seed}, k={k_train:.2f}] Adam phase: {n_adam} steps")
    for step in range(n_adam):
        opt_theta.zero_grad()
        opt_lambda.zero_grad()
        total, parts = _sa_total_loss(model, lam_r, lam_b, batch, k_train)
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt_theta.step()
        opt_lambda.step()
        if step % max(1, n_adam // 10) == 0:
            print(f"  step {step:6d}  L={total.item():.4e}  parts={parts}")

    if n_lbfgs > 0:
        print(f"[seed {seed}, k={k_train:.2f}] L-BFGS phase: {n_lbfgs} steps (λ frozen)")
        lam_r.requires_grad_(False); lam_b.requires_grad_(False)
        lbfgs = torch.optim.LBFGS(
            model.parameters(), max_iter=20,
            tolerance_grad=1e-9, tolerance_change=1e-12,
            history_size=50, line_search_fn="strong_wolfe",
        )
        n_outer = max(1, n_lbfgs // 20)
        for outer in range(n_outer):
            def closure():
                lbfgs.zero_grad()
                total, _ = _sa_total_loss(model, lam_r, lam_b, batch, k_train)
                total.backward()
                return total
            loss_val = lbfgs.step(closure)
            if outer % 50 == 0:
                print(f"  outer {outer:4d}  L={float(loss_val):.4e}")

    elapsed = time.time() - t0

    rel_at_k_train = helm.evaluate_at_k(model, k_train, device, is_lc=False)
    ckpt = REPO / "checkpoints" / f"sa_pinn_helmholtz_seed{seed}_k{k_train:.2f}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "rel_l2": rel_at_k_train,
        "k_train": k_train, "elapsed_sec": elapsed, "seed": seed,
        "n_adam": n_adam, "n_lbfgs": n_lbfgs,
        "lr_theta": lr_theta, "lr_lambda": lr_lambda,
    }, ckpt)
    print(f"[seed {seed}, k={k_train:.2f}] done in {elapsed/60:.1f} min  rel-L2 = {rel_at_k_train:.4e}")
    return {
        "seed": seed, "k_train": k_train,
        "rel_l2": rel_at_k_train, "elapsed_sec": elapsed,
        "checkpoint": str(ckpt.relative_to(REPO)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--k-trains", type=float, nargs="+",
                   default=[1.0, 3.25, 5.5, 7.75, 10.0])
    p.add_argument("--n-adam", type=int, default=10_000)
    p.add_argument("--n-lbfgs", type=int, default=5_000)
    p.add_argument("--lr-theta", type=float, default=1e-3)
    p.add_argument("--lr-lambda", type=float, default=5e-3)
    args = p.parse_args()

    device = select_device()
    print(f"Device: {device_info(device)}")
    print(f"Config: seeds={args.seeds} k_trains={args.k_trains} "
          f"n_adam={args.n_adam} n_lbfgs={args.n_lbfgs}\n")

    print(f"Helmholtz family: k ∈ [{helm.K_MIN}, {helm.K_MAX}], domain [{helm.X_MIN}, {helm.X_MAX}]")
    batch = helm.generate_training_data(device=device)

    runs = []
    for s in args.seeds:
        for k in args.k_trains:
            runs.append(run_one(
                s, float(k), batch, device,
                args.n_adam, args.n_lbfgs, args.lr_theta, args.lr_lambda,
            ))

    # Aggregation: per-k mean and overall mean across the k-grid.
    per_k: dict[float, list[float]] = {}
    for r in runs:
        per_k.setdefault(float(r["k_train"]), []).append(r["rel_l2"])
    per_k_summary = {
        f"{k:.2f}": {
            "mean": float(np.mean(v)), "std": float(np.std(v)),
            "n_seeds": len(v),
        }
        for k, v in sorted(per_k.items())
    }
    grid_means_per_seed: dict[int, float] = {}
    for r in runs:
        grid_means_per_seed.setdefault(r["seed"], []).append(r["rel_l2"])
    seed_means = [float(np.mean(v)) for v in grid_means_per_seed.values()]
    elapsed = [r["elapsed_sec"] for r in runs]

    summary = {
        "method": "sa-pinn", "equation": "helmholtz",
        "config": vars(args), "runs": runs,
        "per_k": per_k_summary,
        "summary": {
            "n_seeds": len(args.seeds), "k_trains": list(map(float, args.k_trains)),
            "rel_l2_mean_over_k_then_seeds": float(np.mean(seed_means)),
            "rel_l2_std_over_seeds":         float(np.std(seed_means)),
            "elapsed_total_sec":             float(np.sum(elapsed)),
        },
    }
    out = REPO / "results" / "sa_pinn_helmholtz.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out.relative_to(REPO)}")
    print(f"  rel-L2 (mean over k-grid then over seeds): "
          f"{summary['summary']['rel_l2_mean_over_k_then_seeds']:.4e} ± "
          f"{summary['summary']['rel_l2_std_over_seeds']:.4e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
