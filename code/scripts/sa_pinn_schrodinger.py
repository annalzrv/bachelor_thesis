"""SA-PINN baseline on 1D driven Schrödinger with parametric harmonic well.

Per (seed, α_train), trains a fresh SA-PINN at that fixed α.

Usage:
    python scripts/sa_pinn_schrodinger.py --seeds 0 1 \\
        --alpha-trains 0.5 5.0 10.0 --n-adam 5000 --n-lbfgs 2500

Output:
    results/sa_pinn_schrodinger.json
    checkpoints/sa_pinn_schrodinger_seed{s}_a{a:.2f}.pt
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
from pinns.equations import schrodinger_1d as sch


HIDDEN_DIMS = [64, 64, 64, 64]
REPO = pathlib.Path(__file__).resolve().parent.parent


def _residual_pde(model, coords, a_fixed):
    coords = coords.requires_grad_(True)
    u = model(coords)
    g = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(g, coords, torch.ones_like(g), create_graph=True)[0]
    V = sch.potential(coords, a_fixed)
    f = sch.forcing(coords, a_fixed)
    return -uxx + V * u - f


def _sq_err(model, coords, target):
    return (model(coords) - target) ** 2


def _sa_total(model, lam_r, lam_b, batch, a_fixed, q=2):
    res_pde = _residual_pde(model, batch["coords_pde"], a_fixed) ** 2
    res_bc = _sq_err(model, batch["coords_bc"], batch["u_bc"])
    L_pde = 0.5 * ((lam_r ** q).unsqueeze(1) * res_pde).mean()
    L_bc = 0.5 * ((lam_b ** q).unsqueeze(1) * res_bc).mean()
    mask = torch.isclose(batch["k_data"], torch.tensor(a_fixed, device=batch["k_data"].device))
    if mask.any():
        L_dat = torch.mean((model(batch["coords_data"][mask]) - batch["u_data"][mask]) ** 2)
    else:
        L_dat = torch.zeros(1, device=L_pde.device).squeeze()
    return L_pde + L_bc + L_dat, {
        "pde": float(L_pde.item()), "bc": float(L_bc.item()), "data": float(L_dat.item())
    }


def run_one(seed, a_train, batch, device, n_adam, n_lbfgs, lr_theta, lr_lambda):
    torch.manual_seed(seed); np.random.seed(seed)
    model = FixedWeightPINN(sch.DIM_PHYS, HIDDEN_DIMS).to(device)

    N_r = batch["coords_pde"].shape[0]
    N_b = batch["coords_bc"].shape[0]
    lam_r = nn.Parameter(torch.rand(N_r, device=device))
    lam_b = nn.Parameter(torch.rand(N_b, device=device))

    opt_th = torch.optim.Adam(model.parameters(), lr=lr_theta)
    opt_la = torch.optim.Adam([lam_r, lam_b], lr=lr_lambda, maximize=True)

    t0 = time.time()
    print(f"[seed {seed}, α={a_train:.2f}] Adam phase: {n_adam} steps")
    for step in range(n_adam):
        opt_th.zero_grad(); opt_la.zero_grad()
        total, parts = _sa_total(model, lam_r, lam_b, batch, a_train)
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt_th.step(); opt_la.step()
        if step % max(1, n_adam // 10) == 0:
            print(f"  step {step:6d}  L={total.item():.4e}  parts={parts}")

    if n_lbfgs > 0:
        print(f"[seed {seed}, α={a_train:.2f}] L-BFGS phase: {n_lbfgs} steps (λ frozen)")
        lam_r.requires_grad_(False); lam_b.requires_grad_(False)
        lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=20,
                                  tolerance_grad=1e-9, tolerance_change=1e-12,
                                  history_size=50, line_search_fn="strong_wolfe")
        n_outer = max(1, n_lbfgs // 20)
        for outer in range(n_outer):
            def closure():
                lbfgs.zero_grad()
                total, _ = _sa_total(model, lam_r, lam_b, batch, a_train)
                total.backward()
                return total
            loss_val = lbfgs.step(closure)
            if outer % 50 == 0:
                print(f"  outer {outer:4d}  L={float(loss_val):.4e}")

    elapsed = time.time() - t0
    rel = sch.evaluate_at_alpha(model, a_train, device, is_lc=False)

    ckpt = REPO / "checkpoints" / f"sa_pinn_schrodinger_seed{seed}_a{a_train:.2f}.pt"
    torch.save({"model_state_dict": model.state_dict(), "rel_l2": rel,
                "alpha_train": a_train, "elapsed_sec": elapsed, "seed": seed,
                "n_adam": n_adam, "n_lbfgs": n_lbfgs}, ckpt)
    print(f"[seed {seed}, α={a_train:.2f}] done in {elapsed/60:.1f} min  rel-L2 = {rel:.4e}")
    return {"seed": seed, "alpha_train": a_train, "rel_l2": rel,
            "elapsed_sec": elapsed, "checkpoint": str(ckpt.relative_to(REPO))}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--alpha-trains", type=float, nargs="+",
                   default=[0.5, 5.0, 10.0])
    p.add_argument("--n-adam", type=int, default=5_000)
    p.add_argument("--n-lbfgs", type=int, default=2_500)
    p.add_argument("--lr-theta", type=float, default=1e-3)
    p.add_argument("--lr-lambda", type=float, default=5e-3)
    args = p.parse_args()

    device = select_device()
    print(f"Device: {device_info(device)}")
    print(f"Config: seeds={args.seeds} alpha_trains={args.alpha_trains} "
          f"n_adam={args.n_adam} n_lbfgs={args.n_lbfgs}\n")
    print(f"Schrödinger: α ∈ [{sch.ALPHA_MIN}, {sch.ALPHA_MAX}]")
    batch = sch.generate_training_data(device=device)

    runs = []
    for s in args.seeds:
        for a in args.alpha_trains:
            runs.append(run_one(s, float(a), batch, device,
                                args.n_adam, args.n_lbfgs,
                                args.lr_theta, args.lr_lambda))

    per_a = {}
    for r in runs:
        per_a.setdefault(float(r["alpha_train"]), []).append(r["rel_l2"])
    per_a_summary = {f"{a:.2f}": {"mean": float(np.mean(v)), "std": float(np.std(v)), "n_seeds": len(v)}
                     for a, v in sorted(per_a.items())}
    seed_means = []
    grouped = {}
    for r in runs:
        grouped.setdefault(r["seed"], []).append(r["rel_l2"])
    seed_means = [float(np.mean(v)) for v in grouped.values()]
    elapsed = [r["elapsed_sec"] for r in runs]

    summary = {
        "method": "sa-pinn", "equation": "schrodinger_1d",
        "config": vars(args), "runs": runs, "per_alpha": per_a_summary,
        "summary": {
            "n_seeds": len(args.seeds), "alpha_trains": list(map(float, args.alpha_trains)),
            "rel_l2_mean_over_alpha_then_seeds": float(np.mean(seed_means)),
            "rel_l2_std_over_seeds": float(np.std(seed_means)),
            "elapsed_total_sec": float(np.sum(elapsed)),
        },
    }
    out = REPO / "results" / "sa_pinn_schrodinger.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out.relative_to(REPO)}")
    print(f"  rel-L2: {summary['summary']['rel_l2_mean_over_alpha_then_seeds']:.4e} ± "
          f"{summary['summary']['rel_l2_std_over_seeds']:.4e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
