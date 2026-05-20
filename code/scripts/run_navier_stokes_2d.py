"""Runnable companion to notebooks/07_navier_stokes_2d.ipynb.

Trains an LC-PINN (uniform λ sampling) and an equal-weights baseline on the
2D Taylor-Green vortex, runs a FAIR λ sweep (exclude_terms=set()), evaluates
rel-L2 on (u, v) per snapshot, saves the comparison figure + JSON summary.

Usage (from thesis/code):
    python scripts/run_navier_stokes_2d.py --n-steps 150000 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pinns.baseline import FixedWeightPINN, train_fixed_pinn
from pinns.device import select_device, device_info
from pinns.equations import navier_stokes_2d as ns
from pinns.inference import sweep_lambda
from pinns.lambda_sampler import LambdaSampler
from pinns.model import LossConditionalPINN
from pinns.training import train_lc_pinn

REPO = pathlib.Path(__file__).resolve().parent.parent
HIDDEN_DIMS = [64, 64, 64, 64]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=150_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-pde", type=int, default=4000)
    parser.add_argument("--n-bc", type=int, default=400)
    parser.add_argument("--n-ic", type=int, default=400)
    parser.add_argument("--n-data", type=int, default=200)
    parser.add_argument("--sweep-candidates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = select_device()
    print(f"Device: {device_info(device)}")
    print(f"Config: n_steps={args.n_steps}  lr={args.lr}  seed={args.seed}\n")

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("Building reference + training batch…")
    ref = ns.compute_reference_solution(nx=64, ny=64, snap_times=[0.0, 0.25, 0.5, 1.0])
    batch = ns.generate_training_data(
        n_pde=args.n_pde, n_bc=args.n_bc, n_ic=args.n_ic, n_data=args.n_data,
        seed=args.seed, device=device,
    )

    # --- LC-PINN (uniform)
    print("\n=== LC-PINN (uniform λ) ===")
    torch.manual_seed(args.seed)
    model_lc = LossConditionalPINN(
        dim_phys=ns.DIM_PHYS, dim_lambda=ns.DIM_LAMBDA,
        hidden_dims=HIDDEN_DIMS, dim_out=ns.DIM_OUT,
    ).to(device)
    sampler = LambdaSampler(dim=ns.DIM_LAMBDA, device=device, mode="uniform")

    t0 = time.time()
    history_lc = train_lc_pinn(
        model_lc, sampler, batch, device,
        loss_fn=ns.compute_losses,
        n_epochs=args.n_steps, lr=args.lr, log_every=2_500,
    )
    print(f"LC-PINN training done in {(time.time() - t0)/60:.1f} min")

    # --- FAIR sweep
    print("\n=== FAIR λ sweep ===")
    torch.manual_seed(args.seed)
    best_ll, best_p, _ = sweep_lambda(
        model_lc, batch, sampler, device,
        loss_fn=ns.compute_losses,
        n_candidates=args.sweep_candidates, exclude_terms=set(),
    )
    best_p_np = best_p.cpu().numpy().round(4)
    print(f"Best λ (fair): {best_p_np}")

    lc_errs = ns.evaluate(model_lc, best_ll, ref, device)
    print("\nLC-PINN per-snapshot rel-L2:")
    for t_val, d in lc_errs.items():
        print(f"  t={t_val:.2f}  u={d['u']:.4f}  v={d['v']:.4f}")
    lc_mean = ns.mean_rel_l2(lc_errs)
    print(f"\nLC-PINN mean rel-L2 (u,v): {lc_mean:.4f}")

    torch.save(
        {"model_state_dict": model_lc.state_dict(), "best_log_lambda": best_ll},
        REPO / "checkpoints" / "ns_lc_pinn_uniform.pt",
    )

    # --- Equal-weights baseline
    print("\n=== Equal-weights baseline ===")
    torch.manual_seed(args.seed)
    model_bl = FixedWeightPINN(
        dim_phys=ns.DIM_PHYS, hidden_dims=HIDDEN_DIMS, dim_out=ns.DIM_OUT,
    ).to(device)
    t0 = time.time()
    history_bl = train_fixed_pinn(
        model_bl, weights=[0.25, 0.25, 0.25, 0.25],
        batch=batch, device=device,
        loss_fn=ns.compute_losses_fixed,
        n_epochs=args.n_steps, lr=args.lr, log_every=2_500,
        desc="NS equal-weights baseline",
    )
    print(f"Baseline training done in {(time.time() - t0)/60:.1f} min")

    bl_errs = ns.evaluate(model_bl, None, ref, device)
    print("\nBaseline per-snapshot rel-L2:")
    for t_val, d in bl_errs.items():
        print(f"  t={t_val:.2f}  u={d['u']:.4f}  v={d['v']:.4f}")
    bl_mean = ns.mean_rel_l2(bl_errs)
    print(f"\nBaseline mean rel-L2 (u,v): {bl_mean:.4f}")
    torch.save({"model_state_dict": model_bl.state_dict()},
               REPO / "checkpoints" / "ns_baseline_equal.pt")

    # --- Figure
    snap_times = sorted(ref.keys())
    fig, axes = plt.subplots(3, len(snap_times),
                              figsize=(3.2 * len(snap_times), 9),
                              sharex=True, sharey=True)
    for col, t_val in enumerate(snap_times):
        X, Y, u_ref, v_ref, _ = ref[t_val]
        u_lc, _, _ = ns.predict_uvp(model_lc, best_ll, X, Y, t_val, device)
        u_bl, _, _ = ns.predict_uvp(model_bl, None,    X, Y, t_val, device)
        vmin, vmax = float(u_ref.min()), float(u_ref.max())
        kw = dict(cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower",
                  extent=[ns.X_MIN, ns.X_MAX, ns.Y_MIN, ns.Y_MAX], aspect="equal")
        axes[0, col].imshow(u_ref.T, **kw); axes[0, col].set_title(f"Analytic  t={t_val}")
        axes[1, col].imshow(u_lc.T,  **kw); axes[1, col].set_title(f"LC-PINN  t={t_val}")
        axes[2, col].imshow(u_bl.T,  **kw); axes[2, col].set_title(f"Baseline  t={t_val}")
    for ax in axes[:, 0]: ax.set_ylabel("y")
    for ax in axes[-1]:   ax.set_xlabel("x")
    fig.suptitle("Taylor-Green vortex — u-component", y=1.01)
    plt.tight_layout()
    out_png = REPO / "results" / "fig_ns_taylor_green_u.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_png.relative_to(REPO)}")

    # --- JSON
    summary = {
        "config": {
            "nu": ns.NU,
            "n_steps": args.n_steps, "lr": args.lr, "hidden_dims": HIDDEN_DIMS,
            "n_pde": args.n_pde, "n_bc": args.n_bc, "n_ic": args.n_ic, "n_data": args.n_data,
        },
        "lc_pinn_uniform": {
            "best_lambda": best_p_np.tolist(),
            "per_snapshot": {str(k): v for k, v in lc_errs.items()},
            "mean_rel_l2": lc_mean,
        },
        "baseline_equal": {
            "per_snapshot": {str(k): v for k, v in bl_errs.items()},
            "mean_rel_l2": bl_mean,
        },
    }
    out_json = REPO / "results" / "ns_taylor_green_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out_json.relative_to(REPO)}")
    print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
