"""Runnable companion for the LC-conditioned 1D eigenvalue PINN.

Trains ONE network u_θ(x, k) for the first K Dirichlet-Laplacian eigenmodes
via the Ky Fan variational principle (weighted-Rayleigh + pairwise cos²
orthogonality), saves figures + JSON + a single checkpoint.

Companion to `run_laplace_1d.py` (which trains K separate small networks
sequentially). Both scripts run on the same problem; outputs are named
differently so they can coexist in results/ and checkpoints/.

Usage (from thesis/code):
    python scripts/run_laplace_1d_lc.py --K 5 --n-epochs 20000
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

from pinns.device import select_device, device_info
from pinns.equations import laplace_1d as lap       # for reference eigenmode
from pinns.equations import laplace_1d_lc as lap_lc

REPO = pathlib.Path(__file__).resolve().parent.parent

HIDDEN_DIMS = [64, 64, 64, 64]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=5, help="number of modes")
    parser.add_argument("--n-epochs", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha-orth", type=float, default=100.0)
    parser.add_argument("--w-exp", type=float, default=1.0,
                        help="ordering weights w_k = 1/k^w_exp")
    parser.add_argument("--curriculum", action="store_true",
                        help="grow K_active from 1 to K across training")
    parser.add_argument("--n-interior", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = select_device()
    print(f"Device: {device_info(device)}")
    print(f"K={args.K}  n_epochs={args.n_epochs}  lr={args.lr}  "
          f"alpha_orth={args.alpha_orth}  w_exp={args.w_exp}  "
          f"curriculum={args.curriculum}  seed={args.seed}\n")

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    model = lap_lc.LCEigenmodeNet(K_max=args.K, hidden_dims=HIDDEN_DIMS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LCEigenmodeNet params: {n_params:,}")

    t0 = time.time()
    history = lap_lc.train_lc_eigenmode(
        model, K=args.K, device=device,
        n_epochs=args.n_epochs, lr=args.lr,
        alpha_orth=args.alpha_orth, n_interior=args.n_interior,
        w_exp=args.w_exp, log_every=500,
        curriculum=args.curriculum,
    )
    print(f"LC training done in {(time.time() - t0)/60:.1f} min")

    # --- Evaluation
    results = lap_lc.evaluate_all(model, args.K, device, nx=500)
    order = lap_lc.reorder_by_rayleigh(model, args.K, device, nx=500)

    print(f"\nOrdering check (slot → rank-by-Rayleigh): {order}  "
          f"(identity = {list(range(1, args.K + 1))})")
    identity = list(range(1, args.K + 1))
    if order != identity:
        print("  WARNING: slot k did not land on the k-th mode; Ky Fan weights "
              "didn't break symmetry this run.")
    print(f"\n{'k':>3} {'λ_true':>10} {'λ_hat':>10} {'|Δλ|/λ':>10} {'rel-L2 u':>10}")
    print("-" * 48)
    for r in results:
        print(f"{r['k']:>3} {r['lambda_true']:>10.4f} {r['lambda_hat']:>10.4f} "
              f"{r['lambda_rel_err']:>10.4f} {r['rel_l2']:>10.4f}")

    # --- Figure: modes
    nx = 500
    x_eval = np.linspace(lap.X_MIN, lap.X_MAX, nx)
    fig, axes = plt.subplots(1, args.K, figsize=(3.2 * args.K, 3), sharey=True)
    if args.K == 1:
        axes = [axes]
    for k_idx in range(args.K):
        k = k_idx + 1
        u_ref, _ = lap.reference_eigenmode(k, x_eval)
        u_pred = lap_lc.predict_eigenmode(model, k, args.K, x_eval, device)
        u_pred_n = lap_lc._l2_normalise(u_pred)
        if float(np.dot(u_pred_n, u_ref)) < 0.0:
            u_pred_n = -u_pred_n
        ax = axes[k_idx]
        r = results[k_idx]
        ax.plot(x_eval, u_ref, "k-", lw=1.5, label="analytic")
        ax.plot(x_eval, u_pred_n, "r--", lw=1.3, label="LC-PINN")
        ax.set_title(f"k={k},  rel-L2={r['rel_l2']:.4f}\n"
                     f"λ̂={r['lambda_hat']:.3f} (true {r['lambda_true']:.0f})")
        ax.set_xlabel("x"); ax.grid(alpha=0.3)
        if k_idx == 0:
            ax.legend(fontsize=8)
    fig.suptitle(
        f"1D Laplace eigenfunctions — LC-conditioned PINN (single network, K={args.K})",
        y=1.03,
    )
    plt.tight_layout()
    out_modes = REPO / "results" / "fig_laplace_1d_lc_modes.png"
    plt.savefig(out_modes, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_modes.relative_to(REPO)}")

    # --- Figure: Rayleigh convergence (per slot)
    ray_traj = np.array(history["lambda_hats"])  # (n_log, K)
    fig, ax = plt.subplots(figsize=(8, 4))
    for k_idx in range(args.K):
        k = k_idx + 1
        ax.plot(history["step"], ray_traj[:, k_idx],
                label=f"slot k={k}, true λ={k**2}", lw=1.3)
        ax.axhline(k ** 2, ls=":", color=f"C{k_idx}", alpha=0.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"Rayleigh quotient $\hat\lambda$")
    ax.set_title("Eigenvalue estimate over training — LC (single network)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    out_ray = REPO / "results" / "fig_laplace_1d_lc_rayleigh.png"
    plt.savefig(out_ray, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_ray.relative_to(REPO)}")

    # --- JSON summary
    summary = {
        "config": {
            "K": args.K,
            "n_epochs": args.n_epochs,
            "lr": args.lr,
            "alpha_orth": args.alpha_orth,
            "w_exp": args.w_exp,
            "n_interior": args.n_interior,
            "hidden_dims": HIDDEN_DIMS,
            "seed": args.seed,
            "n_params": n_params,
        },
        "order_by_rayleigh": order,
        "results": results,
    }
    out_json = REPO / "results" / "laplace_1d_lc_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out_json.relative_to(REPO)}")

    torch.save(
        {"state_dict": model.state_dict(), "K": args.K, "hidden_dims": HIDDEN_DIMS},
        REPO / "checkpoints" / "laplace_1d_lc.pt",
    )
    print("Saved checkpoints/laplace_1d_lc.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
