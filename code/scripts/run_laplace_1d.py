"""Runnable companion to notebooks/06_laplace_1d.ipynb.

Trains the first 5 Dirichlet-Laplacian eigenmodes via the advisor's variational
formulation (Rayleigh quotient + sequential orthogonality penalty), saves the
figures to results/ and the checkpoints to checkpoints/.

Usage (from thesis/code):
    python scripts/run_laplace_1d.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pinns.device import select_device, device_info
from pinns.equations import laplace_1d as lap

REPO = pathlib.Path(__file__).resolve().parent.parent

N_MODES = 5
N_EPOCHS = 5_000
LR = 1e-3
ALPHA_ORTH = 100.0
N_INTERIOR = 1024
HIDDEN_DIMS = [32, 32, 32]


def main() -> int:
    device = select_device()
    print(f"Device: {device_info(device)}")
    torch.manual_seed(0)
    np.random.seed(0)

    print(f"\nTraining first {N_MODES} Laplacian eigenmodes sequentially "
          f"(n_epochs={N_EPOCHS}, lr={LR}, alpha_orth={ALPHA_ORTH})…")
    models, histories = lap.train_first_n_modes(
        n_modes=N_MODES,
        device=device,
        hidden_dims=HIDDEN_DIMS,
        n_epochs=N_EPOCHS,
        lr=LR,
        alpha_orth=ALPHA_ORTH,
        n_interior=N_INTERIOR,
        log_every=500,
    )
    results = lap.evaluate_all(models, device, nx=500)

    print(f"\n{'k':>3} {'λ_true':>10} {'λ_hat':>10} {'|Δλ|/λ':>10} {'rel-L2 u':>10}")
    print("-" * 48)
    for r in results:
        print(f"{r['k']:>3} {r['lambda_true']:>10.4f} {r['lambda_hat']:>10.4f} "
              f"{r['lambda_rel_err']:>10.4f} {r['rel_l2']:>10.4f}")

    # --- Figure: modes
    nx = 500
    x_eval = np.linspace(lap.X_MIN, lap.X_MAX, nx)
    fig, axes = plt.subplots(1, N_MODES, figsize=(3.2 * N_MODES, 3), sharey=True)
    for k_idx, (m, r) in enumerate(zip(models, results)):
        k = k_idx + 1
        u_ref, _ = lap.reference_eigenmode(k, x_eval)
        u_pred = lap.predict_eigenmode(m, x_eval, device)
        u_pred_n = lap._l2_normalise(u_pred)
        if float(np.dot(u_pred_n, u_ref)) < 0.0:
            u_pred_n = -u_pred_n
        ax = axes[k_idx]
        ax.plot(x_eval, u_ref, "k-", lw=1.5, label="analytic")
        ax.plot(x_eval, u_pred_n, "r--", lw=1.3, label="PINN")
        ax.set_title(f"k={k},  rel-L2={r['rel_l2']:.4f}\n"
                     f"λ̂={r['lambda_hat']:.3f} (true {r['lambda_true']:.0f})")
        ax.set_xlabel("x"); ax.grid(alpha=0.3)
        if k_idx == 0:
            ax.legend(fontsize=8)
    fig.suptitle("1D Laplace eigenfunctions — variational PINN vs analytic", y=1.03)
    plt.tight_layout()
    out_modes = REPO / "results" / "fig_laplace_1d_modes.png"
    plt.savefig(out_modes, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_modes.relative_to(REPO)}")

    # --- Figure: Rayleigh convergence
    fig, ax = plt.subplots(figsize=(8, 4))
    for k_idx, h in enumerate(histories):
        k = k_idx + 1
        ax.plot(h["step"], h["lambda_hat"], label=f"k={k}, true λ={k**2}", lw=1.3)
        ax.axhline(k ** 2, ls=":", color=f"C{k_idx}", alpha=0.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"Rayleigh quotient $\hat\lambda$")
    ax.set_title("Eigenvalue estimate over training (sequential)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    out_ray = REPO / "results" / "fig_laplace_1d_rayleigh.png"
    plt.savefig(out_ray, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_ray.relative_to(REPO)}")

    # --- JSON summary
    summary = {
        "config": {
            "n_modes": N_MODES,
            "n_epochs": N_EPOCHS,
            "lr": LR,
            "alpha_orth": ALPHA_ORTH,
            "n_interior": N_INTERIOR,
            "hidden_dims": HIDDEN_DIMS,
        },
        "results": results,
    }
    out_json = REPO / "results" / "laplace_1d_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out_json.relative_to(REPO)}")

    for k_idx, m in enumerate(models):
        ckpt = REPO / "checkpoints" / f"laplace_1d_mode{k_idx+1}.pt"
        torch.save({"state_dict": m.state_dict(), "k": k_idx + 1}, ckpt)
    print(f"Saved checkpoints/laplace_1d_mode{{1..{N_MODES}}}.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
