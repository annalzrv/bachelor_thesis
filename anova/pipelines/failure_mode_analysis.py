"""Failure-mode localization: at specific k values, does the LC-PINN's
per-pixel error align with the spatial structure that the ANOVA pair-effect
heatmap identifies?

For 2D Helmholtz at fixed k, the analytic solution u_ref =
sin(pi x) sin(pi y) cos(kx) cos(ky) has known spatial structure. The LC-PINN
u_theta will deviate from u_ref; the deviation map has its own spatial
structure. We compare:

  1. LC-PINN per-pixel residual: |u_theta(x, y; k) - u_ref(x, y; k)|
  2. Reference solution magnitude: |u_ref(x, y; k)|
  3. Conditional pair-effect from HDMR at fixed k: u_{x, y}^{(k)}(x, y)

Claim: at high k where the LC-PINN per-k rel-L2 is largest (per
OVERNIGHT_RESULTS), the residual concentrates near the oscillation
maxima — exactly where the Sobol pair-effect heatmap has its largest
magnitude. This validates that the decomposition signals where the
LC-PINN has the most trouble.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO_ANOVA = _HERE.parent
_REPO_THESIS_CODE = _REPO_ANOVA.parent / "thesis" / "code"
sys.path.insert(0, str(_REPO_ANOVA))
sys.path.insert(0, str(_REPO_THESIS_CODE))

from lc_anova.pipelines.helmholtz_2d import (  # noqa
    load_lc_pinn, evaluate_lc_pinn_batch, sample_joint,
)
from lc_anova.core.joint_hdmr import JointHDMR  # noqa
from pinns.equations import helmholtz_2d as helm  # noqa

CK_DIR = _REPO_THESIS_CODE / "checkpoints"
RESULTS = _HERE / "results"
PLOT_DIR = RESULTS / "figures"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def make_pair_field_at_fixed_k(jh, k_value, n_grid=80, device=None):
    """Build the conditional pair-effect heatmap u_{x, y, k}(x, y; k_fixed)
    by evaluating the joint HDMR's cross-effect output on a meshgrid with k held fixed."""
    if device is None:
        device = jh.device
    xs = torch.linspace(0.0, 1.0, n_grid, dtype=torch.float32)
    ys = torch.linspace(0.0, 1.0, n_grid, dtype=torch.float32)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    xy_flat = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    k_norm_scalar = float(helm.k_to_norm(np.array([k_value], dtype=np.float32))[0])
    k_flat = torch.full((xy_flat.shape[0], 1), k_norm_scalar, dtype=torch.float32)
    # Cross-effect = parameter-coupled pairs (x, k) and (y, k); not the spatial (x, y) pair
    cross = jh.cross_effect(xy_flat, k_flat).reshape(n_grid, n_grid)
    return X.numpy(), Y.numpy(), cross


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CK_DIR / "lc_pinn_helmholtz_2d_seed0.pt"),
                    help="LC-PINN checkpoint to analyze. Default: base (no L-BFGS), where failure modes are visible.")
    ap.add_argument("--tag", default="basebaseline")
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck_path = Path(args.checkpoint)
    print(f"Loading LC-PINN: {ck_path.name}")
    model, _ = load_lc_pinn(str(ck_path), device)

    # Train a fresh JointHDMR (h=128, L=6, p3=300; fast — ~3 min on MPS)
    print("Training joint (x, y, k) HDMR for visualization (h=128, L=6, p3=300)")
    xy_tr, k_tr = sample_joint(30000, 42, device)
    u_tr = evaluate_lc_pinn_batch(model, xy_tr, k_tr)
    jh = JointHDMR(dim_x=2, dim_lambda=1, hidden=128, layers=3,
                   max_order=3, use_fourier=True, num_freqs=6)
    jh.fit(xy_tr, k_tr, u_tr, phase1_epochs=30, phase2_epochs=80,
           phase3_epochs=300, log_every=100)

    k_values = [1.0, 3.5, 7.0, 10.0]
    n_grid = 96

    # Compute everything on a meshgrid
    panels = []
    for k_value in k_values:
        # Build grid
        xs = np.linspace(0.0, 1.0, n_grid)
        ys = np.linspace(0.0, 1.0, n_grid)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        xy = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1).astype(np.float32)

        # Analytic reference
        u_ref = helm.reference_solution(xy[:, 0], xy[:, 1], k_value).reshape(n_grid, n_grid)

        # LC-PINN output
        with torch.no_grad():
            k_norm = float(helm.k_to_norm(np.array([k_value], dtype=np.float32))[0])
            xy_t = torch.tensor(xy, dtype=torch.float32, device=device)
            k_t = torch.full((xy.shape[0], 1), k_norm, dtype=torch.float32, device=device)
            u_lc = evaluate_lc_pinn_batch(model, xy_t, k_t).cpu().numpy().reshape(n_grid, n_grid)

        # Residual
        err = np.abs(u_lc - u_ref)
        # Pair-effect heatmap from HDMR (cross terms only)
        _, _, pair_field = make_pair_field_at_fixed_k(jh, k_value, n_grid=n_grid, device=device)
        pair_field = pair_field.cpu().numpy() if hasattr(pair_field, "cpu") else pair_field

        panels.append({
            "k": k_value,
            "u_ref": u_ref,
            "u_lc": u_lc,
            "err": err,
            "rel_l2": float(np.linalg.norm(u_lc - u_ref) / (np.linalg.norm(u_ref) + 1e-10)),
            "pair_field": pair_field,
        })

    # ---- Figure ----------------------------------------------------------
    n_rows = len(k_values)
    fig, axes = plt.subplots(n_rows, 3, figsize=(11, 2.6 * n_rows),
                              gridspec_kw={"hspace": 0.45, "wspace": 0.25})

    cmap_ref = "RdBu_r"
    cmap_err = "viridis"
    cmap_pair = "RdBu_r"

    for row, p in enumerate(panels):
        # Column 1: reference solution
        ax = axes[row, 0]
        v = np.max(np.abs(p["u_ref"]))
        im = ax.imshow(p["u_ref"].T, origin="lower", extent=[0, 1, 0, 1],
                       aspect="equal", cmap=cmap_ref, vmin=-v, vmax=v)
        ax.set_title(rf"$u_\mathrm{{ref}}(x, y; k={p['k']:.1f})$")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Column 2: LC-PINN error |u_theta - u_ref|
        ax = axes[row, 1]
        im = ax.imshow(p["err"].T, origin="lower", extent=[0, 1, 0, 1],
                       aspect="equal", cmap=cmap_err)
        ax.set_title(rf"$|u_\theta - u_\mathrm{{ref}}|$  (rel-L²={p['rel_l2']:.3f})")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Column 3: HDMR cross-effect heatmap at fixed k
        ax = axes[row, 2]
        v = np.max(np.abs(p["pair_field"]))
        im = ax.imshow(p["pair_field"].T, origin="lower", extent=[0, 1, 0, 1],
                       aspect="equal", cmap=cmap_pair, vmin=-v, vmax=v)
        ax.set_title(rf"cross-effect $u_{{x,k}} + u_{{y,k}}$  ($k={p['k']:.1f}$)")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Failure-mode localization: LC-PINN per-$k$ residual vs joint-HDMR cross-effect\n"
                 f"({ck_path.stem})",
                 fontsize=13, fontweight="bold", y=1.0)
    out = PLOT_DIR / f"failure_mode_localization_{args.tag}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nWrote {out}")

    # Quantitative correlation: at each k, correlate per-pixel err with |pair_field|
    print("\nCorrelation between LC-PINN per-pixel residual magnitude and |HDMR pair-effect|:")
    print(f"  {'k':>5} {'rel-L²':>10} {'corr(|err|, |pair|)':>22}")
    correlations = []
    for p in panels:
        err_flat = p["err"].flatten()
        pair_flat = np.abs(p["pair_field"]).flatten()
        r = np.corrcoef(err_flat, pair_flat)[0, 1]
        correlations.append({"k": p["k"], "rel_l2": p["rel_l2"], "corr": r})
        print(f"  {p['k']:>5.1f} {p['rel_l2']:>10.4f} {r:>22.4f}")

    out_json = RESULTS / f"failure_mode_correlations_{args.tag}.json"
    out_json.write_text(json.dumps(correlations, indent=2))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
