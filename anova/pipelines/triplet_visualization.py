"""Visualize the order-3 ANOVA triplet term f_{x, y, k}(x, y; k_0) at fixed k_0
and overlay against the closed-form analytic triplet of the manufactured
Helmholtz solution.

Replaces the misleading 'failure-mode localization' figure with the
quantity the paper actually claims dominates: the irreducible triplet.

Closed form (derived in proposition2.md):

  f_{xyk}(x, y; k_0) = tilde_a(x, k_0) * tilde_b(y, k_0) - C(x, y)

where
  tilde_a(x, k) = sin(pi x) cos(k x) - A_k
  tilde_b(y, k) = sin(pi y) cos(k y) - A_k
  A_k = (2 pi cos^2(k/2)) / (pi^2 - k^2)
  C(x, y) = E_{k' ~ p(k)} [tilde_a(x, k') tilde_b(y, k')]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO_ANOVA = _HERE.parent
_REPO_THESIS_CODE = _REPO_ANOVA.parent / "thesis" / "code"
sys.path.insert(0, str(_REPO_ANOVA))
sys.path.insert(0, str(_REPO_THESIS_CODE))

from lc_anova.pipelines.helmholtz_2d import load_lc_pinn, evaluate_lc_pinn_batch, sample_joint  # noqa
from lc_anova.core.joint_hdmr import JointHDMR  # noqa
from pinns.equations import helmholtz_2d as helm  # noqa

CK_DIR = _REPO_THESIS_CODE / "checkpoints"
RESULTS = _HERE / "results"
PAPER_FIGS = RESULTS / "paper_figures"
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 11,
    "legend.fontsize": 9.5, "figure.titlesize": 13,
})


def A_k(k):
    """Closed form: int_0^1 sin(pi x) cos(k x) dx = (2pi cos^2(k/2)) / (pi^2 - k^2)."""
    return 2.0 * np.pi * np.cos(k / 2.0) ** 2 / (np.pi ** 2 - k ** 2)


def analytic_triplet_slice(k0, n_grid=96, k_grid_size=200):
    """Analytic f_{xyk}(x, y; k0) on an n_grid x n_grid mesh.

    C(x, y) is computed via Gauss-Legendre over k' in [K_MIN, K_MAX].
    """
    xs = np.linspace(helm.X_MIN, helm.X_MAX, n_grid)
    ys = np.linspace(helm.Y_MIN, helm.Y_MAX, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    # tilde_a(x, k0) = sin(pi x) cos(k0 x) - A_{k0}, and same for tilde_b on y axis
    A_k0 = A_k(k0)
    tilde_a_k0 = np.sin(np.pi * X) * np.cos(k0 * X) - A_k0
    tilde_b_k0 = np.sin(np.pi * Y) * np.cos(k0 * Y) - A_k0
    spatial_pair_at_k0 = tilde_a_k0 * tilde_b_k0   # (n, n)

    # C(x, y) = E_{k' ~ U[K_MIN, K_MAX]} [tilde_a(x, k') tilde_b(y, k')]
    # Riemann-sum approximation with a fine k' grid.
    k_primes = np.linspace(helm.K_MIN, helm.K_MAX, k_grid_size)
    C_xy = np.zeros_like(spatial_pair_at_k0)
    for kp in k_primes:
        A_kp = A_k(kp)
        ta = np.sin(np.pi * X) * np.cos(kp * X) - A_kp
        tb = np.sin(np.pi * Y) * np.cos(kp * Y) - A_kp
        C_xy += ta * tb
    C_xy /= k_grid_size

    triplet = spatial_pair_at_k0 - C_xy
    return X, Y, triplet


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck_path = CK_DIR / "lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt"
    print(f"Loading LC-PINN: {ck_path.name}")
    model, _ = load_lc_pinn(str(ck_path), device)

    # Train a well-tuned Fourier joint HDMR (~5 min on MPS).
    print("Training joint Fourier HDMR (h=128, L=6, phase3=500)")
    torch.manual_seed(7); np.random.seed(7)
    xy_tr, k_tr = sample_joint(30000, 42, device)
    u_tr = evaluate_lc_pinn_batch(model, xy_tr, k_tr)
    jh = JointHDMR(dim_x=2, dim_lambda=1, hidden=128, layers=3,
                   max_order=3, use_fourier=True, num_freqs=6)
    jh.fit(xy_tr, k_tr, u_tr, phase1_epochs=30, phase2_epochs=80,
           phase3_epochs=500, log_every=100)

    # Evaluate the trained triplet net at fixed k_0 values on a 2D grid.
    # We use a single batch combining the visualization grid + a diverse
    # auxiliary set so the binning-purification within evaluate_terms has
    # enough k-diversity.
    k_values = [1.0, 3.5, 7.0, 10.0]
    n_grid = 96

    # Auxiliary batch with diverse k to make purification bins well-defined
    n_aux = 5000
    xy_aux = np.random.uniform(0.0, 1.0, size=(n_aux, 2)).astype(np.float32)
    k_aux = np.random.uniform(-1.0, 1.0, size=(n_aux, 1)).astype(np.float32)

    panels = []
    for k_value in k_values:
        # Empirical (trained HDMR triplet at fixed k_value)
        xs = np.linspace(0.0, 1.0, n_grid)
        ys = np.linspace(0.0, 1.0, n_grid)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        xy_grid = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1).astype(np.float32)
        k_norm_scalar = float(helm.k_to_norm(np.array([k_value], dtype=np.float32))[0])
        k_grid_col = np.full((xy_grid.shape[0], 1), k_norm_scalar, dtype=np.float32)

        # Combine with aux batch so purification has k-diversity
        xy_batch = np.concatenate([xy_grid, xy_aux], axis=0)
        k_batch = np.concatenate([k_grid_col, k_aux], axis=0)
        xy_t = torch.tensor(xy_batch, device=device)
        k_t = torch.tensor(k_batch, device=device)
        z_t = torch.cat([xy_t, k_t], dim=1)

        jh.model.eval()
        with torch.no_grad():
            _, _, _, t = jh.model.evaluate_terms(z_t, include_triplet=True)
        # Extract just the visualization-grid portion
        triplet_emp = t[:xy_grid.shape[0]].detach().cpu().numpy().reshape(n_grid, n_grid)

        # Analytic
        _, _, triplet_ana = analytic_triplet_slice(k_value, n_grid=n_grid)

        # Statistics: correlation, ratio of magnitudes
        emp_flat = triplet_emp.flatten()
        ana_flat = triplet_ana.flatten()
        corr = float(np.corrcoef(emp_flat, ana_flat)[0, 1])
        emp_std = float(emp_flat.std())
        ana_std = float(ana_flat.std())
        panels.append({
            "k": k_value,
            "triplet_emp": triplet_emp,
            "triplet_ana": triplet_ana,
            "corr": corr,
            "emp_std": emp_std,
            "ana_std": ana_std,
        })
        print(f"  k={k_value:5.1f}: corr={corr:+.4f}  std_emp={emp_std:.3f}  std_ana={ana_std:.3f}")

    # ---- Figure: 4 rows × 2 cols (empirical vs analytic) -----------------
    fig, axes = plt.subplots(len(k_values), 2, figsize=(8.5, 2.5 * len(k_values)),
                              gridspec_kw={"hspace": 0.5, "wspace": 0.30})
    cmap = "RdBu_r"

    for row, p in enumerate(panels):
        vmax = max(np.abs(p["triplet_ana"]).max(), np.abs(p["triplet_emp"]).max())
        for col, (data, label) in enumerate([
            (p["triplet_emp"], "empirical (trained HDMR)"),
            (p["triplet_ana"], "analytic (Prop. 2)")
        ]):
            ax = axes[row, col]
            im = ax.imshow(data.T, origin="lower", extent=[0, 1, 0, 1],
                           aspect="equal", cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.set_xlabel("x"); ax.set_ylabel("y")
            title = rf"$f_{{x,y,k}}(x, y; k={p['k']:.1f})$  {label}"
            if col == 0:
                title += rf"  $\sigma={p['emp_std']:.3f}$"
            else:
                title += rf"  $\sigma={p['ana_std']:.3f}$"
            ax.set_title(title, fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # Row-level annotation: correlation
        fig.text(0.50, axes[row, 0].get_position().y0 + 0.005,
                 rf"corr$($emp$,$ ana$) = {p['corr']:+.3f}$",
                 ha="center", fontsize=10, fontweight="bold", color="#404040",
                 bbox=dict(facecolor="white", edgecolor="none", pad=2))

    plt.suptitle(r"Triplet $f_{x, y, k}(x, y; k_0)$ — trained HDMR vs analytic closed form",
                  fontsize=13, fontweight="bold", y=0.995)
    out = PAPER_FIGS / "fig_triplet_visualization.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nWrote {out} + .pdf")

    # Save numbers
    out_json = RESULTS / "triplet_visualization_correlations.json"
    out_json.write_text(json.dumps([
        {"k": p["k"], "correlation_emp_vs_ana": p["corr"],
         "emp_std": p["emp_std"], "ana_std": p["ana_std"]}
        for p in panels
    ], indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
