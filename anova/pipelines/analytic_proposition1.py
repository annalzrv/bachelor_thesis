"""Proposition 1 — analytic per-k Sobol indices for the 2D Helmholtz
manufactured solution, overlaid on the MC-Sobol measurements from
per_k_sobol.py.

Manufactured solution:
    u(x, y; k) = sin(pi x) sin(pi y) cos(k x) cos(k y)   on [0, 1]^2

The function factors as u = f(x; k) * g(y; k) with
    f(x; k) = sin(pi x) cos(k x).

Under uniform (x, y) ~ U[0, 1]^2 and any fixed k, the ANOVA
decomposition has closed-form variances:

    A_k = E_x[f] = (2 pi cos^2(k/2)) / (pi^2 - k^2)
    B_k = E_x[f^2] = 1/4 - (pi^2 sin(2k)) / (8 k (k^2 - pi^2))
    V_k = B_k - A_k^2

Spatial Sobol indices (closed form):

    S_x^(k) = S_y^(k) = A_k^2 / (B_k + A_k^2)
    S_xy^(k) = (B_k - A_k^2) / (B_k + A_k^2)

This script: (1) evaluates the closed forms at the 11 k values used by
per_k_sobol.py; (2) overlays the analytic curves on the MC-measured
curves from both seed-0 and seed-2; (3) reports the per-k discrepancy
between analytic and empirical.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
PLOT_DIR = RESULTS / "figures"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def A_k(k):
    """Closed form for the marginal mean of f(x; k) = sin(pi x) cos(k x) on [0, 1]."""
    return 2.0 * np.pi * np.cos(k / 2.0) ** 2 / (np.pi ** 2 - k ** 2)


def B_k(k):
    """Closed form for E[f^2]."""
    return 0.25 - (np.pi ** 2 * np.sin(2 * k)) / (8.0 * k * (k ** 2 - np.pi ** 2))


def sobol_xy_analytic(k):
    """S_xy^(k) under uniform (x, y) ~ U[0, 1]^2 with k held fixed."""
    a = A_k(k)
    b = B_k(k)
    return (b - a ** 2) / (b + a ** 2)


def sobol_x_analytic(k):
    """S_x^(k) = S_y^(k) under the same setup."""
    a = A_k(k)
    b = B_k(k)
    return a ** 2 / (b + a ** 2)


def main():
    # k-grid matching what per_k_sobol.py used
    k_grid = np.linspace(1.0, 10.0, 11)

    S_xy_analytic = np.array([sobol_xy_analytic(k) for k in k_grid])
    S_x_analytic = np.array([sobol_x_analytic(k) for k in k_grid])

    # Pull empirical measurements
    empirical = {}
    for path in sorted(RESULTS.glob("results_per_k_helm2d_seed*.json")):
        if "hires" in path.stem:
            continue
        data = json.loads(path.read_text())
        seed = path.stem.split("seed")[-1]
        empirical[seed] = {
            "k": np.array(data["k_values"]),
            "S_x": np.array(data["S_x_at_k"]),
            "S_y": np.array(data["S_y_at_k"]),
            "S_xy": np.array(data["S_xy_at_k"]),
        }

    print("Analytic per-k Sobol indices for u = sin(pi x) sin(pi y) cos(kx) cos(ky):")
    print(f"  {'k':>5} {'S_x analytic':>14} {'S_y analytic':>14} {'S_xy analytic':>14}")
    for k, sxy, sx in zip(k_grid, S_xy_analytic, S_x_analytic):
        print(f"  {k:>5.2f} {sx:>14.4f} {sx:>14.4f} {sxy:>14.4f}")

    print("\nDiscrepancy (analytic - empirical, averaged over seeds):")
    if empirical:
        S_xy_emp = np.mean([empirical[s]["S_xy"] for s in empirical], axis=0)
        S_x_emp = np.mean([empirical[s]["S_x"] for s in empirical], axis=0)
        S_y_emp = np.mean([empirical[s]["S_y"] for s in empirical], axis=0)
        diff_xy = S_xy_analytic - S_xy_emp
        diff_x = S_x_analytic - S_x_emp
        print(f"  {'k':>5} {'Δ S_x':>10} {'Δ S_y':>10} {'Δ S_xy':>10}")
        for i, k in enumerate(k_grid):
            print(f"  {k:>5.2f} {diff_x[i]:>10.4f} {S_x_analytic[i] - S_y_emp[i]:>10.4f} {diff_xy[i]:>10.4f}")
        max_abs = max(np.abs(diff_xy).max(), np.abs(diff_x).max())
        print(f"\n  Max |analytic − empirical| = {max_abs:.4f}")

    # ---- Headline figure ----
    fig, ax = plt.subplots(figsize=(8, 4.8))

    # Analytic curves (solid, thick)
    ax.plot(k_grid, S_x_analytic, "-", color="C0", linewidth=2.5,
            label=r"$S_x^{(k)} = S_y^{(k)}$ analytic", zorder=3)
    ax.plot(k_grid, S_xy_analytic, "-", color="C2", linewidth=2.5,
            label=r"$S_{x,y}^{(k)}$ analytic", zorder=3)

    # Empirical points (one marker style per seed)
    markers = ["o", "s", "D", "^"]
    for i, seed in enumerate(sorted(empirical.keys())):
        m = markers[i % len(markers)]
        e = empirical[seed]
        ax.plot(e["k"], e["S_x"], m, color="C0", alpha=0.55,
                markersize=7, label=rf"$S_x$ LC-PINN seed {seed}", zorder=2)
        ax.plot(e["k"], e["S_y"], m, color="C1", alpha=0.55,
                markersize=5, label=rf"$S_y$ LC-PINN seed {seed}", zorder=2)
        ax.plot(e["k"], e["S_xy"], m, color="C2", alpha=0.55,
                markersize=7, label=rf"$S_{{x,y}}$ LC-PINN seed {seed}", zorder=2)

    ax.set_xlabel(r"wavenumber  $k$")
    ax.set_ylabel(r"conditional spatial Sobol index  $S_{\cdot}^{(k)}$")
    ax.set_title(r"Per-$k$ Sobol: closed form (Prop. 1) vs measurement on LC-PINN")
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlim(0.8, 10.2)

    # Manual deduplicated legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        key = l.split("seed")[0].strip().rstrip(",")
        if key not in seen:
            seen.add(key)
            h2.append(h)
            l2.append(l.split("seed")[0].strip().rstrip(",") + (" (seeds 0,2)" if "LC-PINN" in l else ""))
    ax.legend(h2, l2, loc="center right", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    out_path = PLOT_DIR / "proposition1_analytic_vs_empirical.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nWrote {out_path}")

    # Save the per-k table as JSON for the paper
    table = {
        "k_values": list(k_grid),
        "S_x_analytic": list(S_x_analytic),
        "S_xy_analytic": list(S_xy_analytic),
        "A_k": [float(A_k(k)) for k in k_grid],
        "B_k": [float(B_k(k)) for k in k_grid],
    }
    if empirical:
        table["S_x_empirical_mean"] = list(np.mean([empirical[s]["S_x"] for s in empirical], axis=0))
        table["S_xy_empirical_mean"] = list(np.mean([empirical[s]["S_xy"] for s in empirical], axis=0))
        table["max_abs_diff_xy"] = float(np.abs(np.array(table["S_xy_analytic"]) - np.array(table["S_xy_empirical_mean"])).max())
    json_path = RESULTS / "proposition1_analytic.json"
    json_path.write_text(json.dumps(table, indent=2))
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
