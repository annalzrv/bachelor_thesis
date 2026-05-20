"""Side-by-side comparison: per-k Sobol on 2D Helmholtz vs per-alpha
variance on Schrödinger. The two parametric PDEs have qualitatively
different ANOVA structures, and this figure makes that visible.

2D Helm: pair S_{x, y}(k) sweeps from ~0.10 at k=1 to ~1.00 at k=10.
Schrödinger: spatial main S_x stays near 1.0 across all alpha;
the conditional variance Var(u | alpha) varies, but the *structure*
of the variance (mostly in x) is preserved.

Built from existing JSON outputs of per_k_sobol.py and
per_alpha_hires_schr1d_seed*.json from phase 5/8.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
PLOT_DIR = RESULTS / "figures"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # 2D Helmholtz per-k (existing data)
    helm_seeds = {}
    for path in sorted(RESULTS.glob("results_per_k_helm2d_seed*.json")):
        if "hires" in path.stem:
            continue
        d = json.loads(path.read_text())
        seed = path.stem.split("seed")[-1]
        helm_seeds[seed] = {
            "k": np.array(d["k_values"]),
            "S_x": np.array(d["S_x_at_k"]),
            "S_y": np.array(d["S_y_at_k"]),
            "S_xy": np.array(d["S_xy_at_k"]),
            "var_Y": np.array(d["var_Y_at_k"]),
        }

    # Schrödinger per-alpha (existing data)
    schr_seeds = {}
    for path in sorted(RESULTS.glob("per_alpha_hires_schr1d_seed*.json")):
        d = json.loads(path.read_text())
        seed = path.stem.split("seed")[-1]
        schr_seeds[seed] = {
            "alpha": np.array(d["alpha_values"]),
            "var_Y": np.array(d["var_Y"]),
        }

    # ---- Figure: 2 rows × 2 cols ----------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 7),
                              gridspec_kw={"hspace": 0.35, "wspace": 0.28})

    # Row 1: 2D Helmholtz
    ax = axes[0, 0]
    for seed, d in helm_seeds.items():
        ax.plot(d["k"], d["S_x"], "o-", alpha=0.65, color="C0",
                markersize=5, label=r"$S_x^{(k)}$" if seed == sorted(helm_seeds)[0] else None)
        ax.plot(d["k"], d["S_y"], "s-", alpha=0.65, color="C1",
                markersize=5, label=r"$S_y^{(k)}$" if seed == sorted(helm_seeds)[0] else None)
        ax.plot(d["k"], d["S_xy"], "^-", alpha=0.85, color="C2",
                markersize=6, label=r"$S_{x,y}^{(k)}$" if seed == sorted(helm_seeds)[0] else None)
    ax.set_xlabel(r"wavenumber $k$")
    ax.set_ylabel(r"conditional spatial Sobol index")
    ax.set_title(r"2D Helmholtz LC-PINN  (parametric $k$, $d=3$)" + "\n"
                  r"Pair $(x, y)$ emerges as $k$ grows")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.3); ax.set_ylim(-0.05, 1.10)

    ax = axes[0, 1]
    for seed, d in helm_seeds.items():
        ax.plot(d["k"], d["var_Y"], "o-", alpha=0.7, label=f"seed {seed}")
    ax.set_xlabel(r"wavenumber $k$")
    ax.set_ylabel(r"$\mathrm{Var}(u \mid k)$")
    ax.set_title(r"2D Helmholtz: conditional output variance")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # Row 2: Schrödinger
    ax = axes[1, 0]
    # For Schrödinger d=2 with parametric alpha, at fixed alpha the Sobol decomp over x alone is
    # trivially S_x = 1.0 (only one input). So instead we plot the per-alpha Var(u | alpha)
    # alongside a textbox stating S_x(α) = 1 by construction.
    for seed, d in schr_seeds.items():
        ax.plot(d["alpha"], d["var_Y"], "o-", alpha=0.7, label=f"seed {seed}")
    ax.set_xlabel(r"trap stiffness $\alpha$")
    ax.set_ylabel(r"$\mathrm{Var}(u \mid \alpha)$")
    ax.set_title(r"Schrödinger LC-PINN  (parametric $\alpha$, $d=2$)" + "\n"
                  r"No structural transition with $\alpha$ — only amplitude")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    # Joint d=2 Sobol indices for Schrödinger (constant across alpha because of multiplicative structure)
    # Pull these from results_schr1d_seed*.json
    schr_joint = []
    for path in sorted(RESULTS.glob("results_schr1d_seed*.json")):
        if "smoke" in path.stem:
            continue
        d = json.loads(path.read_text())
        s = d.get("sobol_indices", {})
        schr_joint.append({
            "seed": path.stem.split("seed")[-1].rstrip(".json"),
            "S_x": s.get("x", float("nan")),
            "S_alpha": s.get("alpha", float("nan")),
            "S_x_alpha": s.get("x/alpha", float("nan")),
        })
    if schr_joint:
        names = ["$S_x$", r"$S_\alpha$", r"$S_{x, \alpha}$"]
        means = [np.mean([r["S_x"] for r in schr_joint]),
                 np.mean([r["S_alpha"] for r in schr_joint]),
                 np.mean([r["S_x_alpha"] for r in schr_joint])]
        stds = [np.std([r["S_x"] for r in schr_joint]),
                np.std([r["S_alpha"] for r in schr_joint]),
                np.std([r["S_x_alpha"] for r in schr_joint])]
        bars = ax.bar(names, means, yerr=stds, color=["C0", "C3", "C2"],
                       capsize=4, edgecolor="white", linewidth=0.8)
        for bar, m in zip(bars, means):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel(r"Sobol index over joint $(x, \alpha)$")
    ax.set_title(r"Schrödinger joint Sobol decomposition" + "\n"
                  r"Spatial main effect dominates; cross-pair $\to 0$")
    ax.set_ylim(0, 1.10)
    ax.grid(alpha=0.3, axis="y")

    plt.suptitle("Parametric PDE family ANOVA structure: contrast between Helmholtz and Schrödinger",
                  fontsize=13, fontweight="bold", y=1.02)
    out = PLOT_DIR / "contrast_per_param.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
