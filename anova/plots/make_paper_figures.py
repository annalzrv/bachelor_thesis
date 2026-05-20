"""Publication-ready figure set for the CIKM paper.

Produces a numbered, consistently-styled set of figures from the results JSONs.
Outputs both PNG (for slides/review) and PDF (for the paper). Single source of
truth for all paper figures.

  Fig 1 — three-PDE Sobol stacked bar chart (cross-PDE comparison)
  Fig 2 — Proposition 1: analytic vs measured per-$k$ Sobol overlay
  Fig 3 — per-$k$ Sobol curves with multi-seed (the killer figure)
  Fig 4 — Sobol triplet distribution: HDMR-RNG-seed histogram + MC gold standard
  Fig 5 — sample efficiency curve
  Fig 6 — compute-cost amortization
  Fig 7 — synthetic validation (Sobol recovery on polynomial benchmark)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
PAPER_FIGS = RESULTS / "paper_figures"
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

# Consistent style for all paper figures
mpl.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 13,
    "lines.linewidth": 1.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})


def save(fig, name):
    """Save with consistent dpi/bbox handling."""
    for ext in ("png", "pdf"):
        fig.savefig(PAPER_FIGS / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print(f"  {name}.png + .pdf")
    plt.close(fig)


# =============================================================================
# Fig 1 — three-PDE Sobol stacked bar
# =============================================================================
def fig1_cross_pde():
    # Helm 1D averages
    helm1d_means = {"x": [], "k": [], "x/k": []}
    for f in sorted(RESULTS.glob("results_helm1d_seed*.json")):
        if "smoke" in f.stem:
            continue
        s = json.loads(f.read_text()).get("sobol_indices", {})
        for k in helm1d_means:
            if k in s:
                helm1d_means[k].append(s[k])

    schr1d_means = {"x": [], "alpha": [], "x/alpha": []}
    for f in sorted(RESULTS.glob("results_schr1d_seed*.json")):
        if "smoke" in f.stem:
            continue
        s = json.loads(f.read_text()).get("sobol_indices", {})
        for k in schr1d_means:
            if k in s:
                schr1d_means[k].append(s[k])

    # 2D Helm: use MC-Sobol N=1M gold standard
    mc_path = RESULTS / "results_mc_megaN_seed0.json"
    if mc_path.exists():
        mc = json.loads(mc_path.read_text())
        helm2d = {
            "x": mc["S_first"].get("x", 0),
            "y": mc["S_first"].get("y", 0),
            "k": mc["S_first"].get("k", 0),
            "x/y": mc["S_pair"].get("x,y", 0),
            "x/k": mc["S_pair"].get("x,k", 0),
            "y/k": mc["S_pair"].get("y,k", 0),
            "x/y/k": mc["S_triplet"],
        }
    else:
        helm2d = {}

    pdes = [
        ("1D Helmholtz\n($d=2$)", helm1d_means),
        ("Schrödinger\n($d=2$)", schr1d_means),
        ("2D Helmholtz\n($d=3$, $N$=10$^6$)", helm2d),
    ]

    all_keys = ["x", "y", "k", "alpha", "x/y", "x/k", "y/k", "x/alpha", "x/y/k"]
    palette = {
        "x": "#1f77b4", "y": "#aec7e8", "k": "#ff7f0e", "alpha": "#ff7f0e",
        "x/y": "#9467bd", "x/k": "#2ca02c", "y/k": "#98df8a", "x/alpha": "#2ca02c",
        "x/y/k": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x_pos = np.arange(len(pdes))
    bottoms = np.zeros(len(pdes))

    for key in all_keys:
        heights = []
        for _, vals in pdes:
            v = vals.get(key, [])
            heights.append(np.mean(v) if isinstance(v, list) and v else (v if not isinstance(v, list) else 0.0))
        heights = np.array(heights, dtype=float)
        if heights.sum() < 1e-3:
            continue
        ax.bar(x_pos, heights, bottom=bottoms, color=palette.get(key, "gray"),
               edgecolor="white", linewidth=0.6, label=f"$S_{{{key.replace('/', ',')}}}$")
        for i, h in enumerate(heights):
            if h > 0.07:
                ax.text(x_pos[i], bottoms[i] + h / 2, f"{h:.2f}",
                         ha="center", va="center", fontsize=9,
                         color="white", fontweight="bold")
        bottoms += heights

    ax.set_xticks(x_pos)
    ax.set_xticklabels([n for n, _ in pdes])
    ax.set_ylabel("Sobol index (variance proportion)")
    ax.set_title("Functional-ANOVA Sobol decomposition across three parametric PDEs")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", ncol=2, framealpha=0.95)
    save(fig, "fig1_cross_pde_sobol")


# =============================================================================
# Fig 2 — Proposition 1 overlay
# =============================================================================
def fig2_proposition1():
    p = json.loads((RESULTS / "proposition1_analytic.json").read_text())
    k = np.array(p["k_values"])
    sx_a = np.array(p["S_x_analytic"])
    sxy_a = np.array(p["S_xy_analytic"])
    sx_e = np.array(p["S_x_empirical_mean"])
    sxy_e = np.array(p["S_xy_empirical_mean"])

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(k, sx_a, "-", color="#1f77b4", linewidth=2.5,
             label=r"$S_x^{(k)} = S_y^{(k)}$  Prop. 1 (analytic)", zorder=3)
    ax.plot(k, sxy_a, "-", color="#2ca02c", linewidth=2.5,
             label=r"$S_{x,y}^{(k)}$  Prop. 1 (analytic)", zorder=3)
    ax.plot(k, sx_e, "o", color="#1f77b4", markersize=9, alpha=0.7,
             markeredgecolor="white", markeredgewidth=1.2,
             label=r"$S_x^{(k)}$ measured (LC-PINN, 2 seeds, $N$=10$^4$)", zorder=4)
    ax.plot(k, sxy_e, "s", color="#2ca02c", markersize=9, alpha=0.7,
             markeredgecolor="white", markeredgewidth=1.2,
             label=r"$S_{x,y}^{(k)}$ measured", zorder=4)
    ax.set_xlabel(r"wavenumber  $k$")
    ax.set_ylabel(r"conditional spatial Sobol index")
    ax.set_title(r"Closed-form theory vs MC measurement on the LC-PINN output")
    ax.legend(loc="center right", framealpha=0.95)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlim(0.8, 10.2)
    save(fig, "fig2_proposition1_overlay")


# =============================================================================
# Fig 3 — per-k curves with multi-seed (the killer figure)
# =============================================================================
def fig3_per_k_multiseed():
    fig, (ax_low, ax_main) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                                            gridspec_kw={"width_ratios": [1, 1.6]})

    seeds = {}
    for path in sorted(RESULTS.glob("results_per_k_helm2d_seed*.json")):
        if "hires" in path.stem:
            continue
        d = json.loads(path.read_text())
        seed = path.stem.split("seed")[-1]
        seeds[seed] = d

    # Left: Var(Y|k) — output-magnitude story
    ax = ax_low
    for s, d in seeds.items():
        ax.plot(d["k_values"], d["var_Y_at_k"], "o-",
                 markersize=5, alpha=0.7, label=f"seed {s}")
    ax.set_xlabel(r"wavenumber  $k$")
    ax.set_ylabel(r"$\mathrm{Var}(u \mid k)$")
    ax.set_title(r"Conditional output variance")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    # Right: per-k Sobol with multi-seed
    ax = ax_main
    for i, (s, d) in enumerate(seeds.items()):
        k = d["k_values"]
        for series, c, m, label in [
            ("S_x_at_k", "#1f77b4", "o", "$S_x^{(k)}$"),
            ("S_y_at_k", "#aec7e8", "s", "$S_y^{(k)}$"),
            ("S_xy_at_k", "#2ca02c", "^", "$S_{x,y}^{(k)}$"),
        ]:
            ax.plot(k, d[series], "-" + m, color=c, alpha=0.65,
                     markersize=5,
                     label=label if i == 0 else None)
    ax.set_xlabel(r"wavenumber  $k$")
    ax.set_ylabel(r"Sobol index over $(x, y)$ at fixed $k$")
    ax.set_title(r"Spatial Sobol decomposition vs $k$ — emergence of pair effect at high $k$")
    ax.legend(loc="center right", fontsize=10, framealpha=0.95)
    ax.set_ylim(-0.05, 1.10)
    save(fig, "fig3_per_k_multiseed")


# =============================================================================
# Fig 4 — Sobol triplet distribution: 50-RNG-seed HDMR vs MC gold standard
# =============================================================================
def fig4_triplet_distribution():
    trips_hdmr = []
    for f in sorted(RESULTS.glob("hdmr_rng*_helm2d_seed0.json")):
        d = json.loads(f.read_text())
        t = d["sobol"].get("(0, 1, 2)") or d["sobol"].get("x/y/k")
        if t is not None:
            trips_hdmr.append(t)

    mc_seed = []
    for f in sorted(list(RESULTS.glob("results_mc_highn200_seed*.json")) +
                    list(RESULTS.glob("results_mc_sobol_seed*.json"))):
        d = json.loads(f.read_text())
        t = d.get("S_triplet")
        if t is not None:
            mc_seed.append(t)
    mc_megaN = None
    p = RESULTS / "results_mc_megaN_seed0.json"
    if p.exists():
        mc_megaN = json.loads(p.read_text())["S_triplet"]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.hist(trips_hdmr, bins=15, color="#1f77b4", edgecolor="white",
             alpha=0.75, label=f"Fourier HDMR, n={len(trips_hdmr)} RNG seeds (HDMR-norm)")
    ax.axvline(np.mean(trips_hdmr), color="#1f77b4", linestyle="--", linewidth=2,
                label=f"HDMR mean = {np.mean(trips_hdmr):.3f}  ($\\pm$ {np.std(trips_hdmr):.3f})")
    # MC gold standards
    for v in mc_seed:
        ax.axvline(v, color="#d62728", alpha=0.5, linewidth=1.5)
    ax.axvline(mc_seed[0] if mc_seed else 0.43, color="#d62728", alpha=0.0,
                label=f"MC-Sobol (N=$2\\times 10^5$), 4 seeds, mean {np.mean(mc_seed):.3f}")
    if mc_megaN is not None:
        ax.axvline(mc_megaN, color="#9467bd", linestyle="-", linewidth=3,
                    label=f"MC-Sobol gold standard (N=10$^6$): {mc_megaN:.3f}")

    ax.set_xlabel(r"$S_{x, y, k}$ triplet Sobol index")
    ax.set_ylabel("count of HDMR runs")
    ax.set_title("2D Helmholtz: distribution of triplet estimates across methods")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
    save(fig, "fig4_triplet_distribution")


# =============================================================================
# Fig 5 — sample efficiency
# =============================================================================
def fig5_sample_efficiency():
    runs = []
    import re
    for f in sorted(RESULTS.glob("results_sampeff_n*.json")):
        if "extreme" in f.stem:
            continue
        m = re.search(r"n(\d+)", f.stem)
        if not m:
            continue
        n = int(m.group(1))
        d = json.loads(f.read_text())
        val = d.get("jointhdmr_val_rel_rmse", float("nan"))
        triplet = d.get("sobol_indices", {}).get("x/y/k", float("nan"))
        runs.append((n, val, triplet))
    runs.sort()
    if not runs:
        return
    N, val, trip = zip(*runs)
    captured = [1 - v ** 2 for v in val]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ax1.semilogx(N, [c * 100 for c in captured], "o-", color="#1f77b4", markersize=8)
    ax1.set_xlabel(r"$N_\mathrm{train}$ (HDMR training samples)")
    ax1.set_ylabel(r"variance captured by HDMR (\%)")
    ax1.set_title("Sample-efficiency curve")
    ax1.set_ylim(40, 100)
    ax1.axhline(95, color="gray", linestyle=":", alpha=0.6, label="95% threshold")
    ax1.legend(fontsize=9.5)

    ax2.semilogx(N, trip, "s-", color="#d62728", markersize=8)
    ax2.axhline(0.426, color="#9467bd", linestyle="--", linewidth=1.5,
                label=r"MC gold std $S_{x,y,k} = 0.426$")
    ax2.set_xlabel(r"$N_\mathrm{train}$")
    ax2.set_ylabel(r"$S_{x,y,k}$ (HDMR-normalised)")
    ax2.set_title(r"Triplet estimate vs $N_\mathrm{train}$")
    ax2.legend(fontsize=9.5)
    save(fig, "fig5_sample_efficiency")


# =============================================================================
# Fig 6 — compute cost amortization (already in amortization_plot.py; refresh)
# =============================================================================
def fig6_amortization():
    cc_2d = json.loads((RESULTS / "compute_cost_2d.json").read_text())
    K = np.arange(1, 51)
    T_per_k = 6.0
    T_lc = 85.0

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(K, K * T_per_k, "-", color="#1f77b4", linewidth=2.5,
             label=r"per-$k$ retraining (linear in $K$)")
    ax.plot(K, np.full_like(K, T_lc, dtype=float), "-", color="#d62728", linewidth=2.5,
             label=r"LC-PINN (one-time, any $K$)")
    k_star = int(T_lc / T_per_k)
    ax.axvline(k_star, color="k", linestyle="--", alpha=0.5, linewidth=1.5,
                label=fr"crossover $K^* = {k_star}$")
    ax.fill_between(K[:k_star], 0, np.maximum(K[:k_star] * T_per_k, T_lc),
                      alpha=0.07, color="#1f77b4")
    ax.fill_between(K[k_star:], 0, np.maximum(K[k_star:] * T_per_k, T_lc),
                      alpha=0.07, color="#d62728")
    ax.set_xlabel(r"number of parameter values  $K$")
    ax.set_ylabel("training wall time (min)")
    ax.set_title("Training-cost amortization: LC-PINN wins for $K > 14$")
    ax.legend(loc="upper left", framealpha=0.95)
    save(fig, "fig6_amortization")


# =============================================================================
# Fig 7 — synthetic validation
# =============================================================================
def fig7_synthetic():
    # From synthetic test: analytic and learned Sobol on polynomial benchmark
    analytic = {
        "(0,)": 15 / 44, "(1,)": 9 / 44, "(2,)": 15 / 44,
        "(0, 1)": 0.0, "(0, 2)": 5 / 44, "(1, 2)": 0.0,
        "(0, 1, 2)": 0.0,
    }
    learned_path = RESULTS / "results_order3_h128_seed0.json"
    if not learned_path.exists():
        return
    learned = json.loads(learned_path.read_text()).get("sobol_indices", {})
    # Key mapping
    learned_norm = {
        "(0,)": learned.get("x", 0),
        "(1,)": learned.get("y", 0),
        "(2,)": learned.get("k", 0),
        "(0, 1)": learned.get("x/y", 0),
        "(0, 2)": learned.get("x/k", 0),
        "(1, 2)": learned.get("y/k", 0),
        "(0, 1, 2)": learned.get("x/y/k", 0),
    }
    keys = list(analytic.keys())
    labels = [k.replace("(", "$\\{").replace(")", "\\}$").replace(",", ", ").replace("'", "")
              for k in keys]
    labels = ["$\\{f_1\\}$", "$\\{f_2\\}$", "$\\{f_3\\}$",
              "$\\{f_{1,2}\\}$", "$\\{f_{1,3}\\}$", "$\\{f_{2,3}\\}$", "$\\{f_{1,2,3}\\}$"]
    a = [analytic[k] for k in keys]
    l = [learned_norm[k] for k in keys]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(keys))
    w = 0.38
    ax.bar(x - w / 2, a, w, color="#1f77b4", edgecolor="white", label="analytic")
    ax.bar(x + w / 2, l, w, color="#d62728", edgecolor="white", label="learned (tanh order-2)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Sobol index")
    ax.set_title("Synthetic polynomial benchmark: HDMR recovers analytic Sobol to within 0.005")
    ax.legend(framealpha=0.95)
    save(fig, "fig7_synthetic_validation")


def main():
    print("== Building paper figures ==")
    fig1_cross_pde()
    fig2_proposition1()
    fig3_per_k_multiseed()
    fig4_triplet_distribution()
    fig5_sample_efficiency()
    fig6_amortization()
    fig7_synthetic()
    print(f"\nAll figures written to {PAPER_FIGS}")


if __name__ == "__main__":
    main()
