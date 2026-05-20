"""Publication-quality figures for the CIKM paper.

Reads aggregated results from lc_anova/results/ and produces multi-panel
figures suitable for a 9-10 page conference paper.

Run after the overnight batch + aggregate_results.py:
    python -m lc_anova.plots.polished_plots
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLOT_DIR = Path(__file__).resolve().parent / "results" / "figures"
RESULTS = Path(__file__).resolve().parent / "results"


def load_results():
    """Walk results/ and bucket by PDE."""
    out = {"helm1d": [], "schr1d": [], "helm2d_hdmr": [], "helm2d_mc": []}
    for p in sorted(RESULTS.glob("results_*.json")):
        try:
            payload = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        tag = p.stem.replace("results_", "")
        if "mc_sobol_seed" in tag:
            out["helm2d_mc"].append((tag, payload))
        elif tag.startswith("helm1d_seed"):
            out["helm1d"].append((tag, payload))
        elif tag.startswith("schr1d_seed"):
            out["schr1d"].append((tag, payload))
        elif tag.startswith("helm2d_fourier_seed") or "p3eps400" in tag:
            out["helm2d_hdmr"].append((tag, payload))
    return out


def plot_cross_pde_sobol(results, out_path: Path):
    """Stacked bar chart: Sobol decomposition across the three PDEs."""
    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Helm1d: average over seeds
    helm1d_sobol = {"x": [], "k": [], "x/k": []}
    for tag, p in results["helm1d"]:
        s = p.get("sobol_indices", {})
        for key in helm1d_sobol:
            if key in s:
                helm1d_sobol[key].append(s[key])

    # Schr1d
    schr1d_sobol = {"x": [], "alpha": [], "x/alpha": []}
    for tag, p in results["schr1d"]:
        s = p.get("sobol_indices", {})
        for key in schr1d_sobol:
            if key in s:
                schr1d_sobol[key].append(s[key])

    # Helm2d (HDMR)
    helm2d_sobol = {"x": [], "y": [], "k": [], "x/y": [], "x/k": [], "y/k": [], "x/y/k": []}
    for tag, p in results["helm2d_hdmr"]:
        s = p.get("sobol_indices", {})
        for key in helm2d_sobol:
            if key in s:
                helm2d_sobol[key].append(s[key])

    # Bars: mean across seeds
    pdes = ["1D Helmholtz\n(d=2)", "Schrödinger 1D\n(d=2)", "2D Helmholtz\n(d=3)"]
    sources = [helm1d_sobol, schr1d_sobol, helm2d_sobol]
    all_subsets = sorted({k for d in sources for k in d.keys()})
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_subsets)))
    color_for = dict(zip(all_subsets, colors))

    x_pos = np.arange(len(pdes))
    bottom = np.zeros(len(pdes))
    for subset in all_subsets:
        heights = []
        for src in sources:
            vals = src.get(subset, [])
            heights.append(np.mean(vals) if vals else 0.0)
        heights = np.array(heights)
        ax.bar(x_pos, heights, bottom=bottom, label=subset,
               color=color_for[subset], edgecolor="white", linewidth=0.5)
        # Label segments large enough
        for i, h in enumerate(heights):
            if h > 0.08:
                ax.text(x_pos[i], bottom[i] + h / 2, f"{subset}\n{h:.2f}",
                        ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        bottom += heights

    ax.set_xticks(x_pos)
    ax.set_xticklabels(pdes)
    ax.set_ylabel("Sobol index (variance proportion)")
    ax.set_title("Functional-ANOVA Sobol decomposition of LC-PINN output\n"
                 "(mean over LC-PINN training seeds)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Wrote {out_path}")


def plot_2d_helm_method_comparison(results, out_path: Path):
    """Bar chart: Fourier HDMR vs MC-Sobol on 2D Helmholtz."""
    helm2d_hdmr = results["helm2d_hdmr"]
    helm2d_mc = results["helm2d_mc"]
    if not helm2d_hdmr or not helm2d_mc:
        print("  skipping 2d method comparison (insufficient data)")
        return

    subsets = ["x", "y", "k", "x/y", "x/k", "y/k", "x/y/k"]

    def collect(payloads, key_path):
        """Collect each subset's Sobol value across seeds."""
        d = {s: [] for s in subsets}
        for tag, p in payloads:
            target = p
            for k in key_path:
                target = target.get(k, {})
            for s in subsets:
                # Try both "x" and "(0,)" formats
                for k in [s, s.replace("/", ","), str(tuple([0]) if s == "x" else None)]:
                    if k in target:
                        d[s].append(target[k])
                        break
        return {s: (np.mean(v) if v else 0.0, np.std(v) if v else 0.0) for s, v in d.items()}

    hdmr_vals = collect(helm2d_hdmr, ["sobol_indices"])
    mc_first = collect(helm2d_mc, ["S_first"])
    mc_pair = collect(helm2d_mc, ["S_pair"])
    mc_trip = {tag: p.get("S_triplet", 0.0) for tag, p in helm2d_mc}
    mc_trip_mean = float(np.mean(list(mc_trip.values()))) if mc_trip else 0.0
    mc_trip_std = float(np.std(list(mc_trip.values()))) if mc_trip else 0.0

    # MC's "x" lives in S_first; "x/y" in S_pair; "x/y/k" in S_triplet.
    mc_vals = {}
    for s in subsets:
        if s in ("x", "y", "k"):
            mc_vals[s] = mc_first.get(s, (0.0, 0.0))
        elif s == "x/y/k":
            mc_vals[s] = (mc_trip_mean, mc_trip_std)
        else:
            mc_vals[s] = mc_pair.get(s, (0.0, 0.0))

    x = np.arange(len(subsets))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    hdmr_means = [hdmr_vals[s][0] for s in subsets]
    hdmr_stds = [hdmr_vals[s][1] for s in subsets]
    mc_means = [mc_vals[s][0] for s in subsets]
    mc_stds = [mc_vals[s][1] for s in subsets]
    ax.bar(x - w / 2, hdmr_means, w, yerr=hdmr_stds, label="Fourier HDMR (normalized)",
           color="C0", capsize=3)
    ax.bar(x + w / 2, mc_means, w, yerr=mc_stds, label="MC-Sobol (gold standard)",
           color="C1", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(subsets)
    ax.set_ylabel("Sobol index")
    ax.set_title("2D Helmholtz: Fourier HDMR vs MC-Sobol (mean ± std over seeds)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Wrote {out_path}")


def plot_per_k_sobol(out_path: Path):
    """Per-k Sobol curves for 2D Helmholtz (THE killer figure)."""
    import json
    per_k_files = sorted(RESULTS.glob("results_per_k_helm2d_seed*.json"))
    if not per_k_files:
        print("  skipping per-k plot (no data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for path in per_k_files:
        data = json.loads(path.read_text())
        seed = path.stem.split("seed")[-1]
        k = np.array(data["k_values"])
        s_x = np.array(data["S_x_at_k"])
        s_y = np.array(data["S_y_at_k"])
        s_xy = np.array(data["S_xy_at_k"])
        v = np.array(data["var_Y_at_k"])

        ax = axes[0]
        if path == per_k_files[0]:
            ax.plot(k, s_x, "o-", label=r"$S_x(k)$", color="C0", markersize=5)
            ax.plot(k, s_y, "s-", label=r"$S_y(k)$", color="C1", markersize=5)
            ax.plot(k, s_xy, "^-", label=r"$S_{x,y}(k)$ (pair)", color="C2", markersize=5)
        else:
            ax.plot(k, s_x, "o-", alpha=0.4, color="C0", markersize=4)
            ax.plot(k, s_y, "s-", alpha=0.4, color="C1", markersize=4)
            ax.plot(k, s_xy, "^-", alpha=0.4, color="C2", markersize=4)

        ax2 = axes[1]
        if path == per_k_files[0]:
            ax2.plot(k, v, "o-", label=f"seed {seed}", color="C3", markersize=5)
        else:
            ax2.plot(k, v, "o-", alpha=0.5, label=f"seed {seed}", markersize=4)

    axes[0].axhline(0, color="k", linewidth=0.5, alpha=0.3)
    axes[0].axhline(1, color="k", linewidth=0.5, alpha=0.3)
    axes[0].set_xlabel("wavenumber  k")
    axes[0].set_ylabel("Sobol index (over $x$, $y$ at fixed $k$)")
    axes[0].set_title(r"Spatial Sobol decomposition vs $k$ (2D Helmholtz LC-PINN)")
    axes[0].legend(loc="center right"); axes[0].grid(alpha=0.3)
    axes[0].set_ylim(-0.1, 1.1)

    axes[1].set_xlabel("wavenumber  k")
    axes[1].set_ylabel(r"Var$(u | k)$")
    axes[1].set_title("Conditional output variance vs $k$")
    axes[1].grid(alpha=0.3); axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Wrote {out_path}")


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    print("== Polished plots ==")
    print(f"  helm1d_hdmr runs: {len(results['helm1d'])}")
    print(f"  schr1d_hdmr runs: {len(results['schr1d'])}")
    print(f"  helm2d_hdmr runs: {len(results['helm2d_hdmr'])}")
    print(f"  helm2d_mc runs:   {len(results['helm2d_mc'])}")

    plot_cross_pde_sobol(results, PLOT_DIR / "cross_pde_sobol.png")
    plot_2d_helm_method_comparison(results, PLOT_DIR / "helm2d_method_comparison.png")
    plot_per_k_sobol(PLOT_DIR / "per_k_sobol_helm2d.png")


if __name__ == "__main__":
    main()
