"""Per-k accuracy plot for 1D Helmholtz.

For each method, plots rel-L^2 (mean across seeds) vs k, with shaded
mean +/- std band. The story: LC-PINN stays within one order of
magnitude across the entire k-range while per-k retrained baselines
collapse at the difficult end.

Output: paper/figures/per_k_helmholtz.pdf
"""
from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np


REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = REPO / "paper" / "figures" / "per_k_helmholtz.pdf"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads((RESULTS / "per_k_breakdown_helmholtz.json").read_text())
    k_grid = np.array(d["k_grid"], dtype=float)

    methods = [
        ("LC-PINN (one network)",   "lc_pinn_per_k",   "tab:blue",   "*", "-"),
        ("SA-PINN (per-$k$)",       "sa_pinn_per_k",   "tab:orange", "o", "--"),
        ("ReLoBRaLo (per-$k$)",     "relobralo_per_k", "tab:red",    "s", "--"),
    ]

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    for label, key, color, marker, ls in methods:
        means = np.array([d[key][f"{k:.2f}"]["mean"] for k in k_grid])
        stds  = np.array([d[key][f"{k:.2f}"]["std"]  for k in k_grid])
        # Asymmetric log-friendly error bars: clip lower bar to a small
        # fraction of the mean so the marker stays visible on log axis.
        lo = np.minimum(stds, 0.9 * means)
        ax.errorbar(k_grid, means, yerr=[lo, stds],
                    color=color, marker=marker, ms=6,
                    linewidth=1.6, linestyle=ls, label=label,
                    capsize=3, elinewidth=1.0, zorder=3)

    ax.set_yscale("log")
    ax.set_ylim(3e-5, 3e0)
    ax.set_xlabel("wavenumber $k$")
    ax.set_ylabel("rel-$L^2$ (log)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.set_xticks(k_grid)
    ax.set_xticklabels([f"{k:g}" for k in k_grid])

    plt.tight_layout()
    plt.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
