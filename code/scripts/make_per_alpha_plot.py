"""Per-alpha accuracy plot for 1D Schrödinger.

Plots rel-L^2 (mean across seeds, +/-std error bars) vs alpha for
LC-PINN, PI-DeepONet, and SA-PINN. LC and DON have a 20-point grid;
SA only has 3 training points.

Output: paper/figures/per_alpha_schrodinger.pdf
"""
from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np


REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = REPO / "paper" / "figures" / "per_alpha_schrodinger.pdf"


def _grid_stats(runs: list[dict], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alphas = sorted(float(a) for a in runs[0][key].keys())
    matrix = np.array([[r[key][f"{a}"] for a in alphas] for r in runs])
    return np.array(alphas), matrix.mean(axis=0), matrix.std(axis=0)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    lc = json.loads((RESULTS / "lc_pinn_schrodinger_film_lbfgs.json").read_text())
    don = json.loads((RESULTS / "pi_deeponet_schrodinger_matched.json").read_text())
    sa = json.loads((RESULTS / "sa_pinn_schrodinger.json").read_text())

    alpha_lc, mean_lc, std_lc = _grid_stats(lc["runs"], "rel_l2_per_alpha")
    alpha_don, mean_don, std_don = _grid_stats(don["runs"], "rel_l2_per_k")

    alpha_sa = np.array([float(a) for a in sa["per_alpha"]])
    mean_sa = np.array([sa["per_alpha"][f"{a:.2f}"]["mean"] for a in alpha_sa])
    std_sa = np.array([sa["per_alpha"][f"{a:.2f}"]["std"] for a in alpha_sa])

    fig, ax = plt.subplots(figsize=(6.0, 3.0))

    lo_lc = np.minimum(std_lc, 0.9 * mean_lc)
    ax.errorbar(alpha_lc, mean_lc, yerr=[lo_lc, std_lc],
                color="tab:blue", marker="*", ms=7,
                linewidth=1.6, linestyle="-",
                label="LC-PINN (one network, 20-pt grid)",
                capsize=2.5, elinewidth=0.9, zorder=4)

    lo_don = np.minimum(std_don, 0.9 * mean_don)
    ax.errorbar(alpha_don, mean_don, yerr=[lo_don, std_don],
                color="tab:purple", marker="D", ms=4,
                linewidth=1.4, linestyle="-",
                label="PI-DeepONet (one network)",
                capsize=2.5, elinewidth=0.9, zorder=3)

    lo_sa = np.minimum(std_sa, 0.9 * mean_sa)
    ax.errorbar(alpha_sa, mean_sa, yerr=[lo_sa, std_sa],
                color="tab:orange", marker="o", ms=6,
                linewidth=1.6, linestyle="--",
                label="SA-PINN (per-$\\alpha$, 3 retrainings)",
                capsize=3, elinewidth=1.0, zorder=3)

    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1e-1)
    ax.set_xlabel("trap stiffness $\\alpha$")
    ax.set_ylabel("rel-$L^2$ (log)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
