"""Plot: analytic Sobol floor matches measured projection floor exactly."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "results" / "figures"


def main():
    data = json.loads((RESULTS / "architectural_floor_analytic.json").read_text())

    labels = {
        "order_0_constant": "Constant\n(no inputs)",
        "order_1_mains": "Mains only\n$f_x+f_y+f_k$",
        "additive_xy_plus_k": "Additive\n$f_{xy}+f_k$",
        "order_2_all_pairs": "All pairs\n$\\dots+f_{xk}+f_{yk}$",
        "order_3_full": "Full\n$\\dots+f_{xyk}$",
    }
    order = ["order_0_constant", "order_1_mains", "additive_xy_plus_k",
             "order_2_all_pairs", "order_3_full"]
    predicted = [data["results"][k]["predicted_floor"] for k in order]
    measured = [data["results"][k]["measured_rel_l2"] for k in order]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(order)); w = 0.36
    ax.bar(x - w/2, predicted, w,
           label=r"Sobol-derived floor $\sqrt{\sum_{S \notin \mathcal{F}} S_S}$",
           color="#4C72B0", alpha=0.85)
    ax.bar(x + w/2, measured, w,
           label=r"Measured $\|u_\mathrm{ref}-u^{(k)}\|_2 / \|u_\mathrm{ref}\|_2$",
           color="#DD8452", alpha=0.85)
    for i, v in enumerate(predicted):
        ax.text(i - w/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(measured):
        ax.text(i + w/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels([labels[k] for k in order], fontsize=9)
    ax.set_ylabel("Relative $L^2$ error", fontsize=11)
    ax.set_title("Sobol indices give exact architectural-complexity floor\n"
                 "(2D Helmholtz $u_\\mathrm{ref} = \\sin\\pi x\\sin\\pi y\\cos kx\\cos ky$, $N=10^6$)",
                 fontsize=10.5)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    ax.set_ylim(0, 1.15)

    fig.tight_layout()
    out = FIGS / "architectural_floor_analytic.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
