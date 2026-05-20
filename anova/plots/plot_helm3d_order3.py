"""Plot all 4 mains, 6 pairs, 4 triplets + 1 quadruplet for 3D Helm."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "results" / "figures"


def main():
    data = json.loads((RESULTS / "helm3d_order3_full.json").read_text())
    gold = data["gold_mc_sobol"]
    # group by order
    mains, pairs, trips = [], [], []
    quad = None
    for k, v in gold.items():
        if k == "var_Y":
            continue
        nm = k[2:]  # strip "S_"
        if nm == "xyzk":
            quad = (nm, v)
        elif len(nm) == 1:
            mains.append((nm, v))
        elif len(nm) == 2:
            pairs.append((nm, v))
        elif len(nm) == 3:
            trips.append((nm, v))

    mains.sort(key=lambda t: -t[1])
    pairs.sort(key=lambda t: -t[1])
    trips.sort(key=lambda t: -t[1])
    all_items = mains + pairs + trips + ([quad] if quad else [])
    names = [t[0] for t in all_items]
    vals = [t[1] for t in all_items]
    n_m, n_p, n_t = len(mains), len(pairs), len(trips)
    colors = (["#4C72B0"] * n_m + ["#DD8452"] * n_p + ["#55A868"] * n_t
              + (["#C44E52"] if quad else []))

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    x = np.arange(len(names))
    ax.bar(x, vals, color=colors, alpha=0.88)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5, rotation=0)
    ax.set_xticks(x); ax.set_xticklabels([f"$S_{{{nm}}}$" for nm in names], rotation=0)
    ax.set_ylabel("Sobol index", fontsize=10)
    ax.set_title("3D Helmholtz: full order-3 decomposition (MC-Sobol gold $N=300k$)",
                 fontsize=10.5)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)

    # Legend by color
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#4C72B0", label=f"main ({n_m})"),
        Patch(color="#DD8452", label=f"pair ({n_p})"),
        Patch(color="#55A868", label=f"triplet ({n_t})"),
    ]
    if quad:
        legend.append(Patch(color="#C44E52", label="quadruplet (1)"))
    ax.legend(handles=legend, loc="upper right", fontsize=9, ncol=2)

    fig.tight_layout()
    out = FIGS / "helm3d_order3.pdf"
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
