"""Plot HDMR truncation floors: post-hoc rel-L² of order-k projection."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "results" / "figures"


def main():
    data = json.loads((RESULTS / "hdmr_truncation_floors.json").read_text())
    orders = [0, 1, 2, 3]
    predicted = [data["predicted_floor"][f"order_{k}"] for k in orders]
    measured = [data["measured_truncation_rel_l2"][f"order_{k}"] for k in orders]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    x = np.arange(len(orders)); w = 0.36
    ax.bar(x - w/2, predicted, w,
           label=r"Sobol predicted $\sqrt{\sum_{|S| > k} S_S}$",
           color="#4C72B0", alpha=0.85)
    ax.bar(x + w/2, measured, w,
           label="Measured rel-$L^2$ of order-$k$ HDMR truncation",
           color="#DD8452", alpha=0.85)
    for i, v in enumerate(predicted):
        ax.text(i - w/2, v + 0.018, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(measured):
        ax.text(i + w/2, v + 0.018, f"{v:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels([f"$k={k}$" for k in orders])
    ax.set_xlabel("Truncation order $k$")
    ax.set_ylabel("Relative $L^2$ error")
    ax.set_title("Order-$k$ HDMR truncation matches Sobol-predicted floor\n"
                 "(2D Helmholtz LC-PINN, post-hoc projection)", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    ax.set_ylim(0, 1.1)

    fig.tight_layout()
    out = FIGS / "truncation_floors.pdf"
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
