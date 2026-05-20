"""Plot for Section 'Application: actionable dimensionality reduction'.

Side-by-side bar chart: Sobol-predicted approximation cost vs measured
rel-L2 of the marginal approximation, for Schrödinger (drop α) and
Helmholtz (drop k)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def main():
    data = json.loads((RESULTS / "dim_reduction_demo.json").read_text())
    schr = data["schrodinger"]; helm = data["helmholtz_2d"]

    labels = ["Schrödinger\ndrop α", "2D Helmholtz\ndrop k"]
    predicted = [schr["predicted_rel_l2"], helm["predicted_rel_l2"]]
    measured = [schr["rel_l2"], helm["rel_l2"]]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    b1 = ax.bar(x - width/2, predicted, width, label="Sobol-predicted\n$\\sqrt{\\sum_{S\\ni \\lambda} S_S}$",
                color="#4C72B0", alpha=0.85)
    b2 = ax.bar(x + width/2, measured, width, label="Measured rel-$L^2$\nof marginal $\\bar u$",
                color="#DD8452", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r"Relative $L^2$ error", fontsize=11)
    ax.set_title("Sobol indices predict the cost of dropping an axis", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for b, val in zip(b1, predicted):
        ax.text(b.get_x() + b.get_width()/2, val + 0.015, f"{val:.3f}",
                ha="center", fontsize=8.5, color="#1F2A44")
    for b, val in zip(b2, measured):
        ax.text(b.get_x() + b.get_width()/2, val + 0.015, f"{val:.3f}",
                ha="center", fontsize=8.5, color="#1F2A44")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.1, ls="--", color="gray", alpha=0.4, lw=0.8)
    ax.text(1.95, 0.105, "10%", fontsize=8, color="gray", ha="right")

    fig.tight_layout()
    out = FIGS / "dim_reduction_demo.pdf"
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
