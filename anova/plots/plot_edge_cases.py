"""Plot predicted-vs-measured marginal approximation cost across priors on k."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "results" / "figures"


def main():
    data = json.loads((RESULTS / "edge_cases_priors.json").read_text())
    order = ["training", "near_zero", "near_boundary", "ood_extrapolation"]
    labels = {
        "training": "$k \\sim U(-1, 1)$\nTraining",
        "near_zero": "$k \\sim U(-0.3, 0.3)$\nNear zero",
        "near_boundary": "$k \\sim U(0.7, 1.0)$\nBoundary",
        "ood_extrapolation": "$k \\sim U(1.0, 1.5)$\nOOD (+50%)",
    }
    predicted = [data[k]["predicted_rel_l2"] for k in order]
    measured = [data[k]["measured_rel_l2"] for k in order]

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    x = np.arange(len(order)); width = 0.36
    ax.bar(x - width/2, predicted, width, label=r"Sobol predicted $\sqrt{S_k + S_{xk} + S_{yk} + S_{xyk}}$",
           color="#4C72B0", alpha=0.85)
    ax.bar(x + width/2, measured, width, label=r"Measured rel-$L^2$",
           color="#DD8452", alpha=0.85)
    for i, v in enumerate(predicted):
        ax.text(i - width/2, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.2)
    for i, v in enumerate(measured):
        ax.text(i + width/2, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.2)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in order], fontsize=9)
    ax.set_ylabel("Relative $L^2$ error", fontsize=11)
    ax.set_title("Bound holds under shifted and OOD priors on $k$", fontsize=11)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)

    # Shade OOD region
    ax.axvspan(2.5, 3.5, color="gray", alpha=0.08, zorder=0)
    ax.text(3.0, 0.9, "OOD", fontsize=9, color="gray", ha="center", style="italic")

    fig.tight_layout()
    out = FIGS / "edge_cases_priors.pdf"
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
