"""Plot for the method ablation: rel-RMSE and S_xyk recovery across variants."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "results" / "figures"


def main():
    data = json.loads((RESULTS / "method_ablation.json").read_text())
    # Gold S_xyk from MC at N=1M
    mc_path = RESULTS / "results_mc_megaN_seed0.json"
    if mc_path.exists():
        mc = json.loads(mc_path.read_text())
        gold_sxyk = mc["S_triplet"]
    else:
        gold_sxyk = 0.426

    labels = {
        "fourier_order3_full": "Fourier\norder-3\nfull",
        "tanh_order3_no_fourier": "Tanh\n(no Fourier)\norder-3",
        "fourier_order2_no_triplet": "Fourier\norder-2\n(no triplet)",
        "fourier_order3_mains_only": "Fourier\norder-3\n(mains only)",
    }
    variants = data["variants"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # rel-RMSE
    names = [v["name"] for v in variants]
    short = [labels.get(n, n) for n in names]
    rel = [v["val_rel_rmse"] for v in variants]
    bars1 = ax1.bar(np.arange(len(names)), rel, color="#4C72B0", alpha=0.85)
    for i, v in enumerate(rel):
        ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8.5)
    ax1.set_xticks(np.arange(len(names))); ax1.set_xticklabels(short, fontsize=9)
    ax1.set_ylabel("Val rel-RMSE on LC-PINN output", fontsize=10)
    ax1.set_title("Reconstruction quality", fontsize=10)
    ax1.grid(axis="y", alpha=0.3); ax1.set_axisbelow(True)
    ax1.set_ylim(0, max(rel) * 1.2 + 0.05)

    # S_xyk
    sxyk = [v["sobol"].get("S_xyk") for v in variants]
    sxyk_clean = [s if s is not None else 0 for s in sxyk]
    bars2 = ax2.bar(np.arange(len(names)), sxyk_clean, color="#DD8452", alpha=0.85)
    ax2.axhline(gold_sxyk, ls="--", color="black", lw=1.5,
                label=f"MC-Sobol gold ($N=10^6$): {gold_sxyk:.3f}")
    for i, v in enumerate(sxyk):
        if v is None:
            ax2.text(i, 0.02, "n/a\n(no triplet term)", ha="center", fontsize=8.5)
        else:
            ax2.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.5)
    ax2.set_xticks(np.arange(len(names))); ax2.set_xticklabels(short, fontsize=9)
    ax2.set_ylabel("$S_{xyk}$ recovered", fontsize=10)
    ax2.set_title("Triplet recovery", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(axis="y", alpha=0.3); ax2.set_axisbelow(True)
    ax2.set_ylim(0, max(0.55, gold_sxyk * 1.2))

    fig.suptitle("Method ablation on 2D Helmholtz LC-PINN", fontsize=11, y=1.02)
    fig.tight_layout()
    out = FIGS / "method_ablation.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
