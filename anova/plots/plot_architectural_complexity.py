"""Plot Sobol-predicted floor vs measured rel-L² for restricted LC-PINNs.

Reads:
  - results/restricted_lcpinn.json (from architectural_restriction.py)
  - results/hdmr_truncation_floors.json (post-hoc projection, for comparison)
"""

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
    restricted = json.loads((RESULTS / "restricted_lcpinn.json").read_text())
    trunc_path = RESULTS / "hdmr_truncation_floors.json"
    trunc = json.loads(trunc_path.read_text()) if trunc_path.exists() else None

    pred_add = restricted["predictions"]["additive"]
    pred_o2 = restricted["predictions"]["order2"]

    add_result = next(r for r in restricted["results"] if r["name"] == "additive")
    o2_result = next(r for r in restricted["results"] if r["name"] == "order2")
    meas_add = add_result["eval"]["mean_rel_l2"]
    meas_o2 = o2_result["eval"]["mean_rel_l2"]

    # Post-hoc truncation values (if available)
    if trunc:
        post_add = trunc["measured_truncation_rel_l2"]["order_1"]
        post_o2 = trunc["measured_truncation_rel_l2"]["order_2"]
    else:
        post_add = None; post_o2 = None

    labels = ["Additive\n$f_{xy}(x,y)+f_k(k)$", "Order-2\n$+f_{xk}+f_{yk}$", "Full LC-PINN\n(unrestricted)"]
    predicted = [pred_add, pred_o2, 0.0]
    trained = [meas_add, meas_o2, 0.022]
    post_hoc = [post_add, post_o2, None]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(3); w = 0.27
    b1 = ax.bar(x - w, predicted, w, label="Sobol-predicted floor\n$\\sqrt{\\sum_{|S|>k} S_S}$",
                color="#4C72B0", alpha=0.85)
    b2 = ax.bar(x, trained, w, label="Trained restricted PINN\n(mean rel-$L^2$)",
                color="#DD8452", alpha=0.85)
    has_post = any(p is not None for p in post_hoc)
    if has_post:
        post_vals = [p if p is not None else 0 for p in post_hoc]
        ax.bar(x + w, post_vals, w, label="Post-hoc HDMR truncation\nof full LC-PINN",
               color="#55A868", alpha=0.55, hatch="//")
        for i, v in enumerate(post_hoc):
            if v is None:
                continue
            ax.text(i + w, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(predicted):
        ax.text(i - w, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(trained):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Relative $L^2$ error  (mean across 21 $k$ values)", fontsize=10)
    ax.set_title("Sobol indices predict architectural complexity for 2D Helmholtz",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    out = FIGS / "architectural_complexity.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
