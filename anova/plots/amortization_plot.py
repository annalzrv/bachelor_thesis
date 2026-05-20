"""Plot LC-PINN training amortization vs per-k retraining.

Honest cost analysis: per-k retraining requires K trained models for
K parameter values; LC-PINN trains once and serves any k at inference.
The crossover happens at K ≈ (T_LC) / (T_per_k) where T_LC is LC-PINN
training time and T_per_k is per-k training time.

From OVERNIGHT_RESULTS.md: LC-PINN 2D Helm FiLM+L-BFGS ≈ 85 min/seed,
ReLoBRaLo per-k ≈ 6 min/seed. So crossover K* ≈ 14.

The "infinite speedup" framing kicks in for continuous parameter
sweeps (no discrete K — you can evaluate the LC-PINN at any k value
at inference time).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    # Wall-times from measured benchmarks
    cc_2d_path = RESULTS / "compute_cost_2d.json"
    cc_1d_path = RESULTS / "compute_cost_1d.json"

    cc_2d = json.loads(cc_2d_path.read_text())
    cc_1d = json.loads(cc_1d_path.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel 1: training cost (per-k vs LC-PINN) as function of K
    K = np.arange(1, 51)
    T_per_k = 6.0
    T_lc = 85.0
    train_per_k = K * T_per_k
    train_lc = np.full_like(K, T_lc, dtype=float)

    ax = axes[0]
    ax.plot(K, train_per_k, "-", linewidth=2, color="C0", label="per-$k$ retraining (× K)")
    ax.plot(K, train_lc, "-", linewidth=2, color="C1", label="LC-PINN (one-time)")
    ax.axvline(int(T_lc / T_per_k), color="k", linestyle="--", alpha=0.5,
               label=f"crossover  K* = {int(T_lc/T_per_k)}")
    ax.set_xlabel(r"number of parameter values  $K$")
    ax.set_ylabel("training wall time (min)")
    ax.set_title("Training cost: LC-PINN amortizes for K > 14")
    ax.legend(); ax.grid(alpha=0.3)

    # Panel 2: inference cost per Sobol-analysis call
    ax2 = axes[1]
    K_dense = np.linspace(1, 100, 100)
    # 1D Helm numbers
    t_lc_1d = cc_1d["lc_pinn"]["wall_seconds"]
    t_per_k_avg_1d = np.mean([x["wall_seconds"] for x in cc_1d["per_k_models"]["per_k"]])
    inf_per_k_1d = K_dense * t_per_k_avg_1d * 1000  # ms
    inf_lc_1d = np.full_like(K_dense, t_lc_1d * 1000)  # one MC-Sobol pass

    ax2.plot(K_dense, inf_per_k_1d, "-", linewidth=2, color="C0",
             label=f"per-$k$ MC-Sobol (avg {t_per_k_avg_1d*1000:.2f}ms × K)")
    ax2.plot(K_dense, inf_lc_1d, "-", linewidth=2, color="C1",
             label=f"LC-PINN joint MC-Sobol ({t_lc_1d*1000:.1f}ms one-shot)")
    ax2.set_xlabel(r"number of parameter values  $K$")
    ax2.set_ylabel("inference wall time per analysis (ms)")
    ax2.set_title("Inference cost: per-k wins at low K, LC-PINN gives joint decomp")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = RESULTS / "figures" / "compute_cost_amortization.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
