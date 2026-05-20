"""Validate MC-Sobol on the synthetic polynomial benchmark.

The synthetic function on [0,1]^3 (see synthetic.py) has known analytic
Sobol indices. This script runs Saltelli MC-Sobol on it and checks
recovery.

Analytic ground truth (from synthetic.py):
    S_1 = S_3 = 15/44 ≈ 0.3409   (first-order)
    S_2       = 9/44  ≈ 0.2045
    S_13      = 5/44  ≈ 0.1136   (pure pair)
    S_12 = S_23 = 0
    S_123      = 0               (no third-order term)
    Total indices:
    ST_1 = ST_3 = (15 + 5)/44 = 20/44 ≈ 0.4545
    ST_2       = 9/44  ≈ 0.2045   (no interactions involving z_2)

Run:
    cd /Users/anna/Desktop/research/anova
    python -m lc_anova.tests.test_mc_sobol_synthetic
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_REPO = os.path.dirname(_PKG)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lc_anova.core.mc_sobol import mc_sobol_full, uniform_unit_box_sampler
from lc_anova.core.synthetic import ANALYTIC_SOBOL, u_synthetic


def main(N: int = 50_000, seed: int = 0):
    print(f"MC-Sobol on synthetic polynomial: N={N}, seed={seed}")

    # u_synthetic accepts arrays of shape (N, 3) and returns (N,)
    def model_fn(z):
        return u_synthetic(z)

    def sampler(rng, n):
        return uniform_unit_box_sampler(rng, n, d=3)

    out = mc_sobol_full(model_fn, sampler, N=N, d=3, seed=seed)

    print(f"\nVar(Y) (estimated): {out['var_Y']:.6f}    (analytic 44/45 = {44/45:.6f})")

    print("\nFirst-order Sobol indices:")
    print(f"  {'subset':<6} {'mc':>8} {'analytic':>10} {'abs err':>8}")
    for k in [(0,), (1,), (2,)]:
        mc = out["S_first"][k]
        ana = ANALYTIC_SOBOL[k]
        print(f"  {str(k):<6} {mc:>8.4f} {ana:>10.4f} {abs(mc - ana):>8.4f}")

    print("\nTotal Sobol indices:")
    print(f"  {'subset':<6} {'mc':>8} {'analytic':>10} {'abs err':>8}")
    analytic_total = {(0,): 20/44, (1,): 9/44, (2,): 20/44}  # ST_i = S_i + sum interactions
    for k in [(0,), (1,), (2,)]:
        mc = out["S_total"][k]
        ana = analytic_total[k]
        print(f"  {str(k):<6} {mc:>8.4f} {ana:>10.4f} {abs(mc - ana):>8.4f}")

    print("\nPure pair Sobol indices:")
    print(f"  {'subset':<8} {'mc':>8} {'analytic':>10} {'abs err':>8}")
    for k in [(0, 1), (0, 2), (1, 2)]:
        mc = out["S_pair"][k]
        ana = ANALYTIC_SOBOL[k]
        print(f"  {str(k):<8} {mc:>8.4f} {ana:>10.4f} {abs(mc - ana):>8.4f}")

    print("\nTriplet Sobol index (residual):")
    print(f"  S_(0,1,2) MC = {out['S_triplet']:.4f}   (analytic 0)")

    # ---- Pass / fail report -----------------------------------------------
    checks = {
        "S_1 vs analytic 0.34 (within 0.02)": abs(out["S_first"][(0,)] - 15/44) < 0.02,
        "S_2 vs analytic 0.20 (within 0.02)": abs(out["S_first"][(1,)] - 9/44) < 0.02,
        "S_3 vs analytic 0.34 (within 0.02)": abs(out["S_first"][(2,)] - 15/44) < 0.02,
        "S_13 vs analytic 0.11 (within 0.02)": abs(out["S_pair"][(0, 2)] - 5/44) < 0.02,
        "S_12 ~ 0 (within 0.02)":             abs(out["S_pair"][(0, 1)]) < 0.02,
        "S_23 ~ 0 (within 0.02)":             abs(out["S_pair"][(1, 2)]) < 0.02,
        "S_triplet ~ 0 (within 0.02)":         abs(out["S_triplet"]) < 0.02,
    }
    print("\n" + "=" * 60)
    for name, ok in checks.items():
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {name}")
    all_pass = all(checks.values())
    print("=" * 60)
    print("ALL PASS" if all_pass else "SOME FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
