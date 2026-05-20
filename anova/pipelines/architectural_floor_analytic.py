"""Analytic verification of architectural-complexity Sobol bound.

Claim: For any function $u(x, y, k)$, the best $L^2$-projection onto the
function class $\\mathcal{F}_k = \\{u : u = \\sum_{|S| \\le k} f_S(z_S)\\}$
has $L^2$ residual $\\sqrt{\\sum_{|S| > k} V_S(u)}$, by ANOVA orthogonality.

We verify directly on the 2D Helmholtz analytic reference
$u_\\text{ref}(x, y; k) = \\sin(\\pi x)\\sin(\\pi y)\\cos(kx)\\cos(ky)$
under $x, y \\sim U(0,1)$ and $k \\sim U(1, 10)$:

  Order 0 (constant)                     floor = sqrt(Var(u))/sqrt(Var(u)) = 1
  Order 1 (mains only): u_x + u_y + u_k  floor = sqrt(1 - S_x - S_y - S_k)
  Order 1 + (x,y) pair:                  floor = sqrt(S_xk + S_yk + S_xyk)
  Order 2 (all pairs):                   floor = sqrt(S_xyk)
  Order 3 (all triplets):                floor = 0

Construction:
  1. Draw N=$10^6$ samples (x, y, k).
  2. Compute u_ref on all samples.
  3. For each ANOVA subset, compute the conditional mean via binning.
  4. Subtract the order-k truncation from u_ref; measure residual rel-L².
  5. Compare to the Sobol-derived prediction.

The analytic projection IS achievable by ANY additive/pair-only/triplet-free
network of sufficient capacity trained on $u_\\text{ref}$ via L² regression.
Therefore the floor is an a priori lower bound on the achievable rel-L² of
any such architecture.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ANOVA = _HERE.parent.parent
sys.path.insert(0, str(_REPO_ANOVA))


def u_ref_2d_helm(x, y, k):
    """Analytic 2D Helm reference: sin(pi x) sin(pi y) cos(kx) cos(ky)."""
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(k * x) * np.cos(k * y)


def k_range_uniform(n, rng, k_lo=1.0, k_hi=10.0):
    return rng.uniform(k_lo, k_hi, size=(n,)).astype(np.float64)


def compute_anova_terms(samples: dict, n_bins: int = 40):
    """Compute conditional means via binning to get ANOVA decomposition.

    samples: dict with keys 'x', 'y', 'k', 'u'.
    Returns dict mapping subset (tuple of axis names) -> (N,) array of term values.
    """
    x, y, k_arr, u = samples["x"], samples["y"], samples["k"], samples["u"]
    N = len(u)
    f0 = float(u.mean())
    u_c = u - f0  # centered

    # Bin each axis
    def bin_idx(arr, n_bins_):
        lo, hi = arr.min(), arr.max() + 1e-9
        return np.clip(((arr - lo) / (hi - lo) * n_bins_).astype(int), 0, n_bins_ - 1)

    bi = {"x": bin_idx(x, n_bins), "y": bin_idx(y, n_bins), "k": bin_idx(k_arr, n_bins)}

    # Main effect of axis a: f_a(a) = E[u | a] - f0
    def main(a):
        avg = np.zeros(n_bins); cnt = np.zeros(n_bins)
        np.add.at(avg, bi[a], u_c); np.add.at(cnt, bi[a], 1)
        avg = avg / np.maximum(cnt, 1)
        return avg[bi[a]]

    mains = {a: main(a) for a in ["x", "y", "k"]}

    # Pair effect: f_ab = E[u | a, b] - f_a - f_b - f0
    def pair(a, b):
        idx = bi[a] * n_bins + bi[b]
        avg = np.zeros(n_bins * n_bins); cnt = np.zeros(n_bins * n_bins)
        np.add.at(avg, idx, u_c); np.add.at(cnt, idx, 1)
        avg = avg / np.maximum(cnt, 1)
        f_ab = avg[idx]
        return f_ab - mains[a] - mains[b]

    pairs = {(a, b): pair(a, b) for a, b in [("x", "y"), ("x", "k"), ("y", "k")]}

    # Triplet: u_c - (mains) - (pairs)
    triplet = u_c - sum(mains.values()) - sum(pairs.values())

    return {"f0": f0, "mains": mains, "pairs": pairs, "triplet": triplet,
            "u_centered": u_c}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1_000_000)
    ap.add_argument("--n-bins", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="lc_anova/results/architectural_floor_analytic.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"Sampling N={args.N:,} from u_ref")
    x = rng.uniform(0, 1, size=args.N).astype(np.float64)
    y = rng.uniform(0, 1, size=args.N).astype(np.float64)
    k = k_range_uniform(args.N, rng)
    u = u_ref_2d_helm(x, y, k)
    samples = {"x": x, "y": y, "k": k, "u": u}
    var_u = float(u.var())
    print(f"  Var(u_ref) = {var_u:.6f}")

    print(f"Computing ANOVA terms via {args.n_bins}-bin conditional means")
    terms = compute_anova_terms(samples, n_bins=args.n_bins)

    # Sobol indices from binned terms
    sobol = {"x": float(terms["mains"]["x"].var() / var_u),
             "y": float(terms["mains"]["y"].var() / var_u),
             "k": float(terms["mains"]["k"].var() / var_u),
             "xy": float(terms["pairs"][("x", "y")].var() / var_u),
             "xk": float(terms["pairs"][("x", "k")].var() / var_u),
             "yk": float(terms["pairs"][("y", "k")].var() / var_u),
             "xyk": float(terms["triplet"].var() / var_u)}
    total = sum(sobol.values())
    print(f"  Sobol indices (sum should be ~1.0):")
    for s, v in sobol.items():
        print(f"    S_{s:<5} = {v:.4f}")
    print(f"    sum     = {total:.4f}")

    # Build truncations and measure achievable floor
    u_c = terms["u_centered"]
    norm_full = float(np.linalg.norm(u_c))

    def truncation_residual_rel_l2(subsets_included: list) -> float:
        """Sum over subsets in `subsets_included`, subtract from u_c, return rel-L²."""
        approx = np.zeros_like(u_c)
        if ("x",) in subsets_included: approx += terms["mains"]["x"]
        if ("y",) in subsets_included: approx += terms["mains"]["y"]
        if ("k",) in subsets_included: approx += terms["mains"]["k"]
        if ("x", "y") in subsets_included: approx += terms["pairs"][("x", "y")]
        if ("x", "k") in subsets_included: approx += terms["pairs"][("x", "k")]
        if ("y", "k") in subsets_included: approx += terms["pairs"][("y", "k")]
        if ("x", "y", "k") in subsets_included: approx += terms["triplet"]
        err = u_c - approx
        return float(np.linalg.norm(err) / (norm_full + 1e-12))

    architectures = {
        "order_0_constant": [],
        "order_1_mains": [("x",), ("y",), ("k",)],
        "additive_xy_plus_k": [("x",), ("y",), ("k",), ("x", "y")],   # f_xy + f_k
        "order_2_all_pairs": [("x",), ("y",), ("k",), ("x", "y"), ("x", "k"), ("y", "k")],
        "order_3_full": [("x",), ("y",), ("k",), ("x", "y"), ("x", "k"), ("y", "k"), ("x", "y", "k")],
    }

    pred_floor = {
        "order_0_constant": 1.0,
        "order_1_mains": float(np.sqrt(sobol["xy"] + sobol["xk"] + sobol["yk"] + sobol["xyk"])),
        "additive_xy_plus_k": float(np.sqrt(sobol["xk"] + sobol["yk"] + sobol["xyk"])),
        "order_2_all_pairs": float(np.sqrt(sobol["xyk"])),
        "order_3_full": 0.0,
    }

    print(f"\n{'Architecture':<24} {'predicted floor':>17} {'measured rel-L²':>17}  {'agreement':>11}")
    results = {}
    for name, subsets in architectures.items():
        meas = truncation_residual_rel_l2(subsets)
        pred = pred_floor[name]
        agree = (abs(meas - pred) < 0.02) if pred > 0 else (meas < 0.05)
        print(f"{name:<24} {pred:>17.4f} {meas:>17.4f}  {'✓' if agree else '✗':>11}")
        results[name] = {"predicted_floor": pred, "measured_rel_l2": meas,
                          "agreement": bool(agree)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "N": args.N, "n_bins": args.n_bins, "seed": args.seed,
        "var_u_ref": var_u, "f0": terms["f0"], "sobol": sobol,
        "sobol_sum": total,
        "predictions": pred_floor, "results": results,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
