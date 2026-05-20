"""HDMR truncation floors: post-hoc architectural complexity bound.

For 2D Helm LC-PINN, project the trained network onto subspaces of bounded
ANOVA order. For each order k ∈ {0, 1, 2, 3}, compute rel-L² of the order-k
truncation u^{(k)} vs the full LC-PINN u_θ.

The order-k truncation u^{(k)} = sum_{|S| ≤ k} f_S includes only ANOVA terms of
order ≤ k. By orthogonality, ||u_θ - u^{(k)}||² / ||u_θ||² = sum_{|S| > k} S_S,
exactly. So the floor that an architecturally-order-k network cannot beat is
sqrt(sum_{|S| > k} S_S).

Predictions (using MC-Sobol gold from results_mc_megaN_seed0.json):
  S_x = 0.066, S_y = 0.066, S_k = 0.075
  S_xy = 0.219, S_xk = 0.075, S_yk = 0.073
  S_xyk = 0.426

  Order 0 (constant in everything):  floor = sqrt(1 - mean²-rel) ≈ 1.0
  Order 1 (mains only):              floor = sqrt(S_xy + S_xk + S_yk + S_xyk)
                                            = sqrt(0.793) = 0.890
  Order 2 (mains + pairs):           floor = sqrt(S_xyk) = 0.653
  Order 3 (everything):              floor = 0

We MEASURE these floors empirically by:
  1. Fit Fourier joint HDMR on LC-PINN samples (we already have this).
  2. Compute u^{(k)} = mean + sum of order-≤-k terms, evaluated on val set.
  3. Compute ||u_θ - u^{(k)}|| / ||u_θ - mean||.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_REPO_ANOVA = _HERE.parent.parent
_REPO_THESIS_CODE = _REPO_ANOVA.parent / "thesis" / "code"
sys.path.insert(0, str(_REPO_ANOVA))
sys.path.insert(0, str(_REPO_THESIS_CODE))

from lc_anova.core.joint_hdmr import JointHDMR  # noqa
from lc_anova.pipelines.helmholtz_2d import load_lc_pinn, evaluate_lc_pinn_batch  # noqa

CK = _REPO_THESIS_CODE / "checkpoints"
RESULTS = _REPO_ANOVA / "lc_anova" / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default=str(CK / "lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt"))
    ap.add_argument("--N-train", type=int, default=200_000)
    ap.add_argument("--N-val", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="lc_anova/results/hdmr_truncation_floors.json")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, _ = load_lc_pinn(args.checkpoint, device)

    rng = np.random.default_rng(args.seed)
    print(f"Sampling N_train={args.N_train:,}  N_val={args.N_val:,}")
    X_tr = rng.uniform(0, 1, size=(args.N_train, 3)).astype(np.float32)
    X_tr[:, 2] = X_tr[:, 2] * 2 - 1.0  # k in [-1, 1]
    X_va = rng.uniform(0, 1, size=(args.N_val, 3)).astype(np.float32)
    X_va[:, 2] = X_va[:, 2] * 2 - 1.0

    x_tr = torch.tensor(X_tr[:, :2], device=device)
    k_tr = torch.tensor(X_tr[:, 2:3], device=device)
    x_va = torch.tensor(X_va[:, :2], device=device)
    k_va = torch.tensor(X_va[:, 2:3], device=device)
    with torch.no_grad():
        y_tr = evaluate_lc_pinn_batch(model, x_tr, k_tr)
        y_va = evaluate_lc_pinn_batch(model, x_va, k_va)

    print("Fitting Fourier joint HDMR order-3")
    jh = JointHDMR(dim_x=2, dim_lambda=1, hidden=64, layers=2,
                   max_order=3, use_fourier=True, num_freqs=4)
    jh.fit(x_tr, k_tr, y_tr, phase1_epochs=40, phase2_epochs=120, log_every=40)

    # Evaluate per-term components on val set
    print("Evaluating per-order truncations on val set")
    terms = jh.evaluate_terms(x_va, k_va)
    # main_terms: (N, d) — per-sample value of centered f_k(z_k)
    # pair_terms: (N, n_pairs) — per-sample value of f_{ij}(z_i, z_j)
    # triplet_terms: (N,) — f_{xyk}
    y_va_centered = (y_va - jh.y_mean).cpu().numpy()
    var_full = float(np.var(y_va_centered))
    norm_full = float(np.linalg.norm(y_va_centered))

    mains = terms["main_terms"]            # (N, 3)
    pairs = terms["pair_terms"]            # (N, 3)
    triplet = terms["triplet_terms"]       # (N,) or None

    truncations = {}
    for order in [0, 1, 2, 3]:
        approx = np.zeros_like(y_va_centered)
        if order >= 1:
            approx = approx + mains.sum(axis=1)
        if order >= 2:
            approx = approx + pairs.sum(axis=1)
        if order >= 3 and triplet is not None:
            approx = approx + triplet
        err = y_va_centered - approx
        rel_l2 = float(np.linalg.norm(err) / (norm_full + 1e-10))
        truncations[f"order_{order}"] = {
            "rel_l2": rel_l2,
            "var_remaining_ratio": float(np.var(err) / var_full),
        }
        print(f"  order {order}: rel-L² = {rel_l2:.4f}  (var-remaining = {np.var(err)/var_full:.4f})")

    # Sobol-derived predictions
    sobol = terms["sobol"]
    sobol_named = {}
    for s, v in sobol.items():
        if len(s) == 1:
            key = ["S_x", "S_y", "S_k"][s[0]]
        elif len(s) == 2:
            key = {(0, 1): "S_xy", (0, 2): "S_xk", (1, 2): "S_yk"}[tuple(sorted(s))]
        elif len(s) == 3:
            key = "S_xyk"
        sobol_named[key] = float(v)

    print(f"\nSobol-derived predictions for floor of each architectural restriction:")
    predicted = {}
    # order-0 floor: full variance (no terms used; only mean is subtracted)
    predicted["order_0"] = 1.0  # we subtracted the mean; the rest is the variance
    # order-1 (mains only) floor: cannot capture pairs or triplets
    high1 = sum(v for k, v in sobol_named.items() if k not in ["S_x", "S_y", "S_k"])
    predicted["order_1"] = float(np.sqrt(max(0.0, high1)))
    # order-2 (mains + pairs) floor: cannot capture triplet
    high2 = sobol_named.get("S_xyk", 0.0)
    predicted["order_2"] = float(np.sqrt(max(0.0, high2)))
    predicted["order_3"] = 0.0

    print(f"  Predicted floor  | Measured rel-L²")
    for order in [0, 1, 2, 3]:
        p = predicted[f"order_{order}"]
        m = truncations[f"order_{order}"]["rel_l2"]
        print(f"    order {order}: {p:.4f}      {m:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "N_train": args.N_train,
        "N_val": args.N_val,
        "sobol_indices_measured": sobol_named,
        "predicted_floor": predicted,
        "measured_truncation_rel_l2": {k: v["rel_l2"] for k, v in truncations.items()},
        "var_remaining_ratio": {k: v["var_remaining_ratio"] for k, v in truncations.items()},
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
