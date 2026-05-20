"""Synthetic-truth validation: f(x1, x2, x3) with closed-form Sobol structure.

Construction: let phi(z) = z - 1/2 on z ∈ [0, 1]; phi has mean 0, Var(phi) = 1/12.
  f(x, y, z) = x + y + 2*phi(x)*phi(y) + 12*phi(x)*phi(y)*phi(z)
             = 1 + phi(x) + phi(y) + 2*phi(x)*phi(y) + 12*phi(x)*phi(y)*phi(z)

ANOVA components (all orthogonal under uniform [0,1]^3):
  f_0   = 1
  f_x   = phi(x);            Var = 1/12
  f_y   = phi(y);            Var = 1/12
  f_z   = 0
  f_xy  = 2*phi(x)*phi(y);   Var = 4/144 = 1/36
  f_xz, f_yz = 0
  f_xyz = 12*phi(x)*phi(y)*phi(z);  Var = 144/1728 = 1/12

Total variance = 1/12 + 1/12 + 1/36 + 1/12 = 10/36 = 5/18.

Closed-form Sobol indices (in [0, 1]):
  S_x  = (1/12) / (5/18) = 0.300
  S_y  = 0.300
  S_z  = 0.000
  S_xy = (1/36) / (5/18) = 0.100
  S_xz = 0.000
  S_yz = 0.000
  S_xyz = (1/12) / (5/18) = 0.300

We fit Fourier joint HDMR to N=100k samples and check recovery.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_REPO_ANOVA = _HERE.parent.parent
sys.path.insert(0, str(_REPO_ANOVA))

from lc_anova.core.joint_hdmr import JointHDMR  # noqa

RESULTS = _REPO_ANOVA / "lc_anova" / "results"


ANALYTIC = {
    "S_x": 0.30, "S_y": 0.30, "S_z": 0.0,
    "S_xy": 0.10, "S_xz": 0.0, "S_yz": 0.0,
    "S_xyz": 0.30,
}


def truth(x):
    """x: (N, 3) in [0,1]^3. Returns (N,)."""
    p = x - 0.5
    px, py, pz = p[:, 0], p[:, 1], p[:, 2]
    return 1.0 + px + py + 2.0 * px * py + 12.0 * px * py * pz


def main():
    rng = np.random.default_rng(42)
    N_tr, N_va = 100_000, 50_000
    X_tr = rng.uniform(0, 1, size=(N_tr, 3)).astype(np.float32)
    X_va = rng.uniform(0, 1, size=(N_va, 3)).astype(np.float32)
    y_tr = truth(X_tr).astype(np.float32)
    y_va = truth(X_va).astype(np.float32)

    print(f"Synthetic truth: N_train={N_tr}, N_val={N_va}, d=3 with x, y, lambda=z")
    print(f"  analytic Sobol: {ANALYTIC}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Use dim_x=2, dim_lambda=1 (treat z as the parameter axis)
    jh = JointHDMR(dim_x=2, dim_lambda=1, hidden=32, layers=2,
                   max_order=3, use_fourier=True, num_freqs=4)
    xt = torch.tensor(X_tr[:, :2], device=device)
    lt = torch.tensor(X_tr[:, 2:3], device=device)
    yt = torch.tensor(y_tr, device=device)
    # Lighter schedule — phase 3 overfits past ep=120 on this synthetic.
    jh.fit(xt, lt, yt, phase1_epochs=40, phase2_epochs=120, phase3_epochs=120,
           log_every=40)

    xv = torch.tensor(X_va[:, :2], device=device)
    lv = torch.tensor(X_va[:, 2:3], device=device)
    yv = torch.tensor(y_va, device=device) - jh.y_mean
    z_va = torch.cat([xv, lv], dim=1)
    jh.model.eval()
    with torch.no_grad():
        if jh._has_triplet:
            pred, *_ = jh.model(z_va, include_pairs=True, include_triplet=True, purify=True)
        else:
            pred, _, _ = jh.model(z_va, include_pairs=True, purify=True)
    val_rel = (torch.sqrt(F.mse_loss(pred, yv)) / yv.std()).item()
    print(f"  val rel-RMSE = {val_rel:.4f}")

    terms = jh.evaluate_terms(xv, lv)
    # Map subset (x_indices in 0..1, lambda_indices = 0 → z) to named keys
    sobol_named = {}
    for subset, val in terms["sobol"].items():
        if len(subset) == 1:
            i = subset[0]
            name = ["S_x", "S_y", "S_z"][i]
        elif len(subset) == 2:
            tup = tuple(sorted(subset))
            name = {(0, 1): "S_xy", (0, 2): "S_xz", (1, 2): "S_yz"}[tup]
        elif len(subset) == 3:
            name = "S_xyz"
        else:
            continue
        sobol_named[name] = float(val)

    print(f"\nMeasured Sobol indices:")
    diffs = {}
    for k in sorted(ANALYTIC):
        meas = sobol_named.get(k, 0.0)
        true = ANALYTIC[k]
        diff = meas - true
        diffs[k] = diff
        print(f"  {k:<6}  measured={meas:>7.4f}   analytic={true:>7.4f}   diff={diff:+.4f}")

    max_abs_diff = max(abs(v) for v in diffs.values())
    print(f"\nMax |diff| = {max_abs_diff:.4f}")
    print(f"Pass (max < 0.05): {max_abs_diff < 0.05}")

    out = RESULTS / "synthetic_truth.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_train": N_tr, "n_val": N_va,
        "analytic": ANALYTIC,
        "measured": sobol_named,
        "diffs": diffs,
        "max_abs_diff": max_abs_diff,
        "val_rel_rmse": val_rel,
        "pass_within_0p05": bool(max_abs_diff < 0.05),
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
