"""Edge cases: robustness of the Sobol-derived approximation bound under
non-uniform priors on the conditioning parameter, including OOD.

For 2D Helmholtz LC-PINN, we already showed (in application_dim_reduction.py)
that under the training prior k ~ U(-1, 1):
  - Sobol predicts cost-of-dropping-k ≈ 0.80
  - Measured marginal approximation rel-L² ≈ 0.79

Here we ask: does this prediction-vs-measurement relationship hold under
*different* priors on k, including out-of-distribution priors? Sobol indices
depend on the input distribution, so:
  1. For each prior, recompute Sobol indices restricted to that distribution.
  2. For each prior, build u_bar(x, y) = E_{k ~ prior}[u_theta(x, y; k)] by MC.
  3. Measure rel-L² of marginal vs full LC-PINN over the prior's support.
  4. Verify: measured rel-L² ≤ sqrt(sum_{S contains k} S_S^{prior}).

Priors tested:
  - k ~ U(-1, 1)        (training-distribution baseline)
  - k ~ U(-0.3, 0.3)    (near zero — k=0 means no wave, smooth Helmholtz)
  - k ~ U(0.7, 1.0)     (near training boundary)
  - k ~ U(1.0, 1.5)     (OOD: extrapolation by 50%)
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO_ANOVA = _HERE.parent.parent
_REPO_THESIS_CODE = _REPO_ANOVA.parent / "thesis" / "code"
sys.path.insert(0, str(_REPO_ANOVA))
sys.path.insert(0, str(_REPO_THESIS_CODE))

from lc_anova.pipelines.helmholtz_2d import load_lc_pinn, evaluate_lc_pinn_batch  # noqa

CK_DIR = _REPO_THESIS_CODE / "checkpoints"
RESULTS = _REPO_ANOVA / "lc_anova" / "results"


def make_sampler(k_lo: float, k_hi: float):
    """Saltelli sampler for (x, y, k) with x, y ~ U(0, 1) and k ~ U(k_lo, k_hi)."""
    def sampler(rng, N):
        x = rng.uniform(0.0, 1.0, size=(N,))
        y = rng.uniform(0.0, 1.0, size=(N,))
        k = rng.uniform(k_lo, k_hi, size=(N,))
        return np.stack([x, y, k], axis=1)
    return sampler


def saltelli_sobol(model_fn, sampler, N: int, d: int = 3, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    A = sampler(rng, N).astype(np.float64)
    B = sampler(rng, N).astype(np.float64)
    Y_A = np.asarray(model_fn(A), dtype=np.float64)
    Y_B = np.asarray(model_fn(B), dtype=np.float64)
    f0 = 0.5 * (Y_A.mean() + Y_B.mean())
    var_Y = 0.5 * (Y_A.var() + Y_B.var())

    V = {}
    for i in range(d):
        C = A.copy(); C[:, i] = B[:, i]
        Y_C = np.asarray(model_fn(C), dtype=np.float64)
        V[(i,)] = float(np.mean(Y_B * Y_C) - f0 * f0)

    V_closed = {}
    for i, j in itertools.combinations(range(d), 2):
        C = A.copy(); C[:, i] = B[:, i]; C[:, j] = B[:, j]
        Y_C = np.asarray(model_fn(C), dtype=np.float64)
        V_closed[(i, j)] = float(np.mean(Y_B * Y_C) - f0 * f0)

    V_pair = {(i, j): V_closed[(i, j)] - V[(i,)] - V[(j,)] for i, j in V_closed}
    V_triplet = var_Y - sum(V.values()) - sum(V_pair.values())

    return {
        "var_Y": float(var_Y),
        "S_x": V[(0,)] / var_Y, "S_y": V[(1,)] / var_Y, "S_k": V[(2,)] / var_Y,
        "S_xy": V_pair[(0, 1)] / var_Y,
        "S_xk": V_pair[(0, 2)] / var_Y,
        "S_yk": V_pair[(1, 2)] / var_Y,
        "S_xyk": V_triplet / var_Y,
    }


def marginal_approx_cost(model_fn, k_lo: float, k_hi: float,
                          n_k: int = 200, n_x: int = 64, n_test: int = 20_000,
                          seed: int = 42) -> dict:
    """Build u_bar(x, y) by averaging over k ~ U(k_lo, k_hi), then measure
    rel-L² of u_full - u_bar on a test set drawn from the same prior."""
    xs = np.linspace(0.0, 1.0, n_x)
    ys = np.linspace(0.0, 1.0, n_x)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    xy_flat = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)

    rng = np.random.default_rng(seed)
    k_samples = rng.uniform(k_lo, k_hi, size=n_k)
    u_bar = np.zeros(xy_flat.shape[0])
    for k in k_samples:
        Z = np.concatenate([xy_flat, np.full((xy_flat.shape[0], 1), k)], axis=1)
        u_k = np.asarray(model_fn(Z))
        u_bar += u_k
    u_bar /= n_k
    u_bar_grid = u_bar.reshape(n_x, n_x)

    # Test set under same prior
    rng2 = np.random.default_rng(seed + 1)
    xy_test = rng2.uniform(0.0, 1.0, size=(n_test, 2))
    k_test = rng2.uniform(k_lo, k_hi, size=(n_test, 1))
    Z_test = np.concatenate([xy_test, k_test], axis=1)
    u_full = np.asarray(model_fn(Z_test))

    # Bilinear interp of u_bar at xy_test
    xv = xy_test[:, 0] * (n_x - 1)
    yv = xy_test[:, 1] * (n_x - 1)
    x0 = np.clip(np.floor(xv).astype(int), 0, n_x - 2)
    y0 = np.clip(np.floor(yv).astype(int), 0, n_x - 2)
    dx = xv - x0; dy = yv - y0
    u_bar_at_test = (
        (1 - dx) * (1 - dy) * u_bar_grid[x0, y0]
        + dx * (1 - dy) * u_bar_grid[x0 + 1, y0]
        + (1 - dx) * dy * u_bar_grid[x0, y0 + 1]
        + dx * dy * u_bar_grid[x0 + 1, y0 + 1]
    )
    err = u_full - u_bar_at_test
    rel_l2 = float(np.linalg.norm(err) / (np.linalg.norm(u_full) + 1e-10))
    return {"rel_l2": rel_l2,
            "var_full": float(u_full.var()),
            "var_err": float(err.var())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default=str(CK_DIR / "lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt"))
    ap.add_argument("--N-sobol", type=int, default=80_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="lc_anova/results/edge_cases_priors.json")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, _ = load_lc_pinn(args.checkpoint, device)

    @torch.no_grad()
    def model_fn(z_np: np.ndarray) -> np.ndarray:
        z = torch.tensor(z_np, dtype=torch.float32, device=device)
        u = evaluate_lc_pinn_batch(model, z[:, :2], z[:, 2:3])
        return u.detach().cpu().numpy()

    priors = [
        ("training", -1.0, 1.0),
        ("near_zero", -0.3, 0.3),
        ("near_boundary", 0.7, 1.0),
        ("ood_extrapolation", 1.0, 1.5),
    ]

    all_results = {}
    for name, k_lo, k_hi in priors:
        print(f"\n=== Prior '{name}'  k ~ U({k_lo}, {k_hi})  ===")
        print(f"  Saltelli MC-Sobol N={args.N_sobol:,}")
        sobol = saltelli_sobol(model_fn, make_sampler(k_lo, k_hi),
                                N=args.N_sobol, d=3, seed=args.seed)
        S_total_k = sobol["S_k"] + sobol["S_xk"] + sobol["S_yk"] + sobol["S_xyk"]
        predicted = float(np.sqrt(max(0.0, S_total_k)))
        print(f"  S_k={sobol['S_k']:.3f}  S_xk={sobol['S_xk']:.3f}  "
              f"S_yk={sobol['S_yk']:.3f}  S_xyk={sobol['S_xyk']:.3f}")
        print(f"  total k-contribution = {S_total_k:.3f}  →  predicted rel-L² = {predicted:.3f}")

        print(f"  Measuring marginal approximation cost")
        marg = marginal_approx_cost(model_fn, k_lo, k_hi,
                                     n_k=200, n_x=64, n_test=20_000, seed=args.seed + 7)
        print(f"  measured rel-L² = {marg['rel_l2']:.3f}")
        print(f"  bound holds: {marg['rel_l2'] <= predicted + 0.02}")

        all_results[name] = {
            "k_range": [k_lo, k_hi],
            "sobol": sobol,
            "total_k_contribution": S_total_k,
            "predicted_rel_l2": predicted,
            "measured_rel_l2": marg["rel_l2"],
            "var_full": marg["var_full"],
            "var_err": marg["var_err"],
            "bound_holds": bool(marg["rel_l2"] <= predicted + 0.02),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {out}")

    print("\n=== SUMMARY ===")
    print(f"  {'prior':<20} {'k range':<14} {'pred rel-L²':>12} {'meas rel-L²':>12} {'bound':>8}")
    for name, _, _ in priors:
        r = all_results[name]
        k_range = f"[{r['k_range'][0]:+.1f}, {r['k_range'][1]:+.1f}]"
        print(f"  {name:<20} {k_range:<14} {r['predicted_rel_l2']:>12.3f} "
              f"{r['measured_rel_l2']:>12.3f} {'✓' if r['bound_holds'] else '✗':>8}")


if __name__ == "__main__":
    main()
