"""Bootstrap confidence intervals for MC-Sobol estimates on the 2D Helm LC-PINN.

Resamples the per-sample MC-Sobol output arrays (Y_A, Y_B, Y_C^(i),
Y_C^(i, j)) with replacement and recomputes Sobol indices on each
resample. Reports a 95% CI for every Sobol index.

Run:
    python -m lc_anova.pipelines.bootstrap_ci \\
        --checkpoint .../lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt \\
        --N 200000 --n-boot 500
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
from lc_anova.core.mc_sobol import helmholtz_2d_sampler  # noqa

CK_DIR = _REPO_THESIS_CODE / "checkpoints"
RESULTS = _REPO_ANOVA / "lc_anova" / "results"


def mc_sobol_components(model_fn, sampler, N: int, d: int = 3, seed: int = 42) -> dict:
    """Run Saltelli MC-Sobol and return the per-sample arrays needed for bootstrap.

    Returns:
      Y_A, Y_B           — (N,) base evaluations
      Y_C[i]             — (N,) evaluation with column i swapped
      Y_C_pair[(i, j)]   — (N,) evaluation with columns (i, j) swapped
    """
    rng = np.random.default_rng(seed)
    A = sampler(rng, N).astype(np.float64)
    B = sampler(rng, N).astype(np.float64)
    Y_A = np.asarray(model_fn(A), dtype=np.float64)
    Y_B = np.asarray(model_fn(B), dtype=np.float64)

    Y_C = {}
    for i in range(d):
        C = A.copy(); C[:, i] = B[:, i]
        Y_C[i] = np.asarray(model_fn(C), dtype=np.float64)
    Y_C_pair = {}
    for i, j in itertools.combinations(range(d), 2):
        C = A.copy(); C[:, i] = B[:, i]; C[:, j] = B[:, j]
        Y_C_pair[(i, j)] = np.asarray(model_fn(C), dtype=np.float64)

    return {"Y_A": Y_A, "Y_B": Y_B, "Y_C": Y_C, "Y_C_pair": Y_C_pair}


def sobol_from_components(components: dict, indices: np.ndarray, d: int = 3) -> dict:
    """Compute Sobol indices using only the rows indicated by `indices`."""
    Y_A = components["Y_A"][indices]
    Y_B = components["Y_B"][indices]
    f0 = 0.5 * (Y_A.mean() + Y_B.mean())
    var_Y = 0.5 * (Y_A.var() + Y_B.var())
    if var_Y <= 0:
        return None

    V = {}
    V_T = {}
    for i in range(d):
        Y_C = components["Y_C"][i][indices]
        V[(i,)] = float(np.mean(Y_B * Y_C) - f0 * f0)
        V_T[(i,)] = float(0.5 * np.mean((Y_A - Y_C) ** 2))

    V_closed = {}
    for i, j in itertools.combinations(range(d), 2):
        Y_C_ij = components["Y_C_pair"][(i, j)][indices]
        V_closed[(i, j)] = float(np.mean(Y_B * Y_C_ij) - f0 * f0)

    V_pair = {(i, j): V_closed[(i, j)] - V[(i,)] - V[(j,)] for i, j in V_closed}
    V_triplet = var_Y - sum(V.values()) - sum(V_pair.values())

    return {
        "var_Y": var_Y,
        "S_x": V[(0,)] / var_Y, "S_y": V[(1,)] / var_Y, "S_k": V[(2,)] / var_Y,
        "ST_x": V_T[(0,)] / var_Y, "ST_y": V_T[(1,)] / var_Y, "ST_k": V_T[(2,)] / var_Y,
        "S_xy": V_pair[(0, 1)] / var_Y,
        "S_xk": V_pair[(0, 2)] / var_Y,
        "S_yk": V_pair[(1, 2)] / var_Y,
        "S_xyk": V_triplet / var_Y,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default=str(CK_DIR / "lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt"))
    ap.add_argument("--N", type=int, default=200_000)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="lc_anova/results/bootstrap_ci_helm2d_seed0.json")
    args = ap.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Loading {args.checkpoint}")
    model, _ = load_lc_pinn(args.checkpoint, device)

    @torch.no_grad()
    def model_fn(z_np: np.ndarray) -> np.ndarray:
        z = torch.tensor(z_np, dtype=torch.float32, device=device)
        u = evaluate_lc_pinn_batch(model, z[:, :2], z[:, 2:3])
        return u.detach().cpu().numpy()

    print(f"Running Saltelli MC-Sobol with N={args.N:,} (caching per-sample arrays)")
    components = mc_sobol_components(model_fn, helmholtz_2d_sampler, N=args.N, d=3, seed=args.seed)
    print(f"  ran model {2 + 3 + 3} times × {args.N:,} samples = {(2 + 3 + 3) * args.N:,} evals")

    # Point estimate (full sample)
    full_idx = np.arange(args.N)
    point_est = sobol_from_components(components, full_idx, d=3)
    print("\nPoint estimate (full sample):")
    for k, v in point_est.items():
        if k != "var_Y":
            print(f"  {k:<8} {v:>8.4f}")

    # Bootstrap loop
    print(f"\nBootstrap: {args.n_boot} resamples")
    rng = np.random.default_rng(args.seed + 99)
    keys = [k for k in point_est if k != "var_Y"]
    bootstrap_dist = {k: [] for k in keys}
    for b in range(args.n_boot):
        idx = rng.integers(0, args.N, size=args.N)
        est = sobol_from_components(components, idx, d=3)
        if est is None:
            continue
        for k in keys:
            bootstrap_dist[k].append(est[k])
        if (b + 1) % 100 == 0:
            print(f"  ...{b+1}/{args.n_boot}")

    # Compute CIs
    print("\n95% bootstrap CIs:")
    print(f"  {'index':<8} {'mean':>8} {'std':>8} {'CI_lo':>8} {'CI_hi':>8}")
    ci = {}
    for k in keys:
        arr = np.array(bootstrap_dist[k])
        mean = float(arr.mean())
        std = float(arr.std())
        lo = float(np.percentile(arr, 2.5))
        hi = float(np.percentile(arr, 97.5))
        ci[k] = {"mean": mean, "std": std, "ci_lo": lo, "ci_hi": hi,
                 "point_est": point_est[k]}
        print(f"  {k:<8} {mean:>8.4f} {std:>8.4f} {lo:>8.4f} {hi:>8.4f}")

    # Save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "N": args.N,
        "n_boot": args.n_boot,
        "point_estimate": {k: point_est[k] for k in keys},
        "var_Y": point_est["var_Y"],
        "bootstrap_ci": ci,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
