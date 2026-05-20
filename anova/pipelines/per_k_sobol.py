"""Per-parameter Sobol indices for a parametric LC-PINN.

For the 2D Helmholtz LC-PINN u_theta(x, y; k), at each fixed k we have
a 2-D function over (x, y). We can compute MC-Sobol on (x, y) at that k:

  S_x(k), S_y(k)  — first-order over spatial axes at this k
  S_T_x(k), S_T_y(k)  — total
  S_{x,y}(k)  — pair interaction at this k

Plotted as functions of k, these tell you HOW the spatial sensitivity
structure varies with the parameter — likely the most striking figure
in the paper.

Output is a JSON with per-k Sobol curves, plus an automatically rendered
side-by-side plot.

Run:
    cd /Users/anna/Desktop/research/anova
    python -m lc_anova.pipelines.per_k_sobol \\
        --checkpoint .../lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt \\
        --n-k 11 --N 10000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO_ANOVA = _HERE.parent
_REPO_THESIS_CODE = _REPO_ANOVA.parent / "thesis" / "code"
if str(_REPO_ANOVA) not in sys.path:
    sys.path.insert(0, str(_REPO_ANOVA))
if str(_REPO_THESIS_CODE) not in sys.path:
    sys.path.insert(0, str(_REPO_THESIS_CODE))

from lc_anova.pipelines.helmholtz_2d import load_lc_pinn, evaluate_lc_pinn_batch  # noqa
from lc_anova.core.mc_sobol import mc_sobol_full  # noqa
from pinns.equations import helmholtz_2d as helm  # noqa


def per_k_sobol_helm2d(checkpoint_path: str, k_values, N: int = 10000,
                       seed: int = 42) -> dict:
    """For each k in k_values, compute spatial-only Sobol (over x, y) on the LC-PINN.

    Returns:
        dict with k_values list and corresponding lists of S_x, S_y, S_T_x,
        S_T_y, S_pair, plus Var(Y|k).
    """
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, meta = load_lc_pinn(checkpoint_path, device)
    print(f"  loaded {Path(checkpoint_path).name}: cond={meta['conditioning']} hidden={meta['hidden_dims']}")

    def make_model_fn(k_value: float):
        k_norm_scalar = helm.k_to_norm(np.array([k_value], dtype=np.float32))[0]

        @torch.no_grad()
        def fn(xy_np: np.ndarray) -> np.ndarray:
            xy_t = torch.tensor(xy_np, dtype=torch.float32, device=device)
            k_t = torch.full((xy_np.shape[0], 1), float(k_norm_scalar),
                             dtype=torch.float32, device=device)
            u = evaluate_lc_pinn_batch(model, xy_t, k_t)
            return u.detach().cpu().numpy()
        return fn

    def spatial_sampler(rng, n):
        return rng.uniform(0.0, 1.0, size=(n, 2)).astype(np.float32)

    results = {
        "k_values": [float(k) for k in k_values],
        "var_Y_at_k": [],
        "S_x_at_k": [], "S_y_at_k": [],
        "S_T_x_at_k": [], "S_T_y_at_k": [],
        "S_xy_at_k": [],
    }
    for i, k_value in enumerate(k_values):
        out = mc_sobol_full(make_model_fn(float(k_value)), spatial_sampler,
                            N=N, d=2, seed=seed + i)
        results["var_Y_at_k"].append(out["var_Y"])
        results["S_x_at_k"].append(out["S_first"][(0,)])
        results["S_y_at_k"].append(out["S_first"][(1,)])
        results["S_T_x_at_k"].append(out["S_total"][(0,)])
        results["S_T_y_at_k"].append(out["S_total"][(1,)])
        results["S_xy_at_k"].append(out["S_pair"][(0, 1)])
        print(f"  k={k_value:5.2f}  Var(Y|k)={out['var_Y']:.4f}  "
              f"S_x={out['S_first'][(0,)]:.3f}  S_y={out['S_first'][(1,)]:.3f}  "
              f"S_xy={out['S_pair'][(0, 1)]:.3f}")
    return results


def render_per_k_plot(results: dict, out_path: Path, tag: str):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    k_vals = results["k_values"]
    ax = axes[0]
    ax.plot(k_vals, results["S_x_at_k"], "o-", label=r"$S_x(k)$", color="C0")
    ax.plot(k_vals, results["S_y_at_k"], "s-", label=r"$S_y(k)$", color="C1")
    ax.plot(k_vals, results["S_xy_at_k"], "^-", label=r"$S_{x,y}(k)$", color="C2")
    ax.set_xlabel("wavenumber  k"); ax.set_ylabel("Sobol index over (x, y)")
    ax.set_title(rf"Per-$k$ spatial Sobol ({tag})")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    ax = axes[1]
    ax.plot(k_vals, results["var_Y_at_k"], "o-", color="C3")
    ax.set_xlabel("wavenumber  k"); ax.set_ylabel(r"Var$(u | k)$")
    ax.set_title("Conditional output variance vs k")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-k", type=int, default=11, help="number of k values")
    ap.add_argument("--N", type=int, default=10_000, help="MC samples per k")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="lc_anova/results")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"per_k_{Path(args.checkpoint).stem}"

    k_values = np.linspace(helm.K_MIN, helm.K_MAX, args.n_k)
    print(f"== per-k Sobol on 2D Helmholtz LC-PINN  ({tag}, N={args.N} per k) ==")
    print(f"  k_values: {list(k_values.round(2))}")

    results = per_k_sobol_helm2d(args.checkpoint, k_values, N=args.N, seed=args.seed)

    out_json = out_dir / f"results_{tag}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_json}")

    out_png = out_dir / f"plot_{tag}.png"
    render_per_k_plot(results, out_png, tag=tag)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
