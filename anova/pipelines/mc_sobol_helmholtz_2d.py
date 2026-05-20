"""Run Saltelli MC-Sobol on a trained 2D Helmholtz LC-PINN.

This is the model-agnostic baseline that the Fourier HDMR's Sobol
numbers should agree with (in the asymptotic-N limit).

Run:
    cd /Users/anna/Desktop/research/anova
    python -m lc_anova.pipelines.mc_sobol_helmholtz_2d \\
        --checkpoint /Users/anna/Desktop/research/thesis/code/checkpoints/lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt \\
        --N 50000
"""

from __future__ import annotations

import argparse
import json
import os
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

from lc_anova.pipelines.helmholtz_2d import (  # noqa: E402
    load_lc_pinn,
    evaluate_lc_pinn_batch,
)
from lc_anova.core.mc_sobol import mc_sobol_full, helmholtz_2d_sampler  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--N", type=int, default=20_000, help="MC samples per matrix")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="lc_anova/results")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"mc_sobol_{Path(args.checkpoint).stem}"
    print(f"== MC-Sobol on 2D Helmholtz LC-PINN  (tag={tag}, N={args.N}) ==")

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, meta = load_lc_pinn(args.checkpoint, device)
    print(f"  cond={meta['conditioning']}  hidden={meta['hidden_dims']}  params={meta['n_params']}")

    @torch.no_grad()
    def model_fn(z_np: np.ndarray) -> np.ndarray:
        z = torch.tensor(z_np, dtype=torch.float32, device=device)
        xy = z[:, :2]
        k_norm = z[:, 2:3]
        u = evaluate_lc_pinn_batch(model, xy, k_norm)
        return u.detach().cpu().numpy()

    out = mc_sobol_full(model_fn, helmholtz_2d_sampler, N=args.N, d=3, seed=args.seed)

    name_for = {0: "x", 1: "y", 2: "k"}
    print(f"\nVar(Y) (LC-PINN output): {out['var_Y']:.6f}")
    print(f"f0 (mean of Y):          {out['f0']:.6f}")

    print("\nFirst-order Sobol indices:")
    print(f"  {'subset':<6} {'S':>9}")
    for k in [(0,), (1,), (2,)]:
        name = name_for[k[0]]
        print(f"  {name:<6} {out['S_first'][k]:>9.4f}")

    print("\nTotal Sobol indices:")
    print(f"  {'subset':<6} {'ST':>9}")
    for k in [(0,), (1,), (2,)]:
        name = name_for[k[0]]
        print(f"  {name:<6} {out['S_total'][k]:>9.4f}")

    print("\nPure pair Sobol indices:")
    print(f"  {'subset':<8} {'S_ij':>9}")
    for k in [(0, 1), (0, 2), (1, 2)]:
        name = "/".join(name_for[a] for a in k)
        print(f"  {name:<8} {out['S_pair'][k]:>9.4f}")

    print(f"\nTriplet (residual): S_(x,y,k) = {out['S_triplet']:.4f}")

    # Save JSON
    payload = {
        "checkpoint": str(args.checkpoint),
        "tag": tag,
        "N": args.N,
        "seed": args.seed,
        "var_Y": out["var_Y"],
        "f0": out["f0"],
        "S_first":   {",".join(name_for[a] for a in k): v for k, v in out["S_first"].items()},
        "S_total":   {",".join(name_for[a] for a in k): v for k, v in out["S_total"].items()},
        "S_pair":    {",".join(name_for[a] for a in k): v for k, v in out["S_pair"].items()},
        "S_triplet": out["S_triplet"],
    }
    out_json = out_dir / f"results_{tag}.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
