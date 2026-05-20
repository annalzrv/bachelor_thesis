"""K-eval sweep ablation: rel-L2 vs K for the trained LC-PINN.

Loads existing LC-PINN Burgers checkpoints (`lc_pinn_burgers_seeds_seed{s}.pt`)
and runs K-shot inference at K ∈ {25, 50, 100, 200, 400}. No retraining.

Usage:
    python scripts/ablation_k_eval.py \\
        --K-values 25 50 100 200 400 --seeds 0 1 2 3

Output:
    results/ablation_k_eval.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pinns.device import select_device, device_info
from pinns.equations import burgers as burg
from pinns.model import LossConditionalPINN


HIDDEN_DIMS = [64, 64, 64, 64]
REPO = pathlib.Path(__file__).resolve().parent.parent


def evaluate_at_K(model, ref, device, K: int, seed: int = 0) -> tuple[float, float]:
    """Mean and std of rel-L² over K random λ samples drawn from the
    training prior (LambdaSampler mode='uniform': λ_p ~ U(0,1)^4,
    network input = log λ_p)."""
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(K):
        lam_p = rng.uniform(0.0, 1.0, size=burg.DIM_LAMBDA).astype(np.float32)
        log_lam = torch.tensor(np.log(np.clip(lam_p, 1e-8, None)), device=device)
        per_snap = burg.evaluate(model, log_lam, ref, device)
        errs.append(float(np.mean(list(per_snap.values()))))
    return float(np.mean(errs)), float(np.std(errs))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--K-values", type=int, nargs="+",
                   default=[25, 50, 100, 200, 400])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--ckpt-pattern", type=str,
                   default="checkpoints/lc_pinn_burgers_seeds_seed{seed}.pt")
    args = p.parse_args()

    device = select_device()
    print(f"Device: {device_info(device)}")
    print(f"K-values: {args.K_values}, seeds: {args.seeds}\n")

    print("Building Burgers reference …")
    ref = burg.compute_reference_solution()

    out: dict[int, dict] = {}
    for s in args.seeds:
        ckpt_path = REPO / args.ckpt_pattern.format(seed=s)
        if not ckpt_path.exists():
            print(f"[seed {s}] checkpoint missing: {ckpt_path}; skip")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = LossConditionalPINN(
            dim_phys=burg.DIM_PHYS, dim_lambda=burg.DIM_LAMBDA,
            hidden_dims=HIDDEN_DIMS,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        per_K = {}
        for K in args.K_values:
            mean, std = evaluate_at_K(model, ref, device, K=K, seed=999 + s)
            per_K[K] = {"mean": mean, "std": std}
            print(f"  [seed {s}] K={K:>4d}  rel-L2 mean={mean:.4e}  std={std:.4e}")
        out[s] = per_K

    # Aggregate across seeds: mean ± std at each K.
    agg: dict[int, dict] = {}
    for K in args.K_values:
        means = [out[s][K]["mean"] for s in out if K in out[s]]
        agg[K] = {
            "n_seeds": len(means),
            "mean_over_seeds": float(np.mean(means)) if means else None,
            "std_over_seeds":  float(np.std(means))  if means else None,
            "min": float(np.min(means)) if means else None,
            "max": float(np.max(means)) if means else None,
        }

    out_path = REPO / "results" / "ablation_k_eval.json"
    out_path.write_text(json.dumps({
        "config": vars(args), "per_seed": out, "summary": agg,
    }, indent=2))
    print(f"\nWrote {out_path.relative_to(REPO)}")
    for K, v in agg.items():
        if v["mean_over_seeds"] is not None:
            print(f"  K={K:>4d}  rel-L2 = {v['mean_over_seeds']:.4e} ± {v['std_over_seeds']:.4e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
