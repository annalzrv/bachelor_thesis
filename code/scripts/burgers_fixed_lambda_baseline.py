"""
Random-but-fixed λ baseline for Burgers.

Addresses the "equal-weights baseline is a strawman" gap in EXPERIMENTS.md:
LC-PINN uniform was trained with λ drawn independently from U(0,1)^4 with NO
sum-to-one constraint. Previous baseline fixed λ=(0.25,0.25,0.25,0.25) then
normalised. This script instead draws many random λ from the SAME distribution
LC-PINN saw during training and trains a separate fixed-weight PINN for each,
so we can see whether LC-PINN is actually better than an arbitrary fixed draw.

Usage (from thesis/code):
    python scripts/burgers_fixed_lambda_baseline.py --n-runs 10 --n-epochs 200000

Output:
    results/burgers_fixed_lambda_baseline.json
    checkpoints/burgers_fixed_lambda_run{i}.pt  for i in 0..n_runs-1
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pinns.baseline import FixedWeightPINN, train_fixed_pinn
from pinns.device import select_device, device_info
from pinns.equations import burgers as burg


HIDDEN_DIMS = [64, 64, 64, 64]
REPO = pathlib.Path(__file__).resolve().parent.parent


def draw_weights(rng: np.random.Generator) -> np.ndarray:
    """Draw λ ~ U(0,1)^4, matching LambdaSampler mode='uniform'."""
    return rng.uniform(0.0, 1.0, size=4).astype(np.float32)


def run_one(
    weights: np.ndarray,
    run_idx: int,
    batch,
    ref,
    device,
    n_epochs: int,
    lr: float,
) -> dict:
    """Train one fixed-λ PINN and return its rel-L2 per snapshot + config."""
    model = FixedWeightPINN(burg.DIM_PHYS, HIDDEN_DIMS).to(device)
    desc = f"Run {run_idx} λ={np.round(weights, 3).tolist()}"

    t0 = time.time()
    history = train_fixed_pinn(
        model,
        weights=list(weights),  # pde, bc, ic, data order
        batch=batch,
        device=device,
        loss_fn=burg.compute_losses_fixed,
        n_epochs=n_epochs,
        lr=lr,
        log_every=5_000,
        desc=desc,
        normalize=False,  # fair comparison to LC-PINN uniform
    )
    elapsed = time.time() - t0

    errors = burg.evaluate(model, None, ref, device)
    mean_rel = float(np.mean(list(errors.values())))

    ckpt_path = REPO / "checkpoints" / f"burgers_fixed_lambda_run{run_idx}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "weights": weights.tolist(),
            "rel_l2_per_snapshot": {float(k): float(v) for k, v in errors.items()},
            "mean_rel_l2": mean_rel,
            "elapsed_sec": elapsed,
            "n_epochs": n_epochs,
            "lr": lr,
        },
        ckpt_path,
    )

    return {
        "run_idx": run_idx,
        "weights": weights.tolist(),
        "rel_l2_per_snapshot": {float(k): float(v) for k, v in errors.items()},
        "mean_rel_l2": mean_rel,
        "elapsed_sec": elapsed,
        "checkpoint": str(ckpt_path.relative_to(REPO)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--n-epochs", type=int, default=200_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    device = select_device()
    print(f"Device: {device_info(device)}")
    print(f"Config: n_runs={args.n_runs}  n_epochs={args.n_epochs}  lr={args.lr}  seed={args.seed}\n")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print("Building Burgers reference & training batch…")
    ref = burg.compute_reference_solution()
    batch = burg.generate_training_data(ref, device=device)

    runs = []
    for i in range(args.n_runs):
        weights = draw_weights(rng)
        print(f"\n=== Run {i}/{args.n_runs-1}  λ = {np.round(weights, 3).tolist()} ===")
        result = run_one(
            weights=weights,
            run_idx=i,
            batch=batch,
            ref=ref,
            device=device,
            n_epochs=args.n_epochs,
            lr=args.lr,
        )
        runs.append(result)
        print(f"  rel-L2 mean = {result['mean_rel_l2']:.4f}  elapsed = {result['elapsed_sec']/60:.1f} min")

    means = [r["mean_rel_l2"] for r in runs]
    summary = {
        "config": vars(args),
        "runs": runs,
        "summary": {
            "n_runs": len(runs),
            "rel_l2_mean": float(np.mean(means)),
            "rel_l2_std":  float(np.std(means)),
            "rel_l2_min":  float(np.min(means)),
            "rel_l2_max":  float(np.max(means)),
            "rel_l2_median": float(np.median(means)),
        },
        "references": {
            "lc_pinn_uniform_rel_l2_mean": 0.0004,
            "lc_pinn_logspace_rel_l2_mean": 0.0009,
            "equal_weights_baseline_rel_l2_mean": 0.1472,
        },
    }

    out = REPO / "results" / "burgers_fixed_lambda_baseline.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n\nWrote {out.relative_to(REPO)}")
    print("\nSummary (fixed-λ random draws, NOT normalized):")
    print(f"  n_runs    = {summary['summary']['n_runs']}")
    print(f"  mean      = {summary['summary']['rel_l2_mean']:.4f}")
    print(f"  std       = {summary['summary']['rel_l2_std']:.4f}")
    print(f"  median    = {summary['summary']['rel_l2_median']:.4f}")
    print(f"  [min,max] = [{summary['summary']['rel_l2_min']:.4f}, {summary['summary']['rel_l2_max']:.4f}]")
    print(f"\nReferences:")
    print(f"  LC-PINN uniform       = 0.0004")
    print(f"  LC-PINN logspace      = 0.0009")
    print(f"  Equal-weights baseline = 0.1472")

    return 0


if __name__ == "__main__":
    sys.exit(main())
