"""Test stability of Fourier HDMR Sobol indices under HDMR-training randomness.

For a fixed LC-PINN checkpoint, run the joint HDMR multiple times with
different RNG seeds (for HDMR network init, sample order, dropout, etc.).
Report mean +/- std of each Sobol index across HDMR runs.

This is *seed stability of the analysis method*, not the LC-PINN's seed
stability (which is already tested via MC-Sobol across LC-PINN seeds).

Run:
    python -m lc_anova.pipelines.hdmr_stability \\
        --pde helm2d --checkpoint .../lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt \\
        --n-runs 5
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

from lc_anova.core.joint_hdmr import JointHDMR  # noqa
from pinns.equations import helmholtz_2d as helm2d  # noqa
from pinns.equations import helmholtz as helm1d  # noqa
from pinns.equations import schrodinger_1d as schr  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", required=True, choices=["helm1d", "helm2d", "schr1d"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--phase1-epochs", type=int, default=30)
    ap.add_argument("--phase2-epochs", type=int, default=80)
    ap.add_argument("--phase3-epochs", type=int, default=300)
    ap.add_argument("--n-samples", type=int, default=30_000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--num-freqs", type=int, default=6)
    ap.add_argument("--out-dir", default="lc_anova/results")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"stability_{args.pde}_{Path(args.checkpoint).stem}"
    print(f"== HDMR-seed stability  pde={args.pde}  n_runs={args.n_runs} ==")

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.pde == "helm2d":
        from lc_anova.pipelines.helmholtz_2d import load_lc_pinn, sample_joint, evaluate_lc_pinn_batch
        max_order, use_fourier = 3, True
        dim_x, dim_lambda = 2, 1
        model, _ = load_lc_pinn(args.checkpoint, device)
        sampler = lambda n, seed: sample_joint(n, seed, device)
    elif args.pde == "helm1d":
        from lc_anova.pipelines.pde1d import load_lc_pinn, sample_joint, evaluate_lc_pinn_batch, pde_config
        pde = pde_config("helmholtz")
        max_order, use_fourier = 2, True
        dim_x, dim_lambda = 1, 1
        model, _ = load_lc_pinn(args.checkpoint, pde, device)
        sampler = lambda n, seed: sample_joint(n, seed, pde, device)
    elif args.pde == "schr1d":
        from lc_anova.pipelines.pde1d import load_lc_pinn, sample_joint, evaluate_lc_pinn_batch, pde_config
        pde = pde_config("schrodinger")
        max_order, use_fourier = 2, True
        dim_x, dim_lambda = 1, 1
        model, _ = load_lc_pinn(args.checkpoint, pde, device)
        sampler = lambda n, seed: sample_joint(n, seed, pde, device)
    else:
        raise ValueError(args.pde)

    # Evaluate LC-PINN on a fixed sample set once (independent of HDMR seed).
    x_tr, p_tr = sampler(args.n_samples, seed=42)
    u_tr = evaluate_lc_pinn_batch(model, x_tr, p_tr)
    x_va, p_va = sampler(args.n_samples, seed=43)
    u_va = evaluate_lc_pinn_batch(model, x_va, p_va)

    runs = []
    for run_i in range(args.n_runs):
        rng_seed = 100 + run_i
        torch.manual_seed(rng_seed)
        np.random.seed(rng_seed)
        print(f"\n--- HDMR run {run_i+1}/{args.n_runs} (torch seed {rng_seed}) ---")

        jh = JointHDMR(
            dim_x=dim_x, dim_lambda=dim_lambda,
            hidden=args.hidden, layers=args.layers,
            max_order=max_order, use_fourier=use_fourier,
            num_freqs=args.num_freqs,
        )
        kwargs = {"phase1_epochs": args.phase1_epochs,
                  "phase2_epochs": args.phase2_epochs, "log_every": 30}
        if max_order >= 3:
            kwargs["phase3_epochs"] = args.phase3_epochs
        history = jh.fit(x_samples=x_tr, lambda_samples=p_tr,
                         u_targets=u_tr, **kwargs)

        # Eval on val
        z_va = torch.cat([x_va, p_va], dim=1).to(device)
        y_va_centered = u_va.to(device) - jh.y_mean
        jh.model.eval()
        with torch.no_grad():
            if jh._has_triplet:
                pred, *_ = jh.model(z_va, include_pairs=True, include_triplet=True, purify=True)
            else:
                pred, _, _ = jh.model(z_va, include_pairs=True, purify=True)
            val_rel = (torch.sqrt(torch.mean((pred - y_va_centered) ** 2))
                       / y_va_centered.std()).item()
        terms = jh.evaluate_terms(x_va, p_va)
        runs.append({
            "rng_seed": rng_seed,
            "val_rel_rmse": val_rel,
            "sobol": {str(k): v for k, v in terms["sobol"].items()},
        })
        print(f"  val rel-RMSE = {val_rel:.4f}")

    # Aggregate
    sobol_keys = list(runs[0]["sobol"].keys())
    summary = {
        "checkpoint": str(args.checkpoint),
        "pde": args.pde,
        "n_runs": args.n_runs,
        "val_rel_rmse_mean": float(np.mean([r["val_rel_rmse"] for r in runs])),
        "val_rel_rmse_std":  float(np.std([r["val_rel_rmse"] for r in runs])),
        "sobol_mean": {},
        "sobol_std":  {},
        "runs": runs,
    }
    for key in sobol_keys:
        vals = [r["sobol"][key] for r in runs]
        summary["sobol_mean"][key] = float(np.mean(vals))
        summary["sobol_std"][key]  = float(np.std(vals))

    print(f"\nVal rel-RMSE: {summary['val_rel_rmse_mean']:.4f} ± {summary['val_rel_rmse_std']:.4f}")
    print("\nSobol mean ± std over HDMR seeds:")
    for key in sobol_keys:
        print(f"  {key:<14}  {summary['sobol_mean'][key]:.4f} ± {summary['sobol_std'][key]:.4f}")

    out_json = out_dir / f"stability_{tag}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
