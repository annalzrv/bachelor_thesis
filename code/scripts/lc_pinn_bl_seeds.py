"""LC-PINN Buckley-Leverett reference run, multiple seeds (matched bookkeeping).

Trains one LC-PINN per seed with uniform λ ∈ U(0,1)^4 sampling, identical
backbone to SA-PINN and ReLoBRaLo BL baselines.

After training, evaluates the network at K random uniform λ-vectors so we
can also report a "mean over inference λ" rel-L2 for the amortised-cost
analysis.

Optional knobs (added for the FiLM/L-BFGS sweep):
    --conditioning  concat | film
    --hidden-width  hidden width (default 64)
    --hidden-depth  number of hidden layers (default 4)
    --n-lbfgs       L-BFGS finishing iterations (default 0)

Usage:
    python scripts/lc_pinn_bl_seeds.py --seeds 0 1 2 3 --n-epochs 50000

Output:
    results/lc_pinn_bl_seeds{_tag}.json
    checkpoints/lc_pinn_bl_seed{s}{_tag}.pt
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

from pinns.device import select_device, device_info
from pinns.equations import buckley_leverett as bl
from pinns.lambda_sampler import LambdaSampler
from pinns.model import LossConditionalPINN
from pinns.training import train_lc_pinn, lbfgs_finish_lc


REPO = pathlib.Path(__file__).resolve().parent.parent


def evaluate_at_K_lambdas(model, ref, device, K: int, eval_seed: int = 999):
    rng = np.random.default_rng(eval_seed)
    errs = []
    for _ in range(K):
        lam = rng.uniform(0.0, 1.0, size=4).astype(np.float32)
        log_lam = torch.tensor(np.log(np.clip(lam, 1e-8, None)), device=device)
        e = bl.evaluate(model, log_lam, ref, device)
        errs.append(float(np.mean(list(e.values()))))
    return errs


def run_one_seed(seed, batch, ref, device, n_epochs, lr, n_lambda_samples, K_eval,
                 domain: bl.BLDomain = bl.DEFAULT_DOMAIN,
                 conditioning: str = "concat",
                 hidden_dims: list[int] | None = None,
                 n_lbfgs: int = 0,
                 tag: str = ""):
    if hidden_dims is None:
        hidden_dims = [64, 64, 64, 64]
    torch.manual_seed(seed); np.random.seed(seed)

    model = LossConditionalPINN(
        dim_phys=bl.DIM_PHYS, dim_lambda=bl.DIM_LAMBDA,
        hidden_dims=hidden_dims,
        conditioning=conditioning,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    sampler = LambdaSampler(dim=bl.DIM_LAMBDA, mode="uniform", device=device)

    t0 = time.time()
    print(f"[seed {seed}] BL LC-PINN: cond={conditioning} hidden={hidden_dims} "
          f"params={n_params}  {n_epochs} Adam steps + {n_lbfgs} L-BFGS  "
          f"(λ samples per step = {n_lambda_samples})", flush=True)
    loss_fn = lambda m, log_lam, b: bl.compute_losses(m, log_lam, b, domain=domain)
    history = train_lc_pinn(
        model=model, sampler=sampler, batch=batch, device=device,
        loss_fn=loss_fn,
        n_epochs=n_epochs, lr=lr, n_lambda_samples=n_lambda_samples,
        log_every=2000,
    )
    lbfgs_info = None
    if n_lbfgs > 0:
        n_lambda_support = max(16, n_lambda_samples)
        print(f"[seed {seed}] starting L-BFGS finishing ({n_lbfgs} iter, "
              f"λ support={n_lambda_support})", flush=True)
        pre_lbfgs_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        pre_lbfgs_err = float(np.mean(evaluate_at_K_lambdas(model, ref, device, K=K_eval)))
        lbfgs_info = lbfgs_finish_lc(
            model=model, sampler=sampler, batch=batch, device=device,
            loss_fn=loss_fn,
            n_iter=n_lbfgs, n_lambda_support=n_lambda_support,
            log_every=max(1, n_lbfgs // 20),
        )
        post_err = float(np.mean(evaluate_at_K_lambdas(model, ref, device, K=K_eval)))
        lbfgs_info["pre_lbfgs_rel_l2"] = pre_lbfgs_err
        lbfgs_info["post_lbfgs_rel_l2"] = post_err
        if not (post_err == post_err) or post_err > pre_lbfgs_err:
            print(f"[seed {seed}] L-BFGS made things worse "
                  f"(pre={pre_lbfgs_err:.4e}, post={post_err:.4e}); reverting",
                  flush=True)
            model.load_state_dict(pre_lbfgs_state)
            lbfgs_info["reverted"] = True
        else:
            lbfgs_info["reverted"] = False
            print(f"[seed {seed}] L-BFGS improved rel-L2: "
                  f"{pre_lbfgs_err:.4e} -> {post_err:.4e}", flush=True)
    elapsed = time.time() - t0

    errs_K = evaluate_at_K_lambdas(model, ref, device, K=K_eval)
    mean_rel = float(np.mean(errs_K))
    std_rel  = float(np.std(errs_K))

    lam_centre = torch.tensor(np.log([0.5, 0.5, 0.5, 0.5]), dtype=torch.float32, device=device)
    err_centre = bl.evaluate(model, lam_centre, ref, device)
    mean_centre = float(np.mean(list(err_centre.values())))

    suffix = f"_{tag}" if tag else ""
    ckpt = REPO / "checkpoints" / f"lc_pinn_bl_seed{seed}{suffix}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "rel_l2_K_eval": errs_K,
        "rel_l2_K_mean": mean_rel,
        "rel_l2_K_std": std_rel,
        "rel_l2_centre": mean_centre,
        "rel_l2_per_snapshot_centre": {float(k): float(v) for k, v in err_centre.items()},
        "elapsed_sec": elapsed, "seed": seed,
        "n_epochs": n_epochs, "lr": lr,
        "n_lambda_samples": n_lambda_samples, "K_eval": K_eval,
        "conditioning": conditioning,
        "hidden_dims": hidden_dims,
        "n_params": n_params,
        "n_lbfgs": n_lbfgs,
        "lbfgs": lbfgs_info,
    }, ckpt)

    print(f"[seed {seed}] done in {elapsed/60:.1f} min  "
          f"rel-L2 over K={K_eval}: {mean_rel:.4e} ± {std_rel:.4e}", flush=True)
    return {
        "seed": seed,
        "mean_rel_l2": mean_rel,
        "std_rel_l2_over_K": std_rel,
        "rel_l2_centre": mean_centre,
        "elapsed_sec": elapsed,
        "checkpoint": str(ckpt.relative_to(REPO)),
        "n_params": n_params,
        "lbfgs": lbfgs_info,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--n-epochs", type=int, default=50_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-lambda-samples", type=int, default=4)
    p.add_argument("--K-eval", type=int, default=25)
    p.add_argument("--epsilon", type=float, default=0.0,
                   help="viscous regularisation; >0 uses viscous reference solver")
    p.add_argument("--tag", type=str, default="",
                   help="suffix for output filenames, e.g. 'viscous'")
    p.add_argument("--conditioning", type=str, default="concat",
                   choices=["concat", "film"])
    p.add_argument("--hidden-width", type=int, default=64)
    p.add_argument("--hidden-depth", type=int, default=4)
    p.add_argument("--n-lbfgs", type=int, default=0,
                   help="L-BFGS finishing iterations after Adam (0 disables)")
    args = p.parse_args()

    hidden_dims = [args.hidden_width] * args.hidden_depth

    device = select_device()
    print(f"Device: {device_info(device)}", flush=True)
    print(f"Config: seeds={args.seeds} n_epochs={args.n_epochs} K_eval={args.K_eval} "
          f"epsilon={args.epsilon} cond={args.conditioning} hidden={hidden_dims} "
          f"n_lbfgs={args.n_lbfgs}\n", flush=True)

    domain = bl.BLDomain(epsilon=args.epsilon) if args.epsilon > 0 else bl.DEFAULT_DOMAIN

    print(f"Building BL reference & training batch (epsilon={args.epsilon})…", flush=True)
    if args.epsilon > 0:
        ref = bl.compute_viscous_reference_solution(domain=domain)
    else:
        ref = bl.compute_reference_solution(domain=domain)
    batch = bl.generate_training_data(ref, device=device, domain=domain)

    runs = [run_one_seed(s, batch, ref, device, args.n_epochs, args.lr,
                         args.n_lambda_samples, args.K_eval, domain=domain,
                         conditioning=args.conditioning, hidden_dims=hidden_dims,
                         n_lbfgs=args.n_lbfgs, tag=args.tag)
            for s in args.seeds]
    means = [r["mean_rel_l2"] for r in runs]
    elapsed_total = [r["elapsed_sec"] for r in runs]
    summary = {
        "method": "lc-pinn", "equation": "buckley-leverett",
        "config": vars(args), "runs": runs,
        "summary": {
            "n_seeds": len(runs),
            "rel_l2_mean": float(np.mean(means)), "rel_l2_std":  float(np.std(means)),
            "rel_l2_min":  float(np.min(means)),  "rel_l2_max":  float(np.max(means)),
            "elapsed_mean_sec":  float(np.mean(elapsed_total)),
            "elapsed_total_sec": float(np.sum(elapsed_total)),
            "n_params": runs[0].get("n_params"),
        },
    }
    suffix = f"_{args.tag}" if args.tag else ""
    out = REPO / "results" / f"lc_pinn_bl_seeds{suffix}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out.relative_to(REPO)}", flush=True)
    print(f"  rel-L2 (mean over K={args.K_eval} inference λ, then over seeds): "
          f"{summary['summary']['rel_l2_mean']:.4e} ± {summary['summary']['rel_l2_std']:.4e}",
          flush=True)
    print(f"  mean wall time per seed: {summary['summary']['elapsed_mean_sec']/60:.1f} min",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
