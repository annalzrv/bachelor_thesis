"""LC-PINN on 1D driven Schrödinger with parametric harmonic potential.

Network input is (x, α_norm). Same FiLM/L-BFGS sweep machinery as the
Helmholtz scripts.

Usage:
    python scripts/lc_pinn_schrodinger.py --seeds 0 1 2 3 --n-epochs 50000 \
        --conditioning film --hidden-width 64 --n-lbfgs 1500 --tag film_lbfgs

Output:
    results/lc_pinn_schrodinger{_tag}.json
    checkpoints/lc_pinn_schrodinger_seed{s}{_tag}.pt
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pinns.device import select_device, device_info
from pinns.equations import schrodinger_1d as sch
from pinns.model import LossConditionalPINN


REPO = pathlib.Path(__file__).resolve().parent.parent


def train_lc(model, batch, device, n_epochs, lr, n_a_per_step,
             w_pde, w_bc, w_data, log_every):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    history = {"step": [], "total": [], "pde": [], "bc": [], "data": []}
    t0 = time.time()

    for step in range(n_epochs):
        opt.zero_grad()
        total = torch.zeros(1, device=device).squeeze()
        last_parts = None
        for _ in range(n_a_per_step):
            a_norm = (torch.rand(1, device=device) * 2.0) - 1.0
            losses = sch.compute_losses(model, a_norm, batch)
            weighted = w_pde * losses["pde"] + w_bc * losses["bc"] + w_data * losses["data"]
            total = total + weighted
            last_parts = {k: float(v.item()) for k, v in losses.items()}
        total = total / n_a_per_step
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

        if step % log_every == 0:
            history["step"].append(step)
            history["total"].append(float(total.item()))
            for k_name in ("pde", "bc", "data"):
                history[k_name].append(last_parts[k_name])
            print(f"  step {step:6d}  L={total.item():.4e}  parts={last_parts}", flush=True)

    history["elapsed_sec"] = time.time() - t0
    return history


def lbfgs_finish(model, batch, device, n_iter, n_a_support,
                 w_pde, w_bc, w_data, log_every=50, resample_every=100):
    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0, max_iter=20, max_eval=25,
        tolerance_grad=1e-8, tolerance_change=1e-10,
        history_size=50, line_search_fn="strong_wolfe",
    )
    state = {"step": 0, "last_loss": float("nan"), "last_parts": None,
             "a_support": None}

    def refresh():
        state["a_support"] = (torch.rand(n_a_support, device=device) * 2.0) - 1.0

    refresh()

    def closure():
        opt.zero_grad()
        total = torch.zeros(1, device=device).squeeze()
        last_parts = None
        for a_norm in state["a_support"]:
            losses = sch.compute_losses(model, a_norm.unsqueeze(0), batch)
            weighted = w_pde * losses["pde"] + w_bc * losses["bc"] + w_data * losses["data"]
            total = total + weighted
            last_parts = {k: float(v.item()) for k, v in losses.items()}
        total = total / n_a_support
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        state["last_loss"] = float(total.item())
        state["last_parts"] = last_parts
        return total

    t0 = time.time()
    for it in range(n_iter):
        if it > 0 and it % resample_every == 0:
            refresh()
        opt.step(closure)
        state["step"] = it
        loss = state["last_loss"]
        if not (loss == loss):
            print(f"  [lbfgs] NaN at iter {it} — aborting", flush=True)
            break
        if it % log_every == 0:
            print(f"  [lbfgs] step {it:5d}  L={loss:.4e}  parts={state['last_parts']}",
                  flush=True)
    return {"elapsed_sec": time.time() - t0, "final_loss": state["last_loss"]}


def evaluate_at_K(model, device, K):
    a_grid = np.linspace(sch.ALPHA_MIN, sch.ALPHA_MAX, K, dtype=np.float32)
    errs = [sch.evaluate_at_alpha(model, float(a), device, is_lc=True) for a in a_grid]
    return list(map(float, a_grid)), list(map(float, errs))


def run_one_seed(seed, batch, device, n_epochs, lr, n_a_per_step, K_eval,
                 w_pde, w_bc, w_data, conditioning, hidden_dims, n_lbfgs, tag):
    torch.manual_seed(seed); np.random.seed(seed)
    model = LossConditionalPINN(
        dim_phys=sch.DIM_PHYS, dim_lambda=sch.DIM_LAMBDA,
        hidden_dims=hidden_dims, conditioning=conditioning,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    t0 = time.time()
    print(f"[seed {seed}] LC-PINN Schrödinger: cond={conditioning} hidden={hidden_dims} "
          f"params={n_params}  {n_epochs} Adam + {n_lbfgs} L-BFGS  "
          f"(α samples/step = {n_a_per_step})", flush=True)
    train_lc(model, batch, device, n_epochs, lr, n_a_per_step,
             w_pde, w_bc, w_data, log_every=max(1, n_epochs // 25))

    lbfgs_info = None
    if n_lbfgs > 0:
        n_a_support = max(16, n_a_per_step)
        pre_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        pre_err = float(np.mean(evaluate_at_K(model, device, K=K_eval)[1]))
        print(f"[seed {seed}] L-BFGS finishing ({n_lbfgs} iter, support={n_a_support}, pre={pre_err:.4e})",
              flush=True)
        lbfgs_info = lbfgs_finish(model, batch, device, n_lbfgs, n_a_support,
                                  w_pde, w_bc, w_data, log_every=max(1, n_lbfgs // 20))
        post_err = float(np.mean(evaluate_at_K(model, device, K=K_eval)[1]))
        lbfgs_info["pre_lbfgs_rel_l2"] = pre_err
        lbfgs_info["post_lbfgs_rel_l2"] = post_err
        if not (post_err == post_err) or post_err > pre_err:
            print(f"[seed {seed}] L-BFGS made things worse "
                  f"(pre={pre_err:.4e}, post={post_err:.4e}); reverting", flush=True)
            model.load_state_dict(pre_state)
            lbfgs_info["reverted"] = True
        else:
            lbfgs_info["reverted"] = False
            print(f"[seed {seed}] L-BFGS lift: {pre_err:.4e} -> {post_err:.4e}", flush=True)

    elapsed = time.time() - t0
    a_vals, errs_K = evaluate_at_K(model, device, K=K_eval)
    mean_rel = float(np.mean(errs_K))
    std_rel = float(np.std(errs_K))

    suffix = f"_{tag}" if tag else ""
    ckpt = REPO / "checkpoints" / f"lc_pinn_schrodinger_seed{seed}{suffix}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "rel_l2_K_eval": errs_K, "alpha_grid": a_vals,
        "rel_l2_K_mean": mean_rel, "rel_l2_K_std": std_rel,
        "elapsed_sec": elapsed, "seed": seed,
        "n_epochs": n_epochs, "lr": lr,
        "n_a_per_step": n_a_per_step, "K_eval": K_eval,
        "weights": {"pde": w_pde, "bc": w_bc, "data": w_data},
        "conditioning": conditioning, "hidden_dims": hidden_dims,
        "n_params": n_params, "n_lbfgs": n_lbfgs, "lbfgs": lbfgs_info,
    }, ckpt)
    print(f"[seed {seed}] done in {elapsed/60:.1f} min  rel-L2 mean (K={K_eval}): "
          f"{mean_rel:.4e} ± {std_rel:.4e}", flush=True)
    return {
        "seed": seed,
        "mean_rel_l2": mean_rel,
        "std_rel_l2_over_K": std_rel,
        "rel_l2_per_alpha": dict(zip(a_vals, errs_K)),
        "elapsed_sec": elapsed,
        "checkpoint": str(ckpt.relative_to(REPO)),
        "n_params": n_params,
        "lbfgs": lbfgs_info,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--n-epochs", type=int, default=50_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-a-per-step", type=int, default=4)
    p.add_argument("--K-eval", type=int, default=20)
    p.add_argument("--w-pde", type=float, default=1.0)
    p.add_argument("--w-bc", type=float, default=100.0)
    p.add_argument("--w-data", type=float, default=1.0)
    p.add_argument("--conditioning", type=str, default="concat",
                   choices=["concat", "film"])
    p.add_argument("--hidden-width", type=int, default=64)
    p.add_argument("--hidden-depth", type=int, default=4)
    p.add_argument("--n-lbfgs", type=int, default=0)
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()

    hidden_dims = [args.hidden_width] * args.hidden_depth

    device = select_device()
    print(f"Device: {device_info(device)}", flush=True)
    print(f"Config: seeds={args.seeds} n_epochs={args.n_epochs} K_eval={args.K_eval} "
          f"cond={args.conditioning} hidden={hidden_dims} n_lbfgs={args.n_lbfgs} "
          f"tag={args.tag!r}\n", flush=True)
    print(f"Schrödinger family: α ∈ [{sch.ALPHA_MIN}, {sch.ALPHA_MAX}]", flush=True)
    batch = sch.generate_training_data(device=device)

    runs = []
    for s in args.seeds:
        runs.append(run_one_seed(
            s, batch, device, args.n_epochs, args.lr,
            args.n_a_per_step, args.K_eval,
            args.w_pde, args.w_bc, args.w_data,
            conditioning=args.conditioning, hidden_dims=hidden_dims,
            n_lbfgs=args.n_lbfgs, tag=args.tag,
        ))

    means = [r["mean_rel_l2"] for r in runs]
    elapsed = [r["elapsed_sec"] for r in runs]
    summary = {
        "method": "lc-pinn", "equation": "schrodinger_1d",
        "config": vars(args), "runs": runs,
        "summary": {
            "n_seeds": len(runs),
            "rel_l2_mean": float(np.mean(means)),
            "rel_l2_std": float(np.std(means)),
            "rel_l2_min": float(np.min(means)),
            "rel_l2_max": float(np.max(means)),
            "elapsed_mean_sec": float(np.mean(elapsed)),
            "elapsed_total_sec": float(np.sum(elapsed)),
            "n_params": runs[0].get("n_params"),
        },
    }
    suffix = f"_{args.tag}" if args.tag else ""
    out = REPO / "results" / f"lc_pinn_schrodinger{suffix}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out.relative_to(REPO)}", flush=True)
    print(f"  rel-L2: {summary['summary']['rel_l2_mean']:.4e} ± "
          f"{summary['summary']['rel_l2_std']:.4e}", flush=True)
    print(f"  mean wall time per seed: "
          f"{summary['summary']['elapsed_mean_sec']/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
