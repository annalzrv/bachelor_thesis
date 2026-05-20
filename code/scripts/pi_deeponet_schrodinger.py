"""PI-DeepONet baseline on 1D parametric Schrödinger (driven, harmonic).

Trains a Physics-Informed DeepONet (branch over alpha_norm, trunk over x)
using the same residual loss as LC-PINN. No solver-generated paired data.
Apples-to-apples competitor on the parametric-coefficient family.

Usage:
    python scripts/pi_deeponet_schrodinger.py --seeds 0 1 2 3 --n-epochs 50000

Output:
    results/pi_deeponet_schrodinger{_tag}.json
    checkpoints/pi_deeponet_schrodinger_seed{s}{_tag}.pt
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
from pinns.equations import schrodinger_1d as helm
from pinns.pi_deeponet import PIDeepONet


REPO = pathlib.Path(__file__).resolve().parent.parent


def _u_xx_1d(u, coords):
    g = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(g, coords, torch.ones_like(g), create_graph=True)[0]
    return u_xx


def compute_losses(model, k_norm, batch):
    """Mirror helm.compute_losses but call DeepONet API (coords, alpha_norm)."""
    k_val = helm.norm_to_k(k_norm.item() if k_norm.numel() == 1 else k_norm[0].item())

    coords = batch["coords_pde"].requires_grad_(True)
    u = model(coords, k_norm.view(1, -1))
    uxx = _u_xx_1d(u, coords)
    V = helm.potential(coords, k_val)
    f = helm.forcing(coords, k_val)
    residual = -uxx + V * u - f
    L_pde = torch.mean(residual ** 2)

    L_bc = torch.mean((model(batch["coords_bc"], k_norm.view(1, -1)) - batch["u_bc"]) ** 2)

    k_data = batch["k_data"]
    k_norm_per_point = helm.k_to_norm(k_data).unsqueeze(-1)
    u_pred_data = model(batch["coords_data"], k_norm_per_point)
    L_data = torch.mean((u_pred_data - batch["u_data"]) ** 2)

    return {"pde": L_pde, "bc": L_bc, "data": L_data}


def train_one(model, batch, device, n_epochs, lr, n_k_per_step,
              w_pde, w_bc, w_data, log_every):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    history = {"step": [], "total": [], "pde": [], "bc": [], "data": []}
    t0 = time.time()

    for step in range(n_epochs):
        opt.zero_grad()
        total = torch.zeros(1, device=device).squeeze()
        last_parts = None
        for _ in range(n_k_per_step):
            k_norm = (torch.rand(1, device=device) * 2.0) - 1.0
            losses = compute_losses(model, k_norm, batch)
            weighted = w_pde * losses["pde"] + w_bc * losses["bc"] + w_data * losses["data"]
            total = total + weighted
            last_parts = {k: float(v.item()) for k, v in losses.items()}
        total = total / n_k_per_step
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


def lbfgs_finish(model, batch, device, n_iter, n_k_support,
                 w_pde, w_bc, w_data, log_every=50, resample_every=100):
    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0, max_iter=20, max_eval=25,
        tolerance_grad=1e-8, tolerance_change=1e-10,
        history_size=50, line_search_fn="strong_wolfe",
    )
    state = {"step": 0, "last_loss": float("nan"), "last_parts": None,
             "k_support": None}

    def refresh_support():
        ks = (torch.rand(n_k_support, device=device) * 2.0) - 1.0
        state["k_support"] = ks

    refresh_support()

    def closure():
        opt.zero_grad()
        total = torch.zeros(1, device=device).squeeze()
        last_parts = None
        for k_norm in state["k_support"]:
            losses = compute_losses(model, k_norm.unsqueeze(0), batch)
            weighted = w_pde * losses["pde"] + w_bc * losses["bc"] + w_data * losses["data"]
            total = total + weighted
            last_parts = {k: float(v.item()) for k, v in losses.items()}
        total = total / n_k_support
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        state["last_loss"] = float(total.item())
        state["last_parts"] = last_parts
        return total

    t0 = time.time()
    for it in range(n_iter):
        if it > 0 and it % resample_every == 0:
            refresh_support()
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


@torch.no_grad()
def evaluate_at_K_k_values(model, device, K):
    k_grid = np.linspace(helm.K_MIN, helm.K_MAX, K, dtype=np.float32)
    errs = []
    for k in k_grid:
        x = np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(-1, 1)
        u_ref = helm.reference_solution(x.flatten(), float(k))
        x_t = torch.tensor(x, device=device)
        k_norm = torch.tensor([[helm.k_to_norm(float(k))]], dtype=torch.float32, device=device)
        u_pred = model(x_t, k_norm).cpu().numpy().flatten()
        rel = float(np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-10))
        errs.append(rel)
    return list(map(float, k_grid)), errs


def run_one_seed(seed, batch, device, n_epochs, lr, n_k_per_step, K_eval,
                 w_pde, w_bc, w_data, n_basis, hidden_dims, n_lbfgs, tag):
    torch.manual_seed(seed); np.random.seed(seed)
    model = PIDeepONet(
        dim_phys=helm.DIM_PHYS, dim_lambda=helm.DIM_LAMBDA,
        n_basis=n_basis,
        branch_hidden=hidden_dims, trunk_hidden=hidden_dims,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    t0 = time.time()
    print(f"[seed {seed}] PI-DeepONet 1D Schrod: hidden={hidden_dims} p={n_basis} "
          f"params={n_params}  {n_epochs} Adam + {n_lbfgs} L-BFGS  "
          f"(k samples/step = {n_k_per_step})", flush=True)
    history = train_one(model, batch, device, n_epochs, lr, n_k_per_step,
                        w_pde, w_bc, w_data, log_every=max(1, n_epochs // 25))

    lbfgs_info = None
    if n_lbfgs > 0:
        n_k_support = max(16, n_k_per_step)
        pre_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        pre_err = float(np.mean(evaluate_at_K_k_values(model, device, K=K_eval)[1]))
        print(f"[seed {seed}] L-BFGS finishing ({n_lbfgs} iter, k_support={n_k_support}, pre={pre_err:.4e})", flush=True)
        lbfgs_info = lbfgs_finish(model, batch, device, n_lbfgs, n_k_support,
                                  w_pde, w_bc, w_data, log_every=max(1, n_lbfgs // 20))
        post_err = float(np.mean(evaluate_at_K_k_values(model, device, K=K_eval)[1]))
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
    k_vals, errs_K = evaluate_at_K_k_values(model, device, K=K_eval)
    mean_rel = float(np.mean(errs_K))
    std_rel = float(np.std(errs_K))

    suffix = f"_{tag}" if tag else ""
    ckpt = REPO / "checkpoints" / f"pi_deeponet_schrodinger_seed{seed}{suffix}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "rel_l2_K_eval": errs_K, "k_grid": k_vals,
        "rel_l2_K_mean": mean_rel, "rel_l2_K_std": std_rel,
        "elapsed_sec": elapsed, "seed": seed,
        "n_epochs": n_epochs, "lr": lr,
        "n_k_per_step": n_k_per_step, "K_eval": K_eval,
        "weights": {"pde": w_pde, "bc": w_bc, "data": w_data},
        "n_basis": n_basis,
        "hidden_dims": hidden_dims,
        "n_params": n_params,
        "n_lbfgs": n_lbfgs,
        "lbfgs": lbfgs_info,
    }, ckpt)
    print(f"[seed {seed}] done in {elapsed/60:.1f} min  rel-L2 mean (K={K_eval}): "
          f"{mean_rel:.4e} ± {std_rel:.4e}", flush=True)
    return {
        "seed": seed,
        "mean_rel_l2": mean_rel,
        "std_rel_l2_over_K": std_rel,
        "rel_l2_per_k": dict(zip(k_vals, errs_K)),
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
    p.add_argument("--n-k-per-step", type=int, default=4)
    p.add_argument("--K-eval", type=int, default=20)
    p.add_argument("--w-pde", type=float, default=1.0)
    p.add_argument("--w-bc", type=float, default=100.0)
    p.add_argument("--w-data", type=float, default=1.0)
    p.add_argument("--n-basis", type=int, default=64)
    p.add_argument("--hidden-width", type=int, default=64)
    p.add_argument("--hidden-depth", type=int, default=4)
    p.add_argument("--n-lbfgs", type=int, default=1500,
                   help="L-BFGS finishing iters (apples-to-apples with LC-PINN)")
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()

    hidden_dims = [args.hidden_width] * args.hidden_depth

    device = select_device()
    print(f"Device: {device_info(device)}", flush=True)
    print(f"Config: seeds={args.seeds} n_epochs={args.n_epochs} K_eval={args.K_eval} "
          f"hidden={hidden_dims} n_basis={args.n_basis} n_lbfgs={args.n_lbfgs} "
          f"tag={args.tag!r}\n", flush=True)
    print(f"1D Schrödinger family: α ∈ [{helm.K_MIN}, {helm.K_MAX}]", flush=True)
    batch = helm.generate_training_data(device=device)

    runs = []
    for s in args.seeds:
        runs.append(run_one_seed(
            s, batch, device, args.n_epochs, args.lr,
            args.n_k_per_step, args.K_eval,
            args.w_pde, args.w_bc, args.w_data,
            args.n_basis, hidden_dims, args.n_lbfgs, args.tag,
        ))

    means = [r["mean_rel_l2"] for r in runs]
    elapsed = [r["elapsed_sec"] for r in runs]
    summary = {
        "method": "pi-deeponet", "equation": "schrodinger_1d",
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
    out = REPO / "results" / f"pi_deeponet_schrodinger{suffix}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out.relative_to(REPO)}", flush=True)
    print(f"  rel-L2: {summary['summary']['rel_l2_mean']:.4e} ± "
          f"{summary['summary']['rel_l2_std']:.4e}", flush=True)
    print(f"  mean wall time per seed: "
          f"{summary['summary']['elapsed_mean_sec']/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
