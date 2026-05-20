"""ReLoBRaLo baseline on Buckley-Leverett (m=2).

Adaptive softmax-based balancing of N loss terms (PDE, BC, IC, data).
Same backbone as LC-PINN (hidden_dims = [64,64,64,64]) for fair comparison.

Usage:
    python scripts/relobralo_bl.py --seeds 0 1 2 3 --n-epochs 50000

Output:
    results/relobralo_bl.json
    checkpoints/relobralo_bl_seed{s}.pt
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

from pinns.baseline import FixedWeightPINN
from pinns.device import select_device, device_info
from pinns.equations import buckley_leverett as bl


HIDDEN_DIMS = [64, 64, 64, 64]
REPO = pathlib.Path(__file__).resolve().parent.parent


class ReLoBRaLoBalancer:
    def __init__(self, n_terms: int, alpha: float = 0.999,
                 tau: float = 0.1, rho_mean: float = 0.999,
                 device: torch.device = torch.device("cpu")):
        self.n = n_terms
        self.alpha = alpha
        self.tau = tau
        self.rho_mean = rho_mean
        self.device = device
        self.L_init: torch.Tensor | None = None
        self.L_prev: torch.Tensor | None = None
        self.lam_prev = torch.ones(n_terms, device=device)

    def __call__(self, losses_now: torch.Tensor) -> torch.Tensor:
        Lt = losses_now.detach()
        if self.L_init is None:
            self.L_init = Lt.clone()
            self.L_prev = Lt.clone()
            return self.lam_prev.clone()

        rho = float(torch.bernoulli(torch.tensor(self.rho_mean)).item())
        lam_bal_prev = self.n * torch.softmax(Lt / (self.tau * self.L_prev + 1e-12), dim=0)
        lam_bal_init = self.n * torch.softmax(Lt / (self.tau * self.L_init + 1e-12), dim=0)
        lam_hist = rho * self.lam_prev + (1.0 - rho) * lam_bal_init
        lam_new  = self.alpha * lam_hist + (1.0 - self.alpha) * lam_bal_prev

        self.lam_prev = lam_new.detach()
        self.L_prev = Lt.clone()
        return lam_new


def run_one_seed(seed, batch, ref, device, n_epochs, lr, alpha, tau, rho_mean,
                 domain: bl.BLDomain = bl.DEFAULT_DOMAIN):
    torch.manual_seed(seed); np.random.seed(seed)

    model = FixedWeightPINN(bl.DIM_PHYS, HIDDEN_DIMS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    balancer = ReLoBRaLoBalancer(n_terms=4, alpha=alpha, tau=tau,
                                 rho_mean=rho_mean, device=device)

    t0 = time.time()
    log_steps, log_lambda, log_total = [], [], []
    print(f"[seed {seed}] BL ReLoBRaLo Adam: {n_epochs} steps")
    for step in range(n_epochs):
        opt.zero_grad()
        losses = bl.compute_losses_fixed(model, batch, domain=domain)
        L_vec = torch.stack([losses["pde"], losses["bc"], losses["ic"], losses["data"]])
        lam = balancer(L_vec)
        total = (lam * L_vec).sum()
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step(); sched.step()
        if step % 2000 == 0:
            log_steps.append(step)
            log_lambda.append([float(x) for x in lam.detach().cpu().tolist()])
            log_total.append(float(total.item()))
            print(f"  step {step:6d}  L={total.item():.4e}  lam={[round(float(x),3) for x in lam.detach().cpu().tolist()]}")

    elapsed = time.time() - t0
    errors = bl.evaluate(model, None, ref, device)
    mean_rel = float(np.mean(list(errors.values())))

    ckpt = REPO / "checkpoints" / f"relobralo_bl_seed{seed}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "lambda_final": balancer.lam_prev.cpu(),
        "rel_l2_per_snapshot": {float(k): float(v) for k, v in errors.items()},
        "mean_rel_l2": mean_rel,
        "elapsed_sec": elapsed, "seed": seed,
        "n_epochs": n_epochs, "lr": lr,
        "alpha": alpha, "tau": tau, "rho_mean": rho_mean,
        "log_lambda_trace": log_lambda, "log_total": log_total, "log_steps": log_steps,
    }, ckpt)

    print(f"[seed {seed}] done in {elapsed/60:.1f} min  rel-L2 = {mean_rel:.4e}")
    return {
        "seed": seed,
        "rel_l2_per_snapshot": {float(k): float(v) for k, v in errors.items()},
        "mean_rel_l2": mean_rel, "elapsed_sec": elapsed,
        "checkpoint": str(ckpt.relative_to(REPO)),
        "lambda_final": [float(x) for x in balancer.lam_prev.cpu().tolist()],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--n-epochs", type=int, default=50_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.999)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--rho-mean", type=float, default=0.999)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()

    device = select_device()
    print(f"Device: {device_info(device)}")
    print(f"Config: seeds={args.seeds} n_epochs={args.n_epochs} epsilon={args.epsilon}\n")

    domain = bl.BLDomain(epsilon=args.epsilon) if args.epsilon > 0 else bl.DEFAULT_DOMAIN

    print(f"Building BL reference & training batch (epsilon={args.epsilon})…")
    if args.epsilon > 0:
        ref = bl.compute_viscous_reference_solution(domain=domain)
    else:
        ref = bl.compute_reference_solution(domain=domain)
    batch = bl.generate_training_data(ref, device=device, domain=domain)

    runs = [run_one_seed(s, batch, ref, device, args.n_epochs, args.lr,
                         args.alpha, args.tau, args.rho_mean, domain=domain) for s in args.seeds]
    means = [r["mean_rel_l2"] for r in runs]
    elapsed_total = [r["elapsed_sec"] for r in runs]
    summary = {
        "method": "relobralo", "equation": "buckley-leverett",
        "config": vars(args), "runs": runs,
        "summary": {
            "n_seeds": len(runs),
            "rel_l2_mean": float(np.mean(means)), "rel_l2_std":  float(np.std(means)),
            "rel_l2_min":  float(np.min(means)),  "rel_l2_max":  float(np.max(means)),
            "elapsed_mean_sec":  float(np.mean(elapsed_total)),
            "elapsed_total_sec": float(np.sum(elapsed_total)),
        },
    }
    suffix = f"_{args.tag}" if args.tag else ""
    out = REPO / "results" / f"relobralo_bl{suffix}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out.relative_to(REPO)}")
    print(f"  rel-L2: {summary['summary']['rel_l2_mean']:.4e} ± {summary['summary']['rel_l2_std']:.4e}")
    print(f"  mean wall time per seed: {summary['summary']['elapsed_mean_sec']/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
