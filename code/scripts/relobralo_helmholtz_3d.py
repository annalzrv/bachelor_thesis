"""ReLoBRaLo baseline on 3D parametric Helmholtz.

Per (seed, k_train), trains a ReLoBRaLo-balanced fixed-weight PINN at that
wavenumber. Three-term loss vector (pde, bc, data) — no IC for elliptic.

Usage:
    python scripts/relobralo_helmholtz_3d.py --seeds 0 1 \\
        --k-trains 1.0 3.0 5.0 --n-epochs 25000

Output:
    results/relobralo_helmholtz_3d.json
    checkpoints/relobralo_helmholtz_3d_seed{s}_k{k:.2f}.pt
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
from pinns.equations import helmholtz_3d as helm


HIDDEN_DIMS = [64, 64, 64, 64]
REPO = pathlib.Path(__file__).resolve().parent.parent


class ReLoBRaLoBalancer:
    def __init__(self, n_terms, alpha=0.999, tau=0.1, rho_mean=0.999,
                 device=torch.device("cpu")):
        self.n = n_terms
        self.alpha = alpha
        self.tau = tau
        self.rho_mean = rho_mean
        self.device = device
        self.L_init = None
        self.L_prev = None
        self.lam_prev = torch.ones(n_terms, device=device)

    def __call__(self, losses_now):
        Lt = losses_now.detach()
        if self.L_init is None:
            self.L_init = Lt.clone()
            self.L_prev = Lt.clone()
            return self.lam_prev.clone()

        rho = float(torch.bernoulli(torch.tensor(self.rho_mean)).item())
        lam_bal_prev = self.n * torch.softmax(Lt / (self.tau * self.L_prev + 1e-12), dim=0)
        lam_bal_init = self.n * torch.softmax(Lt / (self.tau * self.L_init + 1e-12), dim=0)
        lam_hist = rho * self.lam_prev + (1.0 - rho) * lam_bal_init
        lam_new = self.alpha * lam_hist + (1.0 - self.alpha) * lam_bal_prev
        self.lam_prev = lam_new.detach()
        self.L_prev = Lt.clone()
        return lam_new


def run_one(seed, k_train, batch, device, n_epochs, lr,
            alpha, tau, rho_mean, eval_nx):
    torch.manual_seed(seed); np.random.seed(seed)

    model = FixedWeightPINN(helm.DIM_PHYS, HIDDEN_DIMS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    balancer = ReLoBRaLoBalancer(n_terms=3, alpha=alpha, tau=tau,
                                 rho_mean=rho_mean, device=device)

    t0 = time.time()
    print(f"[seed {seed}, k={k_train:.2f}] ReLoBRaLo Adam: {n_epochs} steps")
    for step in range(n_epochs):
        opt.zero_grad()
        losses = helm.compute_losses_fixed(model, batch, k_train)
        L_vec = torch.stack([losses["pde"], losses["bc"], losses["data"]])
        lam = balancer(L_vec)
        total = (lam * L_vec).sum()
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()
        if step % max(1, n_epochs // 10) == 0:
            print(f"  step {step:6d}  L={total.item():.4e}  "
                  f"lam={[round(float(x), 3) for x in lam.detach().cpu().tolist()]}")

    elapsed = time.time() - t0
    rel_at_k_train = helm.evaluate_at_k(model, k_train, device, nx=eval_nx, is_lc=False)

    ckpt = REPO / "checkpoints" / f"relobralo_helmholtz_3d_seed{seed}_k{k_train:.2f}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "lambda_final": balancer.lam_prev.cpu(),
        "rel_l2": rel_at_k_train,
        "k_train": k_train, "elapsed_sec": elapsed, "seed": seed,
        "n_epochs": n_epochs, "lr": lr,
        "alpha": alpha, "tau": tau, "rho_mean": rho_mean,
    }, ckpt)
    print(f"[seed {seed}, k={k_train:.2f}] done in {elapsed/60:.1f} min  rel-L2 = {rel_at_k_train:.4e}")
    return {
        "seed": seed, "k_train": k_train,
        "rel_l2": rel_at_k_train, "elapsed_sec": elapsed,
        "checkpoint": str(ckpt.relative_to(REPO)),
        "lambda_final": [float(x) for x in balancer.lam_prev.cpu().tolist()],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--k-trains", type=float, nargs="+",
                   default=[1.0, 3.0, 5.0])
    p.add_argument("--n-epochs", type=int, default=25_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.999)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--rho-mean", type=float, default=0.999)
    p.add_argument("--eval-nx", type=int, default=24)
    args = p.parse_args()

    device = select_device()
    print(f"Device: {device_info(device)}")
    print(f"Config: seeds={args.seeds} k_trains={args.k_trains} n_epochs={args.n_epochs}\n")
    print(f"3D Helmholtz family: k ∈ [{helm.K_MIN}, {helm.K_MAX}]")
    batch = helm.generate_training_data(device=device)

    runs = []
    for s in args.seeds:
        for k in args.k_trains:
            runs.append(run_one(
                s, float(k), batch, device,
                args.n_epochs, args.lr, args.alpha, args.tau, args.rho_mean,
                args.eval_nx,
            ))

    per_k = {}
    for r in runs:
        per_k.setdefault(float(r["k_train"]), []).append(r["rel_l2"])
    per_k_summary = {
        f"{k:.2f}": {"mean": float(np.mean(v)), "std": float(np.std(v)), "n_seeds": len(v)}
        for k, v in sorted(per_k.items())
    }
    grid_means_per_seed = {}
    for r in runs:
        grid_means_per_seed.setdefault(r["seed"], []).append(r["rel_l2"])
    seed_means = [float(np.mean(v)) for v in grid_means_per_seed.values()]
    elapsed = [r["elapsed_sec"] for r in runs]

    summary = {
        "method": "relobralo", "equation": "helmholtz_3d",
        "config": vars(args), "runs": runs,
        "per_k": per_k_summary,
        "summary": {
            "n_seeds": len(args.seeds), "k_trains": list(map(float, args.k_trains)),
            "rel_l2_mean_over_k_then_seeds": float(np.mean(seed_means)),
            "rel_l2_std_over_seeds": float(np.std(seed_means)),
            "elapsed_total_sec": float(np.sum(elapsed)),
        },
    }
    out = REPO / "results" / "relobralo_helmholtz_3d.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out.relative_to(REPO)}")
    print(f"  rel-L2 (mean over k-grid then over seeds): "
          f"{summary['summary']['rel_l2_mean_over_k_then_seeds']:.4e} ± "
          f"{summary['summary']['rel_l2_std_over_seeds']:.4e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
