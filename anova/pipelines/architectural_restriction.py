"""Architecturally-restricted LC-PINNs: Sobol predicts the floor.

Sobol indices say:
  S_x + S_y + S_xy + S_k = 0.428  (additive can capture)
  S_xk + S_yk          = 0.146  (only pair architecture can capture)
  S_xyk                = 0.426  (only triplet architecture can capture)

Therefore — *as a prediction made before training any network* — we have:

  Additive  arch:  rel-L² floor ≥ sqrt(0.572) = 0.757
  Order-2   arch:  rel-L² floor ≥ sqrt(0.426) = 0.653
  Order-3   arch:  rel-L² floor ≥ 0           (no restriction)

We train each architecture with the same PINN loss as the standard
2D Helm LC-PINN and check the resulting mean rel-L² across 21 evenly
spaced k values.

If the trained restricted networks hit these floors, we've shown Sobol
indices give an a priori architectural complexity bound for solving the
PDE — a striking new connection between functional ANOVA and
neural-network expressivity.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_REPO_ANOVA = _HERE.parent.parent
_REPO_THESIS_CODE = _REPO_ANOVA.parent / "thesis" / "code"
sys.path.insert(0, str(_REPO_ANOVA))
sys.path.insert(0, str(_REPO_THESIS_CODE))

from pinns.equations import helmholtz_2d as helm  # noqa
from pinns.model import LossConditionalPINN  # noqa

RESULTS = _REPO_ANOVA / "lc_anova" / "results"


# ---------------------------------------------------------------------------
# Restricted architectures (forward signature compatible with helm.compute_losses)
# ---------------------------------------------------------------------------

class _FourierFeatures(nn.Module):
    """Positional encoding: z, sin(omega_l * z), cos(omega_l * z) for omega_l = 2^l * pi."""

    def __init__(self, in_dim: int, num_freqs: int = 6):
        super().__init__()
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        self.register_buffer(
            "freqs",
            torch.tensor([float(np.pi) * (2 ** l) for l in range(num_freqs)],
                          dtype=torch.float32))
        self.feat_dim = in_dim * (1 + 2 * num_freqs)

    def forward(self, x):
        # x: (N, in_dim)
        N, d = x.shape
        x_freq = x.unsqueeze(-1) * self.freqs.view(1, 1, -1)  # (N, d, L)
        sin_f = torch.sin(x_freq).reshape(N, d * self.num_freqs)
        cos_f = torch.cos(x_freq).reshape(N, d * self.num_freqs)
        return torch.cat([x, sin_f, cos_f], dim=-1)


def _mlp(in_dim: int, out_dim: int, hidden: int, layers: int,
         fourier_freqs: int = 0) -> nn.Module:
    """Optionally prefix with Fourier positional encoding (helps with high-k spatial)."""
    if fourier_freqs > 0:
        ff = _FourierFeatures(in_dim, num_freqs=fourier_freqs)
        prev = ff.feat_dim
    else:
        ff = None
        prev = in_dim
    mods = []
    for _ in range(layers):
        mods.append(nn.Linear(prev, hidden))
        mods.append(nn.Tanh())
        prev = hidden
    mods.append(nn.Linear(prev, out_dim))
    seq = nn.Sequential(*mods)
    for m in seq:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    if ff is None:
        return seq
    return nn.Sequential(ff, seq)


def _broadcast(k_norm, n):
    if k_norm.dim() == 1:
        k_norm = k_norm.unsqueeze(0)
    if k_norm.shape[0] == 1 and n > 1:
        k_norm = k_norm.expand(n, -1)
    return k_norm


class AdditiveLCPINN(nn.Module):
    """u(x, y, k) = f_xy(x, y) + f_k(k)
    Fourier features applied on spatial inputs only (k is scalar, no encoding needed).
    """

    def __init__(self, hidden: int = 64, layers: int = 4, num_freqs: int = 6):
        super().__init__()
        self.f_xy = _mlp(2, 1, hidden, layers, fourier_freqs=num_freqs)
        self.f_k = _mlp(1, 1, hidden, layers, fourier_freqs=0)

    def forward(self, coords, k_norm):
        k_norm = _broadcast(k_norm, coords.shape[0])
        return self.f_xy(coords) + self.f_k(k_norm)


class Order2LCPINN(nn.Module):
    """u(x, y, k) = f_xy(x, y) + f_k(k) + f_xk(x, k) + f_yk(y, k)
    Fourier features on spatial inputs (incl. spatial part of mixed pairs).
    """

    def __init__(self, hidden: int = 64, layers: int = 4, num_freqs: int = 6):
        super().__init__()
        self.f_xy = _mlp(2, 1, hidden, layers, fourier_freqs=num_freqs)
        self.f_k = _mlp(1, 1, hidden, layers, fourier_freqs=0)
        self.f_xk = _mlp(2, 1, hidden, layers, fourier_freqs=num_freqs)
        self.f_yk = _mlp(2, 1, hidden, layers, fourier_freqs=num_freqs)

    def forward(self, coords, k_norm):
        k_norm = _broadcast(k_norm, coords.shape[0])
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        return (self.f_xy(coords) + self.f_k(k_norm)
                + self.f_xk(torch.cat([x, k_norm], -1))
                + self.f_yk(torch.cat([y, k_norm], -1)))


# ---------------------------------------------------------------------------
# Training (Adam + cosine LR, optional L-BFGS finishing)
# ---------------------------------------------------------------------------

def train_lc_helm2d(
    model, batch, device, n_epochs: int, lr: float = 5e-4,
    n_k_per_step: int = 4, w_pde: float = 1.0, w_bc: float = 10.0,
    w_data: float = 10.0, log_every: int = 1000,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    history = []
    t0 = time.time()
    for step in range(n_epochs):
        opt.zero_grad()
        total = torch.zeros(1, device=device).squeeze()
        for _ in range(n_k_per_step):
            k_norm = (torch.rand(1, device=device) * 2.0) - 1.0
            losses = helm.compute_losses(model, k_norm, batch)
            total = total + (w_pde * losses["pde"] + w_bc * losses["bc"]
                             + w_data * losses["data"])
        total = total / n_k_per_step
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0:
            history.append({"step": step, "loss": float(total.item())})
            # Periodically also measure rel-L² to track convergence to floor
            if step % (log_every * 5) == 0:
                with torch.no_grad():
                    quick_rel = helm.evaluate_at_k(model, 5.5, device, nx=32, is_lc=True)
                print(f"  step {step:>6}  L={total.item():.4e}  rel-L²@k=5.5: {quick_rel:.3f}", flush=True)
            else:
                print(f"  step {step:>6}  L={total.item():.4e}", flush=True)
    return {"history": history, "elapsed_sec": time.time() - t0}


def lbfgs_finish(model, batch, device, n_iter: int = 500, n_k: int = 8,
                  w_pde: float = 1.0, w_bc: float = 1.0, w_data: float = 1.0,
                  resample_every: int = 100, log_every: int = 50):
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, max_eval=25,
                            tolerance_grad=1e-8, tolerance_change=1e-10,
                            history_size=50, line_search_fn="strong_wolfe")
    state = {"k_support": None, "last_loss": float("nan")}

    def refresh():
        state["k_support"] = (torch.rand(n_k, device=device) * 2.0) - 1.0

    refresh()

    def closure():
        opt.zero_grad()
        total = torch.zeros(1, device=device).squeeze()
        for k_norm in state["k_support"]:
            losses = helm.compute_losses(model, k_norm.unsqueeze(0), batch)
            total = total + (w_pde * losses["pde"] + w_bc * losses["bc"]
                             + w_data * losses["data"])
        total = total / n_k
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        state["last_loss"] = float(total.item())
        return total

    t0 = time.time()
    for it in range(n_iter):
        if it > 0 and it % resample_every == 0:
            refresh()
        opt.step(closure)
        if it % log_every == 0:
            print(f"  [lbfgs] iter {it:>4}  L={state['last_loss']:.4e}", flush=True)
        if not (state["last_loss"] == state["last_loss"]):
            print(f"  [lbfgs] NaN at iter {it} — abort"); break
    return {"elapsed_sec": time.time() - t0, "final_loss": state["last_loss"]}


def eval_rel_l2_grid(model, device, K_eval: int = 21):
    """Mean rel-L² over K_eval evenly spaced k values, on 64x64 grid."""
    k_grid = np.linspace(helm.K_MIN, helm.K_MAX, K_eval, dtype=np.float32)
    errs = [helm.evaluate_at_k(model, float(k), device, nx=64, is_lc=True)
            for k in k_grid]
    return {"k_grid": k_grid.tolist(), "rel_l2_per_k": errs,
            "mean_rel_l2": float(np.mean(errs))}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_arch(name: str, model_cls, batch, device,
              n_adam: int, n_lbfgs: int, seed: int):
    torch.manual_seed(seed)
    print(f"\n=== Training '{name}' ===")
    model = model_cls(hidden=64, layers=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  n_params = {n_params:,}")
    adam_out = train_lc_helm2d(model, batch, device, n_epochs=n_adam, lr=5e-4,
                                n_k_per_step=4, log_every=1000)
    # L-BFGS with NaN guard and fewer iterations to avoid divergence
    if n_lbfgs > 0:
        lbfgs_out = lbfgs_finish(model, batch, device, n_iter=n_lbfgs,
                                  n_k=8, resample_every=100, log_every=50)
        if lbfgs_out["final_loss"] != lbfgs_out["final_loss"]:  # NaN
            print(f"  L-BFGS NaN'd; relying on Adam result", flush=True)
    else:
        lbfgs_out = {"elapsed_sec": 0.0, "final_loss": 0.0}
    eval_out = eval_rel_l2_grid(model, device, K_eval=21)
    print(f"  mean rel-L² over k grid: {eval_out['mean_rel_l2']:.4f}")
    return {
        "name": name, "n_params": n_params,
        "adam_elapsed_sec": adam_out["elapsed_sec"],
        "lbfgs_elapsed_sec": lbfgs_out["elapsed_sec"],
        "adam_history_tail": adam_out["history"][-5:],
        "lbfgs_final_loss": lbfgs_out["final_loss"],
        "eval": eval_out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-adam", type=int, default=25000)
    ap.add_argument("--n-lbfgs", type=int, default=500)
    ap.add_argument("--out", default="lc_anova/results/restricted_lcpinn.json")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Sobol-derived predictions (from results_mc_megaN_seed0.json)
    S_x, S_y, S_k = 0.038, 0.039, 0.073
    S_xy, S_xk, S_yk = 0.278, 0.073, 0.073
    S_xyk = 0.426
    pred_additive = float(np.sqrt(S_xk + S_yk + S_xyk))
    pred_order2 = float(np.sqrt(S_xyk))
    print(f"\nSobol-derived predictions for rel-L² floor:")
    print(f"  Additive  arch:  >= sqrt(S_xk + S_yk + S_xyk) = sqrt({S_xk + S_yk + S_xyk:.3f}) = {pred_additive:.4f}")
    print(f"  Order-2   arch:  >= sqrt(S_xyk) = sqrt({S_xyk:.3f}) = {pred_order2:.4f}")

    batch = helm.generate_training_data(
        n_pde=4096, n_bc=400, n_data=500, n_data_k_values=5,
        seed=args.seed, device=device,
    )

    results = []
    results.append(run_arch("additive", AdditiveLCPINN, batch, device,
                             args.n_adam, args.n_lbfgs, args.seed))
    results.append(run_arch("order2", Order2LCPINN, batch, device,
                             args.n_adam, args.n_lbfgs, args.seed))

    # Compare measured vs predicted floor
    print(f"\n{'arch':<12} {'n_params':>10} {'predicted floor':>17} {'measured mean rel-L²':>22} {'bound holds':>12}")
    summary = []
    for r in results:
        name = r["name"]
        meas = r["eval"]["mean_rel_l2"]
        pred = pred_additive if name == "additive" else pred_order2
        bound_ok = meas >= 0.9 * pred  # the floor is a LOWER bound, ≥ is correct direction
        print(f"{name:<12} {r['n_params']:>10,} {pred:>17.4f} {meas:>22.4f} {str(bound_ok):>12}")
        summary.append({"name": name, "predicted_floor": pred, "measured_rel_l2": meas,
                        "ratio_meas_over_pred": meas / pred})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "seed": args.seed, "n_adam": args.n_adam, "n_lbfgs": args.n_lbfgs,
        "sobol_used": {"S_x": S_x, "S_y": S_y, "S_k": S_k,
                        "S_xy": S_xy, "S_xk": S_xk, "S_yk": S_yk,
                        "S_xyk": S_xyk},
        "predictions": {"additive": pred_additive, "order2": pred_order2},
        "summary": summary,
        "results": results,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
