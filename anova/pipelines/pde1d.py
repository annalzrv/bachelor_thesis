"""Generic pipeline for d=2 parametric PDEs (1D Helmholtz, Schrödinger 1D).

Both PDEs have 1 spatial dim + 1 physical parameter. The LC-PINN is
$u_\\theta(x, \\lambda_\\mathrm{norm})$ with $\\lambda \\in [\\lambda_\\min, \\lambda_\\max]$
and $\\lambda_\\mathrm{norm} \\in [-1, 1]$.

Joint ANOVA decomposition:
  $u_x(x)$       — main effect over spatial axis
  $u_\\lambda(\\lambda)$ — main effect over parameter axis
  $u_{x, \\lambda}(x, \\lambda)$ — pair effect (the only non-trivial pair)

In d=2 there is no triplet, so this is order-2. We still apply Fourier
features to handle high-frequency $\\cos(kx)$ etc.

Run:
    python -m lc_anova.pipelines.pde1d --pde helmholtz \\
        --checkpoint .../lc_pinn_helmholtz_seed0_film_lbfgs.pt \\
        --fourier --num-freqs 6 --hidden 128 --layers 3
    python -m lc_anova.pipelines.pde1d --pde schrodinger ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO_ANOVA = _HERE.parent
_REPO_THESIS_CODE = _REPO_ANOVA.parent / "thesis" / "code"
if str(_REPO_ANOVA) not in sys.path:
    sys.path.insert(0, str(_REPO_ANOVA))
if str(_REPO_THESIS_CODE) not in sys.path:
    sys.path.insert(0, str(_REPO_THESIS_CODE))

from lc_anova.core.joint_hdmr import JointHDMR  # noqa: E402
from pinns.equations import helmholtz as helm1d  # noqa: E402
from pinns.equations import schrodinger_1d as schr  # noqa: E402
from pinns.model import LossConditionalPINN  # noqa: E402


def pde_config(name: str) -> dict:
    """Return PDE-specific configuration."""
    if name == "helmholtz":
        return {
            "name": "helmholtz_1d",
            "module": helm1d,
            "param_name": "k",
            "param_min": helm1d.K_MIN,
            "param_max": helm1d.K_MAX,
            "x_min": helm1d.X_MIN,
            "x_max": helm1d.X_MAX,
            "param_to_norm": helm1d.k_to_norm,
            "norm_to_param": helm1d.norm_to_k,
            "ref_fn": helm1d.reference_solution,  # (x_array, k_scalar) -> u_array
        }
    elif name == "schrodinger":
        return {
            "name": "schrodinger_1d",
            "module": schr,
            "param_name": "alpha",
            "param_min": schr.ALPHA_MIN,
            "param_max": schr.ALPHA_MAX,
            "x_min": schr.X_MIN,
            "x_max": schr.X_MAX,
            "param_to_norm": schr.alpha_to_norm,
            "norm_to_param": schr.norm_to_alpha,
            "ref_fn": schr.reference_solution,  # (x_array, alpha_scalar) -> u_array
        }
    raise ValueError(f"unknown PDE {name!r}; expected helmholtz or schrodinger")


def load_lc_pinn(checkpoint_path: str, pde: dict, device: torch.device) -> tuple[LossConditionalPINN, dict]:
    """Load a d=2 LC-PINN from checkpoint."""
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dims = ck.get("hidden_dims", [64, 64, 64, 64])
    conditioning = ck.get("conditioning", "film")
    model = LossConditionalPINN(
        dim_phys=pde["module"].DIM_PHYS,
        dim_lambda=pde["module"].DIM_LAMBDA,
        hidden_dims=hidden_dims,
        conditioning=conditioning,
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, {
        "conditioning": conditioning,
        "hidden_dims": hidden_dims,
        "n_params": ck.get("n_params"),
    }


def sample_joint(n: int, seed: int, pde: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(pde["x_min"], pde["x_max"], size=(n, 1)).astype(np.float32)
    p_norm = rng.uniform(-1.0, 1.0, size=(n, 1)).astype(np.float32)
    return torch.tensor(x, device=device), torch.tensor(p_norm, device=device)


@torch.no_grad()
def evaluate_lc_pinn_batch(
    model: LossConditionalPINN,
    x: torch.Tensor,
    p_norm: torch.Tensor,
    batch_size: int = 8192,
) -> torch.Tensor:
    n = x.shape[0]
    out = torch.empty(n, device=x.device)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        u = model(x[start:end], p_norm[start:end])
        out[start:end] = u.squeeze(-1)
    return out


def reference_solution_on_samples(x: torch.Tensor, p_norm: torch.Tensor, pde: dict) -> np.ndarray:
    """Vectorize the per-(x, param) reference call across the batch."""
    x_np = x.detach().cpu().numpy().squeeze(-1)
    p_np = p_norm.detach().cpu().numpy().squeeze(-1)
    p_vals = pde["norm_to_param"](p_np)
    # Most reference_solution functions accept array x and scalar param.
    # We need to vectorize over (x, param) pairs: compute u(x_i, p_i) for each i.
    out = np.empty_like(x_np)
    # Group by parameter for efficiency (each unique p uses one call)
    # But since params are continuous, just call per-row. Slow but correct.
    for i in range(len(x_np)):
        out[i] = pde["ref_fn"](np.array([x_np[i]]), float(p_vals[i]))[0]
    return out


def render_d2_signatures(jh: JointHDMR, out_dir: Path, tag: str, pde: dict):
    """Three signature plots for d=2: u_x(x), u_lambda(lambda), u_{x,lambda}(x, lambda)."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    device = jh.device

    # u_x(x) on a fine x grid; lambda zeroed
    nx = 200
    x_axis = torch.linspace(pde["x_min"], pde["x_max"], nx, dtype=torch.float32)
    x_grid = x_axis.unsqueeze(-1)
    u_x = jh.spatial_main_effect(x_grid)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x_axis.numpy(), u_x, "-", linewidth=2)
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.set_title(rf"Spatial main effect  $u_x(x)$  ({tag})")
    ax.set_xlabel("x"); ax.set_ylabel(r"$u_x$")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"sig_d2_spatial_main_{tag}.png", dpi=150)
    plt.close()

    # u_lambda(lambda) over the parameter range
    nk = 200
    p_norm_grid = torch.linspace(-1.0, 1.0, nk, dtype=torch.float32).unsqueeze(-1)
    u_p = jh.parameter_main_effect(p_norm_grid)
    p_vals = pde["norm_to_param"](p_norm_grid.squeeze(-1).numpy())

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(p_vals, u_p, "-", linewidth=2)
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.set_title(rf"Parameter main effect  $u_{{{pde['param_name']}}}({pde['param_name']})$  ({tag})")
    ax.set_xlabel(pde["param_name"]); ax.set_ylabel(rf"$u_{{{pde['param_name']}}}$")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"sig_d2_param_main_{tag}.png", dpi=150)
    plt.close()

    # Cross-effect heatmap u_{x, lambda}(x, lambda)
    n_grid = 80
    xx = torch.linspace(pde["x_min"], pde["x_max"], n_grid, dtype=torch.float32)
    pp = torch.linspace(-1.0, 1.0, n_grid, dtype=torch.float32)
    XX, PP = torch.meshgrid(xx, pp, indexing="ij")
    x_flat = XX.reshape(-1, 1)
    p_flat = PP.reshape(-1, 1)
    cross = jh.cross_effect(x_flat, p_flat).reshape(n_grid, n_grid)
    p_vals = pde["norm_to_param"](pp.numpy())

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        cross.T, origin="lower",
        extent=[pde["x_min"], pde["x_max"], p_vals[0], p_vals[-1]],
        aspect="auto", cmap="RdBu_r",
    )
    ax.set_title(rf"Cross effect  $u_{{x, {pde['param_name']}}}$ ({tag})")
    ax.set_xlabel("x"); ax.set_ylabel(pde["param_name"])
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_dir / f"sig_d2_cross_{tag}.png", dpi=150)
    plt.close()
    print(f"  Wrote signature plots to {out_dir} (tag={tag!r})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", required=True, choices=["helmholtz", "schrodinger"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default="lc_anova/results")
    ap.add_argument("--n-samples", type=int, default=30_000)
    ap.add_argument("--n-val", type=int, default=30_000)
    ap.add_argument("--phase1-epochs", type=int, default=40)
    ap.add_argument("--phase2-epochs", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--fourier", action="store_true")
    ap.add_argument("--num-freqs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    pde = pde_config(args.pde)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{pde['name']}_{Path(args.checkpoint).stem}"
    print(f"== Pipeline: {pde['name']} LC-PINN -> JointHDMR  (tag={tag}) ==")

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"PDE: {pde['name']}  param={pde['param_name']} in [{pde['param_min']}, {pde['param_max']}]")

    model, meta = load_lc_pinn(args.checkpoint, pde, device)
    print(f"  conditioning={meta['conditioning']}  hidden={meta['hidden_dims']}  params={meta['n_params']}")

    x_tr, p_tr = sample_joint(args.n_samples, args.seed, pde, device)
    print(f"Evaluating LC-PINN on {args.n_samples} training points")
    u_tr = evaluate_lc_pinn_batch(model, x_tr, p_tr)

    x_va, p_va = sample_joint(args.n_val, args.seed + 1, pde, device)
    u_va = evaluate_lc_pinn_batch(model, x_va, p_va)

    u_ref = reference_solution_on_samples(x_va, p_va, pde)
    u_va_np = u_va.detach().cpu().numpy()
    rel = float(np.linalg.norm(u_va_np - u_ref) / (np.linalg.norm(u_ref) + 1e-10))
    print(f"  LC-PINN vs analytic reference, rel L2 = {rel:.4f}")

    jh = JointHDMR(
        dim_x=1, dim_lambda=1,
        hidden=args.hidden, layers=args.layers,
        max_order=2,  # d=2 has no triplet
        use_fourier=args.fourier, num_freqs=args.num_freqs,
    )
    print(f"\nJointHDMR  d={jh.d}  max_order=2  hidden={args.hidden}  layers={args.layers}  "
          f"features={'fourier' if args.fourier else 'tanh'}")

    history = jh.fit(
        x_samples=x_tr, lambda_samples=p_tr, u_targets=u_tr,
        phase1_epochs=args.phase1_epochs, phase2_epochs=args.phase2_epochs,
        log_every=20,
    )

    z_va = torch.cat([x_va, p_va], dim=1).to(device)
    y_va_t = u_va.to(device)
    y_va_centered = y_va_t - jh.y_mean
    jh.model.eval()
    with torch.no_grad():
        pred, _, _ = jh.model(z_va, include_pairs=True, purify=True)
        val_rel = (torch.sqrt(torch.mean((pred - y_va_centered) ** 2))
                   / y_va_centered.std()).item()
    print(f"\nJointHDMR validation rel-RMSE on LC-PINN output: {val_rel:.4f}")

    terms = jh.evaluate_terms(x_va, p_va)
    sobol = terms["sobol"]
    name_for = {0: "x", 1: pde["param_name"]}

    print(f"\nSobol indices on LC-PINN output:")
    print(f"  {'subset':<10} {'sobol':>9}")
    sobol_named = {}
    for key, val in sorted(sobol.items(), key=lambda kv: (len(kv[0]), kv[0])):
        names = "/".join(name_for[a] for a in key)
        sobol_named[names] = val
        print(f"  {names:<10} {val:>9.4f}")

    out_json = out_dir / f"results_{tag}.json"
    payload = {
        "pde": pde["name"],
        "checkpoint": str(args.checkpoint),
        "tag": tag,
        "meta": meta,
        "lc_pinn_vs_reference_rel_l2": rel,
        "jointhdmr_val_rel_rmse": val_rel,
        "sobol_indices": sobol_named,
        "history": history,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_json}")

    print("Rendering signature plots...")
    render_d2_signatures(jh, out_dir, tag, pde)


if __name__ == "__main__":
    main()
