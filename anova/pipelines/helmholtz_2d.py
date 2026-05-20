"""Apply JointHDMR to a trained 2D Helmholtz LC-PINN checkpoint.

Pipeline:
    1. Load a 2D Helmholtz FiLM+L-BFGS LC-PINN checkpoint.
    2. Wrap the network as a black-box callable u_theta(x, y, k_norm).
    3. Sample 30k joint points (x, y, k_norm) ~ U[0,1]^2 x U[-1,1].
    4. Evaluate u_theta on the samples -> training targets.
    5. Fit JointHDMR (dim_x=2, dim_lambda=1).
    6. Save the learned terms + Sobol indices + render three signature plots.

Run with:
    cd /Users/anna/Desktop/research/anova
    python -m lc_anova.pipelines.helmholtz_2d \
        --checkpoint /Users/anna/Desktop/research/thesis/code/checkpoints/lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt \
        --out-dir lc_anova/results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Path setup
_HERE = Path(__file__).resolve().parent
_REPO_ANOVA = _HERE.parent
_REPO_THESIS_CODE = _REPO_ANOVA.parent / "thesis" / "code"
if str(_REPO_ANOVA) not in sys.path:
    sys.path.insert(0, str(_REPO_ANOVA))
if str(_REPO_THESIS_CODE) not in sys.path:
    sys.path.insert(0, str(_REPO_THESIS_CODE))

from lc_anova.core.joint_hdmr import JointHDMR  # noqa: E402
from pinns.equations import helmholtz_2d as helm  # noqa: E402
from pinns.model import LossConditionalPINN  # noqa: E402


def load_lc_pinn(checkpoint_path: str, device: torch.device) -> tuple[LossConditionalPINN, dict]:
    """Load a 2D Helmholtz LC-PINN from checkpoint with matching architecture."""
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dims = ck.get("hidden_dims", [64, 64, 64, 64])
    conditioning = ck.get("conditioning", "film")
    model = LossConditionalPINN(
        dim_phys=helm.DIM_PHYS,
        dim_lambda=helm.DIM_LAMBDA,
        hidden_dims=hidden_dims,
        conditioning=conditioning,
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    meta = {
        "conditioning": conditioning,
        "hidden_dims": hidden_dims,
        "n_params": ck.get("n_params"),
        "n_lbfgs": ck.get("n_lbfgs"),
        "rel_l2_K_eval": ck.get("rel_l2_K_eval"),
        "k_grid": ck.get("k_grid"),
    }
    return model, meta


def sample_joint(n: int, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample n joint points (x, y, k_norm) ~ U[0,1]^2 x U[-1, 1]."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.0, 1.0, size=(n, 2)).astype(np.float32)
    k_norm = rng.uniform(-1.0, 1.0, size=(n, 1)).astype(np.float32)
    return (
        torch.tensor(xy, device=device),
        torch.tensor(k_norm, device=device),
    )


@torch.no_grad()
def evaluate_lc_pinn_batch(
    model: LossConditionalPINN,
    xy: torch.Tensor,
    k_norm: torch.Tensor,
    batch_size: int = 8192,
) -> torch.Tensor:
    """Apply LC-PINN row-wise; return u_theta(x_i, y_i, k_i) of shape (N,)."""
    n = xy.shape[0]
    out = torch.empty(n, device=xy.device)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        u = model(xy[start:end], k_norm[start:end])
        out[start:end] = u.squeeze(-1)
    return out


def reference_solution_on_samples(xy: torch.Tensor, k_norm: torch.Tensor) -> np.ndarray:
    """Analytic manufactured solution u_ref(x, y; k) at each (xy, k_norm) row."""
    xy_np = xy.detach().cpu().numpy()
    k_norm_np = k_norm.detach().cpu().numpy().squeeze(-1)
    k_np = helm.norm_to_k(k_norm_np)
    return helm.reference_solution(xy_np[:, 0], xy_np[:, 1], k_np)


def render_signatures(jh: JointHDMR, out_dir: Path, tag: str = ""):
    """Produce the three canonical-signature plots."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    device = jh.device

    # ---- (a) spatial main effect u_x(x, y) on a 64x64 grid ----------------
    nx = 64
    xs = torch.linspace(0.0, 1.0, nx, dtype=torch.float32)
    ys = torch.linspace(0.0, 1.0, nx, dtype=torch.float32)
    XX, YY = torch.meshgrid(xs, ys, indexing="ij")
    xy_grid = torch.stack([XX.reshape(-1), YY.reshape(-1)], dim=1)
    u_x_flat = jh.spatial_main_effect(xy_grid)
    u_x = u_x_flat.reshape(nx, nx)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(u_x.T, origin="lower", extent=[0, 1, 0, 1], aspect="equal", cmap="RdBu_r")
    ax.set_title(rf"Spatial main effect  $u_x(x, y)$  ({tag})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_dir / f"sig_spatial_main_{tag}.png", dpi=150)
    plt.close()

    # ---- (b) parameter main effect u_k(k) on a 100-pt grid ----------------
    nk = 100
    k_norm_grid = torch.linspace(-1.0, 1.0, nk, dtype=torch.float32).unsqueeze(-1)
    u_k = jh.parameter_main_effect(k_norm_grid)
    k_vals = helm.norm_to_k(k_norm_grid.squeeze(-1).numpy())

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(k_vals, u_k, "-", linewidth=2)
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.set_title(rf"Parameter main effect  $u_k(k)$  ({tag})")
    ax.set_xlabel("wavenumber  k"); ax.set_ylabel(r"$u_k$")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"sig_param_main_{tag}.png", dpi=150)
    plt.close()

    # ---- (c) cross-effect heatmaps in (x, k) and (y, k) -------------------
    # Sample on a coarser grid for the cross-effect render (50 x 50).
    n_xk = 50
    x_axis = torch.linspace(0.0, 1.0, n_xk, dtype=torch.float32)
    k_axis_norm = torch.linspace(-1.0, 1.0, n_xk, dtype=torch.float32)
    XX2, KK2 = torch.meshgrid(x_axis, k_axis_norm, indexing="ij")

    # u_{x_1, k}: fix y at 0.5
    xy_xk = torch.stack([XX2.reshape(-1), 0.5 * torch.ones(n_xk * n_xk)], dim=1)
    k_xk = KK2.reshape(-1, 1)
    cross_xk = jh.cross_effect(xy_xk, k_xk).reshape(n_xk, n_xk)

    # u_{y, k}: fix x at 0.5
    xy_yk = torch.stack([0.5 * torch.ones(n_xk * n_xk), XX2.reshape(-1)], dim=1)
    cross_yk = jh.cross_effect(xy_yk, k_xk).reshape(n_xk, n_xk)

    k_grid_vals = helm.norm_to_k(k_axis_norm.numpy())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, label in [(axes[0], cross_xk, "x"), (axes[1], cross_yk, "y")]:
        im = ax.imshow(
            data.T, origin="lower",
            extent=[0, 1, k_grid_vals[0], k_grid_vals[-1]],
            aspect="auto", cmap="RdBu_r",
        )
        ax.set_title(rf"Cross effect  $u_{{{label}, k}}({label}, k)$  ({tag})")
        ax.set_xlabel(label); ax.set_ylabel("k")
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_dir / f"sig_cross_{tag}.png", dpi=150)
    plt.close()

    print(f"  Wrote signature plots to {out_dir} (tag={tag!r})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to LC-PINN 2D Helmholtz .pt")
    ap.add_argument("--out-dir", default="lc_anova/results", help="where to write outputs")
    ap.add_argument("--n-samples", type=int, default=30_000)
    ap.add_argument("--n-val", type=int, default=30_000)
    ap.add_argument("--phase1-epochs", type=int, default=40)
    ap.add_argument("--phase2-epochs", type=int, default=80)
    ap.add_argument("--phase3-epochs", type=int, default=None, help="default: same as phase2")
    ap.add_argument("--max-order", type=int, default=2, help="HDMR truncation order (2 or 3)")
    ap.add_argument("--hidden", type=int, default=32, help="hidden width per subset MLP")
    ap.add_argument("--layers", type=int, default=2, help="hidden depth per subset MLP")
    ap.add_argument("--fourier", action="store_true", help="use Fourier-feature subset nets")
    ap.add_argument("--num-freqs", type=int, default=4, help="number of Fourier-feature octaves")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None, help="label for outputs (default: derived from checkpoint name)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or Path(args.checkpoint).stem
    print(f"== Pipeline: 2D Helmholtz LC-PINN -> JointHDMR  (tag={tag}) ==")

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # --- load LC-PINN ------------------------------------------------------
    print(f"Loading {args.checkpoint}")
    model, meta = load_lc_pinn(args.checkpoint, device)
    print(f"  conditioning={meta['conditioning']}  hidden={meta['hidden_dims']}  "
          f"params={meta['n_params']}")

    # --- sample joint (x, y, k_norm) --------------------------------------
    print(f"Sampling {args.n_samples} joint training points")
    xy_tr, k_tr = sample_joint(args.n_samples, args.seed, device)
    print(f"Evaluating LC-PINN on training points")
    u_tr = evaluate_lc_pinn_batch(model, xy_tr, k_tr)

    print(f"Sampling {args.n_val} joint validation points")
    xy_va, k_va = sample_joint(args.n_val, args.seed + 1, device)
    u_va = evaluate_lc_pinn_batch(model, xy_va, k_va)

    # Sanity: compare LC-PINN output to analytic reference on the val set
    u_ref = reference_solution_on_samples(xy_va, k_va)
    u_va_np = u_va.detach().cpu().numpy()
    rel = float(np.linalg.norm(u_va_np - u_ref) / (np.linalg.norm(u_ref) + 1e-10))
    print(f"  LC-PINN vs analytic reference, rel L2 = {rel:.4f}  "
          f"(this is the family-averaged rel-L^2; per-k may differ)")

    # --- fit JointHDMR -----------------------------------------------------
    jh = JointHDMR(dim_x=2, dim_lambda=1, hidden=args.hidden, layers=args.layers,
                   max_order=args.max_order, use_fourier=args.fourier,
                   num_freqs=args.num_freqs)
    fourier_tag = f"fourier(L={args.num_freqs})" if args.fourier else "tanh"
    print(f"\nJointHDMR  d={jh.d}  max_order={args.max_order}  "
          f"hidden={args.hidden}  layers={args.layers}  features={fourier_tag}  "
          f"pairs={jh.model.pair_indices}  pair_class={jh.pair_classification}")

    history = jh.fit(
        x_samples=xy_tr,
        lambda_samples=k_tr,
        u_targets=u_tr,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase3_epochs=args.phase3_epochs,
        log_every=10,
    )

    # Validation reconstruction
    z_va = torch.cat([xy_va, k_va], dim=1).to(device)
    y_va_t = u_va.to(device)
    y_va_centered = y_va_t - jh.y_mean
    jh.model.eval()
    with torch.no_grad():
        if jh._has_triplet:
            pred, *_ = jh.model(z_va, include_pairs=True, include_triplet=True, purify=True)
        else:
            pred, _, _ = jh.model(z_va, include_pairs=True, purify=True)
        val_rel = (torch.sqrt(torch.mean((pred - y_va_centered) ** 2))
                   / y_va_centered.std()).item()
    print(f"\nJointHDMR validation rel-RMSE on LC-PINN output: {val_rel:.4f}")

    # --- evaluate Sobol indices on val set --------------------------------
    terms = jh.evaluate_terms(xy_va, k_va)
    sobol = terms["sobol"]

    # Tag each subset so the table is readable
    name_for = {0: "x", 1: "y", 2: "k"}
    print("\nSobol indices on LC-PINN output:")
    print(f"  {'subset':<14} {'sobol':>9}")
    sobol_named = {}
    for key, val in sorted(sobol.items(), key=lambda kv: (len(kv[0]), kv[0])):
        names = "/".join(name_for[a] for a in key)
        sobol_named[names] = val
        print(f"  {names:<14} {val:>9.4f}")

    # --- save outputs ------------------------------------------------------
    out_json = out_dir / f"results_{tag}.json"
    payload = {
        "checkpoint": str(args.checkpoint),
        "tag": tag,
        "meta": {
            "conditioning": meta["conditioning"],
            "hidden_dims": meta["hidden_dims"],
            "n_params": meta["n_params"],
        },
        "lc_pinn_vs_reference_rel_l2": rel,
        "jointhdmr_val_rel_rmse": val_rel,
        "sobol_indices": sobol_named,
        "history": history,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_json}")

    print("Rendering signature plots...")
    render_signatures(jh, out_dir, tag=tag)

    # Save the trained HDMR for later analysis
    torch.save(jh.model.state_dict(), out_dir / f"hdmr_{tag}.pt")
    print(f"Wrote HDMR weights to {out_dir / f'hdmr_{tag}.pt'}")


if __name__ == "__main__":
    main()
