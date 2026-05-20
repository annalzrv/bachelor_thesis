"""Runnable companion for the lid-driven cavity benchmark.

Trains LC-PINN (uniform λ sampling) and an equal-weights baseline on the
Re=400 steady lid-driven cavity, runs a FAIR λ sweep (exclude_terms=set()),
evaluates rel-L2 against Ghia et al. (1982) centerline reference, and saves
figures + JSON to results/.

Usage (from thesis/code):
    python scripts/run_navier_stokes_cavity.py --n-steps 200000 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pinns.baseline import FixedWeightPINN, train_fixed_pinn
from pinns.device import select_device, device_info
from pinns.equations import navier_stokes_cavity as ns
from pinns.inference import sweep_lambda
from pinns.lambda_sampler import LambdaSampler
from pinns.model import LossConditionalPINN
from pinns.training import train_lc_pinn

REPO = pathlib.Path(__file__).resolve().parent.parent
HIDDEN_DIMS = [64, 64, 64, 64]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=200_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-pde", type=int, default=8000)
    parser.add_argument("--n-bc", type=int, default=200, help="per wall")
    parser.add_argument("--sweep-candidates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-lc", action="store_true", help="skip LC-PINN, baseline only")
    parser.add_argument("--skip-baseline", action="store_true", help="skip baseline, LC only")
    args = parser.parse_args()

    device = select_device()
    print(f"Device: {device_info(device)}")
    print(f"Re={ns.RE:.0f}  n_steps={args.n_steps}  lr={args.lr}  seed={args.seed}\n")

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("Building training batch (Ghia centerline data)…")
    batch = ns.generate_training_data(
        n_pde=args.n_pde, n_bc_per_wall=args.n_bc,
        n_data_ghia=True, seed=args.seed, device=device,
    )
    print({k: tuple(v.shape) for k, v in batch.items()})

    results: dict = {
        "config": {
            "re": ns.RE, "nu": ns.NU,
            "n_steps": args.n_steps, "lr": args.lr, "hidden_dims": HIDDEN_DIMS,
            "n_pde": args.n_pde, "n_bc_per_wall": args.n_bc,
            "n_data_ghia": True, "seed": args.seed,
        },
    }

    model_lc = None
    best_ll = None
    if not args.skip_lc:
        print("\n=== LC-PINN (uniform λ) ===")
        torch.manual_seed(args.seed)
        model_lc = LossConditionalPINN(
            dim_phys=ns.DIM_PHYS, dim_lambda=ns.DIM_LAMBDA,
            hidden_dims=HIDDEN_DIMS, dim_out=ns.DIM_OUT,
        ).to(device)
        sampler = LambdaSampler(dim=ns.DIM_LAMBDA, device=device, mode="uniform")

        t0 = time.time()
        _ = train_lc_pinn(
            model_lc, sampler, batch, device,
            loss_fn=ns.compute_losses,
            n_epochs=args.n_steps, lr=args.lr, log_every=5_000,
        )
        print(f"LC-PINN training: {(time.time() - t0)/60:.1f} min")

        print("\n=== FAIR λ sweep ===")
        torch.manual_seed(args.seed)
        best_ll, best_p, _ = sweep_lambda(
            model_lc, batch, sampler, device,
            loss_fn=ns.compute_losses,
            n_candidates=args.sweep_candidates, exclude_terms=set(),
        )
        best_p_np = best_p.cpu().numpy().round(4)
        print(f"Best λ (fair, [pde, bc, data]): {best_p_np}")

        lc_errs = ns.evaluate(model_lc, best_ll, device)
        print("\nLC-PINN centerline rel-L2:")
        print(f"  u-centerline (x=0.5): {lc_errs['u_centerline_rel_l2']:.4f}")
        print(f"  v-centerline (y=0.5): {lc_errs['v_centerline_rel_l2']:.4f}")
        print(f"  mean:                 {lc_errs['mean_rel_l2']:.4f}")

        torch.save(
            {"model_state_dict": model_lc.state_dict(), "best_log_lambda": best_ll},
            REPO / "checkpoints" / "ns_cavity_lc_pinn_uniform.pt",
        )
        results["lc_pinn_uniform"] = {
            "best_lambda": best_p_np.tolist(),
            "centerline_rel_l2": lc_errs,
        }

    model_bl = None
    if not args.skip_baseline:
        print("\n=== Equal-weights baseline ===")
        torch.manual_seed(args.seed)
        model_bl = FixedWeightPINN(
            dim_phys=ns.DIM_PHYS, hidden_dims=HIDDEN_DIMS, dim_out=ns.DIM_OUT,
        ).to(device)
        t0 = time.time()
        _ = train_fixed_pinn(
            model_bl, weights=[1.0 / 3, 1.0 / 3, 1.0 / 3],
            batch=batch, device=device,
            loss_fn=ns.compute_losses_fixed,
            n_epochs=args.n_steps, lr=args.lr, log_every=5_000,
            desc="Cavity equal-weights baseline",
        )
        print(f"Baseline training: {(time.time() - t0)/60:.1f} min")

        bl_errs = ns.evaluate(model_bl, None, device)
        print("\nBaseline centerline rel-L2:")
        print(f"  u-centerline (x=0.5): {bl_errs['u_centerline_rel_l2']:.4f}")
        print(f"  v-centerline (y=0.5): {bl_errs['v_centerline_rel_l2']:.4f}")
        print(f"  mean:                 {bl_errs['mean_rel_l2']:.4f}")
        torch.save({"model_state_dict": model_bl.state_dict()},
                   REPO / "checkpoints" / "ns_cavity_baseline_equal.pt")
        results["baseline_equal"] = {
            "centerline_rel_l2": bl_errs,
        }

    # --- Figure: centerline profiles + u-field
    y_c, u_c_ref = ns.ghia_u_centerline()
    x_c, v_c_ref = ns.ghia_v_centerline()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(u_c_ref, y_c, "ko", label="Ghia 1982", ms=5)
    if model_lc is not None:
        pts = np.column_stack([np.full_like(y_c, 0.5), y_c]).astype(np.float32)
        u_lc = ns.predict_at_points(model_lc, best_ll, pts, device)[:, 0]
        ax.plot(u_lc, y_c, "r-", lw=1.3, label="LC-PINN")
    if model_bl is not None:
        pts = np.column_stack([np.full_like(y_c, 0.5), y_c]).astype(np.float32)
        u_bl = ns.predict_at_points(model_bl, None, pts, device)[:, 0]
        ax.plot(u_bl, y_c, "b--", lw=1.3, label="baseline")
    ax.set_xlabel("u"); ax.set_ylabel("y"); ax.set_title("u on x=0.5 vs Ghia Re=400")
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[1]
    ax.plot(x_c, v_c_ref, "ko", label="Ghia 1982", ms=5)
    if model_lc is not None:
        pts = np.column_stack([x_c, np.full_like(x_c, 0.5)]).astype(np.float32)
        v_lc = ns.predict_at_points(model_lc, best_ll, pts, device)[:, 1]
        ax.plot(x_c, v_lc, "r-", lw=1.3, label="LC-PINN")
    if model_bl is not None:
        pts = np.column_stack([x_c, np.full_like(x_c, 0.5)]).astype(np.float32)
        v_bl = ns.predict_at_points(model_bl, None, pts, device)[:, 1]
        ax.plot(x_c, v_bl, "b--", lw=1.3, label="baseline")
    ax.set_xlabel("x"); ax.set_ylabel("v"); ax.set_title("v on y=0.5 vs Ghia Re=400")
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[2]
    ref_model = model_lc if model_lc is not None else model_bl
    ref_ll = best_ll if model_lc is not None else None
    if ref_model is not None:
        X, Y, u_g, v_g, _ = ns.predict_uvp_grid(ref_model, ref_ll, nx=129, ny=129, device=device)
        speed = np.sqrt(u_g ** 2 + v_g ** 2)
        im = ax.imshow(speed.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", aspect="equal")
        plt.colorbar(im, ax=ax, fraction=0.046)
        label = "LC-PINN" if model_lc is not None else "baseline"
        ax.set_title(f"speed |u| — {label}")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    fig.suptitle(f"Lid-driven cavity Re={ns.RE:.0f}", y=1.02)
    plt.tight_layout()
    out_png = REPO / "results" / "fig_ns_cavity_re400.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_png.relative_to(REPO)}")

    out_json = REPO / "results" / "ns_cavity_re400_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_json.relative_to(REPO)}")
    print("\n" + json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
