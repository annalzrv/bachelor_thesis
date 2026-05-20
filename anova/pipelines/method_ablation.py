"""Method ablation on 2D Helm LC-PINN: every component justified.

Ablations compared (all on the same 2D Helm seed 0 LC-PINN, same N_train):
  (a) Fourier, order-3, full pipeline  — baseline
  (b) tanh    (no Fourier features), order-3
  (c) Fourier, order-2  (no triplet subnet in HDMR)
  (d) Fourier, order-3, mains only  (phase 2 & 3 = 0)

For each variant we report:
  - val rel-RMSE on the LC-PINN output
  - S_xyk recovered (or "n/a" if architecturally absent)
  - Total Sobol mass captured

Headline: only (a) recovers $S_{xyk} \\approx 0.43$ close to gold MC.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_REPO_ANOVA = _HERE.parent.parent
_REPO_THESIS_CODE = _REPO_ANOVA.parent / "thesis" / "code"
sys.path.insert(0, str(_REPO_ANOVA))
sys.path.insert(0, str(_REPO_THESIS_CODE))

from lc_anova.core.joint_hdmr import JointHDMR  # noqa
from lc_anova.pipelines.helmholtz_2d import load_lc_pinn, evaluate_lc_pinn_batch  # noqa

CK = _REPO_THESIS_CODE / "checkpoints"
RESULTS = _REPO_ANOVA / "lc_anova" / "results"


def run_variant(name: str, x_tr, k_tr, y_tr, x_va, k_va, y_va, *,
                 use_fourier: bool, max_order: int,
                 phase1_epochs: int, phase2_epochs: int, phase3_epochs: int | None,
                 device) -> dict:
    print(f"\n=== Variant: {name} ===")
    print(f"  use_fourier={use_fourier}  max_order={max_order}  "
          f"phases=({phase1_epochs}, {phase2_epochs}, {phase3_epochs})")
    t0 = time.time()
    jh = JointHDMR(dim_x=2, dim_lambda=1, hidden=64, layers=2,
                   max_order=max_order, use_fourier=use_fourier, num_freqs=4)
    jh.fit(x_tr, k_tr, y_tr,
           phase1_epochs=phase1_epochs,
           phase2_epochs=phase2_epochs,
           phase3_epochs=phase3_epochs,
           log_every=40)
    wall = time.time() - t0

    # Eval val rel-RMSE
    z_va = torch.cat([x_va, k_va], dim=1).to(device)
    y_va_c = (y_va.to(device) - jh.y_mean)
    jh.model.eval()
    with torch.no_grad():
        include_triplet = (max_order >= 3 and phase3_epochs is not None and phase3_epochs > 0)
        include_pairs = (phase2_epochs > 0)
        if jh._has_triplet:
            pred, *_ = jh.model(z_va, include_pairs=include_pairs,
                                 include_triplet=include_triplet, purify=True)
        else:
            pred, _, _ = jh.model(z_va, include_pairs=include_pairs, purify=True)
        val_rel = (torch.sqrt(F.mse_loss(pred, y_va_c)) / y_va_c.std()).item()

    terms = jh.evaluate_terms(x_va, k_va)
    sobol = terms["sobol"]
    sobol_named = {}
    for s, v in sobol.items():
        if len(s) == 1:
            key = ["S_x", "S_y", "S_k"][s[0]]
        elif len(s) == 2:
            key = {(0, 1): "S_xy", (0, 2): "S_xk", (1, 2): "S_yk"}[tuple(sorted(s))]
        elif len(s) == 3:
            key = "S_xyk"
        sobol_named[key] = float(v)

    total_captured = sum(sobol_named.values())
    print(f"  val rel-RMSE = {val_rel:.4f}   wall = {wall:.1f}s")
    print(f"  S_xyk = {sobol_named.get('S_xyk', 'n/a')}")
    print(f"  total Sobol mass = {total_captured:.4f}")

    return {
        "name": name,
        "use_fourier": use_fourier,
        "max_order": max_order,
        "phases": [phase1_epochs, phase2_epochs, phase3_epochs],
        "wall_seconds": wall,
        "val_rel_rmse": val_rel,
        "sobol": sobol_named,
        "total_captured": total_captured,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default=str(CK / "lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt"))
    ap.add_argument("--N-train", type=int, default=100_000)
    ap.add_argument("--N-val", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="lc_anova/results/method_ablation.json")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading {args.checkpoint}")
    model, _ = load_lc_pinn(args.checkpoint, device)

    rng = np.random.default_rng(args.seed)
    X_tr = rng.uniform(0, 1, size=(args.N_train, 3)).astype(np.float32)
    X_tr[:, 2] = X_tr[:, 2] * 2 - 1
    X_va = rng.uniform(0, 1, size=(args.N_val, 3)).astype(np.float32)
    X_va[:, 2] = X_va[:, 2] * 2 - 1

    x_tr = torch.tensor(X_tr[:, :2], device=device)
    k_tr = torch.tensor(X_tr[:, 2:3], device=device)
    x_va = torch.tensor(X_va[:, :2], device=device)
    k_va = torch.tensor(X_va[:, 2:3], device=device)
    with torch.no_grad():
        y_tr = evaluate_lc_pinn_batch(model, x_tr, k_tr)
        y_va = evaluate_lc_pinn_batch(model, x_va, k_va)
    print(f"sampled LC-PINN N_train={args.N_train:,} N_val={args.N_val:,}")

    variants = []

    # (a) Fourier order-3 full
    variants.append(run_variant(
        "fourier_order3_full", x_tr, k_tr, y_tr, x_va, k_va, y_va,
        use_fourier=True, max_order=3,
        phase1_epochs=40, phase2_epochs=80, phase3_epochs=120,
        device=device))

    # (b) tanh order-3 (no Fourier)
    variants.append(run_variant(
        "tanh_order3_no_fourier", x_tr, k_tr, y_tr, x_va, k_va, y_va,
        use_fourier=False, max_order=3,
        phase1_epochs=40, phase2_epochs=80, phase3_epochs=120,
        device=device))

    # (c) Fourier order-2 (no triplet subnet)
    variants.append(run_variant(
        "fourier_order2_no_triplet", x_tr, k_tr, y_tr, x_va, k_va, y_va,
        use_fourier=True, max_order=2,
        phase1_epochs=40, phase2_epochs=200, phase3_epochs=None,
        device=device))

    # (d) Fourier order-3, mains only
    variants.append(run_variant(
        "fourier_order3_mains_only", x_tr, k_tr, y_tr, x_va, k_va, y_va,
        use_fourier=True, max_order=3,
        phase1_epochs=200, phase2_epochs=0, phase3_epochs=0,
        device=device))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "N_train": args.N_train,
        "N_val": args.N_val,
        "variants": variants,
    }, indent=2))
    print(f"\nWrote {out}")

    print(f"\n{'variant':<30} {'rel-RMSE':>10} {'S_xyk':>9} {'total':>9}")
    for v in variants:
        sxyk = v["sobol"].get("S_xyk")
        sxyk_str = f"{sxyk:.4f}" if sxyk is not None else "n/a"
        print(f"{v['name']:<30} {v['val_rel_rmse']:>10.4f} {sxyk_str:>9} "
              f"{v['total_captured']:>9.4f}")


if __name__ == "__main__":
    main()
