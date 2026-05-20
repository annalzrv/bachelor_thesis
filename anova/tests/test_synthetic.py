"""Validate the joint (x, lambda) HDMR on a synthetic polynomial benchmark.

The function u(z_1, z_2, z_3) = L1(z_1) + L2(z_2) + L1(z_3) + L1(z_1)*L1(z_3)
on [0,1]^3 has known analytic ANOVA terms and Sobol indices (see
synthetic.py). This script trains a JointHDMR on i.i.d. samples from
U[0,1]^3 and checks:

1. The learned reconstruction has rel-RMSE < 5% on the validation set.
2. The recovered Sobol indices match the analytic ones to < 3%
   absolute error on every subset.
3. The recovered main-effect functions f_1, f_2, f_3 match the
   analytic forms (L1, L2, L1) to < 5% point-wise L2 error on a dense
   grid.

Run with:
    cd /Users/anna/Desktop/research/anova
    python -m lc_anova.tests.test_synthetic

(Or from inside lc_anova/:  python tests/test_synthetic.py)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

# Path setup so we can run from inside the package
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)               # .../anova/lc_anova
_REPO = os.path.dirname(_PKG)               # .../anova
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lc_anova.core.joint_hdmr import JointHDMR
from lc_anova.core.synthetic import (
    ANALYTIC_SOBOL,
    L1,
    L2,
    TOTAL_VARIANCE,
    analytic_terms_on_grid,
    u_synthetic,
)


def main(
    n_train: int = 30_000,
    n_val: int = 30_000,
    phase1_epochs: int = 40,
    phase2_epochs: int = 80,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    # ---- Sample U[0,1]^3 (treat z_1, z_2 as spatial, z_3 as lambda) ------
    z_tr = rng.uniform(0.0, 1.0, size=(n_train, 3)).astype(np.float32)
    z_va = rng.uniform(0.0, 1.0, size=(n_val, 3)).astype(np.float32)
    y_tr = u_synthetic(z_tr).astype(np.float32)
    y_va = u_synthetic(z_va).astype(np.float32)

    print(f"Train: N={n_train}  y range [{y_tr.min():.3f}, {y_tr.max():.3f}]")
    print(f"Total analytic variance V = {TOTAL_VARIANCE:.5f} (= 44/45)\n")

    # ---- Fit JointHDMR -----------------------------------------------------
    jh = JointHDMR(dim_x=2, dim_lambda=1, hidden=32, layers=2)
    print(f"JointHDMR  d={jh.d}  pairs={jh.model.pair_indices}  "
          f"pair_class={jh.pair_classification}")
    x_tr = torch.tensor(z_tr[:, :2])
    l_tr = torch.tensor(z_tr[:, 2:3])
    y_tr_t = torch.tensor(y_tr)
    history = jh.fit(
        x_samples=x_tr,
        lambda_samples=l_tr,
        u_targets=y_tr_t,
        phase1_epochs=phase1_epochs,
        phase2_epochs=phase2_epochs,
        log_every=10,
    )
    final_rel = history[-1]["rel"]
    print(f"\nFinal train rel-RMSE: {final_rel:.4f}")

    # ---- Validation rel-RMSE (recompute on val set) ------------------------
    x_va = torch.tensor(z_va[:, :2])
    l_va = torch.tensor(z_va[:, 2:3])
    z_va_t = torch.cat([x_va, l_va], dim=1).to(jh.device)
    y_va_t = torch.tensor(y_va).to(jh.device)
    y_va_centered = y_va_t - jh.y_mean
    jh.model.eval()
    with torch.no_grad():
        pred, _, _ = jh.model(z_va_t, include_pairs=True, purify=True)
        val_rel = (torch.sqrt(torch.mean((pred - y_va_centered) ** 2)) /
                   y_va_centered.std()).item()
    print(f"Validation rel-RMSE: {val_rel:.4f}")

    # ---- Recovered Sobol indices vs analytic ------------------------------
    terms = jh.evaluate_terms(x_va, l_va)
    learned_sobol = terms["sobol"]

    print("\nSobol indices (variance proportions):")
    print(f"  {'subset':<8} {'analytic':>9} {'learned':>9} {'abs err':>8}")
    sobol_errors = {}
    for key in sorted(ANALYTIC_SOBOL.keys()):
        s_ana = ANALYTIC_SOBOL[key]
        s_lrn = learned_sobol.get(key, 0.0)
        err = abs(s_lrn - s_ana)
        sobol_errors[key] = err
        print(f"  {str(key):<8} {s_ana:>9.4f} {s_lrn:>9.4f} {err:>8.4f}")

    # ---- Main-effect function recovery ------------------------------------
    # Evaluate learned f_1, f_2, f_3 on a dense grid; compare to L1/L2/L1.
    N_grid = 200
    grid = np.linspace(0.0, 1.0, N_grid, dtype=np.float32)

    # Use a representative joint sample so the per-subset evaluator sees it
    fixed_other = 0.5  # at the marginal mean for the other axes
    z_for_f1 = np.stack([grid, np.full(N_grid, fixed_other), np.full(N_grid, fixed_other)], axis=1)
    z_for_f2 = np.stack([np.full(N_grid, fixed_other), grid, np.full(N_grid, fixed_other)], axis=1)
    z_for_f3 = np.stack([np.full(N_grid, fixed_other), np.full(N_grid, fixed_other), grid], axis=1)

    def eval_main(z_arr, axis):
        z_t = torch.tensor(z_arr, dtype=torch.float32).to(jh.device)
        with torch.no_grad():
            _, m, _ = jh.model.evaluate_terms(z_t)
        return m[:, axis].cpu().numpy()

    f1_lrn = eval_main(z_for_f1, 0)
    f2_lrn = eval_main(z_for_f2, 1)
    f3_lrn = eval_main(z_for_f3, 2)

    f1_ana = L1(grid).astype(np.float32)
    f2_ana = L2(grid).astype(np.float32)
    f3_ana = L1(grid).astype(np.float32)

    def rel_l2(pred, ref):
        return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-10))

    err_f1 = rel_l2(f1_lrn, f1_ana)
    err_f2 = rel_l2(f2_lrn, f2_ana)
    err_f3 = rel_l2(f3_lrn, f3_ana)
    print("\nMain-effect function recovery (rel L2 error on 200-pt grid):")
    print(f"  f_1 ~ L1(z_1):  {err_f1:.4f}")
    print(f"  f_2 ~ L2(z_2):  {err_f2:.4f}")
    print(f"  f_3 ~ L1(z_3):  {err_f3:.4f}")

    # ---- Pass / fail report -----------------------------------------------
    print("\n" + "=" * 60)
    checks = {
        "Val rel-RMSE < 0.05":          val_rel < 0.05,
        "Sobol abs err < 0.03 each":    all(e < 0.03 for e in sobol_errors.values()),
        "f_1 rel L2 err < 0.05":        err_f1 < 0.05,
        "f_2 rel L2 err < 0.05":        err_f2 < 0.05,
        "f_3 rel L2 err < 0.05":        err_f3 < 0.05,
    }
    for name, ok in checks.items():
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {name}")
    all_pass = all(checks.values())
    print("=" * 60)
    print("ALL PASS" if all_pass else "SOME FAILED — debug the orthogonalization or training.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
