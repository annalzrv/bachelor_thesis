"""Validate the joint (x, lambda) HDMR with Fourier features and order-3 truncation.

Same synthetic polynomial benchmark as test_synthetic.py:

    u(z) = L1(z_1) + L2(z_2) + L1(z_3) + L1(z_1) * L1(z_3)

The order-3 Fourier HDMR should:
1. Reconstruct u with rel-RMSE < 5% (as in the tanh order-2 case)
2. Recover analytic Sobol indices within 0.03 absolute on every subset
3. Correctly recover the TRIPLET as ~0 (since there is no third-order
   structure in the synthetic).

This last check is critical: the Fourier-feature + order-3 architecture
adds capacity. We need to confirm it doesn't *hallucinate* a triplet
signal where none exists.

Run:
    cd /Users/anna/Desktop/research/anova
    python -m lc_anova.tests.test_synthetic_fourier
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_REPO = os.path.dirname(_PKG)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lc_anova.core.joint_hdmr import JointHDMR
from lc_anova.core.synthetic import (
    ANALYTIC_SOBOL,
    L1,
    L2,
    TOTAL_VARIANCE,
    u_synthetic,
)


def main(
    n_train: int = 30_000,
    n_val: int = 30_000,
    phase1_epochs: int = 40,
    phase2_epochs: int = 80,
    phase3_epochs: int = 80,
    hidden: int = 64,
    layers: int = 2,
    num_freqs: int = 4,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    z_tr = rng.uniform(0.0, 1.0, size=(n_train, 3)).astype(np.float32)
    z_va = rng.uniform(0.0, 1.0, size=(n_val, 3)).astype(np.float32)
    y_tr = u_synthetic(z_tr).astype(np.float32)
    y_va = u_synthetic(z_va).astype(np.float32)

    print(f"Train: N={n_train}  y range [{y_tr.min():.3f}, {y_tr.max():.3f}]")
    print(f"Total analytic variance V = {TOTAL_VARIANCE:.5f} (= 44/45)\n")

    jh = JointHDMR(
        dim_x=2, dim_lambda=1,
        hidden=hidden, layers=layers,
        max_order=3, use_fourier=True, num_freqs=num_freqs,
    )
    print(f"JointHDMR  d={jh.d}  max_order=3  hidden={hidden}  num_freqs={num_freqs}")
    print(f"  pairs={jh.model.pair_indices}  pair_class={jh.pair_classification}")

    x_tr = torch.tensor(z_tr[:, :2])
    l_tr = torch.tensor(z_tr[:, 2:3])
    y_tr_t = torch.tensor(y_tr)
    history = jh.fit(
        x_samples=x_tr,
        lambda_samples=l_tr,
        u_targets=y_tr_t,
        phase1_epochs=phase1_epochs,
        phase2_epochs=phase2_epochs,
        phase3_epochs=phase3_epochs,
        log_every=20,
    )
    final_rel = history[-1]["rel"]
    print(f"\nFinal train rel-RMSE: {final_rel:.4f}")

    x_va = torch.tensor(z_va[:, :2])
    l_va = torch.tensor(z_va[:, 2:3])
    z_va_t = torch.cat([x_va, l_va], dim=1).to(jh.device)
    y_va_t = torch.tensor(y_va).to(jh.device)
    y_va_centered = y_va_t - jh.y_mean
    jh.model.eval()
    with torch.no_grad():
        pred, *_ = jh.model(z_va_t, include_pairs=True, include_triplet=True, purify=True)
        val_rel = (torch.sqrt(torch.mean((pred - y_va_centered) ** 2))
                   / y_va_centered.std()).item()
    print(f"Validation rel-RMSE: {val_rel:.4f}")

    terms = jh.evaluate_terms(x_va, l_va)
    learned_sobol = terms["sobol"]

    analytic_sobol_with_triplet = dict(ANALYTIC_SOBOL)
    analytic_sobol_with_triplet[(0, 1, 2)] = 0.0

    print("\nSobol indices (variance proportions):")
    print(f"  {'subset':<10} {'analytic':>9} {'learned':>9} {'abs err':>8}")
    sobol_errors = {}
    for key in sorted(analytic_sobol_with_triplet.keys(), key=lambda kv: (len(kv), kv)):
        s_ana = analytic_sobol_with_triplet[key]
        s_lrn = learned_sobol.get(key, 0.0)
        err = abs(s_lrn - s_ana)
        sobol_errors[key] = err
        print(f"  {str(key):<10} {s_ana:>9.4f} {s_lrn:>9.4f} {err:>8.4f}")

    # Function-form recovery: f_1, f_2, f_3 against L1, L2, L1
    N_grid = 200
    grid = np.linspace(0.0, 1.0, N_grid, dtype=np.float32)
    fixed_other = 0.5
    z_for_f1 = np.stack([grid, np.full(N_grid, fixed_other, dtype=np.float32),
                         np.full(N_grid, fixed_other, dtype=np.float32)], axis=1)
    z_for_f2 = np.stack([np.full(N_grid, fixed_other, dtype=np.float32), grid,
                         np.full(N_grid, fixed_other, dtype=np.float32)], axis=1)
    z_for_f3 = np.stack([np.full(N_grid, fixed_other, dtype=np.float32),
                         np.full(N_grid, fixed_other, dtype=np.float32), grid], axis=1)

    def eval_main(z_arr, axis):
        z_t = torch.tensor(z_arr, dtype=torch.float32).to(jh.device)
        with torch.no_grad():
            _, m, _, _ = jh.model.evaluate_terms(z_t, include_triplet=True)
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
    checks = {
        "Val rel-RMSE < 0.05":           val_rel < 0.05,
        "Sobol abs err < 0.05 each":     all(e < 0.05 for e in sobol_errors.values()),
        "Triplet Sobol ~ 0 (<0.05)":     sobol_errors[(0, 1, 2)] < 0.05,
        "f_1 rel L2 err < 0.07":         err_f1 < 0.07,
        "f_2 rel L2 err < 0.07":         err_f2 < 0.07,
        "f_3 rel L2 err < 0.07":         err_f3 < 0.07,
    }
    print("\n" + "=" * 60)
    for name, ok in checks.items():
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {name}")
    all_pass = all(checks.values())
    print("=" * 60)
    print("ALL PASS" if all_pass else "SOME FAILED — Fourier+order-3 has regressed")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
