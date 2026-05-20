"""Sparse PCE baseline via LASSO on the Legendre basis.

Full-tensor OLS PCE (`pce_baseline.py` / experiments.ipynb Experiment 1) is
the textbook construction but not how modern sensitivity analysis actually
uses PCE. The fair baseline for high-d sparse problems is `LASSO-PCE`: same
Legendre basis, fit by L1-regularised regression so only a few coefficients
survive.

At d=20 with five active inputs (Sobol G, a=[0,1,4.5,9,99,99,...,99]) most
of the variance lives in low-order terms of just five inputs. Full OLS at
P=4 has to fit 10 626 coefficients with N=30k samples — overdetermined but
near the noise floor. LASSO should be able to do as well with far fewer
active coefficients, and may close the gap to neural HDMR.

Procedure:
  - Build same Legendre design matrix as the OLS baseline.
  - Carve off 10% of the training set for alpha selection.
  - Sweep alpha over a small geometric grid; pick the one that minimises
    held-out RMSE.
  - Refit on the full training set with that alpha.
  - Evaluate on the validation set; compute Sobol indices from the
    surviving coefficients.
"""

import itertools
import json
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import Legendre
from sklearn.linear_model import Lasso

from sobol_g import g_function, main_effect_sobol, pair_sobol


def legendre_normalised(n, x):
    x_mapped = 2.0 * x - 1.0
    c = np.zeros(n + 1); c[n] = 1.0
    return np.sqrt(2 * n + 1) * Legendre(c)(x_mapped)


def multi_indices(d, P):
    out = [(0,) * d]
    for total in range(1, P + 1):
        for positions in itertools.combinations_with_replacement(range(d), total):
            alpha = [0] * d
            for p in positions:
                alpha[p] += 1
            out.append(tuple(alpha))
    return out


def design_matrix(X, multi_idx, P):
    N, d = X.shape
    L = np.zeros((P + 1, d, N), dtype=np.float32)
    for j in range(d):
        for n in range(P + 1):
            L[n, j, :] = legendre_normalised(n, X[:, j])
    K = len(multi_idx)
    Phi = np.ones((N, K), dtype=np.float32)
    for k, alpha in enumerate(multi_idx):
        col = np.ones(N, dtype=np.float32)
        for j in range(d):
            if alpha[j] > 0:
                col *= L[alpha[j], j, :]
        Phi[:, k] = col
    return Phi


def sobol_from_pce(coeffs, multi_idx, d):
    """Main + pair Sobol indices from PCE coefficients on normalised basis."""
    total_var = 0.0
    main_var = np.zeros(d)
    pair_var = {}
    for c, alpha in zip(coeffs, multi_idx):
        if all(a == 0 for a in alpha):
            continue
        support = tuple(j for j in range(d) if alpha[j] > 0)
        total_var += c**2
        if len(support) == 1:
            main_var[support[0]] += c**2
        elif len(support) == 2:
            pair_var[support] = pair_var.get(support, 0.0) + c**2
    if total_var == 0:
        return main_var, {}
    return main_var / total_var, {k: v / total_var for k, v in pair_var.items()}


def fit_lasso_with_alpha_search(Phi_tr_full, y_tr_full, alphas, val_frac=0.1, seed=0):
    """Sweep alpha, pick by held-out RMSE, refit on full training set."""
    rng = np.random.default_rng(seed)
    n = len(y_tr_full)
    idx = rng.permutation(n)
    cut = int(n * (1 - val_frac))
    tr_idx, va_idx = idx[:cut], idx[cut:]
    Phi_tr = Phi_tr_full[tr_idx]; y_tr = y_tr_full[tr_idx]
    Phi_va = Phi_tr_full[va_idx]; y_va = y_tr_full[va_idx]

    best_alpha, best_rmse = None, np.inf
    for alpha in alphas:
        m = Lasso(alpha=alpha, fit_intercept=False, max_iter=5000, tol=1e-3)
        m.fit(Phi_tr, y_tr)
        rmse = float(np.sqrt(np.mean((Phi_va @ m.coef_ - y_va) ** 2)))
        nz = int((np.abs(m.coef_) > 1e-8).sum())
        print(f"    alpha={alpha:.2e}  held-out RMSE={rmse:.4f}  nnz={nz}")
        if rmse < best_rmse:
            best_rmse, best_alpha = rmse, alpha

    # Refit on full training set with chosen alpha
    final = Lasso(alpha=best_alpha, fit_intercept=False, max_iter=10000, tol=1e-4)
    final.fit(Phi_tr_full, y_tr_full)
    return final.coef_, best_alpha


def main():
    d = 20
    a_g = np.array([0, 1, 4.5, 9, 99] + [99] * (d - 5), dtype=np.float32)
    rng = np.random.default_rng(42)
    N_train, N_val = 30_000, 30_000
    X_tr = rng.uniform(0, 1, size=(N_train, d)).astype(np.float32)
    y_tr = g_function(X_tr, a_g).astype(np.float32)
    X_va = rng.uniform(0, 1, size=(N_val, d)).astype(np.float32)
    y_va = g_function(X_va, a_g).astype(np.float32)
    y_mean = y_tr.mean()
    y_tr -= y_mean; y_va -= y_mean

    ana_main = main_effect_sobol(a_g)
    ana_pair = pair_sobol(a_g)

    results = []
    # Alpha grid scaled by basis size: smaller alpha for larger basis
    alpha_grids = {
        2: [1e-2, 3e-3, 1e-3, 3e-4, 1e-4],
        3: [3e-3, 1e-3, 3e-4, 1e-4, 3e-5],
        4: [1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
    }

    for P in [2, 3, 4]:
        multi_idx = multi_indices(d, P)
        K = len(multi_idx)
        print(f"\n=== P={P}, basis size {K:,} ===")
        t0 = time.time()
        Phi_tr = design_matrix(X_tr, multi_idx, P)
        print(f"  built design matrix in {time.time() - t0:.1f}s "
              f"({Phi_tr.nbytes/1e9:.2f} GB)")

        t1 = time.time()
        coeffs, alpha = fit_lasso_with_alpha_search(Phi_tr, y_tr, alpha_grids[P])
        fit_time = time.time() - t1
        n_active = int((np.abs(coeffs) > 1e-8).sum())
        print(f"  selected alpha={alpha:.2e}  active basis {n_active}/{K}  "
              f"({fit_time:.1f}s)")
        del Phi_tr

        Phi_va = design_matrix(X_va, multi_idx, P)
        y_pred = Phi_va @ coeffs
        del Phi_va
        rmse = float(np.sqrt(np.mean((y_pred - y_va) ** 2)))
        rel = rmse / float(y_va.std())

        pred_main, pred_pair = sobol_from_pce(coeffs, multi_idx, d)
        main_err = float(np.abs(pred_main - ana_main).sum())
        pair_err = sum(abs(pred_pair.get(ij, 0.0) - v) for ij, v in ana_pair.items())

        print(f"  rel_rmse={rel:.4f}  main_abs_err={main_err:.4f}  "
              f"pair_abs_err={pair_err:.4f}")

        results.append({
            "P": P,
            "basis_size": K,
            "active_basis": n_active,
            "alpha": float(alpha),
            "rel_rmse": rel,
            "main_abs_err": main_err,
            "pair_abs_err": float(pair_err),
            "fit_time_s": fit_time,
            "predicted_main": pred_main.tolist(),
        })

    out_path = Path(__file__).resolve().parent / "results_experiments" / "sparse_pce_d20.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")

    # Comparison table
    print("\n=== summary ===")
    print(f"{'P':>2}  {'basis':>6}  {'active':>7}  {'rel_rmse':>9}  "
          f"{'main_err':>9}  {'pair_err':>9}")
    for r in results:
        print(f"{r['P']:>2}  {r['basis_size']:>6}  {r['active_basis']:>7}  "
              f"{r['rel_rmse']:>9.4f}  {r['main_abs_err']:>9.4f}  "
              f"{r['pair_abs_err']:>9.4f}")


if __name__ == "__main__":
    main()
