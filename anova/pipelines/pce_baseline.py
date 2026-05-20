"""PCE (Polynomial Chaos Expansion) baseline on 2D Helmholtz LC-PINN output.

Comparison against our Fourier-feature joint HDMR and MC-Sobol gold standard.
PCE is the classical sensitivity-analysis baseline; reviewers from the SA
community expect this comparison.

Method:
  1. Sample N joint points (x, y, k_norm) ~ U[0,1]^2 x U[-1,1].
  2. Map (x, y) -> [-1, 1] for Legendre basis (k_norm already in [-1, 1]).
  3. Build multivariate normalised-Legendre design matrix up to total degree P.
  4. Fit coefficients via OLS (with regularisation if conditioning is bad)
     and LASSO (sparse PCE) for sparsity.
  5. Compute Sobol indices from coefficients: V_S = sum_alpha:supp(alpha)=S c_alpha^2.
  6. Compare against MC-Sobol gold standard (N = 1e6 from results_mc_megaN_seed0.json).

Expected: PCE order P captures the function only up to polynomial frequency ~P.
For cos(kx) at k=10 on the unit-length spatial domain, the effective wavenumber
in [-1, 1] is 10 (after the change of variable z = 2x - 1, frequency in z is
10/2 = 5). So we need P >= ~10 to start capturing the cos(kx) component, and
the triplet S_{x, y, k} needs simultaneous high orders in three axes. The
basis explodes: P=10 in d=3 has C(13, 3) = 286 coefficients.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from numpy.polynomial.legendre import Legendre

_HERE = Path(__file__).resolve().parent
_REPO_ANOVA = _HERE.parent.parent
_REPO_THESIS_CODE = _REPO_ANOVA.parent / "thesis" / "code"
sys.path.insert(0, str(_REPO_ANOVA))
sys.path.insert(0, str(_REPO_THESIS_CODE))

from lc_anova.pipelines.helmholtz_2d import load_lc_pinn, evaluate_lc_pinn_batch  # noqa
from pinns.equations import helmholtz_2d as helm  # noqa

CK_DIR = _REPO_THESIS_CODE / "checkpoints"
RESULTS = _REPO_ANOVA / "lc_anova" / "results"


def legendre_normalised(n: int, z: np.ndarray) -> np.ndarray:
    """Normalised Legendre P_n on [-1, 1] (sqrt(2n+1) factor)."""
    coeffs = np.zeros(n + 1)
    coeffs[n] = 1.0
    return np.sqrt(2 * n + 1) * Legendre(coeffs)(z)


def multi_indices(d: int, P: int) -> list[tuple[int, ...]]:
    """Multi-indices alpha with |alpha| <= P."""
    out = [(0,) * d]
    for total in range(1, P + 1):
        for positions in itertools.combinations_with_replacement(range(d), total):
            a = [0] * d
            for p in positions:
                a[p] += 1
            out.append(tuple(a))
    return out


def build_design_matrix(Z: np.ndarray, multi_idx: list[tuple[int, ...]], P: int) -> np.ndarray:
    """Z: (N, d) in [-1, 1]^d. Returns (N, len(multi_idx)) normalised-Legendre design."""
    N, d = Z.shape
    K = len(multi_idx)
    # cache 1D Legendre values
    L = np.zeros((P + 1, d, N))
    for j in range(d):
        for n in range(P + 1):
            L[n, j, :] = legendre_normalised(n, Z[:, j])
    Phi = np.ones((N, K))
    for k, alpha in enumerate(multi_idx):
        col = np.ones(N)
        for j in range(d):
            if alpha[j] > 0:
                col = col * L[alpha[j], j, :]
        Phi[:, k] = col
    return Phi


def sobol_from_pce(coeffs: np.ndarray, multi_idx: list[tuple[int, ...]], d: int = 3) -> dict:
    """Sobol indices from PCE coefficients. d=3 specialisation."""
    names = {(0,): "x", (1,): "y", (2,): "k",
             (0, 1): "x,y", (0, 2): "x,k", (1, 2): "y,k",
             (0, 1, 2): "x,y,k"}
    subset_var = {s: 0.0 for s in names.keys()}
    total_var = 0.0
    for c, alpha in zip(coeffs, multi_idx):
        if all(a == 0 for a in alpha):
            continue
        support = tuple(j for j in range(d) if alpha[j] > 0)
        total_var += c * c
        if support in subset_var:
            subset_var[support] += c * c
    if total_var <= 0:
        return {names[s]: 0.0 for s in names}
    return {names[s]: subset_var[s] / total_var for s in names}


def fit_pce_ols(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    coeffs, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    return coeffs


def fit_pce_lasso(Phi: np.ndarray, y: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    try:
        from sklearn.linear_model import Lasso
    except Exception as e:
        print(f"  sklearn unavailable ({e}); skipping LASSO")
        return None
    m = Lasso(alpha=alpha, fit_intercept=False, max_iter=20000, tol=1e-4)
    m.fit(Phi, y)
    return m.coef_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default=str(CK_DIR / "lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt"))
    ap.add_argument("--N-train", type=int, default=30000)
    ap.add_argument("--N-val", type=int, default=30000)
    ap.add_argument("--degrees", nargs="+", type=int, default=[2, 4, 6, 8, 10])
    ap.add_argument("--lasso-alphas", nargs="+", type=float,
                    default=[1e-2, 3e-3, 1e-3, 3e-4, 1e-4])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="lc_anova/results/pce_baseline_helm2d.json")
    args = ap.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Loading {args.checkpoint}")
    model, meta = load_lc_pinn(args.checkpoint, device)

    rng = np.random.default_rng(args.seed)

    # Sample training and validation points
    print(f"Sampling N_train={args.N_train}, N_val={args.N_val}")
    xy_tr = rng.uniform(0, 1, size=(args.N_train, 2)).astype(np.float32)
    k_tr  = rng.uniform(-1, 1, size=(args.N_train, 1)).astype(np.float32)
    xy_va = rng.uniform(0, 1, size=(args.N_val, 2)).astype(np.float32)
    k_va  = rng.uniform(-1, 1, size=(args.N_val, 1)).astype(np.float32)

    @torch.no_grad()
    def f(xy, k):
        xy_t = torch.tensor(xy, device=device)
        k_t = torch.tensor(k, device=device)
        return evaluate_lc_pinn_batch(model, xy_t, k_t).cpu().numpy()

    print("Evaluating LC-PINN on samples")
    y_tr = f(xy_tr, k_tr).astype(np.float64)
    y_va = f(xy_va, k_va).astype(np.float64)
    y_tr_centered = y_tr - y_tr.mean()
    y_va_centered = y_va - y_tr.mean()  # use training-set mean
    var_y_val = float(y_va_centered.var())
    print(f"  Var(Y val) = {var_y_val:.4f}")

    # Map (x, y) from [0, 1] to [-1, 1] for Legendre basis
    Z_tr = np.column_stack([2.0 * xy_tr[:, 0] - 1.0,
                              2.0 * xy_tr[:, 1] - 1.0,
                              k_tr.squeeze(-1)]).astype(np.float64)
    Z_va = np.column_stack([2.0 * xy_va[:, 0] - 1.0,
                              2.0 * xy_va[:, 1] - 1.0,
                              k_va.squeeze(-1)]).astype(np.float64)

    # MC-Sobol gold standard for reference
    mc_path = RESULTS / "results_mc_megaN_seed0.json"
    mc = json.loads(mc_path.read_text()) if mc_path.exists() else None
    if mc:
        gold = {
            "x": mc["S_first"]["x"], "y": mc["S_first"]["y"], "k": mc["S_first"]["k"],
            "x,y": mc["S_pair"]["x,y"], "x,k": mc["S_pair"]["x,k"], "y,k": mc["S_pair"]["y,k"],
            "x,y,k": mc["S_triplet"],
        }
        print(f"\nMC-Sobol gold standard (N=1e6): {gold}")

    # --- OLS PCE at each total degree ---
    results = []
    for P in args.degrees:
        midx = multi_indices(3, P)
        K = len(midx)
        print(f"\n=== PCE order P={P}  (basis size {K}) ===")
        t0 = time.perf_counter()
        Phi_tr = build_design_matrix(Z_tr, midx, P)
        t_design = time.perf_counter() - t0
        print(f"  built design matrix {Phi_tr.shape} in {t_design:.2f}s")

        # OLS
        t0 = time.perf_counter()
        coeffs = fit_pce_ols(Phi_tr, y_tr_centered)
        t_fit = time.perf_counter() - t0
        Phi_va = build_design_matrix(Z_va, midx, P)
        y_pred = Phi_va @ coeffs
        rmse = float(np.sqrt(np.mean((y_pred - y_va_centered) ** 2)))
        rel = rmse / float(np.std(y_va_centered))
        sobol = sobol_from_pce(coeffs, midx)
        # captured variance fraction
        captured = 1.0 - rel * rel
        print(f"  OLS  rel-RMSE={rel:.4f}  captured={captured*100:.1f}%  fit={t_fit:.1f}s")
        print(f"       Sobol: {sobol}")
        ols_row = {
            "method": "ols", "P": P, "n_basis": K,
            "rel_rmse": rel, "captured": captured, "fit_seconds": t_fit,
            "sobol": sobol,
        }
        if mc:
            ols_row["err_vs_gold"] = {key: sobol[key] - gold[key] for key in gold}
            ols_row["max_abs_err"] = max(abs(v) for v in ols_row["err_vs_gold"].values())
        results.append(ols_row)

        # LASSO sweep (only at moderate P to avoid blowup)
        if P <= 8:
            best_lasso = None
            for alpha in args.lasso_alphas:
                coef = fit_pce_lasso(Phi_tr, y_tr_centered, alpha=alpha)
                if coef is None:
                    break
                y_pred = Phi_va @ coef
                rmse = float(np.sqrt(np.mean((y_pred - y_va_centered) ** 2)))
                rel = rmse / float(np.std(y_va_centered))
                nnz = int((np.abs(coef) > 1e-8).sum())
                sobol_l = sobol_from_pce(coef, midx)
                captured = 1.0 - rel * rel
                if best_lasso is None or rel < best_lasso["rel_rmse"]:
                    best_lasso = {
                        "method": "lasso", "P": P, "alpha": alpha, "n_basis": K,
                        "n_active": nnz, "rel_rmse": rel, "captured": captured,
                        "sobol": sobol_l,
                    }
                    if mc:
                        best_lasso["err_vs_gold"] = {key: sobol_l[key] - gold[key] for key in gold}
                        best_lasso["max_abs_err"] = max(abs(v) for v in best_lasso["err_vs_gold"].values())
            if best_lasso is not None:
                print(f"  LASSO best alpha={best_lasso['alpha']:.0e}  "
                      f"rel-RMSE={best_lasso['rel_rmse']:.4f}  captured={best_lasso['captured']*100:.1f}%  "
                      f"active={best_lasso['n_active']}/{best_lasso['n_basis']}")
                results.append(best_lasso)

    # Save and pretty-print
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "N_train": args.N_train, "N_val": args.N_val,
        "mc_gold_standard": gold if mc else None,
        "var_y_val": var_y_val,
        "results": results,
    }, indent=2))
    print(f"\nWrote {out_path}")

    # Summary table
    print("\n=== SUMMARY ===")
    print(f"  {'method':<6} {'P':>3} {'basis':>5} {'active':>6} "
          f"{'captured%':>9} {'S_xyk':>7} {'|Δ vs gold|':>11}")
    for r in results:
        active = r.get("n_active", r["n_basis"])
        sxyk = r["sobol"]["x,y,k"]
        err = r.get("max_abs_err", float("nan"))
        print(f"  {r['method']:<6} {r['P']:>3} {r['n_basis']:>5} {active:>6} "
              f"{r['captured']*100:>9.1f} {sxyk:>7.3f} {err:>11.4f}")


if __name__ == "__main__":
    main()
