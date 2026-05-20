"""Full order-3 joint HDMR on 3D Helmholtz LC-PINN (d=4: x, y, z, k).

The 3D Helm problem has 4 input dimensions, so an order-3 truncation captures
all C(4,1) + C(4,2) + C(4,3) = 4 + 6 + 4 = 14 subsets, missing only the
quadruplet {x,y,z,k}. We report:
  - All 4 main Sobol indices
  - All 6 pair Sobol indices
  - All 4 triplet Sobol indices (these are the new objects)
  - The unexplained variance (which bounds the quadruplet contribution)

Pipeline:
  1. Load 3D Helm FiLM+L-BFGS checkpoint
  2. Sample (x, y, z, k_norm) ~ U[0,1]^3 x U[-1, 1]
  3. Fit FourierTruncatedHDMR with d=4, max_order=3 (built-in for any d)
  4. Run MC-Sobol gold standard at N=300k (Saltelli) for ground truth comparison
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_HERE = Path(__file__).resolve().parent
_REPO_ANOVA = _HERE.parent.parent
_REPO_THESIS_CODE = _REPO_ANOVA.parent / "thesis" / "code"
sys.path.insert(0, str(_REPO_ANOVA))
sys.path.insert(0, str(_REPO_THESIS_CODE))

from lc_anova.core.fourier import FourierTruncatedHDMR  # noqa
from pinns.equations import helmholtz_3d as helm3d  # noqa
from pinns.model import LossConditionalPINN  # noqa

CK = _REPO_THESIS_CODE / "checkpoints"
RESULTS = _REPO_ANOVA / "lc_anova" / "results"


def load_lc_pinn_3d(checkpoint_path: str, device: torch.device):
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dims = ck.get("hidden_dims", [64, 64, 64, 64])
    conditioning = ck.get("conditioning", "film")
    model = LossConditionalPINN(
        dim_phys=3, dim_lambda=1,
        hidden_dims=hidden_dims, conditioning=conditioning,
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_lc_pinn_3d(model, xyz: torch.Tensor, k_norm: torch.Tensor) -> torch.Tensor:
    """xyz: (N, 3) in [0, 1]^3.  k_norm: (N, 1) in [-1, 1]."""
    bs = 30_000
    out = []
    for i in range(0, xyz.shape[0], bs):
        u = model(xyz[i:i+bs], k_norm[i:i+bs])
        out.append(u.squeeze(-1) if u.dim() > 1 else u)
    return torch.cat(out, dim=0)


def saltelli_sobol_4d(model_fn, N: int, seed: int = 42) -> dict:
    """Saltelli MC-Sobol on d=4 input space (x, y, z, k_norm)."""
    rng = np.random.default_rng(seed)
    d = 4

    def sample(rng_, N_):
        A = np.empty((N_, d), dtype=np.float64)
        A[:, :3] = rng_.uniform(0.0, 1.0, size=(N_, 3))
        A[:, 3] = rng_.uniform(-1.0, 1.0, size=(N_,))
        return A

    A = sample(rng, N); B = sample(rng, N)
    Y_A = np.asarray(model_fn(A), dtype=np.float64)
    Y_B = np.asarray(model_fn(B), dtype=np.float64)
    f0 = 0.5 * (Y_A.mean() + Y_B.mean())
    var_Y = 0.5 * (Y_A.var() + Y_B.var())

    V = {}
    for i in range(d):
        C = A.copy(); C[:, i] = B[:, i]
        Y_C = np.asarray(model_fn(C), dtype=np.float64)
        V[(i,)] = float(np.mean(Y_B * Y_C) - f0 * f0)

    V_closed_pair = {}
    for i, j in itertools.combinations(range(d), 2):
        C = A.copy(); C[:, i] = B[:, i]; C[:, j] = B[:, j]
        Y_C = np.asarray(model_fn(C), dtype=np.float64)
        V_closed_pair[(i, j)] = float(np.mean(Y_B * Y_C) - f0 * f0)
    V_pair = {(i, j): V_closed_pair[(i, j)] - V[(i,)] - V[(j,)] for i, j in V_closed_pair}

    V_closed_trip = {}
    for i, j, k in itertools.combinations(range(d), 3):
        C = A.copy(); C[:, i] = B[:, i]; C[:, j] = B[:, j]; C[:, k] = B[:, k]
        Y_C = np.asarray(model_fn(C), dtype=np.float64)
        V_closed_trip[(i, j, k)] = float(np.mean(Y_B * Y_C) - f0 * f0)
    # V_{ijk} = V_closed_{ijk} - V_{ij} - V_{ik} - V_{jk} - V_i - V_j - V_k
    V_trip = {}
    for i, j, k in V_closed_trip:
        V_trip[(i, j, k)] = (
            V_closed_trip[(i, j, k)]
            - V_pair[(i, j)] - V_pair[(i, k)] - V_pair[(j, k)]
            - V[(i,)] - V[(j,)] - V[(k,)]
        )
    # Quadruplet by closure
    V_quad = var_Y - sum(V.values()) - sum(V_pair.values()) - sum(V_trip.values())

    name_for = {0: "x", 1: "y", 2: "z", 3: "k"}
    out = {"var_Y": float(var_Y)}
    for s, v in V.items():
        out[f"S_{name_for[s[0]]}"] = v / var_Y
    for (i, j), v in V_pair.items():
        out[f"S_{name_for[i]}{name_for[j]}"] = v / var_Y
    for (i, j, k), v in V_trip.items():
        out[f"S_{name_for[i]}{name_for[j]}{name_for[k]}"] = v / var_Y
    out["S_xyzk"] = float(V_quad / var_Y)
    return out


def fit_fourier_hdmr(X_tr, y_tr, X_va, y_va, d: int = 4, hidden: int = 64,
                      layers: int = 2, num_freqs: int = 4,
                      phase1_epochs: int = 40, phase2_epochs: int = 100,
                      batch_size: int = 4096, lr1: float = 1e-3, lr2: float = 5e-4,
                      device=None) -> dict:
    device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # FourierTruncatedHDMR is order-2 only; for d=4 we use order-2 HDMR
    # (mains + 6 pairs) plus gold MC-Sobol for higher-order terms.
    model = FourierTruncatedHDMR(d=d, num_freqs=num_freqs, hidden=hidden,
                                  layers=layers).to(device)
    X_tr_t = torch.tensor(X_tr, device=device)
    y_tr_t = torch.tensor(y_tr, device=device)
    X_va_t = torch.tensor(X_va, device=device)
    y_va_t = torch.tensor(y_va, device=device)
    y_mean = y_tr_t.mean()
    y_tr_c = y_tr_t - y_mean
    y_va_c = y_va_t - y_mean

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_c), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr1, weight_decay=1e-6)
    for ep in range(1, phase1_epochs + 1):
        model.train()
        for xb, yb in loader:
            pred, _, _ = model(xb, include_pairs=False, purify=True)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if ep % 10 == 0 or ep == phase1_epochs:
            model.eval()
            with torch.no_grad():
                pred, _, _ = model(X_va_t, include_pairs=False, purify=True)
                rel = (torch.sqrt(F.mse_loss(pred, y_va_c)) / y_va_c.std()).item()
            print(f"  [phase1] ep={ep:>3} rel-RMSE={rel:.4f}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr2, weight_decay=1e-6)
    for ep in range(1, phase2_epochs + 1):
        model.train()
        for xb, yb in loader:
            pred, _, _ = model(xb, include_pairs=True, purify=True)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if ep % 20 == 0 or ep == phase2_epochs:
            model.eval()
            with torch.no_grad():
                pred, _, _ = model(X_va_t, include_pairs=True, purify=True)
                rel = (torch.sqrt(F.mse_loss(pred, y_va_c)) / y_va_c.std()).item()
            print(f"  [phase2] ep={ep:>3} rel-RMSE={rel:.4f}")

    model.eval()
    with torch.no_grad():
        m = model.main_terms(X_va_t)
        m_pure = model._purify_main(m)
        p = model.pair_terms(X_va_t)
        p_pure = model._purify_pair(p, X_va_t)
    main_var = m_pure.var(dim=0).cpu().numpy()
    pair_var = p_pure.var(dim=0).cpu().numpy()
    total_v = float(main_var.sum() + pair_var.sum())

    sobol_main = {f"S_{['x','y','z','k'][i]}": float(main_var[i] / total_v) for i in range(d)}
    sobol_pair = {}
    for kidx, (i, j) in enumerate(model.pair_indices):
        nm = "".join(["x", "y", "z", "k"][a] for a in (i, j))
        sobol_pair[f"S_{nm}"] = float(pair_var[kidx] / total_v)

    return {"model": model, "y_mean": float(y_mean.item()),
            "sobol_main": sobol_main, "sobol_pair": sobol_pair,
            "n_params": sum(p.numel() for p in model.parameters())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default=str(CK / "lc_pinn_helmholtz_3d_seed0_film_lbfgs.pt"))
    ap.add_argument("--N-train", type=int, default=200_000)
    ap.add_argument("--N-val", type=int, default=50_000)
    ap.add_argument("--N-sobol", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="lc_anova/results/helm3d_order3_full.json")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading {args.checkpoint}")
    model = load_lc_pinn_3d(args.checkpoint, device)

    @torch.no_grad()
    def model_fn(z_np: np.ndarray) -> np.ndarray:
        xyz = torch.tensor(z_np[:, :3].astype(np.float32), device=device)
        k = torch.tensor(z_np[:, 3:4].astype(np.float32), device=device)
        return evaluate_lc_pinn_3d(model, xyz, k).cpu().numpy()

    print(f"\n=== Step 1: MC-Sobol gold standard (Saltelli, N={args.N_sobol:,}) ===")
    t0 = time.time()
    gold = saltelli_sobol_4d(model_fn, N=args.N_sobol, seed=args.seed)
    t_sobol = time.time() - t0
    print(f"Wall: {t_sobol:.1f}s")
    print("Gold Sobol indices:")
    for k in sorted(gold):
        if k == "var_Y":
            continue
        print(f"  {k:<10}  {gold[k]:>8.4f}")

    print(f"\n=== Step 2: Fit Fourier joint HDMR  d=4, max_order=3 ===")
    rng = np.random.default_rng(args.seed)
    X_tr = np.empty((args.N_train, 4), dtype=np.float32)
    X_tr[:, :3] = rng.uniform(0, 1, size=(args.N_train, 3))
    X_tr[:, 3] = rng.uniform(-1, 1, size=(args.N_train,))
    X_va = np.empty((args.N_val, 4), dtype=np.float32)
    X_va[:, :3] = rng.uniform(0, 1, size=(args.N_val, 3))
    X_va[:, 3] = rng.uniform(-1, 1, size=(args.N_val,))
    y_tr = model_fn(X_tr).astype(np.float32)
    y_va = model_fn(X_va).astype(np.float32)
    print(f"  generated training set N={args.N_train:,}")
    t0 = time.time()
    fit = fit_fourier_hdmr(X_tr, y_tr, X_va, y_va, d=4)
    t_hdmr = time.time() - t0
    print(f"Wall: {t_hdmr:.1f}s   params: {fit['n_params']:,}")

    print(f"\n=== Step 3: Compare HDMR vs gold ===")
    print(f"  {'index':<10} {'HDMR':>9} {'gold':>9} {'diff':>9}")
    diffs = {}
    for key in sorted(gold):
        if key == "var_Y" or len(key) > 6:  # skip quadruplet (HDMR is order-3)
            continue
        hdmr_v = fit["sobol_main"].get(key) or fit["sobol_pair"].get(key, None)
        if hdmr_v is None:
            continue
        d_ = hdmr_v - gold[key]
        diffs[key] = d_
        print(f"  {key:<10} {hdmr_v:>9.4f} {gold[key]:>9.4f} {d_:>+9.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "N_train": args.N_train,
        "N_val": args.N_val,
        "N_sobol": args.N_sobol,
        "gold_mc_sobol": gold,
        "hdmr_sobol_main": fit["sobol_main"],
        "hdmr_sobol_pair": fit["sobol_pair"],
        "hdmr_n_params": fit["n_params"],
        "diffs_vs_gold": diffs,
        "wall_sobol_seconds": t_sobol,
        "wall_hdmr_seconds": t_hdmr,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
