"""Head-to-head: Fourier-feature joint HDMR vs tanh HDMR on Sobol-G d=20.

The ICDM Neural HDMR paper baseline (results_hdmr_sobolg_d20.json) achieved
rel-RMSE 0.065, main_abs_err 0.0037 with tanh hidden=32 layers=2 at
N_train=100k for d=20 with 5 active inputs.

We re-run the same Sobol-G benchmark with our Fourier-feature
TruncatedHDMR to see whether positional encoding also helps on the
non-smooth |4x - 2| structure (kink at x=0.5).
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
from torch.utils.data import DataLoader, TensorDataset

_HERE = Path(__file__).resolve().parent
_REPO_ANOVA = _HERE.parent.parent
sys.path.insert(0, str(_REPO_ANOVA))

from lc_anova.core.fourier import FourierTruncatedHDMR  # noqa
from hdmr_net import select_device  # noqa
from sobol_g import g_function, main_effect_sobol, pair_sobol  # noqa


def train_fourier_hdmr(
    X_tr, y_tr, X_va, y_va, d: int,
    hidden: int = 32, layers: int = 2, num_freqs: int = 4,
    phase1_epochs: int = 40, phase2_epochs: int = 80,
    batch_size: int = 4096, lr1: float = 1e-3, lr2: float = 5e-4,
    device=None,
) -> dict:
    device = device or select_device()
    model = FourierTruncatedHDMR(d=d, num_freqs=num_freqs, hidden=hidden, layers=layers).to(device)
    X_tr_t = torch.tensor(X_tr, device=device)
    y_tr_t = torch.tensor(y_tr, device=device)
    X_va_t = torch.tensor(X_va, device=device)
    y_va_t = torch.tensor(y_va, device=device)

    y_mean = y_tr_t.mean()
    y_tr_c = y_tr_t - y_mean
    y_va_c = y_va_t - y_mean

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_c), batch_size=batch_size, shuffle=True)
    history = []

    # Phase 1: mains only
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
            history.append({"phase": 1, "ep": ep, "rel": rel})
            print(f"  [phase1] ep={ep:>3} rel-RMSE={rel:.4f}")

    # Phase 2: mains + pairs
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
            history.append({"phase": 2, "ep": ep, "rel": rel})
            print(f"  [phase2] ep={ep:>3} rel-RMSE={rel:.4f}")

    model.eval()
    with torch.no_grad():
        # Re-evaluate decomposed terms on val set for Sobol estimation
        m = model.main_terms(X_va_t)
        m_pure = model._purify_main(m)
        p = model.pair_terms(X_va_t)
        p_pure = model._purify_pair(p, X_va_t)
    main_var = m_pure.var(dim=0).cpu().numpy()
    pair_var = p_pure.var(dim=0).cpu().numpy()
    total_v = float(main_var.sum() + pair_var.sum())
    pred_main = main_var / total_v
    pred_pair_named = {model.pair_indices[k]: pair_var[k] / total_v for k in range(len(model.pair_indices))}

    return {
        "model": model,
        "n_params": sum(p.numel() for p in model.parameters()),
        "final_rel_rmse": history[-1]["rel"],
        "predicted_main": pred_main.tolist(),
        "predicted_pair": {str(k): float(v) for k, v in pred_pair_named.items()},
        "history": history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=20)
    ap.add_argument("--N-train", type=int, default=100000)
    ap.add_argument("--N-val", type=int, default=50000)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--num-freqs", type=int, default=4)
    ap.add_argument("--phase1-epochs", type=int, default=40)
    ap.add_argument("--phase2-epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="lc_anova/results/sobol_g_fourier_d20.json")
    args = ap.parse_args()

    d = args.d
    a_g = np.array([0, 1, 4.5, 9] + [99] * (d - 4), dtype=np.float32)[:d]

    rng = np.random.default_rng(args.seed)
    X_tr = rng.uniform(0, 1, size=(args.N_train, d)).astype(np.float32)
    X_va = rng.uniform(0, 1, size=(args.N_val, d)).astype(np.float32)
    y_tr = g_function(X_tr, a_g).astype(np.float32)
    y_va = g_function(X_va, a_g).astype(np.float32)

    print(f"Sobol-G d={d} with {sum(a < 50 for a in a_g)} active inputs (a values < 50)")
    print(f"  N_train={args.N_train}  N_val={args.N_val}")
    print(f"  Fourier features: L={args.num_freqs}  hidden={args.hidden}  layers={args.layers}")

    t0 = time.perf_counter()
    out = train_fourier_hdmr(
        X_tr, y_tr, X_va, y_va, d=d,
        hidden=args.hidden, layers=args.layers, num_freqs=args.num_freqs,
        phase1_epochs=args.phase1_epochs, phase2_epochs=args.phase2_epochs,
    )
    wall = time.perf_counter() - t0
    print(f"\nWall: {wall:.1f}s  params: {out['n_params']:,}  rel-RMSE: {out['final_rel_rmse']:.4f}")

    # Analytic Sobol
    ana_main = main_effect_sobol(a_g)
    ana_pair = pair_sobol(a_g)
    main_err = float(np.abs(np.array(out["predicted_main"]) - ana_main).sum())
    pair_err_total = 0.0
    for (i, j), v_ana in ana_pair.items():
        v_pred = out["predicted_pair"].get(str((i, j)), 0.0)
        pair_err_total += abs(v_pred - v_ana)
    print(f"  main_abs_err (sum) = {main_err:.4f}")
    print(f"  pair_abs_err (sum) = {pair_err_total:.4f}")

    # Compare to tanh baseline
    tanh_path = _REPO_ANOVA / "results_hdmr_sobolg_d20.json"
    tanh = None
    if tanh_path.exists():
        tanh = json.loads(tanh_path.read_text())
        print(f"\nTanh baseline (anova/results_hdmr_sobolg_d20.json):")
        print(f"  rel-RMSE = {tanh['final_rel_rmse']:.4f}")
        print(f"  main_abs_err = {tanh['main_abs_err']:.4f}")
        print(f"  pair_abs_err = {tanh['pair_abs_err']:.4f}")
        print(f"\nFourier vs tanh:")
        print(f"  rel-RMSE:    {out['final_rel_rmse']:.4f}  vs  {tanh['final_rel_rmse']:.4f}  "
              f"({'better' if out['final_rel_rmse'] < tanh['final_rel_rmse'] else 'worse'})")
        print(f"  main_err:    {main_err:.4f}  vs  {tanh['main_abs_err']:.4f}")

    # Save (without model)
    out_dict = {k: v for k, v in out.items() if k != "model"}
    out_dict["a_g"] = a_g.tolist()
    out_dict["analytic_main"] = ana_main.tolist()
    out_dict["analytic_pair"] = {str(k): float(v) for k, v in ana_pair.items()}
    out_dict["main_abs_err"] = main_err
    out_dict["pair_abs_err"] = pair_err_total
    out_dict["wall_seconds"] = wall
    out_dict["config"] = {
        "d": d, "N_train": args.N_train, "N_val": args.N_val,
        "hidden": args.hidden, "layers": args.layers,
        "num_freqs": args.num_freqs,
        "phase1_epochs": args.phase1_epochs, "phase2_epochs": args.phase2_epochs,
    }
    if tanh is not None:
        out_dict["tanh_baseline"] = {
            "rel_rmse": tanh["final_rel_rmse"],
            "main_abs_err": tanh["main_abs_err"],
            "pair_abs_err": tanh["pair_abs_err"],
        }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_dict, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
