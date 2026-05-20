"""Non-PINN demo: joint (x, lambda) HDMR on a conditional regression model
trained on tabular data. Tests the method's generality outside of PINNs.

Dataset: California Housing (8 features + 1 target via sklearn fetch).
Setup:
  - Standardize all features and target.
  - Pick MedInc (median income) as the conditioning axis lambda.
  - The other 7 features are x.
  - Train a 'conditional MLP' that predicts median house value from
    (x_7, lambda).
  - Apply our Fourier joint HDMR with dim_x=7, dim_lambda=1, max_order=2.

Expected: Sobol decomposition reveals which feature pairs interact most with
median-income. Validates the method works on real tabular data, not just on
PINN-derived parametric functions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_HERE = Path(__file__).resolve().parent
_REPO_ANOVA = _HERE.parent.parent
sys.path.insert(0, str(_REPO_ANOVA))

from lc_anova.core.joint_hdmr import JointHDMR  # noqa
from hdmr_net import select_device  # noqa


class ConditionalMLP(nn.Module):
    """Simple MLP regressor on concatenated (x, lambda)."""

    def __init__(self, dim_x: int, dim_lambda: int, hidden: int = 128, layers: int = 3):
        super().__init__()
        in_dim = dim_x + dim_lambda
        modules = []
        last = in_dim
        for _ in range(layers):
            modules += [nn.Linear(last, hidden), nn.SiLU()]
            last = hidden
        modules.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*modules)
        self.dim_x = dim_x
        self.dim_lambda = dim_lambda

    def forward(self, x, lam):
        z = torch.cat([x, lam], dim=1)
        return self.net(z).squeeze(-1)


def load_california_housing():
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)
    feature_names = list(data.feature_names)
    return X, y, feature_names


def standardize(X, y):
    """Standardize features and target to unit variance, zero mean, then
    map features to [0, 1] (preserving relative spacing) and target to centred.
    JointHDMR needs inputs in a bounded box for binning purification."""
    Xc = (X - X.mean(0)) / X.std(0)
    # map to roughly [0, 1] via empirical quantile (robust)
    lo, hi = np.percentile(Xc, [1, 99], axis=0)
    Xc = np.clip((Xc - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    yc = (y - y.mean()) / y.std()
    return Xc.astype(np.float32), yc.astype(np.float32)


def train_conditional_mlp(X_tr, y_tr, X_va, y_va, lambda_idx: int,
                            hidden: int = 128, layers: int = 3,
                            epochs: int = 100, lr: float = 1e-3,
                            batch_size: int = 512, device=None):
    device = device or select_device()
    n_feat = X_tr.shape[1]
    mask = np.array([i != lambda_idx for i in range(n_feat)])
    Xx_tr = X_tr[:, mask]; lam_tr = X_tr[:, [lambda_idx]]
    Xx_va = X_va[:, mask]; lam_va = X_va[:, [lambda_idx]]

    model = ConditionalMLP(Xx_tr.shape[1], 1, hidden=hidden, layers=layers).to(device)
    Xx_tr_t = torch.tensor(Xx_tr, device=device)
    lam_tr_t = torch.tensor(lam_tr, device=device)
    y_tr_t = torch.tensor(y_tr, device=device)
    Xx_va_t = torch.tensor(Xx_va, device=device)
    lam_va_t = torch.tensor(lam_va, device=device)
    y_va_t = torch.tensor(y_va, device=device)

    loader = DataLoader(TensorDataset(Xx_tr_t, lam_tr_t, y_tr_t),
                          batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    for ep in range(1, epochs + 1):
        model.train()
        for xb, lb, yb in loader:
            pred = model(xb, lb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                pred = model(Xx_va_t, lam_va_t)
                val_rel = (torch.sqrt(F.mse_loss(pred, y_va_t)) / y_va_t.std()).item()
            print(f"  conditional-MLP ep={ep:>3} val rel-RMSE = {val_rel:.4f}")

    return model, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-feature", default="MedInc",
                    help="Name of the feature to treat as the parameter axis.")
    ap.add_argument("--mlp-epochs", type=int, default=100)
    ap.add_argument("--hdmr-phase1", type=int, default=40)
    ap.add_argument("--hdmr-phase2", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--num-freqs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="lc_anova/results/conditional_mlp_california.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print("Loading California Housing")
    X, y, feature_names = load_california_housing()
    print(f"  N={X.shape[0]}  features: {feature_names}")
    lambda_idx = feature_names.index(args.lambda_feature)
    print(f"  treating '{args.lambda_feature}' (idx {lambda_idx}) as the parameter axis")

    Xc, yc = standardize(X, y)
    n = len(yc); perm = rng.permutation(n)
    n_tr = int(0.7 * n)
    tr_idx = perm[:n_tr]; va_idx = perm[n_tr:]
    X_tr = Xc[tr_idx]; y_tr = yc[tr_idx]
    X_va = Xc[va_idx]; y_va = yc[va_idx]

    print(f"\nTraining conditional MLP")
    model, mask = train_conditional_mlp(
        X_tr, y_tr, X_va, y_va, lambda_idx=lambda_idx,
        hidden=128, layers=3, epochs=args.mlp_epochs,
    )

    # Apply joint HDMR
    device = select_device()
    n_feat = X_tr.shape[1]
    Xx_va = X_va[:, mask]; lam_va = X_va[:, [lambda_idx]]
    x_t = torch.tensor(Xx_va, device=device)
    l_t = torch.tensor(lam_va, device=device)
    with torch.no_grad():
        u_va = model(x_t, l_t)

    Xx_tr = X_tr[:, mask]; lam_tr = X_tr[:, [lambda_idx]]
    xt = torch.tensor(Xx_tr, device=device)
    lt = torch.tensor(lam_tr, device=device)
    with torch.no_grad():
        u_tr = model(xt, lt)

    print(f"\nFitting Fourier joint HDMR  dim_x={Xx_tr.shape[1]}  dim_lambda=1  "
          f"hidden={args.hidden}  L={args.num_freqs}")
    jh = JointHDMR(dim_x=Xx_tr.shape[1], dim_lambda=1, hidden=args.hidden,
                   layers=args.layers, max_order=2, use_fourier=True,
                   num_freqs=args.num_freqs)
    jh.fit(xt, lt, u_tr, phase1_epochs=args.hdmr_phase1,
           phase2_epochs=args.hdmr_phase2, log_every=40)

    z_va = torch.cat([x_t, l_t], dim=1).to(device)
    y_va_c = u_va.to(device) - jh.y_mean
    jh.model.eval()
    with torch.no_grad():
        pred, _, _ = jh.model(z_va, include_pairs=True, purify=True)
        val_rel = (torch.sqrt(torch.mean((pred - y_va_c) ** 2)) / y_va_c.std()).item()
    print(f"\nJointHDMR val rel-RMSE on conditional MLP: {val_rel:.4f}")

    terms = jh.evaluate_terms(x_t, l_t)
    # Re-label subsets with feature names
    x_features = [feature_names[i] for i in range(n_feat) if i != lambda_idx]
    lambda_name = args.lambda_feature
    name_for = {i: x_features[i] for i in range(len(x_features))}
    name_for[len(x_features)] = lambda_name  # the parameter axis

    sobol_named = {}
    for key, val in sorted(terms["sobol"].items(), key=lambda kv: (-len(kv[0]), kv[0])):
        names = " × ".join(name_for[a] for a in key)
        sobol_named[names] = val

    print("\nSobol indices (top 10 by magnitude):")
    for name, val in sorted(sobol_named.items(), key=lambda kv: -kv[1])[:10]:
        kind = "main" if "×" not in name else "pair"
        print(f"  {name:<35}  {val:>9.4f}  ({kind})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dataset": "california_housing",
        "lambda_feature": args.lambda_feature,
        "x_features": x_features,
        "n_train": len(y_tr), "n_val": len(y_va),
        "mlp_val_rel_rmse_predicting_y": None,  # we didn't track this separately
        "jointhdmr_val_rel_rmse_on_mlp": val_rel,
        "sobol_indices": sobol_named,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
