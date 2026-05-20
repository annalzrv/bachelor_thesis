"""LC-conditioned 1D Dirichlet Laplacian eigenvalue PINN.

One network `u_θ(x, k)` conditioned on a (normalised) mode index k ∈ {1..K}
replaces the K independent networks from the sequential-deflation baseline
(`laplace_1d.py`). Trained jointly via the Ky Fan variational principle:

    L = Σ_k w_k · R(u_θ(·, k))  +  α · Σ_{k≠j} cos²(∠(u_θ(·, k), u_θ(·, j)))

where R is the Rayleigh quotient and the weights `w_k = 1/k` break the
permutation symmetry of the Σ R term, so slot k maps to the k-th lowest
eigenvalue in order.

Design choices:
  • Hard Dirichlet BC `u = x(π-x) · N_θ(x, k_norm)`. No BC loss term.
  • k encoded as normalised scalar `(k-1)/(K-1) ∈ [0, 1]` — standard
    LC-PINN input convention.
  • cos² orthogonality (not ⟨u, u_j⟩²) — Rayleigh is scale-invariant so
    only direction-based penalties survive, same fix as Apr 21.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from pinns.equations.laplace_1d import (
    DOMAIN_LEN, X_MAX, X_MIN,
    B, reference_eigenmode,
)
from pinns.model import LossConditionalPINN


DIM_PHYS = 1
DIM_OUT = 1


def lambda_dim_for(K_max: int, encoding: str = "onehot") -> int:
    """Size of the k-conditioning vector for a given encoding."""
    if encoding == "scalar":
        return 1
    if encoding == "onehot":
        return K_max
    raise ValueError(f"unknown encoding {encoding!r}")


# ---------------------------------------------------------------------------
# k encoding
# ---------------------------------------------------------------------------

def normalise_k(
    k: int, K_max: int, device: torch.device, encoding: str = "onehot",
) -> torch.Tensor:
    """Encode integer mode k ∈ {1..K_max} as a conditioning tensor.

    `encoding="onehot"` (default) — length-K_max one-hot vector. Exact
    separation between modes, no tanh-saturation compression, lets each
    slot claim its own affine projection of the first LC-PINN layer.
    Previous experiments with `encoding="scalar"` (raw integer k as a
    single input) showed eigenvalue-accurate but shape-contaminated
    outputs for k ≥ 2: the network could not place neighbouring slots
    far enough apart in feature space to suppress mode mixing.

    `encoding="scalar"` — single real k. Kept for ablation.
    """
    if encoding == "scalar":
        return torch.tensor([float(k)], dtype=torch.float32, device=device)
    if encoding == "onehot":
        v = torch.zeros(K_max, dtype=torch.float32, device=device)
        v[k - 1] = 1.0
        return v
    raise ValueError(f"unknown encoding {encoding!r}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LCEigenmodeNet(nn.Module):
    """u_θ(x, k_enc) = B(x) · N_θ(x, k_enc), one network for all modes."""

    def __init__(
        self, K_max: int,
        hidden_dims: list[int] | None = None,
        encoding: str = "onehot",
    ):
        super().__init__()
        self.K_max = K_max
        self.encoding = encoding
        if hidden_dims is None:
            hidden_dims = [64, 64, 64, 64]
        self.core = LossConditionalPINN(
            dim_phys=DIM_PHYS,
            dim_lambda=lambda_dim_for(K_max, encoding=encoding),
            hidden_dims=hidden_dims, dim_out=DIM_OUT,
        )

    def forward(self, x: torch.Tensor, k_enc: torch.Tensor) -> torch.Tensor:
        return B(x) * self.core(x, k_enc)

    def encode_k(self, k: int, device: torch.device) -> torch.Tensor:
        return normalise_k(k, self.K_max, device, encoding=self.encoding)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def rayleigh_quotient(
    model: nn.Module, x: torch.Tensor, k_norm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u = model(x, k_norm)
    du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    num = (du_dx ** 2).mean()
    den = (u ** 2).mean()
    return num, den, num / den.clamp(min=1e-12)


def _all_modes_outputs(
    model: "LCEigenmodeNet", x: torch.Tensor, K: int, device: torch.device,
) -> list[torch.Tensor]:
    return [model(x, model.encode_k(k, device)) for k in range(1, K + 1)]


def cos2_orthogonality(us: list[torch.Tensor]) -> torch.Tensor:
    """Σ_{i<j} cos²(u_i, u_j) — scale-invariant pairwise penalty."""
    norms = [torch.sqrt((u ** 2).mean().clamp(min=1e-12)) for u in us]
    pen = torch.zeros((), device=us[0].device)
    for i in range(len(us)):
        for j in range(i + 1, len(us)):
            cos_sq = ((us[i] * us[j]).mean() / (norms[i] * norms[j])) ** 2
            pen = pen + cos_sq
    return pen


def compute_losses(
    model: nn.Module,
    x: torch.Tensor,
    K: int,
    device: torch.device,
    w_exp: float = 1.0,
    K_active: int | None = None,
    K_max_encoding: int | None = None,
) -> dict[str, torch.Tensor]:
    """Ky Fan objective. `w_exp` controls weight decay: w_k = 1/k^w_exp.

    `K_active` (curriculum): if set, only slots 1..K_active contribute to the
    loss; other slots are ignored. Encoding uses `K_max_encoding` (default K)
    so the k-normalisation stays consistent as K_active grows.
    """
    if K_active is None:
        K_active = K
    if K_max_encoding is None:
        K_max_encoding = K

    rayleighs: list[torch.Tensor] = []
    us: list[torch.Tensor] = []
    for k in range(1, K_active + 1):
        k_enc = model.encode_k(k, device)
        u = model(x, k_enc)
        us.append(u)
        du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        num = (du_dx ** 2).mean()
        den = (u ** 2).mean().clamp(min=1e-12)
        rayleighs.append(num / den)

    w = torch.tensor(
        [1.0 / (k ** w_exp) for k in range(1, K_active + 1)],
        dtype=torch.float32, device=device,
    )
    weighted_sum = sum(wk * rk for wk, rk in zip(w, rayleighs))
    orth = (
        cos2_orthogonality(us) if len(us) > 1
        else torch.zeros((), device=device)
    )

    # Pad per-slot Rayleigh history out to K (NaN for inactive slots)
    pad = [float("nan")] * (K - K_active)
    all_rays = [r.detach() for r in rayleighs] + [
        torch.tensor(float("nan"), device=device) for _ in pad
    ]
    return {
        "rayleigh_sum": weighted_sum,
        "orth": orth,
        "_rayleighs": torch.stack(all_rays),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lc_eigenmode(
    model: nn.Module,
    K: int,
    device: torch.device,
    n_epochs: int = 20_000,
    lr: float = 1e-3,
    alpha_orth: float = 100.0,
    n_interior: int = 1024,
    w_exp: float = 1.0,
    log_every: int = 500,
    curriculum: bool = False,
) -> dict[str, list]:
    """Train LC eigenmode network.

    If `curriculum=True`, training is split into K equal phases:
    phase j uses K_active = j (only slots 1..j contribute to the loss). This
    mirrors sequential deflation but with shared network weights: mode 1
    locks in during phase 1, then mode 2 is added while mode 1 is
    maintained (via its Rayleigh term + pairwise cos²), and so on.
    """
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list] = {
        "step": [], "rayleigh_sum": [], "orth": [],
        "total": [], "lambda_hats": [], "K_active": [],
    }

    def _phase_for_step(step: int) -> int:
        if not curriculum:
            return K
        # Split n_epochs into K equal phases; phase j covers slots 1..j.
        phase_len = max(n_epochs // K, 1)
        return min(K, step // phase_len + 1)

    for step in range(n_epochs):
        optimiser.zero_grad()
        x = torch.rand(n_interior, 1, device=device) * DOMAIN_LEN + X_MIN
        x.requires_grad_(True)
        K_active = _phase_for_step(step)
        losses = compute_losses(
            model, x, K=K, device=device, w_exp=w_exp,
            K_active=K_active, K_max_encoding=K,
        )
        total = losses["rayleigh_sum"] + alpha_orth * losses["orth"]
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        if step % log_every == 0:
            history["step"].append(step)
            history["rayleigh_sum"].append(float(losses["rayleigh_sum"].item()))
            history["orth"].append(float(losses["orth"].item()))
            history["total"].append(float(total.item()))
            history["lambda_hats"].append(losses["_rayleighs"].cpu().numpy().tolist())
            history["K_active"].append(K_active)

    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_eigenmode(
    model: "LCEigenmodeNet", k: int, K_max: int,
    x_pts: np.ndarray, device: torch.device,
) -> np.ndarray:
    del K_max  # read from model
    k_enc = model.encode_k(k, device)
    x = torch.tensor(x_pts, dtype=torch.float32, device=device).unsqueeze(1)
    return model(x, k_enc).cpu().numpy().flatten()


def _l2_normalise(u: np.ndarray) -> np.ndarray:
    x = np.linspace(X_MIN, X_MAX, len(u))
    integral = np.trapz(u ** 2, x)
    return u / (np.sqrt(integral) + 1e-12)


def evaluate_eigenmode(
    model: nn.Module, k: int, K_max: int,
    device: torch.device, nx: int = 500,
) -> dict[str, float]:
    x_pts = np.linspace(X_MIN, X_MAX, nx)
    u_ref, lambda_k = reference_eigenmode(k, x_pts)
    u_pred = predict_eigenmode(model, k, K_max, x_pts, device)

    u_pred_n = _l2_normalise(u_pred)
    if float(np.dot(u_pred_n, u_ref)) < 0.0:
        u_pred_n = -u_pred_n
    rel_l2 = float(np.linalg.norm(u_pred_n - u_ref) / (np.linalg.norm(u_ref) + 1e-10))

    k_enc = model.encode_k(k, device)
    x_t = torch.tensor(x_pts, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
    _, _, lam_hat = rayleigh_quotient(model, x_t, k_enc)
    lam_hat_val = float(lam_hat.item())

    return {
        "k": k,
        "rel_l2": rel_l2,
        "lambda_true": lambda_k,
        "lambda_hat": lam_hat_val,
        "lambda_rel_err": abs(lam_hat_val - lambda_k) / lambda_k,
    }


def evaluate_all(
    model: nn.Module, K: int, device: torch.device, nx: int = 500,
) -> list[dict[str, float]]:
    return [evaluate_eigenmode(model, k, K, device, nx=nx) for k in range(1, K + 1)]


def reorder_by_rayleigh(
    model: "LCEigenmodeNet", K: int, device: torch.device, nx: int = 500,
) -> list[int]:
    """Diagnostic: ranks slots by Rayleigh so we can check slot k → mode k."""
    x_pts = np.linspace(X_MIN, X_MAX, nx)
    x_t = torch.tensor(x_pts, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
    ray = []
    for k in range(1, K + 1):
        _, _, lam = rayleigh_quotient(model, x_t, model.encode_k(k, device))
        ray.append(float(lam.item()))
    order = sorted(range(K), key=lambda i: ray[i])
    return [o + 1 for o in order]
