"""1D Dirichlet Laplacian eigenvalue problem — variational PINN.

-u''(x) = λ u(x),  x ∈ (0, π),  u(0) = u(π) = 0.

Analytic eigenpairs (L²-normalised on (0, π)):
    u_k(x) = sqrt(2/π) · sin(k·x),    λ_k = k²,    k = 1, 2, 3, …

Implementation follows the advisor's formulation:
  1. Hard Dirichlet BC via B(x) = x(π - x).
  2. Rayleigh-quotient objective (avoids second derivatives in the loss).
  3. Sequential training with orthogonality penalty against previously
     trained modes — robust to mode collapse.

Each mode has its own small network; modes are trained one after another.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Domain & analytic reference
# ---------------------------------------------------------------------------

X_MIN: float = 0.0
X_MAX: float = float(np.pi)
DOMAIN_LEN: float = X_MAX - X_MIN  # = π

DIM_PHYS = 1


def B(x: torch.Tensor) -> torch.Tensor:
    """Hard Dirichlet multiplier: B(x) = x · (π − x) vanishes on ∂Ω."""
    return x * (X_MAX - x)


def reference_eigenmode(k: int, x_pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Analytic eigenpair normalised so ∫_0^π u_k² dx = 1."""
    u_k = np.sqrt(2.0 / np.pi) * np.sin(k * x_pts)
    lambda_k = float(k ** 2)
    return u_k, lambda_k


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class EigenmodeNet(nn.Module):
    """Small MLP for a single eigenfunction with hard-BC enforcement."""

    def __init__(self, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 32, 32]
        layers: list[nn.Module] = []
        prev = 1
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.Tanh()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return B(x) * self.net(x)


# ---------------------------------------------------------------------------
# Interior sampling & loss terms
# ---------------------------------------------------------------------------

def sample_interior(
    n: int,
    device: torch.device,
    requires_grad: bool = True,
) -> torch.Tensor:
    """MC sample of (0, π), shape (n, 1)."""
    x = torch.rand(n, 1, device=device) * DOMAIN_LEN + X_MIN
    if requires_grad:
        x.requires_grad_(True)
    return x


def rayleigh_quotient(
    model: nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (num = E[|u'|²], den = E[u²], λ̂ = num/den)."""
    u = model(x)
    du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    num = (du_dx ** 2).mean()
    den = (u ** 2).mean()
    return num, den, num / den.clamp(min=1e-12)


def orthogonality_penalty(
    model: nn.Module,
    prev_models: list[nn.Module],
    x: torch.Tensor,
) -> torch.Tensor:
    """Sum_j cos²(angle(u, u_j)) — scale-invariant penalty.

    Using the unnormalised inner product would let the net escape the penalty
    by shrinking ‖u‖ (the Rayleigh quotient is itself scale-invariant, so
    shrinking doesn't cost anything). The cosine form keeps pressure on the
    *direction* regardless of scale.
    """
    if not prev_models:
        return torch.zeros(1, device=x.device).squeeze()
    u = model(x)
    u_norm = torch.sqrt((u ** 2).mean().clamp(min=1e-12))
    pen = torch.zeros(1, device=x.device).squeeze()
    for pm in prev_models:
        with torch.no_grad():
            u_prev = pm(x)
            up_norm = torch.sqrt((u_prev ** 2).mean().clamp(min=1e-12))
        cos_sq = ((u * u_prev).mean() / (u_norm * up_norm)) ** 2
        pen = pen + cos_sq
    return pen


def compute_losses(
    model: nn.Module,
    prev_models: list[nn.Module],
    x: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Variational loss dict: {rayleigh, orth}.

    _num / _den / _lambda_hat are detached diagnostics — not used in backprop.
    """
    num, den, lam_hat = rayleigh_quotient(model, x)
    orth = orthogonality_penalty(model, prev_models, x)
    return {
        "rayleigh": lam_hat,
        "orth": orth,
        "_num": num.detach(),
        "_den": den.detach(),
        "_lambda_hat": lam_hat.detach(),
    }


# ---------------------------------------------------------------------------
# Sequential training
# ---------------------------------------------------------------------------

def train_eigenmode(
    model: nn.Module,
    prev_models: list[nn.Module],
    device: torch.device,
    n_epochs: int = 5_000,
    lr: float = 1e-3,
    alpha_orth: float = 100.0,
    n_interior: int = 1024,
    log_every: int = 500,
    desc: str = "Mode",
) -> dict[str, list]:
    """Train one eigenmode network by Rayleigh + α·orth on resampled interior."""
    for pm in prev_models:
        for p in pm.parameters():
            p.requires_grad_(False)
        pm.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list] = {
        "step": [], "rayleigh": [], "orth": [], "total": [], "lambda_hat": [],
    }

    for step in range(n_epochs):
        optimizer.zero_grad()
        x = sample_interior(n_interior, device, requires_grad=True)
        losses = compute_losses(model, prev_models, x)
        total = losses["rayleigh"] + alpha_orth * losses["orth"]
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_every == 0:
            history["step"].append(step)
            history["rayleigh"].append(float(losses["rayleigh"].item()))
            history["orth"].append(float(losses["orth"].item()))
            history["total"].append(float(total.item()))
            history["lambda_hat"].append(float(losses["_lambda_hat"].item()))

    return history


def train_first_n_modes(
    n_modes: int,
    device: torch.device,
    hidden_dims: list[int] | None = None,
    n_epochs: int = 5_000,
    lr: float = 1e-3,
    alpha_orth: float = 100.0,
    n_interior: int = 1024,
    log_every: int = 500,
) -> tuple[list[nn.Module], list[dict[str, list]]]:
    """Sequentially train modes 1, 2, …, n_modes. Returns (models, histories)."""
    models: list[nn.Module] = []
    histories: list[dict[str, list]] = []
    for k in range(1, n_modes + 1):
        torch.manual_seed(k)
        model = EigenmodeNet(hidden_dims=hidden_dims).to(device)
        history = train_eigenmode(
            model,
            prev_models=models,
            device=device,
            n_epochs=n_epochs,
            lr=lr,
            alpha_orth=alpha_orth,
            n_interior=n_interior,
            log_every=log_every,
            desc=f"Mode {k}",
        )
        models.append(model)
        histories.append(history)
    return models, histories


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_eigenmode(
    model: nn.Module,
    x_pts: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x = torch.tensor(x_pts, dtype=torch.float32, device=device).unsqueeze(1)
    return model(x).cpu().numpy().flatten()


def _l2_normalise(u: np.ndarray) -> np.ndarray:
    """Rescale u so that ∫_0^π u² dx = 1 (trapezoid rule on the input grid)."""
    x = np.linspace(X_MIN, X_MAX, len(u))
    integral = np.trapz(u ** 2, x)
    return u / (np.sqrt(integral) + 1e-12)


def evaluate_eigenmode(
    model: nn.Module,
    k: int,
    device: torch.device,
    nx: int = 500,
) -> dict[str, float]:
    """
    Relative L² error of the k-th mode (after L² normalisation & sign fix)
    plus the Rayleigh-quotient estimate of λ_k.
    """
    x_pts = np.linspace(X_MIN, X_MAX, nx)
    u_ref, lambda_k = reference_eigenmode(k, x_pts)
    u_pred = predict_eigenmode(model, x_pts, device)

    # L² normalise prediction on the evaluation grid
    u_pred_n = _l2_normalise(u_pred)
    if float(np.dot(u_pred_n, u_ref)) < 0.0:
        u_pred_n = -u_pred_n

    rel_l2 = float(np.linalg.norm(u_pred_n - u_ref) / (np.linalg.norm(u_ref) + 1e-10))

    x_t = torch.tensor(x_pts, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
    _, _, lam_hat = rayleigh_quotient(model, x_t)
    lam_hat_val = float(lam_hat.item())

    return {
        "k": k,
        "rel_l2": rel_l2,
        "lambda_true": lambda_k,
        "lambda_hat": lam_hat_val,
        "lambda_rel_err": abs(lam_hat_val - lambda_k) / lambda_k,
    }


def evaluate_all(
    models: list[nn.Module],
    device: torch.device,
    nx: int = 500,
) -> list[dict[str, float]]:
    return [evaluate_eigenmode(m, k + 1, device, nx=nx) for k, m in enumerate(models)]
