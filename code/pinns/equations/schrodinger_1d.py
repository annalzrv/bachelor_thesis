"""1D driven stationary Schrödinger with parametric harmonic potential.

    -u''(x) + V(x; α) u(x) = f(x; α),    x ∈ (0, 1),   u(0) = u(1) = 0,
    V(x; α) = α² (x − 0.5)²,             α ∈ [α_MIN, α_MAX].

This is the time-independent Schrödinger operator -d²/dx² + V acting on a
forced state — physically, a driven quantum system in a parametric harmonic
well centred at x = 1/2. The parameter α controls trap stiffness; the
solution localises more sharply near x=1/2 as α grows.

Manufactured solution:

    u_ref(x; α) = sin(π x) · exp(-α (x - 0.5)² / 2).

Closed-form forcing:

    f(x; α) = (π² + α) · u_ref(x; α)
              + 2π α (x − 0.5) · cos(π x) · exp(-α (x − 0.5)² / 2).

Derivation: with u = sin(πx) · g(x;α) where g = exp(-α(x-0.5)²/2),

    g'  = -α(x-0.5) g
    g'' = -α g + α² (x-0.5)² g
    u'' = [-π² - α + α²(x-0.5)²] sin(π x) g  -  2π α (x-0.5) cos(π x) g
    -u'' + V u = (π² + α) u  +  2π α (x-0.5) cos(π x) g.

Network input is (x, α_norm) where α_norm ∈ [-1, 1] and dim_lambda=1.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------

X_MIN: float = 0.0
X_MAX: float = 1.0

ALPHA_MIN: float = 0.5
ALPHA_MAX: float = 10.0

DIM_PHYS: int = 1     # network coords: x
DIM_LAMBDA: int = 1   # condition on α_norm only


# ---------------------------------------------------------------------------
# Manufactured ground truth & forcing
# ---------------------------------------------------------------------------

def alpha_to_norm(alpha):
    return 2.0 * (alpha - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN) - 1.0


def norm_to_alpha(alpha_norm):
    return ALPHA_MIN + 0.5 * (ALPHA_MAX - ALPHA_MIN) * (alpha_norm + 1.0)


def potential(x, alpha: float):
    """V(x; α) = α² (x - 0.5)²."""
    if isinstance(x, torch.Tensor):
        return (alpha ** 2) * (x - 0.5) ** 2
    return (alpha ** 2) * (x - 0.5) ** 2


def reference_solution(x, alpha: float):
    if isinstance(x, torch.Tensor):
        return torch.sin(torch.pi * x) * torch.exp(-alpha * (x - 0.5) ** 2 / 2.0)
    return np.sin(np.pi * x) * np.exp(-alpha * (x - 0.5) ** 2 / 2.0)


def forcing(x, alpha: float):
    """f = -u'' + V u for the manufactured solution."""
    if isinstance(x, torch.Tensor):
        pi = float(np.pi)
        sx = torch.sin(pi * x)
        cx = torch.cos(pi * x)
        g = torch.exp(-alpha * (x - 0.5) ** 2 / 2.0)
        u = sx * g
        return (pi ** 2 + alpha) * u + 2.0 * pi * alpha * (x - 0.5) * cx * g
    sx = np.sin(np.pi * x)
    cx = np.cos(np.pi * x)
    g = np.exp(-alpha * (x - 0.5) ** 2 / 2.0)
    u = sx * g
    return (np.pi ** 2 + alpha) * u + 2.0 * np.pi * alpha * (x - 0.5) * cx * g


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def generate_training_data(
    n_pde: int = 1024,
    n_bc: int = 64,
    n_data: int = 200,
    n_data_alpha_values: int = 5,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    if isinstance(device, str):
        device = torch.device(device)

    x_pde = rng.uniform(X_MIN, X_MAX, size=(n_pde, 1)).astype(np.float32)
    coords_pde = torch.tensor(x_pde, device=device)

    half = n_bc // 2
    coords_bc = torch.tensor(
        np.concatenate([
            np.full((half, 1), X_MIN, dtype=np.float32),
            np.full((n_bc - half, 1), X_MAX, dtype=np.float32),
        ], axis=0),
        device=device,
    )
    u_bc = torch.zeros(n_bc, 1, device=device)

    alpha_vals = np.linspace(ALPHA_MIN, ALPHA_MAX, n_data_alpha_values, dtype=np.float32)
    pts_per_alpha = max(n_data // n_data_alpha_values, 5)
    x_list, a_list, u_list = [], [], []
    for a in alpha_vals:
        x = rng.uniform(X_MIN, X_MAX, size=(pts_per_alpha, 1)).astype(np.float32)
        u = reference_solution(x.flatten(), float(a)).astype(np.float32)
        x_list.append(x)
        a_list.append(np.full(pts_per_alpha, a, dtype=np.float32))
        u_list.append(u)
    coords_data = torch.tensor(np.concatenate(x_list, axis=0), device=device)
    alpha_data = torch.tensor(np.concatenate(a_list), device=device)
    u_data = torch.tensor(np.concatenate(u_list)[:, None], device=device)

    return {
        "coords_pde": coords_pde,
        "coords_bc":  coords_bc,
        "u_bc":       u_bc,
        "coords_data":  coords_data,
        "k_data":       alpha_data,  # named 'k_data' for compatibility with helm scripts
        "u_data":       u_data,
    }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _u_xx(u, coords):
    g = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(g, coords, torch.ones_like(g), create_graph=True)[0]
    return u_xx


def compute_losses(
    model: torch.nn.Module,
    alpha_norm: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    a = norm_to_alpha(alpha_norm.item() if alpha_norm.numel() == 1 else alpha_norm[0].item())

    coords = batch["coords_pde"].requires_grad_(True)
    u = model(coords, alpha_norm)
    uxx = _u_xx(u, coords)
    V = potential(coords, a)
    f = forcing(coords, a)
    residual = -uxx + V * u - f
    L_pde = torch.mean(residual ** 2)

    L_bc = torch.mean((model(batch["coords_bc"], alpha_norm) - batch["u_bc"]) ** 2)

    a_data = batch["k_data"]
    a_norm_per_point = alpha_to_norm(a_data).unsqueeze(-1)
    u_pred_data = model(batch["coords_data"], a_norm_per_point)
    L_data = torch.mean((u_pred_data - batch["u_data"]) ** 2)

    return {"pde": L_pde, "bc": L_bc, "data": L_data}


def compute_losses_fixed(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    alpha_fixed: float,
) -> dict[str, torch.Tensor]:
    coords = batch["coords_pde"].requires_grad_(True)
    u = model(coords)
    uxx = _u_xx(u, coords)
    V = potential(coords, alpha_fixed)
    f = forcing(coords, alpha_fixed)
    residual = -uxx + V * u - f
    L_pde = torch.mean(residual ** 2)

    L_bc = torch.mean((model(batch["coords_bc"]) - batch["u_bc"]) ** 2)

    mask = torch.isclose(batch["k_data"], torch.tensor(alpha_fixed, device=batch["k_data"].device))
    if mask.any():
        coords_d = batch["coords_data"][mask]
        u_d = batch["u_data"][mask]
        L_data = torch.mean((model(coords_d) - u_d) ** 2)
    else:
        L_data = torch.zeros(1, device=L_pde.device).squeeze()

    return {"pde": L_pde, "bc": L_bc, "data": L_data}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_solution(
    model: torch.nn.Module,
    alpha: float,
    x_pts: np.ndarray,
    device: torch.device,
    is_lc: bool = True,
) -> np.ndarray:
    x_t = torch.tensor(x_pts, dtype=torch.float32, device=device).reshape(-1, 1)
    if is_lc:
        a_norm = torch.tensor([alpha_to_norm(alpha)], dtype=torch.float32, device=device)
        return model(x_t, a_norm).cpu().numpy().flatten()
    return model(x_t).cpu().numpy().flatten()


def evaluate_at_alpha(
    model: torch.nn.Module,
    alpha: float,
    device: torch.device,
    nx: int = 256,
    is_lc: bool = True,
) -> float:
    x = np.linspace(X_MIN, X_MAX, nx, dtype=np.float32)
    u_ref = reference_solution(x, alpha).astype(np.float32)
    u_pred = predict_solution(model, alpha, x, device, is_lc=is_lc)
    return float(np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-10))


# Aliases so harvest / generic helpers work uniformly with helmholtz modules.
K_MIN = ALPHA_MIN
K_MAX = ALPHA_MAX
k_to_norm = alpha_to_norm
norm_to_k = norm_to_alpha
evaluate_at_k = evaluate_at_alpha
