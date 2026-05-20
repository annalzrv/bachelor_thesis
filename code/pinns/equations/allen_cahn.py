"""Allen-Cahn equation — reaction-diffusion with stiff interface (PINN benchmark)."""

from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------

EPS2 = 0.01          # diffusion coefficient epsilon^2
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

DIM_PHYS = 2    # inputs: x, t
DIM_LAMBDA = 3  # IC, periodic BC penalty, PDE residual


# ---------------------------------------------------------------------------
# Reference solution (Fourier-spectral + scipy ODE)
# ---------------------------------------------------------------------------

def compute_reference_solution(
    nx: int = 512,
    snap_times: list[float] | None = None,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """
    Spectral reference for u_t - eps^2 * u_xx + 5u^3 - 5u = 0,
    x in [-1,1], periodic BCs.  IC: u(0,x) = x^2 * cos(pi*x).
    """
    if snap_times is None:
        snap_times = [0.1, 0.25, 0.5, 0.75, 1.0]

    x_grid = np.linspace(X_MIN, X_MAX, nx, endpoint=False)
    dx = x_grid[1] - x_grid[0]
    u0 = (x_grid ** 2) * np.cos(np.pi * x_grid)

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)

    def rhs(t, u_flat):
        u_hat = np.fft.fft(u_flat)
        laplacian_u = np.real(np.fft.ifft(-(kx ** 2) * u_hat))
        return EPS2 * laplacian_u - 5.0 * u_flat ** 3 + 5.0 * u_flat

    sol = solve_ivp(
        rhs, [T_MIN, T_MAX], u0,
        method="Radau",
        t_eval=snap_times,
        rtol=1e-6, atol=1e-8,
    )
    snapshots: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for i, t_val in enumerate(snap_times):
        snapshots[t_val] = (x_grid.copy(), sol.y[:, i])
    return snapshots


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def generate_training_data(
    ref_snapshots: dict[float, tuple[np.ndarray, np.ndarray]],
    n_pde: int = 4000,
    n_bc: int = 400,
    n_ic: int = 800,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """PDE collocation, periodic BC penalty points, IC points.

    Defaults are larger than Burgers/BL because Allen-Cahn with eps^2=1e-4
    is stiff and the trivial solution u=0 is a strong attractor. More IC
    points and early-time PDE collocation help prevent collapse.
    """
    rng = np.random.default_rng(seed)
    if isinstance(device, str):
        device = torch.device(device)

    # PDE collocation — bias half the points toward early times (t < 0.2)
    # to help the network learn the IC-driven dynamics before the reaction
    # term drives everything to +/-1
    n_early = n_pde // 2
    n_late = n_pde - n_early
    x_early = rng.uniform(X_MIN, X_MAX, n_early).astype(np.float32)
    t_early = rng.uniform(T_MIN, 0.2, n_early).astype(np.float32)
    x_late = rng.uniform(X_MIN, X_MAX, n_late).astype(np.float32)
    t_late = rng.uniform(T_MIN, T_MAX, n_late).astype(np.float32)
    x_pde = np.concatenate([x_early, x_late])
    t_pde = np.concatenate([t_early, t_late])
    coords_pde = torch.tensor(np.column_stack([x_pde, t_pde])).to(device)

    # Periodic BC: pairs at x=-1 and x=+1
    t_bc = rng.uniform(T_MIN, T_MAX, n_bc).astype(np.float32)
    coords_bc_left = torch.tensor(
        np.column_stack([np.full(n_bc, X_MIN, dtype=np.float32), t_bc])
    ).to(device)
    coords_bc_right = torch.tensor(
        np.column_stack([np.full(n_bc, X_MAX, dtype=np.float32), t_bc])
    ).to(device)

    # IC: u(0, x) = x^2 * cos(pi*x)
    x_ic = rng.uniform(X_MIN, X_MAX, n_ic).astype(np.float32)
    coords_ic = torch.tensor(np.column_stack([x_ic, np.zeros(n_ic, dtype=np.float32)])).to(device)
    u_ic = torch.tensor(
        (x_ic ** 2) * np.cos(np.pi * x_ic), dtype=torch.float32
    ).unsqueeze(1).to(device)

    return {
        "coords_pde":       coords_pde,
        "coords_bc_left":   coords_bc_left,
        "coords_bc_right":  coords_bc_right,
        "coords_ic":        coords_ic,
        "u_ic":             u_ic,
    }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _periodic_bc_loss(
    model: torch.nn.Module,
    coords_bc_left: torch.Tensor,
    coords_bc_right: torch.Tensor,
    log_lambda: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalise u(-1,t) != u(+1,t) and u_x(-1,t) != u_x(+1,t)."""
    def _predict_with_grad(coords):
        c = coords.requires_grad_(True)
        u = model(c, log_lambda) if log_lambda is not None else model(c)
        u_x = torch.autograd.grad(u, c, torch.ones_like(u), create_graph=True)[0][:, 0:1]
        return u, u_x

    u_left,  ux_left  = _predict_with_grad(coords_bc_left)
    u_right, ux_right = _predict_with_grad(coords_bc_right)

    return torch.mean((u_left - u_right) ** 2) + torch.mean((ux_left - ux_right) ** 2)


def compute_losses(
    model: torch.nn.Module,
    log_lambda: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """IC + periodic BC + PDE residual for LC-PINN."""
    coords = batch["coords_pde"].requires_grad_(True)
    u = model(coords, log_lambda)
    grads = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = grads[:, 0:1], grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, coords, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    residual = u_t - EPS2 * u_xx + 5.0 * u ** 3 - 5.0 * u
    L_pde = torch.mean(residual ** 2)

    L_bc = _periodic_bc_loss(
        model, batch["coords_bc_left"], batch["coords_bc_right"], log_lambda
    )
    L_ic = torch.mean((model(batch["coords_ic"], log_lambda) - batch["u_ic"]) ** 2)

    return {"pde": L_pde, "bc": L_bc, "ic": L_ic}


def compute_losses_fixed(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Same three terms for a fixed-weight PINN."""
    coords = batch["coords_pde"].requires_grad_(True)
    u = model(coords)
    grads = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = grads[:, 0:1], grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, coords, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    residual = u_t - EPS2 * u_xx + 5.0 * u ** 3 - 5.0 * u
    L_pde = torch.mean(residual ** 2)

    L_bc = _periodic_bc_loss(model, batch["coords_bc_left"], batch["coords_bc_right"])
    L_ic = torch.mean((model(batch["coords_ic"]) - batch["u_ic"]) ** 2)

    return {"pde": L_pde, "bc": L_bc, "ic": L_ic}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_solution(
    model: torch.nn.Module,
    log_lambda: torch.Tensor | None,
    x_pts: np.ndarray,
    t_val: float,
    device: torch.device,
) -> np.ndarray:
    """Predict u(x, t_val)."""
    x_t = torch.tensor(x_pts, dtype=torch.float32).unsqueeze(1)
    t_t = torch.full((len(x_pts), 1), t_val, dtype=torch.float32)
    coords = torch.cat([x_t, t_t], dim=1).to(device)
    if log_lambda is not None:
        return model(coords, log_lambda).cpu().numpy().flatten()
    return model(coords).cpu().numpy().flatten()


def evaluate(
    model: torch.nn.Module,
    log_lambda: torch.Tensor | None,
    ref_snapshots: dict[float, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> dict[float, float]:
    """Relative L2 error vs spectral reference at each snapshot time."""
    errors: dict[float, float] = {}
    for t_val, (x_ref, u_ref) in sorted(ref_snapshots.items()):
        u_pred = predict_solution(model, log_lambda, x_ref, t_val, device)
        errors[t_val] = relative_l2(u_pred, u_ref)
    return errors


def relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-10))
