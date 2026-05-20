"""1D viscous Burgers equation — core PINN benchmark (Raissi et al. setup)."""

from __future__ import annotations

import numpy as np
import torch
from scipy.fft import fft, ifft


# ---------------------------------------------------------------------------
# Problem parameters (standard Raissi setup)
# ---------------------------------------------------------------------------

NU = 0.01 / np.pi   # viscosity
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

DIM_PHYS = 2    # inputs: x, t
DIM_LAMBDA = 4  # IC, BC, PDE residual, sparse interior data


# ---------------------------------------------------------------------------
# Reference solution (pseudo-spectral)
# ---------------------------------------------------------------------------

def compute_reference_solution(
    nx: int = 512,
    snap_times: list[float] | None = None,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """
    Fourier-spectral reference solved with Radau (implicit, stiff-stable).
    IC: u(0, x) = -sin(pi*x), periodic domain.

    Returns snapshots {t_val: (x_array, u_array)} on the spectral grid.
    """
    from scipy.integrate import solve_ivp

    if snap_times is None:
        snap_times = [0.25, 0.50, 0.75, 1.00]

    x_grid = np.linspace(X_MIN, X_MAX, nx, endpoint=False)
    dx = x_grid[1] - x_grid[0]
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)

    def rhs(t_val, u_cur):
        u_hat = fft(u_cur)
        dudx = np.real(ifft(1j * kx * u_hat))
        d2udx2 = np.real(ifft(-(kx ** 2) * u_hat))
        return -u_cur * dudx + NU * d2udx2

    u0 = -np.sin(np.pi * x_grid)

    sol = solve_ivp(
        rhs, [T_MIN, T_MAX], u0,
        method="Radau",
        t_eval=sorted(snap_times),
        rtol=1e-6, atol=1e-8,
    )

    snapshots: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for i, t_val in enumerate(sorted(snap_times)):
        snapshots[t_val] = (x_grid.copy(), sol.y[:, i])
    return snapshots


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def generate_training_data(
    ref_snapshots: dict[float, tuple[np.ndarray, np.ndarray]],
    n_pde: int = 2000,
    n_bc: int = 200,
    n_ic: int = 200,
    n_data: int = 200,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """PDE collocation, Dirichlet BCs at x=+-1, IC u(0,x)=-sin(pi*x), sparse data."""
    rng = np.random.default_rng(seed)
    if isinstance(device, str):
        device = torch.device(device)

    # PDE collocation
    x_pde = rng.uniform(X_MIN, X_MAX, n_pde).astype(np.float32)
    t_pde = rng.uniform(T_MIN, T_MAX, n_pde).astype(np.float32)
    coords_pde = torch.tensor(np.column_stack([x_pde, t_pde])).to(device)

    # BC: u(t, -1) = 0 and u(t, 1) = 0
    t_bc = rng.uniform(T_MIN, T_MAX, n_bc).astype(np.float32)
    x_left = np.full(n_bc // 2, X_MIN, dtype=np.float32)
    x_right = np.full(n_bc - n_bc // 2, X_MAX, dtype=np.float32)
    x_bc = np.concatenate([x_left, x_right])
    t_bc_full = np.concatenate([t_bc[:n_bc // 2], t_bc[n_bc // 2:]])
    coords_bc = torch.tensor(np.column_stack([x_bc, t_bc_full])).to(device)
    u_bc = torch.zeros(n_bc, 1).to(device)

    # IC: u(0, x) = -sin(pi*x)
    x_ic = rng.uniform(X_MIN, X_MAX, n_ic).astype(np.float32)
    coords_ic = torch.tensor(np.column_stack([x_ic, np.zeros(n_ic, dtype=np.float32)])).to(device)
    u_ic = torch.tensor(-np.sin(np.pi * x_ic), dtype=torch.float32).unsqueeze(1).to(device)

    # Sparse interior data from reference snapshots
    snap_times = sorted(ref_snapshots.keys())
    pts_per_snap = max(n_data // len(snap_times), 10)
    x_list, t_list, u_list = [], [], []
    for t_val in snap_times:
        xr, ur = ref_snapshots[t_val]
        idx = rng.integers(0, len(xr), pts_per_snap)
        x_list.append(xr[idx].astype(np.float32))
        t_list.append(np.full(pts_per_snap, t_val, dtype=np.float32))
        u_list.append(ur[idx].astype(np.float32))
    coords_data = torch.tensor(np.column_stack([
        np.concatenate(x_list), np.concatenate(t_list)
    ])).to(device)
    u_data = torch.tensor(np.concatenate(u_list), dtype=torch.float32).unsqueeze(1).to(device)

    return {
        "coords_pde":  coords_pde,
        "coords_bc":   coords_bc,
        "u_bc":        u_bc,
        "coords_ic":   coords_ic,
        "u_ic":        u_ic,
        "coords_data": coords_data,
        "u_data":      u_data,
    }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def compute_losses(
    model: torch.nn.Module,
    log_lambda: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """IC + BC + PDE residual + sparse data for LC-PINN."""
    coords = batch["coords_pde"].requires_grad_(True)
    u = model(coords, log_lambda)
    grads = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = grads[:, 0:1], grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, coords, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    residual = u_t + u * u_x - NU * u_xx
    L_pde = torch.mean(residual ** 2)

    L_bc   = torch.mean((model(batch["coords_bc"],   log_lambda) - batch["u_bc"])   ** 2)
    L_ic   = torch.mean((model(batch["coords_ic"],   log_lambda) - batch["u_ic"])   ** 2)
    L_data = torch.mean((model(batch["coords_data"], log_lambda) - batch["u_data"]) ** 2)

    return {"pde": L_pde, "bc": L_bc, "ic": L_ic, "data": L_data}


def compute_losses_fixed(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Same four terms for a fixed-weight PINN."""
    coords = batch["coords_pde"].requires_grad_(True)
    u = model(coords)
    grads = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = grads[:, 0:1], grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, coords, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    residual = u_t + u * u_x - NU * u_xx
    L_pde = torch.mean(residual ** 2)

    L_bc   = torch.mean((model(batch["coords_bc"])   - batch["u_bc"])   ** 2)
    L_ic   = torch.mean((model(batch["coords_ic"])   - batch["u_ic"])   ** 2)
    L_data = torch.mean((model(batch["coords_data"]) - batch["u_data"]) ** 2)

    return {"pde": L_pde, "bc": L_bc, "ic": L_ic, "data": L_data}


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
