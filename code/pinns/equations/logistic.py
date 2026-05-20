"""Logistic ODE: u' = r·u·(1 - u/K), sanity-check benchmark for LC-PINN."""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------

R = 2.0       # growth rate
K = 1.0       # carrying capacity
U0 = 0.1      # initial condition
T_MAX = 2.0   # time domain [0, T_MAX]
N_OBS = 15    # noisy observations used as data term
OBS_NOISE = 0.02

DIM_PHYS = 1    # input: t only
DIM_LAMBDA = 3  # three loss terms: ODE residual, IC, data


# ---------------------------------------------------------------------------
# Exact solution
# ---------------------------------------------------------------------------

def exact_solution(t: np.ndarray) -> np.ndarray:
    """u(t) = K·u0·e^(rt) / (K + u0·(e^(rt) - 1))"""
    exp_rt = np.exp(R * t)
    return K * U0 * exp_rt / (K + U0 * (exp_rt - 1.0))


def compute_reference_solution(
    n_pts: int = 500,
    snap_times: list[float] | None = None,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Returns snapshots {t_val: (t_array, u_array)} for evaluation.

    For the 1D ODE, each "snapshot" is just a single time point with the
    exact solution evaluated on a dense grid around it. We return the full
    t grid at each snapshot time so evaluation uses the same interface as PDEs.
    """
    if snap_times is None:
        snap_times = [0.5, 1.0, 1.5, 2.0]
    t_eval = np.linspace(0, T_MAX, n_pts)
    u_exact = exact_solution(t_eval)
    snapshots: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for t_val in snap_times:
        snapshots[t_val] = (t_eval, u_exact)
    return snapshots


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def generate_training_data(
    n_pde: int = 2000,
    n_ic: int = 200,
    n_data: int = 200,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Collocation points (ODE residual), IC point, noisy observations."""
    rng = np.random.default_rng(seed)
    if isinstance(device, str):
        device = torch.device(device)

    # ODE collocation: t ~ Uniform(0, T_MAX)
    coords_pde = (torch.rand(n_pde, 1) * T_MAX).to(device)

    # IC: t = 0, u = U0
    coords_ic = torch.zeros(n_ic, 1).to(device)
    u_ic = torch.full((n_ic, 1), U0).to(device)

    # Sparse noisy observations
    t_obs = np.sort(rng.uniform(0.0, T_MAX, size=N_OBS))
    u_obs = exact_solution(t_obs) + rng.normal(0.0, OBS_NOISE, size=N_OBS)
    idx = rng.integers(0, N_OBS, size=n_data)
    coords_data = torch.tensor(t_obs[idx], dtype=torch.float32).unsqueeze(1).to(device)
    u_data = torch.tensor(u_obs[idx], dtype=torch.float32).unsqueeze(1).to(device)

    return {
        "coords_pde":  coords_pde,
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
    """ODE residual + IC + data losses for LC-PINN."""
    coords = batch["coords_pde"].requires_grad_(True)
    u = model(coords, log_lambda)
    u_t = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0]
    residual = u_t - R * u * (1.0 - u / K)
    L_ode = torch.mean(residual ** 2)

    L_ic = torch.mean((model(batch["coords_ic"], log_lambda) - batch["u_ic"]) ** 2)
    L_data = torch.mean((model(batch["coords_data"], log_lambda) - batch["u_data"]) ** 2)

    return {"pde": L_ode, "ic": L_ic, "data": L_data}


def compute_losses_fixed(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Same three terms for a fixed-weight PINN."""
    coords = batch["coords_pde"].requires_grad_(True)
    u = model(coords)
    u_t = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0]
    residual = u_t - R * u * (1.0 - u / K)
    L_ode = torch.mean(residual ** 2)

    L_ic = torch.mean((model(batch["coords_ic"]) - batch["u_ic"]) ** 2)
    L_data = torch.mean((model(batch["coords_data"]) - batch["u_data"]) ** 2)

    return {"pde": L_ode, "ic": L_ic, "data": L_data}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_solution(
    model: torch.nn.Module,
    log_lambda: torch.Tensor | None,
    t_eval: np.ndarray,
    t_val: float,  # unused for ODE, kept for interface consistency
    device: torch.device,
) -> np.ndarray:
    """Predict u(t) for t_eval array."""
    t_tensor = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1).to(device)
    if log_lambda is not None:
        return model(t_tensor, log_lambda).cpu().numpy().flatten()
    return model(t_tensor).cpu().numpy().flatten()


def evaluate(
    model: torch.nn.Module,
    log_lambda: torch.Tensor | None,
    ref_snapshots: dict[float, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> dict[float, float]:
    """Relative L2 error vs exact solution. For ODE, we evaluate on full t grid."""
    t_eval = np.linspace(0, T_MAX, 500)
    u_exact = exact_solution(t_eval)
    u_pred = predict_solution(model, log_lambda, t_eval, 0.0, device)
    err = relative_l2(u_pred, u_exact)
    # Return single entry keyed by T_MAX for consistency
    return {T_MAX: err}


def relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-10))
