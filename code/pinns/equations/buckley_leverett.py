"""Buckley-Leverett equation — two-phase flow benchmark for LC-PINN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BLDomain:
    """Physical domain, mobility ratio, and network input sizes.

    `epsilon`: viscous regularisation coefficient for −ε·s_xx (capillary
    diffusion proxy). 0.0 → inviscid BL (default, original behaviour).
    Realistic capillary range: 1e−3…1e−2.
    """
    m_ratio: float = 2.0
    x_min: float = 0.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 0.5
    dim_phys: int = 2
    dim_lambda: int = 4
    epsilon: float = 0.0


DEFAULT_DOMAIN = BLDomain()

DIM_PHYS = 2
DIM_LAMBDA = 4

_EPS = 1e-10


# ---------------------------------------------------------------------------
# Fractional flow
# ---------------------------------------------------------------------------

def f_bl_torch(s: torch.Tensor, domain: BLDomain = DEFAULT_DOMAIN) -> torch.Tensor:
    m = domain.m_ratio
    return s**2 / (s**2 + m * (1.0 - s) ** 2 + _EPS)


def df_ds_torch(s: torch.Tensor, domain: BLDomain = DEFAULT_DOMAIN) -> torch.Tensor:
    m = domain.m_ratio
    num = 2.0 * m * s * (1.0 - s)
    den = (s**2 + m * (1.0 - s) ** 2 + _EPS) ** 2
    return num / den


def f_bl_np(s: np.ndarray, domain: BLDomain = DEFAULT_DOMAIN) -> np.ndarray:
    m = domain.m_ratio
    return s**2 / (s**2 + m * (1.0 - s) ** 2 + _EPS)


def df_ds_np(s: np.ndarray, domain: BLDomain = DEFAULT_DOMAIN) -> np.ndarray:
    m = domain.m_ratio
    num = 2.0 * m * s * (1.0 - s)
    den = (s**2 + m * (1.0 - s) ** 2 + _EPS) ** 2
    return num / den


# ---------------------------------------------------------------------------
# Reference solution (Rusanov FVM)
# ---------------------------------------------------------------------------

def compute_reference_solution(
    nx: int = 500,
    domain: BLDomain = DEFAULT_DOMAIN,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Rusanov FVM for ds/dt + df/dx = 0. Returns snapshots[t] = (x, s(x))."""
    x_min, x_max = domain.x_min, domain.x_max
    t_max = domain.t_max

    dx = (x_max - x_min) / nx
    x = np.linspace(x_min + dx / 2, x_max - dx / 2, nx)
    max_speed = np.max(np.abs(df_ds_np(np.linspace(0.01, 0.99, 1000), domain)))
    dt = 0.4 * dx / max_speed
    nt = int(np.ceil(t_max / dt))
    dt = t_max / nt

    s = np.zeros(nx)
    save_times = [0.1, 0.2, 0.3, 0.4, 0.5]
    snapshots: dict[float, tuple[np.ndarray, np.ndarray]] = {0.0: (x.copy(), s.copy())}

    t_cur = 0.0
    for _ in range(nt):
        s_ext = np.concatenate([[1.0], s, [s[-1]]])
        f_ext = f_bl_np(s_ext, domain)
        sL, sR = s_ext[:-1], s_ext[1:]
        fL, fR = f_ext[:-1], f_ext[1:]
        aL = np.abs(df_ds_np(np.clip(sL, 0.01, 0.99), domain))
        aR = np.abs(df_ds_np(np.clip(sR, 0.01, 0.99), domain))
        alpha = np.maximum(aL, aR)
        flux = 0.5 * (fL + fR) - 0.5 * alpha * (sR - sL)
        s = s - dt / dx * (flux[1:] - flux[:-1])
        s = np.clip(s, 0.0, 1.0)
        t_cur += dt
        for ts in save_times:
            if abs(t_cur - ts) < dt * 0.6 and ts not in snapshots:
                snapshots[ts] = (x.copy(), s.copy())

    if t_max not in snapshots:
        snapshots[t_max] = (x.copy(), s.copy())
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
    device: str | torch.device = "cpu",
    domain: BLDomain = DEFAULT_DOMAIN,
) -> dict[str, torch.Tensor]:
    """Create all training batches: PDE collocation, BC, IC, reference data."""
    x_min, x_max = domain.x_min, domain.x_max
    t_min, t_max = domain.t_min, domain.t_max

    if isinstance(device, str):
        device = torch.device(device)

    x_pde = torch.rand(n_pde, 1) * (x_max - x_min) + x_min
    t_pde = torch.rand(n_pde, 1) * (t_max - t_min) + t_min
    coords_pde = torch.cat([x_pde, t_pde], dim=1).to(device)

    t_bc = torch.rand(n_bc, 1) * t_max
    coords_bc = torch.cat([torch.zeros(n_bc, 1), t_bc], dim=1).to(device)
    u_bc = torch.ones(n_bc, 1).to(device)

    x_ic = torch.rand(n_ic, 1) * (x_max - x_min) + x_min
    coords_ic = torch.cat([x_ic, torch.zeros(n_ic, 1)], dim=1).to(device)
    u_ic = torch.zeros(n_ic, 1).to(device)

    x_list, t_list, u_list = [], [], []
    snap_times = sorted([t for t in ref_snapshots if t > 0])
    pts_per_snap = max(n_data // len(snap_times), 10)
    for t_val in snap_times:
        xr, sr = ref_snapshots[t_val]
        idx = np.random.choice(len(xr), min(pts_per_snap, len(xr)), replace=False)
        x_list.append(xr[idx])
        t_list.append(np.full(len(idx), t_val))
        u_list.append(sr[idx])
    coords_data = torch.cat(
        [
            torch.tensor(np.concatenate(x_list), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.concatenate(t_list), dtype=torch.float32).unsqueeze(1),
        ],
        dim=1,
    ).to(device)
    u_data = torch.tensor(np.concatenate(u_list), dtype=torch.float32).unsqueeze(1).to(device)

    return {
        "coords_pde": coords_pde,
        "coords_bc": coords_bc,
        "u_bc": u_bc,
        "coords_ic": coords_ic,
        "u_ic": u_ic,
        "coords_data": coords_data,
        "u_data": u_data,
    }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def compute_losses(
    model: torch.nn.Module,
    log_lambda: torch.Tensor,
    batch: dict[str, torch.Tensor],
    domain: BLDomain = DEFAULT_DOMAIN,
) -> dict[str, torch.Tensor]:
    """PDE residual + BC + IC + data losses for LC-PINN."""
    coords = batch["coords_pde"].requires_grad_(True)
    s = model(coords, log_lambda)
    grads = torch.autograd.grad(s, coords, torch.ones_like(s), create_graph=True)[0]
    s_x, s_t = grads[:, 0:1], grads[:, 1:2]
    residual = s_t + df_ds_torch(s, domain) * s_x
    if domain.epsilon > 0.0:
        s_xx = torch.autograd.grad(s_x, coords, torch.ones_like(s_x), create_graph=True)[0][:, 0:1]
        residual = residual - domain.epsilon * s_xx
    L_pde = torch.mean(residual**2)

    L_bc = torch.mean((model(batch["coords_bc"], log_lambda) - batch["u_bc"]) ** 2)
    L_ic = torch.mean((model(batch["coords_ic"], log_lambda) - batch["u_ic"]) ** 2)
    L_data = torch.mean((model(batch["coords_data"], log_lambda) - batch["u_data"]) ** 2)

    return {"pde": L_pde, "bc": L_bc, "ic": L_ic, "data": L_data}


def compute_losses_fixed(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    domain: BLDomain = DEFAULT_DOMAIN,
) -> dict[str, torch.Tensor]:
    """Same four terms for a fixed-weight PINN."""
    coords = batch["coords_pde"].requires_grad_(True)
    s = model(coords)
    grads = torch.autograd.grad(s, coords, torch.ones_like(s), create_graph=True)[0]
    s_x, s_t = grads[:, 0:1], grads[:, 1:2]
    residual = s_t + df_ds_torch(s, domain) * s_x
    if domain.epsilon > 0.0:
        s_xx = torch.autograd.grad(s_x, coords, torch.ones_like(s_x), create_graph=True)[0][:, 0:1]
        residual = residual - domain.epsilon * s_xx
    L_pde = torch.mean(residual**2)

    L_bc = torch.mean((model(batch["coords_bc"]) - batch["u_bc"]) ** 2)
    L_ic = torch.mean((model(batch["coords_ic"]) - batch["u_ic"]) ** 2)
    L_data = torch.mean((model(batch["coords_data"]) - batch["u_data"]) ** 2)

    return {"pde": L_pde, "bc": L_bc, "ic": L_ic, "data": L_data}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_solution(
    model: torch.nn.Module,
    log_lambda: torch.Tensor | None,
    x_pts: np.ndarray,
    t_val: float,
    device: torch.device,
) -> np.ndarray:
    """Predict s(x, t_val); clamped to [0,1] for BL physical consistency."""
    x_t = torch.tensor(x_pts, dtype=torch.float32).unsqueeze(1)
    t_t = torch.full((len(x_pts), 1), t_val, dtype=torch.float32)
    coords = torch.cat([x_t, t_t], dim=1).to(device)
    if log_lambda is not None:
        out = model(coords, log_lambda)
    else:
        out = model(coords)
    return out.clamp(0, 1).cpu().numpy().flatten()


def evaluate(
    model: torch.nn.Module,
    log_lambda: torch.Tensor | None,
    ref_snapshots: dict[float, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> dict[float, float]:
    """Relative L2 error vs FVM reference at each snapshot time (skips t=0)."""
    errors: dict[float, float] = {}
    for t_val, (x_ref, s_ref) in sorted(ref_snapshots.items()):
        if t_val == 0.0:
            continue
        s_pred = predict_solution(model, log_lambda, x_ref, t_val, device)
        errors[t_val] = float(np.linalg.norm(s_pred - s_ref) / (np.linalg.norm(s_ref) + 1e-10))
    return errors


def relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-10))


# ---------------------------------------------------------------------------
# Viscous reference solver (capillary-corrected BL)
# ---------------------------------------------------------------------------

def compute_viscous_reference_solution(
    nx: int = 1000,
    domain: BLDomain | None = None,
    epsilon: float | None = None,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Finite-difference solver for s_t + f'(s)·s_x = ε·s_xx.

    Uses Lax–Friedrichs for the convective flux and centred FD for the
    diffusive term. Time step satisfies CFL_conv (∝ Δx / max|f'|) AND
    CFL_diff (∝ Δx² / (2ε)); the diffusive bound dominates for ε ≳ 1e−2
    on fine grids, so the timestep can be small. nx=1000 keeps Δx²/(2ε)
    near 5e−6 for ε=0.01, which is fine.
    """
    if domain is None:
        domain = DEFAULT_DOMAIN
    eps = epsilon if epsilon is not None else domain.epsilon
    if eps <= 0:
        raise ValueError("compute_viscous_reference_solution requires epsilon > 0")

    x_min, x_max = domain.x_min, domain.x_max
    t_max = domain.t_max
    dx = (x_max - x_min) / nx
    x = np.linspace(x_min + dx / 2, x_max - dx / 2, nx)

    max_speed = float(np.max(np.abs(df_ds_np(np.linspace(0.01, 0.99, 1000), domain))))
    dt_conv = 0.4 * dx / max_speed
    dt_diff = 0.4 * dx**2 / (2.0 * eps)
    dt = min(dt_conv, dt_diff)
    nt = int(np.ceil(t_max / dt))
    dt = t_max / nt

    s = np.zeros(nx)
    save_times = [0.1, 0.2, 0.3, 0.4, 0.5]
    snapshots: dict[float, tuple[np.ndarray, np.ndarray]] = {0.0: (x.copy(), s.copy())}

    t_cur = 0.0
    for _ in range(nt):
        # Lax–Friedrichs convective flux (Dirichlet: s=1 at x_min, Neumann at x_max).
        s_ext = np.concatenate([[1.0], s, [s[-1]]])
        f_ext = f_bl_np(s_ext, domain)
        sL, sR = s_ext[:-1], s_ext[1:]
        fL, fR = f_ext[:-1], f_ext[1:]
        aL = np.abs(df_ds_np(np.clip(sL, 0.01, 0.99), domain))
        aR = np.abs(df_ds_np(np.clip(sR, 0.01, 0.99), domain))
        alpha = np.maximum(aL, aR)
        flux = 0.5 * (fL + fR) - 0.5 * alpha * (sR - sL)
        adv = -(flux[1:] - flux[:-1]) / dx

        # Centred-FD diffusion ε·s_xx with the same ghost cells.
        diff = eps * (s_ext[2:] - 2.0 * s_ext[1:-1] + s_ext[:-2]) / dx**2

        s = s + dt * (adv + diff)
        s = np.clip(s, 0.0, 1.0)
        t_cur += dt
        for ts in save_times:
            if abs(t_cur - ts) < dt * 0.6 and ts not in snapshots:
                snapshots[ts] = (x.copy(), s.copy())

    if t_max not in snapshots:
        snapshots[t_max] = (x.copy(), s.copy())
    return snapshots
