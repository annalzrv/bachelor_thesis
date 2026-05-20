"""2D incompressible Navier-Stokes — Taylor-Green vortex benchmark.

Governing equations on Ω = [0, 2π]² × [0, T_MAX]:
    u_t + u·u_x + v·u_y = -p_x + ν·(u_xx + u_yy)
    v_t + u·v_x + v·v_y = -p_y + ν·(v_xx + v_yy)
    u_x + v_y = 0                                         (continuity)

Analytic Taylor-Green vortex solution (with periodic BC on [0, 2π]²):
    u(x, y, t) = -cos(x)·sin(y)·exp(-2 ν t)
    v(x, y, t) =  sin(x)·cos(y)·exp(-2 ν t)
    p(x, y, t) = -(1/4)·(cos(2x) + cos(2y))·exp(-4 ν t)

The network outputs (u, v, p) simultaneously.  Loss terms:
    pde  — continuity + x-momentum + y-momentum residuals (averaged)
    ic   — match analytic IC at t=0
    bc   — periodic boundary via |u(0,y,t) - u(2π,y,t)|² + |v(0,y,t) - v(2π,y,t)|²
                                 + |u(x,0,t) - u(x,2π,t)|² + |v(x,0,t) - v(x,2π,t)|²
    data — sparse interior measurements of (u, v) from the analytic solution

λ order for LC-PINN: [pde, bc, ic, data].
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------

NU: float = 0.05                    # viscosity (moderate Re)
X_MIN: float = 0.0
X_MAX: float = 2.0 * float(np.pi)
Y_MIN: float = 0.0
Y_MAX: float = 2.0 * float(np.pi)
T_MIN: float = 0.0
T_MAX: float = 1.0

DIM_PHYS = 3     # inputs:  (x, y, t)
DIM_LAMBDA = 4   # loss terms: pde, bc, ic, data
DIM_OUT = 3      # outputs: (u, v, p)


# ---------------------------------------------------------------------------
# Analytic solution
# ---------------------------------------------------------------------------

def analytic_uvp(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray | float,
    nu: float = NU,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    decay_u = np.exp(-2.0 * nu * np.asarray(t))
    decay_p = np.exp(-4.0 * nu * np.asarray(t))
    u = -np.cos(x) * np.sin(y) * decay_u
    v =  np.sin(x) * np.cos(y) * decay_u
    p = -0.25 * (np.cos(2.0 * x) + np.cos(2.0 * y)) * decay_p
    return u, v, p


def compute_reference_solution(
    nx: int = 64,
    ny: int = 64,
    snap_times: list[float] | None = None,
) -> dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns {t_val: (X, Y, u, v, p)} on a regular nx×ny grid for each snapshot.
    X, Y are 2D meshgrids; u, v, p are 2D arrays of the same shape.
    """
    if snap_times is None:
        snap_times = [0.0, 0.25, 0.5, 1.0]

    x = np.linspace(X_MIN, X_MAX, nx, endpoint=False)
    y = np.linspace(Y_MIN, Y_MAX, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    snaps: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for t_val in sorted(snap_times):
        u, v, p = analytic_uvp(X, Y, t_val)
        snaps[float(t_val)] = (X.copy(), Y.copy(), u, v, p)
    return snaps


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def generate_training_data(
    n_pde: int = 4000,
    n_bc: int = 400,
    n_ic: int = 400,
    n_data: int = 200,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Returns:
      coords_pde:  (n_pde, 3)     random interior (x, y, t)
      coords_bc_left / _right / _bottom / _top: (n_bc, 3) matched periodic pairs
      coords_ic:   (n_ic, 3)      with t=0
      u_ic, v_ic:  (n_ic, 1)
      coords_data: (n_data, 3)
      uv_data:     (n_data, 2)
    """
    rng = np.random.default_rng(seed)
    if isinstance(device, str):
        device = torch.device(device)

    def to(arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype, device=device)

    # PDE collocation
    x_pde = rng.uniform(X_MIN, X_MAX, n_pde).astype(np.float32)
    y_pde = rng.uniform(Y_MIN, Y_MAX, n_pde).astype(np.float32)
    t_pde = rng.uniform(T_MIN, T_MAX, n_pde).astype(np.float32)
    coords_pde = to(np.column_stack([x_pde, y_pde, t_pde]))

    # Periodic BC pairs: same (y, t) at x=0 and x=2π; same (x, t) at y=0 and y=2π
    n_half = n_bc // 2
    # x pair
    y_x_pair = rng.uniform(Y_MIN, Y_MAX, n_half).astype(np.float32)
    t_x_pair = rng.uniform(T_MIN, T_MAX, n_half).astype(np.float32)
    coords_bc_left  = to(np.column_stack([np.full(n_half, X_MIN, np.float32), y_x_pair, t_x_pair]))
    coords_bc_right = to(np.column_stack([np.full(n_half, X_MAX, np.float32), y_x_pair, t_x_pair]))
    # y pair
    x_y_pair = rng.uniform(X_MIN, X_MAX, n_half).astype(np.float32)
    t_y_pair = rng.uniform(T_MIN, T_MAX, n_half).astype(np.float32)
    coords_bc_bot = to(np.column_stack([x_y_pair, np.full(n_half, Y_MIN, np.float32), t_y_pair]))
    coords_bc_top = to(np.column_stack([x_y_pair, np.full(n_half, Y_MAX, np.float32), t_y_pair]))

    # IC at t=0
    x_ic = rng.uniform(X_MIN, X_MAX, n_ic).astype(np.float32)
    y_ic = rng.uniform(Y_MIN, Y_MAX, n_ic).astype(np.float32)
    t_ic = np.zeros(n_ic, dtype=np.float32)
    u_ic, v_ic, _ = analytic_uvp(x_ic, y_ic, 0.0)
    coords_ic = to(np.column_stack([x_ic, y_ic, t_ic]))
    u_ic_t = to(u_ic).unsqueeze(1)
    v_ic_t = to(v_ic).unsqueeze(1)

    # Sparse data: random (x, y, t) with analytic (u, v)
    x_d = rng.uniform(X_MIN, X_MAX, n_data).astype(np.float32)
    y_d = rng.uniform(Y_MIN, Y_MAX, n_data).astype(np.float32)
    t_d = rng.uniform(T_MIN, T_MAX, n_data).astype(np.float32)
    u_d, v_d, _ = analytic_uvp(x_d, y_d, t_d)
    coords_data = to(np.column_stack([x_d, y_d, t_d]))
    uv_data = to(np.column_stack([u_d, v_d]))

    return {
        "coords_pde":       coords_pde,
        "coords_bc_left":   coords_bc_left,
        "coords_bc_right":  coords_bc_right,
        "coords_bc_bot":    coords_bc_bot,
        "coords_bc_top":    coords_bc_top,
        "coords_ic":        coords_ic,
        "u_ic":             u_ic_t,
        "v_ic":             v_ic_t,
        "coords_data":      coords_data,
        "uv_data":          uv_data,
    }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _pde_residual(
    out: torch.Tensor,
    coords: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Continuity and momentum residuals from a (B, 3) network output (u, v, p)."""
    u = out[:, 0:1]
    v = out[:, 1:2]
    p = out[:, 2:3]
    ones = torch.ones_like(u)

    grads_u = torch.autograd.grad(u, coords, ones, create_graph=True)[0]
    grads_v = torch.autograd.grad(v, coords, ones, create_graph=True)[0]
    grads_p = torch.autograd.grad(p, coords, ones, create_graph=True)[0]

    u_x, u_y, u_t = grads_u[:, 0:1], grads_u[:, 1:2], grads_u[:, 2:3]
    v_x, v_y, v_t = grads_v[:, 0:1], grads_v[:, 1:2], grads_v[:, 2:3]
    p_x, p_y      = grads_p[:, 0:1], grads_p[:, 1:2]

    u_xx = torch.autograd.grad(u_x, coords, ones, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, coords, ones, create_graph=True)[0][:, 1:2]
    v_xx = torch.autograd.grad(v_x, coords, ones, create_graph=True)[0][:, 0:1]
    v_yy = torch.autograd.grad(v_y, coords, ones, create_graph=True)[0][:, 1:2]

    r_cont = u_x + v_y
    r_mom_x = u_t + u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
    r_mom_y = v_t + u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)
    return r_cont, r_mom_x, r_mom_y


def _forward_lc(model, coords, log_lambda):
    return model(coords, log_lambda)


def _forward_fx(model, coords, _):
    return model(coords)


def _compute_losses_core(
    model: torch.nn.Module,
    forward,
    log_lambda: torch.Tensor | None,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    coords_pde = batch["coords_pde"].detach().clone().requires_grad_(True)
    out_pde = forward(model, coords_pde, log_lambda)
    r_cont, r_mx, r_my = _pde_residual(out_pde, coords_pde)
    L_pde = (r_cont ** 2).mean() + (r_mx ** 2).mean() + (r_my ** 2).mean()

    out_left  = forward(model, batch["coords_bc_left"],  log_lambda)
    out_right = forward(model, batch["coords_bc_right"], log_lambda)
    out_bot   = forward(model, batch["coords_bc_bot"],   log_lambda)
    out_top   = forward(model, batch["coords_bc_top"],   log_lambda)
    # Match u and v across periodic pairs (skip p — only defined up to a constant)
    L_bc = (
        ((out_left[:, :2] - out_right[:, :2]) ** 2).mean()
        + ((out_bot[:, :2] - out_top[:, :2]) ** 2).mean()
    )

    out_ic = forward(model, batch["coords_ic"], log_lambda)
    L_ic = ((out_ic[:, 0:1] - batch["u_ic"]) ** 2).mean() + ((out_ic[:, 1:2] - batch["v_ic"]) ** 2).mean()

    out_data = forward(model, batch["coords_data"], log_lambda)
    L_data = ((out_data[:, :2] - batch["uv_data"]) ** 2).mean()

    return {"pde": L_pde, "bc": L_bc, "ic": L_ic, "data": L_data}


def compute_losses(
    model: torch.nn.Module,
    log_lambda: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return _compute_losses_core(model, _forward_lc, log_lambda, batch)


def compute_losses_fixed(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return _compute_losses_core(model, _forward_fx, None, batch)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_uvp(
    model: torch.nn.Module,
    log_lambda: torch.Tensor | None,
    X: np.ndarray,
    Y: np.ndarray,
    t_val: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict (u, v, p) on a 2D grid at time t_val."""
    flat_x = X.reshape(-1).astype(np.float32)
    flat_y = Y.reshape(-1).astype(np.float32)
    flat_t = np.full_like(flat_x, t_val, dtype=np.float32)
    coords = torch.tensor(np.column_stack([flat_x, flat_y, flat_t]), device=device)
    out = model(coords, log_lambda) if log_lambda is not None else model(coords)
    out = out.cpu().numpy()
    u = out[:, 0].reshape(X.shape)
    v = out[:, 1].reshape(X.shape)
    p = out[:, 2].reshape(X.shape)
    return u, v, p


def relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-10))


def evaluate(
    model: torch.nn.Module,
    log_lambda: torch.Tensor | None,
    ref_snapshots: dict,
    device: torch.device,
) -> dict[float, dict[str, float]]:
    """Per-snapshot rel-L2 of u, v (pressure is only defined up to a constant)."""
    out: dict[float, dict[str, float]] = {}
    for t_val, (X, Y, u_ref, v_ref, _) in sorted(ref_snapshots.items()):
        u_pred, v_pred, _ = predict_uvp(model, log_lambda, X, Y, t_val, device)
        out[t_val] = {
            "u": relative_l2(u_pred, u_ref),
            "v": relative_l2(v_pred, v_ref),
        }
    return out


def mean_rel_l2(per_snap: dict[float, dict[str, float]]) -> float:
    """Average of u and v rel-L2 across all snapshots."""
    vals = []
    for _, d in per_snap.items():
        vals.extend(d.values())
    return float(np.mean(vals))
