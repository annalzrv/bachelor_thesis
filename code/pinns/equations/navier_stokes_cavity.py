"""Steady 2D lid-driven cavity — canonical NS benchmark.

Governing equations on Ω = [0, 1]² (steady, incompressible):
    u·u_x + v·u_y = -p_x + ν·(u_xx + u_yy)         (x-momentum)
    u·v_x + v·v_y = -p_y + ν·(v_xx + v_yy)         (y-momentum)
    u_x + v_y = 0                                   (continuity)

Boundary conditions (driven lid on top):
    top    y=1:  u=1,  v=0
    bottom y=0:  u=0,  v=0
    left   x=0:  u=0,  v=0
    right  x=1:  u=0,  v=0

Reference: Ghia, Ghia, Shin (1982), "High-Re solutions for incompressible
flow using the Navier-Stokes equations and a multigrid method",
J. Comput. Phys. 48(3), 387-411. Centerline velocity tables at Re=400 are
hardcoded below from Table I / II of that paper.

The corners at y=1 have a BC jump (u goes from 1 on the lid to 0 on the side
walls). We sample top-wall BC coords away from the corners (x ∈ [0.02, 0.98])
to avoid forcing the network onto the singularity; this is standard PINN
practice for cavity.

λ order for LC-PINN: [pde, bc, data].  (No IC term — steady.)
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------

RE: float = 400.0
NU: float = 1.0 / RE
X_MIN: float = 0.0
X_MAX: float = 1.0
Y_MIN: float = 0.0
Y_MAX: float = 1.0

CORNER_EPS: float = 0.02  # skip this much near corners when sampling top-wall BC

DIM_PHYS = 2     # inputs:  (x, y)  — steady, no time
DIM_LAMBDA = 3   # loss terms: pde, bc, data
DIM_OUT = 3      # outputs: (u, v, p)


# ---------------------------------------------------------------------------
# Ghia et al. (1982) Re=400 reference — centerline velocity profiles
# ---------------------------------------------------------------------------

# u-velocity on vertical centerline x=0.5, at selected y values (Table I).
_GHIA_U_CENTERLINE_RE400 = np.array([
    [0.0000,  0.00000],
    [0.0547, -0.08186],
    [0.0625, -0.09266],
    [0.0703, -0.10338],
    [0.1016, -0.14612],
    [0.1719, -0.24299],
    [0.2813, -0.32726],
    [0.4531, -0.17119],
    [0.5000, -0.11477],
    [0.6172,  0.02135],
    [0.7344,  0.16256],
    [0.8516,  0.29093],
    [0.9531,  0.55892],
    [0.9609,  0.61756],
    [0.9688,  0.68439],
    [0.9766,  0.75837],
    [1.0000,  1.00000],
], dtype=np.float32)

# v-velocity on horizontal centerline y=0.5, at selected x values (Table II).
_GHIA_V_CENTERLINE_RE400 = np.array([
    [0.0000,  0.00000],
    [0.0625,  0.18360],
    [0.0703,  0.19713],
    [0.0781,  0.20920],
    [0.0938,  0.22965],
    [0.1563,  0.28124],
    [0.2266,  0.30203],
    [0.2344,  0.30174],
    [0.5000,  0.05186],
    [0.8047, -0.38598],
    [0.8594, -0.44993],
    [0.9063, -0.33827],
    [0.9453, -0.22847],
    [0.9531, -0.19254],
    [0.9609, -0.15663],
    [0.9688, -0.12146],
    [1.0000,  0.00000],
], dtype=np.float32)


def ghia_u_centerline() -> tuple[np.ndarray, np.ndarray]:
    """(y, u) pairs along x=0.5 from Ghia et al. 1982 Re=400."""
    return _GHIA_U_CENTERLINE_RE400[:, 0].copy(), _GHIA_U_CENTERLINE_RE400[:, 1].copy()


def ghia_v_centerline() -> tuple[np.ndarray, np.ndarray]:
    """(x, v) pairs along y=0.5 from Ghia et al. 1982 Re=400."""
    return _GHIA_V_CENTERLINE_RE400[:, 0].copy(), _GHIA_V_CENTERLINE_RE400[:, 1].copy()


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def generate_training_data(
    n_pde: int = 8000,
    n_bc_per_wall: int = 200,
    n_data_ghia: bool = True,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Returns a dict with interior collocation, per-wall BC coords+targets,
    and sparse data from the Ghia reference centerlines.

    Keys:
      coords_pde:  (n_pde, 2)
      coords_bc:   (4*n_bc_per_wall, 2)
      uv_bc:       (4*n_bc_per_wall, 2)         targets (u, v) at BC points
      coords_data: (n_data, 2)                  Ghia centerline points
      uv_data:     (n_data, 2)                  (u, v) targets from Ghia
    """
    rng = np.random.default_rng(seed)
    if isinstance(device, str):
        device = torch.device(device)

    def to(arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype, device=device)

    # Interior collocation
    x_pde = rng.uniform(X_MIN, X_MAX, n_pde).astype(np.float32)
    y_pde = rng.uniform(Y_MIN, Y_MAX, n_pde).astype(np.float32)
    coords_pde = to(np.column_stack([x_pde, y_pde]))

    # BC — four walls. Skip a small neighbourhood of the top corners to avoid
    # forcing the network onto the u=1 vs u=0 discontinuity.
    n = n_bc_per_wall
    # bottom: y=0, u=0, v=0
    xb = rng.uniform(X_MIN, X_MAX, n).astype(np.float32)
    yb = np.zeros(n, dtype=np.float32)
    # left: x=0, u=0, v=0
    xl = np.zeros(n, dtype=np.float32)
    yl = rng.uniform(Y_MIN, Y_MAX, n).astype(np.float32)
    # right: x=1, u=0, v=0
    xr = np.ones(n, dtype=np.float32)
    yr = rng.uniform(Y_MIN, Y_MAX, n).astype(np.float32)
    # top: y=1, u=1, v=0  (sample away from corners)
    xt = rng.uniform(X_MIN + CORNER_EPS, X_MAX - CORNER_EPS, n).astype(np.float32)
    yt = np.ones(n, dtype=np.float32)

    coords_bc = np.concatenate([
        np.column_stack([xb, yb]),
        np.column_stack([xl, yl]),
        np.column_stack([xr, yr]),
        np.column_stack([xt, yt]),
    ], axis=0).astype(np.float32)
    uv_bc = np.concatenate([
        np.zeros((n, 2), dtype=np.float32),           # bottom: u=0, v=0
        np.zeros((n, 2), dtype=np.float32),           # left:   u=0, v=0
        np.zeros((n, 2), dtype=np.float32),           # right:  u=0, v=0
        np.column_stack([np.ones(n, dtype=np.float32), np.zeros(n, dtype=np.float32)]),  # top: u=1, v=0
    ], axis=0)

    # Sparse data: Ghia centerlines
    if n_data_ghia:
        y_c, u_c = ghia_u_centerline()
        x_c, v_c = ghia_v_centerline()
        # vertical centerline (x=0.5): (0.5, y_c) with (u_c, v=?). We only supervise u.
        # We pair it with NaN for v and handle in the loss — simpler to just stack both
        # centerlines and supervise only the relevant component using a mask.
        coords_data_list = []
        target_list = []
        mask_list = []  # per-row [m_u, m_v]
        for yi, ui in zip(y_c, u_c):
            coords_data_list.append([0.5, yi])
            target_list.append([ui, 0.0])
            mask_list.append([1.0, 0.0])
        for xi, vi in zip(x_c, v_c):
            coords_data_list.append([xi, 0.5])
            target_list.append([0.0, vi])
            mask_list.append([0.0, 1.0])
        coords_data = np.array(coords_data_list, dtype=np.float32)
        uv_data = np.array(target_list, dtype=np.float32)
        uv_mask = np.array(mask_list, dtype=np.float32)
    else:
        coords_data = np.zeros((0, 2), dtype=np.float32)
        uv_data = np.zeros((0, 2), dtype=np.float32)
        uv_mask = np.zeros((0, 2), dtype=np.float32)

    return {
        "coords_pde": to(coords_pde) if isinstance(coords_pde, np.ndarray) else coords_pde,
        "coords_bc":  to(coords_bc),
        "uv_bc":      to(uv_bc),
        "coords_data": to(coords_data),
        "uv_data":    to(uv_data),
        "uv_mask":    to(uv_mask),
    }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _pde_residual(
    out: torch.Tensor,
    coords: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Steady continuity + x/y-momentum residuals from (B, 3) output (u, v, p)."""
    u = out[:, 0:1]
    v = out[:, 1:2]
    p = out[:, 2:3]
    ones = torch.ones_like(u)

    grads_u = torch.autograd.grad(u, coords, ones, create_graph=True)[0]
    grads_v = torch.autograd.grad(v, coords, ones, create_graph=True)[0]
    grads_p = torch.autograd.grad(p, coords, ones, create_graph=True)[0]

    u_x, u_y = grads_u[:, 0:1], grads_u[:, 1:2]
    v_x, v_y = grads_v[:, 0:1], grads_v[:, 1:2]
    p_x, p_y = grads_p[:, 0:1], grads_p[:, 1:2]

    u_xx = torch.autograd.grad(u_x, coords, ones, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, coords, ones, create_graph=True)[0][:, 1:2]
    v_xx = torch.autograd.grad(v_x, coords, ones, create_graph=True)[0][:, 0:1]
    v_yy = torch.autograd.grad(v_y, coords, ones, create_graph=True)[0][:, 1:2]

    r_cont = u_x + v_y
    r_mom_x = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
    r_mom_y = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)
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

    out_bc = forward(model, batch["coords_bc"], log_lambda)
    L_bc = ((out_bc[:, :2] - batch["uv_bc"]) ** 2).mean()

    out_data = forward(model, batch["coords_data"], log_lambda)
    sq = (out_data[:, :2] - batch["uv_data"]) ** 2
    mask = batch["uv_mask"]
    L_data = (sq * mask).sum() / mask.sum().clamp(min=1.0)

    return {"pde": L_pde, "bc": L_bc, "data": L_data}


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
# Evaluation: centerline profiles vs Ghia
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_uvp_grid(
    model: torch.nn.Module,
    log_lambda: torch.Tensor | None,
    nx: int = 129,
    ny: int = 129,
    device: torch.device = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict (u, v, p) on a regular nx×ny grid covering [0,1]²."""
    if device is None:
        device = next(model.parameters()).device
    x = np.linspace(X_MIN, X_MAX, nx, dtype=np.float32)
    y = np.linspace(Y_MIN, Y_MAX, ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")
    coords = torch.tensor(np.column_stack([X.reshape(-1), Y.reshape(-1)]), device=device)
    out = model(coords, log_lambda) if log_lambda is not None else model(coords)
    out = out.cpu().numpy()
    u = out[:, 0].reshape(X.shape)
    v = out[:, 1].reshape(X.shape)
    p = out[:, 2].reshape(X.shape)
    return X, Y, u, v, p


@torch.no_grad()
def predict_at_points(
    model: torch.nn.Module,
    log_lambda: torch.Tensor | None,
    pts: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """pts shape (N, 2). Returns (N, 3) array (u, v, p)."""
    coords = torch.tensor(pts.astype(np.float32), device=device)
    out = model(coords, log_lambda) if log_lambda is not None else model(coords)
    return out.cpu().numpy()


def relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-10))


def evaluate(
    model: torch.nn.Module,
    log_lambda: torch.Tensor | None,
    device: torch.device,
) -> dict[str, float]:
    """Rel-L2 of predicted centerlines vs Ghia Re=400 reference."""
    y_c, u_c_ref = ghia_u_centerline()
    x_c, v_c_ref = ghia_v_centerline()
    # vertical centerline x=0.5: predict u at (0.5, y_c)
    pts_u = np.column_stack([np.full_like(y_c, 0.5), y_c])
    uv_pred_u = predict_at_points(model, log_lambda, pts_u, device)
    u_pred = uv_pred_u[:, 0]
    # horizontal centerline y=0.5: predict v at (x_c, 0.5)
    pts_v = np.column_stack([x_c, np.full_like(x_c, 0.5)])
    uv_pred_v = predict_at_points(model, log_lambda, pts_v, device)
    v_pred = uv_pred_v[:, 1]
    return {
        "u_centerline_rel_l2": relative_l2(u_pred, u_c_ref),
        "v_centerline_rel_l2": relative_l2(v_pred, v_c_ref),
        "mean_rel_l2": 0.5 * (relative_l2(u_pred, u_c_ref) + relative_l2(v_pred, v_c_ref)),
    }
