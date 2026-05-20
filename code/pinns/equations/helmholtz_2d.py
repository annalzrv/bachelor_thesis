"""2D Helmholtz with parametric wavenumber.

    Δu(x,y) + k²·u(x,y) = f(x,y; k),    (x,y) ∈ (0,1)²,
    u = 0 on ∂Ω,                        k ∈ [K_MIN, K_MAX].

Manufactured solution (Dirichlet BC holds automatically on all four edges):

    u_ref(x, y; k) = sin(πx) · sin(πy) · cos(kx) · cos(ky).

Forcing computed in closed form. Letting a(x;k) = sin(πx)cos(kx) and
b(y;k) = sin(πy)cos(ky), so u = a·b:

    a''(x;k) = -(π² + k²) sin(πx)cos(kx) - 2πk cos(πx)sin(kx)
    b''(y;k) = -(π² + k²) sin(πy)cos(ky) - 2πk cos(πy)sin(ky)
    Δu = a''·b + a·b''
    f  = Δu + k² u
       = -(2π² + k²) u
         - 2πk · [cos(πx)sin(kx) sin(πy)cos(ky)
                  + sin(πx)cos(kx) cos(πy)sin(ky)] .
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------

X_MIN: float = 0.0
X_MAX: float = 1.0
Y_MIN: float = 0.0
Y_MAX: float = 1.0

K_MIN: float = 1.0
K_MAX: float = 10.0

DIM_PHYS: int = 2     # network coords: (x, y)
DIM_LAMBDA: int = 1   # network conditions on k_norm only


# ---------------------------------------------------------------------------
# Manufactured ground truth & forcing
# ---------------------------------------------------------------------------

def k_to_norm(k: float | np.ndarray | torch.Tensor):
    return 2.0 * (k - K_MIN) / (K_MAX - K_MIN) - 1.0


def norm_to_k(k_norm: float | np.ndarray | torch.Tensor):
    return K_MIN + 0.5 * (K_MAX - K_MIN) * (k_norm + 1.0)


def reference_solution(x: np.ndarray, y: np.ndarray, k: float) -> np.ndarray:
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(k * x) * np.cos(k * y)


def forcing(x, y, k):
    """f(x, y; k) = Δu + k² u for the manufactured solution."""
    if isinstance(x, torch.Tensor):
        pi = float(np.pi)
        sx, cx = torch.sin(pi * x), torch.cos(pi * x)
        sy, cy = torch.sin(pi * y), torch.cos(pi * y)
        skx, ckx = torch.sin(k * x), torch.cos(k * x)
        sky, cky = torch.sin(k * y), torch.cos(k * y)
        u = sx * sy * ckx * cky
        cross = cx * skx * sy * cky + sx * ckx * cy * sky
        return -(2.0 * pi * pi + k ** 2) * u - 2.0 * pi * k * cross
    sx, cx = np.sin(np.pi * x), np.cos(np.pi * x)
    sy, cy = np.sin(np.pi * y), np.cos(np.pi * y)
    skx, ckx = np.sin(k * x), np.cos(k * x)
    sky, cky = np.sin(k * y), np.cos(k * y)
    u = sx * sy * ckx * cky
    cross = cx * skx * sy * cky + sx * ckx * cy * sky
    return -(2.0 * np.pi ** 2 + k ** 2) * u - 2.0 * np.pi * k * cross


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def generate_training_data(
    n_pde: int = 4096,
    n_bc: int = 400,
    n_data: int = 500,
    n_data_k_values: int = 5,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Collocation in unit square; BC sampled on 4 edges; sparse data tagged with k."""
    rng = np.random.default_rng(seed)
    if isinstance(device, str):
        device = torch.device(device)

    # PDE collocation: uniform on (0,1)²
    xy_pde = rng.uniform(low=[X_MIN, Y_MIN], high=[X_MAX, Y_MAX],
                         size=(n_pde, 2)).astype(np.float32)
    coords_pde = torch.tensor(xy_pde, device=device)

    # BC: split equally across 4 edges
    n_per_edge = n_bc // 4
    edges = []
    # x = X_MIN
    t = rng.uniform(Y_MIN, Y_MAX, n_per_edge).astype(np.float32)
    edges.append(np.stack([np.full(n_per_edge, X_MIN, dtype=np.float32), t], axis=1))
    # x = X_MAX
    t = rng.uniform(Y_MIN, Y_MAX, n_per_edge).astype(np.float32)
    edges.append(np.stack([np.full(n_per_edge, X_MAX, dtype=np.float32), t], axis=1))
    # y = Y_MIN
    t = rng.uniform(X_MIN, X_MAX, n_per_edge).astype(np.float32)
    edges.append(np.stack([t, np.full(n_per_edge, Y_MIN, dtype=np.float32)], axis=1))
    # y = Y_MAX
    rest = n_bc - 3 * n_per_edge
    t = rng.uniform(X_MIN, X_MAX, rest).astype(np.float32)
    edges.append(np.stack([t, np.full(rest, Y_MAX, dtype=np.float32)], axis=1))
    coords_bc = torch.tensor(np.concatenate(edges, axis=0), device=device)
    u_bc = torch.zeros(coords_bc.shape[0], 1, device=device)

    # Data: sparse interior samples at a few k values (gives the network a
    # concrete supervision signal anchoring the family).
    k_vals = np.linspace(K_MIN, K_MAX, n_data_k_values, dtype=np.float32)
    pts_per_k = max(n_data // n_data_k_values, 5)
    xy_list, k_list, u_list = [], [], []
    for k in k_vals:
        xy = rng.uniform(low=[X_MIN, Y_MIN], high=[X_MAX, Y_MAX],
                         size=(pts_per_k, 2)).astype(np.float32)
        u = reference_solution(xy[:, 0], xy[:, 1], float(k)).astype(np.float32)
        xy_list.append(xy)
        k_list.append(np.full(pts_per_k, k, dtype=np.float32))
        u_list.append(u)
    coords_data = torch.tensor(np.concatenate(xy_list, axis=0), device=device)
    k_data = torch.tensor(np.concatenate(k_list), device=device)
    u_data = torch.tensor(np.concatenate(u_list)[:, None], device=device)

    return {
        "coords_pde": coords_pde,
        "coords_bc":  coords_bc,
        "u_bc":       u_bc,
        "coords_data": coords_data,
        "k_data":      k_data,
        "u_data":      u_data,
    }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _laplacian_2d(u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Δu = u_xx + u_yy via two autograd passes."""
    grads = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, coords, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, coords, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    return u_xx + u_yy


def compute_losses(
    model: torch.nn.Module,
    k_norm: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    k_val = norm_to_k(k_norm.item() if k_norm.numel() == 1 else k_norm[0].item())

    coords = batch["coords_pde"].requires_grad_(True)
    u = model(coords, k_norm)
    lap = _laplacian_2d(u, coords)
    f = forcing(coords[:, 0:1], coords[:, 1:2], k_val)
    residual = lap + (k_val ** 2) * u - f
    L_pde = torch.mean(residual ** 2)

    L_bc = torch.mean((model(batch["coords_bc"], k_norm) - batch["u_bc"]) ** 2)

    k_data = batch["k_data"]
    k_norm_per_point = k_to_norm(k_data).unsqueeze(-1)
    u_pred_data = model(batch["coords_data"], k_norm_per_point)
    L_data = torch.mean((u_pred_data - batch["u_data"]) ** 2)

    return {"pde": L_pde, "bc": L_bc, "data": L_data}


def compute_losses_fixed(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    k_fixed: float,
) -> dict[str, torch.Tensor]:
    coords = batch["coords_pde"].requires_grad_(True)
    u = model(coords)
    lap = _laplacian_2d(u, coords)
    f = forcing(coords[:, 0:1], coords[:, 1:2], k_fixed)
    residual = lap + (k_fixed ** 2) * u - f
    L_pde = torch.mean(residual ** 2)

    L_bc = torch.mean((model(batch["coords_bc"]) - batch["u_bc"]) ** 2)

    mask = torch.isclose(batch["k_data"], torch.tensor(k_fixed, device=batch["k_data"].device))
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
    k: float,
    xy_pts: np.ndarray,
    device: torch.device,
    is_lc: bool = True,
) -> np.ndarray:
    xy_t = torch.tensor(xy_pts, dtype=torch.float32, device=device)
    if is_lc:
        k_norm = torch.tensor([k_to_norm(k)], dtype=torch.float32, device=device)
        return model(xy_t, k_norm).cpu().numpy().flatten()
    return model(xy_t).cpu().numpy().flatten()


def evaluate_at_k(
    model: torch.nn.Module,
    k: float,
    device: torch.device,
    nx: int = 64,
    is_lc: bool = True,
) -> float:
    x = np.linspace(X_MIN, X_MAX, nx, dtype=np.float32)
    y = np.linspace(Y_MIN, Y_MAX, nx, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")
    xy = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    u_ref = reference_solution(xy[:, 0], xy[:, 1], k).astype(np.float32)
    u_pred = predict_solution(model, k, xy, device, is_lc=is_lc)
    return relative_l2(u_pred, u_ref)


def evaluate_grid(
    model: torch.nn.Module,
    k_values: list[float] | np.ndarray,
    device: torch.device,
    nx: int = 64,
    is_lc: bool = True,
) -> dict[float, float]:
    return {float(k): evaluate_at_k(model, float(k), device, nx=nx, is_lc=is_lc)
            for k in k_values}


def relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-10))
