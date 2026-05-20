"""3D Helmholtz with parametric wavenumber.

    Δu(x,y,z) + k²·u(x,y,z) = f(x,y,z; k),    (x,y,z) ∈ (0,1)³,
    u = 0 on ∂Ω,                              k ∈ [K_MIN, K_MAX].

Manufactured solution (Dirichlet BC holds automatically on all six faces):

    u_ref(x,y,z; k) = sin(πx)sin(πy)sin(πz)·cos(kx)cos(ky)cos(kz).

Letting a(x;k)=sin(πx)cos(kx), b(y;k)=sin(πy)cos(ky), c(z;k)=sin(πz)cos(kz),
so u=abc:

    a''(x;k) = -(π²+k²)·a - 2πk·cos(πx)sin(kx)
    Δu = a''bc + ab''c + abc''
       = -3(π²+k²)·u - 2πk·[A* bc + a B* c + ab C*]
    f  = Δu + k²·u = -(3π² + 2k²)·u - 2πk·[A* bc + a B* c + ab C*]

where A* = cos(πx)sin(kx), B* = cos(πy)sin(ky), C* = cos(πz)sin(kz).

K range defaults to [1,5] — narrower than 2D's [1,10] to keep the 3D
forcing tractable for PINNs. K_MAX can be raised at config time.
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
Z_MIN: float = 0.0
Z_MAX: float = 1.0

K_MIN: float = 1.0
K_MAX: float = 5.0

DIM_PHYS: int = 3     # network coords: (x, y, z)
DIM_LAMBDA: int = 1   # network conditions on k_norm only


# ---------------------------------------------------------------------------
# Manufactured ground truth & forcing
# ---------------------------------------------------------------------------

def k_to_norm(k):
    return 2.0 * (k - K_MIN) / (K_MAX - K_MIN) - 1.0


def norm_to_k(k_norm):
    return K_MIN + 0.5 * (K_MAX - K_MIN) * (k_norm + 1.0)


def reference_solution(x, y, z, k: float):
    return (np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
            * np.cos(k * x) * np.cos(k * y) * np.cos(k * z))


def forcing(x, y, z, k):
    """f = Δu + k² u for the manufactured solution."""
    if isinstance(x, torch.Tensor):
        pi = float(np.pi)
        sx, cx = torch.sin(pi * x), torch.cos(pi * x)
        sy, cy = torch.sin(pi * y), torch.cos(pi * y)
        sz, cz = torch.sin(pi * z), torch.cos(pi * z)
        skx, ckx = torch.sin(k * x), torch.cos(k * x)
        sky, cky = torch.sin(k * y), torch.cos(k * y)
        skz, ckz = torch.sin(k * z), torch.cos(k * z)
        u = sx * sy * sz * ckx * cky * ckz
        cross = (cx * skx * sy * cky * sz * ckz
                 + sx * ckx * cy * sky * sz * ckz
                 + sx * ckx * sy * cky * cz * skz)
        return -(3.0 * pi * pi + 2.0 * k ** 2) * u - 2.0 * pi * k * cross
    sx, cx = np.sin(np.pi * x), np.cos(np.pi * x)
    sy, cy = np.sin(np.pi * y), np.cos(np.pi * y)
    sz, cz = np.sin(np.pi * z), np.cos(np.pi * z)
    skx, ckx = np.sin(k * x), np.cos(k * x)
    sky, cky = np.sin(k * y), np.cos(k * y)
    skz, ckz = np.sin(k * z), np.cos(k * z)
    u = sx * sy * sz * ckx * cky * ckz
    cross = (cx * skx * sy * cky * sz * ckz
             + sx * ckx * cy * sky * sz * ckz
             + sx * ckx * sy * cky * cz * skz)
    return -(3.0 * np.pi ** 2 + 2.0 * k ** 2) * u - 2.0 * np.pi * k * cross


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def generate_training_data(
    n_pde: int = 8192,
    n_bc: int = 600,
    n_data: int = 600,
    n_data_k_values: int = 5,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Collocation in unit cube; BC sampled on 6 faces; sparse data tagged with k."""
    rng = np.random.default_rng(seed)
    if isinstance(device, str):
        device = torch.device(device)

    xyz_pde = rng.uniform(low=[X_MIN, Y_MIN, Z_MIN], high=[X_MAX, Y_MAX, Z_MAX],
                          size=(n_pde, 3)).astype(np.float32)
    coords_pde = torch.tensor(xyz_pde, device=device)

    n_per_face = n_bc // 6
    faces = []
    for axis in range(3):
        for val in (0.0, 1.0):
            other = [i for i in range(3) if i != axis]
            samples = rng.uniform(0.0, 1.0, size=(n_per_face, 2)).astype(np.float32)
            face = np.zeros((n_per_face, 3), dtype=np.float32)
            face[:, axis] = val
            face[:, other[0]] = samples[:, 0]
            face[:, other[1]] = samples[:, 1]
            faces.append(face)
    coords_bc = torch.tensor(np.concatenate(faces, axis=0), device=device)
    u_bc = torch.zeros(coords_bc.shape[0], 1, device=device)

    k_vals = np.linspace(K_MIN, K_MAX, n_data_k_values, dtype=np.float32)
    pts_per_k = max(n_data // n_data_k_values, 5)
    xyz_list, k_list, u_list = [], [], []
    for k in k_vals:
        xyz = rng.uniform(low=[X_MIN, Y_MIN, Z_MIN], high=[X_MAX, Y_MAX, Z_MAX],
                          size=(pts_per_k, 3)).astype(np.float32)
        u = reference_solution(xyz[:, 0], xyz[:, 1], xyz[:, 2], float(k)).astype(np.float32)
        xyz_list.append(xyz)
        k_list.append(np.full(pts_per_k, k, dtype=np.float32))
        u_list.append(u)
    coords_data = torch.tensor(np.concatenate(xyz_list, axis=0), device=device)
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

def _laplacian_3d(u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Δu = u_xx + u_yy + u_zz via two autograd passes."""
    grads = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_z = grads[:, 2:3]
    u_xx = torch.autograd.grad(u_x, coords, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, coords, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    u_zz = torch.autograd.grad(u_z, coords, torch.ones_like(u_z), create_graph=True)[0][:, 2:3]
    return u_xx + u_yy + u_zz


def compute_losses(
    model: torch.nn.Module,
    k_norm: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    k_val = norm_to_k(k_norm.item() if k_norm.numel() == 1 else k_norm[0].item())

    coords = batch["coords_pde"].requires_grad_(True)
    u = model(coords, k_norm)
    lap = _laplacian_3d(u, coords)
    f = forcing(coords[:, 0:1], coords[:, 1:2], coords[:, 2:3], k_val)
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
    lap = _laplacian_3d(u, coords)
    f = forcing(coords[:, 0:1], coords[:, 1:2], coords[:, 2:3], k_fixed)
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
    xyz_pts: np.ndarray,
    device: torch.device,
    is_lc: bool = True,
) -> np.ndarray:
    xyz_t = torch.tensor(xyz_pts, dtype=torch.float32, device=device)
    if is_lc:
        k_norm = torch.tensor([k_to_norm(k)], dtype=torch.float32, device=device)
        return model(xyz_t, k_norm).cpu().numpy().flatten()
    return model(xyz_t).cpu().numpy().flatten()


def evaluate_at_k(
    model: torch.nn.Module,
    k: float,
    device: torch.device,
    nx: int = 32,
    is_lc: bool = True,
) -> float:
    x = np.linspace(X_MIN, X_MAX, nx, dtype=np.float32)
    y = np.linspace(Y_MIN, Y_MAX, nx, dtype=np.float32)
    z = np.linspace(Z_MIN, Z_MAX, nx, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    xyz = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
    u_ref = reference_solution(xyz[:, 0], xyz[:, 1], xyz[:, 2], k).astype(np.float32)
    u_pred = predict_solution(model, k, xyz, device, is_lc=is_lc)
    return relative_l2(u_pred, u_ref)


def evaluate_grid(
    model: torch.nn.Module,
    k_values,
    device: torch.device,
    nx: int = 32,
    is_lc: bool = True,
) -> dict[float, float]:
    return {float(k): evaluate_at_k(model, float(k), device, nx=nx, is_lc=is_lc)
            for k in k_values}


def relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-10))
