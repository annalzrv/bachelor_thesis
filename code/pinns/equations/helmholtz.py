"""1D Helmholtz with parametric wavenumber — second PDE family for LC-PINN.

    u''(x) + k²·u(x) = f(x; k),    x ∈ (0, 1),   u(0) = u(1) = 0,
    k ∈ [K_MIN, K_MAX].

Manufactured solution (chosen so Dirichlet BC holds for every k automatically):

    u_ref(x; k) = sin(π x) · cos(k x).

Forcing computed analytically:
    u_x   =  π cos(πx) cos(kx) − k sin(πx) sin(kx)
    u_xx  = −π² sin(πx) cos(kx) − 2πk cos(πx) sin(kx) − k² sin(πx) cos(kx)
    f     = u_xx + k² u
          = −π² sin(πx) cos(kx) − 2πk cos(πx) sin(kx).

The LC-PINN here conditions on the **physical parameter k** (not loss weights).
Network input is (x, k_norm) where k_norm = 2(k − K_MIN)/(K_MAX − K_MIN) − 1
∈ [−1, 1]. So `dim_lambda = 1` for the model, and the training loop samples
k uniformly each step rather than running through `LambdaSampler` (which is
built for loss-weight conditioning, a different concern).

Loss term naming follows the Burgers convention so the same Adam loop
infrastructure (e.g. tqdm progress, Adam + cosine LR) can be reused with
fixed loss weights.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------

X_MIN: float = 0.0
X_MAX: float = 1.0

K_MIN: float = 1.0
K_MAX: float = 10.0

DIM_PHYS: int = 1     # network coords: just x
DIM_LAMBDA: int = 1   # network conditions on k_norm only


# ---------------------------------------------------------------------------
# Manufactured ground truth & forcing
# ---------------------------------------------------------------------------

def k_to_norm(k: float | np.ndarray | torch.Tensor):
    """Map k ∈ [K_MIN, K_MAX] to k_norm ∈ [-1, 1]."""
    return 2.0 * (k - K_MIN) / (K_MAX - K_MIN) - 1.0


def norm_to_k(k_norm: float | np.ndarray | torch.Tensor):
    """Inverse of k_to_norm."""
    return K_MIN + 0.5 * (K_MAX - K_MIN) * (k_norm + 1.0)


def reference_solution(x: np.ndarray, k: float) -> np.ndarray:
    """Manufactured u_ref(x; k) = sin(πx)·cos(kx)."""
    return np.sin(np.pi * x) * np.cos(k * x)


def forcing(x: np.ndarray | torch.Tensor, k: float | torch.Tensor):
    """f(x; k) = u'' + k²u for the manufactured solution."""
    if isinstance(x, torch.Tensor):
        pi = float(np.pi)
        return (
            -pi ** 2 * torch.sin(pi * x) * torch.cos(k * x)
            - 2.0 * pi * k * torch.cos(pi * x) * torch.sin(k * x)
        )
    return (
        -np.pi ** 2 * np.sin(np.pi * x) * np.cos(k * x)
        - 2.0 * np.pi * k * np.cos(np.pi * x) * np.sin(k * x)
    )


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def generate_training_data(
    n_pde: int = 2000,
    n_bc: int = 100,
    n_data: int = 200,
    n_data_k_values: int = 5,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Collocation points and (sparse interior) data points across k-range.

    PDE collocation points carry no k information by themselves — at each
    training step the script will pair them with a freshly-sampled k.
    Data points come pre-tagged with their k value (so we can compute the
    reference solution) and the corresponding k_norm for model conditioning.
    """
    rng = np.random.default_rng(seed)
    if isinstance(device, str):
        device = torch.device(device)

    # PDE collocation: just x in [0, 1]; k will be sampled per training step.
    x_pde = rng.uniform(X_MIN, X_MAX, n_pde).astype(np.float32)
    coords_pde = torch.tensor(x_pde[:, None], device=device)

    # BC collocation: x = 0 and x = 1 (k-independent target u = 0).
    x_bc = np.concatenate([
        np.full(n_bc // 2, X_MIN, dtype=np.float32),
        np.full(n_bc - n_bc // 2, X_MAX, dtype=np.float32),
    ])
    coords_bc = torch.tensor(x_bc[:, None], device=device)
    u_bc = torch.zeros(n_bc, 1, device=device)

    # Data points: a few k values from the family, sparse interior samples
    # so the network has a concrete "this is what the family looks like"
    # supervision signal at training time.
    k_vals = np.linspace(K_MIN, K_MAX, n_data_k_values, dtype=np.float32)
    pts_per_k = max(n_data // n_data_k_values, 5)
    x_list, k_list, u_list = [], [], []
    for k in k_vals:
        x_smp = rng.uniform(X_MIN, X_MAX, pts_per_k).astype(np.float32)
        x_list.append(x_smp)
        k_list.append(np.full(pts_per_k, k, dtype=np.float32))
        u_list.append(reference_solution(x_smp, float(k)).astype(np.float32))
    coords_data = torch.tensor(np.concatenate(x_list)[:, None], device=device)
    k_data = torch.tensor(np.concatenate(k_list), device=device)
    u_data = torch.tensor(np.concatenate(u_list)[:, None], device=device)

    return {
        "coords_pde": coords_pde,            # (n_pde, 1)
        "coords_bc":  coords_bc,             # (n_bc, 1)
        "u_bc":       u_bc,                  # (n_bc, 1)
        "coords_data": coords_data,          # (n_d, 1)
        "k_data":      k_data,               # (n_d,)  per-point k value
        "u_data":      u_data,               # (n_d, 1)
    }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def compute_losses(
    model: torch.nn.Module,
    k_norm: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """LC-style losses at a single sampled k (passed as 1-vector k_norm).

    PDE & BC are evaluated at the same k. Data term uses the pre-tagged k
    of each data point — different k_norms per point — so the model must
    actually learn the parametric family, not just one k.
    """
    k_val = norm_to_k(k_norm.item() if k_norm.numel() == 1 else k_norm[0].item())

    # PDE residual at sampled k
    coords = batch["coords_pde"].requires_grad_(True)
    u = model(coords, k_norm)
    grads = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_xx = torch.autograd.grad(u_x, coords, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    f = forcing(coords[:, 0:1], k_val)
    residual = u_xx + (k_val ** 2) * u - f
    L_pde = torch.mean(residual ** 2)

    # BC at sampled k (target is 0 regardless of k)
    L_bc = torch.mean((model(batch["coords_bc"], k_norm) - batch["u_bc"]) ** 2)

    # Data: each point has its own k; loop or batched call. Batched per-point
    # k_norm passes a (B, 1) lambda to the model.
    k_data = batch["k_data"]
    k_norm_per_point = k_to_norm(k_data).unsqueeze(-1)            # (n_d, 1)
    u_pred_data = model(batch["coords_data"], k_norm_per_point)
    L_data = torch.mean((u_pred_data - batch["u_data"]) ** 2)

    return {"pde": L_pde, "bc": L_bc, "data": L_data}


def compute_losses_fixed(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    k_fixed: float,
) -> dict[str, torch.Tensor]:
    """Same three terms for a single-k baseline PINN (no LC conditioning).

    The data term here is restricted to the points whose k matches k_fixed.
    """
    coords = batch["coords_pde"].requires_grad_(True)
    u = model(coords)
    grads = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_xx = torch.autograd.grad(u_x, coords, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    f = forcing(coords[:, 0:1], k_fixed)
    residual = u_xx + (k_fixed ** 2) * u - f
    L_pde = torch.mean(residual ** 2)

    L_bc = torch.mean((model(batch["coords_bc"]) - batch["u_bc"]) ** 2)

    # Data: only the points sampled at this fixed k
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
    x_pts: np.ndarray,
    device: torch.device,
    is_lc: bool = True,
) -> np.ndarray:
    """Predict u(x; k). For LC: passes k_norm as the conditioning vector."""
    x_t = torch.tensor(x_pts, dtype=torch.float32, device=device).unsqueeze(1)
    if is_lc:
        k_norm = torch.tensor([k_to_norm(k)], dtype=torch.float32, device=device)
        return model(x_t, k_norm).cpu().numpy().flatten()
    return model(x_t).cpu().numpy().flatten()


def evaluate_at_k(
    model: torch.nn.Module,
    k: float,
    device: torch.device,
    nx: int = 256,
    is_lc: bool = True,
) -> float:
    """Relative L² error of u_pred(·; k) vs the manufactured u_ref(·; k)."""
    x_pts = np.linspace(X_MIN, X_MAX, nx, dtype=np.float32)
    u_ref = reference_solution(x_pts, k).astype(np.float32)
    u_pred = predict_solution(model, k, x_pts, device, is_lc=is_lc)
    return relative_l2(u_pred, u_ref)


def evaluate_grid(
    model: torch.nn.Module,
    k_values: list[float] | np.ndarray,
    device: torch.device,
    nx: int = 256,
    is_lc: bool = True,
) -> dict[float, float]:
    """rel-L² at each k ∈ k_values."""
    return {float(k): evaluate_at_k(model, float(k), device, nx=nx, is_lc=is_lc)
            for k in k_values}


def relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-10))
