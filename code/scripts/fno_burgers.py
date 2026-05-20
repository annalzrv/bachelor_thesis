"""FNO baseline on Burgers (Li et al. 2021, "Fourier Neural Operator for parametric PDEs").

Operator-learning angle differs from PINN baselines and from LC-PINN:

  - SA-PINN / ReLoBRaLo / Causal-PINN / vanilla-PINN
        learn one solution u(x,t) for one specific IC and one specific viscosity
        via residual minimisation; "amortise" only across loss-weight choices.

  - LC-PINN
        amortises across loss-weight space by conditioning the network on λ.

  - FNO (this file)
        learns the SOLUTION OPERATOR S: u(0,·) -> {u(t,·)}_{t in T} at fixed
        viscosity ν = 0.01/π. Training amortises across a *function-space*
        distribution of initial conditions; inference on our specific test IC
        u(0,x) = -sin(πx) is a single forward pass.

These are different amortisation axes. The honest paper framing is: operator
learners amortise over function space, LC-PINN amortises over loss-weight (or
parametric-coefficient) space. Both produce a usable predictor for the
specific test problem; the comparison is on training-cost-to-target-accuracy
on that test problem.

Self-contained: no `neuralop` dependency. SpectralConv1d + FNO1d implemented
inline (~80 lines) so the behaviour is pinned and reproducible.

Usage:
    python scripts/fno_burgers.py --seeds 0 1 2 3 --n-train 256 --n-epochs 200

Output:
    results/fno_burgers.json
    checkpoints/fno_burgers_seed{s}.pt
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.fft import fft, ifft
from scipy.integrate import solve_ivp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pinns.device import select_device, device_info
from pinns.equations import burgers as burg


REPO = pathlib.Path(__file__).resolve().parent.parent
SNAP_TIMES = [0.25, 0.50, 0.75, 1.00]
T_TARGET = 1.00     # FNO predicts the final snapshot only (Li 2020 1D-Burgers benchmark).
                    # Multi-time output would need FNO-1D-time; left to that variant.


# ---------------------------------------------------------------------------
# FNO architecture (self-contained, Li et al. 2021 1D variant)
# ---------------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    """Linear layer in Fourier space on the lowest `modes` frequencies."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, N)
        B, _, N = x.shape
        x_ft = torch.fft.rfft(x, n=N)                                 # (B, C_in, N//2+1)
        out_ft = torch.zeros(B, self.out_channels, N // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum(
            "bcm,com->bom", x_ft[:, :, :self.modes], self.weight
        )
        return torch.fft.irfft(out_ft, n=N)


class FNOBlock(nn.Module):
    """SpectralConv + 1x1 Conv shortcut + GELU activation."""

    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(channels, channels, modes)
        self.shortcut = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.spectral(x) + self.shortcut(x))


class FNO1d(nn.Module):
    """1D FNO: lift -> N FNO blocks -> project. Predicts multi-time output."""

    def __init__(
        self,
        in_channels: int = 2,    # u_0(x) and a positional grid feature
        out_channels: int = 1,   # single snapshot — u(x, T_TARGET)
        width: int = 64,
        modes: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, width, 1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes) for _ in range(n_layers)])
        self.proj = nn.Sequential(
            nn.Conv1d(width, 128, 1),
            nn.GELU(),
            nn.Conv1d(128, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, N) -> (B, out_channels, N)
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        return self.proj(h)


# ---------------------------------------------------------------------------
# Operator-learning training data: solve random ICs with the same Fourier-
# spectral / Radau scheme used elsewhere in the project.
# ---------------------------------------------------------------------------

def _solve_burgers_for_ic(
    u0: np.ndarray, x_grid: np.ndarray, snap_times: list[float], nu: float = burg.NU,
) -> np.ndarray:
    """Same spectral solver as burg.compute_reference_solution, arbitrary IC."""
    nx = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)

    def rhs(_t, u_cur):
        u_hat = fft(u_cur)
        dudx = np.real(ifft(1j * kx * u_hat))
        d2udx2 = np.real(ifft(-(kx ** 2) * u_hat))
        return -u_cur * dudx + nu * d2udx2

    sol = solve_ivp(
        rhs, [burg.T_MIN, burg.T_MAX], u0,
        method="Radau", t_eval=sorted(snap_times), rtol=1e-6, atol=1e-8,
    )
    # shape: (len(snap_times), nx)
    return sol.y.T.astype(np.float32)


def _sample_random_ic(rng: np.random.Generator, x_grid: np.ndarray, K_max: int = 6) -> np.ndarray:
    """Random truncated Fourier IC: sum_{k=1..K_max} (a_k sin(kπx) + b_k cos(kπx)) / k.

    Decaying spectrum (∝ 1/k) keeps ICs smooth. Periodic on [-1, 1].
    """
    a = rng.uniform(-1.0, 1.0, size=K_max).astype(np.float32)
    b = rng.uniform(-1.0, 1.0, size=K_max).astype(np.float32)
    # Force amplitude ≤ 1 so test IC -sin(πx) (amplitude 1) lies in-distribution.
    u = np.zeros_like(x_grid, dtype=np.float32)
    for k in range(1, K_max + 1):
        u += (a[k - 1] * np.sin(k * np.pi * x_grid) + b[k - 1] * np.cos(k * np.pi * x_grid)) / k
    # Normalise to max abs amplitude 1.
    peak = float(np.max(np.abs(u)))
    if peak > 1e-6:
        u = u / peak
    return u


def build_operator_dataset(
    n_train: int, nx: int = 512, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        x_grid:  (nx,)
        ICs:     (n_train, nx)         u(0, x) per sample
        targets: (n_train, T, nx)      u(t_k, x) per sample, T = len(SNAP_TIMES)
    """
    rng = np.random.default_rng(seed)
    x_grid = np.linspace(burg.X_MIN, burg.X_MAX, nx, endpoint=False).astype(np.float32)
    ICs = np.empty((n_train, nx), dtype=np.float32)
    targets = np.empty((n_train, len(SNAP_TIMES), nx), dtype=np.float32)

    print(f"  Generating {n_train} reference solutions (nx={nx})…")
    for i in range(n_train):
        u0 = _sample_random_ic(rng, x_grid)
        ICs[i] = u0
        targets[i] = _solve_burgers_for_ic(u0, x_grid, SNAP_TIMES)
        if (i + 1) % max(1, n_train // 10) == 0:
            print(f"    {i + 1}/{n_train}")
    return x_grid, ICs, targets


def _to_input(ICs: np.ndarray, x_grid: np.ndarray) -> torch.Tensor:
    """Stack u_0(x) with a positional feature x to give FNO 2 input channels."""
    n_train, nx = ICs.shape
    x_feat = np.broadcast_to(x_grid, (n_train, nx))
    inp = np.stack([ICs, x_feat], axis=1)        # (n_train, 2, nx)
    return torch.tensor(inp, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def _clip_grad_norm_complex_safe(parameters, max_norm: float, eps: float = 1e-6) -> torch.Tensor:
    """clip_grad_norm_ that works when some params have complex grads (FNO spectral weights).

    torch.linalg.vector_norm on complex tensors raises 'norm ops are not supported for
    complex yet'. Compute the L2 norm on .abs() in that case.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    parts = []
    for g in grads:
        if g.is_complex():
            parts.append(torch.linalg.vector_norm(g.abs()))
        else:
            parts.append(torch.linalg.vector_norm(g))
    total_norm = torch.linalg.vector_norm(torch.stack(parts))
    clip_coef = max_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)
    return total_norm


def relative_l2_single(pred: np.ndarray, ref: np.ndarray) -> float:
    """pred, ref: (nx,) — rel-L2 at T_TARGET."""
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-10))


def run_one_seed(
    seed: int,
    train_input: torch.Tensor,
    train_target: torch.Tensor,
    test_input: torch.Tensor,
    test_target: np.ndarray,
    device: torch.device,
    n_epochs: int,
    batch_size: int,
    lr: float,
    width: int,
    modes: int,
    n_layers: int,
) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)

    model = FNO1d(in_channels=2, out_channels=1,
                  width=width, modes=modes, n_layers=n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # FNO recipe (Li et al 2020): Adam, weight_decay=1e-4, StepLR(100, 0.5),
    # rel-L2 loss (not MSE). Grad clipping is not used in the reference impl.
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, n_epochs // 5), gamma=0.5)

    train_input_d = train_input.to(device)
    train_target_d = train_target.to(device)
    n_train = train_input.shape[0]

    t0 = time.time()
    print(f"[seed {seed}] FNO Adam: {n_epochs} epochs  N_train={n_train}  "
          f"params={n_params:,}  batch={batch_size}")
    for epoch in range(n_epochs):
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb = train_input_d[idx]
            yb = train_target_d[idx]
            opt.zero_grad()
            pred = model(xb)
            # Per-sample rel-L2 then averaged over batch — Li et al 2020 recipe.
            num = torch.linalg.vector_norm(pred - yb, dim=(-2, -1))
            den = torch.linalg.vector_norm(yb,        dim=(-2, -1)) + 1e-10
            loss = (num / den).mean()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        sched.step()
        if epoch % max(1, n_epochs // 10) == 0:
            print(f"  epoch {epoch:5d}  L={epoch_loss / n_batches:.4e}")

    elapsed = time.time() - t0

    # Inference on test IC (- sin(πx)) — single snapshot at T_TARGET.
    model.eval()
    with torch.no_grad():
        pred = model(test_input.to(device)).cpu().numpy()[0, 0]   # (nx,)
    rel_l2 = relative_l2_single(pred, test_target)

    ckpt = REPO / "checkpoints" / f"fno_burgers_seed{seed}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "rel_l2_at_T": rel_l2,
        "T_target": T_TARGET,
        "elapsed_sec": elapsed, "seed": seed,
        "n_epochs": n_epochs, "lr": lr, "n_params": n_params,
        "width": width, "modes": modes, "n_layers": n_layers,
    }, ckpt)

    print(f"[seed {seed}] done in {elapsed/60:.1f} min  rel-L2 @ t={T_TARGET} = {rel_l2:.4e}")
    return {
        "seed": seed,
        "rel_l2_at_T": rel_l2,
        "T_target": T_TARGET,
        "elapsed_sec": elapsed,
        "checkpoint": str(ckpt.relative_to(REPO)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--n-train", type=int, default=256, help="number of operator-learning ICs")
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--modes", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--data-seed", type=int, default=0,
                   help="seed for IC sampling; shared across model seeds")
    args = p.parse_args()

    device = select_device()
    print(f"Device: {device_info(device)}")
    print(f"Config: seeds={args.seeds} n_train={args.n_train} n_epochs={args.n_epochs}\n")

    # Reference for evaluation: the standard Burgers reference (same one used by
    # SA-PINN, ReLoBRaLo, Causal-PINN, LC-PINN).
    print("Building Burgers test reference (-sin(πx) IC) …")
    ref_snapshots = burg.compute_reference_solution()
    x_grid_ref = ref_snapshots[T_TARGET][0]
    nx_ref = len(x_grid_ref)
    test_target = ref_snapshots[T_TARGET][1].astype(np.float32)        # (nx,)
    test_ic = (-np.sin(np.pi * x_grid_ref)).astype(np.float32)
    test_input = _to_input(test_ic[None], x_grid_ref)                  # (1, 2, nx)

    # Operator-learning training set: random ICs, same physics (ν, domain, T).
    cache_path = REPO / "results" / f"_fno_burgers_dataset_n{args.n_train}_seed{args.data_seed}.npz"
    if cache_path.exists():
        print(f"\nLoading cached dataset from {cache_path.relative_to(REPO)}…")
        cached = np.load(cache_path)
        x_grid, ICs, targets = cached["x_grid"], cached["ICs"], cached["targets"]
    else:
        print(f"\nBuilding operator-learning dataset (data_seed={args.data_seed})…")
        x_grid, ICs, targets = build_operator_dataset(
            n_train=args.n_train, nx=nx_ref, seed=args.data_seed,
        )
        np.savez(cache_path, x_grid=x_grid, ICs=ICs, targets=targets)
        print(f"  cached → {cache_path.relative_to(REPO)}")
    # Use only the final-time snapshot from cached targets (last index = SNAP_TIMES[-1] = 1.0).
    final_targets = targets[:, -1:, :]                                  # (n_train, 1, nx)
    train_input = _to_input(ICs, x_grid)                                # (n_train, 2, nx)
    train_target = torch.tensor(final_targets, dtype=torch.float32)     # (n_train, 1, nx)

    # Run all seeds (model-init varies; data is shared).
    runs = []
    for s in args.seeds:
        runs.append(run_one_seed(
            s, train_input, train_target, test_input, test_target,
            device, args.n_epochs, args.batch_size, args.lr,
            args.width, args.modes, args.n_layers,
        ))

    means = [r["rel_l2_at_T"] for r in runs]
    elapsed = [r["elapsed_sec"] for r in runs]
    summary = {
        "method": "fno", "equation": "burgers",
        "config": vars(args),
        "runs": runs,
        "summary": {
            "n_seeds": len(runs),
            "rel_l2_mean": float(np.mean(means)),
            "rel_l2_std":  float(np.std(means)),
            "rel_l2_min":  float(np.min(means)),
            "rel_l2_max":  float(np.max(means)),
            "elapsed_mean_sec": float(np.mean(elapsed)),
            "elapsed_total_sec": float(np.sum(elapsed)),
        },
    }
    out = REPO / "results" / "fno_burgers.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out.relative_to(REPO)}")
    print(f"  rel-L2: {summary['summary']['rel_l2_mean']:.4e} ± {summary['summary']['rel_l2_std']:.4e}")
    print(f"  mean wall time per seed: {summary['summary']['elapsed_mean_sec']/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
