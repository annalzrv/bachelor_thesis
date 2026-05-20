"""Fixed-weight baseline PINN (equation-agnostic)."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.auto import tqdm


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    warmup_frac: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup then cosine annealing. If warmup_frac=0, cosine only."""
    if warmup_frac <= 0.0:
        return CosineAnnealingLR(optimizer, T_max=n_epochs)
    warmup_steps = max(1, int(n_epochs * warmup_frac))
    cosine_steps = max(1, n_epochs - warmup_steps)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


class FixedWeightPINN(nn.Module):
    """
    Standard PINN with fixed loss weights: F(x, t) -> u(x,t).

    Same MLP backbone as LossConditionalPINN (same hidden_dims, Tanh,
    Xavier init) but lambda is NOT an input.
    """

    def __init__(
        self,
        dim_phys: int = 2,
        hidden_dims: list[int] | None = None,
        output: str = "identity",
        dim_out: int = 1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 64, 64]
        self.dim_out = dim_out
        layers: list[nn.Module] = []
        prev = dim_phys
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.Tanh()])
            prev = h
        layers.append(nn.Linear(prev, dim_out))
        self.net = nn.Sequential(*layers)
        self._output_mode = output
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        raw = self.net(coords)
        if self._output_mode == "sigmoid":
            return torch.sigmoid(raw)
        return raw


def train_fixed_pinn(
    model: FixedWeightPINN,
    weights: list[float] | torch.Tensor,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    loss_fn: Callable[..., dict[str, torch.Tensor]],
    n_epochs: int = 30_000,
    lr: float = 1e-3,
    log_every: int = 2000,
    desc: str = "Baseline",
    warmup_frac: float = 0.0,
    on_log: Callable[[dict], None] | None = None,
    normalize: bool = True,
) -> dict[str, list]:
    """
    Training loop for a fixed-weight PINN.

    loss_fn(model, batch) -> dict[str, Tensor]
        Must return named loss terms. The weighted sum uses the provided weights
        in dict insertion order.

    normalize: if True (default) weights are rescaled to sum to 1 before use;
        if False they are applied as-is (required for fair comparison to an
        LC-PINN trained in uniform mode where each weight is independent in
        [0, 1] and the sum varies across steps).
    """
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, dtype=torch.float32)
    w = (weights / weights.sum()) if normalize else weights
    w = w.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = _make_scheduler(optimizer, n_epochs, warmup_frac)
    history: dict[str, list] = {"step": [], "total": []}
    t0 = time.time()

    pbar = tqdm(range(n_epochs), desc=desc, unit="step", miniters=log_every, smoothing=0.05)
    for step in pbar:
        optimizer.zero_grad()
        losses = loss_fn(model, batch)
        loss_values = list(losses.values())
        total = sum(wi * Li for wi, Li in zip(w, loss_values))
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % log_every == 0:
            vals = {k: v.item() for k, v in losses.items()}
            history["step"].append(step)
            history["total"].append(total.item())
            for k, v in vals.items():
                history.setdefault(k, []).append(v)
            pbar.set_postfix(L=f"{total.item():.2e}", refresh=False)
            if on_log is not None:
                on_log(history)

    elapsed = time.time() - t0
    history["elapsed_sec"] = elapsed
    print(f"{desc} done in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    return history
