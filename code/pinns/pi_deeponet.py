"""Physics-Informed DeepONet (Lu, Jin, Karniadakis 2021; Wang, Wang, Perdikaris 2021).

Standard branch–trunk decomposition:

    u(x; lambda) = sum_i  branch_i(lambda) * trunk_i(x) + b

trained on the *residual* loss only — no solver-generated paired data, like
LC-PINN. Apples-to-apples comparison: same residual signal, same lambda
sampling, only the architecture differs (factored bilinear vs FiLM-MLP).

For 1D Helmholtz: branch input = k_norm (dim_lambda=1), trunk input = x
(dim_phys=1). Output is a scalar u value.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int):
        super().__init__()
        widths = [in_dim, *hidden_dims, out_dim]
        layers: list[nn.Module] = []
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(widths) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PIDeepONet(nn.Module):
    """Branch over (lambda), trunk over (coords). Output u(x; lambda) scalar."""

    def __init__(
        self,
        dim_phys: int,
        dim_lambda: int,
        n_basis: int = 64,
        branch_hidden: list[int] | None = None,
        trunk_hidden: list[int] | None = None,
    ):
        super().__init__()
        if branch_hidden is None:
            branch_hidden = [64, 64, 64, 64]
        if trunk_hidden is None:
            trunk_hidden = [64, 64, 64, 64]
        self.branch = _MLP(dim_lambda, branch_hidden, n_basis)
        self.trunk = _MLP(dim_phys, trunk_hidden, n_basis)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, coords: torch.Tensor, log_lambda: torch.Tensor) -> torch.Tensor:
        """Evaluate u at a batch of coords given a single lambda (or per-coord lambda).

        coords: (N, dim_phys)
        log_lambda: (1, dim_lambda) or (N, dim_lambda) — broadcast handled below.
        Returns: (N, 1)
        """
        if log_lambda.dim() == 1:
            log_lambda = log_lambda.unsqueeze(0)
        b = self.branch(log_lambda)              # (1, p) or (N, p)
        t = self.trunk(coords)                    # (N, p)
        if b.shape[0] == 1:
            u = (b * t).sum(dim=-1, keepdim=True) + self.bias
        else:
            u = (b * t).sum(dim=-1, keepdim=True) + self.bias
        return u
