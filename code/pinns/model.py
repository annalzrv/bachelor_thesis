"""Loss-conditional PINN: network takes (x, t) and log_lambda as inputs.

Two conditioning styles are supported:

* ``concat`` (legacy default): ``log_lambda`` is concatenated to ``coords``
  before the first MLP layer. Simple, weak conditioning signal.
* ``film``: feature-wise linear modulation (Perez et al. 2018). A small
  hypernet ``g(log_lambda)`` produces per-layer scale ``gamma`` and shift
  ``beta``, applied as ``h <- gamma * h + beta`` between linear-and-Tanh
  blocks. Stronger conditioning, comparable parameter count.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

OutputMode = Literal["identity", "sigmoid"]
ConditioningMode = Literal["concat", "film"]


class _FiLMHyperNet(nn.Module):
    """One hypernetwork producing (gamma, beta) for every modulated layer."""

    def __init__(self, dim_lambda: int, layer_widths: list[int], hidden: int = 64):
        super().__init__()
        self.layer_widths = layer_widths
        total = 2 * sum(layer_widths)
        self.trunk = nn.Sequential(
            nn.Linear(dim_lambda, hidden), nn.Tanh(),
            nn.Linear(hidden, total),
        )
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        # Initialise so gamma starts ~1, beta ~0 (identity modulation).
        with torch.no_grad():
            self.trunk[-1].weight.mul_(0.01)

    def forward(self, log_lambda: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Returns a list of (gamma, beta) per modulated layer, each (B, h)."""
        out = self.trunk(log_lambda)
        # Split into per-layer (gamma, beta), each of width ``layer_widths[i]``.
        chunks = []
        offset = 0
        for h in self.layer_widths:
            gamma = 1.0 + out[..., offset:offset + h]
            offset += h
            beta = out[..., offset:offset + h]
            offset += h
            chunks.append((gamma, beta))
        return chunks


class LossConditionalPINN(nn.Module):
    """
    PINN conditioned on loss weights or a physical scalar parameter.
    F_c(x, t, log_lambda) -> u(x,t). Lambda is a NETWORK INPUT (not
    nn.Parameter), so the network learns a family of solutions indexed
    by the loss-weight vector or coefficient.
    """

    def __init__(
        self,
        dim_phys: int = 2,
        dim_lambda: int = 4,
        hidden_dims: list[int] | None = None,
        output: OutputMode = "identity",
        dim_out: int = 1,
        conditioning: ConditioningMode = "concat",
        film_hyper_hidden: int = 64,
    ):
        super().__init__()
        if output not in ("identity", "sigmoid"):
            raise ValueError(f"output must be 'identity' or 'sigmoid', got {output!r}")
        if conditioning not in ("concat", "film"):
            raise ValueError(f"conditioning must be 'concat' or 'film', got {conditioning!r}")
        self._output_mode: OutputMode = output
        self._conditioning: ConditioningMode = conditioning
        self.dim_out = dim_out
        if hidden_dims is None:
            hidden_dims = [64, 64, 64, 64]
        self.hidden_dims = hidden_dims

        if conditioning == "concat":
            layers: list[nn.Module] = []
            prev = dim_phys + dim_lambda
            for h in hidden_dims:
                layers.extend([nn.Linear(prev, h), nn.Tanh()])
                prev = h
            layers.append(nn.Linear(prev, dim_out))
            self.net = nn.Sequential(*layers)
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
            self.linears = None
            self.film_hyper = None
        else:
            self.linears = nn.ModuleList()
            prev = dim_phys
            for h in hidden_dims:
                self.linears.append(nn.Linear(prev, h))
                prev = h
            self.head = nn.Linear(prev, dim_out)
            for m in list(self.linears) + [self.head]:
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            self.film_hyper = _FiLMHyperNet(
                dim_lambda=dim_lambda,
                layer_widths=list(hidden_dims),
                hidden=film_hyper_hidden,
            )
            self.net = None

    def forward(self, coords: torch.Tensor, log_lambda: torch.Tensor) -> torch.Tensor:
        """
        coords:     (B, dim_phys)
        log_lambda: (dim_lambda,) or (B, dim_lambda)
        """
        if log_lambda.dim() == 1:
            log_lambda = log_lambda.unsqueeze(0).expand(coords.shape[0], -1)

        if self._conditioning == "concat":
            raw = self.net(torch.cat([coords, log_lambda], dim=-1))
        else:
            mods = self.film_hyper(log_lambda)
            h = coords
            for layer, (gamma, beta) in zip(self.linears, mods):
                h = layer(h)
                h = gamma * h + beta
                h = torch.tanh(h)
            raw = self.head(h)

        if self._output_mode == "sigmoid":
            return torch.sigmoid(raw)
        return raw
