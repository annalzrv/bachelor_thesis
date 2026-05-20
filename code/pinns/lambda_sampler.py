"""Curriculum-aware sampler for loss-weight vectors log(lambda)."""

from __future__ import annotations

import numpy as np
import torch


class LambdaSampler:
    """
    Curriculum-aware sampler for loss weight vectors.

    Three sampling modes (controlled by `mode`):

    "logspace" (default):
        Draws log(lambda) ~ Uniform(center - hw, center + hw), then
        returns softmax(log_lambda) as the weight vector.  Half-width (hw)
        expands linearly from hw_init to hw_final over curriculum_steps.

    "simplex":
        Draws lambda directly from Dirichlet(alpha * ones), so weights
        always sum to 1 and cover the full probability simplex.
        alpha starts at alpha_init (concentrated near equal weights) and
        decays to alpha_final (more spread / corner-friendly) over
        curriculum_steps.  log_lambda returned as log(lambda) for
        consistency with the model's input convention.

    "uniform":
        Draws each weight independently from U(0, 1).  Weights do NOT
        sum to 1 — each loss term is scaled independently.  This
        decouples the loss terms more aggressively than simplex/logspace.
        log_lambda = log(weight) is passed to the model.

    "uniform_normalized":
        Draws each weight independently from U(0, 1) then normalises so
        the vector lies on the simplex (Σpᵢ = 1).  Keeps the independent
        draw of uniform but enforces sum-to-one so the network sees
        contrastive trade-offs (when one weight is up, others are down).
        Different from Dirichlet(1,…,1): the normalisation biases samples
        toward the simplex centre.
    """

    def __init__(
        self,
        dim: int = 4,
        center: torch.Tensor | None = None,
        hw_init: float = 0.1,
        hw_final: float = 3.0,
        curriculum_steps: int = 8000,
        device: str | torch.device = "cpu",
        mode: str = "logspace",
        alpha_init: float = 10.0,
        alpha_final: float = 0.5,
    ):
        if mode not in ("logspace", "simplex", "uniform", "uniform_normalized"):
            raise ValueError(
                f"mode must be 'logspace', 'simplex', 'uniform', or 'uniform_normalized', "
                f"got {mode!r}"
            )
        self.mode = mode
        self.dim = dim
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.hw_init = hw_init
        self.hw_final = hw_final
        self.curriculum_steps = curriculum_steps
        self.alpha_init = alpha_init
        self.alpha_final = alpha_final
        if center is None:
            if dim == 4:
                # BL-tuned default: emphasise BC and IC over PDE and data
                center = torch.tensor(
                    [np.log(1.0), np.log(10.0), np.log(10.0), np.log(1.0)],
                    dtype=torch.float32,
                )
            else:
                # Generic default: log-uniform (all weights equal in softmax)
                center = torch.zeros(dim, dtype=torch.float32)
        self.center = center.to(self.device)

    def half_width(self, step: int) -> float:
        progress = min(step / max(self.curriculum_steps, 1), 1.0)
        return self.hw_init + progress * (self.hw_final - self.hw_init)

    def _dirichlet_alpha(self, step: int) -> float:
        """Concentration parameter: high (uniform-ish) → low (spread) over curriculum."""
        progress = min(step / max(self.curriculum_steps, 1), 1.0)
        return self.alpha_init + progress * (self.alpha_final - self.alpha_init)

    def sample(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (log_lambda, weights) for a single λ vector."""
        if self.mode == "uniform":
            p_lam = torch.rand(self.dim, device=self.device)
            log_lam = torch.log(p_lam.clamp(min=1e-8))
            return log_lam, p_lam

        if self.mode == "uniform_normalized":
            p_raw = torch.rand(self.dim, device=self.device)
            p_lam = p_raw / p_raw.sum().clamp(min=1e-8)
            log_lam = torch.log(p_lam.clamp(min=1e-8))
            return log_lam, p_lam

        if self.mode == "simplex":
            alpha = self._dirichlet_alpha(step)
            # Dirichlet not supported on MPS; sample on CPU and move.
            dist = torch.distributions.Dirichlet(torch.full((self.dim,), alpha))
            p_lam = dist.sample().to(self.device)
            log_lam = torch.log(p_lam.clamp(min=1e-8))
            return log_lam, p_lam

        hw = self.half_width(step)
        noise = (2.0 * torch.rand(self.dim, device=self.device) - 1.0) * hw
        log_lam = self.center + noise
        return log_lam, torch.softmax(log_lam, dim=0)

    def sample_batch(self, n: int, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample n different lambda vectors. Returns (log_lams, p_lams) each (n, dim)."""
        if self.mode == "uniform":
            p_lams = torch.rand(n, self.dim, device=self.device)
            log_lams = torch.log(p_lams.clamp(min=1e-8))
            return log_lams, p_lams

        if self.mode == "uniform_normalized":
            p_raw = torch.rand(n, self.dim, device=self.device)
            p_lams = p_raw / p_raw.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            log_lams = torch.log(p_lams.clamp(min=1e-8))
            return log_lams, p_lams

        if self.mode == "simplex":
            alpha = self._dirichlet_alpha(step)
            # Dirichlet not supported on MPS; sample on CPU and move.
            dist = torch.distributions.Dirichlet(torch.full((self.dim,), alpha))
            p_lams = dist.sample((n,)).to(self.device)
            log_lams = torch.log(p_lams.clamp(min=1e-8))
            return log_lams, p_lams

        hw = self.half_width(step)
        noise = (2.0 * torch.rand(n, self.dim, device=self.device) - 1.0) * hw
        log_lams = self.center.unsqueeze(0) + noise
        return log_lams, torch.softmax(log_lams, dim=-1)
