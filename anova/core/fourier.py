"""Fourier-feature per-subset MLPs + a 3-D, order-3 HDMR using them.

Diagnosis (2026-05-13): order-2 + order-3 HDMR with tanh-MLPs at hidden up
to 128 captures only ~55% of variance on a 2D Helmholtz LC-PINN
($u = \\sin\\pi x \\sin\\pi y \\cos(kx)\\cos(ky)$, $k \\in [1,10]$). The
issue is that tanh-MLPs of modest width can't represent oscillations at
frequency 10 over a unit-length domain — a known limitation that NeRF /
PINN-RFF / SIREN solved with positional encodings.

So: replace the raw input $z_i$ to each subset MLP with a Fourier
encoding $\\phi(z_i) = [z_i, \\sin(\\omega_1 z_i), \\cos(\\omega_1 z_i),
\\ldots, \\sin(\\omega_L z_i), \\cos(\\omega_L z_i)]$, where the
frequencies $\\omega_l$ span the frequency range of the underlying
function.

The rest of the HDMR machinery (per-subset MLPs, purification) is
unchanged.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn as nn


class FourierSubsetNet(nn.Module):
    """MLP with positional-encoding input transform.

    Input: (N, in_dim) raw values (already normalised to roughly [-1, 1]
    or [0, 1]).
    Encoding: each scalar $z_i$ -> $[z_i, \\sin(\\omega_l z_i), \\cos(\\omega_l z_i)]_{l=1..L}$.
    Default frequencies: $\\omega_l = \\pi \\cdot 2^{l-1}$ for $l=1..L$, so
    L=4 covers $\\omega \\in \\{\\pi, 2\\pi, 4\\pi, 8\\pi\\}$ — enough to
    represent oscillations up to frequency ~10 over a unit-length domain.
    """

    def __init__(self, in_dim: int, num_freqs: int = 4, hidden: int = 64,
                 layers: int = 2):
        super().__init__()
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        # Geometric frequency schedule. Buffer so it moves with .to(device)
        # but is not learned.
        freqs = torch.tensor(
            [np.pi * (2 ** l) for l in range(num_freqs)],
            dtype=torch.float32,
        )
        self.register_buffer("freqs", freqs)
        feat_dim = in_dim * (1 + 2 * num_freqs)
        modules: list[nn.Module] = []
        last = feat_dim
        for _ in range(layers):
            modules.append(nn.Linear(last, hidden))
            modules.append(nn.Tanh())
            last = hidden
        modules.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_dim)
        N, d = x.shape
        # x_freq: (N, d, num_freqs)
        x_freq = x.unsqueeze(-1) * self.freqs.view(1, 1, -1)
        sin_feat = torch.sin(x_freq).reshape(N, d * self.num_freqs)
        cos_feat = torch.cos(x_freq).reshape(N, d * self.num_freqs)
        features = torch.cat([x, sin_feat, cos_feat], dim=-1)
        return self.net(features).squeeze(-1)


class FourierTruncatedHDMR(nn.Module):
    """Order-2 truncated HDMR for any d with Fourier-feature subset MLPs.

    Same API as anova/hdmr_net.py's TruncatedHDMR — drop-in swap for
    high-frequency function targets where tanh-MLPs of modest width
    can't represent oscillations.
    """

    def __init__(self, d: int, num_freqs: int = 4, hidden: int = 64,
                 layers: int = 2):
        super().__init__()
        self.d = d
        self.max_order = 2
        self.f0 = nn.Parameter(torch.zeros(1))
        self.mains = nn.ModuleList([
            FourierSubsetNet(1, num_freqs=num_freqs, hidden=hidden, layers=layers)
            for _ in range(d)
        ])
        self.pair_indices = list(itertools.combinations(range(d), 2))
        self.pairs = nn.ModuleList([
            FourierSubsetNet(2, num_freqs=num_freqs, hidden=hidden, layers=layers)
            for _ in self.pair_indices
        ])

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def main_terms(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.mains[i](x[:, i:i+1]) for i in range(self.d)], dim=1)

    def pair_terms(self, x: torch.Tensor) -> torch.Tensor:
        if not self.pair_indices:
            return torch.zeros(x.size(0), 0, device=x.device)
        return torch.stack([self.pairs[k](x[:, [i, j]])
                            for k, (i, j) in enumerate(self.pair_indices)], dim=1)

    def _purify_main(self, m: torch.Tensor) -> torch.Tensor:
        return m - m.mean(dim=0, keepdim=True)

    def _purify_pair(self, p_terms: torch.Tensor, x: torch.Tensor,
                     n_bins: int = 10) -> torch.Tensor:
        if not self.pair_indices:
            return p_terms
        N, P = p_terms.shape
        edges = torch.linspace(x.min().item(), x.max().item() + 1e-6,
                               n_bins + 1, device=x.device)
        out = p_terms.clone()
        for k, (i, j) in enumerate(self.pair_indices):
            f = out[:, k]
            with torch.no_grad():
                bi = torch.bucketize(x[:, i].contiguous(), edges) - 1
                bj = torch.bucketize(x[:, j].contiguous(), edges) - 1
                bi = bi.clamp_(0, n_bins - 1)
                bj = bj.clamp_(0, n_bins - 1)
            mean_given_i = torch.zeros(n_bins, device=x.device)
            count_i = torch.zeros(n_bins, device=x.device)
            mean_given_i.scatter_add_(0, bi, f)
            count_i.scatter_add_(0, bi, torch.ones_like(f))
            mean_given_i = mean_given_i / count_i.clamp_min(1.0)
            mean_given_j = torch.zeros(n_bins, device=x.device)
            count_j = torch.zeros(n_bins, device=x.device)
            mean_given_j.scatter_add_(0, bj, f)
            count_j.scatter_add_(0, bj, torch.ones_like(f))
            mean_given_j = mean_given_j / count_j.clamp_min(1.0)
            f_mean = f.mean()
            out[:, k] = f - mean_given_i[bi] - mean_given_j[bj] + f_mean
        return out

    def forward(self, x: torch.Tensor, include_pairs: bool = True,
                purify: bool = True):
        m = self.main_terms(x)
        if purify:
            m = self._purify_main(m)
        m_sum = m.sum(dim=1)
        if include_pairs and self.pair_indices:
            p = self.pair_terms(x)
            if purify:
                p = self._purify_pair(p, x)
            p_sum = p.sum(dim=1)
        else:
            p = torch.zeros(x.size(0), 0, device=x.device)
            p_sum = torch.zeros_like(m_sum)
        return self.f0 + m_sum + p_sum, m, p_sum

    def evaluate_terms(self, x: torch.Tensor, purify: bool = True):
        m = self.main_terms(x)
        if purify:
            m = self._purify_main(m)
        if self.pair_indices:
            p = self.pair_terms(x)
            if purify:
                p = self._purify_pair(p, x)
        else:
            p = torch.zeros(x.size(0), 0, device=x.device)
        return self.f0.detach().clone(), m, p


class _TripletFourierHDMR3D(nn.Module):
    """d=3, max_order=3, all subset nets use FourierSubsetNet.

    Same API as `_TripletHDMR3D` in joint_hdmr.py — drop-in swap.
    """

    def __init__(self, num_freqs: int = 4, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.d = 3
        self.f0 = nn.Parameter(torch.zeros(1))
        self.mains = nn.ModuleList([
            FourierSubsetNet(1, num_freqs=num_freqs, hidden=hidden, layers=layers)
            for _ in range(3)
        ])
        self.pair_indices = list(itertools.combinations(range(3), 2))
        self.pairs = nn.ModuleList([
            FourierSubsetNet(2, num_freqs=num_freqs, hidden=hidden, layers=layers)
            for _ in self.pair_indices
        ])
        self.triplet = FourierSubsetNet(3, num_freqs=num_freqs, hidden=hidden, layers=layers)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def main_terms(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.mains[i](x[:, i:i+1]) for i in range(self.d)], dim=1)

    def pair_terms(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.pairs[k](x[:, [i, j]])
                            for k, (i, j) in enumerate(self.pair_indices)], dim=1)

    # Same purification routines as the tanh version — they operate on
    # per-subset MLP outputs, not on the inputs.
    def _purify_main(self, m: torch.Tensor) -> torch.Tensor:
        return m - m.mean(dim=0, keepdim=True)

    def _purify_pair(self, p_terms: torch.Tensor, x: torch.Tensor,
                     n_bins: int = 10) -> torch.Tensor:
        N, P = p_terms.shape
        edges = torch.linspace(x.min().item(), x.max().item() + 1e-6,
                               n_bins + 1, device=x.device)
        out = p_terms.clone()
        for k, (i, j) in enumerate(self.pair_indices):
            f = out[:, k]
            with torch.no_grad():
                bi = torch.bucketize(x[:, i].contiguous(), edges) - 1
                bj = torch.bucketize(x[:, j].contiguous(), edges) - 1
                bi = bi.clamp_(0, n_bins - 1)
                bj = bj.clamp_(0, n_bins - 1)
            mean_given_i = torch.zeros(n_bins, device=x.device)
            count_i = torch.zeros(n_bins, device=x.device)
            mean_given_i.scatter_add_(0, bi, f)
            count_i.scatter_add_(0, bi, torch.ones_like(f))
            mean_given_i = mean_given_i / count_i.clamp_min(1.0)
            mean_given_j = torch.zeros(n_bins, device=x.device)
            count_j = torch.zeros(n_bins, device=x.device)
            mean_given_j.scatter_add_(0, bj, f)
            count_j.scatter_add_(0, bj, torch.ones_like(f))
            mean_given_j = mean_given_j / count_j.clamp_min(1.0)
            f_mean = f.mean()
            out[:, k] = f - mean_given_i[bi] - mean_given_j[bj] + f_mean
        return out

    def _purify_triplet(self, t: torch.Tensor, z: torch.Tensor,
                        n_bins: int = 10) -> torch.Tensor:
        device = t.device
        bin_idx = []
        for ax in range(3):
            zi = z[:, ax].contiguous()
            lo, hi = zi.min().item(), zi.max().item() + 1e-6
            edges = torch.linspace(lo, hi, n_bins + 1, device=device)
            bi = torch.bucketize(zi, edges) - 1
            bi = bi.clamp_(0, n_bins - 1)
            bin_idx.append(bi)

        def cm1d(values, bi):
            m = torch.zeros(n_bins, device=device); c = torch.zeros(n_bins, device=device)
            m.scatter_add_(0, bi, values); c.scatter_add_(0, bi, torch.ones_like(values))
            return m / c.clamp_min(1.0)

        def cm2d(values, bi, bj):
            flat = bi * n_bins + bj
            m = torch.zeros(n_bins * n_bins, device=device)
            c = torch.zeros(n_bins * n_bins, device=device)
            m.scatter_add_(0, flat, values); c.scatter_add_(0, flat, torch.ones_like(values))
            return m / c.clamp_min(1.0)

        with torch.no_grad():
            m_overall = t.mean()
            m_i = [cm1d(t, bin_idx[ax]) for ax in range(3)]
            m_ij = {(i, j): cm2d(t, bin_idx[i], bin_idx[j])
                    for i, j in [(0, 1), (0, 2), (1, 2)]}
        bx, by, bz = bin_idx
        return (t
                - m_ij[(0, 1)][bx * n_bins + by]
                - m_ij[(0, 2)][bx * n_bins + bz]
                - m_ij[(1, 2)][by * n_bins + bz]
                + m_i[0][bx] + m_i[1][by] + m_i[2][bz]
                - m_overall)

    def forward(self, z: torch.Tensor, include_pairs: bool = True,
                include_triplet: bool = True, purify: bool = True):
        m = self.main_terms(z)
        if purify:
            m = self._purify_main(m)
        m_sum = m.sum(dim=1)
        if include_pairs:
            p = self.pair_terms(z)
            if purify:
                p = self._purify_pair(p, z)
            p_sum = p.sum(dim=1)
        else:
            p = torch.zeros(z.shape[0], 0, device=z.device)
            p_sum = torch.zeros_like(m_sum)
        if include_triplet:
            t = self.triplet(z)
            if purify:
                t = self._purify_triplet(t, z)
        else:
            t = torch.zeros(z.shape[0], device=z.device)
        return self.f0 + m_sum + p_sum + t, m, p, t

    def evaluate_terms(self, z: torch.Tensor, include_triplet: bool = True,
                       purify: bool = True):
        m = self.main_terms(z)
        if purify:
            m = self._purify_main(m)
        p = self.pair_terms(z)
        if purify:
            p = self._purify_pair(p, z)
        if include_triplet:
            t = self.triplet(z)
            if purify:
                t = self._purify_triplet(t, z)
        else:
            t = torch.zeros(z.shape[0], device=z.device)
        return self.f0.detach().clone(), m, p, t
