"""Truncated HDMR neural architecture.

f(x) = f_0 + sum_i f_i(x_i) + sum_{i<j} f_ij(x_i, x_j)

Each f_u is a small MLP that takes ONLY the inputs in u. This enforces
the Jacobian-mask condition structurally (df_u/dx_j = 0 for j not in u
because f_u literally doesn't see x_j). No soft penalty needed for that.

ANOVA orthogonality conditions (E[f_u | x_v] = 0 for v subset u) are
enforced via empirical centering during forward pass — a 'purification'
step. After each forward pass:
  - subtract empirical mean of each f_u  (enforces E[f_u] = 0)
  - for pairs, subtract empirical conditional means against each axis
    (enforces E[f_ij | x_i] = 0 and E[f_ij | x_j] = 0)

This is cleaner than soft penalties: the ANOVA condition is satisfied
*by construction* on the training batch, and the only learning signal
is reconstruction.
"""

import itertools
import numpy as np
import torch
import torch.nn as nn


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SubsetNet(nn.Module):
    def __init__(self, in_dim, hidden=32, layers=2):
        super().__init__()
        modules = []
        last = in_dim
        for _ in range(layers):
            modules.append(nn.Linear(last, hidden))
            modules.append(nn.Tanh())
            last = hidden
        modules.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TruncatedHDMR(nn.Module):
    def __init__(self, d, max_order=2, hidden=32, layers=2):
        super().__init__()
        self.d = d
        self.max_order = max_order
        self.f0 = nn.Parameter(torch.zeros(1))
        self.mains = nn.ModuleList([SubsetNet(1, hidden, layers) for _ in range(d)])
        if max_order >= 2:
            self.pair_indices = list(itertools.combinations(range(d), 2))
            self.pairs = nn.ModuleList([SubsetNet(2, hidden, layers) for _ in self.pair_indices])
        else:
            self.pair_indices = []
            self.pairs = nn.ModuleList()

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def main_terms(self, x):
        return torch.stack([self.mains[i](x[:, i:i+1]) for i in range(self.d)], dim=1)

    def pair_terms(self, x):
        if not self.pair_indices:
            return torch.zeros(x.size(0), 0, device=x.device)
        return torch.stack([self.pairs[k](x[:, [i, j]])
                            for k, (i, j) in enumerate(self.pair_indices)], dim=1)

    def _purify_main(self, m):
        # m: (N, d). Subtract per-column mean so E[f_i] ≈ 0 on this batch.
        return m - m.mean(dim=0, keepdim=True)

    def _purify_pair(self, p_terms, x, n_bins=10):
        # p_terms: (N, n_pairs). For each pair (i, j), subtract E[f_ij | x_i],
        # E[f_ij | x_j], and add back E[f_ij] (inclusion-exclusion to satisfy
        # both zero-conditional conditions simultaneously). Done with binning.
        if not self.pair_indices:
            return p_terms
        N, P = p_terms.shape
        edges = torch.linspace(x.min().item(), x.max().item() + 1e-6, n_bins + 1, device=x.device)
        out = p_terms.clone()
        for k, (i, j) in enumerate(self.pair_indices):
            f = out[:, k]
            with torch.no_grad():
                bi = torch.bucketize(x[:, i].contiguous(), edges) - 1
                bj = torch.bucketize(x[:, j].contiguous(), edges) - 1
                bi = bi.clamp_(0, n_bins - 1)
                bj = bj.clamp_(0, n_bins - 1)
            # E[f | x_i] via scatter-mean over bi
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
            f_purified = f - mean_given_i[bi] - mean_given_j[bj] + f_mean
            out[:, k] = f_purified
        return out

    def forward(self, x, include_pairs=True, purify=True):
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
            p_sum = torch.zeros_like(m_sum)
        return self.f0 + m_sum + p_sum, m, p_sum

    def evaluate_terms(self, x, purify=True):
        """Returns (f0, mains_(N,d), pairs_(N,n_pairs)) for analysis."""
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
