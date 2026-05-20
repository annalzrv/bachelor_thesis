"""Joint (x, lambda) HDMR wrapper.

TruncatedHDMR already decomposes a function of d variables into main +
pair effects under whatever joint distribution we sample from. To do
"joint (x, lambda) ANOVA" we simply concatenate spatial and parameter
inputs into one d-vector and train TruncatedHDMR on samples from the
product prior p(x) * p(lambda).

What's *new* here is the aggregation. After training, we want three
canonical signatures separating the spatial axis from the parameter
axis:

    u_x(x)        = sum of mains over spatial dims + sum of pairs (i,j)
                    where both i,j are spatial dims
    u_lambda(l)   = sum of mains over parameter dims + sum of pairs (i,j)
                    where both i,j are parameter dims
    u_{x,l}(x, l) = sum of cross pairs (i, j) with one spatial + one
                    parameter dim

For 2D Helmholtz: dim_x = 2 (spatial x, y), dim_lambda = 1 (k_norm), so
total d = 3 and there is exactly one same-spatial pair, zero same-param
pairs, and two cross pairs.
"""

from __future__ import annotations

import itertools
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hdmr_net import SubsetNet, TruncatedHDMR, select_device


class _TripletHDMR3D(nn.Module):
    """d=3 specialization adding the {0,1,2} triplet MLP on top of TruncatedHDMR.

    Order-2 mains and pairs are delegated to TruncatedHDMR (so the synthetic
    validation already passes). The new triplet MLP captures the irreducible
    3-way interaction. Purification of the triplet uses 2-D + 1-D empirical
    conditional means via binning under the inclusion-exclusion formula.
    """

    def __init__(self, hidden: int = 32, layers: int = 2):
        super().__init__()
        self.d = 3
        self.order2 = TruncatedHDMR(d=3, max_order=2, hidden=hidden, layers=layers)
        self.triplet = SubsetNet(3, hidden, layers)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def pair_indices(self):
        return self.order2.pair_indices

    @property
    def f0(self):
        return self.order2.f0

    # ---- triplet purification with 2-D and 1-D binning ----------------------
    def _purify_triplet(self, t: torch.Tensor, z: torch.Tensor, n_bins: int = 10):
        """t: (N,) triplet outputs. z: (N, 3) joint inputs. Enforce
        E[t|z_i,z_j] = 0 for all 3 pairs (and downstream the 1-D and 0-D
        conditions) via inclusion-exclusion subtraction.
        """
        N = t.shape[0]
        device = t.device
        # Common bin edges for each axis: use the empirical min/max of z.
        edges_per_axis = []
        bin_idx_per_axis = []
        for ax in range(3):
            zi = z[:, ax].contiguous()
            lo, hi = zi.min().item(), zi.max().item() + 1e-6
            edges = torch.linspace(lo, hi, n_bins + 1, device=device)
            bi = torch.bucketize(zi, edges) - 1
            bi = bi.clamp_(0, n_bins - 1)
            edges_per_axis.append(edges)
            bin_idx_per_axis.append(bi)

        def cond_mean_1d(values, bi):
            mean = torch.zeros(n_bins, device=device)
            count = torch.zeros(n_bins, device=device)
            mean.scatter_add_(0, bi, values)
            count.scatter_add_(0, bi, torch.ones_like(values))
            return mean / count.clamp_min(1.0)

        def cond_mean_2d(values, bi, bj):
            flat = bi * n_bins + bj
            mean = torch.zeros(n_bins * n_bins, device=device)
            count = torch.zeros(n_bins * n_bins, device=device)
            mean.scatter_add_(0, flat, values)
            count.scatter_add_(0, flat, torch.ones_like(values))
            return mean / count.clamp_min(1.0)

        with torch.no_grad():
            m_overall = t.mean()
            m_i = [cond_mean_1d(t, bin_idx_per_axis[ax]) for ax in range(3)]
            m_ij = {}
            for i, j in [(0, 1), (0, 2), (1, 2)]:
                m_ij[(i, j)] = cond_mean_2d(t, bin_idx_per_axis[i], bin_idx_per_axis[j])

        # Inclusion-exclusion:
        # f_pure = f - E[f|x,y] - E[f|x,z] - E[f|y,z]
        #            + E[f|x] + E[f|y] + E[f|z] - E[f]
        bi_x = bin_idx_per_axis[0]
        bi_y = bin_idx_per_axis[1]
        bi_z = bin_idx_per_axis[2]
        f_xy = m_ij[(0, 1)][bi_x * n_bins + bi_y]
        f_xz = m_ij[(0, 2)][bi_x * n_bins + bi_z]
        f_yz = m_ij[(1, 2)][bi_y * n_bins + bi_z]
        f_x = m_i[0][bi_x]
        f_y = m_i[1][bi_y]
        f_z = m_i[2][bi_z]
        return t - f_xy - f_xz - f_yz + f_x + f_y + f_z - m_overall

    # ---- forward + helpers --------------------------------------------------
    def forward(self, z: torch.Tensor, include_pairs: bool = True,
                include_triplet: bool = True, purify: bool = True):
        # Order-2 part
        m = self.order2.main_terms(z)
        if purify:
            m = self.order2._purify_main(m)
        m_sum = m.sum(dim=1)
        if include_pairs and self.order2.pair_indices:
            p = self.order2.pair_terms(z)
            if purify:
                p = self.order2._purify_pair(p, z)
            p_sum = p.sum(dim=1)
        else:
            p = torch.zeros(z.shape[0], 0, device=z.device)
            p_sum = torch.zeros_like(m_sum)
        # Triplet
        if include_triplet:
            t = self.triplet(z)
            if purify:
                t = self._purify_triplet(t, z)
        else:
            t = torch.zeros(z.shape[0], device=z.device)
        return self.order2.f0 + m_sum + p_sum + t, m, p, t

    def evaluate_terms(self, z: torch.Tensor, include_triplet: bool = True,
                       purify: bool = True):
        m = self.order2.main_terms(z)
        if purify:
            m = self.order2._purify_main(m)
        if self.order2.pair_indices:
            p = self.order2.pair_terms(z)
            if purify:
                p = self.order2._purify_pair(p, z)
        else:
            p = torch.zeros(z.shape[0], 0, device=z.device)
        if include_triplet:
            t = self.triplet(z)
            if purify:
                t = self._purify_triplet(t, z)
        else:
            t = torch.zeros(z.shape[0], device=z.device)
        return self.order2.f0.detach().clone(), m, p, t


class JointHDMR:
    """Joint-input ANOVA decomposer.

    Parameters
    ----------
    dim_x : int
        Number of spatial input dimensions (treated as "x").
    dim_lambda : int
        Number of parameter input dimensions (treated as "lambda").
    hidden : int
        Hidden width of each per-subset MLP.
    layers : int
        Hidden depth of each per-subset MLP.
    """

    def __init__(self, dim_x: int, dim_lambda: int, hidden: int = 32, layers: int = 2,
                 max_order: int = 2, use_fourier: bool = False, num_freqs: int = 4):
        self.dim_x = dim_x
        self.dim_lambda = dim_lambda
        self.d = dim_x + dim_lambda
        self.max_order = max_order
        self.use_fourier = use_fourier
        self.x_axes = tuple(range(dim_x))
        self.lambda_axes = tuple(range(dim_x, self.d))
        self.device = select_device()
        if max_order >= 3 and self.d == 3:
            self._has_triplet = True
            if use_fourier:
                from lc_anova.core.fourier import _TripletFourierHDMR3D
                self.model = _TripletFourierHDMR3D(
                    num_freqs=num_freqs, hidden=hidden, layers=layers
                ).to(self.device)
            else:
                self.model = _TripletHDMR3D(hidden=hidden, layers=layers).to(self.device)
        elif max_order >= 3:
            raise NotImplementedError(
                f"max_order=3 currently supported only for d=3; got d={self.d}"
            )
        else:
            self._has_triplet = False
            if use_fourier:
                from lc_anova.core.fourier import FourierTruncatedHDMR
                self.model = FourierTruncatedHDMR(
                    d=self.d, num_freqs=num_freqs, hidden=hidden, layers=layers
                ).to(self.device)
            else:
                self.model = TruncatedHDMR(d=self.d, max_order=2, hidden=hidden, layers=layers).to(self.device)
        self.pair_classification = self._classify_pairs()

    def _classify_pairs(self) -> dict[tuple[int, int], str]:
        """Tag each pair index as 'spatial', 'parameter', or 'cross'."""
        out = {}
        for i, j in self.model.pair_indices:
            i_is_x = i in self.x_axes
            j_is_x = j in self.x_axes
            if i_is_x and j_is_x:
                out[(i, j)] = "spatial"
            elif (not i_is_x) and (not j_is_x):
                out[(i, j)] = "parameter"
            else:
                out[(i, j)] = "cross"
        return out

    # --- training -----------------------------------------------------------

    def fit(
        self,
        x_samples: torch.Tensor,
        lambda_samples: torch.Tensor,
        u_targets: torch.Tensor,
        phase1_epochs: int = 40,
        phase2_epochs: int = 80,
        phase3_epochs: int | None = None,
        batch_size: int = 4096,
        lr1: float = 1e-3,
        lr2: float = 5e-4,
        log_every: int = 20,
        verbose: bool = True,
    ) -> list[dict]:
        """Train per-subset MLPs to reconstruct u on the joint samples.

        Phase 1 trains mains only (purified). Phase 2 adds pair terms.

        x_samples       : (N, dim_x)
        lambda_samples  : (N, dim_lambda)
        u_targets       : (N,)  — scalar target u(x, lambda)
        """
        assert x_samples.shape[1] == self.dim_x
        assert lambda_samples.shape[1] == self.dim_lambda
        assert u_targets.shape[0] == x_samples.shape[0]

        # Concatenate into a single (N, d) batch sampled from p(x) * p(lambda)
        z = torch.cat([x_samples, lambda_samples], dim=1).to(self.device)
        y = u_targets.to(self.device)
        y_mean = y.mean()
        y_centered = y - y_mean

        loader = DataLoader(
            TensorDataset(z, y_centered),
            batch_size=batch_size,
            shuffle=True,
        )

        history: list[dict] = []

        def _forward(zb, include_pairs, include_triplet):
            if self._has_triplet:
                pred, *_ = self.model(zb, include_pairs=include_pairs,
                                      include_triplet=include_triplet, purify=True)
            else:
                pred, _, _ = self.model(zb, include_pairs=include_pairs, purify=True)
            return pred

        # Phase 1: mains only
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr1, weight_decay=1e-6)
        for ep in range(1, phase1_epochs + 1):
            self.model.train()
            for zb, yb in loader:
                pred = _forward(zb, include_pairs=False, include_triplet=False)
                loss = F.mse_loss(pred, yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
            if verbose and (ep % log_every == 0 or ep == phase1_epochs):
                with torch.no_grad():
                    pred = _forward(z, include_pairs=False, include_triplet=False)
                    rel = (torch.sqrt(F.mse_loss(pred, y_centered)) / y_centered.std()).item()
                history.append({"phase": 1, "ep": ep, "rel": rel})
                print(f"  [phase1] ep={ep:>3}  rel-RMSE={rel:.4f}")

        # Phase 2: mains + pairs
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr2, weight_decay=1e-6)
        for ep in range(1, phase2_epochs + 1):
            self.model.train()
            for zb, yb in loader:
                pred = _forward(zb, include_pairs=True, include_triplet=False)
                loss = F.mse_loss(pred, yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
            if verbose and (ep % log_every == 0 or ep == phase2_epochs):
                with torch.no_grad():
                    pred = _forward(z, include_pairs=True, include_triplet=False)
                    rel = (torch.sqrt(F.mse_loss(pred, y_centered)) / y_centered.std()).item()
                history.append({"phase": 2, "ep": ep, "rel": rel})
                print(f"  [phase2] ep={ep:>3}  rel-RMSE={rel:.4f}")

        # Phase 3 (optional): mains + pairs + triplet
        if self._has_triplet:
            p3_eps = phase3_epochs if phase3_epochs is not None else phase2_epochs
            opt = torch.optim.AdamW(self.model.parameters(), lr=lr2, weight_decay=1e-6)
            for ep in range(1, p3_eps + 1):
                self.model.train()
                for zb, yb in loader:
                    pred = _forward(zb, include_pairs=True, include_triplet=True)
                    loss = F.mse_loss(pred, yb)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    opt.step()
                if verbose and (ep % log_every == 0 or ep == p3_eps):
                    with torch.no_grad():
                        pred = _forward(z, include_pairs=True, include_triplet=True)
                        rel = (torch.sqrt(F.mse_loss(pred, y_centered)) / y_centered.std()).item()
                    history.append({"phase": 3, "ep": ep, "rel": rel})
                    print(f"  [phase3] ep={ep:>3}  rel-RMSE={rel:.4f}")

        # Cache the offset so evaluate_terms can recover it
        self.y_mean = y_mean.detach()
        return history

    # --- evaluation ---------------------------------------------------------

    def evaluate_terms(self, x: torch.Tensor, lambda_: torch.Tensor) -> dict:
        """Evaluate every learned ANOVA term on a batch of joint samples.

        Returns a dict with:
            f0:              scalar baseline
            main_terms:      (N, d) — column k is f_k(z_k)
            pair_terms:      (N, n_pairs) — column k is f_{ij}(z_i, z_j)
            pair_indices:    list[(i, j)]
            variances:       dict subset -> variance over the input batch
            sobol:           dict subset -> Sobol index (variance / total)
        """
        assert x.shape[1] == self.dim_x
        assert lambda_.shape[1] == self.dim_lambda
        z = torch.cat([x, lambda_], dim=1).to(self.device)
        self.model.eval()
        with torch.no_grad():
            if self._has_triplet:
                f0, m, p, t = self.model.evaluate_terms(z, include_triplet=True)
                t_np = t.detach().cpu().numpy()
            else:
                f0, m, p = self.model.evaluate_terms(z)
                t_np = None
        m_np = m.detach().cpu().numpy()
        p_np = p.detach().cpu().numpy() if p.numel() else np.zeros((z.shape[0], 0))
        variances = {(k,): float(m_np[:, k].var()) for k in range(self.d)}
        for k, (i, j) in enumerate(self.model.pair_indices):
            variances[(i, j)] = float(p_np[:, k].var())
        if t_np is not None:
            variances[tuple(range(self.d))] = float(t_np.var())
        total = sum(variances.values())
        sobol = {key: (v / total if total > 0 else 0.0) for key, v in variances.items()}
        return {
            "f0": float(f0.item()) + float(self.y_mean.item()),
            "main_terms": m_np,
            "pair_terms": p_np,
            "triplet_terms": t_np,
            "pair_indices": list(self.model.pair_indices),
            "variances": variances,
            "sobol": sobol,
        }

    # --- the three canonical signatures ------------------------------------

    def spatial_main_effect(self, x_grid: torch.Tensor) -> np.ndarray:
        """u_x(x) on a fixed x grid: aggregate spatial mains + spatial-spatial pairs.

        Cross terms are excluded (those go into u_{x, lambda}).
        Parameter-axes contribute nothing because we hold them at their
        marginal mean (zero, after purification of the mains).
        """
        N = x_grid.shape[0]
        zero_lambda = torch.zeros(N, self.dim_lambda)
        z = torch.cat([x_grid.cpu(), zero_lambda], dim=1).to(self.device)
        self.model.eval()
        with torch.no_grad():
            _, m, _ = self.model.evaluate_terms(z)
            spatial_main = m[:, list(self.x_axes)].sum(dim=1)
            # Add spatial-spatial pair contributions
            pair_idx = [k for k, (i, j) in enumerate(self.model.pair_indices)
                        if self.pair_classification[(i, j)] == "spatial"]
            if pair_idx:
                _, _, p = self.model.evaluate_terms(z)
                spatial_pair = p[:, pair_idx].sum(dim=1)
                return (spatial_main + spatial_pair).cpu().numpy()
            return spatial_main.cpu().numpy()

    def parameter_main_effect(self, lambda_grid: torch.Tensor) -> np.ndarray:
        """u_lambda(lambda) on a parameter grid."""
        N = lambda_grid.shape[0]
        zero_x = torch.zeros(N, self.dim_x)
        z = torch.cat([zero_x, lambda_grid.cpu()], dim=1).to(self.device)
        self.model.eval()
        with torch.no_grad():
            _, m, _ = self.model.evaluate_terms(z)
            param_main = m[:, list(self.lambda_axes)].sum(dim=1)
            pair_idx = [k for k, (i, j) in enumerate(self.model.pair_indices)
                        if self.pair_classification[(i, j)] == "parameter"]
            if pair_idx:
                _, _, p = self.model.evaluate_terms(z)
                param_pair = p[:, pair_idx].sum(dim=1)
                return (param_main + param_pair).cpu().numpy()
            return param_main.cpu().numpy()

    def cross_effect(self, x_grid: torch.Tensor, lambda_grid: torch.Tensor) -> np.ndarray:
        """u_{x, lambda}(x, lambda) on a joint grid: only cross pair terms.

        x_grid: (N, dim_x), lambda_grid: (N, dim_lambda) — same N (one
        joint sample per row).
        """
        assert x_grid.shape[0] == lambda_grid.shape[0]
        z = torch.cat([x_grid.cpu(), lambda_grid.cpu()], dim=1).to(self.device)
        self.model.eval()
        with torch.no_grad():
            if self._has_triplet:
                _, _, p, _ = self.model.evaluate_terms(z, include_triplet=True)
            else:
                _, _, p = self.model.evaluate_terms(z)
            cross_idx = [k for k, (i, j) in enumerate(self.model.pair_indices)
                         if self.pair_classification[(i, j)] == "cross"]
            if not cross_idx:
                return np.zeros(z.shape[0])
            return p[:, cross_idx].sum(dim=1).cpu().numpy()
