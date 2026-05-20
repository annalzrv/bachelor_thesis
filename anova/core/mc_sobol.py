"""Monte-Carlo Sobol' sensitivity analysis (Saltelli/Jansen estimators).

Goal: an independent, model-agnostic baseline for sensitivity indices.
Given any black-box function $u: \\mathbb{R}^d \\to \\mathbb{R}$ and a
prior $p(\\mathbf{x})$ on the inputs, compute:

  S_i   — first-order Sobol' index (variance fraction explained by X_i alone)
  ST_i  — total Sobol' index (variance from X_i, including all interactions)
  S_ij  — pure pair interaction (variance from X_i × X_j minus first-order)
  S_uvw — for d=3, the residual triple interaction

This is the gold standard. Our Fourier HDMR's Sobol numbers should agree
with these in the asymptotic-N limit.

Cost: 2 + d + C(d,2) model evaluations of N samples each. For d=3 that's
8 N total. With N = 50_000 on a small PINN that's ~30 seconds on MPS.
"""

from __future__ import annotations

import itertools
from typing import Callable

import numpy as np


def mc_sobol_full(
    model_fn: Callable[[np.ndarray], np.ndarray],
    sampler: Callable[[np.random.Generator, int], np.ndarray],
    N: int = 20_000,
    d: int = 3,
    seed: int = 42,
) -> dict:
    """Compute Sobol' indices via Monte-Carlo.

    Parameters
    ----------
    model_fn : (N, d) -> (N,)
        Evaluates u(z) for a batch of joint samples z.
    sampler  : (rng, n) -> (n, d)
        Returns n i.i.d. samples from the joint prior.
    N : int
        Number of MC samples per matrix. Total model calls = (d + d_pairs + 2) * N.
    d : int
        Input dimension.
    seed : int
        RNG seed.

    Returns
    -------
    Dict with:
        var_Y       : total variance (denominator of Sobol indices)
        S_first     : { (i,) -> S_i }                     (first-order)
        S_total     : { (i,) -> ST_i }                    (total)
        S_pair      : { (i, j) -> V_ij / Var(Y) }         (pure pair interaction)
        S_triplet   : float, only meaningful for d == 3   (residual triple)
    """
    rng = np.random.default_rng(seed)
    A = sampler(rng, N).astype(np.float64)
    B = sampler(rng, N).astype(np.float64)

    Y_A = np.asarray(model_fn(A), dtype=np.float64)
    Y_B = np.asarray(model_fn(B), dtype=np.float64)
    Y_all = np.concatenate([Y_A, Y_B])
    f0 = Y_all.mean()
    var_Y = Y_all.var()

    if var_Y <= 0:
        raise ValueError(f"Var(Y) is non-positive ({var_Y}); model output is constant?")

    # First-order V_i  (Sobol' 1993; Y_B Y_{C^(i)} shares column i, others independent)
    # and total ST_i  (Jansen 1999, more numerically stable)
    # Note: C^(i) = A with column i replaced by B's column i.
    # So Y_B and Y_{C^(i)} share their column-i values (both come from B).
    V_i: dict[tuple[int, ...], float] = {}
    V_T_i: dict[tuple[int, ...], float] = {}
    Y_C_single: dict[int, np.ndarray] = {}
    for i in range(d):
        C = A.copy()
        C[:, i] = B[:, i]
        Y_C = np.asarray(model_fn(C), dtype=np.float64)
        Y_C_single[i] = Y_C
        V_i[(i,)] = float(np.mean(Y_B * Y_C) - f0 * f0)         # Var(E[Y|X_i])
        V_T_i[(i,)] = float(0.5 * np.mean((Y_A - Y_C) ** 2))    # Total

    # Closed pair indices V_c({i, j}) = Var(E[Y | X_i, X_j])
    V_closed: dict[tuple[int, int], float] = {}
    pair_indices = list(itertools.combinations(range(d), 2))
    for i, j in pair_indices:
        C = A.copy()
        C[:, i] = B[:, i]
        C[:, j] = B[:, j]
        Y_C = np.asarray(model_fn(C), dtype=np.float64)
        V_closed[(i, j)] = float(np.mean(Y_B * Y_C) - f0 * f0)

    # Pure pair interaction V_{ij}^pure = V_c({i,j}) - V_i - V_j
    V_pair = {
        (i, j): V_closed[(i, j)] - V_i[(i,)] - V_i[(j,)]
        for i, j in pair_indices
    }

    # Residual triple (only meaningful for d=3): everything not in lower orders
    sum_first = sum(V_i.values())
    sum_pair = sum(V_pair.values())
    V_triplet = var_Y - sum_first - sum_pair  # equals V_{0,1,...,d-1} only when d == 3

    return {
        "var_Y": float(var_Y),
        "f0": float(f0),
        "N": N,
        "S_first": {k: v / var_Y for k, v in V_i.items()},
        "S_total": {k: v / var_Y for k, v in V_T_i.items()},
        "S_pair":  {k: v / var_Y for k, v in V_pair.items()},
        "S_triplet": float(V_triplet / var_Y) if d == 3 else None,
    }


def uniform_unit_box_sampler(rng: np.random.Generator, n: int, d: int = 3) -> np.ndarray:
    """Joint U[0, 1]^d sampler — useful for synthetic validation."""
    return rng.uniform(0.0, 1.0, size=(n, d)).astype(np.float32)


def helmholtz_2d_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sampler for 2D Helmholtz LC-PINN: U[0,1]^2 (spatial) x U[-1, 1] (k_norm)."""
    z = np.empty((n, 3), dtype=np.float32)
    z[:, 0:2] = rng.uniform(0.0, 1.0, size=(n, 2)).astype(np.float32)
    z[:, 2] = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
    return z
