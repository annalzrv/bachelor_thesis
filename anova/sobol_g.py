"""Sobol G-function: canonical d-scalable sensitivity benchmark.

f(x) = prod_i (|4*x_i - 2| + a_i) / (1 + a_i),   x in [0, 1]^d

Closed-form Sobol indices:
  V_i = 1 / (3 * (1 + a_i)^2)
  for subset u: V_u = prod_{i in u} V_i
  total variance: V_total = prod_i (1 + V_i) - 1

The factor for input i has mean 1 (by construction of the (1 + a_i)
normaliser) and variance V_i. Because factors are independent and
mean-1, every non-empty subset contributes a non-zero ANOVA term:
the function exercises the full 2^d - 1 ANOVA structure.
"""

import itertools
import numpy as np


def g_function(x, a):
    """Sobol G. x: (N, d), a: (d,). Returns (N,)."""
    factors = (np.abs(4.0 * x - 2.0) + a) / (1.0 + a)
    return np.prod(factors, axis=1)


def variance_per_input(a):
    """V_i = 1 / (3 * (1 + a_i)^2)."""
    return 1.0 / (3.0 * (1.0 + np.asarray(a)) ** 2)


def total_variance(a):
    V_i = variance_per_input(a)
    return float(np.prod(1.0 + V_i) - 1.0)


def sobol_indices(a, max_order=None):
    """All non-empty Sobol indices up to max_order (default = d).

    Returns a dict mapping tuple (sorted indices, 0-based) -> S_u.
    """
    d = len(a)
    V_i = variance_per_input(a)
    V_T = total_variance(a)
    indices = {}
    cap = d if max_order is None else max_order
    for r in range(1, cap + 1):
        for s in itertools.combinations(range(d), r):
            V_s = float(np.prod(V_i[list(s)]))
            indices[s] = V_s / V_T
    return indices


def main_effect_sobol(a):
    V_i = variance_per_input(a)
    V_T = total_variance(a)
    return V_i / V_T


def pair_sobol(a):
    """Returns a dict (i, j) -> S_{ij} for all i < j."""
    d = len(a)
    V_i = variance_per_input(a)
    V_T = total_variance(a)
    out = {}
    for i in range(d):
        for j in range(i + 1, d):
            out[(i, j)] = float(V_i[i] * V_i[j] / V_T)
    return out


def sum_of_low_order_indices(a, max_order):
    """Fraction of total variance captured by interactions up to max_order."""
    s = 0.0
    for u, val in sobol_indices(a, max_order=max_order).items():
        s += val
    return s


if __name__ == "__main__":
    # Standard Saltelli-Sobol benchmark parameters.
    # First 5 inputs are active (small a), rest are inert.
    d = 20
    a = np.array([0, 1, 4.5, 9, 99] + [99] * (d - 5), dtype=np.float32)
    print(f"d = {d}")
    print(f"a = {a.tolist()}")
    print(f"V_total = {total_variance(a):.6f}")
    main = main_effect_sobol(a)
    print("\nMain-effect Sobol indices (top 10):")
    order = np.argsort(-main)
    for idx in order[:10]:
        print(f"  S_{idx+1:2d} = {main[idx]:.4f}   (a = {a[idx]})")
    print(f"\nSum of order-1 indices: {main.sum():.4f}")
    print(f"Sum of orders 1+2:      {sum_of_low_order_indices(a, 2):.4f}")
    print(f"Sum of orders 1+2+3:    {sum_of_low_order_indices(a, 3):.4f}")
    print(f"Sum of all orders (should be 1): {sum_of_low_order_indices(a, d):.4f}")

    # Quick sanity: empirical variance ≈ analytic
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(200_000, d)).astype(np.float32)
    y = g_function(X, a)
    print(f"\nEmpirical variance (200k samples): {y.var():.6f}")
    print(f"Analytic variance:                 {total_variance(a):.6f}")
