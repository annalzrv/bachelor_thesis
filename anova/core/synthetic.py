"""Synthetic polynomial benchmark with analytic ANOVA terms.

The benchmark function on z in [0, 1]^3 (we treat z_1, z_2 as "spatial"
and z_3 as "parameter" — exactly the structure of 2D Helmholtz after
normalization):

    u(z_1, z_2, z_3) = L1(z_1) + L2(z_2) + L1(z_3) + L1(z_1)*L1(z_3)

where L1(z) = 2z - 1 and L2(z) = 6z^2 - 6z + 1 are shifted Legendre
polynomials on [0, 1]. They are mutually orthogonal under U[0,1] with:

    E[L1] = 0,  Var(L1) = 1/3
    E[L2] = 0,  Var(L2) = 1/5
    E[L1 * L2] = 0

So the analytic functional-ANOVA decomposition is exactly:

    f_0       = 0
    f_1(z_1)  = L1(z_1)                  Var = 1/3
    f_2(z_2)  = L2(z_2)                  Var = 1/5
    f_3(z_3)  = L1(z_3)                  Var = 1/3
    f_12      = 0
    f_13(z_1, z_3) = L1(z_1)*L1(z_3)     Var = 1/9
    f_23      = 0

Total variance V = 1/3 + 1/5 + 1/3 + 1/9 = 44/45.

Sobol indices:
    S_1  = (1/3) / V = 15/44   ≈ 0.34091
    S_2  = (1/5) / V =  9/44   ≈ 0.20455
    S_3  = (1/3) / V = 15/44   ≈ 0.34091
    S_13 = (1/9) / V =  5/44   ≈ 0.11364
    S_12 = S_23 = 0
"""

import numpy as np
import torch


def L1(z):
    """Shifted Legendre L1 on [0, 1]: zero mean, variance 1/3."""
    return 2.0 * z - 1.0


def L2(z):
    """Shifted Legendre L2 on [0, 1]: zero mean, variance 1/5."""
    return 6.0 * z * z - 6.0 * z + 1.0


def u_synthetic(z):
    """Target function on [0,1]^3.

    z: tensor or array of shape (N, 3).
    Returns scalar per sample, shape (N,).
    """
    if isinstance(z, torch.Tensor):
        z1, z2, z3 = z[:, 0], z[:, 1], z[:, 2]
        return L1(z1) + L2(z2) + L1(z3) + L1(z1) * L1(z3)
    z = np.asarray(z, dtype=np.float64)
    z1, z2, z3 = z[:, 0], z[:, 1], z[:, 2]
    return L1(z1) + L2(z2) + L1(z3) + L1(z1) * L1(z3)


# Analytic ANOVA term variances and the corresponding Sobol indices.
ANALYTIC_VARIANCES = {
    (0,):    1.0 / 3.0,   # f_1
    (1,):    1.0 / 5.0,   # f_2
    (2,):    1.0 / 3.0,   # f_3
    (0, 1):  0.0,         # f_12
    (0, 2):  1.0 / 9.0,   # f_13
    (1, 2):  0.0,         # f_23
}

TOTAL_VARIANCE = sum(ANALYTIC_VARIANCES.values())  # 44/45

ANALYTIC_SOBOL = {
    key: var / TOTAL_VARIANCE for key, var in ANALYTIC_VARIANCES.items()
}


def analytic_terms_on_grid(z):
    """Evaluate each analytic ANOVA term on the same grid as z.

    Returns dict: subset_tuple -> array of shape (N,).
    """
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    z = np.asarray(z, dtype=np.float64)
    z1, z2, z3 = z[:, 0], z[:, 1], z[:, 2]
    return {
        (0,):    L1(z1),
        (1,):    L2(z2),
        (2,):    L1(z3),
        (0, 1):  np.zeros_like(z1),
        (0, 2):  L1(z1) * L1(z3),
        (1, 2):  np.zeros_like(z1),
    }
