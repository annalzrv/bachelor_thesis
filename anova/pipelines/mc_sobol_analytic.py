"""MC-Sobol on the analytic 2D Helmholtz reference solution.

Establishes the "ground-truth" Sobol indices for u_ref(x, y, k) =
sin(pi x) sin(pi y) cos(kx) cos(ky) on [0,1]^2 x [1, 10], independent
of any trained LC-PINN. Differences from the per-seed LC-PINN MC-Sobol
numbers tell us how faithfully the LC-PINN reproduces the underlying
ANOVA structure.

Run:
    cd /Users/anna/Desktop/research/anova
    python -m lc_anova.pipelines.mc_sobol_analytic
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ANOVA = _HERE.parent
_REPO_THESIS_CODE = _REPO_ANOVA.parent / "thesis" / "code"
if str(_REPO_ANOVA) not in sys.path:
    sys.path.insert(0, str(_REPO_ANOVA))
if str(_REPO_THESIS_CODE) not in sys.path:
    sys.path.insert(0, str(_REPO_THESIS_CODE))

from lc_anova.core.mc_sobol import mc_sobol_full, helmholtz_2d_sampler  # noqa: E402
from pinns.equations import helmholtz_2d as helm  # noqa: E402


def main(N: int = 50_000, seed: int = 42):
    print(f"== MC-Sobol on analytic Helmholtz reference  N={N}  seed={seed} ==")

    def model_fn(z_np: np.ndarray) -> np.ndarray:
        # z_np: (N, 3) = (x, y, k_norm)
        k = helm.norm_to_k(z_np[:, 2])
        return helm.reference_solution(z_np[:, 0], z_np[:, 1], k)

    out = mc_sobol_full(model_fn, helmholtz_2d_sampler, N=N, d=3, seed=seed)

    name_for = {0: "x", 1: "y", 2: "k"}
    print(f"\nVar(Y) (analytic): {out['var_Y']:.6f}")
    print(f"f0 (mean):         {out['f0']:.6f}")

    print("\nFirst-order Sobol:")
    for k in [(0,), (1,), (2,)]:
        print(f"  {name_for[k[0]]:<3}  {out['S_first'][k]:>9.4f}")

    print("\nTotal Sobol:")
    for k in [(0,), (1,), (2,)]:
        print(f"  {name_for[k[0]]:<3}  {out['S_total'][k]:>9.4f}")

    print("\nPure pair Sobol:")
    for k in [(0, 1), (0, 2), (1, 2)]:
        n = "/".join(name_for[a] for a in k)
        print(f"  {n:<5}  {out['S_pair'][k]:>9.4f}")

    print(f"\nTriplet (residual): S_(x,y,k) = {out['S_triplet']:.4f}")


if __name__ == "__main__":
    main()
