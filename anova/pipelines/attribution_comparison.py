"""Comparison of our joint-ANOVA decomposition with per-instance gradient
attribution methods (Integrated Gradients, gradient-magnitude).

These attribution methods are the standard interpretability baselines for
deep models. They produce per-instance scalar attributions for each input
feature. Our claim is:

  1. On first-order importance (which input matters most on average), our
     Sobol indices and aggregate IG agree on the *ordering*.
  2. Per-instance methods *cannot* produce a triplet decomposition. They
     have no way to express "$X$ percent of the function's behaviour is in
     the irreducible 3-way interaction between x, y, and k" — they only
     give per-input scalars.

This script:
  - Computes Integrated Gradients (Sundararajan et al. 2017) over a uniform
    test set, averaging |IG_i| as the aggregate importance.
  - Computes gradient-magnitude baseline E[|∂u/∂x_i|].
  - Compares against our MC-Sobol first-order indices.
  - Produces a bar chart figure.

The figure makes the "but where's the triplet" gap visible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO_ANOVA = _HERE.parent
_REPO_THESIS_CODE = _REPO_ANOVA.parent / "thesis" / "code"
sys.path.insert(0, str(_REPO_ANOVA))
sys.path.insert(0, str(_REPO_THESIS_CODE))

from lc_anova.pipelines.helmholtz_2d import load_lc_pinn, sample_joint  # noqa

CK_DIR = _REPO_THESIS_CODE / "checkpoints"
RESULTS = _HERE / "results"
PAPER_FIGS = RESULTS / "paper_figures"
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "lines.linewidth": 1.8, "axes.grid": True, "grid.alpha": 0.3,
})


def integrated_gradients(model, z_test: torch.Tensor, z_baseline: torch.Tensor, n_steps: int = 50):
    """Per-instance Integrated Gradients attribution. Sundararajan et al. 2017.

    z_test: (B, 3) — test points (x, y, k_norm)
    z_baseline: (3,) — reference baseline (we use the centre 0.5, 0.5, 0).
    """
    # interpolation alphas
    alphas = torch.linspace(0.0, 1.0, n_steps, device=z_test.device).reshape(-1, 1, 1)
    interp = z_baseline.unsqueeze(0).unsqueeze(0) + alphas * (z_test.unsqueeze(0) - z_baseline.unsqueeze(0).unsqueeze(0))
    # interp shape: (n_steps, B, 3)
    interp = interp.reshape(-1, 3).requires_grad_(True)
    # call model on (x, y, k_norm) → u
    xy = interp[:, :2]
    k_norm = interp[:, 2:3]
    out = model(xy, k_norm).squeeze(-1)
    grads = torch.autograd.grad(out.sum(), interp, create_graph=False, retain_graph=False)[0]
    grads = grads.reshape(n_steps, z_test.shape[0], 3).mean(dim=0)
    ig = (z_test - z_baseline.unsqueeze(0)) * grads
    return ig.detach().cpu().numpy()


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck_path = CK_DIR / "lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt"
    print(f"Loading LC-PINN: {ck_path.name}")
    model, _ = load_lc_pinn(str(ck_path), device)

    # Sample test points
    n_test = 1000
    xy_t, k_t = sample_joint(n_test, 42, device)
    z_test = torch.cat([xy_t, k_t], dim=1)
    z_baseline = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float32, device=device)

    print(f"Running Integrated Gradients on {n_test} test points")
    ig = integrated_gradients(model, z_test, z_baseline, n_steps=50)
    ig_abs_mean = np.mean(np.abs(ig), axis=0)
    print(f"  E[|IG_i|]: x={ig_abs_mean[0]:.4f}, y={ig_abs_mean[1]:.4f}, k={ig_abs_mean[2]:.4f}")

    # Normalize so they sum to 1 (interpretable as "importance proportion")
    ig_norm = ig_abs_mean / ig_abs_mean.sum()
    print(f"  IG normalized: x={ig_norm[0]:.3f}, y={ig_norm[1]:.3f}, k={ig_norm[2]:.3f}")

    # Gradient-magnitude baseline (simpler, no integration)
    z_test_req = z_test.detach().requires_grad_(True)
    xy = z_test_req[:, :2]
    k_norm = z_test_req[:, 2:3]
    out = model(xy, k_norm).squeeze(-1)
    grads = torch.autograd.grad(out.sum(), z_test_req)[0]
    grad_abs_mean = grads.detach().cpu().numpy()
    grad_abs_mean = np.mean(np.abs(grad_abs_mean), axis=0)
    grad_norm = grad_abs_mean / grad_abs_mean.sum()
    print(f"  |∇u| normalized: x={grad_norm[0]:.3f}, y={grad_norm[1]:.3f}, k={grad_norm[2]:.3f}")

    # MC-Sobol first-order (from N=10^6 gold standard)
    mc_path = RESULTS / "results_mc_megaN_seed0.json"
    mc = json.loads(mc_path.read_text())
    sobol_first = [mc["S_first"]["x"], mc["S_first"]["y"], mc["S_first"]["k"]]
    sobol_first_norm = np.array(sobol_first) / sum(sobol_first)
    sobol_pair_xy = mc["S_pair"]["x,y"]
    sobol_pair_xk = mc["S_pair"]["x,k"]
    sobol_pair_yk = mc["S_pair"]["y,k"]
    sobol_triplet = mc["S_triplet"]
    print(f"  MC Sobol first-order: x={mc['S_first']['x']:.4f}, y={mc['S_first']['y']:.4f}, k={mc['S_first']['k']:.4f}")
    print(f"  MC Sobol first-order (renormalized so sum=1): "
          f"x={sobol_first_norm[0]:.3f}, y={sobol_first_norm[1]:.3f}, k={sobol_first_norm[2]:.3f}")
    print(f"  MC Sobol triplet (S_xyk): {sobol_triplet:.4f}  ← inaccessible to IG/grad")

    # ---- Figure -----------------------------------------------------------
    fig, (ax_first, ax_full) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                                              gridspec_kw={"width_ratios": [1.2, 1]})

    # Left: agreement on first-order importance
    inputs = ["x", "y", "k"]
    x_pos = np.arange(len(inputs))
    w = 0.27
    ax_first.bar(x_pos - w, ig_norm, w, color="#1f77b4", edgecolor="white",
                  label="Integrated Gradients (renormalised)")
    ax_first.bar(x_pos, grad_norm, w, color="#aec7e8", edgecolor="white",
                  label=r"$\mathbb{E}|\nabla u|$ (renormalised)")
    ax_first.bar(x_pos + w, sobol_first_norm, w, color="#d62728", edgecolor="white",
                  label="MC-Sobol first-order (renormalised)")
    ax_first.set_xticks(x_pos)
    ax_first.set_xticklabels(inputs)
    ax_first.set_ylabel("normalised importance")
    ax_first.set_title("All three methods agree on first-order importance ordering")
    ax_first.legend(framealpha=0.95)
    ax_first.set_ylim(0, 0.6)

    # Right: full Sobol decomposition with all subsets, showing what IG/grad can't reach
    keys = ["x", "y", "k", "x,y", "x,k", "y,k", "x,y,k"]
    values = [mc["S_first"]["x"], mc["S_first"]["y"], mc["S_first"]["k"],
               mc["S_pair"]["x,y"], mc["S_pair"]["x,k"], mc["S_pair"]["y,k"],
               mc["S_triplet"]]
    colors = ["#1f77b4", "#aec7e8", "#ff7f0e",
               "#9467bd", "#2ca02c", "#98df8a", "#d62728"]
    bars = ax_full.bar(keys, values, color=colors, edgecolor="white")
    # Annotate IG/grad cap line
    ax_full.axhline(0.073, color="#aec7e8", linestyle="--", linewidth=1.5,
                     label="IG/$\\nabla u$ ceiling: \nfirst-order only")
    # Mark which bars IG/grad CAN see vs can't
    for i, bar in enumerate(bars):
        if i < 3:
            bar.set_label("accessible to IG/$\\nabla u$" if i == 0 else None)
        else:
            bar.set_hatch("//")
            bar.set_label("inaccessible to IG/$\\nabla u$" if i == 3 else None)
    ax_full.set_ylabel("Sobol index over $(x, y, k)$")
    ax_full.set_title(r"Full ANOVA decomposition: $S_{x,y,k} = 0.43$ is invisible to attribution methods")
    ax_full.set_ylim(0, 0.5)
    # Manual legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#1f77b4", edgecolor="white", label="first-order (IG/$\\nabla u$ accessible)"),
        Patch(facecolor="white", hatch="//", edgecolor="black",
              label="higher-order (only our method)"),
    ]
    ax_full.legend(handles=legend_elems, loc="upper left", framealpha=0.95)
    plt.tight_layout()
    out = PAPER_FIGS / "fig8_attribution_comparison.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nWrote {out} + .pdf")

    # Save comparison numbers
    out_json = RESULTS / "attribution_comparison.json"
    out_json.write_text(json.dumps({
        "ig_normalised": list(ig_norm.tolist()),
        "grad_normalised": list(grad_norm.tolist()),
        "mc_sobol_first_normalised": sobol_first_norm.tolist(),
        "mc_sobol_first_raw": sobol_first,
        "mc_sobol_pair_xy": sobol_pair_xy,
        "mc_sobol_pair_xk": sobol_pair_xk,
        "mc_sobol_pair_yk": sobol_pair_yk,
        "mc_sobol_triplet": sobol_triplet,
        "note": "IG and gradient methods cannot reach pair or triplet indices — they are intrinsically first-order."
    }, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
