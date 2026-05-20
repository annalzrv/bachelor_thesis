"""Per-k breakdown table for 1D parametric Helmholtz.

Re-evaluates the trained LC-PINN seeds on the same 5-point k-grid that
SA-PINN and ReLoBRaLo used (k ∈ {1.00, 3.25, 5.50, 7.75, 10.00}), so
the per-k comparison is on identical evaluation points. Loads the
existing per_k blocks for SA-PINN and ReLoBRaLo from their JSONs.

Outputs:
    1. Per-k LaTeX table to stdout (paste into results.tex)
    2. results/per_k_breakdown_helmholtz.json with the aligned numbers
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pinns.device import select_device, device_info
from pinns.equations import helmholtz as helm
from pinns.model import LossConditionalPINN

K_GRID = [1.00, 3.25, 5.50, 7.75, 10.00]


def evaluate_lc_seed(ckpt_path: pathlib.Path, device: torch.device) -> dict[float, float]:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = LossConditionalPINN(
        dim_phys=helm.DIM_PHYS, dim_lambda=helm.DIM_LAMBDA,
        hidden_dims=state["hidden_dims"],
        conditioning=state["conditioning"],
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return {k: helm.evaluate_at_k(model, k, device, is_lc=True) for k in K_GRID}


def fmt_sci(x: float) -> str:
    if x == 0:
        return r"$0$"
    sign = "-" if x < 0 else ""
    x = abs(x)
    exp = 0
    while x >= 10:
        x /= 10; exp += 1
    while x < 1:
        x *= 10; exp -= 1
    return rf"${sign}{x:.2f}\!\times\!10^{{{exp}}}$"


def fmt_pm(mean: float, std: float, bold: bool = False) -> str:
    """Compact $(mean \\pm std)\\times 10^{exp}$ in shared-exponent notation."""
    if mean == 0:
        return r"$0$"
    sign = "-" if mean < 0 else ""
    m = abs(mean)
    exp = 0
    while m >= 10:
        m /= 10; exp += 1
    while m < 1:
        m *= 10; exp -= 1
    s = std / (10 ** exp)
    body = rf"({sign}{m:.2f}\!\pm\!{s:.2f})\!\times\!10^{{{exp}}}"
    if bold:
        body = rf"\mathbf{{{body}}}"
    return rf"${body}$"


def main() -> int:
    device = select_device()
    print(f"Device: {device_info(device)}", flush=True)

    ckpt_pat = "checkpoints/lc_pinn_helmholtz_seed{s}_film_lbfgs.pt"
    seeds = [0, 1, 2, 3]

    lc_per_seed: list[dict[float, float]] = []
    for s in seeds:
        p = REPO / ckpt_pat.format(s=s)
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 1
        per_k = evaluate_lc_seed(p, device)
        lc_per_seed.append(per_k)
        print(f"  seed {s}  " + "  ".join(f"k={k}: {per_k[k]:.3e}" for k in K_GRID),
              flush=True)

    lc_table: dict[float, dict[str, float]] = {}
    for k in K_GRID:
        vals = np.array([d[k] for d in lc_per_seed])
        lc_table[k] = {"mean": float(vals.mean()), "std": float(vals.std()),
                       "n_seeds": len(seeds)}

    sa = json.loads((REPO / "results/sa_pinn_helmholtz.json").read_text())["per_k"]
    rb = json.loads((REPO / "results/relobralo_helmholtz.json").read_text())["per_k"]

    rows = []
    for k in K_GRID:
        sa_e = sa[f"{k:.2f}"]
        rb_e = rb[f"{k:.2f}"]
        lc_e = lc_table[k]
        rows.append({
            "k": k,
            "sa_pinn":   fmt_pm(sa_e["mean"],  sa_e["std"]),
            "relobralo": fmt_pm(rb_e["mean"],  rb_e["std"]),
            "lc_pinn":   fmt_pm(lc_e["mean"],  lc_e["std"]),
            "lc_dominates": lc_e["mean"] < min(sa_e["mean"], rb_e["mean"]),
        })

    print()
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Per-$k$ rel-$L^2$ on 1D parametric Helmholtz (mean $\pm$ std over 4 seeds, "
          r"shared exponent omitted in each cell). Bold: best per row.}")
    print(r"\label{tab:helm-per-k}")
    print(r"\begin{tabular}{r|ccc}")
    print(r"\toprule")
    print(r"$k$ & SA-PINN & ReLoBRaLo & LC-PINN \\")
    print(r"\midrule")
    for row in rows:
        sa_e = sa[f"{row['k']:.2f}"]
        rb_e = rb[f"{row['k']:.2f}"]
        lc_e = lc_table[row["k"]]
        winner = min(("sa", sa_e["mean"]), ("rb", rb_e["mean"]), ("lc", lc_e["mean"]),
                     key=lambda x: x[1])[0]
        sa_cell = fmt_pm(sa_e["mean"], sa_e["std"], bold=(winner == "sa"))
        rb_cell = fmt_pm(rb_e["mean"], rb_e["std"], bold=(winner == "rb"))
        lc_cell = fmt_pm(lc_e["mean"], lc_e["std"], bold=(winner == "lc"))
        print(rf"{row['k']:.2f} & {sa_cell} & {rb_cell} & {lc_cell} \\")
    print(r"\midrule")
    sa_mean = np.mean([sa[f"{k:.2f}"]["mean"] for k in K_GRID])
    rb_mean = np.mean([rb[f"{k:.2f}"]["mean"] for k in K_GRID])
    lc_mean = np.mean([lc_table[k]["mean"] for k in K_GRID])
    grid_winner = min(("sa", sa_mean), ("rb", rb_mean), ("lc", lc_mean),
                      key=lambda x: x[1])[0]
    sa_grid = fmt_sci(sa_mean) if grid_winner != "sa" else rf"$\mathbf{{{fmt_sci(sa_mean).strip('$')}}}$"
    rb_grid = fmt_sci(rb_mean) if grid_winner != "rb" else rf"$\mathbf{{{fmt_sci(rb_mean).strip('$')}}}$"
    lc_grid = fmt_sci(lc_mean) if grid_winner != "lc" else rf"$\mathbf{{{fmt_sci(lc_mean).strip('$')}}}$"
    print(rf"\textbf{{grid mean}} & {sa_grid} & {rb_grid} & {lc_grid} \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    out = {
        "k_grid": K_GRID,
        "lc_pinn_per_k": {f"{k:.2f}": lc_table[k] for k in K_GRID},
        "sa_pinn_per_k": {f"{k:.2f}": sa[f"{k:.2f}"] for k in K_GRID},
        "relobralo_per_k": {f"{k:.2f}": rb[f"{k:.2f}"] for k in K_GRID},
        "grid_means": {
            "lc_pinn":   float(lc_mean),
            "sa_pinn":   float(sa_mean),
            "relobralo": float(rb_mean),
        },
    }
    out_path = REPO / "results" / "per_k_breakdown_helmholtz.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path.relative_to(REPO)}", flush=True)

    print("\n--- summary ---")
    for k in K_GRID:
        sa_e = sa[f"{k:.2f}"]["mean"]
        rb_e = rb[f"{k:.2f}"]["mean"]
        lc_e = lc_table[k]["mean"]
        winner = "LC" if lc_e < min(sa_e, rb_e) else ("SA" if sa_e < rb_e else "ReLo")
        print(f"  k={k:5.2f}  LC={lc_e:.3e}  SA={sa_e:.3e}  ReLo={rb_e:.3e}  -> {winner}")
    print(f"  grid-mean  LC={lc_mean:.3e}  SA={sa_mean:.3e}  ReLo={rb_mean:.3e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
