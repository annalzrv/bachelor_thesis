"""Aggregate all overnight results JSONs into a master summary table.

Walks lc_anova/results/, parses every JSON, and produces:
- A single markdown table per PDE collating seeds and metrics
- Cross-PDE comparison table
- Multi-seed mean ± std summary

Output: lc_anova/results/MASTER_SUMMARY.md
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    results_dir = Path(__file__).resolve().parent / "results"
    out = ["# Master summary of LC-PINN × ANOVA experiments\n"]
    out.append(f"_Auto-generated from `{results_dir.name}/` JSON files._\n")

    # --- Collect ---
    runs = defaultdict(list)  # category -> list of (tag, payload)
    for path in sorted(results_dir.glob("results_*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        tag = path.stem.replace("results_", "")
        # Skip smoke tests and legacy non-canonical runs from earlier today.
        if "smoke" in tag:
            continue
        # Categorize. Canonical 2D Helm runs are Fourier with phase3-400 epochs.
        if "mc_sobol_seed" in tag:
            runs["helm2d_mc_sobol"].append((tag, payload))
        elif tag.startswith("helm1d_seed"):
            runs["helm1d_hdmr"].append((tag, payload))
        elif tag.startswith("schr1d_seed"):
            runs["schr1d_hdmr"].append((tag, payload))
        elif tag.startswith("helm2d_fourier_seed") or "fourier_h128_L6_p3eps400_seed" in tag:
            runs["helm2d_hdmr"].append((tag, payload))
        elif tag.startswith("order3") or tag.startswith("lc_pinn_helmholtz_2d") or "fourier_h128_L6_seed" in tag or "fourier_h64" in tag:
            runs["helm2d_hdmr_ablation"].append((tag, payload))

    # --- 1D Helmholtz ---
    if runs["helm1d_hdmr"]:
        out.append("\n## 1D Helmholtz — Fourier joint HDMR (d=2)\n")
        out.append("| seed-tag | LC-PINN rel-L² | HDMR val rel-RMSE | S_x | S_k | S_{x,k} |")
        out.append("|---|---|---|---|---|---|")
        s_xk_vals = []
        for tag, p in runs["helm1d_hdmr"]:
            s = p.get("sobol_indices", {})
            row = f"| {tag} | {p.get('lc_pinn_vs_reference_rel_l2', float('nan')):.4f} | "
            row += f"{p.get('jointhdmr_val_rel_rmse', float('nan')):.4f} | "
            row += f"{s.get('x', float('nan')):.3f} | {s.get('k', float('nan')):.3f} | {s.get('x/k', float('nan')):.3f} |"
            out.append(row)
            if 'x/k' in s:
                s_xk_vals.append(s['x/k'])
        if s_xk_vals:
            out.append(f"\n**Cross-pair Sobol $S_{{x, k}}$:** mean={np.mean(s_xk_vals):.4f}  std={np.std(s_xk_vals):.4f}  (across {len(s_xk_vals)} seeds)")

    # --- Schrödinger 1D ---
    if runs["schr1d_hdmr"]:
        out.append("\n## Schrödinger 1D — Fourier joint HDMR (d=2)\n")
        out.append("| seed-tag | LC-PINN rel-L² | HDMR val rel-RMSE | S_x | S_α | S_{x,α} |")
        out.append("|---|---|---|---|---|---|")
        s_xa_vals = []
        s_x_vals = []
        for tag, p in runs["schr1d_hdmr"]:
            s = p.get("sobol_indices", {})
            row = f"| {tag} | {p.get('lc_pinn_vs_reference_rel_l2', float('nan')):.4f} | "
            row += f"{p.get('jointhdmr_val_rel_rmse', float('nan')):.4f} | "
            row += f"{s.get('x', float('nan')):.3f} | {s.get('alpha', float('nan')):.3f} | {s.get('x/alpha', float('nan')):.3f} |"
            out.append(row)
            if 'x' in s: s_x_vals.append(s['x'])
            if 'x/alpha' in s: s_xa_vals.append(s['x/alpha'])
        if s_x_vals:
            out.append(f"\n**Spatial main $S_x$:** mean={np.mean(s_x_vals):.4f}  std={np.std(s_x_vals):.4f}")
        if s_xa_vals:
            out.append(f"**Cross-pair $S_{{x, \\alpha}}$:** mean={np.mean(s_xa_vals):.4f}  std={np.std(s_xa_vals):.4f}  (across {len(s_xa_vals)} seeds)")

    # --- 2D Helmholtz HDMR ---
    if runs["helm2d_hdmr"]:
        out.append("\n## 2D Helmholtz — Fourier joint HDMR (d=3, order-3)\n")
        out.append("| tag | val rel-RMSE | S_x | S_y | S_k | S_{x,y} | S_{x,k} | S_{y,k} | **S_{x,y,k}** |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        s_xyk_vals = []
        for tag, p in runs["helm2d_hdmr"]:
            s = p.get("sobol_indices", {})
            row = f"| {tag} | {p.get('jointhdmr_val_rel_rmse', float('nan')):.4f} | "
            for key in ["x", "y", "k", "x/y", "x/k", "y/k", "x/y/k"]:
                row += f"{s.get(key, float('nan')):.3f} | "
            out.append(row[:-1] + "|")  # close last cell
            if 'x/y/k' in s:
                s_xyk_vals.append(s['x/y/k'])
        if s_xyk_vals:
            out.append(f"\n**Triplet $S_{{x,y,k}}$ (HDMR-normalised):** mean={np.mean(s_xyk_vals):.4f}  std={np.std(s_xyk_vals):.4f}")

    # --- 2D Helmholtz MC-Sobol ---
    if runs["helm2d_mc_sobol"]:
        out.append("\n## 2D Helmholtz — MC-Sobol (gold standard, d=3)\n")
        out.append("| seed-tag | S_x | S_y | S_k | S_{x,y} | S_{x,k} | S_{y,k} | **S_{x,y,k}** |")
        out.append("|---|---|---|---|---|---|---|---|")
        triplet_mc = []
        for tag, p in runs["helm2d_mc_sobol"]:
            s_first = p.get("S_first", {})
            s_pair = p.get("S_pair", {})
            s_trip = p.get("S_triplet", float('nan'))
            row = f"| {tag} | {s_first.get('x', 0):.3f} | {s_first.get('y', 0):.3f} | {s_first.get('k', 0):.3f} | "
            row += f"{s_pair.get('x,y', 0):.3f} | {s_pair.get('x,k', 0):.3f} | {s_pair.get('y,k', 0):.3f} | "
            row += f"{s_trip:.3f} |"
            out.append(row)
            triplet_mc.append(s_trip)
        if triplet_mc:
            out.append(f"\n**MC triplet $S_{{x,y,k}}$:** mean={np.mean(triplet_mc):.4f}  std={np.std(triplet_mc):.4f}")

    # Save
    summary_path = results_dir / "MASTER_SUMMARY.md"
    summary_path.write_text("\n".join(out) + "\n")
    print(f"Wrote {summary_path}")
    print(f"  helm1d HDMR runs: {len(runs['helm1d_hdmr'])}")
    print(f"  schr1d HDMR runs: {len(runs['schr1d_hdmr'])}")
    print(f"  helm2d HDMR runs: {len(runs['helm2d_hdmr'])}")
    print(f"  helm2d MC-Sobol runs: {len(runs['helm2d_mc_sobol'])}")


if __name__ == "__main__":
    main()
