"""Compare baseline (concat) and FiLM+L-BFGS LC-PINN runs across all four
equation families. Run after the overnight chain finishes — outputs a
markdown summary so the paper update is mechanical.

Usage:
    python scripts/harvest_film_lbfgs.py
"""
from __future__ import annotations

import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
RES = REPO / "results"


def load(name: str) -> dict | None:
    p = RES / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fmt(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.4e}"


def row(label: str, baseline_path: str, treatment_path: str,
        elapsed_min: bool = False) -> str:
    bl = load(baseline_path)
    tr = load(treatment_path)
    if bl is None and tr is None:
        return f"| {label} | (no data) | (no data) | — |"
    bl_mean = bl["summary"]["rel_l2_mean"] if bl else None
    bl_std  = bl["summary"]["rel_l2_std"]  if bl else None
    tr_mean = tr["summary"]["rel_l2_mean"] if tr else None
    tr_std  = tr["summary"]["rel_l2_std"]  if tr else None
    bl_t = bl["summary"]["elapsed_mean_sec"] / 60 if bl else None
    tr_t = tr["summary"]["elapsed_mean_sec"] / 60 if tr else None
    lift = ""
    if bl_mean and tr_mean and bl_mean > 0:
        ratio = bl_mean / tr_mean
        lift = f"{ratio:.2f}× lift" if ratio > 1 else f"{1/ratio:.2f}× regression"
    bl_s = f"{fmt(bl_mean)} ± {fmt(bl_std)}" if bl_mean else "—"
    tr_s = f"{fmt(tr_mean)} ± {fmt(tr_std)}" if tr_mean else "—"
    bl_ts = f"{bl_t:.1f} min" if bl_t else "—"
    tr_ts = f"{tr_t:.1f} min" if tr_t else "—"
    return (f"| {label} | {bl_s} ({bl_ts}) | {tr_s} ({tr_ts}) | {lift} |")


def main() -> None:
    print("# FiLM + L-BFGS sweep: rel-L² lift over baseline LC-PINN\n")
    print("Headline: rel-L² (mean ± std over seeds), wall time per seed.\n")
    print("| Equation | Baseline (concat, no L-BFGS) | FiLM + L-BFGS (w64) | Lift |")
    print("|----------|------------------------------|---------------------|------|")
    print(row("1D Helmholtz",
              "lc_pinn_helmholtz.json",
              "lc_pinn_helmholtz_film_lbfgs.json"))
    print(row("2D Helmholtz (w64)",
              "lc_pinn_helmholtz_2d.json",
              "lc_pinn_helmholtz_2d_film_lbfgs_w64.json"))
    print(row("2D Helmholtz (w128)",
              "lc_pinn_helmholtz_2d.json",
              "lc_pinn_helmholtz_2d_film_lbfgs_w128.json"))
    print(row("Burgers",
              "lc_pinn_burgers_seeds.json",
              "lc_pinn_burgers_film_lbfgs.json"))
    print(row("BL (viscous)",
              "lc_pinn_bl_seeds_viscous.json",
              "lc_pinn_bl_seeds_film_lbfgs.json"))

    print("\n## Schrödinger family (LC vs per-α SA)\n")
    sch = load("lc_pinn_schrodinger_film_lbfgs.json")
    sa = load("sa_pinn_schrodinger.json")
    if sch is not None:
        s = sch["summary"]
        print(f"- **LC-PINN Schrödinger (FiLM+L-BFGS, 4 seeds)**: "
              f"rel-L² {s['rel_l2_mean']:.4e} ± {s['rel_l2_std']:.4e} "
              f"({s['elapsed_mean_sec']/60:.1f} min/seed)")
    else:
        print("- LC-PINN Schrödinger: (no data)")
    if sa is not None:
        s = sa["summary"]
        print(f"- **SA-PINN per-α Schrödinger**: "
              f"rel-L² {s['rel_l2_mean_over_alpha_then_seeds']:.4e} ± "
              f"{s['rel_l2_std_over_seeds']:.4e} "
              f"(α-trains={s['alpha_trains']}, n_seeds={s['n_seeds']})")
        for k, v in sa.get("per_alpha", {}).items():
            print(f"  - α={k}: {v['mean']:.4e} ± {v['std']:.4e}")
    else:
        print("- SA-PINN Schrödinger: (no data)")

    print("\n## PI-DeepONet vs LC-PINN on 1D Helmholtz (residual-only, matched)\n")
    don = load("pi_deeponet_helmholtz_matched.json") or load("pi_deeponet_helmholtz.json")
    if don is not None:
        s = don["summary"]
        print(f"- **PI-DeepONet 1D Helm (4 seeds, residual-only)**: "
              f"rel-L² {s['rel_l2_mean']:.4e} ± {s['rel_l2_std']:.4e} "
              f"({s['elapsed_mean_sec']/60:.1f} min/seed)")
    else:
        print("- PI-DeepONet 1D Helm: (no data)")
    lc1d = load("lc_pinn_helmholtz_film_lbfgs.json")
    if lc1d is not None:
        s = lc1d["summary"]
        print(f"- **LC-PINN 1D Helm (FiLM+L-BFGS)**: "
              f"rel-L² {s['rel_l2_mean']:.4e} ± {s['rel_l2_std']:.4e} "
              f"({s['elapsed_mean_sec']/60:.1f} min/seed)")

    print("\n## Per-seed L-BFGS revert flags\n")
    for tag, path in [
        ("1D Helmholtz", "lc_pinn_helmholtz_film_lbfgs.json"),
        ("2D Helmholtz w64", "lc_pinn_helmholtz_2d_film_lbfgs_w64.json"),
        ("2D Helmholtz w128", "lc_pinn_helmholtz_2d_film_lbfgs_w128.json"),
        ("Burgers", "lc_pinn_burgers_film_lbfgs.json"),
        ("BL", "lc_pinn_bl_seeds_film_lbfgs.json"),
    ]:
        d = load(path)
        if d is None:
            print(f"- **{tag}**: (no data — chain may not have reached this run)")
            continue
        flags = []
        for r in d.get("runs", []):
            seed = r.get("seed")
            lb = r.get("lbfgs") or {}
            flag = "REVERTED" if lb.get("reverted") else "ok"
            pre = lb.get("pre_lbfgs_rel_l2")
            post = lb.get("post_lbfgs_rel_l2")
            if pre is not None and post is not None:
                flags.append(f"seed{seed}={flag} ({pre:.2e}→{post:.2e})")
            else:
                flags.append(f"seed{seed}={flag}")
        print(f"- **{tag}**: {', '.join(flags)}")


if __name__ == "__main__":
    main()
