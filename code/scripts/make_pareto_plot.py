"""Pareto plot: total wall time to produce K predictions vs rel-L^2.

For LC-PINN, K is the number of test-time evaluations across the lambda
family (essentially free at inference). For per-lambda baselines,
K corresponds to K retrainings, so the wall time scales as K * T_base.

Output: paper/figures/pareto.pdf

Usage:
    python scripts/make_pareto_plot.py
"""
from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np


REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = REPO / "paper" / "figures" / "pareto.pdf"

K_GRID = np.array([1, 4, 10, 25, 50, 100, 200])


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def _per_seed_train_time(d: dict) -> float:
    """Mean training wall time per seed (one full retraining)."""
    if "elapsed_mean_sec" in d.get("summary", {}):
        return float(d["summary"]["elapsed_mean_sec"])
    runs = d.get("runs", [])
    times = [r["elapsed_sec"] for r in runs if "elapsed_sec" in r]
    return float(np.mean(times)) if times else float("nan")


def _per_k_train_time(d: dict) -> float:
    """For per-(seed, k_train) baselines (Helmholtz): mean per-(seed,k) time."""
    runs = d.get("runs", [])
    times = [r["elapsed_sec"] for r in runs if "elapsed_sec" in r]
    return float(np.mean(times)) if times else float("nan")


def _rel_l2(d: dict) -> tuple[float, float]:
    s = d["summary"]
    if "rel_l2_mean" in s:
        return float(s["rel_l2_mean"]), float(s.get("rel_l2_std", 0.0))
    if "rel_l2_mean_over_k_then_seeds" in s:
        return (float(s["rel_l2_mean_over_k_then_seeds"]),
                float(s.get("rel_l2_std_over_seeds", 0.0)))
    raise KeyError(name="rel_l2 not in summary")


def _curve(t_train: float, err: float, K: np.ndarray, *,
           amortised: bool) -> tuple[np.ndarray, np.ndarray]:
    if amortised:
        return np.full_like(K, t_train, dtype=float), np.full_like(K, err, dtype=float)
    return K * t_train, np.full_like(K, err, dtype=float)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), sharey=False)

    # ---------- Panel 1: Burgers (loss-weight mode) ----------
    burgers_lc = _load("lc_pinn_burgers_seeds.json")
    burgers_sa = _load("sa_pinn_burgers.json")
    burgers_relo = _load("relobralo_burgers.json")
    burgers_causal = _load("causal_pinn_burgers.json")

    ax = axes[0]
    methods_b = [
        ("LC-PINN (one network)",   burgers_lc,     "tab:blue",   True),
        ("Causal-PINN (per-$w$)",   burgers_causal, "tab:green",  False),
        ("SA-PINN (per-$w$)",       burgers_sa,     "tab:orange", False),
        ("ReLoBRaLo (per-$w$)",     burgers_relo,   "tab:red",    False),
    ]
    lc_t_burgers = None
    for label, d, color, amortised in methods_b:
        t = _per_seed_train_time(d) / 60.0
        err, _ = _rel_l2(d)
        x, y = _curve(t, err, K_GRID, amortised=amortised)
        if amortised:
            lc_t_burgers = float(x[0])
            ax.scatter([x[0]], [y[0]], s=60, color=color, label=label, zorder=3,
                       marker="*", edgecolors="black", linewidths=0.5)
            ax.axhline(y[0], color=color, linestyle=":", alpha=0.4)
        else:
            ax.plot(x, y, color=color, linewidth=1.5, label=label, marker="o", ms=3)

    if lc_t_burgers is not None:
        ax.axvline(lc_t_burgers, color="tab:blue", linestyle="--", alpha=0.55,
                   linewidth=1.0, zorder=2)
        ax.text(lc_t_burgers * 1.08, 1.2e-2, "$K^\\star$",
                fontsize=8, ha="left", va="center", color="tab:blue",
                fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall time to produce $K$ predictions (min, log)")
    ax.set_ylabel("rel-$L^2$ (log)")
    ax.set_title("Burgers (loss-weight family, $d_\\lambda{=}4$)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)

    # ---------- Panel 2: Helmholtz (parametric-coefficient mode) ----------
    helm_lc = _load("lc_pinn_helmholtz_film_lbfgs.json")
    helm_sa = _load("sa_pinn_helmholtz.json")
    helm_relo = _load("relobralo_helmholtz.json")
    helm_breakdown = _load("per_k_breakdown_helmholtz.json")

    ax = axes[1]
    # LC-PINN: one training, evaluate at any k in [1,10] for free.
    # Use the per-k breakdown grid mean (matches Table 1).
    lc_t = _per_seed_train_time(helm_lc) / 60.0
    lc_err = float(helm_breakdown["grid_means"]["lc_pinn"])
    ax.scatter([lc_t], [lc_err], s=60, color="tab:blue", label="LC-PINN (one network)",
               zorder=3, marker="*", edgecolors="black", linewidths=0.5)
    ax.axhline(lc_err, color="tab:blue", linestyle=":", alpha=0.4)

    # Per-k baselines: walking right as the k-grid grows
    for label, d, color in [
        ("SA-PINN (per-$k$)",   helm_sa,   "tab:orange"),
        ("ReLoBRaLo (per-$k$)", helm_relo, "tab:red"),
    ]:
        t_per_k = _per_k_train_time(d) / 60.0  # min per (seed, k)
        err, _ = _rel_l2(d)
        x, y = _curve(t_per_k, err, K_GRID, amortised=False)
        ax.plot(x, y, color=color, linewidth=1.5, label=label, marker="o", ms=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(5e-4, 2e-2)
    ax.set_xlabel("wall time to produce $K$ predictions (min, log)")
    ax.set_title("Helmholtz (parametric-coefficient $k\\in[1,10]$)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
