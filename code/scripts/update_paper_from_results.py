"""Fill paper placeholders from result JSONs.

After a training run completes, this script substitutes the
"<don-mean>", "<lc-mean>", etc. placeholders in
paper/sections/results.tex with the actual rel-L² and wall-time
numbers. Idempotent: re-running with the same JSON produces the
same output, and unfilled placeholders survive untouched.

Usage:
    python scripts/update_paper_from_results.py
"""
from __future__ import annotations

import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
RES = REPO / "results"
TEX = REPO / "paper" / "sections" / "results.tex"


def fmt_sci(x: float) -> str:
    """LaTeX-formatted scientific notation, two decimals."""
    if x == 0:
        return r"0"
    sign = "-" if x < 0 else ""
    x = abs(x)
    exp = 0
    while x >= 10:
        x /= 10; exp += 1
    while x < 1:
        x *= 10; exp -= 1
    return rf"${sign}{x:.2f} \times 10^{{{exp}}}$"


def substitute(tex: str, key: str, value: str) -> str:
    pat = re.escape(rf"\texttt{{<{key}>}}")
    return re.sub(pat, value, tex)


def load(name: str) -> dict | None:
    p = RES / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> None:
    tex = TEX.read_text()
    n_subs = 0

    don = load("pi_deeponet_helmholtz_matched.json") or load("pi_deeponet_helmholtz.json")
    if don is not None:
        s = don["summary"]
        tex_new = tex
        tex_new = substitute(tex_new, "don-mean", fmt_sci(s["rel_l2_mean"]))
        tex_new = substitute(tex_new, "don-std",  fmt_sci(s["rel_l2_std"]))
        tex_new = substitute(tex_new, "don-min",  f"{s['elapsed_mean_sec']/60:.1f}")
        if tex_new != tex:
            n_subs += 1
            print(f"PI-DeepONet 1D Helm: rel-L² = {s['rel_l2_mean']:.4e}, "
                  f"{s['elapsed_mean_sec']/60:.1f} min/seed")
        tex = tex_new

    sch = load("lc_pinn_schrodinger_film_lbfgs.json")
    if sch is not None:
        s = sch["summary"]
        tex_new = tex
        tex_new = substitute(tex_new, "lc-mean", fmt_sci(s["rel_l2_mean"]))
        tex_new = substitute(tex_new, "lc-std",  fmt_sci(s["rel_l2_std"]))
        tex_new = substitute(tex_new, "lc-min",  f"{s['elapsed_mean_sec']/60:.1f}")
        if tex_new != tex:
            n_subs += 1
            print(f"LC-PINN Schrödinger: rel-L² = {s['rel_l2_mean']:.4e}, "
                  f"{s['elapsed_mean_sec']/60:.1f} min/seed")
        tex = tex_new

    sa = load("sa_pinn_schrodinger.json")
    if sa is not None:
        s = sa["summary"]
        mean_per_seed = s["rel_l2_mean_over_alpha_then_seeds"]
        std_over_seeds = s["rel_l2_std_over_seeds"]
        avg_min = s["elapsed_total_sec"] / 60 / len(s["alpha_trains"]) / s["n_seeds"]
        tex_new = tex
        tex_new = substitute(tex_new, "sa-mean", fmt_sci(mean_per_seed))
        tex_new = substitute(tex_new, "sa-std",  fmt_sci(std_over_seeds))
        tex_new = substitute(tex_new, "sa-min",  f"{avg_min:.1f}")
        if tex_new != tex:
            n_subs += 1
            print(f"SA-PINN Schrödinger: rel-L² = {mean_per_seed:.4e}, "
                  f"{avg_min:.1f} min/α")
        tex = tex_new

    if n_subs > 0:
        TEX.write_text(tex)
        print(f"\nWrote {TEX.relative_to(REPO)} ({n_subs} substitution group(s))")
    else:
        print("No JSON results found yet — nothing to substitute.")
        # report which placeholders remain
        leftover = re.findall(r"<[a-z\-]+>", tex)
        if leftover:
            print(f"Pending placeholders: {sorted(set(leftover))}")


if __name__ == "__main__":
    main()
