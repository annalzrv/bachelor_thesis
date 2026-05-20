# NeurIPS 2026 — OpenReview submission fields

## Title

Loss-Conditional PINNs: One Training Run for a Parametric PDE Family

## Authors

- Anna Lazareva
- Alexander Tarakanov

## TL;DR

LC-PINN amortises a single residual-loss PINN training across a parametric PDE family — no solver-generated paired data, no test-time adaptation — with a provable per-λ optimality guarantee at the global optimum and a finite-capacity bound on every λ in the support.

## Abstract

Physics-informed neural networks (PINNs) require a fresh training run for any change in the residual: a new wavenumber, a new mobility ratio, a new loss-weight balance. Operator-learning methods (FNO, DeepONet) amortise across a parametric family but require solver-generated paired (input, solution) examples *in their training loop*. We close that gap. LC-PINN is a single residual-loss training run that produces a continuous family of PDE solutions over a parameter λ — a physical scalar coefficient or a loss-weight vector — without paired training data, test-time adaptation, or per-λ retuning. Reference solutions are used only to compute test-time error. The construction adapts loss-conditional networks (Dosovitskiy & Djolonga, 2020) to PINN training: λ becomes a network input drawn fresh from a fixed prior at every optimisation step. We prove a λ-invariance result — at a global optimum of the λ-expected loss the network is a per-λ residual minimiser for p_λ-almost every λ — with a finite-capacity corollary that bounds the suboptimality at every λ in the support. Across 1D, 2D, and 3D parametric Helmholtz, 1D stationary Schrödinger with a multiplicative parametric potential, and loss-weight amortisation on viscous Burgers and Buckley–Leverett, LC reaches rel-L² within a factor of two of the strongest single-λ baseline. On 1D Helmholtz it is the only method whose rel-L² stays within one order of magnitude across k ∈ [1, 10] — per-k retrained baselines vary by three.

## Primary subject area

Applications → Physical sciences (PDE solvers, scientific machine learning)

## Secondary subject area

Deep learning → Algorithms (multi-task / amortised training)

## Keywords

physics-informed neural networks, parametric PDEs, amortised inference, loss-conditional training, Helmholtz, Schrödinger, Burgers, operator learning

## Conflicts

(populated in OpenReview UI from author profiles)

## Format checklist

- [x] 9-page main-content limit (paper ends at p9 with Discussion; refs start p10)
- [x] References, NeurIPS Checklist, and Appendix do not count toward page limit
- [x] Anonymised author block + anonymous bibliography entries
- [x] `\linenumbers` enabled by default in `neurips_2026.sty` submission mode
- [x] Compiled with `tectonic` against `neurips_2026.sty` (2026-01-29 release)
- [x] PDF size under 10 MB (current: ~220 KB)
- [x] No external supplementary code submission yet — code release is anonymous-friendly placeholder in checklist

## Files to upload

- `paper/main.pdf` — anonymised submission
- (Optional supplementary ZIP) — code + result JSONs, deferred unless requested
