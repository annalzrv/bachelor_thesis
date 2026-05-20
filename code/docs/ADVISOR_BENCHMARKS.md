# Advisor notes: benchmark suite for loss-weighting PINNs

Persistent reference from thesis supervision. **Goal:** evaluate a loss-weighting scheme on more than one equation, using problems where composite losses (PDE, IC, BC, data) are genuinely hard to balance.

---

## Why not one equation only?

For a *new weighting scheme*, a single PDE can show a proof of concept but does not demonstrate that the method handles the failure modes discussed in the PINN literature (gradient pathologies, loss imbalance). Papers on loss balancing often use **several** classical problems for that reason.

---

## What makes a good benchmark

A problem is informative if it has **at least one** of:

| Property | Why it matters |
|----------|----------------|
| Loss terms on **very different scales** | Direct stress on weighting / gradients |
| **Sharp gradients** or internal layers | Hard for networks + residual minimization |
| **Oscillatory** or high-frequency solutions | Residual vs fitting tension |
| **BC/IC + sparse interior data** | Physics vs data competition |
| **Exact solution** or very accurate numerical reference | Measurable, comparable errors |

---

## Recommended ladder (2 ODEs + 4 PDEs)

Order matters: **easy first** (sanity), then add pathologies.

### 1. Logistic ODE — sanity check

\[
u'(t) = r\, u(t)\left(1 - \frac{u(t)}{K}\right), \quad t \in [0,T]
\]

Suggested parameters: \(r=2,\, K=1,\, u_0=0.1,\, t \in [0,2]\).

- **Data term:** 10–20 noisy observations \(\{(t_i, u(t_i))\}\).
- **Role:** verify the method does not break an easy nonlinear problem; mostly one IC.

### 2. Harmonic / damped oscillator — oscillatory ODE

Undamped: \(u'' + \omega^2 u = 0\), e.g. \(\omega=5,\, t\in[0,2]\), \(u(0)=1,\, u'(0)=0\).

Damped example: \(u'' + 0.4 u' + 25 u = 0\), \(t\in[0,4]\), same ICs.

- **Role:** oscillations and \(u''\) make residual matching harder; sparse data can pull the solution away from the ODE.

### 3. 1D viscous Burgers — core PDE benchmark

\[
u_t + u u_x - \nu u_{xx} = 0, \quad x \in [-1,1],\; t \in [0,1]
\]

Standard Raissi setup: \(\nu = 0.01/\pi\), \(u(0,x) = -\sin(\pi x)\), \(u(t,-1)=u(t,1)=0\).

- **Data:** sparse interior points from exact/reference solution.
- **Role:** convection–diffusion, **sharp internal layer**, standard in loss-balancing work (e.g. ReLoBRaLo).

### 4. Allen–Cahn — reaction–diffusion, stiff interface

\[
u_t - 10^{-4} u_{xx} + 5u^3 - 5u = 0, \quad x \in [-1,1],\; t \in [0,1]
\]

IC: \(u(0,x) = x^2 \cos(\pi x)\). **Periodic** BCs: \(u\) and \(u_x\) match at \(\pm 1\).

- **Role:** very different scales (reaction vs diffusion), interface dynamics; appears in the original PINN paper.

### 5. 2D Helmholtz — oscillatory elliptic PDE

Manufactured solution on \((0,1)^2\):

\[
u(x,y) = \sin(\pi x)\sin(4\pi y), \quad
f = u_{xx} + u_{yy} + k^2 u = \left[-\pi^2 - (4\pi)^2 + k^2\right] u
\]

Dirichlet \(u=0\) on \(\partial(0,1)^2\). Example: \(k = 5\pi\).

- **Role:** **BC vs PDE** competition; high-frequency content in the interior.

### 6. 2D Poisson + sparse interior data — clean data-assimilation case

\[
-\Delta u = f \quad \text{on } (0,1)^2
\]

Example exact: \(u = \sin(\pi x)\sin(\pi y)\), \(f = 2\pi^2 \sin(\pi x)\sin(\pi y)\), \(u=0\) on boundary.

- **Data:** 20–100 interior sensor points.
- **Role:** simple physics, **interpretable** tradeoff between PDE, BC, and data (good for ablations before harder nonlinear PDEs).

### Optional seventh — nonlinear Schrödinger (advanced)

Raissi-style setup on \(x \in [-5,5]\), \(t \in [0, \pi/2]\):  
\(i h_t + \frac{1}{2} h_{xx} + |h|^2 h = 0\), with sech initial data and periodic BCs in \(x\).

- **Role:** time evolution + periodic BCs + oscillatory nonlinear structure; strong stress test.

---

## Loss decomposition (same template everywhere)

Use a composite loss with explicit weights:

\[
\mathcal{L} = \lambda_{\mathrm{PDE}}\mathcal{L}_{\mathrm{PDE}}
  + \lambda_{\mathrm{IC}}\mathcal{L}_{\mathrm{IC}}
  + \lambda_{\mathrm{BC}}\mathcal{L}_{\mathrm{BC}}
  + \lambda_{\mathrm{data}}\mathcal{L}_{\mathrm{data}}
\]

(Adapt naming if a problem has no IC or no data term.)

---

## Comparisons to report

For each problem, compare at least:

1. **Fixed equal weights** (naive baseline PINN)  
2. **Hand-tuned weights** (manual oracle / best effort)  
3. **Your method** (e.g. loss-conditional / adaptive weighting as in your thesis)

**Metrics (advisor):** relative \(L_2\) error on the solution, PDE residual error, BC/IC violation, data fit error, and **evolution of weights** over training (or behavior over \(\lambda\) if weights are inputs).

---

## Relation to this repo

- **`loss_conditional_pinn.ipynb`**: Buckley–Leverett + FVM reference — strong **application** story (fronts, multiphase); good as one anchor in a broader suite.
- **Next research steps:** implement the same loss template + baselines on several problems above; align tables/figures with the three-way comparison and metrics listed here.
- **For a workshop / short paper (e.g. AI4Physics):** prioritize a **quantitative three-way comparison** on Buckley–Leverett + a **relative \(L_2\)** table vs FVM; detailed checklist and workshop pointer live in [`README.md`](../README.md).

---

## Ideas from advising sessions (backlog)

### Two-phase optimizer / LR warmup (training efficiency)

Use a higher learning rate for the first ~30% of training (warmup), then schedule normally. Motivation: the first few thousand steps are spent reducing a large initial residual — a larger LR early accelerates this escape without hurting the final refinement phase. Advisor also suggested a **10× LR increase** from the current value as a concrete starting point.

**Status (Apr 15 2026):** Implemented and tested. `SequentialLR` (LinearLR warmup → CosineAnnealingLR) added to both `train_lc_pinn` and `train_fixed_pinn`.

**lr=1e-2 (10× default) — FAILED.** 300k steps, loss oscillated 3–4 OOM throughout, never converged. Model still beat equal-weight baseline numerically (rel-L2 0.144 vs 0.192) but was visually worse than the Mar 22 reference. LR=1e-2 is outside the stable range for BL with Adam + grad clip 1.0.

**v2: lr=3e-3 (3× default), warmup=30%, 300k steps — DONE.** Mean rel-L2 0.161 — worse than v1 (0.125, no warmup). Cannot separate LR vs warmup effect from a single run. Ablations v7 (lr=1e-3 + warmup) and v8 (lr=3e-3, no warmup) in the overnight suite will isolate each factor.

**Do not try lr=1e-2 again on BL** without switching to L-BFGS or tighter gradient clipping.

### Weight-stability / robustness experiment

Investigate whether LC-PINN predictions are stable under different inference-time λ values:

1. **Variance across λ** — run inference for many λ vectors; measure prediction variance. If variance is low, that is itself a strong result (robustness).
2. **LC-PINN at equal weights** — plug equal weights (0.25 each) into the trained LC-PINN and compare against the equal-weight baseline trained from scratch. How different are the predictions?
3. **Cross-validate λ at inference** — since inference is cheap, treat λ as a hyperparameter to tune post-training (e.g. grid search or cross-validation on a held-out set). This is the budget-friendly hyperparameter optimization path.

**Status (Apr 15 2026):** Implemented. Section 11 in notebook: 50-λ prediction ribbon (mean ± 1σ) + equal-weights-LC-PINN vs equal-weight-baseline table. Not yet re-run with the best overnight checkpoint — do after Section 13b results tomorrow.

### Prediction averaging over λ

Instead of querying LC-PINN at a single λ, average predictions over a distribution of λ values (e.g. simplex uniform). Motivation: the strange "step" artifact visible in the middle time-slice may be a sensitivity artefact of a single λ choice and could wash out under averaging.

**Status (Apr 16 2026):** Not implemented. Was planned but removed during Apr 16 restructure. Low priority — single best-λ works well.

### Small debug case

Create a tiny version of the BL problem (e.g. coarser grid, 10k steps) that runs in ~15 minutes. Use exclusively for debugging code changes before committing to 1.5 h full runs.

**Status (Apr 16 2026):** Removed during Apr 16 restructure. `pinns/config.py` was deleted — debug settings are now controlled directly via notebook constants (`N_STEPS`, `LR`, etc.).

### Bad-λ paper experiment (killer demo for the paper)

The central claim of the LC-PINN paper story is: *one trained network covers the entire λ space at inference.* The advisor wants to see this demonstrated dramatically:

1. **Find a bad λ** — a fixed weight vector where a standard PINN (equal-weight baseline or hand-tuned) fails: ignores the IC/BC, memorises data noise, or produces a physically wrong shock.
2. **Show LC-PINN recovers** — at the same λ, query the LC-PINN. It should produce a correct solution because it has seen many λ values during training.
3. **Contrast side-by-side**: bad fixed PINN at λ_bad vs LC-PINN at λ_bad → LC-PINN wins.

This directly illustrates the value of conditioning on λ rather than fixing it. Without this experiment the paper shows LC-PINN is *as good as* a well-tuned baseline; with it, the paper shows it is *robust where the baseline breaks*.

**Implementation sketch:** use `sweep_lambda` to identify the worst-performing λ for the baseline, retrain a `FixedWeightPINN` with those exact weights, plot the resulting prediction next to the LC-PINN prediction at the same λ.

---

## Confirmed ideas (from advisor talk, now in backlog)

| Idea | Status | Where / result |
|------|--------|----------------|
| Hard-example mining: keep top 30% highest-loss collocation points per batch | ✅ Done, **harmful** | v3 (0.199) and v5 (0.164) — shock-biased resampling starves smooth region. Do not use on BL. |
| Sample λ uniformly on the simplex (sum-to-one) instead of softmax of log_λ | ✅ Done, **marginal gain** | v4 (0.124) ≈ v1 (0.125). Different optimal λ found but similar accuracy. Either strategy works. |
| Verify optimal λ ≠ center λ accidentally (shift center and recheck) | ⬜ Not yet run | Low priority — best/worst λ gap is negligible (0.001) on BL |
| Benchmark ladder: Logistic ODE → Burgers → Allen–Cahn → BL (4 problems) | ✅ All 4 done | Logistic: baseline wins (expected). Burgers: **160x improvement** (star result). BL: 1.4x improvement. Allen-Cahn: updated to ε²=0.01, re-run pending. |
| Pinnacle benchmark suite (standardised PINN evaluation) | ⬜ Not started | Low priority, do only if time before Apr 24 |
| LR warmup | ✅ Done, **neutral/harmful** | v7 (lr=1e-3+warmup) = 0.139 vs v1 (no warmup) = 0.125. Does not help at optimal LR. |
| 500k steps | ✅ Done, **harmful** | v6 (0.198) — overfitting past 300k steps due to cosine annealing decay. |
| lr=3e-3 (vs 1e-3) | ✅ Done, **harmful** | v8 (0.157) > v1 (0.125). Low sweep loss (5.47e-6) but worse rel-L2 — proxy doesn't correlate. |
| Weight-stability experiment | ✅ Done (Apr 15) | Notebook Section 11; rerun with best checkpoint (v4 or v1) this week |
| Prediction averaging over λ at inference | ⬜ Not implemented | Was planned for `pinns/inference.py` but not yet added |
| Comparison with other PINNs | ⬜ Low priority | Do only if time before Apr 24 |
| Bad-λ paper experiment (find worst λ for baseline, show LC-PINN recovers) | ✅ Done (Apr 16) | `00_buckley_leverett.ipynb` Section 4. LC-PINN best/worst gap = 0.001, fixed-PINN@bad-λ = 0.171 |
| **Allen-Cahn failure analysis** | ✅ Documented + fixed | u=0 collapse at ε²=1e-4 — fundamental PINN causality limitation, not a weighting issue. Updated to ε²=0.01 (tractable interfaces ~0.1 wide). Re-run pending. |
| **Advisor idea: uniform [0,1] weight sampling** | ✅ Done (Apr 17), **2x improvement on Burgers** | `04_burgers_uniform_sampling.ipynb`: mean rel-L2 **0.0004** vs 0.0009 logspace. Each λ sampled independently from U(0,1), no sum-to-1 constraint. Better at every snapshot. Checkpoint: `burgers_lc_pinn_uniform.pt`. |

### Key empirical finding (session Apr 14 2026)

The **gap between LC-PINN and equal-weight baseline grows with T** (visible in cell 15 of the notebook, statistically significant). At T=0.3 the difference is ~0.07 rel-L2; at earlier T it is ~0.02. Advisor interpretation: at larger T the shock has evolved further and weight misspecification hurts more — exactly where the LC-PINN's coverage of λ-space pays off. This is a strong narrative point for the paper.

**Paper framing (advisor):** "we train one model but cover the entire λ space — information from good-λ regions leaks into bad-λ predictions at inference." This is the differentiating story vs any single fixed-weight PINN.

---

## References (topics to cite in the thesis)

PINN loss balancing, ReLoBRaLo, Raissi et al. benchmarks (Burgers, Allen–Cahn, NLS), unified ODE/PDE benchmark papers — use your advisor’s reading list and recent survey papers for exact citations.
