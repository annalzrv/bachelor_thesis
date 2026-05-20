# v1 push plan — NeurIPS/AI4Physics scope expansion

**Today: 2026-04-27.** Targets: NeurIPS abstract May 4, NeurIPS full May 6, AI4Physics @ ICML May 7. All AoE.

Scope decision on 2026-04-27: include both must-haves AND former nice-to-haves in v1. This is aggressive and the calendar is tight — see "Critical path & risks" below for honest assessment.

## Scope (what's in, what's deferred)

**In v1:**
- FNO baseline on Burgers
- DeepONet baseline on Burgers
- Helmholtz with parametric wavenumber (full method comparison)
- Viscous-regularised Buckley-Leverett (full method comparison)
- Ablation study (K_eval, n_λ_samples, sampling distribution)
- Error bars / std on all tables
- λ-invariance proposition + sketch proof

**Deferred to v2 (ML4PS Aug-Sep):**
- HyperNetwork baseline (separate research project)
- 2D PDEs (new solver, new sampling — not feasible in 4 days)
- Full theorem proof with rigour
- Eigenvalue-on-circle (advisor's S¹ suggestion)
- Third additional PDE (KdV solitons or parametric heat)

## Why each piece (the "зачем")

### Operator-learning baselines (FNO + DeepONet)
FNO and DeepONet are the canonical operator-learning pair. Reviewers reading "we beat SA-PINN/ReLoBRaLo/Causal-PINN on amortisation" will ask: but how do you compare to actual operator learners? Showing only one of FNO/DeepONet looks like cherry-picking; both is the standard pairing. Same compute budget across all methods, same data, same reference.

### Helmholtz with parametric wavenumber
Currently the paper wins on **one** PDE (Burgers). One example is a special case; two is a pattern. Helmholtz is the canonical-second-PDE in PINN literature: linear elliptic, smooth, parametric coefficient (k) maps cleanly to LC-PINN's λ-input. Reference solution available analytically for simple geometries. Note: Causal-PINN doesn't apply to elliptic problems (no time-causality), so the comparison is SA + ReLo + LC + FNO/DeepONet only.

### Viscous-regularised Buckley-Leverett
The inviscid BL currently fails uniformly across all methods (filed as caveat in EXPERIMENTS.md). Adding viscous term ε·s_xx with ε ≈ 0.01 turns the equation into a smoothed parabolic problem.

**Honest framing of physical interpretation.** Real porous media have capillary pressure, which produces a diffusion term D(s)·s_xx with saturation-dependent D(s). Constant-ε viscous-BL is a first-order proxy for this capillary diffusion — closer to reservoir physics than inviscid BL, but still a simplification of full capillary-BL. ε = 0.01 is in the physically defensible range for typical capillary diffusion. Saturation-dependent D(s) deferred to v2/future work.

Two outcomes possible:
- LC-PINN wins on viscous-BL → caveat becomes a partial-physics win on the capillarity-corrected model.
- LC-PINN still loses → confirms it's the LC formulation, not the regularisation, and we file as honest limitation.
Either way more informative than the current uniform failure. Paper text should not overclaim "porous media solved" — the claim is "capillarity-corrected BL family solved by LC-PINN".

### Ablations
Three sweeps:
- **K_eval ∈ {25, 50, 100, 200, 400}**: shows where amortisation pays off vs cost. Pure inference — no retraining needed, runs in minutes per K.
- **n_λ_samples per step ∈ {1, 4, 16}**: design parameter analysis. Requires retraining (3 retrains).
- **Sampling distribution: uniform vs log-uniform**: ties directly to the λ-invariance result. Already partially covered (we have both 50k uniform and 200k uniform); add log-uniform run for comparison.

### Error bars
4 seeds already exist for all main results. Cheapest possible move: present mean±std instead of mean. Zero new compute, professional appearance.

### λ-invariance theorem
Formal proposition: "At a global optimum of L_LC(θ), the conditional network u_θ(·, λ) is, for almost every λ ∈ supp(p_λ), a residual-minimiser of the parametric PDE indexed by λ." Sketch proof via dominated convergence + first-order stationarity. Half a page total. Converts the work from purely empirical to "empirical + theoretical" — qualitative shift in how reviewers read the paper.

## Task breakdown by track

Tasks are grouped by independent track. Within a track, tasks are sequential. Across tracks, work parallelises (modulo GPU contention).

### Track A — Operator-learning baselines
- **A1**: `scripts/fno_burgers.py` via `neuraloperator` package; 4 seeds × periodic Burgers; same reference solution as SA/ReLo/Causal/LC; same training budget. Output: `results/fno_burgers.json`.
- **A2**: `scripts/deeponet_burgers.py`; branch+trunk architecture; 4 seeds; same reference and budget. Output: `results/deeponet_burgers.json`.
- **A3**: Add operator-learning columns to summary tables in `notebooks/09_baseline_comparison.ipynb`.

### Track B — Helmholtz parametric wavenumber
- **B1**: `pinns/equations/helmholtz.py` — equation class for ∂²u/∂x² + k²u = f, parametric k ∈ [k_min, k_max], analytical reference solution for simple BC.
- **B2**: `scripts/lc_pinn_helmholtz.py` — LC training, λ = (k - k_min)/(k_max - k_min); 4 seeds × 50k Adam.
- **B3**: `scripts/sa_pinn_helmholtz.py` — SA-PINN baseline; 4 seeds × (10k Adam + 5k LBFGS); skip Causal-PINN (elliptic, no time direction).
- **B4**: `scripts/relobralo_helmholtz.py` — ReLoBRaLo baseline; 4 seeds × 50k Adam.
- **B5**: Helmholtz section in notebook with table + per-k breakdown.

### Track C — Viscous-regularised BL
- **C1**: Add `epsilon` parameter to `pinns/equations/buckley_leverett.py`; viscous residual is s_t + f'(s)·s_x − ε·s_xx.
- **C2**: New reference solver for viscous-BL (finite-difference with mesh fine enough for ε = 0.01 diffusion).
- **C3**: Re-run all four methods (SA, ReLo, Causal, LC) on viscous-BL using existing scripts with the modified equation; 4 seeds each.
- **C4**: Update BL section in notebook; replace current "uniform failure" caveat with viscous results.

### Track D — Ablations
- **D1**: K_eval sweep — load LC-PINN Burgers checkpoint, run inference at K ∈ {25, 50, 100, 200, 400}, table of rel-L2 vs K.
- **D2**: n_λ_samples sweep — retrain LC-PINN Burgers at n ∈ {1, 4, 16} with shorter budget (e.g. 25k Adam) for clean comparison.
- **D3**: Log-uniform vs uniform sampling — single retrain at log-uniform; compare against existing uniform run.

### Track E — Statistical presentation
- **E1**: Update notebook tables with mean±std formatting using existing 4-seed JSONs. No new runs.

### Track F — Theory
- **F1**: Write formal proposition statement in paper Methods section.
- **F2**: Write sketch proof (dominated convergence + first-order stationarity argument).

### Track G — Writing
- **G1**: NeurIPS LaTeX template setup; bibliography file.
- **G2**: Abstract (last to write but first deadline — May 4).
- **G3**: Introduction (motivation: amortisation across PDE-parameter families; gap; contributions list).
- **G4**: Related work (PINN adaptive-loss methods; operator learning; loss-conditional networks; meta-learning).
- **G5**: Method (LC-PINN formalism + theorem F1/F2 + λ-invariance).
- **G6**: Experiments (Burgers main, Helmholtz, viscous-BL, Laplace eigenvalue extension; protocol; baselines).
- **G7**: Results (tables with E1 stats, figures, ablations from D).
- **G8**: Discussion + limitations (honest BL story, scope limits).
- **G9**: References, formatting pass, page-count fit.

## Critical path & risks

**Critical path:** Tracks A, B, C are compute-heavy; G depends on their outputs. M4 Max is a single GPU — runs are serial, not parallel. Compute budget approximation:
- Burgers 4-seed × 50k ≈ 4–6 hours wall.
- Helmholtz expected lighter (smaller solver, smooth solutions).
- Viscous-BL similar to Burgers.
- FNO/DeepONet are fast (10–30 min each).

Realistic compute budget: 3 overnight runs (~18 hours each = 54 GPU-hours total).

**Most likely failure modes (ranked):**
1. **Viscous-BL reference solver bug** — finite-difference with ε = 0.01 needs fine mesh; numerical diffusion may swamp the front. Mitigation: validate reference against a known similarity solution before using it as ground truth. If solver is wrong, keep BL as caveat in v1, do viscous-BL properly in v2.
2. **DeepONet underperforms FNO** so badly that the comparison is uninteresting. Mitigation: report honestly; DeepONet's known weakness on shock problems is itself informative.
3. **Helmholtz LC doesn't win cleanly** — Helmholtz is well-conditioned, baselines may match LC. Mitigation: pick k-range wide enough that single networks struggle (e.g. k ∈ [1, 10] gives 10× wavelength variation).
4. **Writing time gets eaten by experiments** — most likely failure. Mitigation: write Methods + Related Work BEFORE experiments finish (those don't depend on results).

**Drop-list if behind schedule** (in this order):
- Drop D2/D3 (ablations beyond K_eval sweep) — keep only K_eval.
- Drop DeepONet, keep FNO only.
- Drop viscous-BL, keep BL caveat.
- Drop Helmholtz baseline comparison; keep Helmholtz LC-only as proof-of-concept.

Hard floor (must NOT drop): FNO Burgers, error bars, theorem.

## Files to create / modify

**New scripts:**
- `scripts/fno_burgers.py`
- `scripts/deeponet_burgers.py`
- `scripts/lc_pinn_helmholtz.py`
- `scripts/sa_pinn_helmholtz.py`
- `scripts/relobralo_helmholtz.py`

**New equations:**
- `pinns/equations/helmholtz.py`

**Modified equations:**
- `pinns/equations/buckley_leverett.py` (add `epsilon` parameter)

**Reference solvers:**
- `references/helmholtz_analytical.py`
- `references/viscous_bl_fd.py`

**Modified notebook:**
- `notebooks/09_baseline_comparison.ipynb` (mean±std, FNO/DeepONet columns, Helmholtz section, viscous-BL section, ablation cells)

**New paper directory:**
- `paper/main.tex`, `paper/sections/*.tex`, `paper/figures/`, `paper/refs.bib`

## Resolved setup decisions (2026-04-27)

1. **Helmholtz domain: Dirichlet on [0,1] with manufactured solution.** Take u(x; k) = sin(πx)·cos(kx), compute f(x; k) = −u_xx − k²u analytically; reference is exact, no numerical solver needed. Periodic-S¹ rejected because of resonances at integer k (homogeneous problem nontrivial → ill-posed without damping). Advisor's S¹ hint deferred to v2 eigenvalue work where it's natural (∇² on S¹).
2. **k-range: k ∈ [1, 10].** Hidden_dims=[64,64,64,64] resolves k up to ≈10–15 without catastrophic spectral-bias collapse. Want regime where baselines succeed (slowly, per-k retraining) and LC wins by amortisation, not regime where all methods fail. k > 10 noted as future work in Discussion.
3. **Viscous-BL ε = 0.01** (final). Reference solver first validated at ε=0.05 (easier) before switching to 0.01. ε=0.01 is in physically defensible range for capillary diffusion (10⁻³–10⁻²); ε=0.05 would smear the front too much and make the comparison uninteresting.
4. **Template: NeurIPS 2026.** Trim to AI4Physics (8 pages, more flexible) is trivial; the reverse direction would be painful.
