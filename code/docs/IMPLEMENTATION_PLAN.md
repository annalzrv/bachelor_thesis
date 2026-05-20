# Implementation Plan: Full Backlog

## Context

Advisor session (Apr 14) produced a set of concrete improvements to the LC-PINN codebase before the AI4Physics workshop deadline (Apr 24). Items span training efficiency, robustness analysis, inference utilities, and benchmark expansion. This plan covers all confirmed backlog items in ADVISOR_BENCHMARKS.md.

## Status (Apr 16 2026)

| Group | Item | Status |
|-------|------|--------|
| A1 | LR warmup | ✅ Implemented and ablated. **Not helpful** on BL (v7=0.139 > v1=0.125). lr=1e-2 failed. |
| A2 | Hard-example mining | ✅ Implemented and ablated. **Harmful** on BL (v3=0.199, v5=0.164). Shock-biased resampling. |
| A3 | Debug config | ✅ Done. |
| B1 | Simplex sampling | ✅ Implemented and ablated. **Marginal gain** (v4=0.124 ≈ v1=0.125). |
| B2 | Verify center λ | ⬜ Not yet run. |
| C1 | Prediction averaging | ✅ Implemented, notebook cell pending. |
| C2 | CV-λ | ✅ Implemented in `inference.py`. |
| D1 | Weight-stability | ✅ Section 11 done. Rerun with best checkpoint pending. |
| D2 | Bad-λ experiment | 🔄 Implemented in `run_comparison.py`, **never executed**. Priority this week. |
| E1 | Logistic ODE | 🔄 Equation code done, notebook pending. |
| E2 | Burgers | 🔄 Equation code done, notebook pending. |
| E3 | Allen–Cahn | 🔄 Equation code done, notebook pending. |
| F | Pinnacle comparison | ⬜ Not started, low priority. |

---

## Group A — Training improvements

### A1 · LR warmup + 10× learning rate (`pinns/training.py`)

**What:** Add a linear warmup phase covering the first 30% of training, then cosine anneal. Default LR stays at `1e-3` but we try `1e-2` (10×) as first experiment.

**How:**
- Add `warmup_frac: float = 0.0` parameter to `train_lc_pinn` (default 0 = no warmup, backward-compatible).
- When `warmup_frac > 0`, build a `SequentialLR` from:
  1. `LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)` — ramps from 0.1× to 1× of `lr`
  2. `CosineAnnealingLR(optimizer, T_max=n_epochs - warmup_steps)` — then cosine decays
- Replace the current single `CosineAnnealingLR` scheduler with this `SequentialLR` when `warmup_frac > 0`.
- `warmup_steps = int(n_epochs * warmup_frac)`
- Also apply same pattern to `train_fixed_pinn` in `pinns/baseline.py` for fair comparison.
- **Notebook CONFIG change:** add `LR = 1e-2`, `WARMUP_FRAC = 0.3`, `N_STEPS = 500_000` as tunable fields.

### A2 · Hard-example mining (`pinns/training.py` + `pinns/losses.py`)

**What:** Each training step, after computing per-point PDE residuals, keep only the top-`hard_frac` fraction of highest-residual collocation points and resample the rest.

**How:**
- Add `hard_frac: float = 0.0` parameter to `train_lc_pinn` (0 = off, backward-compatible).
- Add `compute_pde_residuals_pointwise(model, log_lambda, coords_pde, domain)` to `pinns/losses.py` — returns `(N,)` tensor of per-point `|residual|²` without reducing to mean.
- In `train_lc_pinn`, before the loss accumulation loop, when `hard_frac > 0`:
  1. With `torch.no_grad()`, compute per-point residuals on current `batch["coords_pde"]`.
  2. Keep top `int(n_pde * hard_frac)` worst points; resample the rest uniformly from the domain.
  3. Replace `batch["coords_pde"]` in place.
- Resample every step (cheap: it's just uniform random coords).
- Domain bounds come from `DEFAULT_DOMAIN` already in scope.

### A3 · Small debug config (`pinns/config.py`)

**What:** A config that runs in ~15 min for fast debugging.

**How:**
- Add `DEBUG_CONFIG` dict (or dataclass) near top of `pinns/config.py`:
  ```python
  DEBUG_N_STEPS = 10_000
  DEBUG_N_PDE   = 500
  DEBUG_N_BC    = 50
  DEBUG_N_IC    = 50
  DEBUG_N_DATA  = 50
  ```
- Add a single cell at the top of `loss_conditional_pinn.ipynb` with a `DEBUG = False` flag that overrides `CONFIG` when `True`.

---

## Group B — Lambda sampler

### B1 · Simplex uniform sampling (`pinns/lambda_sampler.py`)

**What:** Add an alternative sampling mode that draws λ directly from a Dirichlet(1,1,1,1) distribution (= uniform on the probability simplex), instead of softmax of log_λ. The key difference: simplex sampling always sums to 1 by construction; the current softmax-of-uniform approach does not cover extreme corners well.

**How:**
- Add `mode: str = "logspace"` parameter to `LambdaSampler.__init__` (values: `"logspace"` = current behaviour, `"simplex"` = new).
- In `LambdaSampler.sample()`:
  - If `mode == "simplex"`: draw `u ~ Dirichlet(alpha)` where `alpha = [1,1,1,1]` (use `torch.distributions.Dirichlet`). Return `(log(u), u)` so the call signature stays the same (callers already unpack `log_lam, p_lam`; `p_lam` is what gets used for weighting).
  - Half-width curriculum still applies: when `mode == "simplex"`, use it to concentrate sampling — scale alpha as `alpha = [1 + hw, ...]` so early training is more uniform and later more spread. (Or simply ignore hw for simplex mode — keep it simple first.)
- `sample_batch` gets the same treatment.

### B2 · Verify optimal λ ≠ center λ (`run_comparison.py` or notebook)

**What:** Confirm that the sweep-found best λ is genuinely different from the sampler center `[log(1), log(10), log(10), log(1)]` (i.e., we're not just rediscovering the prior).

**How:**
- Add a `--verify-center` flag to `run_comparison.py`.
- When set: evaluate LC-PINN at `sampler.center` (softmax) and at `best_ll` (from sweep), print both error tables, print cosine similarity between the two λ vectors.
- Also add this as a notebook cell after section 10 (side-by-side comparison).

---

## Group C — Inference enhancements (`pinns/inference.py`)

### C1 · Prediction averaging over λ

**What:** `predict_solution_averaged(model, sampler, n_samples, x_pts, t_val, device, step=None)` — draws `n_samples` λ vectors, gets predictions for each, returns the mean (and optionally std).

**How:**
- Add to `pinns/inference.py`:
  ```python
  @torch.no_grad()
  def predict_solution_averaged(model, sampler, n_samples, x_pts, t_val, device, step=None):
      ...
      log_lams, _ = sampler.sample_batch(n_samples, step=step or sampler.curriculum_steps)
      preds = [predict_solution(model, log_lams[i], x_pts, t_val, device) for i in range(n_samples)]
      stack = np.stack(preds)   # (n_samples, N)
      return stack.mean(axis=0), stack.std(axis=0)
  ```
- Also add `evaluate_lc_pinn_averaged` — same as `evaluate_lc_pinn` but uses the averaged prediction.
- Expose in notebook: compare averaged prediction vs single best-λ prediction at each snapshot.

### C2 · Cross-validate λ at inference

**What:** Given a held-out validation set, find the λ that minimises validation error without touching training data. Cheap because inference only.

**How:**
- Add `find_best_lambda_cv(model, val_snapshots, sampler, device, n_candidates=1000)` to `pinns/inference.py`.
- `val_snapshots` = a subset of FVM snapshots held out from training (or the full ref set for now).
- Loop over `n_candidates` λ samples, compute rel-L2 on val set for each, return best.
- This is distinct from `sweep_lambda` (which minimises BC+IC+data MSE on training points). CV uses the FVM reference directly.
- Wire up in `run_comparison.py` as a fourth method: `LC-PINN (CV-λ)`.

---

## Group D — Robustness experiments

### D1 · Weight-stability experiment (notebook + `run_comparison.py`)

**What:** How stable are LC-PINN predictions across different λ values?

**How:**
- New notebook section (e.g. section 11):
  1. Draw 50 random λ vectors from the full-width sampler.
  2. For each, call `predict_solution` at every snapshot time.
  3. Compute prediction mean and ±1σ band across λ. Plot as shaded ribbon over FVM reference.
  4. Separately: evaluate LC-PINN at exact equal weights `[0.25, 0.25, 0.25, 0.25]` and compare rel-L2 to equal-weight baseline. Answers: "what happens if you plug equal weights into the LC-PINN?"
- Report: variance table across λ (std of rel-L2 for each snapshot T).

### D2 · Bad-λ paper experiment (notebook + `run_comparison.py`)

**What:** Find the worst λ for a fixed-weight PINN, show LC-PINN recovers at that λ. Already partially designed in ADVISOR_BENCHMARKS.md.

**How:**
- In `run_comparison.py`, add `--bad-lambda` flag:
  1. Use existing `sweep_lambda` on the **equal-weight baseline** (not LC-PINN) to find the λ that maximises validation error (worst-case).
  2. Retrain a `FixedWeightPINN` with those exact weights.
  3. Evaluate both `FixedWeightPINN(bad_λ)` and `LC-PINN(bad_λ)` at same snapshots.
  4. Print three-row table: equal-weight baseline, bad-λ baseline, LC-PINN at bad-λ.
- For finding worst λ: add `find_worst_lambda(model_fixed, batch, sampler, device, n_candidates)` to `pinns/inference.py` — same loop as `sweep_lambda` but maximises rather than minimises.

---

## Group E — Benchmark ladder

Create a `pinns/equations/` subpackage. Each equation gets its own module with: `physics.py`-style PDE residual, FVM/exact reference, `generate_training_data`, and a training script.

### E1 · Logistic ODE (`pinns/equations/logistic.py` + `notebooks/01_logistic.ipynb`)

**PDE:** `u' = r·u·(1 - u/K)`, `r=2, K=1, u₀=0.1, t∈[0,2]`, 10–20 noisy observations.

**Exact solution:** `u(t) = K·u₀·e^(rt) / (K + u₀·(e^(rt) - 1))` — use this as reference.

**Losses:** IC + ODE residual + sparse data (no BC term). `dim_lambda = 3`.

**Model:** `LossConditionalPINN(dim_phys=1, dim_lambda=3)`.

**Deliverable:** training + comparison table (equal-weight, best-of-random, LC-PINN) in notebook.

### E2 · Burgers (`pinns/equations/burgers.py` + `notebooks/02_burgers.ipynb`)

**PDE:** `u_t + u·u_x - ν·u_xx = 0`, ν=0.01/π, Raissi setup (`x∈[-1,1], t∈[0,1]`).

**Reference:** use the known exact solution (series / `scipy.fft` approach) or load Raissi's dataset if available.

**Losses:** IC + BC (Dirichlet 0 at ±1) + PDE + sparse interior data. `dim_lambda = 4` (same as BL).

**Deliverable:** same comparison table format as BL.

### E3 · Allen–Cahn (`pinns/equations/allen_cahn.py` + `notebooks/03_allen_cahn.ipynb`)

**PDE:** `u_t - 10⁻⁴·u_xx + 5u³ - 5u = 0`, periodic BCs, `x∈[-1,1], t∈[0,1]`.

**Reference:** fine-grid numerical solution with `scipy.integrate.solve_ivp` on the Fourier-spectral semi-discrete system, or load existing dataset.

**Losses:** IC + periodic BC (penalise `u(-1,t) - u(1,t)` and `u_x(-1,t) - u_x(1,t)`) + PDE. `dim_lambda = 3` (no data term unless we add sparse observations).

**Deliverable:** same format.

---

## Group F — Comparison with other PINNs (`scripts/compare_pinnacle.py`)

**What:** Run LC-PINN alongside published PINN implementations on the same equations using identical train-test-split and sampling.

**How:**
1. Identify 1–2 open-source PINN repos for Burgers / Allen-Cahn (e.g. Raissi's original, PINNacle, DeepXDE).
2. Create `scripts/compare_pinnacle.py` that:
   - Generates a fixed seed dataset (`train_test_split` saved to `results/splits/{eq_name}_seed42.npz`).
   - Trains baseline PINNs from the external repo (subprocess call or direct import).
   - Trains LC-PINN on the same split.
   - Reports rel-L2 in same table format.
3. The split file is the single source of truth — both models load from it.
4. Document the external repo commit hash in `ADVISOR_BENCHMARKS.md`.

---

## Implementation order (given Apr 24 deadline)

Priority is paper-critical first:

| # | Item | Files | Why now |
|---|------|-------|---------|
| 1 | A1 LR warmup + 10× LR | `training.py`, `baseline.py` | Advisor said do this first; could unlock better convergence for everything else |
| 2 | A3 Small debug config | `config.py`, notebook | Needed to iterate quickly on everything below |
| 3 | D1 Weight-stability | notebook section 11 | Fast (inference only); strong paper result already in hand |
| 4 | D2 Bad-λ experiment | `inference.py`, `run_comparison.py` | The "killer demo" for the paper |
| 5 | C1 Prediction averaging | `inference.py` | Potentially removes the mid-T artifact; fast |
| 6 | B1 Simplex sampling | `lambda_sampler.py` | Ablation: does sampling strategy matter? |
| 7 | A2 Hard-example mining | `training.py`, `losses.py` | Requires retraining — schedule after LR warmup run finishes |
| 8 | C2 Cross-validate λ | `inference.py` | Needs averaging infrastructure from C1 |
| 9 | B2 Verify center λ | notebook / `run_comparison.py` | Quick sanity check |
| 10 | E1 Logistic ODE | new files | Simplest new equation; good smoke test for the framework |
| 11 | E2 Burgers | new files | Core benchmark for paper |
| 12 | E3 Allen–Cahn | new files | Stiffest; do last |
| 13 | F Pinnacle comparison | `scripts/` | Do only if time permits before deadline |

---

## Critical files

| File | Changes |
|------|---------|
| `pinns/training.py` | A1 (warmup), A2 (hard-example mining) |
| `pinns/baseline.py` | A1 (warmup for fair comparison) |
| `pinns/losses.py` | A2 (pointwise residual helper) |
| `pinns/config.py` | A3 (debug constants) |
| `pinns/lambda_sampler.py` | B1 (simplex mode) |
| `pinns/inference.py` | C1 (averaging), C2 (CV-λ), D2 (worst-λ finder) |
| `run_comparison.py` | B2, D2, C2 (new flags/methods) |
| `loss_conditional_pinn.ipynb` | A3 (DEBUG flag), D1 (section 11), B2 (center-λ check), C1 (averaged plots) |
| `pinns/equations/logistic.py` | E1 |
| `pinns/equations/burgers.py` | E2 |
| `pinns/equations/allen_cahn.py` | E3 |
| `notebooks/01_logistic.ipynb` | E1 |
| `notebooks/02_burgers.ipynb` | E2 |
| `notebooks/03_allen_cahn.ipynb` | E3 |
| `scripts/compare_pinnacle.py` | F |

---

## Verification

- After A1: retrain LC-PINN with `lr=1e-2, warmup_frac=0.3`; compare loss curve to `lr=1e-3` baseline — expect faster early descent, same or better final loss.
- After A2: run with `hard_frac=0.3`; confirm PDE loss decreases faster than without.
- After B1: check that simplex samples cover corners (min/max weight ≈ 0.05/0.85 in some samples); confirm training does not diverge.
- After C1: plot prediction ribbon; verify std is low at T=0.1 (well-learned region) and higher near T=0.5.
- After D2: bad-λ baseline should show visually broken prediction; LC-PINN at same λ should recover. This is the key figure.
- After E1–E3: each benchmark notebook should produce a three-row comparison table with rel-L2 for equal-weight, best-of-random, and LC-PINN.
- After F: single table comparing LC-PINN against at least one external PINN on Burgers.
- All existing tests in `tests/` must pass after every change: `poetry run pytest`.
