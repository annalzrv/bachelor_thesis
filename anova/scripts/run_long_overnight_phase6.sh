#!/bin/bash
# Phase 6: kicks in after phase 5 finishes, packs the remaining 12+ hours
# with aggressive multi-seed + capacity work.
set +e
set -u

cd "$(dirname "$0")/../.."
LOG_DIR="lc_anova/results/overnight_logs"
mkdir -p "$LOG_DIR" "lc_anova/results/figures" "lc_anova/results/sample_efficiency"

CK_DIR="/Users/anna/Desktop/research/thesis/code/checkpoints"
MASTER="$LOG_DIR/master_phase6.log"

log() { echo "[$(date)] $*" | tee -a "$MASTER"; }

# Wait for phase 5 (long overnight) to finish before starting.
log "phase 6 waiting for phase 5 LONG OVERNIGHT BATCH DONE..."
while ! grep -q "LONG OVERNIGHT BATCH DONE" "$LOG_DIR/master_long.log" 2>/dev/null; do
    sleep 30
done

log "=== PHASE 6 BATCH started ==="

# ============================================================
# P6.A. High-N MC-Sobol on every LC-PINN checkpoint we have
# ============================================================
for seed in 0 1 2 3; do
    log "P6.A  high-N MC-Sobol 2D Helm seed${seed}"
    python -m lc_anova.pipelines.mc_sobol_helmholtz_2d \
        --checkpoint "$CK_DIR/lc_pinn_helmholtz_2d_seed${seed}_film_lbfgs_w64.pt" \
        --N 200000 --tag "mc_highn200_seed${seed}" \
        > "$LOG_DIR/P6A_2d_seed${seed}.log" 2>&1
done

# ============================================================
# P6.B. Per-k Sobol on 2D Helm seeds 1, 3 (we already have 0, 2)
# ============================================================
# Already done in phase 5 step B. Skip.

# ============================================================
# P6.C. Per-alpha Sobol on Schrödinger seeds 1, 2, 3
# ============================================================
for seed in 1 2 3; do
    log "P6.C  per-alpha Sobol Schrödinger seed${seed}"
    python -c "
import sys
sys.path.insert(0, '/Users/anna/Desktop/research/anova')
sys.path.insert(0, '/Users/anna/Desktop/research/thesis/code')
import json, numpy as np, torch
from lc_anova.core.mc_sobol import mc_sobol_full
from lc_anova.pipelines.pde1d import load_lc_pinn, evaluate_lc_pinn_batch, pde_config
pde = pde_config('schrodinger')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model, _ = load_lc_pinn('$CK_DIR/lc_pinn_schrodinger_seed${seed}_film_lbfgs.pt', pde, device)
alpha_vals = np.linspace(0.5, 10.0, 21)
results = {'alpha_values': list(alpha_vals), 'var_Y': []}
for i, a in enumerate(alpha_vals):
    a_norm = pde['param_to_norm'](float(a))
    @torch.no_grad()
    def fn(x_np):
        x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
        a_t = torch.full((x_np.shape[0], 1), float(a_norm), dtype=torch.float32, device=device)
        return evaluate_lc_pinn_batch(model, x_t, a_t).cpu().numpy()
    sampler = lambda rng, n: rng.uniform(0,1,(n,1)).astype(np.float32)
    out = mc_sobol_full(fn, sampler, N=50000, d=1, seed=42+i)
    results['var_Y'].append(out['var_Y'])
json.dump(results, open('lc_anova/results/per_alpha_hires_schr1d_seed${seed}.json','w'), indent=2)
print('done seed${seed}')
" > "$LOG_DIR/P6C_seed${seed}.log" 2>&1
done

# ============================================================
# P6.D. Per-k Sobol on all 1D Helmholtz seeds  (we have 0, do 1-3)
# ============================================================
for seed in 0 1 2 3; do
    log "P6.D  per-k Sobol 1D Helm seed${seed}"
    python -c "
import sys
sys.path.insert(0, '/Users/anna/Desktop/research/anova')
sys.path.insert(0, '/Users/anna/Desktop/research/thesis/code')
import json, numpy as np, torch
from lc_anova.core.mc_sobol import mc_sobol_full
from lc_anova.pipelines.pde1d import load_lc_pinn, evaluate_lc_pinn_batch, pde_config
pde = pde_config('helmholtz')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model, _ = load_lc_pinn('$CK_DIR/lc_pinn_helmholtz_seed${seed}_film_lbfgs.pt', pde, device)
k_vals = np.linspace(1.0, 10.0, 21)
results = {'k_values': list(k_vals), 'var_Y': [], 'S_x': []}
for i, k in enumerate(k_vals):
    k_norm = pde['param_to_norm'](float(k))
    @torch.no_grad()
    def fn(x_np):
        x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
        k_t = torch.full((x_np.shape[0], 1), float(k_norm), dtype=torch.float32, device=device)
        return evaluate_lc_pinn_batch(model, x_t, k_t).cpu().numpy()
    sampler = lambda rng, n: rng.uniform(0,1,(n,1)).astype(np.float32)
    out = mc_sobol_full(fn, sampler, N=50000, d=1, seed=42+i)
    results['var_Y'].append(out['var_Y'])
    results['S_x'].append(1.0)  # d=1: all variance in single input
json.dump(results, open('lc_anova/results/per_k_hires_helm1d_seed${seed}.json','w'), indent=2)
print('done seed${seed}')
" > "$LOG_DIR/P6D_seed${seed}.log" 2>&1
done

# ============================================================
# P6.E. 3D Helmholtz Fourier HDMR (d=4, order-2)
# ============================================================
for seed in 0 1 2 3; do
    log "P6.E  HDMR on 3D Helm seed${seed} (d=4 order-2 Fourier)"
    python -c "
import sys
sys.path.insert(0, '/Users/anna/Desktop/research/anova')
sys.path.insert(0, '/Users/anna/Desktop/research/thesis/code')
import json, numpy as np, torch
from lc_anova.core.joint_hdmr import JointHDMR
from pinns.equations import helmholtz_3d as helm3d
from pinns.model import LossConditionalPINN

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
ck = torch.load('$CK_DIR/lc_pinn_helmholtz_3d_seed${seed}_film_lbfgs.pt', map_location=device, weights_only=False)
model = LossConditionalPINN(
    dim_phys=helm3d.DIM_PHYS, dim_lambda=helm3d.DIM_LAMBDA,
    hidden_dims=ck['hidden_dims'], conditioning=ck['conditioning'],
).to(device)
model.load_state_dict(ck['model_state_dict']); model.eval()

rng = np.random.default_rng(42)
N = 30000
xy_tr = rng.uniform(0, 1, (N, 3)).astype(np.float32)
k_tr = rng.uniform(-1, 1, (N, 1)).astype(np.float32)
xy_t = torch.tensor(xy_tr, device=device)
k_t = torch.tensor(k_tr, device=device)
with torch.no_grad():
    u = model(xy_t, k_t).squeeze(-1).detach().cpu()
u_tr = u

jh = JointHDMR(dim_x=3, dim_lambda=1, hidden=128, layers=3,
               max_order=2, use_fourier=True, num_freqs=6)
hist = jh.fit(xy_t, k_t, u_tr.to(device), phase1_epochs=40, phase2_epochs=120, log_every=30)

# val
xy_va = rng.uniform(0, 1, (N, 3)).astype(np.float32)
k_va = rng.uniform(-1, 1, (N, 1)).astype(np.float32)
xy_va_t = torch.tensor(xy_va, device=device)
k_va_t = torch.tensor(k_va, device=device)
with torch.no_grad():
    u_va = model(xy_va_t, k_va_t).squeeze(-1).cpu()
z_va = torch.cat([xy_va_t, k_va_t], dim=1).to(device)
y_va_c = u_va.to(device) - jh.y_mean
jh.model.eval()
with torch.no_grad():
    pred, _, _ = jh.model(z_va, include_pairs=True, purify=True)
    val_rel = (torch.sqrt(torch.mean((pred - y_va_c) ** 2)) / y_va_c.std()).item()

terms = jh.evaluate_terms(xy_va_t, k_va_t)
sobol = {str(k): v for k, v in terms['sobol'].items()}

json.dump({
    'pde': 'helmholtz_3d', 'seed': ${seed}, 'val_rel_rmse': val_rel,
    'sobol_indices': sobol, 'history': hist,
}, open('lc_anova/results/results_helm3d_hdmr_seed${seed}.json','w'), indent=2)
print(f'helm3d seed${seed} val_rel_rmse={val_rel:.4f}  sobol_n={len(sobol)}')
" > "$LOG_DIR/P6E_helm3d_seed${seed}.log" 2>&1
done

# ============================================================
# P6.F. Best-capacity HDMR on all 2D Helm seeds
# ============================================================
for seed in 0 2 3; do
    log "P6.F  best-cap HDMR 2D Helm seed${seed} (h=256, L=8, p3=800)"
    python -m lc_anova.pipelines.helmholtz_2d \
        --checkpoint "$CK_DIR/lc_pinn_helmholtz_2d_seed${seed}_film_lbfgs_w64.pt" \
        --max-order 3 --fourier --num-freqs 8 \
        --hidden 256 --layers 3 \
        --phase1-epochs 40 --phase2-epochs 80 --phase3-epochs 800 \
        --tag "bestcap_helm2d_seed${seed}" \
        > "$LOG_DIR/P6F_seed${seed}.log" 2>&1
done

# ============================================================
# P6.G. HDMR-RNG-seed distribution at fixed best config (LC-PINN seed 0)
# ============================================================
for rng_seed in 0 1 2 3 4 5 6 7 8 9; do
    log "P6.G  HDMR-RNG seed ${rng_seed} on 2D Helm LC-PINN seed 0"
    python -c "
import sys, json, numpy as np, torch
sys.path.insert(0, '/Users/anna/Desktop/research/anova')
sys.path.insert(0, '/Users/anna/Desktop/research/thesis/code')
torch.manual_seed(1000 + ${rng_seed}); np.random.seed(1000 + ${rng_seed})
from lc_anova.pipelines.helmholtz_2d import load_lc_pinn, evaluate_lc_pinn_batch, sample_joint
from lc_anova.core.joint_hdmr import JointHDMR
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model, _ = load_lc_pinn('$CK_DIR/lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt', device)
xy_tr, k_tr = sample_joint(30000, 42, device)
u_tr = evaluate_lc_pinn_batch(model, xy_tr, k_tr)
xy_va, k_va = sample_joint(30000, 43, device)
u_va = evaluate_lc_pinn_batch(model, xy_va, k_va)
jh = JointHDMR(dim_x=2, dim_lambda=1, hidden=128, layers=3,
               max_order=3, use_fourier=True, num_freqs=6)
hist = jh.fit(xy_tr, k_tr, u_tr, phase1_epochs=30, phase2_epochs=80, phase3_epochs=300, log_every=30)
z_va = torch.cat([xy_va, k_va], dim=1).to(device)
y_va_c = u_va.to(device) - jh.y_mean
jh.model.eval()
with torch.no_grad():
    pred, *_ = jh.model(z_va, include_pairs=True, include_triplet=True, purify=True)
    val_rel = (torch.sqrt(torch.mean((pred - y_va_c) ** 2)) / y_va_c.std()).item()
terms = jh.evaluate_terms(xy_va, k_va)
json.dump({'rng_seed': ${rng_seed}, 'val_rel_rmse': val_rel,
           'sobol': {str(k): v for k, v in terms['sobol'].items()}},
          open('lc_anova/results/hdmr_rng${rng_seed}_helm2d_seed0.json','w'), indent=2)
print(f'rng_seed ${rng_seed}  val_rel_rmse={val_rel:.4f}')
" > "$LOG_DIR/P6G_rng${rng_seed}.log" 2>&1
done

# ============================================================
# P6.H. Sample efficiency at extreme N
# ============================================================
for n_train in 300000 1000000; do
    log "P6.H  sample efficiency 2D Helm N=${n_train}"
    python -m lc_anova.pipelines.helmholtz_2d \
        --checkpoint "$CK_DIR/lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt" \
        --max-order 3 --fourier --num-freqs 6 \
        --hidden 128 --layers 3 \
        --n-samples "$n_train" --n-val 30000 \
        --phase1-epochs 30 --phase2-epochs 60 --phase3-epochs 200 \
        --tag "sampeff_n${n_train}" \
        > "$LOG_DIR/P6H_n${n_train}.log" 2>&1
done

# ============================================================
# P6.I. Multi-LC-PINN-seed + multi-HDMR-seed for d=2 PDEs
# ============================================================
for pde in helmholtz schrodinger; do
    for seed in 0 1 2 3; do
        for rng in 0 1 2; do
            tag_pde=$([ "$pde" = "helmholtz" ] && echo "helm1d" || echo "schr1d")
            log "P6.I  ${tag_pde} LC seed${seed} RNG${rng}"
            python -c "
import sys, json, numpy as np, torch
sys.path.insert(0, '/Users/anna/Desktop/research/anova')
sys.path.insert(0, '/Users/anna/Desktop/research/thesis/code')
torch.manual_seed(${rng}*100); np.random.seed(${rng}*100)
from lc_anova.pipelines.pde1d import load_lc_pinn, evaluate_lc_pinn_batch, sample_joint, pde_config
from lc_anova.core.joint_hdmr import JointHDMR
pde = pde_config('${pde}')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model, _ = load_lc_pinn('$CK_DIR/lc_pinn_${pde}_seed${seed}_film_lbfgs.pt', pde, device)
x_tr, p_tr = sample_joint(20000, 42, pde, device)
u_tr = evaluate_lc_pinn_batch(model, x_tr, p_tr)
x_va, p_va = sample_joint(20000, 43, pde, device)
u_va = evaluate_lc_pinn_batch(model, x_va, p_va)
jh = JointHDMR(dim_x=1, dim_lambda=1, hidden=64, layers=2,
               max_order=2, use_fourier=True, num_freqs=4)
hist = jh.fit(x_tr, p_tr, u_tr, phase1_epochs=30, phase2_epochs=60, log_every=60)
z_va = torch.cat([x_va, p_va], dim=1).to(device)
y_va_c = u_va.to(device) - jh.y_mean
jh.model.eval()
with torch.no_grad():
    pred, _, _ = jh.model(z_va, include_pairs=True, purify=True)
    val_rel = (torch.sqrt(torch.mean((pred - y_va_c) ** 2)) / y_va_c.std()).item()
terms = jh.evaluate_terms(x_va, p_va)
json.dump({'pde': '${pde}', 'lc_seed': ${seed}, 'rng_seed': ${rng}, 'val_rel_rmse': val_rel,
           'sobol': {str(k): v for k, v in terms['sobol'].items()}},
          open('lc_anova/results/multiseed_${tag_pde}_lc${seed}_rng${rng}.json','w'), indent=2)
print(f'${tag_pde} lc${seed} rng${rng}  val_rel_rmse={val_rel:.4f}')
" > "$LOG_DIR/P6I_${tag_pde}_lc${seed}_rng${rng}.log" 2>&1
        done
    done
done

# ============================================================
# P6.J. Convergence study: Sobol vs N_MC
# ============================================================
log "P6.J  MC-Sobol convergence study on 2D Helm seed 0"
python -c "
import sys, json, time, numpy as np, torch
sys.path.insert(0, '/Users/anna/Desktop/research/anova')
sys.path.insert(0, '/Users/anna/Desktop/research/thesis/code')
from lc_anova.core.mc_sobol import mc_sobol_full, helmholtz_2d_sampler
from lc_anova.pipelines.helmholtz_2d import load_lc_pinn, evaluate_lc_pinn_batch
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model, _ = load_lc_pinn('$CK_DIR/lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt', device)
@torch.no_grad()
def fn(z_np):
    z = torch.tensor(z_np, dtype=torch.float32, device=device)
    return evaluate_lc_pinn_batch(model, z[:,:2], z[:,2:3]).cpu().numpy()
results = []
for N in [1000, 3000, 10000, 30000, 100000, 300000, 1000000]:
    t0 = time.perf_counter()
    out = mc_sobol_full(fn, helmholtz_2d_sampler, N=N, d=3, seed=42)
    wall = time.perf_counter() - t0
    results.append({'N': N, 'wall': wall, 'triplet': out['S_triplet'],
                    'S_xy': out['S_pair'][(0, 1)],
                    'S_x': out['S_first'][(0,)]})
    print(f'N={N}  wall={wall:.2f}s  triplet={out[\"S_triplet\"]:.4f}')
json.dump(results, open('lc_anova/results/mc_convergence.json','w'), indent=2)
" > "$LOG_DIR/P6J_convergence.log" 2>&1

# ============================================================
# P6.K. Aggregator + polished plots regeneration
# ============================================================
log "P6.K  regenerate aggregator + polished plots"
python -m lc_anova.plots.aggregate_results > "$LOG_DIR/P6K_aggregate.log" 2>&1
python -m lc_anova.plots.polished_plots > "$LOG_DIR/P6K_plots.log" 2>&1
python -m lc_anova.plots.amortization_plot > "$LOG_DIR/P6K_amort.log" 2>&1

log "=== PHASE 6 BATCH DONE ==="
