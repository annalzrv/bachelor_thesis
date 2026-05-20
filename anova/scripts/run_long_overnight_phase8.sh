#!/bin/bash
# Phase 8: final overflow. Queued behind phase 7.
set +e
set -u

cd "$(dirname "$0")/../.."
LOG_DIR="lc_anova/results/overnight_logs"
mkdir -p "$LOG_DIR"

CK_DIR="/Users/anna/Desktop/research/thesis/code/checkpoints"
MASTER="$LOG_DIR/master_phase8.log"

log() { echo "[$(date)] $*" | tee -a "$MASTER"; }

log "phase 8 waiting for PHASE 7 BATCH DONE..."
while ! grep -q "PHASE 7 BATCH DONE" "$LOG_DIR/master_phase7.log" 2>/dev/null; do
    sleep 60
done

log "=== PHASE 8 BATCH started ==="

# ============================================================
# P8.A. Extra HDMR-RNG seeds (30-49) at fixed config — wide distribution
# ============================================================
for rng_seed in 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49; do
    log "P8.A  HDMR-RNG seed ${rng_seed}"
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
hist = jh.fit(xy_tr, k_tr, u_tr, phase1_epochs=30, phase2_epochs=80, phase3_epochs=300, log_every=80)
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
" > "$LOG_DIR/P8A_rng${rng_seed}.log" 2>&1
done

# ============================================================
# P8.B. Extra LC-PINN-seed × HDMR-RNG-seed for Schrödinger and 1D Helm
# (push to 4 RNG seeds at each LC-PINN seed)
# ============================================================
for pde in helmholtz schrodinger; do
    tag_pde=$([ "$pde" = "helmholtz" ] && echo "helm1d" || echo "schr1d")
    for seed in 0 1 2 3; do
        for rng in 3 4 5; do
            log "P8.B  ${tag_pde} LC${seed} RNG${rng}"
            python -c "
import sys, json, numpy as np, torch
sys.path.insert(0, '/Users/anna/Desktop/research/anova')
sys.path.insert(0, '/Users/anna/Desktop/research/thesis/code')
torch.manual_seed(${rng}*100 + ${seed}); np.random.seed(${rng}*100 + ${seed})
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
" > "$LOG_DIR/P8B_${tag_pde}_lc${seed}_rng${rng}.log" 2>&1
        done
    done
done

# ============================================================
# P8.C. High-resolution per-k Sobol on ALL 2D Helm seeds (21 k values × N=50k)
# ============================================================
for seed in 0 1 2 3; do
    log "P8.C  per-k hi-res Sobol 2D Helm seed${seed} (21 k values, N=50k)"
    python -m lc_anova.pipelines.per_k_sobol \
        --checkpoint "$CK_DIR/lc_pinn_helmholtz_2d_seed${seed}_film_lbfgs_w64.pt" \
        --n-k 21 --N 50000 \
        --tag "per_k_hires_helm2d_seed${seed}" \
        > "$LOG_DIR/P8C_seed${seed}.log" 2>&1
done

# ============================================================
# P8.D. Mega-N MC-Sobol on 2D Helm seed 0 (N=1M)
# ============================================================
log "P8.D  mega-N MC-Sobol 2D Helm seed 0 (N=1000000)"
python -m lc_anova.pipelines.mc_sobol_helmholtz_2d \
    --checkpoint "$CK_DIR/lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt" \
    --N 1000000 --tag "mc_megaN_seed0" \
    > "$LOG_DIR/P8D_megaN.log" 2>&1

# ============================================================
# P8.E. Synthetic stress test — higher-dim, higher-order polynomial
# ============================================================
log "P8.E  synthetic stress test (d=4 polynomial)"
python -c "
import sys, json, numpy as np, torch
sys.path.insert(0, '/Users/anna/Desktop/research/anova')
from lc_anova.core.joint_hdmr import JointHDMR

# d=4 polynomial: u(z) = L1(z0) + L2(z1) + L1(z2) + L1(z3) + L1(z0)*L1(z2)
torch.manual_seed(42); np.random.seed(42)
rng = np.random.default_rng(42)
z_tr = rng.uniform(0, 1, (30000, 4)).astype(np.float32)
z_va = rng.uniform(0, 1, (30000, 4)).astype(np.float32)
def L1(z): return 2*z - 1
def L2(z): return 6*z*z - 6*z + 1
def u_fn(z): return L1(z[:,0]) + L2(z[:,1]) + L1(z[:,2]) + L1(z[:,3]) + L1(z[:,0])*L1(z[:,2])
u_tr = u_fn(z_tr).astype(np.float32)
u_va = u_fn(z_va).astype(np.float32)

# Sanity: build a d=4 order-2 HDMR. JointHDMR with dim_x=2, dim_lambda=2 -> d=4.
jh = JointHDMR(dim_x=2, dim_lambda=2, hidden=64, layers=2,
               max_order=2, use_fourier=False, num_freqs=4)
x_tr = torch.tensor(z_tr[:,:2]); l_tr = torch.tensor(z_tr[:,2:])
x_va = torch.tensor(z_va[:,:2]); l_va = torch.tensor(z_va[:,2:])
hist = jh.fit(x_tr, l_tr, torch.tensor(u_tr), phase1_epochs=30, phase2_epochs=80, log_every=60)
z_va_t = torch.cat([x_va, l_va], dim=1).to(jh.device)
y_va_c = torch.tensor(u_va).to(jh.device) - jh.y_mean
jh.model.eval()
with torch.no_grad():
    pred, _, _ = jh.model(z_va_t, include_pairs=True, purify=True)
    val_rel = (torch.sqrt(torch.mean((pred - y_va_c) ** 2)) / y_va_c.std()).item()
terms = jh.evaluate_terms(x_va, l_va)
json.dump({'d': 4, 'val_rel_rmse': val_rel,
           'sobol': {str(k): v for k, v in terms['sobol'].items()}},
          open('lc_anova/results/synthetic_d4_stress.json','w'), indent=2)
print(f'd=4 synthetic stress  val_rel_rmse={val_rel:.4f}')
" > "$LOG_DIR/P8E_synthetic_d4.log" 2>&1

# ============================================================
# P8.F. Final aggregation
# ============================================================
log "P8.F  final aggregation"
python -m lc_anova.plots.aggregate_results > "$LOG_DIR/P8F_aggregate.log" 2>&1
python -m lc_anova.plots.polished_plots > "$LOG_DIR/P8F_plots.log" 2>&1
python -m lc_anova.plots.amortization_plot > "$LOG_DIR/P8F_amort.log" 2>&1

log "=== PHASE 8 BATCH DONE ==="
