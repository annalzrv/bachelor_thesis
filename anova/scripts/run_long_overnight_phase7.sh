#!/bin/bash
# Phase 7: queued behind phase 6. Heavy multi-seed, multi-config, ablation work.
set +e
set -u

cd "$(dirname "$0")/../.."
LOG_DIR="lc_anova/results/overnight_logs"
mkdir -p "$LOG_DIR"

CK_DIR="/Users/anna/Desktop/research/thesis/code/checkpoints"
MASTER="$LOG_DIR/master_phase7.log"

log() { echo "[$(date)] $*" | tee -a "$MASTER"; }

log "phase 7 waiting for PHASE 6 BATCH DONE..."
while ! grep -q "PHASE 6 BATCH DONE" "$LOG_DIR/master_phase6.log" 2>/dev/null; do
    sleep 30
done

log "=== PHASE 7 BATCH started ==="

# ============================================================
# P7.A. Very-high-capacity HDMR on 2D Helm — push capture past 90% if possible
# ============================================================
declare -a CONFIGS=(
    "512 4 10 1500 vhigh_h512_L10_d4_p1500"
    "384 4 12 1500 vhigh_h384_L12_d4_p1500"
    "256 5 10 2000 vhigh_h256_L10_d5_p2000"
)
for cfg in "${CONFIGS[@]}"; do
    read -r hidden layers nfreq p3 tag <<< "$cfg"
    log "P7.A  ${tag}  (h=${hidden} L=${nfreq} layers=${layers} p3=${p3})"
    python -m lc_anova.pipelines.helmholtz_2d \
        --checkpoint "$CK_DIR/lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt" \
        --max-order 3 --fourier --num-freqs "$nfreq" \
        --hidden "$hidden" --layers "$layers" \
        --phase1-epochs 40 --phase2-epochs 100 --phase3-epochs "$p3" \
        --tag "${tag}" \
        > "$LOG_DIR/P7A_${tag}.log" 2>&1
done

# ============================================================
# P7.B. Extra HDMR-RNG seeds (10-29) at fixed best config on 2D Helm seed 0
# ============================================================
for rng_seed in 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29; do
    log "P7.B  HDMR-RNG seed ${rng_seed} on 2D Helm LC-PINN seed 0"
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
hist = jh.fit(xy_tr, k_tr, u_tr, phase1_epochs=30, phase2_epochs=80, phase3_epochs=300, log_every=60)
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
" > "$LOG_DIR/P7B_rng${rng_seed}.log" 2>&1
done

# ============================================================
# P7.C. Ablation: tanh-only HDMR at various capacities on 2D Helm
# ============================================================
declare -a TANH_CONFIGS=(
    "64 2  tanh_h64_d2"
    "128 3 tanh_h128_d3"
    "256 3 tanh_h256_d3"
    "256 4 tanh_h256_d4"
)
for cfg in "${TANH_CONFIGS[@]}"; do
    read -r hidden layers tag <<< "$cfg"
    log "P7.C  tanh ablation ${tag}"
    python -m lc_anova.pipelines.helmholtz_2d \
        --checkpoint "$CK_DIR/lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt" \
        --max-order 3 \
        --hidden "$hidden" --layers "$layers" \
        --phase1-epochs 40 --phase2-epochs 80 --phase3-epochs 400 \
        --tag "${tag}" \
        > "$LOG_DIR/P7C_${tag}.log" 2>&1
done

# ============================================================
# P7.D. Ablation: order-2 only (no triplet) Fourier HDMR on 2D Helm
# ============================================================
log "P7.D  order-2 Fourier ablation on 2D Helm seed 0"
python -c "
import sys, json, numpy as np, torch
sys.path.insert(0, '/Users/anna/Desktop/research/anova')
sys.path.insert(0, '/Users/anna/Desktop/research/thesis/code')
from lc_anova.pipelines.helmholtz_2d import load_lc_pinn, evaluate_lc_pinn_batch, sample_joint
from lc_anova.core.joint_hdmr import JointHDMR
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model, _ = load_lc_pinn('$CK_DIR/lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt', device)
xy_tr, k_tr = sample_joint(30000, 42, device)
u_tr = evaluate_lc_pinn_batch(model, xy_tr, k_tr)
xy_va, k_va = sample_joint(30000, 43, device)
u_va = evaluate_lc_pinn_batch(model, xy_va, k_va)
jh = JointHDMR(dim_x=2, dim_lambda=1, hidden=128, layers=3,
               max_order=2, use_fourier=True, num_freqs=6)
hist = jh.fit(xy_tr, k_tr, u_tr, phase1_epochs=40, phase2_epochs=120, log_every=40)
z_va = torch.cat([xy_va, k_va], dim=1).to(device)
y_va_c = u_va.to(device) - jh.y_mean
jh.model.eval()
with torch.no_grad():
    pred, _, _ = jh.model(z_va, include_pairs=True, purify=True)
    val_rel = (torch.sqrt(torch.mean((pred - y_va_c) ** 2)) / y_va_c.std()).item()
terms = jh.evaluate_terms(xy_va, k_va)
json.dump({'config': 'order2_fourier', 'val_rel_rmse': val_rel,
           'sobol': {str(k): v for k, v in terms['sobol'].items()}},
          open('lc_anova/results/ablation_order2_fourier.json','w'), indent=2)
print(f'order-2 Fourier  val_rel_rmse={val_rel:.4f}')
" > "$LOG_DIR/P7D_order2_fourier.log" 2>&1

# ============================================================
# P7.E. Cross-seed HDMR generalization: train on seed 0 LC-PINN, evaluate on seed 2 data
# ============================================================
log "P7.E  cross-seed HDMR generalization"
python -c "
import sys, json, numpy as np, torch
sys.path.insert(0, '/Users/anna/Desktop/research/anova')
sys.path.insert(0, '/Users/anna/Desktop/research/thesis/code')
from lc_anova.pipelines.helmholtz_2d import load_lc_pinn, evaluate_lc_pinn_batch, sample_joint
from lc_anova.core.joint_hdmr import JointHDMR
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Train HDMR on seed 0 LC-PINN
model_train, _ = load_lc_pinn('$CK_DIR/lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt', device)
xy_tr, k_tr = sample_joint(30000, 42, device)
u_tr = evaluate_lc_pinn_batch(model_train, xy_tr, k_tr)

jh = JointHDMR(dim_x=2, dim_lambda=1, hidden=128, layers=3,
               max_order=3, use_fourier=True, num_freqs=6)
jh.fit(xy_tr, k_tr, u_tr, phase1_epochs=30, phase2_epochs=80, phase3_epochs=300, log_every=60)

# Evaluate on seed 2 LC-PINN samples
model_test, _ = load_lc_pinn('$CK_DIR/lc_pinn_helmholtz_2d_seed2_film_lbfgs_w64.pt', device)
xy_va, k_va = sample_joint(30000, 99, device)
u_va_test = evaluate_lc_pinn_batch(model_test, xy_va, k_va)

z_va = torch.cat([xy_va, k_va], dim=1).to(device)
y_va_c = u_va_test.to(device) - jh.y_mean
jh.model.eval()
with torch.no_grad():
    pred, *_ = jh.model(z_va, include_pairs=True, include_triplet=True, purify=True)
    val_rel = (torch.sqrt(torch.mean((pred - y_va_c) ** 2)) / y_va_c.std()).item()

# Also evaluate on own (seed-0) samples for baseline
xy_v0, k_v0 = sample_joint(30000, 43, device)
u_v0 = evaluate_lc_pinn_batch(model_train, xy_v0, k_v0)
z_v0 = torch.cat([xy_v0, k_v0], dim=1).to(device)
y_v0_c = u_v0.to(device) - jh.y_mean
with torch.no_grad():
    pred0, *_ = jh.model(z_v0, include_pairs=True, include_triplet=True, purify=True)
    val_rel_own = (torch.sqrt(torch.mean((pred0 - y_v0_c) ** 2)) / y_v0_c.std()).item()

json.dump({
    'train_on_seed': 0,
    'eval_on_seed': 2,
    'val_rel_rmse_own_seed': val_rel_own,
    'val_rel_rmse_cross_seed': val_rel,
}, open('lc_anova/results/cross_seed_hdmr.json','w'), indent=2)
print(f'own-seed val_rel: {val_rel_own:.4f}, cross-seed val_rel: {val_rel:.4f}')
" > "$LOG_DIR/P7E_cross_seed.log" 2>&1

# ============================================================
# P7.F. Multi-LC-PINN-seed × bestcap on Schrödinger and 1D Helm
# ============================================================
for pde in helmholtz schrodinger; do
    tag_pde=$([ "$pde" = "helmholtz" ] && echo "helm1d" || echo "schr1d")
    for seed in 0 1 2 3; do
        log "P7.F  bestcap ${tag_pde} LC seed${seed}"
        python -m lc_anova.pipelines.pde1d \
            --pde "$pde" \
            --checkpoint "$CK_DIR/lc_pinn_${pde}_seed${seed}_film_lbfgs.pt" \
            --fourier --hidden 256 --layers 3 --num-freqs 8 \
            --phase1-epochs 60 --phase2-epochs 200 \
            --tag "bestcap_${tag_pde}_seed${seed}" \
            > "$LOG_DIR/P7F_${tag_pde}_seed${seed}.log" 2>&1
    done
done

# ============================================================
# P7.G. Sample efficiency at very extreme N (multi-seed)
# ============================================================
for n_train in 5000000; do
    log "P7.G  extreme sample efficiency 2D Helm N=${n_train}"
    python -m lc_anova.pipelines.helmholtz_2d \
        --checkpoint "$CK_DIR/lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt" \
        --max-order 3 --fourier --num-freqs 6 \
        --hidden 128 --layers 3 \
        --n-samples "$n_train" --n-val 30000 \
        --phase1-epochs 20 --phase2-epochs 40 --phase3-epochs 100 \
        --tag "sampeff_extreme_n${n_train}" \
        > "$LOG_DIR/P7G_n${n_train}.log" 2>&1
done

# ============================================================
# P7.H. Final aggregation + plot regeneration
# ============================================================
log "P7.H  final aggregation"
python -m lc_anova.plots.aggregate_results > "$LOG_DIR/P7H_aggregate.log" 2>&1
python -m lc_anova.plots.polished_plots > "$LOG_DIR/P7H_plots.log" 2>&1
python -m lc_anova.plots.amortization_plot > "$LOG_DIR/P7H_amort.log" 2>&1

log "=== PHASE 7 BATCH DONE ==="
