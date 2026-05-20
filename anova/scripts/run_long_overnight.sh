#!/bin/bash
# 15-hour autonomous overnight on MPS.
# Sequential — MPS can't share between processes. Each step is in `set +e`
# so failures don't kill the batch. All output goes to lc_anova/results/.
#
# Plan (rough timing assumes hidden=128, L=6, 540 epochs ≈ 15 min for d=3
# Fourier HDMR; d=2 ≈ 1 min):
#
#   A. Quick MC-Sobol fillins                       ~20 min
#   B. MC-Sobol per-k on remaining 2D Helm seeds    ~5 min
#   C. HDMR capacity sweep on 2D Helm seed 0        ~5 hours
#   D. Multi-seed high-capacity HDMR (seeds 2, 3)   ~4 hours
#   E. Sample-efficiency study                       ~3 hours
#   F. Per-alpha hi-res for Schrödinger             ~30 min
#   G. 3D Helmholtz MC-Sobol (partial gold standard) ~10 min
#   H. Failure-mode renders                         ~20 min
#   I. Polished plots regeneration                  ~5 min

set +e   # Continue on failures — better to lose a step than the batch.
set -u

cd "$(dirname "$0")/../.."

LOG_DIR="lc_anova/results/overnight_logs"
mkdir -p "$LOG_DIR" "lc_anova/results/figures" "lc_anova/results/sample_efficiency"

CK_DIR="/Users/anna/Desktop/research/thesis/code/checkpoints"
MASTER="$LOG_DIR/master_long.log"

log() { echo "[$(date)] $*" | tee -a "$MASTER"; }

log "=== LONG OVERNIGHT BATCH started ==="

# ============================================================
# A. Quick MC-Sobol fill-ins (high-N gold standard)
# ============================================================
log "A.1  high-N MC-Sobol Schrödinger seed0 (N=100000)"
python -c "
import sys
sys.path.insert(0, '/Users/anna/Desktop/research/anova')
sys.path.insert(0, '/Users/anna/Desktop/research/thesis/code')
import json, numpy as np, torch
from lc_anova.core.mc_sobol import mc_sobol_full
from lc_anova.pipelines.pde1d import load_lc_pinn, evaluate_lc_pinn_batch, pde_config
pde = pde_config('schrodinger')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model, _ = load_lc_pinn('$CK_DIR/lc_pinn_schrodinger_seed0_film_lbfgs.pt', pde, device)
def sampler(rng, n):
    return np.concatenate([rng.uniform(0,1,(n,1)), rng.uniform(-1,1,(n,1))], axis=1).astype(np.float32)
@torch.no_grad()
def fn(z_np):
    z = torch.tensor(z_np, dtype=torch.float32, device=device)
    return evaluate_lc_pinn_batch(model, z[:,:1], z[:,1:2]).cpu().numpy()
out = mc_sobol_full(fn, sampler, N=100000, d=2, seed=42)
json.dump({'S_first': {str(k):v for k,v in out['S_first'].items()},
           'S_total': {str(k):v for k,v in out['S_total'].items()},
           'S_pair':  {str(k):v for k,v in out['S_pair'].items()},
           'var_Y': out['var_Y']},
          open('lc_anova/results/mc_sobol_schr1d_highn_seed0.json','w'), indent=2)
" > "$LOG_DIR/A1_schr_highn.log" 2>&1

log "A.2  high-N MC-Sobol 1D Helmholtz seed0 (N=100000)"
python -c "
import sys
sys.path.insert(0, '/Users/anna/Desktop/research/anova')
sys.path.insert(0, '/Users/anna/Desktop/research/thesis/code')
import json, numpy as np, torch
from lc_anova.core.mc_sobol import mc_sobol_full
from lc_anova.pipelines.pde1d import load_lc_pinn, evaluate_lc_pinn_batch, pde_config
pde = pde_config('helmholtz')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model, _ = load_lc_pinn('$CK_DIR/lc_pinn_helmholtz_seed0_film_lbfgs.pt', pde, device)
def sampler(rng, n):
    return np.concatenate([rng.uniform(0,1,(n,1)), rng.uniform(-1,1,(n,1))], axis=1).astype(np.float32)
@torch.no_grad()
def fn(z_np):
    z = torch.tensor(z_np, dtype=torch.float32, device=device)
    return evaluate_lc_pinn_batch(model, z[:,:1], z[:,1:2]).cpu().numpy()
out = mc_sobol_full(fn, sampler, N=100000, d=2, seed=42)
json.dump({'S_first': {str(k):v for k,v in out['S_first'].items()},
           'S_total': {str(k):v for k,v in out['S_total'].items()},
           'S_pair':  {str(k):v for k,v in out['S_pair'].items()},
           'var_Y': out['var_Y']},
          open('lc_anova/results/mc_sobol_helm1d_highn_seed0.json','w'), indent=2)
" > "$LOG_DIR/A2_helm1d_highn.log" 2>&1

# ============================================================
# B. Per-k Sobol on remaining 2D Helm seeds
# ============================================================
for seed in 1 3; do
    log "B  per-k Sobol 2D Helm seed${seed}"
    python -m lc_anova.pipelines.per_k_sobol \
        --checkpoint "$CK_DIR/lc_pinn_helmholtz_2d_seed${seed}_film_lbfgs_w64.pt" \
        --n-k 11 --N 10000 \
        --tag "per_k_helm2d_seed${seed}" \
        > "$LOG_DIR/B_per_k_seed${seed}.log" 2>&1
done

# ============================================================
# C. HDMR capacity sweep on 2D Helm seed 0 — push past 80% capture
# ============================================================
declare -a CONFIGS=(
    "256 3 6 600 cap_h256_L6_p600"
    "128 3 8 600 cap_h128_L8_p600"
    "256 3 8 600 cap_h256_L8_p600"
    "256 4 8 600 cap_h256_L8_d4_p600"
    "128 3 12 600 cap_h128_L12_p600"
    "192 3 10 800 cap_h192_L10_p800"
)
for cfg in "${CONFIGS[@]}"; do
    read -r hidden layers nfreq p3 tag <<< "$cfg"
    log "C  HDMR capacity sweep  ${tag}  (h=${hidden} L=${nfreq} layers=${layers} p3=${p3})"
    python -m lc_anova.pipelines.helmholtz_2d \
        --checkpoint "$CK_DIR/lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt" \
        --max-order 3 --fourier --num-freqs "$nfreq" \
        --hidden "$hidden" --layers "$layers" \
        --phase1-epochs 40 --phase2-epochs 80 --phase3-epochs "$p3" \
        --tag "${tag}" \
        > "$LOG_DIR/C_${tag}.log" 2>&1
done

# ============================================================
# D. Multi-seed HDMR at best capacity (seeds 2, 3)
# ============================================================
for seed in 2 3; do
    log "D  multi-seed high-cap HDMR 2D Helm seed${seed}"
    python -m lc_anova.pipelines.helmholtz_2d \
        --checkpoint "$CK_DIR/lc_pinn_helmholtz_2d_seed${seed}_film_lbfgs_w64.pt" \
        --max-order 3 --fourier --num-freqs 8 \
        --hidden 256 --layers 3 \
        --phase1-epochs 40 --phase2-epochs 80 --phase3-epochs 600 \
        --tag "highcap_helm2d_seed${seed}" \
        > "$LOG_DIR/D_seed${seed}.log" 2>&1
done

# ============================================================
# E. Sample-efficiency study on 2D Helm seed 0
# ============================================================
for n_train in 5000 10000 30000 100000; do
    log "E  sample efficiency 2D Helm n_train=${n_train}"
    python -m lc_anova.pipelines.helmholtz_2d \
        --checkpoint "$CK_DIR/lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt" \
        --max-order 3 --fourier --num-freqs 6 \
        --hidden 128 --layers 3 \
        --n-samples "$n_train" --n-val 30000 \
        --phase1-epochs 40 --phase2-epochs 80 --phase3-epochs 400 \
        --tag "sampeff_n${n_train}_seed0" \
        > "$LOG_DIR/E_n${n_train}.log" 2>&1
done

# ============================================================
# F. Per-alpha Sobol on Schrödinger LC-PINN (hi-res)
# ============================================================
log "F  per-alpha hi-res Sobol Schrödinger seed0 (N=50k, 21 alpha values)"
python -c "
import sys
sys.path.insert(0, '/Users/anna/Desktop/research/anova')
sys.path.insert(0, '/Users/anna/Desktop/research/thesis/code')
import json, numpy as np, torch
from lc_anova.core.mc_sobol import mc_sobol_full
from lc_anova.pipelines.pde1d import load_lc_pinn, evaluate_lc_pinn_batch, pde_config
pde = pde_config('schrodinger')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model, _ = load_lc_pinn('$CK_DIR/lc_pinn_schrodinger_seed0_film_lbfgs.pt', pde, device)
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
    print(f'alpha={a:.2f}  Var={out[\"var_Y\"]:.4f}')
json.dump(results, open('lc_anova/results/per_alpha_hires_schr1d_seed0.json','w'), indent=2)
" > "$LOG_DIR/F_per_alpha_hires.log" 2>&1

# ============================================================
# G. 3D Helmholtz MC-Sobol (d=4 first-order + total + partial pairs)
# ============================================================
log "G  3D Helmholtz MC-Sobol seed0"
python -c "
import sys
sys.path.insert(0, '/Users/anna/Desktop/research/anova')
sys.path.insert(0, '/Users/anna/Desktop/research/thesis/code')
import json, numpy as np, torch
from lc_anova.core.mc_sobol import mc_sobol_full
from pinns.equations import helmholtz_3d as helm3d
from pinns.model import LossConditionalPINN

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
ck = torch.load('$CK_DIR/lc_pinn_helmholtz_3d_seed0_film_lbfgs.pt', map_location=device, weights_only=False)
model = LossConditionalPINN(
    dim_phys=helm3d.DIM_PHYS, dim_lambda=helm3d.DIM_LAMBDA,
    hidden_dims=ck['hidden_dims'], conditioning=ck['conditioning'],
).to(device)
model.load_state_dict(ck['model_state_dict']); model.eval()

def sampler(rng, n):
    z = np.empty((n, 4), dtype=np.float32)
    z[:, :3] = rng.uniform(0, 1, size=(n, 3))
    z[:, 3] = rng.uniform(-1, 1, size=n)
    return z

@torch.no_grad()
def fn(z_np):
    z = torch.tensor(z_np, dtype=torch.float32, device=device)
    u = model(z[:, :3], z[:, 3:4]).squeeze(-1)
    return u.detach().cpu().numpy()

out = mc_sobol_full(fn, sampler, N=30000, d=4, seed=42)
json.dump({
    'd': 4,
    'S_first': {str(k): v for k, v in out['S_first'].items()},
    'S_total': {str(k): v for k, v in out['S_total'].items()},
    'S_pair':  {str(k): v for k, v in out['S_pair'].items()},
    'var_Y': out['var_Y'],
}, open('lc_anova/results/mc_sobol_helm3d_seed0.json', 'w'), indent=2)
print('3D Helmholtz MC-Sobol done')
" > "$LOG_DIR/G_helm3d.log" 2>&1

# ============================================================
# H. Regenerate aggregator + polished plots
# ============================================================
log "H  regenerate aggregator + polished plots"
python -m lc_anova.plots.aggregate_results > "$LOG_DIR/H_aggregate.log" 2>&1
python -m lc_anova.plots.polished_plots > "$LOG_DIR/H_plots.log" 2>&1
python -m lc_anova.plots.amortization_plot > "$LOG_DIR/H_amort.log" 2>&1

log "=== LONG OVERNIGHT BATCH DONE ==="
