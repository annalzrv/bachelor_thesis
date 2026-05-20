#!/bin/bash
# Phase 4: cross-method MC-Sobol on d=2 PDEs + per-alpha Schrödinger Sobol +
# higher-N gold standard for 2D Helm seed 0.
set -u

cd "$(dirname "$0")/../.."
LOG_DIR="lc_anova/results/overnight_logs"
mkdir -p "$LOG_DIR"

CK_DIR="/Users/anna/Desktop/research/thesis/code/checkpoints"
MASTER="$LOG_DIR/master_phase4.log"

echo "=== phase4 batch started $(date) ===" | tee -a "$MASTER"

# === MC-Sobol cross-method on 1D Helmholtz LC-PINNs ===
for seed in 0 1 2 3; do
    echo "[$(date)] MC-Sobol 1D Helm seed${seed}" | tee -a "$MASTER"
    python -m lc_anova.pipelines.compute_cost_benchmark \
        --lc-checkpoint "$CK_DIR/lc_pinn_helmholtz_seed${seed}_film_lbfgs.pt" \
        --relobralo-pattern "relobralo_helmholtz_seed${seed}_k.*\.pt" \
        --N 20000 --dim-phys 1 \
        --out "lc_anova/results/mc_sobol_helm1d_seed${seed}.json" \
        > "$LOG_DIR/mc_sobol_helm1d_seed${seed}.log" 2>&1
done

# === MC-Sobol on Schrödinger LC-PINNs ===
# Note: no per-alpha retrained models exist; just LC-PINN MC-Sobol.
for seed in 0 1 2 3; do
    echo "[$(date)] MC-Sobol Schrödinger seed${seed}" | tee -a "$MASTER"
    python -c "
import sys
sys.path.insert(0, '/Users/anna/Desktop/research/anova')
sys.path.insert(0, '/Users/anna/Desktop/research/thesis/code')
import json
import numpy as np
import torch
from lc_anova.core.mc_sobol import mc_sobol_full
from lc_anova.pipelines.pde1d import load_lc_pinn, evaluate_lc_pinn_batch, pde_config

pde = pde_config('schrodinger')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
ck = '$CK_DIR/lc_pinn_schrodinger_seed${seed}_film_lbfgs.pt'
model, _ = load_lc_pinn(ck, pde, device)

def sampler(rng, n):
    x = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
    a = rng.uniform(-1.0, 1.0, size=(n, 1)).astype(np.float32)
    return np.concatenate([x, a], axis=1)

@torch.no_grad()
def model_fn(z_np):
    z = torch.tensor(z_np, dtype=torch.float32, device=device)
    u = evaluate_lc_pinn_batch(model, z[:, :1], z[:, 1:2])
    return u.detach().cpu().numpy()

out = mc_sobol_full(model_fn, sampler, N=20000, d=2, seed=42)
payload = {
    'checkpoint': ck,
    'var_Y': out['var_Y'],
    'S_first': {str(k): v for k, v in out['S_first'].items()},
    'S_total': {str(k): v for k, v in out['S_total'].items()},
    'S_pair': {str(k): v for k, v in out['S_pair'].items()},
}
with open('lc_anova/results/mc_sobol_schr1d_seed${seed}.json', 'w') as f:
    json.dump(payload, f, indent=2)
print('S_first:', payload['S_first'])
print('S_pair:', payload['S_pair'])
" > "$LOG_DIR/mc_sobol_schr1d_seed${seed}.log" 2>&1
done

# === Per-alpha Sobol for Schrödinger (parallel to per-k for 2D Helm) ===
echo "[$(date)] per-alpha Sobol Schrödinger seed0" | tee -a "$MASTER"
python -c "
import sys
sys.path.insert(0, '/Users/anna/Desktop/research/anova')
sys.path.insert(0, '/Users/anna/Desktop/research/thesis/code')
import json
import numpy as np
import torch
from lc_anova.core.mc_sobol import mc_sobol_full
from lc_anova.pipelines.pde1d import load_lc_pinn, evaluate_lc_pinn_batch, pde_config

pde = pde_config('schrodinger')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model, _ = load_lc_pinn('$CK_DIR/lc_pinn_schrodinger_seed0_film_lbfgs.pt', pde, device)

alpha_values = np.linspace(0.5, 10.0, 11)
results = {'alpha_values': list(alpha_values), 'var_Y_at_a': [], 'S_x_at_a': []}
for i, a in enumerate(alpha_values):
    a_norm = pde['param_to_norm'](float(a))
    @torch.no_grad()
    def fn(x_np):
        x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
        a_t = torch.full((x_np.shape[0], 1), float(a_norm), dtype=torch.float32, device=device)
        u = evaluate_lc_pinn_batch(model, x_t, a_t)
        return u.detach().cpu().numpy()
    sampler = lambda rng, n: rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
    out = mc_sobol_full(fn, sampler, N=20000, d=1, seed=42+i)
    results['var_Y_at_a'].append(out['var_Y'])
    results['S_x_at_a'].append(out['S_first'][(0,)])
    print(f'  alpha={a:.2f}  Var={out[\"var_Y\"]:.4f}  S_x={out[\"S_first\"][(0,)]:.3f}')

with open('lc_anova/results/per_alpha_sobol_schr1d_seed0.json', 'w') as f:
    json.dump(results, f, indent=2)
" > "$LOG_DIR/per_alpha_schr1d.log" 2>&1

# === Higher-N MC-Sobol on 2D Helm seed 0 for tighter gold standard ===
echo "[$(date)] high-N MC-Sobol 2D Helm seed0 (N=100000)" | tee -a "$MASTER"
python -m lc_anova.pipelines.mc_sobol_helmholtz_2d \
    --checkpoint "$CK_DIR/lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt" \
    --N 100000 --tag "mc_sobol_highn_seed0" \
    > "$LOG_DIR/mc_sobol_highn.log" 2>&1

echo "=== phase4 batch DONE $(date) ===" | tee -a "$MASTER"
