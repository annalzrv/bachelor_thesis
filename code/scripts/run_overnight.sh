#!/bin/zsh
# Overnight comparison kit. Runs all baselines vs LC-PINN on Burgers and BL,
# plus a single 200k-step LC-PINN long reference and a Causal-PINN baseline.
# Identical backbone everywhere: hidden_dims = [64,64,64,64].
#
# Total wall-time estimate on M4 Max MPS: ~9-10 hours.
# Run from thesis/code/ as:    bash scripts/run_overnight.sh
# Live output goes to results/overnight_*.log so you can check progress.
#
# Kill: Ctrl-C in the foreground terminal, or `pkill -f run_overnight.sh`
#       (also kills the python child via `set -e` propagation).

set -e
cd "$(dirname "$0")/.."
mkdir -p results checkpoints

LOG_DIR=results
DATE=$(date +%Y%m%d_%H%M)

echo "=========================================="
echo "Overnight run started: $(date)"
echo "Logs: $LOG_DIR/overnight_${DATE}_*.log"
echo "=========================================="

# --- Burgers baselines ------------------------------------------------------

echo
echo "[1/7] SA-PINN on Burgers: 4 seeds × (10k Adam + 5k L-BFGS)"
python -u scripts/sa_pinn_burgers.py \
    --seeds 0 1 2 3 \
    --n-adam 10000 \
    --n-lbfgs 5000 \
    --lr-theta 1e-3 \
    --lr-lambda 5e-3 \
    2>&1 | tee "$LOG_DIR/overnight_${DATE}_sa_pinn_burgers.log"

echo
echo "[2/7] ReLoBRaLo on Burgers: 4 seeds × 50k Adam"
python -u scripts/relobralo_burgers.py \
    --seeds 0 1 2 3 \
    --n-epochs 50000 \
    --lr 1e-3 \
    --alpha 0.999 --tau 0.1 --rho-mean 0.999 \
    2>&1 | tee "$LOG_DIR/overnight_${DATE}_relobralo_burgers.log"

echo
echo "[3/7] Causal-PINN on Burgers: 4 seeds × 50k Adam, M=32 bins, eps=100"
python -u scripts/causal_pinn_burgers.py \
    --seeds 0 1 2 3 \
    --n-epochs 50000 \
    --lr 1e-3 \
    --M 32 --eps 100 \
    2>&1 | tee "$LOG_DIR/overnight_${DATE}_causal_pinn_burgers.log"

echo
echo "[4/7] LC-PINN Burgers reference (4 seeds × 50k, K_eval=200)"
python -u scripts/lc_pinn_burgers_seeds.py \
    --seeds 0 1 2 3 \
    --n-epochs 50000 \
    --lr 1e-3 \
    --n-lambda-samples 4 \
    --K-eval 200 \
    --tag seeds \
    2>&1 | tee "$LOG_DIR/overnight_${DATE}_lc_pinn_burgers.log"

echo
echo "[5/7] LC-PINN Burgers long run (1 seed × 200k, K_eval=200)"
python -u scripts/lc_pinn_burgers_seeds.py \
    --seeds 0 \
    --n-epochs 200000 \
    --lr 1e-3 \
    --n-lambda-samples 4 \
    --K-eval 200 \
    --tag long \
    2>&1 | tee "$LOG_DIR/overnight_${DATE}_lc_pinn_burgers_long.log"

# --- Buckley-Leverett baselines --------------------------------------------

echo
echo "[6/7] BL baselines: SA-PINN + ReLoBRaLo + LC-PINN (4 seeds each)"

python -u scripts/sa_pinn_bl.py \
    --seeds 0 1 2 3 \
    --n-adam 10000 \
    --n-lbfgs 5000 \
    --lr-theta 1e-3 \
    --lr-lambda 5e-3 \
    2>&1 | tee "$LOG_DIR/overnight_${DATE}_sa_pinn_bl.log"

python -u scripts/relobralo_bl.py \
    --seeds 0 1 2 3 \
    --n-epochs 50000 \
    --lr 1e-3 \
    --alpha 0.999 --tau 0.1 --rho-mean 0.999 \
    2>&1 | tee "$LOG_DIR/overnight_${DATE}_relobralo_bl.log"

echo
echo "[7/7] LC-PINN on Buckley-Leverett (4 seeds × 50k, K_eval=200)"
python -u scripts/lc_pinn_bl_seeds.py \
    --seeds 0 1 2 3 \
    --n-epochs 50000 \
    --lr 1e-3 \
    --n-lambda-samples 4 \
    --K-eval 200 \
    2>&1 | tee "$LOG_DIR/overnight_${DATE}_lc_pinn_bl.log"

echo
echo "=========================================="
echo "Overnight run complete: $(date)"
echo "Open notebooks/09_baseline_comparison.ipynb to see results."
echo "=========================================="
