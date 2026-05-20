#!/bin/bash
# Wait for the 2D Helmholtz FiLM+L-BFGS sweep to finish, then run the same
# treatment on 1D Helmholtz, Burgers, and BL. All outputs tagged film_lbfgs
# so the existing baseline JSONs are not overwritten.

set -u
cd "$(dirname "$0")/.."
LOG=logs/overnight_chain.log
mkdir -p logs

echo "=== overnight chain: started $(date) ===" > "$LOG"

WAIT_PID=70693
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done
echo "=== PID $WAIT_PID finished at $(date); starting chain ===" >> "$LOG"

# 1. 1D Helmholtz FiLM + L-BFGS, matched arch (width 64).
echo "--- RUN 1: 1D Helmholtz FiLM+L-BFGS w64 — start $(date) ---" >> "$LOG"
python -u scripts/lc_pinn_helmholtz.py \
  --seeds 0 1 2 3 --n-epochs 50000 --n-lbfgs 1500 \
  --conditioning film --hidden-width 64 --hidden-depth 4 \
  --tag film_lbfgs >> "$LOG" 2>&1
echo "--- RUN 1 done $(date) (exit $?) ---" >> "$LOG"

# 2. Burgers FiLM + L-BFGS, matched arch.
echo "--- RUN 2: Burgers FiLM+L-BFGS w64 — start $(date) ---" >> "$LOG"
python -u scripts/lc_pinn_burgers_seeds.py \
  --seeds 0 1 2 3 --n-epochs 50000 --n-lbfgs 1500 \
  --conditioning film --hidden-width 64 --hidden-depth 4 \
  --tag film_lbfgs >> "$LOG" 2>&1
echo "--- RUN 2 done $(date) (exit $?) ---" >> "$LOG"

# 3. BL FiLM + L-BFGS, matched arch.
echo "--- RUN 3: BL FiLM+L-BFGS w64 — start $(date) ---" >> "$LOG"
python -u scripts/lc_pinn_bl_seeds.py \
  --seeds 0 1 2 3 --n-epochs 50000 --n-lbfgs 1500 \
  --conditioning film --hidden-width 64 --hidden-depth 4 \
  --tag film_lbfgs >> "$LOG" 2>&1
echo "--- RUN 3 done $(date) (exit $?) ---" >> "$LOG"

echo "=== overnight chain: ALL DONE $(date) ===" >> "$LOG"
