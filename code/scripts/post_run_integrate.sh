#!/bin/bash
# After a follow-on training step lands, integrate it into the paper:
# 1. Print a markdown harvest summary
# 2. Substitute placeholders in results.tex from the new JSONs
# 3. Recompile the PDF
# Usage:  bash scripts/post_run_integrate.sh

set -u
cd "$(dirname "$0")/.."

echo "=== harvest summary ==="
python scripts/harvest_film_lbfgs.py
echo
echo "=== auto-fill placeholders ==="
python scripts/update_paper_from_results.py
echo
echo "=== recompile paper ==="
cd paper
tectonic main.tex 2>&1 | tail -3
cd ..

echo
echo "=== remaining placeholders (if any) ==="
grep -oE "<[a-z\-]+>" paper/sections/results.tex | sort -u || true

echo
echo "=== git status (paper only) ==="
git status --short paper/sections/
