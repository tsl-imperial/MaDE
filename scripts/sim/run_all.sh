#!/usr/bin/env bash
# Reproduces every E01 (simulated) table in the paper, end to end, on CPU.
# Run from the repository root:
#   cd MaDE-release && bash scripts/sim/run_all.sh
set -euo pipefail

echo "=== [1/6] generate simulated data ==="
python scripts/sim/generate_data.py --system all --seed 0

echo "=== [2/6] train the MaDE / ablation / MLP / clamp matrix (all systems, seeds 0-4) ==="
python scripts/sim/run_matrix.py --seeds 0,1,2,3,4

echo "=== [3/6] train FAB (one process per system x seed cell, as the paper's runs) ==="
for system in double_integrator unicycle kinematic_bicycle dynamic_bicycle; do
  for seed in 0 1 2 3 4; do
    echo "--- FAB: system=$system seed=$seed ---"
    python scripts/sim/train_fab.py --system "$system" --seed "$seed"
  done
done

echo "=== [4/6] score every trained cell (this is the table input, not the training-time metrics) ==="
python scripts/sim/score_all.py --seeds 0,1,2,3,4

echo "=== [5/6] control recovery (Table 5) ==="
python scripts/sim/control_recovery.py --seeds 0,1,2,3,4

echo "=== [6/6] build the E01 tables ==="
python scripts/sim/build_tables.py

echo "=== done: see outputs/tables/tab_e01_results.tex, outputs/tables/tab_e01_ablations.tex, and outputs/tables/tab_control_recovery.tex ==="
