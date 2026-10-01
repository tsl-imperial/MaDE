#!/usr/bin/env bash
# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

# Reproduces every inD (real-data) table in the paper, end to end.
# Run from the repository root:
#   uv run bash scripts/ind/run_all.sh
#
# Requires a GPU for training in reasonable time and the raw inD recordings unpacked to
# data/inD-raw/ (the inD dataset is not redistributed here -- see README.md). Step 0 is left
# commented out because it needs the user's own raw-data path.
set -euo pipefail

# echo "=== [0/8] preprocess the raw inD recordings ==="
# python scripts/ind/preprocess.py --raw-dir data/inD-raw --output-dir data/inD-preprocessed
# (writes under data/inD-preprocessed/v1/, since --version defaults to v1)

echo "=== [1/8] train MaDE (seeds 0, 1, 2) ==="
for seed in 0 1 2; do
  echo "--- MaDE: seed=$seed ---"
  python scripts/ind/train_made.py --config configs/ind/made.json \
    --data-dir data/inD-preprocessed/v1 --output-dir outputs/ind/made/seed"$seed" \
    --stationary-filter-m 0.5 --seed "$seed"
done

echo "=== [2/8] train the 15 predictors (lstm, ssm, transformer x seeds 0-4) ==="
for family in lstm ssm transformer; do
  for seed in 0 1 2 3 4; do
    echo "--- predictor: family=$family seed=$seed ---"
    python scripts/ind/train_predictor.py --data-dir data/inD-preprocessed/v1 \
      --output-dir outputs/ind/predictors/"$family"_stage1_seed"$seed" \
      --predictor "$family" --seed "$seed" --epochs 200 --batch-size 32 --lr 0.001 \
      --history 10 --horizon 15 --stride 5 --val-chunk-size 2048 --stationary-filter-m 0.5
  done
done

echo "=== [3/8] tune the smoother baseline's noise covariance on the train split ==="
python scripts/ind/tune_smoother.py

echo "=== [4/8] evaluate the filtered panel (raw / clamp / smoother / MaDE) ==="
python scripts/ind/evaluate.py --smoother-noise outputs/ind/smoother_noise.json

echo "=== [5/8] evaluate the completion-only attribution arm ==="
python scripts/ind/evaluate_completion_only.py

echo "=== [6/8] build the inD tables ==="
python scripts/ind/build_table_main.py
python scripts/ind/build_table_breakdown.py
python scripts/ind/build_table_completion_only.py

echo "=== [7/8] probe gradient norms through the recursive corrector ==="
python scripts/ind/gradient_depth_probe.py --config outputs/ind/made/seed0/config.json \
  --made-checkpoint outputs/ind/made/seed0/checkpoints --data-dir data/inD-preprocessed/v1 \
  --output outputs/ind/grad_norm_probe.json

echo "=== [8/8] corrector-iteration counts, then latency (run alone on an idle GPU) ==="
python scripts/ind/corrector_iterations.py
python scripts/ind/measure_latency.py --idle-devices 0 \
  --smoother-noise outputs/ind/smoother_noise.json

echo "=== done: see outputs/tables/tab_e05_ind.tex, tab_ineq_breakdown_ind_rate.tex, tab_ineq_breakdown_ind_magnitude.tex, tab_completion_only.tex ==="
