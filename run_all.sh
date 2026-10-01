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

# Reproduces the paper's tables. One flag per table: each runs the steps its table needs, in
# order, and steps shared between requested tables run once. The script calls `python`, so run
# it inside the project environment:
#   uv run ./run_all.sh --table1 --table3     (or activate .venv first)
#   ./run_all.sh --help                       (every flag, outputs and skip rules)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

SIM_SEEDS=0,1,2,3,4
SIM_SYSTEMS="double_integrator unicycle kinematic_bicycle dynamic_bicycle"
FAB_SEEDS="0 1 2 3 4"
MADE_SEEDS="0 1 2"
PRED_FAMILIES="lstm ssm transformer"
PRED_SEEDS="0 1 2 3 4"
IND_DATA=data/inD-preprocessed/v1
IND_RAW=data/inD-raw
SMOOTHER_NOISE=outputs/ind/smoother_noise.json

# Run order, independent of flag order: simulated tables first, latency (Table 6) last.
TARGETS="table1 table3 table5 table2 table9_10 table11 table8 corrector_iterations table6"
IND_TARGETS="table2 table9_10 table11 table8 corrector_iterations table6"

DRY_RUN=0
FORCE=0
PREDICTORS_TRAINED=0

usage() {
  cat <<'EOF'
Usage: ./run_all.sh [--dry-run] [--force] TARGET...

Reproduce the paper's tables. Run inside the project environment (the script calls `python`):
  uv run ./run_all.sh --table1        or activate .venv first

Targets (combine freely; each runs its prerequisites, shared steps run once):
  --table1                 Table 1, simulated results              -> outputs/table_1/
  --table2                 Table 2, inD results                    -> outputs/table_2/
  --table3                 Table 3, ablations                      -> outputs/table_3/
  --table5                 Table 5, control recovery               -> outputs/table_5/
  --table6                 Table 6, runtime (idle NVIDIA GPU)      -> outputs/table_6/
  --table8                 Table 8, gradient depth                 -> outputs/table_8/
  --table9, --table10      Tables 9 and 10, inequality breakdown   -> outputs/table_9_10/
  --table11                Table 11, completion-only variant       -> outputs/table_11/
  --corrector-iterations   Corrector iteration counts (Appendix)   -> outputs/appendix/
  --all                    Every target above; Table 6 runs last

Options:
  --dry-run    Print the commands in order without running them.
  --force      Disable this script's skips (finished predictors, tuned smoother noise).
               Tuned noise is otherwise reused only when it is complete and newer than
               every finished predictor.
               Scripts that resume or skip on their own (run_matrix.py, train_fab.py,
               train_made.py) are unaffected: delete a run directory to retrain it.
  -h, --help   Show this help.

inD targets read data/inD-preprocessed/v1 (done once its manifest.json exists). If it is
missing and data/inD-raw/ exists, the raw recordings are preprocessed first; otherwise the
script stops (see README.md, "Reproducing the paper"). A v1/ without manifest.json is an
interrupted preprocessing run: the script stops and says how to redo it.
Training runs, scores and evaluation panels stay under data/, outputs/sim/ and outputs/ind/. Run Table 6 alone on an idle GPU: it measures wall-clock time.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

# Print a command, then run it unless --dry-run.
run() {
  printf '+ %s\n' "$*"
  if (( DRY_RUN == 0 )); then
    "$@"
  fi
}

# Create a final output directory (real runs only).
outdir() {
  if (( DRY_RUN == 0 )); then
    mkdir -p "$1"
  fi
}

# once STAGE: succeed the first time STAGE is seen in this invocation, fail afterwards.
once() {
  local var="DONE_$1"
  if [[ -n "${!var:-}" ]]; then
    return 1
  fi
  printf -v "$var" 1
}

want() { printf -v "WANT_$1" 1; }
wanted() {
  local var="WANT_$1"
  [[ -n "${!var:-}" ]]
}

# ---- simulated-system stages ----

stage_sim_data() {
  once sim_data || return 0
  echo "=== generate simulated data ==="
  run python scripts/sim/generate_data.py --system all --seed 0
}

stage_run_matrix() {
  once run_matrix || return 0
  stage_sim_data
  echo "=== train the MaDE / ablation / MLP / clamp matrix (all systems, seeds 0-4) ==="
  run python scripts/sim/run_matrix.py --seeds "$SIM_SEEDS"
}

stage_train_fab() {
  once train_fab || return 0
  stage_sim_data
  echo "=== train FAB (one process per system x seed cell; finished cells skip themselves) ==="
  local system seed
  for system in $SIM_SYSTEMS; do
    for seed in $FAB_SEEDS; do
      run python scripts/sim/train_fab.py --system "$system" --seed "$seed"
    done
  done
}

stage_score_all() {
  once score_all || return 0
  stage_run_matrix
  stage_train_fab
  echo "=== score every trained cell (the table input, not the training-time metrics) ==="
  run python scripts/sim/score_all.py --seeds "$SIM_SEEDS"
}

# One build_tables.py call writes Table 1, Table 3 and the seed-level JSON.
stage_sim_tables() {
  once sim_tables || return 0
  stage_score_all
  echo "=== build Tables 1 and 3 ==="
  outdir outputs/table_1
  outdir outputs/table_3
  run python scripts/sim/build_tables.py --results-root outputs/sim/scores \
    --out-seed-level outputs/table_1/tab_e01_seed_level.json \
    --out-tex outputs/table_1/tab_e01_results.tex \
    --out-json outputs/table_1/tab_e01_results.audit.json \
    --out-ablation-tex outputs/table_3/tab_e01_ablations.tex \
    --out-ablation-json outputs/table_3/tab_e01_ablations.audit.json
}

# ---- inD stages ----

stage_ind_data() {
  once ind_data || return 0
  if [[ -f "$IND_DATA/manifest.json" ]]; then
    echo "skip: $IND_DATA exists (to redo it, run scripts/ind/preprocess.py with --force)"
    return 0
  fi
  echo "=== preprocess the raw inD recordings ==="
  run python scripts/ind/preprocess.py --raw-dir "$IND_RAW" --output-dir data/inD-preprocessed
}

# stage_made SEED: train (or resume) one MaDE seed on inD.
stage_made() {
  local seed=$1
  once "made_seed$seed" || return 0
  stage_ind_data
  echo "=== train MaDE on inD, seed $seed (resumes from its checkpoint) ==="
  run python scripts/ind/train_made.py --config configs/ind/made.json \
    --data-dir "$IND_DATA" --output-dir "outputs/ind/made/seed$seed" \
    --stationary-filter-m 0.5 --seed "$seed"
}

stage_made_all() {
  local seed
  for seed in $MADE_SEEDS; do
    stage_made "$seed"
  done
}

stage_predictors() {
  once predictors || return 0
  stage_ind_data
  echo "=== train the 15 predictors (lstm, ssm, transformer x seeds 0-4) ==="
  local family seed out
  for family in $PRED_FAMILIES; do
    for seed in $PRED_SEEDS; do
      out="outputs/ind/predictors/${family}_stage1_seed${seed}"
      if (( FORCE == 0 )) && [[ -f "$out/training_summary.json" ]]; then
        echo "skip: $out/training_summary.json exists"
        continue
      fi
      PREDICTORS_TRAINED=1
      run python scripts/ind/train_predictor.py --data-dir "$IND_DATA" \
        --output-dir "$out" \
        --predictor "$family" --seed "$seed" --epochs 200 --batch-size 32 --lr 0.001 \
        --history 10 --horizon 15 --stride 5 --val-chunk-size 2048 --stationary-filter-m 0.5
    done
  done
}

# True when the smoother-noise file covers every family with no grid-edge minimum.
smoother_noise_complete() {
  [[ -f "$SMOOTHER_NOISE" ]] || return 1
  local family seed summary
  for family in $PRED_FAMILIES; do
    for seed in $PRED_SEEDS; do
      summary="outputs/ind/predictors/${family}_stage1_seed${seed}/training_summary.json"
      if [[ -f "$summary" && ! "$SMOOTHER_NOISE" -nt "$summary" ]]; then
        return 1
      fi
    done
  done
  python -c 'import json, sys
r = json.load(open(sys.argv[1]))["records"]
ok = {x["family"] for x in r} >= {"lstm", "ssm", "transformer"}
sys.exit(0 if ok and not any(x["minimum_at_grid_edge"] for x in r) else 1)' \
    "$SMOOTHER_NOISE" 2>/dev/null
}

stage_tune_smoother() {
  once tune_smoother || return 0
  stage_predictors
  if (( FORCE == 0 && PREDICTORS_TRAINED == 0 )) && smoother_noise_complete; then
    echo "skip: $SMOOTHER_NOISE is complete and newer than every predictor"
    return 0
  fi
  echo "=== tune the smoother baseline's noise covariance on the train split ==="
  run python scripts/ind/tune_smoother.py
}

stage_evaluate() {
  once evaluate || return 0
  stage_made_all
  stage_predictors
  stage_tune_smoother
  echo "=== evaluate the filtered panel (raw / clamp / smoother / MaDE) ==="
  run python scripts/ind/evaluate.py --smoother-noise "$SMOOTHER_NOISE"
}

stage_completion_eval() {
  once completion_eval || return 0
  stage_made_all
  stage_predictors
  echo "=== evaluate the completion-only attribution arm ==="
  run python scripts/ind/evaluate_completion_only.py
}

# ---- targets (each called at most once) ----

target_table1() { stage_sim_tables; }
target_table3() { stage_sim_tables; }

target_table5() {
  once control_recovery || return 0
  stage_run_matrix
  echo "=== control recovery (Table 5) ==="
  outdir outputs/table_5
  run python scripts/sim/control_recovery.py --seeds "$SIM_SEEDS" \
    --out-json outputs/table_5/control_recovery.json \
    --out-tex outputs/table_5/tab_control_recovery.tex
}

target_table2() {
  stage_evaluate
  echo "=== build Table 2 ==="
  outdir outputs/table_2
  run python scripts/ind/build_table_main.py --source outputs/ind/panel.json \
    --out-dir outputs/table_2
}

target_table9_10() {
  stage_evaluate
  echo "=== build Tables 9 and 10 ==="
  outdir outputs/table_9_10
  run python scripts/ind/build_table_breakdown.py --source outputs/ind/panel.json \
    --out-dir outputs/table_9_10
}

target_table11() {
  stage_completion_eval
  echo "=== build Table 11 ==="
  outdir outputs/table_11
  run python scripts/ind/build_table_completion_only.py \
    --source outputs/ind/completion_only.json --out-dir outputs/table_11
}

target_table8() {
  stage_made 0
  echo "=== probe gradient norms through the recursive corrector (Table 8) ==="
  outdir outputs/table_8
  run python scripts/ind/gradient_depth_probe.py --config outputs/ind/made/seed0/config.json \
    --made-checkpoint outputs/ind/made/seed0/checkpoints --data-dir "$IND_DATA" \
    --output outputs/table_8/grad_norm_probe.json
}

target_corrector_iterations() {
  stage_made_all
  stage_predictors
  echo "=== corrector-iteration counts (Appendix) ==="
  outdir outputs/appendix
  run python scripts/ind/corrector_iterations.py \
    --out outputs/appendix/corrector_iterations.json
}

target_table6() {
  stage_made_all
  stage_predictors
  stage_tune_smoother
  echo "=== latency (Table 6): run alone on an idle GPU ==="
  outdir outputs/table_6
  run python scripts/ind/measure_latency.py --idle-devices 0 \
    --smoother-noise "$SMOOTHER_NOISE" --out outputs/table_6/latency.json
}

# ---- main ----

if (( $# == 0 )); then
  usage >&2
  exit 2
fi
for arg in "$@"; do
  case "$arg" in
    --table1) want table1 ;;
    --table2) want table2 ;;
    --table3) want table3 ;;
    --table5) want table5 ;;
    --table6) want table6 ;;
    --table8) want table8 ;;
    --table9 | --table10) want table9_10 ;;
    --table11) want table11 ;;
    --corrector-iterations) want corrector_iterations ;;
    --all) for t in $TARGETS; do want "$t"; done ;;
    --dry-run) DRY_RUN=1 ;;
    --force) FORCE=1 ;;
    -h | --help) usage; exit 0 ;;
    *) echo "error: unknown argument: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

any=0
need_ind=0
for t in $TARGETS; do
  if wanted "$t"; then any=1; fi
done
for t in $IND_TARGETS; do
  if wanted "$t"; then need_ind=1; fi
done
if (( any == 0 )); then
  echo "error: no target given" >&2
  usage >&2
  exit 2
fi
if (( DRY_RUN == 0 )) && ! command -v python >/dev/null 2>&1; then
  die "python not found: run 'uv run ./run_all.sh ...' or activate .venv first"
fi
if (( need_ind == 1 )) && [[ ! -d "$IND_DATA" && ! -d "$IND_RAW" ]]; then
  die "inD targets need $IND_DATA, or the raw recordings in $IND_RAW/ to preprocess." \
    "inD is not redistributed: see README.md, \"Reproducing the paper\"."
fi
if (( need_ind == 1 )) && [[ -d "$IND_DATA" && ! -f "$IND_DATA/manifest.json" ]]; then
  die "$IND_DATA exists but has no manifest.json (interrupted preprocessing?)." \
    "Rerun: uv run python scripts/ind/preprocess.py --raw-dir $IND_RAW --output-dir data/inD-preprocessed --force"
fi
if (( DRY_RUN == 1 )); then
  echo "dry run: commands are printed, not run"
fi

for t in $TARGETS; do
  if wanted "$t"; then
    "target_$t"
  fi
done
echo "=== done ==="
