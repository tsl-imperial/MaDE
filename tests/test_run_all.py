# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Routing tests for the root run_all.sh, through its --dry-run listing.

Each test copies the script into a temporary directory (it cds to its own directory), so the
on-disk state that drives preprocessing and skips is controlled per test.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "run_all.sh"

GEN = "python scripts/sim/generate_data.py --system all --seed 0"
MATRIX = "python scripts/sim/run_matrix.py --seeds 0,1,2,3,4"
SCORE = "python scripts/sim/score_all.py --seeds 0,1,2,3,4"
CONTROL = (
    "python scripts/sim/control_recovery.py --seeds 0,1,2,3,4"
    " --out-json outputs/table_5/control_recovery.json"
    " --out-tex outputs/table_5/tab_control_recovery.tex"
)
BUILD_SIM = (
    "python scripts/sim/build_tables.py --results-root outputs/sim/scores"
    " --out-seed-level outputs/table_1/tab_e01_seed_level.json"
    " --out-tex outputs/table_1/tab_e01_results.tex"
    " --out-json outputs/table_1/tab_e01_results.audit.json"
    " --out-ablation-tex outputs/table_3/tab_e01_ablations.tex"
    " --out-ablation-json outputs/table_3/tab_e01_ablations.audit.json"
)
PREPROCESS = (
    "python scripts/ind/preprocess.py --raw-dir data/inD-raw --output-dir data/inD-preprocessed"
)
MADE0 = (
    "python scripts/ind/train_made.py --config configs/ind/made.json"
    " --data-dir data/inD-preprocessed/v1 --output-dir outputs/ind/made/seed0"
    " --stationary-filter-m 0.5 --seed 0"
)
PREDICTOR_LSTM0 = (
    "python scripts/ind/train_predictor.py --data-dir data/inD-preprocessed/v1"
    " --output-dir outputs/ind/predictors/lstm_stage1_seed0 --predictor lstm --seed 0"
    " --epochs 200 --batch-size 32 --lr 0.001 --history 10 --horizon 15 --stride 5"
    " --val-chunk-size 2048 --stationary-filter-m 0.5"
)
PROBE = (
    "python scripts/ind/gradient_depth_probe.py --config outputs/ind/made/seed0/config.json"
    " --made-checkpoint outputs/ind/made/seed0/checkpoints"
    " --data-dir data/inD-preprocessed/v1 --output outputs/table_8/grad_norm_probe.json"
)
LATENCY = (
    "python scripts/ind/measure_latency.py --idle-devices 0"
    " --smoother-noise outputs/ind/smoother_noise.json --out outputs/table_6/latency.json"
)
FAMILIES = ("lstm", "ssm", "transformer")


def _sandbox(tmp_path: Path, *, raw: bool, preprocessed: bool = False) -> Path:
    """Copy run_all.sh into tmp_path and create the requested inD data dirs.

    Args:
        tmp_path: Empty directory that becomes the script's working directory.
        raw: Whether to create data/inD-raw.
        preprocessed: Whether to create data/inD-preprocessed/v1 with a manifest.json.

    Returns:
        Path of the copied script.
    """
    script = tmp_path / "run_all.sh"
    shutil.copy2(_SCRIPT, script)
    if raw:
        (tmp_path / "data" / "inD-raw").mkdir(parents=True)
    if preprocessed:
        v1 = tmp_path / "data" / "inD-preprocessed" / "v1"
        v1.mkdir(parents=True)
        (v1 / "manifest.json").write_text("{}")
    return script


def _invoke(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the script with bash, with this interpreter's directory first on PATH.

    Args:
        script: Path of the script to run.
        *args: Command-line arguments passed to the script.

    Returns:
        The completed process with captured text output.
    """
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"}
    return subprocess.run(
        ["bash", str(script), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=60,
    )


def _dry_run(script: Path, *flags: str) -> list[str]:
    """Return the commands a --dry-run would execute, in order.

    Args:
        script: Path of the script to run.
        *flags: Target and option flags passed after --dry-run.

    Returns:
        The echoed commands, without their "+ " prefix.
    """
    proc = _invoke(script, "--dry-run", *flags)
    assert proc.returncode == 0, proc.stderr
    return [line[2:] for line in proc.stdout.splitlines() if line.startswith("+ ")]


def _count(cmds: list[str], needle: str) -> int:
    """Count the commands containing needle.

    Args:
        cmds: Commands as returned by _dry_run.
        needle: Substring to look for.

    Returns:
        Number of commands containing needle.
    """
    return sum(needle in c for c in cmds)


def _mark_predictors_done(root: Path) -> None:
    """Create a training_summary.json for all 15 predictor cells under root."""
    for family in FAMILIES:
        for seed in range(5):
            _summary(root, family, seed).parent.mkdir(parents=True, exist_ok=True)
            _summary(root, family, seed).write_text("{}")


def _summary(root: Path, family: str, seed: int) -> Path:
    """Return the training_summary.json path of one predictor cell under root."""
    return root / "outputs/ind/predictors" / f"{family}_stage1_seed{seed}" / "training_summary.json"


def test_table5_needs_only_data_and_matrix(tmp_path: Path) -> None:
    """--table5 runs data, matrix and control recovery, with no inD data present."""
    s = _sandbox(tmp_path, raw=False)
    assert _dry_run(s, "--table5") == [GEN, MATRIX, CONTROL]


def test_table1_and_table3_share_one_build(tmp_path: Path) -> None:
    """--table1 with --table3 runs the shared chain and one build_tables call."""
    s = _sandbox(tmp_path, raw=False)
    cmds = _dry_run(s, "--table3", "--table1")
    assert len(cmds) == 24
    assert cmds[0] == GEN
    assert cmds[1] == MATRIX
    assert _count(cmds, "scripts/sim/train_fab.py") == 20
    assert cmds.count(SCORE) == 1
    assert cmds.count(BUILD_SIM) == 1
    assert cmds[-1] == BUILD_SIM
    assert _count(cmds, "control_recovery") == 0


def test_table3_alone_runs_fab_and_scoring(tmp_path: Path) -> None:
    """--table3 alone still trains FAB and scores before building."""
    s = _sandbox(tmp_path, raw=False)
    cmds = _dry_run(s, "--table3")
    assert _count(cmds, "scripts/sim/train_fab.py") == 20
    assert SCORE in cmds
    assert cmds[-1] == BUILD_SIM


def test_all_runs_every_stage_once(tmp_path: Path) -> None:
    """--all runs each shared stage once, with preprocessing first and latency last."""
    s = _sandbox(tmp_path, raw=True)
    cmds = _dry_run(s, "--all")
    assert len(cmds) == 53
    assert len(set(cmds)) == len(cmds)
    for cmd in (GEN, MATRIX, SCORE, CONTROL, BUILD_SIM, PREPROCESS, MADE0, PREDICTOR_LSTM0, PROBE, LATENCY):
        assert cmds.count(cmd) == 1
    assert _count(cmds, "scripts/sim/train_fab.py") == 20
    assert _count(cmds, "scripts/ind/train_made.py") == 3
    assert _count(cmds, "scripts/ind/train_predictor.py") == 15
    for needle in (
        "scripts/ind/tune_smoother.py",
        "scripts/ind/evaluate.py",
        "scripts/ind/evaluate_completion_only.py",
        "scripts/ind/build_table_main.py",
        "scripts/ind/build_table_breakdown.py",
        "scripts/ind/build_table_completion_only.py",
        "scripts/ind/corrector_iterations.py",
    ):
        assert _count(cmds, needle) == 1
    assert cmds.index(PREPROCESS) < cmds.index(MADE0)
    assert cmds[-1] == LATENCY


def test_table8_trains_only_made_seed0(tmp_path: Path) -> None:
    """--table8 preprocesses, trains MaDE seed 0 and runs the probe, nothing else."""
    s = _sandbox(tmp_path, raw=True)
    assert _dry_run(s, "--table8") == [PREPROCESS, MADE0, PROBE]


def test_table9_and_table10_are_one_target(tmp_path: Path) -> None:
    """--table9 and --table10 together build the breakdown once."""
    s = _sandbox(tmp_path, raw=False, preprocessed=True)
    cmds = _dry_run(s, "--table9", "--table10")
    assert _count(cmds, "build_table_breakdown.py") == 1
    assert cmds[-1].endswith("--out-dir outputs/table_9_10")
    assert _count(cmds, "preprocess.py") == 0


def test_table11_skips_smoother_and_panel(tmp_path: Path) -> None:
    """--table11 needs neither tune_smoother nor the filtered-panel evaluation."""
    s = _sandbox(tmp_path, raw=False, preprocessed=True)
    cmds = _dry_run(s, "--table11")
    assert len(cmds) == 20
    assert _count(cmds, "tune_smoother.py") == 0
    assert _count(cmds, "scripts/ind/evaluate.py") == 0
    assert _count(cmds, "evaluate_completion_only.py") == 1
    assert cmds[-1] == (
        "python scripts/ind/build_table_completion_only.py"
        " --source outputs/ind/completion_only.json --out-dir outputs/table_11"
    )


def test_missing_ind_data_stops_before_running(tmp_path: Path) -> None:
    """An inD target without inD data exits 1 with a README pointer and runs nothing."""
    s = _sandbox(tmp_path, raw=False)
    proc = _invoke(s, "--dry-run", "--table1", "--table2")
    assert proc.returncode == 1
    assert "README.md" in proc.stderr
    assert not any(line.startswith("+ ") for line in proc.stdout.splitlines())


def test_finished_predictor_skipped_unless_forced(tmp_path: Path) -> None:
    """A finished predictor cell is skipped, and --force retrains it."""
    s = _sandbox(tmp_path, raw=False, preprocessed=True)
    summary = _summary(tmp_path, "lstm", 0)
    summary.parent.mkdir(parents=True)
    summary.write_text("{}")
    cmds = _dry_run(s, "--table11")
    assert _count(cmds, "train_predictor.py") == 14
    assert PREDICTOR_LSTM0 not in cmds
    assert _count(_dry_run(s, "--table11", "--force"), "train_predictor.py") == 15


def test_tuned_smoother_skipped_only_when_predictors_are_done(tmp_path: Path) -> None:
    """tune_smoother is skipped only for complete noise, finished predictors, no edge minimum."""
    s = _sandbox(tmp_path, raw=False, preprocessed=True)
    _mark_predictors_done(tmp_path)
    noise = tmp_path / "outputs/ind/smoother_noise.json"
    records = [{"family": f, "minimum_at_grid_edge": False} for f in FAMILIES]
    noise.write_text(json.dumps({"records": records}))
    older = noise.stat().st_mtime - 100
    for family in FAMILIES:
        for seed in range(5):
            os.utime(_summary(tmp_path, family, seed), (older, older))

    cmds = _dry_run(s, "--table2")
    assert _count(cmds, "train_predictor.py") == 0
    assert _count(cmds, "tune_smoother.py") == 0
    assert _count(cmds, "scripts/ind/evaluate.py") == 1

    last = _summary(tmp_path, "transformer", 4)
    last.unlink()
    cmds = _dry_run(s, "--table2")
    assert _count(cmds, "train_predictor.py") == 1
    assert _count(cmds, "tune_smoother.py") == 1

    last.write_text("{}")
    edge = [{"family": f, "minimum_at_grid_edge": f == "ssm"} for f in FAMILIES]
    noise.write_text(json.dumps({"records": edge}))
    assert _count(_dry_run(s, "--table2"), "tune_smoother.py") == 1

    noise.write_text(json.dumps({"records": records}))
    newer = noise.stat().st_mtime + 100
    os.utime(_summary(tmp_path, "lstm", 0), (newer, newer))
    cmds = _dry_run(s, "--table2")
    assert _count(cmds, "tune_smoother.py") == 1
    assert _count(cmds, "train_predictor.py") == 0


def test_partial_preprocessed_dir_stops_before_running(tmp_path: Path) -> None:
    """A v1 directory without manifest.json exits 1 with the --force hint and runs nothing."""
    s = _sandbox(tmp_path, raw=False, preprocessed=True)
    (tmp_path / "data/inD-preprocessed/v1/manifest.json").unlink()
    proc = _invoke(s, "--dry-run", "--table8")
    assert proc.returncode == 1
    assert "manifest.json" in proc.stderr
    assert "--force" in proc.stderr
    assert not any(line.startswith("+ ") for line in proc.stdout.splitlines())


def test_help_and_bad_arguments(tmp_path: Path) -> None:
    """--help exits 0 and lists targets, while no arguments or an unknown one exit 2."""
    s = _sandbox(tmp_path, raw=False)
    helped = _invoke(s, "--help")
    assert helped.returncode == 0
    assert "--table9" in helped.stdout
    assert "--corrector-iterations" in helped.stdout
    assert _invoke(s).returncode == 2
    assert _invoke(s, "--bogus").returncode == 2
