"""CPU smoke tests for scripts/ind/train_predictor.py (--use-stub, tiny dims).

Exercised via subprocess (all three predictor kinds) to validate the full CLI path.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "ind" / "train_predictor.py"

_TINY_LSTM_ARGS = ["--hidden-size", "8", "--decoder-width", "8", "--decoder-depth", "1"]
_TINY_SSM_ARGS = [
    "--d-model", "8",
    "--num-blocks", "1",
    "--d-state", "4",
    "--decoder-width", "8",
    "--decoder-depth", "1",
]
_TINY_TRANSFORMER_ARGS = [
    "--d-model", "8",
    "--num-layers", "1",
    "--num-heads", "2",
    "--ff-width", "8",
    "--decoder-width", "8",
    "--decoder-depth", "1",
]
_TINY_ARGS_BY_PREDICTOR: dict[str, list[str]] = {
    "lstm": _TINY_LSTM_ARGS,
    "ssm": _TINY_SSM_ARGS,
    "transformer": _TINY_TRANSFORMER_ARGS,
}
_TINY_WINDOW_ARGS = [
    "--history", "4",
    "--horizon", "3",
    "--stride", "2",
    "--epochs", "2",
    "--batch-size", "8",
    "--stub-num-trajectories", "16",
    "--stub-trajectory-length", "12",
]


def _env() -> dict:
    return {**os.environ, "JAX_PLATFORMS": "cpu"}


def _run(out_dir: Path, predictor: str, seed: int = 0) -> subprocess.CompletedProcess:
    tiny = _TINY_ARGS_BY_PREDICTOR[predictor]
    return subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--use-stub",
            "--output-dir",
            str(out_dir),
            "--predictor",
            predictor,
            "--seed",
            str(seed),
            *_TINY_WINDOW_ARGS,
            *tiny,
        ],
        cwd=_REPO_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def test_smoke_writes_expected_files_lstm(tmp_path) -> None:
    out_dir = tmp_path / "lstm"
    proc = _run(out_dir, "lstm")
    assert proc.returncode == 0, proc.stderr[-4000:]
    _assert_outputs(out_dir, "lstm")


def test_smoke_writes_expected_files_ssm(tmp_path) -> None:
    out_dir = tmp_path / "ssm"
    proc = _run(out_dir, "ssm")
    assert proc.returncode == 0, proc.stderr[-4000:]
    _assert_outputs(out_dir, "ssm")


def test_smoke_writes_expected_files_transformer(tmp_path) -> None:
    out_dir = tmp_path / "transformer"
    proc = _run(out_dir, "transformer")
    assert proc.returncode == 0, proc.stderr[-4000:]
    _assert_outputs(out_dir, "transformer")


def _assert_outputs(out_dir: Path, predictor: str) -> None:
    assert (out_dir / "predictor.eqx").exists()
    assert (out_dir / "predictor_config.json").exists()
    assert (out_dir / "training_summary.json").exists()
    assert (out_dir / "config.json").exists()

    summary = json.loads((out_dir / "training_summary.json").read_text())
    assert summary["predictor"] == predictor
    assert math.isfinite(summary["best_val_mse"])
    assert summary["num_train_windows"] > 0
    assert summary["num_val_windows"] > 0
    assert summary["val_chunk_size"] == 2048  # CLI default, unset in _TINY_WINDOW_ARGS
    assert len(summary["history_epochs"]) == 2

    predictor_config = json.loads((out_dir / "predictor_config.json").read_text())
    assert predictor_config["kind"] == predictor
    assert predictor_config["horizon"] == 3
    assert predictor_config["history"] == 4
    assert predictor_config["stride"] == 2
    assert predictor_config["dt"] == 0.2
