# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""End-to-end CPU smoke test for scripts/ind/eval_lib.py (--use-stub, tiny dims).

Trains a tiny Stage-1 LSTM predictor via a subprocess CLI call to
scripts/ind/train_predictor.py (--use-stub, tiny dims), then drives
scripts/ind/eval_lib.py in-process (importlib + monkeypatched sys.argv) with
--smoke-random-made and the raw/clamp/made_pnp rows requested.

Asserts: all requested rows present, every metric value finite, and (schema
check, not a numeric one) every row exposes the identical metric-key set.
made_pnp inequality magnitude <= raw is intentionally NOT asserted — the
frozen MaDE here is an untrained random cell (--smoke-random-made), so there
is no reason to expect it improves feasibility.
"""

from __future__ import annotations

from types import ModuleType
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "ind" / "train_predictor.py"
_EVAL_SCRIPT = _REPO_ROOT / "scripts" / "ind" / "eval_lib.py"

_TINY_LSTM_ARGS = ["--hidden-size", "8", "--decoder-width", "8", "--decoder-depth", "1"]
_TINY_WINDOW_ARGS = [
    "--history", "4",
    "--horizon", "3",
    "--stride", "2",
    "--epochs", "1",
    "--batch-size", "8",
    "--stub-num-trajectories", "16",
    "--stub-trajectory-length", "12",
]
_TINY_STUB_EVAL_ARGS = ["--stub-num-trajectories", "16", "--stub-trajectory-length", "12"]

_EXPECTED_ROWS = {"raw", "clamp", "made_pnp"}
_BASE_METRIC_KEYS = {
    "ade",
    "fde",
    "dynamics_violation",
    "gt_normalised_dynamics_residual",
    "inequality_violation_rate_physical",
    "inequality_violation_magnitude_physical",
    "inequality_violation_rate_envelope",
    "inequality_violation_magnitude_envelope",
    "latency_batch1_median_s",
    "latency_batched_amortised_s",
}
# made_pnp is scored on its own emitted controls (u_for_ineq is not None), which additionally
# yields the state/control inequality breakdown, plus a KB-controls side metric for
# apples-to-apples comparison against raw/clamp (see scripts/ind/eval_lib.py module
# docstring). raw/clamp are scored with u_for_ineq=None (the state-only branch of
# compute_inequality_dual), which does not emit the breakdown keys.
_MADE_ROWS = {"made_pnp"}
_MADE_METRIC_KEYS = _BASE_METRIC_KEYS | {
    "gt_normalised_dynamics_residual_kb_controls",
    "inequality_violation_rate_physical_state",
    "inequality_violation_rate_physical_control",
    "inequality_violation_magnitude_physical_state",
    "inequality_violation_magnitude_physical_control",
}


def _env() -> dict:
    """Return the process environment pinned to the CPU JAX backend.

    Returns:
        Environment mapping for subprocess calls.
    """
    return {**os.environ, "JAX_PLATFORMS": "cpu"}


def _load_eval_module() -> ModuleType:
    """Import the evaluation script as a module by file path.

    Returns:
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location("eval_lib", str(_EVAL_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_lib"] = module
    spec.loader.exec_module(module)
    return module


def test_chunked_predictor_forward_matches_unchunked_on_nonmultiple_chunk_size() -> None:
    """Unit-style check of ``_chunked_predictor_forward`` in isolation (no subprocess
    training needed): a non-multiple chunk size (7 windows, chunk_size=4 -> chunks of
    4 then 3, via one padded-to-4 call each) must produce output numerically identical
    to a single unchunked ``jax.vmap(predictor)`` call. Guards the pad/slice-off logic
    that bounds peak memory for large predictors (e.g. the SSM, which OOM'd the GPU on
    the full ~54k-window inD test split under one giant vmap'd call) against silently
    corrupting output at chunk boundaries.
    """
    module = _load_eval_module()
    jax = module.jax
    jnp = module.jnp

    def _toy_predictor(context: jax.Array) -> jax.Array:
        """Map a context window to a fixed-shape toy prediction.

        Args:
            context: Context window array.

        Returns:
            Prediction array.
        """
        return jnp.tanh(context * 2.0 + 1.0)[..., :3]

    context_all = jax.random.normal(jax.random.key(0), (7, 4, 5), dtype=jnp.float64)

    unchunked = jax.vmap(_toy_predictor)(context_all)
    chunked = module._chunked_predictor_forward(_toy_predictor, context_all, chunk_size=4)

    assert chunked.shape == unchunked.shape
    assert bool(jnp.allclose(chunked, unchunked, rtol=0, atol=0))


def _run_train(args: list[str]) -> subprocess.CompletedProcess:
    """Run the predictor training script in a subprocess.

    Args:
        args: Command-line arguments passed to the script.

    Returns:
        Completed process with captured output.
    """
    return subprocess.run(
        [sys.executable, str(_TRAIN_SCRIPT), *args],
        cwd=_REPO_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def predictor_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Provide a directory holding a tiny trained LSTM predictor."""
    out_dir = tmp_path_factory.mktemp("eval_lib_predictor")
    proc = _run_train(
        [
            "--use-stub", "--output-dir", str(out_dir),
            "--predictor", "lstm", "--seed", "0",
            *_TINY_WINDOW_ARGS, *_TINY_LSTM_ARGS,
        ]
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return out_dir


def test_eval_lib_all_rows_finite_and_schema_consistent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, predictor_dir: Path
) -> None:
    """Verify eval lib all rows finite and schema consistent."""
    module = _load_eval_module()
    output_path = tmp_path / "results.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_lib.py",
            "--use-stub",
            "--predictor-dir", str(predictor_dir),
            "--rows", "raw,clamp,made_pnp",
            "--smoke-random-made",
            "--output", str(output_path),
            *_TINY_STUB_EVAL_ARGS,
        ],
    )
    module.main()

    assert output_path.exists()
    payload = json.loads(output_path.read_text())

    assert payload["metric_version"] == "e05-v3"
    assert set(payload["rows"]) == _EXPECTED_ROWS

    for row_name, metrics in payload["rows"].items():
        expected_keys = _MADE_METRIC_KEYS if row_name in _MADE_ROWS else _BASE_METRIC_KEYS
        assert set(metrics) == expected_keys, (
            f"row {row_name!r} metric keys {set(metrics)} != {expected_keys}"
        )
        for metric_key, value in metrics.items():
            assert isinstance(value, (int, float)), f"{row_name}.{metric_key} is {type(value)}"
            assert math.isfinite(value), f"{row_name}.{metric_key} = {value} is not finite"
        # dynamics_violation is an unnormalised residual (paper-facing number,
        # scored uniformly against KB-recovered controls for every row) — it
        # must be finite and non-negative for every row. The identity
        # dynamics_violation == (gt_normalised_dynamics_residual + 1) *
        # gt_residual is not directly assertable here (gt_residual is not in
        # the JSON payload), so this is the strongest schema-level check
        # available at this layer.
        assert metrics["dynamics_violation"] >= 0.0, (
            f"row {row_name!r} dynamics_violation = {metrics['dynamics_violation']} < 0"
        )

    provenance = payload["provenance"]
    assert provenance["window_spec"] == {"history": 4, "horizon": 3, "stride": 2, "dt": 0.2}
    assert provenance["smoke_random_made"] is True
    assert provenance["use_stub"] is True
    assert provenance["num_test_windows"] > 0
    assert provenance["predictor_kind"] == "lstm"
    assert isinstance(provenance["git_sha"], str) and provenance["git_sha"]
    assert isinstance(provenance["hardware"], str) and provenance["hardware"]


def test_eval_lib_raw_clamp_only_needs_no_made_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, predictor_dir: Path
) -> None:
    """Rows that don't need MaDE must run without --made-checkpoint or --smoke-random-made."""
    module = _load_eval_module()
    output_path = tmp_path / "results.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_lib.py",
            "--use-stub",
            "--predictor-dir", str(predictor_dir),
            "--rows", "raw,clamp",
            "--output", str(output_path),
            *_TINY_STUB_EVAL_ARGS,
        ],
    )
    module.main()

    payload = json.loads(output_path.read_text())
    assert set(payload["rows"]) == {"raw", "clamp"}
    assert payload["provenance"]["made_checkpoint"] is None
    assert payload["provenance"]["smoke_random_made"] is False


def test_validate_args_rejects_made_pnp_without_checkpoint_or_smoke_flag() -> None:
    """Verify validate args rejects made pnp without checkpoint or smoke flag."""
    module = _load_eval_module()
    args = module._build_parser().parse_args(
        ["--predictor-dir", "x", "--output", "y", "--rows", "made_pnp"]
    )
    with pytest.raises(ValueError, match="--made-checkpoint"):
        module._validate_args(args)


def test_validate_args_rejects_unknown_row() -> None:
    """Verify validate args rejects unknown row."""
    module = _load_eval_module()
    args = module._build_parser().parse_args(
        ["--predictor-dir", "x", "--output", "y", "--rows", "bogus"]
    )
    with pytest.raises(ValueError, match="Unknown row"):
        module._validate_args(args)
