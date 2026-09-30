"""inD evaluation library: real-predictor integration on inD (no synthetic perturbation).

Loads a trained upstream predictor and evaluates it on the inD test split
against the predictor's own errors — there is no synthetic perturbation
anywhere in this path; the object of study is the predictor's own
autoregressive drift, and how raw / clamp / frozen-MaDE correction handle it.

Window spec (history/horizon/stride/dt) is read from the predictor's own
``predictor_config.json`` (written by ``made.upstream.factory.save_predictor``),
never re-specified on the CLI, so eval windows always match how the predictor
was trained.

Rows (subset of ``raw,clamp,made_pnp``, default all):
  raw       — predictor output as-is.
  clamp     — ``ClampBaseline`` with ``drop_position_bounds(kinematic_bicycle_constraints())``
              applied per predicted state.
  made_pnp  — frozen trained MaDE applied plug-and-play (``correction_mode="eval_adaptive"``),
              seeded at the last observed context state.

Metrics per row (aggregated over all test windows): ``ade``, ``fde``,
``dynamics_violation`` (unnormalised mean known-physics one-step residual,
computed uniformly from KB-recovered pseudo-controls for every row — this is
the number reported in the paper), ``gt_normalised_dynamics_residual``,
the 4-key ``compute_inequality_dual`` output, and JIT-warmed wall-clock
latency (batch-1 median + batched-amortised per-trajectory), plus a hardware
string. The ``made_pnp`` row additionally carries
``gt_normalised_dynamics_residual_kb_controls``: a side metric that scores
that row the same way raw/clamp are scored (controls recovered by the
KB-exact inverse from the row's own trajectory), because the primary
``gt_normalised_dynamics_residual`` for the made row uses MaDE's own emitted
controls, which is not apples-to-apples against raw/clamp's
KB-inverse-recovered controls. ``dynamics_violation`` sidesteps that
apples-to-apples problem entirely by always scoring against KB-recovered
controls, uniformly across all rows.

This module is loaded by file path (not imported as a package) by the other
inD evaluation scripts (``evaluate.py``, ``evaluate_completion_only.py``,
``measure_latency.py``, ``corrector_iterations.py``).

Engineering rules:
  - jax_enable_x64 enabled before any JAX import.
  - eqx.filter_jit for JIT-compiled functions; vmap for per-sample -> batch.
  - Inequality constraints on the inD path: inD_physical_constraints() only.
  - MaDE is always frozen; never differentiated here (eval-only, no grads).
  - Orbax/checkpoint paths absolutised at the CLI boundary.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TextIO

import jax

from made.utils.jax_setup import configure

configure()

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from made.baselines.clamp_baseline import ClampBaseline  # noqa: E402
from made.data.ind_data import create_ind_data_source  # noqa: E402
from made.data.window_dataset import assemble_context, make_prediction_windows  # noqa: E402
from made.evaluation.metrics import (  # noqa: E402
    EmpiricalEnvelope,
    ade,
    compute_inequality_dual,
    dynamics_violation_known,
    fde,
    gt_normalised_dynamics_residual,
    kinematic_bicycle_inverse_controls,
    position_ade,
    position_fde,
)
from made.evaluation.real_data_eval import L_REF, load_train_envelope_and_residual  # noqa: E402
from made.physics import (  # noqa: E402
    KinematicBicycle,
    _assert_xy_unbounded,
    drop_position_bounds,
    inD_physical_constraints,
    kinematic_bicycle_constraints,
)
from made.upstream import apply_made_trajectory_with_controls  # noqa: E402
from made.upstream.factory import load_predictor  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Bumping this string requires regenerating every result JSON.
METRIC_VERSION: str = "e05-v3"

_ALL_ROWS: tuple[str, ...] = ("raw", "clamp", "smoother", "smoother_dyn", "made_pnp")

# The smoother row uses the KINEMATIC BICYCLE on BOTH panels -- the same known model
# MaDE is given, and L_REF is the same reference wheelbase the metric stencil already
# scores every row against.
_SMOOTHER_PHYSICS = KinematicBicycle
_SMOOTHER_L_REF = L_REF
_SMOOTHER_STATE_DIM = 4
_ROWS_REQUIRING_MADE: frozenset[str] = frozenset({"made_pnp"})


# ---------------------------------------------------------------------------
# CLI parsing / validation
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(_REPO_ROOT / "data" / "inD-preprocessed" / "v1"))
    parser.add_argument(
        "--predictor-dir",
        required=True,
        help="Predictor output directory (contains predictor_config.json).",
    )
    parser.add_argument(
        "--made-checkpoint",
        default=None,
        help="Directory containing a frozen trained MaDE checkpoint. Required when "
        "'made_pnp' in --rows, unless --smoke-random-made is passed.",
    )
    parser.add_argument(
        "--smoother-noise",
        default=None,
        help="Path to the TRAIN-SPLIT tuning artifact written by "
        "scripts/ind/tune_smoother.py. Required iff a smoother row is in --rows. "
        "The evaluation applies these covariances and tunes nothing. 'smoother' uses the "
        "ADE-optimal operating point and 'smoother_dyn' the known-model-residual-optimal one; "
        "the two sit four to seven and a half decades apart in q_scale.",
    )
    parser.add_argument("--output", required=True, help="Output path for the results JSON.")
    parser.add_argument(
        "--rows",
        default="raw,clamp,made_pnp",
        help="Comma-separated subset of {raw,clamp,made_pnp} (default: all).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use-stub",
        action="store_true",
        help="Use made.data.ind_data.create_ind_data_source stub data (CPU smoke).",
    )
    parser.add_argument(
        "--smoke-random-made",
        action="store_true",
        help="TEST-ONLY escape hatch: use an untrained random frozen MaDEModel for "
        "made_pnp instead of --made-checkpoint. Never use for real results.",
    )
    parser.add_argument("--stub-num-trajectories", type=int, default=32)
    parser.add_argument("--stub-trajectory-length", type=int, default=40)
    parser.add_argument(
        "--timing-batch-size",
        type=int,
        default=16,
        help="Batch size for the batched-amortised latency measurement.",
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=5,
        help="Number of repeats for the batch-1 latency median.",
    )
    parser.add_argument(
        "--pred-chunk-size",
        type=int,
        default=4096,
        help="Run the upstream predictor's forward pass over the test windows in fixed-size "
        "chunks of this many windows instead of one giant vmap'd call. A single call over "
        "the full inD test split (~54k windows) can OOM the GPU for predictors with large "
        "scan intermediates (e.g. the SSM). Chunking bounds peak memory to O(chunk_size) "
        "and produces numerically identical output. Does not affect the downstream row "
        "computations (clamp/made/metrics/latency), which already handle the full window "
        "count fine.",
    )
    return parser


def _parse_rows(rows_arg: str) -> list[str]:
    rows = [r.strip() for r in rows_arg.split(",") if r.strip()]
    unknown = [r for r in rows if r not in _ALL_ROWS]
    if unknown:
        raise ValueError(f"Unknown row(s) {unknown} in --rows; must be a subset of {_ALL_ROWS}")
    return rows


def _validate_args(args: argparse.Namespace) -> list[str]:
    """CLI-boundary validation, fired before any data loading / JAX work."""
    rows = _parse_rows(args.rows)
    needs_made = any(r in _ROWS_REQUIRING_MADE for r in rows)
    if needs_made and args.made_checkpoint is None and not args.smoke_random_made:
        raise ValueError(
            "--rows includes made_pnp, which requires --made-checkpoint pointing "
            "at a frozen trained MaDE checkpoint directory. Use --smoke-random-made "
            "only for test-only smoke runs with an untrained random MaDE."
        )
    return rows


def _progress_output() -> tuple[TextIO, bool, TextIO | None]:
    """Mirrors made.training.trainer._progress_output exactly.

    Renders fine under the queue's ``script -e -q -f -c`` pty wrapping (a pty
    reports ``isatty() == True``); falls back to ``/dev/tty`` only when stderr
    itself is redirected (e.g. piped to a file with no controlling terminal).
    """
    if sys.stderr.isatty():
        return sys.stderr, False, None
    try:
        tty = Path("/dev/tty").open("w", encoding="utf-8")
    except OSError:
        return sys.stderr, True, None
    return tty, False, tty


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _hardware_string() -> str:
    devices = ", ".join(str(d) for d in jax.devices())
    return (
        f"{platform.system()} {platform.release()} | {platform.machine()} | "
        f"jax_devices=[{devices}]"
    )


# ---------------------------------------------------------------------------
# Frozen MaDE loading (mirrors scripts/evaluate_e02.py exactly).
# ---------------------------------------------------------------------------


def _load_frozen_made(
    made_checkpoint: str | None,
    smoke_random_made: bool,
    seed: int,
    *,
    select: str = "best",
):
    """Restore a frozen trained MaDE checkpoint for evaluation.

    Stale deserialised constraints are a known trap: the checkpoint's
    constraints must be overwritten with a fresh ``inD_physical_constraints()``
    after restore, not trusted as-is.

    Returns a ``MaDEModel`` (cell + metadata encoder) so per-vehicle physics
    params can be resolved from the window's own metadata.
    """
    if made_checkpoint is not None:
        from made.models.made_model import MaDEModel

        checkpoint_dir = str(Path(made_checkpoint).expanduser().resolve())
        print(f"[eval_lib] loading MaDE checkpoint: {checkpoint_dir}", file=sys.stderr)
        model = MaDEModel.from_checkpoint(checkpoint_dir, select=select)
        new_constraints = inD_physical_constraints()
        _assert_xy_unbounded(new_constraints)
        new_cell = eqx.tree_at(lambda c: c.constraints, model.cell, new_constraints)
        model = eqx.tree_at(lambda m: m.cell, model, new_cell)
        if model.encoder is None:
            raise ValueError(
                f"MaDE checkpoint at {checkpoint_dir!r} has no metadata encoder; "
                "eval_lib.py requires per-vehicle params resolution via the encoder path."
            )
        return model

    if smoke_random_made:
        print(
            "[eval_lib] --smoke-random-made: using an UNTRAINED random frozen "
            "MaDEModel (with metadata encoder). TEST-ONLY — never use for real results.",
            file=sys.stderr,
        )
        from made.data.ind_data import IND_LOCATION_ID_INDEX, IND_METADATA_DIM, IND_NUM_LOCATIONS
        from made.models import MaDEModel
        from made.utils import CorrectorConfig, ModelConfig

        physics = KinematicBicycle()
        constraints = inD_physical_constraints()
        model_config = ModelConfig(
            use_metadata_encoder=True,
            metadata_dim=IND_METADATA_DIM,
            num_locations=IND_NUM_LOCATIONS,
            embedding_dim=8,
            location_id_index=IND_LOCATION_ID_INDEX,
        )
        # mode="disabled" forces MaDECell.__call__ to take the cheap "it_only"
        # path regardless of the requested correction_mode. TEST-SPEED-ONLY —
        # real evaluation runs use a real --made-checkpoint and the corrector
        # stays enabled (the checkpoint's own trained corrector_mode).
        corrector_config = CorrectorConfig(mode="disabled")
        return MaDEModel.from_config(
            physics,
            constraints,
            model_config,
            corrector_config,
            param_scales=physics.param_scales,
            key=jax.random.key(seed),
        )

    raise ValueError(
        "made_pnp requires --made-checkpoint pointing at a frozen trained MaDE "
        "checkpoint directory. Use --smoke-random-made only for test-only smoke runs with "
        "an untrained random MaDE."
    )


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# X1 -- the EKF/RTS smoother rows
# ---------------------------------------------------------------------------

# Which tuned operating point each row uses. The two sit four to seven and a half decades apart
# in q_scale, and Dyn.-K differs between them by a factor of thirty to a hundred and ninety, so
# they are genuinely different arms rather than a sensitivity check.
_SMOOTHER_CRITERION = {"smoother": "ade", "smoother_dyn": "dynk"}


def _load_smoother(path, family: str, panel: str, dt: float, criterion: str):
    """Build one smoother arm from the TRAIN-SPLIT tuning artifact. Nothing is tuned here.

    The evaluation reads the selected covariances and applies them. A missing file, a missing
    record for this (panel, family), or a record whose minimum landed on a grid edge raises
    rather than falling back to a default -- an untuned smoother reported as a tuned baseline is
    worse than no baseline at all.
    """
    import json as _json

    from made.baselines.ekf_rts_smoother import KinodynamicSmoother

    if not path:
        raise ValueError(
            "--rows includes a smoother row but --smoother-noise was not given. The "
            "covariances are tuned on the training split by "
            "scripts/ind/tune_smoother.py and are not defaulted here."
        )
    doc = _json.loads(Path(path).read_text())
    match = [r for r in doc["records"] if r["panel"] == panel and r["family"] == family]
    if not match:
        raise ValueError(f"{path} carries no tuning record for panel={panel} family={family}")
    rec = match[0]
    if rec.get("minimum_at_grid_edge"):
        raise ValueError(
            f"{panel}/{family}: the tuned q_scale {rec['selected_q_scale']} sits on a GRID "
            "EDGE, so the grid did not bracket the optimum. Widen the grid and re-tune; do not "
            "evaluate against an unbracketed value."
        )
    if criterion == "ade":
        q = float(rec["selected_q_scale"])
    elif criterion == "dynk":
        raw_q = rec.get("CRITERION_SENSITIVITY", {}).get("q_scale_by_known_model_residual")
        if raw_q is None:
            raise ValueError(
                f"{path} carries no known-model-residual optimum for {panel}/{family}. It was "
                "written by a tuning pass that did not record the Dyn.-K curve; re-run "
                "scripts/ind/tune_smoother.py."
            )
        q = float(raw_q)
    else:
        raise ValueError(f"unknown smoother criterion {criterion!r}; expected 'ade' or 'dynk'")

    smoother = KinodynamicSmoother(
        physics=_SMOOTHER_PHYSICS(),
        params=jnp.asarray([_SMOOTHER_L_REF], dtype=jnp.float64),
        dt=dt,
        q_diag=jnp.asarray(
            [q * v for v in rec["q_state_base"]] + [q * v for v in rec["q_ctrl_base"]],
            dtype=jnp.float64),
        r_diag=jnp.asarray(rec["r_base"], dtype=jnp.float64),
        p0_diag=jnp.asarray(list(rec["r_base"]) + list(rec["p0_control_block"]),
                            dtype=jnp.float64),
    )
    return smoother, {"criterion": criterion, "q_scale": q, "tuning_record": rec}


def _load_smoother_arms(path, family: str, panel: str, dt: float, rows) -> dict:
    """Build only the arms actually requested, keyed by row name."""
    return {
        r: _load_smoother(path, family, panel, dt, _SMOOTHER_CRITERION[r])
        for r in _SMOOTHER_CRITERION
        if r in rows
    }


def _smoother_trajectory(smoother, x0: jax.Array, x_pred: jax.Array):
    """Smooth one window and return the FUTURE block and ITS CONTROLS, `([F, D], [F, C])`.

    The anchor `x0` is prepended before smoothing, so the smoother sees the same feasible seed
    state every other correction row is given. The caller's `_full_sequence` re-prepends the
    TRUE anchor afterwards -- the anchor is an observation, not something a baseline may move.

    **The controls are returned, not discarded.** The smoother carries the controls
    in its augmented state and its RTS pass produces a smoothed estimate of them, so it IS a row
    that emits controls and is scored on its own controls rather than on pseudo-controls
    recovered by the panel's inverse.
    """
    meas = jnp.concatenate([x0[None, :], x_pred], axis=0)
    x_s, u_s = smoother.smooth_trajectory(meas)
    return x_s[1:], u_s[1:]


def _smoother_rows(smoother, x0_all: jax.Array, x_pred_all: jax.Array):
    """`[K, F, D] -> ([K, F, D], [K, F, C])`, vmapped over the K windows."""
    return jax.vmap(_smoother_trajectory, in_axes=(None, 0, 0))(smoother, x0_all, x_pred_all)


def _load_test_windows(
    args: argparse.Namespace, window_spec: dict[str, Any]
) -> dict[str, jax.Array]:
    states, metadata, lengths = create_ind_data_source(
        args.data_dir,
        "test",
        use_stub=args.use_stub,
        stub_num_trajectories=args.stub_num_trajectories,
        stub_trajectory_length=args.stub_trajectory_length,
        stub_seed=args.seed,
    )
    windows = make_prediction_windows(
        states,
        lengths,
        metadata,
        history=int(window_spec["history"]),
        horizon=int(window_spec["horizon"]),
        stride=int(window_spec["stride"]),
    )
    if windows["context"].shape[0] == 0:
        raise ValueError(
            "make_prediction_windows produced zero test windows for "
            f"history={window_spec['history']} horizon={window_spec['horizon']} "
            f"stride={window_spec['stride']}. Check --data-dir / --stub-trajectory-length."
        )
    return windows


def _assemble_batch_context(context_states: jax.Array, metadata: jax.Array) -> jax.Array:
    return jax.vmap(assemble_context)(context_states, metadata)


def _predictor_forward_chunk(predictor, context_chunk: jax.Array) -> jax.Array:
    return jax.vmap(predictor)(context_chunk)


def _chunked_predictor_forward(
    predictor,
    context_all: jax.Array,
    chunk_size: int,
) -> jax.Array:
    """Run ``jax.vmap(predictor)`` over ``context_all`` in fixed-size chunks.

    A single ``jax.vmap(predictor)(context_all)`` call materialises every window's
    forward-pass intermediates simultaneously. For the SSM predictor over the full inD
    test split (54,465 windows) this OOMs the GPU (measured: an f64[10,54465,128,16]
    scan intermediate is ~8.31 GiB on its own). Chunking bounds peak memory to
    O(chunk_size) instead of O(num_windows). This is a pure element-wise map (no
    reduction), so chunked and unchunked output are numerically identical — unlike
    ``train_upstream_inD.py``'s ``_chunked_weighted_mean``, no sum/count combination is
    needed here.

    The last chunk is zero-padded up to a full chunk of ``chunk_size`` (so every
    ``eqx.filter_jit``'d call traces the same shape once, rather than recompiling per
    chunk) and the padding is sliced off the concatenated result before returning.
    """
    n = int(context_all.shape[0])
    if n == 0:
        return jax.vmap(predictor)(context_all)
    eff_chunk = min(chunk_size, n)
    num_chunks = -(-n // eff_chunk)  # ceil division
    pad_amount = num_chunks * eff_chunk - n

    if pad_amount > 0:
        pad_width = [(0, pad_amount)] + [(0, 0)] * (context_all.ndim - 1)
        padded = jnp.pad(context_all, pad_width)
    else:
        padded = context_all

    chunk_fn = eqx.filter_jit(_predictor_forward_chunk)
    outputs = [
        chunk_fn(predictor, padded[i * eff_chunk : (i + 1) * eff_chunk]) for i in range(num_chunks)
    ]
    return jnp.concatenate(outputs, axis=0)[:n]


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def _clamp_state(clamp_model: ClampBaseline, state: jax.Array) -> jax.Array:
    """Apply ClampBaseline elementwise to a single predicted state.

    correct_pair(x_prev, x_curr) clips both endpoints independently (no
    dynamics dependency between them), so passing the same state twice and
    keeping the x_curr half reuses the class's exact clip logic per state.
    """
    _, clamped = clamp_model.correct_pair(state, state)
    return clamped


def _clamp_trajectory(clamp_model: ClampBaseline, traj: jax.Array) -> jax.Array:
    """``[F, D] -> [F, D]``, vmap over the trajectory's F states."""
    return jax.vmap(_clamp_state, in_axes=(None, 0))(clamp_model, traj)


def _clamp_rows(clamp_model: ClampBaseline, x_pred_all: jax.Array) -> jax.Array:
    """``[K, F, D] -> [K, F, D]``, vmap over the K windows."""
    return jax.vmap(_clamp_trajectory, in_axes=(None, 0))(clamp_model, x_pred_all)


def _made_pnp_single(
    made_cell,
    x_pred: jax.Array,
    params: jax.Array,
    x0: jax.Array,
    dt: float,
) -> tuple[jax.Array, jax.Array]:
    """Single-window frozen-MaDE plug-and-play application."""
    return apply_made_trajectory_with_controls(
        made_cell, x_pred, params, dt, correction_mode="eval_adaptive", x0=x0
    )


def _made_pnp_rows(
    made_model,
    x_pred_all: jax.Array,
    metadata: jax.Array,
    x0_all: jax.Array,
    dt: float,
) -> tuple[jax.Array, jax.Array]:
    """``[K, F, D] -> ([K, F, D], [K, F, U])`` — corrected states and controls."""
    params_all = made_model.params_from_metadata(metadata)  # [K, param_dim]
    x_corr_all, u_corr_all = jax.vmap(
        lambda x_pred, params, x0: _made_pnp_single(made_model.cell, x_pred, params, x0, dt)
    )(x_pred_all, params_all, x0_all)
    return x_corr_all, u_corr_all


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _full_sequence(x0_all: jax.Array, x_row: jax.Array) -> jax.Array:
    """Prepend the feasible seed state: ``[K, F, D] -> [K, F+1, D]``.

    ``dynamics_violation_known`` is a *transition* residual: it needs T states
    paired with T-1 controls (one control explaining each x[t] -> x[t+1] step).
    The row's own F predicted/corrected states have no predecessor of their
    own, so x0 (the last *observed* context state) supplies the T-th state
    and the first of the F transitions.
    """
    return jnp.concatenate([x0_all[:, None, :], x_row], axis=1)


def _derive_dynamics_controls(x_full: jax.Array, dt: float) -> jax.Array:
    """Recover a pseudo-control trajectory via the KB-exact inverse over ``x_full``.

    Used only for rows that do not produce their own controls (raw, clamp);
    ``gt_normalised_dynamics_residual`` has no ``u=None`` mode, so a control
    signal consistent with the row's own state trajectory is required.
    """
    controls, _aux = kinematic_bicycle_inverse_controls(x_full, dt, wheelbase=L_REF)
    return controls  # [K, F, 2]


def _row_metrics(
    x_row: jax.Array,
    x_gt: jax.Array,
    u_for_ineq: jax.Array | None,
    x_full: jax.Array,
    u_for_dyn: jax.Array,
    envelope: EmpiricalEnvelope,
    physical_constraints,
    gt_residual: float,
    dt: float,
    u_dyn_kb: jax.Array | None = None,
    *,
    position_displacement: bool = False,
) -> dict[str, float]:
    """Compute the primary per-row metric dict.

    ``u_dyn_kb``, when given (made_pnp row only), additionally scores
    ``gt_normalised_dynamics_residual`` against KB-inverse-recovered controls
    (rather than the row's own ``u_for_dyn``), under the key
    ``gt_normalised_dynamics_residual_kb_controls`` — the like-for-like
    comparison against raw/clamp, whose primary metric already uses
    KB-inverse-recovered controls.

    ``dynamics_violation`` is the paper-facing unnormalised counterpart:
    the raw (non-normalised) known-physics one-step residual, scored against
    KB-recovered controls uniformly for every row. For raw/clamp, ``u_for_dyn``
    already *is* the KB-recovered control trajectory, so it doubles as the
    source here; for made rows ``u_for_dyn`` is MaDE's own emitted controls,
    so ``u_dyn_kb`` (when given) is used instead.

    ``position_displacement`` selects the ADE/FDE definition. False (the
    default) is the ``e05-v3`` full-state definition that ``METRIC_VERSION``
    locks — it must not change. True is the position-only (x, y) inD
    definition, used by ``evaluate.py`` and ``evaluate_completion_only.py``.
    """
    kb_physics = KinematicBicycle()
    kb_params = jnp.asarray([L_REF], dtype=jnp.float64)
    u_kb_for_dynamics_violation = u_dyn_kb if u_dyn_kb is not None else u_for_dyn
    ade_fn, fde_fn = (position_ade, position_fde) if position_displacement else (ade, fde)
    metrics: dict[str, float] = {
        "ade": float(ade_fn(x_row, x_gt)),
        "fde": float(fde_fn(x_row, x_gt)),
        "dynamics_violation": float(
            dynamics_violation_known(
                x_full, u_kb_for_dynamics_violation, kb_physics, kb_params, dt
            )
        ),
        "gt_normalised_dynamics_residual": float(
            gt_normalised_dynamics_residual(
                x_full, u_for_dyn, kb_physics, kb_params, dt, gt_residual
            )
        ),
    }
    if u_dyn_kb is not None:
        metrics["gt_normalised_dynamics_residual_kb_controls"] = float(
            gt_normalised_dynamics_residual(
                x_full, u_dyn_kb, kb_physics, kb_params, dt, gt_residual
            )
        )
    metrics.update(compute_inequality_dual(x_row, u_for_ineq, envelope, physical_constraints))
    return metrics


# ---------------------------------------------------------------------------
# Latency harness
# ---------------------------------------------------------------------------


def _time_pipeline(
    fn,
    single_args: tuple[jax.Array, ...],
    batched_args: tuple[jax.Array, ...],
    *,
    repeats: int,
) -> dict[str, float]:
    """JIT-warmed batch-1 median and batched-amortised per-trajectory wall time.

    ``fn`` is a single-sample (unbatched) pure function. Timed via
    ``time.perf_counter`` with ``jax.block_until_ready`` around every call so
    async dispatch does not hide the real compute cost.
    """
    single_jit = eqx.filter_jit(fn)
    warm = single_jit(*single_args)
    jax.block_until_ready(warm)

    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = single_jit(*single_args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    batch1_median = float(np.median(times))

    batched_jit = eqx.filter_jit(jax.vmap(fn))
    warm_b = batched_jit(*batched_args)
    jax.block_until_ready(warm_b)
    batch_n = int(batched_args[0].shape[0])
    t0 = time.perf_counter()
    out_b = batched_jit(*batched_args)
    jax.block_until_ready(out_b)
    batched_total = time.perf_counter() - t0
    batched_amortised = batched_total / max(batch_n, 1)

    return {
        "latency_batch1_median_s": batch1_median,
        "latency_batched_amortised_s": batched_amortised,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _build_parser().parse_args()
    rows = _validate_args(args)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.use_stub:
        args.data_dir = str(Path(args.data_dir).expanduser().resolve())
    predictor_dir = str(Path(args.predictor_dir).expanduser().resolve())
    made_checkpoint = (
        str(Path(args.made_checkpoint).expanduser().resolve()) if args.made_checkpoint else None
    )

    print(f"[eval_lib] rows={rows} predictor_dir={predictor_dir}", file=sys.stderr)

    predictor, predictor_config = load_predictor(predictor_dir)
    dt = float(predictor_config["dt"])
    window_spec = {k: predictor_config[k] for k in ("history", "horizon", "stride")}
    window_spec["dt"] = dt

    windows = _load_test_windows(args, window_spec)
    num_windows = int(windows["context"].shape[0])
    print(f"[eval_lib] {num_windows} test windows (spec={window_spec})", file=sys.stderr)

    context_all = _assemble_batch_context(windows["context"], windows["metadata"])
    x_gt_all = windows["future"]
    x0_all = windows["context"][:, -1, :]
    metadata_all = windows["metadata"]

    print(
        "[eval_lib] estimating envelope + GT reference residual from train split",
        file=sys.stderr,
    )
    envelope, gt_residual, _eval_meta = load_train_envelope_and_residual(
        args.data_dir,
        use_stub=args.use_stub,
        dt=dt,
        smoke_seed=args.seed,
        stub_num_trajectories=args.stub_num_trajectories,
        stub_trajectory_length=args.stub_trajectory_length,
    )
    physical_constraints = inD_physical_constraints()

    smoother_arms = _load_smoother_arms(
        args.smoother_noise, predictor_config["kind"], "ind", dt, rows)

    made_model = None
    if any(r in _ROWS_REQUIRING_MADE for r in rows):
        made_model = _load_frozen_made(made_checkpoint, args.smoke_random_made, args.seed)

    x_pred_all = _chunked_predictor_forward(predictor, context_all, args.pred_chunk_size)

    clamp_model = ClampBaseline(constraints=drop_position_bounds(kinematic_bicycle_constraints()))

    timing_repeats = args.timing_repeats
    timing_batch = min(args.timing_batch_size, num_windows)

    result_rows: dict[str, dict[str, float]] = {}

    progress_file, progress_disabled, progress_owned_file = _progress_output()
    progress = tqdm(
        total=len(rows),
        desc="eval_lib",
        unit="row",
        disable=progress_disabled,
        file=progress_file,
        dynamic_ncols=True,
    )

    try:
        if "raw" in rows:
            progress.set_postfix(row="raw")
            x_row = x_pred_all
            x_full = _full_sequence(x0_all, x_row)
            u_dyn = _derive_dynamics_controls(x_full, dt)
            metrics = _row_metrics(
                x_row,
                x_gt_all,
                None,
                x_full,
                u_dyn,
                envelope,
                physical_constraints,
                gt_residual,
                dt,
            )
            metrics.update(
                _time_pipeline(
                    lambda ctx: predictor(ctx),
                    (context_all[0],),
                    (context_all[:timing_batch],),
                    repeats=timing_repeats,
                )
            )
            result_rows["raw"] = metrics
            progress.update(1)

        if "clamp" in rows:
            progress.set_postfix(row="clamp")
            x_row = _clamp_rows(clamp_model, x_pred_all)
            x_full = _full_sequence(x0_all, x_row)
            u_dyn = _derive_dynamics_controls(x_full, dt)
            metrics = _row_metrics(
                x_row,
                x_gt_all,
                None,
                x_full,
                u_dyn,
                envelope,
                physical_constraints,
                gt_residual,
                dt,
            )
            metrics.update(
                _time_pipeline(
                    lambda ctx: _clamp_trajectory(clamp_model, predictor(ctx)),
                    (context_all[0],),
                    (context_all[:timing_batch],),
                    repeats=timing_repeats,
                )
            )
            result_rows["clamp"] = metrics
            progress.update(1)

        if "smoother" in rows:
            progress.set_postfix(row="smoother")
            _sm, _sm_tuning = smoother_arms["smoother"]
            x_row = _smoother_rows(_sm, x0_all, x_pred_all)
            x_full = _full_sequence(x0_all, x_row)
            u_dyn = _derive_dynamics_controls(x_full, dt)
            metrics = _row_metrics(
                x_row,
                x_gt_all,
                None,
                x_full,
                u_dyn,
                envelope,
                physical_constraints,
                gt_residual,
                dt,
            )
            metrics.update(
                _time_pipeline(
                    # `_sm` is bound per row block rather than by a loop variable: a loop would
                    # have every block's lambda close over the LAST arm, and the timing column
                    # would silently belong to the wrong one.
                    lambda ctx, _sm=_sm: _smoother_trajectory(
                        _sm, ctx[-1, :_SMOOTHER_STATE_DIM], predictor(ctx)),
                    (context_all[0],),
                    (context_all[:timing_batch],),
                    repeats=timing_repeats,
                )
            )
            metrics["smoother_tuning"] = _sm_tuning
            result_rows["smoother"] = metrics
            progress.update(1)

        if "smoother_dyn" in rows:
            progress.set_postfix(row="smoother_dyn")
            _sm, _sm_tuning = smoother_arms["smoother_dyn"]
            x_row = _smoother_rows(_sm, x0_all, x_pred_all)
            x_full = _full_sequence(x0_all, x_row)
            u_dyn = _derive_dynamics_controls(x_full, dt)
            metrics = _row_metrics(
                x_row,
                x_gt_all,
                None,
                x_full,
                u_dyn,
                envelope,
                physical_constraints,
                gt_residual,
                dt,
            )
            metrics.update(
                _time_pipeline(
                    # `_sm` is bound per row block rather than by a loop variable: a loop would
                    # have every block's lambda close over the LAST arm, and the timing column
                    # would silently belong to the wrong one.
                    lambda ctx, _sm=_sm: _smoother_trajectory(
                        _sm, ctx[-1, :_SMOOTHER_STATE_DIM], predictor(ctx)),
                    (context_all[0],),
                    (context_all[:timing_batch],),
                    repeats=timing_repeats,
                )
            )
            metrics["smoother_tuning"] = _sm_tuning
            result_rows["smoother_dyn"] = metrics
            progress.update(1)

        if "made_pnp" in rows:
            progress.set_postfix(row="made_pnp")
            x_row, u_row = _made_pnp_rows(made_model, x_pred_all, metadata_all, x0_all, dt)
            x_full = _full_sequence(x0_all, x_row)
            u_kb = _derive_dynamics_controls(x_full, dt)
            metrics = _row_metrics(
                x_row,
                x_gt_all,
                u_row,
                x_full,
                u_row,
                envelope,
                physical_constraints,
                gt_residual,
                dt,
                u_dyn_kb=u_kb,
            )

            def _made_pnp_pipeline(ctx, meta, x0):
                x_pred = predictor(ctx)
                params = made_model.params_from_metadata(meta)
                x_corr, _u = _made_pnp_single(made_model.cell, x_pred, params, x0, dt)
                return x_corr

            metrics.update(
                _time_pipeline(
                    _made_pnp_pipeline,
                    (context_all[0], metadata_all[0], x0_all[0]),
                    (
                        context_all[:timing_batch],
                        metadata_all[:timing_batch],
                        x0_all[:timing_batch],
                    ),
                    repeats=timing_repeats,
                )
            )
            result_rows["made_pnp"] = metrics
            progress.update(1)
    finally:
        progress.close()
        if progress_owned_file is not None:
            progress_owned_file.close()

    payload = {
        "metric_version": METRIC_VERSION,
        "rows": result_rows,
        "provenance": {
            "window_spec": window_spec,
            "data_dir": args.data_dir,
            "predictor_dir": predictor_dir,
            "predictor_kind": predictor_config["kind"],
            "made_checkpoint": made_checkpoint,
            "smoke_random_made": bool(args.smoke_random_made),
            "use_stub": bool(args.use_stub),
            "seed": args.seed,
            "l_ref": L_REF,
            "git_sha": _git_sha(),
            "num_test_windows": num_windows,
            "hardware": _hardware_string(),
            "timing_batch_size": timing_batch,
            "timing_repeats": timing_repeats,
            "pred_chunk_size": int(args.pred_chunk_size),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"[eval_lib] wrote {output_path} (rows={list(result_rows)})", file=sys.stderr)


if __name__ == "__main__":
    main()
