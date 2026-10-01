# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Real-data (inD experiments) evaluation helpers.

Loads training-split inD ground-truth states, runs the deterministic
kinematic-bicycle inverse-control estimator, derives the empirical envelope
and the GT reference dynamics residual, and packages a metadata block for
the result-JSON.

The helper is shared across the inD evaluators so envelope and
GT-residual provenance is identical between them.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import jax.numpy as jnp

from made.data.ind_data import create_ind_data_source
from made.evaluation.metrics import (
    METRIC_VERSION,
    EmpiricalEnvelope,
    estimate_empirical_envelope,
    estimate_gt_reference_residual,
    kinematic_bicycle_inverse_controls,
)
from made.physics import KinematicBicycle

__all__ = [
    "L_REF",
    "ENVELOPE_STATE_ROUNDING",
    "ENVELOPE_CONTROL_ROUNDING",
    "load_train_envelope_and_residual",
]

# Dataset-level proxy wheelbase for the inverse-control estimator and the
# known-physics reference residual. Recorded in result JSON metadata so the
# choice is auditable. Bumping this constant requires bumping
# ``METRIC_VERSION``.
L_REF: float = 2.7

# Outward rounding increments for the envelope. Inputs are inD state ordering
# ``[x, y, heading, speed]`` and KB control ordering ``[δ, a]``.
ENVELOPE_STATE_ROUNDING: tuple[float, ...] = (1.0, 1.0, 0.01, 0.1)
ENVELOPE_CONTROL_ROUNDING: tuple[float, ...] = (0.01, 0.05)

# Stationary-frame threshold (m/s) below which the inverse-control estimator
# carries forward δ instead of dividing by near-zero v_avg.
STATIONARY_SPEED_THRESHOLD: float = 0.5


def _cache_fingerprint(data_dir: str, dt: float) -> str:
    """Stable hash of inputs that determine envelope + GT-residual outputs.

    Includes the data dir's manifest/stats file size+mtime so any
    preprocessing change invalidates the cache automatically.

    Args:
        data_dir: Preprocessed inD root.
        dt: Timestep in seconds.
    Returns:
        Hex digest.
    """
    h = hashlib.sha256()
    h.update(
        f"v1|dt={dt}|L_REF={L_REF}|sst={STATIONARY_SPEED_THRESHOLD}|"
        f"sr={ENVELOPE_STATE_ROUNDING}|cr={ENVELOPE_CONTROL_ROUNDING}|"
        f"mv={METRIC_VERSION}|".encode()
    )
    for fname in ("manifest.json", "stats.json"):
        p = Path(data_dir) / fname
        if p.is_file():
            st = p.stat()
            h.update(f"{fname}:{st.st_size}:{st.st_mtime_ns}|".encode())
    return h.hexdigest()[:16]


def _cache_path(data_dir: str, dt: float) -> Path:
    """Path of the cache file for the given inputs.

    Args:
        data_dir: Preprocessed inD root.
        dt: Timestep in seconds.
    Returns:
        Cache file path.
    """
    return (
        Path(data_dir)
        / "_eval_cache"
        / f"envelope_residual_{_cache_fingerprint(data_dir, dt)}.json"
    )


def _envelope_from_metadata(meta: dict) -> EmpiricalEnvelope:
    """Rebuild an envelope from its JSON metadata.

    Args:
        meta: Metadata dict with an "envelope" entry.
    Returns:
        The envelope.
    """
    env = meta["envelope"]
    return EmpiricalEnvelope(
        state_min=jnp.asarray(env["state_min"]),
        state_max=jnp.asarray(env["state_max"]),
        control_min=jnp.asarray(env["control_min"]),
        control_max=jnp.asarray(env["control_max"]),
    )


def load_train_envelope_and_residual(
    data_dir: str,
    *,
    use_stub: bool,
    dt: float,
    smoke_seed: int = 0,
    split: str = "train",
    stub_num_trajectories: int = 32,
    stub_trajectory_length: int = 20,
) -> tuple[EmpiricalEnvelope, float, dict]:
    """Estimate envelope and GT reference residual from the training split.

    Returns a 3-tuple ``(envelope, gt_reference_residual, metadata)`` where
    ``metadata`` is a JSON-serialisable dict ready to embed under the
    top-level ``"metadata"`` field of an evaluator's result JSON.

    Train-only invariant: ``split`` MUST equal ``"train"``. The argument
    is exposed so the call site is explicit; mismatches raise ``ValueError``
    so test-split contamination cannot slip in silently.

    Args:
        data_dir: Preprocessed inD root.
        use_stub: Use synthetic data instead of the real split.
        dt: Timestep in seconds.
        smoke_seed: Seed for the synthetic data.
        split: Must be "train".
        stub_num_trajectories: Trajectory count for the stub.
        stub_trajectory_length: Trajectory length for the stub.
    Raises:
        ValueError: If `split` is not "train".

    Returns:
        Tuple (envelope, gt_reference_residual, metadata).
    """
    if split != "train":
        raise ValueError(
            f"load_train_envelope_and_residual requires split='train', got {split!r}"
        )

    # On-disk cache: envelope + GT residual depend only on (data_dir contents, dt, the
    # constants above). Reused across every variant/seed eval. Disable via
    # MADE_EVAL_CACHE_DISABLE=1 or for stub data.
    cache_file: Path | None = None
    if not use_stub and os.environ.get("MADE_EVAL_CACHE_DISABLE") != "1":
        try:
            cache_file = _cache_path(data_dir, dt)
            if cache_file.is_file():
                with cache_file.open() as f:
                    cached = json.load(f)
                envelope = _envelope_from_metadata(cached)
                print(
                    f"[real_data_eval] cache hit: {cache_file}",
                    file=sys.stderr,
                )
                return envelope, float(cached["gt_reference_residual"]), cached
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            print(
                f"[real_data_eval] cache load failed ({exc!r}); recomputing",
                file=sys.stderr,
            )

    states, _, lengths = create_ind_data_source(
        data_dir,
        split,
        use_stub=use_stub,
        stub_num_trajectories=stub_num_trajectories,
        stub_trajectory_length=stub_trajectory_length,
        stub_seed=smoke_seed,
    )

    controls, inv_aux = kinematic_bicycle_inverse_controls(
        states,
        dt,
        wheelbase=L_REF,
        stationary_speed_threshold=STATIONARY_SPEED_THRESHOLD,
    )

    envelope = estimate_empirical_envelope(
        states,
        controls,
        state_rounding=ENVELOPE_STATE_ROUNDING,
        control_rounding=ENVELOPE_CONTROL_ROUNDING,
        lengths=lengths,
    )
    gt_residual = estimate_gt_reference_residual(
        states,
        controls,
        KinematicBicycle(),
        jnp.array([L_REF]),
        dt,
        lengths=lengths,
    )

    metadata = {
        "envelope": {
            "state_min": [float(v) for v in envelope.state_min],
            "state_max": [float(v) for v in envelope.state_max],
            "control_min": [float(v) for v in envelope.control_min],
            "control_max": [float(v) for v in envelope.control_max],
            "lower_quantile": 0.01,
            "upper_quantile": 0.99,
            "state_rounding": list(ENVELOPE_STATE_ROUNDING),
            "control_rounding": list(ENVELOPE_CONTROL_ROUNDING),
        },
        "gt_reference_residual": float(gt_residual),
        "inverse_control": {
            "source": "kb_known_control_prior",
            "wheelbase": L_REF,
            "stationary_speed_threshold": STATIONARY_SPEED_THRESHOLD,
            "stationary_frame_count": int(inv_aux["stationary_frame_count"]),
            # Train-side speed > threshold by construction; the inD evaluator
            # overrides this when stationary headings appear in upstream predictions.
            "heading_undefined_frames": 0,
        },
        "wheelbase_rationale": (
            "fixed L_REF for cross-row comparability; per-MaDE-variant "
            "`learned_wheelbase` is reported separately under each model "
            "block and differs from `metadata.inverse_control.wheelbase` "
            "(= L_REF)."
        ),
        "metric_version": METRIC_VERSION,
    }

    if cache_file is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(cache_file.suffix + f".tmp.{os.getpid()}")
            tmp.write_text(json.dumps(metadata, indent=2))
            os.replace(tmp, cache_file)
            print(
                f"[real_data_eval] cache write: {cache_file}",
                file=sys.stderr,
            )
        except OSError as exc:
            print(
                f"[real_data_eval] cache write failed ({exc!r}); continuing",
                file=sys.stderr,
            )

    return envelope, float(gt_residual), metadata
