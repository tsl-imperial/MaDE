# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Checkpoint contract integration test.

Verifies:
1. MaDEModel.from_checkpoint loads a saved checkpoint and runs a forward pass.
2. FABBaseline.from_checkpoint loads a saved checkpoint and runs a forward pass.
3. Both models produce finite output on a synthetic kinematic-bicycle batch.

Uses synthetic data only — no real inD dataset required.
"""

# ruff: noqa: E402
from pathlib import Path
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

jax.config.update("jax_enable_x64", True)

from made.baselines.fab_baseline import FABBaseline, save_fab_checkpoint
from made.models.made_model import MaDEModel
from made.physics import KinematicBicycle, kinematic_bicycle_constraints
from made.utils.checkpointing import CheckpointManager, TrainState
from made.utils.config import CorrectorConfig, ModelConfig


# Helpers


def _make_made_model(key: jax.Array) -> MaDEModel:
    """Small MaDEModel with inD metadata schema (metadata_dim=5, num_locations=4).

    Args:
        key: PRNG key for initialisation.

    Returns:
        Small MaDEModel.
    """
    physics = KinematicBicycle()
    constraints = kinematic_bicycle_constraints()
    model_cfg = ModelConfig(
        inverse_hidden=(16, 16),
        residual_hidden=(16, 16),
        encoder_hidden=(16,),
        use_metadata_encoder=True,
        metadata_dim=5,
        num_locations=4,
        embedding_dim=8,
        location_id_index=4,
    )
    corrector_cfg = CorrectorConfig(mode="enabled", train_steps=2, eval_max_steps=4)
    return MaDEModel.from_config(
        physics,
        constraints,
        model_cfg,
        corrector_cfg,
        key=key,
    )


def _make_fab(key: jax.Array) -> FABBaseline:
    """Build a tiny FABBaseline for tests.

    Args:
        key: PRNG key for initialisation.

    Returns:
        Small FABBaseline.
    """
    return FABBaseline(
        state_dim=4,
        latent_dim=8,
        num_experts=2,
        hidden=(16, 16),
        latent_radius=1.0,
        key=key,
    )


# Tests


def test_made_model_from_checkpoint_roundtrip(tmp_path: Path) -> None:
    """MaDEModel.from_checkpoint loads saved model and produces finite forward output."""
    model = _make_made_model(jax.random.key(0))
    state = TrainState(
        model=model,
        opt_state_I=None,
        opt_state_T=None,
        key=jax.random.key(1),
        step=0,
    )
    ckpt_dir = str(tmp_path / "made_checkpoint")
    cm = CheckpointManager(ckpt_dir)
    cm.save(state, 0)

    loaded = MaDEModel.from_checkpoint(ckpt_dir)
    assert isinstance(loaded, MaDEModel)

    # eval_max_steps == 0 would silently disable correction even though
    # the forward call still returns finite values.
    assert loaded.corrector.eval_max_steps > 0, (
        "Restored corrector has eval_max_steps=0; corrector is silently inactive"
    )
    # Confirm the value matches what _make_made_model sets so a future config
    # drift can't sneak past this test.
    assert loaded.corrector.eval_max_steps == 4

    # Verify forward pass on synthetic kinematic-bicycle batch
    physics = KinematicBicycle()
    batch_size = 4
    dt = 0.2
    x_prev = jnp.zeros((batch_size, physics.state_dim), dtype=jnp.float64)
    x_curr = jnp.ones((batch_size, physics.state_dim), dtype=jnp.float64) * 0.05
    # metadata: [length, width, car=1, truck=0, location_id=1]
    metadata = jnp.tile(
        jnp.array([4.5, 1.8, 1.0, 0.0, 1.0], dtype=jnp.float64),
        (batch_size, 1),
    )

    corrected_x, corrected_u = jax.vmap(
        lambda xp, xc, m: loaded(xp, xc, params=None, dt=dt, metadata=m)
    )(x_prev, x_curr, metadata)

    assert corrected_x.shape == (batch_size, physics.state_dim)
    assert jnp.all(jnp.isfinite(corrected_x)), "MaDEModel forward produced NaN/Inf"


def test_made_model_corrector_reduces_known_violation(tmp_path: Path) -> None:
    """Restored MaDEModel must actually shrink a known constraint violation.

    A passing ``isfinite`` assertion is too weak — a
    corrector with ``eval_max_steps=0`` would still produce finite output
    while doing no work.  Seed a state below the kinematic-bicycle velocity
    floor (``v_min = 0``), run the restored model, and require that the
    ReLU'd violation strictly decreases.
    """
    model = _make_made_model(jax.random.key(11))
    state = TrainState(
        model=model,
        opt_state_I=None,
        opt_state_T=None,
        key=jax.random.key(12),
        step=0,
    )
    ckpt_dir = str(tmp_path / "made_violation")
    cm = CheckpointManager(ckpt_dir)
    cm.save(state, 0)
    loaded = MaDEModel.from_checkpoint(ckpt_dir)

    constraints = kinematic_bicycle_constraints()
    physics = KinematicBicycle()
    dt = 0.2

    # Construct an x_curr that violates v >= 0 — kinematic_bicycle v_min is 0.
    x_prev = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float64)
    x_curr = jnp.array([0.05, 0.0, 0.0, -1.5], dtype=jnp.float64)  # v = -1.5 (< v_min)
    metadata = jnp.array([4.5, 1.8, 1.0, 0.0, 1.0], dtype=jnp.float64)

    # Initial violation magnitude (sum of positive-only ReLU components).
    zero_u = jnp.zeros((physics.control_dim,), dtype=jnp.float64)
    initial_violation = jnp.sum(jnp.maximum(constraints(x_curr, zero_u), 0.0))
    assert float(initial_violation) > 0.0, "Test setup must seed a real violation"

    corrected_x, corrected_u = loaded(x_prev, x_curr, params=None, dt=dt, metadata=metadata)
    final_violation = jnp.sum(jnp.maximum(constraints(corrected_x, corrected_u), 0.0))

    assert jnp.all(jnp.isfinite(corrected_x))
    # The corrector should make progress — final violation strictly less than initial.
    # Use a small epsilon so flaky float64 noise isn't fatal.
    assert float(final_violation) < float(initial_violation) - 1e-6, (
        f"Corrector failed to reduce violation: "
        f"initial={float(initial_violation):.6e}, final={float(final_violation):.6e}"
    )


def test_fab_from_checkpoint_roundtrip(tmp_path: Path) -> None:
    """FABBaseline.from_checkpoint loads saved model and produces finite forward output."""
    model = _make_fab(jax.random.key(2))
    save_fab_checkpoint(model, str(tmp_path / "fab_checkpoint"))

    loaded = FABBaseline.from_checkpoint(str(tmp_path / "fab_checkpoint"))
    assert isinstance(loaded, FABBaseline)

    # Verify forward pass on synthetic batch
    x_prev = jnp.zeros(4, dtype=jnp.float64)
    x_curr = jnp.ones(4, dtype=jnp.float64) * 0.1
    pred_prev, pred_curr = loaded(x_prev, x_curr)

    assert pred_prev.shape == (4,)
    assert pred_curr.shape == (4,)
    assert jnp.all(jnp.isfinite(pred_prev)), "FABBaseline forward produced NaN"
    assert jnp.all(jnp.isfinite(pred_curr)), "FABBaseline forward produced NaN"


def test_made_model_from_checkpoint_raises_on_missing(tmp_path: Path) -> None:
    """MaDEModel.from_checkpoint raises FileNotFoundError when no checkpoint exists."""
    with pytest.raises(FileNotFoundError):
        MaDEModel.from_checkpoint(str(tmp_path / "no_such_dir"))


def test_fab_from_checkpoint_raises_on_missing(tmp_path: Path) -> None:
    """FABBaseline.from_checkpoint raises FileNotFoundError when no checkpoint exists."""
    with pytest.raises(FileNotFoundError):
        FABBaseline.from_checkpoint(str(tmp_path / "no_such_dir"))


def test_made_model_leaves_preserved_after_checkpoint(tmp_path: Path) -> None:
    """MaDEModel leaves are identical before and after checkpoint round-trip."""
    model = _make_made_model(jax.random.key(3))
    state = TrainState(
        model=model,
        opt_state_I=None,
        opt_state_T=None,
        key=jax.random.key(4),
        step=0,
    )
    ckpt_dir = str(tmp_path / "made_leaves")
    cm = CheckpointManager(ckpt_dir)
    cm.save(state, 0)

    loaded = MaDEModel.from_checkpoint(ckpt_dir)

    orig_leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    rest_leaves = jax.tree_util.tree_leaves(eqx.filter(loaded, eqx.is_array))
    assert len(orig_leaves) == len(rest_leaves)
    for i, (a, b) in enumerate(zip(orig_leaves, rest_leaves)):
        assert jnp.allclose(a, b), f"Leaf {i} differs after checkpoint round-trip"
