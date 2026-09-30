"""Clamp baseline smoke tests for the inD pipeline (Plan 1 / inD baseline coverage).

Mirrors test_mlp_inD.py: exercises ClampBaseline with kinematic-bicycle box
constraints on inD-shaped metadata (width 5), verifying shape, finite outputs,
and that states are clipped to the declared bounds.
"""

# ruff: noqa: E402
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from made.baselines.clamp_baseline import ClampBaseline
from made.data.ind_data import IND_STATE_DIM
from made.physics.constraints import drop_position_bounds, kinematic_bicycle_constraints


def _kb_clamp() -> ClampBaseline:
    return ClampBaseline(constraints=drop_position_bounds(kinematic_bicycle_constraints()))


def _ind_metadata(loc_id: int = 1) -> jax.Array:
    """Return a single inD-shaped metadata vector [length, width, car=1, truck=0, loc_id]."""
    return jnp.array([4.5, 1.8, 1.0, 0.0, float(loc_id)], dtype=jnp.float64)


def test_clamp_output_shape():
    model = _kb_clamp()
    x_prev = jnp.zeros(IND_STATE_DIM, dtype=jnp.float64)
    x_curr = jnp.ones(IND_STATE_DIM, dtype=jnp.float64) * 0.1
    out_prev, out_curr = model.correct_pair(x_prev, x_curr, _ind_metadata())
    assert out_prev.shape == (IND_STATE_DIM,)
    assert out_curr.shape == (IND_STATE_DIM,)


def test_clamp_output_finite():
    model = _kb_clamp()
    x_prev = jnp.zeros(IND_STATE_DIM, dtype=jnp.float64)
    x_curr = jnp.ones(IND_STATE_DIM, dtype=jnp.float64) * 0.1
    out_prev, out_curr = model.correct_pair(x_prev, x_curr, _ind_metadata())
    assert jnp.all(jnp.isfinite(out_prev))
    assert jnp.all(jnp.isfinite(out_curr))


def test_clamp_enforces_state_bounds():
    """Output states must lie within [state_min, state_max]."""
    constraints = kinematic_bicycle_constraints()
    model = ClampBaseline(constraints=constraints)

    # Deliberately out-of-bounds inputs (v = 100 > v_max=13.9, x = 200 > 50)
    x_oob = jnp.array([200.0, -200.0, 5.0, 100.0], dtype=jnp.float64)
    _, out_curr = model.correct_pair(x_oob, x_oob, _ind_metadata())

    assert jnp.all(out_curr >= constraints.state_min)
    assert jnp.all(out_curr <= constraints.state_max)


def test_clamp_in_bounds_input_unchanged():
    """States already within bounds must pass through unmodified."""
    model = _kb_clamp()
    x_in = jnp.array([1.0, 2.0, 0.5, 5.0], dtype=jnp.float64)
    out_prev, out_curr = model.correct_pair(x_in, x_in, _ind_metadata())
    assert jnp.allclose(out_prev, x_in)
    assert jnp.allclose(out_curr, x_in)


def test_clamp_metadata_ignored():
    """ClampBaseline must produce the same result regardless of metadata."""
    model = _kb_clamp()
    x = jnp.array([1.0, 2.0, 0.5, 5.0], dtype=jnp.float64)
    out1 = model.correct_pair(x, x, _ind_metadata(loc_id=1))
    out2 = model.correct_pair(x, x, _ind_metadata(loc_id=4))
    assert jnp.allclose(out1[0], out2[0])
    assert jnp.allclose(out1[1], out2[1])


def test_clamp_xy_unchanged_after_correct_pair():
    """x,y components must pass through unchanged when bounds are ±inf (drop_position_bounds)."""
    model = _kb_clamp()
    # Deliberately out-of-position-range x,y values — must NOT be clamped
    x = jnp.array([999.0, -888.0, 0.1, 5.0], dtype=jnp.float64)
    _, out_curr = model.correct_pair(x, x, _ind_metadata())
    assert float(out_curr[0]) == 999.0, f"x was clamped: {out_curr[0]}"
    assert float(out_curr[1]) == -888.0, f"y was clamped: {out_curr[1]}"
