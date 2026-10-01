# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Tests for the smooth-control (Ornstein-Uhlenbeck) control profile.

Covers: bit-compat of the default "iid_uniform" path (the canonical
simulated datasets must reproduce exactly), OU statistical properties (in-box,
lag-1 autocorrelation), and the DB/6D calmer-initial-state narrowing.
"""

# ruff: noqa: E402

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from made.physics import (
    ConstraintSet,
    DoubleIntegrator,
    DynamicBicycle,
    dynamic_bicycle_constraints,
)
from made.physics.simulator import (
    _generate_single_trajectory,
    _sample_control_sequence,
    _sample_control_sequence_ou,
    _sample_controls,
    _sample_initial_state,
    generate_trajectories,
)
from made.utils.config import DataConfig

_DB = DynamicBicycle()
_DB_CONSTRAINTS = dynamic_bicycle_constraints()
_DB_TRUE_PARAMS = jnp.array([20000.0, 20000.0, 1500.0, 3000.0, 1.2, 1.6], dtype=jnp.float64)


def _reference_iid_control_sequence(
    constraints: ConstraintSet, length: int, key: jax.Array
) -> jax.Array:
    """Inline reference reproducing `_sample_control_sequence`'s body, to guard against
    accidental drift.

    The dynamic bicycle's control-sampling bounds are the constraint set's own bounds
    (steering +-0.5, acceleration +-3.0); this reference reproduces that behaviour.

    Args:
        constraints: Constraint set whose control bounds are used.
        length: Number of control steps.
        key: PRNG key.

    Returns:
        Controls of shape `(length, control_dim)`.
    """
    control_min = constraints.control_min
    control_max = constraints.control_max
    shape = (length, control_min.shape[0])
    return jax.random.uniform(key, shape, minval=control_min, maxval=control_max)


class TestDefaultProfileBitCompat:
    def test_sample_controls_dispatch_matches_direct_call(self) -> None:
        """`_sample_controls` with 'iid_uniform' must call `_sample_control_sequence`
        with the exact same key — no extra split, no altered draw count."""
        key = jax.random.key(7)
        direct = _sample_control_sequence(_DB_CONSTRAINTS, 10, key)
        via_dispatch = _sample_controls(_DB, _DB_CONSTRAINTS, 10, 0.1, key, "iid_uniform", 1.5)
        assert jnp.array_equal(direct, via_dispatch)

    def test_iid_matches_inlined_reference(self) -> None:
        """Check that iid matches inlined reference."""
        key = jax.random.key(11)
        reference = _reference_iid_control_sequence(_DB_CONSTRAINTS, 15, key)
        actual = _sample_control_sequence(_DB_CONSTRAINTS, 15, key)
        assert jnp.array_equal(reference, actual)

    def test_generate_single_trajectory_default_matches_no_kwargs_call(self) -> None:
        """Calling with explicit default kwargs must reproduce the positional-only
        pre-existing call path bit-for-bit (same key consumption throughout)."""
        key = jax.random.key(3)
        states_a, controls_a, ok_a = _generate_single_trajectory(
            _DB, _DB_CONSTRAINTS, 10, 0.1, key, _DB_TRUE_PARAMS
        )
        states_b, controls_b, ok_b = _generate_single_trajectory(
            _DB,
            _DB_CONSTRAINTS,
            10,
            0.1,
            key,
            _DB_TRUE_PARAMS,
            control_profile="iid_uniform",
            control_tau=1.5,
        )
        assert jnp.array_equal(states_a, states_b)
        assert jnp.array_equal(controls_a, controls_b)
        assert bool(ok_a) == bool(ok_b)

    def test_sample_initial_state_default_unaffected_by_new_kwarg(self) -> None:
        """Check that sample initial state default unaffected by new kwarg."""
        key = jax.random.key(21)
        default_call = _sample_initial_state(_DB, _DB_CONSTRAINTS, key)
        explicit_default = _sample_initial_state(_DB, _DB_CONSTRAINTS, key, "iid_uniform")
        assert jnp.array_equal(default_call, explicit_default)

    def test_generate_trajectories_default_matches_explicit_default_kwargs(self) -> None:
        """Check that generate trajectories default matches explicit default kwargs."""
        key = jax.random.key(0)
        states_a, controls_a = generate_trajectories(
            _DB, _DB_CONSTRAINTS, 4, 8, 0.05, key, _DB_TRUE_PARAMS
        )
        states_b, controls_b = generate_trajectories(
            _DB,
            _DB_CONSTRAINTS,
            4,
            8,
            0.05,
            key,
            _DB_TRUE_PARAMS,
            control_profile="iid_uniform",
            control_tau=1.5,
        )
        assert jnp.array_equal(states_a, states_b)
        assert jnp.array_equal(controls_a, controls_b)


class TestOUControlProfile:
    def test_ou_controls_within_bounds(self) -> None:
        """Check that ou controls within bounds."""
        key = jax.random.key(5)
        controls = _sample_control_sequence_ou(_DB_CONSTRAINTS, 60, 0.1, 1.5, key)
        assert controls.shape == (60, 2)
        assert jnp.all(jnp.isfinite(controls))
        # The sampler draws from the CONSTRAINT SET's own bounds (+-0.5 / +-3.0 for the dynamic
        # bicycle), not a narrower default.
        assert jnp.all(controls >= _DB_CONSTRAINTS.control_min - 1e-9)
        assert jnp.all(controls <= _DB_CONSTRAINTS.control_max + 1e-9)

    def test_ou_lag1_autocorrelation_high(self) -> None:
        """Check that ou lag1 autocorrelation high."""
        # rho = 1 - dt/tau (continuous-time OU discretisation, drift term
        # linearised in dt). At dt=0.1, tau=1.5: rho = 1 - 0.1/1.5 ~= 0.9333.
        # Use a long sequence and average over many independent trajectories
        # to get a stable empirical estimate, then check it clears a threshold
        # comfortably below the theoretical 0.933 (0.8 leaves slack for
        # clipping-induced attenuation near the box edges).
        keys = jax.random.split(jax.random.key(42), 64)
        all_controls = jax.vmap(
            lambda k: _sample_control_sequence_ou(_DB_CONSTRAINTS, 200, 0.1, 1.5, k)
        )(keys)
        steer = np.asarray(all_controls[:, :, 0])
        x_t = steer[:, :-1]
        x_tp1 = steer[:, 1:]
        x_t_flat = x_t.reshape(-1)
        x_tp1_flat = x_tp1.reshape(-1)
        corr = np.corrcoef(x_t_flat, x_tp1_flat)[0, 1]
        assert corr > 0.8, f"lag-1 autocorrelation {corr:.3f} below 0.8 threshold"

    def test_iid_lag1_autocorrelation_near_zero(self) -> None:
        """Check that iid lag1 autocorrelation near zero."""
        keys = jax.random.split(jax.random.key(43), 64)
        all_controls = jax.vmap(
            lambda k: _sample_control_sequence(_DB_CONSTRAINTS, 200, k)
        )(keys)
        steer = np.asarray(all_controls[:, :, 0])
        x_t = steer[:, :-1].reshape(-1)
        x_tp1 = steer[:, 1:].reshape(-1)
        corr = np.corrcoef(x_t, x_tp1)[0, 1]
        assert abs(corr) < 0.1, f"i.i.d. lag-1 autocorrelation {corr:.3f} not near zero"

    def test_ou_dispatch_matches_direct_call(self) -> None:
        """Check that ou dispatch matches direct call."""
        key = jax.random.key(9)
        direct = _sample_control_sequence_ou(_DB_CONSTRAINTS, 12, 0.1, 1.5, key)
        via_dispatch = _sample_controls(_DB, _DB_CONSTRAINTS, 12, 0.1, key, "smooth_ou", 1.5)
        assert jnp.array_equal(direct, via_dispatch)

    def test_non_db_system_ou_runs_and_stays_in_box(self) -> None:
        """Check that non db system ou runs and stays in box."""
        di = DoubleIntegrator()
        from made.physics import double_integrator_constraints

        constraints = double_integrator_constraints()
        key = jax.random.key(13)
        controls = _sample_control_sequence_ou(constraints, 30, 0.1, 1.5, key)
        assert controls.shape == (30, 2)
        assert jnp.all(controls >= constraints.control_min - 1e-9)
        assert jnp.all(controls <= constraints.control_max + 1e-9)


class TestCalmerInitialState6D:
    def test_smooth_ou_calms_initial_state(self) -> None:
        """Check that smooth ou calms initial state."""
        keys = jax.random.split(jax.random.key(1), 200)
        states = jax.vmap(
            lambda k: _sample_initial_state(_DB, _DB_CONSTRAINTS, k, "smooth_ou")
        )(keys)
        v0 = states[:, 3]
        vy0 = states[:, 4]
        yaw0 = states[:, 5]
        assert jnp.all(v0 <= 15.0 + 1e-9)
        # Narrowed DB bound for vy/yaw-rate before calming is 1.0 / 0.3, so
        # calmed values must sit within 10% of that: |vy0| <= 0.1, |yaw0| <= 0.03.
        assert jnp.all(jnp.abs(vy0) <= 0.1 + 1e-9)
        assert jnp.all(jnp.abs(yaw0) <= 0.03 + 1e-9)

    def test_iid_uniform_does_not_calm_initial_state(self) -> None:
        """Check that iid uniform does not calm initial state."""
        keys = jax.random.split(jax.random.key(2), 200)
        states = jax.vmap(
            lambda k: _sample_initial_state(_DB, _DB_CONSTRAINTS, k, "iid_uniform")
        )(keys)
        # Without calming, vy/yaw-rate should regularly exceed the calmed window
        # (they are drawn from the full narrowed DB box: [-1, 1] / [-0.3, 0.3]).
        vy0 = states[:, 4]
        assert bool(jnp.any(jnp.abs(vy0) > 0.1))


class TestFeasibilitySmoke:
    def test_smooth_ou_db_trajectories_generate_feasible(self) -> None:
        """Check that smooth ou db trajectories generate feasible."""
        states, controls = generate_trajectories(
            _DB,
            _DB_CONSTRAINTS,
            8,
            60,
            0.1,
            jax.random.key(100),
            _DB_TRUE_PARAMS,
            control_profile="smooth_ou",
            control_tau=1.5,
        )
        assert states.shape == (8, 60, 6)
        assert controls.shape == (8, 59, 2)
        assert jnp.all(jnp.isfinite(states))
        assert jnp.all(jnp.isfinite(controls))


class TestMinSpeedFeasibilityFloor:
    def test_default_none_matches_explicit_none(self) -> None:
        """min_speed=None must reproduce the pre-existing call byte-for-byte."""
        key = jax.random.key(0)
        states_a, controls_a = generate_trajectories(
            _DB,
            _DB_CONSTRAINTS,
            4,
            8,
            0.05,
            key,
            _DB_TRUE_PARAMS,
            control_profile="smooth_ou",
            control_tau=1.5,
        )
        states_b, controls_b = generate_trajectories(
            _DB,
            _DB_CONSTRAINTS,
            4,
            8,
            0.05,
            key,
            _DB_TRUE_PARAMS,
            control_profile="smooth_ou",
            control_tau=1.5,
            min_speed=None,
        )
        assert jnp.array_equal(states_a, states_b)
        assert jnp.array_equal(controls_a, controls_b)

    def test_default_none_initial_state_matches_explicit_none(self) -> None:
        """Check that default none initial state matches explicit none."""
        key = jax.random.key(17)
        default_call = _sample_initial_state(_DB, _DB_CONSTRAINTS, key, "smooth_ou")
        explicit_none = _sample_initial_state(_DB, _DB_CONSTRAINTS, key, "smooth_ou", None)
        assert jnp.array_equal(default_call, explicit_none)

    def test_min_speed_enforced_on_generated_trajectories(self) -> None:
        """Check that min speed enforced on generated trajectories."""
        states, controls = generate_trajectories(
            _DB,
            _DB_CONSTRAINTS,
            8,
            40,
            0.1,
            jax.random.key(200),
            _DB_TRUE_PARAMS,
            control_profile="smooth_ou",
            control_tau=1.5,
            min_speed=1.0,
        )
        assert states.shape == (8, 40, 6)
        assert jnp.all(states[:, :, 3] >= 1.0 - 1e-9)

    def test_min_speed_raises_v0_floor(self) -> None:
        """Check that min speed raises v0 floor."""
        keys = jax.random.split(jax.random.key(23), 100)
        states = jax.vmap(
            lambda k: _sample_initial_state(_DB, _DB_CONSTRAINTS, k, "smooth_ou", 5.0)
        )(keys)
        assert jnp.all(states[:, 3] >= 5.0 - 1e-9)

    def test_min_speed_below_default_clamp_does_not_lower_v0_floor(self) -> None:
        """Check that min speed below default clamp does not lower v0 floor."""
        # min_speed=1.0 < the pre-existing 2.0 clamp, so the floor stays 2.0.
        keys = jax.random.split(jax.random.key(24), 100)
        states = jax.vmap(
            lambda k: _sample_initial_state(_DB, _DB_CONSTRAINTS, k, "smooth_ou", 1.0)
        )(keys)
        assert jnp.all(states[:, 3] >= 2.0 - 1e-9)

    def test_min_speed_on_non_db_system_raises(self) -> None:
        """Check that min speed on non db system raises."""
        import pytest

        di = DoubleIntegrator()
        from made.physics import double_integrator_constraints

        constraints = double_integrator_constraints()
        true_params = jnp.zeros((0,), dtype=jnp.float64)
        with pytest.raises(ValueError):
            generate_trajectories(
                di,
                constraints,
                2,
                8,
                0.1,
                jax.random.key(0),
                true_params,
                min_speed=1.0,
            )


def test_data_config_control_profile_defaults() -> None:
    """Check that data config control profile defaults."""
    config = DataConfig()
    assert config.control_profile == "iid_uniform"
    assert config.control_tau == 1.5
    assert config.min_speed is None


def test_data_config_rejects_invalid_control_profile() -> None:
    """Check that data config rejects invalid control profile."""
    import pytest

    with pytest.raises(ValueError):
        DataConfig(control_profile="not_a_profile")


def test_data_config_rejects_non_positive_control_tau() -> None:
    """Check that data config rejects non positive control tau."""
    import pytest

    with pytest.raises(ValueError):
        DataConfig(control_tau=0.0)
