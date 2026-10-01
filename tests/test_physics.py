# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

import jax
import jax.numpy as jnp

from made.physics import (
    DoubleIntegrator,
    DynamicBicycle,
    KinematicBicycle,
    Unicycle,
    double_integrator_constraints,
)


def test_double_integrator_vector_field() -> None:
    """Check that double integrator vector field."""
    model = DoubleIntegrator()
    result = model.vector_field(
        jnp.array([0.0, 0.0, 1.0, 0.0]),
        jnp.array([0.0, 0.0]),
        jnp.array([]),
        0.0,
    )
    assert jnp.allclose(result, jnp.array([1.0, 0.0, 0.0, 0.0]))


def test_vmap_vector_field() -> None:
    """Check that vmap vector field."""
    model = DoubleIntegrator()
    states = jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]), (8, 1))
    controls = jnp.zeros((8, 2))
    params = jnp.zeros((8, 0))
    times = jnp.zeros((8,))
    result = jax.vmap(model.vector_field)(states, controls, params, times)
    assert result.shape == (8, 4)


def test_box_constraints_feasible_and_infeasible() -> None:
    """Check that box constraints feasible and infeasible."""
    constraints = double_integrator_constraints()
    feasible = constraints(jnp.array([0.0, 0.0, 0.5, 0.5]), jnp.array([0.0, 0.0]))
    infeasible = constraints(jnp.array([0.0, 0.0, 10.0, 10.0]), jnp.array([0.0, 0.0]))
    assert jnp.all(feasible <= 0.0)
    assert jnp.any(infeasible > 0.0)


def test_param_scales_shape() -> None:
    """param_scales has shape (param_dim,) for each physics model."""
    di = DoubleIntegrator()
    assert di.param_scales.shape == (0,)

    uni = Unicycle()
    assert uni.param_scales.shape == (0,)

    kin = KinematicBicycle()
    assert kin.param_scales.shape == (1,)
    assert jnp.all(kin.param_scales > 0)

    dyn = DynamicBicycle()
    assert dyn.param_scales.shape == (6,)
    assert jnp.all(dyn.param_scales > 0)


def test_dynamic_bicycle_param_scales_cover_raw_values() -> None:
    """DynamicBicycle param_scales are >= raw default param values so normalized inputs are O(1)."""
    from made.physics import resolve_params
    params = resolve_params("dynamic_bicycle", {})
    scales = DynamicBicycle().param_scales
    # Each normalized component should be <= 1.0 (scale >= raw value)
    assert jnp.all(params / scales <= 1.0)
