import jax
import jax.numpy as jnp

from made.physics import (
    DoubleIntegrator,
    DynamicBicycle,
    KinematicBicycle,
    Unicycle,
    double_integrator_constraints,
)


def test_double_integrator_vector_field():
    model = DoubleIntegrator()
    result = model.vector_field(
        jnp.array([0.0, 0.0, 1.0, 0.0]),
        jnp.array([0.0, 0.0]),
        jnp.array([]),
        0.0,
    )
    assert jnp.allclose(result, jnp.array([1.0, 0.0, 0.0, 0.0]))


def test_vmap_vector_field():
    model = DoubleIntegrator()
    states = jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]), (8, 1))
    controls = jnp.zeros((8, 2))
    params = jnp.zeros((8, 0))
    times = jnp.zeros((8,))
    result = jax.vmap(model.vector_field)(states, controls, params, times)
    assert result.shape == (8, 4)


def test_box_constraints_feasible_and_infeasible():
    constraints = double_integrator_constraints()
    feasible = constraints(jnp.array([0.0, 0.0, 0.5, 0.5]), jnp.array([0.0, 0.0]))
    infeasible = constraints(jnp.array([0.0, 0.0, 10.0, 10.0]), jnp.array([0.0, 0.0]))
    assert jnp.all(feasible <= 0.0)
    assert jnp.any(infeasible > 0.0)


def test_param_scales_shape():
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


def test_dynamic_bicycle_param_scales_cover_raw_values():
    """DynamicBicycle param_scales are >= raw default param values so normalized inputs are O(1)."""
    from made.physics import resolve_params
    params = resolve_params("dynamic_bicycle", {})
    scales = DynamicBicycle().param_scales
    # Each normalized component should be <= 1.0 (scale >= raw value)
    assert jnp.all(params / scales <= 1.0)
