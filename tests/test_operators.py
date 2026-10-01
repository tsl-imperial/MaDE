# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

import jax.numpy as jnp

from made.models import MetadataEncoder
from made.models.inverse_dynamics import InverseDynamics
from made.physics import DynamicBicycle

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from made.models import MaDECell
    import jax


def test_inverse_dynamics_shape(small_cell: "MaDECell") -> None:
    """Check that inverse dynamics shape."""
    control = small_cell.inverse_dynamics(
        jnp.zeros((4,)),
        jnp.ones((4,)),
        jnp.zeros((0,)),
    )
    assert control.shape == (2,)


def test_augmented_vector_field_additive(small_cell: "MaDECell") -> None:
    """Check that augmented vector field additive."""
    state = jnp.zeros((4,))
    control = jnp.zeros((2,))
    params = jnp.zeros((0,))
    physics_value = small_cell.augmented_dynamics.physics.vector_field(state, control, params, 0.0)
    residual_value = small_cell.augmented_dynamics.residual(state, control, params, 0.0)
    total = small_cell.augmented_dynamics.vector_field(state, control, params, 0.0)
    assert jnp.allclose(total, physics_value + residual_value)


def test_metadata_encoder_bounds(small_key: "jax.Array") -> None:
    """Check that metadata encoder bounds."""
    encoder = MetadataEncoder(10, 1, (16, 16), jnp.array([5.0]), key=small_key)
    output = encoder(jnp.ones((10,)))
    assert output.shape == (1,)
    assert jnp.all(output >= 0.0)
    assert jnp.all(output <= 5.0)


def test_learned_component_uses_normalized_params(small_key: "jax.Array") -> None:
    """InverseDynamics.learned_component divides params by param_scales before the MLP."""
    physics = DynamicBicycle()
    inv = InverseDynamics(
        state_dim=physics.state_dim,
        control_dim=physics.control_dim,
        param_dim=physics.param_dim,
        hidden=(16, 16),
        key=small_key,
        known_physics=physics,
        dt=0.1,
        use_residual=True,
    )
    from made.physics import resolve_params
    raw_params = resolve_params("dynamic_bicycle", {})

    # Verify param_scales has the expected magnitude (C_f/C_r ~20K)
    assert float(physics.param_scales[0]) >= 1000.0

    # learned_component should not raise and should return control-shaped output
    x = jnp.zeros((physics.state_dim,))
    delta_i = inv.learned_component(x, x, raw_params)
    assert delta_i.shape == (physics.control_dim,)
    assert jnp.all(jnp.isfinite(delta_i))


def test_normalized_params_are_order_one(small_key: "jax.Array") -> None:
    """params / param_scales should be O(1) for default raw params (the bug-fix invariant)."""
    del small_key
    from made.physics import resolve_params
    physics = DynamicBicycle()
    raw_params = resolve_params("dynamic_bicycle", {})
    normalized = raw_params / physics.param_scales
    # The normalized inputs that flow into the ΔI/F_a MLPs must be bounded near unity;
    # this is the invariant that prevents the O(3,800)-activation NaN cascade.
    assert float(jnp.max(jnp.abs(normalized))) <= 1.0
    assert float(jnp.min(normalized)) > 0.0
