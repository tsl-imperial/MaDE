"""Clamp-only baseline."""

import jax
import jax.numpy as jnp
import equinox as eqx

from made.physics.constraints import BoxConstraints


class ClampBaseline(eqx.Module):
    """Inequality-only baseline that clips consecutive state pairs to box bounds.

    Returns (clamped_x_prev, clamped_x_curr), consistent with the CorrectionBaseline
    protocol where both elements are state-dim arrays representing the corrected pair.
    """

    constraints: BoxConstraints

    def correct_pair(
        self,
        x_prev: jax.Array,
        x_curr: jax.Array,
        metadata: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Return (clamped_x_prev, clamped_x_curr) clipped to state box bounds."""
        del metadata
        return (
            jnp.clip(x_prev, self.constraints.state_min, self.constraints.state_max),
            jnp.clip(x_curr, self.constraints.state_min, self.constraints.state_max),
        )


def clamp_baseline(
    state: jax.Array,
    control: jax.Array,
    constraints: BoxConstraints,
) -> tuple[jax.Array, jax.Array]:
    """Clamp state and control to the box bounds."""
    state_clipped = jnp.clip(state, constraints.state_min, constraints.state_max)
    control_clipped = jnp.clip(control, constraints.control_min, constraints.control_max)
    return state_clipped, control_clipped
