"""Kinematic bicycle dynamics."""

import jax
import jax.numpy as jnp

from made.physics.base import PhysicsModel


class KinematicBicycle(PhysicsModel):
    """Kinematic bicycle with a learnable wheelbase parameter."""

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def control_dim(self) -> int:
        return 2

    @property
    def param_dim(self) -> int:
        return 1

    @property
    def param_scales(self) -> jax.Array:
        return jnp.array([3.0])

    def vector_field(
        self,
        state: jax.Array,
        control: jax.Array,
        params: jax.Array,
        t: float,
    ) -> jax.Array:
        del t
        theta = state[2]
        velocity = state[3]
        delta = control[0]
        accel = control[1]
        wheelbase = jnp.maximum(params[0], 1e-6)
        # Clamp delta only inside tan() to avoid 1/cos²(δ) blowing up near ±π/2. Inactive
        # for |δ| ≤ 1.4 rad (≈80°), well outside the constraint box |δ| ≤ δ_max=0.5. Needed
        # during Phase-2 training: forward_consistency_loss differentiates through
        # T.integrate -> vector_field -> tan(I-output δ) before the corrector runs.
        tan_arg = jnp.clip(delta, -1.4, 1.4)
        return jnp.array(
            [
                velocity * jnp.cos(theta),
                velocity * jnp.sin(theta),
                velocity * jnp.tan(tan_arg) / wheelbase,  # only tan() sees clamped delta
                accel,
            ],
            dtype=state.dtype,
        )

    def known_control_prior(
        self,
        x_prev: jax.Array,
        x_curr: jax.Array,
        params: jax.Array,
        dt: float,
    ) -> jax.Array:
        # Heun integration of theta_dot = v*tan(delta)/L under ZOH constant (delta, a) gives
        # theta_curr = theta_prev + dt * v_avg * tan(delta) / L exactly (v linear in time,
        # tan(delta) constant), so v_avg = 1/2*(v_prev + v_curr) is the exact-against-Heun
        # inverse and accel = (v_curr - v_prev) / dt is exact under ZOH. Analytic inverse:
        # delta = arctan(L * dtheta / (v_avg * dt)), with a 1e-6 floor on |v_avg| to bound
        # the denominator near the singularity. Correct wherever the heading signal is clean
        # (always true on simulated data).
        #
        # The option-D stationary fallback (delta := 0 below 0.5 m/s) lives in
        # KinematicBicycleFieldData below, for FIELD DATA ONLY: sensor jitter in a recorded
        # heading turns the arctan singularity into spurious near-±π/2 steering, which
        # simulated data doesn't have.
        wheelbase = jnp.maximum(params[0], 1e-6)
        dtheta = x_curr[2] - x_prev[2]
        v_avg_raw = 0.5 * (x_prev[3] + x_curr[3])
        v_avg = jnp.where(
            jnp.abs(v_avg_raw) > 1e-6, v_avg_raw, jnp.asarray(1e-6, x_prev.dtype)
        )
        delta = jnp.arctan(wheelbase * dtheta / (v_avg * dt))
        accel = (x_curr[3] - x_prev[3]) / dt
        return jnp.array([delta, accel], dtype=x_prev.dtype)


class KinematicBicycleFieldData(KinematicBicycle):
    """Kinematic bicycle whose control prior carries the option-D stationary fallback.

    Applies to FIELD DATA ONLY. A separate class rather than a threshold argument, since
    it's a convention, not a tuning parameter.

    Three conventions coexist, each correct in its own regime:

    - ``KinematicBicycle``: analytic arctan. Clean heading, no temporal context.
    - ``KinematicBicycleFieldData``: delta = 0 below 0.5 m/s. Jittery heading, no context.
    - ``kinematic_bicycle_inverse_controls``: carry forward the previous delta. Context.

    Measured reach on the filtered inD training set: 14.10% of transitions fall below the
    threshold (``tables/fallback_reach_filtered_inD.json``).
    """

    STATIONARY_SPEED_MS: float = 0.5

    def known_control_prior(
        self,
        x_prev: jax.Array,
        x_curr: jax.Array,
        params: jax.Array,
        dt: float,
    ) -> jax.Array:
        u = super().known_control_prior(x_prev, x_curr, params, dt)
        v_avg_raw = 0.5 * (x_prev[3] + x_curr[3])
        is_stationary = jnp.abs(v_avg_raw) < jnp.asarray(
            self.STATIONARY_SPEED_MS, x_prev.dtype
        )
        delta = jnp.where(is_stationary, jnp.asarray(0.0, x_prev.dtype), u[0])
        return jnp.array([delta, u[1]], dtype=x_prev.dtype)
