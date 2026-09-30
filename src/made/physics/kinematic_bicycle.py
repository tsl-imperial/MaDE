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
        # Layer 1 (defense-in-depth): clamp δ only inside tan() to avoid
        # 1/cos²(δ) blowing up near ±π/2. Inactive for |δ| ≤ 1.4 rad
        # (≈80°), well outside the constraint box |δ| ≤ δ_max=0.5.
        # Load-bearing during Phase-2 training where forward_consistency_loss
        # differentiates through T.integrate → vector_field → tan(I-output δ)
        # BEFORE the corrector ever runs, so the AD path through I-output δ
        # near π/2 needs this clamp regardless of corrector projection.
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
        # Heun integration of theta_dot = v*tan(delta)/L under ZOH constant (delta, a)
        # gives theta_curr = theta_prev + dt * v_avg * tan(delta) / L exactly: v is
        # linear in time and tan(delta) is constant, so Heun reproduces the analytic
        # integral.  Hence v_avg = 1/2 * (v_prev + v_curr) is the exact-against-Heun
        # inverse, and accel = (v_curr - v_prev) / dt is already exact under ZOH.
        #
        # This is the ANALYTIC inverse, and it is what the solver appendix describes:
        # delta = arctan(L * dtheta / (v_avg * dt)) with a 1e-6 floor on |v_avg| to
        # bound the denominator near the singularity. It is correct wherever the
        # heading signal is clean, which on simulated data it always is.
        #
        # The option-D stationary fallback (delta := 0 below 0.5 m/s) lives in
        # KinematicBicycleFieldData below and applies to FIELD DATA ONLY.
        # It was introduced because sensor jitter in a recorded heading turns the arctan
        # singularity into spurious near-±π/2 steering. Simulated data has no such jitter, so
        # a guard against jitter does not belong there -- that premise is the whole reason for
        # the split and it stands on its own.
        #
        # It does not belong here for that reason ALONE, but the fallback IS in fact what moved
        # the simulated kinematic-bicycle numbers, established by bisection measurement:
        # all 50 E01 kinematic-bicycle runs match one of the two conventions at
        # step 1 to 4.9e-14, and the runs that moved are exactly the ones that ran under the
        # fallback. An intermediate version of this comment called that DISPROVED, on a
        # revert-and-rerun test that was performed after the revert had already landed and so
        # reverted nothing.
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

    The fallback applies to FIELD DATA ONLY. It is a *convention* rather than a
    tuning parameter, which is why it is a separate class rather than a threshold argument —
    the same reasoning commit e82c616 used when it deliberately left the trajectory-major
    helper ``kinematic_bicycle_inverse_controls`` on its own carry-forward convention.

    Three conventions now coexist deliberately, each correct in its own regime:

    - ``KinematicBicycle``           : analytic arctan. Clean heading, no temporal context.
    - ``KinematicBicycleFieldData``  : delta = 0 below 0.5 m/s. Jittery heading, no context.
    - ``kinematic_bicycle_inverse_controls`` : carry forward the previous delta. Context.

    Measured reach on the filtered inD training set: 14.10% of transitions fall below the
    threshold, so this is load-bearing where it applies, not decorative
    (``tables/fallback_reach_filtered_inD.json``).

    Note for anyone tracing history: this class exists because a jitter guard does not belong
    on jitter-free simulated data. Separately, the fallback *is* what moved the simulated
    kinematic-bicycle numbers -- established by bisection measurement, after an
    intermediate test wrongly reported it disproved.
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
