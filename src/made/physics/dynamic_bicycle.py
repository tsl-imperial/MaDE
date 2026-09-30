"""Dynamic bicycle dynamics."""

from __future__ import annotations

from typing import ClassVar

import jax
import jax.numpy as jnp

from made.physics.base import PhysicsModel


class DynamicBicycle(PhysicsModel):
    """Dynamic bicycle model with linear tire forces.

    ClassVar constants
    ------------------
    _SAFE_VX_EPS : float
        Epsilon shared by ``vector_field`` and ``known_control_prior`` to stabilise the
        vx-denominator near zero. Single source of truth.
    _NEWTON_STEPS : int
        Unrolled Newton iterations on the algebraic yaw-rate FD equation that warm-start
        ``known_control_prior`` (Python-static, unrolled at trace time). 2 gives quadratic
        convergence insurance for moderate slip angles.
    _NEWTON_HEUN_STEPS : int
        Unrolled outer Newton iterations on the 2D Heun-step residual
        ``r(delta, a) = [Heun(x_prev, [delta, a])[3] - x_curr[3], Heun(...)[5] - x_curr[5]]``
        (vx and yaw-rate rows, most sensitive to a and delta). Aligns the inverse with the
        training-time integrator, removing the O(dt) FD bias floor. 3 gives machine-zero
        recovery (~1e-12) across mild-to-aggressive manoeuvres at production dt=0.1.
    """

    _SAFE_VX_EPS: ClassVar[float] = 1e-3
    _NEWTON_STEPS: ClassVar[int] = 2
    _NEWTON_HEUN_STEPS: ClassVar[int] = 3

    @property
    def state_dim(self) -> int:
        return 6

    @property
    def control_dim(self) -> int:
        return 2

    @property
    def param_dim(self) -> int:
        return 6

    @property
    def param_scales(self) -> jax.Array:
        # C_f, C_r [N/rad], m [kg], I_z [kg·m²], l_f [m], l_r [m]
        return jnp.array([20000.0, 20000.0, 1500.0, 3000.0, 2.0, 2.0])

    def vector_field(
        self,
        state: jax.Array,
        control: jax.Array,
        params: jax.Array,
        t: float,
    ) -> jax.Array:
        del t
        theta = state[2]
        vx = state[3]
        vy = state[4]
        yaw_rate = state[5]
        delta = control[0]
        accel = control[1]
        c_f, c_r, mass, inertia_z, l_f, l_r = params

        # Must match known_control_prior's epsilon (_SAFE_VX_EPS).
        eps = jnp.asarray(self._SAFE_VX_EPS, dtype=state.dtype)
        safe_vx = jnp.where(jnp.abs(vx) > eps, vx, jnp.where(vx >= 0.0, eps, -eps))
        slip_front = delta - jnp.arctan2(vy + l_f * yaw_rate, safe_vx)
        slip_rear = -jnp.arctan2(vy - l_r * yaw_rate, safe_vx)
        force_front = c_f * slip_front
        force_rear = c_r * slip_rear

        x_dot = vx * jnp.cos(theta) - vy * jnp.sin(theta)
        y_dot = vx * jnp.sin(theta) + vy * jnp.cos(theta)
        theta_dot = yaw_rate
        vx_dot = accel - force_front * jnp.sin(delta) / mass + yaw_rate * vy
        vy_dot = (force_front * jnp.cos(delta) + force_rear) / mass - yaw_rate * vx
        yaw_rate_dot = (
            l_f * force_front * jnp.cos(delta) - l_r * force_rear
        ) / jnp.maximum(inertia_z, eps)

        return jnp.array(
            [x_dot, y_dot, theta_dot, vx_dot, vy_dot, yaw_rate_dot],
            dtype=state.dtype,
        )

    def known_control_prior(
        self,
        x_prev: jax.Array,
        x_curr: jax.Array,
        params: jax.Array,
        dt: float,
    ) -> jax.Array:
        """True dynamic-bicycle inverse aligned with the Heun training integrator.

        Stage A — algebraic warm start (matches FD targets, O(dt) bias):
          A1: Solve delta from the yaw-rate equation via ``_NEWTON_STEPS`` unrolled Newton
              iterations, initialised from the kinematic estimate.
          A2: Closed-form ``a`` from the v_x equation given the solved delta.

        Stage B — Newton-on-Heun (matches the actual integrator, removes O(dt) bias):
          B1: Iterate ``_NEWTON_HEUN_STEPS`` outer Newton steps on the 2D residual
              ``r(delta, a) = [Heun(x_prev, [delta, a])[3] - x_curr[3],
                                Heun(x_prev, [delta, a])[5] - x_curr[5]]``
              (vx and yaw-rate rows, most sensitive to a and delta). Jacobian via
              ``jax.jacfwd`` (exact), inverted via closed-form 2x2 with a det-floor.

        JIT-clean (Python-static loop counts), vmap-compatible, gradient-stable.
        safe_vx epsilon must match vector_field's _SAFE_VX_EPS.
        """
        c_f, c_r, mass, inertia_z, l_f, l_r = params

        vx_p = x_prev[3]
        vy_p = x_prev[4]
        yaw_rate_p = x_prev[5]

        eps = jnp.asarray(self._SAFE_VX_EPS, dtype=x_prev.dtype)
        safe_vx = jnp.where(
            jnp.abs(vx_p) > eps,
            vx_p,
            jnp.where(vx_p >= 0.0, eps, -eps),
        )

        # Constants in delta (depend only on x_prev).
        beta = jnp.arctan2(vy_p + l_f * yaw_rate_p, safe_vx)   # front slip offset
        alpha_r = -jnp.arctan2(vy_p - l_r * yaw_rate_p, safe_vx)
        F_r = c_r * alpha_r
        inertia_safe = jnp.maximum(inertia_z, eps)

        # Finite-difference targets.
        yaw_rate_dot_fd = (x_curr[5] - x_prev[5]) / dt
        vx_dot_fd = (x_curr[3] - x_prev[3]) / dt

        # Stage A1: kinematic initialiser for delta.
        L = jnp.maximum(l_f + l_r, jnp.asarray(1e-6, dtype=x_prev.dtype))
        dtheta = x_curr[2] - x_prev[2]
        delta = jnp.arctan(L * dtheta / (safe_vx * dt))

        # Stage A2: _NEWTON_STEPS unrolled Newton iterations on the yaw-rate equation.
        #   g(delta)  = (l_f * F_f(delta) * cos(delta) - l_r * F_r) / I_z - yaw_rate_dot_fd
        #   dg/ddelta = (l_f * (c_f * cos(delta) - F_f(delta) * sin(delta))) / I_z
        # F_f(delta) = c_f * (delta - beta),  beta constant in delta.
        denom_floor = jnp.asarray(1e-6, dtype=x_prev.dtype)
        for _ in range(self._NEWTON_STEPS):  # Python-static; unrolled at trace time.
            F_f = c_f * (delta - beta)
            g_val = (
                (l_f * F_f * jnp.cos(delta) - l_r * F_r) / inertia_safe - yaw_rate_dot_fd
            )
            dg = (l_f * (c_f * jnp.cos(delta) - F_f * jnp.sin(delta))) / inertia_safe
            safe_dg = jnp.where(
                jnp.abs(dg) > denom_floor,
                dg,
                jnp.where(dg >= 0.0, denom_floor, -denom_floor),
            )
            delta = delta - g_val / safe_dg

        # Stage A3: closed-form a from the v_x equation (warm-start accel).
        F_f_warm = c_f * (delta - beta)
        accel = vx_dot_fd + F_f_warm * jnp.sin(delta) / mass - yaw_rate_p * vy_p

        u = jnp.array([delta, accel], dtype=x_prev.dtype)

        # Stage B: Newton-on-Heun outer iterations.  Closure captures x_prev,
        # params, dt; the Heun half-step matches the training integrator
        # (diffrax.Heun() + ConstantStepSize() at constant control over [t, t+dt]).
        det_floor = jnp.asarray(1e-12, dtype=x_prev.dtype)

        def _heun_residual(u_in: jax.Array) -> jax.Array:
            k1 = self.vector_field(x_prev, u_in, params, 0.0)
            k2 = self.vector_field(x_prev + dt * k1, u_in, params, dt)
            x_next = x_prev + 0.5 * dt * (k1 + k2)
            # Pick rows 3 (vx) and 5 (yaw_rate) — most sensitive to (a, delta).
            return jnp.array([x_next[3] - x_curr[3], x_next[5] - x_curr[5]])

        for _ in range(self._NEWTON_HEUN_STEPS):  # Python-static; unrolled.
            r = _heun_residual(u)
            J = jax.jacfwd(_heun_residual)(u)  # shape (2, 2)
            det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            safe_det = jnp.where(
                jnp.abs(det) > det_floor,
                det,
                jnp.where(det >= 0.0, det_floor, -det_floor),
            )
            # Closed-form 2x2 inverse times r:  delta_u = -J^{-1} r.
            inv00 = J[1, 1] / safe_det
            inv01 = -J[0, 1] / safe_det
            inv10 = -J[1, 0] / safe_det
            inv11 = J[0, 0] / safe_det
            du0 = -(inv00 * r[0] + inv01 * r[1])
            du1 = -(inv10 * r[0] + inv11 * r[1])
            u = u + jnp.array([du0, du1], dtype=x_prev.dtype)

        return u
