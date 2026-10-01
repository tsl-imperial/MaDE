# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Fixed-interval EKF/RTS kinodynamic smoother, the classical-engineering baseline.

Deliberately does not implement `CorrectionBaseline`. That protocol's unit is
`correct_pair(x_prev, x_curr)` -- one consecutive pair, no memory of the rest of the trajectory.
A fixed-interval smoother's defining property is that the estimate at step k uses observations
from steps *after* k; forcing it through `correct_pair` would silently degrade it to a forward
filter. It instead exposes a trajectory-level entry point, `smooth_trajectory`.

The process model is the kinematic bicycle on both panels -- the same known model MaDE itself is
given. On the simulated panel the data-generating system is the dynamic bicycle, so the smoother
is misspecified exactly as MaDE is; this answers whether classical filtering matches MaDE under
the same misspecification.

It enforces no inequality constraints. Its inequality rate and magnitude are still measured and
reported like every other row.

## The augmented state, and why the controls are in it

The kinematic bicycle's vector field needs a control `(delta, a)` that the smoother is not
given: the input is a predicted state trajectory, nothing more. The controls therefore enter the
filter as part of the estimated state, propagated as a random walk driven by process noise:

    z = (observed state ..., delta, a),    d/dt (delta, a) = 0 + w

so the filter infers the steering and acceleration that best explain the observed positions
under the kinematic model, which is precisely the classical kinodynamic-smoothing construction.
The observed-state block's drift is `physics.vector_field(z_state, z_control, params, t)`, so on
the simulated panel the lateral-velocity and yaw-rate channels drift at zero exactly as
`KinematicBicycleAsDynamicState` says they do, and are carried by process noise alone. That is
the misspecification, made explicit in the filter rather than hidden in it.

The measurement is the observed state block and nothing else: `H = [I 0]`. The controls are
never measured, only inferred.

## Propagation

One **Heun** step, matching `ConstantStepSize()` Heun everywhere else in this project, so the
smoother's notion of "consistent with the known model" is the same one `dynamics_violation`
scores it against. The transition Jacobian is `jax.jacfwd` of that step -- not a hand-derived
analytic Jacobian, which would be a second place for the model to be wrong.

Joseph form is used for the covariance update. The symmetric form costs one extra matrix product
and keeps `P` positive-definite through the ~200 sequential updates a smoothed window needs;
the short form loses symmetry to rounding and this filter runs in a loop long enough to care.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from made.physics.base import PhysicsModel

# The RTS backward gain needs (P_pred)^-1. P_pred is positive-definite by construction, but a
# channel with ~zero process noise and ~zero observed variation can drive it toward singular.
# This ridge is a conditioning guard on a matrix solve, like `det_floor` elsewhere in this
# project -- not a convergence test or tolerance.
_RTS_RIDGE = 1e-12


class KinodynamicSmoother(eqx.Module):
    """Fixed-interval RTS smoother over an EKF with a kinematic-bicycle process model.

    Attributes:
        physics: the KNOWN model -- `KinematicBicycle` (state_dim 4, inD panel) or
            `KinematicBicycleAsDynamicState` (state_dim 6, simulated panel).
        params: known-model parameters, the reference wheelbase as a length-1 array. The same
            `L_REF` the metric stencil uses, so the smoother is scored against the model it was
            run with.
        dt: window timestep. 0.2 on inD, 0.1 on the simulated panel.
        q_diag: process-noise variances, length `state_dim + control_dim`.
        r_diag: measurement-noise variances, length `state_dim`.
        p0_diag: initial covariance diagonal, length `state_dim + control_dim`.
    """

    physics: PhysicsModel
    params: jax.Array
    dt: float
    q_diag: jax.Array
    r_diag: jax.Array
    p0_diag: jax.Array

    @property
    def state_dim(self) -> int:
        """Dimension of the observed state.

        Returns:
            State dimension.
        """
        return int(self.physics.state_dim)

    @property
    def control_dim(self) -> int:
        """Dimension of the control input.

        Returns:
            Control dimension.
        """
        return int(self.physics.control_dim)

    @property
    def aug_dim(self) -> int:
        """Dimension of the augmented state (state plus control).

        Returns:
            Augmented dimension.
        """
        return self.state_dim + self.control_dim

    # -- process model ------------------------------------------------------------------

    def _augmented_field(self, z: jax.Array) -> jax.Array:
        """d/dt of the augmented state. Controls are a random walk: zero drift.

        Args:
            z: Augmented state, shape (aug_dim,).
        Returns:
            Time derivative of the augmented state, shape (aug_dim,).
        """
        n = self.state_dim
        x_dot = self.physics.vector_field(z[:n], z[n:], self.params, 0.0)
        return jnp.concatenate([x_dot, jnp.zeros((self.control_dim,), dtype=z.dtype)])

    def _step(self, z: jax.Array) -> jax.Array:
        """One Heun step of the augmented field, matching the project's solver convention.

        Args:
            z: Augmented state, shape (aug_dim,).
        Returns:
            Augmented state after one step, shape (aug_dim,).
        """
        k1 = self._augmented_field(z)
        k2 = self._augmented_field(z + self.dt * k1)
        return z + 0.5 * self.dt * (k1 + k2)

    def _transition_jacobian(self, z: jax.Array) -> jax.Array:
        """Jacobian of one augmented step with respect to the augmented state.

        Args:
            z: Augmented state, shape (aug_dim,).
        Returns:
            Jacobian, shape (aug_dim, aug_dim).
        """
        return jax.jacfwd(self._step)(z)

    # -- forward pass -------------------------------------------------------------------

    def _filter(self, measurements: jax.Array) -> tuple[jax.Array, ...]:
        """EKF forward pass over `(T, state_dim)` measurements.

        Returns the filtered means and covariances, the one-step-ahead predicted means and
        covariances, and the transition Jacobians -- all five of which the RTS pass needs.

        Args:
            measurements: Observed states, shape (T, state_dim).
        Returns:
            Tuple (filtered means, filtered covariances, predicted means, predicted covariances,
            Jacobians).
        """
        n, m = self.state_dim, self.aug_dim
        q = jnp.diag(self.q_diag)
        r = jnp.diag(self.r_diag)
        h = jnp.concatenate(
            [jnp.eye(n, dtype=measurements.dtype),
             jnp.zeros((n, self.control_dim), dtype=measurements.dtype)], axis=1)
        eye = jnp.eye(m, dtype=measurements.dtype)

        def update(
            z_pred: jax.Array, p_pred: jax.Array, meas: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            """Measurement update in Joseph form.

            Args:
                z_pred: Predicted augmented mean.
                p_pred: Predicted augmented covariance.
                meas: Observed state at this step.
            Returns:
                Tuple (updated mean, updated covariance).
            """
            innovation = meas - h @ z_pred
            s = h @ p_pred @ h.T + r
            gain = jnp.linalg.solve(s.T, (p_pred @ h.T).T).T
            z_up = z_pred + gain @ innovation
            # Joseph form: symmetric, and stays positive-definite over a long window.
            factor = eye - gain @ h
            p_up = factor @ p_pred @ factor.T + gain @ r @ gain.T
            return z_up, p_up

        # The initial control estimate is the known model's own analytic inverse over the first
        # observed pair. It uses the INPUT trajectory only -- no ground truth, no training data
        # -- so it is available at test time and carries no leakage.
        u0 = self.physics.known_control_prior(
            measurements[0], measurements[1], self.params, self.dt)
        z0_pred = jnp.concatenate([measurements[0], u0])
        p0_pred = jnp.diag(self.p0_diag)
        z0, p0 = update(z0_pred, p0_pred, measurements[0])

        def scan_fn(
            carry: tuple[jax.Array, jax.Array], meas: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, ...]]:
            """One forward EKF step: predict, then update with the next measurement.

            Args:
                carry: Previous (mean, covariance).
                meas: Observed state at this step.
            Returns:
                Tuple (new carry, per-step outputs for the RTS pass).
            """
            z_prev, p_prev = carry
            jac = self._transition_jacobian(z_prev)
            z_pred = self._step(z_prev)
            p_pred = jac @ p_prev @ jac.T + q
            z_up, p_up = update(z_pred, p_pred, meas)
            return (z_up, p_up), (z_up, p_up, z_pred, p_pred, jac)

        _, (zf, pf, zp, pp, jacs) = jax.lax.scan(scan_fn, (z0, p0), measurements[1:])
        # Prepend step 0, which has no predecessor to predict from. Its "predicted" slots are
        # its own prior, and its Jacobian slot is never read by the backward pass.
        zf = jnp.concatenate([z0[None], zf])
        pf = jnp.concatenate([p0[None], pf])
        zp = jnp.concatenate([z0_pred[None], zp])
        pp = jnp.concatenate([p0_pred[None], pp])
        jacs = jnp.concatenate([jnp.eye(m, dtype=measurements.dtype)[None], jacs])
        return zf, pf, zp, pp, jacs

    # -- backward pass ------------------------------------------------------------------

    def smooth_trajectory(self, measurements: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Smooth one `(T, state_dim)` trajectory.

        Returns:
            `(x_smoothed, u_smoothed)` -- the smoothed observed-state block, shape
            `(T, state_dim)`, and the inferred controls, shape `(T, control_dim)`.

        **The returned controls are the filter's own estimate and are NOT what the metric
        stencil scores.** Every no-control row on both panels (raw, clamp) is scored on
        pseudo-controls recovered by the panel's own inverse, and the smoother is scored the
        same way. The inferred controls are returned for diagnosis only.

        Args:
            measurements: Observed states, shape (T, state_dim).
        """
        zf, pf, zp, pp, jacs = self._filter(measurements)
        m = self.aug_dim
        ridge = _RTS_RIDGE * jnp.eye(m, dtype=measurements.dtype)

        def scan_fn(
            carry: tuple[jax.Array, jax.Array], xs: tuple[jax.Array, ...]
        ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
            """One backward RTS step.

            Args:
                carry: Smoothed (mean, covariance) of the following step.
                xs: Filtered and predicted quantities for this step.
            Returns:
                Tuple (new carry, smoothed (mean, covariance)).
            """
            z_next_s, p_next_s = carry
            z_k, p_k, z_next_pred, p_next_pred, jac_next = xs
            # C_k = P_k A_{k+1}^T (P^-_{k+1})^{-1}, solved rather than inverted.
            gain = jnp.linalg.solve((p_next_pred + ridge).T, (p_k @ jac_next.T).T).T
            z_s = z_k + gain @ (z_next_s - z_next_pred)
            p_s = p_k + gain @ (p_next_s - p_next_pred) @ gain.T
            return (z_s, p_s), (z_s, p_s)

        # Run backward over steps 0..T-2, carrying the already-smoothed step T-1.
        _, (zs, _) = jax.lax.scan(
            scan_fn, (zf[-1], pf[-1]),
            (zf[:-1], pf[:-1], zp[1:], pp[1:], jacs[1:]),
            reverse=True,
        )
        zs = jnp.concatenate([zs, zf[-1][None]])
        n = self.state_dim
        return zs[:, :n], zs[:, n:]


def smooth_batch(
    smoother: KinodynamicSmoother,
    measurements: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """`smooth_trajectory` vmapped over a `(B, T, state_dim)` batch of windows.

    Args:
        smoother: Configured smoother.
        measurements: Batch of windows, shape (B, T, state_dim).
    Returns:
        Tuple (x_smoothed, u_smoothed) with a leading batch dimension.
    """
    return jax.vmap(smoother.smooth_trajectory)(measurements)
