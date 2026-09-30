"""Augmented dynamics model."""

from __future__ import annotations

import os

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from made.physics import PhysicsModel


class ResidualNetwork(eqx.Module):
    """Minimum-norm residual vector field."""

    mlp: eqx.nn.MLP

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        param_dim: int,
        hidden: tuple[int, ...],
        *,
        init_scale: float = 0.0,
        key: jax.Array,
    ):
        width = hidden[0] if hidden else max(state_dim, 1)
        depth = len(hidden)
        self.mlp = eqx.nn.MLP(
            in_size=state_dim + control_dim + param_dim,
            out_size=state_dim,
            width_size=width,
            depth=depth,
            activation=jax.nn.relu,
            key=key,
        )
        if init_scale != 1.0:
            final = self.mlp.layers[-1]
            scaled_weight = final.weight * init_scale
            scaled_bias = None if final.bias is None else final.bias * init_scale
            self.mlp = eqx.tree_at(
                lambda mlp: (mlp.layers[-1].weight, mlp.layers[-1].bias),
                self.mlp,
                (scaled_weight, scaled_bias),
            )

    def __call__(
        self,
        state: jax.Array,
        control: jax.Array,
        params: jax.Array,
        t: float,
    ) -> jax.Array:
        del t
        inputs = jnp.concatenate([state, control, params])
        return self.mlp(inputs)


class ZeroResidual(eqx.Module):
    """Residual variant that preserves known-physics dynamics exactly."""

    state_dim: int

    def __call__(
        self,
        state: jax.Array,
        control: jax.Array,
        params: jax.Array,
        t: float,
    ) -> jax.Array:
        del control, params, t
        return jnp.zeros((self.state_dim,), dtype=state.dtype)


def _fast_heun_enabled() -> bool:
    """Whether the single-step Heun fast path in ``integrate`` is active.

    Default on. Set ``MADE_FAST_HEUN=0`` to fall back to driving ``diffeqsolve`` for the
    single-step case. Read per call rather than cached so a test can toggle it; the read is a
    dict lookup and is not on the device path.

    Agreement with the ``diffeqsolve`` path is at machine epsilon (forward 1.1e-16, gradient
    5.6e-17 abs / 6.9e-16 rel), mathematically identical but not bit-identical.
    """
    return os.environ.get("MADE_FAST_HEUN", "1") != "0"


class AugmentedDynamics(eqx.Module):
    """Known physics plus a learned residual."""

    physics: PhysicsModel
    residual: eqx.Module

    def vector_field(
        self,
        state: jax.Array,
        control: jax.Array,
        params: jax.Array,
        t: float,
    ) -> jax.Array:
        params_norm = (
            params / self.physics.param_scales
            if self.physics.param_dim > 0
            else params
        )
        return self.physics.vector_field(state, control, params, t) + self.residual(
            state,
            control,
            params_norm,
            t,
        )

    def integrate(
        self,
        x_prev: jax.Array,
        control: jax.Array,
        params: jax.Array,
        dt: float,
        *,
        solver: diffrax.AbstractSolver | None = None,
        adjoint: diffrax.AbstractAdjoint | None = None,
        fast: bool | None = None,
    ) -> jax.Array:
        # Fast path: the default config is Heun() + ConstantStepSize() over t0=0 -> t1=dt with
        # dt0=dt, i.e. exactly one Heun step -- the explicit trapezoid k1=f(y,u,0),
        # k2=f(y+dt*k1,u,dt), y' = y + dt/2*(k1+k2). Running diffeqsolve's stepping/controller/
        # save machinery to evaluate the same two vector-field calls costs ~28x the arithmetic
        # on the differentiated path. DirectAdjoint is admitted here because it means
        # "differentiate through the solver's operations", exactly what autodiff through this
        # expression does. Mathematically identical, not bit-identical (see
        # `_fast_heun_enabled`); guarded by MADE_FAST_HEUN.
        use_fast = _fast_heun_enabled() if fast is None else fast
        if (
            use_fast
            and solver is None
            and (adjoint is None or isinstance(adjoint, diffrax.DirectAdjoint))
        ):
            k1 = self.vector_field(x_prev, control, params, 0.0)
            k2 = self.vector_field(x_prev + dt * k1, control, params, dt)
            return x_prev + 0.5 * dt * (k1 + k2)

        term = diffrax.ODETerm(
            lambda t, y, args: self.vector_field(y, args["control"], args["params"], t)
        )
        _solver = solver or diffrax.Heun()
        extra: dict = {}
        if solver is None:
            # Heun() is an embedded 2(1) method; without an explicit controller diffrax
            # uses adaptive stepping, which exhausts max_steps on stiff dynamics.
            # Force fixed-step so the training integrator always takes exactly 1 step.
            extra["stepsize_controller"] = diffrax.ConstantStepSize()
        solution = diffrax.diffeqsolve(
            term,
            solver=_solver,
            t0=0.0,
            t1=dt,
            dt0=dt,
            y0=x_prev,
            args={"control": control, "params": params},
            saveat=diffrax.SaveAt(t1=True),
            adjoint=adjoint or diffrax.RecursiveCheckpointAdjoint(),
            max_steps=16384,
            **extra,
        )
        return solution.ys[0]

    def residual_norm(
        self,
        state: jax.Array,
        control: jax.Array,
        params: jax.Array,
    ) -> jax.Array:
        residual = self.residual(state, control, params, 0.0)
        return jnp.sum(residual ** 2)
