"""MADE_FAST_HEUN must be a pure speed optimisation: same maths, machine-epsilon agreement.

The default configuration integrates with Heun() + ConstantStepSize() from t0=0 to
t1=dt with dt0=dt -- exactly ONE step, i.e. the explicit trapezoid. The fast path
evaluates that expression directly instead of driving diffeqsolve's stepping machinery.
DirectAdjoint is admitted because it means "differentiate through the solver's
operations", which is what autodiff through the expression does.

These tests pin BOTH directions: agreement to machine epsilon, and that the flag is
OFF by default; MADE_FAST_HEUN=1 opts in.
"""

from __future__ import annotations

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from made.models import augmented_dynamics as AD
from made.physics import build_system_for_model

TOL = 1e-14  # machine-epsilon band; observed 1.1e-16 forward, 5.6e-17 gradient
DT = 0.2


@pytest.fixture
def dynamics():
    physics, _ = build_system_for_model("kinematic_bicycle", "kinematic_bicycle")
    residual = AD.ResidualNetwork(
        state_dim=4, control_dim=2, param_dim=physics.param_dim,
        hidden=(16, 16), key=jax.random.key(0), init_scale=0.1,
    )
    return AD.AugmentedDynamics(physics=physics, residual=residual)


@pytest.fixture
def batch():
    k = jax.random.split(jax.random.key(1), 2)
    x = jax.random.normal(k[0], (8, 4), dtype=jnp.float64)
    u = 0.05 * jax.random.normal(k[1], (8, 2), dtype=jnp.float64)
    p = jnp.broadcast_to(jnp.asarray([2.7], dtype=jnp.float64), (8, 1))
    return x, u, p


def _with_flag(value, fn):
    prev = os.environ.get("MADE_FAST_HEUN")
    os.environ["MADE_FAST_HEUN"] = value
    try:
        return fn()
    finally:
        if prev is None:
            os.environ.pop("MADE_FAST_HEUN", None)
        else:
            os.environ["MADE_FAST_HEUN"] = prev


def test_flag_is_on_by_default():
    """Default ON for training and evaluation alike; MADE_FAST_HEUN=0 is the documented
    fallback and what the jaxpr structural guards pin."""
    os.environ.pop("MADE_FAST_HEUN", None)
    assert AD._fast_heun_enabled() is True
    assert _with_flag("0", AD._fast_heun_enabled) is False


def test_forward_agrees_to_machine_epsilon(dynamics, batch):
    x, u, p = batch
    run = lambda: jax.vmap(lambda a, b, c: dynamics.integrate(a, b, c, DT))(x, u, p)
    slow = _with_flag("0", run)
    fast = _with_flag("1", run)
    assert float(jnp.max(jnp.abs(slow - fast))) < TOL


def test_gradient_agrees_to_machine_epsilon(dynamics, batch):
    """The differentiated path is the one training actually uses."""
    x, u, p = batch
    grad = lambda: jax.grad(
        lambda uu: jnp.sum(jax.vmap(lambda a, b, c: dynamics.integrate(a, b, c, DT))(x, uu, p))
    )(u)
    slow = _with_flag("0", grad)
    fast = _with_flag("1", grad)
    assert float(jnp.max(jnp.abs(slow - fast))) < TOL


def test_direct_adjoint_takes_the_fast_path(dynamics, batch):
    """The corrector passes DirectAdjoint explicitly; that must still be accelerated."""
    import diffrax
    x, u, p = batch
    run = lambda: jax.vmap(
        lambda a, b, c: dynamics.integrate(a, b, c, DT, adjoint=diffrax.DirectAdjoint())
    )(x, u, p)
    slow = _with_flag("0", run)
    fast = _with_flag("1", run)
    assert float(jnp.max(jnp.abs(slow - fast))) < TOL


def test_explicit_non_default_solver_still_uses_diffrax(dynamics, batch):
    """A caller asking for a different solver must NOT be silently given Heun."""
    import diffrax
    x, u, p = batch
    run = lambda: jax.vmap(
        lambda a, b, c: dynamics.integrate(a, b, c, DT, solver=diffrax.Tsit5())
    )(x, u, p)
    assert float(jnp.max(jnp.abs(_with_flag("0", run) - _with_flag("1", run)))) == 0.0
