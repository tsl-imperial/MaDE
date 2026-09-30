"""Variant/baseline dispatch tests (US-017)."""

# ruff: noqa: E402

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from made.models import MaDECell
from made.training.dispatch import build_trainable
from made.utils.config import CorrectorConfig, ExperimentConfig, ModelConfig, PhysicsConfig


_SMALL_MODEL = ModelConfig(inverse_hidden=(16, 16), residual_hidden=(16, 16))
_SMALL_CORRECTOR = CorrectorConfig(mode="enabled", train_steps=2)
_DI_CFG = ExperimentConfig(
    physics=PhysicsConfig(true_system="double_integrator"),
    model=_SMALL_MODEL,
    corrector=_SMALL_CORRECTOR,
)


@pytest.fixture
def key():
    return jax.random.key(0)


@pytest.mark.parametrize(
    "variant",
    ["made", "made-no-residual", "made-no-corrector", "made-supervised-i", "made-fixed-i", "mlp", "fab", "clamp"],
)
def test_build_trainable_type(variant, key):
    model = build_trainable(_DI_CFG, variant, key)

    if variant == "clamp":
        assert model is None, "clamp should return None"
        return

    assert isinstance(model, eqx.Module), f"{variant} should return an eqx.Module"


@pytest.mark.parametrize(
    "variant",
    ["made", "made-no-residual", "made-no-corrector", "made-supervised-i", "made-fixed-i", "mlp", "fab"],
)
def test_build_trainable_finite_params(variant, key):
    """All trainable parameters should be finite after initialisation."""
    model = build_trainable(_DI_CFG, variant, key)
    assert model is not None
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    assert len(leaves) > 0, f"{variant} has no learnable parameters"
    for leaf in leaves:
        assert jnp.all(jnp.isfinite(leaf)), f"{variant} has non-finite parameter"


def test_made_no_residual_is_zero_residual(key):
    """made-no-residual should have residual='zero' in the config, expressed as ZeroResidual."""
    from made.models.augmented_dynamics import ZeroResidual
    model = build_trainable(_DI_CFG, "made-no-residual", key)
    assert isinstance(model, MaDECell)
    assert isinstance(model.augmented_dynamics.residual, ZeroResidual)


def test_made_no_corrector_is_disabled(key):
    """made-no-corrector should have corrector_mode='disabled'."""
    model = build_trainable(_DI_CFG, "made-no-corrector", key)
    assert isinstance(model, MaDECell)
    assert model.corrector_mode == "disabled"


def test_unknown_variant_raises(key):
    with pytest.raises(ValueError, match="Unknown variant"):
        build_trainable(_DI_CFG, "not-a-variant", key)
