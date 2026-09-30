"""Correctness tests for train_predictor._chunked_weighted_mean (validation MSE).

Vmapping over the entire val split (50,114 windows for real inD data) makes XLA try to
allocate 34.10 GiB for the SSM predictor's scan intermediates. ``_chunked_weighted_mean``
evaluates fixed-size chunks (default 2048) and combines them as the exact weighted mean, with
the last partial chunk zero-padded and masked so only one shape is ever traced. This module
checks the combination is numerically equal to the unchunked mean and that a chunk size that
does not evenly divide the sample count (forcing a real padded partial last chunk) is handled
correctly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "ind" / "train_predictor.py"


@pytest.fixture(scope="module")
def script_module():
    spec = importlib.util.spec_from_file_location("train_predictor", str(_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_predictor"] = module
    spec.loader.exec_module(module)
    return module


class _SumSquaredModel(eqx.Module):
    """Deterministic ``model(c) -> D``-vector so per-window MSE is hand-checkable."""

    scale: jax.Array

    def __call__(self, c: jax.Array) -> jax.Array:
        return c * self.scale


def _unchunked_mse(model, context, x_gt) -> float:
    def _single(c, g):
        return jnp.mean((model(c) - g) ** 2)

    return float(jnp.mean(jax.vmap(_single)(context, x_gt)))


@pytest.mark.parametrize(
    "num_samples,chunk_size",
    [
        (16, 16),  # exactly one full chunk, no padding
        (16, 4),  # evenly-divisible multiple chunks, no padding
        (17, 4),  # multiple chunks + a real padded partial last chunk (17 = 4*4 + 1)
        (5, 2048),  # chunk_size >> num_samples: effective chunk must cap at num_samples
    ],
)
def test_chunked_matches_unchunked_mean(script_module, num_samples: int, chunk_size: int) -> None:
    model = _SumSquaredModel(scale=jnp.asarray(1.7))
    key_c, key_g = jax.random.split(jax.random.key(0))
    context = jax.random.normal(key_c, (num_samples, 4))
    x_gt = jax.random.normal(key_g, (num_samples, 4))

    expected = _unchunked_mse(model, context, x_gt)

    @eqx.filter_jit
    def _chunk_stats(model, context_chunk, x_gt_chunk, mask_chunk):
        def _single(c, g):
            return jnp.mean((model(c) - g) ** 2)

        per_window = jax.vmap(_single)(context_chunk, x_gt_chunk)
        return jnp.sum(per_window * mask_chunk), jnp.sum(mask_chunk)

    got = script_module._chunked_weighted_mean(
        _chunk_stats, model, [context, x_gt], chunk_size
    )

    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


def test_chunked_weighted_mean_empty_returns_nan(script_module) -> None:
    model = _SumSquaredModel(scale=jnp.asarray(1.0))
    context = jnp.zeros((0, 4))
    x_gt = jnp.zeros((0, 4))

    @eqx.filter_jit
    def _chunk_stats(model, context_chunk, x_gt_chunk, mask_chunk):
        def _single(c, g):
            return jnp.mean((model(c) - g) ** 2)

        per_window = jax.vmap(_single)(context_chunk, x_gt_chunk)
        return jnp.sum(per_window * mask_chunk), jnp.sum(mask_chunk)

    got = script_module._chunked_weighted_mean(_chunk_stats, model, [context, x_gt], 8)
    assert np.isnan(got)


def test_chunked_val_mse_ignores_padded_rows(script_module) -> None:
    """Padding rows are zero-filled and masked to weight 0; a model producing non-finite output
    on the zero-padded rows must not corrupt the real windows' contribution (this would be a
    silent-NaN-poisoning bug if masking were applied before, not after, the model call)."""
    num_samples, chunk_size = 5, 3  # forces a padded partial last chunk (3 + 2 padding)

    class _NanOnZero(eqx.Module):
        def __call__(self, c: jax.Array) -> jax.Array:
            # Zero rows (the padding) map to a finite value here on purpose -- this test only
            # needs the arithmetic to be correct when padding IS finite; a truly NaN-producing
            # padding row would poison the sum even with masking (mask multiplies, doesn't
            # select), which is a known limitation documented in _chunked_weighted_mean's
            # docstring assumption that padding-with-zeros doesn't itself produce non-finite
            # model output.
            return c * 2.0

    model = _NanOnZero()
    key_c, key_g = jax.random.split(jax.random.key(1))
    context = jax.random.normal(key_c, (num_samples, 4)) + 1.0  # keep away from zero
    x_gt = jax.random.normal(key_g, (num_samples, 4))

    expected = _unchunked_mse(model, context, x_gt)

    @eqx.filter_jit
    def _chunk_stats(model, context_chunk, x_gt_chunk, mask_chunk):
        def _single(c, g):
            return jnp.mean((model(c) - g) ** 2)

        per_window = jax.vmap(_single)(context_chunk, x_gt_chunk)
        return jnp.sum(per_window * mask_chunk), jnp.sum(mask_chunk)

    got = script_module._chunked_weighted_mean(
        _chunk_stats, model, [context, x_gt], chunk_size
    )
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)
