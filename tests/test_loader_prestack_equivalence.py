"""Pre-stacked batching must be a pure marshalling change: identical batches, faster.

The loader previously called jnp.asarray per sample per key per batch and stacked the
results. Pre-stacking into contiguous numpy columns and fancy-indexing produces the same
values in the same order; these tests pin that, including the noise path and the
shuffle/drop_remainder behaviour that determine WHICH samples land in WHICH batch.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from made.data.grain_pipeline import InMemoryDataLoader


def _samples(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {
            "x_prev": rng.standard_normal(4),
            "x_curr": rng.standard_normal(4),
            "params": np.array([2.7]),
        }
        for _ in range(n)
    ]


def _reference(samples, batch_size, *, shuffle, seed, drop_remainder, noise_scale, epochs=1):
    """The pre-change implementation, inlined, as the oracle."""
    _NOISE_SALT = __import__("made.data.grain_pipeline", fromlist=["_NOISE_SALT"])._NOISE_SALT
    out = []
    for epoch in range(epochs):
        indices = np.arange(len(samples))
        if shuffle:
            np.random.default_rng(seed).shuffle(indices)
        noise_rng = (
            np.random.default_rng(np.array([seed, epoch, _NOISE_SALT], dtype=np.uint64))
            if noise_scale > 0
            else None
        )
        stop = len(indices) - batch_size + 1 if drop_remainder else len(indices)
        for start in range(0, stop, batch_size):
            bi = indices[start : start + batch_size]
            if len(bi) == 0:
                continue
            bs = [samples[int(i)] for i in bi]
            batch = {
                k: jnp.stack([jnp.asarray(s[k]) for s in bs], axis=0) for k in bs[0]
            }
            if noise_rng is not None:
                for k in ("x_prev", "x_curr"):
                    if k in batch:
                        noise = noise_rng.standard_normal(batch[k].shape).astype(np.float64)
                        batch[k] = batch[k] + jnp.asarray(noise_scale * noise)
            out.append(batch)
    return out


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_remainder", [True, False])
@pytest.mark.parametrize("noise_scale", [0.0, 0.02])
def test_batches_are_bit_identical(shuffle, drop_remainder, noise_scale):
    samples, bs, seed = _samples(), 7, 3
    loader = InMemoryDataLoader(samples, bs, shuffle=shuffle, seed=seed,
                                drop_remainder=drop_remainder, noise_scale=noise_scale)
    got = list(loader)
    want = _reference(samples, bs, shuffle=shuffle, seed=seed,
                      drop_remainder=drop_remainder, noise_scale=noise_scale)
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        assert g.keys() == w.keys()
        for k in g:
            assert jnp.array_equal(g[k], w[k]), f"batch differs on key {k}"


def test_multi_epoch_noise_stream_unchanged():
    """noise_rng is seeded from the epoch counter; pre-stacking must not disturb it."""
    samples, bs, seed = _samples(seed=5), 6, 11
    loader = InMemoryDataLoader(samples, bs, shuffle=True, seed=seed,
                                drop_remainder=True, noise_scale=0.05)
    got = [b for _ in range(3) for b in loader]
    want = _reference(samples, bs, shuffle=True, seed=seed, drop_remainder=True,
                      noise_scale=0.05, epochs=3)
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        for k in g:
            assert jnp.array_equal(g[k], w[k])


def test_empty_sample_list_does_not_crash():
    assert list(InMemoryDataLoader([], 4, shuffle=False, seed=0)) == []
