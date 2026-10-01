# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Tests for InMemoryDataLoader noise injection."""

# ruff: noqa: E402

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from made.data.grain_pipeline import InMemoryDataLoader


# Fixtures


@pytest.fixture
def synthetic_samples() -> list[dict]:
    """Small synthetic sample list that does not require disk I/O."""
    rng = np.random.default_rng(7)
    samples = []
    for _ in range(64):
        samples.append(
            {
                "x_prev": jnp.asarray(rng.standard_normal(6)),
                "x_curr": jnp.asarray(rng.standard_normal(6)),
                "params": jnp.asarray(rng.standard_normal(6)),
                "u_gt": jnp.asarray(rng.standard_normal(2)),
            }
        )
    return samples


def _collect_batches(loader: InMemoryDataLoader) -> list[dict]:
    """Materialise every batch from a loader.

    Args:
        loader: Loader to iterate.

    Returns:
        List of batches.
    """
    return list(loader)


# Zero noise: byte-identical to pre-noise behavior


def test_loader_zero_noise_byte_identical(synthetic_samples: list[dict]) -> None:
    """noise_scale=0.0 must produce element-wise identical batches to default (no kwarg)."""
    baseline = InMemoryDataLoader(
        synthetic_samples, batch_size=8, shuffle=False, seed=42
    )
    noisy_zero = InMemoryDataLoader(
        synthetic_samples, batch_size=8, shuffle=False, seed=42, noise_scale=0.0
    )
    for b_base, b_noisy in zip(list(baseline), list(noisy_zero)):
        for key in ("x_prev", "x_curr", "params", "u_gt"):
            assert jnp.array_equal(b_base[key], b_noisy[key]), (
                f"Batch key '{key}' differs between noise_scale=None and noise_scale=0.0"
            )


# Nonzero noise: batches differ from clean


def test_loader_nonzero_noise_changes_batches(synthetic_samples: list[dict]) -> None:
    """noise_scale=0.1 must produce different x_prev/x_curr from noise_scale=0.0."""
    clean = InMemoryDataLoader(
        synthetic_samples, batch_size=8, shuffle=False, seed=42, noise_scale=0.0
    )
    noisy = InMemoryDataLoader(
        synthetic_samples, batch_size=8, shuffle=False, seed=42, noise_scale=0.1
    )
    found_diff = False
    for b_clean, b_noisy in zip(list(clean), list(noisy)):
        if not jnp.array_equal(b_clean["x_prev"], b_noisy["x_prev"]):
            found_diff = True
            break
    assert found_diff, "Expected at least one batch to differ when noise_scale=0.1"


# Empirical std check


def test_loader_noise_empirical_std(synthetic_samples: list[dict]) -> None:
    """Empirical std of (noisy - clean) per state component is within ±15% of noise_scale."""
    noise_scale = 0.1
    clean_loader = InMemoryDataLoader(
        synthetic_samples, batch_size=8, shuffle=False, seed=0, noise_scale=0.0
    )
    noisy_loader = InMemoryDataLoader(
        synthetic_samples, batch_size=8, shuffle=False, seed=0, noise_scale=noise_scale
    )
    residuals = []
    for b_clean, b_noisy in zip(list(clean_loader), list(noisy_loader)):
        diff = np.array(b_noisy["x_prev"] - b_clean["x_prev"])
        residuals.append(diff)
    residuals_arr = np.concatenate(residuals, axis=0)  # (N, 6)
    empirical_std = residuals_arr.std(axis=0)
    for i, s in enumerate(empirical_std):
        assert abs(s - noise_scale) / noise_scale < 0.15, (
            f"State dim {i}: empirical std {s:.4f} is not within 15% of noise_scale={noise_scale}"
        )


# Determinism: same (seed, noise_scale) → identical sequences across epochs


def test_loader_noise_determinism(synthetic_samples: list[dict]) -> None:
    """Two loaders with identical (seed, noise_scale) produce identical batches across 3 epochs."""
    kwargs = dict(batch_size=8, shuffle=True, seed=99, noise_scale=0.05)
    loader_a = InMemoryDataLoader(synthetic_samples, **kwargs)
    loader_b = InMemoryDataLoader(synthetic_samples, **kwargs)
    for _epoch in range(3):
        batches_a = list(loader_a)
        batches_b = list(loader_b)
        assert len(batches_a) == len(batches_b)
        for b_a, b_b in zip(batches_a, batches_b):
            for key in ("x_prev", "x_curr"):
                assert jnp.array_equal(b_a[key], b_b[key]), (
                    f"Determinism failed: epoch {_epoch}, key '{key}'"
                )


# Per-epoch variation: different noise per epoch


def test_loader_noise_distinct_per_epoch(synthetic_samples: list[dict]) -> None:
    """One loader iterated for 2 epochs produces different x_prev in epoch 0 vs epoch 1."""
    loader = InMemoryDataLoader(
        synthetic_samples, batch_size=8, shuffle=False, seed=42, noise_scale=0.1
    )
    epoch0 = list(loader)
    epoch1 = list(loader)
    found_diff = False
    for b0, b1 in zip(epoch0, epoch1):
        if not jnp.array_equal(b0["x_prev"], b1["x_prev"]):
            found_diff = True
            break
    assert found_diff, "Expected noise to differ between epoch 0 and epoch 1"


# State-only noise: params and u_gt untouched


def test_loader_noise_does_not_perturb_params_or_controls(synthetic_samples: list[dict]) -> None:
    """With noise_scale=0.1, params and u_gt must be bit-identical to noise_scale=0.0."""
    clean = InMemoryDataLoader(
        synthetic_samples, batch_size=8, shuffle=False, seed=42, noise_scale=0.0
    )
    noisy = InMemoryDataLoader(
        synthetic_samples, batch_size=8, shuffle=False, seed=42, noise_scale=0.1
    )
    for b_clean, b_noisy in zip(list(clean), list(noisy)):
        for key in ("params", "u_gt"):
            assert jnp.array_equal(b_clean[key], b_noisy[key]), (
                f"Key '{key}' was unexpectedly perturbed by noise"
            )
