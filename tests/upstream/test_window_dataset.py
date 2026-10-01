# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Tests for inD history→future windowing (leakage-safe, boundary-exact)."""

import jax.numpy as jnp
import numpy as np
import pytest

from made.data.window_dataset import (
    assemble_context,
    compute_state_norm_stats,
    make_prediction_windows,
)

H, F, D, M = 4, 3, 4, 5


def _padded_data() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Two trajectories, lengths 12 and 7, padded to 12 with NaN sentinels.

    Returns:
        States, metadata and lengths.
    """
    rng = np.random.default_rng(0)
    states = rng.standard_normal((2, 12, D))
    states[1, 7:] = np.nan  # padding must never leak into any window
    metadata = np.array(
        [
            [4.5, 2.0, 1.0, 0.0, 1.0],
            [10.2, 2.5, 0.0, 1.0, 3.0],
        ]
    )
    lengths = np.array([12, 7], dtype=np.int32)
    return jnp.asarray(states), jnp.asarray(metadata), jnp.asarray(lengths)


def test_window_counts_and_boundary_safety() -> None:
    """Verify window counts and boundary safety."""
    states, metadata, lengths = _padded_data()
    windows = make_prediction_windows(
        states, lengths, metadata, history=H, horizon=F, stride=2
    )
    # traj 0: starts 0, 2, 4 (s + 7 <= 12); traj 1: start 0 only (s + 7 <= 7).
    assert windows["context"].shape == (4, H, D)
    assert windows["future"].shape == (4, F, D)
    assert windows["metadata"].shape == (4, M)
    assert jnp.all(jnp.isfinite(windows["context"]))
    assert jnp.all(jnp.isfinite(windows["future"]))
    np.testing.assert_array_equal(np.asarray(windows["traj_index"]), [0, 0, 0, 1])
    np.testing.assert_array_equal(np.asarray(windows["start"]), [0, 2, 4, 0])


def test_window_values_match_slices() -> None:
    """Verify window values match slices."""
    states, metadata, lengths = _padded_data()
    windows = make_prediction_windows(
        states, lengths, metadata, history=H, horizon=F, stride=2
    )
    np.testing.assert_allclose(np.asarray(windows["context"][1]), np.asarray(states[0, 2 : 2 + H]))
    np.testing.assert_allclose(
        np.asarray(windows["future"][1]), np.asarray(states[0, 2 + H : 2 + H + F])
    )
    np.testing.assert_allclose(np.asarray(windows["metadata"][3]), np.asarray(metadata[1]))


def test_too_short_trajectory_contributes_nothing() -> None:
    """Verify too short trajectory contributes nothing."""
    states = jnp.zeros((1, 10, D))
    metadata = jnp.zeros((1, M))
    lengths = jnp.asarray([H + F - 1], dtype=jnp.int32)
    windows = make_prediction_windows(
        states, lengths, metadata, history=H, horizon=F, stride=1
    )
    assert windows["context"].shape == (0, H, D)
    assert windows["future"].shape == (0, F, D)


@pytest.mark.parametrize("bad", [{"history": 0}, {"horizon": 0}, {"stride": 0}])
def test_invalid_arguments_raise(bad: dict) -> None:
    """Verify invalid arguments raise."""
    states, metadata, lengths = _padded_data()
    kwargs = {"history": H, "horizon": F, "stride": 1, **bad}
    with pytest.raises(ValueError):
        make_prediction_windows(states, lengths, metadata, **kwargs)


def test_norm_stats_values_and_floor() -> None:
    """Verify norm stats values and floor."""
    context = jnp.asarray(
        np.stack(
            [
                np.array([[0.0, 1.0, 5.0, 2.0], [2.0, 3.0, 5.0, 4.0]]),
                np.array([[4.0, 5.0, 5.0, 6.0], [6.0, 7.0, 5.0, 8.0]]),
            ]
        )
    )
    mean, std = compute_state_norm_stats(context)
    np.testing.assert_allclose(np.asarray(mean), [3.0, 4.0, 5.0, 5.0])
    assert std[2] == 1e-6  # constant dimension is floored, not zero
    assert jnp.all(std[jnp.array([0, 1, 3])] > 1.0)


def test_assemble_context_layout() -> None:
    """Verify assemble context layout."""
    context = jnp.arange(H * D, dtype=jnp.float64).reshape(H, D)
    metadata = jnp.asarray([4.5, 2.0, 1.0, 0.0, 2.0])
    combined = assemble_context(context, metadata)
    assert combined.shape == (H, D + M)
    np.testing.assert_allclose(np.asarray(combined[:, :D]), np.asarray(context))
    for row in range(H):
        np.testing.assert_allclose(np.asarray(combined[row, D:]), np.asarray(metadata))
