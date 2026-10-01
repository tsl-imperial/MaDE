# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""The stationary-window filter removes the intended windows and nothing else.

Default is OFF, so every existing reproduction stays byte-identical and the
reproductions are unaffected.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from made.data.window_dataset import make_prediction_windows

H, F, STRIDE = 2, 3, 1
W = H + F


def _tracks() -> tuple[jax.Array, jax.Array, jax.Array]:
    """Two tracks: one parked, one moving 1 m per step.

    Returns:
        The `(states, lengths, metadata)` arrays.
    """
    T = 8
    parked = np.zeros((T, 4), dtype=np.float64)
    parked[:, 0] = 5.0
    parked[:, 1] = -3.0
    moving = np.zeros((T, 4), dtype=np.float64)
    moving[:, 0] = np.arange(T, dtype=np.float64)
    moving[:, 3] = 1.0
    states = jnp.asarray(np.stack([parked, moving]))
    lengths = jnp.asarray(np.array([T, T], dtype=np.int64))
    metadata = jnp.asarray(np.zeros((2, 1), dtype=np.float64))
    return states, lengths, metadata


def _make(**kw: Any) -> dict[str, jax.Array]:
    """Build prediction windows from `_tracks` with extra keyword arguments.

    Args:
        **kw: Extra keyword arguments for `make_prediction_windows`.

    Returns:
        The window dict from `make_prediction_windows`.
    """
    s, l, m = _tracks()
    return make_prediction_windows(s, l, m, history=H, horizon=F, stride=STRIDE, **kw)


def test_default_is_off_and_keeps_every_window() -> None:
    """Check that default is off and keeps every window."""
    base = _make()
    explicit_none = _make(min_displacement_m=None)
    assert base["context"].shape[0] == explicit_none["context"].shape[0]
    # 2 tracks x floor((8-5)/1)+1 = 4 windows each
    assert base["context"].shape[0] == 8


def test_filter_removes_only_the_parked_track() -> None:
    """Check that filter removes only the parked track."""
    out = _make(min_displacement_m=0.5)
    assert out["context"].shape[0] == 4, "all four moving windows should survive"
    assert set(np.asarray(out["traj_index"]).tolist()) == {1}, "only the moving track"


def test_survivors_are_bit_identical_to_their_unfiltered_selves() -> None:
    """The filter must SELECT windows, never alter them."""
    unf, fil = _make(), _make(min_displacement_m=0.5)
    u_keys = list(zip(np.asarray(unf["traj_index"]).tolist(),
                      np.asarray(unf["start"]).tolist()))
    for k, (ti, st) in enumerate(zip(np.asarray(fil["traj_index"]).tolist(),
                                     np.asarray(fil["start"]).tolist())):
        j = u_keys.index((ti, st))
        assert np.array_equal(np.asarray(fil["context"][k]), np.asarray(unf["context"][j]))
        assert np.array_equal(np.asarray(fil["future"][k]), np.asarray(unf["future"][j]))
        assert np.array_equal(np.asarray(fil["metadata"][k]), np.asarray(unf["metadata"][j]))


@pytest.mark.parametrize("thresh,expected", [(0.5, 4), (2.5, 4), (3.5, 0)])
def test_threshold_is_on_net_displacement_over_the_horizon(thresh: float, expected: bool) -> None:
    """Moving track covers exactly (W-H)=3 m per window, so >3 removes everything."""
    assert _make(min_displacement_m=thresh)["context"].shape[0] == expected
