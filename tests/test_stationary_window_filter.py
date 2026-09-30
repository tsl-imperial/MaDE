"""The stationary-window filter removes the intended windows and nothing else.

Default is OFF, so every existing reproduction stays byte-identical and the
R4-REEVAL gate keeps passing untouched.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from made.data.window_dataset import make_prediction_windows

H, F, STRIDE = 2, 3, 1
W = H + F


def _tracks():
    """Two tracks: one parked, one moving 1 m per step."""
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


def _make(**kw):
    s, l, m = _tracks()
    return make_prediction_windows(s, l, m, history=H, horizon=F, stride=STRIDE, **kw)


def test_default_is_off_and_keeps_every_window():
    base = _make()
    explicit_none = _make(min_displacement_m=None)
    assert base["context"].shape[0] == explicit_none["context"].shape[0]
    # 2 tracks x floor((8-5)/1)+1 = 4 windows each
    assert base["context"].shape[0] == 8


def test_filter_removes_only_the_parked_track():
    out = _make(min_displacement_m=0.5)
    assert out["context"].shape[0] == 4, "all four moving windows should survive"
    assert set(np.asarray(out["traj_index"]).tolist()) == {1}, "only the moving track"


def test_survivors_are_bit_identical_to_their_unfiltered_selves():
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
def test_threshold_is_on_net_displacement_over_the_horizon(thresh, expected):
    """Moving track covers exactly (W-H)=3 m per window, so >3 removes everything."""
    assert _make(min_displacement_m=thresh)["context"].shape[0] == expected
