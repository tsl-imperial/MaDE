"""Track-level stationary filter.

The filter that defines "filtered" for MaDE training. MaDE trains on transition pairs over
whole tracks, never on prediction windows, so the window-level `min_displacement_m` in
`make_prediction_windows` does not apply to it and a separate track-level definition is
used. These tests pin the contract, not the implementation.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from made.data.window_dataset import filter_stationary_tracks


def _tracks():
    """Four tracks: moving, parked, short-hop below threshold, and a 1-sample degenerate."""
    states = np.zeros((4, 6, 4), dtype=np.float64)
    states[0, :, 0] = np.linspace(0.0, 10.0, 6)      # moving: 10 m
    states[1, :, 0] = 3.0                             # parked: exactly 0 m
    states[2, :, 0] = np.linspace(0.0, 0.3, 6)        # hop: 0.3 m, below 0.5
    states[3, 0, 0] = 7.0                             # 1 valid sample: no transition pair
    lengths = np.array([6, 6, 6, 1], dtype=np.int64)
    metadata = np.arange(4 * 2, dtype=np.float64).reshape(4, 2)
    return jnp.asarray(states), jnp.asarray(lengths), jnp.asarray(metadata)


def test_keeps_moving_drops_parked_and_below_threshold():
    s, l, m = _tracks()
    st, ln, mt, kept = filter_stationary_tracks(s, l, m, min_displacement_m=0.5)
    assert kept.tolist() == [0], "only the 10 m track should survive a 0.5 m criterion"
    assert st.shape[0] == ln.shape[0] == mt.shape[0] == 1
    np.testing.assert_allclose(np.asarray(mt[0]), [0.0, 1.0])


def test_threshold_is_strict_greater_than():
    """A track displacing EXACTLY the threshold is dropped, matching the window-level
    filter's `<= min_displacement_m: continue`."""
    s, l, m = _tracks()
    kept = filter_stationary_tracks(s, l, m, min_displacement_m=0.3)[3]
    assert 2 not in kept.tolist(), "0.3 m track must be dropped at a 0.3 m threshold"
    kept = filter_stationary_tracks(s, l, m, min_displacement_m=0.29)[3]
    assert 2 in kept.tolist(), "0.3 m track must survive a 0.29 m threshold"


def test_short_tracks_dropped_regardless():
    """A 1-sample track yields no transition pair, so it is dropped even though its
    displacement is undefined rather than small."""
    s, l, m = _tracks()
    kept = filter_stationary_tracks(s, l, m, min_displacement_m=0.0)[3]
    assert 3 not in kept.tolist()


def test_displacement_is_end_to_end_not_path_length():
    """A track that loops back to its origin has large path length and zero net
    displacement, and must be dropped. This is what makes the criterion match the
    horizon-level one rather than a speed test."""
    states = np.zeros((1, 5, 4), dtype=np.float64)
    states[0, :, 0] = [0.0, 5.0, 9.0, 4.0, 0.0]
    kept = filter_stationary_tracks(
        jnp.asarray(states), jnp.asarray(np.array([5])),
        jnp.asarray(np.zeros((1, 2))), min_displacement_m=0.5)[3]
    assert kept.tolist() == []


def test_metadata_and_lengths_stay_aligned_with_states():
    s, l, m = _tracks()
    st, ln, mt, kept = filter_stationary_tracks(s, l, m, min_displacement_m=0.0)
    for out_row, src_row in enumerate(kept.tolist()):
        np.testing.assert_allclose(np.asarray(st[out_row]), np.asarray(s[src_row]))
        np.testing.assert_allclose(np.asarray(mt[out_row]), np.asarray(m[src_row]))
        assert int(ln[out_row]) == int(l[src_row])


def test_empty_result_is_well_formed_not_a_crash():
    s, l, m = _tracks()
    st, ln, mt, kept = filter_stationary_tracks(s, l, m, min_displacement_m=1e9)
    assert st.shape[0] == 0 and ln.shape[0] == 0 and mt.shape[0] == 0
    assert kept.tolist() == []
    assert st.ndim == 3 and mt.ndim == 2, "rank must survive an empty selection"


@pytest.mark.parametrize("split,exp_tracks_pct,exp_pairs_pct", [("train", 4.4, 79.9)])
def test_matches_the_measured_inD_figures(split, exp_tracks_pct, exp_pairs_pct):
    """Regression pin on the measured numbers: track-level filtering removes ~79.9% of
    train transition pairs while removing only ~4.4% of tracks. If this drifts, the
    reasoning behind the filter no longer holds and it must be re-examined."""
    from pathlib import Path
    from made.data.ind_data import create_ind_data_source

    data_dir = Path("data/inD-preprocessed/v1")
    if not data_dir.exists():
        pytest.skip("inD preprocessed data not present")
    s, m, l = create_ind_data_source(str(data_dir), split, use_stub=False,
                                     stub_num_trajectories=0, stub_trajectory_length=0,
                                     stub_seed=0)
    l_np = np.asarray(l)
    pairs_before = int(np.sum(np.maximum(l_np - 1, 0)))
    st, ln, mt, kept = filter_stationary_tracks(s, l, m, min_displacement_m=0.5)
    ln_np = np.asarray(ln)
    pairs_after = int(np.sum(np.maximum(ln_np - 1, 0)))

    tracks_removed = (1 - kept.size / l_np.size) * 100
    pairs_removed = (1 - pairs_after / pairs_before) * 100
    assert tracks_removed == pytest.approx(exp_tracks_pct, abs=0.5), tracks_removed
    assert pairs_removed == pytest.approx(exp_pairs_pct, abs=0.5), pairs_removed
