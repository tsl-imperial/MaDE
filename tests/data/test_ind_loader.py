"""Tests for the inD data loader stub (Phase A6 gate).

These tests run without real inD data by using ``load_ind_split_stub``.
Shape, dtype, and location-id range are verified.
"""

# ruff: noqa: E402
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from made.data.ind_data import (
    IND_LOCATION_ID_INDEX,
    IND_METADATA_DIM,
    IND_NUM_LOCATIONS,
    IND_STATE_DIM,
    load_ind_split_stub,
)


def test_stub_shapes():
    """Stub returns arrays with the expected shapes."""
    n, t = 32, 16
    states, metadata, lengths = load_ind_split_stub(n, t, seed=0)
    assert states.shape == (n, t, IND_STATE_DIM)
    assert metadata.shape == (n, IND_METADATA_DIM)
    assert lengths.shape == (n,)


def test_stub_dtypes():
    """Stub returns float64 states/metadata and int32 lengths."""
    states, metadata, lengths = load_ind_split_stub(8, 10, seed=1)
    assert states.dtype == jnp.float64
    assert metadata.dtype == jnp.float64
    assert lengths.dtype == jnp.int32


def test_stub_lengths_value():
    """Stub lengths equal trajectory_length for all trajectories."""
    n, t = 12, 20
    _, _, lengths = load_ind_split_stub(n, t, seed=2)
    assert jnp.all(lengths == t)


def test_stub_location_id_range():
    """Location IDs (last metadata column) are in {1, 2, 3, 4}."""
    states, metadata, lengths = load_ind_split_stub(128, 10, seed=3)
    loc_ids = metadata[:, IND_LOCATION_ID_INDEX]
    assert jnp.all(loc_ids >= 1)
    assert jnp.all(loc_ids <= IND_NUM_LOCATIONS)


def test_stub_location_id_all_values():
    """With 128 trajectories, all four location IDs appear."""
    _, metadata, _ = load_ind_split_stub(128, 10, seed=4)
    loc_ids = np.asarray(metadata[:, IND_LOCATION_ID_INDEX], dtype=np.int32)
    for loc in range(1, IND_NUM_LOCATIONS + 1):
        assert loc in loc_ids, f"Location ID {loc} missing from stub data"


def test_stub_class_onehot_valid():
    """Class one-hot columns sum to 1 for every trajectory."""
    _, metadata, _ = load_ind_split_stub(32, 10, seed=5)
    # Columns 2 and 3 are the class one-hot
    cls_onehot = metadata[:, 2:4]
    row_sums = jnp.sum(cls_onehot, axis=1)
    assert jnp.allclose(row_sums, jnp.ones_like(row_sums))


def test_stub_metadata_dim_constant():
    """IND_METADATA_DIM equals 5 per Plan 1 contract."""
    assert IND_METADATA_DIM == 5


def test_stub_location_id_index_constant():
    """IND_LOCATION_ID_INDEX equals 4 (trailing column) per Plan 1 contract."""
    assert IND_LOCATION_ID_INDEX == 4


def test_load_ind_split_raises_without_plan0(tmp_path):
    """load_ind_split raises NotImplementedError (or FileNotFoundError) when Plan 0 not run."""
    from made.data.ind_data import load_ind_split

    with pytest.raises((NotImplementedError, FileNotFoundError, Exception)):
        load_ind_split(str(tmp_path / "nonexistent"), "train")
