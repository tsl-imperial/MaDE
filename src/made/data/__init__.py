"""Data pipeline.

Note: there are two per-recording iterators in this package, intentionally
left out of the top-level ``made.data`` namespace to avoid ambiguity:

  * ``made.data.ind.iter_recordings`` — yields strictly-typed
    ``InDRecordingBundle`` dataclasses (NumPy arrays, dtype-stable).
  * ``made.data.ind_data.iter_ind_recordings`` — yields plain dicts with
    JAX-converted float64 arrays for the Plan 1 / inD trainer pipeline.

Import the one you need explicitly from its submodule.

Lazy attribute lookup
---------------------
Most names in ``__all__`` live in modules that import JAX at module-load time
(``grain_pipeline``, ``ind_data``, ``simulation_data``). Importing them
eagerly here would force JAX into every ``import made.data`` caller. We
therefore resolve those names lazily through ``__getattr__``, keeping
``import made.data`` and ``from made.data.ind import ...`` JAX-free.

Maintenance: adding a JAX-side public symbol that should be re-exported from
``made.data`` requires (a) listing in the submodule's ``__all__``, (b) listing
in ``made.data.__all__``, and (c) registering in ``_LAZY_ATTRS``. Every lazy
target module must define ``__all__``. Enforced bidirectionally by
``tests/data/test_lazy_attrs_complete.py``.
"""

from __future__ import annotations

# Eager imports — these submodules are JAX-free at module load (load_ind_split
# inside `made.data.ind.loader` lazily imports jax.numpy at call time only).
from made.data.ind import (
    InDRecordingBundle,
    InDSplitArrays,
    ValidationReport,
    load_manifest,
    preprocess_ind,
    validate_preprocessed,
)

# Lazy imports — each entry maps an attribute name to the module that defines
# it. Resolved on first access via ``__getattr__`` below.
_LAZY_ATTRS: dict[str, str] = {
    "InMemoryDataLoader": "made.data.grain_pipeline",
    "InMemoryDataSource": "made.data.grain_pipeline",
    "MetadataAttachTransform": "made.data.grain_pipeline",
    "TrajectoryWindowTransform": "made.data.grain_pipeline",
    "create_data_loader": "made.data.grain_pipeline",
    "create_data_source": "made.data.grain_pipeline",
    "IND_LOCATION_ID_INDEX": "made.data.ind_data",
    "IND_METADATA_DIM": "made.data.ind_data",
    "IND_NUM_LOCATIONS": "made.data.ind_data",
    "IND_STATE_DIM": "made.data.ind_data",
    "create_ind_data_source": "made.data.ind_data",
    "load_ind_split": "made.data.ind_data",
    "load_ind_split_stub": "made.data.ind_data",
    "generate_and_save": "made.data.simulation_data",
    "load_split": "made.data.simulation_data",
}


def __getattr__(name: str):
    """Resolve JAX-side names lazily so the package stays importable in JAX-free envs."""
    if name in _LAZY_ATTRS:
        import importlib

        module = importlib.import_module(_LAZY_ATTRS[name])
        attr = getattr(module, name)
        globals()[name] = attr  # cache for subsequent lookups
        return attr
    raise AttributeError(f"module 'made.data' has no attribute {name!r}")


__all__ = [
    "IND_LOCATION_ID_INDEX",
    "IND_METADATA_DIM",
    "IND_NUM_LOCATIONS",
    "IND_STATE_DIM",
    "InDRecordingBundle",
    "InDSplitArrays",
    "InMemoryDataLoader",
    "InMemoryDataSource",
    "MetadataAttachTransform",
    "TrajectoryWindowTransform",
    "ValidationReport",
    "create_data_loader",
    "create_data_source",
    "create_ind_data_source",
    "generate_and_save",
    "load_ind_split",
    "load_ind_split_stub",
    "load_manifest",
    "load_split",
    "preprocess_ind",
    "validate_preprocessed",
]
