"""Maintenance test: made.data._LAZY_ATTRS must cover all re-exported names.

This test enforces the bidirectional invariant:
    made.data.__all__ ∩ module.__all__ ≡ _LAZY_ATTRS keys mapping to module

__all__ is required on every lazy-target module (no dir() fallback).
"""

import importlib

import pytest

import made.data as _data_mod


def test_lazy_attrs_keys_covered_by_all():
    """Every key in _LAZY_ATTRS must appear in made.data.__all__."""
    missing = [k for k in _data_mod._LAZY_ATTRS if k not in _data_mod.__all__]
    assert not missing, (
        f"_LAZY_ATTRS keys not in made.data.__all__: {missing}. "
        "Add them to __all__ or remove from _LAZY_ATTRS."
    )


def test_all_lazy_entries_have_submodule_all():
    """Every lazy-target module must define __all__."""
    unique_modules = set(_data_mod._LAZY_ATTRS.values())
    for mod_name in sorted(unique_modules):
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, "__all__"):
            pytest.fail(
                f"{mod_name} must define __all__ to participate in the lazy-attr contract"
            )


def test_submodule_to_made_data_direction():
    """(a) Submodule → made.data: every name in module.__all__ that is also in
    made.data.__all__ must appear in _LAZY_ATTRS mapping to that module."""
    unique_modules = set(_data_mod._LAZY_ATTRS.values())
    violations = []
    for mod_name in sorted(unique_modules):
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, "__all__"), f"{mod_name} missing __all__"
        for name in mod.__all__:
            if name in _data_mod.__all__:
                if _data_mod._LAZY_ATTRS.get(name) != mod_name:
                    violations.append(
                        f"{name!r} is in {mod_name}.__all__ and made.data.__all__ "
                        f"but _LAZY_ATTRS maps it to {_data_mod._LAZY_ATTRS.get(name)!r}"
                    )
    assert not violations, "\n".join(violations)


def test_all_names_in_lazy_or_eager():
    """Every name in made.data.__all__ must be in _LAZY_ATTRS or eagerly in module dict.

    Catches: contributor adds to made.data.__all__ but forgets _LAZY_ATTRS entry.
    """
    lazy_keys = set(_data_mod._LAZY_ATTRS.keys())
    unaccounted = [
        name for name in _data_mod.__all__
        if name not in lazy_keys and name not in vars(_data_mod)
    ]
    assert not unaccounted, (
        f"Names in made.data.__all__ that are neither in _LAZY_ATTRS nor eagerly "
        f"imported into the module namespace: {unaccounted}. "
        f"Add to _LAZY_ATTRS or ensure they are eagerly imported."
    )


def test_lazy_attrs_to_submodule_direction():
    """(b) _LAZY_ATTRS → submodule: every key must (i) resolve to a real attr
    on its declared module, and (ii) appear in that module's __all__."""
    violations = []
    for attr_name, mod_name in sorted(_data_mod._LAZY_ATTRS.items()):
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, "__all__"):
            violations.append(f"{mod_name} missing __all__")
            continue
        if not hasattr(mod, attr_name):
            violations.append(
                f"_LAZY_ATTRS[{attr_name!r}] → {mod_name} but {mod_name}.{attr_name} does not exist"
            )
        if attr_name not in mod.__all__:
            violations.append(
                f"_LAZY_ATTRS[{attr_name!r}] → {mod_name} but {attr_name!r} not in {mod_name}.__all__"
            )
    assert not violations, "\n".join(violations)
