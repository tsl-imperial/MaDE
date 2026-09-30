"""Shared protocol helpers for per-step correction baselines."""

from __future__ import annotations

from typing import Protocol

import jax


class CorrectionBaseline(Protocol):
    """Per-step baseline that returns corrected consecutive states."""

    def correct_pair(
        self,
        x_prev: jax.Array,
        x_curr: jax.Array,
        metadata: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Correct a consecutive state pair."""


def correct_pair(
    model: CorrectionBaseline,
    x_prev: jax.Array,
    x_curr: jax.Array,
    metadata: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Call the shared correction protocol."""
    return model.correct_pair(x_prev, x_curr, metadata)
