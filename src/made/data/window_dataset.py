"""History→future windowing for upstream predictor training on inD-style data (E05).

Windows never cross trajectory boundaries: for trajectory ``i`` with valid length
``L_i``, window starts ``s`` satisfy ``s + history + horizon <= L_i``. Trajectories
shorter than ``history + horizon`` contribute no windows.

Normalisation statistics (:func:`compute_state_norm_stats`) must be computed from
the **train split only** — callers are responsible for never passing val/test
windows to it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "assemble_context",
    "compute_state_norm_stats",
    "filter_stationary_tracks",
    "make_prediction_windows",
]


def make_prediction_windows(
    states: jax.Array,
    lengths: jax.Array,
    metadata: jax.Array,
    *,
    history: int,
    horizon: int,
    stride: int,
    min_displacement_m: float | None = None,
) -> dict[str, jax.Array]:
    """Slice padded trajectories into (context, future) prediction windows.

    Args:
        states: ``[N, T_max, D]`` padded trajectory states.
        lengths: ``[N]`` valid lengths per trajectory.
        metadata: ``[N, M]`` static per-trajectory metadata.
        history: number of context steps ``H``.
        horizon: number of predicted steps ``F``.
        stride: start-to-start spacing between consecutive windows.
        min_displacement_m: stationary-window filter. ``None`` (the default) keeps every
            window, which is the published behaviour and must stay the default so existing
            reproductions remain byte-identical. When a float is given, a window is kept
            only if its ground-truth net displacement over the horizon,
            ``||xy[start+window-1] - xy[start+history-1]||``, is strictly greater than it.
            MaDE enforces dynamics and stationary agents have none worth enforcing, so
            they are out of scope by design.

    Returns:
        Dict with ``context`` ``[K, H, D]``, ``future`` ``[K, F, D]``,
        ``metadata`` ``[K, M]``, ``traj_index`` ``[K]``, ``start`` ``[K]``.
    """
    if history < 1 or horizon < 1 or stride < 1:
        raise ValueError("history, horizon, and stride must all be >= 1.")
    states_np = np.asarray(states, dtype=np.float64)
    lengths_np = np.asarray(lengths, dtype=np.int64)
    metadata_np = np.asarray(metadata, dtype=np.float64)

    contexts: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    metas: list[np.ndarray] = []
    traj_indices: list[int] = []
    starts: list[int] = []
    window = history + horizon
    for i in range(states_np.shape[0]):
        length = int(lengths_np[i])
        for s in range(0, length - window + 1, stride):
            if min_displacement_m is not None:
                anchor = states_np[i, s + history - 1, :2]
                final = states_np[i, s + window - 1, :2]
                if float(np.hypot(*(final - anchor))) <= min_displacement_m:
                    continue
            contexts.append(states_np[i, s : s + history])
            futures.append(states_np[i, s + history : s + window])
            metas.append(metadata_np[i])
            traj_indices.append(i)
            starts.append(s)

    if not contexts:
        d = states_np.shape[-1]
        m = metadata_np.shape[-1]
        return {
            "context": jnp.zeros((0, history, d)),
            "future": jnp.zeros((0, horizon, d)),
            "metadata": jnp.zeros((0, m)),
            "traj_index": jnp.zeros((0,), dtype=jnp.int32),
            "start": jnp.zeros((0,), dtype=jnp.int32),
        }
    return {
        "context": jnp.asarray(np.stack(contexts)),
        "future": jnp.asarray(np.stack(futures)),
        "metadata": jnp.asarray(np.stack(metas)),
        "traj_index": jnp.asarray(np.asarray(traj_indices), dtype=jnp.int32),
        "start": jnp.asarray(np.asarray(starts), dtype=jnp.int32),
    }


def compute_state_norm_stats(context: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Per-dimension mean/std over all context states. Train-split windows only.

    Std is floored at 1e-6 so degenerate dimensions cannot produce division blowups.
    """
    flat = jnp.reshape(context, (-1, context.shape[-1]))
    mean = jnp.mean(flat, axis=0)
    std = jnp.maximum(jnp.std(flat, axis=0), 1e-6)
    return mean, std


def assemble_context(context: jax.Array, metadata: jax.Array) -> jax.Array:
    """Tile static metadata onto every context row: ``[H, D] + [M] → [H, D + M]``.

    Single-sample layout consumed by the E05 predictors; batch with ``jax.vmap``.
    """
    tiled = jnp.broadcast_to(metadata[None, :], (context.shape[0], metadata.shape[0]))
    return jnp.concatenate([context, tiled], axis=1)


def filter_stationary_tracks(
    states: jax.Array,
    lengths: jax.Array,
    metadata: jax.Array,
    *,
    min_displacement_m: float,
) -> tuple[jax.Array, jax.Array, jax.Array, np.ndarray]:
    """Drop whole tracks whose total displacement is at most ``min_displacement_m``.

    TRACK-level stationary filter, the definition of "filtered" for MaDE training. Separate
    from :func:`make_prediction_windows`'s ``min_displacement_m`` because the two act on
    different units and MaDE training never builds prediction windows: it trains on
    transition pairs drawn from whole tracks
    (``TrajectoryWindowTransform(window_size=2)``, see ``scripts/train_made_inD.py``).

    Displacement is measured end to end over the track's valid length,
    ``||xy[length-1] - xy[0]||``, against the same 0.5 m criterion as the horizon-level
    filter, not a second per-step threshold. The two definitions select nearly the same
    data — both remove 79.9% of train transition pairs, since 79.9% of pairs come from 4.4%
    of tracks: the stationary mass is whole parked tracks, not stationary moments inside
    moving ones.

    Known, deliberate asymmetry: a moving track that pauses at a light keeps those
    stationary transitions, while an evaluation window lying entirely inside that pause is
    removed by the window-level filter — training sees slightly more than evaluation does,
    the harmless direction.

    Tracks shorter than 2 samples are dropped regardless: they yield no transition pair.

    Args:
        states: ``[N, T_max, D]`` padded trajectory states.
        lengths: ``[N]`` valid lengths per trajectory.
        metadata: ``[N, M]`` static per-trajectory metadata.
        min_displacement_m: a track is KEPT only if its end-to-end displacement is
            strictly greater than this.

    Returns:
        ``(states, lengths, metadata, kept_index)`` restricted to surviving tracks, with
        ``kept_index`` the original row indices so callers can trace provenance.
    """
    states_np = np.asarray(states, dtype=np.float64)
    lengths_np = np.asarray(lengths, dtype=np.int64)
    metadata_np = np.asarray(metadata, dtype=np.float64)

    keep: list[int] = []
    for i in range(states_np.shape[0]):
        length = int(lengths_np[i])
        if length < 2:
            continue
        displacement = float(
            np.hypot(*(states_np[i, length - 1, :2] - states_np[i, 0, :2]))
        )
        if displacement > min_displacement_m:
            keep.append(i)

    kept_index = np.asarray(keep, dtype=np.int64)
    return (
        jnp.asarray(states_np[kept_index]),
        jnp.asarray(lengths_np[kept_index]),
        jnp.asarray(metadata_np[kept_index]),
        kept_index,
    )
