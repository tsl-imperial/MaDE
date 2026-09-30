"""Baseline models."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from made.baselines.clamp_baseline import ClampBaseline, clamp_baseline
from made.baselines.ekf_rts_smoother import KinodynamicSmoother, smooth_batch
from made.baselines.fab_baseline import FABBaseline, GatingNetwork, train_fab_baseline
from made.baselines.mlp_baseline import MLPBaseline, train_mlp_baseline
from made.baselines.protocol import CorrectionBaseline, correct_pair

__all__ = [
    "ClampBaseline",
    "CorrectionBaseline",
    "FABBaseline",
    "GatingNetwork",
    "KinodynamicSmoother",
    "MLPBaseline",
    "apply_baseline_over_trajectory",
    "clamp_baseline",
    "correct_pair",
    "smooth_batch",
    "train_fab_baseline",
    "train_mlp_baseline",
]


def apply_baseline_over_trajectory(
    baseline: CorrectionBaseline,
    states: jax.Array,
    controls: jax.Array,
    dt: float,
    history_len: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Apply a baseline's correct_pair over a full trajectory.

    Scans over T-1 consecutive pairs. Returns (x_corrected, u_placeholder) where
    u_placeholder is zeros shaped (T-1, control_dim) since baselines do not infer controls.

    Args:
        baseline: CorrectionBaseline with correct_pair(x_prev, x_curr) -> (state, state)
        states: (T, state_dim) trajectory states
        controls: (T-1, control_dim) — used only to derive control_dim and dtype
        dt: timestep (passed for completeness, not used by current baselines)
        history_len: window length for history-dependent baselines (default 1 = stateless)

    Returns:
        x_corr: (T, state_dim) corrected state trajectory
        u_corr: (T-1, control_dim) zero-filled control placeholder
    """
    x_prev_all = states[:-1]  # (T-1, state_dim)
    x_curr_all = states[1:]   # (T-1, state_dim)
    state_dim = states.shape[-1]

    # rolling history window, shape (history_len, state_dim)
    init_carry = jnp.broadcast_to(states[0][None], (history_len, state_dim))

    def scan_fn(
        carry: jax.Array, xs: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        x_prev, x_curr = xs
        x_prev_c, x_curr_c = baseline.correct_pair(carry[-1], x_curr)
        new_carry = jnp.concatenate([carry[1:], x_curr_c[None]], axis=0)
        return new_carry, (x_prev_c, x_curr_c)

    final_carry, (x_prev_c_all, x_curr_c_all) = jax.lax.scan(
        scan_fn, init_carry, (x_prev_all, x_curr_all)
    )
    del final_carry

    x_corr = jnp.concatenate([x_prev_c_all[:1], x_curr_c_all], axis=0)  # (T, state_dim)
    u_corr = jnp.zeros_like(controls)

    return x_corr, u_corr
