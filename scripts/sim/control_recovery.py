"""Control-recovery evaluation: how well does the inverse model recover ground-truth controls?

Measures the control-recovery error of the inverse dynamics against the simulator's
actuated controls, on the CLEAN test split (no perturbation, no observation noise), so the
error isolates the inverse model itself. Per cell: ``u_hat_{t-1} = I(x_{t-1}, x_t,
p_known)`` over consecutive ground-truth state pairs, compared to ``u_gt`` from the
simulator.

Three columns per row: the known-physics control prior alone (deterministic, no
checkpoint), the cycle-trained MaDE inverse, and the supervised-I ablation.

Run layout read: ``<runs-root>/<system>/<condition>/<variant>/seed<seed>/checkpoints``,
one config per system and variant at ``<config-dir>/<prefix>_<variant>.json``.

Outputs: ``<out-json>`` (the full per-cell and per-condition summary) and ``<out-tex>``
(the paper's Table 5 tabular body).

Usage:
    python scripts/sim/control_recovery.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp

from made.utils.jax_setup import configure

configure()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from made.data.simulation_data import load_split  # noqa: E402
from made.models import MaDECell  # noqa: E402
from made.physics import build_system_for_model, resolve_params  # noqa: E402
from made.utils import CheckpointManager, load_config  # noqa: E402

METRIC_VERSION = "control-recovery-v1"

# (config_prefix, system, condition_dir)
_CONDITIONS = [
    ("di", "double_integrator", "fully-specified"),
    ("unicycle", "unicycle", "fully-specified"),
    ("kinbicycle", "kinematic_bicycle", "fully-specified"),
    ("dynbicycle_underspecified", "dynamic_bicycle", "underspecified"),
]

_CHECKPOINT_VARIANTS = ["made", "made-supervised-i"]


def control_recovery_metrics(
    inverse_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    states: jax.Array,
    controls: jax.Array,
    params: jax.Array,
) -> dict:
    """Compare ``inverse_fn(x_prev, x_curr, params)`` against ground-truth controls.

    Args:
        inverse_fn: per-sample inverse ``(x_prev, x_curr, params) -> u_hat``.
        states: ``(N, T, state_dim)`` clean trajectories.
        controls: ``(N, T-1, control_dim)`` actuated ground-truth controls.
        params: ``(param_dim,)`` known parameters passed through to the inverse.

    Returns:
        Dict of per-dimension and aggregate error statistics. ``nrmse`` entries
        are RMSE normalised by the per-dimension std of the ground-truth
        controls (scale-free, comparable across systems).
    """

    def _per_trajectory(trajectory: jax.Array) -> jax.Array:
        return jax.vmap(lambda x_prev, x_curr: inverse_fn(x_prev, x_curr, params))(
            trajectory[:-1], trajectory[1:]
        )

    u_hat = jax.vmap(_per_trajectory)(states)
    error = u_hat - controls

    control_std = jnp.std(controls, axis=(0, 1))
    rmse_per_dim = jnp.sqrt(jnp.mean(error**2, axis=(0, 1)))
    mae_per_dim = jnp.mean(jnp.abs(error), axis=(0, 1))
    bias_per_dim = jnp.mean(error, axis=(0, 1))
    nrmse_per_dim = rmse_per_dim / control_std

    return {
        "rmse_per_dim": [float(v) for v in rmse_per_dim],
        "mae_per_dim": [float(v) for v in mae_per_dim],
        "bias_per_dim": [float(v) for v in bias_per_dim],
        "control_std_per_dim": [float(v) for v in control_std],
        "nrmse_per_dim": [float(v) for v in nrmse_per_dim],
        "rmse": float(jnp.sqrt(jnp.mean(error**2))),
        "nrmse_mean": float(jnp.mean(nrmse_per_dim)),
        "num_trajectories": int(states.shape[0]),
        "num_transitions": int(states.shape[0] * (states.shape[1] - 1)),
    }


def _resolve_cell_physics(cfg):
    """Mirror scripts/evaluate.py's known-system resolution exactly."""
    true_system = cfg.physics.true_system
    known_system = cfg.model.known_system or true_system
    if cfg.model.known_system:
        known_param_overrides = cfg.model.known_params
    else:
        known_param_overrides = cfg.physics.true_params
    known_physics, known_constraints = build_system_for_model(true_system, known_system)
    known_params = resolve_params(known_system, known_param_overrides)
    return known_physics, known_constraints, known_params


def evaluate_checkpoint_cell(
    config_path: Path,
    checkpoint_dir: Path,
    states: jax.Array,
    controls: jax.Array,
) -> dict:
    """Restore a MaDE checkpoint and evaluate its inverse dynamics."""
    cfg = load_config(str(config_path))
    known_physics, known_constraints, known_params = _resolve_cell_physics(cfg)

    cell = MaDECell.from_config(
        known_physics,
        known_constraints,
        cfg.model,
        cfg.corrector,
        key=jax.random.key(0),
        dt=cfg.physics.dt,
    )
    train_state = CheckpointManager(str(checkpoint_dir)).restore()
    if train_state is None:
        raise ValueError(f"Could not restore checkpoint from {checkpoint_dir}")
    cell = train_state.model

    inverse = cell.inverse_dynamics
    metrics = control_recovery_metrics(
        lambda x_prev, x_curr, p: inverse(x_prev, x_curr, p),
        states,
        controls,
        known_params,
    )
    metrics["checkpoint"] = str(checkpoint_dir)
    return metrics


def evaluate_prior_cell(
    config_path: Path,
    states: jax.Array,
    controls: jax.Array,
) -> dict:
    """Evaluate the known-physics control prior alone (no learned component)."""
    cfg = load_config(str(config_path))
    known_physics, _, known_params = _resolve_cell_physics(cfg)
    dt = cfg.physics.dt
    metrics = control_recovery_metrics(
        lambda x_prev, x_curr, p: known_physics.known_control_prior(x_prev, x_curr, p, dt),
        states,
        controls,
        known_params,
    )
    metrics["checkpoint"] = None
    return metrics


def _aggregate(rows: list[dict]) -> dict:
    """Mean ± population std of nrmse/bias across seed rows of one variant."""
    nrmse = jnp.asarray([row["nrmse_per_dim"] for row in rows])
    bias = jnp.asarray([row["bias_per_dim"] for row in rows])
    return {
        "nrmse_per_dim_mean": [float(v) for v in jnp.mean(nrmse, axis=0)],
        "nrmse_per_dim_std": [float(v) for v in jnp.std(nrmse, axis=0)],
        "bias_per_dim_mean": [float(v) for v in jnp.mean(bias, axis=0)],
        "bias_per_dim_std": [float(v) for v in jnp.std(bias, axis=0)],
        "num_seeds": len(rows),
    }


# --- RENDER BLOCK (begin) ---
_ROWS = [  # (condition key, system label, channel label, statistic, dim, number style)
    ("double_integrator/fully-specified", "DI (exact)", r"$a_x$ (nRMSE)", "nrmse", 0, "sci"),
    ("double_integrator/fully-specified", "DI (exact)", r"$a_y$ (nRMSE)", "nrmse", 1, "sci"),
    ("unicycle/fully-specified", "UNI (exact)", r"$\delta$ (nRMSE)", "nrmse", 0, "sci"),
    ("unicycle/fully-specified", "UNI (exact)", r"$a$ (nRMSE)", "nrmse", 1, "sci"),
    ("kinematic_bicycle/fully-specified", "KB (exact)", r"$\delta$ (nRMSE)", "nrmse", 0, "sci"),
    ("dynamic_bicycle/underspecified", "DB (underspec.)", r"$\delta$ (nRMSE)", "nrmse", 0, "fixed"),
    ("dynamic_bicycle/underspecified", "DB (underspec.)", r"$a$ (nRMSE)", "nrmse", 1, "fixed"),
    ("dynamic_bicycle/underspecified", "DB (underspec.)", r"$a$ (bias)", "bias", 1, "fixed"),
]
_COLUMNS = ("prior", "made", "made-supervised-i")
_PM = r"\,{\scriptscriptstyle\pm}\,"


def _fmt(value: float, style: str) -> str:
    """One number in the table's style: 4 decimals, or 3 significant figures as a x 10^e."""
    if style == "fixed":
        return f"{value:.4f}"
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa}\\times 10^{{{int(exponent)}}}"


def _cell(agg: dict | None, stat: str, dim: int, style: str, with_std: bool) -> str:
    if agg is None:
        return "--"
    mean = _fmt(agg[f"{stat}_per_dim_mean"][dim], style)
    if not with_std:
        return f"${mean}$"
    return f"${mean}{_PM}{_fmt(agg[f'{stat}_per_dim_std'][dim], style)}$"


def render_tex(conditions: dict) -> str:
    """The tabular of the control-recovery table (known inverse, MaDE, supervised-I columns)."""
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"System & Channel & known inverse & MaDE & sup.-$\mathcal{I}$ \\",
        r"\midrule",
    ]
    for key, system_label, channel_label, stat, dim, style in _ROWS:
        variants = conditions.get(key, {})
        cells = [_cell(variants.get(v), stat, dim, style, with_std=(v != "prior"))
                 for v in _COLUMNS]
        lines.append(f"{system_label} & {channel_label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"
# --- RENDER BLOCK (end) ---


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", default="double_integrator,unicycle,kinematic_bicycle,"
                                              "dynamic_bicycle")
    parser.add_argument("--variants", default="prior,made,made-supervised-i")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--config-dir", default=str(ROOT / "configs" / "sim"))
    parser.add_argument("--runs-root", default=str(ROOT / "outputs" / "sim" / "runs"))
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "generated"))
    parser.add_argument("--out-json", default=str(ROOT / "outputs" / "sim" / "control_recovery.json"))
    parser.add_argument("--out-tex", default=str(ROOT / "outputs" / "tables" / "tab_control_recovery.tex"))
    args = parser.parse_args(argv)

    systems = set(args.systems.split(","))
    variants = set(args.variants.split(","))
    seeds = [int(s) for s in args.seeds.split(",")]
    config_dir = Path(args.config_dir)
    runs_root = Path(args.runs_root)
    data_root = Path(args.data_dir)
    out_json = Path(args.out_json)
    out_tex = Path(args.out_tex)

    summary = {"metric_version": METRIC_VERSION, "seeds": seeds, "conditions": {}, "cells": [],
               "missing": []}
    for config_prefix, system, condition_dir in _CONDITIONS:
        if system not in systems:
            continue
        data_dir = data_root / system
        if not data_dir.is_dir():
            summary["missing"].append(str(data_dir))
            print(f"[MISSING] {data_dir}")
            continue
        states, controls = load_split(str(data_dir), "test")
        condition_key = f"{system}/{condition_dir}"
        condition_summary = {}
        if "prior" in variants:
            m = evaluate_prior_cell(config_dir / f"{config_prefix}_made.json", states, controls)
            m["metric_version"] = METRIC_VERSION
            m["test_data"] = str(data_dir)
            summary["cells"].append({"condition": condition_key, "variant": "prior",
                                      "seed": None, "metrics": m})
            condition_summary["prior"] = _aggregate([m])
        for variant in _CHECKPOINT_VARIANTS:
            if variant not in variants:
                continue
            rows = []
            for seed in seeds:
                ckpt_dir = runs_root / system / condition_dir / variant / f"seed{seed}" / \
                    "checkpoints"
                if not ckpt_dir.is_dir():
                    summary["missing"].append(str(ckpt_dir))
                    print(f"[MISSING] {ckpt_dir}")
                    continue
                m = evaluate_checkpoint_cell(config_dir / f"{config_prefix}_{variant}.json",
                                              ckpt_dir, states, controls)
                m["metric_version"] = METRIC_VERSION
                m["test_data"] = str(data_dir)
                summary["cells"].append({"condition": condition_key, "variant": variant,
                                          "seed": seed, "metrics": m})
                rows.append(m)
            if rows:
                condition_summary[variant] = _aggregate(rows)
        summary["conditions"][condition_key] = condition_summary

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(render_tex(summary["conditions"]))
    print(f"wrote {out_json}")
    print(f"wrote {out_tex}")
    return 1 if summary["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
