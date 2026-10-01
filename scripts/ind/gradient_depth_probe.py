# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Probe Phase-2 gradient norms through the recursive corrector.

This probe answers empirically whether backpropagation through the recursive
correction loop suffers exploding/vanishing gradients: restore a trained inD
MaDE checkpoint, take one batch of transitions, and compute the EXACT
training gradient (same ``targeted_phase_loss`` + ``eqx.filter_value_and_grad``
path as ``_train_step_impl``) while sweeping the train-time corrector depth
(``corrector.train_steps``). Reported per depth and per alternating-
optimisation target (I / T): loss, global gradient norm, and per-model
subtree norms (inverse_dynamics / augmented_dynamics).

If norms stay the same order of magnitude as depth grows, the recursion is
well-conditioned; the training default depth is 2.

Usage (GPU workstation, real checkpoint):
    python scripts/ind/gradient_depth_probe.py \
        --config outputs/ind/made/seed0/config.json \
        --made-checkpoint outputs/ind/made/seed0/checkpoints \
        --data-dir data/inD-preprocessed/v1 \
        --output outputs/table_8/grad_norm_probe.json

Local CPU smoke (random cell, stub data, tiny batch):
    JAX_PLATFORMS=cpu python scripts/ind/gradient_depth_probe.py \
        --smoke-random-made --use-stub --batch-size 4 --depths 0 1 2 \
        --output /tmp/grad_norm_probe.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax

from made.utils.jax_setup import configure

configure()

if TYPE_CHECKING:
    from types import ModuleType

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from made.physics import _assert_xy_unbounded, inD_physical_constraints  # noqa: E402
from made.training import losses  # noqa: E402
from made.training.trainer import (  # noqa: E402
    _batch_arrays,
    _cell,
    _compute_x_proposal,
    _replace_cell,
    _resolve_params,
    _sample_batch_controls,
)

METRIC_VERSION = "grad-norm-probe-v1"


def _load_train_made_module() -> "ModuleType":
    """Import ``train_made.py`` by file path.

    Returns:
        The loaded ``train_made`` module.
    """
    spec = importlib.util.spec_from_file_location(
        "train_made", ROOT / "scripts" / "ind" / "train_made.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("train_made", module)
    spec.loader.exec_module(module)
    return module


def _tree_norm(tree: Any) -> float:
    """Global L2 norm over the array leaves of a pytree.

    Args:
        tree: Pytree of arrays and other leaves.

    Returns:
        The norm (0.0 when there are no array leaves).
    """
    leaves = [leaf for leaf in jax.tree.leaves(tree) if eqx.is_array(leaf)]
    if not leaves:
        return 0.0
    return float(jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in leaves)))


def _build_model(args: argparse.Namespace, config: Any) -> Any:
    """Build the MaDE model to probe, random for smoke runs or restored from a checkpoint.

    Args:
        args: Parsed command-line arguments.
        config: The experiment configuration.

    Returns:
        The ``MaDECell`` (smoke) or ``MaDEModel`` (checkpoint) with inD constraints.
    """
    if args.smoke_random_made:
        print(
            "[gradient_depth_probe] --smoke-random-made: UNTRAINED random MaDECell. "
            "TEST-ONLY — never quote these numbers."
        )
        from made.models import MaDECell
        from made.physics import KinematicBicycle
        from made.utils import CorrectorConfig, ModelConfig

        return MaDECell.from_config(
            KinematicBicycle(),
            inD_physical_constraints(),
            ModelConfig(),
            CorrectorConfig(),
            key=jax.random.key(args.seed),
            dt=config.physics.dt,
        )

    from made.models.made_model import MaDEModel

    model = MaDEModel.from_checkpoint(args.made_checkpoint)
    new_constraints = inD_physical_constraints()
    _assert_xy_unbounded(new_constraints)
    return _replace_cell(
        model, eqx.tree_at(lambda c: c.constraints, _cell(model), new_constraints)
    )


def main(argv: list[str] | None = None) -> dict:
    """Probe gradient norms through the corrector at several unroll depths and write the report.

    Args:
        argv: Command-line arguments; ``sys.argv`` when None.

    Returns:
        The report dict that was written.

    Raises:
        ValueError: If neither a checkpoint nor the smoke flag is given, or a checkpoint is probed
        without a config.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="ExperimentConfig JSON — use the config.json saved with the training run. "
        "Optional with --smoke-random-made (defaults to a fresh ExperimentConfig).",
    )
    parser.add_argument("--made-checkpoint", default=None)
    parser.add_argument(
        "--smoke-random-made",
        action="store_true",
        help="TEST-ONLY: random untrained cell instead of --made-checkpoint.",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--use-stub", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--depths", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.made_checkpoint is None and not args.smoke_random_made:
        raise ValueError("Pass --made-checkpoint, or --smoke-random-made for smoke runs.")

    if args.config is not None:
        from made.utils import load_config

        config = load_config(args.config)
    elif args.smoke_random_made:
        from made.utils.config import ExperimentConfig

        config = ExperimentConfig()
    else:
        raise ValueError("--config is required when probing a real checkpoint.")

    model = _build_model(args, config)

    train_module = _load_train_made_module()
    from dataclasses import replace

    loader_config = replace(
        config, training=replace(config.training, batch_size=args.batch_size)
    )
    train_loader, _, _, _ = train_module._build_data_loaders(
        args.data_dir if args.data_dir is not None else "",
        loader_config,
        use_stub=args.use_stub,
        num_devices=1,
        seed=args.seed,
    )
    batch = next(iter(train_loader))

    training_config = config.training
    dt = config.physics.dt

    x_prev, x_curr, params, _, metadata = _batch_arrays(batch)
    if args.smoke_random_made and params is not None and params.shape[-1] == 0:
        # Stub inD batches carry empty params; the smoke MaDECell has no encoder, so
        # substitute the kinematic wheelbase (same convention as train_predictor.py).
        params = jnp.broadcast_to(jnp.array([2.7]), (x_prev.shape[0], 1))
        batch = {**batch, "params": params}
    params = _resolve_params(model, params, metadata)
    # step == total_steps pins the sampling schedule at its end-of-training mix.
    u_sampled, _, _ = _sample_batch_controls(
        model, batch, training_config, jax.random.key(args.seed + 1), 1, 1
    )
    x_proposal = _compute_x_proposal(x_curr, 2, training_config, _cell(model))

    report: dict = {
        "metric_version": METRIC_VERSION,
        "note": (
            "grad_norm_total spans ALL array leaves incl. constraint bounds, which the "
            "trainer's I/T partition masks out of updates — quote the per-model "
            "subtree norms (inverse_dynamics for target I, augmented_dynamics for "
            "target T), not the total."
        ),
        "checkpoint": args.made_checkpoint,
        "smoke_random_made": bool(args.smoke_random_made),
        "batch_size": int(x_prev.shape[0]),
        "train_default_depth": int(_cell(model).corrector.train_steps),
        "depths": {},
    }

    for depth in args.depths:
        cell_d = eqx.tree_at(lambda c: c.corrector.train_steps, _cell(model), depth)
        model_d = _replace_cell(model, cell_d)
        depth_report: dict = {}
        for target in ("I", "T"):

            def _loss_fn(m: Any) -> Any:
                """Phase-2 loss for one target, with its auxiliary output.

                Args:
                    m: The model to differentiate.

                Returns:
                    The ``targeted_phase_loss`` result (loss and auxiliaries).
                """
                return losses.targeted_phase_loss(
                    m,
                    x_prev,
                    x_curr,
                    params,
                    dt,
                    u_sampled,
                    training_config,
                    phase=2,
                    target=target,
                    x_proposal=x_proposal,
                )

            (loss, _), grads = eqx.filter_value_and_grad(_loss_fn, has_aux=True)(model_d)
            grads_cell = _cell(grads)
            depth_report[target] = {
                "loss": float(loss),
                "grad_norm_total": _tree_norm(grads),
                "grad_norm_inverse_dynamics": _tree_norm(grads_cell.inverse_dynamics),
                "grad_norm_augmented_dynamics": _tree_norm(grads_cell.augmented_dynamics),
            }
            print(
                f"depth={depth} target={target} loss={float(loss):.6g} "
                f"|g|={depth_report[target]['grad_norm_total']:.6g} "
                f"(I {depth_report[target]['grad_norm_inverse_dynamics']:.4g}, "
                f"T {depth_report[target]['grad_norm_augmented_dynamics']:.4g})"
            )
        report["depths"][str(depth)] = depth_report

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")
    return report


if __name__ == "__main__":
    main()
