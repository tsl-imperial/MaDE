"""E01 evaluator: apply MaDE or a baseline over perturbed test trajectories and report metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from made.utils.jax_setup import configure

configure()

from made.baselines import (  # noqa: E402
    ClampBaseline,
    apply_baseline_over_trajectory,
)
from made.data.simulation_data import load_split  # noqa: E402
from made.evaluation.metrics import compute_metrics  # noqa: E402
from made.evaluation.perturbation import add_observation_noise, perturb_trajectories  # noqa: E402
from made.models import AugmentedDynamics, MaDECell, ZeroResidual  # noqa: E402
from made.physics import build_system, build_system_for_model, resolve_params  # noqa: E402
from made.utils import CheckpointManager, load_config  # noqa: E402
from made.utils.config import ExperimentConfig  # noqa: E402

_MADE_VARIANTS = {
    "made",
    "made-no-residual",
    "made-no-corrector",
    "made-supervised-i",
    "made-fixed-i",
    "made-prior-only",
}
_BASELINE_VARIANTS = {"mlp", "fab", "clamp"}
_ALL_VARIANTS = _MADE_VARIANTS | _BASELINE_VARIANTS


def _restore_model(checkpoint: str | None, variant: str):
    if checkpoint is None:
        raise ValueError(f"--checkpoint required for {variant} baseline")
    train_state = CheckpointManager(checkpoint).restore()
    if train_state is None:
        raise ValueError(f"Could not load checkpoint from {checkpoint}")
    return train_state.model


def main_programmatic(
    cfg: ExperimentConfig,
    checkpoint: str | None,
    variant: str,
    test_data: str,
    output: str,
    eval_regime: str | None = None,
    dyn_learned_from: str | None = None,
) -> dict:
    """Run the E01 evaluation and write results to *output*.

    eval_regime: optional label describing the perturbation regime used at
    evaluation time (e.g. "fully-specified", "underspecified"). Written into
    the output JSON for provenance when provided.

    dyn_learned_from: path to the MaDE checkpoint for this system, condition and
    seed, used ONLY for rows that have no learned model of their own (clamp, the
    per-step MLP, FAB). Given it, Dyn.-L is scored for those rows by recovering
    controls through MaDE's LEARNED inverse and taking the residual under MaDE's
    learned augmented dynamics, with the known model's inverse as the documented
    fallback. Omitted (the default), those rows have no Dyn.-L and the metric is
    omitted rather than filled. The choice of checkpoint is the caller's.

    Returns the result dict.
    """
    # ------------------------------------------------------------------
    # 1. Resolve physics
    # ------------------------------------------------------------------
    true_system = cfg.physics.true_system
    known_system = cfg.model.known_system or true_system
    known_param_overrides = cfg.model.known_params if cfg.model.known_system else cfg.physics.true_params

    true_physics, _ = build_system(true_system)
    known_physics, known_constraints = build_system_for_model(true_system, known_system)

    true_params = resolve_params(true_system, cfg.physics.true_params)
    known_params = resolve_params(known_system, known_param_overrides)

    # ------------------------------------------------------------------
    # 2. Load test data
    # ------------------------------------------------------------------
    states, controls = load_split(test_data, "test")
    # states: (N, T, state_dim), controls: (N, T-1, control_dim)

    # ------------------------------------------------------------------
    # 3. Perturb
    # ------------------------------------------------------------------
    key = jax.random.key(cfg.evaluation.perturbation_seed)
    state_bounds = (known_constraints.state_min, known_constraints.state_max)
    perturbed_states = perturb_trajectories(states, cfg.data, key, state_bounds=state_bounds)

    if cfg.data.noise_scale > 0:
        key, noise_key = jax.random.split(key)
        perturbed_states = add_observation_noise(perturbed_states, cfg.data.noise_scale, noise_key)

    # Rollout convention (ref/notes/E01-experiment-design.md → "Rollout convention"):
    # the scan seed must be a feasible observed state, never a perturbed one. C
    # acts only through u with x re-completed by T(x_prev, u); from an infeasible
    # x_prev the reachable one-step set may not intersect the feasible region, so
    # an infeasible seed propagates through the carry. Restoring the unperturbed
    # ground-truth x_0 here makes x_prev at t=1 feasible by construction.
    perturbed_states = perturbed_states.at[:, 0].set(states[:, 0])

    # ------------------------------------------------------------------
    # 4. Build corrector and apply correction
    # ------------------------------------------------------------------
    if variant in _MADE_VARIANTS:
        cell = MaDECell.from_config(
            known_physics,
            known_constraints,
            cfg.model,
            cfg.corrector,
            key=jax.random.key(0),
            dt=cfg.physics.dt,
        )

        if checkpoint is not None:
            train_state = CheckpointManager(checkpoint).restore()
            if train_state is not None:
                cell = train_state.model

        def _apply_trajectory(trajectory: jax.Array) -> tuple[jax.Array, jax.Array]:
            def _step(x_prev: jax.Array, x_target: jax.Array) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
                x_corr, u_corr = cell(x_prev, x_target, known_params, cfg.physics.dt, training=False)
                return x_corr, (x_corr, u_corr)

            _, (x_corr_tail, u_corr) = jax.lax.scan(_step, trajectory[0], trajectory[1:])
            return jnp.concatenate([trajectory[:1], x_corr_tail], axis=0), u_corr

        x_corrected, u_corrected = jax.vmap(_apply_trajectory)(perturbed_states)

        dynamics_learned = cell.augmented_dynamics

    else:
        # Baseline variants
        if variant == "clamp":
            baseline = ClampBaseline(constraints=known_constraints)
        elif variant in _BASELINE_VARIANTS:
            baseline = _restore_model(checkpoint, variant)
        else:
            raise ValueError(f"Unknown variant '{variant}'")

        x_corrected, u_corrected = jax.vmap(
            lambda s, c: apply_baseline_over_trajectory(baseline, s, c, cfg.physics.dt)
        )(perturbed_states, controls)
        # x_corrected: (N, T, state_dim), u_corrected: (N, T-1, control_dim) zeros

        # Build a stub AugmentedDynamics using the known physics with zero residual
        stub_dynamics = AugmentedDynamics(
            known_physics,
            ZeroResidual(known_physics.state_dim),
        )
        dynamics_learned = stub_dynamics
        # A baseline emits no controls and has no learned model of its own, so Dyn.-L has
        # nothing of its own to check. Borrow the MaDE model for this cell: recover controls
        # through its LEARNED inverse and score the residual under its learned augmented
        # dynamics. The zero placeholder in `u_corrected` would make the metric meaningless, so
        # the controls are passed explicitly below.
        if dyn_learned_from is not None:
            borrowed = CheckpointManager(dyn_learned_from).restore()
            if borrowed is None:
                raise ValueError(f"Could not load MaDE checkpoint from {dyn_learned_from}")
            borrowed_cell = borrowed.model
            dynamics_learned = borrowed_cell.augmented_dynamics

    # ------------------------------------------------------------------
    # 5. Compute metrics
    # ------------------------------------------------------------------
    x_gt = states          # (N, T, state_dim)
    u_gt = controls        # (N, T-1, control_dim)

    # For non-MaDE baselines, u_corrected is a zero placeholder (baselines do not
    # infer controls). Pass the ground-truth control trajectory so dynamics-violation
    # metrics measure consistency with the actual actuated rollout, not a free coast.
    # Three separate control sets are used, because the three metrics need different recovered
    # controls, not one shared placeholder.
    #
    #   inequality  -- the row's own emitted controls where it emits (every MaDE variant does,
    #                  since they all route through the model); controls recovered through the
    #                  condition's KNOWN model where it does not (clamp, MLP, FAB).
    #                  Scoring a baseline at u = 0 would put it inside every control box
    #                  and its control-bound violations would never be counted.
    #   Dyn.-K      -- recovered through the KNOWN model, on EVERY row.
    #   Dyn.-T      -- fully specified (known model IS the true model): recovered through that
    #                  model, so Dyn.-T equals Dyn.-K by construction. Underspecified (the
    #                  dynamic bicycle): the row's OWN emitted controls where it emits them,
    #                  KNOWN-model (kinematic-bicycle) recovered where it emits none. The TRUE
    #                  model's inverse is never used there for the dynamic bicycle.
    #   Dyn.-L      -- the learned model's controls. Defined for MaDE rows only; omitted for
    #                  rows with no learned model, which is reported as pending.
    #
    # NO ROW USES u_gt for any dynamics metric.
    def _recover(physics, phys_params, x_seq):
        """Controls implied by consecutive states through `physics`'s analytic inverse."""
        def one(traj):
            return jax.vmap(
                lambda a, b: physics.known_control_prior(a, b, phys_params, cfg.physics.dt)
            )(traj[:-1], traj[1:])
        return jax.vmap(one)(x_seq)

    is_made = variant not in _BASELINE_VARIANTS
    u_known_recovered = _recover(known_physics, known_params, x_corrected)
    u_true_recovered = (_recover(true_physics, true_params, x_corrected)
                        if true_physics is not None else u_known_recovered)
    u_for_inequality = u_corrected if is_made else u_known_recovered
    # Where the known model is NOT the true model -- the underspecified dynamic
    # bicycle, and only there -- Dyn.-T takes the row's OWN emitted controls where it emits them
    # (every MaDE variant), and controls recovered through the KNOWN (kinematic-bicycle) model's
    # inverse where it emits none (clamp, MLP, FAB, and prior-only if it emits none). The TRUE
    # model's inverse is NOT used on the dynamic bicycle: evaluating the dynamic-bicycle inverse
    # on states that are off its manifold makes DB Dyn.-T explode.
    #
    # The three fully specified systems are untouched. There `known_system` is
    # None, so the known model IS the true model, `u_true_recovered` equals `u_known_recovered`,
    # and Dyn.-T equals Dyn.-K by construction -- which this branch preserves bit for bit.
    misspecified = cfg.model.known_system is not None
    u_for_dyn_true = (
        (u_corrected if is_made else u_known_recovered) if misspecified else u_true_recovered
    )
    u_for_dynamics = None  # superseded by the explicit per-metric sets below

    # Dyn.-L for a row with no learned model of its own.
    #
    # The learned inverse is TRIED and its output tested: every recovered control must be
    # finite. If any is not, the fallback is the KNOWN model's inverse -- the residual is still
    # taken under the LEARNED augmented dynamics either way, because it is Dyn.-L that is being
    # measured; only the control recovery changes. Which path was taken is recorded per cell, so
    # no reader has to infer it from the value.
    u_for_dyn_learned = None
    dyn_learned_recovery = None
    dyn_learned_available = is_made
    if not is_made and dyn_learned_from is not None:
        dyn_learned_available = True

        def _learned_recover(x_seq):
            def one(traj):
                return jax.vmap(
                    lambda a, b: borrowed_cell.inverse_dynamics(a, b, known_params)
                )(traj[:-1], traj[1:])
            return jax.vmap(one)(x_seq)

        u_learned = _learned_recover(x_corrected)
        n_nonfinite = int(jnp.sum(~jnp.isfinite(u_learned)))
        if n_nonfinite == 0:
            u_for_dyn_learned = u_learned
            dyn_learned_recovery = {
                "path": "learned_inverse",
                "what": "controls recovered through MaDE's learned inverse I_phi; residual "
                        "taken under MaDE's learned augmented dynamics",
                "nonfinite_recovered_controls": 0,
                "max_abs_recovered_control": float(jnp.max(jnp.abs(u_learned))),
            }
        else:
            u_for_dyn_learned = u_known_recovered
            dyn_learned_recovery = {
                "path": "known_inverse_fallback",
                "what": "MaDE's learned inverse produced non-finite controls, so the fallback "
                        "applies: controls recovered through the KNOWN model's "
                        "inverse, residual still taken under MaDE's learned augmented dynamics",
                "nonfinite_recovered_controls": n_nonfinite,
                "total_recovered_controls": int(u_learned.size),
            }

    metrics = compute_metrics(
        x_corrected,
        u_corrected,
        x_gt,
        u_gt,
        known_constraints,
        known_physics,
        dynamics_learned,
        known_params,
        cfg.physics.dt,
        dynamics_true=true_physics,
        true_params=true_params,
        u_for_inequality=u_for_inequality,
        u_for_dyn_known=u_known_recovered,
        u_for_dyn_true=u_for_dyn_true,
        u_for_dyn_learned=u_for_dyn_learned,
        dyn_learned_available=dyn_learned_available,
        u_for_dynamics=u_for_dynamics,
    )

    # ------------------------------------------------------------------
    # 6. Write output
    # ------------------------------------------------------------------
    condition = "underspecified" if cfg.model.known_system else "fully_specified"
    result = {
        "system": cfg.physics.true_system,
        "variant": variant,
        "condition": condition,
        "seed": cfg.training.seed,
        "metrics": metrics,
    }
    if eval_regime is not None:
        result["eval_regime"] = eval_regime
    # Which recovery this cell's Dyn.-L used, recorded per cell so the audit
    # can say it rather than a reader inferring it from the value.
    if dyn_learned_recovery is not None:
        result["dyn_learned_recovery"] = dyn_learned_recovery

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="E01 MaDE evaluator")
    parser.add_argument("--config", required=True, help="Path to ExperimentConfig JSON")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint directory")
    parser.add_argument(
        "--variant",
        default="made",
        choices=sorted(_ALL_VARIANTS),
        help="Correction variant to evaluate",
    )
    parser.add_argument("--test-data", default=None, help="Path to test split directory")
    parser.add_argument("--output", default=None, help="Path to write JSON metrics")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the 6 metric keys and exit without loading data",
    )
    args = parser.parse_args()

    if args.dry_run:
        metric_keys = [
            "inequality_violation_rate",
            "inequality_violation_magnitude",
            "dynamics_violation_known",
            "dynamics_violation_learned",
            "dynamics_violation_true",
            "fidelity",
        ]
        print("E01 metric keys:", metric_keys)
        return

    cfg = load_config(args.config)
    if args.test_data is None or args.output is None:
        parser.error("--test-data and --output are required unless --dry-run is set")
    main_programmatic(
        cfg=cfg,
        checkpoint=args.checkpoint,
        variant=args.variant,
        test_data=args.test_data,
        output=args.output,
    )


if __name__ == "__main__":
    main()
