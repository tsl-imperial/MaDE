"""FAB baseline, phase 2: train the phase-1 model, then continue with the phase-2
structuring loss on top of the same weights.

One system, one seed per process. `src/made/baselines/fab_phase2.py` implements the
four loss terms and the infeasible generators used below.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402

from made.utils.jax_setup import configure  # noqa: E402

configure()

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from made.baselines.fab_baseline import train_fab_baseline  # noqa: E402
from made.baselines.fab_baseline import FABBaseline  # noqa: E402
from made.baselines.fab_phase2 import (  # noqa: E402
    K_CRITIC_STEPS,
    Discriminator,
    discriminator_loss,
    make_infeasible_generators,
    sample_ball,
    structuring_loss,
)
from made.data import create_data_loader, create_data_source  # noqa: E402
from made.models.augmented_dynamics import AugmentedDynamics, ZeroResidual  # noqa: E402
from made.physics import CompositeConstraints, build_system, resolve_params  # noqa: E402
from made.utils import CheckpointManager, TrainState, load_config, save_config  # noqa: E402

# Config prefix for each system, matching configs/sim naming (D3).
PREFIX: dict[str, str] = {
    "double_integrator": "di",
    "unicycle": "unicycle",
    "kinematic_bicycle": "kinbicycle",
    "dynamic_bicycle": "dynbicycle_underspecified",
}


def true_state_dim(cfg) -> int:
    """State dimension of the condition's TRUE system, for validating a per-dim override."""
    from made.physics import build_system
    physics, _ = build_system(cfg.physics.true_system)
    return int(physics.state_dim)


def _condition(cfg) -> dict:
    """True system, known system and eval regime, all read from the config.

    Only the dynamic-bicycle condition is underspecified; on the other three the known model
    is the data-generating model, so they are fully specified.
    """
    true_system = cfg.physics.true_system
    known_system = getattr(cfg.model, "known_system", None)
    underspecified = bool(known_system) and known_system != true_system
    regime = "underspecified" if underspecified else "fully-specified"
    return {
        "true_system": true_system,
        "known_system": known_system or true_system,
        "underspecified": underspecified,
        "eval_regime": regime,
    }


# Appendix A.2 setting.
NUM_DECODERS = 1
LATENT_RADIUS = 0.5
HIDDEN = (64, 64)
BATCH_SIZE = 256
PHASE1_EPOCHS = 500
PHASE2_EPOCHS = 100
LR_AE = 1e-4
LR_DISC = 2e-4

LATENT_DIM = 8  # six free state dimensions plus two of control.
GENERATORS = ("box_violating", "off_manifold", "wrong_control", "ambient")


def _box(constraints):
    return constraints.constraints[0] if isinstance(constraints, CompositeConstraints) else constraints


def _true_step_fn(dt: float, true_params: jax.Array, true_system: str):
    """One step of the true system, the manifold generator 3 moves along.

    On the underspecified dynamic bicycle the true model is denied to the model, so using it
    to synthesise negatives is a disclosure. On the three fully-specified conditions the true
    model is the known model, so there is no asymmetry. `_condition` decides which case applies.
    """
    true_physics, _ = build_system(true_system)
    dynamics = AugmentedDynamics(true_physics, ZeroResidual(true_physics.state_dim))

    def step(x_prev: jax.Array, u: jax.Array) -> jax.Array:
        return dynamics.integrate(x_prev, u, true_params, dt)

    return step


def _make_infeasible_batch(generators, key, x_prev, x_curr):
    """Equal shares of the four generators over one feasible batch.

    Returns `(pairs, per_generator)` where `per_generator` keeps each generator's own slice
    so accuracy can be broken down by generator rather than only in aggregate — a high
    aggregate driven by generators 1 and 4 would mean the manifold was never learned.
    """
    n = x_prev.shape[0]
    share = n // len(GENERATORS)
    keys = jax.random.split(key, len(GENERATORS))
    pairs = []
    per_generator: dict[str, jax.Array] = {}
    for i, name in enumerate(GENERATORS):
        lo = i * share
        hi = n if i == len(GENERATORS) - 1 else (i + 1) * share
        xp, xc = x_prev[lo:hi], x_curr[lo:hi]
        sub = jax.random.split(keys[i], xp.shape[0])
        bad = jax.vmap(generators[name])(sub, xp, xc)
        block = jnp.concatenate([xp, bad], axis=1)
        per_generator[name] = block
        pairs.append(block)
    return jnp.concatenate(pairs, axis=0), per_generator


def _train_phase2(
    model: FABBaseline,
    discriminator: Discriminator,
    batches: list[dict],
    generators,
    *,
    epochs: int,
    lr_ae: float,
    lr_disc: float,
    n_latent_samples: int,
    key: jax.Array,
    normalise: tuple[jax.Array, jax.Array] | None = None,
):
    """The two sides alternate with distinct objectives: not a minimax loss -- the
    discriminator's gradient never reaches the autoencoder's parameters, nor the reverse.
    """
    d_opt = optax.adam(lr_disc)
    m_opt = optax.adam(lr_ae)
    d_state = d_opt.init(eqx.filter(discriminator, eqx.is_array))
    m_state = m_opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def _d_step(disc, state, pairs, labels):
        loss, grads = eqx.filter_value_and_grad(discriminator_loss)(disc, pairs, labels)
        updates, state = d_opt.update(grads, state)
        return eqx.apply_updates(disc, updates), state, loss

    @eqx.filter_jit
    def _m_step(mdl, state, disc, pairs, labels, feasible, z):
        def _loss(m):
            return structuring_loss(m, disc, pairs, labels, feasible, z).total

        loss, grads = eqx.filter_value_and_grad(_loss)(mdl)
        updates, state = m_opt.update(grads, state)
        return eqx.apply_updates(mdl, updates), state, loss

    @eqx.filter_jit
    def _terms(mdl, disc, pairs, labels, feasible, z):
        return structuring_loss(mdl, disc, pairs, labels, feasible, z)

    history: list[dict] = []
    for epoch in range(epochs):
        epoch_start = time.time()
        d_epoch, m_epoch = [], []
        for batch in batches:
            key, bad_key, z_key = jax.random.split(key, 3)
            x_prev, x_curr = batch["x_prev"], batch["x_curr"]
            feasible = jnp.concatenate([x_prev, x_curr], axis=1)
            bad, _ = _make_infeasible_batch(generators, bad_key, x_prev, x_curr)
            pairs = jnp.concatenate([feasible, bad], axis=0)
            labels = jnp.concatenate(
                [jnp.ones(feasible.shape[0]), jnp.zeros(bad.shape[0])]
            )
            z = sample_ball(z_key, n_latent_samples, model.latent_dim, model.latent_radius)

            for _ in range(K_CRITIC_STEPS):
                discriminator, d_state, d_loss = _d_step(discriminator, d_state, pairs, labels)
            model, m_state, m_loss = _m_step(
                model, m_state, discriminator, pairs, labels, feasible, z
            )
            d_epoch.append(float(d_loss))
            m_epoch.append(float(m_loss))
        record = {
            "epoch": epoch,
            "L_disc": sum(d_epoch) / len(d_epoch),
            "L_struc": sum(m_epoch) / len(m_epoch),
            "seconds": time.time() - epoch_start,
        }
        history.append(record)
        # Printed per epoch so a cost overrun is visible while it is still cheap to stop.
        print(f"  phase2 epoch {epoch:3d}  L_disc {record['L_disc']:.6f}  "
              f"L_struc {record['L_struc']:.6e}  {record['seconds']:.1f}s", flush=True)

    # The four terms at convergence, on the last batch's construction.
    key, bad_key, z_key = jax.random.split(key, 3)
    last = batches[-1]
    feasible = jnp.concatenate([last["x_prev"], last["x_curr"]], axis=1)
    bad, _ = _make_infeasible_batch(generators, bad_key, last["x_prev"], last["x_curr"])
    pairs = jnp.concatenate([feasible, bad], axis=0)
    labels = jnp.concatenate([jnp.ones(feasible.shape[0]), jnp.zeros(bad.shape[0])])
    z = sample_ball(z_key, n_latent_samples, model.latent_dim, model.latent_radius)
    terms = _terms(model, discriminator, pairs, labels, feasible, z)

    converged = {
        "L_recon": float(terms.recon),
        "L_feasibility": float(terms.feasibility),
        "L_hinge": float(terms.hinge),
        "L_latent": float(terms.latent),
        "L_geom": float(terms.geom),
        "L_struc_total": float(terms.total),
    }
    return model, discriminator, history, converged


def _save(model, out_dir: Path, key) -> str:
    ckpt = out_dir / "checkpoints"
    CheckpointManager(str(ckpt)).save(
        TrainState(model=model, opt_state_I=None, opt_state_T=None, key=key, step=1), 1
    )
    return str(ckpt)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the FAB baseline (phase 1 then phase 2).")
    ap.add_argument(
        "--system",
        required=True,
        choices=["double_integrator", "unicycle", "kinematic_bicycle", "dynamic_bicycle"],
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config-dir", default=str(ROOT / "configs" / "sim"))
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "generated"))
    ap.add_argument("--output-root", default=str(ROOT / "outputs" / "sim" / "runs"))
    ap.add_argument("--phase1-epochs", type=int, default=None, help="default: 500")
    ap.add_argument("--phase2-epochs", type=int, default=None, help="default: 100")
    ap.add_argument("--latent-samples", type=int, default=64, help="|z| per phase-2 step")
    ap.add_argument("--max-batches", type=int, default=None, help="smoke only; cuts the loader")
    a = ap.parse_args()

    config_path = Path(a.config_dir) / f"{PREFIX[a.system]}_fab.json"
    cfg = load_config(str(config_path))
    # Unnoised feasible pairs for both arms: phase 2's hinge would otherwise place noisy
    # near-feasible points inside the sphere while generator 2 places similar-magnitude
    # perturbations outside it, making the labels contradictory.
    #
    # This concerns the training pairs only: the zero is passed to the two loaders as an
    # argument, and `cfg.data.noise_scale` is left at its configured value, since
    # `evaluate.main_programmatic` reads the same field to decide whether to add observation
    # noise at evaluation time.
    cfg = replace(
        cfg,
        training=replace(cfg.training, seed=a.seed),
        evaluation=replace(cfg.evaluation, perturbation_seed=cfg.evaluation.perturbation_seed + a.seed),
    )
    if cfg.data.perturbation_scale_per_dim is None:
        raise SystemExit(
            f"{cfg.physics.true_system}: data.perturbation_scale_per_dim is null. "
            f"Generators 1 and 2 need it.")
    condition = _condition(cfg)
    p1_epochs = a.phase1_epochs if a.phase1_epochs is not None else PHASE1_EPOCHS
    p2_epochs = a.phase2_epochs if a.phase2_epochs is not None else PHASE2_EPOCHS

    cell_dir = (Path(a.output_root) / condition["true_system"] / condition["eval_regime"]
                / "fab" / f"seed{a.seed}")
    print(f"[fab] system={condition['true_system']} regime={condition['eval_regime']} "
          f"seed={a.seed} known={condition['known_system']}", flush=True)

    if (cell_dir / "checkpoints").exists():
        print(f"skip: {cell_dir / 'checkpoints'} already exists")
        return 0

    data_root = Path(a.data_dir)
    data_path = str(data_root / cfg.physics.true_system)
    true_physics, true_constraints = build_system(cfg.physics.true_system)
    true_params = resolve_params(cfg.physics.true_system, cfg.physics.true_params)
    box = _box(true_constraints)

    source = create_data_source(data_path, "train", cfg.data)
    val_source = create_data_source(data_path, "val", cfg.data)
    loader = create_data_loader(
        source, cfg.training.batch_size, num_devices=1, seed=a.seed, noise_scale=0.0
    )
    val_loader = create_data_loader(
        val_source,
        BATCH_SIZE,
        num_devices=1,
        shuffle=False,
        seed=a.seed,
        drop_remainder=False,
        noise_scale=0.0,
    )
    train_batches = list(loader)
    val_batches = list(val_loader)
    if a.max_batches is not None:
        train_batches = train_batches[: a.max_batches]
        val_batches = val_batches[: a.max_batches]

    key = jax.random.key(a.seed)
    init_key, p1_key, disc_key, p2_key, acc_key = jax.random.split(key, 5)

    # ---------------------------------------------------- phase 1, shared ----
    model = FABBaseline(
        true_physics.state_dim,
        latent_dim=LATENT_DIM,
        num_experts=NUM_DECODERS,
        hidden=HIDDEN,
        latent_radius=LATENT_RADIUS,
        key=init_key,
    )
    p1_cfg = replace(cfg.fab_baseline, num_epochs=p1_epochs, lr=1e-3, normalise_inputs=True)
    control = train_fab_baseline(model, train_batches, val_batches, p1_cfg, key=p1_key)

    # ---------------------------------------------------- phase 2, on top ----
    generators = make_infeasible_generators(
        state_min=jnp.asarray(box.state_min),
        state_max=jnp.asarray(box.state_max),
        perturbation_scale_per_dim=jnp.asarray(cfg.data.perturbation_scale_per_dim),
        step_fn=_true_step_fn(cfg.physics.dt, true_params, condition["true_system"]),
        control_min=jnp.asarray(box.control_min),
        control_max=jnp.asarray(box.control_max),
    )
    discriminator = Discriminator(true_physics.state_dim, hidden=HIDDEN, key=disc_key)
    full, discriminator, history, converged = _train_phase2(
        control,
        discriminator,
        train_batches,
        generators,
        epochs=p2_epochs,
        lr_ae=LR_AE,
        lr_disc=LR_DISC,
        n_latent_samples=a.latent_samples,
        key=p2_key,
    )
    full_ckpt = _save(full, cell_dir, p2_key)
    save_config(replace(cfg, fab_baseline=replace(cfg.fab_baseline, num_epochs=p2_epochs)),
                str(cell_dir / "config.json"))

    print(f"wrote {full_ckpt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
