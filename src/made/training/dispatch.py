"""Variant dispatch helper for E01 experiments."""

from __future__ import annotations

from dataclasses import replace

import jax

from made.baselines import FABBaseline, MLPBaseline
from made.models import MaDECell
from made.physics import build_system, build_system_for_model
from made.utils.config import ExperimentConfig

_ALL_VARIANTS: frozenset[str] = frozenset({
    "made",
    "made-no-residual",
    "made-no-corrector",
    "made-supervised-i",
    "made-fixed-i",
    "mlp",
    "fab",
    "clamp",
})


def build_trainable(cfg: ExperimentConfig, variant: str, key: jax.Array):
    """Return an initialised but untrained model for *variant*, or None for clamp.

    Args:
        cfg: Experiment configuration.  ``cfg.physics.true_system`` and
             ``cfg.model.known_system`` drive physics selection.
        variant: One of the E01 variants.
        key: PRNG key for model initialisation.

    Returns:
        An ``eqx.Module`` instance, or ``None`` for the clamp variant.

    Raises:
        ValueError: For unknown variant names.
    """
    if variant not in _ALL_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Supported: {sorted(_ALL_VARIANTS)}")

    true_system_name = cfg.physics.true_system
    known_system_name = cfg.model.known_system or true_system_name
    true_physics, _ = build_system(true_system_name)
    known_physics, known_constraints = build_system_for_model(true_system_name, known_system_name)

    if variant.startswith("made"):
        model_cfg = cfg.model
        corrector_cfg = cfg.corrector
        if variant == "made-no-residual":
            model_cfg = replace(model_cfg, residual="zero")
        elif variant == "made-no-corrector":
            corrector_cfg = replace(corrector_cfg, mode="disabled")
        elif variant == "made-fixed-i":
            model_cfg = replace(model_cfg, use_inverse_residual=False)
        return MaDECell.from_config(
            known_physics, known_constraints, model_cfg, corrector_cfg, key=key,
            dt=cfg.physics.dt,
        )

    # Location-embedding fields are forwarded from cfg.model so the MLP
    # baseline sees the integer ``location_id`` column the same way MaDE does.
    metadata_dim = cfg.model.metadata_dim if cfg.model.use_metadata_encoder else 0
    num_locations = cfg.model.num_locations
    embedding_dim = cfg.model.embedding_dim
    location_id_index = cfg.model.location_id_index

    if variant == "mlp":
        return MLPBaseline(
            state_dim=true_physics.state_dim,
            metadata_dim=metadata_dim,
            num_locations=num_locations,
            embedding_dim=embedding_dim,
            location_id_index=location_id_index,
            key=key,
        )

    if variant == "fab":
        # FAB does not consume metadata; location embedding is irrelevant here.
        return FABBaseline(state_dim=true_physics.state_dim, key=key)

    return None  # clamp
