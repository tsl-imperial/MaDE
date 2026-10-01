# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Thin model wrapper for optional metadata-backed parameter resolution."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import equinox as eqx
import jax

from made.models.encoder import MetadataEncoder
from made.models.made_cell import MaDECell
from made.physics import ConstraintSet, PhysicsModel
from made.utils import CorrectorConfig, ModelConfig

if TYPE_CHECKING:
    from made.models.augmented_dynamics import AugmentedDynamics
    from made.models.corrector import Corrector
    from made.models.inverse_dynamics import InverseDynamics


class MaDEModel(eqx.Module):
    """Own a MaDE cell plus an optional metadata encoder.

    The cell remains the single-step I-T-C map. This wrapper only resolves
    physics parameters from either known per-sample params or metadata.
    """

    cell: MaDECell
    encoder: MetadataEncoder | None

    @property
    def inverse_dynamics(self) -> InverseDynamics:
        """The cell's inverse-dynamics model.

        Returns:
            The inverse-dynamics module.
        """
        return self.cell.inverse_dynamics

    @property
    def augmented_dynamics(self) -> AugmentedDynamics:
        """The cell's augmented dynamics.

        Returns:
            The augmented-dynamics module.
        """
        return self.cell.augmented_dynamics

    @property
    def corrector(self) -> Corrector:
        """The cell's corrector.

        Returns:
            The corrector module.
        """
        return self.cell.corrector

    @property
    def constraints(self) -> ConstraintSet:
        """The cell's constraint set.

        Returns:
            The constraint set.
        """
        return self.cell.constraints

    def params_from_metadata(self, metadata: jax.Array) -> jax.Array:
        """Resolve physics parameters from one metadata vector or a metadata batch.

        Args:
            metadata: Metadata vector ``[metadata_dim]`` or batch ``[B, metadata_dim]``.

        Returns:
            Physics parameters with matching leading dimension.

        Raises:
            ValueError: If the model has no encoder.
        """
        if self.encoder is None:
            raise ValueError("Metadata was provided but this MaDEModel has no encoder.")
        if metadata.ndim == 1:
            return self.encoder(metadata)
        return jax.vmap(self.encoder)(metadata)

    def resolve_params(
        self,
        params: jax.Array | None = None,
        metadata: jax.Array | None = None,
    ) -> jax.Array:
        """Prefer known params, falling back to encoder-derived params when needed.

        Args:
            params: Known per-sample parameters, or None.
            metadata: Metadata for the encoder, or None.

        Returns:
            Resolved physics parameters.

        Raises:
            ValueError: If neither non-empty params nor metadata is available.
        """
        if params is not None and params.shape[-1] > 0:
            return params
        if metadata is not None:
            return self.params_from_metadata(metadata)
        if params is not None:
            return params
        raise ValueError("MaDEModel requires either non-empty params or metadata.")

    def __call__(
        self,
        x_prev: jax.Array,
        x_curr: jax.Array,
        params: jax.Array | None,
        dt: float,
        *,
        metadata: jax.Array | None = None,
        training: bool = True,
        correction_mode: str | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Run the cell after resolving parameters.

        Args:
            x_prev: Previous state.
            x_curr: Current state.
            params: Known parameters, or None to use ``metadata``.
            dt: Step length.
            metadata: Metadata for the encoder, or None.
            training: Selects the training or evaluation corrector loop.
            correction_mode: Explicit corrector mode, or None for the default.

        Returns:
            Tuple ``(x, u)``.
        """
        resolved_params = self.resolve_params(params, metadata)
        return self.cell(
            x_prev,
            x_curr,
            resolved_params,
            dt,
            training=training,
            correction_mode=correction_mode,
        )

    @classmethod
    def from_config(
        cls,
        physics: PhysicsModel,
        constraints: ConstraintSet,
        model_config: ModelConfig,
        corrector_config: CorrectorConfig,
        *,
        param_scales: jax.Array | None = None,
        key: jax.Array,
    ) -> "MaDEModel":
        """Construct a wrapper, creating an encoder only when configured.

        Args:
            physics: Known physics model.
            constraints: Constraint set.
            model_config: Model configuration.
            corrector_config: Corrector configuration.
            param_scales: Per-parameter scales for the encoder; ones when None.
            key: PRNG key for weight initialisation.

        Returns:
            A new ``MaDEModel``.

        Raises:
            ValueError: If the encoder is requested with a non-positive ``metadata_dim``.
        """
        cell_key, encoder_key = jax.random.split(key)
        cell = MaDECell.from_config(
            physics,
            constraints,
            model_config,
            corrector_config,
            key=cell_key,
        )
        encoder = None
        if model_config.use_metadata_encoder:
            if model_config.metadata_dim <= 0:
                raise ValueError("metadata_dim must be positive when use_metadata_encoder=True.")
            scales = param_scales
            if scales is None:
                import jax.numpy as jnp

                scales = jnp.ones((physics.param_dim,))
            encoder = MetadataEncoder(
                model_config.metadata_dim,
                physics.param_dim,
                model_config.encoder_hidden,
                scales,
                num_locations=model_config.num_locations,
                embedding_dim=model_config.embedding_dim,
                unbounded_scale=model_config.encoder_unbounded_scale,
                lref_residual=model_config.encoder_lref_residual,
                location_id_index=model_config.location_id_index,
                key=encoder_key,
            )
        return cls(cell=cell, encoder=encoder)

    @staticmethod
    def _best_validation_step(path: str) -> int | None:
        """The step recorded as ``early_stop_best_step`` by the run that wrote ``path``.

        Read from the HIGHEST step's ``train_meta.json``, which is the most recent record
        of the run's early-stopping state. Returns ``None`` when the field is absent (a
        checkpoint written before this was persisted) or when no checkpoint exists at that
        step, so callers can fall back rather than fail.

        Args:
            path: Checkpoint directory.

        Returns:
            The best step, or None.
        """
        import json

        from made.utils.checkpointing import CheckpointManager

        root = Path(path)
        latest = CheckpointManager(path).latest_step()
        if latest is None:
            return None
        meta_path = root / str(latest) / "train_meta.json"
        if not meta_path.exists():
            return None
        try:
            best = json.loads(meta_path.read_text(encoding="utf-8")).get("early_stop_best_step")
        except (json.JSONDecodeError, OSError):
            return None
        if best is None:
            return None
        best = int(best)
        return best if (root / str(best)).is_dir() else None

    @classmethod
    def from_checkpoint(cls, path: str, *, select: str = "best") -> "MaDEModel":
        """Load a MaDEModel from a checkpoint directory.

        Expects the checkpoint layout written by ``CheckpointManager.save``: a numeric step
        subdirectory containing ``state.pkl``.

        ``select``:

        - ``"best"`` (default) restores the best-validation checkpoint, the step persisted as
          ``early_stop_best_step``, selected on the validation loss of the training objective.
        - ``"last"`` restores the highest available step. Reproducing a published number
          requires this, since published artifacts were evaluated at the final step: on inD
          seed 0 the two differ by roughly 15 epochs and 183,000 steps.

        Falls back to the highest step, with a warning, when ``"best"`` is asked for but no
        ``early_stop_best_step`` is recorded or its checkpoint is absent, so a checkpoint
        written before this field existed still loads.

        Args:
            path: Checkpoint directory.
            select: ``"best"`` or ``"last"``.

        Returns:
            The restored model.

        Raises:
            ValueError: If ``select`` is not ``"best"`` or ``"last"``.
            FileNotFoundError: If no checkpoint exists at ``path``.
            TypeError: If the checkpoint does not contain a ``MaDEModel``.
        """
        import sys

        from made.utils.checkpointing import CheckpointManager

        if select not in {"best", "last"}:
            raise ValueError(f"select must be 'best' or 'last'; got {select!r}")

        cm = CheckpointManager(path)
        step = None
        if select == "best":
            step = cls._best_validation_step(path)
            if step is None:
                print(
                    f"[MaDEModel.from_checkpoint] select='best' but no usable "
                    f"early_stop_best_step at {path!r}; falling back to the highest step.",
                    file=sys.stderr,
                )
        state = cm.restore(step)
        if state is None:
            raise FileNotFoundError(
                f"No checkpoint found at {path!r}. "
                "Run training first to produce a checkpoint."
            )
        model = state.model
        if not isinstance(model, cls):
            raise TypeError(
                f"Checkpoint at {path!r} contains a {type(model).__name__}, "
                f"expected {cls.__name__}."
            )
        return model
