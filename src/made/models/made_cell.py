# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Composed MaDE cell."""

from __future__ import annotations

import diffrax
import equinox as eqx
import jax

from made.models.augmented_dynamics import AugmentedDynamics, ResidualNetwork, ZeroResidual
from made.models.corrector import Corrector, CorrectorDiagnostics
from made.models.inverse_dynamics import InverseDynamics
from made.physics import ConstraintSet, PhysicsModel
from made.utils import CorrectorConfig, ModelConfig


class MaDECell(eqx.Module):
    """Full propose-complete-correct cell."""

    inverse_dynamics: InverseDynamics
    augmented_dynamics: AugmentedDynamics
    corrector: Corrector
    constraints: ConstraintSet
    corrector_mode: str = "enabled"

    def __call__(
        self,
        x_prev: jax.Array,
        x_curr: jax.Array,
        params: jax.Array,
        dt: float,
        *,
        training: bool = True,
        correction_mode: str | None = None,
        solver: diffrax.AbstractSolver | None = None,
        adjoint: diffrax.AbstractAdjoint | None = None,
        return_diagnostics: bool = False,
    ) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, CorrectorDiagnostics]:
        """Propose a control, integrate the dynamics, and correct.

        Args:
            x_prev: Previous state.
            x_curr: Current state.
            params: Physical parameters.
            dt: Step length.
            training: Selects the training or evaluation corrector loop.
            correction_mode: Explicit corrector mode, or None for the default.
            solver: Optional diffrax solver.
            adjoint: Optional diffrax adjoint.
            return_diagnostics: Also return corrector diagnostics.

        Returns:
            ``(x, u)``, plus ``CorrectorDiagnostics`` when requested.
        """
        u = self.inverse_dynamics(x_prev, x_curr, params)
        x_pred = self.augmented_dynamics.integrate(
            x_prev,
            u,
            params,
            dt,
            solver=solver,
            adjoint=adjoint,
        )
        mode = "it_only" if self.corrector_mode == "disabled" else correction_mode
        return self.corrector(
            x_pred,
            u,
            self.constraints,
            self.augmented_dynamics,
            x_prev,
            params,
            dt,
            training=training,
            mode=mode,
            solver=solver,
            adjoint=adjoint,
            return_diagnostics=return_diagnostics,
        )

    @classmethod
    def from_config(
        cls,
        physics: PhysicsModel,
        constraints: ConstraintSet,
        model_config: ModelConfig,
        corrector_config: CorrectorConfig,
        *,
        key: jax.Array,
        dt: float = 0.1,
    ) -> "MaDECell":
        """Build a cell from configs.

        Args:
            physics: Known physics model.
            constraints: Constraint set.
            model_config: Model configuration.
            corrector_config: Corrector configuration.
            key: PRNG key for weight initialisation.
            dt: Step length.

        Returns:
            A new ``MaDECell``.

        Raises:
            ValueError: If ``model_config.residual`` or ``corrector_config.mode`` is invalid.
        """
        inverse_key, residual_key = jax.random.split(key)
        if model_config.residual not in {"learned", "zero"}:
            raise ValueError("model_config.residual must be 'learned' or 'zero'.")
        if corrector_config.mode not in {"enabled", "disabled"}:
            raise ValueError("corrector_config.mode must be 'enabled' or 'disabled'.")
        inverse = InverseDynamics(
            physics.state_dim,
            physics.control_dim,
            physics.param_dim,
            model_config.inverse_hidden,
            key=inverse_key,
            known_physics=physics,
            dt=dt,
            use_residual=model_config.use_inverse_residual,
        )
        residual = (
            ZeroResidual(physics.state_dim)
            if model_config.residual == "zero"
            else ResidualNetwork(
                physics.state_dim,
                physics.control_dim,
                physics.param_dim,
                model_config.residual_hidden,
                init_scale=model_config.residual_init_scale,
                key=residual_key,
            )
        )
        augmented = AugmentedDynamics(physics=physics, residual=residual)
        corrector = Corrector(corrector_config)
        return cls(
            inverse_dynamics=inverse,
            augmented_dynamics=augmented,
            corrector=corrector,
            constraints=constraints,
            corrector_mode=corrector_config.mode,
        )
