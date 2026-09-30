"""Frozen configuration dataclasses and JSON helpers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

_VALID_PHASE2_PERTURBATIONS: frozenset[str] = frozenset({"none", "bound_violation", "gaussian"})
_VALID_PHASE2_SCALE_SAMPLING: frozenset[str] = frozenset({"fixed", "loguniform", "uniform"})

# The discrete `variant` field on ExperimentConfig is the source of truth
# for dispatch in scripts/ind/train_made.py. The set is closed — unknown
# variants raise at dispatch time. Underscore-canonical (not hyphenated) so
# each value is a valid Python identifier we can match in a dict.
_LEGAL_VARIANTS: frozenset[str] = frozenset({
    "made_phase1",
    "made_phase2",
    "made_no_residual",
    "made_no_corrector",
    "mlp",
    "fab",
    "clamp",
})


@dataclass(frozen=True)
class PhysicsConfig:
    """Configuration for the physics layer."""

    true_system: str = "double_integrator"
    dt: float = 0.1
    true_params: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for MaDE network widths."""

    inverse_hidden: tuple[int, ...] = (256, 256)
    residual_hidden: tuple[int, ...] = (256, 256)
    encoder_hidden: tuple[int, ...] = (64, 64)
    residual: str = "learned"
    residual_init_scale: float = 0.0
    use_inverse_residual: bool = True
    use_metadata_encoder: bool = False
    metadata_dim: int = 0
    known_system: str | None = None
    known_params: dict[str, float] = field(default_factory=dict)
    # Location embedding fields (inD). When num_locations <= 0 the
    # MetadataEncoder uses scalar-only mode (backward-compatible default).
    num_locations: int = 0
    embedding_dim: int = 8
    # The metadata encoder's output map. Default False keeps
    # `sigmoid(mlp(...)) * param_scales`, which is what every existing checkpoint was trained
    # under and what E01 uses, so flag-off is bit-identical.
    #
    # True makes it `softplus(mlp(...))`: positive, as a wheelbase must be, and unbounded above.
    # This avoids a bound whose consequence is that, since `param_scales` defaults to ones on
    # the inD path, the learned wheelbase could never reach 1 m against a physical ~3 m.
    #
    # No initial offset is added. softplus(0) = 0.693 against the old sigmoid(0) * 1.0 = 0.5, so
    # training already starts within 0.2 m of the old regime; an offset would be an unwanted
    # extra tuning knob.
    encoder_unbounded_scale: bool = False
    # Supersedes the softplus design above. The inD known model is the
    # kinematic bicycle at L_REF = 2.7 m, and that one constant serves MaDE's internal completion
    # AND every scorer -- so the wheelbase is no longer something the encoder has to discover.
    # The encoder instead emits a signed, unbounded residual and the model uses
    #     L = L_REF + mlp(inputs)
    # per vehicle: no sigmoid, no scale, and deliberately no clamp (the residual can be positive
    # or negative). A clamp would hide a real failure mode, so if any vehicle's L reaches <= 0
    # that is surfaced, not guarded.
    #
    # `encoder_unbounded_scale` is kept in the code but is not used by any run.
    encoder_lref_residual: bool = False
    location_id_index: int = 4


@dataclass(frozen=True)
class CorrectorConfig:
    """Configuration for the inequality corrector."""

    mode: str = "enabled"
    train_steps: int = 5
    eval_max_steps: int = 50
    eval_tol: float = 1e-6
    step_size: float = 0.01
    momentum: float = 0.0
    # Proximity term: J_gamma(u) = ||ReLU(g(T(x,u),u))||^2 + gamma||u - u_hat||^2.
    # Default 0.0 and evaluation-only: the term is inference-only on frozen
    # models, so it is constructed only on the evaluation path -- the training path does
    # not build it at all, rather than building it and multiplying by zero.
    proximity_gamma: float = 0.0
    # The tracking term: the same gamma*||u - u_hat||^2, promoted into the training path so
    # phase 2 differentiates through a corrector whose fixed point has moved.
    #
    # Deliberately a second field rather than a reuse of proximity_gamma, since
    # proximity_gamma is evaluation-time only. A shared field would mean sweeping
    # proximity_gamma silently changed what phase 2 differentiates through -- reopening
    # that question invisibly, since the sweep would still run and still produce a curve.
    # Two fields keep the two claims separable.
    tracking_gamma: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for MaDE training."""

    seed: int = 0
    batch_size: int = 128
    steps_per_epoch: int | None = None
    val_steps_per_epoch: int | None = None
    num_epochs_phase1: int = 15
    num_epochs_phase2: int = 15
    lr_I: float = 1e-3
    lr_T: float = 1e-3
    lambda_min_norm: float = 0.01
    lambda_ineq: float = 1.0
    # When True, L_ineq is stop_gradient'd in the T-side Phase-2 loss, so the
    # inequality term supervises I_phi alone and T_theta sees L_fwd + lambda_a R_a
    # only. Default False preserves the published behaviour exactly -- every published
    # result was produced with the gradient reaching T_theta.
    isolate_ineq_gradient: bool = False
    lambda_inv_consistency: float = 1.0
    lambda_delta_i_norm: float = 0.01
    alternation_period: int = 10
    control_sampling: str = "mixture"
    inverse_training: str = "cycle"
    solver_train: str = "heun"
    # CLAUDE.md mandates Heun() + ConstantStepSize() everywhere — adaptive
    # solvers exhaust max_steps on stiff tan(δ) near ±π/2.  Per-config
    # overrides may still set "tsit5" for ablations, but the safe default
    # matches the project-wide convention.
    solver_eval: str = "heun"
    early_stopping_enabled: bool = False
    early_stopping_min_epochs: int = 0
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    early_stopping_physical_exact: bool = False
    early_stopping_physical_saturation: bool = False
    early_stopping_forward_tol: float = 1e-6
    # Phase 2 starts from whatever state phase 1 ended in, which is the phase-1 final model, not
    # its best. Where early stopping selected an earlier epoch, phase 2 therefore begins from a
    # model the run itself judged worse. With this on, the phase-1 best checkpoint is restored
    # before phase 2 begins.
    #
    # Default stays False. Every existing config and every published result keeps today's
    # behaviour; the option is set explicitly on the runs that need it, so no number moves by
    # accident.
    restore_phase1_best_before_phase2: bool = False
    early_stopping_inverse_tol: float = 1e-6
    early_stopping_minimum_norm_tol: float = 1e-6
    early_stopping_delta_i_norm_tol: float = 1e-6
    early_stopping_saturation_window: int = 5
    early_stopping_saturation_delta: float = 1e-5
    t_side_grad_clip_norm: float | None = 1.0
    i_side_grad_clip_norm: float | None = 1.0
    warmup_steps: int = 100
    zero_nans_enabled: bool = True
    phase2_proposal_perturbation_type: str = "none"
    phase2_proposal_perturbation_scale: float = 0.0
    # Per-dimension proposal-corruption magnitudes for Phase 2.
    #
    # When None (default) the scalar `phase2_proposal_perturbation_scale` above
    # applies and is interpreted as a FRACTION OF THE BOX RANGE per dimension
    # (`scale * (state_max - state_min)`); dimensions whose range is infinite
    # receive zero perturbation. Existing configs are bit-unchanged.
    #
    # When set, this tuple OVERRIDES the scalar and its entries are ABSOLUTE
    # per-dimension magnitudes in state units — sigma for the "gaussian" type,
    # outward push distance for "bound_violation". An entry of 0.0 means that
    # dimension is NEVER perturbed. No box range is consulted on this path, so
    # dimensions with infinite bounds can still be corrupted.
    #
    # Entries must be finite and >= 0 (validated here); the length is checked
    # against the state dimension at use time in
    # `made.training.trainer._compute_x_proposal`, because the config does not
    # know the state dim.
    phase2_proposal_perturbation_scale_per_dim: tuple[float, ...] | None = None
    # Per-sample / per-dimension randomisation of the Phase-2 "gaussian"
    # proposal corruption (domain randomisation). Gaussian-only: the
    # deterministic "bound_violation" family raises rather than silently
    # ignoring them, so a config can never claim randomisation it does not get.
    #
    # `..._scale_per_dim` (or the scalar fallback) becomes the UPPER bound
    # sigma_max per dimension. The realised sigma is `sigma_max[d] * m[i, d]`,
    # with the multiplier m drawn independently per batch element i and per
    # state dimension d:
    #
    #   "fixed"      -> m == 1 exactly; today's behaviour, bit-for-bit, and no
    #                   additional randomness is consumed.
    #   "loguniform" -> log m ~ U(log min_ratio, 0). Preferred for wide
    #                   brackets: plausible prediction errors span orders of
    #                   magnitude, and uniform sampling puts most mass at the
    #                   large end, under-training the near-feasible regime.
    #   "uniform"    -> m ~ U(min_ratio, 1). Offered for ablation.
    #
    # `..._zero_fraction` is P(sample is left exactly clean), drawn per batch
    # element and applied to the WHOLE state (not per dimension) -- the guard
    # pins identity behaviour on feasible inputs, which requires a clean state.
    #
    # A dimension with sigma_max == 0.0 stays exactly unperturbed under every
    # sampling mode.
    phase2_proposal_perturbation_scale_sampling: str = "fixed"
    phase2_proposal_perturbation_scale_min_ratio: float = 0.0
    phase2_proposal_perturbation_zero_fraction: float = 0.0
    pretrained_phase1_path: str | None = None
    validation_interval_epochs: int = 1
    checkpoint_save_interval: int | None = None
    best_checkpoint_interval_epochs: int = 1
    metric_log_interval_steps: int = 1
    # True for inD/field-data configs; enables Phase-2 resume constraints-factory guard in trainer.train()
    is_field_data: bool = False

    def __post_init__(self) -> None:
        if self.steps_per_epoch is not None and self.steps_per_epoch <= 0:
            raise ValueError(
                f"steps_per_epoch must be positive or None, got {self.steps_per_epoch}"
            )
        if self.val_steps_per_epoch is not None and self.val_steps_per_epoch <= 0:
            raise ValueError(
                "val_steps_per_epoch must be positive or None, "
                f"got {self.val_steps_per_epoch}"
            )
        if not math.isfinite(self.lambda_delta_i_norm) or self.lambda_delta_i_norm < 0.0:
            raise ValueError("lambda_delta_i_norm must be non-negative")
        if self.early_stopping_min_epochs < 0:
            raise ValueError("early_stopping_min_epochs must be non-negative")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive or None")
        if not math.isfinite(self.early_stopping_min_delta) or self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if self.early_stopping_saturation_window <= 1:
            raise ValueError("early_stopping_saturation_window must be greater than 1")
        if (
            not math.isfinite(self.early_stopping_saturation_delta)
            or self.early_stopping_saturation_delta < 0.0
        ):
            raise ValueError("early_stopping_saturation_delta must be non-negative")
        tolerances = {
            "early_stopping_forward_tol": self.early_stopping_forward_tol,
            "early_stopping_inverse_tol": self.early_stopping_inverse_tol,
            "early_stopping_minimum_norm_tol": self.early_stopping_minimum_norm_tol,
            "early_stopping_delta_i_norm_tol": self.early_stopping_delta_i_norm_tol,
        }
        for name, value in tolerances.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.validation_interval_epochs < 1:
            raise ValueError("validation_interval_epochs must be >= 1")
        if self.best_checkpoint_interval_epochs < 1:
            raise ValueError("best_checkpoint_interval_epochs must be >= 1")
        if self.checkpoint_save_interval is not None and self.checkpoint_save_interval < 1:
            raise ValueError("checkpoint_save_interval must be >= 1 when set")
        if self.metric_log_interval_steps < 1:
            raise ValueError("metric_log_interval_steps must be >= 1")
        if self.phase2_proposal_perturbation_type not in _VALID_PHASE2_PERTURBATIONS:
            raise ValueError(
                f"phase2_proposal_perturbation_type must be one of "
                f"{sorted(_VALID_PHASE2_PERTURBATIONS)}, "
                f"got {self.phase2_proposal_perturbation_type!r}"
            )
        if (
            not math.isfinite(self.phase2_proposal_perturbation_scale)
            or self.phase2_proposal_perturbation_scale < 0.0
        ):
            raise ValueError("phase2_proposal_perturbation_scale must be non-negative")
        if self.phase2_proposal_perturbation_scale_per_dim is not None:
            for entry in self.phase2_proposal_perturbation_scale_per_dim:
                if not math.isfinite(entry) or entry < 0.0:
                    raise ValueError(
                        "phase2_proposal_perturbation_scale_per_dim entries must be finite "
                        "and non-negative, got "
                        f"{self.phase2_proposal_perturbation_scale_per_dim!r}"
                    )
        _sampling = self.phase2_proposal_perturbation_scale_sampling
        if _sampling not in _VALID_PHASE2_SCALE_SAMPLING:
            raise ValueError(
                "phase2_proposal_perturbation_scale_sampling must be one of "
                f"{sorted(_VALID_PHASE2_SCALE_SAMPLING)}, got {_sampling!r}"
            )
        _min_ratio = self.phase2_proposal_perturbation_scale_min_ratio
        if not math.isfinite(_min_ratio) or not (0.0 <= _min_ratio <= 1.0):
            raise ValueError(
                "phase2_proposal_perturbation_scale_min_ratio must be finite and in "
                f"[0, 1], got {_min_ratio!r}"
            )
        if _sampling != "fixed" and _min_ratio <= 0.0:
            # Log-uniform has no support at zero, and a zero lower bound would
            # make the "uniform" ablation non-comparable with it. Exact-zero
            # mass belongs in phase2_proposal_perturbation_zero_fraction.
            raise ValueError(
                "phase2_proposal_perturbation_scale_min_ratio must be in (0, 1] when "
                f"phase2_proposal_perturbation_scale_sampling is {_sampling!r}; use "
                "phase2_proposal_perturbation_zero_fraction for exact-zero mass, got "
                f"{_min_ratio!r}"
            )
        _zero_fraction = self.phase2_proposal_perturbation_zero_fraction
        if not math.isfinite(_zero_fraction) or not (0.0 <= _zero_fraction <= 1.0):
            raise ValueError(
                "phase2_proposal_perturbation_zero_fraction must be finite and in "
                f"[0, 1], got {_zero_fraction!r}"
            )
        if self.pretrained_phase1_path is not None and not self.pretrained_phase1_path:
            raise ValueError("pretrained_phase1_path must be a non-empty string or None")


_VALID_CONTROL_PROFILES: frozenset[str] = frozenset({"iid_uniform", "smooth_ou"})


@dataclass(frozen=True)
class DataConfig:
    """Configuration for dataset generation and perturbations."""

    num_trajectories_train: int = 1024
    num_trajectories_val: int = 128
    num_trajectories_test: int = 128
    trajectory_length: int = 32
    perturbation_type: str = "gaussian"
    perturbation_scale: float = 0.0
    perturbation_scale_per_dim: tuple[float, ...] | None = None
    noise_scale: float = 0.0
    # "iid_uniform" (default) reproduces E01's canonical white-noise-steering
    # datasets byte-for-byte. "smooth_ou" opts into a temporally-correlated
    # Ornstein-Uhlenbeck control profile for inD-like calm driving (E05).
    control_profile: str = "iid_uniform"
    control_tau: float = 1.5
    # Minimum longitudinal speed a generated trajectory must maintain at every
    # timestep (dynamic-bicycle layout only; index 3 is only speed for that
    # layout). None (default) disables the floor and reproduces byte-identical
    # legacy generation.
    min_speed: float | None = None
    # The control SAMPLING box, per dimension.
    #
    # The dynamic-bicycle sampler narrowed steering to +-0.2 and acceleration to +-1.5 inside a
    # constraint set of +-0.5 and +-3.0, hardcoded and duplicated in both control samplers, so
    # that generated data does not sit trivially at the edges of its own scored box.
    #
    # None (default) keeps that hardcoded narrowing exactly, so every dataset not being
    # regenerated stays byte-for-byte reproducible. Set it and the sampler uses these bounds
    # instead, with the special case skipped rather than retuned.
    control_sample_min: tuple[float, ...] | None = None
    control_sample_max: tuple[float, ...] | None = None
    # Observation noise added at GENERATION time, on every split.
    # `noise_scale` above is the magnitude; this switches it on at generation rather than only
    # at evaluation. False (default) reproduces existing generation, where noise_scale was
    # carried in the metadata of every dataset and never applied at generation.
    add_generation_noise: bool = False

    def __post_init__(self) -> None:
        for name in ("control_sample_min", "control_sample_max"):
            v = getattr(self, name)
            if v is not None and not all(math.isfinite(x) for x in v):
                raise ValueError(f"{name} entries must be finite")
        lo, hi = self.control_sample_min, self.control_sample_max
        if (lo is None) != (hi is None):
            raise ValueError(
                "control_sample_min and control_sample_max must be set together or both None"
            )
        if lo is not None and hi is not None:
            if len(lo) != len(hi):
                raise ValueError("control_sample_min and control_sample_max must be same length")
            if any(a >= b for a, b in zip(lo, hi)):
                raise ValueError("control_sample_min must be strictly below control_sample_max")
        if self.add_generation_noise and self.noise_scale <= 0.0:
            raise ValueError(
                "add_generation_noise=True requires a positive noise_scale; otherwise the flag "
                "claims noise the data does not carry"
            )
        if self.control_profile not in _VALID_CONTROL_PROFILES:
            raise ValueError(
                f"control_profile must be one of {sorted(_VALID_CONTROL_PROFILES)}, "
                f"got {self.control_profile!r}"
            )
        if not math.isfinite(self.control_tau) or self.control_tau <= 0.0:
            raise ValueError("control_tau must be positive")
        if self.min_speed is not None and (
            not math.isfinite(self.min_speed) or self.min_speed <= 0.0
        ):
            raise ValueError("min_speed must be positive or None")


@dataclass(frozen=True)
class UpstreamConfig:
    """Configuration for upstream Stage 1 (MSE + soft-constraint pretraining)."""

    stage1_mu_ineq: float = 1.0
    stage1_mu_dyn: float = 1.0
    stage1_epochs: int = 25
    # MSE only, strictly. "full" would add an inequality penalty and a discretised
    # dynamics penalty at unit weight, which would make every predictor in the paper
    # dynamically consistent with the known model -- the property MaDE exists to supply.
    # "mse_ineq" and "full" remain as dead options; the default must stay "mse".
    stage1_loss: str = "mse"


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for evaluation (perturbation seed, etc.)."""

    perturbation_seed: int = 42


def _baseline_post_init(cfg: Any) -> None:
    """Shared validation for all *BaselineConfig dataclasses."""
    if cfg.lr <= 0:
        raise ValueError(f"{type(cfg).__name__}.lr must be > 0")
    if cfg.num_epochs < 1:
        raise ValueError(f"{type(cfg).__name__}.num_epochs must be >= 1")
    if cfg.steps_per_epoch is not None and cfg.steps_per_epoch < 1:
        raise ValueError(f"{type(cfg).__name__}.steps_per_epoch must be >= 1 when set")
    if cfg.es_patience < 0:
        raise ValueError(f"{type(cfg).__name__}.es_patience must be >= 0")
    if cfg.es_min_delta < 0:
        raise ValueError(f"{type(cfg).__name__}.es_min_delta must be >= 0")
    if cfg.es_min_epochs < 0:
        raise ValueError(f"{type(cfg).__name__}.es_min_epochs must be >= 0")


@dataclass(frozen=True)
class MLPBaselineConfig:
    """Training configuration for the MLP correction baseline."""

    lr: float = 1e-3
    num_epochs: int = 100
    steps_per_epoch: int | None = None
    es_patience: int = 0
    es_min_delta: float = 1e-4
    es_min_epochs: int = 0

    def __post_init__(self) -> None:
        _baseline_post_init(self)


@dataclass(frozen=True)
class FABBaselineConfig:
    """Training configuration for the FAB latent-projection baseline."""

    lr: float = 1e-3
    num_epochs: int = 100
    steps_per_epoch: int | None = None
    es_patience: int = 0
    es_min_delta: float = 1e-4
    es_min_epochs: int = 0
    # Standardise the encoder's input by training-split statistics, and carry those
    # statistics into inference by storing them on the module.
    #
    # Default False. Every published FAB number was produced without it, and flipping the
    # default would silently restate them. Callers that want normalisation pass it explicitly.
    normalise_inputs: bool = False

    def __post_init__(self) -> None:
        _baseline_post_init(self)


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration.

    The ``variant`` field is the dispatch oracle for ``scripts/ind/train_made.py``.
    Legal values live in :data:`_LEGAL_VARIANTS`. Default is ``"made_phase2"``
    (the most common path) so legacy configs that omit the field still parse
    and behave as before.
    """

    experiment_name: str = "default"
    output_dir: str = "outputs/default"
    variant: str = "made_phase2"
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    corrector: CorrectorConfig = field(default_factory=CorrectorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    upstream: UpstreamConfig = field(default_factory=UpstreamConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    mlp_baseline: MLPBaselineConfig = field(default_factory=MLPBaselineConfig)
    fab_baseline: FABBaselineConfig = field(default_factory=FABBaselineConfig)


def _migrate_schema(payload: dict[str, Any]) -> dict[str, Any]:
    """Migrate old-schema JSON fields to current names."""
    payload = dict(payload)

    # Migrate physics.system -> physics.true_system
    physics = dict(payload.get("physics", {}))
    if "system" in physics and "true_system" not in physics:
        physics["true_system"] = physics.pop("system")
    payload["physics"] = physics

    # Add defaults for new model fields
    model = dict(payload.get("model", {}))
    model.setdefault("known_system", None)
    model.setdefault("known_params", {})
    model.setdefault("use_inverse_residual", True)
    # inD location embedding fields (backward-compat: disabled by default)
    model.setdefault("num_locations", 0)
    model.setdefault("embedding_dim", 8)
    model.setdefault("encoder_unbounded_scale", False)
    model.setdefault("encoder_lref_residual", False)
    model.setdefault("location_id_index", 4)
    payload["model"] = model

    # Add defaults for new evaluation section
    evaluation = dict(payload.get("evaluation", {}))
    evaluation.setdefault("perturbation_seed", 42)
    payload["evaluation"] = evaluation

    # Add defaults for capped epoch budgets.
    training = dict(payload.get("training", {}))
    training.setdefault("steps_per_epoch", None)
    training.setdefault("val_steps_per_epoch", None)
    training.setdefault("early_stopping_enabled", False)
    training.setdefault("early_stopping_min_epochs", 0)
    training.setdefault("early_stopping_patience", None)
    training.setdefault("early_stopping_min_delta", 0.0)
    training.setdefault("early_stopping_physical_exact", False)
    training.setdefault("early_stopping_physical_saturation", False)
    training.setdefault("restore_phase1_best_before_phase2", False)
    training.setdefault("early_stopping_forward_tol", 1e-6)
    training.setdefault("early_stopping_inverse_tol", 1e-6)
    training.setdefault("early_stopping_minimum_norm_tol", 1e-6)
    training.setdefault("early_stopping_delta_i_norm_tol", 1e-6)
    training.setdefault("early_stopping_saturation_window", 5)
    training.setdefault("early_stopping_saturation_delta", 1e-5)
    training.setdefault("t_side_grad_clip_norm", 1.0)
    training.setdefault("i_side_grad_clip_norm", 1.0)
    training.setdefault("warmup_steps", 100)
    training.setdefault("zero_nans_enabled", True)
    training.setdefault("phase2_proposal_perturbation_type", "none")
    training.setdefault("phase2_proposal_perturbation_scale", 0.0)
    training.setdefault("phase2_proposal_perturbation_scale_per_dim", None)
    training.setdefault("phase2_proposal_perturbation_scale_sampling", "fixed")
    training.setdefault("phase2_proposal_perturbation_scale_min_ratio", 0.0)
    training.setdefault("phase2_proposal_perturbation_zero_fraction", 0.0)
    training.setdefault("pretrained_phase1_path", None)
    training.setdefault("lambda_delta_i_norm", 0.01)
    training.setdefault("validation_interval_epochs", 1)
    training.setdefault("checkpoint_save_interval", None)
    training.setdefault("best_checkpoint_interval_epochs", 1)
    training.setdefault("metric_log_interval_steps", 1)
    training.setdefault("is_field_data", False)
    payload["training"] = training

    data = dict(payload.get("data", {}))
    data.setdefault("perturbation_scale_per_dim", None)
    data.setdefault("control_profile", "iid_uniform")
    data.setdefault("control_tau", 1.5)
    data.setdefault("control_sample_min", None)
    data.setdefault("control_sample_max", None)
    data.setdefault("add_generation_noise", False)
    data.setdefault("min_speed", None)
    payload["data"] = data

    for baseline_key in ("mlp_baseline", "fab_baseline"):
        baseline = dict(payload.get(baseline_key, {}))
        baseline.setdefault("es_min_epochs", 0)
        payload[baseline_key] = baseline

    return payload


def _coerce_tuple_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert JSON list fields back into tuple-backed config fields."""
    model_payload = dict(payload["model"])
    model_payload["inverse_hidden"] = tuple(model_payload["inverse_hidden"])
    model_payload["residual_hidden"] = tuple(model_payload["residual_hidden"])
    model_payload["encoder_hidden"] = tuple(model_payload["encoder_hidden"])

    data_payload = dict(payload["data"])
    if data_payload.get("perturbation_scale_per_dim") is not None:
        data_payload["perturbation_scale_per_dim"] = tuple(
            float(x) for x in data_payload["perturbation_scale_per_dim"]
        )

    training_payload = dict(payload["training"])
    if training_payload.get("phase2_proposal_perturbation_scale_per_dim") is not None:
        training_payload["phase2_proposal_perturbation_scale_per_dim"] = tuple(
            float(x) for x in training_payload["phase2_proposal_perturbation_scale_per_dim"]
        )

    payload = dict(payload)
    payload["model"] = model_payload
    payload["data"] = data_payload
    payload["training"] = training_payload
    return payload


def to_json(config: ExperimentConfig) -> str:
    """Serialise an experiment config to stable JSON."""
    return json.dumps(asdict(config), indent=2, sort_keys=True)


def from_json(json_str: str) -> ExperimentConfig:
    """Reconstruct an experiment config from JSON."""
    raw = json.loads(
        json_str,
        parse_constant=lambda s: (_ for _ in ()).throw(ValueError(f"Non-finite JSON value: {s}")),
    )
    raw = _migrate_schema(raw)
    payload = _coerce_tuple_fields(raw)
    return ExperimentConfig(
        experiment_name=payload["experiment_name"],
        output_dir=payload["output_dir"],
        variant=payload.get("variant", "made_phase2"),
        physics=PhysicsConfig(**payload["physics"]),
        model=ModelConfig(**payload["model"]),
        corrector=CorrectorConfig(**payload["corrector"]),
        training=TrainingConfig(**payload["training"]),
        data=DataConfig(**payload["data"]),
        upstream=UpstreamConfig(**payload["upstream"]),
        evaluation=EvaluationConfig(**payload.get("evaluation", {})),
        mlp_baseline=MLPBaselineConfig(**payload.get("mlp_baseline", {})),
        fab_baseline=FABBaselineConfig(**payload.get("fab_baseline", {})),
    )


def save_config(config: ExperimentConfig, path: str) -> None:
    """Save a config to disk as JSON."""
    Path(path).write_text(to_json(config) + "\n", encoding="utf-8")


def load_config(path: str) -> ExperimentConfig:
    """Load a config from disk."""
    return from_json(Path(path).read_text(encoding="utf-8"))


def override_config(config: ExperimentConfig, overrides: dict[str, Any]) -> ExperimentConfig:
    """Apply dot-separated overrides to nested frozen dataclasses."""
    updated: ExperimentConfig = config
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        if len(parts) != 2:
            raise ValueError(
                f"Override key '{dotted_key}' must have the form '<section>.<field>'."
            )
        section_name, field_name = parts
        section = getattr(updated, section_name)
        new_section = replace(section, **{field_name: value})
        updated = replace(updated, **{section_name: new_section})
    return updated
