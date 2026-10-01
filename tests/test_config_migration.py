# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Tests for from_json migration shim: old-schema JSON loads correctly."""

import json
from dataclasses import replace

import pytest

from made.utils.config import from_json, to_json, ExperimentConfig, EvaluationConfig, TrainingConfig


_OLD_SCHEMA_MINIMAL = {
    "experiment_name": "test",
    "output_dir": "outputs/test",
    "physics": {
        "system": "double_integrator",
        "dt": 0.1,
        "true_params": {},
    },
    "model": {
        "inverse_hidden": [256, 256],
        "residual_hidden": [256, 256],
        "encoder_hidden": [64, 64],
        "residual": "learned",
        "residual_init_scale": 0.0,
        "use_metadata_encoder": False,
        "metadata_dim": 0,
    },
    "corrector": {
        "mode": "enabled",
        "train_steps": 5,
        "eval_max_steps": 50,
        "eval_tol": 1e-6,
        "step_size": 0.01,
        "momentum": 0.0,
    },
    "training": {
        "seed": 0,
        "batch_size": 128,
        "num_epochs_phase1": 100,
        "num_epochs_phase2": 100,
        "lr_I": 1e-3,
        "lr_T": 1e-3,
        "lambda_min_norm": 1e-4,
        "lambda_ineq": 1.0,
        "lambda_inv_consistency": 1.0,
        "alternation_period": 10,
        "control_sampling": "mixture",
        "inverse_training": "cycle",
        "solver_train": "heun",
        "solver_eval": "tsit5",
    },
    "data": {
        "num_trajectories_train": 1024,
        "num_trajectories_val": 128,
        "num_trajectories_test": 128,
        "trajectory_length": 32,
        "perturbation_type": "gaussian",
        "perturbation_scale": 0.0,
        "noise_scale": 0.0,
    },
    "upstream": {
        "stage1_mu_ineq": 1.0,
        "stage1_mu_dyn": 1.0,
        "stage1_epochs": 25,
        "stage1_loss": "full",
    },
}


def test_old_schema_loads() -> None:
    """Old-schema JSON with 'physics.system' key loads without error."""
    cfg = from_json(json.dumps(_OLD_SCHEMA_MINIMAL))
    assert cfg.physics.true_system == "double_integrator"


def test_old_schema_known_system_defaults_none() -> None:
    """Missing model.known_system in old schema defaults to None."""
    cfg = from_json(json.dumps(_OLD_SCHEMA_MINIMAL))
    assert cfg.model.known_system is None


def test_old_schema_known_params_defaults_empty() -> None:
    """Missing model.known_params in old schema defaults to {}."""
    cfg = from_json(json.dumps(_OLD_SCHEMA_MINIMAL))
    assert cfg.model.known_params == {}


def test_old_schema_use_inverse_residual_defaults_true() -> None:
    """Missing model.use_inverse_residual in old schema defaults to True."""
    cfg = from_json(json.dumps(_OLD_SCHEMA_MINIMAL))
    assert cfg.model.use_inverse_residual is True


def test_old_schema_perturbation_seed_defaults() -> None:
    """Missing evaluation section in old schema defaults perturbation_seed to 42."""
    cfg = from_json(json.dumps(_OLD_SCHEMA_MINIMAL))
    assert cfg.evaluation.perturbation_seed == 42


def test_old_schema_no_evaluation_section() -> None:
    """Old schema without 'evaluation' key entirely still loads."""
    payload = dict(_OLD_SCHEMA_MINIMAL)
    payload.pop("evaluation", None)
    cfg = from_json(json.dumps(payload))
    assert cfg.evaluation == EvaluationConfig()


def test_new_schema_unaffected() -> None:
    """New-schema JSON with 'true_system' roundtrips cleanly (migration is no-op)."""
    original = ExperimentConfig(
        physics=__import__("made.utils.config", fromlist=["PhysicsConfig"]).PhysicsConfig(
            true_system="kinematic_bicycle"
        )
    )
    restored = from_json(to_json(original))
    assert restored.physics.true_system == "kinematic_bicycle"


def test_old_schema_training_step_budget_defaults_none() -> None:
    """Old-schema JSON without capped epoch fields loads with uncapped defaults."""
    cfg = from_json(json.dumps(_OLD_SCHEMA_MINIMAL))
    assert cfg.training.steps_per_epoch is None
    assert cfg.training.val_steps_per_epoch is None
    assert cfg.training.early_stopping_enabled is False
    assert cfg.training.early_stopping_patience is None


def test_old_schema_adds_new_regularization_defaults() -> None:
    """Checks old schema adds new regularization defaults."""
    cfg = from_json(json.dumps(_OLD_SCHEMA_MINIMAL))
    assert cfg.training.i_side_grad_clip_norm == 1.0
    assert cfg.training.lambda_delta_i_norm == 0.01
    assert cfg.training.t_side_grad_clip_norm == 1.0


def test_lambda_min_norm_in_old_schema_is_preserved() -> None:
    """Old schema explicitly sets lambda_min_norm=1e-4; that wins over the new default."""
    cfg = from_json(json.dumps(_OLD_SCHEMA_MINIMAL))
    assert cfg.training.lambda_min_norm == 1e-4


def test_default_training_config_uses_new_regularization() -> None:
    """Checks default training config uses new regularization."""
    cfg = TrainingConfig()
    assert cfg.lambda_min_norm == 0.01
    assert cfg.t_side_grad_clip_norm == 1.0
    assert cfg.i_side_grad_clip_norm == 1.0
    assert cfg.lambda_delta_i_norm == 0.01


def test_lambda_delta_i_norm_negative_rejected() -> None:
    """Checks lambda delta i norm negative rejected."""
    with pytest.raises(ValueError, match="lambda_delta_i_norm"):
        TrainingConfig(lambda_delta_i_norm=-0.1)


def test_training_step_budget_validation() -> None:
    """Checks training step budget validation."""
    with pytest.raises(ValueError, match="steps_per_epoch must be positive"):
        TrainingConfig(steps_per_epoch=0)
    with pytest.raises(ValueError, match="steps_per_epoch must be positive"):
        TrainingConfig(steps_per_epoch=-1)
    with pytest.raises(ValueError, match="val_steps_per_epoch must be positive"):
        TrainingConfig(val_steps_per_epoch=0)


def test_training_early_stopping_validation() -> None:
    """Checks training early stopping validation."""
    with pytest.raises(ValueError, match="early_stopping_patience must be positive"):
        TrainingConfig(early_stopping_patience=0)
    with pytest.raises(ValueError, match="early_stopping_saturation_window must be greater than 1"):
        TrainingConfig(early_stopping_saturation_window=1)
    with pytest.raises(ValueError, match="early_stopping_forward_tol must be non-negative"):
        TrainingConfig(early_stopping_forward_tol=-1.0)


def test_non_finite_json_rejected() -> None:
    """Checks non finite json rejected."""
    raw = to_json(ExperimentConfig())
    with pytest.raises(ValueError, match="Non-finite JSON value"):
        from_json(raw.replace('"early_stopping_min_delta": 0.0', '"early_stopping_min_delta": NaN'))


def test_training_step_budget_json_roundtrip() -> None:
    """Checks training step budget json roundtrip."""
    original = replace(
        ExperimentConfig(),
        training=replace(ExperimentConfig().training, steps_per_epoch=4, val_steps_per_epoch=2),
    )
    restored = from_json(to_json(original))
    assert restored.training.steps_per_epoch == 4
    assert restored.training.val_steps_per_epoch == 2


def test_old_schema_metric_log_interval_default() -> None:
    """Checks old schema metric log interval default."""
    cfg = from_json(json.dumps(_OLD_SCHEMA_MINIMAL))
    assert cfg.training.metric_log_interval_steps == 1


def test_metric_log_interval_validation() -> None:
    """Checks metric log interval validation."""
    with pytest.raises(ValueError, match="metric_log_interval_steps must be >= 1"):
        TrainingConfig(metric_log_interval_steps=0)


def test_old_schema_baseline_configs_default() -> None:
    """Old-schema JSON without baseline sections gets default config objects."""
    from made.utils.config import MLPBaselineConfig, FABBaselineConfig

    cfg = from_json(json.dumps(_OLD_SCHEMA_MINIMAL))
    assert cfg.mlp_baseline == MLPBaselineConfig()
    assert cfg.fab_baseline == FABBaselineConfig()


def test_baseline_config_roundtrip() -> None:
    """Custom baseline config roundtrips through JSON without loss."""
    from dataclasses import replace as dc_replace
    from made.utils.config import MLPBaselineConfig

    original = dc_replace(
        ExperimentConfig(),
        mlp_baseline=MLPBaselineConfig(lr=5e-4, num_epochs=20, es_patience=3, es_min_delta=1e-5),
    )
    restored = from_json(to_json(original))
    assert restored.mlp_baseline.lr == pytest.approx(5e-4)
    assert restored.mlp_baseline.num_epochs == 20
    assert restored.mlp_baseline.es_patience == 3
    assert restored.mlp_baseline.es_min_delta == pytest.approx(1e-5)


def test_baseline_config_validation() -> None:
    """Invalid baseline config fields raise ValueError."""
    from made.utils.config import MLPBaselineConfig, FABBaselineConfig

    with pytest.raises(ValueError, match="lr must be > 0"):
        MLPBaselineConfig(lr=0.0)
    with pytest.raises(ValueError, match="num_epochs must be >= 1"):
        FABBaselineConfig(num_epochs=0)


def test_default_training_config_has_warmup_steps() -> None:
    """Default TrainingConfig has warmup_steps=100."""
    cfg = TrainingConfig()
    assert cfg.warmup_steps == 100  # TrainingConfig is accessed directly here, not via ExperimentConfig


def test_old_schema_warmup_steps_defaults() -> None:
    """Old-schema JSON without warmup_steps defaults to 100 after migration."""
    cfg = from_json(json.dumps(_OLD_SCHEMA_MINIMAL))
    assert cfg.training.warmup_steps == 100


def test_warmup_steps_roundtrip() -> None:
    """warmup_steps survives a to_json/from_json roundtrip."""
    from dataclasses import replace as dc_replace
    original = dc_replace(ExperimentConfig(), training=TrainingConfig(warmup_steps=50))
    restored = from_json(to_json(original))
    assert restored.training.warmup_steps == 50


def test_make_lr_schedule_attenuates_first_step() -> None:
    """_make_lr_schedule(lr, N) returns a schedule that yields 0.0 at step 0 and lr at step N."""
    from made.training.trainer import _make_lr_schedule

    sched = _make_lr_schedule(1e-3, 100)
    assert callable(sched)
    assert float(sched(0)) == pytest.approx(0.0)
    assert float(sched(100)) == pytest.approx(1e-3)
    assert float(sched(50)) == pytest.approx(5e-4)
    # Past warmup, the schedule plateaus at end_value.
    assert float(sched(500)) == pytest.approx(1e-3)


def test_make_lr_schedule_disabled_returns_scalar() -> None:
    """warmup_steps=0 returns the scalar lr directly (bare optax.adam path)."""
    from made.training.trainer import _make_lr_schedule

    out = _make_lr_schedule(1e-3, 0)
    assert isinstance(out, float)
    assert out == pytest.approx(1e-3)


def test_phase2_proposal_perturbation_defaults() -> None:
    """Old-format JSON (no phase2_* keys) migrates to default perturbation fields."""
    cfg = from_json(json.dumps(_OLD_SCHEMA_MINIMAL))
    assert cfg.training.phase2_proposal_perturbation_type == "none"
    assert cfg.training.phase2_proposal_perturbation_scale == 0.0
    assert cfg.training.phase2_proposal_perturbation_scale_per_dim is None
    assert cfg.training.phase2_proposal_perturbation_scale_sampling == "fixed"
    assert cfg.training.phase2_proposal_perturbation_scale_min_ratio == 0.0
    assert cfg.training.phase2_proposal_perturbation_zero_fraction == 0.0
    assert cfg.training.pretrained_phase1_path is None


def test_phase2_proposal_perturbation_validation() -> None:
    """Invalid phase2 perturbation fields raise ValueError with the field name."""
    with pytest.raises(ValueError, match="phase2_proposal_perturbation_type"):
        TrainingConfig(phase2_proposal_perturbation_type="foo")
    with pytest.raises(ValueError, match="phase2_proposal_perturbation_scale"):
        TrainingConfig(phase2_proposal_perturbation_scale=-0.1)
    with pytest.raises(ValueError, match="pretrained_phase1_path"):
        TrainingConfig(pretrained_phase1_path="")


def test_phase2_proposal_perturbation_gaussian_is_accepted() -> None:
    """'gaussian' is a first-class perturbation type; unknown types still reject."""
    cfg = TrainingConfig(
        phase2_proposal_perturbation_type="gaussian",
        phase2_proposal_perturbation_scale=0.1,
    )
    assert cfg.phase2_proposal_perturbation_type == "gaussian"
    with pytest.raises(ValueError, match="phase2_proposal_perturbation_type"):
        TrainingConfig(phase2_proposal_perturbation_type="gausian")


def test_phase2_proposal_perturbation_per_dim_validation() -> None:
    """Negative and non-finite per-dimension entries are rejected."""
    with pytest.raises(ValueError, match="phase2_proposal_perturbation_scale_per_dim"):
        TrainingConfig(phase2_proposal_perturbation_scale_per_dim=(0.1, -0.2))
    with pytest.raises(ValueError, match="phase2_proposal_perturbation_scale_per_dim"):
        TrainingConfig(phase2_proposal_perturbation_scale_per_dim=(0.1, float("inf")))
    with pytest.raises(ValueError, match="phase2_proposal_perturbation_scale_per_dim"):
        TrainingConfig(phase2_proposal_perturbation_scale_per_dim=(float("nan"),))
    ok = TrainingConfig(phase2_proposal_perturbation_scale_per_dim=(0.0, 0.3, 0.0, 0.0))
    assert ok.phase2_proposal_perturbation_scale_per_dim == (0.0, 0.3, 0.0, 0.0)


def test_phase2_proposal_perturbation_round_trip() -> None:
    """phase2 perturbation fields survive a to_json/from_json round-trip."""
    from dataclasses import replace as dc_replace

    original = dc_replace(
        ExperimentConfig(),
        training=TrainingConfig(
            phase2_proposal_perturbation_type="bound_violation",
            phase2_proposal_perturbation_scale=0.1,
            pretrained_phase1_path="/tmp/foo",
        ),
    )
    restored = from_json(to_json(original))
    assert restored.training.phase2_proposal_perturbation_type == "bound_violation"
    assert restored.training.phase2_proposal_perturbation_scale == pytest.approx(0.1)
    assert restored.training.pretrained_phase1_path == "/tmp/foo"


def test_phase2_proposal_perturbation_per_dim_round_trip() -> None:
    """The per-dimension tuple survives to_json/from_json as a tuple of floats."""
    from dataclasses import replace as dc_replace

    original = dc_replace(
        ExperimentConfig(),
        training=TrainingConfig(
            phase2_proposal_perturbation_type="gaussian",
            phase2_proposal_perturbation_scale=0.0,
            phase2_proposal_perturbation_scale_per_dim=(0.3, 0.0, 0.05, 0.0),
        ),
    )
    restored = from_json(to_json(original))
    assert restored.training.phase2_proposal_perturbation_type == "gaussian"
    assert isinstance(restored.training.phase2_proposal_perturbation_scale_per_dim, tuple)
    assert restored.training.phase2_proposal_perturbation_scale_per_dim == (0.3, 0.0, 0.05, 0.0)


def test_phase2_proposal_scale_sampling_validation() -> None:
    """Sampling mode, min_ratio and zero_fraction are validated up front."""
    with pytest.raises(ValueError, match="phase2_proposal_perturbation_scale_sampling"):
        TrainingConfig(phase2_proposal_perturbation_scale_sampling="log-uniform")

    # min_ratio has no support at zero for the non-"fixed" modes.
    with pytest.raises(ValueError, match="phase2_proposal_perturbation_scale_min_ratio"):
        TrainingConfig(
            phase2_proposal_perturbation_scale_sampling="loguniform",
            phase2_proposal_perturbation_scale_min_ratio=0.0,
        )
    with pytest.raises(ValueError, match="phase2_proposal_perturbation_scale_min_ratio"):
        TrainingConfig(
            phase2_proposal_perturbation_scale_sampling="uniform",
            phase2_proposal_perturbation_scale_min_ratio=0.0,
        )
    with pytest.raises(ValueError, match="phase2_proposal_perturbation_scale_min_ratio"):
        TrainingConfig(phase2_proposal_perturbation_scale_min_ratio=1.5)

    with pytest.raises(ValueError, match="phase2_proposal_perturbation_zero_fraction"):
        TrainingConfig(phase2_proposal_perturbation_zero_fraction=-0.1)
    with pytest.raises(ValueError, match="phase2_proposal_perturbation_zero_fraction"):
        TrainingConfig(phase2_proposal_perturbation_zero_fraction=1.1)


def test_phase2_proposal_scale_sampling_accepts_valid_combinations() -> None:
    """The default and both randomised modes round-trip through the config."""
    assert TrainingConfig().phase2_proposal_perturbation_scale_sampling == "fixed"
    for mode in ("loguniform", "uniform"):
        cfg = TrainingConfig(
            phase2_proposal_perturbation_type="gaussian",
            phase2_proposal_perturbation_scale_per_dim=(0.4, 0.2, 0.0, 0.1),
            phase2_proposal_perturbation_scale_sampling=mode,
            phase2_proposal_perturbation_scale_min_ratio=0.01,
            phase2_proposal_perturbation_zero_fraction=0.1,
        )
        assert cfg.phase2_proposal_perturbation_scale_sampling == mode
