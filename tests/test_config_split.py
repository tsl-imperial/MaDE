"""Tests for PhysicsConfig/ModelConfig known-true split and EvaluationConfig."""

from made.utils.config import (
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    PhysicsConfig,
    from_json,
    to_json,
)


def test_true_system_field():
    cfg = PhysicsConfig()
    assert cfg.true_system == "double_integrator"
    assert not hasattr(cfg, "system")


def test_physics_config_custom_true_system():
    cfg = PhysicsConfig(true_system="dynamic_bicycle")
    assert cfg.true_system == "dynamic_bicycle"


def test_known_system_default_none():
    cfg = ModelConfig()
    assert cfg.known_system is None
    assert cfg.known_params == {}


def test_known_system_set():
    cfg = ModelConfig(known_system="kinematic_bicycle", known_params={"L": 2.9})
    assert cfg.known_system == "kinematic_bicycle"
    assert cfg.known_params == {"L": 2.9}


def test_evaluation_config_defaults():
    cfg = EvaluationConfig()
    assert cfg.perturbation_seed == 42


def test_evaluation_config_custom():
    cfg = EvaluationConfig(perturbation_seed=7)
    assert cfg.perturbation_seed == 7


def test_evaluation_config_roundtrip():
    cfg = ExperimentConfig(evaluation=EvaluationConfig(perturbation_seed=99))
    restored = from_json(to_json(cfg))
    assert restored.evaluation.perturbation_seed == 99


def test_known_true_roundtrip():
    cfg = ExperimentConfig(
        physics=PhysicsConfig(true_system="dynamic_bicycle"),
        model=ModelConfig(
            known_system="kinematic_bicycle",
            known_params={"L": 2.7},
        ),
        evaluation=EvaluationConfig(perturbation_seed=13),
    )
    restored = from_json(to_json(cfg))
    assert restored.physics.true_system == "dynamic_bicycle"
    assert restored.model.known_system == "kinematic_bicycle"
    assert restored.model.known_params == {"L": 2.7}
    assert restored.evaluation.perturbation_seed == 13


def test_backward_compat_known_system_none():
    """Config without known_system produces identical semantics to pre-change."""
    cfg = ExperimentConfig()
    assert cfg.model.known_system is None
    restored = from_json(to_json(cfg))
    assert restored.model.known_system is None
    assert restored.physics.true_system == "double_integrator"
