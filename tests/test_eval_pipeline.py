"""Perturb → correct → metrics E2E integration test.

Exercises the full evaluate.main_programmatic pipeline on DoubleIntegrator
data with a freshly-initialised (but corrector-enabled) MaDECell.
"""

# ruff: noqa: E402

import os
import pathlib

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from made.evaluation.metrics import inequality_violation_rate
from made.evaluation.perturbation import perturb_trajectories
from made.models import MaDECell
from made.physics import build_system
from made.utils import CheckpointManager
from made.utils.checkpointing import TrainState
from made.utils.config import (
    CorrectorConfig,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    PhysicsConfig,
)

from scripts.sim.evaluate import main_programmatic

_METRIC_KEYS = frozenset({
    "inequality_violation_rate",
    "inequality_violation_magnitude",
    "dynamics_violation_known",
    "dynamics_violation_learned",
    "dynamics_violation_true",
    "fidelity",
})


def _write_di_data(root: pathlib.Path, n: int = 4, t: int = 8) -> None:
    """Write minimal zero-state DoubleIntegrator test data directly."""
    s, c = 4, 2
    test_dir = root / "test"
    test_dir.mkdir(parents=True)
    np.save(str(test_dir / "states.npy"), np.zeros((n, t, s), dtype=np.float64))
    np.save(str(test_dir / "controls.npy"), np.zeros((n, t - 1, c), dtype=np.float64))


def _build_cell_checkpoint(tmp_path: pathlib.Path) -> tuple[str, ExperimentConfig]:
    """Build a small MaDECell, save a checkpoint, and return (checkpoint_dir, cfg)."""
    physics, constraints = build_system("double_integrator")
    model_cfg = ModelConfig(inverse_hidden=(16, 16), residual_hidden=(16, 16))
    corrector_cfg = CorrectorConfig(mode="enabled", train_steps=3, step_size=0.01)
    cell = MaDECell.from_config(
        physics, constraints, model_cfg, corrector_cfg, key=jax.random.key(0)
    )
    checkpoint_dir = str(tmp_path / "ckpt")
    state = TrainState(
        model=cell, opt_state_I=None, opt_state_T=None, key=jax.random.key(1), step=3
    )
    CheckpointManager(checkpoint_dir).save(state, 3)

    cfg = ExperimentConfig(
        physics=PhysicsConfig(true_system="double_integrator"),
        model=model_cfg,
        corrector=corrector_cfg,
        data=DataConfig(perturbation_scale=0.05, perturbation_type="gaussian"),
        evaluation=EvaluationConfig(perturbation_seed=42),
    )
    return checkpoint_dir, cfg


def test_correction_improves_feasibility(tmp_path):
    """Full evaluate pipeline: perturb DI states, correct with MaDE, check metrics.

    Assertions:
      a) inequality_violation_rate(corrected) <= inequality_violation_rate(perturbed)
      b) All 6 metric keys present and finite.
    """
    data_dir = tmp_path / "data"
    _write_di_data(data_dir)

    checkpoint_dir, cfg = _build_cell_checkpoint(tmp_path)

    output_path = str(tmp_path / "result.json")
    result = main_programmatic(
        cfg=cfg,
        checkpoint=checkpoint_dir,
        variant="made",
        test_data=str(data_dir),
        output=output_path,
    )

    # (b) All 6 metric keys present and finite
    metrics = result["metrics"]
    assert set(metrics.keys()) == _METRIC_KEYS, (
        f"Missing keys: {_METRIC_KEYS - set(metrics.keys())}"
    )
    for k, v in metrics.items():
        assert jnp.isfinite(v), f"Metric '{k}' = {v} is not finite"

    # (a) Correction does not worsen inequality feasibility vs perturbed states
    states = jnp.asarray(np.load(str(data_dir / "test" / "states.npy")))
    controls_np = np.load(str(data_dir / "test" / "controls.npy"))
    u_zeros = jnp.zeros_like(jnp.asarray(controls_np))

    perturbed = perturb_trajectories(states, cfg.data, jax.random.key(cfg.evaluation.perturbation_seed))
    _, known_constraints = build_system("double_integrator")
    perturbed_rate = float(inequality_violation_rate(perturbed, u_zeros, known_constraints))
    corrected_rate = metrics["inequality_violation_rate"]

    assert corrected_rate <= perturbed_rate + 1e-9, (
        f"Correction worsened feasibility: corrected={corrected_rate:.4f} > perturbed={perturbed_rate:.4f}"
    )


def test_metric_shapes_are_scalar(tmp_path):
    """compute_metrics with batched input returns scalar aggregated values."""
    data_dir = tmp_path / "data"
    _write_di_data(data_dir)
    checkpoint_dir, cfg = _build_cell_checkpoint(tmp_path)

    result = main_programmatic(
        cfg=cfg,
        checkpoint=checkpoint_dir,
        variant="made",
        test_data=str(data_dir),
        output=str(tmp_path / "r.json"),
    )
    for k, v in result["metrics"].items():
        assert isinstance(v, float), f"Metric '{k}' should be a float, got {type(v)}"


def test_clamp_variant_requires_no_checkpoint(tmp_path):
    """Clamp baseline runs without --checkpoint."""
    data_dir = tmp_path / "data"
    _write_di_data(data_dir)

    cfg = ExperimentConfig(
        physics=PhysicsConfig(true_system="double_integrator"),
        data=DataConfig(perturbation_scale=0.05),
        evaluation=EvaluationConfig(perturbation_seed=0),
    )
    result = main_programmatic(
        cfg=cfg,
        checkpoint=None,
        variant="clamp",
        test_data=str(data_dir),
        output=str(tmp_path / "clamp_result.json"),
    )
    # Clamp has no learned model, so it does not emit dynamics_violation_learned
    # (the paper borrows that value from the same-seed MaDE model at scoring time).
    assert set(result["metrics"].keys()) == _METRIC_KEYS - {"dynamics_violation_learned"}
    assert result["variant"] == "clamp"


def test_evaluate_creates_output_parent_for_clamp(tmp_path):
    """Clamp evaluation creates nested metrics output directories."""
    data_dir = tmp_path / "data"
    _write_di_data(data_dir)
    output = tmp_path / "nested" / "clamp" / "metrics.json"

    cfg = ExperimentConfig(
        physics=PhysicsConfig(true_system="double_integrator"),
        data=DataConfig(perturbation_scale=0.05),
        evaluation=EvaluationConfig(perturbation_seed=0),
    )
    main_programmatic(
        cfg=cfg,
        checkpoint=None,
        variant="clamp",
        test_data=str(data_dir),
        output=str(output),
    )
    assert output.exists()
