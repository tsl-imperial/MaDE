"""E01 config matrix tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from made.utils import load_config, save_config


CONFIG_DIR = Path("configs/sim")
VARIANTS = {
    "made",
    "made-no-residual",
    "made-no-corrector",
    "made-supervised-i",
    "made-fixed-i",
    "mlp",
    "fab",
    "clamp",
    "made-prior-only",
}
# Variant defined only for the underspecified dynamic bicycle.
DB_ONLY_VARIANTS = {"made-prior-only"}
GROUP_PREFIXES = {
    "di",
    "unicycle",
    "kinbicycle",
    "dynbicycle_underspecified",
}
# Variants trained with the D3 restore-phase-1-best flag (the MaDE/ablation set);
# made-prior-only is evaluation-only and unchanged from its e01-refresh source.
MADE_ABLATION_VARIANTS = VARIANTS - {"mlp", "fab", "clamp", "made-prior-only"}


def _json_paths() -> list[Path]:
    return sorted(p for p in CONFIG_DIR.glob("*.json") if p.name != "data_generation.json")


def test_all_configs_present():
    paths = _json_paths()
    # 3 fully-specified systems x 8 standard variants + 1 underspecified DB
    # system x 9 (standard variants + made-prior-only) = 24 + 9 = 33.
    n_standard = len(VARIANTS - DB_ONLY_VARIANTS)
    expected_count = 3 * n_standard + len(VARIANTS)
    assert len(paths) == expected_count == 33
    names = {path.stem for path in paths}
    expected = set()
    for prefix in GROUP_PREFIXES:
        if prefix == "dynbicycle_underspecified":
            expected.update(f"{prefix}_{variant}" for variant in VARIANTS)
        else:
            expected.update(
                f"{prefix}_{variant}" for variant in VARIANTS - DB_ONLY_VARIANTS
            )
    assert names == expected


def test_all_configs_roundtrip(tmp_path):
    for path in _json_paths():
        cfg = load_config(str(path))
        out = tmp_path / path.name
        save_config(cfg, str(out))
        assert json.loads(path.read_text()) == json.loads(out.read_text())


def test_configs_include_required_known_true_fields():
    for path in _json_paths():
        payload = json.loads(path.read_text())
        assert "true_system" in payload["physics"], path
        assert "known_system" in payload["model"], path
        assert "perturbation_seed" in payload["evaluation"], path


def test_made_ablation_configs_restore_phase1_best_before_phase2():
    for prefix in GROUP_PREFIXES:
        variants = VARIANTS if prefix == "dynbicycle_underspecified" else VARIANTS - DB_ONLY_VARIANTS
        for variant in variants & MADE_ABLATION_VARIANTS:
            path = CONFIG_DIR / f"{prefix}_{variant}.json"
            payload = json.loads(path.read_text())
            assert payload["training"]["restore_phase1_best_before_phase2"] is True, path


def test_configs_enable_documented_early_stopping_defaults():
    expected = {
        "early_stopping_enabled": True,
        "early_stopping_min_epochs": 5,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.0,
        "early_stopping_physical_exact": True,
        "early_stopping_physical_saturation": True,
        "early_stopping_forward_tol": 1e-6,
        "early_stopping_inverse_tol": 1e-6,
        "early_stopping_minimum_norm_tol": 1e-6,
        "early_stopping_delta_i_norm_tol": 1e-6,
        "early_stopping_saturation_delta": 1e-5,
    }
    for path in _json_paths():
        payload = json.loads(path.read_text())
        training = payload["training"]
        for key, value in expected.items():
            assert training[key] == value, path
        # made/ablation configs (from e01-refresh) use a wider saturation window than
        # the mlp/clamp/fab baseline configs (from e01).
        is_baseline = any(path.stem.endswith(f"_{v}") for v in ("mlp", "clamp", "fab"))
        expected_window = 5 if is_baseline else 10
        assert training["early_stopping_saturation_window"] == expected_window, path


def test_configs_use_tightened_regularization_defaults():
    expected = {
        "lambda_min_norm": 0.01,
        "t_side_grad_clip_norm": 1.0,
        "i_side_grad_clip_norm": 1.0,
        "lambda_delta_i_norm": 0.01,
    }
    for path in _json_paths():
        payload = json.loads(path.read_text())
        training = payload["training"]
        for key, value in expected.items():
            assert training[key] == value, f"{path.name}: {key} = {training.get(key)!r}"


def test_dynamic_bicycle_configs_are_all_underspecified_with_kinematic_known_physics():
    paths = sorted(p for p in CONFIG_DIR.glob("dynbicycle_*.json") if p.name != "data_generation.json")
    assert paths
    assert not any(path.name.startswith("dynbicycle_fully-specified") for path in paths)
    for path in paths:
        payload = json.loads(path.read_text())
        assert payload["physics"]["true_system"] == "dynamic_bicycle"
        assert payload["model"]["known_system"] == "kinematic_bicycle"


def test_run_matrix_wandb_run_name_format():
    from scripts.sim.run_matrix import _wandb_run_name

    assert (
        _wandb_run_name("double_integrator", "fully-specified", "made", 2)
        == "e01-double_integrator-fully-specified-made-2"
    )
    assert (
        _wandb_run_name("dynamic_bicycle", "underspecified", "made-no-residual", 2)
        == "e01-dynamic_bicycle-underspecified-made-no-residual-2"
    )


def test_e01_entrypoints_register_all_config_variants():
    from scripts.sim import evaluate, run_matrix, train

    made_variants = {variant for variant in VARIANTS if variant.startswith("made")}
    # run_matrix trains every variant except fab (trained standalone by train_fab.py) and
    # made-prior-only (evaluation-only, never trained).
    run_matrix_variants = set(run_matrix._VARIANTS) | DB_ONLY_VARIANTS
    non_run_matrix_variants = {"fab"}

    assert run_matrix_variants == VARIANTS - non_run_matrix_variants
    assert train._MADE_VARIANTS == made_variants
    assert evaluate._MADE_VARIANTS == made_variants
    assert VARIANTS.issubset(evaluate._ALL_VARIANTS)


def test_base_regime_perturbation_contracts():
    for prefix_glob in (
        "di_*.json",
        "unicycle_*.json",
        "kinbicycle_*.json",
        "dynbicycle_underspecified_*.json",
    ):
        paths = sorted(p for p in CONFIG_DIR.glob(prefix_glob) if p.name != "data_generation.json")
        assert paths, f"no configs matched {prefix_glob}"
        for path in paths:
            cfg = load_config(str(path))
            assert cfg.data.perturbation_type == "bound_violation", (
                f"{path.name}: must have perturbation_type='bound_violation'"
            )
            assert cfg.data.perturbation_scale == 0.10, (
                f"{path.name}: must have perturbation_scale=0.10"
            )


def test_dynbicycle_underspecified_configs_apply_training_noise():
    paths = sorted(p for p in CONFIG_DIR.glob("dynbicycle_underspecified_*.json") if p.name != "data_generation.json")
    assert paths
    for path in paths:
        cfg = load_config(str(path))
        assert cfg.data.noise_scale > 0.0, (
            f"{path.name}: canonical DB config must have data.noise_scale > 0.0"
        )


def test_per_dim_perturbation_scales_match_state_dim():
    expected = {
        "di_": None,
        "unicycle_": (0.10, 0.10, 0.05, 0.10),
        "kinbicycle_": (0.10, 0.10, 0.05, 0.10),
        "dynbicycle_underspecified_": (0.10, 0.10, 0.05, 0.10, 0.10, 0.10),
    }
    # di_fab.json carries the recorded canonical per-dim scale (D3), unlike every other
    # DI config (None).
    per_file_overrides = {"di_fab.json": (0.1, 0.1, 0.1, 0.1)}
    for prefix, exp in expected.items():
        for path in sorted(CONFIG_DIR.glob(f"{prefix}*.json")):
            if path.name == "data_generation.json":
                continue
            cfg = load_config(str(path))
            file_exp = per_file_overrides.get(path.name, exp)
            assert cfg.data.perturbation_scale_per_dim == file_exp, (
                f"{path.name}: per_dim={cfg.data.perturbation_scale_per_dim!r}, "
                f"expected {file_exp!r}"
            )


def test_run_matrix_steps_per_epoch_dry_run_avoids_trainer_import(monkeypatch, capsys):
    sys.modules.pop("made.training.trainer", None)
    from scripts.sim import run_matrix

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_matrix.py",
            "--systems",
            "double_integrator",
            "--variants",
            "clamp",
            "--seeds",
            "0",
            "--steps-per-epoch",
            "4",
            "--val-steps-per-epoch",
            "2",
            "--dry-run",
        ],
    )

    run_matrix.main()

    assert "made.training.trainer" not in sys.modules
    assert "EVALUATE:" in capsys.readouterr().out
