"""Tests for scripts/sim/control_recovery.py: render format, smoke run, missing checkpoint."""

from __future__ import annotations

import json

import jax
import numpy as np

from made.utils import CheckpointManager, load_config
from made.utils.checkpointing import TrainState
from made.models import MaDECell
from scripts.sim import control_recovery


def test_render_matches_published_format():
    NAN = float("nan")

    def _pub(nm, ns, bm=(NAN, NAN), bs=(NAN, NAN)):
        return {"nrmse_per_dim_mean": list(nm), "nrmse_per_dim_std": list(ns),
                "bias_per_dim_mean": list(bm), "bias_per_dim_std": list(bs)}

    # The currently published values (the paper's Table 5), in the summary's shape.
    PUBLISHED = {
        "double_integrator/fully-specified": {
            "prior": _pub([7.90e-16, 9.54e-16], [0.0, 0.0]),
            "made": _pub([2.36e-6, 2.44e-6], [1.14e-6, 1.01e-6]),
            "made-supervised-i": _pub([6.23e-6, 2.36e-6], [6.39e-6, 5.24e-7])},
        "unicycle/fully-specified": {
            "prior": _pub([1.34e-15, 1.17e-15], [0.0, 0.0]),
            "made": _pub([9.97e-6, 6.99e-6], [2.81e-6, 1.63e-6]),
            "made-supervised-i": _pub([1.12e-5, 7.44e-6], [5.45e-6, 1.34e-6])},
        "kinematic_bicycle/fully-specified": {
            "prior": _pub([5.04e-15, NAN], [0.0, NAN]),
            "made": _pub([1.01e-3, NAN], [4.95e-4, NAN]),
            "made-supervised-i": _pub([6.51e-4, NAN], [3.20e-4, NAN])},
        "dynamic_bicycle/underspecified": {
            "prior": _pub([0.9200, 0.2602], [0.0, 0.0], [NAN, -0.1594], [NAN, 0.0]),
            "made": _pub([2.2256, 0.2582], [0.3271, 0.0051], [NAN, -0.1569], [NAN, 0.0058]),
            "made-supervised-i": _pub([2.0045, 0.2586], [0.3555, 0.0027], [NAN, -0.1575],
                                       [NAN, 0.0033])},
    }

    PUBLISHED_TEX = r"""\begin{tabular}{llccc}
\toprule
System & Channel & known inverse & MaDE & sup.-$\mathcal{I}$ \\
\midrule
DI (exact) & $a_x$ (nRMSE) & $7.90\times 10^{-16}$ & $2.36\times 10^{-6}\,{\scriptscriptstyle\pm}\,1.14\times 10^{-6}$ & $6.23\times 10^{-6}\,{\scriptscriptstyle\pm}\,6.39\times 10^{-6}$ \\
DI (exact) & $a_y$ (nRMSE) & $9.54\times 10^{-16}$ & $2.44\times 10^{-6}\,{\scriptscriptstyle\pm}\,1.01\times 10^{-6}$ & $2.36\times 10^{-6}\,{\scriptscriptstyle\pm}\,5.24\times 10^{-7}$ \\
UNI (exact) & $\delta$ (nRMSE) & $1.34\times 10^{-15}$ & $9.97\times 10^{-6}\,{\scriptscriptstyle\pm}\,2.81\times 10^{-6}$ & $1.12\times 10^{-5}\,{\scriptscriptstyle\pm}\,5.45\times 10^{-6}$ \\
UNI (exact) & $a$ (nRMSE) & $1.17\times 10^{-15}$ & $6.99\times 10^{-6}\,{\scriptscriptstyle\pm}\,1.63\times 10^{-6}$ & $7.44\times 10^{-6}\,{\scriptscriptstyle\pm}\,1.34\times 10^{-6}$ \\
KB (exact) & $\delta$ (nRMSE) & $5.04\times 10^{-15}$ & $1.01\times 10^{-3}\,{\scriptscriptstyle\pm}\,4.95\times 10^{-4}$ & $6.51\times 10^{-4}\,{\scriptscriptstyle\pm}\,3.20\times 10^{-4}$ \\
DB (underspec.) & $\delta$ (nRMSE) & $0.9200$ & $2.2256\,{\scriptscriptstyle\pm}\,0.3271$ & $2.0045\,{\scriptscriptstyle\pm}\,0.3555$ \\
DB (underspec.) & $a$ (nRMSE) & $0.2602$ & $0.2582\,{\scriptscriptstyle\pm}\,0.0051$ & $0.2586\,{\scriptscriptstyle\pm}\,0.0027$ \\
DB (underspec.) & $a$ (bias) & $-0.1594$ & $-0.1569\,{\scriptscriptstyle\pm}\,0.0058$ & $-0.1575\,{\scriptscriptstyle\pm}\,0.0033$ \\
\bottomrule
\end{tabular}
"""

    assert control_recovery.render_tex(PUBLISHED) == PUBLISHED_TEX


def _write_di_data(data_dir):
    rng = np.random.default_rng(0)
    states = rng.normal(size=(4, 6, 4))
    controls = rng.normal(size=(4, 5, 2))
    split_dir = data_dir / "double_integrator" / "test"
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "states.npy", states.astype(np.float64))
    np.save(split_dir / "controls.npy", controls.astype(np.float64))


def _make_di_checkpoint(ckpt_dir):
    cfg = load_config("configs/sim/di_made.json")
    known_physics, known_constraints, _ = control_recovery._resolve_cell_physics(cfg)
    cell = MaDECell.from_config(
        known_physics,
        known_constraints,
        cfg.model,
        cfg.corrector,
        key=jax.random.key(0),
        dt=cfg.physics.dt,
    )
    state = TrainState(model=cell, opt_state_I=None, opt_state_T=None,
                        key=jax.random.key(1), step=0)
    CheckpointManager(str(ckpt_dir)).save(state, 0)


def test_prior_and_checkpoint_smoke(tmp_path):
    data_dir = tmp_path / "data"
    _write_di_data(data_dir)
    runs_root = tmp_path / "runs"
    ckpt_dir = runs_root / "double_integrator" / "fully-specified" / "made" / "seed0" / \
        "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _make_di_checkpoint(ckpt_dir)

    out_json = tmp_path / "out" / "control_recovery.json"
    out_tex = tmp_path / "out" / "tab_control_recovery.tex"
    ret = control_recovery.main([
        "--systems", "double_integrator",
        "--variants", "prior,made",
        "--seeds", "0",
        "--data-dir", str(data_dir),
        "--runs-root", str(runs_root),
        "--out-json", str(out_json),
        "--out-tex", str(out_tex),
    ])
    assert ret == 0

    summary = json.loads(out_json.read_text())
    condition = summary["conditions"]["double_integrator/fully-specified"]
    assert "prior" in condition and "made" in condition
    assert condition["prior"]["num_seeds"] == 1
    assert condition["made"]["num_seeds"] == 1
    for cell in summary["cells"]:
        for v in cell["metrics"]["nrmse_per_dim"]:
            assert np.isfinite(v)

    tex_lines = out_tex.read_text().splitlines()
    mid = tex_lines.index(r"\midrule")
    bot = tex_lines.index(r"\bottomrule")
    body = tex_lines[mid + 1:bot]
    assert len(body) == 8
    assert all(line.endswith(r"\\") for line in body)


def test_missing_checkpoint_exits_nonzero(tmp_path):
    data_dir = tmp_path / "data"
    _write_di_data(data_dir)
    runs_root = tmp_path / "runs"

    out_json = tmp_path / "out" / "control_recovery.json"
    out_tex = tmp_path / "out" / "tab_control_recovery.tex"
    ret = control_recovery.main([
        "--systems", "double_integrator",
        "--variants", "made",
        "--seeds", "0",
        "--data-dir", str(data_dir),
        "--runs-root", str(runs_root),
        "--out-json", str(out_json),
        "--out-tex", str(out_tex),
    ])
    assert ret == 1

    summary = json.loads(out_json.read_text())
    assert len(summary["missing"]) == 1
    assert out_tex.exists()
