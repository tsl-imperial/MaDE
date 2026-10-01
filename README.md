# Markovian Dynamics Enforcer: Feasibility Preserving Correction on Learned Dynamics Manifolds

Kevin Yu<sup>1</sup>, Tao Guo<sup>2</sup>, Constantinos Antoniou<sup>2</sup>, Panagiotis Angeloudis<sup>1</sup>

<sup>1</sup>Imperial College London, UK &nbsp; <sup>2</sup>Technical University of Munich, Germany

<p align="center">
  <a href="https://arxiv.org/abs/2609.39888"><img src="https://img.shields.io/badge/arXiv-2609.39888-b31b1b.svg" alt="arXiv"></a>
  <a href="https://neurips.cc/virtual/2026/poster/XXXXX"><img src="https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg" alt="NeurIPS 2026"></a> <!-- TODO: replace -->
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python 3.11"></a>
  <a href="https://github.com/jax-ml/jax"><img src="https://img.shields.io/badge/built%20with-JAX-orange.svg" alt="JAX"></a>
</p>

MaDE is a time-invariant post-hoc operator that maps state-transition proposals onto a learned
feasible dynamics manifold, trained on feasible states without ground-truth controls. For each
transition it infers a control, recomputes the state through a completion model of known
physics plus a learned residual, and corrects that control by gradient-based inequality
reduction.

<p align="center">
  <img src="assets/made.png" alt="Overview of the MaDE cell" width="100%">
  <br>
  <em>Overview of the MaDE cell. The control update is drawn with zero momentum.</em>
</p>

## Installation

```bash
git clone https://github.com/tsl-imperial/MaDE.git && cd MaDE
bash setup.sh                     # needs uv; macOS / CPU
uv sync --frozen --extra cuda12   # or: Linux, CUDA 12 (versions used for the paper)
uv run pytest                     # smoke test: fast CPU tier; `uv run pytest -m slow` for end-to-end
```

Prefix the commands below with `uv run`, or activate `.venv` first.

MaDE runs in float64 only. Every entry-point script sets
`jax.config.update("jax_enable_x64", True)` at startup. Do the same in your own code.

## Using MaDE with your own predictor

From the repository root, train a MaDE model, here on the kinematic bicycle:

```bash
uv run python scripts/sim/generate_data.py --system kinematic_bicycle --seed 0
uv run python scripts/sim/run_matrix.py --systems kinematic_bicycle --variants made --seeds 0
```

Then correct a predicted trajectory with the frozen model:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from made.upstream import apply_made_trajectory
from made.utils import CheckpointManager

ckpt = "outputs/sim/runs/kinematic_bicycle/fully-specified/made/seed0/checkpoints"
made = CheckpointManager(ckpt).restore().model  # frozen MaDE cell

x0 = jnp.array([0.0, 0.0, 0.0, 5.0])  # last observed state (x, y, theta, v)
steps = jnp.arange(1, 16)[:, None]
x_pred = x0 + steps * jnp.array([0.5, 0.0, 0.0, 0.0])  # your predictor's [T, 4] output
params = jnp.array([2.7])  # known physics parameters (wheelbase L)

x_made = apply_made_trajectory(
    made, x_pred, params, dt=0.1, correction_mode="eval_adaptive", x0=x0
)  # corrected states, [T, 4]
```

`apply_made_trajectory_with_controls` takes the same arguments and also returns the corrected
controls. inD models load with `made.models.MaDEModel.from_checkpoint(path)`. Their physics
parameters come from vehicle metadata through `made.params_from_metadata(metadata)`, and their
step is `dt=0.2`.

## Reproducing the paper

### Data

**Simulated systems.** Generated locally by `scripts/sim/generate_data.py --system all --seed 0`,
which `run_all.sh` runs first for every simulated table. Output goes to `data/generated/`.
Dynamic-bicycle controls are sampled in ±0.2 rad and ±1.5 m/s², inside the ±0.5 rad and
±3.0 m/s² constraint box (`configs/sim/data_generation.json`).

**inD.** We do not redistribute inD. Request access from levelXdata at
https://levelxdata.com/ind-dataset/. The dataset is licensed for non-commercial research use.
Place the contents of the dataset's `data/` directory in `data/inD-raw/`. The first inD table
you request preprocesses it into `data/inD-preprocessed/v1`, which every inD command reads. To
run that step yourself:

```bash
uv run python scripts/ind/preprocess.py --raw-dir data/inD-raw --output-dir data/inD-preprocessed
```

### Commands

Run from the repository root. Each flag runs the steps its table needs, and steps shared by
several flags run once. Combine flags, or use `--all`. `--dry-run` prints the commands without
running them, and `--help` lists every option.

```bash
uv run ./run_all.sh --table1 --table3
```

| Paper item | Command | Output | Hardware |
|---|---|---|---|
| Table 1: simulated results | `uv run ./run_all.sh --table1` | `outputs/table_1/` | CPU |
| Table 2: inD results | `uv run ./run_all.sh --table2` | `outputs/table_2/` | GPU |
| Table 3: ablations | `uv run ./run_all.sh --table3` | `outputs/table_3/` | CPU |
| Table 5: control recovery | `uv run ./run_all.sh --table5` | `outputs/table_5/` | CPU |
| Table 6: runtime | `uv run ./run_all.sh --table6` | `outputs/table_6/` | idle NVIDIA GPU |
| Table 8: gradient depth | `uv run ./run_all.sh --table8` | `outputs/table_8/` | GPU |
| Tables 9 and 10: inD inequality breakdown | `uv run ./run_all.sh --table9` (same as `--table10`) | `outputs/table_9_10/` | GPU |
| Table 11: completion-only variant | `uv run ./run_all.sh --table11` | `outputs/table_11/` | GPU |
| Prior-only comparator (Appendix) | `uv run ./run_all.sh --table1` | `prior_only_comparison` in `outputs/table_1/tab_e01_seed_level.json` | CPU |
| Per-model breakdown (Appendix) | `uv run ./run_all.sh --table2` | `outputs/ind/panel.json` | GPU |
| Corrector iteration counts (Section 5, Appendix) | `uv run ./run_all.sh --corrector-iterations` | `outputs/appendix/` | GPU |

Figures 1 and 2 are drawings. Tables 4 and 7 list metric definitions and default
hyperparameters. None of the four has associated code. Run `--table6` alone on an idle GPU,
because it measures wall-clock time. `--all` runs it last.

Data, checkpoints, scores and evaluation panels stay under `data/`, `outputs/sim/` and
`outputs/ind/`. A rerun skips finished predictors and tuned smoother noise (`--force` disables
this), and MaDE training resumes from its checkpoints. The simulated tables come from the
scoring pass in `outputs/sim/scores/`, not from the `metrics.json` that `run_matrix.py` writes
next to each checkpoint. Each table builder (`scripts/sim/build_tables.py`,
`scripts/ind/build_table_*.py`) can be rerun on its own and writes to the same directories.

## Citation

<!-- TODO: replace with the NeurIPS 2026 @inproceedings entry -->
```bibtex
@article{yu2026made,
  title   = {Markovian Dynamics Enforcer: Feasibility Preserving Correction on Learned Dynamics Manifolds},
  author  = {Yu, Kevin and Guo, Tao and Antoniou, Constantinos and Angeloudis, Panagiotis},
  journal = {arXiv preprint arXiv:2609.39888},
  year    = {2026},
}
```

## License

MIT. See [LICENSE](LICENSE). The inD dataset is under its own licence from levelXdata and is not
part of this repository.
