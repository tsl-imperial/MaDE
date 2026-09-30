"""Entry point for upstream predictor training on inD.

Trains an ``LSTMPredictor`` / ``SSMPredictor`` / ``TransformerPredictor`` from
scratch against the inD train split using ``stage1_loss`` (MSE + soft inequality +
discretised dynamics penalty, all through a frozen kinematic-only inverse-dynamics
model).

Usage
-----
  # LSTM, real data:
  python scripts/ind/train_predictor.py \\
      --data-dir data/inD-preprocessed/v1 \\
      --output-dir outputs/ind/predictors/lstm_stage1_seed0 \\
      --predictor lstm --seed 0

  # Transformer, real data:
  python scripts/ind/train_predictor.py \\
      --data-dir data/inD-preprocessed/v1 \\
      --output-dir outputs/ind/predictors/transformer_stage1_seed0 \\
      --predictor transformer --seed 0

  # CPU smoke run (stub data, tiny dims):
  python scripts/ind/train_predictor.py --use-stub --output-dir /tmp/ind_smoke \\
      --predictor lstm --history 4 --horizon 3 --epochs 2 \\
      --hidden-size 8 --decoder-width 8 --decoder-depth 1

Engineering rules:
  - jax_enable_x64 enabled before any JAX import.
  - eqx.filter_jit for all JIT-compiled functions; vmap for per-sample -> batch.
  - Inequality constraints on the inD path: inD_physical_constraints() only.
  - Orbax/checkpoint paths absolutised at the CLI boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

import jax

from made.utils.jax_setup import configure

configure()

ROOT = Path(__file__).resolve().parents[2]

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from made.data.ind_data import create_ind_data_source  # noqa: E402
from made.data.window_dataset import (  # noqa: E402
    assemble_context,
    compute_state_norm_stats,
    make_prediction_windows,
)
from made.models import InverseDynamics  # noqa: E402
from made.physics import (  # noqa: E402
    KinematicBicycle,
    inD_physical_constraints,
)
from made.upstream import stage1_loss  # noqa: E402
from made.upstream.factory import make_predictor, save_predictor  # noqa: E402
from made.utils import UpstreamConfig  # noqa: E402

# Fixed dt for the inD kinematic-bicycle physics — not a CLI knob.
_DT: float = 0.2
# Frozen kinematic-only wheelbase used for the inverse-dynamics prior
# (matches the L_REF used elsewhere in the inD evaluation path).
_L_REF: float = 2.7

_LSTM_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {"hidden_size", "decoder_width", "decoder_depth", "embedding_dim"}
)
_SSM_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
        "d_model",
        "num_blocks",
        "d_state",
        "expand",
        "conv_kernel",
        "decoder_width",
        "decoder_depth",
        "embedding_dim",
    }
)
_TRANSFORMER_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
        "d_model",
        "num_layers",
        "num_heads",
        "ff_width",
        "decoder_width",
        "decoder_depth",
        "embedding_dim",
    }
)
_OVERRIDE_KEYS_BY_KIND: dict[str, frozenset[str]] = {
    "lstm": _LSTM_OVERRIDE_KEYS,
    "ssm": _SSM_OVERRIDE_KEYS,
    "transformer": _TRANSFORMER_OVERRIDE_KEYS,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "inD-preprocessed" / "v1"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--predictor", choices=["lstm", "ssm", "transformer"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--val-chunk-size",
        type=int,
        default=2048,
        help="Evaluate validation MSE in chunks of this many windows instead of one giant "
        "vmap'd call (fixes a 34.1 GiB OOM on the SSM predictor's scan intermediates over the "
        "full 50,114-window inD val split). Combined as the exact weighted mean over all "
        "windows.",
    )
    parser.add_argument(
        "--use-stub",
        action="store_true",
        help="Use made.data.ind_data.create_ind_data_source stub data "
        "(CPU smoke, no real inD dataset).",
    )
    parser.add_argument("--stub-num-trajectories", type=int, default=64)
    parser.add_argument("--stub-trajectory-length", type=int, default=20)
    # Architecture overrides (small-dim smoke runs / hyperparameter sweeps).
    parser.add_argument("--hidden-size", type=int, default=None, help="LSTM hidden size.")
    parser.add_argument("--decoder-width", type=int, default=None)
    parser.add_argument("--decoder-depth", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None, help="SSM/Transformer model width.")
    parser.add_argument("--num-blocks", type=int, default=None, help="SSM block count.")
    parser.add_argument("--d-state", type=int, default=None, help="SSM state dimension.")
    parser.add_argument("--expand", type=int, default=None, help="SSM inner expansion factor.")
    parser.add_argument(
        "--conv-kernel", type=int, default=None, help="SSM causal-conv kernel size."
    )
    parser.add_argument("--num-layers", type=int, default=None, help="Transformer encoder depth.")
    parser.add_argument("--num-heads", type=int, default=None, help="Transformer attention heads.")
    parser.add_argument(
        "--ff-width", type=int, default=None, help="Transformer feed-forward width."
    )
    parser.add_argument(
        "--es-patience",
        type=int,
        default=10,
        help="Early-stopping patience in validation events (default: 10). None disables.",
    )
    parser.add_argument(
        "--es-min-epochs", type=int, default=5,
        help="Minimum-epoch burn-in before a stop may fire (default: 5).",
    )
    parser.add_argument(
        "--es-min-delta", type=float, default=0.0,
        help="Improvement threshold for resetting patience (default: 0).",
    )
    parser.add_argument(
        "--stationary-filter-m",
        type=float,
        default=None,
        help=(
            "Stationary filter threshold in metres. None (default) reproduces published "
            "behaviour byte-for-byte. 0.5 is the criterion used for the paper's results."
        ),
    )
    return parser


def _collect_overrides(args: argparse.Namespace, kind: str) -> dict[str, Any]:
    allowed = _OVERRIDE_KEYS_BY_KIND[kind]
    overrides: dict[str, Any] = {}
    for name in allowed:
        value = getattr(args, name.replace("-", "_"), None)
        if value is not None:
            overrides[name] = value
    return overrides


def _progress_output() -> tuple[TextIO, bool, TextIO | None]:
    """Mirrors made.training.trainer._progress_output exactly.

    Renders fine under the queue's ``script -e -q -f -c`` pty wrapping (a pty
    reports ``isatty() == True``); falls back to ``/dev/tty`` only when stderr
    itself is redirected (e.g. piped to a file with no controlling terminal).
    """
    if sys.stderr.isatty():
        return sys.stderr, False, None
    try:
        tty = Path("/dev/tty").open("w", encoding="utf-8")
    except OSError:
        return sys.stderr, True, None
    return tty, False, tty


def _batches(num_samples: int, batch_size: int, rng: np.random.Generator):
    order = rng.permutation(num_samples)
    for start in range(0, num_samples, batch_size):
        idx = order[start : start + batch_size]
        if idx.size == 0:
            continue
        yield jnp.asarray(idx)


def _assemble_batch_context(context_states: jax.Array, metadata: jax.Array) -> jax.Array:
    """``[K, H, D] + [K, M] -> [K, H, D + M]`` (vmap of assemble_context)."""
    return jax.vmap(assemble_context)(context_states, metadata)


def _load_windows(args: argparse.Namespace, split: str, stub_seed: int) -> dict[str, jax.Array]:
    states, metadata, lengths = create_ind_data_source(
        args.data_dir,
        split,
        use_stub=args.use_stub,
        stub_num_trajectories=args.stub_num_trajectories,
        stub_trajectory_length=args.stub_trajectory_length,
        stub_seed=stub_seed,
    )
    return make_prediction_windows(
        states,
        lengths,
        metadata,
        history=args.history,
        horizon=args.horizon,
        stride=args.stride,
        # The predictors train AND are evaluated on prediction windows, so they take the
        # window-level filter, which keeps their train and eval protocols identical. MaDE
        # takes the track-level one instead because it trains on transition pairs and never
        # builds a prediction window.
        min_displacement_m=args.stationary_filter_m,
    )


def _chunked_weighted_mean(
    chunk_stats_fn,
    model,
    arrays: list[jax.Array],
    chunk_size: int,
) -> float:
    """Combine per-window MSE over ``arrays`` (all sharing axis-0 length) into the exact
    weighted mean, evaluating ``chunk_stats_fn`` over fixed-size chunks instead of one giant
    vmap'd call over the whole split.

    ``chunk_stats_fn(model, *array_chunks, mask_chunk) -> (sum_sq, count)`` must be an
    ``eqx.filter_jit``'d function returning the SUM (not mean) of per-window MSE over the chunk
    and the count of valid (unmasked) windows in it -- this is what makes the combination exact
    rather than a mean-of-means approximation.

    The last chunk is zero-padded up to a full chunk and masked out (contributes exactly 0 to
    both the sum and the count), so every call to ``chunk_stats_fn`` sees the SAME shape --
    only one shape is ever traced, regardless of how many chunks there are. The effective chunk
    size is capped at ``num_samples`` so small test splits don't pad up to a wastefully large
    ``chunk_size``.
    """
    n = arrays[0].shape[0]
    if n == 0:
        return float("nan")
    eff_chunk = min(chunk_size, n)
    num_chunks = -(-n // eff_chunk)  # ceil division
    pad_amount = num_chunks * eff_chunk - n

    padded_arrays = []
    for a in arrays:
        if pad_amount > 0:
            pad_width = [(0, pad_amount)] + [(0, 0)] * (a.ndim - 1)
            a = jnp.pad(a, pad_width)
        padded_arrays.append(a)
    mask = jnp.concatenate(
        [jnp.ones((n,), dtype=jnp.float64), jnp.zeros((pad_amount,), dtype=jnp.float64)]
    )

    total_sum = 0.0
    total_count = 0.0
    for i in range(num_chunks):
        sl = slice(i * eff_chunk, (i + 1) * eff_chunk)
        chunk_args = [a[sl] for a in padded_arrays]
        sum_sq, count = chunk_stats_fn(model, *chunk_args, mask[sl])
        total_sum += float(sum_sq)
        total_count += float(count)
    return total_sum / total_count


def _make_frozen_inverse_dynamics(key: jax.Array) -> InverseDynamics:
    """Frozen kinematic-only I used by Stage-1's inequality/dynamics penalties.

    ``use_residual=False`` with ``known_physics`` set means no MLP is built
    (``I = I_known`` exactly) — the ``key`` is required by the constructor
    signature but unused on this branch.
    """
    return InverseDynamics(
        state_dim=4,
        control_dim=2,
        param_dim=1,
        hidden=(32, 32),
        known_physics=KinematicBicycle(),
        dt=_DT,
        use_residual=False,
        key=key,
    )


def _train_stage1(
    args: argparse.Namespace,
    predictor,
    windows_train: dict[str, jax.Array],
    windows_val: dict[str, jax.Array],
    l_ref_params: jax.Array,
    *,
    key: jax.Array,
) -> tuple[Any, list[dict[str, float]], float]:
    frozen_I = _make_frozen_inverse_dynamics(key)
    physics = KinematicBicycle()
    constraints = inD_physical_constraints()
    config = UpstreamConfig(stage1_loss="mse")  # MSE-only objective

    context_train = _assemble_batch_context(windows_train["context"], windows_train["metadata"])
    x_gt_train = windows_train["future"]
    context_val = _assemble_batch_context(windows_val["context"], windows_val["metadata"])
    x_gt_val = windows_val["future"]

    num_train = context_train.shape[0]
    params_train = jnp.broadcast_to(l_ref_params, (num_train, l_ref_params.shape[0]))

    optim = optax.adam(args.lr)
    opt_state = optim.init(eqx.filter(predictor, eqx.is_array))

    def _loss_batch(model, context, x_gt, params):
        def _single(c, g, p):
            return stage1_loss(model, frozen_I, None, physics, constraints, c, g, p, _DT, config)

        losses, metrics = jax.vmap(_single)(context, x_gt, params)
        return jnp.mean(losses), {k: jnp.mean(v) for k, v in metrics.items()}

    @eqx.filter_jit
    def _step(model, opt_state, context, x_gt, params):
        (loss, metrics), grads = eqx.filter_value_and_grad(_loss_batch, has_aux=True)(
            model, context, x_gt, params
        )
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, metrics

    @eqx.filter_jit
    def _val_chunk_stats(model, context_chunk, x_gt_chunk, mask_chunk):
        def _single(c, g):
            return jnp.mean((model(c) - g) ** 2)

        per_window = jax.vmap(_single)(context_chunk, x_gt_chunk)
        return jnp.sum(per_window * mask_chunk), jnp.sum(mask_chunk)

    def _val_mse(model, context, x_gt):
        return _chunked_weighted_mean(
            _val_chunk_stats, model, [context, x_gt], args.val_chunk_size
        )

    rng = np.random.default_rng(args.seed)
    best_val = float("inf")
    best_predictor = predictor
    wait = 0
    stop_reason: str | None = None
    history: list[dict[str, float]] = []
    progress_file, progress_disabled, progress_owned_file = _progress_output()
    progress = tqdm(
        total=args.epochs,
        desc="train_upstream_inD stage1",
        unit="epoch",
        disable=progress_disabled,
        file=progress_file,
        dynamic_ncols=True,
    )
    try:
        for epoch in range(args.epochs):
            epoch_losses: list[float] = []
            for idx in _batches(num_train, args.batch_size, rng):
                predictor, opt_state, loss, _metrics = _step(
                    predictor, opt_state, context_train[idx], x_gt_train[idx], params_train[idx]
                )
                epoch_losses.append(float(loss))
            val_mse = (
                float(_val_mse(predictor, context_val, x_gt_val))
                if context_val.shape[0] > 0
                else float("nan")
            )
            train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            history.append({"epoch": epoch, "train_loss": train_loss, "val_mse": val_mse})
            print(
                f"[train_upstream_inD] stage1 epoch={epoch} "
                f"train_loss={train_loss:.6g} val_mse={val_mse:.6g}"
            )
            # Early-stopping policy: burn-in 5, patience 10, validated every epoch,
            # min_delta 0 by default. Mirrors made.training.trainer's early-stop rule:
            # wait resets on improvement and increments otherwise, and the burn-in
            # gates the STOP rather than the counter, so epochs before it still
            # accumulate wait.
            if np.isfinite(val_mse) and val_mse < best_val - args.es_min_delta:
                best_val = val_mse
                best_predictor = predictor
                wait = 0
            else:
                wait += 1
            progress.set_postfix(train_loss=f"{train_loss:.4g}", val_mse=f"{val_mse:.4g}")
            progress.update(1)
            if (
                args.es_patience is not None
                and (epoch + 1) >= args.es_min_epochs
                and wait >= args.es_patience
            ):
                stop_reason = f"validation_patience:epoch{epoch}"
                print(
                    f"[train_upstream_inD] EARLY STOP at epoch={epoch} "
                    f"({stop_reason}); best val_mse={best_val:.6g}"
                )
                break
    finally:
        progress.close()
        if progress_owned_file is not None:
            progress_owned_file.close()
    if not np.isfinite(best_val):
        best_val = float(history[-1]["train_loss"]) if history else float("nan")
        best_predictor = predictor
    if stop_reason is None:
        stop_reason = f"epoch_ceiling:{args.epochs}"
    return best_predictor, history, best_val, stop_reason


def main() -> None:
    args = _build_parser().parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir = (
        str(Path(args.data_dir).expanduser().resolve()) if not args.use_stub else args.data_dir
    )

    print(
        f"[train_upstream_inD] predictor={args.predictor} seed={args.seed} "
        f"use_stub={args.use_stub} output_dir={output_dir}"
    )

    windows_train = _load_windows(args, "train", stub_seed=args.seed)
    windows_val = _load_windows(args, "val", stub_seed=args.seed + 1)
    print(
        f"[train_upstream_inD] windows: train={windows_train['context'].shape[0]} "
        f"val={windows_val['context'].shape[0]} (history={args.history} horizon={args.horizon} "
        f"stride={args.stride})"
    )

    l_ref_params = jnp.asarray([_L_REF], dtype=jnp.float64)
    root_key = jax.random.key(args.seed)

    # Train-split-only normalisation stats.
    state_mean, state_std = compute_state_norm_stats(windows_train["context"])
    predictor_key, train_key = jax.random.split(root_key)
    overrides = _collect_overrides(args, args.predictor)
    predictor = make_predictor(
        args.predictor,
        horizon=args.horizon,
        state_mean=state_mean,
        state_std=state_std,
        key=predictor_key,
        **overrides,
    )
    trained, history, best_val, stop_reason = _train_stage1(
        args, predictor, windows_train, windows_val, l_ref_params, key=train_key
    )
    config_dict: dict[str, Any] = {
        "kind": args.predictor,
        "horizon": args.horizon,
        "history": args.history,
        "stride": args.stride,
        "dt": _DT,
        "state_mean": state_mean,
        "state_std": state_std,
        **overrides,
    }

    save_predictor(output_dir, trained, config_dict)

    summary = {
        "predictor": args.predictor,
        "seed": args.seed,
        "use_stub": args.use_stub,
        "best_val_mse": best_val,
        # Every run records WHICH criterion ended it and where, rather than
        # leaving it to be reconstructed from logs afterwards.
        "termination_reason": stop_reason,
        "epochs_run": len(history),
        "epoch_ceiling": args.epochs,
        "early_stopping": {
            "patience": args.es_patience,
            "min_epochs": args.es_min_epochs,
            "min_delta": args.es_min_delta,
        },
        "num_train_windows": int(windows_train["context"].shape[0]),
        "num_val_windows": int(windows_val["context"].shape[0]),
        "val_chunk_size": args.val_chunk_size,
        "history_epochs": history,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))

    cli_config = {k: v for k, v in vars(args).items()}
    (output_dir / "config.json").write_text(json.dumps(cli_config, indent=2))

    print(f"[train_upstream_inD] done. best_val_mse={best_val:.6g} -> {output_dir}")


if __name__ == "__main__":
    main()
