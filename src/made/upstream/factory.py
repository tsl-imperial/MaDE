"""Construction / serialisation factory for E05 upstream predictors.

``make_predictor`` is the single dispatch point from a string ``kind`` to a
concrete :class:`~made.upstream.base.UpstreamPredictor` subclass.
``save_predictor`` / ``load_predictor`` round-trip a trained predictor through
two files in a directory: ``predictor.eqx`` (leaf arrays, via
``eqx.tree_serialise_leaves``) and ``predictor_config.json`` (everything
needed to rebuild the un-trained template before deserialising leaves into
it — constructor kwargs, with the state-normalisation stats written as plain
lists, plus the ``kind`` tag and windowing/dt bookkeeping fields consumed by
callers such as ``scripts/train_upstream_inD.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from made.upstream.base import UpstreamPredictor
from made.upstream.rnn_predictor import LSTMPredictor
from made.upstream.ssm_predictor import SSMPredictor
from made.upstream.transformer_predictor import TransformerPredictor

__all__ = ["load_predictor", "make_predictor", "save_predictor"]

_REGISTRY: dict[str, type[UpstreamPredictor]] = {
    "lstm": LSTMPredictor,
    "ssm": SSMPredictor,
    "transformer": TransformerPredictor,
}

# Bookkeeping-only keys written to predictor_config.json that are NOT
# constructor kwargs of the predictor classes (they describe the windowing
# protocol / training dt a caller used, not the model architecture).
_NON_CTOR_KEYS: frozenset[str] = frozenset({"kind", "history", "stride", "dt"})


def make_predictor(
    kind: str,
    *,
    horizon: int,
    state_mean: jax.Array,
    state_std: jax.Array,
    key: jax.Array,
    **overrides: Any,
) -> UpstreamPredictor:
    """Construct an ``UpstreamPredictor`` of the given ``kind``.

    Args:
        kind: ``"lstm"``, ``"ssm"``, or ``"transformer"``.
        horizon: number of predicted future steps ``F``.
        state_mean, state_std: train-split normalisation stats, shape ``(state_dim,)``.
        key: PRNG key for parameter initialisation.
        **overrides: forwarded verbatim to the predictor constructor (e.g.
            ``hidden_size``, ``decoder_width``, ``d_model``, ``num_blocks``, ...).

    Returns:
        An initialised, untrained predictor instance.
    """
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown predictor kind {kind!r}. Supported: {sorted(_REGISTRY)}")
    cls = _REGISTRY[kind]
    return cls(
        horizon=horizon,
        state_mean=jnp.asarray(state_mean, dtype=jnp.float64),
        state_std=jnp.asarray(state_std, dtype=jnp.float64),
        key=key,
        **overrides,
    )


def save_predictor(
    directory: str | Path,
    model: UpstreamPredictor,
    config_dict: dict[str, Any],
) -> None:
    """Serialise ``model`` to ``<directory>/predictor.eqx`` + ``predictor_config.json``.

    ``config_dict`` must contain ``"kind"`` plus every constructor kwarg that
    ``make_predictor`` needs to rebuild the template (``horizon``,
    ``state_mean``, ``state_std``, and any architecture overrides), and MAY
    additionally carry windowing/dt bookkeeping fields (``history``,
    ``stride``, ``dt``) that are not constructor kwargs — those are written
    through unchanged but ignored by :func:`load_predictor` when rebuilding
    the template.
    """
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(directory / "predictor.eqx", model)

    serialisable = dict(config_dict)
    for key in ("state_mean", "state_std"):
        if key in serialisable:
            serialisable[key] = [float(v) for v in np.asarray(serialisable[key]).tolist()]
    (directory / "predictor_config.json").write_text(json.dumps(serialisable, indent=2))


def load_predictor(directory: str | Path) -> tuple[UpstreamPredictor, dict[str, Any]]:
    """Inverse of :func:`save_predictor`. Returns ``(model, config_dict)``.

    Rebuilds the untrained template from ``predictor_config.json`` (via
    :func:`make_predictor`, using a throwaway key — the real parameters are
    overwritten by ``eqx.tree_deserialise_leaves`` immediately after), then
    deserialises the trained leaves from ``predictor.eqx`` into it.
    """
    directory = Path(directory).expanduser().resolve()
    config = json.loads((directory / "predictor_config.json").read_text())
    kind = config["kind"]
    ctor_kwargs = {k: v for k, v in config.items() if k not in _NON_CTOR_KEYS}
    template = make_predictor(kind, key=jax.random.key(0), **ctor_kwargs)
    model = eqx.tree_deserialise_leaves(directory / "predictor.eqx", template)
    return model, config
