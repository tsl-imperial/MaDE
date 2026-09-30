"""Upstream integration."""

from made.upstream.base import UpstreamPredictor
from made.upstream.factory import load_predictor, make_predictor, save_predictor
from made.upstream.rnn_predictor import LSTMPredictor
from made.upstream.ssm_predictor import SSMPredictor
from made.upstream.stage_training import (
    apply_made_trajectory,
    apply_made_trajectory_with_controls,
    stage1_loss,
)
from made.upstream.transformer_predictor import TransformerPredictor

__all__ = [
    "LSTMPredictor",
    "SSMPredictor",
    "TransformerPredictor",
    "UpstreamPredictor",
    "apply_made_trajectory",
    "apply_made_trajectory_with_controls",
    "load_predictor",
    "make_predictor",
    "save_predictor",
    "stage1_loss",
]
