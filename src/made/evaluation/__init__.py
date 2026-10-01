# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Evaluation utilities."""

from made.evaluation.metrics import (
    METRIC_VERSION,
    EmpiricalEnvelope,
    ade,
    compute_metrics,
    dynamics_violation_known,
    dynamics_violation_learned,
    dynamics_violation_true,
    empirical_envelope_violation_magnitude,
    empirical_envelope_violation_rate,
    estimate_empirical_envelope,
    estimate_gt_reference_residual,
    fde,
    fidelity,
    gt_normalised_dynamics_residual,
    heading_jerk,
    inequality_violation_magnitude,
    inequality_violation_rate,
    jerk,
    kinematic_bicycle_inverse_controls,
)
from made.evaluation.perturbation import add_observation_noise, perturb_trajectories

__all__ = [
    "EmpiricalEnvelope",
    "METRIC_VERSION",
    "add_observation_noise",
    "ade",
    "compute_metrics",
    "dynamics_violation_known",
    "dynamics_violation_learned",
    "dynamics_violation_true",
    "empirical_envelope_violation_magnitude",
    "empirical_envelope_violation_rate",
    "estimate_empirical_envelope",
    "estimate_gt_reference_residual",
    "fde",
    "fidelity",
    "gt_normalised_dynamics_residual",
    "heading_jerk",
    "inequality_violation_magnitude",
    "inequality_violation_rate",
    "jerk",
    "kinematic_bicycle_inverse_controls",
    "perturb_trajectories",
]
