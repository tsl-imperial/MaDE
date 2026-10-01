#!/usr/bin/env bash
# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).
# setup.sh: create the MaDE environment from a fresh clone (CPU). For Linux + CUDA 12 run
# `uv sync --frozen --extra cuda12` afterwards.
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi
uv sync --frozen
uv run python -c "import made; print('made imports OK')"
echo "Environment ready. Smoke test: uv run pytest"
