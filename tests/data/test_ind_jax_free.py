# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Regression test: made.data.ind submodules must be importable without JAX.

Contract: every public symbol on made.data.ind must be importable in an env
without JAX. JAX is allowed only inside function bodies. Enforced here via
subprocess isolation so we never mutate the parent process's sys.modules.
"""

import os
import subprocess
import sys

import pytest

_SUBMODULES = [
    "made.data.ind",
    "made.data.ind.loader",
    "made.data.ind.preprocess",
    "made.data.ind.validate",
    "made.data.ind.ingest",
    "made.data.ind.splits",
    "made.data.ind.transform",
    "made.data.ind.constants",
]

_HELPER = """
import sys

class _BlockJAX:
    def find_spec(self, name, path, target=None):
        if name == 'jax' or name.startswith('jax.'):
            raise ImportError(f'JAX blocked for contract test: {{name}}')
        return None

sys.meta_path.insert(0, _BlockJAX())
sys.modules['jax'] = None  # belt-and-braces

import {module}
"""


@pytest.mark.parametrize("module_name", _SUBMODULES)
def test_submodule_importable_without_jax(module_name: str) -> None:
    """Verify submodule importable without jax."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src_dir = os.path.join(repo_root, "src")

    env = os.environ.copy()
    env["PYTHONPATH"] = src_dir
    env["JAX_PLATFORMS"] = "cpu"

    code = _HELPER.format(module=module_name)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Importing {module_name!r} without JAX failed.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert result.stderr == "", (
        f"Importing {module_name!r} produced stderr output.\n"
        f"stderr: {result.stderr}"
    )
