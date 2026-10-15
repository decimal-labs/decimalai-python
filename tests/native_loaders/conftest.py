"""Session fixtures: the two projects this suite grades, each observed once per run.

``graded``    the project the eight loaders must read cleanly (``source.graded_project``)
``negative``  SDK 0.13.9's output for the same skills (``source.negative_control_project``),
              graded against the same expectation; the contract must fail on it.
"""

from __future__ import annotations

import pytest

from . import contract, oracle, source
from .runner import Run


@pytest.fixture(scope="session")
def graded(tmp_path_factory) -> Run:
    root = tmp_path_factory.mktemp("native-graded").resolve()
    project = source.graded_project(root / "project")
    return Run("graded", project, contract.expectation_from_lock(project), root)


@pytest.fixture(scope="session")
def negative(tmp_path_factory) -> Run:
    root = tmp_path_factory.mktemp("native-0139").resolve()
    project = source.negative_control_project(root / "project")
    return Run("v0139", project, contract.expectation_for(project, oracle.build().lock), root)
