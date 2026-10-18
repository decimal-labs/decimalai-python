"""Do REAL models call each framework's own skill loader, and does DecimalAI record the
version they got? (Phase 4's live check.)

Marked ``live_llm`` (deselected by default): it spends provider money. Run it with
``RUN_LIVE_LLM_TESTS=1`` and ``OPENAI_API_KEY`` in the environment each framework is graded in:

    RUN_LIVE_LLM_TESTS=1 DECIMAL_TEST_ENV=openai3 .venv/bin/python -m pytest tests/native/test_native_live.py -m live_llm -rs
    RUN_LIVE_LLM_TESTS=1 DECIMAL_TEST_ENV=openai2 .venv-openai2/bin/python -m pytest tests/native/test_native_live.py -m live_llm -rs

The release gate runs it as phase P7. ``LIVE_TRIALS`` (default 3) sets the trials per
framework; ``LIVE_LLM_OPENAI_MODEL`` the model.
"""

from __future__ import annotations

import os
from typing import Dict

import pytest

from . import live_contract as contract
from ._runner import run_scenario
from .conftest import missing_imports
from .test_native_column import FRAMEWORKS

pytestmark = pytest.mark.live_llm

#: Where each framework's real model class imports (the design's D3), and the key it needs.
LIVE_ENVS: Dict[str, str] = {
    "openai-agents": "openai3", "deepagents": "openai3", "pydantic-ai-harness": "openai3",
    "agent-framework": "openai3", "claude-agent-sdk": "openai3",
    "crewai": "openai2", "strands-agents": "openai2", "google-adk": "openai2",
}
KEYS: Dict[str, str] = {"claude-agent-sdk": "ANTHROPIC_API_KEY"}


@pytest.mark.parametrize("framework", sorted(FRAMEWORKS))
def test_a_real_model_calls_the_native_loader(native_project, framework):
    if os.environ.get("RUN_LIVE_LLM_TESTS") != "1":
        pytest.skip("live: set RUN_LIVE_LLM_TESTS=1 to spend provider money")
    from tests.conformance.environments import current_env

    if LIVE_ENVS[framework] != current_env():
        pytest.skip(f"{framework} runs live in the {LIVE_ENVS[framework]} environment")
    key = KEYS.get(framework, "OPENAI_API_KEY")
    if not os.environ.get(key):
        pytest.skip(f"{framework}: no {key}")
    missing = missing_imports(framework)
    if missing:
        pytest.fail(f"{framework} is not installed here (no {', '.join(missing)})")
    trials = int(os.environ.get("LIVE_TRIALS") or 3)
    result = run_scenario(f"tests.native.{FRAMEWORKS[framework]}:live", {"trials": trials},
                          cwd=native_project.root, live=True, timeout=600)
    verdict, lines = contract.grade(framework, result)
    report = f"{framework} ({result.get('versions')}):\n  " + "\n  ".join(lines)
    print(report)
    if verdict == "skip":
        pytest.skip(f"provider refused every trial: {lines}")
    assert verdict == "pass", report
