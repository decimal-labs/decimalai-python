"""Which hermetic test environment this is, and which drivers it must run.

No single environment can install every framework the SDK supports. Upstream has
split on openai 3: openai-agents and pydantic-ai-slim[openai] need ``openai>=3``,
while crewai, litellm and llama-index-llms-openai still pin ``openai<3``. So the
conformance matrix is graded in two environments, each built from one extra in
``pyproject.toml`` (``TEST_ENVS`` below):

* ``openai3`` — ``pip install -e ".[dev,conformance-tests]"``. Every driver except
  crewai and adk.
* ``openai2`` — ``pip install -e ".[conformance-tests-openai2]"``. crewai and adk.
* llamaindex and generic-otel need neither openai, and are graded in both.

``DECIMAL_TEST_ENV=openai3|openai2`` says which one a run is in. Unset means
``openai3``, which is what ``.venv`` and a plain ``.[dev]`` install are. It does
two things:

* with ``DECIMAL_CONFORMANCE_REQUIRE_ALL=1``, every driver that BELONGS to this
  environment must be runnable
  (``test_coverage.test_no_driver_is_silently_unavailable``), and the drivers that
  belong only to the other one are named on every run
  (``test_coverage.test_drivers_graded_in_another_environment_are_named``);
* ``openai2`` also narrows the UNIT suite to the files that need the openai 2
  packages (``tests/conftest.py``), because the rest of the unit suite imports
  openai 3 frameworks that environment cannot hold.

Every driver belongs to at least one environment; ``test_coverage`` fails on a
driver with none, so a new framework cannot land graded nowhere.
"""

from __future__ import annotations

import os
from typing import Dict, FrozenSet, Optional

#: The variable that names the environment.
ENV_VAR = "DECIMAL_TEST_ENV"

#: The environment a run is in when ``DECIMAL_TEST_ENV`` is unset.
DEFAULT_ENV = "openai3"

#: Environment name -> the install that builds it. Each extra is the single
#: source for what that environment needs; never retype its list elsewhere.
TEST_ENVS: Dict[str, str] = {
    "openai3": 'pip install -e ".[dev,conformance-tests]"',
    "openai2": 'pip install -e ".[conformance-tests-openai2]"',
}

_OPENAI3 = frozenset({"openai3"})
_OPENAI2 = frozenset({"openai2"})
_BOTH = frozenset(TEST_ENVS)

#: Driver name -> the environment(s) where its column is GUARANTEED to be graded.
#: A driver may also run elsewhere when its imports happen to be installed there;
#: this is where a missing import is a build failure rather than a fact of life.
DRIVER_ENVS: Dict[str, FrozenSet[str]] = {
    "langchain": _OPENAI3,
    "openai-agents": _OPENAI3,      # openai-agents needs openai>=3
    "pydantic-ai": _OPENAI3,        # pydantic-ai-slim[openai] needs openai>=3
    "anthropic": _OPENAI3,
    "claude-agent-sdk": _OPENAI3,
    "crewai": _OPENAI2,             # crewai pins openai<3
    # google-adk itself installs in either environment, but the driver's stub
    # model is LiteLLM's custom provider reached through ADK's LiteLlm, and
    # litellm pins openai<3.
    "adk": _OPENAI2,
    # A stub CustomLLM on llama-index-core, which both install sets carry.
    "llamaindex": _BOTH,
    # Needs only opentelemetry-sdk, a core dependency.
    "generic-otel": _BOTH,
}


def current_env() -> str:
    """The environment this process is in. Raises on a value nobody defined."""
    value = (os.environ.get(ENV_VAR) or "").strip() or DEFAULT_ENV
    if value not in TEST_ENVS:
        raise ValueError(
            f"{ENV_VAR}={value!r} is not a test environment. Known: "
            f"{', '.join(sorted(TEST_ENVS))} (unset means {DEFAULT_ENV!r})."
        )
    return value


def graded_elsewhere(driver_name: str, env: Optional[str] = None) -> Optional[str]:
    """The environment(s) that grade ``driver_name``, if this one does not.

    ``None`` means the driver belongs HERE (or is unmapped, which
    ``test_coverage`` fails on separately) — so a missing import is a failure
    under ``DECIMAL_CONFORMANCE_REQUIRE_ALL=1``.
    """
    env = env or current_env()
    homes = DRIVER_ENVS.get(driver_name)
    if not homes or env in homes:
        return None
    return ", ".join(sorted(homes))
