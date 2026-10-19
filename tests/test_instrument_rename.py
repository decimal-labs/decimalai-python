"""Every framework integration's entry point is `instrument()`.

`install()` was its old name on these modules, a deprecated alias through 0.17,
removed in 0.18: "install" also named adding a skill to a workspace (now
`decimalai skills add`), and one word for two unrelated actions was the problem.
"""
import importlib

import pytest

MODULES = [
    "adk", "anthropic", "autogen", "claude_agent_sdk", "langchain",
    "llamaindex", "openai_agents", "otel", "pydantic_ai",
]


@pytest.mark.parametrize("name", MODULES)
def test_instrument_is_the_one_entry_point(name):
    mod = importlib.import_module(f"decimalai.{name}")
    assert callable(getattr(mod, "instrument", None)), f"{name}.instrument() missing"
    assert not hasattr(mod, "install"), f"{name}.install() is back; the name is instrument()"
