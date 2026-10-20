"""Every tracing integration's entry point is `instrument()`.

`install()` was its old name on these modules, a deprecated alias through 0.17,
removed in 0.18: "install" also named adding a skill to a workspace (now
`decimalai skills add`), and one word for two unrelated actions was the problem.
Pydantic AI has neither since 0.19: `load_agent()` returns a capability, and
tracing is Pydantic AI's own OpenTelemetry. Google ADK has neither since 0.20:
`load_agent()` returns an instruction provider and a tool, and tracing is
`DecimalaiPlugin()` in ADK's `App(plugins=[...])`.
"""
import importlib

import pytest

MODULES = [
    "anthropic", "autogen", "claude_agent_sdk", "langchain",
    "llamaindex", "openai_agents", "otel",
]


@pytest.mark.parametrize("name", MODULES)
def test_instrument_is_the_one_entry_point(name):
    mod = importlib.import_module(f"decimalai.{name}")
    assert callable(getattr(mod, "instrument", None)), f"{name}.instrument() missing"
    assert not hasattr(mod, "install"), f"{name}.install() is back; the name is instrument()"


def test_pydantic_ai_has_load_agent_and_nothing_to_install():
    mod = importlib.import_module("decimalai.pydantic_ai")
    assert callable(mod.load_agent)
    assert not hasattr(mod, "instrument") and not hasattr(mod, "install")


def test_adk_has_load_agent_and_a_plugin_and_nothing_to_install():
    mod = importlib.import_module("decimalai.adk")
    assert callable(mod.load_agent) and callable(mod.DecimalaiPlugin)
    assert not hasattr(mod, "instrument") and not hasattr(mod, "install")
