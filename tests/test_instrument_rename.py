"""Every tracing integration's entry point is `instrument()`.

`install()` was its old name on these modules, a deprecated alias through 0.17,
removed in 0.18: "install" also named adding a skill to a workspace (now
`decimalai skills add`), and one word for two unrelated actions was the problem.
Pydantic AI has neither since 0.19: `agent_config()` returns a capability, and
tracing is Pydantic AI's own OpenTelemetry. Google ADK has neither since 0.20:
`agent_config()` returns an instruction provider and a tool, and tracing is
`DecimalAIPlugin()` in ADK's `App(plugins=[...])`. Anthropic's entry point is
`agent_config()` since 0.21; its `instrument()` is kept only to raise, naming it,
and tracing is `decimalai.init(anthropic=True)`.
"""
import importlib

import pytest

MODULES = [
    "autogen", "claude_agent_sdk", "langchain",
    "llamaindex", "openai_agents", "otel",
]


@pytest.mark.parametrize("name", MODULES)
def test_instrument_is_the_one_entry_point(name):
    mod = importlib.import_module(f"decimalai.{name}")
    assert callable(getattr(mod, "instrument", None)), f"{name}.instrument() missing"
    assert not hasattr(mod, "install"), f"{name}.install() is back; the name is instrument()"


def test_pydantic_ai_has_load_agent_and_nothing_to_install():
    mod = importlib.import_module("decimalai.pydantic_ai")
    assert callable(mod.agent_config)
    assert not hasattr(mod, "instrument") and not hasattr(mod, "install")


def test_adk_has_load_agent_and_a_plugin_and_nothing_to_install():
    mod = importlib.import_module("decimalai.adk")
    assert callable(mod.agent_config) and callable(mod.DecimalAIPlugin)
    assert not hasattr(mod, "instrument") and not hasattr(mod, "install")


def test_anthropic_has_load_agent_and_its_instrument_names_it():
    mod = importlib.import_module("decimalai.anthropic")
    assert callable(mod.agent_config)
    assert not hasattr(mod, "install")
    with pytest.raises(TypeError, match="agent_config"):
        mod.instrument()
