"""`decimalai.agent_config()`: an agent's prompt and skills, for code that calls a model itself.

The per-run machinery every framework's ``agent_config()`` shares is tested in
test_agent_binding.py. These test the public surface of the framework-neutral one:
`agent_config()`, `AgentConfig.run()` and the `Run` it returns (`system`, `tool_spec`,
`load_skill()`), the record a run leaves on a live OpenTelemetry run, and the 0.20
names, which now fail with the new ones in the message.
"""

from __future__ import annotations

import importlib
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import decimalai
import decimalai._agent as agent_mod
import decimalai._agent_binding as binding
from decimalai._agent import SavedPrompt
from decimalai.skill_router import LOAD_SKILL_PROMPT_HINT, LOAD_SKILL_TOOL_NAME, SkillRouter
from tests.test_agent_binding import FakePlatform

BASE = "http://agent-config.test"


@pytest.fixture
def platform(monkeypatch):
    """A configured SDK, a router answering from FakePlatform, background work inline,
    and a saved prompt with one variable."""
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    monkeypatch.setattr(cfg, "_config", DecimalConfig(api_key="dai_sk_test", base_url=BASE, enabled=True))
    monkeypatch.setattr(cfg, "_client", MagicMock())
    monkeypatch.setattr(binding, "_in_background", lambda fn, *args: fn(*args))
    fake = FakePlatform()
    router = SkillRouter(api_key="dai_sk_test", base_url=BASE, inject_body=False)
    monkeypatch.setattr(router, "_request", fake.request)
    monkeypatch.setattr(agent_mod, "_skill_router_singleton", router)
    fake.saved = SavedPrompt(agent_name="support", system_prompt="Hello {{customer.name}}.",
                             version_number=4, content_hash="c" * 64)
    fake.reads = []

    def read_prompt(name, **kwargs):
        fake.reads.append((name, kwargs))
        return fake.saved

    monkeypatch.setattr(agent_mod, "read_prompt", read_prompt)
    return fake


SAM = {"customer": {"name": "Sam"}}


def test_agent_config_reads_the_prompt_once_and_routes_each_run(platform):
    support = decimalai.agent_config("support", fallback="Be brief.")
    assert isinstance(support, decimalai.AgentConfig)
    assert platform.reads == [("support", {"fallback": "Be brief."})]
    assert (support.name, support.prompt, support.prompt_version) == ("support", "Hello {{customer.name}}.", 4)
    assert support.variables == ["customer.name"]

    run = support.run("My keyboard arrived cracked.", variables=SAM)
    assert isinstance(run, decimalai.Run)
    assert run.system.startswith("Hello Sam.\n\n## Available Skills")
    assert "| damaged-item |" in run.system and run.system.endswith(LOAD_SKILL_PROMPT_HINT)
    assert repr(run) == "Run(agent='support', skills=['damaged-item', 'order-support'])"
    support.run("Where is my order?", variables=SAM)
    assert list(platform.queries.values()) == ["My keyboard arrived cracked.", "Where is my order?"]
    assert len(platform.reads) == 1, "the prompt is a local copy, not a read per run"


def test_a_runs_tool_spec_lists_the_agents_skills(platform):
    support = decimalai.agent_config("support")
    spec = support.run("Hi", variables=SAM).tool_spec
    assert spec["name"] == LOAD_SKILL_TOOL_NAME and spec["description"]
    assert spec["parameters"]["properties"]["name"]["enum"] == ["damaged-item", "order-support"]
    assert spec["parameters"]["required"] == ["name"]


def test_load_skill_serves_a_skill_once_per_run(platform):
    support = decimalai.agent_config("support")
    first = support.run("My keyboard arrived cracked.", variables=SAM)
    assert first.load_skill("damaged-item").startswith("## Skill: damaged-item")
    assert "already loaded" in first.load_skill("damaged-item")
    second = support.run("And another thing", variables=SAM)
    assert second.load_skill("damaged-item").startswith("## Skill: damaged-item")
    assert "already loaded" in first.load_skill("damaged-item"), "each run keeps its own loads"


def test_a_run_serves_its_loads_from_any_thread(platform):
    """The run carries its own key: no thread or task has to be the one that made it."""
    support = decimalai.agent_config("support")
    run = support.run("My keyboard arrived cracked.", variables=SAM)
    support.run("Where is my order?", variables=SAM)  # a later run, made in this thread
    out = []
    worker = threading.Thread(target=lambda: out.append(run.load_skill("damaged-item")))
    worker.start()
    worker.join()
    assert out[0].startswith("## Skill: damaged-item")
    assert "already loaded" in run.load_skill("damaged-item"), "the load counted for the run that made it"
    assert agent_mod._RUN.get() is None, "a load leaves no run behind in the caller's context"


def test_a_missing_variable_says_where_the_values_go(platform):
    support = decimalai.agent_config("support")
    with pytest.raises(decimalai.MissingPromptVariable,
                       match=r"customer\.name\. Pass them as support\.run\(\.\.\., variables=\.\.\.\)"):
        support.run("Hi")
    run = support.run("Hi", variables=SimpleNamespace(customer=SimpleNamespace(name="Ann")))
    assert run.system.startswith("Hello Ann.")


def test_on_unavailable_must_be_continue_or_fail(platform):
    with pytest.raises(ValueError, match="on_unavailable"):
        decimalai.agent_config("support", on_unavailable="sometimes")


def test_a_run_inside_an_opentelemetry_run_records_on_that_run(platform):
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import decimalai.otel as otel

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    support = decimalai.agent_config("support")
    with provider.get_tracer("test").start_as_current_span("run"):
        run_key = otel.current_run_key()
        run = support.run("My keyboard arrived cracked.", variables=SAM)
        run.load_skill("damaged-item")
    rail = otel._pop_skill_rail(run_key)
    assert rail["routing_id"] and rail["offered"] == ["damaged-item", "order-support"]
    assert rail["loaded"] == ["damaged-item"]
    (span,) = exporter.get_finished_spans()
    record = otel._decimal_record_attributes(span)
    assert record == {
        "decimal.agent": "support", "decimal.prompt.source": "decimalai",
        "decimal.prompt.version": 4, "decimal.prompt.content_hash": "c" * 64,
        "decimal.prompt.variables": ["customer.name"],
    }, "names and versions only: never a variable's value"


def test_the_exporter_carries_only_decimals_own_record_attributes():
    import decimalai.otel as otel

    span = SimpleNamespace(attributes={
        "decimal.prompt.version": 2, "decimal.prompt.variables": ("a", "b"),
        "gen_ai.prompt": "secret text", "http.url": "https://example.test",
    }, events=())
    assert otel._exported_attributes(span) == {
        "decimal.prompt.version": 2, "decimal.prompt.variables": ["a", "b"],
    }


# ── the 0.20 names ───────────────────────────────────────────


@pytest.mark.parametrize("name, says", [
    ("system", "support.run(message, variables=...).system"),
    ("tool_spec", "run.tool_spec"),
    ("load_skill", "run.load_skill(name)"),
])
def test_what_moved_onto_a_run_says_where_it_went(platform, name, says):
    support = decimalai.agent_config("support")
    with pytest.raises(AttributeError) as raised:
        getattr(support, name)
    assert says in str(raised.value) and "0.21" in str(raised.value)
    assert not hasattr(support, name)
    with pytest.raises(AttributeError, match="has no attribute 'nonsense'"):
        support.nonsense  # noqa: B018 — any other missing name is a plain AttributeError


@pytest.mark.parametrize("module", ["decimalai", "decimalai.openai_agents", "decimalai.pydantic_ai", "decimalai.adk"])
def test_load_agent_still_imports_and_names_agent_config(module):
    """An old tutorial fails at the call, with the new name, rather than at the import."""
    mod = importlib.import_module(module)
    assert "load_agent" not in getattr(mod, "__all__", ())
    with pytest.raises(TypeError, match=r"renamed (decimalai\.)?agent_config\(\) in decimalai 0\.21"):
        mod.load_agent("support", fallback="Be brief.")


def test_the_adk_plugins_old_name_names_the_new_one():
    from decimalai.adk import DecimalaiPlugin

    with pytest.raises(TypeError, match=r"renamed DecimalAIPlugin\(\) in decimalai 0\.21"):
        DecimalaiPlugin()
