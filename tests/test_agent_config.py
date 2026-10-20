"""`decimalai.load_agent()`: an agent's prompt and skills, for code that calls a model itself.

The per-run machinery every framework's ``load_agent()`` shares is tested in
test_agent_binding.py. These test the public surface of the framework-neutral one:
`load_agent()`, `AgentConfig.system()`, `tool_spec`, `load_skill()`, and the
record a turn leaves on a live OpenTelemetry run.
"""

from __future__ import annotations

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
    and a saved prompt with one placeholder."""
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


def test_load_agent_reads_the_prompt_once_and_routes_each_turn(platform):
    support = decimalai.load_agent("support", fallback="Be brief.")
    assert isinstance(support, decimalai.AgentConfig)
    assert platform.reads == [("support", {"fallback": "Be brief."})]
    assert (support.name, support.prompt, support.prompt_version) == ("support", "Hello {{customer.name}}.", 4)
    assert support.variables == ["customer.name"]

    text = support.system("My keyboard arrived cracked.", context=SAM)
    assert text.startswith("Hello Sam.\n\n## Available Skills")
    assert "| damaged-item |" in text and text.endswith(LOAD_SKILL_PROMPT_HINT)
    support.system("Where is my order?", context=SAM)
    assert list(platform.queries.values()) == ["My keyboard arrived cracked.", "Where is my order?"]
    assert len(platform.reads) == 1, "the prompt is a local copy, not a read per turn"


def test_tool_spec_lists_the_agents_skills_once_the_whole_list_is_known(platform):
    support = decimalai.load_agent("support")
    before = support.tool_spec
    assert before["name"] == LOAD_SKILL_TOOL_NAME and before["description"]
    assert "enum" not in before["parameters"]["properties"]["name"]
    support.system("Hi", context=SAM)
    assert support.tool_spec["parameters"]["properties"]["name"]["enum"] == ["damaged-item", "order-support"]


def test_load_skill_serves_a_skill_once_per_turn(platform):
    support = decimalai.load_agent("support")
    support.system("My keyboard arrived cracked.", context=SAM)
    assert support.load_skill("damaged-item").startswith("## Skill: damaged-item")
    assert "already loaded" in support.load_skill("damaged-item")
    support.system("And another thing", context=SAM)
    assert support.load_skill("damaged-item").startswith("## Skill: damaged-item")


def test_a_missing_placeholder_says_where_the_values_go(platform):
    support = decimalai.load_agent("support")
    with pytest.raises(decimalai.MissingPromptVariable, match=r"customer\.name\. Pass them as system\(\.\.\., context=\.\.\.\)"):
        support.system("Hi")
    assert support.system("Hi", context=SimpleNamespace(customer=SimpleNamespace(name="Ann"))).startswith("Hello Ann.")


def test_on_unavailable_must_be_continue_or_fail(platform):
    with pytest.raises(ValueError, match="on_unavailable"):
        decimalai.load_agent("support", on_unavailable="sometimes")


def test_a_turn_inside_an_opentelemetry_run_records_on_that_run(platform):
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import decimalai.otel as otel

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    support = decimalai.load_agent("support")
    with provider.get_tracer("test").start_as_current_span("run"):
        run_key = otel.current_run_key()
        support.system("My keyboard arrived cracked.", context=SAM)
        support.load_skill("damaged-item")
    rail = otel._pop_skill_rail(run_key)
    assert rail["routing_id"] and rail["offered"] == ["damaged-item", "order-support"]
    assert rail["loaded"] == ["damaged-item"]
    (span,) = exporter.get_finished_spans()
    record = otel._decimal_record_attributes(span)
    assert record == {
        "decimal.agent": "support", "decimal.prompt.source": "decimal",
        "decimal.prompt.version": 4, "decimal.prompt.content_hash": "c" * 64,
        "decimal.prompt.variables": ["customer.name"],
    }, "names and versions only: never a placeholder's value"


def test_the_exporter_carries_only_decimals_own_record_attributes():
    import decimalai.otel as otel

    span = SimpleNamespace(attributes={
        "decimal.prompt.version": 2, "decimal.prompt.variables": ("a", "b"),
        "gen_ai.prompt": "secret text", "http.url": "https://example.test",
    }, events=())
    assert otel._exported_attributes(span) == {
        "decimal.prompt.version": 2, "decimal.prompt.variables": ["a", "b"],
    }
