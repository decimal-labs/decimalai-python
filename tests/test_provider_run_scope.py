"""The run scope for the raw-provider rail (``decimalai.providers.agent_run``).

These tests drive the REAL OpenTelemetry SDK — a ``TracerProvider`` with the
real ``DecimalSpanExporter`` on it — and assert on the ``RunTrace`` objects that
come out the far end. That is deliberate: the defect being covered is entirely
about OTel *context* (which span is current, which ContextVar is set, which
trace_id a span lands in), and a mocked tracer has no context to get wrong.

The defect, for anyone reading this later: a provider instrumentor
(OpenInference's openai / anthropic / google-genai) emits one span per SDK call
and nothing above it. Each span is therefore a root, in its own trace_id, and
the exporter finalizes a trace when a parentless root arrives — so a two-call
tool-use loop arrived as TWO one-span traces, both filed under whichever agent
name the exporter was built with. ``agent_run`` supplies the missing parent and
the missing per-run name.

A tracer is used in place of the instrumentors here because the seam under test
is the same one they use: ``tracer.start_as_current_span``, parented by whatever
is current. Installing openai + its instrumentor to assert the same thing would
test OpenInference, not this.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from decimalai import providers
from decimalai.otel import DecimalSpanExporter, _active_agent_name

pytest.importorskip("opentelemetry.sdk.trace")


# ── plumbing ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _sdk_enabled():
    """A configured, mocked SDK — the exporter refuses to assemble without one."""
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    prev_config, prev_client = cfg._config, cfg._client
    cfg._config = DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {
        "manifest_id": "test-manifest-id", "status": "active",
    }
    yield cfg
    cfg._config, cfg._client = prev_config, prev_client


@pytest.fixture(autouse=True)
def _clean_context():
    """The run scope publishes an agent name; never leak it into the next test."""
    token = _active_agent_name.set(None)
    yield
    _active_agent_name.reset(token)


@pytest.fixture(autouse=True)
def _clean_module_state():
    prev = (providers._pipeline_provider, providers._last_provider,
            set(providers._instrumented))
    yield
    (providers._pipeline_provider, providers._last_provider) = prev[0], prev[1]
    providers._instrumented.clear()
    providers._instrumented.update(prev[2])


class _Pipeline:
    """A caller-owned TracerProvider wired exactly as ``_ensure_pipeline`` wires one."""

    def __init__(self, agent_name: str | None = None) -> None:
        from opentelemetry.sdk.trace import TracerProvider

        self.provider = TracerProvider()
        providers._ensure_pipeline(agent_name, self.provider)
        self.tracer = self.provider.get_tracer("test")

    def traces(self, cfg: Any) -> List[Any]:
        from decimalai._config import _sender

        _sender.flush()
        return [c[0][0] for c in cfg._client.ingest_trace.call_args_list]

    def llm_span(self, name: str = "ChatCompletion") -> None:
        """One span shaped like what a provider instrumentor emits."""
        with self.tracer.start_as_current_span(name, attributes={
            "gen_ai.request.model": "stub-model-1",
            "gen_ai.system": "openai",
            "gen_ai.usage.input_tokens": 11,
            "gen_ai.usage.output_tokens": 3,
        }):
            pass


def _by_agent(traces: List[Any]) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {}
    for t in traces:
        out.setdefault(t.agent_name, []).append(t)
    return out


# ── the defect, and the fix ──────────────────────────────────────────────────


class TestFragmentation:
    def test_without_a_scope_each_provider_call_is_its_own_trace(self, _sdk_enabled):
        """The defect, pinned: two calls, two traces, one span each, no parent."""
        pipe = _Pipeline(agent_name="agent-a")

        pipe.llm_span()
        pipe.llm_span()

        traces = pipe.traces(_sdk_enabled)
        assert len(traces) == 2
        assert [len(t.spans) for t in traces] == [1, 1]
        assert all(s.parent_span_id is None for t in traces for s in t.spans)

    def test_a_scope_makes_one_run_one_trace(self, _sdk_enabled):
        """Same two calls inside ``agent_run`` → ONE trace, both calls nested."""
        pipe = _Pipeline(agent_name="agent-a")

        with providers.agent_run("agent-a", tracer_provider=pipe.provider):
            pipe.llm_span()
            pipe.llm_span()

        traces = pipe.traces(_sdk_enabled)
        assert len(traces) == 1
        trace = traces[0]
        assert len(trace.spans) == 3
        ids = {s.id for s in trace.spans}
        children = [s for s in trace.spans if s.parent_span_id in ids]
        assert len(children) == 2
        assert len(trace.llm_calls) == 2

    def test_the_scope_adds_the_parent_and_invents_nothing_else(self, _sdk_enabled):
        """One added span, and it is the one that really wrapped the calls.

        Guards the line this fix is not allowed to cross: the parent span is
        real (it was open for the duration), so it may be reported. Steps the
        framework never emitted — a tool span derived from an LLM span's
        attributes, say — stay absent.
        """
        pipe = _Pipeline(agent_name="agent-a")

        with providers.agent_run("agent-a", tracer_provider=pipe.provider):
            pipe.llm_span()

        (trace,) = pipe.traces(_sdk_enabled)
        assert len(trace.spans) == 2  # the run span + the one call that happened
        roots = [s for s in trace.spans if s.parent_span_id is None]
        assert [s.name for s in roots] == ["agent.run"]

    def test_nesting_does_not_start_a_second_trace(self, _sdk_enabled):
        pipe = _Pipeline(agent_name="agent-a")

        with providers.agent_run("outer", tracer_provider=pipe.provider):
            with providers.agent_run("inner", tracer_provider=pipe.provider):
                pipe.llm_span()

        traces = pipe.traces(_sdk_enabled)
        assert len(traces) == 1
        assert len(traces[0].spans) == 3


class TestIdentity:
    def test_second_agent_in_one_process_is_not_filed_under_the_first(
        self, _sdk_enabled
    ):
        """The exporter's own name is fixed when it is built; the scope's is not."""
        pipe = _Pipeline(agent_name="agent-a")

        with providers.agent_run("agent-a", tracer_provider=pipe.provider):
            pipe.llm_span()
        with providers.agent_run("agent-b", tracer_provider=pipe.provider):
            pipe.llm_span()

        assert sorted(_by_agent(pipe.traces(_sdk_enabled))) == ["agent-a", "agent-b"]

    def test_concurrent_runs_each_keep_their_own_name(self, _sdk_enabled):
        """Eight lanes, eight agents — a module global would hand all eight one name."""
        pipe = _Pipeline(agent_name="agent-a")

        def lane(i: int) -> None:
            with providers.agent_run(f"agent-{i}", tracer_provider=pipe.provider):
                pipe.llm_span()

        threads = [threading.Thread(target=lane, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        by_agent = _by_agent(pipe.traces(_sdk_enabled))
        assert sorted(by_agent) == sorted(f"agent-{i}" for i in range(8))
        assert all(len(v) == 1 for v in by_agent.values())

    def test_no_name_leaves_the_exporter_default_alone(self, _sdk_enabled):
        pipe = _Pipeline(agent_name="agent-a")

        with providers.agent_run(tracer_provider=pipe.provider):
            pipe.llm_span()

        (trace,) = pipe.traces(_sdk_enabled)
        assert trace.agent_name == "agent-a"

    def test_the_name_is_released_on_exit(self, _sdk_enabled):
        pipe = _Pipeline(agent_name="agent-a")
        with providers.agent_run("agent-b", tracer_provider=pipe.provider):
            assert _active_agent_name.get() == "agent-b"
        assert _active_agent_name.get() is None


class TestErrorPath:
    def test_a_raising_run_still_emits_one_errored_trace(self, _sdk_enabled):
        """A failed run recorded as a success is worse than no trace."""
        from decimalai.schema.common import Status

        pipe = _Pipeline(agent_name="agent-a")

        with pytest.raises(RuntimeError):
            with providers.agent_run("agent-a", tracer_provider=pipe.provider):
                pipe.llm_span()
                raise RuntimeError("the provider refused this on purpose")

        (trace,) = pipe.traces(_sdk_enabled)
        assert trace.status == Status.ERROR
        assert _active_agent_name.get() is None  # released despite the raise


class TestPipelineWiring:
    def test_ensure_pipeline_installs_the_stamper_alongside_the_exporter(self):
        """Without the stamper, a per-run name has nowhere to land on a span."""
        from opentelemetry.sdk.trace import TracerProvider

        from decimalai.otel import _AgentNameStamper

        tp = TracerProvider()
        providers._ensure_pipeline("agent-x", tracer_provider=tp)

        installed = getattr(tp, "_active_span_processor")._span_processors
        assert any(isinstance(p, _AgentNameStamper) for p in installed)
        assert any(isinstance(getattr(p, "span_exporter", None), DecimalSpanExporter)
                   for p in installed)

    def test_ensure_pipeline_records_the_provider_agent_run_defaults_to(self):
        from opentelemetry.sdk.trace import TracerProvider

        tp = TracerProvider()
        providers._ensure_pipeline("agent-x", tracer_provider=tp)
        assert providers._last_provider is tp

    def test_a_later_instrument_call_still_publishes_its_agent_name(
        self, monkeypatch, _sdk_enabled
    ):
        """The instrumentors are process-wide singletons, so a second
        ``init(openai=True, agent_name="B")`` has nothing left to instrument.
        The name it carries is the one thing about it that IS new."""
        monkeypatch.setattr(providers, "_sdk_present", lambda module: True)
        monkeypatch.setattr(providers, "_load_instrumentor", lambda spec: MagicMock())
        monkeypatch.setattr(
            providers, "_ensure_pipeline", lambda name, tp=None: MagicMock()
        )

        providers.instrument(openai=True, agent_name="agent-a")
        providers.instrument(openai=True, agent_name="agent-b")

        assert _active_agent_name.get() == "agent-b"
