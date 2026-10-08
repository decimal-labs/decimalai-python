"""A CrewAI run's trace is handed to its caller, filed under its caller, and flushed.

Three gaps the fleet found when it tried to switch its CrewAI sessions onto the
product path (``decimalai.crewai.instrument(enable_skill_loader=True)``), each of
which this file pins:

1. **No observer.** The CrewAI trace is assembled from spans by the OTel
   exporter, on the exporter's thread, under an id minted there, so nothing the
   caller holds names it — and that trace carries the platform's usage receipt
   for the run. ``instrument(on_trace=...)`` now has the LangChain / ADK
   contract: called once per run (failed runs too) with a detached copy of the
   finished ``RunTrace``, before export, and an observer that raises costs
   neither the run nor its trace.
2. **No parent link.** The router stamps a delivery onto the caller's enclosing
   ``decimalai.start_trace()`` too, and the platform drops that duplicate credit
   only for a LINKED child (``public_skill_usage.suppress_wrapper_duplicates``).
   A run is now linked the way LangChain and ADK link theirs: the enclosing
   trace's id, captured when the run starts.
3. **No flush.** ``decimalai.flush()`` drained the sender but never the OTel
   batch processor the CrewAI trace leaves through, so a short-lived process
   could report before the trace existed. It now exports those spans first,
   bounded, and touches nothing for a process that never set up CrewAI tracing.

Two layers, like ``test_crewai_skill_delivery.py``:

* ``TestTheSeam`` / ``TestFlush`` drive the real OpenTelemetry SDK and the real
  ``DecimalSpanExporter`` with spans started under the CrewAI instrumentor's own
  scope name — no crewai needed, so they run everywhere.
* ``TestRealCrewAIRun`` runs a real ``Crew.kickoff`` under the real OpenInference
  CrewAI instrumentor, with a model that answers from a script instead of the
  network.

The wire-level verdict for every observer adapter — the trace on the wire is the
one the observer was handed, and it names the enclosing trace — is the
conformance suite's: C15 and C16 in ``tests/conformance``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import logging
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

#: The instrumentation scope the OpenInference CrewAI instrumentor's spans carry.
CREWAI_SCOPE = "openinference.instrumentation.crewai"


def _real_crewai_installed() -> bool:
    try:
        return (
            importlib.util.find_spec("crewai.hooks") is not None
            and importlib.util.find_spec(CREWAI_SCOPE) is not None
        )
    except (ImportError, ValueError):
        return False


HAS_REAL_CREWAI = _real_crewai_installed()


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_sdk():
    """Fresh SDK config and adapter globals; nothing leaks into other tests."""
    import decimalai._config as cfg
    import decimalai.crewai as dc
    from decimalai._config import DecimalConfig
    from decimalai.otel import _active_agent_name, _reset_run_links, _reset_skill_rails

    saved_cfg = (cfg._config, cfg._client)
    cfg._config = DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True,
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {
        "manifest_id": "test-manifest-id", "status": "active",
    }
    saved = (
        dc._tracing_installed, dc._skill_loader_installed, dc._install_agent_name,
        dc._skill_router_singleton, dc._instrument_config, list(dc._tracer_providers),
    )
    dc._tracing_installed = True  # no global tracer provider from a unit test
    dc._skill_loader_installed = False
    dc._install_agent_name = None
    dc._skill_router_singleton = None
    dc._instrument_config = dc._InstrumentationConfig()
    dc._tracer_providers.clear()
    token = _active_agent_name.set(None)
    _reset_run_links()
    _reset_skill_rails()
    yield
    _reset_run_links()
    _reset_skill_rails()
    _active_agent_name.reset(token)
    (
        dc._tracing_installed, dc._skill_loader_installed, dc._install_agent_name,
        dc._skill_router_singleton, dc._instrument_config, providers,
    ) = saved
    dc._tracer_providers[:] = providers
    cfg._config, cfg._client = saved_cfg


def _sent() -> List[Any]:
    """Every trace handed to the sender so far, once the sender is drained."""
    import decimalai._config as cfg
    from decimalai._config import _sender

    _sender.flush()
    return [call.args[0] for call in cfg._client.ingest_trace.call_args_list]


class _Pipeline:
    """What ``_activate_crewai_instrumentation`` builds, minus the instrumentor.

    A real ``TracerProvider`` and the real ``DecimalSpanExporter``, wired the way
    the SDK wires CrewAI's provider (``_wire_tracer_provider``), and a tracer
    under the CrewAI instrumentor's own scope, so the spans are a CrewAI run's as
    far as anything downstream can tell.

    ``batched=False`` exports each trace the moment its root span ends, on the
    thread that ended it — the strictest case for "never raises into the run".
    ``batched=True`` holds spans in a ``BatchSpanProcessor`` with an hour-long
    schedule, so only an explicit flush can ship them.
    """

    def __init__(self, *, batched: bool = False) -> None:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            SimpleSpanProcessor,
        )

        import decimalai.crewai as dc
        from decimalai.otel import DecimalSpanExporter

        self.provider = TracerProvider()
        exporter = DecimalSpanExporter(agent_name="support")
        self.provider.add_span_processor(
            BatchSpanProcessor(exporter, schedule_delay_millis=3_600_000)
            if batched else SimpleSpanProcessor(exporter)
        )
        dc._wire_tracer_provider(self.provider)
        self.crewai = self.provider.get_tracer(CREWAI_SCOPE)
        self.other = self.provider.get_tracer("openinference.instrumentation.openai")

    def run(self, name: str = "Crew.kickoff", *, children: int = 2, fail: bool = False) -> None:
        """One CrewAI run: a crew span with task/agent spans inside it."""
        with self.crewai.start_as_current_span(name):
            for i in range(children):
                with self.crewai.start_as_current_span(f"Task._execute_core.{i}"):
                    pass
            if fail:
                raise RuntimeError("the crew failed")


def _instrument(**kwargs: Any) -> None:
    from decimalai.crewai import instrument

    instrument(agent_name="support", **kwargs)


def _root_name(trace: Any) -> Optional[str]:
    """The name of a trace's root span — the run it records."""
    return next((s.name for s in trace.spans if s.parent_span_id is None), None)


# ── layer 1: the seam, graded on every run of the suite ─────────────────────


class TestTheSeam:
    def test_on_trace_is_a_named_keyword_parameter(self):
        """What the fleet's arming check reads (``accepts_usage_observer``): a
        named keyword parameter, never a ``**kwargs`` that would swallow it."""
        from decimalai.crewai import instrument

        param = inspect.signature(instrument).parameters.get("on_trace")
        assert param is not None and param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is None

    def test_a_run_inside_start_trace_is_filed_as_its_child(self):
        from decimalai.generic import start_trace

        pipe = _Pipeline()
        with start_trace(agent_name="session", auto_send=False) as outer:
            pipe.run()
        [trace] = _sent()
        assert trace.parent_trace_id == outer.get_trace_id()

    def test_the_parent_is_captured_when_the_run_starts(self):
        """The run's root span ends — and its trace is assembled — after the
        enclosing trace has closed, and on another thread. Reading the parent at
        export instead of at the run's start would silently unlink it."""
        from decimalai.generic import start_trace

        pipe = _Pipeline()
        with start_trace(agent_name="session", auto_send=False) as outer:
            crew_span = pipe.crewai.start_span("Crew.kickoff")
        ender = threading.Thread(target=crew_span.end)
        ender.start()
        ender.join()
        [trace] = _sent()
        assert trace.parent_trace_id == outer.get_trace_id()

    def test_a_run_outside_any_trace_stays_a_root(self):
        _Pipeline().run()
        [trace] = _sent()
        assert trace.parent_trace_id is None

    def test_only_crewai_runs_are_linked_and_observed(self):
        """The pipeline also carries provider-SDK spans (``init(crewai=True)``
        instruments the importable provider SDKs onto it). A bare model call
        outside any crew is not a CrewAI run: no link, no observer."""
        from decimalai.generic import start_trace

        seen: List[Any] = []
        _instrument(on_trace=seen.append)
        pipe = _Pipeline()
        with start_trace(agent_name="session", auto_send=False):
            with pipe.other.start_as_current_span("ChatCompletion"):
                pass
        [trace] = _sent()
        assert trace.parent_trace_id is None
        assert seen == []

    def test_the_observer_is_called_once_per_run_with_a_detached_copy(self):
        """One run, three CrewAI spans, one call. And the observer is handed a
        COPY: what it does to the trace must not reach the trace exported."""
        from decimalai.generic import start_trace
        from decimalai.schema.trace import RunTrace

        seen: List[RunTrace] = []

        def observe(trace: RunTrace) -> None:
            seen.append(trace.model_copy(deep=True))
            trace.agent_name = "observer-mutation"
            trace.skills_delivered.append("observer-mutation")
            trace.spans.clear()

        _instrument(on_trace=observe)
        pipe = _Pipeline()
        with start_trace(agent_name="session", auto_send=False) as outer:
            pipe.run(children=2)
        [trace] = _sent()
        assert len(seen) == 1
        assert isinstance(seen[0], RunTrace)
        assert seen[0].id == trace.id
        assert seen[0].parent_trace_id == trace.parent_trace_id == outer.get_trace_id()
        assert trace.agent_name == "support"
        assert "observer-mutation" not in trace.skills_delivered
        assert len(trace.spans) == len(seen[0].spans) == 3

    def test_an_observer_that_raises_costs_neither_the_run_nor_its_trace(self, caplog):
        """Exported inline here — the root span's end runs the exporter on the
        caller's thread — so an observer failure that escaped would surface
        inside the caller's own run."""
        calls: List[Any] = []

        def broken(trace: Any) -> None:
            calls.append(trace.id)
            raise RuntimeError("observer failed")

        _instrument(on_trace=broken)
        with caplog.at_level(logging.ERROR, logger="decimalai.otel"):
            _Pipeline().run()  # must not raise
        [trace] = _sent()
        assert calls == [trace.id]
        assert "Trace observer failed" in caplog.text

    def test_a_failed_run_is_observed_once_and_marked_errored(self):
        from decimalai.schema.common import Status

        seen: List[Any] = []
        _instrument(on_trace=seen.append)
        with pytest.raises(RuntimeError, match="the crew failed"):
            _Pipeline().run(fail=True)
        [trace] = _sent()
        assert [t.id for t in seen] == [trace.id]
        assert seen[0].status == trace.status == Status.ERROR

    def test_reinstrumenting_moves_only_runs_that_start_afterwards(self):
        """The observer is captured at a run's START. An ``instrument()`` call
        made while a run is in flight must not hand that run to a different
        observer — and ``on_trace=None`` removes it for the runs after."""
        first: List[Any] = []
        second: List[Any] = []
        pipe = _Pipeline()

        _instrument(on_trace=first.append)
        in_flight = pipe.crewai.start_span("Crew.kickoff")
        _instrument(on_trace=second.append)
        pipe.run("Crew.kickoff.second")
        in_flight.end()
        _instrument(on_trace=None)
        pipe.run("Crew.kickoff.third")

        traces = {_root_name(t): t for t in _sent()}
        assert [t.id for t in first] == [traces["Crew.kickoff"].id]
        assert [t.id for t in second] == [traces["Crew.kickoff.second"].id]
        assert len(traces) == 3

    def test_concurrent_runs_keep_their_own_parent(self):
        from decimalai.generic import start_trace

        pipe = _Pipeline()
        parents: Dict[str, str] = {}
        barrier = threading.Barrier(2)

        def lane(name: str) -> None:
            with start_trace(agent_name=name, auto_send=False) as outer:
                parents[name] = outer.get_trace_id()
                with pipe.crewai.start_as_current_span(f"{name}.kickoff"):
                    barrier.wait(timeout=5)  # both runs in flight at once

        threads = [threading.Thread(target=lane, args=(n,)) for n in ("first", "second")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        by_root = {_root_name(t).split(".")[0]: t for t in _sent()}
        assert {n: t.parent_trace_id for n, t in by_root.items()} == parents
        assert len(set(parents.values())) == 2

    def test_a_link_is_released_when_its_trace_cannot_be_assembled(self, monkeypatch):
        """Like the skill rail: a run whose assembly raised must not hold a slot
        in the process-wide store, nor hand its observer to a later trace."""
        from decimalai.otel import DecimalSpanExporter, _run_links

        seen: List[Any] = []
        _instrument(on_trace=seen.append)

        def broken(self, spans):
            raise RuntimeError("assembly failed")

        monkeypatch.setattr(DecimalSpanExporter, "_assemble_trace", broken)
        _Pipeline().run()
        assert len(_run_links) == 0
        assert seen == []

    def test_a_disabled_sdk_observes_nothing(self):
        """No trace is assembled when tracing is off, so there is nothing to
        hand over — the same as the LangChain and ADK observers."""
        import decimalai._config as cfg
        from decimalai._config import DecimalConfig
        from decimalai.otel import _run_links

        seen: List[Any] = []
        _instrument(on_trace=seen.append)
        cfg._config = DecimalConfig(
            api_key="dai_sk_test", base_url="http://localhost:8000", enabled=False,
        )
        _Pipeline().run()
        assert seen == [] and _sent() == []
        assert len(_run_links) == 0


class TestWiring:
    @staticmethod
    def _fake_instrumentor(monkeypatch, *, fail: bool = False) -> MagicMock:
        instrumentor = MagicMock(name="CrewAIInstrumentor_instance")
        if fail:
            instrumentor.instrument.side_effect = RuntimeError("semconv mismatch")
        mod = types.ModuleType(CREWAI_SCOPE)
        mod.CrewAIInstrumentor = MagicMock(return_value=instrumentor)
        for name in ("openinference", "openinference.instrumentation"):
            if name not in sys.modules:
                monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, CREWAI_SCOPE, mod)
        import decimalai.providers as providers

        monkeypatch.setattr(providers, "_sdk_present", lambda _mod: False)
        return instrumentor

    def test_activation_links_and_registers_the_provider_once(self, monkeypatch):
        """``_activate_crewai_instrumentation`` is the one path both
        ``init(crewai=True)`` and ``crewai.instrument()`` take, so it is where
        the run linker goes on — once per provider, however often it runs."""
        from opentelemetry.sdk.trace import TracerProvider

        import decimalai
        import decimalai.crewai as dc

        self._fake_instrumentor(monkeypatch)
        provider = TracerProvider()
        decimalai._activate_crewai_instrumentation(provider)
        decimalai._activate_crewai_instrumentation(provider)
        processors = provider._active_span_processor._span_processors
        assert sum(isinstance(p, dc._RunLinker) for p in processors) == 1
        assert dc._tracer_providers == [provider]

    def test_a_failed_activation_wires_nothing(self, monkeypatch):
        """No CrewAI spans will reach that provider, so there is nothing to link
        and nothing for ``flush()`` to push."""
        import decimalai
        import decimalai.crewai as dc

        self._fake_instrumentor(monkeypatch, fail=True)
        provider = MagicMock()
        decimalai._activate_crewai_instrumentation(provider)
        assert dc._tracer_providers == []
        provider.add_span_processor.assert_not_called()


class TestFlush:
    def test_flush_exports_a_queued_run_before_draining_the_sender(self):
        """The batch schedule here is an hour, so only ``flush()`` can ship the
        run — and by the time it returns, the trace exists, was observed, and
        was sent."""
        import decimalai
        import decimalai._config as cfg

        seen: List[Any] = []
        _instrument(on_trace=seen.append)
        _Pipeline(batched=True).run()
        assert cfg._client.ingest_trace.call_count == 0 and seen == []
        decimalai.flush()
        assert cfg._client.ingest_trace.call_count == 1
        assert [t.id for t in seen] == [cfg._client.ingest_trace.call_args.args[0].id]

    def test_flush_is_bounded_by_its_timeout(self, monkeypatch, caplog):
        """``BatchSpanProcessor.force_flush`` ignores its timeout in
        opentelemetry-sdk 1.42 (it exports inline, under a lock), so the bound
        has to be the SDK's own. A wedged export must not hang the caller."""
        import decimalai
        import decimalai.crewai as dc

        release = threading.Event()
        wedged = MagicMock()
        wedged.force_flush.side_effect = lambda *a, **k: release.wait(10)
        dc._tracer_providers.append(wedged)
        monkeypatch.setattr(dc, "_FLUSH_TIMEOUT_S", 0.2)
        try:
            started = time.monotonic()
            with caplog.at_level(logging.WARNING, logger="decimalai.crewai"):
                decimalai.flush()
            assert time.monotonic() - started < 3
            assert "did not finish exporting within" in caplog.text
        finally:
            release.set()
        wedged.force_flush.assert_called_once()

    def test_flush_touches_only_the_pipelines_this_sdk_wired(self):
        """A provider the SDK did not wire for CrewAI — a user's own, another
        rail's — is never flushed by ``decimalai.flush()``."""
        import decimalai
        import decimalai.crewai as dc

        ours, theirs = MagicMock(), MagicMock()
        dc._tracer_providers.append(ours)
        decimalai.flush()
        ours.force_flush.assert_called_once()
        theirs.force_flush.assert_not_called()

    def test_flush_without_crewai_starts_no_thread_and_imports_nothing(self, monkeypatch):
        """Users who never set up CrewAI tracing are unaffected: the adapter is
        not imported, and with it imported but unwired no helper thread starts."""
        import decimalai
        import decimalai.crewai as dc

        started: List[Any] = []
        real_start = threading.Thread.start

        def counting_start(thread: threading.Thread) -> None:
            started.append(thread.name)
            real_start(thread)

        monkeypatch.setattr(threading.Thread, "start", counting_start)
        decimalai.flush()
        assert "decimalai-crewai-flush" not in started

        monkeypatch.delitem(sys.modules, "decimalai.crewai")
        decimalai.flush()
        assert "decimalai.crewai" not in sys.modules
        monkeypatch.setitem(sys.modules, "decimalai.crewai", dc)

    def test_sigterm_exports_the_run_before_draining_the_sender(self, monkeypatch):
        """The SIGTERM handler re-raises with the default disposition, which
        skips the provider's own exit hook — so without this the queued run
        never became a trace on every container stop."""
        import decimalai
        from decimalai import _config

        order: List[str] = []
        installed: Dict[str, Any] = {}
        monkeypatch.setattr(decimalai, "_sigterm_registered", False)
        monkeypatch.setattr(signal, "getsignal", lambda _sig: lambda *_a: order.append("previous"))
        monkeypatch.setattr(signal, "signal", lambda _sig, handler: installed.setdefault("h", handler))
        monkeypatch.setattr(decimalai, "_flush_otel_pipelines", lambda: order.append("otel"))
        monkeypatch.setattr(_config._sender, "flush", lambda *a, **k: order.append("sender"))
        monkeypatch.setattr(decimalai, "_atexit_flush", lambda: order.append("client"))
        decimalai._register_sigterm_flush()
        installed["h"](signal.SIGTERM, None)
        assert order == ["otel", "sender", "client", "previous"]


    def test_a_normal_exit_still_exports_the_queued_run(self, tmp_path):
        """The exit path needed no change, and this is the proof. The provider
        both setup paths build (``otel.instrument``) is shut down — and so
        flushed — from ``otel._register_flush_atexit``, during
        ``threading._shutdown()``, while the sender can still take the trace. A
        process that never calls ``flush()`` still exports the run, links it and
        hands it to the observer on the way out. In a child process, because
        the provider it builds becomes the process's global one."""
        script = textwrap.dedent(f"""
            import os
            os.environ["DECIMALAI_HANDLE_SIGTERM"] = "0"
            from unittest.mock import MagicMock

            import decimalai
            import decimalai._config as cfg
            import decimalai.crewai as dc
            from decimalai._config import DecimalConfig
            from decimalai.otel import instrument as otel_instrument

            cfg._config = DecimalConfig(
                api_key="dai_sk_test", base_url="http://127.0.0.1:9", enabled=True)
            cfg._client = MagicMock()
            cfg._client.register_manifest.return_value = {{"manifest_id": "m", "status": "active"}}
            cfg._client.ingest_trace.side_effect = (
                lambda t: print("SENT", t.id, t.parent_trace_id, flush=True))
            provider = otel_instrument(agent_name="support")
            dc._wire_tracer_provider(provider)
            dc._tracing_installed = True
            dc.instrument(agent_name="support",
                          on_trace=lambda t: print("OBSERVED", t.id, flush=True))
            tracer = provider.get_tracer({CREWAI_SCOPE!r})
            with decimalai.start_trace(agent_name="session", auto_send=False) as outer:
                print("PARENT", outer.get_trace_id(), flush=True)
                with tracer.start_as_current_span("Crew.kickoff"):
                    pass
            print("EXITING", flush=True)
        """)
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            [str(root)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
        ))
        proc = subprocess.run(
            [sys.executable, "-c", script], cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        lines = [line.split() for line in proc.stdout.splitlines()]
        words = [line[0] for line in lines]
        assert words.index("EXITING") < words.index("OBSERVED") < words.index("SENT"), proc.stdout
        parent = next(line[1] for line in lines if line[0] == "PARENT")
        sent = next(line for line in lines if line[0] == "SENT")
        observed = next(line for line in lines if line[0] == "OBSERVED")
        assert sent[1] == observed[1] and sent[2] == parent, proc.stdout


# ── layer 2: the real thing ─────────────────────────────────────────────────


@pytest.mark.skipif(
    not HAS_REAL_CREWAI,
    reason="crewai and openinference-instrumentation-crewai are not SDK "
           "dependencies; the seam is graded by the classes above on every run "
           "and on the wire by tests/conformance (C15 / C16)",
)
class TestRealCrewAIRun:
    """A real ``Crew.kickoff`` under the real OpenInference CrewAI instrumentor.

    The model is a ``BaseLLM`` subclass answering from a script. Everything
    upstream of the wire — CrewAI's executor, the instrumentor's spans, the
    exporter's assembly — is the shipped code.
    """

    @pytest.fixture(autouse=True)
    def _instrumented(self):
        """The CrewAI instrumentor bound to a LOCAL provider, held for an hour
        so only ``decimalai.flush()`` ships anything, and taken off again after
        so no other test's crews are traced."""
        from openinference.instrumentation.crewai import CrewAIInstrumentor

        instrumentor = CrewAIInstrumentor()
        if instrumentor.is_instrumented_by_opentelemetry:  # pragma: no cover
            pytest.skip("CrewAI is already instrumented in this process")
        self.pipe = _Pipeline(batched=True)
        instrumentor.instrument(tracer_provider=self.pipe.provider)
        yield
        instrumentor.uninstrument()

    @staticmethod
    def _crew(fail: bool = False):
        from crewai import Agent, Crew, Task
        from crewai.llms.base_llm import BaseLLM

        class _ScriptedLLM(BaseLLM):
            def call(self, messages, tools=None, callbacks=None, available_functions=None,
                     from_task=None, from_agent=None, response_model=None, **kw):
                if fail:
                    raise RuntimeError("the model failed")
                return "Thought: I know it\nFinal Answer: 23.5%"

            async def acall(self, messages, tools=None, callbacks=None, available_functions=None,
                            from_task=None, from_agent=None, response_model=None, **kw):
                return self.call(messages)

            def supports_function_calling(self) -> bool:
                return False

            def supports_stop_words(self) -> bool:
                return True

            def get_context_window_size(self) -> int:
                return 8192

        agent = Agent(role="Support", goal="Resolve the ticket", backstory="You are terse.",
                      llm=_ScriptedLLM(model="scripted-stub"), verbose=False, max_retry_limit=0)
        task = Task(description="What fee applies to an opened box?",
                    expected_output="the fee", agent=agent)
        return Crew(agents=[agent], tasks=[task], verbose=False)

    def _observed_run(self, run, **instrument_kwargs: Any) -> tuple:
        import decimalai
        from decimalai.generic import start_trace

        seen: List[Any] = []
        _instrument(on_trace=seen.append, **instrument_kwargs)
        with start_trace(agent_name="session", auto_send=False) as outer:
            try:
                run()
            except Exception:
                pass
        assert seen == [], "the batch processor exported before flush() was called"
        decimalai.flush()
        return outer.get_trace_id(), seen, _sent()

    def test_a_crew_inside_start_trace_is_its_child_and_observed_once(self):
        parent, seen, sent = self._observed_run(lambda: self._crew().kickoff())
        [trace] = sent
        assert [t.id for t in seen] == [trace.id]
        assert seen[0].parent_trace_id == trace.parent_trace_id == parent
        assert trace.agent_name == "support"
        assert any("kickoff" in (s.name or "") for s in trace.spans)

    def test_kickoff_async_carries_the_caller_context_to_its_thread(self):
        """``Crew.kickoff_async`` runs ``kickoff`` through ``asyncio.to_thread``,
        which copies the caller's context — so the run is still linked."""
        parent, seen, sent = self._observed_run(
            lambda: asyncio.run(self._crew().kickoff_async())
        )
        [trace] = sent
        assert trace.parent_trace_id == parent
        assert [t.id for t in seen] == [trace.id]

    def test_a_failed_crew_is_observed_once_and_marked_errored(self):
        from decimalai.schema.common import Status

        parent, seen, sent = self._observed_run(lambda: self._crew(fail=True).kickoff())
        [trace] = sent
        assert [t.id for t in seen] == [trace.id]
        assert trace.status == Status.ERROR
        assert trace.parent_trace_id == parent

    def test_the_observed_trace_carries_the_runs_delivery_rails(self, monkeypatch):
        """With the loader on, the observer is handed the trace the platform will
        credit: the routing id and the delivered names the hook witnessed, on
        the same trace the run is filed under — which is what a caller reads to
        find the usage receipt for the run."""
        from crewai.hooks import unregister_before_llm_call_hook

        import decimalai.crewai as dc
        import decimalai.skill_router as sr

        routed: List[Optional[str]] = []

        class _Router:
            def build_prompt_parts(self, query=None, *, agent_name=None, scope=None, **kw):
                routed.append(scope)
                sr._last_offered_names_ctx.set(["refund-policy"])
                sr._last_delivered_names_ctx.set(["refund-policy"])
                return ("## Skill: refund-policy\n\nOpened boxes carry a 23.5% fee.\n",
                        "Most relevant: refund-policy.", "rt_" + "d" * 24)

        monkeypatch.setattr(dc, "_skill_router_singleton", _Router())
        try:
            parent, seen, sent = self._observed_run(
                lambda: self._crew().kickoff(), enable_skill_loader=True,
            )
        finally:
            unregister_before_llm_call_hook(dc._skill_hook)
        [trace] = sent
        assert routed and routed[0] is not None, "the hook routed outside the crew's span"
        assert [t.id for t in seen] == [trace.id]
        assert seen[0].routing_id == trace.routing_id == "rt_" + "d" * 24
        assert seen[0].skills_delivered == trace.skills_delivered == ["refund-policy"]
        assert seen[0].parent_trace_id == parent
