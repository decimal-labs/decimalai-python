"""`DecimalaiPlugin`'s traces: the observer, the parent link, and a refused manifest.

Every invocation the plugin sees becomes exactly one trace, its observer gets a
detached copy of that trace, and the trace links to the DecimalAI trace that was
open when the invocation STARTED: finalization runs on an executor thread, where
the caller's context no longer exists. The real-run tests drive google-adk 2.12's
own runner (sync and async) with a provider-free model, and post the exported
traces through a real ``DecimalAIClient`` over a mock transport, so what is
checked is the payload the platform would receive.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import MagicMock

import pytest

pytest.importorskip("google.adk")

CALLER_PROMPT = "You are a terse support agent."
USER_QUESTION = "What fee applies to an opened box return?"


@pytest.fixture(autouse=True)
def client(monkeypatch):
    import decimalai._config as cfg
    import decimalai.adk as adk
    from decimalai._config import DecimalConfig

    mock = MagicMock()
    mock.register_manifest.return_value = {"manifest_id": "test-manifest-id", "status": "active"}
    monkeypatch.setattr(cfg, "_config", DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True,
    ))
    monkeypatch.setattr(cfg, "_client", mock)
    for name, value in (("_PluginClass", None), ("_manifest_ids", {}), ("_manifest_trackers", {}),
                        ("_pending_manifests", {}), ("_live_runs", {})):
        monkeypatch.setattr(adk, name, value)
    return mock


def _flushed_traces() -> List[Any]:
    import decimalai._config as cfg

    cfg._sender.flush()
    return [call.args[0] for call in cfg._client.ingest_trace.call_args_list]


def _agent(name: str = "root_agent") -> SimpleNamespace:
    return SimpleNamespace(name=name, model="gemini-3.6-flash", instruction=CALLER_PROMPT,
                           tools=[], sub_agents=[])


def _drive_one_turn(plugin: Any, *, inv_id: str = "inv-1") -> None:
    """before_run -> before_agent -> after_agent, on the plugin's own callbacks."""
    agent = _agent()
    ic = SimpleNamespace(agent=agent, invocation_id=inv_id, user_content=USER_QUESTION)
    cc = SimpleNamespace(invocation_id=inv_id, agent_name=None)

    async def go():
        await plugin.before_run_callback(invocation_context=ic)
        await plugin.before_agent_callback(agent=agent, callback_context=cc)
        await plugin.after_agent_callback(agent=agent, callback_context=cc)

    asyncio.run(go())


class TestInvocationObserver:
    def test_the_observer_gets_one_detached_copy_and_cannot_change_the_export(self):
        from decimalai.adk import DecimalaiPlugin

        seen = []

        def observe(trace):
            seen.append(trace.model_copy(deep=True))
            trace.agent_name = "observer-mutation"

        plugin = DecimalaiPlugin(agent_name="support", on_trace=observe)
        _drive_one_turn(plugin)
        # A late completion signal for the same invocation is not a second trace.
        asyncio.run(plugin.after_run_callback(invocation_context=SimpleNamespace(invocation_id="inv-1")))
        asyncio.run(plugin.on_run_error_callback(
            invocation_context=SimpleNamespace(invocation_id="inv-1"), error=RuntimeError("late"),
        ))
        sent = _flushed_traces()
        assert len(seen) == len(sent) == 1
        assert seen[0].id == sent[0].id
        assert sent[0].agent_name == "support"

    def test_an_error_invocation_is_observed_once_and_a_failing_observer_keeps_it(self):
        from decimalai.adk import DecimalaiPlugin
        from decimalai.schema.common import Status

        seen = []

        def broken_observer(trace):
            seen.append(trace)
            raise RuntimeError("observer failed")

        plugin = DecimalaiPlugin(on_trace=broken_observer)
        ic = SimpleNamespace(agent=_agent(), invocation_id="error", user_content=USER_QUESTION)
        cc = SimpleNamespace(invocation_id="error")
        request = SimpleNamespace(model="gemini-3.6-flash", contents=[],
                                  config=SimpleNamespace(system_instruction=CALLER_PROMPT))

        async def drive():
            await plugin.before_run_callback(invocation_context=ic)
            await plugin.before_model_callback(callback_context=cc, llm_request=request)
            await plugin.on_model_error_callback(callback_context=cc, llm_request=request, error=RuntimeError("429"))
            await plugin.on_run_error_callback(invocation_context=ic, error=RuntimeError("429"))
            await plugin.on_run_error_callback(invocation_context=ic, error=RuntimeError("late"))

        asyncio.run(drive())
        sent = _flushed_traces()
        assert len(seen) == len(sent) == 1
        assert seen[0].status == sent[0].status == Status.ERROR

    def test_the_parent_is_captured_at_invocation_start_before_executor_finalization(self):
        from decimalai.adk import DecimalaiPlugin
        from decimalai.generic import start_trace

        seen = []
        plugin = DecimalaiPlugin(on_trace=seen.append)
        ic = SimpleNamespace(agent=_agent(), invocation_id="parent", user_content=USER_QUESTION)
        with start_trace(agent_name="outer", auto_send=False) as outer:
            parent_id = outer.get_trace_id()
            asyncio.run(plugin.before_run_callback(invocation_context=ic))
        # The generic context has closed; finalization's executor inherits no
        # ContextVar. Reading the parent there would silently unlink.
        asyncio.run(plugin.after_run_callback(invocation_context=ic))
        assert seen[0].parent_trace_id == _flushed_traces()[0].parent_trace_id == parent_id

    def test_an_explicit_parent_wins_and_a_standalone_invocation_stays_a_root(self):
        from uuid import uuid4

        from decimalai.adk import DecimalaiPlugin
        from decimalai.generic import start_trace

        parent_id = str(uuid4())
        seen = []
        with start_trace(agent_name="outer", auto_send=False):
            _drive_one_turn(DecimalaiPlugin(parent_trace_id=parent_id, on_trace=seen.append))
        _drive_one_turn(DecimalaiPlugin(on_trace=seen.append), inv_id="standalone")
        assert [trace.parent_trace_id for trace in seen] == [parent_id, None]

    def test_concurrent_generic_contexts_keep_their_own_parent(self):
        from decimalai.adk import DecimalaiPlugin
        from decimalai.generic import start_trace

        seen, parents = [], {}
        plugin = DecimalaiPlugin(on_trace=seen.append)

        async def run(invocation_id):
            with start_trace(agent_name=invocation_id, auto_send=False) as outer:
                parents[invocation_id] = outer.get_trace_id()
                ic = SimpleNamespace(agent=_agent(invocation_id), invocation_id=invocation_id)
                await plugin.before_run_callback(invocation_context=ic)
                await asyncio.sleep(0)  # overlap starts before either finalizes
                await plugin.after_run_callback(invocation_context=ic)

        async def drive():
            await asyncio.gather(run("first"), run("second"))

        asyncio.run(drive())
        assert len(seen) == 2
        assert {trace.agent_name: trace.parent_trace_id for trace in seen} == parents
        assert len({trace.id for trace in seen}) == 2


def _posted(sent: List[Any]) -> List[dict]:
    """The traces as a real DecimalAIClient would POST them."""
    import httpx

    from decimalai._client import DecimalAIClient

    payloads = []

    def capture(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "accepted"}, request=request)

    client = DecimalAIClient(api_key="dai_sk_test", base_url="http://sdk-proof.test")
    client._http.close()
    client._http = httpx.Client(base_url="http://sdk-proof.test", transport=httpx.MockTransport(capture))
    try:
        for trace in sent:
            client.ingest_trace(trace)
    finally:
        client._http.close()
    return payloads


class TestRealAdkParentLink:
    """The parent is right on google-adk 2.12's own runner, sync and async."""

    @staticmethod
    def _runner(observed, *, explicit_parent=None, barrier=None):
        from google.adk.agents import LlmAgent
        from google.adk.apps import App
        from google.adk.models.base_llm import BaseLlm
        from google.adk.models.llm_response import LlmResponse
        from google.adk.plugins.base_plugin import BasePlugin
        from google.adk.runners import InMemoryRunner
        from google.genai import types

        import decimalai.adk as adk

        configs = []

        class CaptureConfig(BasePlugin):
            def __init__(self):
                super().__init__(name="capture_config")

            async def before_run_callback(self, *, invocation_context):
                configs.append(invocation_context.run_config)

        class StubLlm(BaseLlm):
            async def generate_content_async(self, llm_request, stream=False):
                if barrier is not None:
                    # Each caller has its own event loop. Force both model calls to
                    # overlap before either invocation can finalize.
                    barrier.wait(timeout=10)
                yield LlmResponse(content=types.Content(
                    role="model", parts=[types.Part(text="Provider-free answer")],
                ))

        plugins = [
            adk.DecimalaiPlugin(agent_name="parent_probe", parent_trace_id=explicit_parent,
                                on_trace=observed.append),
            CaptureConfig(),
        ]
        agent = LlmAgent(name="parent_probe", model=StubLlm(model="capturing-stub"), instruction=CALLER_PROMPT)
        runner = InMemoryRunner(app=App(name="parent_probe", root_agent=agent, plugins=plugins))
        return runner, configs

    @staticmethod
    def _run(runner, session_id, *, synchronous, run_config=None):
        from google.genai import types

        arguments = {
            "user_id": "u", "session_id": session_id,
            "new_message": types.Content(role="user", parts=[types.Part(text=session_id)]),
            "run_config": run_config,
        }
        if synchronous:
            return list(runner.run(**arguments))

        async def run():
            return [event async for event in runner.run_async(**arguments)]

        return asyncio.run(run())

    @staticmethod
    def _session(runner, session_id):
        runner.session_service.create_session_sync(app_name="parent_probe", user_id="u", session_id=session_id)

    @staticmethod
    def _assert_parents(observed, expected):
        sent = _flushed_traces()
        assert len(observed) == len(sent) == len(expected)
        assert {trace.user_input_preview: trace.parent_trace_id for trace in observed} == expected
        assert {trace.user_input_preview: trace.parent_trace_id for trace in sent} == expected
        assert {trace.id for trace in observed} == {trace.id for trace in sent}
        payloads = _posted(sent)
        assert {trace["user_input_preview"]: trace.get("parent_trace_id") for trace in payloads} == expected
        assert "_decimalai_invocation_parent_trace_id" not in json.dumps(payloads)

    @pytest.mark.parametrize("synchronous", [False, True], ids=["async", "sync"])
    def test_a_reused_runner_and_config_capture_each_parent_and_a_standalone_root(self, synchronous):
        from google.adk.agents.run_config import RunConfig

        from decimalai.generic import start_trace

        observed = []
        runner, configs = self._runner(observed)
        config = RunConfig(max_llm_calls=5, custom_metadata={"caller": "preserved"})
        original = config.model_dump(mode="json")
        expected = {}
        for session_id in ("first_parent", "second_parent", "standalone"):
            self._session(runner, session_id)
            if session_id == "standalone":
                expected[session_id] = None
                self._run(runner, session_id, synchronous=synchronous, run_config=config)
            else:
                with start_trace(agent_name=session_id, auto_send=False) as outer:
                    expected[session_id] = outer.get_trace_id()
                    self._run(runner, session_id, synchronous=synchronous, run_config=config)
        self._assert_parents(observed, expected)
        assert config.model_dump(mode="json") == original
        assert len(configs) == 3
        assert not any(hasattr(value, "_decimalai_invocation_parent_trace_id") for value in configs)

    @pytest.mark.parametrize("synchronous", [False, True], ids=["async", "sync"])
    def test_concurrent_invocations_on_one_runner_keep_separate_parents(self, synchronous):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier

        from decimalai.generic import start_trace

        observed = []
        runner, _ = self._runner(observed, barrier=Barrier(2))
        for session_id in ("concurrent_first", "concurrent_second"):
            self._session(runner, session_id)

        def invoke(session_id):
            with start_trace(agent_name=session_id, auto_send=False) as parent:
                parent_id = parent.get_trace_id()
                self._run(runner, session_id, synchronous=synchronous)
                return session_id, parent_id

        with ThreadPoolExecutor(max_workers=2) as callers:
            expected = dict(callers.map(invoke, ("concurrent_first", "concurrent_second")))
        self._assert_parents(observed, expected)
        assert len(set(expected.values())) == 2

    @pytest.mark.parametrize("synchronous", [False, True], ids=["async", "sync"])
    def test_the_explicit_parent_wins_over_the_caller_and_standalone(self, synchronous):
        from uuid import uuid4

        from decimalai.generic import start_trace

        observed = []
        explicit = str(uuid4())
        runner, _ = self._runner(observed, explicit_parent=explicit)
        for session_id in ("explicit_nested", "explicit_standalone"):
            self._session(runner, session_id)
        with start_trace(agent_name="ignored_parent", auto_send=False) as outer:
            assert outer.get_trace_id() != explicit
            self._run(runner, "explicit_nested", synchronous=synchronous)
        self._run(runner, "explicit_standalone", synchronous=synchronous)
        self._assert_parents(observed, {"explicit_nested": explicit, "explicit_standalone": explicit})

    def test_a_deferred_sync_iterator_captures_the_parent_when_iteration_starts(self):
        from google.genai import types

        from decimalai.generic import start_trace

        observed = []
        runner, _ = self._runner(observed)
        self._session(runner, "deferred")
        events = runner.run(user_id="u", session_id="deferred", new_message=types.Content(
            role="user", parts=[types.Part(text="deferred")],
        ))
        with start_trace(agent_name="iteration_parent", auto_send=False) as outer:
            expected = outer.get_trace_id()
            assert list(events)
        self._assert_parents(observed, {"deferred": expected})

    @pytest.mark.parametrize("failure", ["extraction", "registration-refusal"])
    def test_a_first_manifest_failure_keeps_both_children_in_a_reused_sync_session(
        self, monkeypatch, caplog, failure,
    ):
        import decimalai._config as cfg
        import decimalai.adk as adk
        from decimalai.generic import start_trace

        pending_submit = MagicMock(wraps=cfg.submit_trace_pending_manifest)
        monkeypatch.setattr(cfg, "submit_trace_pending_manifest", pending_submit)
        if failure == "extraction":
            # A failed CURRENT shape must not inherit or replay an OLD one.
            adk._manifest_ids["parent_probe"] = "previous-manifest-id"
            adk._pending_manifests["parent_probe"] = (MagicMock(), ConnectionError("previous refusal"))
            original_extract = adk.extract_from_config
            calls = []

            def extract(**kwargs):
                calls.append(None)
                if len(calls) == 1:
                    raise ValueError("private metadata must not appear in logs")
                return original_extract(**kwargs)

            monkeypatch.setattr(adk, "extract_from_config", extract)
        else:
            cfg._client.register_manifest.side_effect = [
                ConnectionError("offline refusal"),
                {"manifest_id": "registered-manifest-id", "status": "active"},
                {"manifest_id": "registered-manifest-id", "status": "active"},
            ]
        observed = []
        runner, _ = self._runner(observed)
        self._session(runner, "shared")
        with start_trace(agent_name="generic_parent", auto_send=False) as parent:
            parent_id = parent.get_trace_id()
            for _ in range(2):
                self._run(runner, "shared", synchronous=True)
                cfg._sender.flush()
        sent = _flushed_traces()
        assert len(observed) == len(sent) == 2 and len({trace.id for trace in sent}) == 2
        assert all(trace.parent_trace_id == parent_id for trace in observed + sent)
        if failure == "extraction":
            assert observed[0].manifest_id is sent[0].manifest_id is None
            assert sent[1].manifest_id == "test-manifest-id"
            pending_submit.assert_not_called()
            records = [r for r in caplog.records if "manifest preparation failed" in r.message]
            assert len(records) == 1 and "ValueError" in records[0].message
            assert records[0].exc_info is None
            assert "private metadata" not in caplog.text
        else:
            pending_submit.assert_called_once()
            assert [trace.manifest_id for trace in sent] == ["registered-manifest-id"] * 2
        payloads = _posted(sent)
        assert all(payload["parent_trace_id"] == parent_id for payload in payloads)
