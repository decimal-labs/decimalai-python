"""Exact served UUIDs survive adapter ownership, copied callbacks, and export.

All body responses are synthetic. No provider or backend requests are made.
The concurrent lanes deliberately fetch different versions of the SAME name.
"""

from __future__ import annotations

import contextvars
import importlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pytest

from decimalai import skill_router as sr
from decimalai._skill_witness import copy_delivered_versions
from decimalai.skill_router import SkillRouter

NAME = "same-skill"
SKILL_ID = "f36284bd-aee5-458c-a593-451756c73151"
VERSION_IDS = ["58b198ca-f336-4092-b875-dc20d00d8286", "37a95d98-c167-496e-b206-2b46c4b45007"]
HASHES = ["a" * 64, "b" * 64]
BODIES = ["SYNTHETIC FIRST COMPLETE BODY", "SYNTHETIC SECOND COMPLETE BODY"]
ROUTING_ID = "rt_" + "7" * 24
_lane = contextvars.ContextVar("delivery_test_lane", default=0)


@pytest.fixture(autouse=True)
def sdk(monkeypatch):
    import decimalai._config as cfg
    import decimalai.langchain as lc
    import decimalai.openai_agents as oa
    import decimalai.otel as otel

    cfg._sender.flush()
    monkeypatch.setattr(cfg, "_config", cfg.DecimalConfig(api_key="dai_sk_test", enabled=True))
    client = MagicMock()
    client.register_manifest.return_value = {"manifest_id": str(uuid4()), "status": "active"}
    client.list_manifests.return_value = {"manifests": []}
    monkeypatch.setattr(cfg, "_client", client)
    for module in (lc, oa):
        for field in ("_manifest_ids", "_manifest_hashes", "_pending_manifests", "_pending_snapshots"):
            if hasattr(module, field):
                monkeypatch.setattr(module, field, {})
    for field in ("_skills_delivered_ctx", "_skills_offered_ctx", "_skills_delivered_versions_ctx", "_routing_id_ctx"):
        getattr(lc, field).set(None)
    monkeypatch.setattr(lc, "_manifest_adoption_probed", set())
    monkeypatch.setattr(lc, "_explicit_manifest_config", None)
    oa._run_rails.clear()
    otel._reset_skill_rails()
    otel._reset_run_links()
    sr._body_budget_ctx.set(None)
    sr._last_delivered_versions_ctx.set(None)
    yield cfg
    cfg._sender.flush()
    oa._run_rails.clear()
    otel._reset_skill_rails()
    otel._reset_run_links()
    sr._body_budget_ctx.set(None)
    sr._last_delivered_versions_ctx.set(None)
    for field in ("_skills_delivered_ctx", "_skills_offered_ctx", "_skills_delivered_versions_ctx", "_routing_id_ctx"):
        getattr(lc, field).set(None)


def router(monkeypatch, *, concurrent=True):
    instance = SkillRouter(api_key="dai_sk_test", inject_body=True)
    route = {
        "prompt_fragment": "Available skills: " + NAME, "routing_id": ROUTING_ID,
        "skills": [{"name": NAME}], "stable_menu": "Available skills: " + NAME,
        "stable_menu_skills": [NAME], "routing_hint": "Use " + NAME,
    }
    monkeypatch.setattr(instance, "smart_route", lambda *args, **kwargs: dict(route))

    def record(*args, **kwargs):
        index = _lane.get()
        return {"body": BODIES[index], "content_hash": HASHES[index],
                "skill_id": SKILL_ID, "version_id": VERSION_IDS[index]}

    monkeypatch.setattr(instance, "get_skill_body_record", record)
    if concurrent:
        # Every shared last-seen write has happened before either adapter
        # captures its witness. Looking up the current hash would fail here.
        original = instance.get_skill_body
        barrier = threading.Barrier(2)

        def body(*args, **kwargs):
            result = original(*args, **kwargs)
            barrier.wait(timeout=10)
            return result

        monkeypatch.setattr(instance, "get_skill_body", body)
    return instance


def sent(cfg):
    cfg._sender.flush()
    return [call.args[0] for call in cfg._client.ingest_trace.call_args_list]


def serialized(traces):
    from decimalai._client import DecimalAIClient

    payloads = []

    def capture(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "accepted"}, request=request)

    client = DecimalAIClient(api_key="dai_sk_test", base_url="http://offline.invalid")
    client._http.close()
    client._http = httpx.Client(base_url="http://offline.invalid", transport=httpx.MockTransport(capture))
    try:
        for trace in traces:
            client.ingest_trace(trace)
    finally:
        client._http.close()
    return payloads


def assert_versions(traces, parents, *, observed=None, loaded=False):
    payloads = serialized(traces)
    assert len(traces) == len(payloads) == 2
    by_parent = {trace.parent_trace_id: trace for trace in traces}
    for index, parent in enumerate(parents):
        trace = by_parent[parent]
        expected = {"name": NAME, "hash": HASHES[index], "skill_id": SKILL_ID, "version_id": VERSION_IDS[index]}
        witness = trace.skills_delivered_versions
        assert len(witness) == 1
        assert {key: witness[0][key] for key in expected} == expected
        assert trace.skills_delivered == [NAME]
        assert trace.skills_loaded_by_agent == ([NAME] if loaded else [])
        wire = next(payload for payload in payloads if payload["id"] == str(trace.id))
        assert wire["skills_delivered_versions"] == witness
        if observed is not None:
            callback = next(value for value in observed if value.id == trace.id)
            assert callback.skills_delivered_versions == witness


@pytest.mark.parametrize("adapter", ["anthropic", "crewai"])
def test_otel_adapter_prompt_witnesses_reach_observer_and_export_concurrently(monkeypatch, sdk, adapter):
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    from decimalai.generic import start_trace
    from decimalai.otel import DecimalSpanExporter, agent_run, record_run_link

    module = importlib.import_module("decimalai." + adapter)
    monkeypatch.setattr(module, "_skill_router_singleton", router(monkeypatch))
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(DecimalSpanExporter()))
    tracer = provider.get_tracer("delivery-proof")
    observed = []

    def observe(trace):
        observed.append(trace.model_copy(deep=True))
        trace.skills_delivered_versions[0]["version_id"] = str(uuid4())

    def invoke(index):
        _lane.set(index)
        with start_trace(agent_name=f"parent-{index}", auto_send=False) as parent:
            with agent_run(f"child-{index}", tracer_provider=provider) as root:
                record_run_link(root.get_span_context().trace_id, parent_trace_id=parent.get_trace_id(), on_trace=observe)
                if adapter == "anthropic":
                    prompt = module.skill_system("caller", query=f"request-{index}")
                else:
                    messages = [{"role": "user", "content": f"request-{index}"}]
                    module._skill_hook(SimpleNamespace(
                        executor=object(), messages=messages, task=None, agent=None, crew=None, llm=None,
                    ))
                    prompt = "\n".join(message["content"] for message in messages)
                assert BODIES[index] in prompt and BODIES[1 - index] not in prompt
                with tracer.start_as_current_span("model", attributes={"gen_ai.request.model": "offline-stub"}):
                    pass
            return parent.get_trace_id()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            parents = list(pool.map(invoke, range(2)))
        assert_versions(sent(sdk), parents, observed=observed)
    finally:
        provider.shutdown()


def test_load_agent_load_skill_carries_exact_scoped_witness(monkeypatch, sdk):
    """The load every OpenTelemetry-rail agent_config() shares (decimalai's and
    Pydantic AI's): two runs at once each record their own body's version."""
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    from decimalai._agent import _serve_and_record
    from decimalai.generic import start_trace
    from decimalai.otel import DecimalSpanExporter, agent_run, record_run_link

    shared = router(monkeypatch)
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(DecimalSpanExporter()))

    def invoke(index):
        _lane.set(index)
        with start_trace(agent_name=f"parent-{index}", auto_send=False) as parent:
            with agent_run(f"child-{index}", tracer_provider=provider) as root:
                record_run_link(root.get_span_context().trace_id, parent_trace_id=parent.get_trace_id())
                body = contextvars.copy_context().run(
                    _serve_and_record, shared, NAME, scope=f"run-{index}", agent_name=f"child-{index}")
                assert BODIES[index] in body and BODIES[1 - index] not in body
            return parent.get_trace_id()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            parents = list(pool.map(invoke, range(2)))
        assert_versions(sent(sdk), parents, loaded=True)
    finally:
        provider.shutdown()


@pytest.mark.parametrize("loaded", [False, True], ids=["prompt", "tool"])
def test_real_langchain_calls_keep_exact_witnesses_for_same_name_concurrently(monkeypatch, sdk, loaded):
    pytest.importorskip("langchain_core")
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.runnables import RunnableLambda

    import decimalai.langchain as lc
    from decimalai.generic import start_trace

    shared = router(monkeypatch)
    monkeypatch.setattr(lc, "_skill_router_singleton", shared)
    monkeypatch.setattr(BaseChatModel, "invoke", BaseChatModel.invoke)
    monkeypatch.setattr(BaseChatModel, "ainvoke", BaseChatModel.ainvoke)
    monkeypatch.setattr(lc, "_skill_loader_installed", False)
    if not loaded:
        lc._install_skill_loader()
    observed = []

    def invoke(index):
        _lane.set(index)
        handler = lc.CallbackHandler(agent_name=f"child-{index}")
        handler.on_trace = observed.append
        model = FakeListChatModel(responses=["synthetic answer"])

        def load(value):
            body = shared.load_skill(NAME)
            assert BODIES[index] in body and BODIES[1 - index] not in body
            return value

        chain = RunnableLambda(load) | model if loaded else model
        with start_trace(agent_name=f"parent-{index}", auto_send=False) as parent:
            chain.invoke(f"request-{index}", config={"callbacks": [handler]})
            return parent.get_trace_id()

    with ThreadPoolExecutor(max_workers=2) as pool:
        parents = list(pool.map(invoke, range(2)))
    assert_versions(sent(sdk), parents, observed=observed, loaded=loaded)


# Since 0.18 a body reaches an OpenAI agent only through load_skill: agent_config()
# never puts one in the instructions, so there is no "prompt" case to prove here.
@pytest.mark.parametrize("loaded", [True], ids=["tool"])
def test_openai_copied_callback_uses_its_run_witness_not_shared_current_hash(monkeypatch, sdk, loaded):
    import decimalai.openai_agents as oa
    from decimalai.generic import start_trace

    monkeypatch.setattr(oa, "_skill_router_singleton", router(monkeypatch))
    current = contextvars.ContextVar("openai-proof-run", default=None)
    monkeypatch.setattr(oa, "_current_run_key", current.get)
    processor = oa.DecimalTracingProcessor()

    def invoke(index):
        _lane.set(index)
        key = "trace_" + uuid4().hex
        current.set(key)
        trace = SimpleNamespace(trace_id=key, name=f"child-{index}")
        with start_trace(agent_name=f"parent-{index}", auto_send=False) as parent:
            oa.set_parent_trace(parent.get_trace_id())
            processor.on_trace_start(trace)

            def deliver():
                return oa._handle_load_skill(NAME)

            body = contextvars.copy_context().run(deliver)
            assert BODIES[index] in body and BODIES[1 - index] not in body
            processor.on_trace_end(trace)
            oa.clear_parent_trace()
            return parent.get_trace_id()

    with ThreadPoolExecutor(max_workers=2) as pool:
        parents = list(pool.map(invoke, range(2)))
    assert_versions(sent(sdk), parents, loaded=loaded)


def test_openai_unowned_shared_version_is_never_attached_to_a_later_run(monkeypatch, sdk):
    import decimalai.openai_agents as oa

    shared = router(monkeypatch, concurrent=False)
    monkeypatch.setattr(oa, "_skill_router_singleton", shared)
    shared.load_skill(NAME)  # No invocation owns this body.
    processor = oa.DecimalTracingProcessor()
    trace = SimpleNamespace(trace_id="trace_" + uuid4().hex, name="later-run")
    processor.on_trace_start(trace)
    processor.on_trace_end(trace)
    (exported,) = sent(sdk)
    assert exported.skills_delivered_versions == []


@pytest.mark.parametrize("pair", [
    {"skill_id": SKILL_ID}, {"version_id": VERSION_IDS[0]},
    {"skill_id": SKILL_ID, "version_id": "invalid"},
    {"skill_id": SKILL_ID.upper(), "version_id": VERSION_IDS[0]},
])
def test_partial_or_noncanonical_pair_preserves_legacy_hash_only(pair):
    witness = {"name": NAME, "hash": HASHES[0], **pair}
    assert copy_delivered_versions([witness]) == [{"name": NAME, "hash": HASHES[0]}]


# Not openai_agents or adk: since 0.18 and 0.20 they never build a body into a prompt
# (load_skill only).
@pytest.mark.parametrize("adapter", ["anthropic", "crewai", "langchain"])
@pytest.mark.parametrize("failure", ["writes-then-fails", "stale-before-fails"])
def test_failed_prompt_build_cannot_certify_next_legacy_body(monkeypatch, adapter, failure):
    import decimalai.otel as otel

    if adapter == "langchain":
        pytest.importorskip("langchain_core.messages")
    current = [41]
    monkeypatch.setattr(otel, "current_run_key", lambda: current[0])
    stale = {"name": NAME, "hash": HASHES[0], "skill_id": SKILL_ID, "version_id": VERSION_IDS[0]}

    class FailedThenLegacyRouter:
        build_prompt_parts = None

        def __init__(self):
            self.calls = 0

        def build_prompt_fragment(self, **kwargs):
            self.calls += 1
            sr._last_offered_names_ctx.set([NAME])
            sr._last_delivered_names_ctx.set([NAME])
            if self.calls == 1:
                if failure == "writes-then-fails":
                    sr._last_delivered_versions_ctx.set([dict(stale)])
                raise ValueError("synthetic optional router failure")
            # This body is different, and a legacy router cannot certify its
            # version. Reusing the failed body's UUID would be false evidence.
            return NAME + ": " + BODIES[1], None

    module = importlib.import_module("decimalai." + adapter)
    monkeypatch.setattr(module, "_skill_router_singleton", FailedThenLegacyRouter())
    if failure == "stale-before-fails":
        sr._last_delivered_versions_ctx.set([dict(stale)])

    def build():
        if adapter == "anthropic":
            return module.skill_system("caller", query="synthetic request")
        if adapter == "crewai":
            messages = [{"role": "user", "content": "synthetic request"}]
            module._skill_hook(SimpleNamespace(
                executor=object(), messages=messages, task=None, agent=None, crew=None, llm=None,
            ))
            return "\n".join(message["content"] for message in messages)
        tokens = module._open_call_rails()
        try:
            messages = module._inject_skills_into_input("synthetic request")
            if current[0] == 42:
                assert module._skills_delivered_ctx.get() == {NAME}
                assert module._skills_delivered_versions_ctx.get() == []
            return "\n".join(getattr(message, "content", "") for message in messages) if isinstance(messages, list) else messages
        finally:
            module._close_call_rails(tokens)

    build()
    assert sr._last_delivered_versions_ctx.get() is None
    current[0] = 42
    prompt = build()
    assert BODIES[1] in prompt
    if adapter in {"anthropic", "crewai"}:
        rail = otel._pop_skill_rail(42)
        assert rail["delivered"] == [NAME]
        assert rail["delivered_versions"] == []
