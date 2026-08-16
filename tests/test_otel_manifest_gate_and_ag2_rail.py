"""Regression tests for the OTel / community rail.

Three defects, all reproduced against a live backend before being fixed here:

1. **The manifest gate dropped whole traces.** ``_maybe_register_manifest``
   returned early unless the trace happened to expose a model, tool or prompt,
   so a run that exposed none shipped ``manifest_id=None`` and the backend
   rejected it with a 400 under ``require_manifest_on_ingest``. Every Microsoft
   AutoGen trace (its runtime spans carry no GenAI attributes at all) and every
   AG2/CrewAI turn that neither called a tool nor made a model call was lost.

2. **The manifest shrank between turns.** ``seen_tools`` was rebuilt per trace,
   and AG2 declares a tool only by *executing* it — so two identical turns of an
   unchanged agent registered two versions and the platform reported
   ``tool_registry breaking/major "search removed"`` with a ``replay``
   decision. Fabricated breaking changes are worse than no versioning.

3. **AG2 content never arrived.** AG2's GenAI-semconv message shape (one JSON
   attribute, not indexed keys) was not read — so previews were raw JSON blobs
   and ``rendered_input``/``output`` were ``None``.

The AG2-shaped spans below are FIXTURES for the generic exporter, not an
integration: AutoGen/AG2 was retired as an adapter on 2026-08-16 and the SDK no
longer instruments agents for anyone. They stay because the wire shape they
capture is real, and it is the shape an AG2 user's own ``instrument_agent()``
calls still put on the generic OTel rail. The tests that graded the retired
auto-instrumentation (the ``ConversableAgent.__init__`` hook and the
already-built-agent sweep) were deleted with it.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


# ── Mock OTEL span plumbing (mirrors tests/test_otel_exporter.py) ──


class _Ctx:
    def __init__(self, trace_id: int, span_id: int):
        self.trace_id = trace_id
        self.span_id = span_id
        self.trace_flags = 1


class _Status:
    def __init__(self, code: str = "OK"):
        self.status_code = code


class _Resource:
    def __init__(self, service_name: str = "test-service"):
        self.attributes = {"service.name": service_name}


class _MockSpan:
    def __init__(self, name, trace_id, span_id, parent_span_id=None, attributes=None):
        self.name = name
        self.context = _Ctx(trace_id, span_id)
        self.parent = _Ctx(trace_id, parent_span_id) if parent_span_id else None
        self.attributes = attributes or {}
        self.start_time = int(datetime.now(timezone.utc).timestamp() * 1e9)
        self.end_time = self.start_time + 100_000_000
        self.status = _Status()
        self.resource = _Resource()


_mock_export_result = MagicMock()
_mock_export_result.SUCCESS = "SUCCESS"


def _otel_mock():
    return patch.dict("sys.modules", {
        "opentelemetry": MagicMock(),
        "opentelemetry.sdk": MagicMock(),
        "opentelemetry.sdk.trace": MagicMock(),
        "opentelemetry.sdk.trace.export": MagicMock(
            SpanExportResult=_mock_export_result
        ),
    })


@pytest.fixture(autouse=True)
def _sdk(monkeypatch):
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    cfg._config = DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True
    )
    cfg._client = MagicMock()
    # The backend scopes a manifest by (org, agent_name, hash); mirror that so
    # two agents with identical components still get distinct ids.
    cfg._client.register_manifest.side_effect = (
        lambda snapshot: {
            "manifest_id": f"mf-{snapshot.agent_name}-{snapshot.manifest_hash[:8]}"
        }
    )
    # No pre-existing manifest for the agent unless a test says otherwise.
    cfg._client.list_manifests.return_value = {"manifests": []}
    yield cfg


def _export(exporter, spans):
    from decimalai._config import _sender

    with _otel_mock():
        exporter.export(spans)
    _sender.flush()


def _sent_traces(cfg):
    return [call[0][0] for call in cfg._client.ingest_trace.call_args_list]


# ── 1. The manifest gate ─────────────────────────────────────


def test_trace_with_nothing_to_declare_still_carries_a_manifest_id(_sdk):
    """A Microsoft AutoGen run: messaging spans, no model/tool/prompt anywhere.

    Before the fix this shipped manifest_id=None and the backend 400'd it —
    100% trace loss for that framework.
    """
    from decimalai.otel import DecimalSpanExporter

    exporter = DecimalSpanExporter(agent_name="ms-autogen")
    _export(exporter, [
        _MockSpan("autogen create GroupChatStart", 0x1, 0x1,
                  attributes={"messaging.operation": "create"}),
        _MockSpan("autogen process RequestReply", 0x1, 0x2, parent_span_id=0x1,
                  attributes={"messaging.operation": "process"}),
    ])

    traces = _sent_traces(_sdk)
    assert len(traces) == 1
    assert traces[0].manifest_id, "trace shipped with no manifest_id → 400 on ingest"


def test_undeclared_manifest_has_zero_components_and_says_so(_sdk):
    """"Nothing to declare" is encoded as ABSENT surfaces, not placeholders.

    A placeholder model (provider "unknown") would make the first real
    observation diff as provider 'unknown' → 'openai', i.e. major/breaking.
    Zero components is the only representation the diff engine reads as
    "not declared", and the label says which it is.
    """
    from decimalai.otel import DecimalSpanExporter

    exporter = DecimalSpanExporter(agent_name="ms-autogen")
    _export(exporter, [_MockSpan("process", 0x2, 0x1)])

    snapshot = _sdk._client.register_manifest.call_args[0][0]
    assert snapshot.components == []
    assert snapshot.version_label == "undeclared"
    # Descriptive, never a closed contract — an undeclared manifest must not
    # turn every real tool call into a contract violation.
    assert snapshot.is_closed_world is False


def test_existing_active_manifest_is_adopted_instead_of_minting_an_empty_one(_sdk):
    """The agent already declared a contract (redeploy, other rail, explicit
    register_manifest) → a run with nothing to declare joins it rather than
    stacking an ``undeclared`` version on top of it."""
    from decimalai.otel import DecimalSpanExporter

    _sdk._client.list_manifests.return_value = {
        "manifests": [
            {"id": "already-active", "status": "active", "agent_name": "known-agent"},
            {"id": "older", "status": "superseded", "agent_name": "known-agent"},
        ]
    }
    exporter = DecimalSpanExporter(agent_name="known-agent")
    _export(exporter, [_MockSpan("process", 0x3, 0x1)])

    _sdk._client.register_manifest.assert_not_called()
    assert _sent_traces(_sdk)[0].manifest_id == "already-active"


# ── 2. Never un-declare ──────────────────────────────────────


_AG2_CHAT = {
    "gen_ai.operation.name": "chat",
    "gen_ai.provider.name": "openai",
    "gen_ai.request.model": "gpt-4o-mini",
}


def _ag2_turn(trace_id, tools):
    spans = [
        _MockSpan("invoke_agent assistant", trace_id, 0x1,
                  attributes={"gen_ai.operation.name": "invoke_agent"}),
        _MockSpan("chat gpt-4o-mini", trace_id, 0x2, parent_span_id=0x1,
                  attributes=dict(_AG2_CHAT)),
    ]
    for i, tool in enumerate(tools):
        spans.append(_MockSpan(
            f"execute_tool {tool}", trace_id, 0x10 + i, parent_span_id=0x1,
            attributes={"gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.name": tool},
        ))
    return spans


def test_a_quiet_turn_never_registers_a_shrunken_manifest(_sdk):
    """Turn 1 calls both tools, turn 2 calls one. Nothing about the agent
    changed, so there must be exactly ONE manifest version — not a second one
    the platform reads as ``search removed`` / breaking / replay."""
    from decimalai.otel import DecimalSpanExporter

    exporter = DecimalSpanExporter(agent_name="ag2-agent")
    _export(exporter, _ag2_turn(0x100, ["get_weather", "search"]))
    _export(exporter, _ag2_turn(0x200, ["get_weather"]))

    assert _sdk._client.register_manifest.call_count == 1
    snapshot = _sdk._client.register_manifest.call_args[0][0]
    tools = sorted(c.component_name for c in snapshot.components
                   if c.component_type == "tool")
    assert tools == ["get_weather", "search"]

    ids = {t.manifest_id for t in _sent_traces(_sdk)}
    assert len(ids) == 1, "both turns must land on the same manifest version"


def test_a_turn_that_observes_nothing_keeps_the_populated_manifest(_sdk):
    """The "everything was removed" direction: an empty snapshot registered on
    top of a populated one is the worst version-history poisoning available."""
    from decimalai.otel import DecimalSpanExporter

    exporter = DecimalSpanExporter(agent_name="ag2-agent")
    _export(exporter, _ag2_turn(0x300, ["search"]))
    _export(exporter, [_MockSpan("housekeeping", 0x400, 0x1)])

    assert _sdk._client.register_manifest.call_count == 1
    first, second = _sent_traces(_sdk)
    assert second.manifest_id == first.manifest_id


def test_a_failed_registration_is_retried_on_the_next_trace(_sdk):
    """The fallback id is a client-side uuid the backend has never seen, and it
    rejects an unknown manifest_id exactly like a missing one — so a cached
    failure would turn one blip into permanent trace loss."""
    from decimalai.otel import DecimalSpanExporter

    _sdk._client.register_manifest.side_effect = RuntimeError("backend down")
    exporter = DecimalSpanExporter(agent_name="ag2-agent")
    _export(exporter, _ag2_turn(0x450, ["search"]))

    _sdk._client.register_manifest.side_effect = (
        lambda snapshot: {"manifest_id": "recovered"}
    )
    _export(exporter, _ag2_turn(0x460, ["search"]))

    assert _sent_traces(_sdk)[1].manifest_id == "recovered"


def test_a_newly_seen_tool_still_grows_the_manifest(_sdk):
    """Monotonic must not mean frozen — a genuinely new tool is a real change
    and still registers (as an addition, which diffs non-breaking)."""
    from decimalai.otel import DecimalSpanExporter

    exporter = DecimalSpanExporter(agent_name="ag2-agent")
    _export(exporter, _ag2_turn(0x500, ["search"]))
    _export(exporter, _ag2_turn(0x600, ["search", "book_flight"]))

    assert _sdk._client.register_manifest.call_count == 2
    snapshot = _sdk._client.register_manifest.call_args[0][0]
    assert sorted(c.component_name for c in snapshot.components
                  if c.component_type == "tool") == ["book_flight", "search"]


def test_two_agents_in_one_process_do_not_share_a_manifest(_sdk):
    """The manifest hash does not include the agent name, so a single shared
    tracker handed agent B the id it had minted for agent A."""
    from decimalai.otel import DecimalSpanExporter

    exporter = DecimalSpanExporter()  # name auto-detected per root span
    a = _ag2_turn(0x700, ["search"])
    a[0].attributes["gen_ai.agent.name"] = "alpha"
    a[0].resource = _Resource("alpha")
    b = _ag2_turn(0x800, ["search"])
    b[0].resource = _Resource("beta")
    _export(exporter, a)
    _export(exporter, b)

    by_agent = {t.agent_name: t.manifest_id for t in _sent_traces(_sdk)}
    assert set(by_agent) == {"alpha", "beta"}
    assert by_agent["alpha"] != by_agent["beta"]
    registered = [c[0][0].agent_name
                  for c in _sdk._client.register_manifest.call_args_list]
    assert sorted(registered) == ["alpha", "beta"]


def test_tool_components_use_the_declared_name_not_the_span_name(_sdk):
    """AG2 names tool spans "execute_tool <fn>"; the operation prefix must not
    end up in the manifest's tool registry."""
    from decimalai.otel import DecimalSpanExporter

    exporter = DecimalSpanExporter(agent_name="ag2-agent")
    _export(exporter, _ag2_turn(0x900, ["search"]))

    snapshot = _sdk._client.register_manifest.call_args[0][0]
    assert [c.component_name for c in snapshot.components
            if c.component_type == "tool"] == ["search"]


# ── 4. GenAI-semconv content (AG2's wire shape) ──────────────


_SEMCONV_INPUT = json.dumps([
    {"role": "system", "parts": [{"type": "text", "content": "You are terse."}]},
    {"role": "user", "parts": [{"type": "text", "content": "What is 2+2?"}]},
])
_SEMCONV_OUTPUT = json.dumps([
    {"role": "assistant", "parts": [{"type": "text", "content": "4"}]},
])


def test_semconv_messages_become_previews_not_raw_json(_sdk):
    from decimalai.otel import _preview_from_attrs

    attrs = {**_AG2_CHAT,
             "gen_ai.input.messages": _SEMCONV_INPUT,
             "gen_ai.output.messages": _SEMCONV_OUTPUT}
    assert _preview_from_attrs(attrs, "input") == "You are terse.\nWhat is 2+2?"
    assert _preview_from_attrs(attrs, "output") == "4"


def test_semconv_messages_become_the_sft_artifact(_sdk):
    """rendered_input/output are what SFT derivation reads; on this rail they
    were None for every AG2 call."""
    from decimalai.otel import DecimalSpanExporter

    exporter = DecimalSpanExporter(agent_name="ag2-agent")
    _export(exporter, [
        _MockSpan("invoke_agent assistant", 0xA00, 0x1,
                  attributes={"gen_ai.operation.name": "invoke_agent"}),
        _MockSpan("chat gpt-4o-mini", 0xA00, 0x2, parent_span_id=0x1, attributes={
            **_AG2_CHAT,
            "gen_ai.input.messages": _SEMCONV_INPUT,
            "gen_ai.output.messages": _SEMCONV_OUTPUT,
        }),
    ])

    call = _sent_traces(_sdk)[0].llm_calls[0]
    assert call.rendered_input == [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    assert call.output == {"role": "assistant", "content": "4"}
    assert call.provider == "openai"  # gen_ai.provider.name, not gen_ai.system


def test_semconv_system_prompt_reaches_the_manifest(_sdk):
    from decimalai.otel import DecimalSpanExporter

    exporter = DecimalSpanExporter(agent_name="ag2-agent")
    _export(exporter, [
        _MockSpan("chat gpt-4o-mini", 0xB00, 0x1, attributes={
            **_AG2_CHAT, "gen_ai.input.messages": _SEMCONV_INPUT,
        }),
    ])

    snapshot = _sdk._client.register_manifest.call_args[0][0]
    prompts = [c for c in snapshot.components if c.component_type == "prompt"]
    assert [p.component_name for p in prompts] == ["system"]
    assert prompts[0].schema_json["content"] == "You are terse."


def test_semconv_tool_calls_are_extracted(_sdk):
    from decimalai.otel import _extract_tool_calls

    attrs = {"gen_ai.output.messages": json.dumps([
        {"role": "assistant", "parts": [
            {"type": "tool_call", "id": "c1", "name": "get_weather",
             "arguments": {"city": "SF"}},
        ]},
    ])}
    calls = _extract_tool_calls(attrs)
    assert [(c.tool_name, c.args) for c in calls] == [("get_weather", {"city": "SF"})]


# ── The exit flush ───────────────────────────────────────────


def test_instrument_flushes_early_enough_to_reach_the_sender(monkeypatch):
    """CPython runs threading._shutdown() — which stops the thread pool the SDK
    sends on — BEFORE ordinary atexit callbacks, so the provider's default exit
    flush raised "cannot schedule new futures after interpreter shutdown" and a
    plain script exported nothing."""
    import decimalai.otel as otel_mod

    registered = []
    monkeypatch.setattr(
        threading, "_register_atexit", lambda fn: registered.append(fn)
    )

    provider_cls = MagicMock()
    with patch.dict("sys.modules", {
        "opentelemetry": MagicMock(),
        "opentelemetry.trace": MagicMock(),
        "opentelemetry.sdk": MagicMock(),
        "opentelemetry.sdk.resources": MagicMock(
            Resource=MagicMock(), SERVICE_NAME="service.name"
        ),
        "opentelemetry.sdk.trace": MagicMock(TracerProvider=provider_cls),
        "opentelemetry.sdk.trace.export": MagicMock(BatchSpanProcessor=MagicMock()),
    }):
        provider = otel_mod.instrument(agent_name="a")

    assert provider_cls.call_args.kwargs["shutdown_on_exit"] is False
    assert registered, "no early flush hook registered"
    registered[0]()
    provider.shutdown.assert_called_once()


def test_send_falls_back_inline_when_the_pool_is_already_gone(_sdk):
    """Paths we don't own (a user-built TracerProvider) still flush from an
    ordinary atexit callback. Send on the calling thread rather than drop it."""
    from decimalai.otel import DecimalSpanExporter
    import decimalai._config as cfg

    exporter = DecimalSpanExporter(agent_name="ag2-agent")
    with patch.object(
        cfg._sender, "submit",
        side_effect=RuntimeError("cannot schedule new futures after interpreter shutdown"),
    ):
        with _otel_mock():
            exporter.export(_ag2_turn(0xC00, ["search"]))

    cfg._client.ingest_trace.assert_called_once()
    assert cfg._client.ingest_trace.call_args[0][0].manifest_id


# ── The legacy install_otel() rail ───────────────────────────


def test_install_otel_exporter_stamps_a_manifest_id(_sdk):
    """decimalai.integrations.otel registered no manifest at all, so every
    trace it produced was rejected with a 400."""
    from decimalai.integrations.otel import DecimalSpanExporter as LegacyExporter

    exporter = LegacyExporter(agent_name="legacy-agent")
    with _otel_mock():
        exporter.export([
            _MockSpan("chat", 0xD00, 0x1, attributes={
                "gen_ai.request.model": "gpt-4o-mini",
                "gen_ai.system": "openai",
                "gen_ai.prompt.0.role": "system",
                "gen_ai.prompt.0.content": "You are terse.",
            }),
        ])
    from decimalai._config import _sender
    _sender.flush()

    trace = _sent_traces(_sdk)[0]
    assert trace.manifest_id
    snapshot = _sdk._client.register_manifest.call_args[0][0]
    assert {c.component_type for c in snapshot.components} == {"model", "prompt"}


def test_install_otel_trace_with_nothing_to_declare_still_has_a_manifest(_sdk):
    from decimalai.integrations.otel import DecimalSpanExporter as LegacyExporter

    exporter = LegacyExporter(agent_name="legacy-agent")
    with _otel_mock():
        exporter.export([_MockSpan("plan", 0xE00, 0x1)])
    from decimalai._config import _sender
    _sender.flush()

    assert _sent_traces(_sdk)[0].manifest_id
