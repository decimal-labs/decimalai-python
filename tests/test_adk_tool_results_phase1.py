"""ADK adapter, Phase 1 (2026-10-14): a tool's result reaches the trace.

The adapter builds one ``ToolCallRecord`` per function call when the model
turn comes back (``after_model_callback``), before any tool has run, so that
record has no result. The tool callbacks build a second record that does carry
the result, and used to attach it only when the model turn held no record with
the same tool name. The model turn always holds one, so every tool result was
dropped, including a native ``SkillToolset`` ``load_skill`` result, the one
place an ADK trace can show which skill the model read.

Every test here drives the REAL google-adk 2.12 ``Runner`` with a scripted
``BaseLlm`` (no key, no network, no backend). The records are checked on the
trace the adapter hands to the client.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import textwrap
import threading
from pathlib import Path
from typing import Any, Callable, List
from unittest.mock import MagicMock

import pytest


def _real_adk_installed() -> bool:
    # Asked at import time: sibling ADK test files stub `google.adk` in
    # sys.modules inside fixtures, and a stub must never satisfy this.
    try:
        return importlib.util.find_spec("google.adk.runners") is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(
    not _real_adk_installed(), reason="google-adk is not installed in this environment",
)

USER_TEXT = "What does a widget cost?"


# ── fixtures ────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def client(monkeypatch):
    """A fresh adapter and a mock DecimalAI client that captures every export."""
    import decimalai
    import decimalai._config as cfg
    import decimalai.adk as adk
    from decimalai._config import DecimalConfig

    mock = MagicMock()
    mock.register_manifest.return_value = {"manifest_id": "tool-results-manifest"}
    monkeypatch.setattr(cfg, "_config", DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True,
    ))
    monkeypatch.setattr(cfg, "_client", mock)
    monkeypatch.setattr(decimalai, "_global_client", mock, raising=False)
    for name, value in (
        ("_PluginClass", None),
        ("_manifest_ids", {}),
        ("_manifest_trackers", {}),
        ("_pending_manifests", {}),
        ("_skill_router_singleton", None),
        ("_live_runs", {}),
    ):
        monkeypatch.setattr(adk, name, value)
    yield mock


Step = Callable[[], Any]


def call(tool: str, /, *, call_id: str | None = None, **args: Any) -> Step:
    """A model turn that asks for one tool."""
    return lambda: [_fc(tool, args, call_id)]


def calls(*specs: tuple) -> Step:
    """A model turn that asks for several tools at once: ``(name, args[, id])``."""
    return lambda: [_fc(s[0], s[1], s[2] if len(s) > 2 else None) for s in specs]


def say(text: str) -> Step:
    from google.genai import types

    return lambda: [types.Part(text=text)]


def _fc(name: str, args: dict, call_id: str | None) -> Any:
    from google.genai import types

    return types.Part(function_call=types.FunctionCall(name=name, args=args, id=call_id))


def _scripted_model(steps: List[Step]) -> Any:
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types
    from pydantic import PrivateAttr

    class ScriptedLlm(BaseLlm):
        _turn: int = PrivateAttr(default=0)

        async def generate_content_async(self, llm_request, stream=False):
            step = steps[self._turn]
            self._turn += 1
            yield LlmResponse(content=types.Content(role="model", parts=step()))

    return ScriptedLlm(model="scripted-offline")


def _run(steps: List[Step], tools: List[Any], *, plugins_before: List[Any] = (),
         plugins_after: List[Any] = (), on_trace: Any = None) -> List[Any]:
    """One invocation through a real InMemoryRunner; returns the exported traces."""
    from google.adk.agents import LlmAgent
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    import decimalai._config as cfg
    import decimalai.adk as adk

    agent = LlmAgent(name="shop", model=_scripted_model(steps),
                     instruction="You are a shop assistant.", tools=tools)
    plugin = adk.DecimalaiPlugin(agent_name="shop", on_trace=on_trace)
    runner = InMemoryRunner(agent=agent, app_name="tool-results",
                            plugins=[*plugins_before, plugin, *plugins_after])

    async def go() -> None:
        session = await runner.session_service.create_session(app_name="tool-results", user_id="u")
        async for _ in runner.run_async(
            user_id="u", session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=USER_TEXT)]),
        ):
            pass
        await runner.close()

    asyncio.run(go())
    cfg._sender.flush()
    return [c.args[0] for c in cfg._client.ingest_trace.call_args_list]


def _records(trace: Any) -> List[Any]:
    return [tc for call in trace.llm_calls for tc in call.tool_calls]


def _tool_spans(trace: Any) -> List[Any]:
    return [s for s in trace.spans if s.span_type.value == "tool"]


def _only_trace(traces: List[Any]) -> Any:
    assert len(traces) == 1, f"expected one exported trace, got {len(traces)}"
    trace = traces[0]
    # The real client sends `model_dump(mode="json")`. A record the exporter
    # cannot serialize loses the WHOLE trace, so every test checks it.
    json.dumps(trace.model_dump(mode="json"))
    return trace


# ── tools ───────────────────────────────────────────────────


def get_price(item: str) -> dict:
    """Look up an item's price."""
    return {"item": item, "price": 12.5, "currency": "EUR"}


async def slow_lookup(n: int) -> dict:
    """Look up record n (n=1 is slow, so it finishes last)."""
    await asyncio.sleep(0.2 if n == 1 else 0.0)
    return {"n": n, "value": f"value-{n}"}


def fetch_doc(name: str) -> str:
    """Return a very long document."""
    return f"BEGIN {name} " + "x" * 50_000 + " END"


class _Opaque:
    """A tool result with no JSON form, holding something deepcopy refuses."""

    def __init__(self) -> None:
        self.lock = threading.Lock()

    def __str__(self) -> str:
        return "opaque-handle-42"


def open_handle() -> Any:
    """Return an object that is not JSON."""
    return _Opaque()


def broken_tool(reason: str) -> dict:
    """Always fails."""
    raise RuntimeError(f"tool exploded: {reason}")


# ── 1. a function tool's result lands on the model's record ─


class TestFunctionToolResults:
    def test_the_result_lands_on_the_models_tool_call_record_once(self):
        trace = _only_trace(_run(
            [call("get_price", item="widget"), say("A widget costs 12.50 EUR.")],
            [get_price],
        ))

        records = _records(trace)
        assert [(r.tool_name, r.args) for r in records] == [("get_price", {"item": "widget"})]
        (record,) = records
        assert record.result == {"item": "widget", "price": 12.5, "currency": "EUR"}
        assert record.status.value == "success"
        assert record.latency_ms is not None
        # It is the model turn's own record that carries it, not a second one.
        assert trace.llm_calls[0].tool_calls == [record]
        assert trace.llm_calls[1].tool_calls == []
        (span,) = _tool_spans(trace)
        assert "12.5" in span.output_preview

    def test_parallel_same_name_calls_each_get_their_own_result_out_of_order(self):
        """Neither call carries a model id, and n=1 finishes after n=2: matching
        by name and order alone would swap the two results."""
        trace = _only_trace(_run(
            [calls(("slow_lookup", {"n": 1}), ("slow_lookup", {"n": 2})), say("done")],
            [slow_lookup],
        ))

        records = _records(trace)
        assert [r.args for r in records] == [{"n": 1}, {"n": 2}]
        assert [r.result for r in records] == [
            {"n": 1, "value": "value-1"}, {"n": 2, "value": "value-2"},
        ]
        assert len(_tool_spans(trace)) == 2

    def test_same_name_same_args_parallel_calls_are_matched_by_the_id_adk_assigns(self):
        """Name and args cannot tell these two apart, and they finish out of
        order. The model gave no ids; ADK assigns one to each call after the
        model turn is recorded, and each result must land by THAT id."""
        made: List[Any] = []
        finished: List[int] = []
        started = {"n": 0}

        async def take_ticket(tool_context) -> dict:
            started["n"] += 1
            mine = started["n"]
            await asyncio.sleep(0.2 if mine == 1 else 0.0)
            finished.append(mine)
            return {"ticket": mine, "call_id": tool_context.function_call_id}

        def turn() -> List[Any]:
            parts = [_fc("take_ticket", {}, None), _fc("take_ticket", {}, None)]
            made.extend(p.function_call for p in parts)
            return parts

        trace = _only_trace(_run([turn, say("done")], [take_ticket]))

        assert finished == [2, 1], "the two calls did not finish out of order"
        ids = [fc.id for fc in made]
        assert all(ids) and ids[0] != ids[1]
        assert [r.result["call_id"] for r in _records(trace)] == ids

    def test_a_model_supplied_call_id_is_matched(self):
        trace = _only_trace(_run(
            [calls(("get_price", {"item": "bolt"}, "call_model_7"),
                   ("get_price", {"item": "nut"}, "call_model_8")),
             say("done")],
            [get_price],
        ))

        assert [(r.args["item"], r.result["item"]) for r in _records(trace)] == [
            ("bolt", "bolt"), ("nut", "nut"),
        ]

    def test_results_land_across_several_model_turns(self):
        trace = _only_trace(_run(
            [call("get_price", item="a"), call("get_price", item="b"), say("done")],
            [get_price],
        ))

        assert [[r.result["item"] for r in c.tool_calls] for c in trace.llm_calls] == [
            ["a"], ["b"], [],
        ]

    def test_a_result_answered_by_an_earlier_plugin_still_lands(self):
        """Another plugin listed first answers ``before_tool``, so ADK never
        calls ours there and the tool never runs. ``after_tool`` still runs,
        with that plugin's answer, and that answer is what the model got."""
        from google.adk.plugins.base_plugin import BasePlugin

        class Cache(BasePlugin):
            def __init__(self) -> None:
                super().__init__(name="cache")

            async def before_tool_callback(self, *, tool, tool_args, tool_context):
                return {"item": tool_args["item"], "price": 1.0, "cached": True}

        trace = _only_trace(_run(
            [call("get_price", item="widget"), say("done")],
            [get_price], plugins_before=[Cache()],
        ))

        (record,) = _records(trace)
        assert record.result == {"item": "widget", "price": 1.0, "cached": True}

    def test_a_tool_that_raises_marks_its_own_record_and_adds_none(self):
        trace = _only_trace(_run_expecting_error(
            [call("broken_tool", reason="disk"), say("unreachable")], [broken_tool],
        ))

        (record,) = _records(trace)
        assert record.tool_name == "broken_tool"
        assert record.status.value == "error"
        assert "tool exploded: disk" in str(record.result)
        assert trace.status.value == "error"

    def test_a_recovered_tool_error_is_one_record_carrying_the_recovery(self):
        """Ours sees ``on_tool_error`` first; a plugin after it answers, and ADK
        then runs ``after_tool`` with that answer. One call, one record, marked
        as the error it was, carrying what the model was actually given."""
        from google.adk.plugins.base_plugin import BasePlugin

        class Recover(BasePlugin):
            def __init__(self) -> None:
                super().__init__(name="recover")

            async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error):
                return {"error": f"recovered: {error}"}

        trace = _only_trace(_run(
            [call("broken_tool", reason="net"), say("sorry")],
            [broken_tool], plugins_after=[Recover()],
        ))

        (record,) = _records(trace)
        assert record.status.value == "error"
        assert record.result == {"error": "recovered: tool exploded: net"}


def _run_expecting_error(steps: List[Step], tools: List[Any]) -> List[Any]:
    with pytest.raises(Exception, match="tool exploded"):
        _run(steps, tools)
    import decimalai._config as cfg

    cfg._sender.flush()
    return [c.args[0] for c in cfg._client.ingest_trace.call_args_list]


# ── 2. what a recorded result may hold ──────────────────────


class TestRecordedResultsAreBoundedAndExportable:
    def test_a_huge_result_is_bounded_on_the_record_and_the_span(self):
        import decimalai.adk as adk

        trace = _only_trace(_run([call("fetch_doc", name="manual"), say("done")], [fetch_doc]))

        (record,) = _records(trace)
        assert isinstance(record.result, str)
        assert record.result.startswith("BEGIN manual ")
        assert len(record.result) <= adk._TOOL_RESULT_TOTAL_LIMIT + 64
        assert "truncated" in record.result
        (span,) = _tool_spans(trace)
        assert len(span.output_preview) <= adk._PREVIEW_LIMIT

    def test_a_result_with_no_json_form_is_recorded_as_text_and_the_trace_survives(self):
        """Recorded raw, this object fails ``model_dump(mode="json")`` on the
        export thread and the observer's deep copy, and the trace is lost."""
        observed: List[Any] = []
        trace = _only_trace(_run(
            [call("open_handle"), say("done")], [open_handle], on_trace=observed.append,
        ))

        (record,) = _records(trace)
        assert record.result == "opaque-handle-42"
        assert len(observed) == 1

    def test_a_recording_failure_never_reaches_the_callers_run(self, monkeypatch, caplog):
        """ADK's PluginManager re-raises a plugin callback's exception, which
        would fail the caller's run over a bookkeeping bug. It is logged
        instead, naming the tool whose result the trace now lacks."""
        import logging

        import decimalai.adk as adk

        def boom(tool, result):
            raise RuntimeError("recording bug")

        monkeypatch.setattr(adk, "_recorded_tool_result", boom)
        with caplog.at_level(logging.WARNING, logger="decimalai.adk"):
            trace = _only_trace(_run(
                [call("get_price", item="widget"), say("A widget costs 12.50 EUR.")],
                [get_price],
            ))

        assert trace.status.value == "success"
        assert trace.final_output_preview == "A widget costs 12.50 EUR."
        assert "could not record the result of tool 'get_price'" in caplog.text


class TestMatchingWithoutIds:
    """The fallback when ids do not line up: never a call carrying another
    id, identical args first, then the oldest open call of that tool."""

    @staticmethod
    def _state(*entries: tuple) -> Any:
        import decimalai.adk as adk
        from decimalai.schema.trace import ToolCallRecord

        state = adk._RunState(agent_name="shop", started_at=adk._now())
        for name, args, call_id in entries:
            record = ToolCallRecord(tool_name=name, args=args)
            state.open_tool_calls.append(adk._ModelToolCall(record, None, call_id))
        return state

    @staticmethod
    def _claim(state: Any, name: str, call_id: Any, args: dict) -> Any:
        import decimalai.adk as adk

        return adk._plugin_class()._claim_model_tool_call(state, name, call_id, args)

    def test_identical_args_win_over_order(self):
        state = self._state(("lookup", {"n": 1}, None), ("lookup", {"n": 2}, None))
        assert self._claim(state, "lookup", None, {"n": 2}).args == {"n": 2}
        assert self._claim(state, "lookup", None, {"n": 1}).args == {"n": 1}
        assert state.open_tool_calls == []

    def test_oldest_open_call_when_args_differ(self):
        state = self._state(("lookup", {"n": 1}, None), ("lookup", {"n": 2}, None))
        assert self._claim(state, "lookup", None, {"n": 9}).args == {"n": 1}

    def test_a_call_with_another_id_is_never_taken(self):
        state = self._state(("lookup", {"n": 1}, "call_a"))
        assert self._claim(state, "lookup", "call_b", {"n": 1}) is None
        assert len(state.open_tool_calls) == 1
        assert self._claim(state, "lookup", "call_a", {"n": 1}).args == {"n": 1}

    def test_another_tool_is_never_taken(self):
        state = self._state(("lookup", {}, None))
        assert self._claim(state, "other", None, {}) is None


class TestBoundedConversion:
    def test_a_self_referencing_result_terminates_and_stays_bounded(self):
        import decimalai.adk as adk

        loop: List[Any] = []
        loop.extend([loop] * 50)
        recorded = adk._bounded_tool_result({"tree": loop})
        assert len(adk._json_text(recorded)) <= adk._TOOL_RESULT_TOTAL_LIMIT + 64

    def test_a_str_enum_is_recorded_as_plain_text(self):
        import enum

        import decimalai.adk as adk

        class Color(str, enum.Enum):
            RED = "red"

        recorded = adk._bounded_tool_result({"color": Color.RED})
        assert recorded == {"color": "red"} and type(recorded["color"]) is str

    def test_a_callers_own_tool_named_load_skill_is_not_a_skill_tool(self):
        from google.adk.tools.function_tool import FunctionTool

        import decimalai.adk as adk

        def load_skill(skill_name: str) -> dict:
            return {"skill_name": skill_name}

        assert adk._is_skill_tool(FunctionTool(load_skill)) is False


# ── 3. ADK's native SkillToolset ────────────────────────────


BODY_SENTINEL = "REFUND-BODY-SENTINEL-7f3a"


def _skill_dir(root: Path) -> Path:
    skill = root / "refund-policy"
    (skill / "references").mkdir(parents=True)
    (skill / "scripts").mkdir()
    long_body = "\n".join(f"{i}. Rule {i}: apply the refund window exactly." for i in range(400))
    (skill / "SKILL.md").write_text(textwrap.dedent("""\
        ---
        name: refund-policy
        description: Apply the refund rules when a customer asks for a refund.
        metadata:
          decimalai-skill-id: sk_8f2c
          decimalai-version-id: ver_31d0
          decimalai-body-sha256: 9b8ca481e1741a4713d024f9da2e749ab41385d5004057bd3779be8305fb9cd5
        ---
        """) + f"# Refund policy\n{BODY_SENTINEL}\n{long_body}\n")
    (skill / "references" / "fees.md").write_text("| electronics | 15% |\n" + "y" * 20_000)
    (skill / "scripts" / "compute.py").write_text("print('store credit: 85.00')\n")
    return skill


def _skill_toolset(tmp_path: Path) -> Any:
    from google.adk.code_executors.unsafe_local_code_executor import (
        UnsafeLocalCodeExecutor,
    )
    from google.adk.skills import load_skill_from_dir
    from google.adk.tools.skill_toolset import SkillToolset

    skill = load_skill_from_dir(_skill_dir(tmp_path))
    return SkillToolset(skills=[skill], code_executor=UnsafeLocalCodeExecutor())


class TestNativeSkillToolsetResults:
    METADATA = {
        "decimalai-skill-id": "sk_8f2c",
        "decimalai-version-id": "ver_31d0",
        "decimalai-body-sha256": "9b8ca481e1741a4713d024f9da2e749ab41385d5004057bd3779be8305fb9cd5",
    }

    def test_load_skill_result_names_the_skill_and_its_metadata_in_full(self, tmp_path):
        trace = _only_trace(_run(
            [call("load_skill", skill_name="refund-policy"), say("store credit")],
            [_skill_toolset(tmp_path)],
        ))

        (record,) = [r for r in _records(trace) if r.tool_name == "load_skill"]
        assert record.result["skill_name"] == "refund-policy"
        assert record.result["frontmatter"]["name"] == "refund-policy"
        assert record.result["frontmatter"]["metadata"] == self.METADATA
        # The body is there, but bounded like any other tool output.
        assert BODY_SENTINEL in record.result["instructions"]
        assert "truncated" in record.result["instructions"]
        # The span preview leads with the identity, so it survives the cut.
        (span,) = [s for s in _tool_spans(trace) if s.name == "load_skill"]
        assert len(span.output_preview) <= 500
        assert "refund-policy" in span.output_preview
        assert "ver_31d0" in span.output_preview

    def test_resource_and_script_results_name_the_skill_and_the_file(self, tmp_path):
        trace = _only_trace(_run(
            [call("load_skill", skill_name="refund-policy"),
             call("load_skill_resource", skill_name="refund-policy", file_path="references/fees.md"),
             call("run_skill_script", skill_name="refund-policy", file_path="scripts/compute.py"),
             say("store credit")],
            [_skill_toolset(tmp_path)],
        ))

        by_tool = {r.tool_name: r for r in _records(trace)}
        assert set(by_tool) == {"load_skill", "load_skill_resource", "run_skill_script"}
        assert len(_records(trace)) == 3
        resource = by_tool["load_skill_resource"].result
        assert (resource["skill_name"], resource["file_path"]) == ("refund-policy", "references/fees.md")
        assert resource["content"].startswith("| electronics | 15% |")
        script = by_tool["run_skill_script"].result
        assert (script["skill_name"], script["file_path"]) == ("refund-policy", "scripts/compute.py")
        assert "store credit: 85.00" in script["stdout"]

    def test_a_failed_load_is_recorded_as_what_adk_answered(self, tmp_path):
        trace = _only_trace(_run(
            [call("load_skill", skill_name="no-such-skill"), say("sorry")],
            [_skill_toolset(tmp_path)],
        ))

        (record,) = _records(trace)
        assert record.result["error_code"] == "SKILL_NOT_FOUND"
