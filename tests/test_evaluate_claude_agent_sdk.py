"""`decimalai.evaluate()` on a Claude Agent SDK agent: `evaluate(support, cases, evals=..., run=answer)`.

What these pin, against a fake platform that serves evaluation runs and a stand-in Claude SDK whose
"model" answers each case on its own (`CaseSDK`), through the reader's own `answer()`:

- each case calls `answer()` once, inside its evaluation block: the `support.run()` it opens routes
  with the case's setup, and its trace comes back to evaluate(), which checks it, attaches the
  results and sends it once, marked ``evaluation``;
- a check reads a Claude run's tool calls as any framework's: the tool's own name, not Claude Code's
  ``mcp__<server>__<tool>``, and the text the tool returned, not its content blocks;
- ``prompt_version`` and ``exclude_skills`` reach the Claude run (its system prompt, its list, its
  ``load_skill``), and a setup DecimalAI can't confirm never starts a Claude query;
- a run that ends in an error, a function that makes no run or two, are that case's error;
- concurrent cases each get their own trace.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from unittest.mock import MagicMock

import pytest

import decimalai
import decimalai._agent as core
import decimalai._agent_binding as binding
import decimalai._config as cfg
import decimalai.claude_agent_sdk as claude
from decimalai._agent import SavedPrompt
from decimalai._config import DecimalConfig
from decimalai._evaluation import EvaluationNotApplied
from decimalai.evals import TraceData
from decimalai.evals import eval as check
from decimalai.schema.manifest import ManifestTracker
from decimalai.skill_router import SkillRouter
from tests.test_claude_agent_config import SDK, Options
from tests.test_claude_agent_sdk_handler import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from tests.test_evaluation import EvalPlatform

ORDERS = {"ORD-1042": "shipped Monday", "ORD-1077": "delivered", "ORD-1002": "charged twice"}


def lookup_order(order_id: str) -> str:
    """Look up an order by its id."""
    return ORDERS.get(order_id, "No order with that id.")


def flag_for_billing(order_id: str, note: str) -> str:
    """Flag an order's charges for the billing team."""
    return f"Flagged {order_id} for the billing team: {note}"


TOOLS = [lookup_order, flag_for_billing]


class CaseSDK(SDK):
    """The Claude SDK stand-in, answering any case on its own: it looks up an order the message
    names, flags one charged twice, loads damaged-item when the list shows it (or always, when the
    customer insists), then answers. Prompt one makes it promise a refund; "forever" runs out of turns."""

    async def query(self, *, prompt, options):
        self.calls.append((prompt, options))
        if self.barrier is not None:
            await self.barrier.wait()
        system = options.system_prompt or ""
        tools = {t.name: t for t in options.mcp_servers["support"]["tools"]}
        loader = options.mcp_servers.get("decimalai")
        steps = []
        if "ORD-" in prompt:
            order = "ORD-" + prompt.split("ORD-")[1][:4]
            steps.append(("support", "lookup_order", {"order_id": order}))
            if "twice" in prompt:
                steps.append(("support", "flag_for_billing", {"order_id": order, "note": "charged twice"}))
        if loader is not None and ("broken" in prompt or "insist" in prompt) \
                and ("damaged-item" in system or "insist" in prompt):
            steps.append(("decimalai", "load_skill", {"name": "damaged-item"}))
        for i, (server, name, args) in enumerate(steps):
            call = ToolUseBlock(f"call-{len(self.calls)}-{i}", f"mcp__{server}__{name}", args)
            yield AssistantMessage([call], model=options.model)
            handler = loader["tools"][0].handler if server == "decimalai" else tools[name].handler
            result = await handler(args)
            yield UserMessage([ToolResultBlock(call.id, result["content"], is_error=result.get("is_error", False))])
        if "forever" in prompt:
            end = ResultMessage(result="", is_error=True, session_id="s")
            end.subtype = "error_max_turns"
            yield end
            return
        text = "I'll issue a refund today." if "PROMISE" in system else f"Your order is on its way. ({prompt})"
        yield AssistantMessage([TextBlock(text)], model=options.model)
        yield ResultMessage(result=text, usage={"input_tokens": 5, "output_tokens": 3},
                            session_id=f"session-{len(self.calls)}")


@pytest.fixture
def platform(monkeypatch):
    monkeypatch.setattr(cfg, "_config", DecimalConfig(api_key="dai_sk_test", base_url="http://evaluation.test"))
    client = MagicMock()
    client.register_manifest.return_value = {"manifest_id": "man_test"}
    monkeypatch.setattr(cfg, "_client", client)
    monkeypatch.setattr(claude, "_manifest_tracker", ManifestTracker())
    monkeypatch.setattr(claude, "_manifest_ids", {})
    monkeypatch.setattr(claude, "_manifest_hashes", {})
    monkeypatch.setattr(claude, "_pending_manifests", {})
    monkeypatch.setattr(binding, "_in_background", lambda fn, *args: fn(*args))
    fake = EvalPlatform()
    router = SkillRouter(api_key="dai_sk_test", base_url="http://evaluation.test", inject_body=False)
    monkeypatch.setattr(router, "_request", fake.request)
    monkeypatch.setattr(core, "_skill_router_singleton", router)
    fake.saved = SavedPrompt(agent_name="support", system_prompt="You help customers with orders.",
                             version_number=4, content_hash="c" * 64)
    monkeypatch.setattr(core, "read_prompt", lambda name, **kwargs: fake.saved)
    fake.client = client
    fake.sdk = CaseSDK()
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake.sdk)
    return fake


def program(sdk):
    """The reader's main.py, as the tutorial's Claude Agent SDK tab has it: `support`, `answer()`."""

    def as_tool(fn):
        async def handler(arguments):
            return {"content": [{"type": "text", "text": fn(**arguments)}]}
        return sdk.tool(fn.__name__, fn.__doc__, {n: str for n in inspect.signature(fn).parameters})(handler)

    support = claude.agent_config("support", fallback="fallback")
    options = Options(mcp_servers={"support": sdk.create_sdk_mcp_server("support", tools=[*map(as_tool, TOOLS)])},
                      allowed_tools=[f"mcp__support__{fn.__name__}" for fn in TOOLS])

    async def answer(question):
        reply = None
        async with support.run(question) as run:
            async for message in run.query(options=options):
                if type(message).__name__ == "ResultMessage":
                    reply = message.result
        return reply

    return support, answer


def traces(platform):
    cfg._sender.flush()
    return [call.args[0] for call in platform.client.ingest_trace.call_args_list]


CASES = [
    {"id": "late-order", "input": "Where is ORD-1042? It was due Monday.", "order_id": "ORD-1042"},
    {"id": "broken-mug", "input": "My mug from ORD-1077 arrived broken.", "order_id": "ORD-1077",
     "skill": "damaged-item"},
    {"id": "double-charge", "input": "I was charged twice for ORD-1002.", "order_id": "ORD-1002",
     "action": "flag_for_billing"},
    {"id": "hello", "input": "Hi there"},
]


@check
def looked_up_the_order(trace: TraceData):
    order = trace.case.get("order_id")
    if order is None:
        return None
    return any(c.name == "lookup_order" and c.args.get("order_id") == order for c in trace.tool_calls)


@check
def loaded_the_right_skill(trace: TraceData):
    skill = trace.case.get("skill")
    return None if skill is None else skill in trace.skills_loaded


@check
def did_the_action(trace: TraceData):
    action = trace.case.get("action")
    if action is None:
        return None
    # The tutorial's check, as written: the tool's own name, and the text it returned.
    return any(c.name == action and c.args.get("order_id") == trace.case.get("order_id")
               and str(c.result).startswith("Flagged") for c in trace.tool_calls)


@check
def never_promises_a_refund(trace: TraceData) -> bool:
    return "refund" not in str(trace.output).lower()


CHECKS = [looked_up_the_order, loaded_the_right_skill, did_the_action, never_promises_a_refund]


def test_evaluate_runs_each_case_through_your_function_and_sends_its_trace_once_as_an_evaluation(platform):
    support, answer = program(platform.sdk)
    report = decimalai.evaluate(support, CASES, evals=CHECKS, run=answer)
    assert [c.id for c in report.cases] == ["late-order", "broken-mug", "double-charge", "hello"]
    assert all(c.error is None and c.check_errors == {} for c in report.cases), report
    assert report.summary == {
        "looked_up_the_order": {"passed": 3, "failed": 0, "skipped": 1, "errors": 0},
        "loaded_the_right_skill": {"passed": 1, "failed": 0, "skipped": 3, "errors": 0},
        "did_the_action": {"passed": 1, "failed": 0, "skipped": 3, "errors": 0},
        "never_promises_a_refund": {"passed": 4, "failed": 0, "skipped": 0, "errors": 0},
    }
    assert report.label == "served" and report.prompt_version == 4
    assert report.cases[3].output == "Your order is on its way. (Hi there)"
    sent = traces(platform)
    assert sorted(str(t.id) for t in sent) == sorted(c.trace_id for c in report.cases)
    assert {t.source_type.value for t in sent} == {"evaluation"}
    by_id = {str(t.id): t for t in sent}
    for case, row in zip(report.cases, CASES):
        # Four cases at once (concurrency=4), each with its own trace.
        assert by_id[case.trace_id].user_input_preview == row["input"]
    charge = by_id[report.cases[2].trace_id]
    assert {s["name"]: s["passed"] for s in charge.eval_scores} == {
        "looked_up_the_order": True, "did_the_action": True, "never_promises_a_refund": True}
    assert {s["source"] for s in charge.eval_scores} == {"evaluate"}
    assert len(platform.evaluation_calls) == 4
    assert all(fields.get("evaluation") is True for _p, fields in platform.evaluation_calls)
    assert str(report).splitlines()[0] == "support · served (prompt v4) · 4 cases"


def test_a_check_reads_a_claude_tool_by_its_own_name_and_the_text_it_returned(platform):
    support, answer = program(platform.sdk)
    seen = {}

    @check
    def record(trace: TraceData):
        seen[trace.case["id"]] = trace.tool_calls
        return None

    decimalai.evaluate(support, CASES[2:3], evals=[record], run=answer)
    lookup, flag = seen["double-charge"]
    assert (lookup.name, lookup.mcp_server, lookup.result) == ("lookup_order", "support", "charged twice")
    assert (flag.name, flag.args) == ("flag_for_billing", {"order_id": "ORD-1002", "note": "charged twice"})
    assert flag.result == "Flagged ORD-1002 for the billing team: charged twice"
    # The stored trace keeps what Claude Code called the tool, and the blocks it returned.
    stored = traces(platform)[0].llm_calls[1].tool_calls[0]
    assert stored.tool_name == "mcp__support__flag_for_billing"
    assert stored.result == [{"type": "text", "text": "Flagged ORD-1002 for the billing team: charged twice"}]


def test_a_draft_prompt_version_reaches_the_claude_run(platform):
    support, answer = program(platform.sdk)
    served = decimalai.evaluate(support, CASES, evals=CHECKS, run=answer)
    draft = decimalai.evaluate(support, CASES, evals=CHECKS, run=answer, prompt_version=1)
    assert draft.label == "prompt v1" and draft.prompt_version == 1
    systems = [options.system_prompt for _prompt, options in platform.sdk.calls[len(CASES):]]
    assert all(s.startswith("You are prompt one. PROMISE refunds.") for s in systems)
    assert draft.summary["never_promises_a_refund"] == {"passed": 0, "failed": 4, "skipped": 0, "errors": 0}
    assert [(c[0], c[2], c[3]) for c in draft.compare(served).changed] == [
        ("late-order", "pass", "fail"), ("broken-mug", "pass", "fail"),
        ("double-charge", "pass", "fail"), ("hello", "pass", "fail")]
    # What production serves is untouched: an ordinary run afterwards gets the served prompt.
    asyncio.run(answer("Hi again"))
    assert platform.sdk.calls[-1][1].system_prompt.startswith("You help customers with orders.")


def test_a_skill_left_out_is_not_listed_and_its_load_is_refused(platform):
    support, answer = program(platform.sdk)
    cases = [*CASES[:2], {"id": "insists", "input": "I insist on the damaged-item skill.", "skill": "damaged-item"}]
    report = decimalai.evaluate(support, cases, evals=CHECKS, run=answer, exclude_skills=["damaged-item"])
    assert report.label == "without damaged-item"
    for _prompt, options in platform.sdk.calls:
        assert "damaged-item" not in options.system_prompt
        tool = options.mcp_servers["decimalai"]["tools"][0]
        assert "damaged-item" not in tool.input_schema["properties"]["name"].get("enum", [])
    assert report.summary["loaded_the_right_skill"] == {"passed": 0, "failed": 2, "skipped": 1, "errors": 0}
    insisted = next(t for t in traces(platform) if t.user_input_preview.startswith("I insist"))
    refusal = insisted.llm_calls[0].tool_calls[0].result[0]["text"]
    assert "isn't available for this run" in refusal
    assert insisted.skills_loaded_by_agent == []


def test_a_setup_decimalai_cannot_confirm_never_starts_a_claude_query(platform):
    support, answer = program(platform.sdk)
    platform.echo = False
    with pytest.raises(EvaluationNotApplied, match="predates evaluation runs"):
        decimalai.evaluate(support, CASES, evals=CHECKS, run=answer, prompt_version=3)
    with pytest.raises(EvaluationNotApplied, match="no prompt version 9"):
        platform.echo = True
        decimalai.evaluate(support, CASES, evals=CHECKS, run=answer, prompt_version=9)
    assert platform.sdk.calls == []


def test_a_run_that_ends_in_an_error_is_that_cases_error_and_its_trace_still_goes_out(platform):
    support, answer = program(platform.sdk)
    cases = [CASES[0], {"id": "stuck", "input": "Keep going forever."}]
    report = decimalai.evaluate(support, cases, evals=CHECKS, run=answer)
    assert report.cases[0].error is None
    assert report.cases[1].error == "the Claude Agent SDK run ended in an error: error_max_turns"
    assert report.cases[1].results == {}
    assert "Cases that did not run\n  stuck   the Claude Agent SDK run ended in an error" in str(report)
    stuck = next(t for t in traces(platform) if str(t.id) == report.cases[1].trace_id)
    assert stuck.source_type.value == "evaluation" and stuck.eval_scores == []


def test_a_function_that_makes_no_run_or_two_is_said_so(platform):
    support, answer = program(platform.sdk)

    async def no_run(question):
        return "I answered without the agent."

    async def twice(question):
        await answer(question)
        return await answer(question)

    report = decimalai.evaluate(support, CASES[:1], evals=CHECKS, run=no_run)
    assert report.cases[0].error.startswith("no_run() made no run of 'support'")
    report = decimalai.evaluate(support, CASES[:1], evals=CHECKS, run=twice)
    assert report.cases[0].error == "the case ran 'support' 2 times; a case is one run, so its checks did not run"
    assert len(traces(platform)) == 2  # both went out, unchecked


def test_a_sync_function_runs_off_the_event_loop(platform):
    support, answer = program(platform.sdk)

    def answer_sync(question):
        return asyncio.run(answer(question))

    report = decimalai.evaluate(support, CASES, evals=CHECKS, run=answer_sync, exclude_skills=["damaged-item"])
    assert all(c.error is None for c in report.cases), report
    assert report.summary["loaded_the_right_skill"]["failed"] == 1
    assert {t.source_type.value for t in traces(platform)} == {"evaluation"}


def test_evaluate_async_in_a_running_loop(platform):
    support, answer = program(platform.sdk)
    report = asyncio.run(decimalai.evaluate_async(support, CASES, evals=CHECKS, run=answer, concurrency=1))
    assert report.summary["looked_up_the_order"]["passed"] == 3


def test_evaluate_says_what_a_claude_agent_needs(platform):
    support, answer = program(platform.sdk)
    with pytest.raises(TypeError, match=r"run=reply"):
        decimalai.evaluate(support, CASES, evals=CHECKS)
    with pytest.raises(TypeError, match="takes your function"):
        decimalai.evaluate(support, CASES, evals=CHECKS, run="answer")
    with pytest.raises(TypeError, match="context= is Runner.run"):
        decimalai.evaluate(support, CASES, evals=CHECKS, run=answer, context=lambda case: {})
    with pytest.raises(TypeError, match="or a decimalai.claude_agent_sdk.agent_config"):
        decimalai.evaluate(object(), CASES, evals=CHECKS)
    assert platform.sdk.calls == []


def test_an_ordinary_claude_run_is_unchanged_and_carries_its_whole_answer(platform):
    support, answer = program(platform.sdk)
    long_question = "Hi " + "there " * 200
    reply = asyncio.run(answer(long_question))
    trace, = traces(platform)
    assert trace.source_type.value == "production" and trace.eval_scores in (None, [])
    assert trace.final_output == reply and len(reply) > 1000
    assert trace.final_output_preview == reply[:500]
