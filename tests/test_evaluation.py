"""Evaluation runs and `decimalai.evaluate()` (tutorial Part 4, decimalai._evaluation).

What these pin, against a fake platform that serves evaluation runs (or, with ``echo=False``, one
from before them):

- an override run routes first and sends its override; its prompt is the saved version's, its list
  leaves the excluded skills out, and the platform refuses it their loads;
- its answer never becomes the agent's local copy (list, always-include texts, experiment);
- a run DecimalAI cannot confirm does not start: no echo, a 422, an outage, no key;
- `evaluate()` runs each case once through the real OpenAI Agents runner, checks it, attaches the
  results to the case's trace, sends that trace once as ``evaluation``, and reports and compares;
- `support.override()` hands back the run's trace id; the raw `agent_config().run(...)` overrides
  mark the OpenTelemetry run it is in.
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from uuid import uuid4

import pytest

import decimalai
import decimalai._agent_binding as binding
import decimalai.skill_router as sr
from decimalai._evaluation import EvaluationNotApplied, Override, Trial, block
from decimalai.evals import TraceData, eval as check
from tests.test_agent_binding import _RUN, Binding, FakePlatform, prompt

ROUTES = ("/api/v1/skills/menu", "/api/v1/skills/route")
VERSIONS = {1: "You are prompt one. PROMISE refunds.", 3: "You help customers with orders."}


class EvaluationAnswers:
    """A platform that serves evaluation runs; ``echo = False`` makes it one from before them."""

    def _start_evaluation(self):
        self.echo = True
        self.evaluation_calls: list = []
        self.withheld: dict = {}           # routing id -> the skills its evaluation run left out
        self._swap = threading.Lock()

    def request(self, method, path, json=None, params=None, **kwargs):
        fields = (params if method == "GET" else json) or {}
        if path in ROUTES and fields.get("evaluation"):
            self.evaluation_calls.append((path, dict(fields)))
            if self.echo:
                return self._evaluation_answer(method, path, json, params, fields, **kwargs)
        if path.endswith("/body"):
            name = path.split("/")[-2]
            if name in self.withheld.get((params or {}).get("routing_id"), ()):
                raise sr.SkillRouterError("SkillRouter request failed (404)", status_code=404)
        return super().request(method, path, json=json, params=params, **kwargs)

    def _evaluation_answer(self, method, path, json, params, fields, **kwargs):
        version = fields.get("prompt_version")
        names = list(fields.get("exclude_skills") or [])
        if version is not None and version not in VERSIONS:
            raise sr.SkillRouterError("SkillRouter request failed (422)", status_code=422,
                                      detail=f"support has no prompt version {version}; it has prompt versions 1, 3")
        unknown = [n for n in names if n not in self.skills]
        if unknown:
            raise sr.SkillRouterError("SkillRouter request failed (422)", status_code=422,
                                      detail=f"support has no skill named {unknown[0]!r}")
        with self._swap:
            kept, always = self.skills, getattr(self, "always", {})
            self.skills = [n for n in kept if n not in names]
            if always:
                self.always = {n: v for n, v in always.items() if n not in names}
            try:
                answer = super().request(method, path, json=json, params=params, **kwargs)
            finally:
                self.skills = kept
                if always:
                    self.always = always
        answer["evaluation"] = {"prompt_version": version, "exclude_skills": names}
        if version is not None:
            answer["prompt"] = {"version": version, "content": VERSIONS[version], "content_hash": "e" * 64}
        self.withheld[answer["routing_id"]] = names
        return answer


class EvalPlatform(EvaluationAnswers, FakePlatform):
    def __init__(self):
        super().__init__()
        self._start_evaluation()


class EvalBinding(Binding):
    """`Binding`, keeping each run's state and the trials its runs were noted for."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noted: list = []

    def _note_evaluation(self, run_key, trial):
        self.noted.append(trial)

    def run(self, message, values=None):
        _RUN.set(f"run-{uuid4().hex}")
        context = {"message": message, "values": values}
        self.state = self._prepare_run_sync(context, None, _RUN.get())
        return self._compose(context, None, _RUN.get(), self.state)


@pytest.fixture
def platform(monkeypatch):
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    monkeypatch.setattr(cfg, "_config", DecimalConfig(
        api_key="dai_sk_test", base_url="http://evaluation.test", enabled=True))
    monkeypatch.setattr(binding, "_in_background", lambda fn, *args: fn(*args))
    fake = EvalPlatform()
    router = sr.SkillRouter(api_key="dai_sk_test", base_url="http://evaluation.test", inject_body=False)
    monkeypatch.setattr(router, "_request", fake.request)
    fake.router = router
    return fake


def _in(trial_override: Override, fn, *args):
    trial = Trial(trial_override)
    with block(trial):
        return fn(*args), trial


# ── one evaluation run ──────────────────────────────────────────────────────


def test_an_override_run_routes_first_and_gets_the_version_and_list_it_asked_for(platform):
    support = EvalBinding(prompt(), platform.router)
    support.run("Where is my order?")                      # the agent's own list, kept locally
    text, trial = _in(Override.make("support", 1, ["damaged-item"]), support.run, "My mug arrived broken.")
    assert text.startswith("You are prompt one. PROMISE refunds.\n\n## Available Skills")
    assert "damaged-item" not in text and "| order-support |" in text
    path, fields = platform.evaluation_calls[-1]
    assert path == ROUTES[1] and fields["prompt_version"] == 1 and fields["exclude_skills"] == ["damaged-item"]
    assert "experiments" not in fields
    record = support.records[-1]
    assert record["prompt"]["version"] == 1 and record["prompt"]["content_hash"] == "e" * 64
    assert record["offered"] == ["order-support"] and record["routing_id"] in platform.withheld
    assert support.noted == [trial]
    # The platform refuses the run the skill it left out, and the run hears it isn't this run's.
    refused = support._load_sync("damaged-item", support.state)
    assert refused.startswith("load_skill: 'damaged-item' isn't available for this run.")


def test_an_override_run_never_becomes_the_agents_local_copy(platform):
    platform.always = {"order-support": "v1"}
    support = EvalBinding(prompt(), platform.router)
    first = support.run("Where is my order?")
    text, _trial = _in(Override.make("support", None, ["order-support", "damaged-item"]), support.run, "Hello")
    assert support.state["names"] == [] and "## Skill: order-support" not in text
    assert "## Skill: order-support" in first
    calls = len(platform.menu_calls())
    again = support.run("Where is my order?")
    assert again == first, "the list and the always-include text are the agent's, as before"
    assert support.skill_names == ["damaged-item", "order-support"]
    assert support._experiment is None
    assert len(platform.menu_calls()) == calls + 1, "the local copy, routed in the background"


def test_an_override_with_neither_field_is_an_evaluation_run_on_the_served_setup(platform):
    support = EvalBinding(prompt(), platform.router)
    text, _trial = _in(Override.make("support"), support.run, "Where is my order?")
    assert text.startswith("You help customers with orders.") and "| damaged-item |" in text
    assert platform.evaluation_calls[-1][1] == {
        "query": "Where is my order?", "agent_name": "support", "top_k": 10, "include_attachments": False,
        "evaluation": True}


def test_an_override_for_another_agent_leaves_this_one_alone(platform):
    support = EvalBinding(prompt(), platform.router)
    _in(Override.make("billing", 1), support.run, "Where is my order?")
    assert platform.evaluation_calls == [] and support.noted == []


@pytest.mark.parametrize("setup, says", [
    (lambda p: setattr(p, "echo", False), "the backend predates evaluation runs"),
    (lambda p: setattr(p, "down", True), r"could not set up this evaluation run of 'support' \(prompt v1\): "
                                         "transport error: timed out"),
])
def test_a_run_decimalai_cannot_confirm_does_not_start(platform, setup, says):
    support = EvalBinding(prompt(), platform.router)
    setup(platform)
    with pytest.raises(EvaluationNotApplied, match=says):
        _in(Override.make("support", 1), support.run, "Where is my order?")
    assert support.records == [], "nothing was composed"


def test_a_refusal_says_what_the_platform_said(platform):
    support = EvalBinding(prompt(), platform.router)
    with pytest.raises(EvaluationNotApplied, match="support has no prompt version 9; it has prompt versions 1, 3"):
        _in(Override.make("support", 9), support.run, "Hi")
    with pytest.raises(EvaluationNotApplied, match="support has no skill named 'refund-polcy'"):
        _in(Override.make("support", None, ["refund-polcy"]), support.run, "Hi")


def test_an_answer_for_another_setup_is_refused():
    with pytest.raises(EvaluationNotApplied, match="another setup"):
        Override.make("support", 2).check({"prompt_version": 3, "exclude_skills": []}, {"content": "x"})
    with pytest.raises(EvaluationNotApplied, match="sent no text"):
        Override.make("support", 2).check({"prompt_version": 2, "exclude_skills": []}, None)
    Override.make("support", None, ["b", "a"]).check({"prompt_version": None, "exclude_skills": ["a", "b"]}, None)


def test_a_keyless_or_switched_off_agent_cannot_have_an_evaluation_run(platform):
    support = EvalBinding(SimpleNamespace(**vars(prompt()), keyless=True), platform.router)
    with pytest.raises(EvaluationNotApplied, match="no API key is set"):
        _in(Override.make("support", 1), support.run, "Hi")


def test_override_values_are_checked_and_named():
    assert Override.make("a").label == "served"
    assert Override.make("a", 4).label == "prompt v4"
    assert Override.make("a", None, "refund-policy").exclude_skills == ("refund-policy",)
    assert Override.make("a", 4, ["x", "y", "x"]).label == "prompt v4 without x, y"
    for bad in (0, -1, True, "4"):
        with pytest.raises(ValueError, match="prompt_version"):
            Override.make("a", bad)
    with pytest.raises(ValueError, match="exclude_skills"):
        Override.make("a", None, ["ok", ""])


def test_an_evaluation_menu_is_never_cached_and_its_failure_raises(platform):
    router = platform.router
    router.get_menu(agent_name="support")
    router.get_menu(agent_name="support", evaluation={"prompt_version": None, "exclude_skills": []})
    router.get_menu(agent_name="support", evaluation={"prompt_version": None, "exclude_skills": []})
    assert len(platform.evaluation_calls) == 2
    platform.down = True
    assert router.get_menu(agent_name="support", force_refresh=True)["strategy"] == "none"
    with pytest.raises(sr.SkillRouterError):
        router.get_menu(agent_name="support", evaluation={"exclude_skills": ["damaged-item"]})


# ── the raw agent_config().run(...) ──────────────────────────────────────────


def test_raw_run_overrides_mark_the_opentelemetry_run_they_are_in(platform, monkeypatch):
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace import TracerProvider

    import decimalai._agent as agent_mod
    import decimalai.otel as otel

    monkeypatch.setattr(agent_mod, "_skill_router_singleton", platform.router)
    monkeypatch.setattr(agent_mod, "read_prompt", lambda name, **kw: SimpleNamespace(
        **vars(prompt()), resolved_from=None, label=None, keyless=False))
    support = decimalai.agent_config("support")
    assert support.run("Where is my order?").trace_id is None, "an ordinary run names no trace"
    run = support.run("Where is my order?", prompt_version=1)
    assert run.system.startswith("You are prompt one.") and run.trace_id is None, "no OpenTelemetry run"
    provider = TracerProvider()
    with provider.get_tracer("test").start_as_current_span("run"):
        key = otel.current_run_key()
        run = support.run("My mug arrived broken.", exclude_skills=["damaged-item"])
    assert "damaged-item" not in run.system
    rail = otel._pop_skill_rail(key)
    assert rail["evaluation_trace_id"] == run.trace_id and rail["offered"] == ["order-support"]


# ── evaluate() through the OpenAI Agents SDK ─────────────────────────────────

agents = pytest.importorskip("agents")

from agents import Agent, Model, ModelResponse, Runner, Usage  # noqa: E402
from openai.types.responses import (  # noqa: E402
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

import decimalai.openai_agents as oa  # noqa: E402
from tests.test_openai_agents_load_agent import FakeDecimal, decimal, lookup_order, sent  # noqa: E402, F401


class EvalDecimal(EvaluationAnswers, FakeDecimal):
    def __init__(self):
        super().__init__()
        self._start_evaluation()


@pytest.fixture
def evaldecimal(decimal, monkeypatch):  # noqa: F811 - the imported fixture
    """The `decimal` fixture's SDK, answering from a platform that serves evaluation runs."""
    fake = EvalDecimal()
    fake.client = decimal.client
    decimal.client.get_agent_prompt.side_effect = fake.prompt_payload
    monkeypatch.setattr(oa._skill_router_singleton, "_request", fake.request)
    return fake


class CaseModel(Model):
    """Answers any case on its own: looks up an order the message names, loads a skill the
    message is about, then answers; prompt one makes it promise a refund. Each answer is
    traced as the OpenAI model's are, in a response span."""

    def __init__(self):
        self.instructions: list = []
        self.loops: set = set()

    async def get_response(self, system_instructions, input, model_settings, tools, *args, **kwargs):
        self.instructions.append(system_instructions)
        self.loops.add(id(asyncio.get_running_loop()))
        items = input if isinstance(input, list) else [{"role": "user", "content": input}]
        message = next((i.get("content") for i in items if isinstance(i, dict) and i.get("role") == "user"), "")
        done = {getattr(i, "name", None) or (i.get("name") if isinstance(i, dict) else None)
                for i in items if (i.get("type") if isinstance(i, dict) else getattr(i, "type", None)) == "function_call"}
        tools = {t.name for t in tools}
        call = None
        if "ORD-" in message and "lookup_order" not in done:
            call = ("lookup_order", {"order_id": "ORD-" + message.split("ORD-")[1][:4]})
        elif "broken" in message and "load_skill" not in done and "load_skill" in tools \
                and "damaged-item" in system_instructions:
            call = ("load_skill", {"name": "damaged-item"})
        if call is not None:
            output = [ResponseFunctionToolCall(id=f"fc{uuid4().hex[:6]}", type="function_call",
                                               call_id=f"call{uuid4().hex[:6]}", name=call[0],
                                               arguments=json.dumps(call[1]))]
        else:
            text = "I'll issue a refund today." if "PROMISE" in system_instructions else "Your order is on its way."
            output = [ResponseOutputMessage(id="msg", type="message", role="assistant", status="completed",
                                            content=[ResponseOutputText(type="output_text", text=text, annotations=[])])]
        from agents.tracing import response_span

        with response_span() as span:
            span.span_data.response = SimpleNamespace(
                id=f"resp_{uuid4().hex[:8]}", model="gpt-test", usage=None, output=output,
                temperature=None, max_output_tokens=None)
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError
        yield  # pragma: no cover


CASES = [
    {"id": "late-order", "input": "Where is ORD-1042? It was due Monday.", "order_id": "ORD-1042"},
    {"id": "broken-mug", "input": "My mug from ORD-1077 arrived broken.", "order_id": "ORD-1077",
     "skill": "damaged-item"},
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
def never_promises_a_refund(trace: TraceData) -> bool:
    return "refund" not in str(trace.output).lower()


CHECKS = [looked_up_the_order, loaded_the_right_skill, never_promises_a_refund]


def _agent(model=None):
    support = oa.agent_config("support")
    return Agent(name="support", instructions=support.instructions,
                 tools=[lookup_order, *support.tools], model=model or CaseModel()), support


def test_evaluate_runs_each_case_checks_it_and_sends_its_trace_once_as_an_evaluation(evaldecimal):
    agent, _support = _agent()
    report = decimalai.evaluate(agent, CASES, evals=CHECKS)
    assert [c.id for c in report.cases] == ["late-order", "broken-mug", "hello"]
    assert all(c.error is None and c.check_errors == {} for c in report.cases), report
    assert report.summary == {
        "looked_up_the_order": {"passed": 2, "failed": 0, "skipped": 1, "errors": 0},
        "loaded_the_right_skill": {"passed": 1, "failed": 0, "skipped": 2, "errors": 0},
        "never_promises_a_refund": {"passed": 3, "failed": 0, "skipped": 0, "errors": 0},
    }
    assert report.label == "served" and report.prompt_version == 2
    traces = sent(evaldecimal)
    assert sorted(str(t.id) for t in traces) == sorted(c.trace_id for c in report.cases)
    assert {t.source_type.value for t in traces} == {"evaluation"}
    broken = next(t for t in traces if str(t.id) == report.cases[1].trace_id)
    assert {s["name"]: s["passed"] for s in broken.eval_scores} == {
        "looked_up_the_order": True, "loaded_the_right_skill": True, "never_promises_a_refund": True}
    assert {s["source"] for s in broken.eval_scores} == {"evaluate"}
    assert all(fields.get("evaluation") is True for _p, fields in evaldecimal.evaluation_calls)
    assert len(evaldecimal.evaluation_calls) == 3
    text = str(report)
    assert text.splitlines()[0] == "support · served (prompt v2) · 3 cases"
    assert "looked_up_the_order" in text and "2/2" in text


def test_a_draft_version_compared_with_the_served_one(evaldecimal):
    agent, _support = _agent()
    served = decimalai.evaluate(agent, CASES, evals=CHECKS)
    draft = decimalai.evaluate(agent, CASES, evals=CHECKS, prompt_version=1)   # a second loop run
    assert draft.label == "prompt v1" and draft.prompt_version == 1
    assert draft.summary["never_promises_a_refund"] == {"passed": 0, "failed": 3, "skipped": 0, "errors": 0}
    comparison = draft.compare(served)
    assert [(c[0], c[2], c[3]) for c in comparison.changed] == [
        ("late-order", "pass", "fail"), ("broken-mug", "pass", "fail"), ("hello", "pass", "fail")]
    text = str(comparison)
    assert text.splitlines()[0].startswith("support · 3 cases") and "served (prompt v2)" in text
    assert "prompt v1" in text and "−3" in text and "pass → fail" in text
    assert text.endswith("3 cases, one run each: a smoke test, not a measurement. Models vary run to run.")


def test_a_skill_left_out_is_not_listed_or_loaded(evaldecimal):
    agent, _support = _agent()
    report = decimalai.evaluate(agent, CASES, evals=CHECKS, exclude_skills=["damaged-item"])
    assert report.label == "without damaged-item"
    broken = report.cases[1]
    assert broken.outcome("loaded_the_right_skill") == "fail"
    assert [fields["exclude_skills"] for _p, fields in evaldecimal.evaluation_calls] == [["damaged-item"]] * 3
    assert evaldecimal.body_calls() == [], "nothing loaded the skill left out"


def test_a_check_that_raises_is_an_error_on_its_case_never_a_skip(evaldecimal):
    @check
    def reads_a_missing_field(trace: TraceData) -> bool:
        return trace.case["expected_status"] == "shipped"

    agent, _support = _agent()
    report = decimalai.evaluate(agent, CASES[:1], evals=[reads_a_missing_field])
    case = report.cases[0]
    assert case.outcome("reads_a_missing_field") == "error"
    assert case.check_errors == {"reads_a_missing_field": "KeyError: 'expected_status'"}
    assert report.summary["reads_a_missing_field"]["errors"] == 1
    assert "Checks that raised" in str(report)


def test_when_no_case_can_run_on_the_setup_asked_for_evaluate_raises(evaldecimal):
    agent, _support = _agent()
    with pytest.raises(EvaluationNotApplied, match="no prompt version 9"):
        decimalai.evaluate(agent, CASES, evals=CHECKS, prompt_version=9)
    evaldecimal.echo = False
    with pytest.raises(EvaluationNotApplied, match="predates evaluation runs"):
        decimalai.evaluate(agent, CASES, evals=CHECKS)
    # Each failed run still went out, as an evaluation run, never as one of the agent's own.
    assert {t.source_type.value for t in sent(evaldecimal)} == {"evaluation"}


def test_cases_come_from_a_jsonl_file_and_are_checked(evaldecimal, tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text("# eight cases, one per line\n" + "\n".join(json.dumps(c) for c in CASES[:2]) + "\n\n")
    agent, _support = _agent()
    assert [c.id for c in decimalai.evaluate(agent, str(path), evals=CHECKS).cases] == ["late-order", "broken-mug"]
    path.write_text('{"id": "a", "input": "hi"}\n{"id": "b"}\n')
    with pytest.raises(ValueError, match="line 2: a case needs an \"input\""):
        decimalai.evaluate(agent, path, evals=CHECKS)
    path.write_text('{"id": "a", "input": "hi"}\n{"id": "a", "input": "again"}\n')
    with pytest.raises(ValueError, match="two cases are named 'a'"):
        decimalai.evaluate(agent, path, evals=CHECKS)


def test_evaluate_refuses_what_it_cannot_run(evaldecimal):
    plain = Agent(name="plain", instructions="Answer.", model=CaseModel())
    with pytest.raises(TypeError, match="needs an Agent whose instructions come from agent_config"):
        decimalai.evaluate(plain, CASES, evals=CHECKS)
    with pytest.raises(TypeError, match="OpenAI Agents SDK Agent"):
        decimalai.evaluate(object(), CASES, evals=CHECKS)
    agent, _support = _agent()

    async def inside_a_loop():
        with pytest.raises(RuntimeError, match="evaluate_async"):
            decimalai.evaluate(agent, CASES, evals=CHECKS)
        return await decimalai.evaluate_async(agent, CASES[:1], evals=CHECKS)

    assert asyncio.run(inside_a_loop()).cases[0].error is None


def test_an_override_block_hands_back_its_runs_trace(evaldecimal):
    agent, support = _agent()
    with support.override(prompt_version=1) as trial:
        result = Runner.run_sync(agent, "Where is ORD-1042?")
    assert result.final_output == "I'll issue a refund today."
    assert trial.trace_id and trial.trace_url == f"http://localhost:3000/traces/{trial.trace_id}"
    Runner.run_sync(agent, "Where is ORD-1042?")                    # after the block: ordinary
    traces = {str(t.id): t for t in sent(evaldecimal)}
    assert traces[trial.trace_id].source_type.value == "evaluation"
    assert [t.source_type.value for i, t in traces.items() if i != trial.trace_id] == ["production"]


def test_evaluate_and_run_sync_share_one_loop(evaldecimal):
    # The OpenAI client the Agents SDK shares binds its connections to the loop that opened
    # them: a script mixing evaluate() and Runner.run_sync() failed with "bound to a different
    # event loop" until both ran on the thread's default loop.
    model = CaseModel()
    agent, _support = _agent(model)
    decimalai.evaluate(agent, CASES[:1], evals=CHECKS)
    Runner.run_sync(agent, "Where is ORD-1042?")
    decimalai.evaluate(agent, CASES[:1], evals=CHECKS, prompt_version=1)
    assert len(model.loops) == 1
