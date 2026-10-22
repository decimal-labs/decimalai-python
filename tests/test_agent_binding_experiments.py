"""Experiments in agent_config(): every run gets its own arm (skills-first Phase 3, SDK 0.25).

The platform decides a run's arm in the routing call the run already makes, keyed on that call's
routing id, and only for a call marked ``experiments=this_run``. Before 0.25 the binding showed a
run the PREVIOUS run's decision and routed its own in the background, so an arm landed one run
late. What these pin, against a fake platform:

- once any answer names a running experiment, every run is routed first, and the call is marked
  ``this_run``; background calls are marked ``capable``; when answers stop naming one, runs go
  back to the background;
- a held-out run's text shows no trace of the skill, and load_skill names no skill while an
  experiment runs (the platform checks the name against the run's own decision);
- a refused load in an experiment says the skill isn't available for this run;
- a prompt split's B run uses the prompt its decision names and records that version, while the
  agent's own prompt (what manifests declare) stays the configured one;
- a conversation that started in an experiment keeps its first decision for every message.
"""

from __future__ import annotations

import contextvars
import hashlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

import decimalai._agent_binding as binding
import decimalai.skill_router as sr
from decimalai.skill_router import SkillRouter

DESCRIPTIONS = {"damaged-item": "Handle items that arrive broken.", "order-support": "Order status and tracking."}
BODIES = {
    "damaged-item": "# Damaged item\n\nAsk for photos of the item and packaging.",
    "order-support": "# Order support\n\nUse lookup_order for order facts.",
}
ROUTES = ("/api/v1/skills/menu", "/api/v1/skills/route")


def _menu(names):
    rows = "".join(f"| {n} | {DESCRIPTIONS[n]} |\n" for n in names)
    return f"## Available Skills\n\n| Skill | Description |\n|-------|-------------|\n{rows}" if names else ""


class Platform:
    """Menu, route and body, plus one running experiment the test sets up."""

    def __init__(self):
        self.skills = ["damaged-item", "order-support"]
        self.experiment = None          # {"id", "kind"} while one runs
        self.held_out = None            # a holdout's skill, withheld from this_run answers
        self.b_prompt = None            # a split's B prompt, sent on this_run answers
        self.marks: list = []           # (path, mark) per routing call
        self.loads: list = []
        self._next = 0

    def request(self, method, path, json=None, params=None, **kwargs):
        if path in ROUTES:
            mark = (params or {}).get("experiments") if method == "GET" else (json or {}).get("experiments")
            self.marks.append((path, mark))
            shown = list(self.skills)
            if self.experiment and mark == "this_run" and self.held_out:
                shown = [n for n in shown if n != self.held_out]
            self._next += 1
            out = {
                "skills": [{"name": n, "description": n} for n in shown],
                "stable_menu": _menu(shown), "menu_instruction": "Use a skill when it applies." if shown else "",
                "routing_hint": "", "stable_menu_skills": shown, "routing_id": f"rt_{self._next:024x}",
                "strategy": "full_menu", "query_ranked": False, "truncated": False,
                "rows_total": len(self.skills), "skill_count": len(shown),
            }
            if self.experiment and mark in ("this_run", "capable"):
                out["experiment"] = dict(self.experiment)
            if self.experiment and mark == "this_run" and self.b_prompt:
                out["prompt"] = dict(self.b_prompt)
            return out
        if path.startswith("/api/v1/skills/") and path.endswith("/body"):
            name = path.split("/")[-2]
            self.loads.append((name, (params or {}).get("routing_id")))
            if self.experiment and name == self.held_out:
                raise sr.SkillRouterError("not found", status_code=404)
            body = BODIES[name]
            return {"body": body, "content_hash": hashlib.sha256(body.encode()).hexdigest(),
                    "skill_id": f"11111111-1111-4111-8111-{len(name):012d}",
                    "version_id": "22222222-2222-4222-8222-222222222222", "content_hash_matches_body": True}
        raise AssertionError(f"unexpected request {method} {path}")


_RUN: contextvars.ContextVar = contextvars.ContextVar("test_binding_experiments_run", default=None)


class Binding(binding.AgentBinding):
    """No framework: a run is one call to `run()`; `conversation=` names a conversation to continue."""

    def __init__(self, prompt, router):
        super().__init__(prompt, fallback=None, on_unavailable="continue")
        self.router = router
        self.records: list = []
        self.conversations: dict = {}

    def _router(self):
        return self.router

    def _run_key(self):
        return _RUN.get()

    def _message(self, context, agent):
        return context.get("message")

    def _values(self, context):
        return None

    def _record(self, **fields):
        self.records.append(fields)
        return True

    def _serve_load(self, name):
        return self.router.load_skill(name, scope=self._run_key(), agent_name=self.name)

    def _conversation(self, context, agent):
        return self.conversations.get(context.get("conversation"))

    def _keep_conversation(self, context, agent, run_key, record):
        if context.get("conversation"):
            self.conversations[context["conversation"]] = record

    def run(self, message, conversation=None):
        _RUN.set(f"run-{uuid4().hex}")
        context = {"message": message, "conversation": conversation}
        self.state = self._prepare_run_sync(context, None, _RUN.get())
        # As every adapter does: a continued conversation keeps its first run's text.
        if self.state["text"] is None:
            self.state["text"] = self._compose(context, None, _RUN.get(), self.state)
        return self.state["text"]


def _prompt(text="You help customers with orders.", version=3):
    return SimpleNamespace(agent_name="support", system_prompt=text, version_number=version,
                           content_hash="c" * 64, is_fallback=False)


@pytest.fixture
def platform(monkeypatch):
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    monkeypatch.setattr(cfg, "_config", DecimalConfig(
        api_key="dai_sk_test", base_url="http://binding-experiments.test", enabled=True))
    monkeypatch.setattr(binding, "_in_background", lambda fn, *args: fn(*args))
    fake = Platform()
    router = SkillRouter(api_key="dai_sk_test", base_url="http://binding-experiments.test", inject_body=False)
    monkeypatch.setattr(router, "_request", fake.request)
    fake.router = router
    return fake


def test_background_calls_are_marked_capable_and_an_experiment_makes_every_run_route_first(platform):
    support = Binding(_prompt(), platform.router)
    support.run("first")                     # no menu yet: routed first, for this run
    support.run("second")                    # the local copy, routed in the background
    assert platform.marks == [(ROUTES[1], "this_run"), (ROUTES[1], "capable")]

    platform.experiment = {"id": "x-1", "kind": "skill_holdout"}
    support.run("third")                     # still the local copy; its background call learns of it
    assert platform.marks[-1] == (ROUTES[1], "capable") and support._experiment == platform.experiment
    platform.marks.clear()
    support.run("fourth")
    support.run("fifth")
    assert platform.marks == [(ROUTES[1], "this_run"), (ROUTES[1], "this_run")], "routed first, nothing behind"
    assert support.records[-1]["routing_id"] is not None, "each run names its own decision"

    platform.experiment = None
    support.run("sixth")                     # this answer names none: back to the background
    platform.marks.clear()
    support.run("seventh")
    assert platform.marks == [(ROUTES[1], "capable")]


def test_a_held_out_run_shows_no_trace_of_the_skill_and_its_tool_names_none(platform):
    platform.experiment = {"id": "x-2", "kind": "skill_holdout"}
    platform.held_out = "damaged-item"
    support = Binding(_prompt(), platform.router)
    text = support.run("My keyboard arrived cracked.")
    assert "damaged-item" not in text and "order-support" in text
    assert support.state["tool_names"] == [] and support.state["experiment"] is True
    refused = support._load_sync("damaged-item", support.state)
    assert refused.startswith("load_skill: 'damaged-item' isn't available for this run.")
    assert platform.loads[-1] == ("damaged-item", support.records[-1]["routing_id"])


def test_a_split_runs_the_prompt_its_decision_names_and_keeps_the_agents_own(platform):
    platform.experiment = {"id": "x-3", "kind": "prompt_split"}
    platform.b_prompt = {"version": 1, "content": "You are prompt B.", "content_hash": "b" * 64}
    support = Binding(_prompt(), platform.router)
    text = support.run("Where is my order?")
    assert text.startswith("You are prompt B.\n\n## Available Skills")
    assert support.records[-1]["prompt"]["version"] == 1
    assert support.records[-1]["prompt"]["content_hash"] == "b" * 64
    # What manifests declare: the agent's configured prompt, not the run's.
    assert support._template() == "You help customers with orders." and support.prompt_version == 3

    platform.b_prompt = None                 # an A decision
    assert support.run("Again?").startswith("You help customers with orders.")
    assert support.records[-1]["prompt"]["version"] == 3


def test_a_conversation_in_an_experiment_keeps_its_first_decision(platform):
    platform.experiment = {"id": "x-4", "kind": "skill_holdout"}
    support = Binding(_prompt(), platform.router)
    support.run("first")                      # learns of the experiment
    platform.marks.clear()
    support.run("Hello", conversation="c-1")
    first_decision = support.records[-1]["routing_id"]
    assert support.conversations["c-1"].experiment is True
    support.run("And another thing", conversation="c-1")
    assert platform.marks == [(ROUTES[1], "this_run")], "a later message is not routed again"
    assert support.state["routing_id"] == first_decision
    # The record survives the framework's serialization.
    restored = binding.ConversationRecord.from_state(support.conversations["c-1"].to_state("support"), "support")
    assert restored.experiment is True


def test_an_unmarked_router_call_sends_no_mark(platform):
    platform.router.get_menu(agent_name="support")
    platform.router.smart_route("cracked", agent_name="support")
    assert platform.marks == [(ROUTES[0], None), (ROUTES[1], None)]
