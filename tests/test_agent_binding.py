"""The framework-neutral half of every ``load_agent()``: ``decimalai._agent_binding``.

The OpenAI Agents SDK tests exercise it through ``OpenAIAgentConfig`` on the
async path. These run it through a binding with no framework at all, on the
synchronous path (``_prepare_run_sync`` / ``_load_sync``) that a caller with no
event loop uses. Nothing here needs a framework installed, so it runs in every
test environment.
"""

from __future__ import annotations

import contextvars
import hashlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

import decimalai._agent_binding as binding
import decimalai.skill_router as sr
from decimalai.skill_router import LOAD_SKILL_PROMPT_HINT, SkillRouter

DESCRIPTIONS = {"damaged-item": "Handle items that arrive broken.", "order-support": "Order status and tracking."}
BODIES = {
    "damaged-item": "# Damaged item\n\nAsk for photos of the item and packaging.",
    "order-support": "# Order support\n\nUse lookup_order for order facts.",
}
MENU_INSTRUCTION = "When a skill above applies, follow its instructions and then answer the user directly."


def menu(names):
    rows = "".join(f"| {n} | {DESCRIPTIONS[n]} |\n" for n in names)
    return f"## Available Skills\n\n| Skill | Description |\n|-------|-------------|\n{rows}" if names else ""


class FakePlatform:
    """The router's requests: the agent's menu, a route on a message, a skill's body."""

    def __init__(self):
        self.skills = ["damaged-item", "order-support"]
        self.down = False
        #: A catalog bigger than the menu: which skills make it depends on the message.
        self.truncated = False
        self.ranking = {"cracked": ["damaged-item"], "where": ["order-support"]}
        self.requests: list = []
        self.queries: dict = {}
        self._next = 0

    def request(self, method, path, json=None, params=None, **kwargs):
        self.requests.append((method, path, dict(params or {})))
        if path in ("/api/v1/skills/menu", "/api/v1/skills/route"):
            if self.down:
                raise sr.SkillRouterError("transport error: timed out")
            query = (json or {}).get("query") if path.endswith("/route") else None
            shown = list(self.skills)
            if self.truncated:
                words = (query or "").lower()
                shown = next((names for word, names in self.ranking.items() if word in words), shown[:1])
            self._next += 1
            routing_id = f"rt_{self._next:024x}"
            self.queries[routing_id] = query
            return {
                "skills": [{"name": n, "description": n} for n in shown],
                "stable_menu": menu(shown), "menu_instruction": MENU_INSTRUCTION if shown else "",
                "routing_hint": "", "stable_menu_skills": shown, "routing_id": routing_id,
                "strategy": "full_menu" if not query else "semantic", "query_ranked": bool(query),
                "truncated": self.truncated, "rows_total": len(self.skills), "skill_count": len(shown),
            }
        if path.startswith("/api/v1/skills/") and path.endswith("/body"):
            name = path.split("/")[-2]
            body = BODIES[name]
            return {"body": body, "content_hash": hashlib.sha256(body.encode()).hexdigest(),
                    "skill_id": f"11111111-1111-4111-8111-{len(name):012d}",
                    "version_id": "22222222-2222-4222-8222-222222222222",
                    "content_hash_matches_body": True}
        raise AssertionError(f"unexpected DecimalAI request {method} {path}")

    def menu_calls(self):
        return [r for r in self.requests if r[1] in ("/api/v1/skills/menu", "/api/v1/skills/route")]


_RUN: contextvars.ContextVar = contextvars.ContextVar("test_agent_binding_run", default=None)


class Binding(binding.AgentBinding):
    """A binding with no framework: a run is one call to `run()`."""

    _VALUES_HINT = "Pass them as values=..."

    def __init__(self, prompt, router, *, fallback=None, on_unavailable="continue"):
        super().__init__(prompt, fallback=fallback, on_unavailable=on_unavailable)
        self.router = router
        self.records: list = []
        self.menus: list = []

    def _router(self):
        return self.router

    def _run_key(self):
        return _RUN.get()

    def _message(self, context, agent):
        return context.get("message")

    def _values(self, context):
        return context.get("values")

    def _record(self, **fields):
        self.records.append(fields)
        return True

    def _on_menu(self, names):
        self.menus.append(list(names))

    def _serve_load(self, name):
        return self.router.load_skill(name, scope=self._run_key(), agent_name=self.name)

    def run(self, message, values=None):
        _RUN.set(f"run-{uuid4().hex}")
        context = {"message": message, "values": values}
        state = self._prepare_run_sync(context, None, _RUN.get())
        return self._compose(context, None, _RUN.get(), state)


def prompt(text="You help customers with orders.", version=3):
    return SimpleNamespace(agent_name="support", system_prompt=text, version_number=version,
                           content_hash="c" * 64, is_fallback=False)


@pytest.fixture
def platform(monkeypatch):
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    monkeypatch.setattr(cfg, "_config", DecimalConfig(
        api_key="dai_sk_test", base_url="http://agent-binding.test", enabled=True))
    monkeypatch.setattr(binding, "_in_background", lambda fn, *args: fn(*args))
    fake = FakePlatform()
    router = SkillRouter(api_key="dai_sk_test", base_url="http://agent-binding.test", inject_body=False)
    monkeypatch.setattr(router, "_request", fake.request)
    fake.router = router
    return fake


def test_the_first_run_is_routed_before_it_is_composed_and_later_runs_use_the_local_copy(platform):
    support = Binding(prompt(), platform.router)
    first = support.run("My keyboard arrived cracked.")
    assert first.startswith("You help customers with orders.\n\n## Available Skills")
    assert "| damaged-item |" in first and "| order-support |" in first
    assert first.endswith(LOAD_SKILL_PROMPT_HINT)
    assert platform.queries[support.records[0]["routing_id"]] == "My keyboard arrived cracked."
    assert support.skill_names == ["damaged-item", "order-support"]

    second = support.run("Where is my order?")
    assert second == first, "the local copy: the whole menu, whatever was asked"
    assert support.records[1]["routing_id"] is None
    # ...and the run's own decision, ranked on its message, recorded (here inline).
    assert platform.queries[support.records[2]["routing_id"]] == "Where is my order?"
    assert len(platform.menu_calls()) == 2


def test_a_cut_menu_is_routed_for_each_message_and_the_tool_accepts_any_name(platform):
    platform.truncated = True
    support = Binding(prompt(), platform.router)
    cracked = support.run("My keyboard arrived cracked.")
    where = support.run("Where is my order?")
    assert "| damaged-item |" in cracked and "| order-support |" not in cracked
    assert "| order-support |" in where and "| damaged-item |" not in where
    assert support.menus == [[], []], "a per-message list must not become the tool's name list"


def test_placeholders_are_filled_per_run_and_a_missing_one_stops_the_run(platform):
    support = Binding(prompt("Hello {{customer.name}} ({{customer.tier}})."), platform.router)
    assert support.variables == ["customer.name", "customer.tier"]
    text = support.run("Hi", values={"customer": SimpleNamespace(name="Sam", tier="gold")})
    assert text.startswith("Hello Sam (gold).")
    assert support.records[0]["prompt"]["variables"] == ["customer.name", "customer.tier"]
    with pytest.raises(binding.MissingPromptVariable, match=r"customer\.tier\. Pass them as values=\.\.\."):
        support.run("Hi", values={"customer": {"name": "Sam"}})


def test_skills_that_never_arrived_fail_the_run_only_when_asked(platform):
    platform.down = True
    relaxed = Binding(prompt(), platform.router)
    assert relaxed.run("Hi") == "You help customers with orders."
    assert relaxed.records[0]["prompt"]["skills_unavailable"]
    strict = Binding(prompt(), platform.router, on_unavailable="fail")
    with pytest.raises(RuntimeError, match="unavailable"):
        strict.run("Hi")


def test_a_load_is_served_once_per_run(platform):
    support = Binding(prompt(), platform.router)
    support.run("My keyboard arrived cracked.")
    assert support._load_sync("damaged-item").startswith("## Skill: damaged-item")
    assert "already loaded" in support._load_sync("damaged-item")
    support.run("Again")
    assert support._load_sync("damaged-item").startswith("## Skill: damaged-item"), "a new run loads afresh"


def test_init_enabled_false_stops_routing_and_skills_but_not_the_prompt(platform, monkeypatch):
    import decimalai._config as cfg

    support = Binding(prompt(), platform.router)
    monkeypatch.setattr(cfg, "_disabled_by_init", lambda: True)
    assert support.run("Hi") == "You help customers with orders."
    assert platform.menu_calls() == []


def test_the_fallback_prompt_is_used_when_none_is_saved(platform):
    support = Binding(prompt(text=None, version=None), platform.router, fallback="Be brief.")
    assert support.run("Hi").startswith("Be brief.\n\n## Available Skills")
    assert support.records[0]["prompt"]["source"] == "code"
