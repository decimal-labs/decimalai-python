"""The framework-neutral half of every ``agent_config()``: ``decimalai._agent_binding``.

The OpenAI Agents SDK tests exercise it through ``OpenAIAgentConfig`` on the
async path. These run it through a binding with no framework at all, on the
synchronous path (``_prepare_run_sync`` / ``_load_sync``) that a caller with no
event loop uses. Nothing here needs a framework installed, so it runs in every
test environment.
"""

from __future__ import annotations

import asyncio
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
    assert support.records[0]["prompt"]["source"] == "fallback (none saved)"


@pytest.mark.parametrize("text, version, is_fallback, source", [
    ("Saved.", 3, False, "decimalai"),
    ("Saved.", 3, True, "decimalai (cached copy)"),
    ("Be brief.", None, True, "fallback (unreachable)"),
    (None, None, False, "fallback (none saved)"),
])
def test_prompt_source_says_where_the_prompt_came_from(text, version, is_fallback, source):
    """Each value says what happened, not where the text was typed."""
    read = SimpleNamespace(agent_name="support", system_prompt=text, version_number=version,
                           content_hash=None, is_fallback=is_fallback)
    assert Binding(read, None, fallback="Be brief.").prompt_source == source


# ── conversations ───────────────────────────────────────────


class ChatBinding(Binding):
    """A binding whose runs name their chat: a chat's later runs reuse its first run's record."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chats: dict = {}

    def _conversation(self, context, agent):
        return self.chats.get(context.get("chat"))

    def _keep_conversation(self, context, agent, run_key, record):
        self.chats.setdefault(context.get("chat"), record)

    def chat_run(self, chat, message, values=None):
        _RUN.set(f"run-{uuid4().hex}")
        context = {"message": message, "values": values, "chat": chat}
        state = self._prepare_run_sync(context, None, _RUN.get())
        if state["text"] is None:
            state["text"] = self._compose(context, None, _RUN.get(), state)
        return state


def test_a_conversation_keeps_its_first_runs_text_and_tool_after_the_prompt_and_list_change(platform):
    support = ChatBinding(prompt(), platform.router)
    first = support.chat_run("a", "My keyboard arrived cracked.")
    support._prompt = prompt("You help customers with orders. Be brief.", version=4)  # saved, refreshed
    platform.skills = ["damaged-item"]                                                  # a skill unassigned
    support._route_run(None, [], "anything")                                            # the copy follows

    later = support.chat_run("a", "Where is my order?")
    assert later["text"] == first["text"], "byte for byte: Claude rejects a changed system prompt"
    assert later["tool_names"] == first["tool_names"] == ["damaged-item", "order-support"]
    assert later["tool"] is True
    assert support.records[-1]["prompt"]["version"] == 3, "the record names the version the model saw"

    other = support.chat_run("b", "Where is my order?")
    assert other["text"].startswith("You help customers with orders. Be brief.")
    assert other["tool_names"] == ["damaged-item"], "a new conversation gets the new prompt and list"


def test_a_later_run_of_a_whole_list_is_routed_on_its_own_message_for_the_record(platform):
    support = ChatBinding(prompt(), platform.router)
    support.chat_run("a", "My keyboard arrived cracked.")
    support.chat_run("a", "Where is my order?")
    routed = [r["routing_id"] for r in support.records if r.get("routing_id")]
    assert [platform.queries[r] for r in routed][-1] == "Where is my order?"


def test_a_later_run_of_a_cut_list_names_the_decision_its_list_came_from(platform):
    platform.truncated = True
    support = ChatBinding(prompt(), platform.router)
    first = support.chat_run("a", "My keyboard arrived cracked.")
    calls = len(platform.menu_calls())
    later = support.chat_run("a", "Where is my order?")
    assert later["text"] == first["text"] and "| order-support |" not in later["text"]
    assert len(platform.menu_calls()) == calls, "not routed again: its list would offer other skills"
    assert later["tool_names"] == [] and later["cut"] is True
    assert support.records[-1]["routing_id"] == first["routing_id"]
    assert platform.router._run_routing_ids[_RUN.get()] == first["routing_id"], "its loads name that decision"
    assert support._load_sync("damaged-item").startswith("## Skill: damaged-item")


def test_the_kill_switch_leaves_an_open_conversation_its_text_and_tool(platform, monkeypatch):
    import decimalai._config as cfg

    support = ChatBinding(prompt(), platform.router)
    first = support.chat_run("a", "My keyboard arrived cracked.")
    monkeypatch.setattr(cfg, "_disabled_by_init", lambda: True)
    calls = len(platform.menu_calls())
    later = support.chat_run("a", "Where is my order?")
    assert (later["text"], later["tool"]) == (first["text"], True)
    assert len(platform.menu_calls()) == calls
    assert support.chat_run("b", "Hi")["text"] == "You help customers with orders.", "a new one gets no skills"


def test_one_runs_tool_names_hold_while_a_decision_lands_mid_run(platform):
    support = ChatBinding(prompt(), platform.router)
    state = support.chat_run(None, "My keyboard arrived cracked.")
    platform.skills = ["damaged-item"]
    support._route_run(None, [], "anything")
    assert support._run_state(_RUN.get())["tool_names"] == ["damaged-item", "order-support"]
    assert state["conversation"].tool_names == ["damaged-item", "order-support"]


def test_a_record_survives_json_and_refuses_another_agents_state(platform):
    import json

    support = ChatBinding(prompt(), platform.router)
    record = support.chat_run("a", "My keyboard arrived cracked.")["conversation"]
    state = json.loads(json.dumps(record.to_state("support")))
    assert binding.ConversationRecord.from_state(state, "support") == record
    assert binding.ConversationRecord.from_state(state, "billing") is None
    assert binding.ConversationRecord.from_state({**state, "version": 99}, "support") is None


@pytest.mark.parametrize("name, claude", [
    ("claude-haiku-5-5", True), ("anthropic/claude-sonnet-5-5", True), ("anthropic.claude-opus-5-5", True),
    ("claude-opus-4-8@20260101", True), ("gpt-5.4-mini", False), ("gemini-3.5-flash", False), (None, False),
])
def test_is_claude_reads_every_providers_spelling(name, claude):
    assert binding.is_claude(name) is claude


def _start_async_run(support, message):
    """An async run of the neutral binding: (run key, its state), on the caller's event loop."""
    key = f"run-{uuid4().hex}"
    _RUN.set(key)
    context = {"message": message, "values": None}

    async def prepare():
        state = await support._prepare_run(context, None, key)
        support._compose(context, None, key, state)
        return state

    return key, prepare


def test_an_async_load_that_stops_waiting_leaves_the_runs_decision_running(platform, monkeypatch):
    """`asyncio.wait_for(wrap_future(...))` cancelled a decision still queued in the pool when the
    load stopped waiting for it, and the run's record never got it. Every binding's async load
    (OpenAI Agents, Pydantic AI, Google ADK) waits this way."""
    import concurrent.futures
    import threading

    support = Binding(prompt(), platform.router)
    support.run("My keyboard arrived cracked.")  # the first run routes; later ones in the background
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    gate = threading.Event()
    pool.submit(gate.wait, 5)  # the pool's one worker is busy: the decision queues
    monkeypatch.setattr(binding, "_in_background",
                        lambda fn, *args: pool.submit(contextvars.copy_context().run, fn, *args))
    monkeypatch.setattr(binding, "_LOAD_WAITS_FOR_ROUTING_S", 0.2)

    async def go():
        _key, prepare = _start_async_run(support, "Where is my order?")
        state = await prepare()
        await support._load("order-support")
        return state["pending"]

    pending = asyncio.run(go())
    assert not pending.cancelled(), "the decision is still on its way"
    gate.set()
    pending.result(timeout=5)
    pool.shutdown()


def test_without_an_asyncio_loop_anyio_carries_an_async_run(platform, monkeypatch):
    """An async client also runs under trio, where there is no asyncio loop to hand the routing
    to; anyio's worker thread takes it there. Exercised on anyio's asyncio backend with the
    asyncio branch switched off and asyncio's thread hop failing as it does under trio."""
    monkeypatch.setattr(binding, "_running_asyncio_loop", lambda: None)

    async def no_asyncio_loop(*args, **kwargs):
        raise RuntimeError("no running event loop")

    monkeypatch.setattr(asyncio, "to_thread", no_asyncio_loop)
    support = Binding(prompt(), platform.router)

    async def go():
        _key, prepare = _start_async_run(support, "My keyboard arrived cracked.")
        await prepare()
        return await support._load("damaged-item")

    body = asyncio.run(go())
    assert body.startswith("## Skill: damaged-item"), "the run was routed and its skill served"
    assert platform.queries[support.records[0]["routing_id"]] == "My keyboard arrived cracked."


def test_an_async_load_waiting_for_its_decision_holds_no_thread(platform, monkeypatch):
    """The wait is on the event loop: the loop's default executor stays free for the app's own
    work (DNS lookups, other loads) while a load waits up to 2 s for its decision."""
    import concurrent.futures
    import threading
    import time

    support = Binding(prompt(), platform.router)
    support.run("My keyboard arrived cracked.")
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    gate = threading.Event()
    pool.submit(gate.wait, 5)
    monkeypatch.setattr(binding, "_in_background",
                        lambda fn, *args: pool.submit(contextvars.copy_context().run, fn, *args))
    monkeypatch.setattr(binding, "_LOAD_WAITS_FOR_ROUTING_S", 0.6)

    async def go():
        asyncio.get_running_loop().set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1))
        _key, prepare = _start_async_run(support, "Where is my order?")
        await prepare()
        load = asyncio.ensure_future(support._load("order-support"))
        await asyncio.sleep(0.1)  # the load is waiting now
        started = time.monotonic()
        await asyncio.to_thread(lambda: None)  # the app's own thread work
        took = time.monotonic() - started
        await load
        return took

    took = asyncio.run(go())
    gate.set()
    pool.shutdown()
    assert took < 0.3, f"the app's thread work waited {took:.2f}s behind the load"


def test_without_an_asyncio_loop_a_cancelled_run_does_not_wait_for_its_thread(platform, monkeypatch):
    """Under trio, anyio's thread hop shields the call from cancellation unless told otherwise:
    a cancelled run (a client that hung up) then waits for its routing thread. It is abandoned
    on cancel, as asyncio's is. (On anyio's asyncio backend, which this suite runs, a native
    cancel interrupts either way, so the hop's arguments are what can be checked here.)"""
    import anyio.to_thread

    monkeypatch.setattr(binding, "_running_asyncio_loop", lambda: None)
    seen = []
    real_run_sync = anyio.to_thread.run_sync

    async def run_sync(func, *args, **kwargs):
        seen.append(kwargs)
        return await real_run_sync(func, *args, **kwargs)

    monkeypatch.setattr(anyio.to_thread, "run_sync", run_sync)
    support = Binding(prompt(), platform.router)

    async def go():
        _key, prepare = _start_async_run(support, "My keyboard arrived cracked.")
        await prepare()

    asyncio.run(go())
    assert seen and all(kw.get("abandon_on_cancel") or kw.get("cancellable") for kw in seen)


def test_without_an_asyncio_loop_a_call_that_raises_typeerror_runs_once(monkeypatch):
    """The anyio hop once retried with anyio < 4.1's keyword on any TypeError, including
    one the routed call raised itself: a second routing request for that run."""
    calls = []

    def route():
        calls.append(1)
        raise TypeError("'int' object is not iterable")

    monkeypatch.setattr(binding, "_running_asyncio_loop", lambda: None)

    async def go():
        with pytest.raises(TypeError):
            await binding._in_thread(route)

    asyncio.run(go())
    assert calls == [1]
