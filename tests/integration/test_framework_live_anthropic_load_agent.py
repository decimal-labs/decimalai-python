"""Live-LLM: ``decimalai.anthropic.agent_config()`` against the real Anthropic API.

The unit tests (``tests/test_anthropic_load_agent.py``) drive the same code
through a stub of the Messages API. These send real requests to Claude and check
what only the real API can answer:

1. The Tool Runner, one message: the model calls ``load_skill`` for the skill
   that obviously fits and answers from the text it got back.
2. The same with a hand-written ``client.messages.create`` loop
   (``run.tool_params`` and ``run.tool_result``).
3. A two-message conversation with thinking on, where the saved prompt changes
   between the messages. The conversation keeps its first ``system``, so the
   second message succeeds. The control in the same test proves the hazard:
   the second message sent with a ``system`` rebuilt from a fresh
   ``support.run()`` is refused with HTTP 400, because Anthropic ties a thinking
   block's signature to the conversation's system prompt.
4. Prompt caching: two one-message runs share the cached system prompt.
5. ``AsyncAnthropic`` with the async Tool Runner, one message.

WHERE DECIMALAI'S ANSWERS COME FROM
-----------------------------------
The in-process fake platform the unit tests use (``FakePlatform`` from
``tests/test_agent_binding.py``), not a backend. The one other ``agent_config()``
live test, ADK's skill delivery, uses a real local backend; here the subject is
how Anthropic's API treats what ``agent_config()`` builds, the fake lets case 3
change the saved prompt between two messages without a dashboard, and no local
backend was running when this file was written. So it needs no backend: it
overrides the directory's ``_require_gates`` and ``_reset_sdk`` fixtures below.

WHAT MAKES CASES 1, 2 AND 5 NON-CIRCULAR
----------------------------------------
The skill's text carries a claim code minted for this session (``DMG-`` and
eight random hex digits). It is never in ``system``; the only way to it is the
``tool_result`` that answers ``load_skill``. Each case checks that result on the
wire and that the model's answer quotes the code.

THE MODEL AND THE BUDGET
------------------------
Pinned to ``claude-haiku-5-5`` (the owner's budget tier), not the lane's
``LIVE_LLM_ANTHROPIC_MODEL``: case 3 needs adaptive thinking and case 4 the
512-token caching minimum, and both are Haiku 5.5 facts. Every HTTP attempt is
counted by a client middleware, and the file refuses to send more than
``REQUEST_BUDGET`` requests in one run (a full run took 12 on 2026-10-20).

Marker: live_llm + skill_routing. Needs ``RUN_LIVE_LLM_TESTS=1`` and
``ANTHROPIC_API_KEY``; skips without them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from unittest.mock import MagicMock

import pytest

from . import _live_helpers as h

anthropic = pytest.importorskip("anthropic")

from anthropic import Middleware, beta_async_tool, beta_tool  # noqa: E402

import decimalai  # noqa: E402
import decimalai._agent as agent_mod  # noqa: E402
import decimalai._agent_binding as binding  # noqa: E402
import decimalai.anthropic as da  # noqa: E402
import tests.test_agent_binding as fake_platform  # noqa: E402
from decimalai._agent import SavedPrompt  # noqa: E402
from decimalai.skill_router import SkillRouter  # noqa: E402

pytestmark = [pytest.mark.live_llm, pytest.mark.skill_routing]

MODEL = h._guard_no_opus("claude-haiku-5-5")
REQUEST_BUDGET = 25
FAKE_BASE = "http://anthropic-live.test"
SAM = {"customer": {"name": "Sam"}}

#: Minted per session, so the model can only know it from the skill's text.
TOKEN = uuid.uuid4().hex[:8].upper()
CODE = f"DMG-{TOKEN}"

SKILL_TEXT = f"""# Damaged item

Claims for an item that arrives damaged are accepted within 30 days of delivery.
The claim code to quote on a damaged-item claim is {CODE}. Quote it exactly.
Ask for photos of the item and its packaging.
"""

PROMPT = (
    "You are Acme's customer support agent. The customer is {{customer.name}}.\n"
    "Answer using the procedures available to you. If you do not know something, say so."
)

QUESTION = "My keyboard arrived cracked. What claim code do I quote? Reply with the code only."

#: An ``anthropic.Anthropic`` per test; its timeout keeps a hung call from hanging the run.
TIMEOUT_S = 120.0

#: Requests sent by every test in this run, retries included.
_SPENT = [0]


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _require_gates():
    """This file's DecimalAI is in-process, so only the live switch and the key gate it."""
    if not h.LIVE_TESTS_ENABLED:
        pytest.skip("Set RUN_LIVE_LLM_TESTS=1 to run live LLM tests.")
    h.require_key_for("anthropic")


@pytest.fixture(autouse=True)
def _reset_sdk():
    """No backend to initialize against: ``platform`` sets the SDK's config."""
    yield


@pytest.fixture(scope="module", autouse=True)
def _report_spend():
    yield
    print(f"\nAnthropic requests sent by {__name__.rsplit('.', 1)[-1]}: {_SPENT[0]} "
          f"(budget {REQUEST_BUDGET})")


@pytest.fixture
def platform(monkeypatch):
    """DecimalAI as the unit tests fake it: two skills, a saved prompt, traces sent nowhere."""
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    monkeypatch.setattr(cfg, "_config", DecimalConfig(api_key="dai_sk_test", base_url=FAKE_BASE, enabled=True))
    monkeypatch.setattr(cfg, "_client", MagicMock())
    monkeypatch.setattr(binding, "_in_background", lambda fn, *args: fn(*args))
    monkeypatch.setitem(fake_platform.BODIES, "damaged-item", SKILL_TEXT)
    fake = fake_platform.FakePlatform()
    router = SkillRouter(api_key="dai_sk_test", base_url=FAKE_BASE, inject_body=False)
    monkeypatch.setattr(router, "_request", fake.request)
    monkeypatch.setattr(da, "_skill_router_singleton", router)
    fake.saved = SavedPrompt(agent_name="support", system_prompt=PROMPT, version_number=4,
                             content_hash="c" * 64)
    monkeypatch.setattr(agent_mod, "read_prompt", lambda name, **kwargs: fake.saved)
    return fake


class Wire(Middleware):
    """Every HTTP attempt a test's clients make: the request body and the status that came back."""

    def __init__(self) -> None:
        self.sent: list = []

    def _start(self, request) -> dict:
        if _SPENT[0] >= REQUEST_BUDGET:
            raise RuntimeError(f"this file may send at most {REQUEST_BUDGET} Anthropic requests per run")
        _SPENT[0] += 1
        entry = {"body": _plain(request.json), "status": None}
        self.sent.append(entry)
        return entry

    def handle(self, request, call_next):
        entry = self._start(request)
        response = call_next(request)
        entry["status"] = response.status_code
        return response

    async def handle_async(self, request, call_next):
        entry = self._start(request)
        response = await call_next(request)
        entry["status"] = response.status_code
        return response


@pytest.fixture
def wire():
    return Wire()


def _client(wire: Wire):
    return anthropic.Anthropic(middleware=[wire], timeout=TIMEOUT_S)


# ── reading what was sent ───────────────────────────────────────────────────


def _plain(body):
    """A request body as plain JSON data (SDK blocks become dicts)."""
    def dump(value):
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json", exclude_none=True)
        return str(value)

    return json.loads(json.dumps(body, default=dump))


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [] if isinstance(part, dict))


def _loads(sent: list) -> dict:
    """{tool_use id: (skill asked for, the tool_result text)} for each load_skill call on the wire."""
    asked, loads = {}, {}
    for entry in sent:
        for message in entry["body"].get("messages", []):
            content = message.get("content")
            for block in content if isinstance(content, list) else []:
                if block.get("type") == "tool_use" and block.get("name") == "load_skill":
                    asked[block["id"]] = (block.get("input") or {}).get("name")
                elif block.get("type") == "tool_result" and block.get("tool_use_id") in asked:
                    loads[block["tool_use_id"]] = (asked[block["tool_use_id"]], _text(block.get("content")))
    return loads


def _answer(message) -> str:
    return "".join(block.text for block in message.content if block.type == "text")


def _assert_the_skill_text_reached_the_model(sent: list, answer: str) -> None:
    assert sent and all(CODE not in json.dumps(entry["body"].get("system")) for entry in sent), \
        "the code must reach the model only through load_skill"
    served = [text for name, text in _loads(sent).values() if name == "damaged-item"]
    assert served, f"the model never loaded damaged-item; loads: {_loads(sent)!r}"
    assert served[0].startswith("## Skill: damaged-item") and CODE in served[0], served[0][:300]
    assert CODE in answer, f"the answer does not quote the code from the skill text: {answer!r}"


@beta_tool
def lookup_order(order_id: str) -> str:
    """Look up an order's status.

    Args:
        order_id: The order id, for example ORD-1001.
    """
    return json.dumps({"order_id": order_id, "status": "delivered"})


@beta_async_tool
async def lookup_order_async(order_id: str) -> str:
    """Look up an order's status.

    Args:
        order_id: The order id, for example ORD-1001.
    """
    return json.dumps({"order_id": order_id, "status": "delivered"})


# ── 1. the Tool Runner, one message ────────────────────────────────────────


def test_tool_runner_one_message_loads_the_matching_skill_and_answers_from_it(platform, wire, caplog):
    traces = []
    support = da.agent_config("support", on_trace=traces.append)
    client = _client(wire)
    with caplog.at_level(logging.WARNING, logger="decimalai.anthropic"):
        with support.run(QUESTION, variables=SAM) as run:
            final = client.beta.messages.tool_runner(
                model=MODEL, max_tokens=4096, system=run.system,
                tools=[lookup_order, *run.tools],
                messages=[{"role": "user", "content": QUESTION}],
            ).until_done()
    print(f"\n[tool runner] {len(wire.sent)} requests")
    assert final.stop_reason == "end_turn"
    _assert_the_skill_text_reached_the_model(wire.sent, _answer(final))
    first = wire.sent[0]["body"]
    assert [tool["name"] for tool in first["tools"]] == ["lookup_order", "load_skill"]
    assert not [r for r in caplog.records if "no Claude request was traced" in r.getMessage()]

    decimalai.flush()
    (trace,) = traces
    assert trace.agent_name == "support"
    assert trace.skills_loaded_by_agent == ["damaged-item"] and trace.routing_id
    assert len(trace.llm_calls) >= 2, "each of the Tool Runner's requests is traced inside the run"


# ── 2. a hand-written messages.create loop ─────────────────────────────────


def test_hand_written_create_loop_loads_the_matching_skill_and_answers_from_it(platform, wire):
    support = da.agent_config("support")
    client = _client(wire)
    messages = [{"role": "user", "content": QUESTION}]
    with support.run(QUESTION, variables=SAM) as run:
        for _ in range(6):
            response = client.messages.create(
                model=MODEL, max_tokens=4096, system=run.system,
                tools=[lookup_order.to_dict(), *run.tool_params], messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                break
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = run.tool_result(block)
                if result is None:  # the caller's own tool
                    result = {"type": "tool_result", "tool_use_id": block.id,
                              "content": lookup_order.call(block.input)}
                results.append(result)
            messages.append({"role": "user", "content": results})
    print(f"\n[messages.create loop] {len(wire.sent)} requests")
    assert response.stop_reason == "end_turn"
    _assert_the_skill_text_reached_the_model(wire.sent, _answer(response))


# ── 3. a conversation with thinking on, across a prompt change ─────────────

THINKING = {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}

FIRST = ("My keyboard was delivered on 3 October and arrived cracked. Today is 20 October. "
         "Am I still inside the claim window, and what claim code do I quote?")
SECOND = "Thanks. Repeat the claim code you gave me, and nothing else."
CHANGED = (
    "You are Acme's customer support agent (version 5). The customer is {{customer.name}}.\n"
    "Be brief. Answer using the procedures available to you."
)


def _turn(client, run, history: list) -> None:
    """One message through the Tool Runner, copying every turn into ``history``.

    ``until_done()`` returns only the last message, and the next message needs
    them all, thinking blocks included.
    """
    runner = client.beta.messages.tool_runner(
        model=MODEL, max_tokens=8000, system=run.system, tools=[lookup_order, *run.tools],
        messages=list(history), **THINKING)
    for message in runner:
        history.append({"role": "assistant", "content": message.content})
        response = runner.generate_tool_call_response()
        if response is not None:
            history.append(response)


def _thought(history: list) -> bool:
    return any(getattr(block, "type", None) in ("thinking", "redacted_thinking")
               for message in history if message["role"] == "assistant"
               for block in message["content"])


def test_conversation_keeps_its_system_when_the_prompt_changes_with_thinking_on(platform, wire):
    """Anthropic refuses a message whose ``system`` differs from the one a thinking block in
    the history was made under. A conversation keeps its first message's ``system``, so a
    prompt saved mid-chat cannot break it; a fresh ``support.run()`` per message would."""
    support = da.agent_config("support")
    client = _client(wire)

    # Turn 1. Thinking is adaptive, so a turn can come back without a thinking
    # block, and then there is no hazard to test: retry with a new conversation.
    for attempt in range(3):
        chat = support.conversation(variables=SAM)
        history = [{"role": "user", "content": FIRST}]
        start = len(wire.sent)
        with chat.run(FIRST) as run:
            system, tools = run.system, run.tool_params
            _turn(client, run, history)
        if _thought(history):
            break
    print(f"\n[conversation] turn 1 took {attempt + 1} attempt(s), {len(wire.sent) - start} requests")
    assert _thought(history), "three tries of turn 1 came back without a thinking block"
    turn_one = wire.sent[start:]
    assert "damaged-item" in [name for name, _ in _loads(turn_one).values()], "turn 1 loaded the skill"

    # The saved prompt changes and this process reads it.
    platform.saved = SavedPrompt(agent_name="support", system_prompt=CHANGED, version_number=5,
                                 content_hash="d" * 64)
    support._refresh_prompt()
    assert support.prompt_version == 5
    history.append({"role": "user", "content": SECOND})

    # The control: the same history under a system rebuilt from a fresh run.
    start = len(wire.sent)
    with pytest.raises(anthropic.BadRequestError) as refused:
        with support.run(SECOND, variables=SAM) as fresh:
            assert fresh.system[0]["text"].startswith("You are Acme's customer support agent (version 5)")
            assert fresh.tool_params == tools, "only the system differs, so it is what the API refuses"
            client.beta.messages.tool_runner(
                model=MODEL, max_tokens=8000, system=fresh.system, tools=[lookup_order, *fresh.tools],
                messages=list(history), **THINKING).until_done()
    assert [entry["status"] for entry in wire.sent[start:]] == [400]
    assert "thinking" in str(refused.value).lower(), str(refused.value)

    # Turn 2 in the conversation: the system it started with, so the API accepts it.
    start = len(wire.sent)
    with chat.run(SECOND) as run:
        assert run.system == system and run.tool_params == tools
        _turn(client, run, history)
    turn_two = wire.sent[start:]
    print(f"[conversation] turn 2: {len(turn_two)} requests, statuses {[e['status'] for e in turn_two]}")
    assert turn_two and all(entry["status"] == 200 for entry in turn_two)
    assert turn_two[0]["body"]["system"] == turn_one[0]["body"]["system"], "byte for byte on the wire"
    assert CODE in _text([block.model_dump() for block in history[-1]["content"]])


# ── 4. prompt caching across two one-message runs ──────────────────────────


def test_prompt_caches_across_two_one_message_runs(platform, wire):
    """No placeholders, so every run's ``system`` is the same and DecimalAI's breakpoint
    (on the skill list, the last block) caches the whole of it. The session id makes the
    first run write the cache instead of reading an entry an earlier run left."""
    rules = "\n".join(
        f"Rule {i}: For order question type {i}, check the order record first, quote only facts "
        f"the tools returned, and never promise a refund amount the policy does not state."
        for i in range(1, 51))
    platform.saved = SavedPrompt(
        agent_name="support", version_number=4, content_hash="e" * 64,
        system_prompt=f"You are Acme's customer support agent (session {TOKEN}).\n\n{rules}")
    support = da.agent_config("support")
    client = _client(wire)
    usage = []
    for question in ("Reply with the single word: ready.", "Reply with the single word: steady."):
        with support.run(question) as run:
            assert "cache_control" in run.system[-1]
            response = client.messages.create(
                model=MODEL, max_tokens=1024, system=run.system, tools=run.tool_params,
                messages=[{"role": "user", "content": question}], output_config={"effort": "low"})
        usage.append(response.usage)
    print(f"\n[caching] {len(wire.sent)} requests; "
          f"write {[u.cache_creation_input_tokens for u in usage]}, read {[u.cache_read_input_tokens for u in usage]}")
    first, second = wire.sent
    assert first["body"]["system"] == second["body"]["system"] and first["body"]["tools"] == second["body"]["tools"]
    assert usage[0].cache_creation_input_tokens, "the first run wrote the system prompt to the cache"
    assert (usage[1].cache_read_input_tokens or 0) > 0, "the second run read it back"


# ── 5. AsyncAnthropic and the async Tool Runner ────────────────────────────


def test_async_tool_runner_one_message_loads_the_matching_skill(platform, wire):
    support = da.agent_config("support")

    async def main():
        async with anthropic.AsyncAnthropic(middleware=[wire], timeout=TIMEOUT_S) as client:
            async with support.run(QUESTION, variables=SAM) as run:
                tools = run.tools
                final = await client.beta.messages.tool_runner(
                    model=MODEL, max_tokens=4096, system=run.system,
                    tools=[lookup_order_async, *tools],
                    messages=[{"role": "user", "content": QUESTION}],
                ).until_done()
        return final, tools

    final, (tool,) = asyncio.run(main())
    print(f"\n[async tool runner] {len(wire.sent)} requests")
    assert type(tool).__name__ == "AsyncLoadSkillTool"
    assert final.stop_reason == "end_turn"
    _assert_the_skill_text_reached_the_model(wire.sent, _answer(final))
