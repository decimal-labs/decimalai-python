"""Live-LLM: ``decimalai.langchain.agent_config()`` on ``create_agent``, against Claude.

The unit tests (``tests/test_langchain_agent_config.py``) drive the middleware with a scripted
chat model. These send real requests to Claude through ``langchain-anthropic`` and check what
only the real API can answer:

1. One message (``invoke``): the model calls ``load_skill`` for the skill that obviously fits
   and answers from the text it got back; the run's trace says so.
2. The same with ``ainvoke``.
3. A two-message thread with thinking on, where the saved prompt changes between the
   messages. The thread keeps its first system text and ``load_skill`` definition (in its
   checkpointed state), so the second message succeeds. The control in the same test proves
   the hazard: the thread's own history and tools, sent with the system text a fresh run
   composes, is refused with HTTP 400, because Anthropic ties a thinking block's signature
   to the conversation's system prompt.

The one-message run on OpenAI, against a backend and with the stored trace checked, is a cell
of ``test_framework_live_binding_skill_loading.py``, with the other bindings.

WHERE DECIMALAI'S ANSWERS COME FROM
-----------------------------------
The in-process fake platform the unit tests use (``FakePlatform`` from
``tests/test_agent_binding.py``), as in ``test_framework_live_anthropic_load_agent.py`` and for
the same reasons: case 3 changes the saved prompt between two messages without a dashboard,
and the subject is how Claude treats what the middleware sends. So it needs no backend: it
overrides the directory's ``_require_gates`` and ``_reset_sdk`` fixtures below.

WHAT MAKES CASES 1 AND 2 NON-CIRCULAR
-------------------------------------
The skill's text carries a claim code minted for this session (``DMG-`` and eight random hex
digits). It is never in the system text; the only way to it is the tool result that answers
``load_skill``. Each case checks that result on the wire and that the model's answer quotes
the code.

THE MODEL AND THE BUDGET
------------------------
Pinned to ``claude-haiku-5-5``, as the Anthropic file is: case 3 needs adaptive thinking.
``ChatAnthropic`` gets ``anthropic`` clients whose middleware counts every HTTP attempt, and
the file refuses to send more than ``REQUEST_BUDGET`` requests in one run.

Marker: live_llm + skill_routing. Needs ``RUN_LIVE_LLM_TESTS=1`` and ``ANTHROPIC_API_KEY``;
skips without them.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from unittest.mock import MagicMock

import pytest

from . import _live_helpers as h

anthropic = pytest.importorskip("anthropic")
pytest.importorskip("langchain_anthropic")
pytest.importorskip("langchain.agents.middleware")

from anthropic import Middleware  # noqa: E402
from langchain.agents import create_agent  # noqa: E402
from langchain_anthropic import ChatAnthropic  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

import decimalai._agent as agent_mod  # noqa: E402
import decimalai._agent_binding as binding  # noqa: E402
import decimalai.langchain as lc  # noqa: E402
import tests.test_agent_binding as fake_platform  # noqa: E402
from decimalai._agent import SavedPrompt  # noqa: E402
from decimalai.skill_router import SkillRouter  # noqa: E402

pytestmark = [pytest.mark.live_llm, pytest.mark.skill_routing]

MODEL = h._guard_no_opus("claude-haiku-5-5")
REQUEST_BUDGET = 25
FAKE_BASE = "http://langchain-live.test"
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
    monkeypatch.setattr(lc, "_skill_router_singleton", router)
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
        entry = {"body": json.loads(json.dumps(request.json, default=str)), "status": None}
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


def _claude(wire: Wire, **kwargs) -> ChatAnthropic:
    """``ChatAnthropic`` on clients that count each request (it builds its own lazily)."""
    llm = ChatAnthropic(model=MODEL, max_tokens=8000, max_retries=2, **kwargs)
    llm.__dict__["_client"] = anthropic.Anthropic(middleware=[wire], timeout=TIMEOUT_S, max_retries=2)
    llm.__dict__["_async_client"] = anthropic.AsyncAnthropic(
        middleware=[wire], timeout=TIMEOUT_S, max_retries=2)
    return llm


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order's status.

    Args:
        order_id: The order id, for example ORD-1001.
    """
    return json.dumps({"order_id": order_id, "status": "delivered"})


# ── reading what was sent ───────────────────────────────────────────────────


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


def _answer(messages: list) -> str:
    """The run's final answer: the text of its last AI message."""
    final = next(m for m in reversed(messages) if isinstance(m, AIMessage))
    return final.text if isinstance(final.text, str) else str(final.text)


def _assert_the_skill_text_reached_the_model(sent: list, answer: str) -> None:
    assert sent and all(CODE not in json.dumps(entry["body"].get("system")) for entry in sent), \
        "the code must reach the model only through load_skill"
    served = [text for name, text in _loads(sent).values() if name == "damaged-item"]
    assert served, f"the model never loaded damaged-item; loads: {_loads(sent)!r}"
    assert served[0].startswith("## Skill: damaged-item") and CODE in served[0], served[0][:300]
    assert CODE in answer, f"the answer does not quote the code from the skill text: {answer!r}"


# ── 1. one message ──────────────────────────────────────────────────────────


def test_one_message_loads_the_matching_skill_and_answers_from_it(platform, wire):
    support = lc.agent_config("support")
    agent = create_agent(_claude(wire), tools=[lookup_order], middleware=[support.middleware])
    handler = lc.CallbackHandler(auto_send=False)
    result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]},
                          {"callbacks": [handler]}, context=SAM)
    print(f"\n[invoke] {len(wire.sent)} requests")
    _assert_the_skill_text_reached_the_model(wire.sent, _answer(result["messages"]))
    assert _text(wire.sent[0]["body"]["system"]).startswith(
        "You are Acme's customer support agent. The customer is Sam.")

    trace = handler.get_trace()
    assert trace.agent_name == "support"
    assert trace.skills_loaded_by_agent == ["damaged-item"]
    assert "damaged-item" in trace.skills_offered_in_prompt


# ── 2. ainvoke ──────────────────────────────────────────────────────────────


def test_ainvoke_one_message_loads_the_matching_skill(platform, wire):
    support = lc.agent_config("support")
    agent = create_agent(_claude(wire), tools=[lookup_order], middleware=[support.middleware])
    result = asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": QUESTION}]},
                                       context=SAM))
    print(f"\n[ainvoke] {len(wire.sent)} requests")
    _assert_the_skill_text_reached_the_model(wire.sent, _answer(result["messages"]))


# ── 3. a thread with thinking on, across a prompt change ────────────────────

FIRST = ("My keyboard was delivered on 3 October and arrived cracked. Today is 20 October. "
         "Am I still inside the claim window, and what claim code do I quote?")
SECOND = "Thanks. Repeat the claim code you gave me, and nothing else."
CHANGED = (
    "You are Acme's customer support agent (version 5). The customer is {{customer.name}}.\n"
    "Be brief. Answer using the procedures available to you."
)


def _thought(messages: list) -> bool:
    return any(isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking")
               for message in messages if isinstance(message, AIMessage)
               for block in (message.content if isinstance(message.content, list) else []))


def test_a_thread_keeps_its_system_when_the_prompt_changes_with_thinking_on(platform, wire):
    """Anthropic refuses a message whose system prompt differs from the one a thinking block in
    the history was made under. A thread keeps its first message's system text, so a prompt
    saved mid-thread cannot break it; a system text composed afresh per message would."""
    support = lc.agent_config("support")
    llm = _claude(wire, thinking={"type": "adaptive"}, effort="high")
    agent = create_agent(llm, tools=[lookup_order], middleware=[support.middleware],
                         checkpointer=InMemorySaver())

    # Turn 1. Thinking is adaptive, so a turn can come back without a thinking block, and
    # then there is no hazard to test: retry on a new thread.
    for attempt in range(3):
        thread = {"configurable": {"thread_id": f"thread-{uuid.uuid4().hex[:8]}"}}
        start = len(wire.sent)
        agent.invoke({"messages": [{"role": "user", "content": FIRST}]}, thread, context=SAM)
        history = agent.get_state(thread).values["messages"]
        if _thought(history):
            break
    turn_one = wire.sent[start:]
    print(f"\n[thread] turn 1 took {attempt + 1} attempt(s), {len(turn_one)} requests")
    assert _thought(history), "three tries of turn 1 came back without a thinking block"
    assert "damaged-item" in [name for name, _ in _loads(turn_one).values()], "turn 1 loaded the skill"

    # The saved prompt changes and this process reads it.
    platform.saved = SavedPrompt(agent_name="support", system_prompt=CHANGED, version_number=5,
                                 content_hash="d" * 64)
    support._refresh_prompt()
    assert support.prompt_version == 5

    # The control: the thread's history and tools as sent, under a system text composed afresh.
    fresh = support.system_message([HumanMessage(SECOND)], context=SAM)
    assert fresh.content.startswith("You are Acme's customer support agent (version 5)")
    start = len(wire.sent)
    with pytest.raises(anthropic.BadRequestError) as refused:
        llm.bind_tools(turn_one[0]["body"]["tools"]).invoke([fresh, *history, HumanMessage(SECOND)])
    control = wire.sent[start:]
    assert [entry["status"] for entry in control] == [400]
    assert control[0]["body"]["tools"] == turn_one[0]["body"]["tools"], \
        "only the system differs, so it is what the API refuses"
    assert "thinking" in str(refused.value).lower(), str(refused.value)

    # Turn 2 in the thread: the system text it started with, so the API accepts it.
    start = len(wire.sent)
    result = agent.invoke({"messages": [{"role": "user", "content": SECOND}]}, thread, context=SAM)
    turn_two = wire.sent[start:]
    print(f"[thread] turn 2: {len(turn_two)} requests, statuses {[e['status'] for e in turn_two]}")
    assert turn_two and all(entry["status"] == 200 for entry in turn_two)
    assert turn_two[0]["body"]["system"] == turn_one[0]["body"]["system"], "byte for byte on the wire"
    assert turn_two[0]["body"]["tools"] == turn_one[0]["body"]["tools"]
    assert CODE in _answer(result["messages"])
