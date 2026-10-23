"""Live-LLM: ``decimalai.openai_agents.agent_config()`` on Claude, through LiteLLM.

The unit tests (``tests/test_openai_agents_load_agent.py``) drive the binding with a scripted
model. These send real requests to Claude through the Agents SDK's ``LitellmModel`` and check
what only the real API can answer:

1. One message: the model calls ``load_skill`` for the skill that obviously fits and answers
   from the text it got back.
2. A two-message chat with thinking on, where the saved prompt changes between the messages.
   The chat's runs share a ``RunConfig(group_id=...)``, so the second keeps the first run's
   instructions and ``load_skill`` definition and is accepted. The control in the same test
   proves the hazard: the chat's own history, sent by a run with no conversation to keep (a
   new group id, so its instructions are composed afresh from the new prompt), is refused with
   HTTP 400, because Claude ties a thinking block's signature to the conversation's system
   prompt.

The one-message run on OpenAI, against a backend and with the stored trace checked, is a cell
of ``test_framework_live_binding_skill_loading.py``, with the other bindings.

THE ENVIRONMENT
---------------
``openai-agents`` needs ``openai>=3`` and LiteLLM's current releases pin ``openai<3``, so neither
of the SDK's two test environments holds both: today ``openai-agents[litellm]`` installs LiteLLM
1.83.0, the one release that fits beside openai 3. The ``openai-agents-litellm-tests`` extra
names that set. Run this file in a venv built from it::

    uv venv -p 3.12 .venv-agents-litellm
    uv pip install -p .venv-agents-litellm/bin/python -e ".[openai-agents-litellm-tests]"
    RUN_LIVE_LLM_TESTS=1 ANTHROPIC_API_KEY=... .venv-agents-litellm/bin/python -m pytest \\
        tests/integration/test_framework_live_openai_agents_claude_chat.py -s

It skips elsewhere (no LiteLLM). The release gate runs it the same way, in a venv it builds from
that extra under its own locks with litellm pinned (platform ``release_gate``, the Claude chat
leg), and reads the model and the request count from the line this file prints at the end.

THINKING ON CLAUDE 5.x
----------------------
The Agents SDK sends a turn's thinking back to Claude only when ``ModelSettings.reasoning`` has
an effort. LiteLLM 1.83 turns that effort into ``thinking: {type: enabled, budget_tokens: ...}``,
which Claude 5.x models refuse (HTTP 400, "Use thinking.type.adaptive"), so the chat also passes
``extra_args={"thinking": {"type": "adaptive"}}``, which LiteLLM sends as it is.

WHERE DECIMALAI'S ANSWERS COME FROM
-----------------------------------
The in-process fake platform the unit tests use (``FakePlatform`` from
``tests/test_agent_binding.py``), as in the Anthropic and LangChain live files: case 2 changes
the saved prompt between two messages without a dashboard, and the subject is how Claude treats
what the binding sends. So it needs no backend: it overrides the directory's ``_require_gates``
and ``_reset_sdk`` fixtures below. Only DecimalAI's trace processor is registered (a group id is
read off the run's trace, so tracing stays on); its traces go to a mock client.

WHAT MAKES CASE 1 NON-CIRCULAR
------------------------------
The skill's text carries a claim code minted for this session (``DMG-`` and eight random hex
digits). It is never in the instructions; the only way to it is the ``load_skill`` result. The
case checks that result in the next request's input and that the answer quotes the code.

THE MODEL AND THE BUDGET
------------------------
Pinned to ``claude-haiku-5-5``, as the Anthropic and LangChain files are. Every model request
goes through a ``LitellmModel`` that records it, and the file refuses to send more than
``REQUEST_BUDGET`` requests in one run.

Marker: live_llm + skill_routing. Needs ``RUN_LIVE_LLM_TESTS=1``, ``ANTHROPIC_API_KEY`` and
LiteLLM; skips without them.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import pytest

from . import _live_helpers as h

pytest.importorskip("agents")
pytest.importorskip("litellm")
pytest.importorskip("agents.extensions.models.litellm_model")

from agents import Agent, ModelSettings, RunConfig, Runner, function_tool  # noqa: E402
from agents.extensions.models.litellm_model import LitellmModel  # noqa: E402
from agents.tracing import get_trace_provider, set_trace_processors  # noqa: E402
from openai.types.shared import Reasoning  # noqa: E402

import decimalai._agent as agent_mod  # noqa: E402
import decimalai._agent_binding as binding  # noqa: E402
import decimalai.openai_agents as oa  # noqa: E402
import tests.test_agent_binding as fake_platform  # noqa: E402
from decimalai._agent import SavedPrompt  # noqa: E402
from decimalai.skill_router import SkillRouter  # noqa: E402

pytestmark = [pytest.mark.live_llm, pytest.mark.skill_routing]

MODEL = h._guard_no_opus("claude-haiku-5-5")
REQUEST_BUDGET = 25
FAKE_BASE = "http://openai-agents-claude-live.test"
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

#: Requests sent by every test in this run.
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
    print(f"\nAnthropic requests sent by {__name__.rsplit('.', 1)[-1]} ({MODEL}): {_SPENT[0]} "
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
    monkeypatch.setattr(oa, "_skill_router_singleton", router)
    monkeypatch.setattr(oa, "_tracing_processor", None, raising=False)
    monkeypatch.setattr(oa, "_tracing_processor_added", False, raising=False)
    fake.saved = SavedPrompt(agent_name="support", system_prompt=PROMPT, version_number=4,
                             content_hash="c" * 64)
    monkeypatch.setattr(agent_mod, "read_prompt", lambda name, **kwargs: fake.saved)
    # Only DecimalAI's processor (agent_config() registers it): the Agents SDK's own would
    # upload these traces to OpenAI.
    processors = list(get_trace_provider()._multi_processor._processors)
    set_trace_processors([])
    yield fake
    set_trace_processors(processors)


class Wire(LitellmModel):
    """``LitellmModel`` that records each request as the model got it, and its outcome."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent: list = []

    async def get_response(self, system_instructions, input, model_settings, tools, *args, **kwargs):
        if _SPENT[0] >= REQUEST_BUDGET:
            raise RuntimeError(f"this file may send at most {REQUEST_BUDGET} Anthropic requests per run")
        _SPENT[0] += 1
        entry = {
            "system": system_instructions,
            # Read when the request is built, as the model does.
            "tools": {tool.name: json.loads(json.dumps(tool.params_json_schema)) for tool in tools},
            "input": json.loads(json.dumps(input, default=str)),
            "status": None, "error": None,
        }
        self.sent.append(entry)
        try:
            response = await super().get_response(system_instructions, input, model_settings, tools,
                                                  *args, **kwargs)
        except Exception as exc:
            entry["status"] = getattr(exc, "status_code", None)
            entry["error"] = str(exc)
            raise
        entry["status"] = 200
        return response


@function_tool
def lookup_order(order_id: str) -> str:
    """Look up an order's status.

    Args:
        order_id: The order id, for example ORD-1001.
    """
    return json.dumps({"order_id": order_id, "status": "delivered"})


def _agent(support, model, **settings) -> Agent:
    return Agent(name=support.name, instructions=support.instructions, model=model,
                 tools=[lookup_order, *support.tools],
                 model_settings=ModelSettings(max_tokens=8000, **settings))


def _served(sent: list) -> list:
    """The text each ``load_skill`` call got back, as the next request carried it."""
    calls = set()
    out = []
    for entry in sent:
        for item in entry["input"] if isinstance(entry["input"], list) else []:
            if item.get("type") == "function_call" and item.get("name") == "load_skill":
                calls.add(item.get("call_id"))
            elif item.get("type") == "function_call_output" and item.get("call_id") in calls:
                out.append(str(item.get("output")))
    return out


# ── 1. one message ──────────────────────────────────────────────────────────


def test_one_message_loads_the_matching_skill_and_answers_from_it(platform):
    support = oa.agent_config("support")
    model = Wire(f"anthropic/{MODEL}")
    result = Runner.run_sync(_agent(support, model), QUESTION, context=SAM, max_turns=6)
    print(f"\n[one message] {len(model.sent)} requests")
    assert model.sent and all(CODE not in entry["system"] for entry in model.sent), \
        "the code must reach the model only through load_skill"
    assert model.sent[0]["system"].startswith("You are Acme's customer support agent. The customer is Sam.")
    served = _served(model.sent)
    assert served and served[0].startswith("## Skill: damaged-item") and CODE in served[0], served
    assert CODE in str(result.final_output), f"the answer does not quote the code: {result.final_output!r}"


# ── 2. a chat with thinking on, across a prompt change ──────────────────────

FIRST = ("My keyboard was delivered on 3 October and arrived cracked. Today is 20 October. "
         "Am I still inside the claim window, and what claim code do I quote?")
SECOND = "Thanks. Repeat the claim code you gave me, and nothing else."
CHANGED = (
    "You are Acme's customer support agent (version 5). The customer is {{customer.name}}.\n"
    "Be brief. Answer using the procedures available to you."
)
#: Thinking sent back: an effort for the Agents SDK, adaptive thinking for Claude 5.x.
THINKING = {"reasoning": Reasoning(effort="high"), "extra_args": {"thinking": {"type": "adaptive"}}}


def _thought(items: list) -> bool:
    return any(item.get("type") == "reasoning" and (item.get("provider_data") or {}).get("thinking_blocks")
               for item in items)


def test_a_chat_keeps_its_instructions_when_the_prompt_changes_with_thinking_on(platform):
    """Claude refuses a message whose system prompt differs from the one a thinking block in the
    history was made under. A chat's runs share a group id and keep its first run's
    instructions, so a prompt saved mid-chat cannot break it; instructions composed afresh
    for the chat's history would."""
    support = oa.agent_config("support")
    model = Wire(f"anthropic/{MODEL}")
    agent = _agent(support, model, **THINKING)

    # Turn 1. Thinking is adaptive, so a turn can come back without a thinking block, and then
    # there is no hazard to test: retry as a new chat.
    for attempt in range(3):
        chat = RunConfig(group_id=f"chat-{uuid.uuid4().hex[:8]}")
        start = len(model.sent)
        result = Runner.run_sync(agent, FIRST, context=SAM, run_config=chat, max_turns=6)
        history = result.to_input_list()
        if _thought(history):
            break
    turn_one = model.sent[start:]
    print(f"\n[chat] turn 1 took {attempt + 1} attempt(s), {len(turn_one)} requests")
    assert _thought(history), "three tries of turn 1 came back without a thinking block"
    assert any(CODE in text for text in _served(turn_one)), "turn 1 loaded the skill"

    # The saved prompt changes and this process reads it.
    platform.saved = SavedPrompt(agent_name="support", system_prompt=CHANGED, version_number=5,
                                 content_hash="d" * 64)
    support._refresh_prompt()
    assert support.prompt_version == 5
    follow_up = [*history, {"role": "user", "content": SECOND}]

    # The control: the chat's history, sent by a run with no conversation to keep (another
    # group id), so its instructions are composed afresh from the new prompt.
    start = len(model.sent)
    with pytest.raises(Exception) as refused:
        Runner.run_sync(agent, follow_up, context=SAM, max_turns=6,
                        run_config=RunConfig(group_id=f"control-{uuid.uuid4().hex[:8]}"))
    control = model.sent[start:]
    print(f"[chat] control: statuses {[entry['status'] for entry in control]}: "
          f"{str(refused.value)[:300]}")
    assert [entry["status"] for entry in control] == [400], (refused.value, control)
    assert control[0]["system"].startswith("You are Acme's customer support agent (version 5)")
    assert control[0]["tools"] == turn_one[0]["tools"], "only the instructions differ, so they are what Claude refuses"
    assert "bound to a different conversation" in str(refused.value), str(refused.value)

    # Turn 2 in the chat: the instructions it started with, so Claude accepts it.
    start = len(model.sent)
    result = Runner.run_sync(agent, follow_up, context=SAM, run_config=chat, max_turns=6)
    turn_two = model.sent[start:]
    print(f"[chat] turn 2: {len(turn_two)} requests, statuses {[entry['status'] for entry in turn_two]}")
    assert turn_two and all(entry["status"] == 200 for entry in turn_two)
    assert turn_two[0]["system"] == turn_one[0]["system"], "byte for byte, as the model got it"
    assert turn_two[0]["tools"] == turn_one[0]["tools"]
    assert CODE in str(result.final_output), result.final_output
    assert support.conversation(chat.group_id).state["agents"]["support"]["prompt"]["version"] == 4
