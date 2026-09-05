"""The generated file EXECUTES — not just parses.

`tests/test_cli_init_scaffold.py` proves the template is valid Python and that
`enable_skill_loader=True` and the agent name are really there. That is the
cheap half. It cannot catch the expensive half: a template that compiles fine
while calling an `instrument()` keyword the adapter does not accept, importing
a symbol that moved, or constructing an `Agent(...)` whose signature changed.
Every one of those ships a file that dies on line one for the user, and the
product claim is that the file RUNS.

So this module writes the generated file to a temp directory and runs it in a
SUBPROCESS, with exactly one thing stubbed: the LLM call. Everything else is
the real code path — the real imports, the real `instrument()`, the real
`Agent(...)` / `init_chat_model(...)`, the real entry point.

A subprocess and not an in-process exec, because `instrument()` is a
process-wide install: it monkey-patches `Agent.__init__` (openai-agents) and
`BaseChatModel.invoke` (langchain) and registers a global trace processor.
Running that inside pytest would leak those patches into every test that came
after it.

`decimalai.init()` is the one other stub. It is not what is under test here,
and left alone it makes a real network call to verify the (fake) key.

Each framework skips when its runtime package is absent, so a core-only
checkout still runs the suite. `agents` is in the dev extras and therefore
runs on every commit; `langchain` is not, and runs wherever it is installed.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from decimalai.cli.scaffold import render_agent_file

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT = "refund-bot"
SKILLS = [{"skill_name": "refund-policy", "description": "When a refund is allowed",
           "scope": "agent"}]

# The agent's own system prompt, which `decimalai.load_agent()` reads at run
# time. Every framework below is run TWICE: once with this, and once with the
# prompt unset — the state an agent created without one is in, where the file
# must still run and must not invent instructions.
PROMPT = "Never issue a refund over $500."
NO_PROMPT = None

# What each framework needs installed, and how to stub its one LLM call.
# The stub replaces the model, never the DecimalAI seam — patching the seam
# would make the test pass on a template that never reaches it.
RUNTIMES = {
    "openai-agents": {
        "requires": "agents",
        "stub": """
import agents
class _Result:
    final_output = "stubbed answer"
_stack.enter_context(patch.object(agents.Runner, "run_sync", return_value=_Result()))
""",
        "probe": """
# What the Agent will actually be run with. `Runner.run_sync` is stubbed, so
# the Agent's own instructions are the last observable before the model — and
# they are the exact value the runner passes as `system_instructions`. The
# skill loader wraps a string into a per-run callable, so unwrap it first;
# reading the wrapper would print a function and prove nothing.
from decimalai.openai_agents import _BASE_INSTRUCTIONS_ATTR
_instr = _mod["agent"].instructions
if callable(_instr):
    _instr = getattr(_instr, _BASE_INSTRUCTIONS_ATTR, None)
print("PROMPT_SEEN:", _instr if _instr else "<none>")

import decimalai.openai_agents as adapter
print("LOADER:", adapter._skill_loader_installed)
# Read the name off the processor the adapter actually registered. The
# earlier version of this probe printed the literal "refund-bot", which
# made the assertion below unfalsifiable — it passed whatever instrument()
# did with the name. openai-agents has no module-level equivalent of
# langchain's _install_agent_name; the name lives on the processor as
# default_agent_name, so that is what gets read.
from agents.tracing import get_trace_provider
_procs = get_trace_provider()._multi_processor._processors
_ours = [p for p in _procs if hasattr(p, "default_agent_name")]
print("PROCESSORS:", len(_ours))
print("BOUND:", _ours[0].default_agent_name if _ours else "<no processor>")
""",
    },
    "pydantic-ai": {
        "requires": "pydantic_ai",
        # `infer_model` is where the MODEL string becomes a model object, so
        # patching it swaps the provider and leaves everything else — the real
        # Agent, the real tool loop, the real adapter patches — on the path.
        # Patching `Agent.run_sync` instead (the openai-agents shape) would skip
        # the loop, and the loop is the body channel on this framework.
        "stub": """
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
_sent = []


def _reply(messages, info):
    # The whole request, verbatim: this is the evidence the canary is read from.
    _sent.append((messages, info))
    tools = {t.name for t in (info.function_tools or [])}
    # A tool result already came back → this is the answering turn. Derived from
    # the REQUEST rather than from a call counter, so the script does not have
    # to know how many turns the adapter takes.
    answered = any(
        type(p).__name__ == "ToolReturnPart"
        for m in messages for p in (getattr(m, "parts", None) or [])
    )
    if "load_skill" in tools and not answered:
        return ModelResponse(parts=[
            ToolCallPart("load_skill", {"name": "restocking-policy"}),
        ])
    return ModelResponse(parts=[TextPart("stubbed answer")])


_stack.enter_context(patch(
    "pydantic_ai.models.infer_model", return_value=FunctionModel(_reply),
))
""",
        "probe": """
import decimalai.pydantic_ai as adapter
print("LOADER:", adapter._skill_loader_installed)
# The name the RUN SCOPE will stamp, computed by the adapter's own resolver
# against the Agent the template built — not the literal from the template and
# not the module global either. Pydantic AI fills a missing name from the local
# variable (`agent = Agent(...)` runs as the agent "agent"), so this is the one
# reading that can tell a bound name from an invented one.
print("BOUND:", adapter._run_agent_name(_mod["agent"]))

# Every part the model was handed, across every request, flattened. Parts carry
# their text under `content` whatever their kind (system prompt, user turn, tool
# return), so one accessor covers all three.
def _texts():
    for messages, _info in _sent:
        for m in messages:
            for p in (getattr(m, "parts", None) or []):
                c = getattr(p, "content", None)
                if isinstance(c, str):
                    yield type(p).__name__, c

_parts = list(_texts())

# The agent's own prompt, told apart from the rail's system parts by their
# headings — same filter the langchain probe uses, and for the same reason: the
# rail registers a system_prompt function, so "the first system part" stopped
# being "the agent's prompt" the moment the loader was on.
_system = [
    c for kind, c in _parts
    if kind == "SystemPromptPart"
    and not c.lstrip().startswith(("## Available Skills", "## Skill:"))
]
print("PROMPT_SEEN:", _system[0] if _system else "<none>")

# The load_skill tool was really registered — the body channel on this
# framework IS the tool, so a run without it can only ever offer a menu.
print("TOOL_REGISTERED:", any(
    "load_skill" in {t.name for t in (info.function_tools or [])}
    for _m, info in _sent
))
print("TURNS:", len(_sent))

_all = "\\n".join(c for _kind, c in _parts)
print("SKILL_OFFERED:", "restocking-policy" in _all)
print("SKILL_BODY_DELIVERED:", "23.5% restocking fee" in _all)
""",
    },
    "langchain": {
        "requires": "langchain",
        "stub": """
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
_sent = []

# A REAL BaseChatModel subclass, not a duck-typed object.
#
# The skill rail works by monkey-patching BaseChatModel.invoke. A plain class
# that merely defines its own `invoke` never inherits that patch, so the
# injection code never runs and `_sent` captures the template's messages
# BEFORE any skill routing. That is exactly what this file used to do, and it
# is why a rail that delivered nothing at all kept the whole suite green.
# Overriding `_generate` (the abstract leaf) instead leaves `invoke` — and
# therefore the patch — on the call path.
class _Fake(BaseChatModel):
    @property
    def _llm_type(self):
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        _sent.append(messages)   # what the model was ACTUALLY handed, post-injection
        return ChatResult(generations=[
            ChatGeneration(message=AIMessage(content="stubbed answer"))
        ])
_stack.enter_context(patch("langchain.chat_models.init_chat_model", return_value=_Fake()))
""",
        "probe": """
import decimalai.langchain as adapter
print("LOADER:", adapter._skill_loader_installed)
print("BOUND:", adapter._install_agent_name)

# The messages the model was called with. The prompt has to arrive AS a system
# message and the question has to stay last: the skills rail reads the trailing
# human turn to route, and providers reject a system message after one.
_msgs = _sent[-1]
# The rail injects its own system messages after the caller's leading system
# run, so "the first system message" is no longer the same thing as "the
# agent's prompt". Filter the rail's out by their headings — otherwise a
# no-prompt agent looks like it has one, and the assertion that a no-prompt
# agent sends NO invented instructions silently stops testing anything.
_system = [
    m for m in _msgs
    if getattr(m, "type", None) == "system"
    and not str(getattr(m, "content", "")).lstrip().startswith(
        ("## Available Skills", "## Skill:")
    )
]
print("PROMPT_SEEN:", _system[0].content if _system else "<none>")
print("LAST_TURN:", getattr(_msgs[-1], "type", None))

# THE CANARY. Every message the model was handed, concatenated. The skill's
# body is only in here if the rail DELIVERED it — a menu row carries the
# skill's name and description and never its body.
_all = "\\n".join(str(getattr(m, "content", "")) for m in _msgs)
print("SKILL_OFFERED:", "restocking-policy" in _all)
print("SKILL_BODY_DELIVERED:", "23.5% restocking fee" in _all)
""",
    },
}

#: A fact no model has priors for. If it reaches the model, it reached it as a
#: SKILL BODY and by no other route — which is the only way to assert delivery
#: rather than mere offering.
CANARY = "Opened boxes carry a 23.5% restocking fee."
CANARY_SKILL = "restocking-policy"

#: Frameworks whose stub leaves the model on the call path, so the canary can be
#: read off what the model was handed — and, for each, which channel the body was
#: supposed to arrive by. Two different channels, both graded by one assertion:
#: langchain has no tool loop so injection is all it has, pydantic-ai owns one so
#: the body comes back as a tool result and injection stays OFF
#: (`DecimalConfig.resolve_inject_body(has_tool_loop=True)`).
#:
#: A dict rather than a set because the failure message has to name the channel:
#: "the body never arrived" is a different investigation on each.
BODY_CHANNEL_OBSERVABLE = {
    "langchain": (
        "On an adapter with no tool loop, prompt injection is the only body "
        "channel."
    ),
    "pydantic-ai": (
        "This adapter registers a real load_skill tool and Pydantic AI owns the "
        "loop, so the body should have come back as a tool result mid-turn."
    ),
}

#: The shape `SkillRouter._request` sees from the real backend, reduced to the
#: keys the client actually reads. Patched in rather than reached over HTTP so
#: this test needs no network and no live model.
ROUTER_FIXTURE = """
def _fake_router_request(self, method, path, **kw):
    if path.endswith("/body"):
        return {{"body": {canary!r}, "content_hash": "sha256:canary", "version": 1}}
    return {{
        "skills": [{{"name": {skill!r}, "description": "When a refund is allowed."}}],
        "prompt_fragment": "## Available Skills\\n\\n| Skill | Description |\\n"
                           "| --- | --- |\\n| {skill} | When a refund is allowed. |",
        "strategy": "semantic",
        "routing_id": "rt_" + "0" * 24,
    }}
_stack.enter_context(patch(
    "decimalai.skill_router.SkillRouter._request", _fake_router_request,
))
""".format(canary=CANARY, skill=CANARY_SKILL)

DRIVER = """\
import contextlib, runpy
from unittest.mock import patch

from decimalai import AgentConfig

_stack = contextlib.ExitStack()
{router}
{stub}
# init() runs FOR REAL, with only its network probe turned off.
#
# It used to be replaced wholesale, which meant no DecimalConfig was ever built,
# so the skill router singleton could not be constructed and the skills rail was
# inert for the entire test — the rail was asserted to be INSTALLED while having
# nothing to route with. Wrapping instead of replacing keeps `_init.called` true
# and makes the rail actually run.
import decimalai as _decimalai
_real_init = _decimalai.init
_init = _stack.enter_context(patch(
    "decimalai.init",
    side_effect=lambda *a, **k: _real_init(*a, **{{**k, "verify": False}}),
))
_flush = _stack.enter_context(patch("decimalai.flush"))

# load_agent() is the third network call the generated file makes, and it runs
# at MODULE SCOPE — deliberately, because that is where the prompt is needed
# and where a wrong name should fail. Unstubbed it would reach a real backend
# from the test suite, so it is stubbed exactly like init().
#
# With the REAL AgentConfig, never a MagicMock: a Mock answers to any attribute
# name, so a template that read `config.system_prmopt` would sail through this
# test and ship a file that dies for the user on line one.
_config = AgentConfig(agent_name={agent!r}, system_prompt={prompt!r})
_load = _stack.enter_context(patch("decimalai.load_agent", return_value=_config))

with _stack:
    _mod = runpy.run_path({path!r}, run_name="__main__")
    print("INIT:", _init.called)
    print("FLUSH:", _flush.called)
    _args = _load.call_args
    print("LOAD_AGENT:", (_args.args[0] if _args and _args.args
                          else "<not called with a name>"))
{probe}
"""


#: The environment every subprocess in this module runs under. One copy, because
#: three tests were already carrying four identical keys and the fourth key below
#: has to reach all of them.
_ENV = {
    "PATH": "/usr/bin:/bin",
    "PYTHONPATH": str(REPO_ROOT),
    "DECIMAL_API_KEY": "dai_sk_not_a_real_key",
    # The template's own provider-key guard runs before anything else; every
    # model call below is stubbed, so this value is never sent anywhere.
    "OPENAI_API_KEY": "sk-stub-for-the-guard-only",
    # The generated file legitimately turns the loader on; inside a
    # disk-runtime harness that prints a duplicate-skills warning to
    # stderr, which is correct advice and noise here.
    "DECIMALAI_SUPPRESS_DISK_RUNTIME_WARNING": "1",
    "DECIMAL_AUTOINIT": "false",
    # Port 9 (discard) refuses instantly, which is what makes this hermetic
    # rather than merely fast. The pydantic-ai template calls `init(otel=True)`
    # — the only way that adapter traces at all — and a real span exporter with
    # the hosted default would spend the subprocess's exit trying to POST this
    # run's prompts to api.decimal.ai with a fake key. The router is patched;
    # the exporter is not, and is not the thing under test.
    "DECIMAL_BASE_URL": "http://127.0.0.1:9",
}


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


@pytest.mark.parametrize("prompt", [PROMPT, NO_PROMPT], ids=["prompt", "no-prompt"])
@pytest.mark.parametrize("framework", sorted(RUNTIMES))
def test_generated_file_executes(framework, prompt, tmp_path):
    spec = RUNTIMES[framework]
    if not _have(spec["requires"]):
        pytest.skip(f"{spec['requires']} not installed")

    agent_py = tmp_path / "agent.py"
    agent_py.write_text(render_agent_file(AGENT, framework, SKILLS))
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER.format(
        stub=spec["stub"], probe=spec["probe"], path=str(agent_py),
        agent=AGENT, prompt=prompt, router=ROUTER_FIXTURE,
    ))

    proc = subprocess.run(
        [sys.executable, str(driver)],
        capture_output=True, text=True, cwd=tmp_path, timeout=120,
        env=_ENV,
    )
    assert proc.returncode == 0, (
        f"generated {framework} file failed to run:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    out = proc.stdout

    # The example call ran and produced the model's answer.
    assert "stubbed answer" in out, out
    assert "INIT: True" in out and "FLUSH: True" in out, out

    # The load-bearing one: `instrument(enable_skill_loader=True)` did not
    # merely parse — the adapter accepted it and the loader is INSTALLED.
    # A template that passes an argument the adapter silently ignores would
    # satisfy every AST assertion in the sibling module and fail here.
    assert "LOADER: True" in out, out

    # And the name reached the adapter, so traces file against the agent the
    # user configured rather than an auto-detected one.
    assert f"BOUND: {AGENT}" in out, out

    # The prompt: read for THIS agent, and actually delivered to the model.
    #
    # `LOAD_AGENT` alone would only prove the template called the function.
    # `PROMPT_SEEN` is the half that matters — the value came back as data and
    # the template's own code put it in front of the model. The two templates
    # spent months proving how easily this half goes missing: langchain sent no
    # system message at all, and openai-agents overwrote whatever the user
    # typed with "You are <name>. Use the skills you are given."
    assert f"LOAD_AGENT: {AGENT}" in out, out
    if prompt is None:
        # No prompt set is a REAL state. The file still runs (asserted above),
        # and it must send nothing rather than invent instructions — an agent
        # quietly following a made-up prompt looks exactly like a working one.
        assert "PROMPT_SEEN: <none>" in out, out
    else:
        assert f"PROMPT_SEEN: {prompt}" in out, out

    # THE CANARY — the assertion this whole file existed without.
    #
    # Every other check here proves a rail was INSTALLED or a name was BOUND.
    # None of them proved the thing the product promises: that a skill's
    # knowledge reaches the model. On 2026-08-28 a real agent built by
    # `decimalai init` answered a customer with one sentence about its own
    # tooling, because LangChain owns no tool loop and body injection defaulted
    # off — so the model got a menu of titles it could not read. The suite was
    # green throughout.
    #
    # Offered-but-not-delivered is the exact failure state, so assert both rungs
    # separately: a test that only checked "the skill was mentioned" would have
    # passed against the bug.
    #
    # Asserted for every framework whose stub keeps the model on the call path.
    # openai-agents is the exception and it is a limit of ITS STUB, not of the
    # adapter: replacing `Runner.run_sync` removes the loop, and on that
    # framework the loop IS the body channel. Named here rather than left as a
    # silent gap in an `if`.
    if framework in BODY_CHANNEL_OBSERVABLE:
        assert "SKILL_OFFERED: True" in out, out
        assert "SKILL_BODY_DELIVERED: True" in out, (
            f"the {framework} template offered the skill and never delivered its "
            f"BODY — the model was handed a menu row it has no way to read. "
            f"{BODY_CHANNEL_OBSERVABLE[framework]}\n" + out
        )

    # The question stays the last turn: the skills rail routes on the trailing
    # human message, and langchain_anthropic raises on a system message placed
    # after any human/AI turn.
    if framework == "langchain":
        assert "LAST_TURN: human" in out, out

    # Exactly one DecimalAI processor. Two would mean every span is sent
    # twice — the failure `init(openai_agents=True)` plus `instrument()`
    # produces, and the reason the template passes a bare `init()`.
    if framework == "openai-agents":
        assert "PROCESSORS: 1" in out, out

    if framework == "pydantic-ai":
        # The tool is the body channel here, so its absence explains a missing
        # body — and its presence rules out the interpretation that the canary
        # arrived by injection instead. Both channels reaching the model at once
        # is the duplicate-delivery state `resolve_inject_body(has_tool_loop=…)`
        # exists to prevent.
        assert "TOOL_REGISTERED: True" in out, out
        # Two model requests: ask for the body, then answer with it. One would
        # mean the loop never closed — the shape a bare model call has.
        assert "TURNS: 2" in out, out


# ── the langchain template must be an AGENT, not a chat completion ───
#
# The test above runs the template AS GENERATED, with no tools. That is the
# happy path and it stayed green through the entire defect: until 2026-08-29
# the template emitted `agent = init_chat_model(MODEL)` — a chat completion in
# a variable named `agent` — and answering one question with no tools works
# fine on a bare chat model.
#
# What did not work was the very next thing the template's own docstring told
# the user to do: "Add tools, memory or a graph here". Binding a tool to a bare
# chat model makes the model reply with `tool_calls` and an EMPTY `.content`.
# Nothing runs the tool, nothing feeds a result back, and `run()` returns "".
# So the file shipped by the DEFAULT framework of a product whose whole
# vocabulary is "agent" stopped being an agent the moment it was used as one.
#
# This test does what that user would do — it adds a tool at the seam the
# template advertises — and asserts the loop actually closes.

#: The seam. Asserted to exist before the substitution, so removing `TOOLS`
#: from the template fails here loudly instead of turning the replace below
#: into a silent no-op that leaves the test passing against nothing.
#: Subprocess budget for the two tests below.
#:
#: 300, not the 120 the parametrized test above uses, and measured rather than
#: guessed: each of these spends ~60s importing langchain + langgraph and
#: building a real agent graph in a cold interpreter, which fits inside 120
#: alone and does NOT fit when the rest of the suite is competing for the
#: machine. It failed exactly that way once. A generous ceiling on a subprocess
#: that normally finishes in a minute costs nothing when it passes; a tight one
#: makes the whole suite flaky under load, which is worse than slow.
SLOW_TIMEOUT = 300

TOOL_SEAM = "TOOLS: list = []"

#: A fact only the TOOL can supply — not in the prompt, not in the skill body,
#: not in the model stub's own vocabulary. It reaches the transcript only if
#: the tool was really called and its result was really fed back to the model.
TOOL_FACT = "ORDER-7 shipped on Tuesday"

TOOL_DEFINITION = f'''from langchain_core.tools import tool


@tool
def order_status(order_id: str) -> str:
    """Look up an order's status."""
    return {TOOL_FACT!r}


TOOLS: list = [order_status]'''

#: A model that behaves like gpt-4o-mini does with a tool bound: one turn of
#: `tool_calls` with EMPTY content, then an answer that quotes what came back.
TOOL_LOOP_STUB = '''
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
_sent = []


class _Fake(BaseChatModel):
    """A real BaseChatModel subclass, so the skills rail's `invoke` patch is
    still on the call path (see the sibling stub for why that matters)."""

    @property
    def _llm_type(self):
        return "fake"

    def bind_tools(self, tools, **kwargs):
        from langchain_core.utils.function_calling import convert_to_openai_tool
        return self.bind(
            tools=[convert_to_openai_tool(t) for t in tools], **kwargs
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        _sent.append(messages)
        bound = kwargs.get("tools") or []
        results = [m for m in messages if getattr(m, "type", None) == "tool"]
        if bound and not results:
            # EXACTLY the shape that broke the old template: tool_calls, and
            # content is the empty string.
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content="",
                tool_calls=[{"name": "order_status",
                             "args": {"order_id": "7"}, "id": "call_1"}],
            ))])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content="Status: " + (str(results[-1].content) if results
                                  else "<the tool never ran>"),
        ))])


_stack.enter_context(
    patch("langchain.chat_models.init_chat_model", return_value=_Fake())
)
'''

TOOL_LOOP_PROBE = '''
# What run() actually returned. The old template returned "" here.
_answer = _mod["run"]("Where is order 7?")
print("ANSWER:", repr(_answer))
print("ANSWER_EMPTY:", not str(_answer).strip())

# The tool was bound to the model at all — the old template never passed
# TOOLS anywhere, so nothing was ever bound.
print("TOOLS_BOUND:", any(
    getattr(m, "tool_calls", None) for msgs in _sent for m in msgs
) or len(_sent) > 1)

# And the skill body still arrives with a tool in play: adding a tool must not
# cost the user their skills.
_all = "\\n".join(
    str(getattr(m, "content", "")) for msgs in _sent for m in msgs
)
print("SKILL_BODY_DELIVERED:", "23.5% restocking fee" in _all)
'''


def test_langchain_template_survives_a_bound_tool(tmp_path):
    """Add a tool where the template says to, and the agent still answers.

    RED against the pre-2026-08-29 template, which built a bare chat model:
    `run()` returned "" and the tool's result never reached the model.
    """
    if not _have("langchain"):
        pytest.skip("langchain not installed")

    source = render_agent_file(AGENT, "langchain", SKILLS)
    assert TOOL_SEAM in source, (
        "the langchain template no longer exposes a TOOLS seam — either it "
        "regressed to a bare chat model, or the seam was renamed and this "
        "test must be renamed with it.\n" + source
    )
    agent_py = tmp_path / "agent.py"
    agent_py.write_text(source.replace(TOOL_SEAM, TOOL_DEFINITION))

    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER.format(
        stub=TOOL_LOOP_STUB, probe=TOOL_LOOP_PROBE, path=str(agent_py),
        agent=AGENT, prompt=PROMPT, router=ROUTER_FIXTURE,
    ))

    proc = subprocess.run(
        [sys.executable, str(driver)],
        capture_output=True, text=True, cwd=tmp_path, timeout=SLOW_TIMEOUT,
        env=_ENV,
    )
    assert proc.returncode == 0, (
        f"the generated langchain file died once a tool was added:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    out = proc.stdout

    # THE RATCHET. A bare chat model answers a tool call with empty content,
    # so this is the assertion the old template failed.
    assert "ANSWER_EMPTY: False" in out, (
        "run() returned an empty string once a tool was bound — the template "
        "is a chat completion, not an agent. A bare chat model replies with "
        "tool_calls and no content, and nothing runs the loop.\n" + out
    )
    # And the loop CLOSED: the tool ran and its result went back to the model.
    # Without this, a template that merely returned some non-empty string
    # (an error message, say) would pass the check above.
    assert f"Status: {TOOL_FACT}" in out, (
        "the tool's result never reached the model — the loop did not close.\n"
        + out
    )
    # Adding a tool must not cost the user their skills.
    assert "SKILL_BODY_DELIVERED: True" in out, out


def test_langchain_template_emits_no_deprecation_warnings(tmp_path):
    """The generated file must not warn on every run.

    `langgraph.prebuilt.create_react_agent` — the obvious way to add a loop —
    raises `LangGraphDeprecatedSinceV10` on import-and-call and is removed in
    langgraph 2.0. A scaffold whose first run prints a deprecation notice
    teaches the user their setup is wrong when it is not, so the template uses
    `langchain.agents.create_agent` instead. This is what keeps it there.
    """
    if not _have("langchain"):
        pytest.skip("langchain not installed")

    agent_py = tmp_path / "agent.py"
    agent_py.write_text(render_agent_file(AGENT, "langchain", SKILLS))
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER.format(
        stub=RUNTIMES["langchain"]["stub"], probe="", path=str(agent_py),
        agent=AGENT, prompt=PROMPT, router=ROUTER_FIXTURE,
    ))

    proc = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", str(driver)],
        capture_output=True, text=True, cwd=tmp_path, timeout=SLOW_TIMEOUT,
        env=_ENV,
    )
    # Match on the class name rather than the module: langchain and langgraph
    # each subclass DeprecationWarning under their own name, and -W error
    # surfaces whichever fires.
    #
    # Scoped to warnings THIS FILE causes, which is what the docstring above
    # claims and what a template can actually control. A bare substring over
    # stderr is broader than that, and on 2026-08-30 it turned the floors job
    # red on a warning no template choice can avoid: at our declared floor,
    # `from langchain.agents import create_agent` pulls langgraph 1.0.0, which
    # pins `langgraph-checkpoint<3.0.0`, and every checkpoint below 4.1.0 emits
    # `LangChainPendingDeprecationWarning: The default value of allowed_objects
    # will change` on import. Raising our floor to dodge a third party's warning
    # hygiene would be an arbitrary number that says nothing about our code.
    #
    # A real DeprecationWarning from the generated file is still caught, and by
    # something stronger than this: `-W error::DeprecationWarning` above turns it
    # into a non-zero exit, asserted below. This line remains for the Pending*
    # subclasses that `-W error::DeprecationWarning` does not escalate — the
    # class `create_react_agent` raises — so it must not be deleted.
    ours = [
        ln for ln in proc.stderr.splitlines()
        if "Deprecat" in ln and "site-packages" not in ln and "dist-packages" not in ln
    ]
    assert not ours, (
        "the generated file emits a deprecation warning on a plain run:\n"
        + "\n".join(ours)
    )
    assert proc.returncode == 0, (
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


def test_openai_agents_template_survives_an_unknown_tool_call(tmp_path):
    """The Support pack's starter prompt named `kb_search`; a model that called it
    raised `ModelBehaviorError: Tool kb_search not found` out of Runner.run_sync and
    the generated file died with a traceback (fleet scaffold canary, 2026-09-04).
    run() must answer with a message naming the mismatch, never raise."""
    if not _have("agents"):
        pytest.skip("agents not installed")
    agent_py = tmp_path / "agent.py"
    agent_py.write_text(render_agent_file(AGENT, "openai-agents", SKILLS))
    stub = """
import agents
from agents.exceptions import ModelBehaviorError
_stack.enter_context(patch.object(
    agents.Runner, "run_sync",
    side_effect=ModelBehaviorError("Tool kb_search not found in agent refund-bot"),
))
"""
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER.format(
        stub=stub, probe="", path=str(agent_py), agent=AGENT, prompt=PROMPT,
        router=ROUTER_FIXTURE,
    ))
    proc = subprocess.run(
        [sys.executable, str(driver)], capture_output=True, text=True,
        cwd=tmp_path, timeout=120, env=_ENV,
    )
    assert proc.returncode == 0, proc.stderr
    assert "does not define" in proc.stdout, proc.stdout
    assert "kb_search" in proc.stdout, proc.stdout
    assert "Traceback" not in proc.stderr, proc.stderr


@pytest.mark.parametrize("framework", sorted(RUNTIMES))
def test_missing_provider_key_is_a_one_line_refusal(framework, tmp_path):
    """Without the guard a missing OPENAI_API_KEY is a traceback from inside the
    provider client — after init(), instrument() and load_agent() have all made
    their network calls. The file must refuse on line one and point at the
    `Set:` block `decimalai init` printed."""
    spec = RUNTIMES[framework]
    if not _have(spec["requires"]):
        pytest.skip(f"{spec['requires']} not installed")
    agent_py = tmp_path / "agent.py"
    agent_py.write_text(render_agent_file(AGENT, framework, SKILLS))
    env = {k: v for k, v in _ENV.items() if k != "OPENAI_API_KEY"}
    proc = subprocess.run(
        [sys.executable, str(agent_py)], capture_output=True, text=True,
        cwd=tmp_path, timeout=120, env=env,
    )
    assert proc.returncode == 1, proc.stderr
    assert "OPENAI_API_KEY is not set" in proc.stderr, proc.stderr
    assert "Traceback" not in proc.stderr, proc.stderr   # a refusal, not a stack
    assert "decimalai init" in proc.stderr, proc.stderr
