"""Render a runnable agent file from an agent's DecimalAI configuration.

`decimalai init <agent-name>` closes the gap between "created an agent in the
dashboard" and "an agent is running": the dashboard stores a name, a
description and a set of skills, and until now the user still had to write the
agent themselves. This module turns that stored configuration into a file that
runs.

We generate; they run. Nothing here executes the agent, ships a runtime, or
phones home at run time beyond the SDK the file imports — the model is
`create-next-app`, not hosting.

FIVE THINGS THIS TEMPLATE EXISTS TO GET RIGHT
---------------------------------------------

0. The agent's own SYSTEM PROMPT is what it runs on. Until 2026-08-25 the
   langchain template sent no system message at all and the openai-agents one
   hardcoded "You are <name>. Use the skills you are given." — so whatever the
   user typed at /agents/new was silently discarded by the file we generated
   for them. Both now read it with `decimalai.load_agent(...)`.

   EXPLICITLY, and that is the design, not an implementation detail: the
   prompt lands in a variable in THEIR file and is passed to the model by a
   line they can see. Skills go the other way (instrument() injects them),
   because a skill menu is additive and a system prompt is not — silently
   replacing one means their repo says "Never issue refunds over $500" while
   the model receives something else, which is unfixable from their side.

   `load_agent()` runs at process start, so the no-redeploy property survives:
   edit the prompt in the dashboard, restart, done. And when an agent has NO
   prompt set (a real state — it is optional at creation) the file still runs
   and sends nothing rather than inventing one.

1. `enable_skill_loader=True`. Both skill-capable adapters default it to
   False (decimalai/langchain.py, decimalai/openai_agents.py), so a file that
   merely calls `init()` traces beautifully and delivers NONE of the agent's
   skills. The failure is silent and it is the whole reason the scaffold is
   worth shipping: without the flag the model is handed a list of skill titles
   it cannot read. Every generated file carries it, and
   `tests/test_cli_init_scaffold.py` fails the build if one stops.

2. The agent name is BOUND. An unbound name means the adapter auto-detects
   one off the runnable ("langchain-agent" when it cannot tell), so traces are
   filed under a different agent than the one the user configured — their page
   stays empty forever while traces pile up somewhere else. Same bug the
   dashboard snippets were fixed for on 2026-08-22.

3. What comes out is an AGENT. Until 2026-08-29 the langchain template emitted
   `agent = init_chat_model(MODEL)` — a chat completion with a variable named
   `agent` and no tool loop anywhere — while its own docstring said "Add tools,
   memory or a graph here". Following that advice breaks the file: binding a
   tool to a bare chat model makes the model reply with `tool_calls` and an
   EMPTY `.content`, so `run()` returns `""` and nothing ever runs the tool.
   Measured against gpt-4o-mini and reproduced in
   `tests/test_cli_init_scaffold_runs.py::test_langchain_template_survives_a_bound_tool`.
   It now emits `create_agent(model, tools=TOOLS, system_prompt=...)` with
   `TOOLS = []` — a real loop that happens to have no tools yet, so the seam
   the docstring points at is one the file actually has.

   `langchain.agents.create_agent`, NOT `langgraph.prebuilt.create_react_agent`.
   Two reasons, both checked by running it: the langgraph symbol raises
   `LangGraphDeprecatedSinceV10` on every single run (it moved, and goes away in
   langgraph 2.0), and it lives in a package the template does not otherwise
   import. `langchain.agents` is the same distribution `init_chat_model` already
   comes from, so the install line does not grow. `langchain` 0.3.x has no
   `create_agent`, and cannot be what resolves here: `decimalai[langchain]`
   floors `langchain-core>=1.3.3`, which 0.3.x caps below 1.0.

4. TRACING IS NOT AUTOMATIC ON EVERY ADAPTER, and the pydantic-ai template is
   where that bites. `decimalai init` ends by printing "The trace appears at →
   /agents/<name>", which is a promise about the file it just wrote. The
   langchain and openai-agents adapters keep it themselves — their `instrument()`
   installs a callback handler / a trace processor. `decimalai.pydantic_ai`
   deliberately does not: Pydantic AI emits no spans of its own, so that adapter
   ships the skills rail and the run BOUNDARY and leaves the span content to
   whatever tracing you paired with it. `init()` with no flags installs no
   exporter at all (see `decimalai/__init__.py` — the exporter is built inside
   the `if otel or crewai or autogen:` branch), so a pydantic-ai file that called
   a bare `init()` would run, deliver its skills, and leave the user staring at
   an empty traces page having done everything right.

   The template therefore pairs ONE exporter — `init(otel=True)` — with Pydantic
   AI's own OpenTelemetry instrumentation (`Agent.instrument_all()`), rather than
   the provider-instrumentor pairing the docs show for adding DecimalAI to an
   agent you already have (`init(openai=True)` +
   `openinference-instrumentation-openai`). Three reasons, in the order they
   decided it:

     * it is provider-agnostic. `Agent.instrument_all()` emits the same spans for
       `openai:…`, `google:…` and `anthropic:…`, so `--model google:gemini-…`
       needs no second install line and no different init flag. The provider
       route would need one instrumentor package per provider, chosen by the
       model string — and the fleet that walks this journey is Gemini-only.
     * one install line. `opentelemetry-sdk` is already a core dependency of
       `decimalai`, so `pip install "decimalai[pydantic-ai]"` is the whole
       requirement. The provider route adds a package whose import name
       (`openinference.instrumentation.openai`) does not match its distribution
       name, which is exactly the shape that makes an automated requirement check
       give up.
     * it is ONE exporter, which is the same one-install rule points 1-3
       protect. Adding `init(openai=True)` on top would put an OpenInference
       span and a Pydantic AI span on the same model call.

   `Agent.instrument_all()` lives in the generated file rather than inside
   `decimalai.pydantic_ai.instrument()` on purpose: turning it on inside the
   adapter would silently double up for every existing user who already pairs a
   provider instrumentor, which is the failure this point is about.

RELATIONSHIP TO THE DASHBOARD'S COPY-PASTE SNIPPETS
---------------------------------------------------
The snippets the dashboard hands out and these templates must not contradict each other, and they
do not — they answer different questions. The snippets are "add DecimalAI to
the agent you already have" and therefore end at `agent.invoke(...)` on an
`agent` that is assumed to exist. These templates are "here IS an agent", so
they additionally construct a model and an entry point. Every claim the two
share is identical, deliberately:

  * langchain: `init()` with no framework flag, then a single
    `instrument(agent_name=..., enable_skill_loader=True)`. Passing
    `langchain=True` to `init()` would install tracing a first time with the
    loader off, which resolves `disk_sync` the other way and can leave skills
    arriving from two sources at once.
  * openai-agents: same shape, and for a sharper reason — that adapter's
    `instrument()` has no already-installed guard and calls
    `add_trace_processor()` unconditionally, which appends rather than
    dedupes. `init(openai_agents=True)` followed by `instrument()` registers
    two processors and double-sends every trace.
  * pydantic-ai: the same `instrument(agent_name=..., enable_skill_loader=True)`
    — but `init(otel=True)`, not a bare `init()`. That is not a contradiction of
    the rule above, it is the same rule: exactly one span source per file. On
    the other two the adapter IS the span source; here it is not (point 4).

Two differences are intentional. The openai-agents snippet awaits
`Runner.run(...)` because it is being pasted into an async app, while the
generated file is a standalone script and uses `Runner.run_sync(...)`. And the
pydantic-ai snippet stops at `Agent(...)` without a `name=`, because it is being
pasted into an agent that already has one — the template passes a name, because
an unnamed Pydantic AI Agent is named after the local variable it is assigned to
and would file its traces under `agent`.

WHY ONLY THESE FRAMEWORKS
-------------------------
A scaffold that silently delivers no skills is worse than no scaffold, so a
framework is only offered here if its adapter has a prompt seam the skill
loader can use. `enable_skill_loader` exists on exactly four adapters
(langchain, openai_agents, anthropic, pydantic_ai); the rest — llamaindex,
claude_agent_sdk, crewai/autogen/otel, adk — trace and version but have no
loader, and generating a file for them would hand someone a program that looks
correct and quietly ignores every skill they picked. `--framework` names the
reason rather than printing a bare "invalid choice".

Of the four, three are scaffolded. `anthropic` is the holdout and the reason is
not "nobody got to it": that adapter patches one `messages.create()` call, so it
owns no tool loop — a generated file there would be a hand-rolled while-loop this
template would have to invent, and the body could only arrive by injection.
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

if TYPE_CHECKING:  # keeps this module pure — it imports nothing at run time
    from .._agent import AgentConfig

# Changing this is a one-line edit in the generated file; it lives here so the
# CLI's --model default and the template cannot drift apart.
DEFAULT_MODEL = "gpt-4o-mini"

#: Per-framework overrides of `DEFAULT_MODEL`, for frameworks whose model
#: identifier is not the bare OpenAI name.
#:
#: Pydantic AI resolves `provider:model` and REFUSES a bare one —
#: `infer_model("gpt-4o-mini")` raises `UserError: Unknown model`, checked
#: against pydantic-ai 2.36. So the shared default would have produced a file
#: that dies on the line that builds the Agent, every run, for the default
#: invocation of the command.
_FRAMEWORK_DEFAULT_MODELS: Dict[str, str] = {
    "pydantic-ai": "openai:gpt-4o-mini",
}

DEFAULT_OUTPUT = "agent.py"

#: Frameworks `init` can scaffold, in the order they are offered.
SUPPORTED_FRAMEWORKS: tuple = ("langchain", "openai-agents", "pydantic-ai")

#: Frameworks whose adapter carries `enable_skill_loader` but which have no
#: template yet. Named separately from the seamless ones because the answer to
#: "why not?" is different, and so is what we'd have to do to add them.
UNSCAFFOLDED_WITH_SEAM: Dict[str, str] = {
    "anthropic": "The Anthropic Messages adapter",
}

#: Frameworks deliberately NOT offered: their adapters have no prompt seam, so
#: a generated file would trace correctly and deliver none of the agent's
#: skills — the exact silent failure this scaffold exists to prevent.
NO_PROMPT_SEAM: Dict[str, str] = {
    "llamaindex": "LlamaIndex",
    "claude-agent-sdk": "The Claude Agent SDK",
    "crewai": "CrewAI",
    "autogen": "AutoGen / AG2",
    "otel": "The generic OpenTelemetry rail",
    "adk": "Google ADK",
}


class UnknownFramework(ValueError):
    """`--framework` named something this command will not scaffold.

    Carries a message that says WHY, because the three reasons (typo,
    seam-but-no-template, no-seam-at-all) have three different fixes.
    """


class UnusableModel(ValueError):
    """`--model` named something this framework cannot resolve.

    Same contract as `UnknownFramework` and refused in the same place, one
    round trip before anything is written: a `--model` this framework will
    reject at run time produces a file that dies on the line building the
    agent, EVERY run, reported as `✓ Wrote agent.py`.
    """


def normalize_framework(name: str) -> str:
    """Canonicalize a `--framework` value, or raise `UnknownFramework`.

    Accepts the underscore spellings too (`openai_agents`), because that is
    what the SDK's own `init()` keyword is called and typing it is not a
    mistake worth punishing.
    """
    key = (name or "").strip().lower().replace("_", "-")
    if key in SUPPORTED_FRAMEWORKS:
        return key

    offered = " or ".join(f"--framework {f}" for f in SUPPORTED_FRAMEWORKS)
    if key in UNSCAFFOLDED_WITH_SEAM:
        raise UnknownFramework(
            f"{UNSCAFFOLDED_WITH_SEAM[key]} can deliver skills, but there is no "
            f"scaffold for it yet.\nUse {offered}, or copy the {key} snippet "
            f"from your agent's page in the dashboard."
        )
    if key in NO_PROMPT_SEAM:
        raise UnknownFramework(
            f"{NO_PROMPT_SEAM[key]} has no prompt seam for the skill loader, so "
            f"a generated file would trace correctly and deliver NONE of this "
            f"agent's skills — silently. That is worse than no scaffold, so this "
            f"command refuses to write one.\nUse {offered}. To trace {key} "
            f"without skills, copy its snippet from the dashboard instead."
        )
    supported = ", ".join(SUPPORTED_FRAMEWORKS)
    raise UnknownFramework(
        f"Unknown framework {name!r}. Supported: {supported}."
    )


# ── skill list rendering ─────────────────────────────────────────────


def _clean(text: Any, limit: int = 88) -> str:
    """Flatten a value to one safe comment line.

    Everything in the header is emitted as a `#` comment, which cannot be
    escaped out of except by a newline — so collapsing newlines (and other
    control characters) is the entire escaping story for prose. The
    functional bindings use `repr()` instead; see `_py`.
    """
    s = " ".join(str(text or "").split())
    s = "".join(ch for ch in s if ch.isprintable())
    if len(s) > limit:
        s = s[: limit - 1].rstrip() + "…"
    return s


def _py(value: str) -> str:
    """Embed a string in the generated source, escaped by Python itself.

    `repr()` rather than f-string interpolation, because an agent may legally
    be named `[Demo] support-agent` or contain a quote. Interpolating raw
    would emit a file that does not parse — or, worse, one that parses into
    something else.

    Prefers double quotes when they cost nothing, so the generated file is
    already what `ruff format` / `black` would produce and does not show up as
    a diff the first time someone runs their formatter. `repr()` still has the
    last word on anything it would have to escape.
    """
    s = str(value)
    r = repr(s)
    if r.startswith("'") and '"' not in s and "\\" not in r:
        return '"' + r[1:-1] + '"'
    return r


def _skill_comment_lines(skills: Sequence[Mapping[str, Any]]) -> List[str]:
    """The `# Skills …` block: what this agent will actually use.

    A zero-skill agent gets a sentence saying so rather than an empty list.
    A silently empty section is how someone concludes the scaffold is broken
    when the truth is that they have not attached anything yet.
    """
    if not skills:
        return [
            "# No skills are attached to this agent yet, so it will run as a plain",
            "# model call. Attach some in the dashboard — no need to regenerate this",
            "# file, it loads whatever is attached at run time.",
        ]

    lines = [f"# Skills this agent will use ({len(skills)}):"]
    for s in skills:
        name = _clean(s.get("skill_name") or s.get("name") or "(unnamed)", limit=60)
        desc = _clean(s.get("description"), limit=70)
        # A workspace-scoped subscription applies to EVERY agent in the org,
        # not to this one specifically. Saying so keeps the comment from
        # claiming a deliberate choice that was never made.
        suffix = "  [workspace-wide]" if s.get("scope") == "workspace" else ""
        lines.append(f"#   - {name}" + (f" — {desc}" if desc else "") + suffix)
    return lines


def _prompt_comment_lines(prompt: Optional["AgentConfig"]) -> List[str]:
    """The `# System prompt …` line: whether this agent has one, not what it says.

    Never the prompt TEXT. A copy pasted into the file is a second source of
    truth that goes stale the moment someone edits it in the dashboard — the
    exact problem `load_agent()` exists to remove — and a prompt can carry
    material nobody meant to commit.

    `None` means the scaffold could not read it (an older backend, a 5xx), and
    then this says nothing at all rather than guessing. The generated file
    fetches it at run time either way.
    """
    if prompt is None:
        return []
    if prompt.system_prompt is None:
        return [
            "#",
            "# This agent has no system prompt set. The file below runs anyway and",
            "# sends none — it will not invent one. Write one in the dashboard and it",
            "# takes effect on the next run; no need to regenerate this file.",
        ]
    version = (f", version {prompt.version_number}"
               if prompt.version_number is not None else "")
    return [
        "#",
        f"# System prompt: {len(prompt.system_prompt):,} characters{version}. Read at run",
        "# time by the load_agent() call below, so editing it in the dashboard changes",
        "# the next run — this file does not change and does not carry a copy.",
    ]


def _header_lines(
    agent_name: str,
    framework: str,
    skills: Sequence[Mapping[str, Any]],
    prompt: Optional["AgentConfig"] = None,
) -> List[str]:
    # _clean() before quoting, not after: this string lands inside a `#`
    # comment, so a name carrying a newline would end the comment and take
    # the rest of the line with it as code. shlex.quote guards shell
    # metacharacters, not Python line structure.
    cmd = (f"decimalai init {shlex.quote(_clean(agent_name, limit=70))}"
           f" --framework {framework}")
    lines = [
        f"# {_clean(agent_name, limit=70)} — generated by `{cmd}`.",
        "#",
    ]
    lines += _skill_comment_lines(skills)
    if skills:
        # Only says "that list" when there is a list. The empty-skills block
        # already makes the run-time point in its own words.
        lines += [
            "#",
            "# That list is a snapshot for your eyes only. The agent fetches its skills",
            "# from DecimalAI at run time, so changing what is attached in the dashboard",
            "# changes what this file does — you do not have to regenerate it.",
        ]
    lines += _prompt_comment_lines(prompt)
    lines += [
        "#",
        "# This file is yours now: edit it, commit it, rename it. Nothing runs on",
        "# DecimalAI's side.",
    ]
    return lines


def _init_call(base_url: Optional[str], *flags: str) -> str:
    """`decimalai.init(...)` — with a base_url only when one is needed.

    The key is read from the environment by `init()` itself, so no generated
    file ever contains one. A non-default base_url IS baked in: it is not a
    secret, and someone who scaffolded against a local or self-hosted backend
    gets a file that points at production otherwise.

    `flags` names init keywords to set True. Only the pydantic-ai template
    passes any (`otel`), and only because that adapter installs no exporter of
    its own — see point 4 in the module docstring. A template that can trace
    without one must not pass one: a second exporter double-sends every span.
    """
    args = []
    if base_url:
        args.append(f"base_url={_py(base_url)}")
    args += [f"{flag}=True" for flag in flags]
    return f"decimalai.init({', '.join(args)})"


# ── templates ────────────────────────────────────────────────────────


def _render_langchain(
    agent_name: str, model: str, base_url: Optional[str]
) -> List[str]:
    return [
        "import decimalai",
        "from decimalai.langchain import instrument",
        "from langchain.agents import create_agent",
        "from langchain.chat_models import init_chat_model",
        "from langchain_core.messages import HumanMessage",
        "",
        "# Change this to any chat model you have a key for, e.g.",
        '#   "gpt-4o-mini" · "anthropic:claude-sonnet-4-6" · "google_genai:gemini-3.5-flash"',
        "# (install that provider\'s package too: langchain-anthropic /",
        "#  langchain-google-genai)",
        f"MODEL = {_py(model)}",
        "",
        "# Reads DECIMAL_API_KEY from the environment. Never paste a key into a",
        "# file you are going to commit.",
        _init_call(base_url),
        "",
        "# Two things have to be true for a skill to reach the model on LangChain.",
        "# enable_skill_loader=True puts the skill MENU in the prompt. Delivering the",
        "# skill's BODY is separate: the DecimalAI LangChain adapter registers no",
        "# load_skill tool — it installs on the chat model, below any loop — so",
        "# prompt injection is the only body channel it has. That stays true no",
        "# matter how many tools you add to TOOLS below; the agent loop is yours,",
        "# not a channel the adapter can hand a skill body back through.",
        "# The SDK turns injection on by default for adapters with no load_skill",
        "# tool — pass init(inject_skill_body=False) if you want menu-only.",
        "#",
        "# Do not also pass langchain=True to init() above. That installs tracing a",
        "# first time with the loader off, and the repeat call cannot undo the",
        "# disk-mirror decision it already made.",
        f"instrument(agent_name={_py(agent_name)}, enable_skill_loader=True)",
        "",
        "# Your agent's system prompt, as it stands in the dashboard right now.",
        "# It is handed to you as data and sent by the line YOU can see below —",
        "# nothing injects it behind your back, so what this file sends is what",
        "# this file says. (Skills are the other way round: instrument() delivers",
        "# those for you, because a menu is additive and a system prompt is not.)",
        "#",
        "# Read at run time, so editing the prompt in the dashboard changes the",
        "# next run of this file. Nothing to redeploy, nothing to regenerate.",
        f"config = decimalai.load_agent({_py(agent_name)})",
        "",
        "# Your tools go here. An empty list is a WORKING agent, not a stub: the",
        "# loop runs, the skills rail delivers, and adding a tool later needs no",
        "# other change to this file.",
        "#",
        "# This has to be an agent loop and not a bare chat model, which is what",
        "# this template used to emit. `init_chat_model(MODEL)` on its own has no",
        "# loop, so the moment you bind a tool to it the model replies with",
        "# tool_calls and an EMPTY .content — nothing runs the tool, nothing feeds",
        "# the result back, and run() returns \"\". Measured on gpt-4o-mini,",
        "# 2026-08-28; the old template's own docstring invited exactly that.",
        "TOOLS: list = []",
        "",
        "# create_agent runs the tool loop and returns the final state. The system",
        "# prompt is handed to it BY THIS LINE, from the config read above — and",
        "# None (this agent has no prompt set, a real state; a failed read raises",
        "# instead) means it sends no system message at all, rather than one this",
        "# file invented.",
        "agent = create_agent(",
        "    init_chat_model(MODEL),",
        "    tools=TOOLS,",
        "    system_prompt=config.system_prompt,",
        ")",
        "",
        "",
        "def run(question: str) -> str:",
        '    """One turn: the loop runs until the model answers.',
        "",
        "    Add tools to TOOLS above — the skills rail is installed on the chat",
        "    model itself, so it stays wired through every turn of the loop.",
        '    """',
        "    # A HumanMessage object, not a (\"human\", \"...\") tuple: the skills rail",
        "    # reads the trailing human turn to route, and a tuple carries no role",
        "    # it can read — with tuples every call falls back to the full menu.",
        "    state = agent.invoke({\"messages\": [HumanMessage(content=question)]})",
        "    # The LAST message, which after the loop is the model's answer. An",
        "    # intermediate tool-call turn has empty content; this one does not.",
        "    return state[\"messages\"][-1].content",
        "",
        "",
        'if __name__ == "__main__":',
        '    print(run("What can you help me with?"))',
        "    # Short-lived scripts exit before the background sender drains. init()",
        "    # registers an atexit flush; this makes it explicit.",
        "    decimalai.flush()",
    ]


def _render_openai_agents(
    agent_name: str, model: str, base_url: Optional[str]
) -> List[str]:
    return [
        "import decimalai",
        "from agents import Agent, Runner",
        "from agents.exceptions import MaxTurnsExceeded",
        "from decimalai.openai_agents import instrument",
        "",
        "# Change this to any model your OpenAI key can reach.",
        f"MODEL = {_py(model)}",
        f"AGENT_NAME = {_py(agent_name)}",
        "# The SDK default is 10, which a skill-loading turn can legitimately",
        "# exceed. Raise it if your agent has many tools; lower it to fail faster.",
        "MAX_TURNS = 20",
        "",
        "# Reads DECIMAL_API_KEY from the environment. Never paste a key into a",
        "# file you are going to commit.",
        _init_call(base_url),
        "",
        "# enable_skill_loader=True is what actually delivers this agent's skills.",
        "# It defaults to False, and tracing alone does not deliver them. This",
        "# adapter owns a tool loop, so the model fetches each skill's body on",
        "# demand via the load_skill tool rather than having it injected.",
        "#",
        "# This call must stay ABOVE the Agent(...) below: the loader works by",
        "# wrapping Agent.__init__, and on builds without the run-time hooks an",
        "# agent constructed earlier silently receives nothing.",
        "#",
        "# Do not also pass openai_agents=True to init() above — that registers a",
        "# second trace processor and every trace is sent twice.",
        f"instrument(agent_name={_py(agent_name)}, enable_skill_loader=True)",
        "",
        "# Your agent's system prompt, as it stands in the dashboard right now.",
        "# It is handed to you as data and passed to the Agent by the line YOU can",
        "# see below — nothing injects it behind your back, so what this file sends",
        "# is what this file says. (Skills are the other way round: instrument()",
        "# delivers those for you, because a menu is additive and a prompt is not.)",
        "#",
        "# Read at run time, so editing the prompt in the dashboard changes the",
        "# next run of this file. Nothing to redeploy, nothing to regenerate.",
        f"config = decimalai.load_agent({_py(agent_name)})",
        "",
        "agent = Agent(",
        f"    name={_py(agent_name)},",
        "    model=MODEL,",
        "    # None when this agent has no prompt set — a real state, not a failure",
        "    # (a failed read raises instead). The Agent then runs with no",
        "    # instructions rather than a default this file made up.",
        "    instructions=config.system_prompt,",
        ")",
        "",
        "",
        "def run(question: str) -> str:",
        '    """One turn.',
        "",
        "    Add tools or handoffs to the Agent above — the skills rail rides on",
        "    the agent's instructions, so it survives both.",
        '    """',
        "    try:",
        "        return Runner.run_sync(agent, question, max_turns=MAX_TURNS).final_output",
        "    except MaxTurnsExceeded:",
        "        # The model kept calling tools without answering. Usually it is",
        "        # chasing a tool the prompt names but this file does not define —",
        "        # check the Tools line in your system prompt against the tools",
        "        # actually attached to the Agent above.",
        "        return (",
        "            f\"[{AGENT_NAME}] stopped after {MAX_TURNS} turns without an answer. \"",
        "            \"See the comment in run().\"",
        "        )",
        "",
        "",
        'if __name__ == "__main__":',
        '    print(run("What can you help me with?"))',
        "    # Short-lived scripts exit before the background sender drains. init()",
        "    # registers an atexit flush; this makes it explicit.",
        "    decimalai.flush()",
    ]


def _render_pydantic_ai(
    agent_name: str, model: str, base_url: Optional[str]
) -> List[str]:
    return [
        "import decimalai",
        "from decimalai.pydantic_ai import instrument",
        "from pydantic_ai import Agent",
        "from pydantic_ai.exceptions import UsageLimitExceeded",
        "from pydantic_ai.usage import UsageLimits",
        "",
        "# Pydantic AI resolves a PROVIDER-QUALIFIED model string and refuses a",
        "# bare one, so the provider half is not optional here. Any of:",
        '#   "openai:gpt-4o-mini" · "google:gemini-3.6-flash"',
        '#   "anthropic:claude-sonnet-4-6"',
        "# all three ship with the pydantic-ai install; the key each one reads is",
        "# named in the `Set:` list `decimalai init` printed for your model.",
        f"MODEL = {_py(model)}",
        f"AGENT_NAME = {_py(agent_name)}",
        "# Pydantic AI's own default is 50 model requests. Lower, because a",
        "# runaway loop here costs real tokens; raise it if your agent legitimately",
        "# chains many tool calls.",
        "MAX_REQUESTS = 20",
        "",
        "# Reads DECIMAL_API_KEY from the environment. Never paste a key into a",
        "# file you are going to commit.",
        "#",
        "# otel=True is what makes this agent's traces exist. Unlike the LangChain",
        "# and OpenAI-Agents adapters, decimalai.pydantic_ai installs no exporter:",
        "# Pydantic AI emits no spans of its own, so that adapter ships the skills",
        "# rail and the run boundary and leaves the tracing to whatever you pair.",
        "# A bare init() here would run fine, deliver every skill, and leave your",
        "# traces page empty forever.",
        _init_call(base_url, "otel"),
        "",
        "# Pydantic AI's own OpenTelemetry instrumentation, which is where the",
        "# spans come from. It is a global switch and it is provider-agnostic —",
        "# swapping MODEL to a Gemini or Claude string changes nothing here.",
        "#",
        "# Keep it BELOW init(): init(otel=True) is what installs the DecimalAI",
        "# exporter on the process's tracer provider, and these spans have nowhere",
        "# to go until it has.",
        "Agent.instrument_all()",
        "",
        "# enable_skill_loader=True is what actually delivers this agent's skills.",
        "# It defaults to False, and tracing alone does not deliver them. This",
        "# adapter registers a real load_skill tool on every Agent, and Pydantic AI",
        "# owns its tool loop — so the model reads the menu, asks for the one it",
        "# needs, and the body comes back mid-turn.",
        "#",
        "# agent_name is the fallback for an Agent built without a name=. Pydantic",
        "# AI infers one from the VARIABLE it was assigned to in that case, so",
        "# without this line an agent you renamed files its traces under 'agent'.",
        "#",
        "# This call must stay ABOVE the Agent(...) below: the loader works by",
        "# wrapping Agent.__init__, so an Agent constructed earlier gets neither",
        "# the skills prompt nor the load_skill tool.",
        f"instrument(agent_name={_py(agent_name)}, enable_skill_loader=True)",
        "",
        "# Your agent's system prompt, as it stands in the dashboard right now.",
        "# It is handed to you as data and passed to the Agent by the line YOU can",
        "# see below — nothing injects it behind your back, so what this file sends",
        "# is what this file says. (Skills are the other way round: instrument()",
        "# delivers those for you, because a menu is additive and a prompt is not.)",
        "#",
        "# Read at run time, so editing the prompt in the dashboard changes the",
        "# next run of this file. Nothing to redeploy, nothing to regenerate.",
        f"config = decimalai.load_agent({_py(agent_name)})",
        "",
        "# Your tools go here. An empty list is a WORKING agent, not a stub: the",
        "# loop runs, the skills rail delivers, and adding a tool later needs no",
        "# other change to this file.",
        "TOOLS: list = []",
        "",
        "agent = Agent(",
        "    MODEL,",
        "    # Bound, not inferred. Pydantic AI reads the name off the variable",
        "    # this Agent is assigned to when you leave it out, and the DecimalAI",
        "    # run scope stamps whatever it finds onto the trace — so an unnamed",
        "    # Agent files under 'agent' and your dashboard page stays empty.",
        "    name=AGENT_NAME,",
        "    # `or ()` because None is a REAL state (this agent has no prompt set;",
        "    # a failed read raises instead) and Pydantic AI's system_prompt takes a",
        "    # string or a sequence, never None. An empty tuple sends no system",
        "    # prompt at all, rather than one this file invented.",
        "    system_prompt=config.system_prompt or (),",
        "    tools=TOOLS,",
        ")",
        "",
        "",
        "def run(question: str) -> str:",
        '    """One turn: the loop runs until the model answers.',
        "",
        "    Add tools to TOOLS above — the skills rail rides on the Agent's own",
        "    system prompt and its load_skill tool, so it survives them.",
        '    """',
        "    try:",
        "        return agent.run_sync(",
        "            question, usage_limits=UsageLimits(request_limit=MAX_REQUESTS),",
        "        ).output",
        "    except UsageLimitExceeded:",
        "        # The model kept calling tools without answering. Usually it is",
        "        # chasing a tool the prompt names but this file does not define —",
        "        # check the Tools line in your system prompt against TOOLS above.",
        "        return (",
        "            f\"[{AGENT_NAME}] stopped after {MAX_REQUESTS} model requests \"",
        '            "without an answer. See the comment in run()."',
        "        )",
        "",
        "",
        'if __name__ == "__main__":',
        '    print(run("What can you help me with?"))',
        "    # Short-lived scripts exit before the background sender drains. init()",
        "    # registers an atexit flush; this makes it explicit.",
        "    decimalai.flush()",
    ]


_RENDERERS = {
    "langchain": _render_langchain,
    "openai-agents": _render_openai_agents,
    "pydantic-ai": _render_pydantic_ai,
}

#: What the user has to install and export for the generated file to run.
#: `decimalai[openai]` pulls the openai SDK, NOT the Agents SDK — they are
#: separate extras, and copy-pasting the wrong one leaves the import failing on
#: a machine that looks correctly set up.
INSTALL: Dict[str, str] = {
    "langchain": 'pip install "decimalai[langchain]" langchain langchain-openai',
    "openai-agents": 'pip install "decimalai[openai-agents]"',
    # One line, whatever the model. The `pydantic-ai` distribution (which is what
    # `decimalai[pydantic-ai]` resolves to — not `pydantic-ai-slim`) bundles the
    # openai, google and anthropic provider bindings, and the OpenTelemetry SDK
    # the template's tracing needs is already a core dependency of decimalai.
    # A provider outside those three adds a token; see _PYDANTIC_AI_PROVIDERS.
    "pydantic-ai": 'pip install "decimalai[pydantic-ai]"',
}

ENV_VARS: Dict[str, tuple] = {
    "langchain": ("DECIMAL_API_KEY", "OPENAI_API_KEY"),
    "openai-agents": ("DECIMAL_API_KEY", "OPENAI_API_KEY"),
    "pydantic-ai": ("DECIMAL_API_KEY", "OPENAI_API_KEY"),
}

#: LangChain's `init_chat_model` takes a `provider:model` string. The provider
#: half decides which package to install and which key to export — so the
#: framework alone does not determine either.
_PROVIDER_REQUIREMENTS: Dict[str, tuple] = {
    # provider prefix -> (extra pip packages, model-provider env var)
    "openai": ("langchain-openai", "OPENAI_API_KEY"),
    "anthropic": ("langchain-anthropic", "ANTHROPIC_API_KEY"),
    "google_genai": ("langchain-google-genai", "GOOGLE_API_KEY"),
    "google_vertexai": ("langchain-google-vertexai", "GOOGLE_API_KEY"),
    "groq": ("langchain-groq", "GROQ_API_KEY"),
    "mistralai": ("langchain-mistralai", "MISTRAL_API_KEY"),
    "cohere": ("langchain-cohere", "COHERE_API_KEY"),
    "fireworks": ("langchain-fireworks", "FIREWORKS_API_KEY"),
    "together": ("langchain-together", "TOGETHER_API_KEY"),
    "ollama": ("langchain-ollama", ""),  # local, no key
}


#: Pydantic AI's own provider prefixes — a DIFFERENT vocabulary from LangChain's
#: above, which is why this is a second table rather than a shared one:
#: LangChain spells Gemini `google_genai` and reads `GOOGLE_API_KEY`, Pydantic AI
#: spells it `google` and reads `GEMINI_API_KEY` first
#: (`pydantic_ai.providers.google`, checked on 2.36). Sharing one table would
#: have printed a key the run does not read for exactly the model the fleet uses.
#:
#: The extra is empty for the three providers the `pydantic-ai` distribution
#: already carries; the rest are `pydantic-ai-slim` extras it does not.
_PYDANTIC_AI_PROVIDERS: Dict[str, tuple] = {
    # provider prefix -> (extra pip requirement, model-provider env var)
    "openai": ("", "OPENAI_API_KEY"),
    "openai-chat": ("", "OPENAI_API_KEY"),
    "openai-responses": ("", "OPENAI_API_KEY"),
    "azure": ("", "AZURE_OPENAI_API_KEY"),
    "google": ("", "GEMINI_API_KEY"),
    # The spelling every pydantic-ai before 2.x used for the same provider.
    # Accepted so a model string copied from an older project prints the right
    # key rather than silently falling back to OpenAI's.
    "google-gla": ("", "GEMINI_API_KEY"),
    "google-vertex": ("", "GOOGLE_API_KEY"),
    "anthropic": ("", "ANTHROPIC_API_KEY"),
    "groq": ('"pydantic-ai[groq]"', "GROQ_API_KEY"),
    "mistral": ('"pydantic-ai[mistral]"', "MISTRAL_API_KEY"),
    "cohere": ('"pydantic-ai[cohere]"', "CO_API_KEY"),
    "huggingface": ('"pydantic-ai[huggingface]"', "HF_TOKEN"),
    "openrouter": ('"pydantic-ai[openrouter]"', "OPENROUTER_API_KEY"),
    "deepseek": ("", "DEEPSEEK_API_KEY"),
    "ollama": ("", ""),  # local, no key
}


def model_provider(model: str) -> str:
    """The provider half of a `provider:model` string; "openai" when bare.

    `init_chat_model("gpt-4o-mini")` infers OpenAI from the model name, so a
    bare string means OpenAI in practice.
    """
    raw = str(model or "")
    return raw.split(":", 1)[0].strip().lower() if ":" in raw else "openai"


def normalize_model(framework: str, model: Optional[str]) -> str:
    """This framework's MODEL line for `--model`, or raise `UnusableModel`.

    Only pydantic-ai has anything to check, and it checks the one thing that is
    fatal rather than merely suboptimal. `pydantic_ai.models.infer_model` splits
    on `provider:model` and raises `UserError: Unknown model` when there is no
    provider half — so `--model gpt-4o-mini --framework pydantic-ai` writes a
    file whose `Agent(MODEL, ...)` line cannot run, while `decimalai init`
    reports success.

    Not rewritten to `openai:gpt-4o-mini` silently. A model string is a choice
    with a bill attached, and a command that quietly picks a different provider
    than the one asked for is the same class of surprise as a scaffold that
    delivers no skills. The message names the corrected string; the user types it.

    `"test"` is exempt because `infer_model` exempts it — it is Pydantic AI's own
    built-in `TestModel`, and refusing it would be this function inventing a rule
    the framework does not have.

    STRICTER THAN THE OLDEST SUPPORTED PYDANTIC AI, deliberately, and this is the
    one place that is worth stating. `pydantic-ai` 0.1.0 (the floor
    `decimalai[pydantic-ai]` declares) still inferred OpenAI from a bare name;
    2.x removed that. So on an ancient pin this refuses something that would have
    run — but the fix it names, `openai:<model>`, resolves on BOTH (checked by
    running `infer_model` under 0.1.0 and 2.36), so nobody is ever pushed off a
    working configuration and onto a broken one. The alternative, gating on the
    installed version, would make `decimalai init` behave differently on two
    machines with the same command.
    """
    model = (model or "").strip() or default_model(framework)
    if framework != "pydantic-ai" or model == "test" or ":" in model:
        return model
    raise UnusableModel(
        f"Pydantic AI needs a provider-qualified model, and {model!r} has no "
        f"provider half — `Agent({model!r})` raises `UserError: Unknown model` "
        f"on the line that builds the agent.\n"
        f"Try --model openai:{model}, or name the provider you meant "
        f"(google:… · anthropic:…)."
    )


def default_model(framework: str) -> str:
    """The MODEL line this framework gets when `--model` was not passed.

    Not a single constant, because `DEFAULT_MODEL` is an OpenAI model NAME and
    Pydantic AI wants a provider-qualified IDENTIFIER. The two differ by a
    prefix and the difference is fatal rather than cosmetic: `Agent("gpt-4o-mini")`
    raises `UserError: Unknown model` on the line that builds the agent.
    """
    return _FRAMEWORK_DEFAULT_MODELS.get(framework, DEFAULT_MODEL)


def install_command(framework: str, model: Optional[str] = None) -> str:
    """The pip line for this (framework, model) pair.

    Keyed on the pair, not on the framework: until 2026-08-28 this was a dict
    on framework alone, so `--model anthropic:...` printed
    `pip install langchain-openai` and `export OPENAI_API_KEY` — the wrong
    package and the wrong key, while the key the user actually needed went
    unmentioned.

    `model=None` means "whatever this framework's default is", which is what the
    CLI passes when `--model` was not given.
    """
    model = model or default_model(framework)
    if framework == "langchain":
        package, _ = _PROVIDER_REQUIREMENTS.get(
            model_provider(model), _PROVIDER_REQUIREMENTS["openai"]
        )
        return f'pip install "decimalai[langchain]" langchain {package}'
    if framework == "pydantic-ai":
        extra, _ = _PYDANTIC_AI_PROVIDERS.get(
            model_provider(model), _PYDANTIC_AI_PROVIDERS["openai"]
        )
        return INSTALL[framework] + (f" {extra}" if extra else "")
    return INSTALL[framework]


def env_vars(framework: str, model: Optional[str] = None) -> tuple:
    """The env vars this (framework, model) pair needs, DECIMAL_API_KEY first."""
    model = model or default_model(framework)
    table = {
        "langchain": _PROVIDER_REQUIREMENTS,
        "pydantic-ai": _PYDANTIC_AI_PROVIDERS,
    }.get(framework)
    if table is None:
        return ENV_VARS[framework]
    _, key = table.get(model_provider(model), table["openai"])
    return ("DECIMAL_API_KEY", key) if key else ("DECIMAL_API_KEY",)


def render_agent_file(
    agent_name: str,
    framework: str = "langchain",
    skills: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    prompt: Optional["AgentConfig"] = None,
) -> str:
    """Return the complete source of a runnable agent file.

    Pure: no I/O, no clock, no randomness. Given the same inputs it returns
    the same bytes, which is what lets the test suite compile the output
    instead of pattern-matching a string.

    Args:
        agent_name: The agent as it exists on DecimalAI. Bound into
            `instrument()` so traces file against it and its skills resolve.
        framework: Already normalized by `normalize_framework`.
        skills: Rows from `GET /api/v1/agents/{name}/skills`. Named in a
            comment so the file shows what it will use.
        model: Overrides `default_model(framework)` on the generated MODEL line.
        base_url: Baked in only when it is not the hosted default.
        prompt: The agent's prompt config, from `client.get_agent_prompt`.
            COMMENT ONLY — it decides one header line about whether a prompt
            exists, never what the file sends. The generated file always reads
            the prompt itself at run time, so passing None (the scaffold could
            not read it) costs a comment and changes no behaviour.
    """
    framework = normalize_framework(framework)
    if not str(agent_name).strip():
        raise ValueError("agent_name is required")

    body = _RENDERERS[framework](
        agent_name, normalize_model(framework, model), base_url
    )
    lines = (
        _header_lines(agent_name, framework, skills or [], prompt) + [""] + body
    )
    return "\n".join(lines) + "\n"
