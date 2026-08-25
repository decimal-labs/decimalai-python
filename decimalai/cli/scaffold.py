"""Render a runnable agent file from an agent's DecimalAI configuration.

`decimalai init <agent-name>` closes the gap between "created an agent in the
dashboard" and "an agent is running": the dashboard stores a name, a
description and a set of skills, and until now the user still had to write the
agent themselves. This module turns that stored configuration into a file that
runs.

We generate; they run. Nothing here executes the agent, ships a runtime, or
phones home at run time beyond the SDK the file imports — the model is
`create-next-app`, not hosting.

THREE THINGS THIS TEMPLATE EXISTS TO GET RIGHT
----------------------------------------------

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

One difference is intentional: the openai-agents snippet awaits
`Runner.run(...)` because it is being pasted into an async app, while the
generated file is a standalone script and uses `Runner.run_sync(...)`.

WHY ONLY TWO FRAMEWORKS
-----------------------
A scaffold that silently delivers no skills is worse than no scaffold, so a
framework is only offered here if its adapter has a prompt seam the skill
loader can use. `enable_skill_loader` exists on exactly four adapters
(langchain, openai_agents, anthropic, pydantic_ai); the rest — llamaindex,
claude_agent_sdk, crewai/autogen/otel, adk — trace and version but have no
loader, and generating a file for them would hand someone a program that looks
correct and quietly ignores every skill they picked. `--framework` names the
reason rather than printing a bare "invalid choice".
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

if TYPE_CHECKING:  # keeps this module pure — it imports nothing at run time
    from .._agent import AgentConfig

# Changing this is a one-line edit in the generated file; it lives here so the
# CLI's --model default and the template cannot drift apart.
DEFAULT_MODEL = "gpt-4o-mini"

DEFAULT_OUTPUT = "agent.py"

#: Frameworks `init` can scaffold, in the order they are offered.
SUPPORTED_FRAMEWORKS: tuple = ("langchain", "openai-agents")

#: Frameworks whose adapter carries `enable_skill_loader` but which have no
#: template yet. Named separately from the seamless ones because the answer to
#: "why not?" is different, and so is what we'd have to do to add them.
UNSCAFFOLDED_WITH_SEAM: Dict[str, str] = {
    "anthropic": "The Anthropic Messages adapter",
    "pydantic-ai": "The Pydantic AI adapter",
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


def _init_call(base_url: Optional[str]) -> str:
    """`decimalai.init(...)` — with a base_url only when one is needed.

    The key is read from the environment by `init()` itself, so no generated
    file ever contains one. A non-default base_url IS baked in: it is not a
    secret, and someone who scaffolded against a local or self-hosted backend
    gets a file that points at production otherwise.
    """
    if base_url:
        return f"decimalai.init(base_url={_py(base_url)})"
    return "decimalai.init()"


# ── templates ────────────────────────────────────────────────────────


def _render_langchain(
    agent_name: str, model: str, base_url: Optional[str]
) -> List[str]:
    return [
        "import decimalai",
        "from decimalai.langchain import instrument",
        "from langchain.chat_models import init_chat_model",
        "from langchain_core.messages import HumanMessage, SystemMessage",
        "",
        "# Change this to any chat model you have a key for, e.g.",
        '#   "gpt-4o-mini" · "anthropic:claude-sonnet-4-5" · "google_genai:gemini-2.5-flash"',
        "# (install that provider's package too: pip install langchain-anthropic)",
        f"MODEL = {_py(model)}",
        "",
        "# Reads DECIMAL_API_KEY from the environment. Never paste a key into a",
        "# file you are going to commit.",
        _init_call(base_url),
        "",
        "# enable_skill_loader=True is what actually delivers this agent's skills.",
        "# It defaults to False, and tracing alone does not deliver them: without",
        "# this flag the model is handed a list of skill titles it cannot read.",
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
        "# None means this agent has no prompt set — a real state, not a failure",
        "# (a failed read raises instead). Then we send no system message at all,",
        "# rather than invent one and put words in the agent's mouth.",
        "PREAMBLE = (",
        "    [SystemMessage(content=config.system_prompt)] if config.system_prompt",
        "    else []",
        ")",
        "",
        "agent = init_chat_model(MODEL)",
        "",
        "",
        "def run(question: str) -> str:",
        '    """One turn.',
        "",
        "    Add tools, memory or a graph here — the skills rail is already wired",
        "    and stays wired, because it is installed on the chat model itself.",
        '    """',
        "    # Message objects, not (\"system\", \"...\") tuples: the skills rail reads",
        "    # the human turn off this list to route, and a tuple carries no role",
        "    # it can read — with tuples every call falls back to the full menu.",
        "    return agent.invoke([*PREAMBLE, HumanMessage(content=question)]).content",
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
        "from decimalai.openai_agents import instrument",
        "",
        "# Change this to any model your OpenAI key can reach.",
        f"MODEL = {_py(model)}",
        "",
        "# Reads DECIMAL_API_KEY from the environment. Never paste a key into a",
        "# file you are going to commit.",
        _init_call(base_url),
        "",
        "# enable_skill_loader=True is what actually delivers this agent's skills.",
        "# It defaults to False, and tracing alone does not deliver them.",
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
        "    return Runner.run_sync(agent, question).final_output",
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
}

#: What the user has to install and export for the generated file to run.
#: `decimalai[openai]` pulls the openai SDK, NOT the Agents SDK — they are
#: separate extras, and copy-pasting the wrong one leaves the import failing on
#: a machine that looks correctly set up.
INSTALL: Dict[str, str] = {
    "langchain": 'pip install "decimalai[langchain]" langchain langchain-openai',
    "openai-agents": 'pip install "decimalai[openai-agents]"',
}

ENV_VARS: Dict[str, tuple] = {
    "langchain": ("DECIMAL_API_KEY", "OPENAI_API_KEY"),
    "openai-agents": ("DECIMAL_API_KEY", "OPENAI_API_KEY"),
}


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
        model: Overrides `DEFAULT_MODEL` on the generated MODEL line.
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

    body = _RENDERERS[framework](agent_name, model or DEFAULT_MODEL, base_url)
    lines = (
        _header_lines(agent_name, framework, skills or [], prompt) + [""] + body
    )
    return "\n".join(lines) + "\n"
