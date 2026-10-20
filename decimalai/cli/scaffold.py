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
   line they can see. On langchain and adk skills go the other way
   (instrument() injects them), because a skill menu is additive and a system
   prompt is not — silently replacing one means their repo says "Never issue
   refunds over $500" while the model receives something else, which is
   unfixable from their side. On openai-agents and pydantic-ai nothing is
   injected: `load_agent()` returns the prompt and the skills together, and
   the file hands both to its own `Agent(...)`.

   `load_agent()` runs at process start, so the no-redeploy property survives:
   edit the prompt in the dashboard, restart, done. And when an agent has NO
   prompt set (a real state — it is optional at creation) the file still runs
   and sends nothing rather than inventing one.

1. `enable_skill_loader=True`. Every adapter's `instrument()` defaults it to
   False (decimalai/langchain.py, decimalai/openai_agents.py, …), so a file
   that calls `instrument()` without it traces beautifully and delivers NONE of
   the agent's skills. The failure is silent and it is the whole reason the
   scaffold is worth shipping: without the flag the model is handed a list of
   skill titles it cannot read. Every generated file whose adapter has a
   loader carries it, and `tests/test_cli_init_scaffold.py` fails the build if
   one stops. The openai-agents (0.18+) and pydantic-ai (0.19+) files have no
   loader to turn on: their skills reach the model only through the
   `load_agent()` pieces they pass to `Agent(...)`, which the same tests check.
   (`init(langchain=True)`, `init(openai_agents=True)`, `init(adk=True)` and
   `init(crewai=True)` turn the loader on themselves since 2026-10-14; the
   templates keep the explicit `instrument()` line because it is where the
   adapter's other options go.)

2. The agent name is BOUND, and written ONCE. An unbound name means the
   adapter auto-detects one off the runnable ("langchain-agent" when it cannot
   tell), so traces are filed under a different agent than the one the user
   configured — their page stays empty forever while traces pile up somewhere
   else. Same bug the dashboard snippets were fixed for on 2026-08-22.

   Written once because a second copy is the same bug on a delay. Until
   2026-10-19 the name was a literal in `instrument(agent_name=...)`, in
   `load_agent(...)` and in the framework's own `Agent(name=...)`, beside an
   `AGENT_NAME` constant only error strings read — so renaming the agent meant
   finding every copy, and a missed one filed traces under the old name with no
   error. Now each file assigns `AGENT_NAME` once and passes it to
   `init(agent_name=AGENT_NAME)`, which every adapter's `instrument()` falls
   back to (`decimalai/_config.py` `default_agent_name`); the other uses read
   the constant. `tests/test_cli_init_scaffold.py` fails if the literal
   appears twice.

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
   comes from, so the install line does not grow. `decimalai[langchain]`
   floors `langchain-core>=1.6.10`, so the `langchain` that resolves beside it
   is a 1.x release, which is where `create_agent` lives.

4. TRACING IS NOT AUTOMATIC ON EVERY ADAPTER, and the pydantic-ai template is
   where that bites. `decimalai init` ends by printing "The trace appears at →
   /agents/<name>", which is a promise about the file it just wrote. The
   langchain and openai-agents adapters keep it themselves — their `instrument()`
   installs a callback handler / a trace processor. `decimalai.pydantic_ai`
   deliberately does not: its `load_agent()` capability ships the prompt, the
   skills and the run BOUNDARY (one DecimalAI run span per run) and leaves the
   span content to Pydantic AI's own instrumentation or a provider instrumentor. `init()` with no flags installs no
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
   `decimalai.pydantic_ai.load_agent()` on purpose: turning it on inside the
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
    `instrument(enable_skill_loader=True)`. `init(langchain=True)` is the same
    install, so passing it as well installs twice: the first one wins, and the
    explicit call's other arguments are dropped with a warning.
  * openai-agents: no `instrument()` at all. `load_agent(AGENT_NAME)`'s
    instructions and tools go into the file's own `Agent(...)`, and init()
    carries no framework flag.
  * pydantic-ai: `load_agent(AGENT_NAME)` passed as `Agent(capabilities=[support])`,
    with `init(..., otel=True)` and `Agent.instrument_all()` for the spans.
    That is the same rule: exactly one span source per file. On the other two
    the adapter IS the span source; here it is not (point 4).

Two differences are intentional. The snippets name the agent on
`instrument(agent_name=...)`; the templates name it on `init(agent_name=...)`
(point 2). It is the same binding — `init()`'s name is the one every
`instrument()` falls back to — moved to the one call that reaches every
adapter, so a whole file can name its agent once. The openai-agents snippet awaits
`Runner.run(...)` because it is being pasted into an async app, while the
generated file is a standalone script and uses `Runner.run_sync(...)`.

WHY ONLY THESE FRAMEWORKS
-------------------------
A scaffold that silently delivers no skills is worse than no scaffold, so a
framework is only offered here if its adapter has a prompt seam the skill
loader can use. Six adapters have one: `enable_skill_loader` on four
(langchain, anthropic, adk, crewai) and a `load_agent()` binding on two
(openai_agents, pydantic_ai). The rest — llamaindex, claude_agent_sdk,
autogen/otel — trace and version but have no loader, and generating a file for
them would hand someone a program that looks correct and quietly ignores every
skill they picked. `--framework` names the reason rather than printing a bare
"invalid choice".

Of the six, four are scaffolded. `anthropic` is a holdout and the reason is
not "nobody got to it": that adapter patches one `messages.create()` call, so it
owns no tool loop — a generated file there would be a hand-rolled while-loop this
template would have to invent, and the body could only arrive by injection.
`crewai` is the other, and that one IS "nobody got to it yet": its loader
shipped on 2026-10-08, ahead of a template.
"""

from __future__ import annotations

import re
import shlex
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .. import _frameworks as _registry

if TYPE_CHECKING:  # keeps this module pure — at run time it imports only the registry
    from .._agent import SavedPrompt

# ── what this command knows about each framework ─────────────────────
#
# Every per-framework fact below is DERIVED from `decimalai/_frameworks.py`, the one
# registry the docs' framework tables and the dashboard's snippets are generated from
# too. Edit a fact there, not here; the reasons behind each value (why pydantic-ai's
# default model carries a provider prefix, why adk's env var is GOOGLE_API_KEY, when
# crewai and adk changed category) are written beside it there.

# Changing this is a one-line edit in the generated file; the registry keeps it so the
# CLI's --model default and the template cannot drift apart.
DEFAULT_MODEL = _registry.DEFAULT_MODEL

#: Per-framework overrides of `DEFAULT_MODEL`, for frameworks whose model
#: identifier is not the bare OpenAI name (pydantic-ai's `provider:model`, adk's Gemini).
_FRAMEWORK_DEFAULT_MODELS: Dict[str, str] = {
    f.key: f.default_model for f in _registry.FRAMEWORKS
    if f.default_model and f.default_model != DEFAULT_MODEL
}

DEFAULT_OUTPUT = "agent.py"

#: Frameworks `init` can scaffold, in the order they are offered.
SUPPORTED_FRAMEWORKS: tuple = tuple(
    f.key for f in _registry.FRAMEWORKS if f.scaffold == _registry.TEMPLATE
)

#: Frameworks whose adapter carries `enable_skill_loader` but which have no
#: template yet. Named separately from the seamless ones because the answer to
#: "why not?" is different, and so is what we'd have to do to add them.
UNSCAFFOLDED_WITH_SEAM: Dict[str, str] = {
    f.key: f.cli_label for f in _registry.FRAMEWORKS
    if f.scaffold == _registry.SEAM_NO_TEMPLATE
}

#: Frameworks deliberately NOT offered: their adapters have no prompt seam, so
#: a generated file would trace correctly and deliver none of the agent's
#: skills — the exact silent failure this scaffold exists to prevent.
NO_PROMPT_SEAM: Dict[str, str] = {
    f.key: f.cli_label for f in _registry.FRAMEWORKS
    if f.scaffold == _registry.NO_SEAM
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
    """The `# Skills …` block: what is attached to this agent, and what else
    it is offered.

    `skills` is `GET /api/v1/agents/{name}/skills`: subscription rows, the ones
    made for this agent and the workspace-wide ones. It is NOT everything the
    agent is offered. Every skill the workspace owns reaches every agent with
    no row at all, unless its "Offered to" setting is "Only agents it's
    assigned to" — so a skill created a moment ago is offered and is not in
    this list. The one endpoint that answers with the whole offered set,
    `GET /api/v1/skills/menu`, records a routing decision on every call, and a
    scaffold is not a run: calling it here would count an offer of every skill
    that no run ever made. So the list stays what it is, labelled as such, and
    the rest is said in a sentence.

    A zero-row agent gets that sentence rather than an empty list. A silently
    empty section is how someone concludes the scaffold is broken.
    """
    if not skills:
        return [
            "# No skills are attached to this agent itself. It is still offered the",
            "# skills this workspace owns: by default every one of them, including",
            "# one created a moment ago, unless its \"Offered to\" setting is \"Only",
            "# agents it's assigned to\". The file loads its skills at run time, so",
            "# attaching more in the dashboard needs no regenerating.",
        ]

    lines = [f"# Skills attached to this agent ({len(skills)}):"]
    for s in skills:
        name = _clean(s.get("skill_name") or s.get("name") or "(unnamed)", limit=60)
        desc = _clean(s.get("description"), limit=70)
        # A workspace-scoped subscription applies to EVERY agent in the org,
        # not to this one specifically. Saying so keeps the comment from
        # claiming a deliberate choice that was never made.
        suffix = "  [workspace-wide]" if s.get("scope") == "workspace" else ""
        lines.append(f"#   - {name}" + (f" — {desc}" if desc else "") + suffix)
    lines += [
        "#",
        "# It is also offered the skills this workspace owns: by default every one",
        "# of them, including one created a moment ago, unless its \"Offered to\"",
        "# setting is \"Only agents it's assigned to\". Only attached skills are",
        "# listed above.",
    ]
    return lines


def _prompt_comment_lines(prompt: Optional["SavedPrompt"]) -> List[str]:
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
    prompt: Optional["SavedPrompt"] = None,
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


def _agent_name_lines(agent_name: str, *, load_agent_routes: bool = False) -> List[str]:
    """The one place a generated file writes the agent's name (point 2).

    ``load_agent_routes``: the template's ``load_agent()`` call brings the
    skills as well as the prompt (OpenAI Agents, Pydantic AI), so init() only
    traces.
    """
    second = (
        "# to everything that traces, and load_agent() reads its prompt and skills."
        if load_agent_routes else
        "# to everything that traces or routes skills, and load_agent() reads its prompt."
    )
    return [
        "# The DecimalAI agent this file runs as. Named once, here: init() passes it",
        second,
        f"AGENT_NAME = {_py(agent_name)}",
    ]


def _init_call(base_url: Optional[str], *flags: str) -> str:
    """`decimalai.init(agent_name=AGENT_NAME, ...)` — with a base_url only when
    one is needed.

    `agent_name=AGENT_NAME` names the agent for every adapter's `instrument()`,
    so the template's `instrument()` line does not repeat it (point 2).

    The key is read from the environment by `init()` itself, so no generated
    file ever contains one. A non-default base_url IS baked in: it is not a
    secret, and someone who scaffolded against a local or self-hosted backend
    gets a file that points at production otherwise.

    `flags` names init keywords to set True. Only the pydantic-ai template
    passes any (`otel`), and only because that adapter installs no exporter of
    its own — see point 4 in the module docstring. A template that can trace
    without one must not pass one: a second exporter double-sends every span.
    """
    args = ["agent_name=AGENT_NAME"]
    if base_url:
        args.append(f"base_url={_py(base_url)}")
    args += [f"{flag}=True" for flag in flags]
    return f"decimalai.init({', '.join(args)})"


# ── templates ────────────────────────────────────────────────────────


def _render_langchain(
    agent_name: str, model: str, base_url: Optional[str]
) -> List[str]:
    return [
        "import os",
        "",
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
        *_agent_name_lines(agent_name),
        "",
        *_key_guard_lines("langchain", model),
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
        "# No agent_name here: init() above already named the agent. Do not also",
        "# pass langchain=True to init(). That is this same install, and the first",
        "# one wins: options you add to this call later would be dropped.",
        "instrument(enable_skill_loader=True)",
        "",
        "# Your agent's system prompt, as it stands in the dashboard right now.",
        "# It is handed to you as data and sent by the line YOU can see below —",
        "# nothing injects it behind your back, so what this file sends is what",
        "# this file says. (Skills are the other way round: instrument() delivers",
        "# those for you, because a menu is additive and a system prompt is not.)",
        "#",
        "# Read at run time, so editing the prompt in the dashboard changes the",
        "# next run of this file. Nothing to redeploy, nothing to regenerate.",
        "support = decimalai.load_agent(AGENT_NAME)",
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
        "# prompt is handed to it BY THIS LINE, from the prompt read above — and",
        "# None (this agent has no prompt set, a real state; a failed read raises",
        "# instead) means it sends no system message at all, rather than one this",
        "# file invented.",
        "agent = create_agent(",
        "    init_chat_model(MODEL),",
        "    tools=TOOLS,",
        "    system_prompt=support.prompt,",
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
        "    # No turn cap here, unlike MAX_TURNS / MAX_REQUESTS on the other templates:",
        "    # create_agent compiles its graph with recursion_limit=9_999 (langchain/",
        "    # agents/factory.py) — effectively uncapped, NOT langchain-core's default of",
        "    # 25. With TOOLS=[] the loop is one step and cannot run away; once you add",
        "    # a tool, cap it — each turn is two graph steps (model, tools):",
        "    #   agent.invoke(..., config={\"recursion_limit\": 2 * MAX_TURNS + 1})",
        "    # and catch langgraph.errors.GraphRecursionError the way the openai-agents",
        "    # template catches MaxTurnsExceeded.",
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
        "import os",
        "",
        "import decimalai",
        "from agents import Agent, Runner",
        "from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError",
        "from decimalai.openai_agents import load_agent",
        "",
        "# Change this to any model your OpenAI key can reach.",
        f"MODEL = {_py(model)}",
        *_agent_name_lines(agent_name, load_agent_routes=True),
        "# The SDK default is 10, which a skill-loading turn can legitimately",
        "# exceed. Raise it if your agent has many tools; lower it to fail faster.",
        "MAX_TURNS = 20",
        "",
        *_key_guard_lines("openai-agents", model),
        "# Reads DECIMAL_API_KEY from the environment. Never paste a key into a",
        "# file you are going to commit.",
        _init_call(base_url),
        "",
        "# This agent's prompt and skills, from DecimalAI. Nothing is injected behind",
        "# your back: the two lines in Agent(...) below are where they go in.",
        "#   support.instructions  the prompt saved in the dashboard, then the list of",
        "#                         this agent's skills. Re-read at most once a minute,",
        "#                         so a prompt saved in the dashboard reaches a run",
        "#                         within a minute, with no redeploy. {{placeholders}}",
        "#                         in it are filled from",
        "#                         Runner.run(..., context=...).",
        "#   support.tools         the load_skill tool the model reads a skill with.",
        "# With no prompt saved, the agent gets only the skill list: nothing this file",
        "# made up. Pass load_agent(AGENT_NAME, fallback=...) to supply one.",
        "support = load_agent(AGENT_NAME)",
        "",
        "agent = Agent(",
        "    name=support.name,",
        "    model=MODEL,",
        "    instructions=support.instructions,",
        "    # Your own tools go first; DecimalAI adds load_skill.",
        "    tools=[*support.tools],",
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
        "    except ModelBehaviorError as exc:",
        "        # The model asked for a tool this file does not define (or returned",
        "        # something the SDK could not parse), and the SDK raises rather than",
        "        # looping. Same check as above: the Tools line in your system prompt",
        "        # names a tool this file does not define — compare it against the",
        "        # tools actually attached to the Agent above.",
        "        return (",
        "            f\"[{AGENT_NAME}] the prompt names a tool this file does not define \"",
        "            f\"({exc}). See the comment in run().\"",
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
        "import asyncio",
        "import os",
        "",
        "import decimalai",
        "from decimalai.pydantic_ai import load_agent, retry_model_requests",
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
        *_agent_name_lines(agent_name, load_agent_routes=True),
        "# Pydantic AI's own default is 50 model requests. Lower, because a",
        "# runaway loop here costs real tokens; raise it if your agent legitimately",
        "# chains many tool calls.",
        "MAX_REQUESTS = 20",
        "MAX_RUN_SECONDS = 90",
        "",
        *_key_guard_lines("pydantic-ai", model),
        "# Reads DECIMAL_API_KEY from the environment. Never paste a key into a",
        "# file you are going to commit.",
        "#",
        "# otel=True installs DecimalAI's OpenTelemetry exporter, which turns this",
        "# agent's spans into traces. A bare init() here would run fine and deliver",
        "# every skill, and your traces page would stay empty.",
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
        "# Your agent's prompt and skills, as they stand in the dashboard, as one",
        "# Pydantic AI capability: its instructions are the saved prompt, then the",
        "# skill list picked for each run's prompt, and its toolset is load_skill,",
        "# which the model calls for a listed skill's text. It also wraps each run",
        "# in one DecimalAI run span, so a run is one trace filed under this agent.",
        "# Nothing in Pydantic AI is patched.",
        "#",
        "# The prompt is re-read in the background at most once a minute, so",
        "# editing it in the dashboard changes the next runs of this process.",
        "# {{placeholders}} in it are filled from the run's deps=. With no prompt",
        "# saved, the agent gets only the skill list: nothing this file made up.",
        "# Pass load_agent(AGENT_NAME, fallback=...) to supply one.",
        "support = load_agent(AGENT_NAME)",
        "",
        "# Your tools go here. An empty list is a WORKING agent, not a stub: the",
        "# loop runs, load_skill sits beside your tools, and adding a tool later",
        "# needs no other change to this file.",
        "TOOLS: list = []",
        "",
        "agent = Agent(",
        "    # Retry temporary provider failures inside a model request. Completed",
        "    # tools are not repeated. Each request has a 30-second total budget.",
        "    retry_model_requests(MODEL, max_attempts=3, timeout_s=30),",
        "    name=support.name,",
        "    tools=TOOLS,",
        "    capabilities=[support],",
        ")",
        "",
        "",
        "def run(question: str) -> str:",
        '    """One turn: the loop runs until the model answers.',
        "",
        "    Add tools to TOOLS above; load_agent()'s capability adds load_skill",
        "    beside them.",
        '    """',
        "    async def bounded_turn():",
        "        return await asyncio.wait_for(",
        "            agent.run(question, usage_limits=UsageLimits(request_limit=MAX_REQUESTS)),",
        "            timeout=MAX_RUN_SECONDS,",
        "        )",
        "",
        "    try:",
        "        return asyncio.run(bounded_turn()).output",
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


def _render_adk(
    agent_name: str, model: str, base_url: Optional[str]
) -> List[str]:
    # NODE_NAME is derived IN the file rather than rendered here, so it is not a
    # second copy of the name (point 2). The rule is ADK's own
    # (`BaseAgent.validate_name`: an identifier, and not "user").
    return [
        "import asyncio",
        "import os",
        "import re",
        "",
        "import decimalai",
        "from decimalai.adk import instrument",
        "from google.adk.agents import LlmAgent",
        "from google.adk.runners import InMemoryRunner",
        "from google.genai import types",
        "",
        "# Any Gemini model your key can reach. ADK resolves a bare `gemini-*` name",
        "# natively; for another provider wrap it: model=LiteLlm(\"openai/gpt-4o-mini\").",
        f"MODEL = {_py(model)}",
        *_agent_name_lines(agent_name),
        "# ADK node names must be Python identifiers, and not \"user\"; the DecimalAI",
        "# name need not be. Traces still file under AGENT_NAME — init() carries it.",
        "NODE_NAME = re.sub(r\"\\W\", \"_\", AGENT_NAME)",
        "if not NODE_NAME.isidentifier() or NODE_NAME == \"user\":",
        "    NODE_NAME = \"agent_\" + NODE_NAME",
        "",
        "# ADK's Gemini client reads GOOGLE_API_KEY (or GEMINI_API_KEY) at the FIRST",
        "# model call, deep inside the run loop. Fail here, before anything runs.",
        "if not (os.environ.get(\"GOOGLE_API_KEY\") or os.environ.get(\"GEMINI_API_KEY\")",
        "        or os.environ.get(\"GOOGLE_GENAI_USE_VERTEXAI\", \"\").lower() in (\"1\", \"true\")):",
        "    raise SystemExit(",
        "        \"GOOGLE_API_KEY is not set. Run: export GOOGLE_API_KEY=...  (a Gemini API key; \"",
        "        \"or GOOGLE_GENAI_USE_VERTEXAI=1 with Google Cloud credentials). The `Set:` \"",
        "        \"block `decimalai init` printed lists every variable this file needs.\"",
        "    )",
        "",
        "# Reads DECIMAL_API_KEY from the environment. Never paste a key into a",
        "# file you are going to commit.",
        _init_call(base_url),
        "",
        "# enable_skill_loader=True is what actually delivers this agent's skills:",
        "# every model turn is routed and the routed BODIES are appended to the",
        "# request's system_instruction from the plugin's before_model hook. ADK",
        "# registers no load_skill tool, so injection is the only body channel.",
        "#",
        "# This call patches Runner.__init__ to attach the DecimalAI plugin, so it",
        "# must stay ABOVE the InMemoryRunner(...) below. No agent_name here: init()",
        "# above already named the agent. Do not also pass adk=True to init(): that",
        "# is this same install, so one of the two is noise.",
        "instrument(enable_skill_loader=True)",
        "",
        "# Your agent's system prompt, as it stands in the dashboard right now.",
        "# Read at run time, so editing it in the dashboard changes the next run.",
        "support = decimalai.load_agent(AGENT_NAME)",
        "",
        "# Your tools go here. An empty list is a WORKING agent, not a stub.",
        "TOOLS: list = []",
        "",
        "agent = LlmAgent(",
        "    name=NODE_NAME,",
        "    model=MODEL,",
        "    # `or \"\"` because None is a REAL state (no prompt set; a failed read",
        "    # raises instead) and ADK's instruction field takes a string, never None.",
        "    instruction=support.prompt or \"\",",
        "    tools=TOOLS,",
        ")",
        "",
        "runner = InMemoryRunner(agent=agent, app_name=NODE_NAME)",
        "",
        "",
        "async def _turn(question: str) -> str:",
        "    session = await runner.session_service.create_session(",
        "        app_name=runner.app_name, user_id=\"local\",",
        "    )",
        "    answer = \"\"",
        "    async for event in runner.run_async(",
        "        user_id=\"local\", session_id=session.id,",
        "        new_message=types.Content(role=\"user\", parts=[types.Part(text=question)]),",
        "    ):",
        "        if event.content and event.content.parts:",
        "            text = \"\".join(p.text or \"\" for p in event.content.parts)",
        "            if text:",
        "                answer = text",
        "    return answer",
        "",
        "",
        "def run(question: str) -> str:",
        "    \"\"\"One turn: ADK runs the tool loop until the model answers.\"\"\"",
        "    return asyncio.run(_turn(question))",
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
    "adk": _render_adk,
}

#: What the user has to install and export for the generated file to run.
#: `decimalai[openai]` pulls the openai SDK, NOT the Agents SDK — they are
#: separate extras, and copy-pasting the wrong one leaves the import failing on
#: a machine that looks correctly set up. From the registry, like ENV_VARS below.
INSTALL: Dict[str, str] = {
    f.key: f.install for f in _registry.FRAMEWORKS if f.scaffold == _registry.TEMPLATE
}

#: Extras whose framework floor needs a newer Python than decimalai itself
#: (pyproject's `[adk]` and `[pydantic-ai]`: google-adk and pydantic-ai-slim at
#: their floors require 3.11). pip does not report the mismatch. On an older
#: Python it installs the newest decimalai release whose requirements still
#: resolve, older than the one that wrote the file, and prints nothing.
EXTRA_MIN_PYTHON: Dict[str, Tuple[int, int]] = {
    f.key: f.min_python for f in _registry.FRAMEWORKS if f.min_python
}


def install_python_note(
    framework: str, version_info: Optional[Sequence[int]] = None
) -> Optional[str]:
    """A warning to print under the install line, or None.

    Only when this Python is older than the framework's extra needs, because
    that is the one case where the printed line "works" and installs the wrong
    SDK. ``version_info`` defaults to the running interpreter's.
    """
    need = EXTRA_MIN_PYTHON.get(framework)
    have = tuple((version_info if version_info is not None else sys.version_info)[:2])
    if need is None or have >= need:
        return None
    return (
        f"decimalai[{framework}] needs Python {need[0]}.{need[1]} or newer, and this is "
        f"Python {have[0]}.{have[1]}. Here pip installs an older decimalai release "
        f"instead, without saying so. Run the install line under Python "
        f"{need[0]}.{need[1]}+."
    )


ENV_VARS: Dict[str, tuple] = {
    f.key: f.env_vars for f in _registry.FRAMEWORKS if f.scaffold == _registry.TEMPLATE
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
#: LangChain spells Gemini `google_genai`, Pydantic AI spells it `google`.
#: Pydantic AI's GoogleProvider reads `GOOGLE_API_KEY`, else the legacy
#: `GEMINI_API_KEY` (`pydantic_ai.providers.google`, pydantic-ai-slim 2.56), so
#: the `GEMINI_API_KEY` the table names still runs; when both are set,
#: `GOOGLE_API_KEY` is the one used.
#:
#: The extra is empty for the providers `decimalai[pydantic-ai]` already
#: installs (`pydantic-ai-slim[openai,google,anthropic]`; azure, deepseek and
#: ollama ride the openai binding). The rest are `pydantic-ai-slim` extras —
#: never the `pydantic-ai` meta package, which pulls logfire.
_PYDANTIC_AI_PROVIDERS: Dict[str, tuple] = {
    # provider prefix -> (extra pip requirement, model-provider env var)
    "openai": ("", "OPENAI_API_KEY"),
    "openai-chat": ("", "OPENAI_API_KEY"),
    "openai-responses": ("", "OPENAI_API_KEY"),
    "azure": ("", "AZURE_OPENAI_API_KEY"),
    "google": ("", "GEMINI_API_KEY"),
    "google-vertex": ("", "GOOGLE_API_KEY"),
    "anthropic": ("", "ANTHROPIC_API_KEY"),
    "groq": ('"pydantic-ai-slim[groq]"', "GROQ_API_KEY"),
    "mistral": ('"pydantic-ai-slim[mistral]"', "MISTRAL_API_KEY"),
    "cohere": ('"pydantic-ai-slim[cohere]"', "CO_API_KEY"),
    "huggingface": ('"pydantic-ai-slim[huggingface]"', "HF_TOKEN"),
    "openrouter": ('"pydantic-ai-slim[openrouter]"', "OPENROUTER_API_KEY"),
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

    This matches the pydantic-ai-slim `decimalai[pydantic-ai]` floors (2.56),
    which refuses a bare name too, so it refuses nothing that would have run.
    """
    model = (model or "").strip() or default_model(framework)
    if framework == "adk":
        # Mirrors google_llm.py's native patterns. `decimalai[adk]` installs no
        # other provider, so a non-Gemini name is a file that dies on its first
        # model call while `decimalai init` reports success.
        if re.fullmatch(r"(gemini-.*|gemma-4.*|model-optimizer-.*|projects/.+)", model):
            return model
        raise UnusableModel(
            f"Google ADK resolves Gemini names natively, and {model!r} is not one — "
            f"`decimalai[adk]` installs no other provider, so the file would die on "
            f"the first model call.\nTry --model gemini-3.5-flash, or wrap another "
            f"provider in the file: model=LiteLlm(\"openai/{model}\")."
        )
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


def provider_key(framework: str, model: Optional[str] = None) -> Optional[str]:
    """The ONE provider variable the generated file refuses to run without, or None.

    None where there is nothing safe to insist on: a local provider (ollama),
    Pydantic AI's built-in ``"test"`` model, and any provider prefix the tables
    do not know (azure_openai:, bedrock:, xai:, …). ``env_vars()`` falls back to
    OPENAI_API_KEY for those as ADVICE in the ``Set:`` block; advice that is
    sometimes wrong is fine printed, not as a SystemExit — a guard that refuses
    an Azure user whose key IS set is the "stricter broke a working path" class.
    """
    model = model or default_model(framework)
    table = {
        "langchain": _PROVIDER_REQUIREMENTS,
        "pydantic-ai": _PYDANTIC_AI_PROVIDERS,
    }.get(framework)
    if table is None:  # openai-agents: the default OpenAIProvider; adk: its own guard
        vars_ = ENV_VARS[framework]
        return vars_[1] if len(vars_) > 1 else None
    if framework == "pydantic-ai" and model == "test":
        return None
    entry = table.get(model_provider(model))
    return entry[1] if entry and entry[1] else None


def _key_guard_lines(framework: str, model: str) -> List[str]:
    """The two-line refusal at the top of a template, or nothing (see provider_key).

    Without it a missing provider key is a traceback from inside the provider
    client — after the file's DecimalAI setup has made its network calls
    (openai-agents: only on the first Runner.run_sync).
    """
    key = provider_key(framework, model)
    if not key:
        return []
    return [
        "# The key MODEL's provider reads, checked before anything talks to DecimalAI.",
        "# Without this it is a traceback from inside the provider client, after the",
        "# DecimalAI setup below has run. Change this name if you change MODEL to",
        "# another provider.",
        f"PROVIDER_KEY = {_py(key)}",
        "if not os.environ.get(PROVIDER_KEY):",
        "    raise SystemExit(",
        "        f\"{PROVIDER_KEY} is not set. Run: export {PROVIDER_KEY}=...  \"",
        "        \"(the `Set:` block `decimalai init` printed lists every variable this file needs).\"",
        "    )",
        "",
    ]


def render_agent_file(
    agent_name: str,
    framework: str = "langchain",
    skills: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    prompt: Optional["SavedPrompt"] = None,
) -> str:
    """Return the complete source of a runnable agent file.

    Pure: no I/O, no clock, no randomness. Given the same inputs it returns
    the same bytes, which is what lets the test suite compile the output
    instead of pattern-matching a string.

    Args:
        agent_name: The agent as it exists on DecimalAI. Written once, as
            `AGENT_NAME`, and bound by `init(agent_name=AGENT_NAME)` so traces
            file against it and its skills resolve.
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
