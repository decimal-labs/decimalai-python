"""One registry of the agent frameworks the SDK integrates with.

Every fact a consumer needs about a framework lives here once: its scaffold id, the
adapter module and entry point, the ``init()`` flag, the extra, whether ``decimalai init``
writes a file for it, the default model, the install line and environment variables the
quickstart needs, the native-loader helpers (SDK 0.16+), and the cells of the docs'
capability table. Until this module, each of those was copied by hand into four places
that drifted apart one release at a time.

Who reads it
------------
* ``decimalai/cli/scaffold.py`` derives ``SUPPORTED_FRAMEWORKS``, ``UNSCAFFOLDED_WITH_SEAM``,
  ``NO_PROMPT_SEAM``, ``INSTALL``, ``ENV_VARS``, ``EXTRA_MIN_PYTHON`` and
  ``_FRAMEWORK_DEFAULT_MODELS`` from it, keeping those names and values.
* ``decimalai-docs/sdk/python/frameworks.mdx`` carries the capability table and the module
  reference between ``BEGIN GENERATED`` / ``END GENERATED`` markers, written by
  :func:`render_block`. The docs repo's ``scripts/check_generated_framework_tables.py``
  regenerates them and fails on a difference; ``--write`` rewrites them.
  ``tests/test_frameworks_registry.py`` here makes the same comparison when the docs
  checkout sits beside this one.
* DecimalAI's web app commits ``python -m decimalai._frameworks --json`` as its framework
  snippet data, and a check there fails when its copy differs.

Stdlib only, and nothing imported from the ``decimalai`` package, ON PURPOSE: the docs
checker and the app's check load this file by path, so it has to work without
decimalai's dependencies installed and without running ``decimalai/__init__.py``.

Changing a fact
---------------
Edit the entry below, then regenerate the consumers: ``scripts/check_generated_framework_tables.py
--write`` in decimalai-docs, and the app's sync script.
Each one's check fails until you do. Adding a row to the capability table also adds a slug
to ``tests/conformance/drivers/__init__.py::ADVERTISED_SNAPSHOT``, which then needs a
conformance driver (``tests/conformance/test_coverage.py`` says how).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: ``decimalai init``'s --model default for a framework with no override below.
#: Re-exported by ``decimalai/cli/scaffold.py`` as ``DEFAULT_MODEL``.
DEFAULT_MODEL = "gpt-4o-mini"

# ── what `decimalai init` does with a framework ──────────────────────────────
#: Its adapter has a prompt seam AND ``decimalai init`` has a template for it.
TEMPLATE = "template"
#: Its adapter carries ``enable_skill_loader``, but there is no template yet:
#: ``decimalai init`` says "not yet" (``scaffold.UNSCAFFOLDED_WITH_SEAM``).
SEAM_NO_TEMPLATE = "seam-no-template"
#: Its adapter has no prompt seam, so a generated file would trace and deliver none of
#: the agent's skills: ``decimalai init`` refuses, saying why (``scaffold.NO_PROMPT_SEAM``).
NO_SEAM = "no-seam"
SCAFFOLD_STATUSES = (TEMPLATE, SEAM_NO_TEMPLATE, NO_SEAM)

#: Bumped when a field is renamed or removed, so a consumer of the JSON can refuse a
#: shape it was not written for. Adding a field does not bump it.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Capability:
    """One row of the docs' capability table (``sdk/python/frameworks.mdx``).

    Markdown cells, written for the SDK release this module ships in. The native-loader
    column is not here: it is rendered from :attr:`Framework.native_loaders`.
    """

    tier: str           # First-class | Supported | Community
    manifest: str       # Auto-detected manifest
    schema_depth: str   # Tool schema depth
    handoffs: str       # Handoffs / multi-agent
    skills_rail: str    # how DecimalAI's skill router reaches the model


@dataclass(frozen=True)
class NativeLoader:
    """A framework's OWN skill loader and the DecimalAI helper that points it at the
    skills ``decimalai skills add`` installed (SDK 0.16+, "native mode").

    ``loader_id`` is the id the native-skill-loaders page and
    ``tests/native_loaders/loaders/__init__.py::NATIVE_LOADERS_SNAPSHOT`` use.
    A loader is listed under the framework whose rail traces it: Deep Agents under
    LangChain, Microsoft Agent Framework and Strands under generic OTel.
    """

    loader_id: str
    #: The framework the loader belongs to, as the native-skill-loaders page names it.
    name: str
    #: Dotted paths of the helpers its setup calls, the main one first.
    helpers: Tuple[str, ...]
    #: Shown before the helper in the capability table when the row's framework is not
    #: the whole story ("Deep Agents: ..."). None when the row's own framework loads it.
    label: Optional[str] = None
    #: For a loader whose helpers live in a module of their own: the distribution the
    #: framework installs as, the ``init()`` setup that traces it, and its "Use it for"
    #: cell in the module reference. All three None when the helpers sit in the row's
    #: own adapter module.
    requires: Optional[str] = None
    traced_by: Optional[str] = None
    use_for: Optional[str] = None

    @property
    def module(self) -> str:
        return self.helpers[0].rsplit(".", 1)[0]


@dataclass(frozen=True)
class Framework:
    """Everything a consumer needs to know about one framework integration."""

    #: The ``decimalai init --framework`` id; ``normalize_framework`` also accepts it
    #: with underscores.
    key: str
    #: The capability table's first column. ``tests/conformance/test_coverage.py``
    #: turns it into slugs (parentheticals and inline code dropped, "/" splits).
    title: str
    #: How ``decimalai init`` names it at the start of a sentence when it refuses it.
    cli_label: str
    #: The adapter module and its entry point (a markdown cell; every ``name(`` in it
    #: is an attribute of the module, which the registry test checks).
    module: str
    entry_point: str
    #: ``init(<flag>=True)``, or None when ``init()`` has no flag that installs this
    #: adapter (``init_flag_note`` then fills the module-reference cell).
    init_flag: Optional[str]
    #: ``pip install "decimalai[<extra>]"``, or None when there is no extra
    #: (``extra_note`` then fills the module-reference cell).
    extra: Optional[str]
    scaffold: str
    #: What the quickstart needs installed and exported. For a TEMPLATE framework that
    #: is the generated file at its default model (``scaffold.INSTALL`` /
    #: ``scaffold.ENV_VARS``); otherwise the framework page's own quickstart.
    install: str
    env_vars: Tuple[str, ...]
    #: ``decimalai init``'s MODEL line when --model is not passed; None without a template.
    default_model: Optional[str] = None
    #: Set when the extra's framework floor needs a newer Python than decimalai itself.
    min_python: Optional[Tuple[int, int]] = None
    native_loaders: Tuple[NativeLoader, ...] = ()
    #: None: not a row of the docs' capability table.
    capability: Optional[Capability] = None
    #: The module reference's "Use it for" cell; None: not a row of the module reference.
    use_for: Optional[str] = None
    init_flag_note: Optional[str] = None
    extra_note: Optional[str] = None


# ── the registry ─────────────────────────────────────────────────────────────
#
# Order is meaningful twice: it is the order `decimalai init` offers its templates in
# (langchain first: the --framework default), and the order of both docs tables.

FRAMEWORKS: Tuple[Framework, ...] = (
    Framework(
        key="langchain",
        title="LangChain / LangGraph (with `agent_config(...)`)",
        cli_label="LangChain",
        module="decimalai.langchain",
        entry_point="`agent_config(name).middleware`",
        init_flag="langchain",
        extra="langchain",
        scaffold=TEMPLATE,
        # The extra brings `langchain` (create_agent and its middleware); the generated
        # file calls an OpenAI model, so langchain-openai too. The quotes are load-bearing:
        # zsh globs a bare `decimalai[langchain]` and pip never sees it.
        install='pip install "decimalai[langchain]" langchain-openai',
        env_vars=("DECIMAL_API_KEY", "OPENAI_API_KEY"),
        default_model=DEFAULT_MODEL,
        native_loaders=(
            NativeLoader(
                loader_id="deepagents",
                name="Deep Agents",
                helpers=("decimalai.deepagents.skills",),
                label="Deep Agents",
                requires="deepagents",
                traced_by="`langchain=True`",
                use_for="Deep Agents' own skills middleware, pointed at installed skills; "
                        "each `read_file` of a skill is recorded as an activation (0.16+)",
            ),
        ),
        capability=Capability(
            tier="First-class",
            manifest="✅ full",
            schema_depth="Full schemas",
            handoffs="✅",
            skills_rail="`load_skill` tool",
        ),
        use_for="LangChain / LangGraph — `agent_config(name).middleware` in `create_agent` gives "
                "the agent its prompt and `load_skill`; `init(langchain=True)` traces",
    ),
    Framework(
        key="openai-agents",
        title="OpenAI Agents (with `agent_config(...)`)",
        cli_label="OpenAI Agents",
        module="decimalai.openai_agents",
        entry_point="`agent_config(name)`",
        init_flag="openai_agents",
        extra="openai-agents",
        scaffold=TEMPLATE,
        # `decimalai[openai]` pulls the openai SDK, NOT the Agents SDK: separate extras.
        install='pip install "decimalai[openai-agents]"',
        env_vars=("DECIMAL_API_KEY", "OPENAI_API_KEY"),
        default_model=DEFAULT_MODEL,
        native_loaders=(
            NativeLoader(
                loader_id="openai-agents",
                name="OpenAI Agents SandboxAgent",
                helpers=("decimalai.openai_agents.sandbox_skills",),
                label="`SandboxAgent`",
            ),
        ),
        capability=Capability(
            tier="First-class",
            manifest="✅ full",
            schema_depth="Full schemas",
            handoffs="✅",
            skills_rail="`load_skill` tool",
        ),
        use_for="OpenAI Agents SDK — `agent_config(name)` returns the agent's prompt and "
                "skills to pass as `Agent(instructions=..., tools=[...])`, nothing patched "
                "(0.18+); `sandbox_skills()` is a `SandboxAgent`'s Skills capability over "
                "installed skills (0.16+)",
    ),
    Framework(
        key="claude-agent-sdk",
        title="Claude Agent SDK / Claude Code",
        cli_label="The Claude Agent SDK",
        module="decimalai.claude_agent_sdk",
        entry_point="`instrument()` / `trace_stream()`",
        init_flag="claude_agent_sdk",
        extra="claude-agent-sdk",
        scaffold=NO_SEAM,
        install='pip install "decimalai[claude-agent-sdk]"',
        env_vars=("DECIMAL_API_KEY", "ANTHROPIC_API_KEY"),
        native_loaders=(
            NativeLoader(
                loader_id="claude-agent-sdk",
                name="Claude Agent SDK and Claude Code",
                helpers=("decimalai.claude_agent_sdk.skills_plugin",),
            ),
        ),
        capability=Capability(
            tier="First-class",
            manifest="✅ full (model, prompt, tools, subagents)",
            schema_depth="Names only",
            handoffs="✅ (subagents)",
            # No prompt seam for the router; skills reach Claude through Claude Code's
            # own loader (the native column), which is how they always did.
            skills_rail="—",
        ),
        use_for="Claude Agent SDK — wraps the `query()` stream; one trace per run; "
                "`skills_plugin()` loads installed skills as a Claude Code plugin (0.16+)",
    ),
    Framework(
        key="pydantic-ai",
        title="Pydantic AI (with `agent_config(...)`)",
        cli_label="Pydantic AI",
        module="decimalai.pydantic_ai",
        entry_point="`agent_config(name)`",
        init_flag=None,
        init_flag_note="— (no flag; tracing is `otel=True` + `Agent.instrument_all()`)",
        extra="pydantic-ai",
        scaffold=TEMPLATE,
        # One line, whatever the model. `decimalai[pydantic-ai]` resolves to
        # `pydantic-ai-slim[openai,google,anthropic]`, so those three provider bindings
        # come with it, and the OpenTelemetry SDK the template's tracing needs is already
        # a core dependency of decimalai. Not the `pydantic-ai` meta package: its logfire
        # extra needs an opentelemetry-api that google-adk rules out of `decimalai[all]`.
        # A provider outside those three adds a token; see scaffold._PYDANTIC_AI_PROVIDERS.
        install='pip install "decimalai[pydantic-ai]"',
        env_vars=("DECIMAL_API_KEY", "OPENAI_API_KEY"),
        # Pydantic AI resolves `provider:model` and REFUSES a bare one —
        # `infer_model("gpt-4o-mini")` raises `UserError: Unknown model` (pydantic-ai-slim
        # 2.56), so the shared default would die on the line that builds the Agent.
        default_model="openai:gpt-4o-mini",
        # pydantic-ai-slim at its floor requires Python 3.11; pip does not report the
        # mismatch, it installs an older decimalai instead.
        min_python=(3, 11),
        native_loaders=(
            NativeLoader(
                loader_id="pydantic-ai-harness",
                name="Pydantic AI harness",
                helpers=("decimalai.pydantic_ai_harness.skills",),
                label="Pydantic AI harness",
                requires="pydantic-ai-harness",
                traced_by="`otel=True` + `Agent.instrument_all()`",
                use_for="The Pydantic AI harness's `Skills` capability, pointed at installed "
                        "skills; each load recorded as an activation (0.16+)",
            ),
        ),
        capability=Capability(
            tier="Supported",
            manifest="❌ (register manually)",
            schema_depth="—",
            handoffs="—",
            skills_rail="`load_skill` tool",
        ),
        use_for="Pydantic AI — `agent_config(name)` returns the agent's prompt and skills "
                "as one capability for `Agent(capabilities=[...])`, nothing patched (0.19+)",
    ),
    Framework(
        key="adk",
        title="Google ADK (with `agent_config(...)`)",
        cli_label="Google ADK",
        module="decimalai.adk",
        entry_point="`agent_config(name)` + `DecimalAIPlugin()`",
        init_flag=None,
        init_flag_note="— (no flag; tracing is `DecimalAIPlugin()` in `App(plugins=[...])`)",
        extra="adk",
        # moved from NO_PROMPT_SEAM on 2026-09-03 (seam proven by C14 and a live Gemini
        # end-to-end test) and to TEMPLATE on 2026-09-05: google-genai reads
        # GOOGLE_GEMINI_BASE_URL, so the hermetic journey stub can stand in for Gemini.
        scaffold=TEMPLATE,
        install='pip install "decimalai[adk]"',
        # google-genai reads GOOGLE_API_KEY ahead of GEMINI_API_KEY; name the one that
        # wins so the `Set:` block and the generated file's own guard agree.
        env_vars=("DECIMAL_API_KEY", "GOOGLE_API_KEY"),
        # Never an OpenAI name: `decimalai[adk]` installs google-adk only, and ADK would
        # resolve `gpt-*` to its OpenAI shim and die on the first model call.
        # `scaffold.normalize_model` refuses non-Gemini names for adk so this stays true.
        default_model="gemini-3.5-flash",
        # google-adk at its floor requires Python 3.11.
        min_python=(3, 11),
        native_loaders=(
            NativeLoader(
                loader_id="google-adk",
                name="Google ADK",
                helpers=("decimalai.adk.skills",),
            ),
        ),
        capability=Capability(
            tier="Supported",
            manifest="✅ full (model, instruction, tools, sub-agents)",
            schema_depth="Names + descriptions",
            handoffs="✅",
            skills_rail="`load_skill` tool",
        ),
        use_for="Google ADK — `agent_config(name)` returns the agent's prompt and skills as an "
                "instruction provider and `load_skill` for your own `LlmAgent`; "
                "`DecimalAIPlugin()` in `App(plugins=[...])` makes one trace per invocation, "
                "nothing patched (0.20+); `skills()` feeds ADK's `SkillToolset` (0.16+)",
    ),
    Framework(
        key="anthropic",
        title="Anthropic Messages API (with `agent_config(...)`)",
        cli_label="The Anthropic Messages API",
        module="decimalai.anthropic",
        entry_point="`agent_config(name)`",
        # `init(anthropic=True)` traces every Anthropic call in the process; agent_config()
        # turns the same tracing on for the calls made inside its runs, so it needs no flag.
        init_flag=None,
        init_flag_note="— (no flag; `agent_config()` traces its runs, `anthropic=True` every call)",
        extra="anthropic",
        # a template since 0.23: agent_config() gives each run a load_skill tool, and the
        # SDK's own Tool Runner is the loop. Was SEAM_NO_TEMPLATE while the adapter patched
        # one `messages.create()` call and owned no loop.
        scaffold=TEMPLATE,
        install='pip install "decimalai[anthropic]"',
        env_vars=("DECIMAL_API_KEY", "ANTHROPIC_API_KEY"),
        # The budget tier, like the other templates' defaults. Other names are not refused:
        # a gateway behind ANTHROPIC_BASE_URL can serve models not named `claude-*`.
        default_model="claude-haiku-5-5",
        capability=Capability(
            tier="Supported",
            # What the exporter registers from the traced requests: the model and each
            # tool, by name (no schema hash or description). Not the system text: a
            # run's prompt is the saved one, whose version each trace names, and the
            # text the model saw has its values filled in and the skill list after it.
            manifest="✅ partial (model, tools; the prompt by its saved version)",
            schema_depth="Names only",
            handoffs="—",
            skills_rail="`load_skill` tool",
        ),
        use_for="Anthropic Messages API — `agent_config(name)`; each `support.run(question)` gives "
                "`run.system` (the prompt, then the skill list) and `load_skill` in `run.tools` "
                "for the Tool Runner (`run.tool_params` and `run.tool_result()` for your own "
                "`messages.create()` loop), one trace per run, nothing patched (0.23+)",
    ),
    Framework(
        key="llamaindex",
        title="LlamaIndex",
        cli_label="LlamaIndex",
        module="decimalai.llamaindex",
        entry_point="`instrument()`",
        init_flag="llamaindex",
        extra="llamaindex",
        scaffold=NO_SEAM,
        # LlamaIndex resolves its default LLM and embedding model to OpenAI's, and core
        # ships no provider bindings: the framework page's quickstart installs both.
        install='pip install "decimalai[llamaindex]" llama-index-llms-openai '
                'llama-index-embeddings-openai',
        env_vars=("DECIMAL_API_KEY", "OPENAI_API_KEY"),
        capability=Capability(
            tier="Supported",
            manifest="✅ partial",
            schema_depth="Names only",
            handoffs="—",
            skills_rail="—",
        ),
        use_for="LlamaIndex span handler (`llama-index-core>=0.14.25`)",
    ),
    Framework(
        key="crewai",
        title="CrewAI",
        cli_label="CrewAI",
        module="decimalai.crewai",
        entry_point="`instrument(enable_skill_loader=True)`",
        init_flag="crewai",
        extra=None,
        extra_note="— (install `crewai` and `openinference-instrumentation-crewai`)",
        # moved from NO_PROMPT_SEAM on 2026-10-08: decimalai/crewai.py delivers skills
        # through CrewAI's before_llm_call hook, graded by C8/C14 and the injected
        # delivery cell. No template yet, so `decimalai init` says "not yet".
        scaffold=SEAM_NO_TEMPLATE,
        # The CrewAI page's install line. No extra exists; the OpenAI instrumentor is
        # what carries the model call (CrewAI's own spans have no LLM span), and the
        # page's agent runs `openai/gpt-5-mini`.
        install='pip install "decimalai>=0.14" "crewai>=1.15.27" '
                '"openinference-instrumentation-crewai>=1.1.21" '
                'openinference-instrumentation-openai',
        env_vars=("DECIMAL_API_KEY", "OPENAI_API_KEY"),
        native_loaders=(
            NativeLoader(
                loader_id="crewai",
                name="CrewAI",
                helpers=("decimalai.crewai.skills", "decimalai.crewai.skill_tools"),
            ),
        ),
        capability=Capability(
            tier="Supported",
            manifest="✅ partial",
            schema_depth="Names only",
            handoffs="✅",
            skills_rail="Prompt injection (0.13.9+)",
        ),
        use_for="CrewAI tracing through the OTel exporter, plus skill injection via CrewAI's "
                "`before_llm_call` hook (0.13.9+, `crewai>=1.15.27`); `skills()` and "
                "`skill_tools()` feed CrewAI's own loader (0.16+)",
    ),
    Framework(
        key="otel",
        title="Generic OTel (Haystack, …)",
        cli_label="The generic OpenTelemetry rail",
        module="decimalai.otel",
        entry_point="`instrument()`",
        init_flag="otel",
        extra=None,
        extra_note="—",
        scaffold=NO_SEAM,
        install="pip install decimalai",
        env_vars=("DECIMAL_API_KEY",),
        native_loaders=(
            NativeLoader(
                loader_id="agent-framework",
                name="Microsoft Agent Framework",
                helpers=("decimalai.agent_framework.skills",),
                label="Microsoft Agent Framework",
                requires="agent-framework-core",
                traced_by="`otel=True`",
                use_for="Microsoft Agent Framework's `SkillsProvider`, pointed at installed "
                        "skills, with a script runner; each load recorded as an activation "
                        "(0.16+)",
            ),
            NativeLoader(
                loader_id="strands-agents",
                name="Strands Agents",
                helpers=("decimalai.strands.skills",),
                label="Strands",
                requires="strands-agents",
                traced_by="`otel=True`",
                use_for="Strands' `AgentSkills` plugin, pointed at installed skills; each "
                        "load recorded as an activation (0.16+)",
            ),
        ),
        capability=Capability(
            tier="Community",
            manifest="❌ (register manually)",
            schema_depth="—",
            handoffs="—",
            skills_rail="—",
        ),
        use_for="Generic OpenTelemetry GenAI span exporter; also the trace for Microsoft "
                "Agent Framework and Strands, from their own spans (0.16+)",
    ),
    # ── not rows of the docs' tables ──
    Framework(
        key="autogen",
        title="AutoGen / AG2",
        cli_label="AutoGen / AG2",
        module="decimalai.autogen",
        entry_point="`instrument()`",
        # Retired as an integration: the flag still runs, installs the generic OTel
        # exporter and warns. Not a docs table row (the frameworks page says so in a Note).
        init_flag="autogen",
        extra=None,
        scaffold=NO_SEAM,
        install="pip install decimalai",
        env_vars=("DECIMAL_API_KEY",),
    ),
)

BY_KEY: Dict[str, Framework] = {f.key: f for f in FRAMEWORKS}


def get(key: str) -> Framework:
    """The framework for a ``decimalai init`` id; underscores accepted."""
    return BY_KEY[key.strip().lower().replace("_", "-")]


def native_loaders() -> List[Tuple[Framework, NativeLoader]]:
    """Every native loader, with the framework whose rail traces it."""
    return [(f, n) for f in FRAMEWORKS for n in f.native_loaders]


# ── JSON ─────────────────────────────────────────────────────────────────────


def _loader_dict(n: NativeLoader) -> Dict[str, Any]:
    return {
        "loader_id": n.loader_id,
        "name": n.name,
        "helpers": list(n.helpers),
        "module": n.module,
        "label": n.label,
        "requires": n.requires,
        "traced_by": n.traced_by,
        "use_for": n.use_for,
    }


def _framework_dict(f: Framework) -> Dict[str, Any]:
    cap = f.capability
    return {
        "key": f.key,
        "title": f.title,
        "cli_label": f.cli_label,
        "module": f.module,
        "entry_point": f.entry_point,
        "init_flag": f.init_flag,
        "extra": f.extra,
        "scaffold": f.scaffold,
        "install": f.install,
        "env_vars": list(f.env_vars),
        "default_model": f.default_model,
        "min_python": list(f.min_python) if f.min_python else None,
        "native_loaders": [_loader_dict(n) for n in f.native_loaders],
        "capability": None if cap is None else {
            "tier": cap.tier,
            "manifest": cap.manifest,
            "schema_depth": cap.schema_depth,
            "handoffs": cap.handoffs,
            "skills_rail": cap.skills_rail,
        },
        "use_for": f.use_for,
        "init_flag_note": f.init_flag_note,
        "extra_note": f.extra_note,
    }


def to_dict() -> Dict[str, Any]:
    """The whole registry, as ``python -m decimalai._frameworks --json`` prints it."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "python -m decimalai._frameworks --json",
        "default_model": DEFAULT_MODEL,
        "scaffold_statuses": list(SCAFFOLD_STATUSES),
        "frameworks": {f.key: _framework_dict(f) for f in FRAMEWORKS},
    }


def to_json() -> str:
    """:func:`to_dict` as the bytes consumers commit: sorted nowhere, indented, one
    trailing newline, non-ASCII kept (the table cells carry ✅ and —)."""
    return json.dumps(to_dict(), indent=2, ensure_ascii=False) + "\n"


# ── the docs' two tables ─────────────────────────────────────────────────────


def _row(cells: Sequence[str]) -> str:
    for c in cells:
        if "|" in c.replace("\\|", "") or "\n" in c:
            raise ValueError(f"a table cell may not contain '|' or a newline: {c!r}")
    return "| " + " | ".join(cells) + " |"


def _rule(align: Sequence[str]) -> str:
    return "|" + "|".join(align) + "|"


def _call(dotted: str) -> str:
    return f"`{dotted}()`"


def _short_call(dotted: str) -> str:
    return f"`{dotted.rsplit('.', 1)[1]}()`"


def _native_cell(f: Framework) -> str:
    parts = []
    for n in f.native_loaders:
        helpers = _call(n.helpers[0]) + "".join(" + " + _short_call(h) for h in n.helpers[1:])
        parts.append(f"{n.label}: {helpers}" if n.label else helpers)
    return "; ".join(parts) or "—"


CAPABILITY_HEADER = (
    "Framework", "Tier", "Auto-detected manifest", "Tool schema depth",
    "Handoffs / multi-agent", "Skills rail", "Native skill loader (0.16+)",
)
_CAPABILITY_ALIGN = ("---", "---", ":---:", ":---:", ":---:", ":---:", "---")


def render_capability_table() -> str:
    """The capability-comparison table, every framework with a ``capability``."""
    lines = [_row(CAPABILITY_HEADER), _rule(_CAPABILITY_ALIGN)]
    for f in FRAMEWORKS:
        cap = f.capability
        if cap is None:
            continue
        lines.append(_row((f.title, cap.tier, cap.manifest, cap.schema_depth,
                           cap.handoffs, cap.skills_rail, _native_cell(f))))
    return "\n".join(lines) + "\n"


MODULE_HEADER = ("Module", "Entry point", "`init()` flag", "Extra", "Use it for")


def render_module_reference() -> str:
    """The module-level reference: each adapter module with a ``use_for``, then each
    native-loader module that is not an adapter module of its own."""
    lines = [_row(MODULE_HEADER), _rule(("---",) * len(MODULE_HEADER))]
    for f in FRAMEWORKS:
        if f.use_for is None:
            continue
        own = [h for n in f.native_loaders if n.module == f.module for h in n.helpers]
        entry = f.entry_point + "".join(" / " + _short_call(h) for h in own)
        flag = f"`{f.init_flag}=True`" if f.init_flag else (f.init_flag_note or "—")
        extra = f"`[{f.extra}]`" if f.extra else (f.extra_note or "—")
        lines.append(_row((f"`{f.module}`", entry, flag, extra, f.use_for)))
    for f, n in native_loaders():
        if n.module == f.module:
            continue
        lines.append(_row((
            f"`{n.module}`",
            " / ".join(_short_call(h) for h in n.helpers),
            f"— (traced by {n.traced_by})" if n.traced_by else "—",
            f"— (install `{n.requires}`)" if n.requires else "—",
            n.use_for or "—",
        )))
    return "\n".join(lines) + "\n"


#: The generated blocks of ``decimalai-docs/sdk/python/frameworks.mdx``.
GENERATED_BLOCKS: Dict[str, Callable[[], str]] = {
    "capability-table": render_capability_table,
    "module-reference": render_module_reference,
}

_MARK = "decimalai._frameworks"


def begin_marker(name: str) -> str:
    return (f"{{/* BEGIN GENERATED: {_MARK} {name} — edit decimalai/_frameworks.py in "
            f"decimalai-python, then run scripts/check_generated_framework_tables.py "
            f"--write; never edit this table by hand */}}")


def end_marker(name: str) -> str:
    return f"{{/* END GENERATED: {_MARK} {name} */}}"


def _span(text: str, name: str) -> Optional[Tuple[int, int, int, int]]:
    """(start of BEGIN line, end of BEGIN line, start of END line, end of END line)."""
    begin = re.search(r"(?m)^\{/\* BEGIN GENERATED: " + re.escape(_MARK) + " "
                      + re.escape(name) + r"\b[^\n]*\*/\}[ \t]*$", text)
    if begin is None:
        return None
    end = re.compile(r"(?m)^" + re.escape(end_marker(name)) + r"[ \t]*$").search(text, begin.end())
    if end is None:
        raise ValueError(f"{name}: a BEGIN GENERATED marker with no END marker after it")
    return begin.start(), begin.end(), end.start(), end.end()


def extract_block(text: str, name: str) -> Optional[str]:
    """The text between a block's markers, without its surrounding blank lines; None
    when the page has no such block."""
    span = _span(text, name)
    if span is None:
        return None
    return text[span[1]:span[2]].strip("\n") + "\n"


def render_block(name: str) -> str:
    return GENERATED_BLOCKS[name]()


def replace_block(text: str, name: str, body: Optional[str] = None) -> str:
    """``text`` with block ``name`` rewritten from the registry (or from ``body``).

    The markers are rewritten too, so a hand-edited hint in the BEGIN line is restored.
    MDX needs the blank lines: a table that touches a JSX comment is not parsed as one.
    """
    span = _span(text, name)
    if span is None:
        raise ValueError(f"no BEGIN GENERATED marker for {name!r}")
    body = render_block(name) if body is None else body
    block = f"{begin_marker(name)}\n\n{body.strip(chr(10))}\n\n{end_marker(name)}"
    return text[:span[0]] + block + text[span[3]:]


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m decimalai._frameworks",
        description="Print DecimalAI's framework registry, or one of the docs tables it generates.",
    )
    what = parser.add_mutually_exclusive_group()
    what.add_argument("--json", action="store_true", help="the whole registry as JSON (the default)")
    what.add_argument("--block", choices=sorted(GENERATED_BLOCKS),
                      help="one generated docs table, as markdown")
    args = parser.parse_args(argv)
    sys.stdout.write(render_block(args.block) if args.block else to_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
