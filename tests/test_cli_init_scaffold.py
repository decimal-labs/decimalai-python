"""`decimalai init <agent-name>` — the scaffold that closes the gap between
"created an agent in the dashboard" and "an agent is running".

THE TEMPLATE IS CHECKED AS CODE, NOT AS A STRING. Every generated file in this
module is `compile()`d, and the assertions about what it contains are made
against its AST, not against substrings. A string test passes on a file with an
unbalanced quote or an `enable_skill_loader` that landed inside a comment; both
have to fail the build, because the product claim is that the file RUNS.

The four things that must never regress:

  1. `enable_skill_loader=True` is present, as a real keyword on the real
     `instrument()` call. It defaults to False on both adapters, so a file
     without it traces perfectly and delivers none of the agent's skills —
     silently. That is the single most important line in the template.
  2. The agent name is BOUND, and survives a name that needs escaping.
     Unbound, the adapter auto-detects a name off the runnable and files
     traces against a different agent than the one the user configured.
  3. No API key literal is ever written. The file reads DECIMAL_API_KEY from
     the environment, because people commit these.
  4. Refusing to clobber. A scaffold that silently eats an existing agent.py
     is worse than one that does not exist.

No network: every HTTP call is mocked at `DecimalAIClient`.
"""

from __future__ import annotations

import ast
import re
from unittest.mock import MagicMock, patch

import httpx
import pytest
from click.testing import CliRunner

from decimalai import AgentConfig
from decimalai._client import AgentNotFoundError, DecimalAPIError
from decimalai.cli.main import cli
from decimalai.cli.scaffold import (
    NO_PROMPT_SEAM,
    SUPPORTED_FRAMEWORKS,
    UNSCAFFOLDED_WITH_SEAM,
    UnknownFramework,
    UnusableModel,
    default_model,
    normalize_framework,
    normalize_model,
    render_agent_file,
)

AGENT = "refund-bot"
PROMPT = "Never issue a refund over $500."
#: Distinguishes "this test did not say" from "this test said None", which are
#: different states for a prompt: unreadable versus provably not set.
_MISSING = object()
SKILLS = [
    {"skill_name": "refund-policy", "description": "When a refund is allowed",
     "scope": "agent"},
    {"skill_name": "order-lookup", "description": "Find an order by email",
     "scope": "agent"},
    {"skill_name": "tone-check", "description": "House style", "scope": "workspace"},
]


# ── AST helpers: what the file MEANS, not what it says ───────────────


def parse(source: str) -> ast.Module:
    """Parse, which is also the syntax check. A broken template dies here."""
    return ast.parse(source, filename="agent.py")


def calls(tree: ast.Module, func_name: str) -> list:
    """Every `ast.Call` whose callee ends in `func_name`.

    Matches `instrument(...)` and `mod.instrument(...)` alike, so the test
    does not care which import style the template chose.
    """
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
        if name == func_name:
            out.append(node)
    return out


def kwarg(call: ast.Call, name: str):
    """The literal value of one keyword on a call, or `None` if absent."""
    for kw in call.keywords:
        if kw.arg == name:
            return ast.literal_eval(kw.value)
    return None


def string_literals(tree: ast.Module) -> list:
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def imported_modules(tree: ast.Module) -> set:
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


# ── 1. every framework's template is valid, runnable Python ──────────


class TestTemplateIsCode:
    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_compiles(self, framework):
        """compile(), not a substring match. A syntax error fails the suite."""
        source = render_agent_file(AGENT, framework, SKILLS)
        compile(source, "agent.py", "exec")

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_compiles_with_no_skills(self, framework):
        compile(render_agent_file(AGENT, framework, []), "agent.py", "exec")

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_imports_resolve_to_the_adapter_that_has_the_seam(self, framework):
        mods = imported_modules(parse(render_agent_file(AGENT, framework, SKILLS)))
        assert "decimalai" in mods
        expected = {
            "langchain": "decimalai.langchain",
            "openai-agents": "decimalai.openai_agents",
            "pydantic-ai": "decimalai.pydantic_ai",
        }[framework]
        assert expected in mods, f"{framework} must import {expected}"

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_has_one_entry_point_and_one_example_call(self, framework):
        tree = parse(render_agent_file(AGENT, framework, SKILLS))
        funcs = [n.name for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef)]
        assert funcs == ["run"], f"expected exactly one entry point, got {funcs}"
        assert len(calls(tree, "run")) == 1, "expected exactly one example call"

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_model_is_a_named_constant_the_user_can_change(self, framework):
        """One assignment, one place to edit — not a literal buried in a call."""
        tree = parse(render_agent_file(AGENT, framework, SKILLS))
        assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "MODEL"
                           for t in n.targets)]
        assert len(assigns) == 1
        assert isinstance(ast.literal_eval(assigns[0].value), str)

    #: A `--model` each framework can actually resolve. Not one string for all
    #: three: Pydantic AI splits `provider:model` and raises on a bare name, so
    #: the shared "gpt-5-mini" was a value the product refuses — the override
    #: would have been asserted to "reach the file" for a file that cannot run.
    OVERRIDE_MODEL = {
        "langchain": "gpt-5-mini",
        "openai-agents": "gpt-5-mini",
        "pydantic-ai": "openai:gpt-5-mini",
    }

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_model_override_reaches_the_file(self, framework):
        model = self.OVERRIDE_MODEL[framework]
        source = render_agent_file(AGENT, framework, SKILLS, model=model)
        tree = parse(source)
        assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "MODEL"
                           for t in n.targets)]
        assert ast.literal_eval(assigns[0].value) == model

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_the_default_model_is_one_this_framework_can_resolve(self, framework):
        """The zero-flag invocation must produce a file that runs.

        `DEFAULT_MODEL` is an OpenAI model NAME; Pydantic AI wants a
        provider-qualified IDENTIFIER, and the gap is fatal rather than
        cosmetic. Checked through `normalize_model`, which is the same gate
        `--model` goes through — so a framework whose default its own gate would
        reject cannot ship.
        """
        source = render_agent_file(AGENT, framework, SKILLS)   # no --model
        tree = parse(source)
        assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "MODEL"
                           for t in n.targets)]
        rendered = ast.literal_eval(assigns[0].value)
        assert normalize_model(framework, rendered) == rendered
        assert rendered == default_model(framework)

    def test_a_bare_model_name_is_refused_for_pydantic_ai(self):
        """RED before 2026-08-29: this wrote a file that died on line one.

        Pydantic AI's `infer_model` needs the provider half. Without this the
        command printed `✓ Wrote agent.py` and every run of that file raised
        `UserError: Unknown model` — the shape of failure this whole module
        exists to keep out of generated files.
        """
        with pytest.raises(UnusableModel) as e:
            render_agent_file(AGENT, "pydantic-ai", SKILLS, model="gpt-4o-mini")
        # Names the fix, and names it as a string that works.
        assert "openai:gpt-4o-mini" in str(e.value)
        assert normalize_model("pydantic-ai", "openai:gpt-4o-mini") == (
            "openai:gpt-4o-mini"
        )

    def test_the_frameworks_that_take_a_bare_model_name_still_do(self):
        """The refusal is Pydantic AI's rule, not a new house style.

        LangChain's `init_chat_model("gpt-4o-mini")` infers OpenAI from the
        model name and the OpenAI Agents SDK takes the bare name directly, so
        refusing one there would break a spelling both frameworks document.
        """
        for framework in ("langchain", "openai-agents"):
            assert normalize_model(framework, "gpt-4o-mini") == "gpt-4o-mini"

    def test_pydantic_ais_own_test_model_is_not_refused(self):
        """`infer_model` special-cases "test"; so does the gate.

        A gate stricter than the framework it guards refuses something that
        works, which is how a check earns a `# noqa` instead of a fix.
        """
        assert normalize_model("pydantic-ai", "test") == "test"


# ── 2. the flag that makes the whole thing worth shipping ────────────


class TestSkillLoaderIsOn:
    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_enable_skill_loader_is_true_on_instrument(self, framework):
        """Both adapters default it to False. Without it the file traces and
        delivers no skills, with no error anywhere — the exact silent failure
        the scaffold exists to prevent."""
        tree = parse(render_agent_file(AGENT, framework, SKILLS))
        instruments = calls(tree, "instrument")
        assert len(instruments) == 1, "expected exactly one instrument() call"
        assert kwarg(instruments[0], "enable_skill_loader") is True

    #: Every `init()` keyword that installs a span source, and — per framework —
    #: the ONE this template is allowed to carry.
    #:
    #: An allow-list of one rather than a blanket ban, because "no flag" and
    #: "exactly one flag" are the same rule seen from two adapters. On langchain
    #: and openai-agents `instrument()` installs the span source itself, so any
    #: flag here is a SECOND one: `init(langchain=True)` installs tracing with
    #: the loader off and the later `instrument()` cannot undo the disk-mirror
    #: decision it already made, and `init(openai_agents=True)` registers a
    #: second trace processor that double-sends every trace. On pydantic-ai
    #: `instrument()` installs NO span source (that adapter ships the skills rail
    #: and the run boundary; Pydantic AI emits no spans of its own), so `otel=True`
    #: is the first one rather than the second — and without it the generated file
    #: runs, delivers every skill, and traces nothing at all.
    #:
    #: What both halves forbid is the same thing: two span sources on one file.
    TRACING_FLAGS = (
        "langchain", "openai_agents", "crewai", "otel", "adk", "llamaindex",
        "claude_agent_sdk", "autogen", "openai", "anthropic", "google",
    )
    ALLOWED_TRACING_FLAG = {
        "langchain": None,
        "openai-agents": None,
        "pydantic-ai": "otel",
    }

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_init_carries_at_most_the_one_tracing_flag_this_template_needs(
        self, framework
    ):
        tree = parse(render_agent_file(AGENT, framework, SKILLS))
        init_calls = calls(tree, "init")
        assert len(init_calls) == 1
        allowed = self.ALLOWED_TRACING_FLAG[framework]
        carried = [f for f in self.TRACING_FLAGS
                   if kwarg(init_calls[0], f) is not None]
        assert carried == ([allowed] if allowed else []), (
            f"{framework}: init() carries {carried}, expected "
            f"{[allowed] if allowed else []} — see scaffold.py. Two span "
            f"sources on one file means every span is sent twice."
        )

    def test_openai_agents_instruments_before_constructing_the_agent(self):
        """The loader wraps `Agent.__init__`, so an Agent built above the
        instrument() call silently receives nothing on builds without the
        run-time hooks."""
        tree = parse(render_agent_file(AGENT, "openai-agents", SKILLS))
        instrument_line = calls(tree, "instrument")[0].lineno
        agent_line = calls(tree, "Agent")[0].lineno
        assert instrument_line < agent_line


# ── 2b. the agent's own system prompt is what it runs on ─────────────


class TestPromptIsRead:
    """Until 2026-08-25 the generated file ignored the prompt the user typed
    at /agents/new: langchain sent no system message at all, and openai-agents
    hardcoded "You are <name>. Use the skills you are given." — silently
    overwriting it. `tests/test_cli_init_scaffold_runs.py` proves the value
    reaches the model; these prove the CALL is right in the file itself."""

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_load_agent_is_called_once_with_the_bound_name(self, framework):
        tree = parse(render_agent_file(AGENT, framework, SKILLS))
        loads = calls(tree, "load_agent")
        assert len(loads) == 1, "expected exactly one load_agent() call"
        assert [ast.literal_eval(a) for a in loads[0].args] == [AGENT]

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_load_agent_runs_at_module_scope(self, framework):
        """Not inside `run()`. The prompt is needed where the agent is built,
        it is one request per process rather than one per turn, and a wrong
        name should fail at startup instead of on a customer's first
        question."""
        tree = parse(render_agent_file(AGENT, framework, SKILLS))
        in_a_function = {
            id(c)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for c in calls(node, "load_agent")  # type: ignore[arg-type]
        }
        assert not in_a_function

    def test_openai_agents_instructions_are_the_prompt_not_a_literal(self):
        """The regression that shipped: a hardcoded `instructions=` string.
        Asserted as "not a constant", so any future invented default fails
        too — not just the one sentence that used to be there."""
        tree = parse(render_agent_file(AGENT, "openai-agents", SKILLS))
        instructions = [kw for kw in calls(tree, "Agent")[0].keywords
                        if kw.arg == "instructions"]
        assert len(instructions) == 1
        value = instructions[0].value
        assert not isinstance(value, ast.Constant), (
            "instructions= must come from load_agent(), not a literal"
        )
        assert isinstance(value, ast.Attribute) and value.attr == "system_prompt"

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_the_prompt_body_is_never_baked_into_the_file(self, framework):
        """A copy in the file is a second source of truth that goes stale on
        the next dashboard edit — the whole point of reading it at run time."""
        source = render_agent_file(
            AGENT, framework, SKILLS,
            prompt=AgentConfig(agent_name=AGENT, system_prompt=PROMPT,
                               version_number=3),
        )
        assert PROMPT not in source
        # …but the file does say that there IS one, and which version.
        assert "System prompt: 31 characters, version 3" in source

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_an_agent_with_no_prompt_says_so_and_still_generates(self, framework):
        source = render_agent_file(
            AGENT, framework, SKILLS,
            prompt=AgentConfig(agent_name=AGENT, system_prompt=None),
        )
        compile(source, "agent.py", "exec")
        assert "no system prompt set" in source
        assert len(calls(parse(source), "load_agent")) == 1

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_an_unreadable_prompt_says_nothing_rather_than_guessing(self, framework):
        """`prompt=None` is "the scaffold could not read it" (older backend,
        5xx). It must not render as "no prompt set" — that is a claim, and the
        wrong one sends someone to write a prompt they already have."""
        source = render_agent_file(AGENT, framework, SKILLS, prompt=None)
        assert "system prompt" not in source.lower().split("import decimalai")[0]
        assert len(calls(parse(source), "load_agent")) == 1


# ── 3. the agent name is bound, and survives escaping ────────────────


class TestAgentNameBinding:
    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_name_is_interpolated_into_instrument(self, framework):
        tree = parse(render_agent_file(AGENT, framework, SKILLS))
        assert kwarg(calls(tree, "instrument")[0], "agent_name") == AGENT

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    @pytest.mark.parametrize("name", [
        '[Demo] support-agent',      # the real default demo agent
        'quote"bot',                 # naive f-string interpolation → SyntaxError
        "apostrophe's-bot",
        "back\\slash-bot",
        "emoji-🤖-bot",
        # Control characters. Trace ingest accepts these (validate_agent_name
        # blocks only \x00), so an SDK-minted agent can carry one. The header
        # comment was the one place the name skipped both _clean and repr, and
        # a newline there ended the comment and turned the rest of the line
        # into code — a file that does not parse, reported as "✓ Wrote".
        "line\nbreak-bot",
        "carriage\rreturn-bot",
        "esc\x1bbot",
        "tab\tbot",
    ])
    def test_hostile_names_still_parse_and_round_trip(self, framework, name):
        """Escaped by Python's own `repr`, not by hand. A name that breaks the
        file is a name that breaks the product for that user."""
        tree = parse(render_agent_file(name, framework, SKILLS))
        assert kwarg(calls(tree, "instrument")[0], "agent_name") == name

    def test_empty_name_is_refused_rather_than_emitted(self):
        with pytest.raises(ValueError):
            render_agent_file("   ", "langchain", SKILLS)


# ── 4. no secret is ever written to a file people commit ─────────────


class TestNoKeyInFile:
    # Key SHAPES, not the substring "sk-": the template legitimately contains
    # the word "disk-mirror", and a test that cannot tell those apart teaches
    # people to weaken it rather than to fix a real leak.
    KEY_SHAPES = re.compile(r"dai_sk_[A-Za-z0-9_-]{4,}|\bsk-[A-Za-z0-9]{16,}")

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_no_api_key_literal_anywhere(self, framework):
        source = render_agent_file(AGENT, framework, SKILLS)
        assert not self.KEY_SHAPES.search(source)
        assert "dai_sk_" not in source
        tree = parse(source)
        # init() takes no api_key at all — it reads the env itself, which is
        # both fewer characters and impossible to leak.
        assert kwarg(calls(tree, "init")[0], "api_key") is None
        for literal in string_literals(tree):
            assert "dai_sk_" not in literal

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_scaffold_run_with_a_key_in_env_does_not_bake_it_in(
        self, framework, tmp_path, monkeypatch
    ):
        """End to end, not just the renderer: the CLI holds a real key in
        memory while it writes the file."""
        monkeypatch.chdir(tmp_path)
        result = _run_cli([AGENT, "--framework", framework,
                           "--api-key", "dai_sk_supersecret_value"])
        assert result.exit_code == 0, result.output
        written = (tmp_path / "agent.py").read_text()
        assert "dai_sk_supersecret_value" not in written
        assert "supersecret" not in written

    def test_a_default_base_url_is_not_baked_in(self):
        """No inert line pinning the hosted default — but see below: a
        non-default one IS pinned, deliberately."""
        source = render_agent_file(AGENT, "langchain", SKILLS,
                                   base_url=None)
        assert kwarg(calls(parse(source), "init")[0], "base_url") is None

    def test_a_local_base_url_is_baked_in(self):
        """Otherwise a file scaffolded against a local backend silently points
        at production — a URL is not a secret and the surprise costs more."""
        source = render_agent_file(AGENT, "langchain", SKILLS,
                                   base_url="http://localhost:8000")
        assert kwarg(calls(parse(source), "init")[0],
                     "base_url") == "http://localhost:8000"


# ── 5. the skills are NAMED in the file ──────────────────────────────


class TestSkillsAreNamed:
    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_every_assigned_skill_appears_in_a_comment(self, framework):
        source = render_agent_file(AGENT, framework, SKILLS)
        for s in SKILLS:
            assert s["skill_name"] in source

    def test_workspace_scope_is_labelled_not_claimed_as_a_choice(self):
        """A workspace-scoped subscription applies to EVERY agent in the org.
        Listing it unlabelled would claim a deliberate pick that never
        happened."""
        source = render_agent_file(AGENT, "langchain", SKILLS)
        line = [x for x in source.splitlines() if "tone-check" in x][0]
        assert "workspace-wide" in line

    def test_zero_skills_says_so_instead_of_printing_an_empty_list(self):
        source = render_agent_file(AGENT, "langchain", [])
        assert "No skills are attached" in source

    def test_a_multiline_description_cannot_break_out_of_the_comment(self):
        """The header is `#` comments, so a newline is the ONLY escape — which
        is why every description is flattened before it goes in."""
        evil = [{"skill_name": "x", "description": "line one\nMODEL = 'pwned'",
                 "scope": "agent"}]
        source = render_agent_file(AGENT, "langchain", evil)
        tree = parse(source)
        assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "MODEL"
                           for t in n.targets)]
        assert len(assigns) == 1
        assert ast.literal_eval(assigns[0].value) != "pwned"

    def test_a_skill_name_containing_a_quote_does_not_break_the_file(self):
        evil = [{"skill_name": 'weird"""name', "description": "x'''y",
                 "scope": "agent"}]
        compile(render_agent_file(AGENT, "langchain", evil), "agent.py", "exec")


# ── 6. framework selection refuses unknowns, with the reason ─────────


class TestFrameworkSelection:
    def test_default_is_langchain(self):
        assert SUPPORTED_FRAMEWORKS[0] == "langchain"

    @pytest.mark.parametrize("spelling", [
        "langchain", "LangChain", "openai-agents", "openai_agents", "OPENAI-AGENTS",
    ])
    def test_accepted_spellings(self, spelling):
        assert normalize_framework(spelling) in SUPPORTED_FRAMEWORKS

    def test_unknown_framework_lists_what_is_supported(self):
        with pytest.raises(UnknownFramework) as e:
            normalize_framework("langhcain")
        for f in SUPPORTED_FRAMEWORKS:
            assert f in str(e.value)

    @pytest.mark.parametrize("framework", [
        "llamaindex", "claude-agent-sdk", "crewai", "autogen", "otel", "adk",
    ])
    def test_seamless_frameworks_are_refused_with_the_reason(self, framework):
        """These adapters trace but have no prompt seam. A scaffold that
        silently delivers no skills is worse than no scaffold — so the refusal
        has to say that, not just 'invalid choice'."""
        with pytest.raises(UnknownFramework) as e:
            normalize_framework(framework)
        msg = str(e.value)
        assert "no prompt seam" in msg
        assert "langchain" in msg

    @pytest.mark.parametrize("framework", sorted(UNSCAFFOLDED_WITH_SEAM))
    def test_seamed_but_unscaffolded_frameworks_get_a_different_reason(
        self, framework
    ):
        """Read off the ledger, never a list typed here.

        The day a framework gains a template it leaves `UNSCAFFOLDED_WITH_SEAM`,
        and this case has to disappear with it rather than start asserting that
        the product refuses something it now does. A hardcoded
        `["anthropic", "pydantic-ai"]` did exactly that on 2026-08-29.
        """
        with pytest.raises(UnknownFramework) as e:
            normalize_framework(framework)
        assert "no scaffold for it yet" in str(e.value)

    def test_no_framework_is_in_two_ledgers_at_once(self):
        """Scaffoldable, seam-but-no-template, and no-seam are exclusive.

        A name in two of them makes `normalize_framework` answer by dict order
        rather than by fact — and the fact decides whether the user gets a file,
        a "not yet" or a refusal.
        """
        supported, seam, no_seam = (
            set(SUPPORTED_FRAMEWORKS), set(UNSCAFFOLDED_WITH_SEAM),
            set(NO_PROMPT_SEAM),
        )
        assert not supported & seam
        assert not supported & no_seam
        assert not seam & no_seam

    def test_cli_refuses_unknown_framework_before_touching_the_network(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        with patch("decimalai._client.DecimalAIClient") as client_cls:
            result = CliRunner().invoke(cli, [
                "init", AGENT, "--framework", "llamaindex",
                "--api-key", "dai_sk_x",
            ])
        assert result.exit_code == 1
        assert "no prompt seam" in result.output
        client_cls.assert_not_called()
        assert not (tmp_path / "agent.py").exists()


# ── CLI: mocked HTTP, no network ─────────────────────────────────────


def _prompt_payload(system_prompt=PROMPT, version_number=3):
    """The shape `GET /api/v1/agents/{name}/prompt` returns."""
    return {
        "agent_name": AGENT,
        "agent_id": "5f2c1a90-0f1e-4b6c-9a11-2b3c4d5e6f70",
        "resolved_from": None,
        "system_prompt": system_prompt,
        "version_number": version_number if system_prompt is not None else None,
        "content_hash": "b7f4c1" if system_prompt is not None else None,
        "label": None,
        "provenance": "ui",
        "version_mode": "latest",
        "pinned_version_number": None,
    }


def _response(status, detail):
    """A real `httpx.Response` with its request attached.

    Both error classes below are `httpx.HTTPStatusError` subclasses, so they
    need a real response to construct — and a real one is what proves the CLI
    branches on the exception TYPE the client raises rather than on a message
    a test happened to write.
    """
    request = httpx.Request(
        "GET", f"https://api.decimal.ai/api/v1/agents/{AGENT}/prompt",
    )
    return httpx.Response(status, json={"detail": detail}, request=request)


def _agent_not_found(detail="Not Found"):
    """What `get_agent_prompt` raises on a 404 — including from a backend
    whose router has no prompt route at all, which is FastAPI's exact
    unmatched-route body."""
    return AgentNotFoundError(_response(404, detail), agent_name=AGENT)


def _api_error(status, detail):
    return DecimalAPIError(_response(status, detail))


def _client(agents=None, skills=None, agents_exc=None, skills_exc=None,
            prompt=_MISSING, prompt_exc=None):
    """A DecimalAIClient stub answering the three endpoints `init` calls."""
    if agents is None:
        agents = [{"agent_name": AGENT}, {"agent_name": "billing-agent"}]
    if skills is None:
        skills = SKILLS
    if prompt is _MISSING:
        prompt = _prompt_payload()

    def get(url, **kwargs):
        resp = MagicMock()
        if url.endswith("/api/v1/agents"):
            if agents_exc:
                raise agents_exc
            resp.json.return_value = {"agents": agents}
        elif "/skills" in url:
            if skills_exc:
                raise skills_exc
            resp.json.return_value = {"agent_name": AGENT, "skills": skills}
        else:
            raise AssertionError(f"unexpected GET {url}")
        resp.raise_for_status = MagicMock()
        return resp

    def get_agent_prompt(name, **kwargs):
        if prompt_exc:
            raise prompt_exc
        return prompt

    c = MagicMock()
    c._http.get.side_effect = get
    c.get_agent_prompt.side_effect = get_agent_prompt
    return c


def _run_cli(args, client=None):
    with patch("decimalai._client.DecimalAIClient",
               return_value=client or _client()):
        return CliRunner().invoke(cli, ["init", *args])


class TestCliWritesTheFile:
    def test_writes_agent_py_that_compiles(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _run_cli([AGENT, "--api-key", "dai_sk_x"])
        assert result.exit_code == 0, result.output
        source = (tmp_path / "agent.py").read_text()
        compile(source, "agent.py", "exec")
        tree = parse(source)
        assert kwarg(calls(tree, "instrument")[0], "enable_skill_loader") is True
        assert kwarg(calls(tree, "instrument")[0], "agent_name") == AGENT

    def test_out_path_is_honored(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _run_cli([AGENT, "--api-key", "dai_sk_x", "--out", "bot.py"])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "bot.py").exists()
        assert not (tmp_path / "agent.py").exists()

    def test_next_steps_name_install_run_and_where_the_trace_lands(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        result = _run_cli([AGENT, "--api-key", "dai_sk_x"])
        assert 'pip install "decimalai[langchain]"' in result.output
        assert "python agent.py" in result.output
        assert "DECIMAL_API_KEY" in result.output
        assert f"https://app.decimal.ai/agents/{AGENT}" in result.output

    def test_openai_agents_next_steps_name_the_right_extra(
        self, tmp_path, monkeypatch
    ):
        """`decimalai[openai]` is the openai SDK, NOT the Agents SDK — copying
        the wrong extra leaves the import failing on a machine that looks
        correctly set up."""
        monkeypatch.chdir(tmp_path)
        result = _run_cli([AGENT, "--api-key", "dai_sk_x",
                           "--framework", "openai-agents"])
        assert 'pip install "decimalai[openai-agents]"' in result.output

    def test_the_prompt_is_summarized_in_the_header(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _run_cli([AGENT, "--api-key", "dai_sk_x"])
        assert result.exit_code == 0, result.output
        source = (tmp_path / "agent.py").read_text()
        assert "System prompt: 31 characters, version 3" in source
        assert PROMPT not in source          # a length, never a copy

    def test_an_agent_with_no_prompt_is_told_it_has_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _run_cli([AGENT, "--api-key", "dai_sk_x"],
                          client=_client(prompt=_prompt_payload(None)))
        assert result.exit_code == 0, result.output
        assert "no system prompt set" in (tmp_path / "agent.py").read_text()

    @pytest.mark.parametrize("exc", [
        httpx.ConnectError("refused"),
        _api_error(503, "upstream connect error"),
    ])
    def test_a_transient_prompt_failure_still_scaffolds(
        self, exc, tmp_path, monkeypatch
    ):
        """A timeout or a 5xx does NOT prove the generated file cannot run —
        it reads the prompt itself at run time, and `load_agent()` raises
        there rather than returning an empty prompt. So this read feeds ONE
        header comment, and losing it costs a comment, not the scaffold."""
        monkeypatch.chdir(tmp_path)
        result = _run_cli([AGENT, "--api-key", "dai_sk_x"],
                          client=_client(prompt_exc=exc))
        assert result.exit_code == 0, result.output
        source = (tmp_path / "agent.py").read_text()
        compile(source, "agent.py", "exec")
        # Silent about a prompt it could not read — and still reads it at run time.
        assert "System prompt:" not in source
        assert "no system prompt set" not in source
        assert 'decimalai.load_agent("refund-bot")' in source
        # Said out loud, because a silently missing comment is indistinguishable
        # from an agent that has no prompt.
        assert "Could not read" in result.output

    def test_a_backend_with_no_prompt_route_refuses_to_scaffold(
        self, tmp_path, monkeypatch
    ):
        """The one prompt-read failure that IS fatal, and it is fatal for a
        structural reason rather than a string heuristic: step 1 already
        proved this agent is in this workspace, so a 404 on its prompt can
        only mean the route is missing. The generated file calls
        `load_agent()` at module scope, so writing it anyway would hand
        someone a program that dies on line one, every run, in silence."""
        monkeypatch.chdir(tmp_path)
        result = _run_cli(
            [AGENT, "--api-key", "dai_sk_x"],
            client=_client(prompt_exc=_agent_not_found()),
        )
        assert result.exit_code == 1, result.output
        assert not (tmp_path / "agent.py").exists()
        assert "does not serve agent prompts" in result.output

    def test_skill_count_is_reported(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _run_cli([AGENT, "--api-key", "dai_sk_x"])
        assert "3 skills" in result.output

    @pytest.mark.parametrize("prompt, expected", [
        (_prompt_payload(), "3 skills, system prompt"),
        (_prompt_payload(None), "3 skills, no system prompt"),
    ])
    def test_whether_the_prompt_was_picked_up_is_reported(
        self, prompt, expected, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        result = _run_cli([AGENT, "--api-key", "dai_sk_x"],
                          client=_client(prompt=prompt))
        assert expected in result.output

    def test_an_unreadable_prompt_is_not_reported_as_absent(
        self, tmp_path, monkeypatch
    ):
        """"Could not read it" and "there isn't one" are different facts, and
        only one of them sends someone to go write a prompt they already
        wrote."""
        monkeypatch.chdir(tmp_path)
        result = _run_cli([AGENT, "--api-key", "dai_sk_x"],
                          client=_client(prompt_exc=_api_error(500, "boom")))
        assert result.exit_code == 0, result.output
        assert "no system prompt" not in result.output
        assert "Could not read this agent's system prompt" in result.output

    def test_agent_name_is_url_quoted_in_the_dashboard_link(
        self, tmp_path, monkeypatch
    ):
        """`[Demo] support-agent` is a real default agent name; interpolated
        raw it produces a URL nothing can follow."""
        monkeypatch.chdir(tmp_path)
        name = "[Demo] support-agent"
        result = _run_cli([name, "--api-key", "dai_sk_x"],
                          client=_client(agents=[{"agent_name": name}]))
        assert result.exit_code == 0, result.output
        assert "%5BDemo%5D%20support-agent" in result.output


class TestRefusesToClobber:
    def test_existing_file_is_not_overwritten(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        existing = tmp_path / "agent.py"
        existing.write_text("# my real agent, do not eat\n")
        result = _run_cli([AGENT, "--api-key", "dai_sk_x"])
        assert result.exit_code == 1
        assert "already exists" in result.output
        assert existing.read_text() == "# my real agent, do not eat\n"

    def test_the_refusal_names_all_three_ways_out(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.py").write_text("x = 1\n")
        result = _run_cli([AGENT, "--api-key", "dai_sk_x"])
        assert "--force" in result.output
        assert "--out" in result.output
        assert "--dry-run" in result.output

    def test_force_overwrites(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.py").write_text("# old\n")
        result = _run_cli([AGENT, "--api-key", "dai_sk_x", "--force"])
        assert result.exit_code == 0, result.output
        source = (tmp_path / "agent.py").read_text()
        assert "# old" not in source
        compile(source, "agent.py", "exec")

    def test_out_path_is_also_protected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "bot.py").write_text("# mine\n")
        result = _run_cli([AGENT, "--api-key", "dai_sk_x", "--out", "bot.py"])
        assert result.exit_code == 1
        assert (tmp_path / "bot.py").read_text() == "# mine\n"

    def test_force_refuses_a_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.py").mkdir()
        result = _run_cli([AGENT, "--api-key", "dai_sk_x", "--force"])
        assert result.exit_code == 1
        assert "is a directory" in result.output

    def test_clobber_check_runs_before_the_skills_fetch(self, tmp_path, monkeypatch):
        """Refuse after the name is confirmed, but before the second round
        trip and before anything is written."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.py").write_text("x = 1\n")
        client = _client()
        result = _run_cli([AGENT, "--api-key", "dai_sk_x"], client=client)
        assert result.exit_code == 1
        urls = [c.args[0] for c in client._http.get.call_args_list]
        assert urls == ["/api/v1/agents"], urls

    def test_a_wrong_name_is_diagnosed_as_a_wrong_name_not_a_file_conflict(
        self, tmp_path, monkeypatch
    ):
        """Found by running it: with the file check first, a typo'd name in a
        directory that already had an agent.py was reported as "agent.py
        already exists" and offered `--force` — advice that, if followed,
        produces a completely different error. The name is the thing they got
        wrong and cannot fix locally, so it is diagnosed first."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.py").write_text("x = 1\n")
        result = _run_cli(["refund_bot", "--api-key", "dai_sk_x"])
        assert result.exit_code == 1
        assert "Did you mean" in result.output
        assert "already exists" not in result.output
        assert "--force" not in result.output


class TestDryRun:
    def test_prints_the_file_and_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _run_cli([AGENT, "--api-key", "dai_sk_x", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert not (tmp_path / "agent.py").exists()
        compile(result.output, "agent.py", "exec")

    def test_dry_run_output_is_exactly_the_file(self, tmp_path, monkeypatch):
        """The reviewable path has to show what would land, byte for byte —
        otherwise reviewing it proves nothing."""
        monkeypatch.chdir(tmp_path)
        printed = _run_cli([AGENT, "--api-key", "dai_sk_x", "--dry-run"]).output
        _run_cli([AGENT, "--api-key", "dai_sk_x"])
        assert printed == (tmp_path / "agent.py").read_text()

    def test_dry_run_does_not_clobber_and_does_not_refuse(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.py").write_text("# mine\n")
        result = _run_cli([AGENT, "--api-key", "dai_sk_x", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "agent.py").read_text() == "# mine\n"


class TestAgentResolution:
    def test_unknown_agent_points_at_agents_new_and_writes_nothing(
        self, tmp_path, monkeypatch
    ):
        """Inventing one locally would file traces under a name the workspace
        does not have: empty page, unresolved skills, no error anywhere."""
        monkeypatch.chdir(tmp_path)
        result = _run_cli(["ghost-bot", "--api-key", "dai_sk_x"])
        assert result.exit_code == 1
        assert "No agent named 'ghost-bot'" in result.output
        assert "https://app.decimal.ai/agents/new" in result.output
        assert not (tmp_path / "agent.py").exists()

    def test_a_near_miss_is_named(self, tmp_path, monkeypatch):
        """refund_bot vs refund-bot: identical at a glance, and the difference
        is an agent page that stays empty forever."""
        monkeypatch.chdir(tmp_path)
        result = _run_cli(["refund_bot", "--api-key", "dai_sk_x"])
        assert result.exit_code == 1
        assert "Did you mean" in result.output
        assert AGENT in result.output

    def test_an_unrelated_name_lists_the_workspace_instead_of_guessing(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        result = _run_cli(["totally-different", "--api-key", "dai_sk_x"])
        assert "Did you mean" not in result.output
        assert "billing-agent" in result.output

    def test_empty_workspace_still_points_at_agents_new(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        result = _run_cli([AGENT, "--api-key", "dai_sk_x"],
                          client=_client(agents=[]))
        assert result.exit_code == 1
        assert "/agents/new" in result.output

    def test_an_agent_with_no_skills_still_scaffolds(self, tmp_path, monkeypatch):
        """Zero skills is a legitimate state, not an error — the file says so."""
        monkeypatch.chdir(tmp_path)
        result = _run_cli([AGENT, "--api-key", "dai_sk_x"],
                          client=_client(skills=[]))
        assert result.exit_code == 0, result.output
        source = (tmp_path / "agent.py").read_text()
        compile(source, "agent.py", "exec")
        assert "No skills are attached" in source
        assert "0 skills" in result.output


class TestErrorTriage:
    """Same triage the keyless `init` uses: when the server ANSWERS, it is not
    a connection problem, and saying so sends people off debugging DNS."""

    def _exc(self, status):
        return httpx.HTTPStatusError(
            f"{status}", request=MagicMock(),
            response=MagicMock(status_code=status),
        )

    def test_401_says_invalid_key(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _run_cli([AGENT, "--api-key", "dai_sk_bad"],
                          client=_client(agents_exc=self._exc(401)))
        assert result.exit_code == 1
        assert "Invalid API key" in result.output
        assert "Connection failed" not in result.output

    def test_500_names_the_status(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _run_cli([AGENT, "--api-key", "dai_sk_x"],
                          client=_client(agents_exc=self._exc(500)))
        assert result.exit_code == 1
        assert "HTTP 500" in result.output

    def test_transport_error_says_connection_failed(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _run_cli([AGENT, "--api-key", "dai_sk_x"],
                          client=_client(agents_exc=httpx.ConnectError("refused")))
        assert result.exit_code == 1
        assert "Connection failed" in result.output

    def test_a_failing_skills_call_does_not_leave_a_half_written_file(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        result = _run_cli([AGENT, "--api-key", "dai_sk_x"],
                          client=_client(skills_exc=self._exc(500)))
        assert result.exit_code == 1
        assert not (tmp_path / "agent.py").exists()

    def test_no_key_points_at_settings(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for k in ("DECIMAL_API_KEY", "DECIMALAI_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        result = CliRunner().invoke(cli, ["init", AGENT])
        assert result.exit_code == 1
        assert "No API key found" in result.output
        assert "https://app.decimal.ai/settings" in result.output


class TestBackwardCompatibility:
    """`decimalai init` with no argument keeps doing what it always did —
    the verify-and-test-trace flow every existing doc and script calls."""

    def test_no_argument_still_verifies(self):
        client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {"workspace_id": "ws_1", "scope": "workspace"}
        resp.raise_for_status = MagicMock()
        client._http.get.return_value = resp
        with patch("decimalai._client.DecimalAIClient", return_value=client):
            result = CliRunner().invoke(cli, [
                "init", "--api-key", "dai_sk_x", "--no-test-trace",
            ])
        assert result.exit_code == 0, result.output
        assert "Connected to workspace: ws_1" in result.output

    def test_no_argument_advertises_the_scaffold(self):
        client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {"workspace_id": "ws_1"}
        resp.raise_for_status = MagicMock()
        client._http.get.return_value = resp
        with patch("decimalai._client.DecimalAIClient", return_value=client):
            result = CliRunner().invoke(cli, [
                "init", "--api-key", "dai_sk_x", "--no-test-trace",
            ])
        assert "decimalai init <agent-name>" in result.output

    @pytest.mark.parametrize("flag", [
        ["--force"], ["--dry-run"], ["--out", "x.py"], ["--model", "gpt-4o"],
    ])
    def test_scaffold_flags_without_a_name_fail_loudly(self, flag, monkeypatch):
        """Otherwise they are silent no-ops and the user waits for a file that
        was never going to be written."""
        result = CliRunner().invoke(cli, ["init", "--api-key", "dai_sk_x", *flag])
        assert result.exit_code == 1
        assert "needs an agent name" in result.output

    def test_help_documents_both_modes(self):
        result = CliRunner().invoke(cli, ["init", "--help"])
        assert result.exit_code == 0
        assert "decimalai init refund-bot" in result.output
        assert "--framework" in result.output


class TestConsistencyWithFrontendSnippets:
    """The dashboard's per-framework snippets and this template answer
    different questions ("add it to your agent" vs "here IS an agent") but
    must not contradict each other. Locked here because the two files live in
    different repos and nothing else would notice them drifting."""

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_same_shape_as_the_snippet(self, framework):
        tree = parse(render_agent_file(AGENT, framework, SKILLS))
        # One bare init(), one instrument() carrying both the name and the
        # loader — the shape the dashboard snippets settled on 2026-08-22.
        assert len(calls(tree, "init")) == 1
        inst = calls(tree, "instrument")
        assert len(inst) == 1
        assert kwarg(inst[0], "agent_name") == AGENT
        assert kwarg(inst[0], "enable_skill_loader") is True
