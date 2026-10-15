"""The eight native skill loaders, one module each, and what each one is.

A loader module OBSERVES and returns a dict; it never asserts (an AST guard in
``test_native_coverage.py`` holds it to that). Grading is ``contract.py``'s job, so
adding a loader means writing an observer, never an assertion.

Every module exposes two functions, both run only inside that loader's child
process (``_child.py``), with the project as the working directory:

``discover(ctx) -> dict``
    The loader's own discovery call and nothing else: ``{"listed": [...] | None,
    "raised": str | None, "channel": [...]}``. The ledger matrix (``matrix.py``)
    runs it once per single-defect project.
``observe(ctx) -> dict``
    The documented snippet, built on ``skills_dir()``, driven by a scripted model
    through the scenario in ``oracle.SCENARIO``: load the target skill, read its
    resource, run its script, answer. Returns the observation below.

The observation (every key present; ``None`` means "not observed"):

``listed``          names the loader's discovery lists (N1)
``offered``         names in the listing the model was given, where observable (N1, N7)
``channel``         the loader's own skill-warning channel (N2)
``body_text``       what the model got when it loaded the target (N3)
``body``            the target's body as the loader delivers it, frontmatter removed (N3)
``resource_text``   the resource's bytes through the loader's own route (N4)
``script_stdout``   the script's output through the loader's own route (N5)
``stamp``           ``{"via", "path", "metadata"}`` the target's stamp (N6)
``user_only``       ``{"offered", "refusal"}`` the user-only skill (N7)
``exec_bit``        ``{"path", "source", "copy"}`` the target script's exec bit (N10)
``claude``          the Claude-only facts (N8, N9)
``proofs``          facts behind a declared N/A or a ledgered limit, re-proven each run
``errors``          ``{stage: "Type: message"}`` for every stage that raised
``versions``        the distributions graded

This module imports no framework: the parent process reads it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Tuple


@dataclass(frozen=True)
class Loader:
    """One native loader: how to find it, and what it is graded on."""

    #: The id used everywhere: test ids, ``LOADER_ENVS``, the ledger, the docs table.
    name: str
    #: ``loaders/<module>.py``.
    module: str
    #: Distributions that must be installed; their versions are recorded.
    distributions: Tuple[str, ...]
    #: Top-level imports the loader needs (``find_spec``, never imported in the parent).
    imports: Tuple[str, ...]
    #: The documented entry call, as the docs page states it.
    snippet: str


#: Items every loader is graded on. N8 and N9 are Claude Code's alone (the scoped
#: ``Bash`` permission and the plugin root exist only there).
COMMON_ITEMS = ("N1", "N2", "N3", "N4", "N5", "N6", "N7", "N10")
CLAUDE_ITEMS = ("N8", "N9")

LOADERS: Tuple[Loader, ...] = (
    Loader("claude-agent-sdk", "claude", ("claude-agent-sdk",), ("claude_agent_sdk",),
           'query(prompt, options=ClaudeAgentOptions(plugins=[{"type": "local", '
           '"path": str(decimalai.skills_dir().parent)}]))'),
    Loader("google-adk", "adk", ("google-adk",), ("google.adk",),
           "SkillToolset(skills=load_skills_from_dir(decimalai.skills_dir()), "
           "code_executor=UnsafeLocalCodeExecutor())"),
    Loader("openai-agents", "openai_agents", ("openai-agents",), ("agents",),
           "Skills(lazy_from=LocalDirLazySkillSource(source=LocalDir("
           "src=decimalai.skills_dir(relative_to=Path.cwd()))))"),
    Loader("deepagents", "deepagents", ("deepagents",), ("deepagents",),
           "create_deep_agent(model, skills=[str(decimalai.skills_dir())], "
           "backend=LocalShellBackend(root_dir=project, virtual_mode=False))"),
    Loader("agent-framework", "agent_framework", ("agent-framework-core",), ("agent_framework",),
           "SkillsProvider.from_paths(str(decimalai.skills_dir()), script_runner=runner)"),
    Loader("strands-agents", "strands", ("strands-agents",), ("strands",),
           "AgentSkills(skills=str(decimalai.skills_dir()), strict=True)"),
    Loader("pydantic-ai-harness", "pydantic_harness", ("pydantic-ai-harness", "pydantic-ai-slim"),
           ("pydantic_ai_harness", "pydantic_ai"),
           "Skills(str(decimalai.skills_dir(relative_to=workspace))) inside LocalWorkspace(workspace)"),
    Loader("crewai", "crewai", ("crewai",), ("crewai",),
           "Agent(..., skills=[decimalai.skills_dir()])"),
)

LOADERS_BY_NAME: Dict[str, Loader] = {loader.name: loader for loader in LOADERS}
LOADER_MODULES: Tuple[str, ...] = tuple(loader.module for loader in LOADERS)


def items_for(name: str) -> Tuple[str, ...]:
    return COMMON_ITEMS + (CLAUDE_ITEMS if name == "claude-agent-sdk" else ())


#: The loaders the docs page ``guides/native-skill-loaders.mdx`` advertises, as the
#: ids in its table's first column. Vendored as a literal, edited by hand when the
#: docs table changes, so the coverage guard bites in a checkout without the docs
#: repo (derived from ``LOADERS``, it could never disagree with them); re-derived
#: from the page whenever it is on disk
#: (``test_native_coverage.test_loader_table_matches_the_docs``).
NATIVE_LOADERS_SNAPSHOT: FrozenSet[str] = frozenset({
    "claude-agent-sdk",
    "google-adk",
    "openai-agents",
    "deepagents",
    "agent-framework",
    "strands-agents",
    "pydantic-ai-harness",
    "crewai",
})
