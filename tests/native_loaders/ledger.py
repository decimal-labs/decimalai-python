"""Every exception this suite grants, declared, with the fact that proves it.

Four kinds, each re-proven on every run, so none can outlive the fact behind it:

``DECLARED_NA``
    A cell the loader cannot be graded on the strict way, because the framework
    lacks the capability (CrewAI has no resource reader, no script runner). The
    cell is skipped as N/A only after its proof holds on this run; the day the
    framework ships the capability, the proof fails and the cell must be graded.
``UPSTREAM_GAPS``
    A cell whose strict check fails because of a known upstream behaviour, in
    exactly the documented way (its ``signature``). Reported as xfail. If the
    strict check starts PASSING the cell fails, so the line is deleted on the
    commit that notices (the ledger cleans itself); if it fails some OTHER way,
    the cell fails with the strict detail.
``PROOFS``
    A fact behind how an item is graded on one loader (MAF ships no script
    runner, so N5 uses the suite's own; openai-agents exposes no metadata, so the
    stamp is read from the file). Checked inside the cell; a fact that stops
    holding fails the cell.
``NOISE_ALLOWLIST``
    Lines the loader's own warning channel prints on every healthy run (N2). Each
    entry must still occur on the graded project, or
    ``test_native_loaders.test_every_noise_allowance_still_occurs`` fails.

Plus the negative control: ``NEGATIVE_CONTROL_PREDICTED``, the cells the contract
must fail on SDK 0.13.9's output, and ``NEGATIVE_CONTROL_RED``, every cell it does
fail there (recorded, so a weakened item shows as a cell that stopped failing).

The single-defect ledger matrix (loader x defect, run by ``matrix.py`` in the
release gate, not per commit) is ``ledger_matrix.json`` beside this file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, NoReturn, Tuple

import pytest

from .oracle import SCENARIO

Check = Callable[[Dict[str, Any]], Tuple[bool, str]]


@dataclass(frozen=True)
class Exemption:
    reason: str
    check: Check


def _proof(key: str, ok: Callable[[Any], bool], want: str) -> Check:
    def check(obs: Dict[str, Any]) -> Tuple[bool, str]:
        value = (obs.get("proofs") or {}).get(key)
        return (bool(ok(value)), f"proofs[{key!r}] = {value!r}; expected {want}")
    return check


def _all(*checks: Check) -> Check:
    def check(obs: Dict[str, Any]) -> Tuple[bool, str]:
        results = [c(obs) for c in checks]
        return all(ok for ok, _ in results), "; ".join(d for _, d in results)
    return check


def _field(path: Tuple[str, ...], ok: Callable[[Any], bool], want: str) -> Check:
    def check(obs: Dict[str, Any]) -> Tuple[bool, str]:
        value: Any = obs
        for part in path:
            value = (value or {}).get(part) if isinstance(value, dict) else None
        return bool(ok(value)), f"{'.'.join(path)} = {value!r}; expected {want}"
    return check


# ── declared N/As ────────────────────────────────────────────────────────────

_CREWAI_ONLY_LOAD_SKILL = _proof(
    "skill_tools", lambda v: isinstance(v, list) and set(v) <= {"load_skill"},
    "no skill tool but load_skill (CrewAI's only one)")

DECLARED_NA: Dict[str, Exemption] = {
    "crewai:N4": Exemption(
        "CrewAI has no native resource reader: the only skill tool it gives the model "
        "is load_skill, and load_resources() renders a file listing, never the bytes",
        _all(_CREWAI_ONLY_LOAD_SKILL,
             _proof("resource_listing", lambda v: bool(v) and SCENARIO.resource in v,
                    f"a listing that names {SCENARIO.resource}"))),
    "crewai:N5": Exemption(
        "CrewAI has no native script runner: Agent(allow_code_execution=True) is a "
        "deprecated no-op that yields no tools",
        _all(_CREWAI_ONLY_LOAD_SKILL,
             _proof("code_execution_tools", lambda v: v == 0, "0 tools"))),
}

#: The number above, written down: the set comparison says WHICH exemption moved,
#: this says THAT the total moved, on one line of a diff.
NA_BUDGET = 2

# ── known upstream gaps (xfail, self-cleaning) ───────────────────────────────

_OFFERS_USER_ONLY = Exemption(
    "offers a disable-model-invocation skill to the model; only Claude Code honours "
    "the key (an upstream fact the docs loader page states)",
    _field(("user_only", "offered"), lambda v: v is True, "True (offered to the model)"))

UPSTREAM_GAPS: Dict[str, Exemption] = {
    "openai-agents:N10": Exemption(
        "lazy materialization copies the skill to .agents/<name> without the exec bit",
        _all(_field(("exec_bit", "source"), lambda v: v is True, "True (the folder keeps it)"),
             _field(("exec_bit", "copy"), lambda v: v is False, "False (the copy drops it)"))),
    **{f"{name}:N7": _OFFERS_USER_ONLY for name in (
        "google-adk", "openai-agents", "deepagents", "agent-framework", "strands-agents",
        "pydantic-ai-harness", "crewai")},
}

GAP_BUDGET = 8

# ── facts behind how an item is graded on one loader ────────────────────────

PROOFS: Dict[str, Exemption] = {
    "agent-framework:N5": Exemption(
        "MAF ships no script runner, so N5 is graded with the suite's own runner: MAF "
        "listed the script and handed the runner the right full path",
        _proof("no_runner", lambda v: isinstance(v, str) and "requires a runner" in v,
               "MAF's own 'requires a runner' refusal")),
    "openai-agents:N6": Exemption(
        "the loader exposes no metadata (SkillMetadata is name, description, path), so "
        "the stamp is read from the materialized SKILL.md through exec_command",
        _proof("metadata_fields", lambda v: v == ["description", "name", "path"],
               "['description', 'name', 'path']")),
    "pydantic-ai-harness:N6": Exemption(
        "SkillDefinition keeps no metadata, so the stamp is read from the file at the "
        "path the harness reports",
        _proof("definition_fields", lambda v: isinstance(v, list) and "metadata" not in v,
               "a field list without 'metadata'")),
    "strands-agents:N1": Exemption(
        "AgentSkills(strict=True) skips a bad folder instead of raising; the strict parse "
        "is Skill.from_file(strict=True), which must accept every folder",
        _proof("strict_parse", lambda v: isinstance(v, dict) and bool(v)
               and all(r == "ok" for r in v.values()), "'ok' for every folder")),
}

# ── N2: lines a healthy run prints ──────────────────────────────────────────

NOISE_ALLOWLIST: Dict[str, List[Tuple[str, str]]] = {
    "pydantic-ai-harness": [(
        r"^UserWarning: Ignoring unsupported Agent Skill behavioral frontmatter fields: ",
        "the harness enforces neither allowed-tools nor disable-model-invocation and "
        "warns once per run that it ignores them",
    )],
}


def allowlisted(loader: str, line: str) -> bool:
    return any(re.search(pattern, line) for pattern, _ in NOISE_ALLOWLIST.get(loader, ()))


# ── the negative control: SDK 0.13.9's output ────────────────────────────────

_NON_CLAUDE = ("google-adk", "openai-agents", "deepagents", "agent-framework",
               "strands-agents", "pydantic-ai-harness", "crewai")
_ALL = ("claude-agent-sdk",) + _NON_CLAUDE

#: Predicted before the suite ever ran on 0.13.9's output, and kept as a ratchet:
#: * N1 — ADK, MAF, Strands and pydantic reject the namespaced ``name:``; ADK, MAF
#:   and CrewAI reject the list-typed fields; openai-agents and deepagents list the
#:   skill under the namespaced name (renamed);
#: * N6 — the v1 export stamps no ids into SKILL.md (and its lock carries none);
#: * N9 — no ``plugin.json``, so Claude names the plugin after its folder;
#: * N10 — the script is written without its exec bit.
NEGATIVE_CONTROL_PREDICTED: FrozenSet[str] = frozenset(
    [f"{n}:N1" for n in _NON_CLAUDE] + [f"{n}:N6" for n in _ALL]
    + [f"{n}:N10" for n in _ALL] + ["claude-agent-sdk:N9"])

#: Every cell the contract fails on 0.13.9's output (a superset of the prediction),
#: recorded from a run. A cell that leaves this set is a contract item that went
#: soft, or a loader that got looser; a cell that joins it is a stricter item. Both
#: are worth a look, and re-recording is a deliberate edit here.
#
#: Recorded 2026-10-14 (claude-agent-sdk 0.2.166 / CLI 2.1.299, google-adk 2.12.0,
#: openai-agents 0.23.1, deepagents 0.7.25, agent-framework-core 1.21.0,
#: pydantic-ai-harness 0.56.0, strands-agents 1.60.0, crewai 1.15.27); unchanged on
#: strands-agents 1.61.0 and crewai 1.15.28 (2026-10-15). Beyond the prediction:
#: * Claude N8: 0.13.9 writes ``allowed-tools`` as a YAML list holding a bare
#:   ``Bash``, which Claude Code pre-approves, so a command outside any skill's rule
#:   ran with no prompt;
#: * Claude passes N7 there: 0.13.9 does write ``disable-model-invocation: true``;
#: * N3-N5 fail wherever the loader dropped the target, or listed it under another
#:   name and so never told the model where it is (deepagents); openai-agents still
#:   materializes it by folder name, so its N3-N5 pass;
#: * N2: the loaders that drop a folder say so on their own channel;
#: * N7 elsewhere: the user-only skill was dropped too, so the ledgered upstream fact
#:   cannot be observed;
#: * CrewAI N5 stays a declared N/A: no runner is a framework fact, true of any folder.
NEGATIVE_CONTROL_RED: Dict[str, FrozenSet[str]] = {
    "claude-agent-sdk": frozenset({"N1", "N6", "N8", "N9", "N10"}),
    "google-adk": frozenset({"N1", "N3", "N4", "N5", "N6", "N7", "N10"}),
    "openai-agents": frozenset({"N1", "N6", "N7", "N10"}),
    "deepagents": frozenset({"N1", "N2", "N3", "N4", "N5", "N6", "N7", "N10"}),
    "agent-framework": frozenset({"N1", "N2", "N3", "N4", "N5", "N6", "N7", "N10"}),
    "pydantic-ai-harness": frozenset({"N1", "N2", "N3", "N4", "N5", "N6", "N7", "N10"}),
    "strands-agents": frozenset({"N1", "N2", "N3", "N4", "N5", "N6", "N7", "N10"}),
    "crewai": frozenset({"N1", "N2", "N3", "N4", "N6", "N7", "N10"}),
}


# ── skips ────────────────────────────────────────────────────────────────────

#: Test functions allowed to skip on an ENVIRONMENT condition, each with it.
#: ``test_native_coverage.test_no_undeclared_skip_call_sites`` fails on any other
#: ``pytest.skip`` in this directory.
ENVIRONMENT_SKIPS: Dict[str, str] = {
    "test_loader_table_matches_the_docs": (
        "the decimalai-docs repo is a sibling checkout most clones (and CI) lack, and "
        "until its native-loader page exists there is no table to compare; the "
        "in-repo guards beside it do not skip"
    ),
    "test_the_tokenizer_agrees_with_skillfmt": (
        "skillevaluation.skillfmt arrives in skillevaluation 0.8.0; an environment with "
        "an older skillevaluation has nothing to compare the contract's tokenizer with, "
        "and the synthetic N8 cases in test_contract.py still grade it"
    ),
}

SKIP_KINDS = ("na", "elsewhere", "unavailable")


def skip_declared(kind: str, key: str, reason: str) -> NoReturn:
    """Skip one cell, but only for a reason this ledger declares; anything else FAILS."""
    import os

    from tests.conformance.environments import (
        ENV_VAR,
        current_env,
        loader_graded_elsewhere,
    )

    if kind == "na":
        if key not in DECLARED_NA:
            pytest.fail(f"{key} was graded N/A but is not declared in ledger.DECLARED_NA "
                        f"(reason given: {reason}). Declare it with its proof, or grade it.")
    elif kind == "elsewhere":
        home = loader_graded_elsewhere(key.partition(":")[0])
        if home is None:
            pytest.fail(f"{key} belongs to this environment ({current_env()!r}) and must be "
                        f"graded here, not skipped.")
        reason = f"{reason} (graded in the {home} environment, not {current_env()!r})"
    elif kind == "unavailable":
        if (os.environ.get(ENV_VAR) or "").strip():
            pytest.fail(f"{key}: {reason}, and {ENV_VAR}={current_env()} says this "
                        f"environment holds it. A named environment never grades by skipping.")
    else:
        pytest.fail(f"skip_declared({kind!r}): unknown kind; known: {', '.join(SKIP_KINDS)}")
    pytest.skip(reason)

