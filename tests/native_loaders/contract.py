"""The contract: the ONLY place in this suite that decides pass or fail.

Each loader module observes and returns a dict (see ``loaders/__init__.py``); this
file grades that dict against the expectation, which is the lock v2 entries of the
project being graded. "Install name" means a key of ``.decimal/skills.lock``,
which is also the folder name and the frontmatter ``name``.

========  ==================================================================
N1        the loader lists exactly the lock's install names; none missing, none
          extra, none renamed, and the model's listing (where observable) holds
          every model-invocable one. Seven of the eight loaders drop or rename a
          bad folder without raising, so this set equality is the main detector.
N2        the loader's own skill-warning channel is quiet, apart from the
          self-cleaning allowlist in ``ledger.NOISE_ALLOWLIST``.
N3        the text the model got for the target skill carries THAT skill's
          sentinel and the loader's parsed body verbatim (a delivery cut short after
          the sentinel fails), and ``sha256(body.strip())`` equals
          ``decimalai-body-sha256``.
N4        the target's resource, read through the loader's own route, equals the
          file the lock pins (exact bytes where the route returns bytes; line by
          line where it is a file viewer that numbers lines).
N5        the target's script, run through the loader's own route, prints the
          expected line.
N6        the stamp (``decimalai-skill-id``, ``-version-id``, ``-version``,
          ``-body-sha256``, ``-name``, ``-format``, ``-invocation``) is a flat map of
          strings and equals the lock entry.
N7        the user-only skill is listed but never offered to the model, and a
          model call for it is refused (Claude Code); elsewhere ledgered.
N8        Claude Code only: ``permission_mode="default"`` runs the target's script
          under its scoped ``allowed-tools`` rule, denies a command outside it, and
          denies a foreign skill's own script; and, read from every installed
          ``SKILL.md`` (the runtime half loads only the target), ``allowed-tools`` is
          one space-separated string with no bare Bash (``Bash``, ``Bash(*)`` or
          ``Bash(*:*)``, all unrestricted to Claude Code), no comma outside a scope
          (import damage: Claude Code reads ``Bash,`` as ``Bash``), and a skill the
          org did not author keeps only Read, Grep and Glob.
N9        Claude Code only: the plugin root is ``decimalai-skills``, version
          ``lock-<first 12 hex of the recomputed lock_hash>``, loaded from the
          project's ``.agents`` (the path the CLI reports), with no plugin errors;
          and again with every skill when the documented snippet runs from a
          subdirectory of the project (a relative ``--plugin-dir`` loads nothing
          there, with no visible error).
N10       the target's script is executable in the copy the loader runs.
========  ==================================================================
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import ledger, oracle

ITEMS: Tuple[str, ...] = ("N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10")
PLUGIN_NAME = oracle.PLUGIN_NAME
FOREIGN_TOOLS = frozenset({"Read", "Grep", "Glob"})
SCENARIO = oracle.SCENARIO


class ContractError(AssertionError):
    """A cell failed. Subclasses AssertionError, so pytest reports it as a failure."""


@dataclass(frozen=True)
class Expectation:
    """What a correct folder promises: the lock v2 entries, keyed by install name."""

    project: Path
    root: str
    skills: Dict[str, dict]
    lock_hash: Optional[str]

    @property
    def names(self) -> List[str]:
        return sorted(self.skills)

    def model_invocable(self) -> List[str]:
        return sorted(n for n, e in self.skills.items() if e.get("invocation") != "user")

    def as_child_expect(self) -> Dict[str, dict]:
        return self.skills


def expectation_from_lock(project: Path) -> Expectation:
    """The expectation a v2 ``.decimal/skills.lock`` declares (anything else is an error)."""
    project = Path(project).resolve()
    lock = json.loads((project / ".decimal" / "skills.lock").read_text())
    if lock.get("version") != 2:
        raise ContractError(f"{project}/.decimal/skills.lock is not a v2 lock "
                            f"(version={lock.get('version')!r}); nothing to grade against")
    return expectation_for(project, lock)


def expectation_for(project: Path, lock: dict) -> Expectation:
    return Expectation(project=Path(project).resolve(), root=lock["root"],
                       skills=dict(lock["skills"]), lock_hash=lock.get("lock_hash"))


@dataclass
class Verdict:
    ok: bool
    detail: str = ""
    notes: List[str] = field(default_factory=list)


def _fail(detail: str) -> Verdict:
    return Verdict(False, detail)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _target(ex: Expectation) -> Optional[dict]:
    return ex.skills.get(SCENARIO.target)


def _errors(obs: dict, *stages: str) -> str:
    errs = obs.get("errors") or {}
    found = [f"{s}: {errs[s]}" for s in stages if s in errs]
    return ("; the loader raised during " + "; ".join(found)) if found else ""


# ── the items ────────────────────────────────────────────────────────────────


def n1(obs: dict, ex: Expectation) -> Verdict:
    if (obs.get("errors") or {}).get("discover"):
        return _fail(f"the loader RAISED during discovery: {obs['errors']['discover']}")
    listed = obs.get("listed")
    if listed is None:
        return _fail("the loader's discovery listed nothing observable" + _errors(obs, "run"))
    want, got = set(ex.names), set(listed)
    if want != got:
        return _fail(f"the loader lists {sorted(got)}; the lock installs {sorted(want)} "
                     f"(missing {sorted(want - got)}, not in the lock {sorted(got - want)})")
    offered = obs.get("offered")
    if offered is not None:
        missing = sorted(set(ex.model_invocable()) - set(offered))
        stray = sorted(set(offered) - want)
        if missing or stray:
            return _fail(f"the model's skill listing is {sorted(offered)}: it lacks the "
                         f"model-invocable {missing} and names {stray}, which the lock does not")
    return Verdict(True)


def n2(obs: dict, ex: Expectation) -> Verdict:
    loud = [line for line in obs.get("channel") or [] if not ledger.allowlisted(obs["loader"], line)]
    if loud:
        return _fail("the loader's own warning channel is not quiet:\n  " + "\n  ".join(loud[:12]))
    return Verdict(True)


def n3(obs: dict, ex: Expectation) -> Verdict:
    entry = _target(ex)
    if entry is None:
        return _fail(f"the lock has no {SCENARIO.target!r} to grade")
    sentinel = oracle.sentinel_of(SCENARIO.target)
    text = obs.get("body_text")
    if not text or sentinel not in text:
        return _fail(f"the model was not handed {SCENARIO.target}'s body (sentinel {sentinel!r} "
                     f"absent from {str(text)[:300]!r})" + _errors(obs, "run"))
    body = obs.get("body")
    if body is None:
        return _fail("the loader exposed no body to hash")
    if body.strip() not in text:
        return _fail(f"the model was not handed the body verbatim: the loader parsed "
                     f"{len(body.strip())} characters, and the delivered text "
                     f"({len(text)} characters) does not contain them; it ends "
                     f"{text[-200:]!r}")
    if _sha(body.strip()) != entry["body_sha256"]:
        return _fail(f"sha256(body.strip()) = {_sha(body.strip())}, the lock pins "
                     f"{entry['body_sha256']}")
    return Verdict(True)


def _lines(text: str) -> List[str]:
    return [ln.rstrip() for ln in text.rstrip("\n").split("\n")]


def n4(obs: dict, ex: Expectation) -> Verdict:
    entry = _target(ex)
    if entry is None:
        return _fail(f"the lock has no {SCENARIO.target!r} to grade")
    pinned = (entry.get("files") or {}).get(SCENARIO.resource)
    on_disk = ex.project / entry["path"] / SCENARIO.resource
    try:
        disk = on_disk.read_bytes()
    except OSError as exc:
        return _fail(f"the resource is missing from the folder: {exc}")
    if hashlib.sha256(disk).hexdigest() != pinned:
        return _fail(f"{on_disk} does not hash to the lock's files[{SCENARIO.resource!r}]")
    got = obs.get("resource_text")
    if got is None:
        return _fail("the resource was not readable through the loader's own route"
                     + _errors(obs, "run"))
    if obs.get("resource_exact"):
        if _sha(got) != pinned:
            return _fail(f"the loader returned {got[:200]!r}, not the pinned bytes")
    elif _lines(got) != _lines(disk.decode("utf-8")):
        return _fail(f"the loader's viewer returned {got[:200]!r}, not the file's lines")
    return Verdict(True)


def n5(obs: dict, ex: Expectation) -> Verdict:
    out = obs.get("script_stdout")
    if not out or SCENARIO.script_out not in out:
        return _fail(f"the script did not print {SCENARIO.script_out!r} through the loader's "
                     f"route; got {str(out)[:300]!r}" + _errors(obs, "run"))
    return Verdict(True)


_STAMP_KEYS = ("decimalai-skill-id", "decimalai-version-id", "decimalai-version",
               "decimalai-body-sha256", "decimalai-name", "decimalai-format",
               "decimalai-invocation")


def n6(obs: dict, ex: Expectation) -> Verdict:
    entry = _target(ex)
    if entry is None:
        return _fail(f"the lock has no {SCENARIO.target!r} to grade")
    stamp = (obs.get("stamp") or {}).get("metadata")
    if not isinstance(stamp, dict):
        return _fail(f"no stamp was readable (metadata = {stamp!r})" + _errors(obs, "discover", "run"))
    bad_types = sorted(k for k, v in stamp.items() if not isinstance(v, str))
    if bad_types:
        return _fail(f"metadata is not a flat map of strings: {bad_types}")
    want = {
        "decimalai-skill-id": entry.get("skill_id"),
        "decimalai-version-id": entry.get("version_id"),
        "decimalai-version": None if entry.get("version") is None else str(entry["version"]),
        "decimalai-body-sha256": entry.get("body_sha256"),
        "decimalai-name": entry.get("name"),
        "decimalai-format": str((entry.get("render") or {}).get("format")),
    }
    wrong = {k: (stamp.get(k), v) for k, v in want.items() if stamp.get(k) != v}
    invocation = stamp.get("decimalai-invocation")
    allowed = ("user",) if entry.get("invocation") == "user" else ("model", "any")
    if invocation not in allowed:
        wrong["decimalai-invocation"] = (invocation, "/".join(allowed))
    if wrong:
        return _fail("the stamp disagrees with the lock (got, want): "
                     + ", ".join(f"{k}={g!r}/{w!r}" for k, (g, w) in sorted(wrong.items())))
    return Verdict(True)


def n7(obs: dict, ex: Expectation) -> Verdict:
    name = SCENARIO.user_only
    if name not in ex.skills or ex.skills[name].get("invocation") != "user":
        return _fail(f"the lock has no user-only {name!r} to grade")
    if name not in (obs.get("listed") or []):
        return _fail(f"{name} is not listed at all, so 'hidden from the model' is not observable")
    offered = (obs.get("user_only") or {}).get("offered")
    if offered is not False:
        return _fail(f"the user-only {name} is offered to the model (offered={offered!r})")
    if obs["loader"] == "claude-agent-sdk":
        refusal = (obs.get("user_only") or {}).get("refusal") or ""
        if "disable-model-invocation" not in refusal:
            return _fail(f"a model Skill call for {name} was not refused: {refusal[:300]!r}")
    return Verdict(True)


#: ``Bash`` with no real scope. The same rule as ``skillevaluation.skillfmt``'s
#: ``is_bare_bash`` (``test_contract.py`` keeps the two equal where both exist).
_BARE_BASH = re.compile(r"Bash(?:\(\s*\*\s*(?::\s*\*\s*)?\))?")


def _is_sep(ch: str) -> bool:
    return ch.isspace() or ch == ","


def allowed_tools_tokens(value: str) -> List[str]:
    """The tool tokens of an ``allowed-tools`` string, as Claude Code reads them.

    The same split as ``skillevaluation.skillfmt.tokenize_allowed_tools`` (the
    contract grades the renderer, so it does not import it): whitespace and commas
    outside parentheses separate tokens, so ``"Read, Bash,"`` is ``Read`` and
    ``Bash``; a scope that never closes is cut at its first separator; duplicates
    drop, order stays.
    """
    tokens: List[str] = []
    i, n = 0, len(value)
    while i < n:
        while i < n and _is_sep(value[i]):
            i += 1
        if i >= n:
            break
        start, depth = i, 0
        while i < n:
            ch = value[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif depth == 0 and _is_sep(ch):
                break
            i += 1
        if depth > 0:
            i = start
            while i < n and not _is_sep(value[i]):
                i += 1
        tokens.append(value[start:i])
    return list(dict.fromkeys(tokens))


def is_bare_bash(token: str) -> bool:
    """``Bash``, ``Bash(*)`` and ``Bash(*:*)``: each lets Claude Code run any command."""
    return bool(_BARE_BASH.fullmatch(token))


def comma_damaged(value: str) -> List[str]:
    """The space-separated words of ``value`` holding a comma outside parentheses.

    Format 1 writes ``allowed-tools`` as one space-separated string, so a comma there
    is an importer's comma-separated upstream list leaking through the renderer.
    """
    out: List[str] = []
    depth, word, hit = 0, "", False
    for ch in value + " ":
        if ch.isspace() and depth == 0:
            if hit:
                out.append(word)
            word, hit = "", False
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            hit = True
        word += ch
    return out


def _allowed_tools(skill_md: Path) -> Optional[Any]:
    import yaml

    text = skill_md.read_text(encoding="utf-8")
    end = text.find("\n---\n", 3)
    return (yaml.safe_load(text[4:end + 1]) or {}).get("allowed-tools")


def _allowed_tools_problems(name: str, entry: dict, raw: Optional[Any]) -> List[str]:
    if raw is None:
        return []
    problems = []
    if not isinstance(raw, str):
        problems.append(f"{name}: allowed-tools is a {type(raw).__name__}, not a string")
        raw = " ".join(str(t) for t in raw) if isinstance(raw, list) else str(raw)
    damaged = comma_damaged(raw)
    if damaged:
        problems.append(f"{name}: allowed-tools {raw!r} separates tools with a comma "
                        f"({damaged}); format 1 is one space-separated string, so a comma "
                        "is import damage the renderer let through")
    tokens = allowed_tools_tokens(raw)
    bare = [t for t in tokens if is_bare_bash(t)]
    if bare:
        problems.append(f"{name}: allowed-tools grants a bare Bash ({bare}), which Claude "
                        "Code runs unrestricted")
    if (entry.get("render") or {}).get("variant") == "foreign" and set(tokens) - FOREIGN_TOOLS:
        problems.append(f"{name} (foreign) grants {sorted(set(tokens) - FOREIGN_TOOLS)}")
    return problems


def n8(obs: dict, ex: Expectation) -> Verdict:
    c = obs.get("claude") or {}
    denied = c.get("commands_denied") or []
    problems = []
    if not c.get("scoped_command") or c["scoped_command"] in denied:
        problems.append(f"the target's own script was not pre-approved: {c.get('scoped_command')!r}")
    if SCENARIO.script_out not in (obs.get("script_stdout") or ""):
        problems.append(f"the scoped script did not run: {obs.get('script_stdout')!r}")
    if not c.get("outside_command") or c["outside_command"] not in denied:
        problems.append(f"a command outside the rule was not denied: {c.get('outside_command')!r}")
    foreign = c.get("foreign")
    if foreign:
        if not c.get("foreign_loaded"):
            problems.append(f"the foreign skill {foreign} did not load")
        if not c.get("foreign_command") or c["foreign_command"] not in denied:
            problems.append(f"the foreign skill's own script was not denied: {c.get('foreign_command')!r}")
    root = ex.project / ex.root
    for name, entry in sorted(ex.skills.items()):
        md = root / name / "SKILL.md"
        if not md.is_file():
            problems.append(f"{md} is missing")
            continue
        problems += _allowed_tools_problems(name, entry, _allowed_tools(md))
    return _fail("; ".join(problems)) if problems else Verdict(True)


def n9(obs: dict, ex: Expectation) -> Verdict:
    c = obs.get("claude") or {}
    plugin = c.get("plugin")
    if not plugin:
        return _fail(f"no plugin was loaded from {c.get('plugin_root')!r}; init.plugins = "
                     f"{c.get('plugins')!r}" + _errors(obs, "run"))
    recomputed = oracle.lock_hash(ex.root, ex.skills)
    problems = []
    if plugin.get("name") != PLUGIN_NAME:
        problems.append(f"the plugin is named {plugin.get('name')!r}, not {PLUGIN_NAME!r}")
    if ex.lock_hash != recomputed:
        problems.append(f"the lock's lock_hash {ex.lock_hash!r} does not recompute ({recomputed})")
    if plugin.get("version") != f"lock-{recomputed[:12]}":
        problems.append(f"the plugin version is {plugin.get('version')!r}, not lock-{recomputed[:12]}")
    if c.get("plugin_errors"):
        problems.append(f"plugin_errors: {c['plugin_errors']!r}")
    problems += _plugin_problems(plugin, c.get("skills"), ex)
    problems += _subdirectory_problems(c.get("from_subdirectory"), ex, _errors(obs, "subdirectory"))
    return _fail("; ".join(problems)) if problems else Verdict(True)


def _plugin_problems(plugin: dict, skills: Optional[List[str]], ex: Expectation) -> List[str]:
    """Loaded from the project's ``.agents`` (as the CLI reports it), with every skill."""
    problems = []
    if os.path.realpath(plugin.get("path") or "") != os.path.realpath(ex.project / ".agents"):
        problems.append(f"the plugin was loaded from {plugin.get('path')!r}, not the project's .agents")
    ours = [s for s in skills or [] if s.split(":", 1)[0] == plugin.get("name")]
    if sorted(s.split(":", 1)[1] for s in ours) != ex.names:
        problems.append(f"the plugin's skills are {ours}, not {PLUGIN_NAME}:<each install name>")
    return problems


def _subdirectory_problems(sub: Optional[dict], ex: Expectation, raised: str) -> List[str]:
    """The documented snippet, run from a subdirectory of the project, loads the same plugin.

    ``skills_dir()`` walks up from the working directory, and the CLI gets the
    absolute path. A relative ``--plugin-dir .agents`` from a subdirectory loads
    nothing: exit 0, nothing on stderr, one ``path-not-found`` in ``plugin_errors``.
    """
    if not sub:
        return ["the snippet was not run from a subdirectory of the project" + raised]
    cwd = Path(os.path.realpath(sub.get("cwd") or "/"))
    if Path(os.path.realpath(ex.project)) not in cwd.parents:
        return [f"the subdirectory run's cwd {sub.get('cwd')!r} is not a subdirectory of {ex.project}"]
    plugin = sub.get("plugin")
    where = f"run from the subdirectory {sub.get('cwd')}"
    if not plugin:
        return [f"{where}, no plugin loaded from {sub.get('plugin_root')!r}; init.plugins = "
                f"{sub.get('plugins')!r}, plugin_errors = {sub.get('plugin_errors')!r}"]
    problems = [f"{where}, {p}" for p in _plugin_problems(plugin, sub.get("skills"), ex)]
    if plugin.get("name") != PLUGIN_NAME:
        problems.append(f"{where}, the plugin is named {plugin.get('name')!r}")
    if sub.get("plugin_errors"):
        problems.append(f"{where}, plugin_errors: {sub['plugin_errors']!r}")
    return problems


def n10(obs: dict, ex: Expectation) -> Verdict:
    entry = _target(ex)
    if entry is None:
        return _fail(f"the lock has no {SCENARIO.target!r} to grade")
    if SCENARIO.script not in (entry.get("executable") or []):
        return _fail(f"the lock does not mark {SCENARIO.script} executable")
    bit = obs.get("exec_bit") or {}
    if bit.get("copy") is not True:
        return _fail(f"{bit.get('path')!r} is not executable in the copy the loader uses "
                     f"(source {bit.get('source')!r}, copy {bit.get('copy')!r})")
    return Verdict(True)


GRADERS = {"N1": n1, "N2": n2, "N3": n3, "N4": n4, "N5": n5, "N6": n6, "N7": n7,
           "N8": n8, "N9": n9, "N10": n10}


def verdict(item: str, obs: dict, ex: Expectation) -> Verdict:
    """The strict verdict, before any ledger entry is applied."""
    v = GRADERS[item](obs, ex)
    proof = ledger.PROOFS.get(f"{obs['loader']}:{item}")
    if v.ok and proof is not None:
        held, detail = proof.check(obs)
        if not held:
            return _fail(f"the fact this cell is graded on no longer holds ({proof.reason}): {detail}")
    return v


@dataclass(frozen=True)
class Cell:
    kind: str     # "pass" | "na" | "gap"
    reason: str = ""


def grade(loader: str, item: str, obs: dict, ex: Expectation) -> Cell:
    """Grade one cell with the ledger applied. Raises ``ContractError`` on a failure."""
    key = f"{loader}:{item}"
    if obs.get("loader") != loader:
        raise ContractError(f"{key}: the observation is {obs.get('loader')!r}'s")
    na = ledger.DECLARED_NA.get(key)
    if na is not None:
        held, detail = na.check(obs)
        if not held:
            raise ContractError(f"{key} is declared N/A ({na.reason}), but its proof no longer "
                                f"holds: {detail}. The framework may have shipped the capability: "
                                f"grade the item and delete the ledger line.")
        return Cell("na", na.reason)
    v = verdict(item, obs, ex)
    gap = ledger.UPSTREAM_GAPS.get(key)
    if gap is not None:
        if v.ok:
            raise ContractError(f"{key} PASSES now: the upstream gap ({gap.reason}) is closed. "
                                f"Delete its line in ledger.UPSTREAM_GAPS (it cleans itself).")
        held, detail = gap.check(obs)
        if not held:
            raise ContractError(f"{key} fails, and not in the documented way ({gap.reason}; "
                                f"{detail}): {v.detail}")
        return Cell("gap", f"{gap.reason}: {v.detail}")
    if not v.ok:
        raise ContractError(f"{key}: {v.detail}")
    return Cell("pass")


def failing_items(loader: str, items, obs: dict, ex: Expectation) -> Dict[str, str]:
    """Every item that fails for ``loader`` (with the ledger applied), with why."""
    out = {}
    for item in items:
        try:
            grade(loader, item, obs, ex)
        except ContractError as exc:
            out[item] = str(exc)
    return out


def assert_one_skill_md_per_folder(ex: Expectation) -> None:
    """Never a second SKILL.md anywhere in a managed folder (any case), so no loader
    can list an attachment as one more skill; and every lock entry has its folder."""
    root = ex.project / ex.root
    problems = []
    for name in ex.names:
        folder = root / name
        found = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*")
                       if p.name.lower() == "skill.md")
        if found != ["SKILL.md"]:
            problems.append(f"{name}: {found}")
    if problems:
        raise ContractError("each skill folder must hold exactly one SKILL.md, at its top: "
                            + "; ".join(problems))
