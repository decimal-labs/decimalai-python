"""``decimalai skills verify``: what a native skill loader would reject, said before one does.

Two ways to call it::

    from decimalai import skillpkg

    skillpkg.verify()                     # the project: .agents/skills against .decimal/skills.lock
    skillpkg.verify("skills/")            # any tree, by the rules for unmanaged folders
    skillpkg.verify("skills/", strict=True)

**The rules** (:data:`RULES`). The format rules ``F1``-``F15`` are
:mod:`skillevaluation.skillfmt`'s own, and this module only calls them. The folder and
project rules ``V1``-``V9`` are below (:data:`FOLDER_RULES`). A rule's scope is
``loader`` when at least one native loader rejects, drops or renames a skill that breaks
it, and ``sdk`` when it is this format's own rule.

**Who gets which rules.**

* A folder the lock lists (managed): every rule is an error, whatever its scope.
* A folder the lock does not list (a hand-authored skill sharing the root), and every
  folder ``verify PATH`` checks: ``loader`` rules are errors, ``sdk`` rules warnings, and
  with ``strict`` errors too. The rules that compare a folder with its lock entry
  (``F11``-``F13``, ``V3``, ``V4``) do not apply. The files an author keeps beside
  SKILL.md (``reference.md``, ``LICENSE.txt``, ``eval.yaml``) are the author's business,
  and so is where the other files sit: ``F14`` reports only an unsafe path there (a
  backslash or a control character), ``F15`` a second SKILL.md below the first.
* The project rules (``plugin.json``, the lock, the manifest, an interrupted run) are
  errors.

**A project without a lock.** When ``.agents/skills`` exists but ``.decimal/skills.lock``
does not, the root is graded as one of unmanaged folders and the lock, manifest and
``plugin.json`` are not checked; one notice (:data:`NO_LOCK`, severity :data:`NOTICE`,
never an error) says so. With neither a lock nor a root there is nothing to verify.

``verify PATH`` takes a skill folder, its ``SKILL.md``, or a folder of skill folders
(``<PATH>/<name>/SKILL.md``, the shape every loader reads); hidden folders and OS clutter
are skipped. It never reads a lock.

**The result** is a :class:`~decimalai.skillpkg.errors.Result` whose ``problems`` are
:class:`Finding` s (warnings included), sorted: project findings first, then folder by
folder. Its ``exit_code`` is 0 when no finding is an error, 1 when one is. A lock that
cannot be read at all is a refusal, never a pass: a corrupt one, a version 1 or npx lock
(``decimalai skills migrate``), or no lock and no root give the error alone, with exit
code 3. An unusable ``PATH`` gives exit code 2.

verify only reads: it takes no process lock, settles no journal and writes nothing, so
it is safe beside another ``decimalai skills`` run, and in a read-only checkout.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

from skillevaluation import skillfmt

from . import fs as _fs
from . import lock as _lock
from . import manifest as _manifest
from .errors import (
    ExitCode,
    ForeignLockEntriesError,
    IllegalEntryError,
    JournalConflictError,
    LockHashMismatch,
    LockOutOfDateError,
    ManifestError,
    Result,
    SkillsLockError,
    SkillsNotInstalledError,
    UsageError,
)
from .project import PLUGIN_NAME, ROOT, Project

__all__ = [
    "ERROR",
    "FOLDER_RULES",
    "NOTICE",
    "NO_LOCK",
    "RULES",
    "WARNING",
    "Finding",
    "verify",
    "verify_folder",
]

PathLike = Union[str, "os.PathLike[str]"]

ERROR = "error"
WARNING = "warning"
#: The severity of a notice: information, never an error, whatever ``strict`` says.
NOTICE = "notice"

#: The notice of a project with skills in its root and no lock.
NO_LOCK = "no-lock"

#: The rules of this module: rule id -> (scope, what it checks).
FOLDER_RULES: Mapping[str, Tuple[str, str]] = {
    "V1": ("loader", "no symlink in the skills tree, '.agents/' or '.decimal/': openai-agents "
                     "skips a symlinked skill folder without a word, and agent-framework drops "
                     "a symlinked resource"),
    "V2": ("loader", "each skill is a real folder directly in the root that holds its SKILL.md "
                     "(<root>/<name>/SKILL.md): agent-framework and Claude Code look no deeper. "
                     "The managed root holds nothing else, a folder holds only regular files "
                     "and folders, and no two names differ only by case"),
    "V3": ("sdk", "managed: the folder holds exactly the files the lock lists, with the lock's "
                  "sha256 (a file the lock lists as unavailable may be present or absent)"),
    "V4": ("sdk", "managed: each file's exec bit is what the lock's 'executable' says (POSIX)"),
    "V5": ("loader", "'.agents/.claude-plugin/plugin.json' names the plugin 'decimalai-skills' "
                     "at version 'lock-<first 12 hex of lock_hash>': without it Claude Code "
                     "names every skill after the '.agents' folder"),
    "V6": ("sdk", "the lock is whole: lock_hash recomputes, and every entry was written by "
                  "this engine"),
    "V7": ("sdk", "'.decimal/skills.toml' agrees with the lock: every skill it lists is "
                  "installed at its version, and every installed skill is listed"),
    "V8": ("sdk", "a folder whose SKILL.md carries metadata.decimalai-skill-id is in the lock; "
                  "otherwise it is an orphan of an older install, or a copy"),
    "V9": ("sdk", "no interrupted run: '.decimal/skills.lock.journal' is absent"),
}

#: Every rule verify reports: skillfmt's format rules, then this module's.
RULES: Mapping[str, Tuple[str, str]] = {**skillfmt.RULES, **FOLDER_RULES}

_ORDER = {rule: i for i, rule in enumerate(RULES)}
_PROJECT_RULES = frozenset({"V5", "V6", "V7", "V9"})
_HEX64 = re.compile(r"[0-9a-f]{64}")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

_LOCK = ".decimal/skills.lock"
_MANIFEST = ".decimal/skills.toml"
_PLUGIN = ".agents/.claude-plugin/plugin.json"
_JOURNAL = ".decimal/skills.lock.journal"

_CLUTTER_DIRS = frozenset(c[:-1] for c in _fs.CLUTTER if c.endswith("/"))
_CLUTTER_FILES = tuple(c for c in _fs.CLUTTER if not c.endswith("/"))
#: How far below a folder without a SKILL.md verify looks for the skill that was meant.
_NESTED_DEPTH = 3

_RESTORE = ("restore the locked files with `decimalai skills install --frozen --force`, or copy "
            "the folder to a new name to keep your changes as a skill of your own")
_FETCH = "run `decimalai skills install` to fetch it"
_UNLINK = ("replace the link with the real file or folder; `decimalai skills` never follows "
           "a symlink inside the project")
_PLUGIN_FIX = "run `decimalai skills install` to rewrite it"
_JOURNAL_FIX = ("run `decimalai skills install` to finish or roll back the interrupted change; "
                "`decimalai skills status` shows it")
_NO_MANIFEST_FIX = ("restore `.decimal/skills.toml` from version control: it lists the skills "
                    "the lock installs")
_NO_LOCK_FIX = ("run `decimalai skills install`, or `decimalai skills add <ref>` for a first "
                "skill; to check skill folders without a lock, run "
                "`decimalai skills verify <path>`")
_LOCKLESS_FIX = ("run `decimalai skills add <ref>` to install a registry skill as a managed "
                 "folder, or `decimalai skills install` when `.decimal/skills.toml` lists some")
_PATH_FIX = "pass a skill folder, its SKILL.md, or a folder of skill folders"
_READABLE_FIX = ("make it readable (`chmod u+r`; `sudo chown` a file a container wrote as "
                 "root), then run `decimalai skills verify` again")


# ── findings ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Finding:
    """One rule a folder or the project breaks.

    Attributes:
        rule: the rule id (:data:`RULES`), or for a notice its id (:data:`NO_LOCK`).
        scope: ``loader`` or ``sdk``.
        severity: ``error``, ``warning``, or ``notice`` (information, never an error).
        message: what is wrong, naming the file or the value.
        skill: the folder's name; for a project finding about one skill, its registry
            name (the manifest and the lock) or its folder name (the journal).
        path: the file or folder: relative to the project, or under the ``PATH`` given.
        fix: the command or edit that clears it, when there is one.
        managed: whether the lock lists the folder (project findings count as managed).
    """

    rule: str
    scope: str
    severity: str
    message: str
    skill: Optional[str] = None
    path: Optional[str] = None
    fix: Optional[str] = None
    managed: bool = False

    @property
    def is_error(self) -> bool:
        return self.severity == ERROR

    def __str__(self) -> str:
        where = self.path or self.skill
        return f"{self.severity} {self.rule}{' ' + where if where else ''}: {self.message}"

    def to_dict(self) -> dict:
        return {"rule": self.rule, "scope": self.scope, "severity": self.severity,
                "message": self.message, "skill": self.skill, "path": self.path,
                "fix": self.fix, "managed": self.managed}


def _sort_key(f: Finding) -> tuple:
    folder = f.rule not in _PROJECT_RULES and f.skill is not None
    return (folder, f.skill or "", f.path or "", _ORDER.get(f.rule, -1), f.message)


@dataclass
class _Report:
    strict: bool
    items: List[Finding] = field(default_factory=list)
    _seen: Set[tuple] = field(default_factory=set)

    def add(self, rule: str, message: str, *, managed: bool, skill: Optional[str] = None,
            path: Optional[str] = None, fix: Optional[str] = None) -> None:
        scope = RULES[rule][0]
        problem = skillfmt.Problem(rule, scope, message)
        severity = ERROR if skillfmt.is_error(problem, managed=managed,
                                              strict=self.strict) else WARNING
        key = (rule, skill, path, message)
        if key not in self._seen:
            self._seen.add(key)
            self.items.append(Finding(rule, scope, severity, message, skill, path, fix, managed))

    def notice(self, notice: str, message: str, *, path: Optional[str] = None,
               fix: Optional[str] = None) -> None:
        self.items.append(Finding(notice, "sdk", NOTICE, message, None, path, fix, False))

    def problem(self, p: skillfmt.Problem, *, managed: bool, skill: str, path: str) -> None:
        self.add(p.rule, p.message, managed=managed, skill=skill, path=path)

    @property
    def has_errors(self) -> bool:
        return any(f.is_error for f in self.items)

    def result(self) -> Result:
        items = sorted(self.items, key=_sort_key)
        return Result((), items, ExitCode.PROBLEMS if self.has_errors else ExitCode.OK)


# ── walking a folder ─────────────────────────────────────────────────────────


def _is_clutter(name: str, mode: int) -> bool:
    """Whether a directory entry is OS or tool clutter, ignored as the engine ignores it."""
    if stat.S_ISDIR(mode):
        return name in _CLUTTER_DIRS
    if stat.S_ISREG(mode):
        return any(fnmatch.fnmatchcase(name, pattern) for pattern in _CLUTTER_FILES)
    return False


@dataclass
class _Tree:
    files: Set[str] = field(default_factory=set)              # regular files, relative paths
    links: List[str] = field(default_factory=list)            # symlinks, never followed
    odd: List[Tuple[str, str]] = field(default_factory=list)  # (path, why) of anything else

    @property
    def sound(self) -> bool:
        return not self.links and not self.odd


def _scan(directory: Path) -> List[Tuple[str, int, str]]:
    """``(name, lstat mode, path)`` of each entry of ``directory``, clutter left out, sorted."""
    with os.scandir(directory) as it:
        entries = [(item.name, item.stat(follow_symlinks=False).st_mode, item.path)
                   for item in it]
    return sorted(e for e in entries if not _is_clutter(e[0], e[1]))


def _walk(folder: Path) -> _Tree:
    """Every entry under ``folder``: regular files, symlinks and anything else, apart."""
    tree = _Tree()
    pending: List[Tuple[Path, str]] = [(folder, "")]
    while pending:
        directory, prefix = pending.pop()
        try:
            entries = _scan(directory)
        except OSError as exc:
            tree.odd.append((prefix.rstrip("/") or ".", f"cannot be read ({exc.strerror})"))
            continue
        for name, mode, path in entries:
            rel = prefix + name
            if stat.S_ISLNK(mode):
                tree.links.append(rel)
            elif stat.S_ISDIR(mode):
                pending.append((Path(path), rel + "/"))
            elif stat.S_ISREG(mode):
                tree.files.add(rel)
            else:
                tree.odd.append((rel, "is not a regular file or folder"))
    tree.links.sort()
    tree.odd.sort()
    return tree


def _nested_skill_md(folder: Path) -> Optional[str]:
    """The shallowest ``SKILL.md`` below ``folder``'s own level, if one is near: the skill
    someone put a level (or more) too deep."""
    level: List[Tuple[Path, str]] = [(folder, "")]
    for _ in range(_NESTED_DEPTH):
        below: List[Tuple[Path, str]] = []
        for directory, prefix in level:
            try:
                entries = _scan(directory)
            except OSError:
                continue
            for name, mode, path in entries:
                if stat.S_ISDIR(mode) and not name.startswith("."):
                    below.append((Path(path), f"{prefix}{name}/"))
        for directory, prefix in below:
            try:
                if any(n == "SKILL.md" and stat.S_ISREG(m) for n, m, _ in _scan(directory)):
                    return f"{prefix}SKILL.md"
            except OSError:
                continue
        level = below
    return None


def _entry_names(directory: Path) -> Set[str]:
    try:
        return {name for name, _, _ in _scan(directory)}
    except OSError:
        return set()


# ── one skill folder ─────────────────────────────────────────────────────────


def _grade_folder(report: _Report, folder: Path, *, name: str, where: str,
                  entry: Optional[Mapping[str, Any]], required: bool) -> None:
    """Report every rule ``folder`` breaks. ``entry`` is its lock entry when it is managed;
    ``where`` is how findings name it. ``required`` means the folder must be a skill (it
    sits in the managed root, or was named as one), so a missing SKILL.md is reported even
    when no skill is nested below it."""
    managed = entry is not None
    tree = _walk(folder)
    # Locked files this user cannot open: reported once each, and never hashed.
    hidden = _unreadable_locked(folder, entry, tree) if managed else []
    for rel in hidden:
        _unreadable(report, name=name, where=where, rel=rel)
    for rel in tree.links:
        report.add("V1", f"`{rel}` is a symlink; agent-framework drops a symlinked resource, "
                         "and the engine never follows one",
                   managed=managed, skill=name, path=f"{where}/{rel}", fix=_UNLINK)
    for rel, why in tree.odd:
        report.add("V2", f"`{rel}` {why}", managed=managed, skill=name, path=f"{where}/{rel}",
                   fix="remove it, or replace it with a regular file")

    has_skill_md = "SKILL.md" in tree.files
    if has_skill_md:
        md = f"{where}/SKILL.md"
        try:
            data = None if "SKILL.md" in hidden else (folder / "SKILL.md").read_bytes()
        except OSError as exc:
            if managed:
                _unreadable(report, name=name, where=where, rel="SKILL.md",
                            why=exc.strerror)
            else:
                report.add("F1", f"SKILL.md cannot be read ({exc.strerror})",
                           managed=False, skill=name, path=md, fix=_READABLE_FIX)
        else:
            for p in (() if data is None else skillfmt.validate_skill_md(
                    data, folder_name=name, managed=managed, expect=entry)):
                report.problem(p, managed=managed, skill=name, path=md)
        for rel in sorted(tree.files):
            if rel == "SKILL.md" or (not managed and "/" not in rel):
                continue  # beside SKILL.md, an unmanaged folder holds what its author likes
            for p in skillfmt.validate_attachment_path(rel):
                if managed or p.rule != "F14" or _unsafe(rel):
                    report.problem(p, managed=managed, skill=name, path=f"{where}/{rel}")
    elif not managed and "SKILL.md" not in tree.links:
        _no_skill_md(report, folder, name=name, where=where, required=required)

    # A link or a special file was reported above; the bytes behind it are never compared.
    if managed and tree.sound:
        try:
            state = _fs.check_folder(folder, _without(entry, hidden), name=name)
        except OSError as exc:  # it became unreadable after the probe
            _unreadable(report, name=name, where=where, why=exc.strerror,
                        rel=_rel_to(folder, exc.filename))
            return
        for rel in state.edited:
            report.add("V3", f"`{rel}` differs from the lock", managed=True, skill=name,
                       path=f"{where}/{rel}", fix=_RESTORE)
        for rel in state.missing:
            report.add("V3", f"`{rel}` is missing; the lock lists it", managed=True, skill=name,
                       path=f"{where}/{rel}", fix=_RESTORE)
        for rel in state.extra:
            report.add("V3", f"`{rel}` is not in the lock", managed=True, skill=name,
                       path=f"{where}/{rel}", fix=_RESTORE)
        executable = set(entry.get("executable") or ())
        for rel in state.mode:
            want = ("should be executable: the lock lists it in `executable`" if rel in executable
                    else "should not be executable: the lock does not list it in `executable`")
            report.add("V4", f"`{rel}` {want}", managed=True, skill=name,
                       path=f"{where}/{rel}", fix=_RESTORE)


def _readable(path: Path) -> bool:
    try:
        with open(path, "rb"):
            return True
    except OSError:
        return False


def _unreadable_locked(folder: Path, entry: Mapping[str, Any], tree: _Tree) -> List[str]:
    """The files of ``entry`` present in ``folder`` that cannot be opened, sorted."""
    files = entry.get("files") or {}
    return sorted(rel for rel in files if rel in tree.files and not _readable(folder / rel))


def _without(entry: Mapping[str, Any], hidden: List[str]) -> Mapping[str, Any]:
    """``entry`` with the ``hidden`` files taken out of the comparison: listed as
    unavailable, which a folder may hold or lack, so they are neither hashed nor extra."""
    if not hidden:
        return entry
    out = dict(entry)
    out["files"] = {k: v for k, v in (entry.get("files") or {}).items() if k not in hidden}
    out["unavailable"] = list(entry.get("unavailable") or ()) + [
        {"path": rel, "reason": "unreadable"} for rel in hidden]
    return out


def _rel_to(folder: Path, filename: Any) -> str:
    try:
        return Path(os.fspath(filename)).relative_to(folder).as_posix()
    except (TypeError, ValueError):
        return "."


def _unreadable(report: _Report, *, name: str, where: str, rel: str,
                why: Optional[str] = None) -> None:
    report.add("V3", f"unreadable: `{rel}` ({why or 'permission denied'}); its bytes cannot "
                     "be compared with the lock",
               managed=True, skill=name, path=where if rel == "." else f"{where}/{rel}",
               fix=_READABLE_FIX)


def _unsafe(rel: str) -> bool:
    """Whether a path is unsafe to write or to read back, wherever it sits: the half of
    F14 that applies to every folder (the placement half is the managed format's)."""
    parts = rel.split("/")
    return ("\\" in rel or bool(_CONTROL.search(rel)) or rel.startswith("/")
            or any(part in ("", ".", "..") for part in parts))


def _no_skill_md(report: _Report, folder: Path, *, name: str, where: str,
                 required: bool) -> bool:
    """Report a folder with no SKILL.md of its own; whether there was anything to report.

    Unless the folder is ``required`` to be a skill, one with no skill anywhere near is
    just a folder (``docs/`` beside the skill folders of a repository)."""
    names = _entry_names(folder)
    misnamed = sorted(n for n in names if n.casefold() == "skill.md")
    nested = None if misnamed else _nested_skill_md(folder)
    if misnamed:
        message = (f"`{where}` has `{misnamed[0]}` but no SKILL.md; loaders look for the "
                   "exact name SKILL.md")
        fix = f"rename it to `{where}/SKILL.md`"
    elif nested:
        message = (f"`{where}` holds no SKILL.md; its skill is at `{where}/{nested}`, deeper "
                   "than loaders look (agent-framework and Claude Code read only "
                   "<root>/<name>/SKILL.md)")
        fix = ("move that skill's folder up beside the others" if required else
               "point the loader, and `decimalai skills verify`, at the folder that holds "
               "the skill folders")
    elif required:
        message = f"`{where}` holds no SKILL.md, so no loader lists it"
        fix = f"add a SKILL.md, or move the folder out of `{ROOT}/`"
    else:
        return False
    report.add("V2", message, managed=False, skill=name, path=where, fix=fix)
    return True


def verify_folder(folder: PathLike, *, entry: Optional[Mapping[str, Any]] = None,
                  strict: bool = False, name: Optional[str] = None) -> List[Finding]:
    """The findings for one skill folder, sorted.

    With ``entry`` (its lock entry) the folder is graded as managed: every rule, every one
    an error, its bytes and exec bits compared with the entry. Without it, by the rules
    for unmanaged folders. ``name`` is the folder name SKILL.md's ``name`` must equal
    (default: the folder's own name).
    """
    path = Path(folder)
    report = _Report(strict=bool(strict))
    if os.path.islink(path):
        report.add("V1", f"`{path.as_posix()}` is a symlink", managed=entry is not None,
                   skill=name or path.name, path=path.as_posix(), fix=_UNLINK)
    else:
        _grade_folder(report, path, name=name or Path(os.path.abspath(path)).name,
                      where=path.as_posix(), entry=entry, required=True)
    return sorted(report.items, key=_sort_key)


# ── the project ──────────────────────────────────────────────────────────────


def _is_link(path: Path) -> bool:
    return os.path.islink(path)


def _links_under(report: _Report, project: Project, top: Path, *,
                 skip: Tuple[str, ...] = ()) -> None:
    """V1 for every symlink under ``top`` (itself excluded), ``skip`` names left out."""
    pending = [top]
    while pending:
        directory = pending.pop()
        try:
            entries = _scan(directory)
        except OSError:
            continue
        for name, mode, path in entries:
            if directory == top and name in skip:
                continue
            if stat.S_ISLNK(mode):
                rel = project.rel(path)
                report.add("V1", f"`{rel}` is a symlink", managed=True, path=rel, fix=_UNLINK)
            elif stat.S_ISDIR(mode):
                pending.append(Path(path))


def _check_lock(report: _Report, lock: "_lock.Lock") -> None:
    if lock.dropped:
        names = ", ".join(lock.dropped)
        count = len(lock.dropped)
        report.add("V6", f"`{_LOCK}` holds {count} entr{'y' if count == 1 else 'ies'} an older "
                         f"decimalai wrote ({names}); their folders are unmanaged",
                   managed=True, path=_LOCK, fix=ForeignLockEntriesError.fix)
    if not lock.hash_ok:
        report.add("V6", f"lock_hash is {lock.stored_hash!r}, but the entries hash to "
                         f"{lock.lock_hash!r}: the lock was edited by hand",
                   managed=True, path=_LOCK, fix=LockHashMismatch.fix)


def _check_journal(report: _Report, project: Project) -> None:
    if not os.path.lexists(project.journal_path):
        return
    try:
        journal = _fs.read_journal(project)
    except JournalConflictError as exc:
        report.add("V9", f"a `decimalai skills` run was interrupted, and {exc.message}",
                   managed=True, path=_JOURNAL, fix=_JOURNAL_FIX)
        return
    except OSError as exc:
        report.add("V9", f"a `decimalai skills` run was interrupted, and `{_JOURNAL}` cannot "
                         f"be read ({exc.strerror})", managed=True, path=_JOURNAL,
                   fix=_READABLE_FIX)
        return
    if journal is not None:
        report.add("V9", f"a `decimalai skills` run was interrupted while it changed "
                         f"`{ROOT}/{journal.name}`",
                   managed=True, skill=journal.name, path=_JOURNAL, fix=_JOURNAL_FIX)


def _check_plugin(report: _Report, project: Project, lock: "_lock.Lock") -> None:
    path = project.plugin_json_path
    if _is_link(path.parent) or _is_link(path):
        return  # V1 says so; a link is never read
    stored = lock.stored_hash
    # A hand-edited lock (V6) is compared by the hash its file declares: plugin.json was
    # written with that one, so the edit is reported once.
    want = f"lock-{(stored if stored and _HEX64.fullmatch(stored) else lock.lock_hash)[:12]}"

    def add(message: str) -> None:
        report.add("V5", message, managed=True, path=_PLUGIN, fix=_PLUGIN_FIX)

    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        if lock.skills:
            add(f"`{_PLUGIN}` is missing: Claude Code names each skill `.agents:<name>`, not "
                f"`{PLUGIN_NAME}:<name>` (it should be version {want!r})")
        return
    except OSError as exc:
        add(f"`{_PLUGIN}` cannot be read ({exc.strerror})")
        return
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        add(f"`{_PLUGIN}` is not JSON; it should name {PLUGIN_NAME!r} at version {want!r}")
        return
    if not isinstance(data, dict):
        add(f"`{_PLUGIN}` is not a JSON object; it should name {PLUGIN_NAME!r} at version "
            f"{want!r}")
        return
    if data.get("name") != PLUGIN_NAME:
        add(f"`{_PLUGIN}` names the plugin {data.get('name')!r}, not {PLUGIN_NAME!r}; "
            "`decimalai` is the marketplace plugin's name, and two plugins with one name "
            "replace each other")
    if data.get("version") != want:
        add(f"`{_PLUGIN}` has version {data.get('version')!r}; the lock says {want!r}")


def _spec(entry: "_manifest.ManifestEntry") -> str:
    return "latest" if entry.latest else f"version {entry.spec}"


def _check_manifest(report: _Report, project: Project, lock: "_lock.Lock") -> None:
    def add(message: str, *, skill: Optional[str] = None,
            fix: str = LockOutOfDateError.fix) -> None:
        report.add("V7", message, managed=True, skill=skill, path=_MANIFEST, fix=fix)

    try:
        manifest = _manifest.load(project)
    except SkillsNotInstalledError:
        if lock.skills:
            add(f"`{_MANIFEST}` is missing, and the lock installs {len(lock.skills)} "
                f"skill{'' if len(lock.skills) == 1 else 's'}", fix=_NO_MANIFEST_FIX)
        return
    except IllegalEntryError:
        report.add("V1", f"`{_MANIFEST}` is a symlink", managed=True, path=_MANIFEST,
                   fix=_UNLINK)
        return
    except ManifestError as exc:
        add(exc.message, fix=exc.fix)
        return
    except OSError as exc:
        add(f"`{_MANIFEST}` cannot be read ({exc.strerror})", fix=_READABLE_FIX)
        return

    # Matched as `install --frozen` matches them, so the two agree: by skill id when the
    # manifest gives one, else by registry name; entries with an id first; each lock entry
    # answers one manifest entry at most.
    by_id = {entry["skill_id"]: key for key, entry in lock.skills.items()}
    claimed: Dict[str, str] = {}
    for want in sorted(manifest.entries.values(), key=lambda e: e.id is None):
        if want.id is not None:
            key = by_id.get(want.id)
        else:
            key = next((k for k, e in lock.skills.items()
                        if e.get("name") == want.name and k not in claimed), None)
        if key is None or key in claimed:
            add(f"`{_MANIFEST}` wants {want.name} ({_spec(want)}), which is not installed",
                skill=want.name)
            continue
        claimed[key] = want.name
        entry = lock.skills[key]
        if want.pin != entry["pin"] or (not want.latest and entry["version"] != want.version):
            held = (f"version {entry['version']}, exactly" if entry["pin"] == "exact"
                    else f"version {entry['version']} as latest")
            add(f"`{_MANIFEST}` wants {want.name} at {_spec(want)}, but the lock holds {held}",
                skill=want.name)
    for key, entry in lock.skills.items():
        if key not in claimed:
            add(f"the lock installs {entry['name']} at `{ROOT}/{key}`, but `{_MANIFEST}` does "
                "not list it", skill=entry["name"])


def _stamped_orphan(report: _Report, folder: Path, *, name: str, where: str,
                    skill_id: Optional[str]) -> None:
    registry_name = None
    try:
        fm, _ = skillfmt.parse_skill_md_text((folder / "SKILL.md").read_bytes().decode("utf-8"))
        meta = fm.get("metadata")
        if isinstance(meta, dict) and isinstance(meta.get("decimalai-name"), str):
            registry_name = meta["decimalai-name"]
    except (OSError, UnicodeDecodeError, skillfmt.SkillFmtError):
        pass
    add = (f"run `decimalai skills add {shlex.quote(registry_name)} --force` to manage it"
           if registry_name else "add it with `decimalai skills add <ref> --force` to manage it")
    report.add("V8", f"SKILL.md carries decimalai-skill-id {skill_id}, but the lock does not "
                     "list this folder: an orphan of an older install, or a copy",
               managed=False, skill=name, path=f"{where}/SKILL.md",
               fix=f"{add}, or delete its `decimalai-*` metadata to keep it as a skill of "
                   "your own")


def _check_root(report: _Report, project: Project, lock: "_lock.Lock", *,
                orphans: bool = True) -> None:
    """Grade every folder of the root against ``lock``; ``orphans`` reports stamped folders
    the lock does not list (V8), which says nothing when there is no lock at all."""
    root = project.skills_root
    if _is_link(root):
        report.add("V1", f"`{ROOT}` is a symlink", managed=True, path=ROOT, fix=_UNLINK)
        return
    if os.path.lexists(root) and not root.is_dir():
        report.add("V2", f"`{ROOT}` is not a folder", managed=True, path=ROOT,
                   fix=f"move it away; `{ROOT}` is the folder the skills are installed in")
        return
    # The scan hashes every locked file: one it cannot open would stop it, so those are
    # taken out of the comparison here, and reported per file by _grade_folder.
    skills = {}
    for key, entry in lock.skills.items():
        folder = root / key
        hidden = (_unreadable_locked(folder, entry, _walk(folder))
                  if folder.is_dir() and not _is_link(folder) else [])
        skills[key] = _without(entry, hidden)
    try:
        states = _fs.scan_root(project, skills)
    except IllegalEntryError as exc:
        report.add("V2", exc.message, managed=True, path=ROOT, fix=exc.fix)
        return
    except OSError as exc:  # a folder this user cannot list
        where = _rel_to(project.directory, exc.filename)
        where = ROOT if where == "." else where
        report.add("V2", f"`{where}` cannot be read ({exc.strerror}), so `{ROOT}` cannot be "
                         "checked", managed=True, path=where, fix=_READABLE_FIX)
        return
    for name, state in states.items():
        entry = lock.skills.get(name)
        where = f"{ROOT}/{name}"
        folder = root / name
        if state.state is _fs.State.MISSING:
            report.add("V3", f"the folder is missing; the lock installs {entry['name']} here",
                       managed=True, skill=name, path=where, fix=_FETCH)
        elif state.state is _fs.State.ILLEGAL and _is_link(folder):
            report.add("V1", "the skill folder is a symlink; openai-agents skips a symlinked "
                             "skill folder without a word", managed=entry is not None,
                       skill=name, path=where, fix=_UNLINK)
        elif state.state is _fs.State.ILLEGAL and os.path.lexists(folder) \
                and not folder.is_dir():
            report.add("V2", f"`{where}` is not a folder; the root holds only skill folders",
                       managed=entry is not None, skill=name, path=where,
                       fix=f"move it out of `{ROOT}/`")
        elif state.state is _fs.State.ILLEGAL and (not folder.is_dir() or _walk(folder).sound):
            # Neither a link nor a stray file, and sound inside: a name that differs only by
            # case from another folder, or from the name the lock gives it.
            report.add("V2", f"`{where}`: {state.reason}", managed=entry is not None,
                       skill=name, path=where,
                       fix="rename the folder so that no two names differ only by case")
        else:
            _grade_folder(report, folder, name=name, where=where, entry=entry, required=True)
            if orphans and state.state is _fs.State.UNMANAGED_STAMPED:
                _stamped_orphan(report, folder, name=name, where=where, skill_id=state.skill_id)


def _verify_project(project: Project, *, strict: bool) -> Result:
    report = _Report(strict=strict)
    for path in (project.state_dir, project.lock_path, project.npx_lock_path):
        if _is_link(path):
            rel = project.rel(path)
            report.add("V1", f"`{rel}` is a symlink, so the lock cannot be trusted",
                       managed=True, path=rel, fix=_UNLINK)
            return report.result()
    try:
        lock = _lock.load(project, repair=True)
    except SkillsNotInstalledError as exc:
        if os.path.lexists(project.skills_root):
            return _verify_lockless(project, strict=strict)
        return Result.from_error(SkillsNotInstalledError(exc.message, path=exc.path,
                                                         fix=_NO_LOCK_FIX))
    except SkillsLockError as exc:  # corrupt, another version or root, version 1, npx
        return Result.from_error(exc)
    except OSError as exc:
        return Result.from_error(SkillsLockError(
            f"`{_LOCK}` cannot be read ({exc.strerror})", path=project.lock_path,
            fix=_READABLE_FIX))

    _links_under(report, project, project.state_dir)
    _check_lock(report, lock)
    _check_journal(report, project)
    _check_manifest(report, project, lock)
    if _is_link(project.agents_dir):
        report.add("V1", "`.agents` is a symlink", managed=True, path=".agents", fix=_UNLINK)
    else:
        if project.agents_dir.is_dir():
            _links_under(report, project, project.agents_dir, skip=("skills",))
        _check_plugin(report, project, lock)
        _check_root(report, project, lock)
    return report.result()


def _verify_lockless(project: Project, *, strict: bool) -> Result:
    """A root of skills with no lock: every folder unmanaged, nothing compared with a lock,
    a manifest or ``plugin.json`` that are not this engine's yet."""
    report = _Report(strict=strict)
    report.notice(NO_LOCK, f"the project has no lock (`{_LOCK}`), so every folder in "
                           f"`{ROOT}` is graded as an unmanaged, hand-authored skill; "
                           "`decimalai skills add` installs a registry skill there as a "
                           "managed folder", path=_LOCK, fix=_LOCKLESS_FIX)
    if _is_link(project.agents_dir):
        report.add("V1", "`.agents` is a symlink", managed=False, path=".agents", fix=_UNLINK)
    else:
        _check_root(report, project, _lock.empty(), orphans=False)
    return report.result()


# ── a tree: verify PATH ──────────────────────────────────────────────────────


def _verify_tree(given: Path, *, strict: bool) -> Result:
    report = _Report(strict=strict)
    shown = given.as_posix()
    try:
        mode = os.lstat(given).st_mode
    except FileNotFoundError:
        return Result.from_error(UsageError(f"`{shown}` does not exist", path=given,
                                            fix=_PATH_FIX))
    except OSError as exc:
        return Result.from_error(UsageError(f"`{shown}` cannot be read ({exc.strerror})",
                                            path=given, fix=_PATH_FIX))
    name = Path(os.path.abspath(given)).name
    if stat.S_ISLNK(mode):
        what = ("openai-agents skips a symlinked skill folder without a word"
                if os.path.isdir(given) else "the engine never follows one")
        report.add("V1", f"`{shown}` is a symlink; {what}", managed=False, skill=name,
                   path=shown, fix=_UNLINK)
        return report.result()
    if stat.S_ISREG(mode):
        if given.name != "SKILL.md":
            return Result.from_error(UsageError(
                f"`{shown}` is a file, not a skill folder or a SKILL.md", path=given,
                fix=_PATH_FIX))
        folder = given.parent
        _grade_folder(report, folder, name=Path(os.path.abspath(folder)).name,
                      where=folder.as_posix(), entry=None, required=True)
        return report.result()
    if not stat.S_ISDIR(mode):
        return Result.from_error(UsageError(f"`{shown}` is not a folder", path=given,
                                            fix=_PATH_FIX))
    if "SKILL.md" in _entry_names(given):
        _grade_folder(report, given, name=name, where=shown, entry=None, required=True)
        return report.result()

    # A folder of skill folders: <PATH>/<name>/SKILL.md, what every loader reads.
    found = 0
    try:
        children = _scan(given)
    except OSError as exc:
        return Result.from_error(UsageError(f"`{shown}` cannot be read ({exc.strerror})",
                                            path=given, fix=_PATH_FIX))
    for child, child_mode, child_path in children:
        if child.startswith("."):
            continue
        where = (given / child).as_posix()
        if stat.S_ISLNK(child_mode):
            if os.path.isdir(child_path):  # a link to a file is not a skill folder
                found += 1
                report.add("V1", f"`{where}` is a symlink; openai-agents skips a symlinked "
                                 "skill folder without a word", managed=False, skill=child,
                           path=where, fix=_UNLINK)
        elif stat.S_ISDIR(child_mode):
            if "SKILL.md" in _entry_names(Path(child_path)):
                found += 1
                _grade_folder(report, Path(child_path), name=child, where=where, entry=None,
                              required=True)
            elif _no_skill_md(report, Path(child_path), name=child, where=where,
                              required=False):
                found += 1
    if not found:
        report.add("V2", f"no skill in `{shown}`: no SKILL.md in it, or in the folders "
                         "directly inside it", managed=False, path=shown, fix=_PATH_FIX)
    return report.result()


# ── the entry point ──────────────────────────────────────────────────────────


def verify(path: Optional[PathLike] = None, *, directory: Optional[PathLike] = None,
           strict: bool = False) -> Result:
    """Report what a native skill loader would reject, and what the engine would refuse.

    Args:
        path: check this tree by the unmanaged rules instead of the project: a skill
            folder, its SKILL.md, or a folder of skill folders. A relative path is taken
            from ``directory`` when one is given.
        directory: the project directory (default: the project the working directory
            belongs to, see :meth:`Project.find`).
        strict: make the SDK's own rules errors in unmanaged folders too.

    Returns:
        ``Result(changes=(), problems=findings, exit_code)``: exit code 0 when no finding
        is an error (warnings and notices may be listed), 1 when one is. A refusal holds
        only its error: 3 when the lock is unreadable or a version 1 or npx lock, or when
        the project has neither a lock nor ``.agents/skills``; 2 when ``path`` is unusable.
    """
    strict = bool(strict)
    if path is not None:
        given = Path(path)
        if directory is not None and not given.is_absolute():
            given = Path(directory) / given
        return _verify_tree(given, strict=strict)
    project = Project.at(directory) if directory is not None else Project.find()
    return _verify_project(project, strict=strict)
