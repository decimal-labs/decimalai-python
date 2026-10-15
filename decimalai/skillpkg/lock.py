"""``.decimal/skills.lock`` (version 2): exactly what is installed, and ``plugin.json``.

::

    {
      "lock_hash": "<64 hex>",
      "root": ".agents/skills",
      "skills": {
        "acme-refund-policy": {
          "name": "acme/refund-policy",
          "skill_id": "6f1c…", "version_id": "9b2e…", "version": 3,
          "render": {"format": 1, "hash": "<64 hex>", "variant": "foreign"},
          "body_sha256": "…", "description_sha256": "…",
          "path": ".agents/skills/acme-refund-policy",
          "files": {"SKILL.md": "…", "references/fees.md": "…"},
          "executable": [], "unavailable": [],
          "source": {"base_url": "https://api.decimal.ai", "org": null},
          "pin": "latest", "invocation": "model",
          "installed_at": "2026-10-20T10:00:00Z"
        }
      },
      "version": 2
    }

Keys of ``skills`` are install names, which are also the folder names; each skill id
appears once. ``files`` maps every written file to the sha256 of its exact bytes.

``lock_hash`` is :func:`skillevaluation.skillfmt.lock_hash` of the root and the entries
without ``installed_at``, so two machines installing one lock agree, and
``plugin.json``'s version (``lock-<first 12 hex>``) changes only when what is installed
does.

**Reading never resets.** A corrupt file, a version 1 lock (or no version), npx's
``.decimal/skills.lock.json``, entries an older decimalai merged in, and a hand edit
that breaks ``lock_hash`` each raise their own error. ``load(repair=True)`` instead
drops the merged entries (:attr:`Lock.dropped`) and accepts a stale hash
(:attr:`Lock.hash_ok`), for the one command that repairs them.

**Writing** is sorted, indented by 2, newline-terminated and atomic, and happens only
when the bytes change.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from skillevaluation import skillfmt

from .errors import (
    ForeignLockEntriesError,
    LegacyLockError,
    LockHashMismatch,
    SkillsLockError,
    SkillsNotInstalledError,
)
from .project import PLUGIN_NAME, ROOT, Project

__all__ = [
    "FOREIGN_MARKERS",
    "LOCK_VERSION",
    "Lock",
    "check_entry",
    "check_version",
    "commit_entry",
    "dumps",
    "empty",
    "is_foreign",
    "keep_installed_at",
    "load",
    "loads",
    "make_entry",
    "plugin_json_bytes",
    "save",
    "write_plugin_json",
]

LOCK_VERSION = 2

#: An entry lacking any of these was not written by this engine: SDK 0.13.9 merges its
#: version 1 entries into whatever lock it finds.
FOREIGN_MARKERS = ("version_id", "files", "render")

PINS = ("latest", "exact")
INVOCATIONS = ("model", "user")

#: The fields that, when changed, mean the folder was installed anew.
_INSTALL_IDENTITY = ("version_id", "render", "files")

_SOURCE = ".decimal/skills.lock"
_HEX64 = re.compile(r"[0-9a-f]{64}")

_PLUGIN_DESCRIPTION = "Skills installed by `decimalai skills`. Managed: edit through the CLI."


# ── the Lock value ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Lock:
    """A version 2 lock in memory. Build new ones with :meth:`with_entry`.

    Attributes:
        skills: install name -> entry (a JSON-ready dict).
        root: always ``.agents/skills``.
        dropped: with ``load(repair=True)``, the merged entries that were dropped.
        stored_hash: the ``lock_hash`` the file carried (None for a lock not read
            from a file).
    """

    skills: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    root: str = ROOT
    dropped: Tuple[str, ...] = ()
    stored_hash: Optional[str] = None

    @property
    def lock_hash(self) -> str:
        return skillfmt.lock_hash(self.root, {k: dict(v) for k, v in self.skills.items()})

    @property
    def hash_ok(self) -> bool:
        """Whether the hash the file carried recomputes (True for a lock built in memory)."""
        return self.stored_hash is None or self.stored_hash == self.lock_hash

    @property
    def plugin_version(self) -> str:
        return f"lock-{self.lock_hash[:12]}"

    def with_entry(self, name: str, entry: Optional[Mapping[str, Any]]) -> "Lock":
        """A copy with ``name`` set to ``entry``, or removed when ``entry`` is None."""
        skills = {k: v for k, v in self.skills.items() if k != name}
        if entry is not None:
            skills[name] = copy.deepcopy(dict(entry))
        return Lock(skills=dict(sorted(skills.items())), root=self.root)

    def by_skill_id(self, skill_id: str) -> Optional[Tuple[str, Mapping[str, Any]]]:
        """``(install name, entry)`` of the skill with this id, or None."""
        for name, entry in self.skills.items():
            if entry.get("skill_id") == skill_id:
                return name, entry
        return None

    def to_dict(self) -> dict:
        return {
            "version": LOCK_VERSION,
            "root": self.root,
            "lock_hash": self.lock_hash,
            "skills": {k: copy.deepcopy(dict(v)) for k, v in self.skills.items()},
        }


def empty() -> Lock:
    """A lock with no skills."""
    return Lock()


# ── entries ──────────────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_entry(
    install_name: str,
    *,
    name: str,
    skill_id: str,
    version_id: str,
    version: int,
    render: Mapping[str, Any],
    body_sha256: str,
    description_sha256: str,
    files: Mapping[str, str],
    executable: Iterable[str] = (),
    unavailable: Iterable[Mapping[str, str]] = (),
    source: Mapping[str, Any],
    pin: str = "latest",
    invocation: str = "model",
    installed_at: Optional[str] = None,
) -> dict:
    """A lock entry for ``install_name``, checked like one read from a file.

    ``name`` is the registry name; ``files`` maps each written path to its sha256.
    ``installed_at`` defaults to now; see :func:`keep_installed_at` for when an update
    keeps the old value.

    Raises:
        SkillsLockError: a field the lock would refuse to read back.
    """
    entry = {
        "name": name,
        "skill_id": skill_id,
        "version_id": version_id,
        "version": version,
        "render": dict(render),
        "body_sha256": body_sha256,
        "description_sha256": description_sha256,
        "path": f"{ROOT}/{install_name}",
        "files": dict(sorted(dict(files).items())),
        "executable": sorted(executable),
        "unavailable": sorted((dict(u) for u in unavailable), key=lambda u: str(u.get("path"))),
        "source": dict(source),
        "pin": pin,
        "invocation": invocation,
        "installed_at": installed_at if installed_at is not None else _now(),
    }
    check_entry(install_name, entry)
    return entry


def keep_installed_at(new: Mapping[str, Any], old: Optional[Mapping[str, Any]]) -> dict:
    """``new``, with ``old``'s ``installed_at`` when the folder itself did not change.

    An entry's ``installed_at`` moves only when its ``version_id``, ``render`` or
    ``files`` do; a pin change or a re-run of ``install`` leaves it alone.
    """
    out = copy.deepcopy(dict(new))
    if old is not None and all(old.get(k) == new.get(k) for k in _INSTALL_IDENTITY) \
            and "installed_at" in old:
        out["installed_at"] = old["installed_at"]
    return out


def is_foreign(entry: Any) -> bool:
    """An entry this engine did not write: a map lacking ``version_id``, ``files`` or ``render``."""
    return isinstance(entry, dict) and any(k not in entry for k in FOREIGN_MARKERS)


def _bad(name: str, what: str) -> SkillsLockError:
    return SkillsLockError(f"`{_SOURCE}` entry {name!r}: {what}", skill=name, path=_SOURCE)


def _str(v: Any) -> bool:
    return isinstance(v, str) and bool(v)


def _safe_file(path: Any, *, unavailable: bool = False) -> bool:
    if path == "SKILL.md":
        return not unavailable
    problems = skillfmt.validate_attachment_path(path)
    if unavailable:  # a nested SKILL.md is listed, never written
        problems = [p for p in problems if p.rule != "F15"]
    return not problems


def check_entry(name: str, entry: Any, *, root: str = ROOT) -> None:
    """Refuse an entry the engine could not use safely.

    Every path an entry names is one the engine will read or write, so a hand-edited
    lock must not be able to point outside the skill's folder.

    Raises:
        SkillsLockError: names the entry and the field.
    """
    if not skillfmt.is_valid_install_name(name):
        raise _bad(name, "the key is not an install name (lowercase letters, digits and "
                         "single hyphens, at most 64 characters)")
    if not isinstance(entry, dict):
        raise _bad(name, f"the entry must be a map, not {type(entry).__name__}")
    for key in ("name", "skill_id", "version_id"):
        if not _str(entry.get(key)):
            raise _bad(name, f"{key} must be a non-empty string")
    version = entry.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise _bad(name, f"version must be a positive integer, not {version!r}")
    render = entry.get("render")
    if not isinstance(render, dict):
        raise _bad(name, "render must be a map of format, hash and variant")
    fmt = render.get("format")
    if not isinstance(fmt, int) or isinstance(fmt, bool) or fmt not in skillfmt.SUPPORTED_FORMATS:
        err = _bad(name, f"render format {fmt!r} is not one this decimalai writes")
        err.fix = "run `pip install -U decimalai`, then retry"
        raise err
    if not isinstance(render.get("hash"), str) or not _HEX64.fullmatch(render["hash"]):
        raise _bad(name, "render hash must be 64 lowercase hex")
    if render.get("variant") not in skillfmt.VARIANTS:
        raise _bad(name, f"render variant must be one of {', '.join(skillfmt.VARIANTS)}")
    for key in ("body_sha256", "description_sha256"):
        if not isinstance(entry.get(key), str) or not _HEX64.fullmatch(entry[key]):
            raise _bad(name, f"{key} must be 64 lowercase hex")
    if entry.get("path") != f"{root}/{name}":
        raise _bad(name, f"path must be {root}/{name!s}, not {entry.get('path')!r}")
    files = entry.get("files")
    if not isinstance(files, dict) or "SKILL.md" not in files:
        raise _bad(name, "files must be a map of path to sha256 that includes SKILL.md")
    for path, digest in files.items():
        if not _safe_file(path):
            raise _bad(name, f"file path {path!r} is not a safe path inside the skill folder")
        if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
            raise _bad(name, f"the sha256 of {path!r} must be 64 lowercase hex")
    executable = entry.get("executable")
    if not isinstance(executable, list) or any(p not in files for p in executable):
        raise _bad(name, "executable must list files of the entry")
    unavailable = entry.get("unavailable")
    if not isinstance(unavailable, list):
        raise _bad(name, "unavailable must be a list of {path, reason}")
    for item in unavailable:
        if (not isinstance(item, dict) or not _safe_file(item.get("path"), unavailable=True)
                or not isinstance(item.get("reason"), str) or item.get("path") in files):
            raise _bad(name, f"unavailable item {item!r} is not a safe {{path, reason}} "
                             "outside files")
    if not isinstance(entry.get("source"), dict):
        raise _bad(name, "source must be a map of base_url and org")
    if entry.get("pin") not in PINS:
        raise _bad(name, f"pin must be one of {', '.join(PINS)}, not {entry.get('pin')!r}")
    if entry.get("invocation") not in INVOCATIONS:
        raise _bad(name, f"invocation must be one of {', '.join(INVOCATIONS)}, "
                         f"not {entry.get('invocation')!r}")
    if not isinstance(entry.get("installed_at"), str):
        raise _bad(name, "installed_at must be a string")


# ── reading ──────────────────────────────────────────────────────────────────


def _header(raw: bytes) -> dict:
    """The parsed lock, once its version and root are known to be this engine's.

    Raises:
        SkillsLockError: not JSON, not a map, an unknown version or another root.
        LegacyLockError: version 1, or no version.
    """
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SkillsLockError(f"`{_SOURCE}` is not valid JSON ({exc})", path=_SOURCE) from None
    if not isinstance(data, dict):
        raise SkillsLockError(f"`{_SOURCE}` is not a lock: it holds a JSON "
                              f"{type(data).__name__}", path=_SOURCE)
    version = data.get("version")
    if version is None or (type(version) is int and version == 1):
        raise LegacyLockError(f"`{_SOURCE}` is a version 1 lock (or has no version), "
                              "written by an older decimalai", path=_SOURCE)
    if type(version) is not int or version != LOCK_VERSION:
        err = SkillsLockError(f"`{_SOURCE}` has version {version!r}; this decimalai reads "
                              f"version {LOCK_VERSION}", path=_SOURCE)
        if type(version) is int and version > LOCK_VERSION:
            err.fix = "run `pip install -U decimalai`: a newer decimalai wrote this lock"
        raise err
    root = data.get("root")
    if root != ROOT:
        raise SkillsLockError(f"`{_SOURCE}` has root {root!r}; the managed root is `{ROOT}`",
                              path=_SOURCE)
    return data


def loads(raw: bytes, *, repair: bool = False) -> Lock:
    """Parse lock bytes. See :func:`load` for the errors and for ``repair``."""
    data = _header(raw)
    root = data["root"]
    skills = data.get("skills")
    if not isinstance(skills, dict):
        raise SkillsLockError(f"`{_SOURCE}`: skills must be a map", path=_SOURCE)

    foreign = sorted(k for k, v in skills.items() if is_foreign(v))
    if foreign and not repair:
        raise ForeignLockEntriesError(
            f"`{_SOURCE}` holds {len(foreign)} entr{'y' if len(foreign) == 1 else 'ies'} "
            f"an older decimalai wrote: {', '.join(foreign)}",
            path=_SOURCE, details={"entries": foreign})
    kept = {k: v for k, v in skills.items() if k not in foreign}
    seen_ids: Dict[str, str] = {}
    for name in sorted(kept):
        entry = kept[name]
        check_entry(name, entry, root=root)
        other = seen_ids.get(entry["skill_id"])
        if other is not None:
            raise _bad(name, f"skill {entry['skill_id']} is also installed as {other!r}; "
                             "each skill is installed once")
        seen_ids[entry["skill_id"]] = name
    stored = data.get("lock_hash")
    lock = Lock(skills=dict(sorted(kept.items())), root=root, dropped=tuple(foreign),
                stored_hash=stored if isinstance(stored, str) else "")
    if not lock.hash_ok and not repair:
        raise LockHashMismatch(
            f"`{_SOURCE}` lock_hash is {stored!r} but its entries hash to {lock.lock_hash!r}",
            path=_SOURCE, details={"stored": stored, "computed": lock.lock_hash})
    return lock


def load(project: Project, *, missing_ok: bool = False, repair: bool = False) -> Lock:
    """Read the project's lock.

    Args:
        missing_ok: return :func:`empty` when there is no lock, instead of raising.
            Never applies to a lock that exists but cannot be read.
        repair: drop entries an older decimalai merged in (listed in
            :attr:`Lock.dropped`) and accept a ``lock_hash`` that does not recompute
            (:attr:`Lock.hash_ok` is then False). Saving the result repairs the file.

    Raises:
        LegacyLockError: a version 1 lock, one with no version, or npx's
            ``.decimal/skills.lock.json`` beside it: ``decimalai skills migrate``.
        SkillsLockError: corrupt JSON, an unknown version, another root, or an entry
            the engine cannot use (named).
        ForeignLockEntriesError: entries an older decimalai merged in (named).
        LockHashMismatch: ``lock_hash`` does not recompute.
        SkillsNotInstalledError: no lock, and ``missing_ok`` is false.
        IllegalEntryError: ``.decimal`` or the lock is a symlink.
    """
    raw = _read(project, missing_ok=missing_ok)
    return empty() if raw is None else loads(raw, repair=repair)


def check_version(project: Project) -> None:
    """Refuse a lock that is not a version 2 lock of the managed root; read nothing else.

    For callers that only need to know the project is managed by this engine (such as
    :func:`decimalai.skillpkg.project.skills_dir`): the entries and ``lock_hash`` are
    not checked.

    Raises:
        SkillsNotInstalledError: there is no lock.
        LegacyLockError: a version 1 lock, one with no version, or npx's lock beside it.
        SkillsLockError: corrupt JSON, an unknown version or another root.
        IllegalEntryError: ``.decimal`` or the lock is a symlink.
    """
    raw = _read(project, missing_ok=False)
    assert raw is not None
    _header(raw)


def _read(project: Project, *, missing_ok: bool) -> Optional[bytes]:
    """The lock's bytes; None when it is missing and ``missing_ok``."""
    path = project.lock_path
    project.refuse_symlinks(path)
    project.refuse_symlinks(project.npx_lock_path)
    if project.npx_lock_path.exists():
        raise LegacyLockError(
            f"`{project.rel(project.npx_lock_path)}` is the npx installer's lock",
            path=project.npx_lock_path)
    try:
        return path.read_bytes()
    except FileNotFoundError:
        if missing_ok:
            return None
        raise SkillsNotInstalledError(f"`{_SOURCE}` does not exist in {project.directory}",
                                      path=path) from None


# ── writing ──────────────────────────────────────────────────────────────────


def dumps(lock: Lock) -> bytes:
    """The file bytes: keys sorted, indented by 2, one trailing newline."""
    return (json.dumps(lock.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8")


def save(project: Project, lock: Lock) -> bool:
    """Write the lock if its bytes changed, atomically. Returns whether it wrote.

    The hash is always recomputed, so saving a lock read with ``repair=True`` repairs
    the file.
    """
    data = dumps(lock)
    path = project.lock_path
    project.refuse_symlinks(path)
    try:
        if path.read_bytes() == data:
            return False
    except FileNotFoundError:
        pass
    project.write_file(path, data)
    return True


def _hash_or_none(root: str, skills: Mapping[str, Any]) -> Optional[str]:
    """``lock_hash`` of the entries this engine wrote (as ``load(repair=True)`` counts
    them), or None when an entry is not even a map."""
    kept = {k: v for k, v in skills.items() if not is_foreign(v)}
    if not all(isinstance(v, dict) for v in kept.values()):
        return None
    return skillfmt.lock_hash(root, kept)


def _raw_for_recovery(project: Project) -> Optional[bytes]:
    path = project.lock_path
    project.refuse_symlinks(path)
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def commit_entry(project: Project, name: str, entry: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Set (or, with None, remove) one entry of the lock on disk; leave the rest alone.

    For settling a crash. The lock may hold damage the interrupted run was going to
    repair (entries an older decimalai merged in, a ``lock_hash`` a hand edit made
    stale); recovery must not refuse it, and must not hide it either. So only ``name``
    changes, nothing else is validated, and ``lock_hash`` is recomputed only when it was
    correct before. The next command reports and repairs the rest as it would have.

    Returns:
        The ``lock_hash`` the file now carries, or None when there is no lock (and none
        was needed).

    Raises:
        SkillsLockError: the file is not a version 2 lock of the managed root at all.
        LegacyLockError: it is a version 1 lock.
    """
    raw = _raw_for_recovery(project)
    if raw is None:
        if entry is None:
            return None
        fresh = empty().with_entry(name, entry)
        save(project, fresh)
        return fresh.lock_hash
    data = _header(raw)
    root = data["root"]
    skills = data.get("skills")
    if not isinstance(skills, dict):
        raise SkillsLockError(f"`{_SOURCE}`: skills must be a map", path=_SOURCE)
    hash_was_ok = data.get("lock_hash") == _hash_or_none(root, skills)
    skills = {k: v for k, v in skills.items() if k != name}
    if entry is not None:
        skills[name] = copy.deepcopy(dict(entry))
    data["skills"] = dict(sorted(skills.items()))
    if hash_was_ok:
        recomputed = _hash_or_none(root, data["skills"])
        if recomputed is not None:
            data["lock_hash"] = recomputed
    out = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if out != raw:
        project.write_file(project.lock_path, out)
    stored = data.get("lock_hash")
    return stored if isinstance(stored, str) else None


def locked_entries(project: Project) -> Optional[Dict[str, Mapping[str, Any]]]:
    """The entries this engine wrote, read without judging the rest of the lock.

    Returns None when the lock exists but cannot be read as a version 2 lock, so the
    caller can be careful rather than guess; ``{}`` when there is no lock.
    """
    try:
        raw = _raw_for_recovery(project)
        if raw is None:
            return {}
        skills = _header(raw).get("skills")
    except SkillsLockError:
        return None
    if not isinstance(skills, dict):
        return None
    return {k: v for k, v in skills.items() if isinstance(v, dict) and not is_foreign(v)}


def plugin_json_bytes(lock_hash: str) -> bytes:
    """``.agents/.claude-plugin/plugin.json`` for a lock with this hash."""
    data = {
        "name": PLUGIN_NAME,
        "version": f"lock-{lock_hash[:12]}",
        "description": _PLUGIN_DESCRIPTION,
        "author": {"name": "decimalai"},
    }
    return (json.dumps(data, separators=(",", ":")) + "\n").encode("utf-8")


def write_plugin_json(project: Project, lock_hash: str) -> bool:
    """Write ``plugin.json`` if it does not already match ``lock_hash``. Returns whether it wrote."""
    data = plugin_json_bytes(lock_hash)
    path = project.plugin_json_path
    project.refuse_symlinks(path)
    try:
        if path.read_bytes() == data:
            return False
    except FileNotFoundError:
        pass
    project.write_file(path, data)
    return True
