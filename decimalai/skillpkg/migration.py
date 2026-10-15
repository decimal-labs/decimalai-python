"""``decimalai skills migrate``: a version 1 lock, or npx's lock, becomes a manifest and a version 2 lock.

Releases of decimalai before 0.15 recorded installed skills in ``.decimal/skills.lock``
version 1, keyed by registry name, and the npx installer records them in
``.decimal/skills.lock.json``, keyed by url_slug. Neither says which bytes were written, so
neither can be installed again exactly, and the engine refuses both until the project is
migrated. Migration is one explicit step; nothing runs it implicitly::

    from decimalai import skillpkg
    from decimalai.skillpkg.migration import format_table

    result = skillpkg.migrate()
    print(format_table(result.changes))

What it does:

1. **Seeds the manifest.** A version 1 key becomes ``"latest"``: its version numbers were
   never real (the resolver path always wrote 1). An npx key becomes the exact version npx
   installed. Keys are the canonical registry names the registry answers with. An existing
   ``.decimal/skills.toml`` wins: none of its lines changes, and a skill it already lists is
   installed at the version it asks for.
2. **Resolves every entry**, by ``skill_id`` first and by name when the registry no longer
   knows the id. A name that now resolves to a skill other than the recorded ``skill_id`` is
   skipped and reported. Entries that do not resolve are reported, and nothing changes
   unless ``yes=True`` (exit 7). When the version 1 lock and the npx lock list one skill,
   the npx entry wins (an exact pin npx really installed) and the other is a duplicate.
   Each skill to install passes the trust gate's evaluation, without its question: a
   ``blocked`` skill is refused, and so is a ``withheld`` one unless the caller's
   organization owns or uses it; both are reported like an entry that does not resolve.
   ``caution``, ``unreviewed`` and a withheld skill the organization owns or uses install
   with a warning naming why.
3. **Replaces the folders.** Legacy hashes are never trusted (some version 1 locks pin the
   hash of an empty body in every entry), so every skill is fetched again from the registry:

   * the folders **inside the root** (``.agents/skills``) of every entry whose skill is
     installed (a duplicate entry's too) move to ``.decimal/migrated-v1/<id>/skills/``, so
     a stale copy cannot load beside the new one;
   * folders **outside the root** (``.claude/skills`` and the other agent folders) stay
     exactly where they are and are listed: an interactive agent may be reading them;
   * so do the folders of entries that are not installed (unresolvable, or skipped):
     something may still load them. They are listed, and the engine leaves them alone as
     unmanaged folders.

   Nothing is ever deleted.
4. **Backs up** the legacy locks into the same ``.decimal/migrated-v1/<id>/``, removes the
   originals, writes the manifest, installs each skill with the journaled swap of
   :mod:`~decimalai.skillpkg.fs`, and writes the version 2 lock and ``plugin.json``.

The result's ``changes`` are one :class:`MigratedSkill` per legacy entry and one
:class:`LockBackup` per legacy lock; :func:`format_table` prints them. A second run finds no
legacy lock and changes nothing.

**A crash is recoverable.** Every skill is fetched and staged under ``.decimal/staging/``
before anything else changes. The legacy locks are copied into the backup folder, with a
``migration.json`` record, before the originals are removed, and the record is marked
finished last. A run that finds an unfinished record resumes it: it migrates the backed-up
locks again, against what is already installed. At every point each old file is in its
place or in the backup folder.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from . import fs
from . import lock as _lock
from . import manifest as _manifest
from .client import Bundle, RegistryClient, Resolved, ResolveRequest, trust_verdict
from .errors import (
    BundleContractError,
    ExitCode,
    IllegalEntryError,
    InstallNameCollisionError,
    JournalConflictError,
    Result,
    SkillBlockedError,
    SkillNotFoundError,
    SkillpkgError,
    SkillsLockError,
    SkillWithheldError,
    UnavailableError,
    UsageError,
)
from .project import ROOT, Project, fsync_dir

__all__ = [
    "CHECKPOINTS",
    "LegacyEntry",
    "LockBackup",
    "MigratedSkill",
    "MigrationPlan",
    "OldFolder",
    "V1_AGENT_DIRS",
    "format_table",
    "migrate",
    "plan_migration",
]

PathLike = Union[str, "os.PathLike[str]"]

#: The points between the steps of a migration, in order. Tests stop a run at each one.
CHECKPOINTS = ("staged", "backed_up", "originals_removed", "folder_moved",
               "manifest_written", "skill_installed", "finished")

#: Where releases before 0.15 wrote a skill folder for each agent a version 1 entry names
#: (project scope). It records what those releases did, so it never changes.
V1_AGENT_DIRS: Mapping[str, str] = MappingProxyType({
    **{agent: ".agents/skills" for agent in (
        "aider", "antigravity", "cline", "codex", "cursor", "deepagents", "firebender",
        "gemini-cli", "github-copilot", "goose", "kilo-code", "otto", "pear", "supermaven",
        "tabnine", "universal", "warp", "zed")},
    "amp": ".amp/skills",
    "augment": ".augment/skills",
    "claude-code": ".claude/skills",
    "codebuddy": ".codebuddy/skills",
    "command-code": ".commandcode/skills",
    "continue": ".continue/skills",
    "cortex": ".cortex/skills",
    "crush": ".crush/skills",
    "droid": ".factory/skills",
    "melty": ".melty/skills",
    "roo": ".roo/skills",
    "trae": ".trae/skills",
    "void": ".void/skills",
    "windsurf": ".windsurf/skills",
})

V1, NPX = "v1", "npx"

#: What became of a legacy entry.
INSTALLED = "installed"
ALREADY = "already_installed"   # the version 2 lock already has the skill
SKIPPED = "skipped"             # not installed, on purpose (see the row's note)
UNRESOLVABLE = "unresolvable"
DUPLICATE = "duplicate"         # another legacy entry is the same skill

#: The outcomes whose skill ends up installed. Only their folders in the root move to the
#: backup (a duplicate's skill is installed by its twin entry, so its old folder moves too,
#: or two copies would load). An unresolvable or skipped entry's folder stays where it is,
#: unmanaged, since something may still load it.
_MIGRATES = (INSTALLED, ALREADY, DUPLICATE)

_PURPOSE = "migrate"
_RECORD = "migration.json"
_FOLDERS = "skills"             # <backup>/skills/<name>: the old folders from the root
_V1_COPY = "skills.lock"
_NPX_COPY = "skills.lock.json"
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_NPX_DEFAULT_DIR = ".claude/skills"
_NPX_FIX = ("restore `.decimal/skills.lock.json` from version control, or move it out of "
            "`.decimal/` (its folders stay), then run `decimalai skills migrate` again")
_YES_FIX = "or run `decimalai skills migrate --yes` to migrate the other skills without it"


def _checkpoint(step: str) -> None:
    """Called after each step of a migration; does nothing (tests stop a run here)."""


# ── what a migration reports ─────────────────────────────────────────────────


@dataclass(frozen=True)
class LegacyEntry:
    """One entry of a legacy lock, as the migration reads it."""

    source: str                      # "v1" or "npx"
    key: str                         # a registry name (v1) or a url_slug (npx)
    spec: Optional[str]              # the manifest spec it seeds; None when unusable
    claim: str                       # what the legacy lock claimed, for the table
    skill_id: Optional[str] = None   # v1: the recorded skill id, when there was one
    agents: Tuple[str, ...] = ()     # v1: the agents its folders were written for
    folders: Tuple[str, ...] = ()    # npx: the folders its files are in (project-relative)
    empty_body_pin: bool = False     # a hash it records is the sha256 of the empty body


@dataclass(frozen=True)
class OldFolder:
    """A folder a legacy install wrote, and where the migration put it."""

    path: str                        # relative to the project
    moved_to: Optional[str] = None   # the backup it moved to; None when left in place


@dataclass(frozen=True)
class MigratedSkill:
    """One row of the migration table: a legacy entry, and what became of it.

    ``outcome`` is ``installed``, ``already_installed``, ``skipped``, ``unresolvable`` or
    ``duplicate``; ``note`` says why for the last three.
    """

    name: str
    source: str
    claim: str
    outcome: str
    registry_name: Optional[str] = None
    install_name: Optional[str] = None
    version: Optional[int] = None
    spec: Optional[str] = None
    old_folders: Tuple[OldFolder, ...] = ()
    empty_body_pin: bool = False
    note: Optional[str] = None


@dataclass(frozen=True)
class LockBackup:
    """A legacy lock the migration removed, and where its copy is (project-relative)."""

    path: str
    moved_to: str


# ── the plan ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Source:
    """A legacy lock to migrate: the original's path, and its bytes."""

    kind: str           # V1 or NPX
    path: Path          # where the original is (or was)
    copy_name: str      # its name in the backup folder
    raw: bytes
    present: bool       # the original is still in place


@dataclass
class _Item:
    """A legacy entry while it is planned."""

    entry: LegacyEntry
    outcome: Optional[str] = None
    resolved: Optional[Resolved] = None
    asked: Optional[str] = None          # the version the last resolve asked for
    by_name: bool = False                # resolved by its key, not by skill id
    spec: Optional[str] = None           # the manifest spec it installs at
    lock_name: Optional[str] = None      # ALREADY: its key in the version 2 lock
    error: Optional[SkillpkgError] = None
    note: Optional[str] = None
    caution: Optional[str] = None        # INSTALLED: why its safety review has not cleared it
    old: List[OldFolder] = field(default_factory=list)


@dataclass(frozen=True)
class MigrationPlan:
    """What a migration will do, worked out before anything is written.

    :func:`plan_migration` builds one (it resolves, and writes nothing); :func:`migrate`
    builds one and applies it.
    """

    project: Project
    sources: Tuple[_Source, ...]
    backup_dir: Path
    lock: "_lock.Lock"                       # the version 2 lock as it stands (often empty)
    manifest: "_manifest.Manifest"           # with the seeded entries added
    items: Tuple[_Item, ...]
    moves: Tuple[Tuple[str, str], ...]       # (folder, backup), relative to the project
    warnings: Tuple[str, ...]

    @property
    def empty(self) -> bool:
        """True when there is no legacy lock: nothing to migrate."""
        return not self.sources

    @property
    def unresolved(self) -> Tuple[SkillpkgError, ...]:
        return tuple(i.error for i in self.items if i.outcome == UNRESOLVABLE and i.error)

    @property
    def to_install(self) -> Tuple[_Item, ...]:
        return tuple(i for i in self.items if i.outcome == INSTALLED)

    def rows(self) -> Tuple[MigratedSkill, ...]:
        out = []
        for item in self.items:
            e, r = item.entry, item.resolved
            row: Dict[str, Any] = dict(
                name=e.key, source=e.source, claim=e.claim, outcome=item.outcome,
                old_folders=tuple(item.old), empty_body_pin=e.empty_body_pin, note=item.note)
            if item.outcome == INSTALLED and r is not None:
                row.update(registry_name=r.registry_name, install_name=r.install_name,
                           version=r.version, spec=item.spec)
            elif item.outcome == ALREADY and item.lock_name is not None:
                locked = self.lock.skills[item.lock_name]
                row.update(registry_name=locked["name"], install_name=item.lock_name,
                           version=locked["version"], spec=item.spec)
            elif item.outcome == UNRESOLVABLE and item.error is not None:
                row["note"] = item.error.message
            elif r is not None:
                row["registry_name"] = r.registry_name
            out.append(MigratedSkill(**row))
        return tuple(out)

    def lock_backups(self) -> Tuple[LockBackup, ...]:
        return tuple(LockBackup(self.project.rel(s.path),
                                self.project.rel(self.backup_dir / s.copy_name))
                     for s in self.sources)


def plan_migration(*, directory: Union[PathLike, Project, None] = None,
                   client: RegistryClient) -> MigrationPlan:
    """Work out a migration: read the legacy locks, resolve each entry, place each folder.

    Calls the registry's resolve route only; writes nothing. ``directory`` is the project
    root (default: the project the working directory belongs to).

    Raises:
        SkillsLockError: a legacy lock, or the version 2 lock beside it, cannot be read.
        ManifestError: ``.decimal/skills.toml`` cannot be read.
        IllegalEntryError: a folder to move is a symlink or not a directory, or the root is.
        JournalConflictError: an unfinished migration kept a copy of a legacy lock that
            differs from the one in place.
        InstallNameCollisionError: the registry chose a name the project already uses.
        NetworkError, RegistryTooOldError, AccessDeniedError: from the registry.
    """
    project = _project(directory)
    warnings: List[str] = []
    sources, unfinished = _find_sources(project)
    backup_dir = unfinished if unfinished is not None else project.migrated_dir / fs.new_txid()
    current = _current_lock(project, warnings.append)
    manifest = _manifest.load(project, missing_ok=True)
    entries: List[LegacyEntry] = []
    for src in sources:
        read = _v1_entries if src.kind == V1 else _npx_entries
        entries += read(project, src, warnings.append)

    items = [_Item(entry) for entry in entries]
    for item in items:
        if item.entry.spec is None:
            item.outcome = UNRESOLVABLE
            item.error = UnavailableError(
                f"the npx lock records {item.entry.claim}, which is not a version number",
                skill=item.entry.key, fix="check `.decimal/skills.lock.json`")
        elif item.entry.skill_id and current.by_skill_id(item.entry.skill_id):
            item.outcome = ALREADY
            item.lock_name = current.by_skill_id(item.entry.skill_id)[0]  # type: ignore[index]

    states = fs.scan_root(project, current.skills) if sources else {}
    _resolve(client, items, current, manifest, states)
    _seed_manifest(client, manifest, items, current)
    moves = _place_folders(project, items, current, backup_dir)
    _check_names(items, current, states, moves)
    warnings += _notes(items)
    return MigrationPlan(project, tuple(sources), backup_dir, current, manifest, tuple(items),
                         moves, tuple(warnings))


# ── the operation ────────────────────────────────────────────────────────────


def migrate(
    *,
    directory: Optional[PathLike] = None,
    yes: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    channel: str = "sdk",
    client: Optional[RegistryClient] = None,
) -> Result:
    """Migrate a project from a version 1 lock, or npx's lock, to a manifest and a version 2 lock.

    See the module docstring for what happens. ``directory`` is the project root (default:
    the project the working directory belongs to). ``yes`` migrates the entries that resolve
    even when others do not; those are reported in ``problems``. ``client`` replaces the
    registry client built from ``api_key``, ``base_url`` and ``channel`` (and the
    environment, as :meth:`RegistryClient.from_env` reads it).

    Returns:
        ``Result(changes, problems, exit_code)``. ``changes`` holds a
        :class:`MigratedSkill` per legacy entry and a :class:`LockBackup` per legacy lock;
        :func:`format_table` prints them. ``problems`` holds the warnings, and the errors of
        entries that did not resolve. With no legacy lock it is empty, and nothing on disk
        or on the network was touched. Exit 7 when entries do not resolve and ``yes`` is
        false; then nothing changed.
    """
    own = client is None
    try:
        project = _project(directory)
        if not _has_work(project):
            return Result()
        if client is None:
            client = RegistryClient.from_env(api_key=api_key, base_url=base_url, channel=channel)
        with fs.ProcessLock(project) as process_lock:
            problems: List[Any] = []
            settled = fs.recover(project)
            if settled is not None:
                problems.append(f"{settled.name}: settled an interrupted change "
                                f"(kept the {settled.outcome} folder)")
            fs.discard_staging(project)
            plan = plan_migration(directory=project, client=client)
            if plan.empty:
                return Result((), problems)
            if plan.unresolved and not yes:
                refused = [_with_fix(e, _YES_FIX) for e in plan.unresolved]
                return Result((), refused + problems + list(plan.warnings), ExitCode.UNAVAILABLE)
            return _apply(plan, client, process_lock, problems)
    except SkillpkgError as exc:
        return Result.from_error(exc)
    finally:
        if own and client is not None:
            client.close()


def _apply(plan: MigrationPlan, client: RegistryClient, process_lock: fs.ProcessLock,
           problems: List[Any]) -> Result:
    project = plan.project
    warnings: List[str] = list(plan.warnings)
    txid, staged = _stage_all(plan, client, process_lock, warnings.append)
    _checkpoint("staged")

    # From here the legacy locks live in the backup folder, and its record says so.
    _back_up(plan)
    _checkpoint("backed_up")
    for src in plan.sources:
        if src.present:
            _remove_original(project, src)
    _checkpoint("originals_removed")
    for rel, dest in plan.moves:
        process_lock.refresh()
        fs.move_folder(project, _path(project, rel), _path(project, dest))
        _checkpoint("folder_moved")

    _manifest.save(project, plan.manifest)
    _checkpoint("manifest_written")
    current = plan.lock
    for item, entry in staged:
        process_lock.refresh()
        assert item.resolved is not None
        current = fs.swap(project, txid, item.resolved.install_name, entry, current,
                          on_warning=warnings.append)
        _checkpoint("skill_installed")
    _lock.save(project, current)  # also when nothing was installed: the project is v2 now
    _lock.write_plugin_json(project, current.lock_hash)
    fs.discard_staging(project, txid)

    rows = plan.rows()
    _write_record(project, plan.backup_dir, {
        "finished": True,
        "finished_at": _now(),
        "sources": [project.rel(s.path) for s in plan.sources],
        "skills": [asdict(row) for row in rows],
    })
    _checkpoint("finished")
    return Result(rows + plan.lock_backups(),
                  list(plan.unresolved) + problems + warnings)


def _stage_all(plan: MigrationPlan, client: RegistryClient, process_lock: fs.ProcessLock,
               warn: Callable[[str], None]) -> Tuple[str, List[Tuple[_Item, dict]]]:
    """Fetch and stage every skill to install. Nothing outside ``.decimal/staging/`` changes;
    a failure removes what was staged."""
    project = plan.project
    txid = fs.new_txid()
    staged: List[Tuple[_Item, dict]] = []
    try:
        for item in plan.to_install:
            process_lock.refresh()
            resolved = item.resolved
            assert resolved is not None
            bundle = client.fetch_resolved(resolved, purpose=_PURPOSE)
            entry = _lock_entry(item, bundle, client)
            fs.stage(project, txid, resolved.install_name,
                     {f.path: f.content for f in bundle.files}, entry)
            for note in dict.fromkeys(resolved.policy_notes + bundle.policy_notes):
                warn(f"{resolved.registry_name}: {note}")
            staged.append((item, entry))
    except BaseException:
        with contextlib.suppress(Exception):
            fs.discard_staging(project, txid)
        raise
    return txid, staged


def _lock_entry(item: _Item, bundle: Bundle, client: RegistryClient) -> dict:
    r = item.resolved
    assert r is not None
    invocation = bundle.invocation or r.invocation
    org = r.raw.get("org_id")
    try:
        return _lock.make_entry(
            r.install_name, name=r.registry_name, skill_id=r.skill_id,
            version_id=r.version_id, version=r.version, render=r.render.to_dict(),
            body_sha256=r.body_sha256, description_sha256=r.description_sha256,
            files=bundle.file_hashes, executable=bundle.executable,
            unavailable=[{"path": u.path, "reason": u.reason} for u in bundle.unavailable],
            source={"base_url": client.base_url,
                    "org": org if isinstance(org, str) and org else None},
            pin="latest" if item.spec == "latest" else "exact",
            invocation="user" if invocation == "user" else "model",
        )
    except SkillsLockError as exc:
        raise BundleContractError(f"the registry's answer cannot be locked: {exc.message}",
                                  skill=r.registry_name) from None


# ── the backup folder ────────────────────────────────────────────────────────


def _back_up(plan: MigrationPlan) -> None:
    """Copy the legacy locks into the backup folder, then write its unfinished record.

    The record is written last, so a record means the copies are complete.
    """
    project, backup = plan.project, plan.backup_dir
    project.makedirs(backup)
    ignore = project.migrated_dir / ".gitignore"
    if not os.path.lexists(ignore):
        project.write_file(ignore, b"*\n")  # backups are never committed
    for src in plan.sources:
        target = backup / src.copy_name
        if _read(project, target) != src.raw:
            project.write_file(target, src.raw)
    if not (backup / _RECORD).is_file():
        _write_record(project, backup, {
            "finished": False,
            "started_at": _now(),
            "sources": [project.rel(s.path) for s in plan.sources],
        })


def _write_record(project: Project, backup: Path, data: Mapping[str, Any]) -> None:
    project.write_file(backup / _RECORD,
                       (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _remove_original(project: Project, src: _Source) -> None:
    """Remove a legacy lock whose copy is in the backup, if it is still the file we read."""
    path = src.path
    project.refuse_symlinks(path)
    try:
        now = path.read_bytes()
    except FileNotFoundError:
        return
    if now != src.raw:
        raise JournalConflictError(
            f"`{project.rel(path)}` changed while it was being migrated", path=path,
            fix="run `decimalai skills migrate` again once nothing else writes it")
    os.unlink(path)
    fsync_dir(path.parent)


def _unfinished(project: Project) -> Optional[Path]:
    """The newest backup folder whose record says its migration did not finish."""
    base = project.migrated_dir
    project.refuse_symlinks(base)
    if not base.is_dir():
        return None
    found = []
    with os.scandir(base) as it:
        for child in it:
            if not child.is_dir(follow_symlinks=False):
                continue
            try:
                record = json.loads((Path(child.path) / _RECORD).read_bytes().decode("utf-8"))
            except (OSError, UnicodeDecodeError, ValueError):
                continue
            if isinstance(record, dict) and record.get("finished") is False:
                found.append(Path(child.path))
    return max(found, key=lambda p: p.name) if found else None


# ── reading the legacy locks ─────────────────────────────────────────────────


def _has_work(project: Project) -> bool:
    """Whether there is anything to migrate: read only, so a no-op run writes nothing."""
    if os.path.lexists(project.npx_lock_path):
        return True
    raw = _read(project, project.lock_path)
    return (raw is not None and _is_v1(raw)) or _unfinished(project) is not None


def _find_sources(project: Project) -> Tuple[List[_Source], Optional[Path]]:
    """The legacy locks to migrate, and the unfinished backup folder to resume in."""
    unfinished = _unfinished(project)
    out: List[_Source] = []
    for kind, path, copy_name in ((V1, project.lock_path, _V1_COPY),
                                  (NPX, project.npx_lock_path, _NPX_COPY)):
        project.refuse_symlinks(path)
        original = _read(project, path)
        if kind == V1 and original is not None and not _is_v1(original):
            original = None  # a version 2 lock: the current lock, not a source
        kept = _read(project, unfinished / copy_name) if unfinished is not None else None
        if original is not None and kept is not None and original != kept:
            raise JournalConflictError(
                f"`{project.rel(path)}` differs from the copy an unfinished migration kept in "
                f"`{project.rel(unfinished / copy_name)}`",  # type: ignore[operator]
                path=path,
                fix="move one of the two out of the way, then run `decimalai skills migrate` again")
        raw = original if original is not None else kept
        if raw is not None:
            out.append(_Source(kind, path, copy_name, raw, original is not None))
    return out, unfinished


def _current_lock(project: Project, warn: Callable[[str], None]) -> "_lock.Lock":
    """The version 2 lock beside an npx lock (or left by an interrupted run), else empty."""
    path = project.lock_path
    raw = _read(project, path)
    if raw is None or _is_v1(raw):
        return _lock.empty()
    current = _lock.loads(raw, repair=True)
    if current.dropped:
        warn(f"`.decimal/skills.lock` held {len(current.dropped)} entr"
             f"{'y' if len(current.dropped) == 1 else 'ies'} an older decimalai merged in "
             f"({', '.join(current.dropped)}); they were dropped, and their folders stay, "
             "unmanaged")
    if not current.hash_ok:
        warn("`.decimal/skills.lock` had a lock_hash that did not recompute; it was rewritten")
    return current


def _v1_entries(project: Project, src: _Source, warn: Callable[[str], None]) -> List[LegacyEntry]:
    rel = project.rel(src.path)
    data = json.loads(src.raw.decode("utf-8"))  # parses: _is_v1 said so
    skills = data.get("skills", {})
    if not isinstance(skills, dict):
        raise SkillsLockError(f"`{rel}` (version 1): `skills` must be a map", path=src.path,
                              fix="restore it from version control, then run "
                                  "`decimalai skills migrate` again")
    out = []
    for key, value in skills.items():
        if not key.strip() or not isinstance(value, dict):
            warn(f"`{rel}`: skipped the entry {key!r}, which is not a map")
            continue
        sid = value.get("skill_id")
        agents = value.get("agents")
        agents = (tuple(a for a in agents if isinstance(a, str))
                  if isinstance(agents, list) and agents else ("universal",))
        if value.get("scope", "project") != "project":
            agents = ()  # written outside the project: not this migration's to list
        for agent in agents:
            if agent not in V1_AGENT_DIRS:
                warn(f"{key}: `{rel}` names the agent {agent!r}, which no release wrote "
                     "folders for; none was looked for")
        version = value.get("version")
        out.append(LegacyEntry(
            V1, key, "latest",
            f"version {version}" if _is_int(version) else "no version",
            skill_id=sid.strip() if isinstance(sid, str) and sid.strip() else None,
            agents=agents,
            empty_body_pin=_pins_empty_body(value.get("content_hash"), value.get("body_hash")),
        ))
    return out


def _npx_entries(project: Project, src: _Source, warn: Callable[[str], None]) -> List[LegacyEntry]:
    rel = project.rel(src.path)
    try:
        data = json.loads(src.raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SkillsLockError(f"`{rel}` is not valid JSON ({exc})", path=src.path,
                              fix=_NPX_FIX) from None
    if not isinstance(data, dict):
        raise SkillsLockError(f"`{rel}` is not an npx lock: it holds a JSON "
                              f"{type(data).__name__}", path=src.path, fix=_NPX_FIX)
    out = []
    for key, value in data.items():
        if not key.strip() or not isinstance(value, dict):
            warn(f"`{rel}`: skipped the entry {key!r}, which is not a map")
            continue
        version = value.get("version")
        usable = _is_int(version) and version >= 1
        files = value.get("files")
        folders: List[str] = []
        for listed in files if isinstance(files, list) else ():
            path = _safe_rel(listed)
            if path is None:
                warn(f"{key}: `{rel}` lists {listed!r}, which is not a path inside the "
                     "project; it was not touched")
            elif path.name == "SKILL.md" and len(path.parts) > 1 \
                    and path.parent.as_posix() not in folders:
                folders.append(path.parent.as_posix())
        if not files and _component(key):
            folders.append(f"{_NPX_DEFAULT_DIR}/{key}")  # where npx writes by default
        out.append(LegacyEntry(
            NPX, key, str(version) if usable else None,
            f"version {version}" if usable else f"version {version!r}",
            folders=tuple(folders), empty_body_pin=_pins_empty_body(value.get("sha256")),
        ))
    return out


# ── resolving ────────────────────────────────────────────────────────────────


def _resolve(client: RegistryClient, items: List[_Item], current: "_lock.Lock",
             manifest: "_manifest.Manifest", states: Mapping[str, "fs.FolderState"]) -> None:
    """Resolve every item still open, with the install names the project leaves free.

    The folders that will move out of the root are free; every other folder there is taken.
    Which folders move is only known after resolving: a version 1 folder may be named after
    the skill's url_slug, and only entries that migrate move theirs. So the open items are
    resolved (and settled) again until the freed names stop changing.
    """
    freed = _moving_names(items, states)
    for _ in range(4):
        for item in items:  # settled outcomes depend on the names; worked out again
            if item.outcome in (INSTALLED, DUPLICATE) or (
                    item.outcome == ALREADY and item.resolved is not None):
                item.outcome, item.lock_name, item.note, item.caution = None, None, None, None
        open_items = [i for i in items if i.outcome is None]
        for item in open_items:
            item.resolved = None
        _resolve_open(client, open_items, _taken(current, states, freed), manifest)
        _settle(items, current)
        now = _moving_names(items, states)
        if now == freed:
            return
        freed = now


def _resolve_open(client: RegistryClient, items: List[_Item],
                  taken: Dict[str, Optional[str]], manifest: "_manifest.Manifest") -> None:
    if not items:
        return
    # By the recorded skill id; by the key when there is none.
    for item in items:
        item.asked = _asked(item, manifest)
        item.by_name = not item.entry.skill_id
    answers = client.resolve([_request(i, by_name=i.by_name) for i in items],
                             taken=taken, purpose=_PURPOSE)
    gone: List[_Item] = []
    for item, answer in zip(items, answers):
        if isinstance(answer, Resolved):
            item.resolved = answer
        elif item.entry.skill_id and isinstance(answer, SkillNotFoundError):
            gone.append(item)
        else:
            item.outcome, item.error = UNRESOLVABLE, answer

    # The registry no longer knows the id: the name may still find the same skill.
    if gone:
        answers = client.resolve([_request(i, by_name=True) for i in gone],
                                 taken=_with_chosen(taken, items), purpose=_PURPOSE)
        for item, answer in zip(gone, answers):
            item.by_name = True
            sid = item.entry.skill_id
            if isinstance(answer, Resolved) and answer.skill_id == sid:
                item.resolved = answer
            elif isinstance(answer, Resolved):
                item.outcome = SKIPPED
                item.note = (f"version 1 recorded skill {sid}, which the registry no longer "
                             f"has; the name now finds skill {answer.skill_id}, a different "
                             "skill, so it was not installed")
            elif isinstance(answer, SkillNotFoundError):
                item.outcome = UNRESOLVABLE
                item.error = SkillNotFoundError(
                    f"neither its skill id {sid} nor its name resolves", skill=item.entry.key,
                    details=answer.details or None)
            else:
                item.outcome, item.error = UNRESOLVABLE, answer

    # A skill the manifest already lists installs at the manifest's spec.
    again: List[_Item] = []
    for item in items:
        if item.resolved is None:
            continue
        listed, _ = _listing(manifest, item.resolved)
        item.spec = listed.spec if listed is not None else item.entry.spec
        if item.spec != item.asked and not (
                item.spec != "latest" and item.resolved.version == int(item.spec or 0)):
            again.append(item)
    if again:
        answers = client.resolve(
            [ResolveRequest(skill_id=i.resolved.skill_id, version=i.spec)  # type: ignore[union-attr]
             for i in again],
            taken=_with_chosen(taken, items), purpose=_PURPOSE)
        for item, answer in zip(again, answers):
            item.asked = item.spec
            if isinstance(answer, Resolved):
                item.resolved = answer
            else:
                item.resolved = None
                item.outcome, item.error = UNRESOLVABLE, answer


def _asked(item: _Item, manifest: "_manifest.Manifest") -> Optional[str]:
    """The version to ask for: the manifest's, when it lists the version 1 key as is."""
    e = item.entry
    listed = manifest.get(e.key) if e.source == V1 else None
    if listed is not None and listed.id in (None, e.skill_id):
        return listed.spec
    return e.spec


def _request(item: _Item, *, by_name: bool) -> ResolveRequest:
    version = item.asked or "latest"
    if by_name:
        return ResolveRequest(ref=item.entry.key, version=version)
    return ResolveRequest(skill_id=item.entry.skill_id, version=version)


def _taken(current: "_lock.Lock", states: Mapping[str, "fs.FolderState"],
           freed: Set[str]) -> Dict[str, Optional[str]]:
    """The install names the project uses: locked ones, and folders that stay in the root."""
    taken: Dict[str, Optional[str]] = {k: e["skill_id"] for k, e in current.skills.items()}
    for name, state in states.items():
        if name not in taken and name not in freed and state.state is not fs.State.MISSING:
            taken[name] = None
    return taken


def _with_chosen(taken: Mapping[str, Optional[str]],
                 items: Iterable[_Item]) -> Dict[str, Optional[str]]:
    out = dict(taken)
    for item in items:
        if item.resolved is not None and item.outcome is None:
            out[item.resolved.install_name] = item.resolved.skill_id
    return out


def _listing(manifest: "_manifest.Manifest",
             resolved: Resolved) -> Tuple[Optional["_manifest.ManifestEntry"], bool]:
    """``(the manifest entry for this skill or None, whether its name belongs to another skill)``."""
    for entry in manifest.entries.values():
        if entry.id == resolved.skill_id:
            return entry, False
    entry = manifest.get(resolved.registry_name)
    if entry is None:
        return None, False
    if entry.id is not None:
        return None, True
    return entry, False


def _settle(items: List[_Item], current: "_lock.Lock") -> None:
    """Each skill installs once: one already locked, or met twice, is not installed again,
    and one the trust gate would refuse is not installed at all.

    Of two legacy entries for one skill, the npx entry wins (an exact pin npx really
    installed, where a version 1 version number never meant anything), then the first.
    """
    first: Dict[str, _Item] = {}
    for item in sorted(items, key=lambda i: i.entry.source != NPX):  # stable: npx first
        if item.outcome is not None or item.resolved is None:
            continue
        r = item.resolved
        sid = r.skill_id
        locked = current.by_skill_id(sid)
        refusal, caution = trust_verdict(r)
        if locked is not None:
            item.outcome, item.lock_name = ALREADY, locked[0]  # nothing is installed
        elif refusal == "blocked":
            item.outcome = UNRESOLVABLE
            item.error = SkillBlockedError(
                f"{r.registry_name} v{r.version} failed its safety review and is blocked",
                skill=item.entry.key)
        elif refusal == "withheld":
            item.outcome = UNRESOLVABLE
            item.error = SkillWithheldError(
                f"the registry withholds {r.registry_name} from installs", skill=item.entry.key)
        elif sid in first:
            other = first[sid].entry
            item.outcome = DUPLICATE
            item.note = f"the same skill as the {other.source} entry {other.key!r}"
        else:
            item.outcome, item.caution = INSTALLED, caution
            first[sid] = item


def _seed_manifest(client: RegistryClient, manifest: "_manifest.Manifest", items: List[_Item],
                   current: "_lock.Lock") -> None:
    """Add the skills the manifest does not list yet; never change a line it has."""
    added: Dict[str, str] = {}
    check: List[_Item] = []
    for item in items:
        if item.outcome == INSTALLED and item.resolved is not None:
            r = item.resolved
            listed, clash = _listing(manifest, r)
            if listed is not None and r.registry_name not in added:
                continue
            if clash or added.get(r.registry_name, r.skill_id) != r.skill_id:
                item.outcome = SKIPPED
                item.note = (f"`.decimal/skills.toml` already lists another skill as "
                             f"{r.registry_name!r}")
                continue
            manifest.set(r.registry_name, item.spec or "latest")
            added[r.registry_name] = r.skill_id
            if not (item.by_name and item.entry.key == r.registry_name):
                check.append(item)
        elif item.outcome == ALREADY and item.lock_name is not None:
            locked = current.skills[item.lock_name]
            spec = "latest" if locked["pin"] == "latest" else str(locked["version"])
            listed = next((e for e in manifest.entries.values()
                           if e.id == locked["skill_id"]), None) or manifest.get(locked["name"])
            item.spec = listed.spec if listed is not None else spec
            if listed is None:
                manifest.set(locked["name"], spec)
                added[locked["name"]] = locked["skill_id"]
    # The name alone must find the skill again; when it does not, the id is recorded too.
    if check:
        answers = client.resolve(
            [ResolveRequest(ref=i.resolved.registry_name) for i in check],  # type: ignore[union-attr]
            purpose=_PURPOSE)
        for item, answer in zip(check, answers):
            r = item.resolved
            assert r is not None
            if not isinstance(answer, Resolved) or answer.skill_id != r.skill_id:
                manifest.set(r.registry_name, item.spec or "latest", id=r.skill_id)


# ── folders ──────────────────────────────────────────────────────────────────


def _candidates(item: _Item) -> List[str]:
    """The project-relative folders a legacy entry may have written."""
    e = item.entry
    if e.source == NPX:
        return list(e.folders)
    names = [e.key] if _component(e.key) else []
    slug = item.resolved.url_slug if item.resolved is not None else None
    if slug and _component(slug) and slug not in names:
        names.append(slug)
    out: List[str] = []
    for agent in e.agents:
        base = V1_AGENT_DIRS.get(agent)
        for name in names if base else ():
            rel = f"{base}/{name}"
            if rel not in out:
                out.append(rel)
    return out


def _in_root(rel: str) -> Optional[str]:
    """The folder name, when ``rel`` is a folder directly inside the root."""
    prefix = ROOT + "/"
    name = rel[len(prefix):] if rel.startswith(prefix) else ""
    return name if _component(name) else None


def _moving_names(items: Iterable[_Item], states: Mapping[str, "fs.FolderState"]) -> Set[str]:
    """The folders in the root that move: those an entry still open, or one that migrates,
    declared, and that are there."""
    return {name for item in items if item.outcome is None or item.outcome in _MIGRATES
            for rel in _candidates(item)
            if (name := _in_root(rel)) is not None and name in states
            and states[name].state is not fs.State.MISSING}


def _place_folders(project: Project, items: List[_Item], current: "_lock.Lock",
                   backup: Path) -> Tuple[Tuple[str, str], ...]:
    """Decide each old folder's fate, and record it on every entry that declared it.

    A folder in the root moves to the backup when an entry whose skill is installed
    declared it. Every other folder stays where it is: outside the root, or declared only by
    entries that are not installed. Returns the moves.
    """
    owners: Dict[str, List[_Item]] = {}
    for item in items:
        for rel in _candidates(item):
            owners.setdefault(rel, []).append(item)
    moves: List[Tuple[str, str]] = []
    used: Set[Path] = set()
    for rel, declared_by in owners.items():
        name = _in_root(rel)
        path = _path(project, rel)
        if name is None:
            folder: Optional[OldFolder] = OldFolder(rel) if os.path.lexists(path) else None
        else:
            kept = backup / _FOLDERS / name
            present = name not in current.skills and os.path.lexists(path)
            if present and any(i.outcome in _MIGRATES for i in declared_by):
                project.refuse_symlinks(path)  # refused now, before anything changes
                mode = os.lstat(path).st_mode
                if not stat.S_ISDIR(mode):
                    raise IllegalEntryError(
                        f"`{rel}` is not a directory, so it cannot be moved to the backup",
                        skill=declared_by[0].entry.key, path=path)
                dest = _free(kept, used)
                used.add(dest)
                moves.append((rel, project.rel(dest)))
                folder = OldFolder(rel, project.rel(dest))
            elif present:
                folder = OldFolder(rel)  # not installed: stays, unmanaged
            elif os.path.isdir(kept):
                # An earlier, interrupted run already moved it (and the lock may now manage a
                # new folder under the same name).
                folder = OldFolder(rel, project.rel(kept))
            else:
                folder = None
        for item in declared_by if folder is not None else ():
            item.old.append(folder)
    for item in items:
        item.old.sort(key=lambda f: f.path)
    return tuple(moves)


def _free(path: Path, used: Set[Path]) -> Path:
    """``path``, or ``path.1``, ``path.2``... : the first that is not there yet."""
    candidate, n = path, 0
    while os.path.lexists(candidate) or candidate in used:
        n += 1
        candidate = path.with_name(f"{path.name}.{n}")
    return candidate


def _check_names(items: List[_Item], current: "_lock.Lock",
                 states: Mapping[str, "fs.FolderState"], moves: Iterable[Tuple[str, str]]) -> None:
    """The registry chose names around ``taken``; make sure no install lands on a folder."""
    moving = {(_in_root(rel) or "").casefold() for rel, _ in moves}
    blocked = {name.casefold() for name, s in states.items()
               if s.state is not fs.State.MISSING and name.casefold() not in moving}
    blocked |= {name.casefold() for name in current.skills}
    chosen: Set[str] = set()
    for item in items:
        if item.outcome != INSTALLED or item.resolved is None:
            continue
        name = item.resolved.install_name
        if name.casefold() in blocked or name.casefold() in chosen:
            raise InstallNameCollisionError(
                f"the registry chose the folder name `{name}`, which this project already uses",
                skill=item.resolved.registry_name)
        chosen.add(name.casefold())


# ── the report ───────────────────────────────────────────────────────────────


def _notes(items: List[_Item]) -> List[str]:
    out: List[str] = []
    for item in items:
        key = item.entry.key
        if item.outcome in (SKIPPED, DUPLICATE) and item.note:
            out.append(f"{key}: {'skipped' if item.outcome == SKIPPED else 'installed once'}: "
                       f"{item.note}")
        if item.outcome == INSTALLED and item.caution:
            out.append(f"{key}: its safety review is {item.caution}, so the registry has not "
                       "cleared it; review it before relying on it")
        for folder in item.old:
            if folder.moved_to is not None:
                continue
            if _in_root(folder.path) is not None:
                out.append(f"{key}: `{folder.path}` was left in place, unmanaged, because "
                           f"this entry was not installed ({(item.outcome or '').replace('_', ' ')}); "
                           "`decimalai skills` never touches it")
            else:
                out.append(f"{key}: `{folder.path}` was left in place: it is outside "
                           f"`{ROOT}`, so `decimalai skills` does not manage it; remove it "
                           "once no agent reads it")
    empty = [i.entry.key for i in items if i.entry.empty_body_pin]
    if empty:
        out.append(f"{len(empty)} legacy entr{'y' if len(empty) == 1 else 'ies'} pinned the "
                   f"empty body ({', '.join(empty)}); legacy hashes are never trusted, so "
                   "every skill was fetched again")
    return out


def format_table(changes: Iterable[Any]) -> str:
    """The migration table: each skill, what the legacy lock claimed, the version installed,
    and where its old folders went; then where the legacy locks went."""
    rows = [c for c in changes if isinstance(c, MigratedSkill)]
    backups = [c for c in changes if isinstance(c, LockBackup)]
    table = [("skill", "legacy claim", "installed", "old folders")]
    for r in rows:
        claim = f"{r.source} {r.claim}" + (" (empty body)" if r.empty_body_pin else "")
        if r.outcome in (INSTALLED, ALREADY):
            installed = f"{r.install_name}@{r.version} " + (
                "(latest)" if r.spec == "latest" else "(pinned)")
            if r.outcome == ALREADY:
                installed += ", already installed"
        else:
            installed = r.outcome.replace("_", " ")
        folders = "; ".join(f"{f.path} -> {f.moved_to}" if f.moved_to
                            else f"{f.path} (left in place)" for f in r.old_folders)
        table.append((r.name, claim, installed, folders or "-"))
    widths = [max(len(row[i]) for row in table) for i in range(3)]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)) + "  " + row[3]
             for row in table]
    lines += [f"{b.path} -> {b.moved_to}" for b in backups]
    return "\n".join(lines) + "\n"


# ── small helpers ────────────────────────────────────────────────────────────


def _project(directory: Union[PathLike, Project, None]) -> Project:
    if isinstance(directory, Project):
        return directory
    if directory is None:
        return Project.find()
    project = Project.at(directory)
    if not project.directory.is_dir():
        raise UsageError(f"{project.directory} is not a directory",
                         fix="pass the project directory with `--directory`")
    return project


def _path(project: Project, rel: str) -> Path:
    return project.directory.joinpath(*rel.split("/"))


def _read(project: Project, path: Path) -> Optional[bytes]:
    project.refuse_symlinks(path)
    try:
        return path.read_bytes()
    except (FileNotFoundError, NotADirectoryError):
        return None


def _is_v1(raw: bytes) -> bool:
    """A version 1 lock: a JSON map whose ``version`` is 1 or missing (the lock's own rule)."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    version = data.get("version")
    return version is None or (type(version) is int and version == 1)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _pins_empty_body(*hashes: Any) -> bool:
    """Whether a recorded hash (whole, or its first 12 hex or more) is the empty body's."""
    return any(isinstance(h, str) and len(h) >= 12 and _EMPTY_SHA256.startswith(h.lower())
               for h in hashes)


def _component(name: Any) -> bool:
    """One safe path component, as the legacy writers required of a folder name."""
    return (isinstance(name, str) and name not in ("", ".", "..")
            and not any(c in name for c in "/\\\x00"))


def _safe_rel(text: Any) -> Optional[PurePosixPath]:
    """A path from an npx lock, if it stays inside the project (it is never trusted)."""
    if not isinstance(text, str) or not text or "\x00" in text:
        return None
    path = PurePosixPath(text.replace("\\", "/"))
    if path.is_absolute() or not path.parts or ".." in path.parts or ":" in path.parts[0]:
        return None
    return path


def _with_fix(error: SkillpkgError, extra: str) -> SkillpkgError:
    out = copy.copy(error)
    out.fix = f"{error.fix}; {extra}" if error.fix else extra
    return out


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
