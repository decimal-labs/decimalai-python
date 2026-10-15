"""The operations of the skills engine: add, install, install --frozen, update, remove, status.

Each operation is one call that returns a :class:`~decimalai.skillpkg.errors.Result`:
``changes`` (one :class:`Change` per folder or entry it changed; for :func:`status`, one
:class:`SkillStatus` row per skill), ``problems`` (:class:`Notice` warnings, and the
refusal when there is one) and the documented ``exit_code``. A refusal is returned, not
raised, so the CLI and the Python API report it the same way.

Every operation that writes follows one order, so a failure at any point before the
first folder moves leaves the project exactly as it was:

1. take ``.decimal/.skills.lck`` (one run per project at a time);
2. settle a swap a crashed run left half done, and discard leftover staging;
3. plan in memory: what each install name becomes, from the manifest, the lock and the
   state of every folder; refuse local edits and unmanaged folders the plan would
   replace, and a ``.agents/.claude-plugin/plugin.json`` this engine did not write,
   unless ``force`` (which backs each one up to ``.decimal/backup/<txid>/`` first); a
   folder with a symlink, or with a file the engine cannot read, is refused even then;
4. fetch every bundle and stage every folder under ``.decimal/staging/<txid>/``, each
   byte checked against the registry's declaration (``install --frozen``: the lock's);
5. apply one skill at a time: the folder is swapped whole behind a journal and the lock
   is written after each swap, so a crash leaves every completed skill locked;
6. write the manifest, then ``plugin.json``, then remove the staging.

``install --frozen`` never resolves and never writes the manifest or the lock: it
fetches MISSING folders by the exact lock key and puts each one in place with a rename
(the lock does not change, so there is nothing to journal), and writes ``plugin.json``.
It does not settle an interrupted run either, since that writes the lock. The API key
only ever goes to the configured registry (``base_url``, ``DECIMAL_BASE_URL``); the
base URL a lock records is informational.

The trust gate reads each resolved skill's safety. ``blocked`` is refused for everyone.
Skills the caller's organization owns pass. ``withheld`` is refused unless the
organization uses the skill, and is then asked about like ``caution`` and
``unreviewed``: on a terminal the user is asked, without one the skill is refused unless
``yes=True``. The question comes only for a version the project has not accepted
before, and never in a frozen install: the lock records that the version was accepted.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from skillevaluation import skillfmt

from . import fs
from . import lock as _lock
from . import manifest as _manifest
from .errors import (
    AlreadyInstalledError,
    BundleContractError,
    ConfirmationRequiredError,
    ExitCode,
    ForeignLockEntriesError,
    IllegalEntryError,
    JournalConflictError,
    LegacyLockError,
    LockHashMismatch,
    LockOutOfDateError,
    ModifiedFolderError,
    OfflineError,
    Result,
    SkillBlockedError,
    SkillNotManagedError,
    SkillpkgError,
    SkillsLockError,
    SkillsNotInstalledError,
    SkillWithheldError,
    UnmanagedFolderError,
    UsageError,
)
from .project import GITIGNORE_LINES, ROOT, Project

if TYPE_CHECKING:  # the client imports httpx; operations that need no network never load it
    from .client import Bundle, RegistryClient, Resolved, ResolveRequest

__all__ = [
    "LOCK_TIMEOUT",
    "Change",
    "Notice",
    "SkillStatus",
    "add",
    "install",
    "remove",
    "status",
    "update",
]

PathLike = Union[str, "os.PathLike[str]"]

#: Seconds to wait for another ``decimalai skills`` run to release the project.
LOCK_TIMEOUT = 30.0

_MANIFEST = ".decimal/skills.toml"
_LOCKFILE = ".decimal/skills.lock"
_OLD = ".old"


# ── What an operation reports ────────────────────────────────────────────────


@dataclass(frozen=True)
class Change:
    """One folder or entry an operation changed (or, with ``dry_run``, would change).

    ``action`` is one of:

    - ``add``: a skill new to the project was installed;
    - ``update``: the folder was replaced by another version or render;
    - ``fetch``: a MISSING folder was written from its lock entry;
    - ``restore``: a MODIFIED folder was replaced by its locked bytes (``force``);
    - ``remove``: the folder and its entries are gone;
    - ``pin``: only the lock's pin changed (``latest`` <-> ``exact``), no file moved;
    - ``listed``: the manifest records it; nothing is installed yet (``no_install``);
    - ``unlisted``: removed from the manifest; it was not installed (``install_name``
      is then the manifest key);
    - ``unchanged``: it was asked for and already installed as asked.

    ``backup`` is where ``force`` saved the folder it replaced, relative to the project.
    """

    action: str
    install_name: str
    skill: Optional[str] = None
    version: Optional[int] = None
    old_version: Optional[int] = None
    pin: Optional[str] = None
    description_changed: bool = False
    format: Optional[int] = None
    old_format: Optional[int] = None
    backup: Optional[str] = None
    dry_run: bool = False

    @property
    def path(self) -> str:
        return f"{ROOT}/{self.install_name}"

    def __str__(self) -> str:
        who = self.skill or self.install_name
        if self.action == "add":
            text = f"added {who} v{self.version} as {self.path}"
        elif self.action == "update":
            if self.old_version is not None and self.old_version != self.version:
                text = f"updated {who} v{self.old_version} -> v{self.version}"
            else:
                text = f"refreshed {who} v{self.version}"
            if self.description_changed:
                text += " (description changed)"
            if self.old_format is not None and self.old_format != self.format:
                text += f" (format {self.old_format} -> {self.format})"
        elif self.action == "fetch":
            text = f"fetched {who} v{self.version} into {self.path}"
        elif self.action == "restore":
            text = f"restored {who} v{self.version} in {self.path}"
        elif self.action == "remove":
            text = f"removed {who} ({self.path})"
        elif self.action == "pin":
            text = f"pinned {who} v{self.version} as {self.pin}"
        elif self.action == "listed":
            text = f"listed {who} in {_MANIFEST}"
        elif self.action == "unlisted":
            text = f"removed {who} from {_MANIFEST}"
        else:
            text = f"{who} v{self.version} is already installed"
        if self.backup:
            text += f"; a copy of the folder as it was is saved in {self.backup}"
        if self.dry_run and self.action != "unchanged":
            return f"would have {text}"
        return text


@dataclass(frozen=True)
class Notice:
    """A warning or a finding that does not stop the operation.

    ``code`` is stable and machine-readable. ``drift`` marks a finding that makes
    ``status(check=True)`` exit 1.
    """

    code: str
    message: str
    skill: Optional[str] = None
    path: Optional[str] = None
    fix: Optional[str] = None
    drift: bool = False
    level: str = "warning"

    def __str__(self) -> str:
        text = f"{self.skill}: {self.message}" if self.skill else self.message
        return f"{text}; {self.fix}" if self.fix else text


@dataclass(frozen=True)
class SkillStatus:
    """One row of :func:`status`: a folder in the root, a lock entry or a manifest entry.

    ``state`` is a folder state (``clean``, ``modified``, ``missing``, ``unmanaged``,
    ``unmanaged_stamped``, ``illegal``), ``interrupted`` for the skill a crashed run was
    changing, or ``not_installed`` for a manifest entry with no lock entry (then
    ``install_name`` is None). ``remote`` holds what the registry said, with
    ``status(remote=True)``.
    """

    install_name: Optional[str]
    state: str
    skill: Optional[str] = None
    version: Optional[int] = None
    pin: Optional[str] = None
    spec: Optional[str] = None
    files: Tuple[str, ...] = ()
    reason: Optional[str] = None
    remote: Tuple[str, ...] = ()


# ── The operations ───────────────────────────────────────────────────────────


def add(
    ref: str,
    *,
    version: Union[int, str, None] = None,
    yes: bool = False,
    force: bool = False,
    no_install: bool = False,
    directory: Optional[PathLike] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    client: Optional["RegistryClient"] = None,
    channel: str = "sdk",
    interactive: Optional[bool] = None,
    confirm: Optional[Callable[[str], bool]] = None,
) -> Result:
    """Add a registry skill to the project and install it: ``decimalai skills add``.

    ``ref`` is a registry name, url_slug, install name or skill id, optionally with
    ``@N`` or ``@latest`` (or pass ``version``). The manifest records the skill's
    canonical registry name with ``"latest"`` or ``"N"``. Adding a skill that is already
    there with another spec changes the spec and moves the folder, as ``cargo add`` does.

    Args:
        yes: accept a skill whose safety review is ``caution`` or ``unreviewed``.
        force: replace an unmanaged folder that already holds the skill, or a
            ``plugin.json`` this engine did not write; each is first copied to
            ``.decimal/backup/<txid>/``.
        no_install: record the manifest entry only; ``install`` fetches it later.
        directory: the project directory (default: found from the working directory).
        api_key, base_url: the registry and key (default: ``DECIMAL_API_KEY`` and
            ``DECIMAL_BASE_URL``); no key means the anonymous rails, public skills only.
        client: a registry client to use instead (it is not closed).
        channel: who is fetching, for the registry's counters (the CLI passes ``cli``).
        interactive: whether a question can be asked (default: stdin is a terminal).
        confirm: asks the question; returns whether the user accepted (default: a
            ``[y/N]`` prompt on the terminal).
    """
    def op(run: _Run) -> None:
        name, spec = _parse_ref(ref, version)
        _add(run, name, spec, yes=yes, force=force, no_install=no_install)

    return _call(op, directory, api_key=api_key, base_url=base_url, client=client,
                 channel=channel, interactive=interactive, confirm=confirm)


def install(
    *,
    frozen: bool = False,
    force: bool = False,
    offline: bool = False,
    yes: bool = False,
    directory: Optional[PathLike] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    client: Optional["RegistryClient"] = None,
    channel: str = "sdk",
    interactive: Optional[bool] = None,
    confirm: Optional[Callable[[str], bool]] = None,
) -> Result:
    """Make the folders and the lock agree with the manifest: ``decimalai skills install``.

    New manifest entries are resolved and added; lock entries the manifest no longer
    lists are removed (CLEAN folders only); MISSING folders are fetched at their locked
    version and render. A pin is never moved (that is :func:`update`), so when every
    folder is CLEAN no network call is made. A spec changed by hand in the manifest is
    resolved at the new spec. Entries an older decimalai merged into the lock are
    dropped (their folders stay, unmanaged), and a stale ``lock_hash`` is recomputed.

    With ``frozen=True`` (CI and Docker) nothing is resolved and the manifest and lock
    are never written: the lock must be a version 2 lock that agrees with the manifest,
    CLEAN folders cost nothing, and MISSING ones are fetched by the exact lock key with
    every byte checked against the lock. UNMANAGED folders are reported, never touched.

    Args:
        force: replace MODIFIED folders with their locked bytes (frozen), or let the
            plan touch them, and replace a ``plugin.json`` this engine did not write;
            each is first copied to ``.decimal/backup/<txid>/``.
        offline: fail (exit 8) instead of reaching the registry.
        yes, interactive, confirm: the trust gate for new entries; see :func:`add`.
        directory, api_key, base_url, client, channel: see :func:`add`.
    """
    def op(run: _Run) -> None:
        if frozen:
            _install_frozen(run, force=force)
        else:
            _sync(run, mode="install", force=force, yes=yes)

    return _call(op, directory, api_key=api_key, base_url=base_url, client=client,
                 channel=channel, interactive=interactive, confirm=confirm, offline=offline,
                 must_exist=True)


def update(
    name: Optional[str] = None,
    *,
    force: bool = False,
    dry_run: bool = False,
    yes: bool = False,
    directory: Optional[PathLike] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    client: Optional["RegistryClient"] = None,
    channel: str = "sdk",
    interactive: Optional[bool] = None,
    confirm: Optional[Callable[[str], bool]] = None,
) -> Result:
    """Move skills to newer versions or renders: ``decimalai skills update``.

    With no ``name``, every ``latest`` entry moves to the current version and render,
    and exact pins stay; new and removed manifest entries are reconciled as
    :func:`install` does. ``name@N`` or ``name@latest`` changes that entry's spec, then
    updates it; ``name`` alone on an exact pin refreshes only its render (the same
    version). ``name`` is an install name or a registry name. Each changed folder is
    replaced whole, so files the new version dropped disappear; folder names never
    change. If any skill cannot be updated, none is.

    Args:
        force: replace MODIFIED folders, and a ``plugin.json`` this engine did not
            write; each is first copied to ``.decimal/backup/<txid>/``, and the path is
            in the result.
        dry_run: plan and report, change nothing (no bundle is fetched).
        yes, interactive, confirm: the trust gate for new versions; see :func:`add`.
        directory, api_key, base_url, client, channel: see :func:`add`.
    """
    def op(run: _Run) -> None:
        target = None
        if name is not None:
            target = _parse_ref(name, None)
        _sync(run, mode="update", target=target, force=force, yes=yes, dry_run=dry_run)

    return _call(op, directory, api_key=api_key, base_url=base_url, client=client,
                 channel=channel, interactive=interactive, confirm=confirm, must_exist=True,
                 take_lock=not dry_run)


def remove(
    name: str,
    *,
    force: bool = False,
    directory: Optional[PathLike] = None,
) -> Result:
    """Remove a skill's folder, lock entry and manifest entry: ``decimalai skills remove``.

    ``name`` is the install name or the registry name. No registry call is made; server
    state (Use links, forks) is never changed.

    Args:
        force: remove a MODIFIED folder too, and replace a ``plugin.json`` this engine
            did not write; each is first copied to ``.decimal/backup/<txid>/``. Files
            placed by hand where the lock lists an unavailable file are always copied
            there before the folder goes.
        directory: see :func:`add`.
    """
    def op(run: _Run) -> None:
        _remove(run, name, force=force)

    return _call(op, directory, must_exist=True, missing_error=lambda: SkillNotManagedError(
        f"is not in `{_MANIFEST}` or `{_LOCKFILE}`", skill=str(name)))


def status(
    *,
    remote: bool = False,
    check: bool = False,
    directory: Optional[PathLike] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    client: Optional["RegistryClient"] = None,
    channel: str = "sdk",
) -> Result:
    """Report what is installed and whether it agrees: ``decimalai skills status``.

    Read-only. ``changes`` holds one :class:`SkillStatus` per skill; ``problems`` holds a
    :class:`Notice` per finding: manifest and lock disagreement, MODIFIED, MISSING and
    ILLEGAL folders, an interrupted run, a legacy lock, a lock that git ignores. With
    ``remote=True`` the registry is asked, and outdated skills, descriptions changed
    upstream and revoked versions are added. No other call reaches the network.

    Exit code 0, except with ``check=True``: 1 when any finding is drift.
    """
    def op(run: _Run) -> None:
        _status(run, remote=remote)
        if check and any(isinstance(n, Notice) and n.drift for n in run.notices):
            run.exit_code = ExitCode.PROBLEMS

    return _call(op, directory, api_key=api_key, base_url=base_url, client=client,
                 channel=channel, take_lock=False)


# ── One call ─────────────────────────────────────────────────────────────────


class _Refused(Exception):
    """Every refusal a plan found; the call returns them all."""

    def __init__(self, errors: Sequence[SkillpkgError]) -> None:
        super().__init__(errors[0].message)
        self.errors = list(errors)


class _Run:
    """One operation: the project, the registry client (made on first use), the report."""

    def __init__(self, project: Project, *, client: Optional["RegistryClient"],
                 api_key: Optional[str], base_url: Optional[str], channel: str,
                 interactive: Optional[bool], confirm: Optional[Callable[[str], bool]],
                 offline: bool) -> None:
        self.project = project
        self._client = client
        self._owned = False
        self._api_key = api_key
        self._base_url = base_url
        self._channel = channel
        self.interactive = interactive
        self.confirm = confirm
        self.offline = offline
        self.changes: List[Any] = []
        self.notices: List[Notice] = []
        self.exit_code = ExitCode.OK
        self.plock: Optional[fs.ProcessLock] = None
        #: ``force`` was given and ``plugin.json`` is the user's: back it up, then replace it.
        self.plugin_backup = False

    def client(self, *, skill: Optional[str] = None,
               why: str = "this needs the registry") -> "RegistryClient":
        if self.offline:
            raise OfflineError(f"{why}, and offline mode is on", skill=skill)
        if self._client is None:
            from .client import RegistryClient

            self._client = RegistryClient.from_env(
                api_key=self._api_key, base_url=self._base_url, channel=self._channel)
            self._owned = True
        return self._client

    def note(self, code: str, message: str, **kwargs: Any) -> None:
        self.notices.append(Notice(code, message, **kwargs))

    def refresh(self) -> None:
        """Keep the hold on the project fresh; stop if another run took it over."""
        if self.plock is not None:
            self.plock.refresh()

    def close(self) -> None:
        if self._owned and self._client is not None:
            self._client.close()

    def result(self, errors: Sequence[SkillpkgError] = ()) -> Result:
        if errors:
            return Result(self.changes, [*self.notices, *errors], errors[0].exit_code)
        return Result(self.changes, self.notices, self.exit_code)


def _call(
    op: Callable[[_Run], None],
    directory: Optional[PathLike],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    client: Optional["RegistryClient"] = None,
    channel: str = "sdk",
    interactive: Optional[bool] = None,
    confirm: Optional[Callable[[str], bool]] = None,
    offline: bool = False,
    must_exist: bool = False,
    take_lock: bool = True,
    missing_error: Optional[Callable[[], SkillpkgError]] = None,
) -> Result:
    run: Optional[_Run] = None
    try:
        project = _project(directory)
        run = _Run(project, client=client, api_key=api_key, base_url=base_url, channel=channel,
                   interactive=interactive, confirm=confirm, offline=offline)
        if must_exist and not _has_state(project):
            # Nothing to work on: say so without creating `.decimal/` for the run lock.
            if missing_error is not None:
                raise missing_error()
            raise SkillsNotInstalledError(
                f"`{_MANIFEST}` does not exist in {project.directory}",
                path=project.manifest_path)
        if take_lock:
            with fs.ProcessLock(project, timeout=LOCK_TIMEOUT) as plock:
                run.plock = plock
                try:
                    op(run)
                finally:
                    run.plock = None
        else:
            op(run)
    except _Refused as stop:
        return _result(run, stop.errors)
    except SkillpkgError as exc:
        return _result(run, [exc])
    except OSError as exc:
        return _result(run, [_os_error(exc)])
    finally:
        if run is not None:
            run.close()
    return run.result()


def _os_error(exc: OSError) -> SkillpkgError:
    """A failure of the filesystem itself, reported like any refusal (exit 1).

    What the engine can tell in advance (an unreadable skill folder, a symlink, local
    edits) is refused with its own error first; this covers the rest, such as a
    permission the project's own `.decimal/` lacks or a full disk.
    """
    where = f": {exc.filename}" if exc.filename else ""
    return SkillpkgError(
        f"the filesystem refused the operation ({exc.strerror or exc}{where})",
        path=exc.filename if isinstance(exc.filename, str) else None,
        fix="check the permissions of the project's files and the free disk space, then "
            "run the command again; `decimalai skills status` shows what is installed",
        details={"errno": exc.errno} if exc.errno is not None else None)


def _result(run: Optional[_Run], errors: Sequence[SkillpkgError]) -> Result:
    if run is None:
        return Result((), errors, errors[0].exit_code)
    return run.result(errors)


def _project(directory: Optional[PathLike]) -> Project:
    if directory is None:
        return Project.find()
    project = Project.at(directory)
    if not project.directory.is_dir():
        raise UsageError(f"{project.directory} is not a directory",
                         fix="pass the project directory with `--directory`")
    return project


def _has_state(project: Project) -> bool:
    return any(os.path.lexists(p) for p in (
        project.manifest_path, project.lock_path, project.npx_lock_path, project.journal_path))


def _settle(run: _Run) -> None:
    """Finish what a crashed run left: its half-done swap, then its staging."""
    recovery = fs.recover(run.project)
    if recovery is not None:
        run.note("recovered", f"settled the change an interrupted run left in "
                              f"`{ROOT}/{recovery.name}`: the {recovery.outcome} version stands",
                 skill=recovery.name, level="info")
    fs.discard_staging(run.project)


def _check_plugin_json(run: _Run, *, force: bool) -> None:
    """Refuse a ``plugin.json`` this engine did not write before anything changes; with
    ``force``, remember to back it up before it is replaced."""
    if _lock.plugin_json_is_ours(run.project):
        return
    if not force:
        _lock.check_plugin_json(run.project)
    run.plugin_backup = True


def _write_plugin_json(run: _Run, lock_hash: str, txid: str) -> None:
    project = run.project
    if run.plugin_backup and os.path.lexists(project.plugin_json_path):
        backup = project.backup_dir / txid / "plugin.json"
        project.makedirs(backup.parent)
        shutil.copy2(project.plugin_json_path, backup)
        rel = project.rel(backup)
        run.note("backup", f"`{project.rel(project.plugin_json_path)}` was not the one this "
                           f"engine writes; it is saved in `{rel}`", path=rel)
    _lock.write_plugin_json(project, lock_hash, replace_foreign=run.plugin_backup)


# ── References ───────────────────────────────────────────────────────────────


def _parse_ref(ref: Any, version: Union[int, str, None]) -> Tuple[str, Optional[str]]:
    """``(reference, spec)`` from ``ref[@N|@latest]``; spec None when none was given."""
    if not isinstance(ref, str) or not ref.strip():
        raise UsageError(f"a skill reference must be a non-empty string, not {ref!r}")
    ref = ref.strip()
    spec: Optional[str] = None
    base, at, tail = ref.rpartition("@")
    if at and base and "/" not in tail:
        spec = _manifest.parse_spec(tail)
        ref = base
    if version is not None:
        wanted = _manifest.parse_spec(version)
        if spec is not None and spec != wanted:
            raise UsageError(f"{ref!r} asks for version {spec} and {wanted}; give one")
        spec = wanted
    return ref, spec


def _is_id_ref(ref: str, resolved: "Resolved") -> bool:
    return ref == resolved.skill_id


# ── Reading the project ──────────────────────────────────────────────────────


@dataclass
class _Project:
    """The manifest, the lock and every folder's state, read under the run lock."""

    manifest: _manifest.Manifest
    lock: _lock.Lock
    states: Dict[str, fs.FolderState]
    claims: Dict[str, Optional[str]] = field(default_factory=dict)   # manifest key -> install name
    unclaimed: List[str] = field(default_factory=list)                # locked, not in the manifest

    def key_of(self, install_name: str) -> Optional[str]:
        for key, name in self.claims.items():
            if name == install_name:
                return key
        return None

    def state(self, name: str) -> fs.State:
        st = self.states.get(name)
        return st.state if st is not None else fs.State.MISSING


def _match(manifest: _manifest.Manifest,
           lock: _lock.Lock) -> Tuple[Dict[str, Optional[str]], List[str]]:
    """Which lock entry each manifest entry is: by skill id when the manifest has one,
    else by registry name. Returns the claims and the lock entries no entry claims."""
    claims: Dict[str, Optional[str]] = {}
    claimed: Dict[str, str] = {}
    by_id = {entry["skill_id"]: name for name, entry in lock.skills.items()}
    ordered = sorted(manifest.entries.items(), key=lambda kv: kv[1].id is None)
    for key, entry in ordered:
        found: Optional[str] = None
        if entry.id is not None:
            found = by_id.get(entry.id)
        else:
            for name, locked in lock.skills.items():
                if locked.get("name") == key and name not in claimed:
                    found = name
                    break
        if found is not None and found in claimed:
            found = None
        claims[key] = found
        if found is not None:
            claimed[found] = key
    claims = {key: claims[key] for key in manifest}
    return claims, [name for name in lock.skills if name not in claimed]


def _read(run: _Run, *, repair: bool, manifest_missing_ok: bool,
          lock_missing_ok: bool = True) -> _Project:
    project = run.project
    # The lock first: a project with a version 1 lock and no manifest needs `migrate`.
    lock = _lock.load(project, missing_ok=lock_missing_ok, repair=repair)
    manifest = _manifest.load(project, missing_ok=manifest_missing_ok)
    if lock.dropped:
        for name in lock.dropped:
            run.note("dropped_entry",
                     f"the lock entry `{name}` was written by an older decimalai; it was "
                     f"dropped, and `{ROOT}/{name}` (if present) is now an unmanaged folder",
                     skill=name)
    if not lock.hash_ok:
        run.note("lock_hash", f"`{_LOCKFILE}` had a lock_hash that did not match its "
                              "entries (a hand edit?); it was recomputed",
                 fix=f"review and commit `{_LOCKFILE}`")
    states = fs.scan_root(project, lock.skills)
    claims, unclaimed = _match(manifest, lock)
    return _Project(manifest, lock, states, claims, unclaimed)


def _taken(run: _Run, ctx: _Project) -> Dict[str, Optional[str]]:
    """Install name -> skill id for every name the project already uses.

    A locked name maps to its skill. An unmanaged folder maps to None, unless it is an
    earlier install of a skill left behind (stamped with the skill id, its ``name:``
    still its folder's name): then it maps to that skill, so the registry hands that
    name back and the operation asks before replacing the copy, instead of installing
    a second copy beside it.
    """
    taken: Dict[str, Optional[str]] = {}
    for name, st in ctx.states.items():
        entry = ctx.lock.skills.get(name)
        if entry is not None:
            taken[name] = entry["skill_id"]
        elif st.state is fs.State.UNMANAGED_STAMPED and _names_itself(run.project, name):
            taken[name] = st.skill_id
        else:
            taken[name] = None
    return taken


def _names_itself(project: Project, name: str) -> bool:
    try:
        text = (project.skill_dir(name) / "SKILL.md").read_bytes().decode("utf-8")
        fm, _ = skillfmt.parse_skill_md_text(text)
    except (OSError, UnicodeDecodeError, skillfmt.SkillFmtError):
        return False
    return fm.get("name") == name


# ── Plans ────────────────────────────────────────────────────────────────────


@dataclass
class _Step:
    """What one install name becomes."""

    action: str                                   # add | update | fetch | restore | remove
    name: str                                     # the install name, also the folder
    skill: str                                    # the registry name, for messages
    old: Optional[Mapping[str, Any]]              # its lock entry before, if any
    new: Optional[Dict[str, Any]] = None          # its lock entry after (None: removed)
    resolved: Optional["Resolved"] = None         # fetch this rendering (else: the lock key)
    pin: str = "latest"
    backup: bool = False
    contents: Optional[Dict[str, bytes]] = None
    backup_path: Optional[str] = None

    def change(self, *, dry_run: bool = False) -> Change:
        new = self.new or {}
        old = self.old or {}
        version = new.get("version")
        if version is None and self.resolved is not None:
            version = self.resolved.version
        new_render = new.get("render") or (self.resolved.render.to_dict() if self.resolved else {})
        new_desc = new.get("description_sha256") or (
            self.resolved.description_sha256 if self.resolved else None)
        return Change(
            action=self.action,
            install_name=self.name,
            skill=self.skill,
            version=version if self.action != "remove" else old.get("version"),
            old_version=old.get("version") if self.action != "remove" else None,
            pin=self.pin if self.action != "remove" else old.get("pin"),
            description_changed=bool(old and new_desc and self.action == "update"
                                     and old.get("description_sha256") != new_desc),
            format=new_render.get("format") if self.action != "remove" else None,
            old_format=(old.get("render") or {}).get("format") if old else None,
            backup=self.backup_path,
            dry_run=dry_run,
        )


@dataclass
class _Plan:
    steps: List[_Step] = field(default_factory=list)
    lock_only: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # name -> new entry
    reported: List[Change] = field(default_factory=list)                 # pin, unchanged, listed
    errors: List[SkillpkgError] = field(default_factory=list)


def _satisfied(entry: _manifest.ManifestEntry, locked: Mapping[str, Any]) -> bool:
    """Whether the locked version is one the manifest's spec allows (pin aside)."""
    return entry.latest or locked.get("version") == entry.version


def _modified_error(ctx: _Project, name: str, skill: str) -> ModifiedFolderError:
    st = ctx.states[name]
    return ModifiedFolderError(
        f"`{ROOT}/{name}` has local changes ({', '.join(st.files) or 'its files'})",
        skill=skill, path=f"{ROOT}/{name}",
        details={"files": list(st.files), "edited": list(st.edited),
                 "missing": list(st.missing), "extra": list(st.extra), "mode": list(st.mode)})


def _illegal_error(ctx: _Project, name: str, skill: Optional[str]) -> IllegalEntryError:
    st = ctx.states.get(name)
    reason = (st.reason if st is not None else None) or "not a plain folder"
    fix = None
    if reason.startswith("unreadable:"):
        fix = (f"make it readable (`chmod -R u+rwX {ROOT}/{name}`), or move it out of "
               f"`{ROOT}/`, then run the command again")
    return IllegalEntryError(f"`{ROOT}/{name}`: {reason}", skill=skill or name,
                             path=f"{ROOT}/{name}", fix=fix)


def _check_targets(ctx: _Project, plan: _Plan, *, force: bool) -> None:
    """Refuse a step whose folder holds local state, unless ``force`` (then back it up)."""
    for step in plan.steps:
        state = ctx.state(step.name)
        if state is fs.State.ILLEGAL:
            plan.errors.append(_illegal_error(ctx, step.name, step.skill))
        elif state is fs.State.MODIFIED:
            if force:
                step.backup = True
            else:
                plan.errors.append(_modified_error(ctx, step.name, step.skill))
        elif state in (fs.State.UNMANAGED, fs.State.UNMANAGED_STAMPED):
            if force:
                step.backup = True
            else:
                plan.errors.append(UnmanagedFolderError(
                    f"`{ROOT}/{step.name}` already exists and is not in `{_LOCKFILE}`",
                    skill=step.skill, path=f"{ROOT}/{step.name}"))


def _gate(run: _Run, resolved: "Resolved", *, yes: bool, new_version: bool = True) -> None:
    """The trust gate for one resolved skill. Raises its refusal.

    The verdict is :func:`~decimalai.skillpkg.client.trust_verdict`, which the
    migration shares. Blocked and withheld are checked every time; the question is
    asked only for a version the project has not accepted before (``new_version``),
    and never for a skill the caller's organization owns.
    """
    from . import client

    name = resolved.registry_name
    refusal, reason = client.trust_verdict(resolved)
    if refusal == "blocked":
        raise SkillBlockedError(f"version {resolved.version} failed its safety review and is "
                                "blocked", skill=name)
    if refusal == "withheld":
        raise SkillWithheldError("the registry withholds it from installs", skill=name)
    if reason is None or resolved.relationship == "owned":
        return
    if yes or not new_version:
        return
    interactive = run.interactive if run.interactive is not None else _stdin_is_tty()
    if not interactive:
        raise ConfirmationRequiredError(
            f"its safety review is {reason}, and there is no terminal to confirm it", skill=name)
    confirm = run.confirm or _ask
    question = (f"{name} v{resolved.version} is {reason}: its safety review has not cleared "
                "it. Install it?")
    if not confirm(question):
        raise ConfirmationRequiredError(f"its safety review is {reason}, and it was declined",
                                        skill=name)


def _stdin_is_tty() -> bool:
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError, OSError):
        return False


def _ask(question: str) -> bool:
    sys.stderr.write(f"{question} [y/N] ")
    sys.stderr.flush()
    try:
        answer = sys.stdin.readline()
    except (OSError, ValueError):
        return False
    return answer.strip().lower() in ("y", "yes")


def _relabel(error: SkillpkgError, skill: str) -> SkillpkgError:
    """``error`` naming ``skill`` (a registry name) instead of the id it was asked by.

    A usage error is kept as it is: it names the reference as given, escaped so that it
    prints, and its fix says which one to correct.
    """
    if error.skill == skill or isinstance(error, UsageError):
        return error
    return type(error)(error.message, skill=skill, path=error.path, details=error.details)


def _accept(run: _Run, ctx: _Project, plan: _Plan, key: str, entry: _manifest.ManifestEntry,
            resolved: "Resolved", locked_name: Optional[str], *, yes: bool, force: bool,
            ref: Optional[str] = None, report_unchanged: bool = True) -> None:
    """Turn one resolved manifest entry into a step (or a lock-only change, or nothing).

    The manifest key becomes the skill's canonical registry name. A folder that keeps
    its rendering keeps the name it was locked with, the one its SKILL.md is stamped
    with; a folder that is fetched again takes the registry's current name.
    """
    other = ctx.lock.by_skill_id(resolved.skill_id)
    if other is not None and other[0] != resolved.install_name:
        raise AlreadyInstalledError(
            f"is already installed as `{ROOT}/{other[0]}`; the registry chose "
            f"`{resolved.install_name}`", skill=resolved.registry_name)
    if locked_name is None and other is not None:
        locked_name = other[0]
    old = ctx.lock.skills.get(locked_name) if locked_name else None
    same = (old is not None and old.get("version_id") == resolved.version_id
            and old.get("render") == resolved.render.to_dict())
    _gate(run, resolved, yes=yes,
          new_version=old is None or old.get("version_id") != resolved.version_id)

    canonical = str(old["name"]) if (same and old is not None) else resolved.registry_name
    if canonical != key and canonical in ctx.manifest:
        raise AlreadyInstalledError(
            f"`{_MANIFEST}` lists it twice, as {key!r} and as {canonical!r}", skill=canonical,
            fix=f"delete one of the two lines of `{_MANIFEST}`, then run the command again")
    claimed = ctx.claims.get(canonical)
    if claimed is not None and claimed != resolved.install_name:
        raise AlreadyInstalledError(
            f"`{_MANIFEST}` already lists {canonical!r}, for the skill installed as "
            f"`{ROOT}/{claimed}`", skill=canonical)
    record_id = entry.id
    if ref is not None and _is_id_ref(ref, resolved):
        record_id = resolved.skill_id
    if canonical != key:
        ctx.manifest.remove(key)
        ctx.claims.pop(key, None)
    if ctx.manifest.get(canonical) != _manifest.ManifestEntry(canonical, entry.spec, record_id):
        ctx.manifest.set(canonical, entry.spec, id=record_id)
    ctx.claims[canonical] = resolved.install_name

    name = resolved.install_name
    if not same:
        plan.steps.append(_Step("update" if old is not None else "add", name, canonical, old,
                                resolved=resolved, pin=entry.pin))
        return
    assert old is not None
    _hold(run, ctx, plan, name, canonical, old, pin=entry.pin, force=force,
          report_unchanged=report_unchanged)


def _resolve(run: _Run, ctx: _Project, plan: _Plan,
             items: Sequence[Tuple[str, "ResolveRequest", Optional[str]]],
             *, purpose: str) -> List[Tuple[str, "Resolved", Optional[str]]]:
    """Resolve ``(key, request, locked install name)`` items in one batch; refusals go to
    ``plan.errors``, named by registry name."""
    if not items:
        return []
    client = run.client(skill=items[0][0] if len(items) == 1 else None,
                        why="resolving a skill needs the registry")
    results = client.resolve([request for _, request, _ in items],
                             taken=_taken(run, ctx), purpose=purpose)
    out = []
    for (key, _, locked_name), result in zip(items, results):
        if isinstance(result, SkillpkgError):
            plan.errors.append(_relabel(result, key))
        else:
            out.append((key, result, locked_name))
    return out


def _request(key: str, entry: _manifest.ManifestEntry, locked: Optional[Mapping[str, Any]],
             spec: Optional[str] = None) -> "ResolveRequest":
    from .client import ResolveRequest

    version = spec or entry.spec
    if locked is not None:
        return ResolveRequest(skill_id=locked["skill_id"], version=version)
    if entry.id is not None:
        return ResolveRequest(skill_id=entry.id, version=version)
    return ResolveRequest(ref=key, version=version)


# ── add ──────────────────────────────────────────────────────────────────────


def _add(run: _Run, ref: str, spec: Optional[str], *, yes: bool, force: bool,
         no_install: bool) -> None:
    from .client import ResolveRequest

    _settle(run)
    ctx = _read(run, repair=False, manifest_missing_ok=True)
    if not no_install:
        _check_plugin_json(run, force=force)
    plan = _Plan()
    client = run.client(skill=ref, why="adding a skill needs the registry")
    resolved = client.resolve_one(ResolveRequest(ref=ref, version=spec or "latest"),
                                  taken=_taken(run, ctx), purpose="add")
    key = resolved.registry_name
    hit = ctx.lock.by_skill_id(resolved.skill_id)
    locked_name = hit[0] if hit else None
    if locked_name is not None:
        key = ctx.key_of(locked_name) or key
    existing = ctx.manifest.get(key)
    entry = _manifest.ManifestEntry(key, spec or "latest", existing.id if existing else None)

    if no_install:
        _gate(run, resolved, yes=yes)
        canonical = resolved.registry_name
        if canonical != key:
            ctx.manifest.remove(key)
        record_id = resolved.skill_id if _is_id_ref(ref, resolved) else entry.id
        ctx.manifest.set(canonical, entry.spec, id=record_id)
        _manifest.save(run.project, ctx.manifest)
        _ensure_gitignore(run.project)
        run.changes.append(Change("listed", resolved.install_name, canonical,
                                  version=resolved.version, pin=entry.pin))
        return

    _accept(run, ctx, plan, key, entry, resolved, locked_name, yes=yes, force=force, ref=ref)
    _execute(run, ctx, plan, purpose="add", force=force)


# ── install and update ───────────────────────────────────────────────────────


def _find(ctx: _Project, name: str) -> Tuple[Optional[str], Optional[str]]:
    """``(manifest key, install name)`` for an install name or a registry name."""
    if name in ctx.lock.skills:
        return ctx.key_of(name), name
    if name in ctx.manifest:
        return name, ctx.claims.get(name)
    for install_name, entry in ctx.lock.skills.items():
        if entry.get("name") == name:
            return ctx.key_of(install_name), install_name
    return None, None


def _sync(run: _Run, *, mode: str, force: bool, yes: bool, dry_run: bool = False,
          target: Optional[Tuple[str, Optional[str]]] = None) -> None:
    """``install`` (mode ``install``) and ``update`` (mode ``update``): one planner.

    Both reconcile the manifest and the lock. ``update`` also re-resolves the entries it
    refreshes: every ``latest`` entry, or only ``target`` when one is named.
    """
    if dry_run:
        if fs.read_journal(run.project) is not None:
            run.note("interrupted", "an interrupted run left a change to settle; the next "
                                    "command that writes settles it first", level="info")
    else:
        _settle(run)
    ctx = _read(run, repair=(mode == "install"), manifest_missing_ok=False)
    _check_plugin_json(run, force=force)
    plan = _Plan()

    only: Optional[str] = None
    if target is not None:
        name, spec = target
        key, install_name = _find(ctx, name)
        if key is None:
            raise SkillNotManagedError(f"is not in `{_MANIFEST}`", skill=name)
        if spec is not None and ctx.manifest[key].spec != spec:
            current = ctx.manifest[key]
            ctx.manifest.set(key, spec, id=current.id)
        only = key

    to_resolve: List[Tuple[str, "ResolveRequest", Optional[str]]] = []
    for key, entry in list(ctx.manifest.entries.items()):
        if only is not None and key != only:
            continue
        name = ctx.claims.get(key)
        if name is None:
            to_resolve.append((key, _request(key, entry, None), None))
            continue
        locked = ctx.lock.skills[name]
        refresh = mode == "update" and (only is not None or entry.latest)
        if refresh or not _satisfied(entry, locked):
            to_resolve.append((key, _request(key, entry, locked), name))
            continue
        _hold(run, ctx, plan, name, str(locked.get("name") or key), locked, pin=entry.pin,
              force=force)

    if only is None:
        for name in ctx.unclaimed:
            locked = ctx.lock.skills[name]
            plan.steps.append(_Step("remove", name, str(locked.get("name") or name), locked))

    purpose = "update" if mode == "update" else "install"
    for key, resolved, locked_name in _resolve(run, ctx, plan, to_resolve, purpose=purpose):
        entry = ctx.manifest[key]
        try:
            _accept(run, ctx, plan, key, entry, resolved, locked_name, yes=yes, force=force,
                    report_unchanged=only is not None)
        except SkillpkgError as exc:
            plan.errors.append(exc)
    if plan.errors:
        raise _Refused(plan.errors)
    _execute(run, ctx, plan, purpose=purpose, force=force, dry_run=dry_run, write_lock=True)


def _hold(run: _Run, ctx: _Project, plan: _Plan, name: str, skill: str,
          locked: Mapping[str, Any], *, pin: str, force: bool,
          report_unchanged: bool = False) -> None:
    """A locked skill that stays at its version and render.

    Its pin follows the manifest (a lock-only change). A MISSING folder is fetched by
    its lock key. A MODIFIED folder is reported and left alone, or with ``force``
    restored to the locked bytes; an ILLEGAL one is refused.
    """
    new = dict(locked, pin=pin)
    state = ctx.state(name)
    if state is fs.State.MISSING:
        plan.steps.append(_Step("fetch", name, skill, locked, new=new, pin=pin))
        return
    if state is fs.State.ILLEGAL:
        plan.errors.append(_illegal_error(ctx, name, skill))
        return
    if state is fs.State.MODIFIED:
        if force:
            plan.steps.append(_Step("restore", name, skill, locked, new=new, pin=pin))
            return
        st = ctx.states[name]
        run.note("modified", f"`{ROOT}/{name}` has local changes ({', '.join(st.files)}); "
                             "it was left as it is",
                 skill=skill, path=f"{ROOT}/{name}",
                 fix="run `decimalai skills install --force` to restore the locked files")
    if new != dict(locked):
        plan.lock_only[name] = new
        plan.reported.append(_Step("pin", name, skill, locked, new=new, pin=pin).change())
    elif report_unchanged:
        plan.reported.append(_Step("unchanged", name, skill, locked, new=new, pin=pin).change())


# ── remove ───────────────────────────────────────────────────────────────────


def _remove(run: _Run, name: str, *, force: bool) -> None:
    if not isinstance(name, str) or not name.strip():
        raise UsageError(f"a skill name must be a non-empty string, not {name!r}")
    name = name.strip()
    _settle(run)
    ctx = _read(run, repair=False, manifest_missing_ok=True)
    key, install_name = _find(ctx, name)
    if key is None and install_name is None:
        raise SkillNotManagedError(f"is not in `{_MANIFEST}` or `{_LOCKFILE}`", skill=name)
    _check_plugin_json(run, force=force)
    plan = _Plan()
    if key is not None:
        ctx.manifest.remove(key)
    if install_name is not None:
        locked = ctx.lock.skills[install_name]
        plan.steps.append(_Step("remove", install_name, str(locked.get("name") or name), locked))
    elif key is not None:
        plan.reported.append(Change("unlisted", key, key))
    _execute(run, ctx, plan, purpose="install", force=force)


# ── fetch, stage, apply ──────────────────────────────────────────────────────


def _execute(run: _Run, ctx: _Project, plan: _Plan, *, purpose: str, force: bool,
             dry_run: bool = False, write_lock: bool = False) -> None:
    """Steps 3 to 6 of the module docstring for a non-frozen plan.

    ``write_lock`` writes the lock even when the project had none and still has no
    skill, so that ``install --frozen`` finds one after ``install``.
    """
    _check_targets(ctx, plan, force=force)
    if plan.errors:
        raise _Refused(plan.errors)
    plan.steps.sort(key=lambda s: (s.action != "remove", s.name))
    if dry_run:
        run.changes += [step.change(dry_run=True) for step in plan.steps]
        run.changes += [replace(c, dry_run=True) for c in plan.reported]
        return

    project = run.project
    txid = fs.new_txid()
    try:
        _fetch_all(run, plan, purpose=purpose)
        for step in plan.steps:
            if step.new is not None:
                run.refresh()
                fs.stage(project, txid, step.name, step.contents or {}, step.new)
    except BaseException:
        with contextlib.suppress(SkillpkgError, OSError):
            fs.discard_staging(project, txid)
        raise

    lock = ctx.lock
    for name, entry in sorted(plan.lock_only.items()):
        lock = lock.with_entry(name, entry)
    if plan.steps:
        project.makedirs(project.skills_root)
    for step in plan.steps:
        run.refresh()
        if step.backup and os.path.lexists(project.skill_dir(step.name)):
            step.backup_path = project.rel(fs.backup_folder(project, txid, step.name))
            run.note("backup", f"`{ROOT}/{step.name}` had local changes; they are saved in "
                               f"`{step.backup_path}`", skill=step.skill, path=step.backup_path)
        lock = fs.swap(project, txid, step.name, step.new, lock,
                       on_warning=lambda message: run.note("hand_placed", message))
        run.changes.append(step.change())
    run.changes += plan.reported
    _finish(run, ctx, lock, txid, write_lock=write_lock)


def _fetch_all(run: _Run, plan: _Plan, *, purpose: str) -> None:
    """Fetch every step's bundle and build its lock entry. Stops at the first refusal."""
    for step in plan.steps:
        if step.action == "remove":
            continue
        run.refresh()
        client = run.client(skill=step.skill, why=f"`{ROOT}/{step.name}` must be fetched")
        if step.resolved is not None:
            bundle = client.fetch_resolved(step.resolved, purpose=purpose)
            step.new = _new_entry(client, step, bundle)
            for note in dict.fromkeys((*step.resolved.policy_notes, *bundle.policy_notes)):
                run.note("policy", note, skill=step.skill, level="info")
            for item in bundle.unavailable:
                run.note("unavailable", f"`{item.path}` is not installed: {item.reason}",
                         skill=step.skill)
        else:
            bundle = _fetch_locked(run, client, step, purpose=purpose)
        step.contents = {f.path: f.content for f in bundle.files}


def _fetch_locked(run: _Run, client: "RegistryClient", step: _Step, *, purpose: str) -> "Bundle":
    """The bundle of exactly what the lock records for ``step``, every byte checked."""
    entry = step.old
    assert entry is not None
    render = entry["render"]
    locked_url = str((entry.get("source") or {}).get("base_url") or "").rstrip("/")
    if locked_url and locked_url != client.base_url:
        run.note("base_url", f"it was locked from {locked_url}; it is fetched from the "
                             f"configured registry {client.base_url}",
                 skill=step.skill, level="info")
    return client.fetch_bundle(
        entry["version_id"], install_name=step.name, purpose=purpose,
        render_hash=render["hash"], format=render["format"], variant=render["variant"],
        expect_files=entry["files"], expect_executable=entry["executable"],
        expected_from="the lock", skill=step.skill, version=entry["version"])


def _new_entry(client: "RegistryClient", step: _Step, bundle: "Bundle") -> Dict[str, Any]:
    resolved = step.resolved
    assert resolved is not None
    invocation = bundle.invocation or resolved.invocation
    try:
        entry = _lock.make_entry(
            resolved.install_name,
            name=resolved.registry_name,
            skill_id=resolved.skill_id,
            version_id=resolved.version_id,
            version=resolved.version,
            render=resolved.render.to_dict(),
            body_sha256=resolved.body_sha256,
            description_sha256=resolved.description_sha256,
            files=bundle.file_hashes,
            executable=bundle.executable,
            unavailable=[{"path": u.path, "reason": u.reason} for u in bundle.unavailable],
            source={"base_url": client.base_url, "org": resolved.org_id},
            pin=step.pin,
            invocation="user" if invocation == "user" else "model",
        )
    except SkillsLockError as exc:
        raise BundleContractError(f"the registry's answer cannot be locked: {exc.message}",
                                  skill=resolved.registry_name) from None
    return _lock.keep_installed_at(entry, step.old)


def _finish(run: _Run, ctx: _Project, lock: _lock.Lock, txid: Optional[str], *,
            write_lock: bool = False) -> None:
    project = run.project
    has_lock = write_lock or bool(lock.skills) or project.lock_path.exists()
    if has_lock:
        _lock.save(project, lock)
    _manifest.save(project, ctx.manifest)
    if has_lock:
        _write_plugin_json(run, lock.lock_hash, txid or fs.new_txid())
        project.makedirs(project.skills_root)
    _ensure_gitignore(project)
    if txid is not None:
        fs.discard_staging(project, txid)


def _ensure_gitignore(project: Project) -> bool:
    """Append the engine's lines to ``.decimal/.gitignore`` that it lacks. Idempotent."""
    path = project.state_dir / ".gitignore"
    project.refuse_symlinks(path)
    try:
        text = path.read_bytes().decode("utf-8")
    except FileNotFoundError:
        text = ""
    except (OSError, UnicodeDecodeError):
        return False
    have = {line.strip() for line in text.splitlines()}
    missing = [line for line in GITIGNORE_LINES if line not in have]
    if not missing:
        return False
    if text and not text.endswith("\n"):
        text += "\n"
    project.write_file(path, (text + "".join(f"{line}\n" for line in missing)).encode("utf-8"))
    return True


# ── install --frozen ─────────────────────────────────────────────────────────


def _install_frozen(run: _Run, *, force: bool) -> None:
    project = run.project
    if fs.read_journal(project) is not None:
        raise JournalConflictError(
            "an interrupted run left a change to settle, and a frozen install never writes "
            "the lock", path=project.journal_path,
            fix="run `decimalai skills install` (without --frozen) to settle it, then commit "
                f"`{_LOCKFILE}` if it changed")
    fs.discard_staging(project)
    lock = _lock.load(project)
    manifest = _manifest.load(project)
    claims, unclaimed = _match(manifest, lock)
    _check_agreement(manifest, lock, claims, unclaimed)
    states = fs.scan_root(project, lock.skills)
    ctx = _Project(manifest, lock, states, claims, unclaimed)

    plan = _Plan()
    for name, entry in lock.skills.items():
        skill = str(entry.get("name") or name)
        state = ctx.state(name)
        if state is fs.State.CLEAN:
            continue
        if state is fs.State.MISSING:
            plan.steps.append(_Step("fetch", name, skill, entry, new=dict(entry), pin=entry["pin"]))
        elif state is fs.State.MODIFIED and force:
            plan.steps.append(_Step("restore", name, skill, entry, new=dict(entry),
                                    pin=entry["pin"], backup=True))
        elif state is fs.State.MODIFIED:
            plan.errors.append(_modified_error(ctx, name, skill))
        else:
            plan.errors.append(_illegal_error(ctx, name, skill))
    for name, st in states.items():
        if name in lock.skills:
            continue
        if st.state is fs.State.ILLEGAL:
            run.note("illegal", f"`{ROOT}/{name}` is not a plain folder ({st.reason}); it was "
                                "left as it is", skill=name, path=f"{ROOT}/{name}")
        else:
            run.note("unmanaged", f"`{ROOT}/{name}` is not in `{_LOCKFILE}`; it was left as "
                                  "it is", skill=name, path=f"{ROOT}/{name}", level="info")
    if plan.errors:
        raise _Refused(plan.errors)
    _check_plugin_json(run, force=force)
    if run.offline and plan.steps:
        names = ", ".join(f"`{ROOT}/{s.name}`" for s in plan.steps)
        raise OfflineError(f"{names} must be fetched, and offline mode is on")

    txid = fs.new_txid()
    try:
        _fetch_all(run, plan, purpose="frozen")
        for step in plan.steps:
            run.refresh()
            fs.stage(project, txid, step.name, step.contents or {}, step.old or {})
    except BaseException:
        with contextlib.suppress(SkillpkgError, OSError):
            fs.discard_staging(project, txid)
        raise

    if plan.steps:
        project.makedirs(project.skills_root)
    for step in plan.steps:
        run.refresh()
        if step.backup:
            step.backup_path = project.rel(fs.backup_folder(project, txid, step.name))
            run.note("backup", f"`{ROOT}/{step.name}` had local changes; they are saved in "
                               f"`{step.backup_path}`", skill=step.skill, path=step.backup_path)
        _place(project, txid, step)
        run.changes.append(step.change())
    _write_plugin_json(run, lock.lock_hash, txid)
    project.makedirs(project.skills_root)
    fs.discard_staging(project, txid)


def _place(project: Project, txid: str, step: _Step) -> None:
    """Put a staged folder at ``<root>/<name>`` without touching the lock.

    A frozen install never changes the lock, so it needs no journal: a crash leaves the
    old folder, no folder (MISSING, fetched again by the next run) or the new one. A
    replaced MODIFIED folder was copied to the backup first. Files the user placed where
    the lock lists an unavailable file are carried into the new folder.
    """
    target = project.skill_dir(step.name)
    staging = project.staging_dir / txid
    staged = staging / step.name
    project.refuse_symlinks(target)
    if os.path.lexists(target):
        unavailable = [u.get("path") for u in (step.old or {}).get("unavailable") or ()]
        for rel in unavailable:
            if not isinstance(rel, str):
                continue
            source = target.joinpath(*rel.split("/"))
            dest = staged.joinpath(*rel.split("/"))
            if os.path.isfile(source) and not os.path.islink(source) and not os.path.lexists(dest):
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
        fs.move_folder(project, target, staging / _OLD / step.name)
    fs.move_folder(project, staged, target)


def _check_agreement(manifest: _manifest.Manifest, lock: _lock.Lock,
                     claims: Mapping[str, Optional[str]], unclaimed: Sequence[str]) -> None:
    """Refuse a lock that does not satisfy the manifest (exit 4)."""
    not_locked = [key for key, name in claims.items() if name is None]
    differ = []
    for key, name in claims.items():
        if name is None:
            continue
        entry, locked = manifest[key], lock.skills[name]
        if not _satisfied(entry, locked) or locked.get("pin") != entry.pin:
            differ.append(f"{key} = {entry.spec!r} but the lock has v{locked.get('version')} "
                          f"({locked.get('pin')})")
    if not (not_locked or unclaimed or differ):
        return
    parts = []
    if not_locked:
        parts.append(f"not in the lock: {', '.join(not_locked)}")
    if unclaimed:
        parts.append(f"not in `{_MANIFEST}`: {', '.join(unclaimed)}")
    if differ:
        parts.append("; ".join(differ))
    raise LockOutOfDateError(
        f"`{_LOCKFILE}` is out of date with `{_MANIFEST}` ({'; '.join(parts)})",
        details={"not_locked": not_locked, "not_in_manifest": list(unclaimed),
                 "spec_mismatch": differ})


# ── status ───────────────────────────────────────────────────────────────────


def _status(run: _Run, *, remote: bool) -> None:
    project = run.project
    interrupted: Optional[str] = None
    try:
        journal = fs.read_journal(project)
        if journal is not None:
            interrupted = journal.name
            run.note("interrupted", f"an interrupted run left `{ROOT}/{journal.name}` half "
                                    "changed", skill=journal.name, drift=True,
                     fix="run `decimalai skills install` to settle it")
    except JournalConflictError as exc:
        run.note("interrupted", exc.message, drift=True,
                 fix="run `decimalai skills install` to settle it")

    manifest: Optional[_manifest.Manifest]
    try:
        manifest = _manifest.load(project, missing_ok=True)
    except SkillpkgError as exc:
        manifest = None
        run.note("manifest_invalid", exc.message, drift=True, fix=exc.fix)

    lock: Optional[_lock.Lock] = None
    try:
        lock = _lock.load(project, missing_ok=True)
    except LegacyLockError as exc:
        run.note("legacy_lock", exc.message, drift=True, fix="run `decimalai skills migrate`")
    except (ForeignLockEntriesError, LockHashMismatch) as exc:
        code = "foreign_entries" if isinstance(exc, ForeignLockEntriesError) else "lock_hash"
        run.note(code, exc.message, drift=True, fix=exc.fix)
        with contextlib.suppress(SkillpkgError):
            lock = _lock.load(project, missing_ok=True, repair=True)
    except SkillsLockError as exc:
        run.note("lock_unreadable", exc.message, drift=True, fix=exc.fix)

    skills = lock.skills if lock is not None else {}
    states = fs.scan_root(project, skills)
    claims: Dict[str, Optional[str]] = {}
    unclaimed: List[str] = []
    if manifest is not None and lock is not None:
        claims, unclaimed = _match(manifest, lock)
    specs = {name: manifest[key].spec for key, name in claims.items()
             if name is not None and manifest is not None}

    findings = _remote_findings(run, skills, states) if remote and skills else {}

    rows: List[SkillStatus] = []
    for name, st in states.items():
        entry = skills.get(name)
        state = "interrupted" if name == interrupted else st.state.value
        rows.append(SkillStatus(
            install_name=name, state=state,
            skill=str(entry.get("name")) if entry else None,
            version=entry.get("version") if entry else None,
            pin=entry.get("pin") if entry else None,
            spec=specs.get(name),
            files=st.files, reason=st.reason, remote=tuple(findings.get(name, ()))))
        who = str(entry.get("name")) if entry else name
        if st.state is fs.State.MODIFIED:
            run.note("modified", f"`{ROOT}/{name}` has local changes ({', '.join(st.files)})",
                     skill=who, path=f"{ROOT}/{name}", drift=True,
                     fix="run `decimalai skills install --force` to restore the locked files")
        elif st.state is fs.State.MISSING:
            run.note("missing", f"`{ROOT}/{name}` is locked but not on disk", skill=who,
                     path=f"{ROOT}/{name}", drift=True, fix="run `decimalai skills install`")
        elif st.state is fs.State.ILLEGAL:
            run.note("illegal", f"`{ROOT}/{name}`: {st.reason}", skill=who,
                     path=f"{ROOT}/{name}", drift=True)
        elif st.state is fs.State.UNMANAGED_STAMPED:
            run.note("unmanaged", f"`{ROOT}/{name}` carries a registry stamp but is not in "
                                  f"`{_LOCKFILE}` (an older install, or a copy)",
                     skill=name, path=f"{ROOT}/{name}", level="info")
    if manifest is not None and lock is not None:
        for key, name in claims.items():
            entry = manifest[key]
            if name is None:
                rows.append(SkillStatus(None, "not_installed", skill=key, spec=entry.spec))
                run.note("not_installed", f"`{_MANIFEST}` lists it, and it is not installed",
                         skill=key, drift=True, fix="run `decimalai skills install`")
                continue
            locked = skills[name]
            if not _satisfied(entry, locked) or locked.get("pin") != entry.pin:
                run.note("spec_mismatch", f"`{_MANIFEST}` asks for {entry.spec}, and the lock "
                                          f"has v{locked.get('version')} ({locked.get('pin')})",
                         skill=key, drift=True, fix="run `decimalai skills install`")
        for name in unclaimed:
            run.note("not_in_manifest", f"`{ROOT}/{name}` is locked but not in `{_MANIFEST}`",
                     skill=str(skills[name].get("name") or name), drift=True,
                     fix="run `decimalai skills install` to remove it, or "
                         "`decimalai skills add` to keep it")
    for rel in _git_ignored(project, (_LOCKFILE, _MANIFEST)):
        run.note("lock_ignored", f"git ignores `{rel}`, so it is never committed",
                 path=rel, drift=True, fix=f"remove the .gitignore line that matches `{rel}`")
    run.changes += rows


def _remote_findings(run: _Run, skills: Mapping[str, Mapping[str, Any]],
                     states: Mapping[str, fs.FolderState]) -> Dict[str, List[str]]:
    """What the registry says about each locked skill: revoked, outdated, re-rendered."""
    from .client import ResolveRequest

    items: List[Tuple[str, str, "ResolveRequest"]] = []
    for name, entry in skills.items():
        items.append(("locked", name, ResolveRequest(skill_id=entry["skill_id"],
                                                     version_id=entry["version_id"])))
        if entry.get("pin") == "latest":
            items.append(("latest", name, ResolveRequest(skill_id=entry["skill_id"])))
    taken = {name: (skills[name]["skill_id"] if name in skills else None) for name in states}
    client = run.client(why="`status --remote` needs the registry")
    results = client.resolve([request for _, _, request in items], taken=taken,
                             purpose="update")
    out: Dict[str, List[str]] = {}
    for (kind, name, _), result in zip(items, results):
        entry = skills[name]
        who = str(entry.get("name") or name)
        found: Optional[Tuple[str, str]] = None
        if isinstance(result, SkillpkgError):
            if kind == "locked":
                reason = result.code if result.code != "unavailable" else (
                    result.details.get("registry_code") or "unavailable")
                found = ("revoked", f"revoked ({reason})")
        elif kind == "locked":
            locked_render = entry.get("render") or {}
            if result.description_sha256 != entry.get("description_sha256"):
                found = ("description_changed", "description changed upstream (same version)")
            elif (result.render.variant == locked_render.get("variant")
                  and result.render.to_dict() != locked_render):
                # Without the key that installed it, an authored skill resolves as the
                # foreign variant: that is not a change upstream.
                found = ("render_changed", "rendered differently upstream (same version)")
        elif result.version > int(entry.get("version") or 0):
            found = ("outdated", f"outdated (v{entry.get('version')} -> v{result.version})")
        if found is None:
            continue
        code, text = found
        out.setdefault(name, []).append(text)
        fix = ("run `decimalai skills update` or `decimalai skills remove`" if code == "revoked"
               else "run `decimalai skills update`")
        run.note(code, text, skill=who, drift=True, fix=fix)
    return out


def _git_ignored(project: Project, paths: Sequence[str]) -> List[str]:
    """The paths git ignores in the project's repository (none outside a repository)."""
    git = shutil.which("git")
    if git is None:
        return []
    try:
        proc = subprocess.run(
            [git, "-C", str(project.directory), "check-ignore", "--", *paths],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode not in (0, 1):
        return []  # not a repository, or git failed
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
