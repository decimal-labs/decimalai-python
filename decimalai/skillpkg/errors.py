"""Exit codes, errors and the result type of ``decimalai skills`` and ``decimalai.skillpkg``.

The exit codes are stable and documented; CI scripts and Dockerfiles branch on them:

====  =====================================================================================
code  meaning
====  =====================================================================================
0     ok
1     ``status --check`` or ``verify`` found problems; a journal conflict; another run holds
      the project
2     usage error
3     the lock or the manifest is missing or unreadable, or the lock is a version 1 or npx lock
4     the lock is out of date with ``.decimal/skills.toml`` (``install --frozen``)
5     local edits, an unmanaged folder or a symlink block the operation
6     integrity: served bytes differ from the declared or locked hashes; ``lock_hash`` mismatch
7     unavailable: not found, ambiguous, withheld, blocked, version unsafe, no access, name
      taken, render gone or invalid
8     network failure, or ``--offline`` needed a fetch
====  =====================================================================================

Each exception class carries one of those codes (``exit_code``), a stable machine-readable
``code`` and a ``fix``: the command that gets the user unstuck. Every refusal names the skill,
the reason and the fix::

    >>> err = LockedVersionRevokedError(
    ...     "locked version 3 is no longer available (blocked)", skill="acme/refund-policy")
    >>> print(err)
    acme/refund-policy: locked version 3 is no longer available (blocked); run `decimalai skills update acme/refund-policy` or `decimalai skills remove acme/refund-policy`
    >>> err.exit_code
    <ExitCode.UNAVAILABLE: 7>

This module is the leaf of the engine. It imports nothing outside the standard library, so
every other module of the engine, and the CLI, can import it at no cost.
"""
from __future__ import annotations

import dataclasses
import enum
import os
from dataclasses import dataclass
from typing import Any, ClassVar, Iterable, Mapping, Optional, Sequence, Union

__all__ = [
    "EXIT_CODE_MEANINGS",
    "ExitCode",
    "Result",
    "SkillpkgError",
    # 1
    "JournalConflictError",
    "SkillsBusyError",
    # 2
    "UsageError",
    # 3
    "SkillsLockError",
    "LegacyLockError",
    "ForeignLockEntriesError",
    "ManifestError",
    "SkillsNotInstalledError",
    # 4
    "LockOutOfDateError",
    # 5
    "LocalChangesError",
    "ModifiedFolderError",
    "UnmanagedFolderError",
    "IllegalEntryError",
    # 6
    "IntegrityError",
    "LockHashMismatch",
    "BundleContractError",
    # 7
    "UnavailableError",
    "SkillNotFoundError",
    "AmbiguousRefError",
    "SkillWithheldError",
    "SkillBlockedError",
    "VersionUnsafeError",
    "InstallNameCollisionError",
    "RenderInvalidError",
    "RenderGoneError",
    "LockedVersionRevokedError",
    "AccessDeniedError",
    "AlreadyInstalledError",
    "ConfirmationRequiredError",
    "SkillNotManagedError",
    "RegistryTooOldError",
    # 8
    "NetworkError",
    "OfflineError",
]


class ExitCode(enum.IntEnum):
    """The documented process exit codes. An ``int``, so ``sys.exit(code)`` works."""

    OK = 0
    PROBLEMS = 1
    USAGE = 2
    LOCK = 3
    LOCK_OUT_OF_DATE = 4
    LOCAL_CHANGES = 5
    INTEGRITY = 6
    UNAVAILABLE = 7
    NETWORK = 8


#: One line per exit code, for ``--help`` text and the docs.
EXIT_CODE_MEANINGS: Mapping[ExitCode, str] = {
    ExitCode.OK: "ok",
    ExitCode.PROBLEMS: (
        "`status --check` or `verify` found problems; a journal conflict; "
        "another run holds the project"
    ),
    ExitCode.USAGE: "usage error",
    ExitCode.LOCK: (
        "the lock or the manifest is missing or unreadable, or the lock is a "
        "version 1 or npx lock"
    ),
    ExitCode.LOCK_OUT_OF_DATE: "the lock is out of date with `.decimal/skills.toml`",
    ExitCode.LOCAL_CHANGES: "local edits, an unmanaged folder or a symlink block the operation",
    ExitCode.INTEGRITY: (
        "integrity: served bytes differ from the declared or locked hashes, "
        "or `lock_hash` does not recompute"
    ),
    ExitCode.UNAVAILABLE: (
        "unavailable: not found, ambiguous, withheld, blocked, version unsafe, "
        "no access, name taken, render gone or invalid"
    ),
    ExitCode.NETWORK: "network failure, or `--offline` needed a fetch",
}


# ── The base class ───────────────────────────────────────────────────────────


class SkillpkgError(Exception):
    """Base class of every refusal the skills engine reports.

    ``exit_code`` and ``code`` belong to the class and never change for it. ``fix`` is a
    class default that a raise site may replace with a more precise command. A ``{skill}``
    in the default is filled with the skill's name quoted for a POSIX shell, so the printed
    command can be pasted as it is: registry names may contain spaces and quotes
    (``decimalai skills remove 'composiohq/Docker Hub Automation'``), while a plain name
    such as ``acme/refund-policy`` is printed unchanged.

    Args:
        message: the reason, as a clause (``"2 files differ from the lock"``).
        skill: the registry or install name the error is about, when there is one.
        path: the file or folder the error is about, when there is one.
        fix: replaces the class's default fix. A raise site that puts a name into its
            own fix quotes it the same way (``shlex.quote``).
        details: JSON-ready facts for ``--json`` output (files, entries, near misses).
    """

    exit_code: ClassVar[ExitCode] = ExitCode.PROBLEMS
    code: ClassVar[str] = "error"
    fix: str = "run `decimalai skills status` to see the state of the project"

    def __init__(
        self,
        message: str,
        *,
        skill: Optional[str] = None,
        path: Union[str, "os.PathLike[str]", None] = None,
        fix: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.skill = skill
        self.path = None if path is None else os.fspath(path)
        if fix is None:
            fix = type(self).fix.replace("{skill}", _shell_word(skill) if skill else "<name>")
        self.fix = fix
        self.details: dict = _plain(details) if details else {}

    def __str__(self) -> str:
        text = f"{self.skill}: {self.message}" if self.skill else self.message
        return f"{text}; {self.fix}" if self.fix else text

    def to_dict(self) -> dict:
        """The error as JSON-ready data, for ``--json`` output."""
        return {
            "error": type(self).__name__,
            "code": self.code,
            "exit_code": int(self.exit_code),
            "message": self.message,
            "skill": self.skill,
            "path": self.path,
            "fix": self.fix,
            "details": _plain(self.details),
        }


# ── 1: problems found, or the project is busy ───────────────────────────────


class JournalConflictError(SkillpkgError):
    """A crashed run left a journal, and the folder matches neither its old nor its new entry."""

    exit_code = ExitCode.PROBLEMS
    code = "journal_conflict"
    fix = (
        "move the skill's folder out of `.agents/skills/`, delete "
        "`.decimal/skills.lock.journal`, then run `decimalai skills install`"
    )


class SkillsBusyError(SkillpkgError):
    """Another ``decimalai skills`` run holds ``.decimal/.skills.lck`` past the wait timeout."""

    exit_code = ExitCode.PROBLEMS
    code = "busy"
    fix = (
        "wait for the other `decimalai skills` command to finish; if none is running, "
        "delete `.decimal/.skills.lck`"
    )


# ── 2: usage ─────────────────────────────────────────────────────────────────


class UsageError(SkillpkgError, ValueError):
    """An argument the engine cannot accept, such as a malformed ``<ref>@<version>``."""

    exit_code = ExitCode.USAGE
    code = "usage"
    fix = "run `decimalai skills --help` for the accepted forms"


# ── 3: the lock or the manifest cannot be used ───────────────────────────────


class SkillsLockError(SkillpkgError):
    """``.decimal/skills.lock`` cannot be read: corrupt JSON, or not a lock at all."""

    exit_code = ExitCode.LOCK
    code = "lock_unreadable"
    fix = (
        "restore `.decimal/skills.lock` from version control, or delete it and run "
        "`decimalai skills install` to write a new one"
    )


class LegacyLockError(SkillsLockError):
    """A version 1 lock (or one with no version), or npx's ``.decimal/skills.lock.json``."""

    code = "legacy_lock"
    fix = "run `decimalai skills migrate`"


class ForeignLockEntriesError(SkillsLockError):
    """Lock entries without ``version_id``, ``files`` or ``render``: an older decimalai wrote them."""

    code = "foreign_lock_entries"
    fix = (
        "run `decimalai skills install` to drop them (their folders stay, unmanaged), "
        "then commit `.decimal/skills.lock`"
    )


class ManifestError(SkillpkgError):
    """``.decimal/skills.toml`` cannot be used: invalid TOML, a bad spec, or an unsupported root."""

    exit_code = ExitCode.LOCK
    code = "manifest_invalid"
    fix = "correct the named line of `.decimal/skills.toml`, then run `decimalai skills install`"


class SkillsNotInstalledError(SkillpkgError):
    """The project has no lock or no skills root yet."""

    exit_code = ExitCode.LOCK
    code = "not_installed"
    fix = "run `decimalai skills install`, or `decimalai skills add <ref>` for a first skill"


# ── 4: the lock disagrees with the manifest ──────────────────────────────────


class LockOutOfDateError(SkillpkgError):
    """``install --frozen``: the lock does not satisfy ``.decimal/skills.toml``."""

    exit_code = ExitCode.LOCK_OUT_OF_DATE
    code = "lock_out_of_date"
    fix = (
        "run `decimalai skills install`, then commit `.decimal/skills.toml` and "
        "`.decimal/skills.lock`"
    )


# ── 5: local state blocks the operation ──────────────────────────────────────


class LocalChangesError(SkillpkgError):
    """Something on disk that the engine will not overwrite without ``--force``."""

    exit_code = ExitCode.LOCAL_CHANGES
    code = "local_changes"
    fix = "move your changes out of the way, or re-run with `--force` to replace them"


class ModifiedFolderError(LocalChangesError):
    """A managed folder differs from the lock: a file was edited, added or removed."""

    code = "modified"
    fix = (
        "keep your edits by copying the folder to a new name (a skill of your own), "
        "or re-run with `--force` to replace them"
    )


class UnmanagedFolderError(LocalChangesError):
    """A folder the lock does not list already holds the name the operation needs."""

    code = "unmanaged_folder"
    fix = (
        "rename or remove that folder (it is not in `.decimal/skills.lock`), "
        "or re-run with `--force` to replace it"
    )


class IllegalEntryError(SkillpkgError):
    """A symlink, or a file where a skill folder belongs. ``--force`` never overrides it."""

    exit_code = ExitCode.LOCAL_CHANGES
    code = "illegal_entry"
    fix = (
        "replace it with a real directory, or move it out of `.agents/skills/`; "
        "`decimalai skills` never follows a symlink inside the project"
    )


# ── 6: integrity ─────────────────────────────────────────────────────────────


class IntegrityError(SkillpkgError):
    """Served bytes differ from the hashes the registry declared or the lock recorded."""

    exit_code = ExitCode.INTEGRITY
    code = "integrity"
    fix = (
        "nothing was written; if the change is expected, run `decimalai skills update "
        "{skill}` and review the new files before committing the lock"
    )


class LockHashMismatch(IntegrityError):
    """``lock_hash`` does not recompute: the lock was edited by hand."""

    code = "lock_hash_mismatch"
    fix = (
        "run `decimalai skills install` to recompute it, then review and commit "
        "`.decimal/skills.lock`"
    )


class BundleContractError(IntegrityError):
    """A served bundle breaks the on-disk contract: an unsafe path, or a SKILL.md that fails verify."""

    code = "bundle_invalid"
    fix = "nothing was written; run `pip install -U decimalai` and retry, and report it if it persists"


# ── 7: unavailable ───────────────────────────────────────────────────────────


class UnavailableError(SkillpkgError):
    """The registry will not serve the skill or version asked for.

    Build one from a per-item error code of the install routes with
    :meth:`UnavailableError.for_code`.
    """

    exit_code = ExitCode.UNAVAILABLE
    code = "unavailable"
    fix = "check the reference and your API key (`DECIMAL_API_KEY`), then retry"

    @classmethod
    def for_code(
        cls, registry_code: str, message: str, **kwargs: Any,
    ) -> "UnavailableError":
        """The error class for a registry error code; an unknown code stays ``UnavailableError``.

        Takes the same keyword arguments as the constructor. An unknown code is kept in
        ``details["registry_code"]``, so a newer server's refusal still exits 7 and says
        what it was.
        """
        target = _BY_REGISTRY_CODE.get(registry_code)
        if target is not None:
            return target(message, **kwargs)
        details = dict(kwargs.pop("details", None) or {})
        details["registry_code"] = registry_code
        return UnavailableError(message, details=details, **kwargs)


class SkillNotFoundError(UnavailableError):
    """No skill the caller can see matches the reference."""

    code = "not_found"
    fix = (
        "check the spelling, or pass the skill's url_slug or id; `decimalai skills list` "
        "shows your organization's skills"
    )


class AmbiguousRefError(UnavailableError):
    """The reference matches more than one skill."""

    code = "ambiguous"
    fix = "pass the full registry name (`owner/name`) or the skill's id"


class SkillWithheldError(UnavailableError):
    """The registry withholds the skill from installs."""

    code = "withheld"
    fix = "choose another skill, and run `decimalai skills remove {skill}` if it is installed"


class SkillBlockedError(UnavailableError):
    """The skill failed its safety review and is blocked."""

    code = "blocked"
    fix = "choose another skill, and run `decimalai skills remove {skill}` if it is installed"


class VersionUnsafeError(UnavailableError):
    """The requested version is marked unsafe, though the skill has other versions."""

    code = "version_unsafe"
    fix = "install another version with `decimalai skills add {skill}@latest` or `@<N>`"


class InstallNameCollisionError(UnavailableError):
    """Both forms of the skill's install name are already used in this project."""

    code = "install_name_collision"
    fix = (
        "rename or remove the folder that already uses the name, then retry; "
        "`decimalai skills status` lists the folders"
    )


class RenderInvalidError(UnavailableError):
    """The registry cannot render a valid SKILL.md for this version."""

    code = "render_invalid"
    fix = "try another version with `decimalai skills add {skill}@<N>`, or report it to the skill's owner"


class RenderGoneError(UnavailableError):
    """The render the lock names no longer exists on the registry."""

    code = "render_gone"
    fix = "run `decimalai skills update {skill}` to take the current render, then commit `.decimal/skills.lock`"


class LockedVersionRevokedError(UnavailableError):
    """A locked version the registry has since blocked, withheld or marked unsafe."""

    code = "revoked"
    fix = "run `decimalai skills update {skill}` or `decimalai skills remove {skill}`"


class AccessDeniedError(UnavailableError):
    """The API key in use cannot fetch this skill, or this variant of it."""

    code = "no_access"
    fix = "set `DECIMAL_API_KEY` to a key of an organization that can see this skill"


class AlreadyInstalledError(UnavailableError):
    """The skill is already installed in this project under another name."""

    code = "already_installed"
    fix = (
        "run `decimalai skills status` to find its folder; `decimalai skills update "
        "<name>@<N>` changes its version"
    )


class ConfirmationRequiredError(UnavailableError):
    """The skill needs a confirmation (caution or unreviewed) and there is no terminal to ask."""

    code = "confirmation_required"
    fix = "review the skill, then re-run with `--yes` to accept it"


class SkillNotManagedError(UnavailableError):
    """No skill of that name is in this project's manifest or lock."""

    code = "not_managed"
    fix = "run `decimalai skills status` to list the installed skills"


class RegistryTooOldError(UnavailableError):
    """The registry has no install routes; this client never falls back to older ones."""

    code = "registry_too_old"
    fix = "check `--base-url` and `DECIMAL_BASE_URL`; the default registry is https://api.decimal.ai"


# ── 8: network ───────────────────────────────────────────────────────────────


class NetworkError(SkillpkgError):
    """The registry could not be reached, or did not answer in time."""

    exit_code = ExitCode.NETWORK
    code = "network"
    fix = "check the connection and `--base-url`, then run the command again"


class OfflineError(NetworkError):
    """``--offline`` was given, and a folder had to be fetched."""

    code = "offline"
    fix = (
        "run it without `--offline`, or commit `.agents/skills` so the folders "
        "are there without a fetch"
    )


#: The per-item error codes of the install routes, each with its own class.
_BY_REGISTRY_CODE: Mapping[str, type] = {
    cls.code: cls
    for cls in (
        SkillNotFoundError, AmbiguousRefError, SkillWithheldError, SkillBlockedError,
        VersionUnsafeError, InstallNameCollisionError, RenderInvalidError, RenderGoneError,
    )
}


# ── Result ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Result:
    """What one engine call did: ``Result(changes, problems, exit_code)``.

    ``changes`` and ``problems`` are stored as tuples. ``problems`` holds errors and
    verify findings alike; warnings may sit there with an ``exit_code`` of 0.
    """

    changes: Sequence[Any] = ()
    problems: Sequence[Any] = ()
    exit_code: ExitCode = ExitCode.OK

    def __post_init__(self) -> None:
        object.__setattr__(self, "changes", tuple(self.changes))
        object.__setattr__(self, "problems", tuple(self.problems))
        # Refuses an undocumented code with ValueError.
        object.__setattr__(self, "exit_code", ExitCode(self.exit_code))

    @property
    def ok(self) -> bool:
        return self.exit_code == ExitCode.OK

    @classmethod
    def from_error(cls, error: SkillpkgError, *, changes: Iterable[Any] = ()) -> "Result":
        """The result of a call that stopped at ``error``, after ``changes``."""
        return cls(tuple(changes), (error,), error.exit_code)

    def to_dict(self) -> dict:
        """The result as JSON-ready data: the ``--json`` output of a command."""
        return {
            "ok": self.ok,
            "exit_code": int(self.exit_code),
            "changes": [_plain(change) for change in self.changes],
            "problems": [_plain(problem) for problem in self.problems],
        }


def _shell_word(text: str) -> str:
    """``text`` as one word of a POSIX shell command; a plain name comes back unchanged."""
    # Imported on first use, not at the top: ``import decimalai`` does not load shlex, and
    # importing this module must load no module but itself.
    from shlex import quote

    return quote(text)


def _plain(value: Any) -> Any:
    """``value`` as JSON-ready data: dicts, lists, strings, numbers, booleans and None."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return int(value) if isinstance(value, enum.IntEnum) else value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    as_dict = getattr(value, "_asdict", None)  # a NamedTuple
    if callable(as_dict):
        return _plain(as_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(item) for item in value), key=repr)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, enum.Enum):
        return _plain(value.value)
    return str(value)
