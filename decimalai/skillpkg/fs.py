"""Skill folders on disk: their states, staging, the journaled swap, recovery, the run lock.

**States** (:func:`scan_root`). Every child of ``.agents/skills``, and every lock entry
without a folder, is in one :class:`State`: ``CLEAN`` (exactly the locked files, hashes
and exec bits), ``MODIFIED`` (some file differs, is missing or is extra), ``MISSING``,
``UNMANAGED`` (a folder the lock does not list; ``UNMANAGED_STAMPED`` when its SKILL.md
carries a ``decimalai-skill-id``) or ``ILLEGAL`` (a symlink anywhere, a file where a
folder belongs, or a file or folder the engine cannot read, with a reason starting
``unreadable:``). OS and tool clutter (:data:`CLUTTER`) is ignored everywhere, and so are
the ``.<name>.partial`` folders an interrupted copy across devices leaves in the root.

**Changing a folder** is staged, then swapped::

    txid = fs.new_txid()
    fs.stage(project, txid, name, contents, entry)    # every byte checked; root untouched
    lock = fs.swap(project, txid, name, entry, lock)  # journal, rename, rename, lock, done
    fs.discard_staging(project, txid)

:func:`stage` writes the folder under ``.decimal/staging/<txid>/`` and checks each file
against the hashes ``entry`` declares, and SKILL.md against the format's rules. Nothing
under the root changes until :func:`swap`, which replaces one whole folder:

1. write ``.decimal/skills.lock.journal`` = ``{txid, name, old_entry, new_entry, old_present}``;
2. rename ``<root>/<name>`` to ``.decimal/staging/<txid>/.old/<name>``;
3. rename the staged folder to ``<root>/<name>``;
4. write the lock with this entry changed;
5. delete the journal.

A whole-folder swap is what removes files a new version dropped. Removing a skill is
step 2 alone. One kind of file crosses over: a file the user placed by hand where the
lock lists an ``unavailable`` (binary) file is copied into the new folder when the new
version still lists it as unavailable; when the new version serves that path, no longer
lists it, or the skill is removed, the hand-placed copy is saved to
``.decimal/backup/<txid>/<name>/`` (unless it equals the served bytes), with a warning
naming it.

**Recovery** (:func:`recover`). A journal at startup means a run died inside a swap.
The folder decides: if it is the new one, the new entry is committed; if the old one
still stands, or waits in ``.old``, it is put back and the old entry kept; anything else
is a :class:`~decimalai.skillpkg.errors.JournalConflictError` and nothing is touched. So
a crash leaves the old folder and entry or the new ones, never a folder that looks like
a local edit.

**Moving a folder without recording anything** (:func:`move_folder`): the same
cross-device-safe move a swap uses, with no journal and no lock write, for a staged
folder the lock already describes (``install --frozen``) or a folder leaving the root.

**One run at a time** (:class:`ProcessLock`): ``.decimal/.skills.lck``, created
exclusively, holding the pid. A dead holder's lock, or one older than ten minutes, is
taken over.

Run :func:`recover` and the swaps while holding the :class:`ProcessLock`.
"""
from __future__ import annotations

import contextlib
import enum
import fnmatch
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from skillevaluation import skillfmt

from . import lock as _lock
from .errors import (
    BundleContractError,
    IllegalEntryError,
    IntegrityError,
    JournalConflictError,
    SkillsBusyError,
    SkillsLockError,
    UsageError,
)
from .project import ROOT, Project, fsync_dir

__all__ = [
    "CHECKPOINTS",
    "CLUTTER",
    "FolderState",
    "Journal",
    "ProcessLock",
    "Recovery",
    "State",
    "backup_folder",
    "check_folder",
    "discard_staging",
    "is_clutter_path",
    "move_folder",
    "new_txid",
    "read_journal",
    "recover",
    "scan_root",
    "stage",
    "swap",
]

#: The points between the steps of :func:`swap`, in order. Tests stop a swap at each one.
CHECKPOINTS = ("journal", "old_moved", "new_moved", "lock_written")

#: OS and tool clutter: never part of a skill. Inside a skill folder, and in the root, it
#: never makes a folder MODIFIED or the root ILLEGAL, and a swap drops it with the old
#: folder. A trailing ``/`` names a directory, skipped with everything under it; the
#: other entries are file names, ``*`` a wildcard. Symlinks are never clutter.
CLUTTER = (".DS_Store", "Thumbs.db", "desktop.ini", "__pycache__/", "*.pyc")
_CLUTTER_DIRS = frozenset(c[:-1] for c in CLUTTER if c.endswith("/"))
_CLUTTER_FILES = tuple(c for c in CLUTTER if not c.endswith("/"))

_OLD = ".old"
_PARTIAL = ".partial"
#: What :func:`_move` leaves in the root when a copy across devices is interrupted.
_DEBRIS = re.compile(r"\.(.+)\.partial")
_CHUNK = 1 << 20

_log = logging.getLogger("decimalai.skillpkg")
_POSIX = os.name == "posix"


def _checkpoint(step: str) -> None:
    """Called after each step of :func:`swap`; does nothing (tests stop a swap here)."""


# ── folder states ────────────────────────────────────────────────────────────


class State(enum.Enum):
    CLEAN = "clean"
    MODIFIED = "modified"
    MISSING = "missing"
    UNMANAGED = "unmanaged"
    UNMANAGED_STAMPED = "unmanaged_stamped"
    ILLEGAL = "illegal"


@dataclass(frozen=True)
class FolderState:
    """One folder's state. For ``MODIFIED``, which files differ and how (relative paths)."""

    name: str
    state: State
    edited: Tuple[str, ...] = ()    # content differs from the lock
    missing: Tuple[str, ...] = ()   # locked, not on disk
    extra: Tuple[str, ...] = ()     # on disk, not locked
    mode: Tuple[str, ...] = ()      # the exec bit differs from the lock (POSIX)
    skill_id: Optional[str] = None  # UNMANAGED_STAMPED: the id its SKILL.md carries
    reason: Optional[str] = None    # ILLEGAL: why

    @property
    def files(self) -> Tuple[str, ...]:
        """Every file that makes the folder MODIFIED, sorted."""
        return tuple(sorted(set(self.edited) | set(self.missing) | set(self.extra)
                            | set(self.mode)))

    @property
    def managed(self) -> bool:
        return self.state in (State.CLEAN, State.MODIFIED, State.MISSING)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_clutter(name: str, mode: int) -> bool:
    """Whether a directory entry with this name and ``lstat`` mode is :data:`CLUTTER`."""
    if stat.S_ISDIR(mode):
        return name in _CLUTTER_DIRS
    if stat.S_ISREG(mode):
        return any(fnmatch.fnmatchcase(name, pattern) for pattern in _CLUTTER_FILES)
    return False


def is_clutter_path(rel: str) -> bool:
    """Whether the file at relative path ``rel`` would be ignored as :data:`CLUTTER`."""
    parts = rel.split("/")
    return (any(part in _CLUTTER_DIRS for part in parts[:-1])
            or any(fnmatch.fnmatchcase(parts[-1], pattern) for pattern in _CLUTTER_FILES))


def _is_debris(name: str, mode: int) -> bool:
    """A ``.<install-name>.partial`` directory: an interrupted copy, never the only copy."""
    m = _DEBRIS.fullmatch(name)
    return m is not None and stat.S_ISDIR(mode) and skillfmt.is_valid_install_name(m.group(1))


def _walk(folder: Path) -> Tuple[Dict[str, os.stat_result], Optional[str]]:
    """Every regular file under ``folder`` (relative POSIX path -> lstat), or why it is
    ILLEGAL. :data:`CLUTTER` is skipped."""
    files: Dict[str, os.stat_result] = {}
    pending: List[Tuple[Path, str]] = [(folder, "")]
    while pending:
        directory, prefix = pending.pop()
        try:
            with os.scandir(directory) as it:
                entries = [(item, item.stat(follow_symlinks=False)) for item in it]
        except OSError as exc:
            return files, _unreadable(prefix.rstrip("/") or ".", exc)
        for item, st in entries:
            rel = prefix + item.name
            if _is_clutter(item.name, st.st_mode):
                continue
            if stat.S_ISLNK(st.st_mode):
                return files, f"`{rel}` is a symlink"
            if stat.S_ISDIR(st.st_mode):
                pending.append((Path(item.path), rel + "/"))
            elif stat.S_ISREG(st.st_mode):
                files[rel] = st
            else:
                return files, f"`{rel}` is not a regular file"
    return files, None


def _unreadable(rel: str, exc: OSError) -> str:
    """The ILLEGAL reason for a file or folder the engine cannot read."""
    return f"unreadable: `{rel}` ({exc.strerror or exc})"


def _rel_in(folder: Path, filename: Any) -> str:
    """``filename`` relative to ``folder``, as an ILLEGAL reason names it."""
    try:
        return Path(os.fspath(filename)).relative_to(folder).as_posix()
    except (TypeError, ValueError):
        return "."


def _illegal_entry(path: Path) -> Optional[str]:
    """Why ``path`` cannot be a skill folder (a symlink, not a directory), or None."""
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        return "a symlink"
    if not stat.S_ISDIR(st.st_mode):
        return "not a directory"
    return None


def check_folder(path: Path, entry: Mapping[str, Any], *, name: Optional[str] = None) -> FolderState:
    """The state of the folder at ``path`` against its lock ``entry``.

    One of ``CLEAN``, ``MODIFIED``, ``MISSING`` or ``ILLEGAL``. Files the entry lists as
    ``unavailable`` may be present or absent. A folder inside it that cannot be listed
    makes it ``ILLEGAL`` with a reason starting ``unreadable:``.

    Raises:
        OSError: a file cannot be opened to hash it. :func:`scan_root` reports that folder
            as ``ILLEGAL`` (``unreadable: <path>``); callers that grade the bytes
            themselves catch it here.
    """
    name = name or path.name
    if not os.path.lexists(path):
        return FolderState(name, State.MISSING)
    why = _illegal_entry(path)
    if why:
        return FolderState(name, State.ILLEGAL, reason=why)
    on_disk, why = _walk(path)
    if why:
        return FolderState(name, State.ILLEGAL, reason=why)
    expected: Mapping[str, str] = entry.get("files") or {}
    executable = set(entry.get("executable") or ())
    unavailable = {u.get("path") for u in entry.get("unavailable") or () if isinstance(u, dict)}
    edited, extra, mode = [], [], []
    for rel, st in sorted(on_disk.items()):
        if rel not in expected:
            if rel not in unavailable:
                extra.append(rel)
            continue
        if _sha256_file(path / rel) != expected[rel]:
            edited.append(rel)
        if _POSIX and bool(st.st_mode & stat.S_IXUSR) != (rel in executable):
            mode.append(rel)
    missing = sorted(set(expected) - set(on_disk))
    if edited or extra or mode or missing:
        return FolderState(name, State.MODIFIED, edited=tuple(edited), missing=tuple(missing),
                           extra=tuple(extra), mode=tuple(mode))
    return FolderState(name, State.CLEAN)


def _stamp(folder: Path) -> Optional[str]:
    """The ``metadata.decimalai-skill-id`` of ``folder/SKILL.md``, if it parses and has one."""
    try:
        text = (folder / "SKILL.md").read_bytes().decode("utf-8")
        fm, _ = skillfmt.parse_skill_md_text(text)
    except (OSError, UnicodeDecodeError, skillfmt.SkillFmtError):
        return None
    meta = fm.get("metadata")
    sid = meta.get("decimalai-skill-id") if isinstance(meta, dict) else None
    return sid if isinstance(sid, str) and sid else None


def _check_root(project: Project) -> bool:
    """Refuse a symlinked or non-directory root; whether it exists."""
    root = project.skills_root
    project.refuse_symlinks(root)
    if not os.path.lexists(root):
        return False
    if not root.is_dir():
        raise IllegalEntryError(f"`{ROOT}` is not a directory", path=root)
    return True


def _root_children(root: Path) -> List[Tuple[str, int]]:
    """``(name, lstat mode)`` of each entry of the root, clutter and debris left out."""
    out: List[Tuple[str, int]] = []
    with os.scandir(root) as it:
        for item in it:
            mode = item.stat(follow_symlinks=False).st_mode
            if not (_is_clutter(item.name, mode) or _is_debris(item.name, mode)):
                out.append((item.name, mode))
    return out


def scan_root(project: Project, skills: Mapping[str, Mapping[str, Any]]) -> Dict[str, FolderState]:
    """The state of every folder in the root and every locked skill, by name (sorted).

    ``skills`` is the lock's entries (``lock.skills``). Names that differ only by case
    (two folders, or a folder and a locked name) are ``ILLEGAL``: on a case-insensitive
    filesystem they are one folder, so nothing may be fetched, replaced or named there.

    Raises:
        IllegalEntryError: ``.agents`` or ``.agents/skills`` is a symlink, or the root
            is not a directory: no folder in it can be trusted.
    """
    out: Dict[str, FolderState] = {}
    root = project.skills_root
    children = [name for name, _ in _root_children(root)] if _check_root(project) else []
    # On a macOS or Windows checkout, names that differ only by case are one folder.
    by_fold: Dict[str, List[str]] = {}
    for name in children:
        by_fold.setdefault(name.casefold(), []).append(name)
    locked = {key.casefold(): key for key in skills}
    for name in children:
        twins = [n for n in by_fold[name.casefold()] if n != name]
        if twins:
            out[name] = FolderState(name, State.ILLEGAL, reason=(
                f"its name differs only by case from `{twins[0]}`; on a case-insensitive "
                "filesystem they are one folder"))
            continue
        if name not in skills and name.casefold() in locked:
            out[name] = FolderState(name, State.ILLEGAL, reason=(
                f"the lock names this skill `{locked[name.casefold()]}`; the folder's name "
                "differs only by case"))
            continue
        path = root / name
        entry = skills.get(name)
        if entry is not None:
            try:
                out[name] = check_folder(path, entry, name=name)
            except OSError as exc:
                # It cannot be hashed, so it can be neither trusted nor safely replaced.
                out[name] = FolderState(name, State.ILLEGAL,
                                        reason=_unreadable(_rel_in(path, exc.filename), exc))
            continue
        why = _illegal_entry(path)
        if why is None:
            _, why = _walk(path)
        if why:
            out[name] = FolderState(name, State.ILLEGAL, reason=why)
            continue
        sid = _stamp(path)
        out[name] = (FolderState(name, State.UNMANAGED_STAMPED, skill_id=sid) if sid
                     else FolderState(name, State.UNMANAGED))
    for name in skills:
        if name in out:
            continue
        variants = by_fold.get(name.casefold())
        out[name] = (FolderState(name, State.ILLEGAL, reason=(
            f"the folder is named `{variants[0]}`; its name differs only by case"))
            if variants else FolderState(name, State.MISSING))
    return dict(sorted(out.items()))


# ── staging ──────────────────────────────────────────────────────────────────


def new_txid() -> str:
    """A transaction id: sortable by time, unique, and safe as one path component."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(4)


def _txid_dir(project: Project, txid: str) -> Path:
    if (not isinstance(txid, str) or not txid or txid.startswith(".") or "/" in txid
            or "\\" in txid or "\x00" in txid):
        raise UsageError(f"{txid!r} is not a transaction id")
    return project.staging_dir / txid


def _rmtree(path: Path) -> None:
    if os.path.lexists(path):
        if os.path.islink(path) or not os.path.isdir(path):
            os.unlink(path)
        else:
            shutil.rmtree(path)


def stage(project: Project, txid: str, name: str, contents: Mapping[str, bytes],
          entry: Mapping[str, Any]) -> Path:
    """Write a skill folder into ``.decimal/staging/<txid>/<name>/`` and check it.

    ``contents`` maps each relative path to its exact bytes; ``entry`` is the lock entry
    the folder must match (from the registry's declarations, or the lock for a frozen
    install). Files are written in binary mode with 0755 or 0644, whatever the umask.
    Nothing under the skills root is touched.

    Returns:
        The staged folder.

    Raises:
        IntegrityError: a file's sha256 differs from the one ``entry`` declares.
        BundleContractError: the file set differs from the entry's, a path is unsafe, or
            SKILL.md breaks the format's rules for a managed folder.
        IllegalEntryError: a symlink inside ``.decimal``.
    """
    folder = _txid_dir(project, txid) / project.skill_dir(name).name
    try:
        _lock.check_entry(name, entry)
    except SkillsLockError as exc:
        raise BundleContractError(exc.message, skill=name) from None
    declared: Mapping[str, str] = entry["files"]
    if set(contents) != set(declared):
        raise BundleContractError(
            "the files served differ from the files declared",
            skill=name, details={"served_only": sorted(set(contents) - set(declared)),
                                 "declared_only": sorted(set(declared) - set(contents))})
    for rel in contents:
        if rel != "SKILL.md" and skillfmt.validate_attachment_path(rel):
            raise BundleContractError(f"{rel!r} is not a safe path inside a skill folder",
                                      skill=name, details={"path": rel})
        if is_clutter_path(rel):
            raise BundleContractError(f"{rel!r} is OS or tool clutter, which the engine "
                                      "never checks, so it is never installed",
                                      skill=name, details={"path": rel})
    for rel, data in sorted(contents.items()):
        actual = hashlib.sha256(data).hexdigest()
        if actual != declared[rel]:
            raise IntegrityError(f"`{rel}` hashes to {actual}, not the declared {declared[rel]}",
                                 skill=entry.get("name") or name,
                                 details={"path": rel, "expected": declared[rel],
                                          "actual": actual})
    problems = skillfmt.validate_skill_md(contents["SKILL.md"], folder_name=name,
                                          managed=True, expect=entry)
    if problems:
        raise BundleContractError(
            f"SKILL.md breaks the format: {problems[0].rule} {problems[0].message}",
            skill=name, details={"problems": [
                {"rule": p.rule, "scope": p.scope, "message": p.message} for p in problems]})

    project.refuse_symlinks(folder)
    _rmtree(folder)
    project.makedirs(folder)
    executable = set(entry["executable"])
    for rel, data in sorted(contents.items()):
        path = folder.joinpath(*rel.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "xb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(path, 0o755 if rel in executable else 0o644)
    state = check_folder(folder, entry, name=name)
    if state.state is not State.CLEAN:
        raise IntegrityError(f"the staged folder does not read back as written ({state.files})",
                             skill=name)
    return folder


def backup_folder(project: Project, txid: str, name: str) -> Path:
    """Copy ``<root>/<name>`` to ``.decimal/backup/<txid>/<name>/`` before a forced replace."""
    source = project.skill_dir(name)
    project.refuse_symlinks(source)
    dest = project.backup_dir / _txid_dir(project, txid).name / source.name
    project.makedirs(dest.parent)
    shutil.copytree(source, dest, symlinks=True)
    return dest


def _remove_debris(project: Project) -> None:
    """Remove the ``.<name>.partial`` folders an interrupted copy left in the root.

    Safe whenever no swap is in flight (under the :class:`ProcessLock`): a copy is
    renamed into place before its source is removed, so debris is never the only copy.
    Nothing is removed through a symlinked root.
    """
    root = project.skills_root
    try:
        project.refuse_symlinks(root)
        if not root.is_dir():
            return
    except IllegalEntryError:
        return
    with os.scandir(root) as it:
        debris = [Path(item.path) for item in it
                  if _is_debris(item.name, item.stat(follow_symlinks=False).st_mode)]
    for path in debris:
        shutil.rmtree(path)


def _only_copies(project: Project, txdirs: List[Path]) -> List[Tuple[str, Path]]:
    """``(name, path)`` of each ``.old/<name>`` that is the only good copy of a locked folder.

    That is a copy that matches the lock's entry while the root's folder does not. When
    the lock cannot be read, every copy whose root folder is gone counts.
    """
    entries = _lock.locked_entries(project)
    found: List[Tuple[str, Path]] = []
    for txdir in txdirs:
        olds = txdir / _OLD
        if os.path.islink(olds) or not olds.is_dir():
            continue
        with os.scandir(olds) as it:
            names = sorted(item.name for item in it
                           if item.is_dir(follow_symlinks=False) and not item.name.startswith("."))
        for name in names:
            copy, here = olds / name, project.skills_root / name
            if entries is None:
                if not os.path.lexists(here):
                    found.append((name, copy))
                continue
            entry = entries.get(name)
            if entry is not None and _matches(copy, entry) and not _matches(here, entry):
                found.append((name, copy))
    return found


def discard_staging(project: Project, txid: Optional[str] = None) -> None:
    """Remove ``.decimal/staging/<txid>/``, or with no txid all of ``.decimal/staging/``,
    and any ``.<name>.partial`` debris in the root.

    A ``.old/<name>`` that is the only good copy of a locked folder (a crashed run moved
    it there, and the journal that would have put it back is gone) is never deleted: its
    transaction is kept, and the error says where it is.

    Raises:
        JournalConflictError: a kept copy needs the user; nothing of its transaction was
            removed.
    """
    target = project.staging_dir if txid is None else _txid_dir(project, txid)
    project.refuse_symlinks(target)
    if txid is not None:
        txdirs = [target]
    elif os.path.isdir(target):
        with os.scandir(target) as it:
            txdirs = [Path(item.path) for item in it if item.is_dir(follow_symlinks=False)]
    else:
        txdirs = []
    kept = _only_copies(project, txdirs)
    keep = {path.parent.parent for _, path in kept}
    if not keep:
        _rmtree(target)
    else:
        for txdir in txdirs:
            if txdir not in keep:
                _rmtree(txdir)
    with contextlib.suppress(OSError):
        project.staging_dir.rmdir()  # only when empty
    _remove_debris(project)
    if kept:
        name, path = kept[0]
        where = project.rel(path)
        raise JournalConflictError(
            f"`{where}` is the only copy of the locked folder `{ROOT}/{name}`, so it was "
            "not deleted",
            skill=name, path=path, details={"copies": [project.rel(p) for _, p in kept]},
            fix=(f"move `{where}` back to `{ROOT}/{name}` (it is the version the lock "
                 "describes), or delete it if you do not want it, then run the command again"))


# ── the swap ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Journal:
    """``.decimal/skills.lock.journal``: the one swap in flight.

    ``old_present`` records whether ``<root>/<name>`` existed when the swap began, so
    recovery can tell "nothing has moved yet" from "a new folder arrived".
    """

    txid: str
    name: str
    old_entry: Optional[Dict[str, Any]]
    new_entry: Optional[Dict[str, Any]]
    old_present: bool

    def to_dict(self) -> dict:
        return {"txid": self.txid, "name": self.name, "old_entry": self.old_entry,
                "new_entry": self.new_entry, "old_present": self.old_present}


def _journal_error(project: Project, why: str, *, skill: Optional[str] = None) -> JournalConflictError:
    return JournalConflictError(f"`{project.rel(project.journal_path)}` {why}",
                                skill=skill, path=project.journal_path)


def read_journal(project: Project) -> Optional[Journal]:
    """The journal of an interrupted swap, or None.

    Raises:
        JournalConflictError: there is a journal, but it cannot be read.
    """
    path = project.journal_path
    project.refuse_symlinks(path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
        journal = Journal(txid=data["txid"], name=data["name"], old_entry=data["old_entry"],
                          new_entry=data["new_entry"], old_present=data["old_present"])
        _txid_dir(project, journal.txid)
        project.skill_dir(journal.name)
        for entry in (journal.old_entry, journal.new_entry):
            if entry is not None:
                _lock.check_entry(journal.name, entry)
        if not isinstance(journal.old_present, bool):
            raise ValueError("old_present")
    except (UnicodeDecodeError, ValueError, KeyError, TypeError, UsageError,
            SkillsLockError) as exc:
        raise _journal_error(project, f"cannot be read ({exc})") from None
    return journal


def _same_device(a: Path, b: Path) -> bool:
    return os.lstat(a).st_dev == os.lstat(b).st_dev


def _move(src: Path, dst: Path) -> None:
    """Move a folder to ``dst`` (which must not exist), in one rename where possible.

    Across devices the folder is copied beside ``dst``, checked file by file and renamed
    into place; then ``src`` is renamed aside (``.<name>.partial`` in its own directory,
    one step) and only then deleted. A crash leaves ``src`` whole or gone, never half.
    """
    if os.path.lexists(dst):
        raise UsageError(f"{dst} already exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _same_device(src.parent, dst.parent):
        os.rename(src, dst)
        return
    partial = dst.with_name(f".{dst.name}{_PARTIAL}")
    _rmtree(partial)
    shutil.copytree(src, partial, symlinks=True)
    for rel, _ in _walk(src)[0].items():
        if _sha256_file(src / rel) != _sha256_file(partial / rel) or \
                os.lstat(src / rel).st_mode != os.lstat(partial / rel).st_mode:
            raise IntegrityError(f"copying {rel} to {dst} did not reproduce it")
    os.rename(partial, dst)
    fsync_dir(dst.parent)
    aside = src.with_name(f".{src.name}{_PARTIAL}")
    _rmtree(aside)
    os.rename(src, aside)
    fsync_dir(src.parent)
    shutil.rmtree(aside)


def move_folder(project: Project, src: Union[str, "os.PathLike[str]"],
                dst: Union[str, "os.PathLike[str]"]) -> None:
    """Move the folder ``src`` to ``dst``, both inside the project, in one rename where
    possible. The lock, the manifest and the journal are not touched.

    For moves nothing needs to record: putting a staged folder in place when the lock
    already describes it (``install --frozen``, where the lock never changes), or moving a
    folder out of the skills root into a backup. A change the lock must follow goes
    through :func:`swap`, whose journal lets :func:`recover` settle a crash.

    ``dst`` must not exist; its parent directories are created. Across devices the folder
    is copied beside ``dst`` (as ``.<name>.partial``), checked file by file, bytes and mode,
    and renamed into place; only then is ``src`` renamed aside and deleted. A crash leaves
    ``src`` whole, or gone with ``dst`` complete, never a half-copied ``dst``. A
    ``.<name>.partial`` left in the skills root is removed by :func:`recover` and
    :func:`discard_staging`; one under ``.decimal/staging/<txid>/`` goes with its staging.

    Raises:
        UsageError: ``dst`` exists, ``src`` is not a folder, or a path is outside the project.
        IllegalEntryError: ``src`` or ``dst``, or a directory on the way to either, is a
            symlink.
        IntegrityError: a copy across devices did not reproduce a file.
    """
    source, target = Path(src), Path(dst)
    project.refuse_symlinks(source)
    project.refuse_symlinks(target)
    try:
        is_dir = stat.S_ISDIR(os.lstat(source).st_mode)
    except FileNotFoundError:
        is_dir = False
    if not is_dir:
        raise UsageError(f"`{project.rel(source)}` is not a folder", path=source)
    if os.path.lexists(target):
        raise UsageError(f"`{project.rel(target)}` already exists", path=target)
    project.makedirs(target.parent)
    _move(source, target)


def _carry_hand_placed(project: Project, txid: str, name: str, target: Path,
                       staged: Optional[Path], old_entry: Optional[Mapping[str, Any]],
                       new_entry: Optional[Mapping[str, Any]],
                       warn: Callable[[str], None]) -> None:
    """Keep the files a user placed by hand where the lock lists an unavailable file.

    Still unavailable in the new entry: copied into the staged folder (bytes and mode).
    Served by the new entry, not listed by it at all, or the skill removed (``new_entry``
    None, no staged folder): the hand-placed copy does not go into a new folder; unless
    it equals the served bytes it is saved to ``.decimal/backup/<txid>/<name>/`` and
    ``warn`` names it. Runs before the journal, so only ``.decimal/`` changes.
    """
    on_disk, why = _walk(target)
    if why:
        raise IllegalEntryError(f"`{project.rel(target)}`: {why}", skill=name, path=target)
    new: Mapping[str, Any] = new_entry or {}
    still = {u.get("path") for u in new.get("unavailable") or ()} if staged is not None else set()
    was = {u.get("path") for u in (old_entry or {}).get("unavailable") or ()}
    served: Mapping[str, str] = new.get("files") or {}
    for rel in sorted(on_disk):
        source = target.joinpath(*rel.split("/"))
        if rel in still and staged is not None:
            dest = staged.joinpath(*rel.split("/"))
            if not os.path.lexists(dest):
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
        elif rel in was:
            if rel in served and _sha256_file(source) == served[rel]:
                continue
            backup = project.backup_dir / txid / name / rel
            project.makedirs(backup.parent)
            shutil.copy2(source, backup)
            version = new.get("version")
            if new_entry is None:
                why = "the skill is removed with its folder"
            elif rel in served:
                why = f"version {version} serves it now, so the served file replaces it"
            else:
                why = f"version {version} no longer has that file"
            warn(f"{name}: `{rel}` was placed by hand; {why}. Your copy is saved in "
                 f"`{project.rel(backup)}`")


def swap(project: Project, txid: str, name: str, new_entry: Optional[Mapping[str, Any]],
         lock: "_lock.Lock", *,
         on_warning: Optional[Callable[[str], None]] = None) -> "_lock.Lock":
    """Replace ``<root>/<name>`` with its staged folder (or remove it when ``new_entry`` is
    None) and commit ``new_entry`` to the lock. Returns the new lock.

    The folder must have been staged with :func:`stage` under the same ``txid``. A crash
    at any point is settled by :func:`recover`. Hand-placed unavailable files are carried
    over or backed up (see the module docstring); each warning goes to ``on_warning``,
    or is logged on ``decimalai.skillpkg`` when there is none.

    Raises:
        JournalConflictError: a journal from an earlier run is still there: run
            :func:`recover` first.
        IllegalEntryError: ``<root>/<name>`` is a symlink or not a directory.
        UsageError: nothing was staged for ``name``.
    """
    target = project.skill_dir(name)
    txdir = _txid_dir(project, txid)
    staged = txdir / target.name
    old_dir = txdir / _OLD / target.name
    _check_root(project)
    project.refuse_symlinks(target)
    project.refuse_symlinks(old_dir)
    if os.path.lexists(project.journal_path):
        raise _journal_error(project, "is left from an interrupted run", skill=name)
    if new_entry is not None and not staged.is_dir():
        raise UsageError(f"{name} was not staged under {txid}", skill=name)
    old_present = os.path.lexists(target)
    if old_present and _illegal_entry(target):
        raise IllegalEntryError(f"`{project.rel(target)}` is {_illegal_entry(target)}",
                                skill=name, path=target)
    old_entry = lock.skills.get(name)
    if old_present:
        _carry_hand_placed(project, txid, name, target,
                           staged if new_entry is not None else None, old_entry, new_entry,
                           on_warning or _log.warning)
    journal = Journal(txid, name, None if old_entry is None else dict(old_entry),
                      None if new_entry is None else dict(new_entry), old_present)
    project.write_file(project.journal_path,
                       (json.dumps(journal.to_dict(), indent=2, sort_keys=True) + "\n").encode())
    _checkpoint("journal")
    if old_present:
        _move(target, old_dir)
    _checkpoint("old_moved")
    if new_entry is not None:
        _move(staged, target)
    _checkpoint("new_moved")
    new_lock = lock.with_entry(name, new_entry)
    _lock.save(project, new_lock)
    _checkpoint("lock_written")
    os.unlink(project.journal_path)
    fsync_dir(project.state_dir)
    return new_lock


# ── recovery ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Recovery:
    """What :func:`recover` did: which skill, and whether its ``old`` or ``new`` state stands."""

    txid: str
    name: str
    outcome: str  # "old" or "new"
    lock: Optional["_lock.Lock"]  # as ``load(repair=True)`` reads it; None if it cannot


def _matches(path: Path, entry: Optional[Mapping[str, Any]]) -> bool:
    if entry is None:
        return not os.path.lexists(path)
    return check_folder(path, entry).state is State.CLEAN


def _files(folder: Path) -> Dict[str, str]:
    """Every file under ``folder`` (clutter left out): relative path -> sha256."""
    return {rel: _sha256_file(folder / rel) for rel in _walk(folder)[0]}


def _replace(target: Path, source: Path) -> None:
    """Put ``source`` where ``target`` is: ``target`` is renamed aside first, then deleted."""
    aside = target.with_name(f".{target.name}{_PARTIAL}")
    _rmtree(aside)
    os.rename(target, aside)
    _move(source, target)
    shutil.rmtree(aside)


def recover(project: Project) -> Optional[Recovery]:
    """Settle a swap a crashed run left half done. Returns None when there was none.

    The folder decides, and the lock is made to agree with it:

    * it matches the new entry → the new entry is committed;
    * it still matches the old entry, or nothing had moved yet → the old entry stands;
    * it is gone and the old folder waits in ``.old`` → the old folder is put back;
    * otherwise → conflict, and nothing is touched.

    Only the journal's one entry of the lock is changed, and no other entry is checked,
    so a lock the crashed run had read with ``repair=True`` does not block recovery.
    Then the journal, the transaction's staging and any ``.<name>.partial`` debris in the
    root are removed, and ``plugin.json`` is brought in line with the lock (unless it is
    a plugin.json this engine did not write: that is left for the next command to refuse
    or, with ``--force``, back up). With no journal, only the debris is removed. Safe to run again after a crash of its own.

    Raises:
        JournalConflictError: the folder matches neither entry, or the journal is unreadable.
    """
    journal = read_journal(project)
    if journal is None:
        _remove_debris(project)
        return None
    name = journal.name
    target = project.skill_dir(name)
    txdir = _txid_dir(project, journal.txid)
    old_dir = txdir / _OLD / target.name
    _check_root(project)
    project.refuse_symlinks(target)
    project.refuse_symlinks(old_dir)
    old_waits = os.path.lexists(old_dir)
    present = os.path.lexists(target)

    if _matches(target, journal.new_entry):
        outcome = "new"
    elif journal.old_entry is not None and _matches(target, journal.old_entry):
        # Still the old folder. A copy across devices may also wait in .old; when the two
        # differ, the folder in the root lost files the .old copy still has (a CLEAN
        # folder may lack unavailable files), so the complete copy is put back.
        if old_waits and _matches(old_dir, journal.old_entry):
            here, there = _files(target), _files(old_dir)
            if here != there:
                if not here.items() <= there.items():
                    raise _journal_error(
                        project, "records an interrupted change, and the folder and its "
                                 "copy in .old differ in a way recovery cannot settle",
                        skill=name)
                _replace(target, old_dir)
        outcome = "old"
    elif old_waits and not present:
        _move(old_dir, target)
        outcome = "old"
    elif not old_waits and (journal.old_present or not present):
        # The old folder never moved (or there was none): whatever is there is the old state.
        outcome = "old"
    else:
        raise _journal_error(
            project, "records an interrupted change, and the folder matches neither the old "
                     "nor the new version", skill=name)

    entry = journal.new_entry if outcome == "new" else journal.old_entry
    # Only this entry changes. The lock may hold damage the crashed run was repairing
    # (merged entries, a stale lock_hash): it is left for the next command to report.
    lock_hash = _lock.commit_entry(project, name, entry)
    if lock_hash is not None:
        if _lock.plugin_json_is_ours(project):  # a user's own plugin.json is never replaced here
            _lock.write_plugin_json(project, lock_hash)
    with contextlib.suppress(FileNotFoundError):
        os.unlink(project.journal_path)
    fsync_dir(project.state_dir)
    discard_staging(project, journal.txid)  # and any .<name>.partial debris
    try:
        settled: Optional[_lock.Lock] = _lock.load(project, missing_ok=True, repair=True)
    except SkillsLockError:
        settled = None
    return Recovery(journal.txid, name, outcome, settled)


# ── one run at a time ────────────────────────────────────────────────────────

#: Lock file -> token of the ProcessLock in this process that holds it, so a lock carrying
#: our own pid can be told apart from one a dead predecessor with the same pid left
#: behind (a restarted container).
_HELD: Dict[str, str] = {}


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) terminates the process on Windows. Without a safe check, a
        # Windows lock is taken over by age alone.
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # EPERM: alive, owned by someone else
        return True
    return True


class ProcessLock:
    """``.decimal/.skills.lck``: one ``decimalai skills`` run per project at a time.

    The file is created with ``O_CREAT | O_EXCL`` and holds ``<pid> <token>``. A lock
    whose pid is dead, or that is older than ``stale_after`` seconds, is taken over;
    otherwise acquiring waits up to ``timeout`` seconds and then raises
    :class:`~decimalai.skillpkg.errors.SkillsBusyError`. A long run calls
    :meth:`refresh` to keep its lock young. Use it as a context manager::

        with fs.ProcessLock(project):
            fs.recover(project)
            ...
    """

    def __init__(self, project: Project, *, timeout: float = 30.0, stale_after: float = 600.0,
                 poll: float = 0.05) -> None:
        self.project = project
        self.path = project.process_lock_path
        self.timeout = timeout
        self.stale_after = stale_after
        self.poll = poll
        self._token = secrets.token_hex(8)
        self.held = False

    def __enter__(self) -> "ProcessLock":
        return self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self.release()

    def _read(self) -> Tuple[Optional[int], Optional[str], float]:
        """``(pid, token, age in seconds)``; pid and token are None when unreadable."""
        st = os.lstat(self.path)
        age = time.time() - st.st_mtime
        try:
            parts = self.path.read_text(encoding="utf-8").split()
            return int(parts[0]), parts[1] if len(parts) > 1 else None, age
        except (OSError, ValueError, IndexError):
            return None, None, age

    def _is_stale(self) -> bool:
        try:
            pid, token, age = self._read()
        except FileNotFoundError:
            return False  # released meanwhile: just try again
        if age > self.stale_after:
            return True
        if pid is None:
            return False  # being written right now, or garbage: only age decides
        if pid == os.getpid():
            return _HELD.get(str(self.path)) != token
        return not _pid_alive(pid)

    def _break(self) -> None:
        """Remove a stale lock, under a short-lived second lock so two breakers cannot
        both remove it (the second would remove the first's fresh lock)."""
        breaker = self.path.with_name(self.path.name + ".break")
        try:
            fd = os.open(breaker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            with contextlib.suppress(FileNotFoundError):
                if time.time() - os.lstat(breaker).st_mtime > 10:
                    os.unlink(breaker)
            time.sleep(self.poll)
            return
        try:
            os.close(fd)
            if self._is_stale():
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(self.path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(breaker)

    def acquire(self) -> "ProcessLock":
        """Take the lock, waiting up to ``timeout`` seconds. Returns ``self``."""
        if self.held:
            return self
        self.project.makedirs(self.path.parent)
        self.project.refuse_symlinks(self.path)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                if os.path.islink(self.path):
                    raise IllegalEntryError(
                        f"`{self.project.rel(self.path)}` is a symlink", path=self.path) from None
                if self._is_stale():
                    self._break()
                    continue
                if time.monotonic() >= deadline:
                    pid = None
                    with contextlib.suppress(OSError):
                        pid = self._read()[0]
                    holder = f" (pid {pid})" if pid is not None else ""
                    raise SkillsBusyError(
                        f"another `decimalai skills` run{holder} holds "
                        f"`{self.project.rel(self.path)}`", path=self.path) from None
                time.sleep(self.poll)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(f"{os.getpid()} {self._token}\n")
            _HELD[str(self.path)] = self._token
            self.held = True
            return self

    def _let_go(self) -> None:
        self.held = False
        if _HELD.get(str(self.path)) == self._token:
            del _HELD[str(self.path)]

    def refresh(self) -> None:
        """Mark the lock as fresh, so a long run is not taken for a dead one.

        Raises:
            SkillsBusyError: another run took the lock over while this one was stalled
                (past ``stale_after``). This run no longer holds it and must stop
                writing; ``held`` is False.
        """
        if not self.held:
            return
        try:
            _, token, _ = self._read()
        except FileNotFoundError:
            token = None
        if token != self._token:
            self._let_go()
            raise SkillsBusyError(
                f"another `decimalai skills` run took over `{self.project.rel(self.path)}` "
                "while this one was stalled; this run stopped so the two cannot interleave",
                path=self.path)
        os.utime(self.path)

    def release(self) -> None:
        """Remove the lock, if it is still ours."""
        if not self.held:
            return
        self._let_go()
        try:
            _, token, _ = self._read()
        except FileNotFoundError:
            return
        if token == self._token:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self.path)
