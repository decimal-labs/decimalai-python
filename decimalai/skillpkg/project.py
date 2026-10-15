"""Where a project's managed skills live, and how the engine writes a project file.

A project is the directory that holds ``.decimal/``::

    <project>/
      .agents/
        .claude-plugin/plugin.json      the Claude Code plugin root (``--plugin-dir <abs>/.agents``)
        skills/<install-name>/          one real directory per skill, never a symlink
      .decimal/
        skills.toml                     the manifest: what the project wants      (commit)
        skills.lock                     the lock: exactly what is installed        (commit)
        install.json, staging/, backup/, migrated-v1/,
        skills.lock.journal, .skills.lck                                         (gitignored)

The root is always ``.agents/skills``: ``plugin.json`` only works when the root is the
plugin's ``skills/`` folder, and one root is what the native-loader check grades.

Two rules hold for every path below the project directory: the engine never follows a
symlink there (:meth:`Project.refuse_symlinks`), and it replaces a file only by writing
a temporary file beside it and renaming it into place (:meth:`Project.write_file`), so
a reader sees the old bytes or the new ones.
"""
from __future__ import annotations

import contextlib
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional, Union

from .errors import IllegalEntryError, SkillsNotInstalledError, UsageError

__all__ = [
    "GITIGNORE_LINES",
    "PLUGIN_NAME",
    "ROOT",
    "STATE_DIR",
    "Project",
    "find_project",
    "skills_dir",
]

PathLike = Union[str, "os.PathLike[str]"]

#: The managed root, relative to the project. The only root this engine writes.
ROOT = ".agents/skills"
#: The name Claude Code loads ``.agents`` under (``decimalai-skills:<install-name>``). It
#: is not ``decimalai``: that is the marketplace plugin's name, and two plugins with one
#: name replace each other.
PLUGIN_NAME = "decimalai-skills"
STATE_DIR = ".decimal"

#: What ``.decimal/.gitignore`` lists: everything the engine keeps under ``.decimal/``
#: except the manifest and the lock, which are committed.
GITIGNORE_LINES = (
    "install.json",
    "staging/",
    "backup/",
    "migrated-v1/",
    "skills.lock.journal",
    ".skills.lck",
)


@dataclass(frozen=True)
class Project:
    """A project directory and the paths the engine uses inside it.

    Build one with :meth:`at` (this directory) or :meth:`find` (search upwards). The
    directory is absolute and symlink-resolved, so every path derived from it is too.
    """

    directory: Path

    @classmethod
    def at(cls, directory: PathLike) -> "Project":
        """The project rooted exactly at ``directory`` (``--directory DIR``)."""
        return cls(Path(directory).resolve())

    @classmethod
    def find(cls, start: Optional[PathLike] = None) -> "Project":
        """The project ``start`` (default: the working directory) belongs to.

        The search walks up from ``start`` and stops at the repository: the first
        directory with a ``.git`` entry (a directory, or the file a git worktree has). It
        stops below the home directory: home is the project only when ``start`` is home
        itself. Within that range the project is the nearest directory holding
        ``.decimal/skills.toml`` or ``.decimal/skills.lock``, else the nearest holding any
        ``.decimal/``, else the repository root, else ``start``.

        So neither a folder of repositories with a ``.decimal/`` of its own above them, nor
        a home directory (where running the SDK leaves ``.decimal/install.json``, and where
        ``~/.claude/skills`` holds personal skills), is mistaken for the project of a
        directory inside it.
        """
        base = Path(start if start is not None else os.getcwd()).resolve()
        home = _home()
        candidates = []
        for candidate in (base, *base.parents):
            if candidate == home and candidate != base:
                break
            candidates.append(candidate)
            if os.path.lexists(candidate / ".git") or candidate == home:
                break
        for candidate in candidates:
            state = candidate / STATE_DIR
            if (state / "skills.toml").is_file() or (state / "skills.lock").is_file():
                return cls(candidate)
        for candidate in candidates:
            if (candidate / STATE_DIR).is_dir():
                return cls(candidate)
        top = candidates[-1]
        return cls(top if os.path.lexists(top / ".git") else base)

    # ── paths ────────────────────────────────────────────────────────────────

    @property
    def state_dir(self) -> Path:
        return self.directory / STATE_DIR

    @property
    def manifest_path(self) -> Path:
        return self.state_dir / "skills.toml"

    @property
    def lock_path(self) -> Path:
        return self.state_dir / "skills.lock"

    @property
    def npx_lock_path(self) -> Path:
        """The lock the npx installer writes; its presence means ``skills migrate`` is due."""
        return self.state_dir / "skills.lock.json"

    @property
    def journal_path(self) -> Path:
        return self.state_dir / "skills.lock.journal"

    @property
    def process_lock_path(self) -> Path:
        return self.state_dir / ".skills.lck"

    @property
    def staging_dir(self) -> Path:
        return self.state_dir / "staging"

    @property
    def backup_dir(self) -> Path:
        return self.state_dir / "backup"

    @property
    def migrated_dir(self) -> Path:
        return self.state_dir / "migrated-v1"

    @property
    def agents_dir(self) -> Path:
        return self.directory / ".agents"

    @property
    def skills_root(self) -> Path:
        return self.directory.joinpath(*PurePosixPath(ROOT).parts)

    @property
    def plugin_json_path(self) -> Path:
        return self.agents_dir / ".claude-plugin" / "plugin.json"

    def skill_dir(self, name: str) -> Path:
        """``<root>/<name>``. ``name`` must be one path component; anything else is refused."""
        if (not isinstance(name, str) or name in ("", ".", "..") or "/" in name
                or "\\" in name or "\x00" in name):
            raise UsageError(f"{name!r} is not a folder name", skill=str(name))
        return self.skills_root / name

    def rel(self, path: PathLike) -> str:
        """``path`` relative to the project, with ``/`` separators: for messages and the lock."""
        return Path(path).relative_to(self.directory).as_posix()

    # ── safety ───────────────────────────────────────────────────────────────

    def refuse_symlinks(self, path: PathLike) -> None:
        """Refuse ``path`` if it, or any directory between the project and it, is a symlink.

        Checked with ``lstat``, component by component; components that do not exist
        yet are fine. The project directory itself, and anything above it, may be a
        link (``/tmp`` is one on macOS).

        Raises:
            IllegalEntryError: a component is a symlink.
            UsageError: ``path`` is not inside the project.
        """
        target = Path(path)
        try:
            parts = target.relative_to(self.directory).parts
        except ValueError:
            raise UsageError(f"{target} is not inside the project {self.directory}") from None
        current = self.directory
        for part in parts:
            current = current / part
            try:
                mode = os.lstat(current).st_mode
            except FileNotFoundError:
                return
            if stat.S_ISLNK(mode):
                raise IllegalEntryError(f"`{self.rel(current)}` is a symlink", path=current)

    def makedirs(self, path: PathLike) -> None:
        """Create ``path`` and its parents inside the project, never through a symlink."""
        self.refuse_symlinks(path)
        Path(path).mkdir(parents=True, exist_ok=True)

    def write_file(self, path: PathLike, data: bytes, *, mode: int = 0o644) -> None:
        """Replace ``path`` with exactly ``data``, atomically.

        The bytes go to a temporary file in the same directory, are flushed to disk and
        renamed over ``path``: a crash leaves the old file or the new one. The mode is
        set explicitly, so it does not depend on the umask.
        """
        target = Path(path)
        self.makedirs(target.parent)
        self.refuse_symlinks(target)
        fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp, mode)
            os.replace(tmp, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        fsync_dir(target.parent)


def _home() -> Optional[Path]:
    try:
        return Path.home().resolve()
    except (RuntimeError, KeyError, OSError):  # no home directory can be determined
        return None


def fsync_dir(directory: PathLike) -> None:
    """Flush a directory entry change (a rename) to disk, where the platform allows it."""
    if os.name != "posix":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def find_project(start: Optional[PathLike] = None) -> Project:
    """The project ``start`` belongs to; see :meth:`Project.find`."""
    return Project.find(start)


def skills_dir(directory: Optional[PathLike] = None, *,
               relative_to: Optional[PathLike] = None) -> Path:
    """The project's managed skills root, ``<project>/.agents/skills``.

    Point a framework's skill loader at it::

        import decimalai

        decimalai.skills_dir()                        # /abs/path/to/project/.agents/skills
        decimalai.skills_dir(relative_to=Path.cwd())  # .agents/skills

    Args:
        directory: where to start looking for the project (default: the working
            directory); see :meth:`Project.find`.
        relative_to: return the root relative to this directory instead, for loaders
            that read paths relative to a working directory or workspace.

    Returns:
        The absolute, symlink-resolved root; or, with ``relative_to``, the root relative
        to that directory (both sides resolved first).

    Raises:
        SkillsNotInstalledError: the project has no ``.decimal/skills.lock`` or no root.
        LegacyLockError: the lock is a version 1 lock (or has no version), or npx's lock
            is beside it: run ``decimalai skills migrate``.
        SkillsLockError: the lock is not JSON, or has an unknown version or root.
        IllegalEntryError: ``.agents`` or ``.agents/skills`` is a symlink, or the root is
            not a directory.
        UsageError: with ``relative_to``, the root is not under that directory.
    """
    from .lock import check_version  # lock imports this module

    project = Project.find(directory)
    root = project.skills_root
    project.refuse_symlinks(root)
    # Only a version 2 lock means this engine manages the root; the entries are not read.
    check_version(project)
    if not os.path.lexists(root):
        raise SkillsNotInstalledError(
            f"`{ROOT}` does not exist in {project.directory}", path=root)
    if not root.is_dir():
        raise IllegalEntryError(f"`{ROOT}` is not a directory", path=root)
    if relative_to is None:
        return root
    base = Path(relative_to).resolve()
    try:
        return root.relative_to(base)
    except ValueError:
        raise UsageError(
            f"the skills root {root} is not under {base}",
            path=root,
            fix="call `skills_dir(relative_to=...)` with a directory that contains the project",
        ) from None
