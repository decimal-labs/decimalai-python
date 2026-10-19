"""Agent skill directories, and the deprecated writers that put registry skills in them.

``AGENT_PATHS`` maps each agent runtime to the folder it reads ``SKILL.md`` files from.
The SDK uses it to detect a runtime and to find skills; nothing writes there on its own.

Registry skills reach disk through the package manager: ``decimalai skills add <name>``
(or ``decimalai.skillpkg.add()``) installs the folder the registry renders under
``.agents/skills`` and pins it in ``.decimal/skills.lock``. :func:`export_skill_to_disk`
and :func:`export_skills_to_disk` are kept for code written against 0.14 and earlier, and
warn once that they are deprecated:

- they render ``SKILL.md`` locally with ``skillevaluation.skillfmt``, the module the
  registry renders with: format 1, with the registry's stamp under ``metadata`` and
  ``allowed-tools`` and ``compatibility`` written as strings. The skill dict must carry
  its registry ids (``id`` and a version id), because the stamp is made of them;
- they place each folder whole, the way the engine does: staged under
  ``.decimal/staging/``, checked, then renamed into place, never through a symlink;
- the folders they write are unmanaged: no lock entry vouches for them. They write
  a lock only where a version 1 lock already exists (one an older SDK wrote), and keep
  it up to date so ``decimalai skills migrate`` can convert it. A project with no lock
  gets none, so a fresh project never starts with legacy state, and a version 2 lock
  is never written by them.

Based on the agentskills.io / npx skills conventions:
https://github.com/vercel-labs/skills
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shlex
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple

if TYPE_CHECKING:  # skillevaluation is imported when a writer runs, not with this module
    from skillevaluation.skillfmt import Attachment, RenderFields

    from .skillpkg.project import Project

logger = logging.getLogger("decimalai.disk_export")

#: Writers that warned already in this process.
_warned_once: set = set()


def _warn_deprecated(key: str, message: str) -> None:
    """Warn once per process that the writer ``key`` is deprecated."""
    if key in _warned_once:
        return
    _warned_once.add(key)
    warnings.warn(message, DeprecationWarning, stacklevel=3)



# ── Agent Path Registry ───────────────────────────────────────
# Maps agent name → (project-local path, global path)
# Based on npx skills conventions for 43+ agent runtimes.

AGENT_PATHS: Dict[str, Dict[str, str]] = {
    # Major coding agents
    "claude-code":      {"project": ".claude/skills",      "global": "~/.claude/skills"},
    "cursor":           {"project": ".agents/skills",      "global": "~/.cursor/skills"},
    "github-copilot":   {"project": ".agents/skills",      "global": "~/.copilot/skills"},
    "codex":            {"project": ".agents/skills",      "global": "~/.codex/skills"},
    "gemini-cli":       {"project": ".agents/skills",      "global": "~/.gemini/skills"},
    "windsurf":         {"project": ".windsurf/skills",    "global": "~/.windsurf/skills"},
    "cline":            {"project": ".agents/skills",      "global": "~/.agents/skills"},
    "continue":         {"project": ".continue/skills",    "global": "~/.continue/skills"},
    "warp":             {"project": ".agents/skills",      "global": "~/.agents/skills"},
    "antigravity":      {"project": ".agents/skills",      "global": "~/.gemini/antigravity/skills"},
    # Additional agents
    "aider":            {"project": ".agents/skills",      "global": "~/.aider/skills"},
    "amp":              {"project": ".amp/skills",         "global": "~/.amp/skills"},
    "augment":          {"project": ".augment/skills",     "global": "~/.augment/skills"},
    "codebuddy":        {"project": ".codebuddy/skills",   "global": "~/.codebuddy/skills"},
    "command-code":     {"project": ".commandcode/skills", "global": "~/.commandcode/skills"},
    "cortex":           {"project": ".cortex/skills",      "global": "~/.snowflake/cortex/skills"},
    "crush":            {"project": ".crush/skills",       "global": "~/.config/crush/skills"},
    "deepagents":       {"project": ".agents/skills",      "global": "~/.deepagents/agent/skills"},
    "droid":            {"project": ".factory/skills",     "global": "~/.factory/skills"},
    "firebender":       {"project": ".agents/skills",      "global": "~/.firebender/skills"},
    "goose":            {"project": ".agents/skills",      "global": "~/.config/goose/skills"},
    "kilo-code":        {"project": ".agents/skills",      "global": "~/.kilo-code/skills"},
    "melty":            {"project": ".melty/skills",       "global": "~/.melty/skills"},
    "otto":             {"project": ".agents/skills",      "global": "~/.otto/skills"},
    "pear":             {"project": ".agents/skills",      "global": "~/.pear/skills"},
    "roo":              {"project": ".roo/skills",         "global": "~/.roo/skills"},
    "supermaven":       {"project": ".agents/skills",      "global": "~/.supermaven/skills"},
    "tabnine":          {"project": ".agents/skills",      "global": "~/.tabnine/skills"},
    "trae":             {"project": ".trae/skills",        "global": "~/.trae/skills"},
    "void":             {"project": ".void/skills",        "global": "~/.void/skills"},
    "zed":              {"project": ".agents/skills",      "global": "~/.zed/skills"},
    # Universal fallback (covers most .agents/skills readers)
    "universal":        {"project": ".agents/skills",      "global": "~/.config/agents/skills"},
}


def get_agent_paths(
    agent: str,
    scope: str = "project",
    project_root: Optional[str] = None,
) -> str:
    """Resolve the disk path for a given agent and scope.

    Args:
        agent: Agent runtime name (e.g., 'claude-code', 'cursor').
        scope: 'project' for project-local, 'global' for user-global.
        project_root: Project root directory (used when scope='project').
                      Defaults to current working directory.

    Returns:
        Resolved absolute path to the skills directory.

    Raises:
        ValueError: If the agent name is not recognized.
    """
    if agent not in AGENT_PATHS:
        raise ValueError(
            f"Unknown agent '{agent}'. "
            f"Known agents: {', '.join(sorted(AGENT_PATHS.keys()))}"
        )

    paths = AGENT_PATHS[agent]
    raw_path = paths[scope]

    if raw_path.startswith("~"):
        return os.path.expanduser(raw_path)
    elif scope == "project":
        root = project_root or os.getcwd()
        return os.path.join(root, raw_path)
    else:
        return os.path.expanduser(raw_path)


def list_supported_agents() -> List[str]:
    """Return all supported agent runtime names."""
    return sorted(AGENT_PATHS.keys())



# ── SKILL.md for a body that has none ─────────────────────────


_YAML_SAFE_PLAIN = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ._/()+-"
)


def _yaml_scalar(value: str) -> str:
    """Render a string as a safe YAML scalar.

    Plain style when it can't be misparsed; otherwise a JSON string, which
    is valid YAML (double-quoted flow scalar) and round-trips through
    yaml.safe_load exactly.
    """
    value = "" if value is None else str(value)
    if (
        value
        and not value[0].isspace()
        and not value[-1].isspace()
        and all(c in _YAML_SAFE_PLAIN for c in value)
        and value.lower() not in ("true", "false", "null", "yes", "no", "on", "off")
        # Leading indicator chars parse as structure, not a scalar, under
        # strict YAML ("- x" = sequence entry, "? x"/": x" = mapping key,
        # "---" = document fence) even when every char is individually safe.
        and not value.startswith(("- ", "? ", ": "))
        and value not in ("-", "?", ":", "---", "...")
    ):
        return value
    return json.dumps(value, ensure_ascii=False)


def _joined(value: Any, sep: str) -> str:
    """A string, or a list of strings joined with ``sep``; never one item per character."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return sep.join(str(v).strip() for v in value if str(v).strip())


def _reconstruct_skill_md(
    name: str,
    description: str,
    body_markdown: str,
    *,
    license_info: Optional[str] = None,
    compatibility: Any = None,
    allowed_tools: Any = None,
    category: Optional[str] = None,
    skill_type: Optional[str] = None,
    skill_scope: Optional[str] = None,
    invocation: Optional[str] = None,
) -> str:
    """Frontmatter for a skill body that arrived without any.

    For a body a person wrote, not for a registry skill: the registry serves the whole
    ``SKILL.md`` (``decimalai skills add`` and ``skills pull`` write its bytes). Not the
    registry's format either: there is no stamp, and the taxonomy labels are top-level
    keys, as older SDKs wrote them.

    ``compatibility`` and ``allowed_tools`` may each be a string or a list of strings;
    both are written as one string (a list joined with ``"; "`` and spaces), because
    loaders refuse or drop a list there.

    Args:
        name: Skill name.
        description: Short description.
        body_markdown: Full body content.
        license_info: Optional license (e.g., 'MIT').
        compatibility: Optional requirements.
        allowed_tools: Optional allowed tools.
        category: Optional category grouping.
        skill_type: Optional taxonomy class ('capability' | 'preference').
        skill_scope: Optional visibility scope ('public' | 'private').
        invocation: Optional invocation mode ('model' | 'user' | 'any') —
            ``invocation: user`` additionally emits
            ``disable-model-invocation: true`` (Claude Code's spelling).

    Returns:
        Complete SKILL.md file content.
    """
    # Values are YAML-escaped: an unquoted `description:` containing ':' (e.g.
    # "Google Classroom: Manage classes…") is invalid YAML, and the file is then
    # skipped by discover_skills() on re-parse.
    lines = ["---"]
    lines.append(f"name: {_yaml_scalar(name)}")
    lines.append(f"description: {_yaml_scalar(description)}")
    if category:
        lines.append(f"category: {_yaml_scalar(category)}")
    if skill_type:
        lines.append(f"skill-type: {_yaml_scalar(skill_type)}")
    if skill_scope:
        lines.append(f"skill-scope: {_yaml_scalar(skill_scope)}")
    if invocation:
        # Normalize case on the way out ("User" → "user") so re-imports and
        # platform CHECK constraints see the canonical spelling.
        invocation = str(invocation).strip().lower()
        lines.append(f"invocation: {_yaml_scalar(invocation)}")
        if invocation == "user":
            # Claude Code's spelling of "not model-invocable"; other runtimes ignore it.
            lines.append("disable-model-invocation: true")
    if license_info:
        lines.append(f"license: {_yaml_scalar(license_info)}")
    compat = _joined(compatibility, "; ")
    if compat:
        lines.append(f"compatibility: {_yaml_scalar(compat)}")
    tools = _joined(allowed_tools, " ")
    if tools:
        lines.append(f"allowed-tools: {_yaml_scalar(tools)}")
    lines.append("---")
    lines.append("")

    # Append body
    lines.append(body_markdown)

    return "\n".join(lines)


# ── Lockfile ──────────────────────────────────────────────────

LOCKFILE_NAME = ".decimal/skills.lock"


def _read_lockfile(project_root: str) -> Dict[str, Any]:
    """Read a version 1 skills lockfile; an empty one when it is missing or unreadable.

    Only for the version 1 lock this module writes. A project managed by
    ``decimalai skills`` has a version 2 lock, which ``decimalai.skillpkg.lock`` reads.
    """
    lockfile_path = os.path.join(project_root, LOCKFILE_NAME)
    if not os.path.exists(lockfile_path):
        return {"version": 1, "skills": {}}
    try:
        with open(lockfile_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"version": 1, "skills": {}}


def _lock_kind(project_root: str) -> str:
    """What ``.decimal/skills.lock`` is: ``absent``, ``v1``, ``v2`` or ``unreadable``.

    ``v1`` covers a lock with no version, as the oldest SDKs wrote it. ``unreadable`` is
    anything else: not JSON, not a map, an unknown version, or a symlink.
    :func:`export_skills_to_disk` updates only a ``v1`` lock.
    """
    path = os.path.join(project_root, LOCKFILE_NAME)
    if not os.path.lexists(path):
        return "absent"
    if os.path.islink(path):
        return "unreadable"
    try:
        with open(path, "rb") as f:
            data = json.loads(f.read().decode("utf-8"))
    except (OSError, ValueError):
        return "unreadable"
    if not isinstance(data, dict):
        return "unreadable"
    version = data.get("version")
    if version is None or (type(version) is int and version == 1):
        return "v1"
    if type(version) is int and version == 2:
        return "v2"
    return "unreadable"


def _write_lockfile(project_root: str, lockdata: Dict[str, Any]) -> None:
    """Write a version 1 skills lockfile, atomically.

    Never over a lock this module did not write: a version 2 lock (``decimalai skills``
    manages that project) or one it cannot read is left exactly as it is.

    Raises:
        ValueError: ``lockdata`` is not a version 1 lock, or the file on disk is a
            version 2 or unreadable lock.
    """
    if lockdata.get("version", 1) != 1:
        raise ValueError("disk_export writes only version 1 locks")
    kind = _lock_kind(project_root)
    if kind not in ("absent", "v1"):
        what = "a version 2 lock" if kind == "v2" else "a lock it cannot read"
        raise ValueError(f"{LOCKFILE_NAME} in {project_root} is {what}; disk_export never "
                         "writes over it")
    from .skillpkg.project import Project

    project = Project.at(project_root)
    lockdata = {**lockdata, "updated_at": datetime.now(timezone.utc).isoformat()}
    project.write_file(project.lock_path,
                       (json.dumps(lockdata, indent=2) + "\n").encode("utf-8"))


# ── Errors ────────────────────────────────────────────────────


class SkillExportFileExistsError(Exception):
    """Raised when an export would replace a folder or file that already exists.

    Installing a skill never touches files that already exist on disk unless the caller
    opts in with ``force=True``; a folder ``decimalai skills`` manages is never replaced,
    ``force`` or not.
    """

    def __init__(self, path: str, message: Optional[str] = None):
        super().__init__(message or (
            f"Refusing to overwrite existing file: {path}. "
            "Pass force=True to overwrite (e.g., decimalai skills pull --force)."
        ))
        self.path = path


class SkillExportUnsafePathError(ValueError):
    """Raised when a SERVER-supplied skill name or
    attachment file_path would escape the target skill directory (path traversal
    or absolute path). The installer must never trust backend-supplied filenames —
    a malicious/compromised registry skill could otherwise write into ~/.bashrc,
    .git/hooks, cron.d, SSH config, etc. and gain code execution on the installer's
    machine. Subclasses ValueError so existing `except ValueError` handlers catch it.
    """

    def __init__(self, value: str, kind: str = "path"):
        self.value = value
        super().__init__(
            f"Refusing to write outside the skill directory: unsafe {kind} {value!r} "
            "(path traversal / absolute path). The skill source may be malicious."
        )


def _safe_skill_dirname(name: str) -> str:
    """Validate a server-supplied skill name is a single safe path component.

    Skill slugs are `[A-Za-z0-9._-]`; a path separator, a `.`/`..` segment, a null
    byte, or an absolute path is a traversal attempt. Raises rather than silently
    sanitizing so the caller sees exactly why an install was refused.
    """
    if not name or name in (".", "..") or os.path.isabs(name) or "\x00" in name:
        raise SkillExportUnsafePathError(name, "skill name")
    if "/" in name or "\\" in name or ".." in name.replace("\\", "/").split("/"):
        raise SkillExportUnsafePathError(name, "skill name")
    return name


def _safe_join_within(base_dir: str, rel_path: str) -> str:
    """Join ``rel_path`` under ``base_dir`` and confirm the result stays inside it.

    Allows legitimate nested attachment paths (e.g. ``scripts/run.py``) while
    rejecting ``../`` escapes and absolute paths from a malicious registry skill.
    """
    if not rel_path or "\x00" in rel_path:
        raise SkillExportUnsafePathError(rel_path, "attachment path")
    base = os.path.realpath(base_dir)
    target = os.path.realpath(os.path.join(base, rel_path))
    if target != base and not target.startswith(base + os.sep):
        raise SkillExportUnsafePathError(rel_path, "attachment path")
    return target


def _is_traversal(path: str) -> bool:
    """A path that names somewhere outside the folder it is joined to."""
    return (os.path.isabs(path) or path.startswith(("/", "\\")) or "\\" in path
            or "\x00" in path or ".." in path.split("/"))


# ── Reading a skill dict, strictly ────────────────────────────

_NEEDS_IDS = (
    "skill {name!r} has no {what}: disk_export writes the registry's format, whose stamp "
    "is the skill's registry ids (`id`, and `version_id` or `latest_version.id`). To "
    "install a registry skill run `decimalai skills add {quoted}` "
    "(decimalai.skillpkg.add({name!r}))"
)

_GLOBAL_SCOPE = (
    "scope={scope!r} is not supported any more: skills install into the project's "
    "`.agents/skills`, managed by `decimalai skills`. For a personal copy run "
    "`decimalai skills pull {quoted} --out ~/.claude/skills`"
)


def _text(skill: Mapping[str, Any], *keys: str) -> Optional[str]:
    """The first of ``keys`` that is set, as a string; a value of another type is refused."""
    for key in keys:
        value = skill.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"skill field {key!r} must be a string, not {type(value).__name__}")
        return value
    return None


def _strings(skill: Mapping[str, Any], key: str) -> Any:
    """A string, or a list of strings, kept as given; never split into characters."""
    value = skill.get(key)
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return list(value)
    raise ValueError(f"skill field {key!r} must be a string or a list of strings")


def _ident(value: Any) -> Optional[str]:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None


def _version_number(skill: Mapping[str, Any], latest: Mapping[str, Any]) -> int:
    for value in (latest.get("version_number"), skill.get("version")):
        if value is None:
            continue
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return value
        raise ValueError(f"skill version must be a positive integer, not {value!r}")
    return 1


def _render_fields(skill: Mapping[str, Any]) -> Tuple["RenderFields", str]:
    """The render fields of one skill dict, and its folder name. Strict: a field of the
    wrong type is refused, never coerced into something else."""
    from skillevaluation import skillfmt

    if not isinstance(skill, Mapping):
        raise ValueError(f"a skill must be a dict, not {type(skill).__name__}")
    name = skill.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("the skill has no name")
    if _is_traversal(name) or any(seg in ("", ".") for seg in name.split("/")):
        raise SkillExportUnsafePathError(name, "skill name")
    slug = skill.get("url_slug")
    if slug is not None:
        _safe_skill_dirname(slug if isinstance(slug, str) else "")
    latest = skill.get("latest_version")
    latest = latest if isinstance(latest, Mapping) else {}
    skill_id = _ident(skill.get("id")) or _ident(skill.get("skill_id"))
    version_id = (_ident(skill.get("version_id")) or _ident(skill.get("latest_version_id"))
                  or _ident(latest.get("id")))
    for what, value in (("id", skill_id), ("version id", version_id)):
        if value is None:
            raise ValueError(_NEEDS_IDS.format(name=name, what=what, quoted=shlex.quote(name)))
    assert skill_id is not None and version_id is not None
    invocation = _text(skill, "invocation", "invocation_mode")
    invocation = (invocation or "model").strip().lower()
    if invocation not in skillfmt.INVOCATION_MODES:
        raise ValueError(f"skill invocation must be one of {', '.join(skillfmt.INVOCATION_MODES)}, "
                         f"not {invocation!r}")
    fields = skillfmt.RenderFields(
        skill_id=skill_id,
        version_id=version_id,
        version=_version_number(skill, latest),
        registry_name=name,
        body=_text(skill, "body_markdown") or "",
        description=_text(skill, "description"),
        license=_text(skill, "license"),
        license_spdx=_text(skill, "license_spdx"),
        compatibility=_strings(skill, "compatibility"),
        allowed_tools=_strings(skill, "allowed_tools"),
        invocation_mode=invocation,
        category=_text(skill, "category"),
        # The platform spells these snake_case; older payloads used the frontmatter's
        # kebab-case keys.
        skill_type=_text(skill, "skill_type", "skill-type"),
        skill_scope=_text(skill, "skill_scope", "skill-scope"),
    )
    return fields, skillfmt.install_name_for(name, skill_id)


def _attachment(att: Any) -> Optional["Attachment"]:
    """One attachment dict as a bundle file; None for a row with no path."""
    from skillevaluation import skillfmt

    if not isinstance(att, Mapping):
        raise ValueError(f"an attachment must be a dict, not {type(att).__name__}")
    path = att.get("file_path") or att.get("path")
    if not path:
        return None
    if not isinstance(path, str) or _is_traversal(path):
        raise SkillExportUnsafePathError(str(path), "attachment path")
    unsafe = [p for p in skillfmt.validate_attachment_path(path) if p.rule == "F14"]
    if unsafe:
        raise ValueError(unsafe[0].message)
    content = att["content_text"] if "content_text" in att else att.get("content")
    if content is not None and not isinstance(content, str):
        raise ValueError(f"attachment {path!r}: content must be text")
    language = att.get("execution_language")
    return skillfmt.Attachment(
        path=path, content=content,
        executable=bool(att.get("is_executable") or att.get("executable")),
        execution_language=language if isinstance(language, str) else None)


# ── Export to disk (deprecated) ───────────────────────────────


def _managed_names(project: "Project") -> List[str]:
    """Install names a version 2 lock lists; none for any other lock."""
    from .skillpkg import lock as _lock

    try:
        return list(_lock.load(project, missing_ok=True, repair=True).skills)
    except Exception:
        return []


def _folders(project: "Project", install_name: str,
             agents: Optional[List[str]]) -> Tuple[List[Path], List[Path]]:
    """``(one folder per agent, the distinct folders)``: cursor, codex and github-copilot
    all read ``.agents/skills``, and an export for all three writes it once."""
    per_agent: List[Path] = []
    folders: List[Path] = []
    for agent in agents or ["universal"]:
        folder = Path(get_agent_paths(agent, "project", str(project.directory))) / install_name
        per_agent.append(folder)
        if folder not in folders:
            folders.append(folder)
    return per_agent, folders


def _place(
    project: "Project",
    install_name: str,
    contents: Mapping[str, bytes],
    entry: Mapping[str, Any],
    folders: List[Path],
    *,
    replace: Callable[[Path], Optional[str]],
) -> List[str]:
    """Write ``contents`` as each of ``folders``, whole, the way the engine places a folder.

    Every folder is checked before any is written. A folder that exists is replaced
    only as ``replace(folder)`` says: ``"backup"`` moves it to ``.decimal/backup/<txid>/``
    first, ``"discard"`` (a copy this module wrote and nobody edited) drops it, None
    refuses. A folder ``decimalai skills`` manages is never replaced. Returns the
    backups made.
    """
    from .skillpkg import fs
    from .skillpkg.errors import IllegalEntryError

    txid = fs.new_txid()
    backups: List[str] = []
    with fs.ProcessLock(project):
        managed = _managed_names(project)
        how: Dict[Path, Optional[str]] = {}
        for folder in folders:
            try:
                project.refuse_symlinks(folder)
            except IllegalEntryError:
                raise SkillExportUnsafePathError(str(folder), "symlinked path") from None
            if folder == project.skill_dir(install_name) and install_name in managed:
                raise SkillExportFileExistsError(str(folder), (
                    f"Refusing to replace {folder}: `decimalai skills` manages it "
                    "(it is in .decimal/skills.lock). Change it with "
                    f"`decimalai skills update {install_name}`."))
            if not os.path.lexists(folder):
                continue
            if os.path.islink(folder) or not folder.is_dir():
                raise SkillExportFileExistsError(str(folder), (
                    f"Refusing to replace {folder}: it is not a folder."))
            how[folder] = replace(folder)
            if how[folder] is None:
                skill_md = folder / "SKILL.md"
                raise SkillExportFileExistsError(
                    str(skill_md if os.path.lexists(skill_md) else folder))
        try:
            for folder in folders:
                if folder in how:
                    rel = folder.relative_to(project.directory)
                    if how[folder] == "backup":
                        # The folder as it was goes to the backup, never away.
                        aside = project.backup_dir / txid / rel
                        backups.append(str(aside))
                    else:
                        aside = project.staging_dir / txid / ".replaced" / rel
                    fs.move_folder(project, folder, aside)
                staged = fs.stage(project, txid, install_name, contents, entry)
                fs.move_folder(project, staged, folder)
                logger.debug("Wrote %s", folder)
        finally:
            fs.discard_staging(project, txid)
    from . import _install

    _install._ensure_gitignore(project.state_dir)
    return backups


def _export(
    skill: Mapping[str, Any],
    *,
    agents: Optional[List[str]],
    scope: str,
    project_root: Optional[str],
    attachments: Optional[List[Dict[str, Any]]],
    force: bool,
) -> Dict[str, Any]:
    from skillevaluation import skillfmt

    from .skillpkg import lock as _lock
    from .skillpkg.project import Project

    name = skill.get("name") if isinstance(skill, Mapping) else None
    if scope != "project":
        raise ValueError(_GLOBAL_SCOPE.format(scope=scope, quoted=shlex.quote(str(name))))
    fields, install_name = _render_fields(skill)
    files = [f for f in (_attachment(a) for a in (attachments or ())) if f is not None]
    bundle = skillfmt.render_bundle(fields, files, install_name=install_name, variant="foreign")
    contents = {f.path: f.content.encode("utf-8") for f in bundle.files}
    entry = _lock.make_entry(
        install_name,
        name=fields.registry_name,
        skill_id=fields.skill_id,
        version_id=fields.version_id,
        version=fields.version,
        render={"format": bundle.format, "hash": bundle.render_hash, "variant": bundle.variant},
        body_sha256=bundle.body_sha256,
        description_sha256=bundle.description_sha256,
        files={f.path: f.sha256 for f in bundle.files},
        executable=[f.path for f in bundle.files if f.mode == "0755"],
        unavailable=[{"path": u.path, "reason": u.reason} for u in bundle.unavailable],
        source={"base_url": None, "org": None},
        invocation=bundle.invocation,
    )

    project = Project.at(project_root or os.getcwd())
    per_agent, folders = _folders(project, install_name, agents)
    backups = _place(project, install_name, contents, entry, folders,
                     replace=lambda _folder: "backup" if force else None)
    logger.info(
        "Wrote %s to %s as an unmanaged folder (no lock entry vouches for it); "
        "`decimalai skills add %s` installs it managed",
        fields.registry_name, ", ".join(str(f) for f in folders), shlex.quote(fields.registry_name),
    )

    return {
        "written_paths": [str(folder / "SKILL.md") for folder in per_agent],
        "attachment_count": (len(bundle.files) - 1) * len(folders),
        # The skill's IDENTITY (may be namespaced). The folder on disk is `skill_dirname`,
        # which is also the `name:` its SKILL.md declares.
        "skill_name": fields.registry_name,
        "skill_dirname": install_name,
        # FULL sha256 of the exact SKILL.md written.
        "content_hash": bundle.files[0].sha256,
        "unmanaged": [str(folder) for folder in folders],
        "unavailable": [u.path for u in bundle.unavailable],
        "policy_notes": list(bundle.policy_notes),
        "backups": backups,
    }


# ── A project 0.14 wrote: the version 1 lock fallback ─────────
#
# SkillRouter's old disk methods use these in a project whose lock is a version 1
# lock, instead of the engine (which refuses that lock until `decimalai skills migrate`
# runs, and must not migrate implicitly). The bytes are the registry's own, fetched and
# checked by the engine's client; the folders go where 0.14 put them, one per agent
# directory; and the version 1 lock is kept up to date. 0.18 removes this fallback.


def _sha256_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _v1_clean(folder: Path, entry: Optional[Mapping[str, Any]]) -> bool:
    """Whether ``folder`` is still the copy its version 1 entry records: its SKILL.md
    hashes to the entry's ``content_hash`` (the full sha256 of the file written)."""
    if not entry:
        return False
    recorded = entry.get("content_hash")
    return isinstance(recorded, str) and _sha256_file(folder / "SKILL.md") == recorded


def _v1_skills(project_root: str) -> Dict[str, Dict[str, Any]]:
    skills = _read_lockfile(project_root).get("skills")
    return {k: dict(v) for k, v in skills.items() if isinstance(v, dict)} \
        if isinstance(skills, dict) else {}


def _v1_find(skills: Mapping[str, Mapping[str, Any]], skill_id: str,
             registry_name: str) -> Tuple[Optional[str], Optional[Mapping[str, Any]]]:
    """The version 1 entry of a skill: by its id, else by its registry name."""
    for key, entry in skills.items():
        if entry.get("skill_id") == skill_id:
            return key, entry
    if registry_name in skills:
        return registry_name, skills[registry_name]
    return None, None


def _v1_taken(skills: Mapping[str, Mapping[str, Any]]) -> Dict[str, Optional[str]]:
    """Folder names the version 1 entries hold, for the registry to keep."""
    from skillevaluation import skillfmt

    return {key: entry["skill_id"] for key, entry in skills.items()
            if skillfmt.is_valid_install_name(key) and isinstance(entry.get("skill_id"), str)
            and entry["skill_id"]}


def _v1_trust(resolved: Any) -> None:
    """The engine's verdict: blocked and withheld are refused; anything else to ask
    about is a warning here, since 0.14 never asked."""
    from .skillpkg.client import trust_verdict
    from .skillpkg.errors import SkillBlockedError, SkillWithheldError

    refusal, caution = trust_verdict(resolved)
    if refusal == "blocked":
        raise SkillBlockedError(f"version {resolved.version} failed its safety review and is "
                                "blocked", skill=resolved.registry_name)
    if refusal == "withheld":
        raise SkillWithheldError("the registry withholds it from installs",
                                 skill=resolved.registry_name)
    if caution and resolved.relationship != "owned":
        logger.warning("%s v%s: its safety review is %s; it was written anyway, as 0.14 did",
                       resolved.registry_name, resolved.version, caution)


def _v1_write(client: Any, project: "Project", resolved: Any, folders: List[Path], *,
              purpose: str, replace: Callable[[Path], Optional[str]]) -> Dict[str, Any]:
    """Fetch the registry's folder for ``resolved`` and place it as each of ``folders``."""
    from .skillpkg import lock as _lock

    bundle = client.fetch_resolved(resolved, purpose=purpose)
    contents = {f.path: f.content for f in bundle.files}
    invocation = bundle.invocation or resolved.invocation
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
        invocation="user" if invocation == "user" else "model",
    )
    backups = _place(project, resolved.install_name, contents, entry, folders, replace=replace)
    return {"skill_md_sha256": bundle.file_hashes["SKILL.md"], "files": len(bundle.files),
            "backups": backups}


def _v1_entry(resolved: Any, skill_md_sha256: str, agents: List[str]) -> Dict[str, Any]:
    """A version 1 lock entry, in the shape 0.14 wrote."""
    return {
        "version": resolved.version,
        "content_hash": skill_md_sha256,
        "body_hash": resolved.body_sha256,
        "source": "registry",
        "skill_id": resolved.skill_id,
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "agents": list(agents),
        "scope": "project",
    }


def _v1_record(project_root: str, *, put: Mapping[str, Dict[str, Any]],
               drop: Optional[List[str]] = None) -> None:
    lockdata = _read_lockfile(project_root)
    skills = lockdata.setdefault("skills", {})
    for key in drop or ():
        skills.pop(key, None)
    skills.update(put)
    _write_lockfile(project_root, lockdata)


def _v1_add(client: Any, project_root: str, ref: str, *, agents: Optional[List[str]],
            force: bool) -> Dict[str, Any]:
    """Install one registry skill into a version 1 project, as 0.14's export did.

    A folder that exists is replaced when it is this skill's own copy and nobody edited
    it; any other folder is refused unless ``force`` (then backed up first).

    Raises:
        SkillpkgError: the registry refused, or the bytes failed their checks.
        SkillExportFileExistsError: a folder is in the way.
    """
    from .skillpkg.client import ResolveRequest
    from .skillpkg.project import Project

    project = Project.at(project_root)
    root = str(project.directory)
    skills = _v1_skills(root)
    resolved = client.resolve_one(ResolveRequest(ref=ref, version="latest"),
                                  taken=_v1_taken(skills), purpose="add")
    _v1_trust(resolved)
    key, old = _v1_find(skills, resolved.skill_id, resolved.registry_name)
    targets = list(agents or ["universal"])
    per_agent, folders = _folders(project, resolved.install_name, targets)

    def replace(folder: Path) -> Optional[str]:
        if _v1_clean(folder, old):
            return "discard"
        return "backup" if force else None

    wrote = _v1_write(client, project, resolved, folders, purpose="add", replace=replace)
    entry = _v1_entry(resolved, wrote["skill_md_sha256"], targets)
    _v1_record(root, put={resolved.registry_name: entry},
               drop=[key] if key and key != resolved.registry_name else None)
    return {
        "skills_written": 1,
        "attachments_written": (wrote["files"] - 1) * len(folders),
        "paths": [str(f / "SKILL.md") for f in folders],
        "written_paths": [str(f / "SKILL.md") for f in per_agent],
        "skill_name": resolved.registry_name,
        "skill_dirname": resolved.install_name,
        "errors": [],
        "lock_entry": entry,
        "lockfile": "v1",
        "backups": wrote["backups"],
        "result": None,
    }


def _v1_refresh(client: Any, project_root: str, *, agents: Optional[List[str]], force: bool,
                replace_existing: bool, purpose: str) -> Dict[str, Any]:
    """Bring the folders a version 1 lock lists up to the registry's current version.

    A folder that is gone is written (``pulled``); one whose SKILL.md differs from the
    registry's is replaced (``updated``) when it is still the copy the lock records, or
    with ``force`` (backed up first); one edited by hand is otherwise left alone and
    reported in ``failures``. With ``replace_existing=False`` no folder that exists is
    replaced. Never adds a skill the lock does not list.
    """
    from .skillpkg.client import ResolveRequest
    from .skillpkg.errors import SkillpkgError
    from .skillpkg.project import Project

    project = Project.at(project_root)
    root = str(project.directory)
    skills = _v1_skills(root)
    keys = sorted(skills)
    requests = [ResolveRequest(skill_id=skills[k]["skill_id"], version="latest")
                if isinstance(skills[k].get("skill_id"), str) and skills[k]["skill_id"]
                else ResolveRequest(ref=k, version="latest") for k in keys]
    results = client.resolve(requests, taken=_v1_taken(skills), purpose=purpose) if keys else []
    out: Dict[str, Any] = {"pulled": [], "updated": [], "skipped": [], "failures": []}
    for key, resolved in zip(keys, results):
        old = skills[key]
        if isinstance(resolved, SkillpkgError):
            out["failures"].append((key, str(resolved)))
            continue
        try:
            _v1_trust(resolved)
        except SkillpkgError as exc:
            out["failures"].append((key, str(exc)))
            continue
        targets = list(agents or old.get("agents") or ["universal"])
        _, folders = _folders(project, resolved.install_name, targets)
        existing = [f for f in folders if os.path.lexists(f)]
        served = resolved.file_hashes.get("SKILL.md")
        if existing and not replace_existing:
            out["skipped"].append(key)
            continue
        if len(existing) == len(folders) and all(
                _sha256_file(f / "SKILL.md") == served for f in existing):
            out["skipped"].append(key)
            continue
        edited = [f for f in existing if not _v1_clean(f, old)]
        if edited and not force:
            out["failures"].append((key, (
                f"{', '.join(str(f) for f in edited)} was edited by hand and was left as it "
                "is; force=True replaces it (backed up first)")))
            continue
        try:
            wrote = _v1_write(
                client, project, resolved, folders, purpose=purpose,
                replace=lambda f: "discard" if _v1_clean(f, old) else "backup")
        except (SkillpkgError, SkillExportFileExistsError, ValueError) as exc:
            out["failures"].append((key, str(exc)))
            continue
        _v1_record(root, put={key: _v1_entry(resolved, wrote["skill_md_sha256"], targets)})
        out["updated" if existing else "pulled"].append(key)
    return out


def export_skill_to_disk(
    skill: Dict[str, Any],
    *,
    agents: Optional[List[str]] = None,
    scope: str = "project",
    project_root: Optional[str] = None,
    attachments: Optional[List[Dict[str, Any]]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Write one registry skill (SKILL.md + attachments) to disk, unmanaged. Deprecated.

    .. deprecated:: 0.15
        Use ``decimalai skills add <name>`` or :func:`decimalai.skillpkg.add`, which
        install the registry's own bytes and pin them in ``.decimal/skills.lock``. This
        stops working in 0.18.

    The SKILL.md is rendered locally in the registry's format (see the module notes)
    and written into the skills folder of each agent runtime, one folder per distinct
    directory.

    Args:
        skill: the skill as the platform serves it: ``name``, ``description``,
            ``body_markdown``, ``id`` and a version id (``version_id``,
            ``latest_version_id`` or ``latest_version.id``); optionally ``url_slug``,
            ``latest_version.version_number`` or ``version``, ``license``,
            ``license_spdx``, ``compatibility`` and ``allowed_tools`` (a string or a
            list of strings), ``invocation``, ``category``, ``skill_type`` and
            ``skill_scope``. Only Read, Grep and Glob of ``allowed_tools`` are written,
            as the registry does for a skill installed from it.
        agents: agent runtimes to write for (default ``["universal"]``).
        scope: ``"project"``. ``"global"`` is refused with the command that writes a
            personal copy.
        project_root: the project (default: the working directory).
        attachments: ``{file_path, content_text, is_executable}`` dicts, each path under
            ``scripts/``, ``references/``, ``templates/`` or ``assets/``.
        force: replace a folder that already exists; the old one is moved to
            ``.decimal/backup/<txid>/`` first. A folder ``decimalai skills`` manages is
            never replaced.

    Returns:
        ``written_paths`` (the SKILL.md written for each agent), ``attachment_count``,
        ``skill_name`` (the registry name), ``skill_dirname`` (the folder, also the
        frontmatter ``name``), ``content_hash`` (sha256 of SKILL.md), ``unmanaged``
        (the folders written), ``unavailable``, ``policy_notes`` and ``backups``.

    Raises:
        SkillExportFileExistsError: a folder exists (without ``force``), or is managed.
        SkillExportUnsafePathError: a name or path would leave the skill folder.
        ValueError: a field has the wrong type, the ids are missing, the body or the
            description is empty, or ``scope`` is not ``"project"``.
    """
    _warn_deprecated(
        "export_skill_to_disk",
        "disk_export.export_skill_to_disk() is deprecated and stops working in 0.18: "
        "install registry skills with `decimalai skills add <name>` "
        "(decimalai.skillpkg.add()), which pins what it writes in .decimal/skills.lock.")
    return _export(skill, agents=agents, scope=scope, project_root=project_root,
                   attachments=attachments, force=force)


def export_skills_to_disk(
    skills: List[Dict[str, Any]],
    *,
    agents: Optional[List[str]] = None,
    scope: str = "project",
    project_root: Optional[str] = None,
    update_lockfile: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    """Write several registry skills to disk, unmanaged. Deprecated.

    .. deprecated:: 0.15
        Use ``decimalai skills add <name>`` for each skill, or
        :func:`decimalai.skillpkg.add`. This stops working in 0.18.

    Each skill is written as :func:`export_skill_to_disk` writes it, with the dict's own
    ``attachments``.

    The lock: only a version 1 ``.decimal/skills.lock`` (one an older SDK wrote) is
    updated, so ``decimalai skills migrate`` can turn its entries into managed skills.
    A project with no lock gets none: a fresh project never starts with legacy state.
    A version 2 lock (one ``decimalai skills`` manages) and an unreadable one are left
    exactly as they are. Either way the folders written stay unmanaged. Nothing is
    written with ``update_lockfile=False``.

    Returns:
        ``skills_written``, ``attachments_written``, ``paths``, ``agents``, ``scope``,
        ``unmanaged`` (the folders written) and ``lockfile`` (``"v1"`` when the version 1
        lock was updated, else None).
    """
    _warn_deprecated(
        "export_skills_to_disk",
        "disk_export.export_skills_to_disk() is deprecated and stops working in 0.18: "
        "install registry skills with `decimalai skills add <name>` "
        "(decimalai.skillpkg.add()), which pins what it writes in .decimal/skills.lock.")
    root = project_root or os.getcwd()
    written = 0
    total_attachments = 0
    all_paths: List[str] = []
    unmanaged: List[str] = []
    skill_entries: Dict[str, Dict[str, Any]] = {}

    for skill in skills:
        result = _export(skill, agents=agents, scope=scope, project_root=root,
                         attachments=skill.get("attachments") or [], force=force)
        written += 1
        total_attachments += result["attachment_count"]
        all_paths.extend(result["written_paths"])
        unmanaged.extend(result["unmanaged"])
        latest = skill.get("latest_version") if isinstance(skill.get("latest_version"), Mapping) else {}
        skill_entries[skill["name"]] = {
            "version": _version_number(skill, latest),
            # The FILE: the full sha256 of the exact SKILL.md written.
            "content_hash": result["content_hash"],
            # The platform's axis: the sha256 of body_markdown, which `GET /skills/hashes`
            # serves.
            "body_hash": hashlib.sha256((skill.get("body_markdown") or "").encode("utf-8")).hexdigest(),
            "source": skill.get("source", "platform"),
            "skill_id": skill.get("id", ""),
            "installed_at": datetime.now(timezone.utc).isoformat(),
            "agents": agents or ["universal"],
            "scope": scope,
        }

    lockfile = None
    if update_lockfile and skill_entries:
        kind = _lock_kind(root)
        if kind == "v1":
            lockdata = _read_lockfile(root)
            lockdata.setdefault("skills", {}).update(skill_entries)
            _write_lockfile(root, lockdata)
            lockfile = "v1"
            logger.info("Updated lockfile: %d skills tracked", len(lockdata["skills"]))
        elif kind == "absent":
            logger.info(
                "Not creating %s: a project starts with `decimalai skills`, not a version 1 "
                "lock. `decimalai skills add <name>` installs a skill managed.", LOCKFILE_NAME)
        elif kind == "v2":
            logger.warning(
                "Not recording %s in %s: `decimalai skills` manages that lock, so the "
                "folders just written are unmanaged. `decimalai skills add <name>` "
                "installs a skill managed.", ", ".join(sorted(skill_entries)), LOCKFILE_NAME)
        else:
            logger.warning("Not recording %s: %s cannot be read, so it was left as it is.",
                           ", ".join(sorted(skill_entries)), LOCKFILE_NAME)

    return {
        "skills_written": written,
        "attachments_written": total_attachments,
        "paths": all_paths,
        "agents": agents or ["universal"],
        "scope": scope,
        "unmanaged": unmanaged,
        "lockfile": lockfile,
    }
