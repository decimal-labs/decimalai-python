"""Agent path registry and disk export utilities.

Maps agent runtime names to their SKILL.md directory conventions,
and provides utilities to write skills from the platform to disk.

Based on the agentskills.io / npx skills conventions:
https://github.com/vercel-labs/skills
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("decimalai.disk_export")


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


# ── SKILL.md Reconstruction ──────────────────────────────────


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
    import json
    return json.dumps(value, ensure_ascii=False)


def _reconstruct_skill_md(
    name: str,
    description: str,
    body_markdown: str,
    *,
    license_info: Optional[str] = None,
    compatibility: Optional[List[str]] = None,
    allowed_tools: Optional[List[str]] = None,
    category: Optional[str] = None,
    skill_type: Optional[str] = None,
    skill_scope: Optional[str] = None,
    invocation: Optional[str] = None,
) -> str:
    """Reconstruct a SKILL.md file from DB fields.

    Builds a proper SKILL.md with YAML frontmatter + body.

    Args:
        name: Skill name.
        description: Short description.
        body_markdown: Full body content.
        license_info: Optional license (e.g., 'MIT').
        compatibility: Optional list of requirements.
        allowed_tools: Optional list of allowed tools.
        category: Optional category grouping.
        skill_type: Optional taxonomy class ('capability' | 'preference').
            Values have no underscores, so they are emitted verbatim (no
            kebab conversion).
        skill_scope: Optional visibility scope ('public' | 'private') — the
            second, orthogonal taxonomy axis.
        invocation: Optional invocation mode ('model' | 'user' | 'any') —
            ``invocation: user`` additionally
            emits ``disable-model-invocation: true`` (Claude Code's spelling)
            so the label round-trips through harness-native machinery.

    Returns:
        Complete SKILL.md file content.
    """
    # Build frontmatter. Values are YAML-escaped (fixed 2026-06-10) — an
    # unquoted description containing ':' (e.g. "Google Classroom: Manage
    # classes…") produced invalid YAML, so the exported SKILL.md was silently
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
        if str(invocation).strip().lower() == "user":
            # Claude Code's spelling of "not model-invocable". Written for
            # ALL runtimes, not just claude-code exports: the key is
            # namespaced to Claude-Code semantics and ignored elsewhere,
            # and a single reconstruction is shared by every agent in one
            # export call (see paths_written_this_call below) — per-agent
            # content would break that dedup and the multi-agent-symmetry
            # guarantee.
            lines.append("disable-model-invocation: true")
    if license_info:
        lines.append(f"license: {_yaml_scalar(license_info)}")
    if compatibility:
        lines.append("compatibility:")
        for req in compatibility:
            lines.append(f"  - {_yaml_scalar(req)}")
    if allowed_tools:
        lines.append("allowed-tools:")
        for tool in allowed_tools:
            lines.append(f"  - {_yaml_scalar(tool)}")
    lines.append("---")
    lines.append("")

    # Append body
    lines.append(body_markdown)

    return "\n".join(lines)


# ── Lockfile ──────────────────────────────────────────────────

LOCKFILE_NAME = ".decimal/skills.lock"


def _read_lockfile(project_root: str) -> Dict[str, Any]:
    """Read the skills lockfile."""
    lockfile_path = os.path.join(project_root, LOCKFILE_NAME)
    if not os.path.exists(lockfile_path):
        return {"version": 1, "skills": {}}
    try:
        with open(lockfile_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"version": 1, "skills": {}}


def _write_lockfile(project_root: str, lockdata: Dict[str, Any]) -> None:
    """Write the skills lockfile."""
    lockfile_path = os.path.join(project_root, LOCKFILE_NAME)
    os.makedirs(os.path.dirname(lockfile_path), exist_ok=True)
    lockdata["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(lockfile_path, "w") as f:
        json.dump(lockdata, f, indent=2)


# ── Export to Disk ────────────────────────────────────────────


class SkillExportFileExistsError(Exception):
    """Raised when export would overwrite an existing local file.

    Enforces the documented guarantee that installing a skill never touches
    files that already exist on disk. Previously the SKILL.md write at
    open(path, 'w') would silently truncate-and-overwrite any pre-existing file
    at the target path, even one created manually by the user. Now the caller
    must opt in to overwriting via the `force=True` kwarg.
    """

    def __init__(self, path: str):
        super().__init__(
            f"Refusing to overwrite existing file: {path}. "
            "Pass force=True to overwrite (e.g., decimal registry install --force)."
        )
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


def export_skill_to_disk(
    skill: Dict[str, Any],
    *,
    agents: Optional[List[str]] = None,
    scope: str = "project",
    project_root: Optional[str] = None,
    attachments: Optional[List[Dict[str, Any]]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Write a single skill (SKILL.md + attachments) to disk.

    Creates the skill directory structure for each specified agent runtime.

    Args:
        skill: Skill dict with at minimum: name, description, body_markdown.
               May also include: license, compatibility, allowed_tools, id,
               category, skill_type, skill_scope, invocation.
        agents: List of agent runtimes to export for (e.g., ['claude-code', 'cursor']).
                Defaults to ['universal'] if not specified.
        scope: 'project' or 'global'.
        project_root: Project root (used when scope='project').
        attachments: Optional list of attachment dicts with file_path, directory,
                     content_text. Written to subdirectories under the skill dir.
        force: When False (default), raise SkillExportFileExistsError if any
            target file (SKILL.md or an attachment) already exists. This is
            what mechanizes the documented "existing files are never touched"
            guarantee — before it, silent overwrites were possible whenever a
            local file pre-existed. When True, fall back to the prior
            truncate-and-overwrite behavior.

    Returns:
        Dict with:
            - written_paths: list of paths where SKILL.md was written
            - attachment_count: number of attachment files written
            - skill_name: the skill name
    """
    if not agents:
        agents = ["universal"]

    root = project_root or os.getcwd()
    # DIRECTORY NAME vs DECLARED NAME are two different things, and conflating them
    # in one variable is why `decimalai skills pull` refused 72.6% of the registry.
    #
    # A skill's `name` may be namespaced (`microsoft/azure-kusto`) — that is its
    # identity, and it is what the SKILL.md frontmatter must keep declaring. But a
    # slash cannot be a single path component, so `_safe_skill_dirname` correctly
    # rejected it: "unsafe skill name ... The skill source may be malicious." The
    # guard was never wrong; it was being fed the wrong field.
    #
    # `url_slug` is the slash-free identifier the backend mints for
    # exactly this. Falling back to `name` keeps older backends working — and for a
    # plain name the two are identical anyway.
    dirname = _safe_skill_dirname(
        skill.get("url_slug") or skill["name"]
    )  # reject traversal in the server-supplied value
    name = skill["name"]  # the DECLARED identity, written into the frontmatter
    description = skill.get("description", "")
    body_markdown = skill.get("body_markdown", "")

    # Reconstruct SKILL.md content
    skill_md_content = _reconstruct_skill_md(
        name=name,
        description=description,
        body_markdown=body_markdown,
        license_info=skill.get("license"),
        compatibility=skill.get("compatibility"),
        allowed_tools=skill.get("allowed_tools"),
        category=skill.get("category"),
        # Taxonomy labels. The platform payload
        # spells these snake_case; tolerate the kebab frontmatter spelling and
        # the `invocation_mode` column name too so exports keep working across
        # payload-shape iterations. `skill-type` (capability | preference) and
        # `skill-scope` (public | private) are the two orthogonal axes.
        skill_type=skill.get("skill_type") or skill.get("skill-type"),
        skill_scope=skill.get("skill_scope") or skill.get("skill-scope"),
        invocation=skill.get("invocation") or skill.get("invocation_mode"),
    )

    written_paths = []
    attachment_count = 0
    # Track canonical SKILL.md paths we've written THIS call so multiple
    # agent runtimes sharing the same on-disk layout (e.g., cursor +
    # github-copilot + codex all default to .agents/skills/) don't
    # collide on the second write. Without this, a single export call
    # with ``agents=["cursor", "github-copilot"]`` raises
    # SkillExportFileExistsError on the second agent against the file
    # the first agent just wrote. Covered by
    # ``test_top_five_agent_runtimes_get_their_own_skill_md``.
    paths_written_this_call: set[str] = set()

    for agent in agents:
        base_dir = get_agent_paths(agent, scope=scope, project_root=root)
        # dirname, not name: a namespaced `owner/skill` name would nest a
        # directory one level below where agents discover skills.
        skill_dir = os.path.join(base_dir, dirname)
        os.makedirs(skill_dir, exist_ok=True)

        # Write SKILL.md
        skill_md_path = os.path.join(skill_dir, "SKILL.md")
        if skill_md_path in paths_written_this_call:
            # Already written by an earlier agent in this same call —
            # the canonical paths overlap. Record the path in the result
            # (so callers see "claude-code AND cursor both got it") but
            # skip the I/O + collision check.
            written_paths.append(skill_md_path)
            continue
        if not force and os.path.exists(skill_md_path):
            raise SkillExportFileExistsError(skill_md_path)
        with open(skill_md_path, "w") as f:
            f.write(skill_md_content)
        paths_written_this_call.add(skill_md_path)
        written_paths.append(skill_md_path)
        logger.debug("Wrote %s", skill_md_path)

        # Write attachments
        if attachments:
            for att in attachments:
                file_path = att.get("file_path", "")
                content = att.get("content_text", "")
                if not file_path or not content:
                    continue

                att_path = _safe_join_within(skill_dir, file_path)  # reject traversal/absolute attachment paths
                os.makedirs(os.path.dirname(att_path), exist_ok=True)
                if not force and os.path.exists(att_path):
                    raise SkillExportFileExistsError(att_path)
                with open(att_path, "w") as f:
                    f.write(content)
                attachment_count += 1
                logger.debug("Wrote attachment %s", att_path)

    return {
        "written_paths": written_paths,
        "attachment_count": attachment_count,
        # The skill's IDENTITY (may be namespaced). Callers that need the folder
        # on disk want `skill_dirname`.
        "skill_name": name,
        "skill_dirname": dirname,
        # FULL sha256 of the exact SKILL.md content written (frontmatter+body).
        # The lockfile pins THIS — hashing anything else pins a file that was
        # never on disk: sha256(body)[:12] over an empty body is
        # e3b0c44298fc, which pins nothing.
        "content_hash": hashlib.sha256(skill_md_content.encode("utf-8")).hexdigest(),
    }


def export_skills_to_disk(
    skills: List[Dict[str, Any]],
    *,
    agents: Optional[List[str]] = None,
    scope: str = "project",
    project_root: Optional[str] = None,
    update_lockfile: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    """Write multiple skills to disk and update the lockfile.

    Args:
        skills: List of skill dicts from the platform (with body_markdown,
                and optionally attachments embedded).
        agents: Agent runtimes to export for.
        scope: 'project' or 'global'.
        project_root: Project root directory.
        update_lockfile: Whether to update .decimal/skills.lock.
        force: When False (default), raise SkillExportFileExistsError on first
            collision. Threaded through to export_skill_to_disk.

    Returns:
        Summary dict with total skills written, paths, etc.
    """
    root = project_root or os.getcwd()
    total_written = 0
    total_attachments = 0
    all_paths = []
    skill_entries = {}

    for skill in skills:
        attachments = skill.get("attachments", [])
        result = export_skill_to_disk(
            skill,
            agents=agents,
            scope=scope,
            project_root=root,
            attachments=attachments,
            force=force,
        )
        total_written += 1
        total_attachments += result["attachment_count"]
        all_paths.extend(result["written_paths"])

        # Build lockfile entry
        body = skill.get("body_markdown", "")
        skill_entries[skill["name"]] = {
            "version": (
                (skill.get("latest_version") or {}).get("version_number")
                or skill.get("version", 1)
            ),
            # content_hash pins the FILE: full sha256 of the exact SKILL.md
            # content written to disk (frontmatter+body as written).
            "content_hash": result["content_hash"],
            # body_hash pins the PLATFORM axis: full sha256 of body_markdown,
            # the same value `GET /skills/hashes` serves — update_skills
            # compares this one.
            "body_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "source": skill.get("source", "platform"),
            "skill_id": skill.get("id", ""),
            "installed_at": datetime.now(timezone.utc).isoformat(),
            "agents": agents or ["universal"],
            "scope": scope,
        }

    # Update lockfile
    if update_lockfile and scope == "project":
        lockdata = _read_lockfile(root)
        lockdata["skills"].update(skill_entries)
        _write_lockfile(root, lockdata)
        logger.info(
            "Updated lockfile: %d skills tracked",
            len(lockdata["skills"]),
        )

    return {
        "skills_written": total_written,
        "attachments_written": total_attachments,
        "paths": all_paths,
        "agents": agents or ["universal"],
        "scope": scope,
    }
