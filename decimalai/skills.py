"""Skill discovery and activation detection for DecimalAI SDK.

Implements the three-tier skill tracking model:

1. **Auto-discovery** — Scans for SKILL.md files following the agentskills.io spec
2. **Prompt-diff detection** — Matches rendered prompts against known skill content
3. **Explicit declaration** — ``install(skills=[...])`` or ``log_skill_activation()``

See https://docs.decimal.ai/guides/skills for the full user-facing documentation.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("decimalai.skills")

# Default scan paths for SKILL.md auto-discovery.
# Covers project-local paths for all known agent runtimes.
# Deduplicated — many agents share `.agents/skills/`.
DEFAULT_SKILL_PATHS = [
    # Primary agent skill directories (alphabetical by agent)
    ".agents/skills",          # universal, cursor, copilot, cline, warp, etc.
    ".claude/skills",          # claude-code
    ".continue/skills",        # continue
    ".windsurf/skills",        # windsurf
    ".amp/skills",             # amp
    ".augment/skills",         # augment
    ".roo/skills",             # roo
    ".trae/skills",            # trae
    ".melty/skills",           # melty
    ".void/skills",            # void
    # Legacy / alternate
    ".skills",                 # generic fallback
]

# Global skill dirs (user-wide, not project-scoped)
GLOBAL_SKILL_PATHS = [
    "~/.claude/skills",
    "~/.cursor/skills",
    "~/.copilot/skills",
    "~/.agents/skills",
    "~/.config/agents/skills",
    "~/.gemini/skills",
    "~/.gemini/antigravity/skills",
    "~/.windsurf/skills",
    "~/.continue/skills",
]
# Primary global path (backward compat)
GLOBAL_SKILL_PATH = os.path.expanduser("~/.claude/skills")

# Minimum size for a SKILL.md body to be considered valid
_MIN_BODY_LENGTH = 10


# ── Skill taxonomy ────────────────────────────────────────────
# The taxonomy split from a single 3-way `skill-type` into two orthogonal
# frontmatter fields:
#   - skill-type:  capability | preference   (what kind of skill it is)
#   - skill-scope: public | private          (who is allowed to see it)
# Older on-disk SKILL.md files still spell the pre-split 3-way `skill-type`
# values; each maps to a (type, scope) pair so they keep parsing. Mirrors the
# backend's normalization rule.
_VALID_SKILL_TYPES = {"capability", "preference"}
_VALID_SKILL_SCOPES = {"public", "private"}
_LEGACY_SKILL_TYPES = {
    "model_gap": ("capability", "public"),
    "proprietary": ("capability", "private"),
    "convention": ("preference", "public"),
}


def _resolve_skill_taxonomy(
    raw_type: Any, raw_scope: Any
) -> Tuple[Optional[str], Optional[str]]:
    """Normalize on-disk ``skill-type`` / ``skill-scope`` to the current taxonomy.

    Returns ``(skill_type, skill_scope)`` where each is a canonical new value or
    ``None``:

    - New spellings pass through: ``capability`` | ``preference`` for the type,
      ``public`` | ``private`` for the scope.
    - A LEGACY ``skill-type`` (``model-gap`` / ``proprietary`` / ``convention``)
      resolves to a ``(type, scope)`` pair — so an older SKILL.md keeps working.
    - An explicit ``skill-scope`` always wins over the scope a legacy type
      implies.
    - Anything unrecognized is ignored (never raises).
    """
    skill_type: Optional[str] = None
    skill_scope: Optional[str] = None

    if raw_type:
        norm = str(raw_type).strip().lower().replace("-", "_")
        if norm in _VALID_SKILL_TYPES:
            skill_type = norm
        elif norm in _LEGACY_SKILL_TYPES:
            skill_type, skill_scope = _LEGACY_SKILL_TYPES[norm]

    if raw_scope:
        norm_scope = str(raw_scope).strip().lower().replace("-", "_")
        if norm_scope in _VALID_SKILL_SCOPES:
            skill_scope = norm_scope  # explicit scope overrides legacy-implied

    return skill_type, skill_scope


def discover_skills(
    search_paths: Optional[List[str]] = None,
    *,
    include_global: bool = False,
) -> List[Dict[str, Any]]:
    """Scan for SKILL.md files following the agentskills.io spec.

    Searches project-level directories by default. Global / personal
    skill directories (``~/.claude/skills`` and friends) are OPT-IN.
    Earlier SDK releases defaulted to ``include_global=True``, and
    running ``decimalai skills sync`` from a project with no local skills
    could sweep personal agent-tool skills (e.g.
    ``~/.claude/skills/my-personal-helper``) into the org's skill
    registry. The two concerns are kept separate now: project skills go
    up automatically, personal/global skills are a deliberate opt-in.

    Args:
        search_paths: Override the list of directories to scan.
            Defaults to project-local paths from ``DEFAULT_SKILL_PATHS``
            (``.claude/skills``, ``.agents/skills``, etc.).
        include_global: Whether to also scan personal/global skill
            directories (``~/.claude/skills``, ``~/.agents/skills``, …).
            Defaults to ``False``, so only project-local skills are
            returned and nothing from your home directory is picked up
            unless you ask for it. Pass ``True`` to include them.

    Returns:
        List of skill descriptors with ``name``, ``description``, ``version``,
        ``hash``, ``stability``, ``category``, ``skill_type``, ``skill_scope``,
        ``invocation``, and ``source_path`` keys.
    """
    paths = list(search_paths or DEFAULT_SKILL_PATHS)
    if include_global:
        paths.extend(GLOBAL_SKILL_PATHS)

    skills: List[Dict[str, Any]] = []
    seen_names: set = set()

    for base_path in paths:
        expanded = os.path.expanduser(base_path)
        if not os.path.isdir(expanded):
            continue
        for skill_dir in _find_skill_dirs(expanded):
            skill_md = os.path.join(skill_dir, "SKILL.md")
            if not os.path.isfile(skill_md):
                continue
            try:
                descriptor = parse_skill_md(skill_md)
                if descriptor and descriptor["name"] not in seen_names:
                    skills.append(descriptor)
                    seen_names.add(descriptor["name"])
            except Exception:
                logger.warning("Failed to parse %s", skill_md, exc_info=True)

    if skills:
        logger.info("Auto-discovered %d skills from SKILL.md files", len(skills))
    return skills


def _title_from_skill_md(frontmatter: Dict[str, Any], body: str) -> Optional[str]:
    """Human display title for a SKILL.md.

    The SKILL.md / agentskills.io spec has no title field, so prefer an explicit
    ``title:``/``display_name:`` frontmatter key, else lift the body's first H1
    heading (skipping a heading that's itself a bare kebab slug — that adds
    nothing over the slug). Returns None when neither yields a human title.
    """
    title = frontmatter.get("title") or frontmatter.get("display_name")
    if title:
        return str(title)[:200]
    h1 = re.match(r"^#\s+(.+)", body.strip()) if body else None
    if h1:
        heading = h1.group(1).strip()
        # Skip a bare-slug H1 and an instructional-sentence H1 (too long / too
        # many words to be a title) — the registry humanizes the slug instead.
        if (heading
                and not re.fullmatch(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", heading)
                and len(heading) <= 64 and len(heading.split()) <= 7):
            return heading
    return None


def parse_skill_md(path: str) -> Optional[Dict[str, Any]]:
    """Parse a SKILL.md file into a skill descriptor.

    Extracts YAML frontmatter (name, description, metadata.version, plus the
    taxonomy labels category / skill-type / skill-scope / invocation when
    present) and hashes the markdown body.

    Args:
        path: Absolute or relative path to the SKILL.md file.

    Returns:
        Skill descriptor dict, or None if the file is invalid.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    frontmatter, body = _split_frontmatter(content)
    if not frontmatter or not body or len(body.strip()) < _MIN_BODY_LENGTH:
        return None

    # Slug source: frontmatter `name:`, else the skill directory's basename —
    # matching the `decimalai skills sync` CLI (cli/main.py), so the same tree
    # syncs identically whether discovered programmatically or via the CLI.
    name = (frontmatter.get("name") or "").strip() or Path(path).parent.name
    if not name:
        return None

    # Validate name per agentskills.io spec: lowercase a-z, digits, single hyphens
    _SPEC_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
    if not _SPEC_NAME_RE.match(name) or len(name) > 64:
        logger.warning(
            "Skipping skill '%s' — name does not match agentskills.io spec "
            "(must be 1-64 chars, lowercase a-z/0-9 and single hyphens, starting with a letter)",
            name,
        )
        return None

    metadata = frontmatter.get("metadata", {})
    version = metadata.get("version") if isinstance(metadata, dict) else None

    # Validate description length (spec: 1-1024 chars)
    description = frontmatter.get("description", "")
    if len(description) > 1024:
        logger.warning(
            "Skill '%s' description exceeds 1024 chars (%d) — truncating",
            name, len(description),
        )
        description = description[:1024]

    # Taxonomy labels: `skill-type`, the new
    # orthogonal `skill-scope`, and `invocation` frontmatter travel with the
    # descriptor so sync_to_platform can thread them to the platform.
    # `_resolve_skill_taxonomy` accepts the new values (`capability` /
    # `preference`, `public` / `private`) AND the legacy 3-way `skill-type`
    # spellings (`model-gap` / `proprietary` / `convention`), mapping each
    # legacy value to a (type, scope) pair so an older on-disk SKILL.md keeps
    # working. Claude Code's `disable-model-invocation: true` is accepted as
    # `invocation: user` (round-trip both spellings). Absent keys stay None —
    # no default is stamped, so a legacy SKILL.md gains no frontmatter on
    # re-export.
    raw_skill_type = frontmatter.get("skill-type") or frontmatter.get("skill_type")
    raw_skill_scope = frontmatter.get("skill-scope") or frontmatter.get("skill_scope")
    skill_type, skill_scope = _resolve_skill_taxonomy(raw_skill_type, raw_skill_scope)
    invocation = frontmatter.get("invocation")

    # The backend's SyncSkillItem caps category at 100 chars and requires a
    # string — an oversized or nested-map category would 422 the WHOLE sync
    # batch. Clamp here (same pattern as the description cap above) so every
    # surface that syncs a descriptor is covered.
    category = frontmatter.get("category")
    if category is not None and not isinstance(category, str):
        logger.warning(
            "Skill '%s' category is not a string (%s) — dropping", name, type(category).__name__,
        )
        category = None
    elif category and len(category) > 100:
        logger.warning(
            "Skill '%s' category exceeds 100 chars (%d) — truncating", name, len(category),
        )
        category = category[:100]
    if not invocation and str(
        frontmatter.get("disable-model-invocation") or ""
    ).strip().lower() in ("true", "yes", "1"):
        invocation = "user"

    return {
        "name": name,
        "display_name": _title_from_skill_md(frontmatter, body),
        "description": description,
        "version": version,
        "hash": _hash_content(body),
        "stability": "stable",
        "category": category,
        # Already normalized to the current taxonomy by _resolve_skill_taxonomy.
        "skill_type": skill_type,
        "skill_scope": skill_scope,
        "invocation": str(invocation).strip().lower() if invocation else None,
        "source_path": str(Path(path).parent),
        # Retain the body so Tier-2 fuzzy body matching in
        # detect_skill_activations can actually fire for auto-discovered
        # skills. Without it the descriptor carried no body and Tier-2 was
        # dead — only Tier-1 name matching ever ran. The body is already
        # read above to compute the hash, so this is no extra I/O.
        "body": body,
    }


class SkillRegistry:
    """Holds discovered skills and supports merge with explicit overrides.

    Priority: explicit skills override auto-discovered skills when names match.
    """

    def __init__(
        self,
        auto_discovered: Optional[List[Dict[str, Any]]] = None,
        explicit: Optional[List[Dict[str, Any]]] = None,
    ):
        self._skills: Dict[str, Dict[str, Any]] = {}

        # Auto-discovered first (lower priority)
        for skill in (auto_discovered or []):
            self._skills[skill["name"]] = skill

        # Explicit overrides (higher priority)
        for skill in (explicit or []):
            self._skills[skill["name"]] = skill

    @property
    def skills(self) -> List[Dict[str, Any]]:
        """Return the merged skill list."""
        return list(self._skills.values())

    @property
    def names(self) -> List[str]:
        """Return sorted skill names."""
        return sorted(self._skills.keys())

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a skill by name."""
        return self._skills.get(name)

    def __len__(self) -> int:
        return len(self._skills)

    def __bool__(self) -> bool:
        return bool(self._skills)


def detect_skill_activations(
    rendered_input: Any,
    skill_registry: List[Dict[str, Any]],
    *,
    fuzzy_match: bool = True,
    fuzzy_threshold: float = 0.6,
) -> List[str]:
    """Detect which skills were activated by matching prompt content.

    Detection tiers (in order):
    1. **Name-pattern matching** — looks for ``## Skill: code-review``,
       ``[code-review]``, etc. in system/developer messages.
    2. **Fuzzy body matching** (optional, default on) — checks whether
       ≥ ``fuzzy_threshold`` of a skill's body lines appear in the prompt.
       Catches cases where skill content is injected without a header.

    Args:
        rendered_input: The rendered_input from an LlmCallRecord
            (list of message dicts or string).
        skill_registry: The skill registry list from SkillRegistry.skills.
        fuzzy_match: Enable fuzzy body matching fallback. Default True.
            Set to False to use name-pattern matching only.
        fuzzy_threshold: Minimum line-overlap ratio (0.0–1.0) for fuzzy
            matching. Default 0.6 (60% of skill body lines must appear
            in the prompt). Set to 0.0 to effectively disable.

    Returns:
        List of activated skill names.
    """
    if not skill_registry or not rendered_input:
        return []

    system_text = _extract_system_text(rendered_input)
    if not system_text:
        return []

    activated = []
    for skill in skill_registry:
        name = skill.get("name", "")
        if not name:
            continue

        # Tier 1: Name-pattern matching (fast, high precision)
        if _skill_appears_in_text(name, system_text):
            activated.append(name)
            continue

        # Tier 2: Fuzzy body matching (fallback)
        if fuzzy_match and fuzzy_threshold > 0.0:
            body = skill.get("body") or skill.get("body_markdown") or ""
            if body and _fuzzy_body_match(body, system_text, fuzzy_threshold):
                activated.append(name)

    return activated


# ── Internal Helpers ────────────────────────────────────────


def _find_skill_dirs(base_path: str) -> List[str]:
    """Find directories containing SKILL.md files."""
    dirs = []
    base = Path(base_path)
    if not base.exists():
        return dirs

    # Direct children (e.g., .claude/skills/code-review/SKILL.md)
    for child in sorted(base.iterdir()):
        if child.is_dir() and (child / "SKILL.md").exists():
            dirs.append(str(child))

    # Also check if SKILL.md is directly in base_path (single-skill dir)
    if (base / "SKILL.md").exists():
        dirs.append(str(base))

    return dirs


def _split_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Split YAML frontmatter from markdown body.

    Returns (frontmatter_dict, body_string). If no frontmatter, returns ({}, content).
    """
    content = content.strip()
    if not content.startswith("---"):
        return {}, content

    # Find the closing fence — a line that is exactly "---" (YAML frontmatter
    # delimiter semantics). A raw substring search would mis-split on a "---"
    # that appears inside a frontmatter scalar value (e.g. a description), which
    # truncates the YAML and silently drops the rest of the frontmatter.
    lines = content.split("\n")
    end_line = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_line = i
            break
    if end_line is None:
        return {}, content

    yaml_str = "\n".join(lines[1:end_line]).strip()
    body = "\n".join(lines[end_line + 1:]).strip()

    # Parse simple YAML (key: value) without requiring PyYAML dependency
    frontmatter = _parse_simple_yaml(yaml_str)
    return frontmatter, body


def _parse_simple_yaml(yaml_str: str) -> Dict[str, Any]:
    """Parse simple key: value YAML without external deps.

    Handles basic scalar values and simple nested maps (metadata block).
    Not a full YAML parser — sufficient for SKILL.md frontmatter.
    """
    result: Dict[str, Any] = {}
    current_key = None
    current_map: Optional[Dict[str, str]] = None

    for line in yaml_str.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Check for nested map entry (2-space or tab indent)
        if (line.startswith("  ") or line.startswith("\t")) and current_key:
            match = re.match(r'\s+(\w[\w-]*):\s*(.*)', line)
            if match:
                if current_map is None:
                    current_map = {}
                    result[current_key] = current_map
                current_map[match.group(1)] = match.group(2).strip().strip('"').strip("'")
                continue

        # Top-level key: value
        match = re.match(r'([\w-]+):\s*(.*)', stripped)
        if match:
            current_key = match.group(1)
            value = match.group(2).strip()
            current_map = None
            if value:
                result[current_key] = value.strip('"').strip("'")
            else:
                result[current_key] = None  # Will be replaced if nested map follows

    return result


def _hash_content(text: str) -> str:
    """SHA-256 hash of text content."""
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _extract_system_text(rendered_input: Any) -> str:
    """Extract system/developer message text from rendered_input."""
    if isinstance(rendered_input, str):
        return rendered_input

    if isinstance(rendered_input, list):
        parts = []
        for msg in rendered_input:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                if role in ("system", "developer"):
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        parts.append(content)
        return "\n".join(parts)

    return ""


def _skill_appears_in_text(skill_name: str, text: str) -> bool:
    """Check if a skill's name appears as a reference in the prompt text.

    Looks for patterns like:
    - ## Skill: code-review
    - ## Active Skill: code-review
    - # code-review
    - [code-review]
    """
    if not skill_name or not text:
        return False

    # Direct name reference (case-insensitive)
    patterns = [
        rf"##?\s+(?:Active\s+)?Skill:\s*{re.escape(skill_name)}",
        rf"##?\s+{re.escape(skill_name)}",
        rf"\[{re.escape(skill_name)}\]",
    ]
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def _fuzzy_body_match(skill_body: str, prompt_text: str, threshold: float) -> bool:
    """Check if a skill's body content appears in the prompt text via fuzzy line matching.

    Normalizes both texts (lowercase, strip whitespace), splits the skill body
    into significant lines (>10 chars to skip headers/blanks), and computes
    what fraction of those lines appear as substrings in the prompt.

    Args:
        skill_body: The skill's markdown body content.
        prompt_text: The full system/developer prompt text.
        threshold: Minimum overlap ratio (0.0–1.0).

    Returns:
        True if the line-overlap ratio meets the threshold.
    """
    if not skill_body or not prompt_text:
        return False

    # Normalize both texts
    prompt_lower = prompt_text.lower()

    # Split skill body into significant lines
    body_lines = [
        line.strip().lower()
        for line in skill_body.split("\n")
        if len(line.strip()) > 10  # Skip short lines (headers, blanks, bullets)
    ]

    if not body_lines:
        return False

    # Count how many body lines appear in the prompt
    matches = sum(1 for line in body_lines if line in prompt_lower)
    ratio = matches / len(body_lines)

    return ratio >= threshold


def sync_to_platform(
    api_key: str,
    base_url: str = "https://api.decimal.ai",
    *,
    search_paths: Optional[List[str]] = None,
    include_global: bool = False,
    author: Optional[str] = None,
    conflict_policy: str = "newer_wins",
) -> Dict[str, Any]:
    """Discover local SKILL.md files and push them to the platform.

    Bridges local skill discovery with the platform skill management API.
    Skills are upserted — existing skills are updated if body changed,
    new skills are created.

    ``include_global`` defaults to ``False`` so a developer's personal
    agent-tool skills (``~/.claude/skills/*``) are NOT swept into the
    org registry by an accidental ``decimalai skills sync``. Set
    ``include_global=True`` to opt back in.

    Args:
        api_key: DecimalAI API key.
        base_url: Platform API base URL.
        search_paths: Override skill search directories.
        include_global: Also scan personal/global skill directories
            (``~/.claude/skills`` etc.). Default ``False`` — opt-in.
        author: Attribution for the upload.
        conflict_policy: Hash-mismatch resolution. Defaults to ``"newer_wins"``
            (git-aware timestamps decide; never blind-clobber a newer remote
            edit). Pass ``"local_wins"`` for CI where the repo is the source of
            truth, or ``"remote_wins"`` to never overwrite.

    Returns:
        Summary dict with created, updated, unchanged counts.
    """
    from .skill_router import SkillRouter

    discovered = discover_skills(search_paths, include_global=include_global)
    if not discovered:
        logger.info("No local skills discovered — nothing to sync")
        return {"created": 0, "updated": 0, "unchanged": 0}

    # Convert to platform format: include body_markdown from SKILL.md files
    skills_payload = []
    for skill in discovered:
        body = _read_skill_body(skill.get("source_path", ""))
        if not body:
            continue
        skills_payload.append({
            "name": skill["name"],
            "display_name": skill.get("display_name"),
            "description": skill.get("description", skill["name"]),
            "body_markdown": body,
            "category": skill.get("category"),
            "trigger_phrases": skill.get("trigger_phrases"),
            # Taxonomy labels: thread a
            # locally-authored skill's `skill-type` / `skill-scope` /
            # `invocation` frontmatter to the platform. The sync endpoint's
            # Pydantic model ignores unknown fields, so this degrades
            # gracefully across payload-shape iterations.
            "skill_type": skill.get("skill_type"),
            "skill_scope": skill.get("skill_scope"),
            "invocation": skill.get("invocation"),
            # git-aware mtime so a fresh checkout doesn't look "newest" and
            # clobber a more-recent dashboard edit under newer_wins.
            "local_updated_at": _local_updated_at_iso(
                os.path.join(skill.get("source_path", ""), "SKILL.md")
            ),
        })

    if not skills_payload:
        return {"created": 0, "updated": 0, "unchanged": 0}

    router = SkillRouter(api_key=api_key, base_url=base_url)
    # Stamp this checkout's install identity so the platform can flag
    # local/remote drift per install. Best-effort: identity I/O never blocks a
    # sync.
    install_id = install_label = None
    try:
        from ._install import get_install_identity

        identity = get_install_identity()
        install_id = identity.get("install_id")
        install_label = identity.get("install_label")
    except Exception:  # pragma: no cover - identity is non-essential
        pass
    result = router.sync_skills(
        skills_payload,
        author=author,
        # Default newer_wins (not the SDK-level local_wins) so an
        # auto/programmatic sync never blind-overwrites a teammate's more-recent
        # dashboard edit; with git-aware local_updated_at above, the genuinely
        # newer side wins. CI that wants repo-is-truth can pass local_wins.
        conflict_policy=conflict_policy,
        install_id=install_id,
        install_label=install_label,
    )

    logger.info(
        "Synced %d skills to platform: %s",
        len(skills_payload), result,
    )
    return result


def _read_skill_body(skill_dir: str) -> Optional[str]:
    """Read the body markdown from a SKILL.md file in the given directory."""
    skill_md = os.path.join(skill_dir, "SKILL.md")
    if not os.path.isfile(skill_md):
        return None

    with open(skill_md, "r", encoding="utf-8") as f:
        content = f.read()

    _, body = _split_frontmatter(content)
    return body if body and len(body.strip()) >= _MIN_BODY_LENGTH else None


def _local_updated_at_iso(skill_md_path: str) -> Optional[str]:
    """ISO-8601 last-modified time for a SKILL.md, preferring the git commit time
    over the filesystem mtime.

    A fresh ``git clone`` / CI checkout resets every file's mtime to "now", which
    makes an mtime-based ``newer_wins`` sync treat the local copy as newest and
    clobber a more-recent dashboard edit. The git commit time is stable across
    checkouts, so we use it when the file is tracked; otherwise fall back to the
    filesystem mtime. Best-effort — never raises.
    """
    import datetime as _dt

    if not skill_md_path or not os.path.isfile(skill_md_path):
        return None
    try:
        import subprocess

        out = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", os.path.basename(skill_md_path)],
            cwd=os.path.dirname(skill_md_path) or ".",
            capture_output=True, text=True, timeout=3,
        )
        ts = (out.stdout or "").strip()
        if out.returncode == 0 and ts:
            return ts
    except Exception:  # pragma: no cover - git absent / not a repo / timeout
        pass
    try:
        return _dt.datetime.fromtimestamp(
            os.stat(skill_md_path).st_mtime, tz=_dt.timezone.utc
        ).isoformat()
    except OSError:
        return None


def _with_local_timestamps(skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add a git-aware ``local_updated_at`` to each skill dict that has a known
    ``source_path`` and doesn't already carry one — so an auto/background
    ``newer_wins`` sync compares real ages instead of falling back to local-wins
    (which silently clobbers newer remote edits). Non-destructive."""
    out: List[Dict[str, Any]] = []
    for s in skills:
        if s.get("local_updated_at") or not s.get("source_path"):
            out.append(s)
            continue
        ts = _local_updated_at_iso(os.path.join(s["source_path"], "SKILL.md"))
        out.append({**s, "local_updated_at": ts} if ts else s)
    return out

