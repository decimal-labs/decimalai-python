"""Skills you wrote, as a project lists them for its agents.

``.decimal/skills.toml`` lists two kinds of skill. Registry skills sit under ``[skills]``,
which the package manager owns (``decimalai skills add acme/refund-policy``). A skill you
wrote lives beside them in ``.agents/skills/<name>/``, and the project's agents use it once
its folder name is listed under ``[authored]``::

    [authored]
    skills = ["refund-policy"]

``decimalai skills add refund-policy`` writes that line. A folder that is not listed stays
off the agents' list: ``.agents/skills`` is also the folder Codex, Cursor and other coding
tools read, so it can hold skills meant for them.

Released SDKs (0.24 and older) read ``[root]`` and ``[skills]`` only, and the package
manager's own saves keep every other table byte for byte, so this table travels safely.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("decimalai")

TABLE = "authored"
KEY = "skills"

_TABLE_COMMENT = "# Skills you wrote: their folder names under .agents/skills."


def _toml_loads(text: str) -> Dict[str, Any]:
    from .skillpkg.manifest import _toml

    return _toml.loads(text[1:] if text.startswith("﻿") else text)


def _read_text(project: Any) -> Optional[str]:
    path = project.manifest_path
    try:
        return path.read_bytes().decode("utf-8")
    except FileNotFoundError:
        return None


def listed(project: Any) -> List[str]:
    """The folder names ``[authored]`` lists, in file order, each once.

    Empty when there is no manifest, no table, or the file cannot be read: a broken
    manifest is the package manager's error to report, and an agent's run must not fail
    on it.
    """
    try:
        text = _read_text(project)
        if text is None:
            return []
        table = _toml_loads(text).get(TABLE)
    except Exception:
        logger.debug("authored: could not read %s", project.manifest_path, exc_info=True)
        return []
    names = table.get(KEY) if isinstance(table, dict) else None
    if not isinstance(names, list):
        return []
    out: List[str] = []
    for name in names:
        name = name.strip() if isinstance(name, str) else ""
        # A folder name: one path component, as `Project.skill_dir` takes it.
        if (not name or name in (".", "..") or "/" in name or "\\" in name
                or "\x00" in name or name in out):
            continue
        out.append(name)
    return out


def managed(project: Any) -> Dict[str, str]:
    """The folders the lock records as installed from the registry: install name -> registry name."""
    from .skillpkg import lock as _lock

    try:
        entries = _lock.locked_entries(project) or {}
    except Exception:
        logger.debug("authored: could not read the lock", exc_info=True)
        return {}
    return {folder: str(entry.get("name") or folder) for folder, entry in entries.items()}


def skill_md(project: Any, name: str) -> Path:
    return project.skill_dir(name) / "SKILL.md"


def is_own_folder(project: Any, name: str) -> bool:
    """``.agents/skills/<name>/SKILL.md`` exists and the lock did not install it."""
    try:
        return skill_md(project, name).is_file() and name not in managed(project)
    except Exception:
        return False


def unlisted(project: Any) -> List[str]:
    """Folders in ``.agents/skills`` holding a ``SKILL.md`` that are neither installed
    from the registry nor listed under ``[authored]``, sorted."""
    try:
        root = project.skills_root
        if not root.is_dir():
            return []
        taken = set(managed(project)) | set(listed(project))
        return sorted(
            d.name for d in root.iterdir()
            if d.is_dir() and not d.is_symlink() and not d.name.startswith(".")
            and d.name not in taken and (d / "SKILL.md").is_file()
        )
    except Exception:
        logger.debug("authored: could not list %s", getattr(project, "skills_root", "?"),
                     exc_info=True)
        return []


def body_hash(project: Any, name: str) -> Optional[str]:
    """The SHA-256 of the skill's instructions (its body, stripped), as the platform compares them."""
    from skillevaluation import skillfmt

    try:
        text = skill_md(project, name).read_text(encoding="utf-8")
    except OSError:
        return None
    split = skillfmt.split_frontmatter(text)
    body = split[1] if split else text
    return skillfmt.body_sha256(body)


def _array(names: List[str]) -> str:
    from .skillpkg.manifest import _quote

    return "[" + ", ".join(_quote(n) for n in names) + "]"


def _edit(text: Optional[str], names: List[str]) -> str:
    """The manifest text with ``[authored].skills`` set to ``names``; every other byte kept."""
    from .skillpkg.manifest import HEADER, _lines, _quote, _scan
    from .skillpkg.project import ROOT

    line = f"{KEY} = {_array(names)}"
    if text is None:
        return (HEADER + f"root = {_quote(ROOT)}\n\n[skills]\n\n"
                + f"{_TABLE_COMMENT}\n[{TABLE}]\n{line}\n")
    bom = "﻿" if text.startswith("﻿") else ""
    body = text[len(bom):]
    newline = "\r\n" if "\r\n" in body else "\n"
    lines = _lines(body)
    stmts = _scan(lines)
    kv = next((s for s in stmts if s.kind == "kv" and s.path == (TABLE, KEY)), None)
    if kv is not None:
        if kv.table != (TABLE,):
            raise ValueError(f"[{TABLE}] {KEY} is written in a form this editor does not rewrite "
                             f"(line {kv.line}); write it as `{KEY} = [\"<name>\"]` under [{TABLE}]")
        out = lines[:kv.start] + [line + newline] + lines[kv.end:]
    else:
        header = next((s for s in stmts if s.kind == "header" and s.table == (TABLE,)
                       and not s.array), None)
        if header is not None:
            out = lines[:header.end] + [line + newline] + lines[header.end:]
        else:
            out = list(lines)
            if out and not out[-1].endswith("\n"):
                out[-1] += newline
            if out and out[-1].strip():
                out.append(newline)
            out += [_TABLE_COMMENT + newline, f"[{TABLE}]" + newline, line + newline]
    return bom + "".join(out)


def _write(project: Any, names: List[str]) -> bool:
    """Set ``[authored].skills``; True when the file changed. Refuses an edit that does not
    read back exactly (the registry entries and the new list), writing nothing."""
    from .skillpkg import manifest as _manifest

    path = project.manifest_path
    project.refuse_symlinks(path)
    text = _read_text(project)
    before = _manifest.loads(text) if text is not None else _manifest.new()
    new_text = _edit(text, names)
    after = _manifest.loads(new_text)
    data = _toml_loads(new_text)
    table = data.get(TABLE)
    if (dict(after.entries) != dict(before.entries) or not isinstance(table, dict)
            or table.get(KEY) != names):
        raise ValueError(f"an edit of {project.rel(path)} did not read back as written; "
                         "nothing was saved")
    if text == new_text:
        return False
    project.makedirs(project.state_dir)
    project.write_file(path, new_text.encode("utf-8"))
    return True


def add(project: Any, name: str) -> bool:
    """List ``name`` under ``[authored]``; True when it was not listed yet."""
    names = listed(project)
    if name in names:
        return False
    _write(project, names + [name])
    return True


def remove(project: Any, name: str) -> bool:
    """Drop ``name`` from ``[authored]`` (its folder stays); True when it was listed."""
    names = listed(project)
    if name not in names:
        return False
    _write(project, [n for n in names if n != name])
    return True
