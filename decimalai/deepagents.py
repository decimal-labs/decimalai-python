"""Deep Agents native mode: its own skill loader reads the DecimalAI-installed skills.

Setup::

    import decimalai
    decimalai.init(langchain=True)

    from deepagents import create_deep_agent
    import decimalai.deepagents

    agent = create_deep_agent(model, **decimalai.deepagents.skills())

Deep Agents is a LangGraph graph, so the LangChain adapter traces it. Its skills
middleware lists each skill (name, description, ``SKILL.md`` path) in the system prompt,
and the model loads one by calling ``read_file`` on that path. The adapter sees the read
in its tool callbacks and reports it as an activation carrying the registry identity of
the bytes ``read_file`` returned (:mod:`decimalai._native`), never of the file as it is
on disk by the time the trace is built. The adapter puts nothing into the agent's model
calls (``init(langchain=True)`` traces only). ``agent_config()``'s middleware on the same
agent would show the model a second skill list, and it says so when it sees Deep Agents'.

What ``read_file`` hands the model, measured on deepagents 0.7.25: optional bracketed
notices, a status header ``@@ lines 1-25 of 25 @@`` (more fields when the window stops
short: ``next offset N``, ``truncated ...``), then the file's lines verbatim, with no
line numbers. A read that is not the whole file delivers part of the body, whose hash
then names no version: the activation is reported unverified, which is the truth.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger("decimalai.deepagents")

#: The tool the model reads a skill with, and the state key of the skills middleware's
#: parsed listing (``SkillsState.skills_metadata``).
READ_TOOL = "read_file"
LISTING_KEY = "skills_metadata"
#: The middleware's graph nodes are named ``<middleware name>.<hook>``.
MIDDLEWARE_NAME = "SkillsMiddleware"

_HEADER = re.compile(r"^@@ [^\n]* @@$", re.MULTILINE)


def skills(*, backend: Any = None) -> Dict[str, Any]:
    """The ``create_deep_agent`` arguments that make the installed skills visible.

    Returns ``{"skills": [<skills root>], "backend": <backend>}``, to splat:
    ``create_deep_agent(model, **decimalai.deepagents.skills())``. Both are needed:
    with Deep Agents' default backend (agent state, not the disk) the skills
    middleware finds no ``SKILL.md`` and the prompt silently lists nothing.

    Args:
        backend: your own backend, returned as given. Pass one when the agent also
            needs a shell, for example
            ``LocalShellBackend(root_dir=project, virtual_mode=False)`` so the model
            can run a skill's scripts. It must read real paths (``virtual_mode=False``):
            in virtual mode the listing advertises paths a shell cannot open.
            Default: a ``FilesystemBackend`` over the project, which reads files and
            runs nothing, because a helper that only points at skills should not
            hand the model a shell.

    Raises:
        decimalai.skillpkg.errors.SkillsNotInstalledError: no ``decimalai skills``
            install in this project (see :func:`decimalai.skills_dir`).
    """
    from .skillpkg.project import Project, skills_dir

    root = skills_dir()
    if backend is None:
        from deepagents.backends import FilesystemBackend

        # Rooted at the project, not at the skills root, so a skill's instructions can
        # point the model at project files too; the listing's paths are absolute either way.
        backend = FilesystemBackend(root_dir=str(Project.find().directory), virtual_mode=False)
    return {"skills": [str(root)], "backend": backend}


# ── what the LangChain adapter asks of a Deep Agents run ─────────────────────


def is_skills_node(name: Any) -> bool:
    """True for a graph node the skills middleware adds (``SkillsMiddleware.before_agent``)."""
    return isinstance(name, str) and name.split(".", 1)[0] == MIDDLEWARE_NAME


def listing(inputs: Any) -> Optional[List[Dict[str, str]]]:
    """``[{"name", "path"}, ...]`` from a node's input state, when it carries the skills
    middleware's parsed listing; None when it does not (no skills middleware here)."""
    if not isinstance(inputs, dict) or LISTING_KEY not in inputs:
        return None
    out: List[Dict[str, str]] = []
    for item in inputs.get(LISTING_KEY) or ():
        if isinstance(item, dict) and isinstance(item.get("name"), str) \
                and isinstance(item.get("path"), str):
            out.append({"name": item["name"], "path": item["path"]})
    return out


def skill_read_path(tool_name: Any, inputs: Any) -> Optional[str]:
    """The ``SKILL.md`` path a ``read_file`` call reads, or None for any other call.

    Any ``SKILL.md`` counts, under whatever skills source the agent was given: the
    model reads the path its listing shows, and the listing is the middleware's.
    """
    if tool_name != READ_TOOL or not isinstance(inputs, dict):
        return None
    path = inputs.get("file_path")
    if isinstance(path, str) and os.path.basename(path.rstrip("/")) == "SKILL.md":
        return path
    return None


def delivered_text(output: Any) -> Optional[str]:
    """The file text a ``read_file`` result handed the model, without the viewer's
    decoration (notices and the ``@@ ... @@`` status header), or None when the read
    failed or returned no text."""
    message = _tool_message(output)
    if message is None:
        content = output if isinstance(output, str) else None
    else:
        if getattr(message, "status", "success") == "error":
            return None
        content = _text(getattr(message, "content", None))
    if not content:
        return None
    header = _HEADER.search(content)
    # No header: an empty file's system reminder, or a backend that renders reads its
    # own way. Either way it is all the model got.
    return content[header.end() + 1:] if header else content


def listed_on_prompt(skills: Iterable[Dict[str, str]], system_text: str) -> List[str]:
    """The names of the listed skills whose ``SKILL.md`` path is in the system prompt:
    what the middleware put in front of the model (it renders ``Read `<path>```)."""
    return [s["name"] for s in skills if s.get("path") and f"`{s['path']}`" in system_text]


def pinned_skill(message: Any) -> Optional[Dict[str, str]]:
    """``{"name", "path", "body"}`` of a message the middleware appended to pin a skill
    (``pinned_skills=[...]``), or None.

    The middleware wraps the body as ``<skill name=".." path="..">\\n<body>\\n</skill>``
    after cutting the frontmatter, so the body is what sits between the two lines.
    """
    extra = getattr(message, "additional_kwargs", None)
    if not isinstance(extra, dict) or extra.get("lc_source") != "pinned_skill":
        return None
    marker = extra.get("skill")
    content = _text(getattr(message, "content", None))
    if not isinstance(marker, dict) or not content:
        return None
    first, _, rest = content.partition("\n")
    if not first.startswith("<skill ") or not rest.endswith("</skill>"):
        return None
    body = rest[: -len("</skill>")]
    return {"name": str(marker.get("name") or ""), "path": str(marker.get("path") or ""),
            "body": body[:-1] if body.endswith("\n") else body}


def install_name_of(path: str) -> Optional[str]:
    """The skill folder's name for a ``SKILL.md`` path."""
    folder = Path(path).parent.name
    return folder or None


def _tool_message(output: Any) -> Any:
    """The ``ToolMessage`` in a tool's output: the output itself, or the one a
    ``Command(update={"messages": [...]})`` carries."""
    if hasattr(output, "tool_call_id") and hasattr(output, "content"):
        return output
    update = getattr(output, "update", None)
    messages = update.get("messages") if isinstance(update, dict) else None
    for message in messages or ():
        if hasattr(message, "tool_call_id") and hasattr(message, "content"):
            return message
    return None


def _text(content: Any) -> Optional[str]:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        texts = [p for p in parts if isinstance(p, str)]
        return "".join(texts) if texts else None
    return None
