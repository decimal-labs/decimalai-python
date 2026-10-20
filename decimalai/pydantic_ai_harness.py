"""Pydantic AI harness native mode: its own ``Skills`` reads the DecimalAI-installed skills.

Setup::

    import decimalai
    decimalai.init(otel=True)
    Agent.instrument_all()                      # Pydantic AI's own spans: one trace per run

    from pydantic_ai_harness import FileSystem, Shell
    import decimalai.pydantic_ai_harness

    agent = Agent(model, capabilities=[
        LocalWorkspace(project),
        decimalai.pydantic_ai_harness.skills(project),
        FileSystem(), Shell()])

:func:`skills` is the harness's ``Skills`` over ``decimalai.skills_dir()``, read relative to
the workspace as the harness requires, carrying an observer. The harness offers each skill
as a deferred capability and the model loads one with ``load_capability``, whose result is
the skill's rendered instructions::

    # Skill: acme-refund-policy

    Skill directory: `/abs/.agents/skills/acme-refund-policy`. Relative paths ...

    <the SKILL.md body>

The observer takes that result, cuts the harness's heading back off, and reports the body
as an activation carrying the registry identity of exactly those bytes
(:mod:`decimalai._native`). The harness parses the frontmatter and keeps none of
``metadata``, so the stamp never reaches the model: identity comes from the body's hash in
the lock. The harness reads its skill libraries at the start of every run, so a run after
``decimalai skills update`` delivers the new version, and the activation says so.

The observer records onto the run that is live when the run starts: a
``decimalai.start_trace()`` trace, or the OpenTelemetry run (Pydantic AI's own run span,
with ``Agent.instrument_all()``) whose trace the exporter assembles. With neither, a load
is dropped and a warning says so once.
"""

from __future__ import annotations

import logging
import re
import sys
import threading
from pathlib import Path
from typing import Any, List, Optional, Union

logger = logging.getLogger("decimalai.pydantic_ai_harness")

FRAMEWORK = "pydantic-ai-harness"

#: ``SkillDefinition.render`` puts this line between the heading and the body when the
#: skill sits in the run's workspace.
_DIRECTORY_LINE = re.compile(
    r"Skill directory: `([^`\n]*)`\. Relative paths in this skill resolve against it\.\n\n")

_double_catalog_warned = False
_warn_lock = threading.Lock()


def skills(workspace: Union[str, Path, None] = None, **options: Any) -> Any:
    """The harness's ``Skills`` capability over the installed skills, observed.

    Args:
        workspace: the directory the agent's ``LocalWorkspace`` is rooted at (default:
            the working directory). The harness reads skill libraries relative to the
            run's workspace; a library outside it still lists, but the model's
            ``read_file`` refuses its resources.
        **options: passed to ``Skills`` as given (``include``, ``exclude``,
            ``missing_directories``, ``duplicate_names``).

    Raises:
        decimalai.skillpkg.errors.SkillsNotInstalledError: no ``decimalai skills``
            install in this project (see :func:`decimalai.skills_dir`).
        decimalai.skillpkg.errors.UsageError: the skills root is not inside ``workspace``.
    """
    from .skillpkg.project import skills_dir

    root = Path(workspace) if workspace is not None else Path.cwd()
    return _observed_skills_class()(str(skills_dir(relative_to=root)), **options)


def is_skills(capability: Any) -> bool:
    """True for the harness's ``Skills`` (observed or not). False without the harness."""
    harness = sys.modules.get("pydantic_ai_harness")
    if harness is None:
        return False  # never imported, so no agent in this process can hold one
    cls = getattr(harness, "Skills", None)
    return isinstance(cls, type) and isinstance(capability, cls)


def is_observed(capability: Any) -> bool:
    """True for a ``Skills`` made by :func:`skills`, whose loads are reported."""
    cls = _classes.get("skills")
    return cls is not None and isinstance(capability, cls)


# ── the observer ─────────────────────────────────────────────────────────────

_classes: dict = {}


def _observed_skills_class() -> type:
    """Built on first use: the harness is an optional dependency of this SDK."""
    cls = _classes.get("skills")
    if cls is not None:
        return cls
    from pydantic_ai.capabilities import AbstractCapability, CombinedCapability
    from pydantic_ai_harness import Skills

    class _LoadObserver(AbstractCapability):
        """One run's observer: knows the run's skill catalog and, once the run starts,
        the run's collector."""

        def __init__(self, ids: List[str]) -> None:
            self.catalog = list(ids)
            self.ids = frozenset(ids)
            self.loads: Any = None

        async def before_run(self, ctx: Any) -> None:
            # Inside every wrap_run, so Pydantic AI's own run span is open here. It is
            # not yet open in Skills.for_run, which runs while the run is being set up.
            try:
                self.loads = _run_carrier()
                if self.loads is not None:
                    self.loads.listed(_wire_names(self.catalog))
            except Exception:  # noqa: BLE001 - observing must never cost the run
                logger.debug("could not record the skill catalog (non-fatal)", exc_info=True)

        async def after_tool_execute(self, ctx: Any, *, call: Any, tool_def: Any,
                                     args: Any, result: Any) -> Any:
            try:
                _observe_load(self, ctx, tool_def, args, result)
            except Exception:  # noqa: BLE001 - observing must never cost the run
                logger.debug("load_capability observation failed (non-fatal)", exc_info=True)
            return result

    class ObservedSkills(Skills):
        """``Skills``, whose per-run capability also reports each skill the model loads."""

        async def for_run(self, ctx: Any) -> Any:
            run_capability = await super().for_run(ctx)
            try:
                ids = [c.id for c in getattr(run_capability, "capabilities", ())
                       if isinstance(getattr(c, "id", None), str)]
                if not ids:
                    return run_capability  # no skills this run: nothing to observe
                return CombinedCapability([run_capability, _LoadObserver(ids)])
            except Exception:  # noqa: BLE001
                logger.debug("could not attach the skills observer (non-fatal)", exc_info=True)
                return run_capability

    _classes["skills"] = ObservedSkills
    return ObservedSkills


def _observe_load(observer: Any, ctx: Any, tool_def: Any, args: Any, result: Any) -> None:
    if getattr(tool_def, "tool_kind", None) != "capability-load":
        return
    skill = args.get("id") if isinstance(args, dict) else None
    if skill not in observer.ids:
        return  # another deferred capability, not a skill
    text = _instructions(result)
    if text is None:
        return
    body, directory = _body(skill, text)
    from ._native import Load, record

    load = Load(framework=FRAMEWORK, name=skill, body=body, path=directory)
    activation = record(load, loads=observer.loads) if observer.loads is not None else record(load)
    if activation is not None:
        _warn_if_router_injected(ctx, activation.name)


def _instructions(result: Any) -> Optional[str]:
    """The text ``load_capability`` returned to the model."""
    value = getattr(result, "return_value", result)  # a ToolReturn, or the bare dict
    text = value.get("instructions") if isinstance(value, dict) else None
    return text if isinstance(text, str) else None


def _body(skill: str, text: str) -> tuple:
    """``(body, skill directory)`` from what ``SkillDefinition.render`` produced: the
    heading and the directory line cut off. A text in any other shape is returned
    whole, which hashes to no version: the load is then reported unverified."""
    heading = f"# Skill: {skill}"
    if not text.startswith(heading):
        logger.debug("load_capability(%r) returned no harness heading; reporting it as is", skill)
        return text, None
    rest = text[len(heading):]
    if rest == "":
        return "", None  # an empty body renders as the heading alone
    if not rest.startswith("\n\n"):
        return text, None
    rest = rest[2:]
    directory = _DIRECTORY_LINE.match(rest)
    if directory is None:
        return rest, None
    return rest[directory.end():], directory.group(1)


def _run_carrier() -> Any:
    """The collector of the run starting now: the open ``start_trace()`` trace's, else the
    OpenTelemetry run's. Taken at the run's start (``before_run``), so a tool that runs in
    another task still records onto its own run."""
    from .generic import _get_current_trace

    trace = _get_current_trace()
    if trace is not None:
        return trace._native_loads()
    try:
        from .otel import native_loads_for_current_run
    except ImportError:
        return None
    return native_loads_for_current_run()


def _wire_names(names: List[str]) -> List[str]:
    """Install names as a trace reports them: a DecimalAI-installed skill by its wire
    name, the name its activation carries."""
    try:
        from .skillpkg.identity import current_index

        index = current_index()
    except Exception:  # noqa: BLE001
        return list(names)
    out = []
    for name in names:
        managed = index.by_install_name(name)
        out.append(managed.wire_name if managed is not None else name)
    return out


def _warn_if_router_injected(ctx: Any, name: str) -> None:
    """Two catalogs reached the model: the router's menu and the harness's."""
    global _double_catalog_warned
    adapter = sys.modules.get("decimalai.pydantic_ai")
    if adapter is None or _double_catalog_warned:
        return
    if not adapter._menu_shown(ctx):
        return
    with _warn_lock:
        if _double_catalog_warned:
            return
        _double_catalog_warned = True
    logger.warning(
        "decimalai: the model loaded the skill %s through the harness's Skills, on a run "
        "decimalai.pydantic_ai.load_agent() also put its skill list into, so it saw two "
        "catalogs of skills. load_agent() leaves an Agent built with Skills alone; this "
        "one got its Skills some other way (a capability function, or capabilities passed "
        "to the run). Logged once.", name)
