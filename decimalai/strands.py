"""Strands Agents: its own skill loader, with DecimalAI seeing what it loads.

Setup::

    import decimalai
    from strands import Agent
    from decimalai.strands import skills

    decimalai.init(otel=True, agent_name="support")
    agent = Agent(model=model, system_prompt="Answer refund questions.",
                  tools=[file_editor, shell], plugins=[skills()])

``skills()`` is ``AgentSkills(skills=decimalai.skills_dir(), strict=True)``: Strands' own
plugin, pointed at the folders ``decimalai skills add`` wrote, by absolute path (Strands'
``file_editor`` refuses a relative one). It lists the skills in the system prompt, and the
model activates one with the ``skills`` tool, which answers with the skill's instructions,
its ``Location:`` and its files. Reading a file or running a script then goes through the
agent's own tools (``file_editor``, ``shell``); the plugin has none of its own.

What DecimalAI adds is an observer, on the same object: the plugin returned is a subclass
of ``AgentSkills`` with two more hooks, so it is still the one plugin Strands knows by the
name ``agent_skills``, and nothing else has to be registered or kept in step with it.

* After each ``skills`` call (``AfterToolCallEvent``) it hands the body the model received
  to :func:`decimalai._native.record`, with the metadata Strands parsed from the same
  ``SKILL.md``: the tool's answer carries no frontmatter, so the stamp comes from the parsed
  skill, and the answer is checked to begin with that exact body first. Strands reads an
  agent's skills once and keeps them, so after ``decimalai skills update`` the same agent
  keeps serving the old version, and the activation says the old version.
* Before each invocation (``BeforeInvocationEvent``) it records the skills the plugin lists
  in the prompt as offered.

Where the activation goes: Strands emits OpenTelemetry spans of its own (``invoke_agent``,
``execute_event_loop_cycle``, ``chat``, ``execute_tool``) to the global tracer provider,
and ``decimalai.init(otel=True)`` installs DecimalAI's exporter there, so a run arrives as
one trace with its model calls and tool spans, and the load is recorded on that run. Inside
a ``decimalai.start_trace()`` it is recorded on that trace instead. With neither, the load
is dropped and a warning says so once. Pass ``agent_name`` to ``init``: the exporter names
a trace by it, not by the ``Agent``'s own name. By default Strands puts message and tool
content in span EVENTS, which the exporter does not read; set
``OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental,gen_ai_span_attributes_only``
before the first agent is built (Strands reads it once) for tool arguments, tool results
and messages as span attributes.

There is no router to turn off here: DecimalAI has no Strands adapter that injects skills,
so the plugin above is the only catalog the model sees.

SUPPORTED: strands-agents 1.61.0. The seams, read out of that release: ``Plugin`` hooks
declared with ``@hook`` are discovered over the class's MRO, parents first, so these run
after ``AgentSkills``' own before-invocation hook; ``AgentSkills.get_available_skills(agent)``
returns the agent's own skill set, the one its ``skills`` tool serves from; and that tool
answers ``skill.instructions`` first, then its metadata lines and resource listing.
"""

import functools
import logging
from pathlib import Path
from typing import Any, List, Optional, Union

logger = logging.getLogger("decimalai.strands")

FRAMEWORK = "strands"
#: The tool ``AgentSkills`` registers for activating a skill.
SKILLS_TOOL = "skills"


def skills(path: Union[str, Path, None] = None, *, strict: bool = True, **kwargs: Any) -> Any:
    """Strands' ``AgentSkills`` plugin for the project's installed skills, observed.

    Args:
        path: the skills root (default: :func:`decimalai.skills_dir`, which is absolute).
        strict: Strands' strict parse: a skill that fails validation is skipped with a log
            line instead of loaded with a warning.
        **kwargs: passed to ``AgentSkills`` (``state_key``, ``max_resource_files``).

    Returns:
        An ``AgentSkills`` to put in the agent's ``plugins``.
    """
    if path is None:
        from .skillpkg.project import skills_dir

        path = skills_dir()
    return _plugin_class()(skills=str(Path(path).resolve()), strict=strict, **kwargs)


@functools.lru_cache(maxsize=1)
def _plugin_class() -> type:
    """The plugin subclass, built on first use so importing this module never needs
    strands-agents installed. (No ``from __future__ import annotations`` in this module:
    ``@hook`` reads the event type from the annotation, which must be the class.)"""
    from strands import AgentSkills
    from strands.hooks import AfterToolCallEvent, BeforeInvocationEvent
    from strands.plugins import hook

    class DecimalAgentSkills(AgentSkills):
        """``AgentSkills`` that reports each skill it lists and each one the model loads."""

        @hook
        async def _decimalai_listed(self, event: BeforeInvocationEvent) -> None:
            try:
                _record_listed([s.name for s in self.get_available_skills(event.agent)])
            except Exception:  # noqa: BLE001 - observing must never break the agent's run
                logger.debug("could not record the listed skills", exc_info=True)

        @hook
        async def _decimalai_loaded(self, event: AfterToolCallEvent) -> None:
            try:
                _observe_load(self, event)
            except Exception:  # noqa: BLE001
                logger.debug("could not observe a skills call", exc_info=True)

    return DecimalAgentSkills


def _observe_load(plugin: Any, event: Any) -> None:
    use = event.tool_use or {}
    if use.get("name") != SKILLS_TOOL:
        return
    result = event.result
    if not isinstance(result, dict) or result.get("status") != "success":
        return
    name = (use.get("input") or {}).get("skill_name")
    skill = next((s for s in plugin.get_available_skills(event.agent) if s.name == name), None)
    if skill is None or not skill.instructions:
        return  # an unknown name, or a skill with no body: nothing was delivered
    text = "\n".join(c["text"] for c in result.get("content") or []
                     if isinstance(c, dict) and isinstance(c.get("text"), str))
    if not text.startswith(skill.instructions):
        # Another hook rewrote the answer: the parsed body is not what the model read.
        logger.debug("skills(%s) answered something other than the skill's body", name)
        return
    from ._native import Load, record

    record(Load(framework=FRAMEWORK, name=name, body=skill.instructions,
                metadata=skill.metadata if isinstance(skill.metadata, dict) else None,
                path=str(skill.path) if skill.path is not None else None))


def _record_listed(names: List[str]) -> None:
    if not names:
        return
    loads = _live_loads()
    if loads is not None:
        loads.listed(names)


def _live_loads() -> Optional[Any]:
    """The run's native-load collector, found the way :func:`decimalai._native.record`
    finds it: the open ``start_trace()`` trace, else the OpenTelemetry run that is live."""
    from .generic import _get_current_trace

    ctx = _get_current_trace()
    if ctx is not None:
        return ctx._native_loads()
    try:
        from .otel import native_loads_for_current_run
    except ImportError:
        return None
    return native_loads_for_current_run()


__all__ = ["skills"]
