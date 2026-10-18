"""Microsoft Agent Framework: its own skill loader, with DecimalAI seeing what it loads.

Setup::

    import decimalai
    from agent_framework import Agent
    from decimalai.agent_framework import skills

    decimalai.init(otel=True, agent_name="support")
    agent = Agent(client=client, instructions="Answer refund questions.",
                  context_providers=[skills()])

``skills()`` is ``SkillsProvider.from_paths(decimalai.skills_dir(), ...)``: Agent
Framework's own loader, pointed at the folders ``decimalai skills add`` wrote. It lists
them in the system prompt, and the model loads one with ``load_skill``, reads a file with
``read_skill_resource`` and runs a script with ``run_skill_script``. DecimalAI adds three
things, all on that one object:

* **The activation.** A function middleware the provider adds to every run watches
  ``load_skill`` and hands what the tool returned (the ``SKILL.md`` as Agent Framework
  served it) to :func:`decimalai._native.record`, which names the skill and the version
  from those bytes, never from the folder on disk. Agent Framework caches skills by
  default, so after ``decimalai skills update`` a long-lived provider keeps serving the old
  version, and the activation says the old version, because that is what the model read.
  The skills the provider listed in the prompt are recorded as offered.
* **A script runner.** Agent Framework ships none: without one a file script answers that
  it "requires a runner". :func:`script_runner` runs a ``.py`` script with this
  interpreter, from the working directory, with a timeout.
* **An approval rule.** Every skill tool waits for approval by default, so a plain
  provider stops at the first ``load_skill``. Loading a skill and reading its files only
  put text in front of the model that the listing already offered, so those run unasked;
  running a script executes code on this machine, so it still waits for approval unless
  you pass ``auto_approve_scripts=True``.

Where the activation goes: Agent Framework emits OpenTelemetry spans of its own
(``invoke_agent``, ``chat``, ``execute_tool``) to the global tracer provider, and
``decimalai.init(otel=True)`` installs DecimalAI's exporter there, so a run arrives as one
trace with its model calls and tool spans, and the load is recorded on that run. Inside a
``decimalai.start_trace()`` the load is recorded on that trace instead. With neither, the
load is dropped and a warning says so once. Pass ``agent_name`` to ``init``: the exporter
names a trace by it, not by the ``Agent``'s own name. Agent Framework keeps tool arguments
and results out of its spans unless you call
``agent_framework.observability.enable_sensitive_telemetry()``; without it the activation is
still recorded, but the trace does not show what each tool call read or returned.

There is no router to turn off here: DecimalAI has no Agent Framework adapter that injects
skills, so the loader above is the only catalog the model sees.

SUPPORTED: agent-framework-core 1.21.0. The seams, read out of that release:
``ContextProvider.before_run`` and ``SessionContext.extend_middleware`` (a provider may
add function middleware for the run it prepares), ``FunctionMiddleware``, the tool names
``SkillsProvider.LOAD_SKILL_TOOL_NAME`` etc., and ``FileSkill.get_content``, which returns
the raw ``SKILL.md`` followed by an ``<available_resources>`` and an
``<available_scripts>`` block, always both, always last.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import re
import sys
from html import unescape
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

logger = logging.getLogger("decimalai.agent_framework")

FRAMEWORK = "agent-framework"

#: Agent Framework appends these after the raw ``SKILL.md`` in ``load_skill``'s answer
#: (``FileSkill.get_content``). The resources block always comes first of the two.
_MAF_BLOCKS = "\n\n<available_resources"
_LISTED_NAME = re.compile(r"<name>([^<]+)</name>")

#: Seconds a skill script may run before :func:`script_runner` stops it.
DEFAULT_SCRIPT_TIMEOUT = 60.0


def skills(path: Union[str, Path, None] = None, *, auto_approve_reads: bool = True,
           auto_approve_scripts: bool = False, script_runner: Optional[Callable] = None,
           **kwargs: Any) -> Any:
    """Agent Framework's ``SkillsProvider`` for the project's installed skills, observed.

    Args:
        path: the skills root (default: :func:`decimalai.skills_dir`).
        auto_approve_reads: run ``load_skill`` and ``read_skill_resource`` without asking.
        auto_approve_scripts: run ``run_skill_script`` without asking. Off by default: a
            script runs code on this machine, and Agent Framework's own default is to ask.
        script_runner: how file scripts run (default: :func:`script_runner`).
        **kwargs: passed to ``SkillsProvider.from_paths`` (``disable_caching``,
            ``cache_refresh_interval``, ``instruction_template``, ...). An explicit
            ``disable_*_approval`` wins over the two ``auto_approve_*`` flags.

    Returns:
        A ``SkillsProvider`` to put in the agent's ``context_providers``.
    """
    if path is None:
        from .skillpkg.project import skills_dir

        path = skills_dir()
    kwargs.setdefault("disable_load_skill_approval", auto_approve_reads)
    kwargs.setdefault("disable_read_skill_resource_approval", auto_approve_reads)
    kwargs.setdefault("disable_run_skill_script_approval", auto_approve_scripts)
    runner = script_runner if script_runner is not None else globals()["script_runner"]()
    return _provider_class().from_paths(str(path), script_runner=runner, **kwargs)


def script_runner(*, timeout: float = DEFAULT_SCRIPT_TIMEOUT,
                  python: Optional[str] = None) -> Callable:
    """A ``SkillScriptRunner`` that runs a skill's script as a subprocess.

    A ``.py`` script runs with ``python`` (default: this interpreter, so a script can import
    what the agent can); any other script runs directly when it is executable. The
    arguments are the model's: a list is passed as is, a mapping as ``--key value`` flags
    (``True`` as a bare ``--key``). The script runs from the working directory, as Claude
    Code and the other loaders run them, so a path argument means what the user meant.
    Its stdout is the answer; a failure, or a run longer than ``timeout`` seconds, answers
    with what went wrong instead of raising, so the model can read it.
    """

    async def run(skill: Any, script: Any, args: Any = None) -> str:
        full_path = str(script.full_path)
        argv = _argv(args)
        if full_path.endswith(".py"):
            command = [python or sys.executable, full_path, *argv]
        elif os.access(full_path, os.X_OK):
            command = [full_path, *argv]
        else:
            return (f"Error: {script.name} is neither a .py script nor executable, so it "
                    "cannot be run.")
        proc = await asyncio.create_subprocess_exec(
            *command, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Error: {script.name} did not finish within {timeout:g}s and was stopped."
        stdout = out.decode("utf-8", "replace")
        if proc.returncode == 0:
            return stdout
        stderr = err.decode("utf-8", "replace")
        return (f"Error: {script.name} exited with status {proc.returncode}.\n{stdout}"
                + (f"\n--- stderr ---\n{stderr}" if stderr else ""))

    return run


def _argv(args: Any) -> List[str]:
    if args is None:
        return []
    if isinstance(args, str):
        return [args]
    if isinstance(args, dict):
        out: List[str] = []
        for key, value in args.items():
            flag = f"--{key}"
            if value is True:
                out.append(flag)
            elif value is None or value is False:
                continue
            elif isinstance(value, (list, tuple)):
                for item in value:
                    out += [flag, str(item)]
            else:
                out += [flag, str(value)]
        return out
    return [str(a) for a in args]


# ── the observer ─────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def _provider_class() -> type:
    """The provider subclass, built on first use so importing this module never needs
    agent-framework installed."""
    from agent_framework import FunctionMiddleware, SkillsProvider

    class _LoadObserver(FunctionMiddleware):
        """Records each ``load_skill`` answer of one provider, after the tool ran."""

        def __init__(self, source_id: str) -> None:
            self._source_id = source_id

        async def process(self, context: Any, call_next: Callable) -> None:
            # A raise in the tool or a later middleware propagates untouched; only what
            # this observer itself does is guarded.
            await call_next()
            try:
                _observe_load(context, self._source_id)
            except Exception:  # noqa: BLE001 - observing must never break the agent's run
                logger.debug("could not observe a load_skill call", exc_info=True)

    class DecimalSkillsProvider(SkillsProvider):
        """``SkillsProvider`` that reports each skill it lists and each one the model loads."""

        async def before_run(self, **kwargs: Any) -> None:
            context = kwargs.get("context")
            already = len(getattr(context, "instructions", None) or [])
            await super().before_run(**kwargs)
            try:
                added = list(context.instructions[already:])
                _record_listed(_listed_names(added))
                # The tools were built for this run, so the middleware goes on this run
                # too: an agent-level one would watch every other provider's tools.
                context.extend_middleware(self.source_id, _LoadObserver(self.source_id))
            except Exception:  # noqa: BLE001
                logger.debug("could not attach the skill observer", exc_info=True)

    return DecimalSkillsProvider


def _observe_load(context: Any, source_id: str) -> None:
    from agent_framework import SkillsProvider

    function = context.function
    if getattr(function, "name", None) != SkillsProvider.LOAD_SKILL_TOOL_NAME:
        return
    # Tools a provider adds carry its source id; a load_skill of someone else's is not ours.
    props = getattr(function, "additional_properties", None) or {}
    if props.get("context_source", source_id) != source_id:
        return
    text = _result_text(context.result)
    if not text or text.startswith("Error:"):
        return  # an empty or unknown name: nothing was delivered
    arguments = context.arguments
    name = (arguments.get("skill_name") if hasattr(arguments, "get")
            else getattr(arguments, "skill_name", None))
    from ._native import Load, record

    record(Load(framework=FRAMEWORK, name=name if isinstance(name, str) else None,
                text=_skill_md(text)))


def _skill_md(delivered: str) -> str:
    """The ``SKILL.md`` part of a ``load_skill`` answer: the blocks Agent Framework appends
    are not the skill's, and hashing them would make every load unverifiable."""
    cut = delivered.rfind(_MAF_BLOCKS)
    return delivered[:cut] if cut != -1 else delivered


def _result_text(result: Any) -> Optional[str]:
    """The text a function result gives the model: a string, or ``list[Content]``."""
    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)):
        parts = []
        for item in result:
            if isinstance(item, str):
                parts.append(item)
            elif getattr(item, "type", None) == "text":
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts) if parts else None
    return None


def _listed_names(instructions: List[Any]) -> List[str]:
    names: List[str] = []
    for text in instructions:
        if isinstance(text, str):
            names += [unescape(n).strip() for n in _LISTED_NAME.findall(text)]
    return [n for n in names if n]


def _record_listed(names: List[str]) -> None:
    if not names:
        return
    loads = _live_loads()
    if loads is not None:
        loads.listed(names)


def _live_loads() -> Any:
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


__all__ = ["skills", "script_runner", "DEFAULT_SCRIPT_TIMEOUT"]
