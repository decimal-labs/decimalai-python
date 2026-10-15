"""pydantic-ai-harness: ``Skills`` inside a ``LocalWorkspace``, with ``FileSystem`` and ``Shell``.

Documented snippet::

    Agent(model, capabilities=[LocalWorkspace(project),
                               Skills(str(decimalai.skills_dir(relative_to=project))),
                               FileSystem(), Shell()])

The skills path is relative to the WORKSPACE: a library outside it still lists,
but ``read_file`` refuses its resources. A bad folder goes into the catalog's
``skipped`` list and a ``UserWarning``, never an exception.

The harness keeps only a skill's name, description and body (``SkillDefinition``);
``metadata`` is parsed and dropped. So the stamp is read from the file at the
path the harness reports, and the "not exposed" fact is re-proven each run
(``proofs["definition_fields"]``). ``allowed-tools`` and
``disable-model-invocation`` are not enforced; the harness says so with a
``UserWarning`` on every run (allowlisted in the ledger, self-cleaning).
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
import warnings
from pathlib import Path
from typing import Any, Dict

from ._observe import (
    Ctx,
    attempt,
    executable,
    new_observation,
    read_text,
    stamp_of,
    unique,
    without_line_numbers,
)

NAME = "pydantic-ai-harness"


def _relative_root(ctx: Ctx) -> str:
    return str(ctx.skills_dir(relative_to=ctx.project))


def discover(ctx: Ctx) -> Dict[str, Any]:
    out: Dict[str, Any] = {"listed": None, "raised": None, "channel": []}
    try:
        from pydantic_ai.workspaces import LocalWorkspaceBackend
        from pydantic_ai_harness import Skills

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cat = asyncio.run(Skills(_relative_root(ctx)).load(LocalWorkspaceBackend(ctx.project)))
        out["listed"] = sorted(s.name for s in cat.skills)
        out["channel"] = [f"skipped: {s}"[:300] for s in cat.skipped] + [
            f"{w.category.__name__}: {w.message}"[:300] for w in caught
            if issubclass(w.category, UserWarning)]
    except Exception as exc:  # noqa: BLE001
        out["raised"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    return out


def observe(ctx: Ctx) -> Dict[str, Any]:
    sc = ctx.scenario
    obs = new_observation(NAME, "pydantic-ai-harness", "pydantic-ai-slim")

    from pydantic_ai import Agent
    from pydantic_ai.capabilities import LocalWorkspace
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
    )
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from pydantic_ai.workspaces import LocalWorkspaceBackend
    from pydantic_ai_harness import FileSystem, Shell, Skills
    from pydantic_ai_harness.skills import SkillDefinition

    obs["proofs"]["definition_fields"] = sorted(f.name for f in dataclasses.fields(SkillDefinition))
    rel = _relative_root(ctx)
    seen: Dict[str, Any] = {}

    def last_return(messages):
        for m in reversed(messages):
            if isinstance(m, ModelRequest):
                for p in m.parts:
                    if isinstance(p, ToolReturnPart):
                        return p.content
        return None

    def scripted(messages, info: AgentInfo) -> ModelResponse:
        n = sum(isinstance(m, ModelResponse) for m in messages)
        if n == 0:
            seen["instructions"] = info.instructions or ""
            return ModelResponse(parts=[ToolCallPart("load_capability", {"id": sc.target},
                                                     tool_call_id="c1")])
        r = last_return(messages)
        if n == 1:
            seen["load"] = r
            text = r.get("instructions", "") if isinstance(r, dict) else str(r)
            m = re.search(r"Skill directory: `([^`]+)`", text)
            if not m:   # the harness did not load it, so the model has no directory to use
                return ModelResponse(parts=[TextPart(f"could not load {sc.target}")])
            seen["dir"] = m.group(1)
            return ModelResponse(parts=[ToolCallPart(
                "read_file", {"path": f"{seen['dir']}/{sc.resource}"}, tool_call_id="c2")])
        if n == 2:
            seen["fees"] = r
            cmd = " ".join(("python3", f"{seen['dir']}/{sc.script}", *sc.script_args))
            return ModelResponse(parts=[ToolCallPart("run_command", {"command": cmd},
                                                     tool_call_id="c3")])
        seen["script"] = r
        return ModelResponse(parts=[TextPart("done")])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        catalog = attempt(obs, "discover", lambda: asyncio.run(
            Skills(rel).load(LocalWorkspaceBackend(ctx.project))))
        if catalog is not None:
            agent = Agent(FunctionModel(scripted), capabilities=[
                LocalWorkspace(ctx.project), Skills(rel), FileSystem(), Shell()])
            attempt(obs, "run", lambda: asyncio.run(agent.run("refund for $100 after 45 days?")))
    obs["channel"] = ([f"skipped: {s}"[:300] for s in (catalog.skipped if catalog else ())]
                      + [f"{w.category.__name__}: {w.message}"[:400] for w in caught
                         if issubclass(w.category, UserWarning)])
    if catalog is None:
        return obs
    by = {s.name: s for s in catalog.skills}
    obs["listed"] = sorted(by)
    instructions = seen.get("instructions") or ""
    if instructions:
        obs["offered"] = unique(re.findall(r"(?m)^- ([^\s:]+): ", instructions))
        obs["user_only"]["offered"] = sc.user_only in obs["offered"]
        obs["notes"]["instructions"] = instructions[-1500:]
    load = seen.get("load")
    obs["body_text"] = (load.get("instructions") if isinstance(load, dict) else
                        (str(load) if load is not None else None))
    target = by.get(sc.target)
    if target is not None:
        obs["body"] = target.body
        reported = _host_path(ctx, target.path)
        obs["stamp"] = {"via": "file", "path": str(reported),
                        "metadata": stamp_of(read_text(reported))}
        script = reported.parent / sc.script
        bit = executable(script)
        obs["exec_bit"] = {"path": str(script), "source": bit, "copy": bit}
        obs["notes"]["ignored_behavioral_fields"] = list(target.ignored_behavioral_fields)
    fees = seen.get("fees")
    obs["notes"]["resource_raw"] = None if fees is None else str(fees)[:600]
    obs["resource_text"] = _file_text(fees)
    obs["resource_exact"] = False
    script_out = seen.get("script")
    obs["script_stdout"] = None if script_out is None else str(script_out)
    return obs


def _host_path(ctx: Ctx, workspace_path: str) -> Path:
    p = Path(workspace_path)
    return p if p.is_absolute() and p.exists() else (ctx.project / workspace_path.lstrip("/"))


_VIEW_HEADER = re.compile(r"\A\[[^\]\n]*\| \d+ lines?[^\]\n]*\]\n")


def _file_text(result):
    """read_file's output with its viewer decoration (a header line, line numbers) removed."""
    if result is None:
        return None
    text = result if isinstance(result, str) else str(result)
    return without_line_numbers(_VIEW_HEADER.sub("", text))
