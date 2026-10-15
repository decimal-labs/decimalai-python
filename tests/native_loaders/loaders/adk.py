"""google-adk: ``load_skills_from_dir`` plus ``SkillToolset`` on an ``LlmAgent``.

Documented snippet::

    skills = load_skills_from_dir(decimalai.skills_dir())
    SkillToolset(skills=skills, code_executor=UnsafeLocalCodeExecutor())

``load_skills_from_dir`` is the only loader call of the eight that RAISES on one
bad folder (the whole load fails), which is why it, and not
``list_skills_in_dir(on_error=...)``, is the documented call. Without a code
executor ADK offers no ``run_skill_script``. A scripted ``BaseLlm`` drives the
native tools: ``load_skill``, ``load_skill_resource``, ``run_skill_script``.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict

from ._observe import Ctx, attempt, executable, new_observation, watch_loggers

NAME = "google-adk"
CHANNEL = ("google_adk", "google.adk")


def discover(ctx: Ctx) -> Dict[str, Any]:
    channel = watch_loggers(*CHANNEL)
    out: Dict[str, Any] = {"listed": None, "raised": None, "channel": channel.lines}
    try:
        from google.adk.skills import load_skills_from_dir

        out["listed"] = sorted(s.name for s in load_skills_from_dir(ctx.skills_dir()))
    except Exception as exc:  # noqa: BLE001
        out["raised"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    return out


def observe(ctx: Ctx) -> Dict[str, Any]:
    sc = ctx.scenario
    obs = new_observation(NAME, "google-adk")
    channel = watch_loggers(*CHANNEL)
    obs["channel"] = channel.lines

    from google.adk.agents import LlmAgent
    from google.adk.code_executors.unsafe_local_code_executor import (
        UnsafeLocalCodeExecutor,
    )
    from google.adk.runners import InMemoryRunner
    from google.adk.skills import load_skills_from_dir
    from google.adk.tools.skill_toolset import SkillToolset
    from google.genai import types

    from ..stubs.adk_llm import ScriptedLlm, answer, call

    skills = attempt(obs, "discover", lambda: load_skills_from_dir(ctx.skills_dir()))
    if skills is None:
        return obs
    by = {s.name: s for s in skills}
    obs["listed"] = sorted(by)
    target = by.get(sc.target)
    if target is not None:
        obs["body"] = target.instructions
        md = target.frontmatter.metadata
        obs["stamp"] = {"via": "loader", "path": None,
                        "metadata": dict(md) if isinstance(md, dict) else md}
        script = ctx.skills_dir() / sc.target / sc.script
        # ADK reads the folder in place; the executor runs the script from there.
        bit = executable(script)
        obs["exec_bit"] = {"path": str(script), "source": bit, "copy": bit}
        obs["notes"]["allowed_tools_as_parsed"] = target.frontmatter.allowed_tools

    model = ScriptedLlm(model="scripted")
    # ADK offers skills through its own list_skills tool, so the model asks first.
    model._steps = [
        call("list_skills"),
        call("load_skill", skill_name=sc.target),
        call("load_skill_resource", skill_name=sc.target, file_path=sc.resource),
        call("run_skill_script", skill_name=sc.target, file_path=sc.script, args=list(sc.script_args)),
        answer("done"),
    ]
    agent = LlmAgent(name="support", model=model, instruction="Answer refund questions.",
                     tools=[SkillToolset(skills=list(skills), code_executor=UnsafeLocalCodeExecutor())])

    async def run() -> None:
        runner = InMemoryRunner(agent=agent, app_name="native")
        sess = await runner.session_service.create_session(app_name="native", user_id="u")
        msg = types.Content(role="user", parts=[types.Part(text="refund for $100 after 45 days?")])
        async for _ in runner.run_async(user_id="u", session_id=sess.id, new_message=msg):
            pass
        await runner.close()

    attempt(obs, "run", lambda: asyncio.run(run()))
    seen = model._responses

    def response(i: int) -> dict:
        return (seen[i][0] if i < len(seen) and seen[i] else {}) or {}

    listing, loaded, res, scr = response(1), response(2), response(3), response(4)
    result = listing.get("result") if isinstance(listing, dict) else None
    listing_text = result if isinstance(result, str) else (str(listing) if listing else "")
    obs["notes"]["list_skills"] = listing_text[:2000]
    offered = re.findall(r"<name>\s*([^<\s]+)\s*</name>", listing_text)
    obs["offered"] = sorted(set(offered)) if listing else None
    obs["body_text"] = loaded.get("instructions") or (str(loaded) if loaded else None)
    obs["resource_text"] = res.get("content")
    obs["resource_exact"] = True
    obs["script_stdout"] = scr.get("stdout")
    obs["notes"]["script_result"] = {k: v for k, v in scr.items() if k != "stdout"}
    obs["user_only"] = {"offered": (sc.user_only in offered) if listing else None,
                        "refusal": None}
    return obs
