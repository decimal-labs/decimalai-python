"""openai-agents: the sandbox ``Skills`` capability over a lazy local source.

Documented snippet::

    src = LocalDirLazySkillSource(source=LocalDir(
        src=decimalai.skills_dir(relative_to=Path.cwd())))
    SandboxAgent(..., capabilities=[Shell(), Skills(lazy_from=src)],
                 default_manifest=Manifest())

Every hazard here is silent, which is why N1 (set equality) is the only detector:

* a source outside the cwd, reached through the macOS ``/tmp`` alias, or behind a
  symlink lists ZERO skills;
* with no ``Manifest()`` the skills vanish;
* ``load_skill`` of an unlisted name raises mid-run;
* lazy materialization copies a skill to ``.agents/<name>`` and drops the exec bit.

On macOS commands run under ``sandbox-exec``, where only a ``python3`` on ``PATH``
from a granted venv can start (the parent puts this venv's ``bin/`` first).
The loader exposes no metadata (its frontmatter parser is a flat string map), so
the stamp is read from the MATERIALIZED ``SKILL.md`` through ``exec_command``,
which also proves the copy kept it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
import sys
from pathlib import Path
from typing import Any, Dict

from ._observe import (
    Ctx,
    attempt,
    body_of,
    executable,
    new_observation,
    stamp_of,
    unique,
    watch_loggers,
)

NAME = "openai-agents"
CHANNEL = ("openai.agents", "agents")
KEPT, LOST = "EXEC_BIT_KEPT", "EXEC_BIT_LOST"


def _source(ctx: Ctx):
    from agents.sandbox.capabilities import LocalDirLazySkillSource
    from agents.sandbox.entries import LocalDir

    return LocalDirLazySkillSource(source=LocalDir(src=ctx.skills_dir(relative_to=Path.cwd())))


def discover(ctx: Ctx) -> Dict[str, Any]:
    channel = watch_loggers(*CHANNEL)
    out: Dict[str, Any] = {"listed": None, "raised": None, "channel": channel.lines}
    try:
        out["listed"] = sorted(m.name for m in _source(ctx).list_skill_metadata(skills_path=".agents"))
    except Exception as exc:  # noqa: BLE001
        out["raised"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    return out


def observe(ctx: Ctx) -> Dict[str, Any]:
    sc = ctx.scenario
    obs = new_observation(NAME, "openai-agents")
    channel = watch_loggers(*CHANNEL)
    obs["channel"] = channel.lines

    from agents import RunConfig, Runner, set_trace_processors
    from agents.sandbox import SandboxAgent, SandboxRunConfig
    from agents.sandbox.capabilities import Shell, Skills
    from agents.sandbox.capabilities.skills import SkillMetadata
    from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient
    from agents.testing import ScriptedModel, assistant_message, function_call

    set_trace_processors([])
    obs["proofs"]["metadata_fields"] = sorted(f.name for f in dataclasses.fields(SkillMetadata))

    src = attempt(obs, "discover", lambda: _source(ctx))
    listed = attempt(obs, "discover", lambda: src.list_skill_metadata(skills_path=".agents")) if src else None
    if listed is None:
        return obs
    obs["listed"] = sorted(m.name for m in listed)

    d = f".agents/{sc.target}"
    script = " ".join(("python3", f"{d}/{sc.script}", *sc.script_args))
    model = ScriptedModel([
        [function_call("load_skill", {"skill_name": sc.target}, call_id="c1")],
        [function_call("exec_command", {"cmd": f"cat {d}/SKILL.md"}, call_id="c2")],
        [function_call("exec_command", {"cmd": f"cat {d}/{sc.resource}"}, call_id="c3")],
        [function_call("exec_command", {"cmd": script}, call_id="c4")],
        [function_call("exec_command", {"cmd": f"test -x {d}/{sc.script} && echo {KEPT} || echo {LOST}"},
                       call_id="c5")],
        [assistant_message("done")],
    ])
    agent = SandboxAgent(name="native", instructions="Answer refund questions.", model=model,
                         capabilities=[Shell(), Skills(lazy_from=src)],
                         default_manifest=_manifest(obs))
    config = RunConfig(sandbox=SandboxRunConfig(client=UnixLocalSandboxClient()))
    attempt(obs, "run", lambda: asyncio.run(
        Runner.run(agent, "refund for $100 after 45 days?", run_config=config, max_turns=10)))

    outs: Dict[str, str] = {}
    if model.calls:
        for item in model.calls[-1].input:
            dct = item if isinstance(item, dict) else item.model_dump()
            if dct.get("type") == "function_call_output":
                outs[dct["call_id"]] = str(dct.get("output"))
    system = (model.calls[0].system_instructions or "") if model.calls else ""
    obs["notes"]["system_instructions"] = system[-2500:]
    obs["notes"]["load_skill"] = outs.get("c1")
    if system:
        obs["offered"] = unique(re.findall(r"(?m)^- ([^\s:]+): .*\(file: [^)\n]*\)$", system))
        obs["user_only"]["offered"] = sc.user_only in obs["offered"]

    def output(cid: str) -> str:
        # exec_command reports "Process exited with code N" and then the output.
        return outs.get(cid, "")

    skill_md = output("c2")
    text = skill_md[skill_md.find("---\n"):] if "---\n" in skill_md else None
    obs["body_text"] = skill_md or None
    obs["body"] = body_of(text)
    obs["stamp"] = {"via": "file", "path": f"{d}/SKILL.md", "metadata": stamp_of(text)}
    fees = output("c3")
    obs["notes"]["resource_raw"] = fees
    obs["resource_text"] = _after_header(fees)
    obs["resource_exact"] = True
    obs["script_stdout"] = output("c4") or None
    bit = output("c5")
    obs["exec_bit"] = {
        "path": f"{d}/{sc.script}",
        "source": executable(ctx.skills_dir() / sc.target / sc.script),
        "copy": True if KEPT in bit else False if LOST in bit else None,
    }
    return obs


def _manifest(obs: Dict[str, Any]):
    """``Manifest()``, plus one read-only grant this suite's isolation makes necessary.

    On macOS the sandbox lets a command read the interpreter it runs only when the
    interpreter is a system or Homebrew one, or lives under ``$HOME/.<dir>``. Each
    child here runs with a fresh, empty ``HOME``, so a uv- or pyenv-managed Python
    under the real home loses its standard library and ``libpython``. A user's own
    HOME grants it, so the documented snippet needs no grant; this one stands in
    for that rule and nothing else.
    """
    from agents.sandbox import Manifest, SandboxPathGrant

    if sys.platform != "darwin":
        return Manifest()
    base = str(Path(sys.base_prefix).resolve())
    obs["notes"]["interpreter_grant"] = base
    return Manifest(extra_path_grants=(SandboxPathGrant(path=base, read_only=True),))


def _after_header(out: str):
    """exec_command output with its status header removed, the command's bytes kept."""
    if not out:
        return None
    m = re.search(r"\nOutput:\n", out)
    return out[m.end():] if m else out
