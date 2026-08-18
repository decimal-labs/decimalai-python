"""Claude Agent SDK / Claude Code driver.

Runs the snippets documented at
``https://docs.decimal.ai/sdk/python/frameworks/claude-agent-sdk``: a real
``claude_agent_sdk.query(...)`` wrapped in ``trace_stream(...)`` — the
"Explicit form — wrap a stream you already have" section, verbatim in shape.

**The stub is the CLI, not the model.** The Claude Agent SDK has no model
object to swap: ``query()`` speaks stream-JSON over a ``Transport`` to the
``claude`` CLI subprocess, which is where the model lives. So the stub sits at
that seam — ``query(..., transport=...)`` is a first-class public parameter, and
``_StubCLI`` below answers the control protocol and replays the same
stream-JSON frames the real CLI emits. Everything above the transport is the
real SDK: the real control handshake, the real ``message_parser``, the real
message classes. No CLI binary, no ``ANTHROPIC_API_KEY``, no network — the
hermetic tier runs this on every commit.

**Why the explicit form and not ``instrument()``.** The global form is a
one-line monkeypatch whose body is ``trace_stream(stream,
agent_name=_install_agent_name, ...)`` — the same function this driver calls, so
the tracing path under test is identical. What differs is that
``_install_agent_name`` is a single module global: under the concurrency phase
all eight lanes would report whichever name was assigned last, which is a
property of a process-wide switch and not a defect the contract should be
told about. The docs give ``trace_stream(agent_name=...)`` as the per-run
answer, so that is the form driven here. The 12-line patch wrapper is the only
adapter code this leaves uncovered — and driving it would also leave
``claude_agent_sdk.query`` permanently patched for every driver that runs after
this one.

NO ASSERTIONS BELOW THIS LINE. That is the driver contract.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from . import (
    STUB_MODEL_NAME,
    SYSTEM_PROMPT,
    Capabilities,
    Ctx,
    Driver,
    DriverError,
    stub_script,
    tool_result,
    user_message,
)

# ── the stub CLI ─────────────────────────────────────────────────────────────


def _stub_transport(frames: Sequence[Dict[str, Any]], *, fail_after: Optional[int] = None) -> Any:
    """A ``Transport`` that replays ``frames`` instead of spawning the CLI.

    Implements the same contract ``SubprocessCLITransport`` does: answer the
    control protocol on ``write()``, hand parsed JSON frames back from
    ``read_messages()``. ``fail_after`` raises after N frames, which is how a
    run fails the way a crashed CLI fails — mid-stream, not at connect.
    """
    import json

    from claude_agent_sdk._internal.transport import Transport

    class _StubCLI(Transport):
        def __init__(self) -> None:
            self._queue: List[Dict[str, Any]] = []
            self._wake = asyncio.Event()
            self._ready = False
            self._prompt_seen = False
            self._sent = 0

        async def connect(self) -> None:
            self._ready = True

        async def write(self, data: str) -> None:
            for line in data.splitlines():
                if not line.strip():
                    continue
                message = json.loads(line)
                if message.get("type") == "control_request":
                    # The SDK blocks on `initialize` for 60s if nobody answers.
                    self._queue.append({
                        "type": "control_response",
                        "response": {
                            "subtype": "success",
                            "request_id": message["request_id"],
                            "response": {"commands": [], "output_style": "default"},
                        },
                    })
                elif message.get("type") == "user":
                    # The prompt landed — the "model" may now answer.
                    self._queue.extend(frames)
                    self._prompt_seen = True
                self._wake.set()

        async def read_messages(self) -> Any:
            while True:
                if not self._queue:
                    if self._prompt_seen:
                        return
                    await self._wake.wait()
                    self._wake.clear()
                    continue
                frame = self._queue.pop(0)
                if fail_after is not None and self._sent >= fail_after:
                    raise DriverError("conformance: the CLI died mid-stream on purpose")
                self._sent += 1
                yield frame

        async def close(self) -> None:
            self._ready = False
            self._prompt_seen = True
            self._wake.set()

        def is_ready(self) -> bool:
            return self._ready

        async def end_input(self) -> None:
            self._prompt_seen = True
            self._wake.set()

    return _StubCLI()


def _frames(ctx: Ctx, *, use_tool: bool = True, declare: bool = True) -> List[Dict[str, Any]]:
    """The shared stub script as stream-JSON, in the CLI's own wire shape.

    ``declare=False`` is the degenerate run: an init frame that announces no
    model and no tools, so ``ClaudeAgentOptions()`` plus this leaves the adapter
    nothing to build a manifest from.
    """
    session = "conformance-" + uuid4().hex[:12]
    init: Dict[str, Any] = {"type": "system", "subtype": "init", "session_id": session}
    if declare:
        init["model"] = STUB_MODEL_NAME
        init["tools"] = [ctx.tool_name]
    frames: List[Dict[str, Any]] = [init]

    total_in = total_out = 0
    for index, turn in enumerate(stub_script(ctx, use_tool=use_tool)):
        blocks: List[Dict[str, Any]] = []
        tool_use_id = f"toolu_conformance_{index}"
        if turn.tool_call:
            name, args = turn.tool_call
            blocks.append({"type": "tool_use", "id": tool_use_id, "name": name, "input": args})
        if turn.content:
            blocks.append({"type": "text", "text": turn.content})
        frames.append({
            "type": "assistant",
            "session_id": session,
            "message": {
                "id": f"msg_conformance_{index}",
                "type": "message",
                "role": "assistant",
                "model": STUB_MODEL_NAME,
                "stop_reason": "tool_use" if turn.tool_call else "end_turn",
                "content": blocks,
                # The real CLI reports per-turn usage on every assistant frame.
                "usage": {
                    "input_tokens": turn.input_tokens,
                    "output_tokens": turn.output_tokens,
                },
            },
        })
        if turn.tool_call:
            _, args = turn.tool_call
            frames.append({
                "type": "user",
                "session_id": session,
                "message": {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": tool_result(ctx, str(args.get("query", ""))),
                    }],
                },
            })
        total_in += turn.input_tokens
        total_out += turn.output_tokens

    frames.append({
        "type": "result",
        "subtype": "success",
        "session_id": session,
        "duration_ms": 12,
        "duration_api_ms": 9,
        "is_error": False,
        "num_turns": len(frames),
        "result": ctx.reply_sentinel,
        "total_cost_usd": 0.00042,
        "usage": {"input_tokens": total_in, "output_tokens": total_out},
    })
    return frames


def _options(ctx: Ctx, *, declare: bool = True) -> Any:
    from claude_agent_sdk import ClaudeAgentOptions

    if not declare:
        # Nothing to declare: no model, no system prompt, no tools, no subagents.
        return ClaudeAgentOptions()
    return ClaudeAgentOptions(
        model=STUB_MODEL_NAME,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=[ctx.tool_name],
    )


# ── the documented snippet ───────────────────────────────────────────────────


async def _drain(
    ctx: Ctx, *, use_tool: bool = True, declare: bool = True, fail_after: Optional[int] = None
) -> List[Any]:
    import claude_agent_sdk

    from decimalai.claude_agent_sdk import trace_stream

    options = _options(ctx, declare=declare)
    prompt = user_message(ctx)
    stream = claude_agent_sdk.query(
        prompt=prompt,
        options=options,
        transport=_stub_transport(
            _frames(ctx, use_tool=use_tool, declare=declare), fail_after=fail_after
        ),
    )
    seen: List[Any] = []
    async for message in trace_stream(
        stream, agent_name=ctx.agent_name, user_input=prompt, options=options
    ):
        seen.append(message)
    return seen


def run(ctx: Ctx) -> Any:
    """One traced ``query()`` run: a tool turn, its result, then the answer."""
    return asyncio.run(_drain(ctx))


def run_error(ctx: Ctx) -> Any:
    """The same run with the CLI dying after the first model turn."""
    return asyncio.run(_drain(ctx, fail_after=2))


def run_degenerate(ctx: Ctx) -> Any:
    """A run with nothing to declare: bare options, no model/tools in init."""
    return asyncio.run(_drain(ctx, use_tool=False, declare=False))


def run_concurrent(ctxs: Sequence[Ctx]) -> Any:
    """N lanes at once the way this SDK does concurrency — one event loop, gathered."""

    async def _all() -> List[Any]:
        return list(await asyncio.gather(*(_drain(c) for c in ctxs)))

    return asyncio.run(_all())


DRIVER = Driver(
    name="claude-agent-sdk",
    covers=frozenset({"claude-agent-sdk", "claude-code"}),
    requires=("claude_agent_sdk",),
    entrypoint="decimalai.claude_agent_sdk.trace_stream()",
    run=run,
    run_concurrent=run_concurrent,
    run_error=run_error,
    run_degenerate=run_degenerate,
    capabilities=Capabilities(
        has_tools=True,
        has_skills_rail=False,
        supports_concurrency=True,
        supports_error_path=True,
        supports_degenerate=True,
        reasons={
            "has_skills_rail": (
                "Claude Code loads skills from disk (.claude/skills/), not from a hosted "
                "menu, and the docs say so in as many words: 'There is no hosted-routing "
                "path on this integration… router-side effectiveness data isn't collected "
                "for skills used this way.' A disk install produces no routing decision at "
                "prompt-assembly time, so there is no routing_id and no offered set for a "
                "trace to carry — the rail C8 grades does not exist here. The install side "
                "(SkillRouter.install(agents=['claude-code']) writing SKILL.md) is a "
                "filesystem operation this adapter never sees. It silences the "
                "activation items (C13/C13b) for the same reason the docs give: a disk "
                "skill produces no offered set, so nothing here could be joined to a "
                "routing decision, and the adapter maps no Skill tool use onto an "
                "activation today — recording one from prompt presence would be a "
                "fabrication."
            ),
        },
    ),
)
