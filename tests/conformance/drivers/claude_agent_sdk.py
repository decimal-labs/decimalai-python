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

from ..delivery import INJECTED
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
            self._mcp_waiting = False
            self._after_mcp: List[Dict[str, Any]] = []
            self._mcp_marker: Dict[str, Any] = {}

        def _mcp(self, stage: str, payload: Dict[str, Any]) -> None:
            self._queue.append({
                "type": "control_request", "request_id": f"conformance_mcp_{stage}",
                "request": {"subtype": "mcp_message", "server_name": "decimalai",
                            "message": payload},
            })

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
                elif message.get("type") == "control_response":
                    response = message.get("response") or {}
                    request_id = response.get("request_id")
                    if request_id == "conformance_mcp_init":
                        self._mcp("ready", {"jsonrpc": "2.0", "method": "notifications/initialized"})
                    elif request_id == "conformance_mcp_ready":
                        self._mcp("call", {
                            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                            "params": {"name": "load_skill", "arguments": {"name": self._mcp_marker["_skill"]}},
                        })
                    elif request_id == "conformance_mcp_call":
                        result = (response.get("response") or {}).get("mcp_response") or {}
                        self._queue.append({
                            "type": "user", "session_id": self._mcp_marker["_session"],
                            "message": {"role": "user", "content": [{
                                "type": "tool_result", "tool_use_id": self._mcp_marker["_tool_id"],
                                "content": (result.get("result") or {}).get("content") or str(result),
                            }]},
                        })
                        self._queue.extend(self._after_mcp)
                        self._mcp_waiting = False
                self._wake.set()

        async def read_messages(self) -> Any:
            while True:
                if not self._queue:
                    if self._prompt_seen and not self._mcp_waiting:
                        return
                    await self._wake.wait()
                    self._wake.clear()
                    continue
                frame = self._queue.pop(0)
                if "_skill" in frame:
                    self._mcp_marker = frame
                    self._mcp_waiting = True
                    self._after_mcp, self._queue = self._queue, []
                    self._mcp("init", {
                        "jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                                   "clientInfo": {"name": "conformance-cli", "version": "1"}},
                    })
                    continue
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


def _frames(ctx: Ctx, *, use_tool: bool = True, declare: bool = True,
            skill: Optional[str] = None, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """The shared stub script as stream-JSON, in the CLI's own wire shape.

    ``declare=False`` is the degenerate run: an init frame that announces no
    model and no tools, so ``ClaudeAgentOptions()`` plus this leaves the adapter
    nothing to build a manifest from.
    """
    session = session_id or "conformance-" + uuid4().hex[:12]
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
            if skill is not None:
                name, args = "mcp__decimalai__load_skill", {"name": skill}
            blocks.append({"type": "tool_use", "id": tool_use_id, "name": name, "input": args})
        if turn.content:
            blocks.append({"type": "text", "text": turn.content})
        # The real CLI streams ONE request as one assistant frame PER CONTENT BLOCK,
        # every frame carrying the request's id and the same usage, whose output
        # count is where the stream started, not what it produced; the result frame
        # carries the run's real totals (measured: claude-agent-sdk 0.2.166, CLI
        # 2.1.299). Replay that shape, not one tidy frame per turn.
        for block in blocks:
            frames.append({
                "type": "assistant",
                "session_id": session,
                "message": {
                    "id": f"msg_conformance_{index}",
                    "type": "message",
                    "role": "assistant",
                    "model": STUB_MODEL_NAME,
                    "stop_reason": None,
                    "content": [block],
                    "usage": {"input_tokens": turn.input_tokens, "output_tokens": 1},
                },
            })
        if turn.tool_call:
            if skill is not None:
                frames.append({"_skill": skill, "_tool_id": tool_use_id, "_session": session})
                total_in += turn.input_tokens
                total_out += turn.output_tokens
                continue
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


async def _configured(ctx: Ctx, support: Any) -> List[Any]:
    """The config tutorial, over the real SDK/control protocol and MCP server."""
    skill = ctx.skills[0]["name"] if ctx.skills and ctx.delivery_mode != INJECTED else None
    async with support.run(user_message(ctx)) as run:
        return [message async for message in run.query(
            options=_options(ctx), transport=_stub_transport(_frames(
                ctx, skill=skill, session_id=getattr(support, "_session_id", None))),
        )]


def run_skills(ctxs: Sequence[Ctx]) -> Any:
    from decimalai.claude_agent_sdk import agent_config

    configs: Dict[str, Any] = {}
    pairs = []
    for ctx in ctxs:
        if ctx.agent_name not in configs:
            configs[ctx.agent_name] = agent_config(ctx.agent_name)
        pairs.append((ctx, configs[ctx.agent_name]))

    async def _all() -> List[Any]:
        return list(await asyncio.gather(*(_configured(ctx, support) for ctx, support in pairs)))
    return asyncio.run(_all())


def run_unnamed(ctx: Ctx) -> Any:
    from decimalai.claude_agent_sdk import agent_config
    return asyncio.run(_configured(ctx, agent_config(ctx.agent_name)))


def run_conversation(ctxs: Sequence[Ctx]) -> Any:
    """The same config conversation, keeping the first prompt and native session."""
    from decimalai.claude_agent_sdk import agent_config

    async def _run() -> List[Any]:
        chat = agent_config(ctxs[0].agent_name).conversation()
        result = []
        for ctx in ctxs:
            result.append(await _configured(ctx, chat))
        return result
    return asyncio.run(_run())


DRIVER = Driver(
    name="claude-agent-sdk",
    covers=frozenset({"claude-agent-sdk", "claude-code"}),
    requires=("claude_agent_sdk",),
    frameworks=("claude-agent-sdk",),
    entrypoint="decimalai.claude_agent_sdk.trace_stream() / agent_config(name).run(message).query()",
    run=run,
    run_concurrent=run_concurrent,
    run_error=run_error,
    run_degenerate=run_degenerate,
    run_skills=run_skills,
    run_unnamed=run_unnamed,
    run_conversation=run_conversation,
    loads_agent_from_platform=True,
    capabilities=Capabilities(
        has_tools=True,
        has_skills_rail=True,
        supports_concurrency=True,
        supports_error_path=True,
        supports_degenerate=True,
        has_invocation_observer=False,
        reasons={
            "has_invocation_observer": (
                "decimalai/claude_agent_sdk.py hands its caller no run: instrument() "
                "takes no on_trace, so C15 has no observer to grade, and a run is "
                "linked to a parent only through the explicit parent_trace_id= of "
                "trace_stream() / traced_query(), never from an enclosing "
                "decimalai.start_trace(). C16 grades that link as the second half of "
                "the same hand-over, so it is N/A with C15. Not a framework limit and "
                "not this suite's opinion: test_coverage reads this flag off "
                "instrument()'s own signature, so the commit that gives it on_trace "
                "turns both items on."
            ),
        },
    ),
)
