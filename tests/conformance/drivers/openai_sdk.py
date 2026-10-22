"""Raw OpenAI SDK driver — the no-framework rail for the most common agent loop.

Runs the two snippets the docs give for code that calls the ``openai`` SDK
directly, with no agent framework in between:

* tracing — ``init(openai=True)`` (here: its body,
  ``decimalai.providers.instrument(openai=True)``), which turns on the
  OpenInference OpenAI instrumentor and routes its OTel spans through
  ``decimalai.otel.DecimalSpanExporter``, with ``providers.agent_run(name)``
  around an ordinary Chat Completions tool loop as the run boundary. The phases
  up to ``concurrent`` run this and nothing else of DecimalAI;
* an agent's prompt and skills — ``decimalai.openai.agent_config(name)``, one run
  per user message, around the hand-written Chat Completions loop the module
  documents::

      support = agent_config(name)
      with support.run(question) as run:
          messages = [{"role": "system", "content": run.system}, run.message]
          ...client.chat.completions.create(model=..., messages=messages,
                                            tools=[lookup, *run.tools])...
          # each tool call: run.tool_result(call) or the caller's own answer

  ``nested`` and every phase after it run this. ``run.system`` is the saved
  prompt, then the skill list routed on the run's message; ``run.tools`` is
  ``load_skill``, answered by ``run.tool_result(call)``, so a body comes back as
  a ``tool`` message mid-run. Opening the run opens its trace, filed under the
  DecimalAI agent, and ``agent_config()`` turns on the same tracing as
  ``init(openai=True)``. Nothing in the SDK is patched, so every rail phase builds
  its run that way (``Driver.loads_agent_from_platform``: the harness puts each
  phase's agent in the workspace first, as the dashboard would).

The second shapes are the module's other documented loops: the same run on the
Responses API (``instructions=run.system``, ``tools=[*mine, *run.response_tools]``,
``previous_response_id``), and ``async with`` around an ``AsyncOpenAI`` loop that
answers with ``await run.atool_result(call)``.

**Where the stub sits.** The shared ``_openai_wire.OpenAIWire``: a real HTTP
server on 127.0.0.1 answering ``POST /v1/chat/completions`` and
``POST /v1/responses`` with the shared ``stub_script``, and ``base_url=`` points
the real client at it. Every line of the real ``openai`` SDK runs, and the
OpenInference instrumentor sees exactly what it would see in production.

**Instrumented once, on purpose.** The OpenInference instrumentor binds its
tracer the first time it is enabled, so the exporter's default ``agent_name`` is
fixed for the process at that moment. Each run is still named where the docs
name it: ``agent_run(name)`` on the tracing half, ``agent_config(name)`` on the
rail.

NO ASSERTIONS BELOW THIS LINE. That is the driver contract.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Dict, List, Optional, Sequence

from ..delivery import INJECTED
from . import (
    SYSTEM_PROMPT,
    Capabilities,
    Ctx,
    Driver,
    Shape,
    StubTurn,
    fanout_threads,
    stub_script,
    tool_result,
    trace_observer,
    user_message,
)
from ._openai_wire import STUB_API_KEY, OpenAIWire

#: The model id every request names. OpenAI-shaped, but not a model that exists.
STUB_MODEL = "gpt-conformance-stub"

#: A hand-written loop's bound on model calls. The stub answers on the second,
#: so this only stops a script defect from spinning forever.
_MAX_MODEL_CALLS = 4

_WIRE: Optional[OpenAIWire] = None
_WIRE_LOCK = threading.Lock()


def _wire() -> OpenAIWire:
    """The process's stub OpenAI service, started on first use."""
    global _WIRE
    with _WIRE_LOCK:
        if _WIRE is None:
            _WIRE = OpenAIWire().start()
        return _WIRE


def _client() -> Any:
    from openai import OpenAI

    # Retries off: a retried 400 would turn the error phase's one failure into three.
    return OpenAI(base_url=_wire().base_url, api_key=STUB_API_KEY, max_retries=0)


def _tool(ctx: Ctx) -> Dict[str, Any]:
    """The conformance tool, as a Chat Completions tool definition."""
    return {"type": "function", "function": {
        "name": ctx.tool_name,
        "description": "Look a value up for the conformance run.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }}


def _response_tool(ctx: Ctx) -> Dict[str, Any]:
    """The same tool for the Responses API: the function's fields at the top level."""
    return {"type": "function", **_tool(ctx)["function"], "strict": False}


def _skill_script(ctx: Ctx) -> List[StubTurn]:
    """The rail's script: call ``load_skill``, then answer.

    In the ``injected`` delivery cell the ``load_skill`` tool is switched OFF in
    this process's environment, so the model is scripted onto the ordinary
    conformance tool instead, and the body has to arrive in ``run.system``.
    """
    turns = stub_script(ctx)
    if not ctx.skills or ctx.delivery_mode == INJECTED:
        return turns
    ask, answer = turns[0], turns[-1]
    return [
        StubTurn(
            tool_call=("load_skill", {"name": ctx.skills[0]["name"]}),
            content=ask.content,
            input_tokens=ask.input_tokens,
            output_tokens=ask.output_tokens,
        ),
        answer,
    ]


def _query(arguments: Any) -> str:
    try:
        parsed = json.loads(arguments or "{}")
    except ValueError:
        return ""
    return str(parsed.get("query", "")) if isinstance(parsed, dict) else ""


# ── the tracing snippet: init(openai=True) ───────────────────────────────────

_pipeline_lock = threading.Lock()
_instrumented = False


def _instrument(ctx: Ctx) -> None:
    """``init(openai=True)``: its body, on the process-global provider. Once."""
    global _instrumented
    with _pipeline_lock:
        if _instrumented:
            return
        from decimalai import providers

        providers.instrument(openai=True, agent_name=ctx.agent_name)
        _instrumented = True


def _loop(ctx: Ctx, client: Any, *, tools: bool = True, system: Optional[str] = SYSTEM_PROMPT) -> Any:
    """The documented Chat Completions tool loop, verbatim in shape."""
    kwargs: Dict[str, Any] = {"model": STUB_MODEL}
    if tools:
        kwargs["tools"] = [_tool(ctx)]
    messages: List[Any] = [{"role": "system", "content": system}] if system is not None else []
    messages.append({"role": "user", "content": user_message(ctx)})
    first = client.chat.completions.create(messages=messages, **kwargs)
    reply = first.choices[0].message
    if not reply.tool_calls:
        return first
    messages.append(reply)
    for call in reply.tool_calls:
        messages.append({"role": "tool", "tool_call_id": call.id,
                         "content": tool_result(ctx, _query(call.function.arguments))})
    return client.chat.completions.create(messages=messages, **kwargs)


def _traced(ctx: Ctx, *, fail: bool = False, tools: bool = True,
            system: Optional[str] = SYSTEM_PROMPT) -> Any:
    _instrument(ctx)
    _wire().register(ctx, script=None if tools else stub_script(ctx, use_tool=False), fail=fail)
    client = _client()
    try:
        from decimalai import providers

        # The documented run boundary: without it a two-call tool loop arrives as
        # two unrelated single-span traces. (agent_config()'s run declares it.)
        with providers.agent_run(ctx.agent_name):
            return _loop(ctx, client, tools=tools, system=system)
    finally:
        client.close()


def run(ctx: Ctx) -> Any:
    """One traced run: the tool-use loop against the stub service."""
    return _traced(ctx)


def run_error(ctx: Ctx) -> Any:
    """The same run with the provider refusing the request."""
    return _traced(ctx, fail=True)


def run_degenerate(ctx: Ctx) -> Any:
    """A bare ``chat.completions.create`` — no tools, no system prompt."""
    return _traced(ctx, tools=False, system=None)


# ── the rail snippet: agent_config() ───────────────────────────────────────────


def _config(ctx: Ctx, configs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    """This agent's ``agent_config()`` config. ``configs`` shares one per agent
    name, as a module-level one would be."""
    from decimalai.openai import agent_config

    configs = {} if configs is None else configs
    if ctx.agent_name not in configs:
        configs[ctx.agent_name] = agent_config(ctx.agent_name, **kwargs)
    return configs[ctx.agent_name]


def _chat_loop(ctx: Ctx, client: Any, run: Any, history: Optional[List[Any]] = None) -> Any:
    """The module's documented loop: ``tools=[*mine, *run.tools]``, and
    ``run.tool_result(call)`` for each call — the answer for ``load_skill``,
    ``None`` for the caller's own tool."""
    history = [] if history is None else history
    history.append(run.message)
    response = None
    for _ in range(_MAX_MODEL_CALLS):
        response = client.chat.completions.create(
            model=STUB_MODEL, tools=[_tool(ctx), *run.tools],
            messages=[{"role": "system", "content": run.system}, *history])
        reply = response.choices[0].message
        history.append(reply)
        if not reply.tool_calls:
            break
        for call in reply.tool_calls:
            history.append(run.tool_result(call) or {
                "role": "tool", "tool_call_id": call.id,
                "content": tool_result(ctx, _query(call.function.arguments))})
    return response


def _one_run(ctx: Ctx, support: Any, *, script: Optional[List[StubTurn]] = None) -> Any:
    """One user message: ``with support.run(question) as run:`` around the loop."""
    _wire().register(ctx, script=script)
    client = _client()
    try:
        with support.run(user_message(ctx)) as run:
            return _chat_loop(ctx, client, run)
    finally:
        client.close()


def run_nested(ctx: Ctx) -> Any:
    """One run with the suite's observer as ``agent_config(on_trace=...)``.

    The harness runs this inside a ``decimalai.start_trace()`` of its own; the
    run reads the enclosing trace when it opens, which is where its parent link
    is recorded.
    """
    return _one_run(ctx, _config(ctx, on_trace=trace_observer(ctx)))


def run_skills(ctxs: Sequence[Ctx]) -> Any:
    """The skills rail, N lanes at once, sharing one config per agent name."""
    configs: Dict[str, Any] = {}
    pairs = [(ctx, _config(ctx, configs)) for ctx in ctxs]

    def _go(pair: Any) -> Any:
        ctx, support = pair
        return _one_run(ctx, support, script=_skill_script(ctx))

    return fanout_threads(_go)(pairs)


def run_unnamed(ctx: Ctx) -> Any:
    """The rail where the only name the run has is the DecimalAI one.

    The Chat Completions API has no agent object, and the tracing phases fixed
    the exporter's default name to another agent. Nothing here names this run
    but ``agent_config(name)``, so the trace and the routing call must both
    carry that name, not the process's.
    """
    return _one_run(ctx, _config(ctx), script=_skill_script(ctx))


def run_conversation(ctxs: Sequence[Ctx]) -> Any:
    """Two turns of one ``support.conversation()``: the second turn is sent the
    first turn's whole exchange ahead of its own ``run.message``."""
    first, second = ctxs
    chat = _config(first).conversation()
    for ctx in ctxs:
        _wire().register(ctx, script=_skill_script(ctx))
    history: List[Any] = []
    client = _client()
    try:
        with chat.run(user_message(first)) as run:
            _chat_loop(first, client, run, history)
        with chat.run(user_message(second)) as run:
            return _chat_loop(second, client, run, history)
    finally:
        client.close()


# ── second documented shapes ─────────────────────────────────────────────────


def _responses_shape(ctxs: Sequence[Ctx]) -> Any:
    """The Responses API: ``instructions=run.system``, ``tools=[*mine,
    *run.response_tools]``, and ``run.tool_result(call)`` for each ``function_call``
    item, with the conversation's items sent back in ``input`` (the stub keeps no
    state, so ``previous_response_id``, the other documented form, has nothing to
    point at here; the unit tests send that one)."""
    out = []
    for ctx in ctxs:
        support = _config(ctx)
        _wire().register(ctx, script=_skill_script(ctx))
        client = _client()
        try:
            with support.run(user_message(ctx)) as run:
                tools = [_response_tool(ctx), *run.response_tools]
                items: List[Any] = [run.message]
                response = None
                for _ in range(_MAX_MODEL_CALLS):
                    response = client.responses.create(model=STUB_MODEL, instructions=run.system,
                                                       input=items, tools=tools)
                    calls = [item for item in response.output if item.type == "function_call"]
                    if not calls:
                        break
                    items += [item.model_dump(exclude_none=True) for item in response.output]
                    items += [run.tool_result(call) or {
                        "type": "function_call_output", "call_id": call.call_id,
                        "output": tool_result(ctx, _query(call.arguments))} for call in calls]
                out.append(response)
        finally:
            client.close()
    return out


async def _async_run(ctx: Ctx, support: Any) -> Any:
    from openai import AsyncOpenAI

    _wire().register(ctx, script=_skill_script(ctx))
    client = AsyncOpenAI(base_url=_wire().base_url, api_key=STUB_API_KEY, max_retries=0)
    try:
        async with support.run(user_message(ctx)) as run:
            messages: List[Any] = [{"role": "system", "content": run.system}, run.message]
            response = None
            for _ in range(_MAX_MODEL_CALLS):
                response = await client.chat.completions.create(
                    model=STUB_MODEL, messages=messages, tools=[_tool(ctx), *run.tools])
                reply = response.choices[0].message
                messages.append(reply)
                if not reply.tool_calls:
                    break
                for call in reply.tool_calls:
                    messages.append(await run.atool_result(call) or {
                        "role": "tool", "tool_call_id": call.id,
                        "content": tool_result(ctx, _query(call.function.arguments))})
            return response
    finally:
        await client.close()


def _async_shape(ctxs: Sequence[Ctx]) -> Any:
    """``async with support.run(...)`` around an ``AsyncOpenAI`` Chat Completions loop."""
    return [asyncio.run(_async_run(ctx, _config(ctx))) for ctx in ctxs]


DRIVER = Driver(
    name="openai",
    # The capability table's "OpenAI SDK (with `agent_config(...)`)" row
    # (decimalai/_frameworks.py). The tracing half, init(openai=True), is advertised
    # in prose and recorded in test_coverage.PROVIDER_FLAG_DRIVERS.
    covers=frozenset({"openai-sdk"}),
    requires=("openai", "openinference.instrumentation.openai"),
    # Gated on the instrumentor, not on `openai`: openai's floor is the openai 3
    # environment's, and this driver runs on openai 2 as well.
    frameworks=("openinference-instrumentation-openai",),
    entrypoint=(
        "decimalai.providers.instrument(openai=True) / "
        "decimalai.openai.agent_config(name).run(message)"
    ),
    run=run,
    run_concurrent=fanout_threads(run),
    run_error=run_error,
    run_degenerate=run_degenerate,
    run_skills=run_skills,
    run_nested=run_nested,
    run_unnamed=run_unnamed,
    run_conversation=run_conversation,
    loads_agent_from_platform=True,
    shapes=(
        Shape(
            name="responses_api_loop",
            doc=(
                "agent_config(name).run(message) on the Responses API: instructions=run.system, "
                "tools=[*mine, *run.response_tools], the items sent back in input, and "
                "run.tool_result(call) answering load_skill's function_call"
            ),
            run=_responses_shape,
            routed=True,
        ),
        Shape(
            name="async_chat_loop",
            doc=(
                "async with agent_config(name).run(message) around an AsyncOpenAI "
                "chat.completions loop answering load_skill with await run.atool_result(call)"
            ),
            run=_async_shape,
            routed=True,
        ),
    ),
    capabilities=Capabilities(
        has_tools=True,
        has_skills_rail=True,
        supports_concurrency=True,
        supports_error_path=True,
        supports_degenerate=True,
    ),
)
