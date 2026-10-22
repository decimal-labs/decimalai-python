"""Gemini API driver (``google-genai``) — the no-framework rail for Gemini.

Runs the two snippets the docs give for code that calls the ``google-genai`` SDK directly:

* tracing — ``init(google=True)`` (here: its body,
  ``decimalai.providers.instrument(google=True)``), the OpenInference google-genai
  instrumentor routed through ``decimalai.otel.DecimalSpanExporter``, with
  ``providers.agent_run(name)`` around a hand-written function-calling loop as the run
  boundary. The phases up to ``concurrent`` run this and nothing else of DecimalAI;
* an agent's prompt and skills — ``decimalai.google_genai.agent_config(name)``, one run
  per user message, in the documented shape: a chat with automatic function calling::

      support = agent_config(name)
      with support.run(question) as run:
          chat = client.chats.create(model=..., config=types.GenerateContentConfig(
              system_instruction=run.system, tools=[lookup, *run.tools]))
          chat.send_message(run.message)

  ``nested`` and every phase after it run this. ``run.tools`` is ``load_skill`` as a
  Python function, which the SDK calls in its own loop, so a body comes back as a
  ``functionResponse`` mid-run. Opening the run opens its trace, filed under the
  DecimalAI agent, and ``agent_config()`` turns on the same tracing as
  ``init(google=True)`` (``Driver.loads_agent_from_platform``).

The second shapes are the module's other documented forms: a hand-written
``generate_content`` loop with ``run.tool_params`` and ``run.tool_result(call)``, and
``async with`` around ``client.aio``'s chat.

**Where the stub sits.** ``_gemini_wire.GeminiWire``: a real HTTP server on 127.0.0.1
answering ``generateContent``, reached through ``HttpOptions(base_url=...)``. Every line of
the real SDK and the instrumentor runs; no key is used and no Google request is made.

NO ASSERTIONS BELOW THIS LINE. That is the driver contract.
"""

from __future__ import annotations

import asyncio
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
from ._gemini_wire import GeminiWire

#: The model every request names: Gemini-shaped, not a model that exists.
STUB_MODEL = "gemini-conformance-stub"

#: A hand-written loop's bound on model calls. The stub answers on the second.
_MAX_MODEL_CALLS = 4

_WIRE: Optional[GeminiWire] = None
_WIRE_LOCK = threading.Lock()


def _wire() -> GeminiWire:
    global _WIRE
    with _WIRE_LOCK:
        if _WIRE is None:
            _WIRE = GeminiWire().start()
        return _WIRE


def _declared(ctx: Ctx) -> Dict[str, Any]:
    """The conformance tool as a ``types.Tool``, for a hand-written loop."""
    return {"function_declarations": [{
        "name": ctx.tool_name,
        "description": "Look a value up for the conformance run.",
        "parameters_json_schema": {"type": "object", "properties": {"query": {"type": "string"}},
                                   "required": ["query"]},
    }]}


def _function(ctx: Ctx, *, asynchronous: bool = False) -> Any:
    """The same tool as a Python function, for automatic function calling."""
    if asynchronous:
        async def lookup(query: str) -> str:
            return tool_result(ctx, query)
    else:
        def lookup(query: str) -> str:  # type: ignore[misc]
            return tool_result(ctx, query)
    lookup.__name__ = lookup.__qualname__ = ctx.tool_name
    lookup.__doc__ = "Look a value up for the conformance run."
    return lookup


def _skill_script(ctx: Ctx) -> List[StubTurn]:
    """The rail's script: call ``load_skill``, then answer.

    In the ``injected`` delivery cell the ``load_skill`` tool is switched OFF in this
    process's environment, so the model is scripted onto the ordinary conformance tool.
    """
    turns = stub_script(ctx)
    if not ctx.skills or ctx.delivery_mode == INJECTED:
        return turns
    ask, answer = turns[0], turns[-1]
    return [
        StubTurn(tool_call=("load_skill", {"name": ctx.skills[0]["name"]}), content=ask.content,
                 input_tokens=ask.input_tokens, output_tokens=ask.output_tokens),
        answer,
    ]


def _own_answer(ctx: Ctx, call: Any) -> Dict[str, Any]:
    """The function_response part for the conformance tool's call."""
    query = str((call.args or {}).get("query", ""))
    response: Dict[str, Any] = {"name": call.name, "response": {"result": tool_result(ctx, query)}}
    if call.id:
        response["id"] = call.id
    return {"function_response": response}


# ── the tracing snippet: init(google=True) ───────────────────────────────────

_pipeline_lock = threading.Lock()
_instrumented = False


def _instrument(ctx: Ctx) -> None:
    """``init(google=True)``: its body, on the process-global provider. Once."""
    global _instrumented
    with _pipeline_lock:
        if _instrumented:
            return
        from decimalai import providers

        providers.instrument(google=True, agent_name=ctx.agent_name)
        _instrumented = True


def _loop(ctx: Ctx, client: Any, *, tools: bool = True, system: Optional[str] = SYSTEM_PROMPT) -> Any:
    """The documented hand-written function-calling loop."""
    from google.genai import types

    config = types.GenerateContentConfig(system_instruction=system,
                                         tools=[_declared(ctx)] if tools else None)
    contents: List[Any] = [{"role": "user", "parts": [{"text": user_message(ctx)}]}]
    first = client.models.generate_content(model=STUB_MODEL, contents=contents, config=config)
    if not first.function_calls:
        return first
    contents.append(first.candidates[0].content)
    contents.append({"role": "user", "parts": [_own_answer(ctx, call) for call in first.function_calls]})
    return client.models.generate_content(model=STUB_MODEL, contents=contents, config=config)


def _traced(ctx: Ctx, *, fail: bool = False, tools: bool = True,
            system: Optional[str] = SYSTEM_PROMPT) -> Any:
    _instrument(ctx)
    _wire().register(ctx, script=None if tools else stub_script(ctx, use_tool=False), fail=fail)
    from decimalai import providers

    # The documented run boundary: without it the loop's two calls arrive as two traces.
    with providers.agent_run(ctx.agent_name):
        return _loop(ctx, _wire().client(), tools=tools, system=system)


def run(ctx: Ctx) -> Any:
    """One traced run: the function-calling loop against the stub service."""
    return _traced(ctx)


def run_error(ctx: Ctx) -> Any:
    """The same run with the provider refusing the request."""
    return _traced(ctx, fail=True)


def run_degenerate(ctx: Ctx) -> Any:
    """A bare ``generate_content`` — no tools, no system instruction."""
    return _traced(ctx, tools=False, system=None)


# ── the rail snippet: agent_config() ───────────────────────────────────────────


def _config(ctx: Ctx, configs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    from decimalai.google_genai import agent_config

    configs = {} if configs is None else configs
    if ctx.agent_name not in configs:
        configs[ctx.agent_name] = agent_config(ctx.agent_name, **kwargs)
    return configs[ctx.agent_name]


def _chat(ctx: Ctx, client: Any, run: Any) -> Any:
    """The documented snippet: a chat with automatic function calling."""
    from google.genai import types

    chat = client.chats.create(model=STUB_MODEL, config=types.GenerateContentConfig(
        system_instruction=run.system, tools=[_function(ctx), *run.tools]))
    return chat.send_message(run.message)


def _one_run(ctx: Ctx, support: Any, *, script: Optional[List[StubTurn]] = None) -> Any:
    _wire().register(ctx, script=script)
    with support.run(user_message(ctx)) as run:
        return _chat(ctx, _wire().client(), run)


def run_nested(ctx: Ctx) -> Any:
    """One run with the suite's observer as ``agent_config(on_trace=...)``."""
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
    """The rail where the only name the run has is the DecimalAI one."""
    return _one_run(ctx, _config(ctx), script=_skill_script(ctx))


def _hand_loop(ctx: Ctx, client: Any, run: Any, history: Optional[List[Any]] = None) -> Any:
    """A loop of your own: ``run.tool_params``, and ``run.tool_result(call)`` for each call."""
    from google.genai import types

    history = [] if history is None else history
    history.append({"role": "user", "parts": [{"text": run.message}]})
    config = types.GenerateContentConfig(system_instruction=run.system,
                                         tools=[_declared(ctx), *run.tool_params])
    response = None
    for _ in range(_MAX_MODEL_CALLS):
        response = client.models.generate_content(model=STUB_MODEL, contents=history, config=config)
        if not response.function_calls:
            break
        history.append(response.candidates[0].content)
        history.append({"role": "user", "parts": [
            run.tool_result(call) or _own_answer(ctx, call) for call in response.function_calls]})
    return response


def run_conversation(ctxs: Sequence[Ctx]) -> Any:
    """Two turns of one ``support.conversation()``, the second sent the first's exchange."""
    first, second = ctxs
    conversation = _config(first).conversation()
    for ctx in ctxs:
        _wire().register(ctx, script=_skill_script(ctx))
    client = _wire().client()
    history: List[Any] = []
    with conversation.run(user_message(first)) as run:
        _hand_loop(first, client, run, history)
    with conversation.run(user_message(second)) as run:
        return _hand_loop(second, client, run, history)


# ── second documented shapes ─────────────────────────────────────────────────


def _hand_written_shape(ctxs: Sequence[Ctx]) -> Any:
    out = []
    for ctx in ctxs:
        support = _config(ctx)
        _wire().register(ctx, script=_skill_script(ctx))
        with support.run(user_message(ctx)) as run:
            out.append(_hand_loop(ctx, _wire().client(), run))
    return out


async def _async_run(ctx: Ctx, support: Any) -> Any:
    from google.genai import types

    _wire().register(ctx, script=_skill_script(ctx))
    client = _wire().client()
    async with support.run(user_message(ctx)) as run:
        chat = client.aio.chats.create(model=STUB_MODEL, config=types.GenerateContentConfig(
            system_instruction=run.system, tools=[_function(ctx, asynchronous=True), *run.tools]))
        return await chat.send_message(run.message)


def _async_shape(ctxs: Sequence[Ctx]) -> Any:
    return [asyncio.run(_async_run(ctx, _config(ctx))) for ctx in ctxs]


DRIVER = Driver(
    name="google-genai",
    # The capability table's "Gemini API (with `agent_config(...)`)" row
    # (decimalai/_frameworks.py). The tracing half, init(google=True), is advertised in
    # prose and recorded in test_coverage.PROVIDER_FLAG_DRIVERS.
    covers=frozenset({"gemini-api"}),
    requires=("google.genai", "openinference.instrumentation.google_genai"),
    frameworks=("google-genai", "openinference-instrumentation-google-genai"),
    entrypoint=(
        "decimalai.providers.instrument(google=True) / "
        "decimalai.google_genai.agent_config(name).run(message)"
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
            name="hand_written_loop",
            doc=(
                "agent_config(name).run(message) around a client.models.generate_content loop "
                "with tools=[*mine, *run.tool_params], answering load_skill with "
                "run.tool_result(call)"
            ),
            run=_hand_written_shape,
            routed=True,
        ),
        Shape(
            name="async_chat",
            doc=(
                "async with agent_config(name).run(message) around a client.aio chat with "
                "automatic function calling, tools=[*mine, *run.tools]"
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
