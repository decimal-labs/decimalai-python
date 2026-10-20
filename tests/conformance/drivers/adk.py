"""Google ADK driver.

Runs the snippet documented at ``https://docs.decimal.ai/sdk/python/frameworks/adk``:

    support = agent_config("support")
    agent = LlmAgent(name=..., model=..., instruction=support.instruction,
                     tools=[lookup, *support.tools])
    app = App(name=..., root_agent=agent, plugins=[DecimalAIPlugin()])
    runner = Runner(app=app, session_service=...)

The agent's prompt and skills come from the platform through ``agent_config()`` (an
instruction provider and ``load_skill``), and its traces from ``DecimalAIPlugin`` in
the ``App``, ADK's own place for plugins. Nothing in ADK is patched, so a runner is
traced only when its app carries the plugin.

ADK is Gemini-native and Gemini is quota-dead for this account, so the model is
LiteLLM's custom-provider hook (``litellm.custom_provider_map``) wired to the
shared stub script and reached through ``google.adk.models.lite_llm.LiteLlm`` —
the exact escape hatch the docs name for non-Gemini models. No key, no network.

The skills rail is graded whole: the skill list in the instruction, and
``load_skill``, which ADK runs in its own tool loop, so the body comes back as a
tool result in the same invocation.

NO ASSERTIONS BELOW THIS LINE. That is the driver contract.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import uuid
from typing import Any, Dict, List, Optional, Sequence

from ..delivery import INJECTED
from . import (
    STUB_MODEL_NAME,
    Capabilities,
    Ctx,
    Driver,
    DriverError,
    Shape,
    StubTurn,
    stub_script,
    tool_result,
    trace_observer,
    user_message,
)

#: LiteLLM provider prefix the stub registers itself under. ADK sees a model id
#: of ``<provider>/<model>`` and routes it to our handler instead of the network.
_PROVIDER = "conformance-stub"
_MODEL_ID = f"{_PROVIDER}/{STUB_MODEL_NAME}"

_REGISTER_LOCK = threading.Lock()
_REGISTERED = False


# ── the stub model ───────────────────────────────────────────────────────────


def _model_response(turn: Any, ctx: Ctx) -> Any:
    """Map one ``StubTurn`` onto a LiteLLM ``ModelResponse``."""
    from litellm.types.utils import ModelResponse

    message: Dict[str, Any] = {"role": "assistant", "content": turn.content or None}
    if turn.tool_call:
        name, args = turn.tool_call
        message["tool_calls"] = [{
            "id": "call_conformance_1",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]
    return ModelResponse(
        id=f"chatcmpl-conformance-{ctx.lane}",
        created=0,
        model=_MODEL_ID,
        object="chat.completion",
        choices=[{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if turn.tool_call else "stop",
        }],
        usage={
            "prompt_tokens": turn.input_tokens,
            "completion_tokens": turn.output_tokens,
            "total_tokens": turn.input_tokens + turn.output_tokens,
        },
    )


class _StubHandler:
    """One scripted conversation, keyed by the lane that owns it.

    LiteLLM's ``custom_provider_map`` is process-global, so every lane's
    requests arrive at one handler object. It routes on the sentinel carried in
    the request messages — which is also what makes a lane reading another
    lane's script impossible to hide.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ctxs: Dict[str, Ctx] = {}
        self._turns: Dict[str, int] = {}
        self._order: Dict[str, int] = {}
        self._failing: set = set()
        self._scripts: Dict[str, List[Any]] = {}
        self._seq = 0

    # -- registration ------------------------------------------------
    def register(self, ctx: Ctx, *, fail: bool = False, script: Optional[List[Any]] = None) -> str:
        key = uuid.uuid4().hex
        with self._lock:
            self._seq += 1
            self._ctxs[key] = ctx
            self._turns[key] = 0
            self._order[key] = self._seq
            if fail:
                self._failing.add(key)
            if script is not None:
                self._scripts[key] = script
        return key

    # -- lookup ------------------------------------------------------
    def _key_for(self, messages: Any) -> Optional[str]:
        """The registration this conversation belongs to.

        Two collisions have to be resolved, and getting either wrong silently
        answers one phase with another phase's script:

        * ``…-lane3`` CONTAINS the base run's sentinel, so a plain substring
          scan hands every lane the base run's turns — pick the LONGEST
          matching sentinel.
        * ``main``, ``repeat`` and ``error`` share one ctx, so their sentinels
          are identical — pick the most RECENT registration, which is the phase
          currently running.
        """
        blob = json.dumps(messages, default=str)
        with self._lock:
            matches = [
                key for key, ctx in self._ctxs.items() if ctx.prompt_sentinel in blob
            ]
            if not matches:
                return None
            return max(
                matches,
                key=lambda k: (len(self._ctxs[k].prompt_sentinel), self._order[k]),
            )

    def _next(self, messages: Any) -> Any:
        from . import StubTurn

        key = self._key_for(messages)
        if key is None:
            # No lane owns this conversation. Answer something inert rather than
            # guessing — a trace built on it fails C4 on the missing sentinel,
            # which is the honest report.
            return _model_response(
                StubTurn(None, "conformance: unrouted request", 1, 1), _LANELESS
            )
        with self._lock:
            ctx = self._ctxs[key]
            index = self._turns[key]
            self._turns[key] = index + 1
            failing = key in self._failing
            script = self._scripts.get(key)
        if failing:
            raise DriverError("conformance: the model failed on purpose")
        script = script or stub_script(ctx)
        return _model_response(script[min(index, len(script) - 1)], ctx)


#: Placeholder ctx for a request no lane claims — only its ``lane`` is read.
_LANELESS = Ctx(
    base_url="", api_key="", agent_name="", prompt_sentinel="\x00none\x00",
    reply_sentinel="", tool_name="", tool_sentinel="", workdir="",
)


_HANDLER = _StubHandler()


def _stub_provider() -> Any:
    """Register the stub with LiteLLM once, and hand back the ADK model object."""
    global _REGISTERED
    import litellm
    from litellm import CustomLLM

    with _REGISTER_LOCK:
        if not _REGISTERED:
            class _ConformanceLLM(CustomLLM):
                def completion(self, *args: Any, **kwargs: Any) -> Any:
                    return _HANDLER._next(kwargs.get("messages"))

                async def acompletion(self, *args: Any, **kwargs: Any) -> Any:
                    return _HANDLER._next(kwargs.get("messages"))

            existing = list(getattr(litellm, "custom_provider_map", None) or [])
            existing.append({"provider": _PROVIDER, "custom_handler": _ConformanceLLM()})
            litellm.custom_provider_map = existing
            # The error phase raises on purpose; without this LiteLLM prints its
            # "Give Feedback / Get Help" banner to stderr for a failure the suite
            # asked for. Suppresses the banner only, never the exception.
            litellm.suppress_debug_info = True
            _REGISTERED = True

    from google.adk.models.lite_llm import LiteLlm

    return LiteLlm(model=_MODEL_ID)


# ── the documented snippet ───────────────────────────────────────────────────


def _node_name(ctx: Ctx) -> str:
    """ADK node names must be Python identifiers; the DecimalAI label need not."""
    return re.sub(r"\W", "_", ctx.agent_name)


def _agent(ctx: Ctx, *, configs: Optional[Dict[str, Any]] = None) -> Any:
    """This agent's ``agent_config()`` instruction and tools on the developer's own
    ``LlmAgent``. ``configs`` shares one config per agent name, as a module-level
    one would be."""
    from google.adk.agents import LlmAgent

    from decimalai.adk import agent_config

    configs = {} if configs is None else configs
    if ctx.agent_name not in configs:
        configs[ctx.agent_name] = agent_config(ctx.agent_name)
    support = configs[ctx.agent_name]

    def lookup(query: str) -> dict:
        """Look a value up for the conformance run."""
        return {"result": tool_result(ctx, query)}

    lookup.__name__ = ctx.tool_name

    return LlmAgent(
        name=_node_name(ctx),
        model=_stub_provider(),
        instruction=support.instruction,
        tools=[lookup, *support.tools],
    )


def _skill_script(ctx: Ctx) -> List[Any]:
    """The rail's script: call ``load_skill``, then answer.

    ``load_skill`` is the rail this framework advertises, and ADK owns its tool
    loop, so the body comes back mid-invocation. Same shape and reply text as the
    shared script; only the tool called differs, because the tool IS the mechanism
    under test. In the ``injected`` delivery cell the tool is switched OFF in this
    process (``DECIMALAI_LOAD_SKILL_TOOL=0``), so the model is scripted onto the
    ordinary conformance tool and the body has to arrive in the instruction.
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


def _runner(ctx: Ctx, agent: Any, *, on_trace: Any = None) -> Any:
    """``InMemoryRunner(app=App(..., plugins=[DecimalAIPlugin(...)]))``."""
    from google.adk.apps import App
    from google.adk.runners import InMemoryRunner

    from decimalai.adk import DecimalAIPlugin

    return InMemoryRunner(app=App(
        name=_node_name(ctx), root_agent=agent,
        plugins=[DecimalAIPlugin(agent_name=ctx.agent_name, on_trace=on_trace)],
    ))


async def _drive(ctx: Ctx, runner: Any) -> str:
    from google.genai import types

    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="conformance",
    )
    message = types.Content(role="user", parts=[types.Part(text=user_message(ctx))])
    final = ""
    async for event in runner.run_async(
        user_id="conformance", session_id=session.id, new_message=message,
    ):
        if event.content and event.content.parts:
            text = "".join(p.text or "" for p in event.content.parts)
            if text:
                final = text
    return final


def run(ctx: Ctx) -> Any:
    """One ADK invocation of the documented snippet."""
    _HANDLER.register(ctx)
    return asyncio.run(_drive(ctx, _runner(ctx, _agent(ctx))))


async def _gather(ctxs: Sequence[Ctx]) -> Any:
    configs: Dict[str, Any] = {}
    return await asyncio.gather(
        *(_drive(c, _runner(c, _agent(c, configs=configs))) for c in ctxs),
        return_exceptions=True,
    )


def run_concurrent(ctxs: Sequence[Ctx]) -> Any:
    """N invocations at once, on ONE event loop — ADK's own concurrency model.

    Not ``fanout_threads``: ADK and LiteLLM are async-native, and a thread per
    lane gives each lane its own event loop, which LiteLLM's process-global
    logging worker then complains about in a way that has nothing to do with
    the adapter under test. One loop, N concurrent invocations, is also what an
    ADK server actually does.
    """
    for ctx in ctxs:
        _HANDLER.register(ctx)
    return asyncio.run(_gather(ctxs))


def run_nested(ctx: Ctx) -> Any:
    """One invocation with the suite's observer as the plugin's ``on_trace``.

    The harness runs this inside a ``decimalai.start_trace()`` of its own;
    ``asyncio.run`` copies the caller's context into the invocation, which is where
    the plugin reads the enclosing trace at ``before_run_callback``.
    """
    _HANDLER.register(ctx)
    runner = _runner(ctx, _agent(ctx), on_trace=trace_observer(ctx))
    return asyncio.run(_drive(ctx, runner))


def run_error(ctx: Ctx) -> Any:
    """The same invocation, with the model raising on the first generation."""
    _HANDLER.register(ctx, fail=True)
    return asyncio.run(_drive(ctx, _runner(ctx, _agent(ctx))))


def run_degenerate(ctx: Ctx) -> Any:
    """A custom ADK agent with no model, no tools and no instruction.

    ``BaseAgent`` is ADK's escape hatch for an agent that is just code, and it
    is the manifest-gate case here: the plugin introspects the root agent and
    finds nothing to declare, so it must not mint a manifest version that reads
    as "the model and tools were deleted".
    """
    from google.adk.agents import BaseAgent
    from google.adk.events import Event
    from google.genai import types

    sentinel = ctx.prompt_sentinel

    class EchoAgent(BaseAgent):
        async def _run_async_impl(self, invocation_context: Any):  # noqa: ANN202
            yield Event(
                author=self.name,
                invocation_id=invocation_context.invocation_id,
                content=types.Content(
                    role="model", parts=[types.Part(text=f"echo {sentinel}")],
                ),
            )

    agent = EchoAgent(name=_node_name(ctx))
    return asyncio.run(_drive(ctx, _runner(ctx, agent)))


def run_skills(ctxs: Sequence[Ctx]) -> Any:
    """The skills rail, N lanes at once, on one event loop.

    The lanes share one agent name and so one config, the way every invocation of
    a module-level agent does. Each lane asks a different question, so the router
    mints a routing decision per lane and a leaked one is visible.
    """
    for ctx in ctxs:
        _HANDLER.register(ctx, script=_skill_script(ctx))
    return asyncio.run(_gather(ctxs))


def run_unnamed(ctx: Ctx) -> Any:
    """The rail on an agent ADK names one way and DecimalAI another.

    ADK requires every agent to have a name, and it must be a Python identifier,
    so the ADK agent here is called ``conformance_adk_…`` while the DecimalAI agent
    is ``conformance-adk-…``. Nothing on the plugin names it: the DecimalAI name is
    given once, in ``agent_config()``, and the trace and the routing call must both
    carry it.
    """
    from google.adk.apps import App
    from google.adk.runners import InMemoryRunner

    from decimalai.adk import DecimalAIPlugin

    _HANDLER.register(ctx, script=_skill_script(ctx))
    runner = InMemoryRunner(app=App(
        name=_node_name(ctx), root_agent=_agent(ctx), plugins=[DecimalAIPlugin()],
    ))
    return asyncio.run(_drive(ctx, runner))


async def _converse(runner: Any, first: Ctx, second: Ctx) -> Any:
    from google.genai import types

    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="conformance",
    )
    out = []
    for ctx in (first, second):
        message = types.Content(role="user", parts=[types.Part(text=user_message(ctx))])
        async for event in runner.run_async(
            user_id="conformance", session_id=session.id, new_message=message,
        ):
            out.append(event)
    return out


def run_conversation(ctxs: Sequence[Ctx]) -> Any:
    """Two turns in ONE session: the second ``run_async`` on the same session id,
    which is how ADK carries a conversation's history. Each invocation calls the
    instruction provider again, so the second turn is routed on its own message."""
    first, second = ctxs
    for ctx in ctxs:
        _HANDLER.register(ctx, script=_skill_script(ctx))
    runner = _runner(first, _agent(first))
    return asyncio.run(_converse(runner, first, second))


def _app_runner_shape(ctxs: Sequence[Ctx]) -> Any:
    """``Runner(app=App(...))`` — the shape ``adk web``, ``adk run`` and
    ``adk api_server`` build — through both runner classes.

    Lane 0: ``InMemoryRunner(app=App(name, root_agent, plugins=[DecimalAIPlugin()]))``,
    the plugin naming nothing, so the trace is ``agent_config()``'s agent. Lane 1:
    ``Runner(app=App(..., plugins=[DecimalAIPlugin(agent_name=...)]),
    session_service=...)``, the SDK's documented path.
    """
    from google.adk.apps import App
    from google.adk.runners import InMemoryRunner, Runner
    from google.adk.sessions import InMemorySessionService

    from decimalai.adk import DecimalAIPlugin

    out = []
    first, second = ctxs
    _HANDLER.register(first)
    in_memory = InMemoryRunner(app=App(
        name=_node_name(first), root_agent=_agent(first), plugins=[DecimalAIPlugin()],
    ))
    out.append(asyncio.run(_drive(first, in_memory)))

    _HANDLER.register(second)
    explicit = Runner(
        app=App(
            name=_node_name(second), root_agent=_agent(second),
            plugins=[DecimalAIPlugin(agent_name=second.agent_name)],
        ),
        session_service=InMemorySessionService(),
    )
    out.append(asyncio.run(_drive(second, explicit)))
    return out


DRIVER = Driver(
    name="adk",
    covers=frozenset({"google-adk"}),
    requires=("google.adk", "litellm"),
    frameworks=("google-adk",),
    entrypoint="decimalai.adk.agent_config(name) + DecimalAIPlugin() in App(plugins=[...])",
    run=run,
    run_concurrent=run_concurrent,
    run_error=run_error,
    run_degenerate=run_degenerate,
    run_skills=run_skills,
    run_nested=run_nested,
    run_unnamed=run_unnamed,
    run_conversation=run_conversation,
    loads_agent_from_platform=True,
    shapes=(
        Shape(
            name="app_runner",
            doc=(
                "Runner(app=App(...)): InMemoryRunner(app=App(name, root_agent, "
                "plugins=[DecimalAIPlugin()])), then "
                "Runner(app=App(..., plugins=[DecimalAIPlugin(agent_name=...)]), session_service=...)"
            ),
            run=_app_runner_shape,
            lanes=2,
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
