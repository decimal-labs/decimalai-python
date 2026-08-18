"""Google ADK driver.

Runs the snippet documented at ``decimalai-docs/sdk/python/frameworks/adk.mdx``,
in its **explicit** form:

    runner = Runner(agent=..., app_name=..., session_service=...,
                    plugins=[DecimalaiPlugin(agent_name="support")])

That is the form the docs give for "per-Runner control over naming", and it is
the only one that can name a second agent in the same process: the ``adk=True``
flag form monkeypatches ``Runner.__init__`` to inject ONE shared plugin, so
every Runner in the process would report the same agent. Both are documented;
this driver exercises the one that can actually express what the contract asks
for, and says so rather than declaring the identity items N/A.

ADK is Gemini-native and Gemini is quota-dead for this account, so the model is
LiteLLM's custom-provider hook (``litellm.custom_provider_map``) wired to the
shared stub script and reached through ``google.adk.models.lite_llm.LiteLlm`` —
the exact escape hatch the docs name for non-Gemini models. No key, no network.

There is no skills rail on this adapter: the docs capability table records
ADK's skills-rail column as "—" and the page says so in words, so C8 is
declared N/A with that reason.

NO ASSERTIONS BELOW THIS LINE. That is the driver contract.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import uuid
from typing import Any, Dict, Optional, Sequence

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
        self._seq = 0

    # -- registration ------------------------------------------------
    def register(self, ctx: Ctx, *, fail: bool = False) -> str:
        key = uuid.uuid4().hex
        with self._lock:
            self._seq += 1
            self._ctxs[key] = ctx
            self._turns[key] = 0
            self._order[key] = self._seq
            if fail:
                self._failing.add(key)
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
        if failing:
            raise DriverError("conformance: the model failed on purpose")
        script = stub_script(ctx)
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


def _agent(ctx: Ctx) -> Any:
    from google.adk.agents import LlmAgent

    def lookup(query: str) -> dict:
        """Look a value up for the conformance run."""
        return {"result": tool_result(ctx, query)}

    lookup.__name__ = ctx.tool_name

    return LlmAgent(
        name=_node_name(ctx),
        model=_stub_provider(),
        instruction=SYSTEM_PROMPT,
        tools=[lookup],
    )


def _runner(ctx: Ctx, agent: Any) -> Any:
    from google.adk.runners import InMemoryRunner

    from decimalai.adk import DecimalaiPlugin

    return InMemoryRunner(
        agent=agent,
        app_name=_node_name(ctx),
        plugins=[DecimalaiPlugin(agent_name=ctx.agent_name)],
    )


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
    """One ADK invocation through an explicitly-attached DecimalaiPlugin."""
    _HANDLER.register(ctx)
    return asyncio.run(_drive(ctx, _runner(ctx, _agent(ctx))))


async def _gather(ctxs: Sequence[Ctx]) -> Any:
    return await asyncio.gather(
        *(_drive(c, _runner(c, _agent(c))) for c in ctxs), return_exceptions=True
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


DRIVER = Driver(
    name="adk",
    covers=frozenset({"google-adk"}),
    requires=("google.adk", "litellm"),
    entrypoint="decimalai.adk.DecimalaiPlugin",
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
                "the adapter has no skills rail — the docs page says so in words "
                "(\"No skills rail on this adapter yet\") and the capability table "
                "records ADK's skills-rail column as '—'. Skills reach an ADK agent "
                "only by hand, via SkillRouter.build_prompt_fragment(). This silences "
                "the activation items (C13/C13b) as well as C8: with nothing offered "
                "and no loader tool, there is no model action that could constitute an "
                "activation, and recording one from prompt presence would be a "
                "fabrication."
            ),
        },
    ),
)
