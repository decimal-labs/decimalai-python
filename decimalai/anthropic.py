"""Anthropic SDK integration.

One-line setup::

    import decimalai
    decimalai.init()

    from decimalai.anthropic import instrument
    instrument(enable_skill_loader=True)

    import anthropic
    client = anthropic.Anthropic()
    # Every `client.messages.create()`, `.stream()` and `.parse()` call now
    # has skills injected into `system` — sync and async clients alike.

`instrument()` monkey-patches ``create``, ``stream`` and ``parse`` on
`anthropic.resources.messages.Messages` and `AsyncMessages`, so the `system`
argument is augmented with the platform-routed skill fragment before the
request is built. All three are patched because in anthropic 1.x the
``stream()`` and ``parse()`` helpers build their own request and never go
through ``create()``. The beta surface (``client.beta.messages.*``, including
``tool_runner()``) is NOT patched; use :func:`skill_system` there.

User ``system`` content is preserved and kept FIRST; the skill fragment is
appended after it, so the caller's own instructions stay in the cacheable
prefix (the fragment is rebuilt per query and so differs on every call —
in front, it would break the prompt cache for everything behind it).

On the async client the routing call (a blocking HTTP request) runs on a
worker thread, so it never stalls the event loop. ``AsyncMessages.stream()``
therefore routes when its ``async with`` block is entered, not when
``stream()`` is called.

Manual / explicit path::

    from decimalai.anthropic import skill_system

    resp = client.messages.create(
        model="claude-opus-5-5",
        system=skill_system("You are helpful", query=user_msg),
        messages=[{"role": "user", "content": user_msg}],
    )

Note on Anthropic's own Agent Skills (generally available since 2026-08-19:
``client.skills`` and ``container={"skills": [...]}`` on the Messages API):
this adapter does not upload to or attach those. It delivers DecimalAI-routed
skills through the system prompt only, and leaves a ``container`` you pass
untouched.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from contextvars import Context, ContextVar, copy_context
from typing import Any, Callable, List, Optional, Union

logger = logging.getLogger("decimalai.anthropic")


# ── SkillRouter routing-id context ──────────────────────────
_routing_id_ctx: ContextVar[Optional[str]] = ContextVar(
    "decimalai_skill_router_routing_id_anthropic", default=None,
)


def _set_routing_id(routing_id: Optional[str]) -> None:
    _routing_id_ctx.set(routing_id)


def get_current_routing_id() -> Optional[str]:
    """Read (without clearing) the current routing_id, for trace stamping."""
    return _routing_id_ctx.get()


def _scope() -> Optional[str]:
    """This run's routing scope — the live OTel trace id, or None.

    Passed to the Router so two concurrent runs of one agent do not share a
    fragment-cache slot (and therefore a routing decision). Outside a run there
    is nothing to scope to, and None keeps the pre-existing behaviour.
    """
    try:
        from .otel import current_run_key
        key = current_run_key()
    except Exception:
        return None
    return None if key is None else f"{key:032x}"


# ── SkillRouter dynamic loader ──────────────────────────────
_skill_loader_installed = False
# Set by `instrument(agent_name=...)`. Forwarded on every auto-injected
# `messages.create()` / `.stream()` / `.parse()` so agent-scope skills
# resolve; None keeps the old org-wide-only behaviour.
_install_agent_name: Optional[str] = None
_skill_router_singleton: Any = None


def _get_skill_router() -> Any:
    """Lazily construct a SkillRouter using the SDK's global config."""
    global _skill_router_singleton
    # The kill switch: a later init(enabled=False) withholds the router (kept
    # for a re-enable) from a loader an earlier init installed.
    from ._config import _disabled_by_init

    if _disabled_by_init():
        return None
    if _skill_router_singleton is not None:
        return _skill_router_singleton
    try:
        from ._config import _get_config
        from .skill_router import SkillRouter
        config = _get_config()
        _skill_router_singleton = SkillRouter(
            api_key=config.api_key,
            base_url=config.base_url,
            # Every routing call names its agent; no router default (see
            # `skill_router._AGENT_FROM_INIT`).
            agent_name=None,
            inject_body=config.resolve_inject_body(has_tool_loop=False),
        )
        return _skill_router_singleton
    except Exception:
        logger.debug("SkillRouter singleton init failed", exc_info=True)
        return None


def _extract_query_from_anthropic_messages(messages: Any) -> Optional[str]:
    """Walk an Anthropic-format messages list backward for the latest user msg."""
    if not isinstance(messages, list):
        return None
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            # Multi-modal: concatenate text blocks.
            parts = [
                blk.get("text", "")
                for blk in content
                if isinstance(blk, dict) and blk.get("type") == "text"
            ]
            joined = " ".join(p for p in parts if p).strip()
            if joined:
                return joined
    return None


def skill_system(
    base: Optional[Union[str, List[Any]]] = None,
    *,
    query: Optional[str] = None,
    agent_name: Optional[str] = None,
) -> Union[str, List[Any]]:
    """Build a ``system`` parameter for ``messages.create()`` with skills appended.

    The function returns whichever shape best matches the input:

    - If ``base`` is None or a string → returns a string with the skill
      fragment appended after it.
    - If ``base`` is a list of content blocks → returns a new list that is
      the caller's blocks unchanged, in their original order, with ONE skill
      block added at the end.

    Order matters for cost, not just for reading. The fragment is rebuilt from
    ``query`` on every call, so it is different bytes each time; the caller's
    ``base`` is the same bytes each time. Anthropic matches a cached prefix up
    to each ``cache_control`` breakpoint, so a varying block in FRONT of the
    caller's blocks moves every one of them off their cached offset and turns
    a hit into a full miss. Appending leaves the caller's blocks — and any
    ``cache_control`` hints they carry — byte-identical and in place.

    Args:
        base: The user's normal system prompt — string or content-block list.
        query: Optional query for semantic routing. None = full menu.
        agent_name: Optional override forwarded to the Router.

    Returns:
        The augmented system payload, ready to pass straight to
        ``client.messages.create(system=...)``.
    """
    router = _get_skill_router()
    if router is None:
        return base if base is not None else ""

    from .skill_router import (
        _release_scoped_routing_rail,
        consume_last_delivered_names,
        consume_last_delivered_versions,
        consume_last_offered_names,
    )

    scope = _scope()
    consume_last_delivered_versions()
    try:
        fragment, routing_id = router.build_prompt_fragment(
            query=query, agent_name=agent_name, scope=scope,
        )
    except Exception:
        consume_last_delivered_versions()
        logger.debug("skill_system build_prompt_fragment failed (non-fatal)", exc_info=True)
        return base if base is not None else ""
    finally:
        # The router keeps its own copy of this decision under the run's scope,
        # for adapters that read it back from there. This one reads the
        # per-call rails below, so the copy is released here or never.
        _release_scoped_routing_rail(router, scope)

    offered = consume_last_offered_names()
    delivered = consume_last_delivered_names()
    delivered_versions = consume_last_delivered_versions()

    # Claim nothing for a call we inject nothing into. This guard used to sit
    # BELOW the routing-id stamp, so a call that got an empty fragment still
    # reported a routing decision — a trace asserting the model was routed when
    # its prompt was untouched.
    if not fragment:
        return base if base is not None else ""

    if routing_id:
        _set_routing_id(routing_id)

    # Record against the run's trace, in the same breath as returning the text
    # that goes into `system`. The drains below are the per-call CONTEXTVAR
    # rails, written by build_prompt_fragment one statement ago on this thread —
    # NOT the router's instance rails, which are process-global clear-on-read
    # state that under a concurrent fanout hands lane 1 everybody's names.
    try:
        from .otel import record_skill_rail
        record_skill_rail(
            routing_id=routing_id,
            offered=offered,
            delivered=delivered,
            delivered_versions=delivered_versions,
            prompt_text=fragment,
        )
    except Exception:
        logger.debug("skill rail recording failed (non-fatal)", exc_info=True)

    # The caller's blocks are copied through untouched and in order — no
    # rewriting, no re-tagging — so any `cache_control` breakpoint they carry
    # still sits at the same offset in the request it did before we ran.
    if isinstance(base, list):
        return [*base, {"type": "text", "text": fragment}]
    if isinstance(base, str) and base:
        return f"{base}\n\n{fragment}"
    return fragment


def _routing_agent_name() -> Optional[str]:
    """The DecimalAI agent an auto-patched call's skills are routed for.

    The name the TRACE is filed under, so routing and attribution agree: the
    run-scoped name the OTel rail stamps onto every span of this run
    (``decimalai.otel._active_agent_name``, set by ``providers.agent_run(...)``
    and by ``providers.instrument(agent_name=...)``) first, then this adapter's
    own ``instrument(agent_name=...)``. Agent-scope skills (every registry
    skill pulled onto one agent), version pins and restricted skills only
    resolve when the name is sent, so routing a named run as nobody offered
    none of them while its trace said it had been routed.
    """
    try:
        from .otel import _active_agent_name

        name = _active_agent_name.get()
    except Exception:
        name = None
    return name or _install_agent_name


def _inject_skills_into_create_kwargs(kwargs: dict) -> None:
    """Mutate create/stream/parse kwargs in place: add/augment the system param."""
    base = kwargs.get("system")
    messages = kwargs.get("messages") or []
    query = _extract_query_from_anthropic_messages(messages)
    # Forward the agent's name so agent-scope Use rows resolve — see
    # `_routing_agent_name`. `skill_system` has always accepted `agent_name`;
    # the auto-patch path first sent none, then only the install name, so a run
    # named with `agent_run(...)` was routed for no agent (or the wrong one).
    kwargs["system"] = skill_system(
        base, query=query, agent_name=_routing_agent_name(),
    )


#: The ``Messages`` / ``AsyncMessages`` methods that send a message. In
#: anthropic 1.x each builds its own request, so each needs its own patch.
_PATCHED_METHODS = ("create", "stream", "parse")

# Set while a patched method is injecting, so a patched method that calls
# another one (say a ``stream()`` built on ``create(stream=True)``) injects
# once, in the outer call. A ContextVar rather than a global: concurrent calls
# on other threads or asyncio tasks each start with it unset.
_injecting: ContextVar[bool] = ContextVar(
    "decimalai_anthropic_injecting", default=False,
)


def _inject_best_effort(kwargs: dict) -> None:
    """Inject into ``kwargs``, never raising: a router failure sends the
    caller's own prompt, untouched."""
    try:
        _inject_skills_into_create_kwargs(kwargs)
    except Exception:
        logger.debug("Skill injection failed (non-fatal)", exc_info=True)


def _inject_once(kwargs: dict) -> Any:
    """Inject unless an outer patched call already did. Returns a reset token,
    or None when this call is the inner one and must leave kwargs alone."""
    if _injecting.get():
        return None
    token = _injecting.set(True)
    _inject_best_effort(kwargs)
    return token


# ── the async client: route on a worker thread ──────────────────────────────
#
# The router is a synchronous HTTP client: one routing call is a route request
# plus a body fetch, each with its own read budget, plus admission retries.
# Made inline from a coroutine, it stalled every other coroutine on the event
# loop (in an async server, every other conversation in the process) until the
# platform answered.

_UNSET = object()


def _running_asyncio_loop() -> Optional[asyncio.AbstractEventLoop]:
    """The asyncio loop running on this thread, or None.

    None under another async library: anthropic's async client is built on
    anyio and also runs under trio.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _adopt_sdk_context_writes(ctx: Context) -> None:
    """Copy into the current Context what routing wrote into ``ctx``.

    Only this SDK's own variables (named ``decimalai_*``): this adapter's
    routing id (read back with :func:`get_current_routing_id`), the router's
    per-call rails that :func:`skill_system` drains, and the router's per-turn
    body budget, which a later ``router.load_skill()`` in the same coroutine
    reads. The caller is left with the Context an inline call would have left.
    Anything else a worker thread set stays in that thread's copy.
    """
    for var, value in ctx.items():
        if var.name.startswith("decimalai_") and var.get(_UNSET) is not value:
            var.set(value)


async def _inject_once_off_loop(kwargs: dict) -> Any:
    """`_inject_once` for the async client, with the routing call on a worker
    thread. Returns a reset token for the caller to reset, or None.

    The thread runs in a COPY of the caller's Context, so everything routing
    reads from it is the caller's: the run's agent name set by
    ``providers.agent_run()``, and the live OTel span that the run's routing
    decision is filed under. Its writes are then copied back
    (`_adopt_sdk_context_writes`).

    On asyncio, cancelling the caller while the router is in flight cancels
    the call at once (anyio, under trio, waits for the thread). The abandoned
    routing call finishes on its thread and the message is never sent, though
    the decision it made is still recorded against the run.
    """
    if _injecting.get():
        return None
    ctx = copy_context()
    started: List[bool] = []

    def route() -> None:
        started.append(True)
        _inject_best_effort(kwargs)

    try:
        loop = _running_asyncio_loop()
        if loop is not None:
            await loop.run_in_executor(None, ctx.run, route)
        else:
            import anyio.to_thread

            await anyio.to_thread.run_sync(ctx.run, route)
    except Exception:
        # Only the hand-off can fail here (`route` never raises), for example
        # an executor that is shutting down. If routing never started, route
        # inline: a blocked loop for one call is better than a call that
        # silently goes out without its skills.
        logger.debug("off-loop skill routing hand-off failed", exc_info=True)
        if not started:
            _inject_best_effort(kwargs)
    else:
        _adopt_sdk_context_writes(ctx)
    return _injecting.set(True)


def _patch_async_call(original: Any) -> Any:
    """Wrap ``AsyncMessages.create`` or ``.parse``.

    ``parse`` is a coroutine function. ``create`` is anthropic's
    ``@required_args`` wrapper, a plain function that returns the coroutine.
    Both bind ``kwargs`` when called, so this awaits the routing first and calls
    the original after. The guard stays up while the request is awaited, so a
    helper built on another patched method does not route again.
    """
    @functools.wraps(original)
    async def patched(self, *args, **kwargs):
        token = await _inject_once_off_loop(kwargs)
        try:
            result = original(self, *args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        finally:
            if token is not None:
                _injecting.reset(token)
    return patched


class _OffLoopMessageStreamManager:
    """What the patched ``AsyncMessages.stream()`` returns in place of the
    SDK's ``AsyncMessageStreamManager``.

    The SDK's ``stream()`` is a plain function that puts ``system`` into the
    request body when it is called, so nothing can be awaited before the body
    is built. This manager calls the real ``stream()`` in ``__aenter__``, after
    the routing has been awaited on a worker thread, and hands back the SDK's
    own ``AsyncMessageStream``. It has the manager's interface
    (``async with``), not its class.
    """

    def __init__(self, open_stream: Callable[[], Any], kwargs: dict) -> None:
        self._open_stream = open_stream
        self._kwargs = kwargs
        self._manager: Any = None

    async def __aenter__(self) -> Any:
        if self._manager is not None:
            # Entered twice: let the SDK's manager answer as it would.
            return await self._manager.__aenter__()
        token = await _inject_once_off_loop(self._kwargs)
        try:
            self._manager = self._open_stream()
            # The guard stays up until the request is sent. A stream() built
            # on create(stream=True) runs that create inside this __aenter__,
            # and it must see that this call has already routed.
            return await self._manager.__aenter__()
        finally:
            if token is not None:
                _injecting.reset(token)

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        if self._manager is None:
            return None
        return await self._manager.__aexit__(exc_type, exc, tb)


def _patch_async_stream(original: Any) -> Any:
    """Wrap ``AsyncMessages.stream``: routing moves to ``__aenter__``."""
    @functools.wraps(original)
    def patched(self, *args, **kwargs):
        # A closure over this call's `kwargs` dict, not functools.partial: a
        # partial copies the dict now, and the routing writes `system` into it
        # later.
        return _OffLoopMessageStreamManager(
            lambda: original(self, *args, **kwargs), kwargs,
        )
    return patched


def _patch(original: Any, *, name: str = "create", async_client: bool = False) -> Any:
    """Wrap one SDK method so skills are injected into its ``system`` kwarg."""
    if async_client and name == "stream":
        return _patch_async_stream(original)
    if async_client or inspect.iscoroutinefunction(original):
        return _patch_async_call(original)

    # The sync client: every method binds kwargs when called, so injecting
    # before the call is enough.
    @functools.wraps(original)
    def patched(self, *args, **kwargs):
        token = _inject_once(kwargs)
        try:
            return original(self, *args, **kwargs)
        finally:
            if token is not None:
                _injecting.reset(token)
    return patched


def _install_skill_loader() -> None:
    """Monkey-patch create/stream/parse on Anthropic's Messages and AsyncMessages."""
    global _skill_loader_installed
    if _skill_loader_installed:
        return
    try:
        from anthropic.resources.messages import AsyncMessages, Messages
    except ImportError:
        logger.warning(
            "enable_skill_loader=True but anthropic SDK not installed; "
            "skipping skill loader install"
        )
        return

    for cls in (Messages, AsyncMessages):
        for name in _PATCHED_METHODS:
            original = getattr(cls, name, None)
            if original is None:
                # Say so rather than skip quietly: calls through this method
                # would otherwise run without skills and nothing would show it.
                logger.warning(
                    "anthropic %s.%s not found; calls through it will NOT "
                    "receive DecimalAI skills",
                    cls.__name__, name,
                )
                continue
            setattr(cls, name, _patch(
                original, name=name, async_client=cls is AsyncMessages,
            ))
    _skill_loader_installed = True
    logger.info("DecimalAI SkillRouter loader installed (Anthropic)")


def instrument(
    *,
    enable_skill_loader: bool = False,
    enable_load_skill_tool: bool = False,
    agent_name: Optional[str] = None,
) -> None:
    """Install DecimalAI integration for the Anthropic SDK.

    Currently this surface is skill-loader-only. Tracing for raw
    Anthropic SDK calls flows through the generic ``decimalai.trace``
    decorator or the OpenTelemetry exporter; this module focuses on
    the prompt-assembly side.

    Note: this adapter never reads from or writes to disk, so there is
    no ``disk_sync`` parameter. If you're running inside a disk-loading
    runtime (Claude Code, Cursor) — which is especially likely for the
    Anthropic SDK — ``_warn_if_disk_runtime_detected`` will log a
    one-shot warning on enable to flag the duplicate-injection risk.

    Args:
        enable_skill_loader: When True, monkey-patch
            ``client.messages.create()``, ``.stream()`` and ``.parse()`` (sync
            and async) so skills auto-inject into ``system`` before each
            request.
        agent_name: Scopes the routed skill menu to this agent. Required for
            agent-scope skills (a registry skill pulled onto one agent) to be
            offered at all — without it the router only sees org-owned
            workspace-scope skills. A run named with
            ``decimalai.providers.agent_run(name)`` is routed for THAT name,
            the one its trace is filed under; this is the fallback outside
            one. Defaults to the name an earlier call set, else the agent
            ``decimalai.init()`` named (or ``DECIMAL_AGENT_NAME``). Callers who
            assemble ``system`` themselves can keep passing it per-call to
            :func:`skill_system` instead.
        enable_load_skill_tool: Accepted but DORMANT on this adapter —
            the patch layer is a single
            ``messages.create()`` call, which cannot route a tool result
            back mid-turn — that needs a tool loop the caller owns.
            Skills stay prompt-injected here (``inject_skill_body`` now
            trims + budgets bodies); the live load_skill tool ships on
            ``decimalai.openai_agents`` and ``decimalai.pydantic_ai``.
    """
    global _install_agent_name
    if agent_name is None and _install_agent_name is None:
        # Named by neither this call nor an earlier one: the agent init()
        # named (or DECIMAL_AGENT_NAME). `init(anthropic=True)` installs the
        # tracing half (decimalai.providers), never this module, so this is
        # the only way init()'s name reaches the skill loader.
        from ._config import default_agent_name
        agent_name = default_agent_name()
    if agent_name is not None:
        _install_agent_name = agent_name
    if enable_load_skill_tool:
        logger.warning(
            "enable_load_skill_tool is not supported on the anthropic adapter "
            "(no tool loop to route the result); staying on prompt injection. "
            "Use openai_agents or pydantic_ai for the native load_skill tool."
        )
    if enable_skill_loader:
        from .skill_router import _warn_if_disk_runtime_detected
        _warn_if_disk_runtime_detected("anthropic")
        _install_skill_loader()
    logger.info(
        "DecimalAI Anthropic integration installed (skill_loader=%s)",
        enable_skill_loader,
    )
