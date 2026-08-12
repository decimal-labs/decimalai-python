"""Pydantic AI integration.

One-line install path::

    import decimalai
    decimalai.init()

    from decimalai.pydantic_ai import instrument
    instrument(enable_skill_loader=True)

    from pydantic_ai import Agent
    agent = Agent("openai:gpt-4o", system_prompt="You are helpful")
    # Skills are now auto-loaded into every agent.run() call.

The install monkey-patches `pydantic_ai.Agent.__init__` so each newly
constructed Agent gets an extra `@agent.system_prompt`-registered
function that calls `SkillRouter.build_prompt_fragment()` per turn and
prepends the result to the base system prompt.

Tracing for Pydantic AI is observed via the underlying provider SDK
(OpenAI/Anthropic) — install the matching tracing adapter alongside.
"""

from __future__ import annotations

import warnings

import logging
from contextvars import ContextVar
from typing import Any, Optional

logger = logging.getLogger("decimalai.pydantic_ai")


# ── SkillRouter routing-id context ──────────────────────────
# Pydantic AI doesn't have its own DecimalAI trace processor (yet);
# downstream provider adapters (openai_agents / anthropic) read from
# their own contextvars. We surface our routing_id via a getter so
# the provider adapter can pull it in if both are installed.
_routing_id_ctx: ContextVar[Optional[str]] = ContextVar(
    "decimalai_skill_router_routing_id_pydantic_ai", default=None,
)


def _set_routing_id(routing_id: Optional[str]) -> None:
    _routing_id_ctx.set(routing_id)


def get_current_routing_id() -> Optional[str]:
    """Read (without clearing) the current routing_id.

    Exposed so downstream framework adapters can read the routing_id
    set by the Pydantic AI skill loader and stamp it onto a trace.
    """
    return _routing_id_ctx.get()


# ── SkillRouter dynamic loader ──────────────────────────────
_skill_loader_installed = False
_skill_router_singleton: Any = None


def _get_skill_router() -> Any:
    """Lazily construct a SkillRouter using the SDK's global config."""
    global _skill_router_singleton
    if _skill_router_singleton is not None:
        return _skill_router_singleton
    try:
        from ._config import _get_config
        from .skill_router import SkillRouter
        config = _get_config()
        _skill_router_singleton = SkillRouter(
            api_key=config.api_key,
            base_url=config.base_url,
            inject_body=getattr(config, "inject_skill_body", False),
        )
        return _skill_router_singleton
    except Exception:
        logger.debug("SkillRouter singleton init failed", exc_info=True)
        return None


# True once at least one Agent got the load_skill tool — gates the prompt
# hint in _skills_system_prompt (per-agent introspection of registered tools
# is private API in pydantic_ai; a module flag is the stable alternative).
_load_skill_tool_active = False


def _load_skill_tool_enabled() -> bool:
    """Config gate for the load_skill tool (kill switch:
    DECIMALAI_LOAD_SKILL_TOOL=0 / init(load_skill_tool=False))."""
    try:
        from ._config import _get_config
        return bool(getattr(_get_config(), "load_skill_tool", True))
    except Exception:
        return True


def _handle_load_skill(name: str) -> str:
    """Tool callback — always returns a string the model can act on."""
    router = _get_skill_router()
    if router is None:
        return "load_skill error: skill router unavailable."
    try:
        return router.load_skill(name)
    except Exception:
        logger.debug("load_skill handler failed (non-fatal)", exc_info=True)
        return f"load_skill error: could not load {name!r} (transient error)."


async def _skills_system_prompt(ctx: Any) -> str:
    """Async system-prompt function registered on every Agent.

    The user message isn't directly available on RunContext, so we
    default to full-menu mode (query=None) — every active skill's
    name + description is added to the prompt. Users who want smart
    routing can disable the loader and call
    `SkillRouter.build_prompt_fragment()` themselves with explicit
    context.
    """
    router = _get_skill_router()
    if router is None:
        return ""
    try:
        agent_name = None
        agent_obj = getattr(ctx, "agent", None)
        if agent_obj is not None:
            agent_name = getattr(agent_obj, "name", None)
        fragment, routing_id = router.build_prompt_fragment(
            query=None, agent_name=agent_name,
        )
        if routing_id:
            _set_routing_id(routing_id)
        # Tell the model how bodies arrive when the tool exists.
        # The server fragment keeps the activation-statement instruction
        # unchanged (Stage-M parity).
        if fragment and _load_skill_tool_active:
            from .skill_router import LOAD_SKILL_PROMPT_HINT
            fragment = f"{fragment}\n{LOAD_SKILL_PROMPT_HINT}"
        return fragment or ""
    except Exception:
        logger.debug("Pydantic AI skill loader failed (non-fatal)", exc_info=True)
        return ""


def _install_skill_loader() -> None:
    """Monkey-patch `pydantic_ai.Agent.__init__` to auto-register skills.

    Also registers the load_skill tool on every new Agent (the progressive-disclosure path):
    Pydantic AI owns its tool loop, so the tool result routes back mid-turn
    for free — this adapter ships the tool live."""
    global _skill_loader_installed
    if _skill_loader_installed:
        return
    try:
        from pydantic_ai import Agent
    except ImportError:
        logger.warning(
            "enable_skill_loader=True but pydantic-ai not installed; "
            "skipping skill loader install. "
            "Install it with: pip install \"decimalai[pydantic-ai]\""
        )
        return

    original_init = Agent.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            # Pydantic AI exposes the registration as a decorator AND
            # as a function call. We use the function-call form here.
            register = getattr(self, "system_prompt", None)
            if callable(register):
                register(_skills_system_prompt)
        except Exception:
            logger.debug(
                "Failed to register skills system_prompt on Agent (non-fatal)",
                exc_info=True,
            )
        # Register load_skill so surfaced descriptions are
        # executable — an agent that can see a skill can pull its body.
        if _load_skill_tool_enabled():
            try:
                tool_plain = getattr(self, "tool_plain", None)
                if callable(tool_plain):
                    from .skill_router import LOAD_SKILL_TOOL_DESCRIPTION

                    def load_skill(name: str) -> str:
                        return _handle_load_skill(name)

                    load_skill.__doc__ = LOAD_SKILL_TOOL_DESCRIPTION
                    tool_plain(load_skill)
                    global _load_skill_tool_active
                    _load_skill_tool_active = True
            except Exception:
                logger.debug(
                    "load_skill tool registration failed (non-fatal)", exc_info=True
                )

    Agent.__init__ = patched_init  # type: ignore[method-assign]
    _skill_loader_installed = True
    logger.info("DecimalAI SkillRouter loader installed (Pydantic AI)")


def instrument(*, enable_skill_loader: bool = False) -> None:
    """Install DecimalAI integration for Pydantic AI.

    Tracing for Pydantic AI flows through the underlying provider SDK,
    so this instrument() currently focuses on the skill loader. Pair with
    `decimalai.openai_agents.instrument()` or `decimalai.anthropic` for
    full tracing coverage.

    Note: this adapter never reads from or writes to disk, so there is
    no ``disk_sync`` parameter. If you're running inside a disk-loading
    runtime (Claude Code, Cursor), ``_warn_if_disk_runtime_detected``
    will log a one-shot warning on enable to flag the duplicate-injection
    risk.

    Args:
        enable_skill_loader: When True, monkey-patch Agent so new
            instances auto-load skills into the system prompt.
    """
    if enable_skill_loader:
        from .skill_router import _warn_if_disk_runtime_detected
        _warn_if_disk_runtime_detected("pydantic_ai")
        _install_skill_loader()
    logger.info(
        "DecimalAI Pydantic AI integration installed (skill_loader=%s)",
        enable_skill_loader,
    )


# ── Deprecated: install() ────────────────────────────────────────────────────
#
# Renamed to `instrument()` 2026-08-11. "install" was doing double duty across
# this SDK: here it turned on TRACING for a framework, while
# `SkillRouter.install()` added a SKILL to a workspace. Two unrelated actions
# under one word, in one package — and the skill sense is the one users arrive
# with, because it is what every extension marketplace means by install.
#
# Behaviour is unchanged and this alias is not going away soon; it warns so the
# docs and the code agree on one name.
def install(*args, **kwargs):  # pragma: no cover - thin deprecation shim
    warnings.warn(
        "decimalai.pydantic_ai.install() is deprecated; use "
        "decimalai.pydantic_ai.instrument() instead. It turns on tracing for pydantic_ai "
        "and has never had anything to do with installing a skill.",
        DeprecationWarning,
        stacklevel=2,
    )
    return instrument(*args, **kwargs)
