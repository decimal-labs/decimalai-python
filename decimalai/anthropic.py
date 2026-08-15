"""Anthropic SDK integration.

One-line setup::

    import decimalai
    decimalai.init()

    from decimalai.anthropic import instrument
    instrument(enable_skill_loader=True)

    import anthropic
    client = anthropic.Anthropic()
    # Every `client.messages.create(...)` now has skills injected into `system`.

`instrument()` monkey-patches `anthropic.resources.messages.Messages.create`
(and the async counterpart) so the `system` argument is augmented with
the platform-routed skill fragment before the request is sent. User
``system`` content is preserved and placed AFTER the skill fragment.

Manual / explicit path::

    from decimalai.anthropic import skill_system

    resp = client.messages.create(
        model="claude-opus-4-7",
        system=skill_system("You are helpful", query=user_msg),
        messages=[{"role": "user", "content": user_msg}],
    )

Note on Anthropic Skills (beta): Anthropic's first-party Skills
primitive is intentionally not intercepted by this adapter. If you
use Skills, treat DecimalAI as the registry/router that Skills point
at, not as a replacement for Anthropic's runtime skill plumbing.
"""

from __future__ import annotations

import logging
import warnings
from contextvars import ContextVar
from typing import Any, List, Optional, Union

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
    """Build a ``system`` parameter for ``messages.create()`` with skills prepended.

    The function returns whichever shape best matches the input:

    - If ``base`` is None or a string → returns a string with the skill
      fragment prepended.
    - If ``base`` is a list of content blocks → returns a new list with
      a skill content block prepended (so user cache_control hints are
      preserved on the trailing blocks).

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

    try:
        fragment, routing_id = router.build_prompt_fragment(
            query=query, agent_name=agent_name,
        )
    except Exception:
        logger.debug("skill_system build_prompt_fragment failed (non-fatal)", exc_info=True)
        return base if base is not None else ""

    if routing_id:
        _set_routing_id(routing_id)

    if not fragment:
        return base if base is not None else ""

    if isinstance(base, list):
        return [{"type": "text", "text": fragment}, *base]
    if isinstance(base, str) and base:
        return f"{fragment}\n\n{base}"
    return fragment


def _inject_skills_into_create_kwargs(kwargs: dict) -> None:
    """Mutate Messages.create kwargs in place: add/augment the system param."""
    base = kwargs.get("system")
    messages = kwargs.get("messages") or []
    query = _extract_query_from_anthropic_messages(messages)
    kwargs["system"] = skill_system(base, query=query)


def _install_skill_loader() -> None:
    """Monkey-patch Anthropic's Messages.create / AsyncMessages.create."""
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

    original_create = Messages.create
    original_acreate = AsyncMessages.create

    def patched_create(self, *args, **kwargs):
        try:
            _inject_skills_into_create_kwargs(kwargs)
        except Exception:
            logger.debug("Skill injection failed (non-fatal)", exc_info=True)
        return original_create(self, *args, **kwargs)

    async def patched_acreate(self, *args, **kwargs):
        try:
            _inject_skills_into_create_kwargs(kwargs)
        except Exception:
            logger.debug("Skill injection failed (non-fatal)", exc_info=True)
        return await original_acreate(self, *args, **kwargs)

    Messages.create = patched_create  # type: ignore[method-assign]
    AsyncMessages.create = patched_acreate  # type: ignore[method-assign]
    _skill_loader_installed = True
    logger.info("DecimalAI SkillRouter loader installed (Anthropic)")


def instrument(
    *, enable_skill_loader: bool = False, enable_load_skill_tool: bool = False,
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
            ``client.messages.create()`` so skills auto-inject into
            ``system`` before each request.
        enable_load_skill_tool: Accepted but DORMANT on this adapter —
            the patch layer is a single
            ``messages.create()`` call, which cannot route a tool result
            back mid-turn — that needs a tool loop the caller owns.
            Skills stay prompt-injected here (``inject_skill_body`` now
            trims + budgets bodies); the live load_skill tool ships on
            ``decimalai.openai_agents`` and ``decimalai.pydantic_ai``.
    """
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


# ── Deprecated: install() ────────────────────────────────────────────────────
#
# `instrument()` is the current name. Behaviour is unchanged and this alias is
# not going away soon; it warns so the docs and the code agree on one name.
def install(*args, **kwargs):  # pragma: no cover - thin deprecation shim
    warnings.warn(
        "decimalai.anthropic.install() is deprecated; use "
        "decimalai.anthropic.instrument() instead. It installs the SkillRouter "
        "prompt-injection loader for the Anthropic SDK only when "
        "enable_skill_loader=True (default False), and has never added a "
        "skill to a workspace the way SkillRouter.install() does.",
        DeprecationWarning,
        stacklevel=2,
    )
    return instrument(*args, **kwargs)
