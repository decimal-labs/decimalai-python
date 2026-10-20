"""`decimalai.load_agent()` — an agent's prompt and skills, read at run time.

An agent's prompt is typed in the dashboard and stored as a versioned object on
the platform, and its skills are assigned there. This is how a running process
gets both, for code that calls a model itself (a raw provider SDK, a hand-written
tool loop)::

    support = decimalai.load_agent("refund-bot", fallback=BASE_PROMPT)
    system = support.system(question, context=customer)  # prompt, then skill list
    tools = [support.tool_spec]                          # the load_skill tool
    text = support.load_skill(name)                      # when the model calls it

A framework with its own adapter has its own ``load_agent()`` that returns the
same two things in that framework's shape: ``decimalai.openai_agents.load_agent``.
All of them share `decimalai._agent_binding.AgentBinding`.

EXPLICIT, NEVER AUTO-INJECTED
-----------------------------
The prompt comes back as data and the caller hands it to the model themselves.
Injecting it silently would mean their repo reads "Never issue refunds over
$500" while the model was handed "You are a helpful assistant", and nothing in
their own source would show it.

Explicit does NOT cost the no-redeploy property: the prompt is read at run time
and re-read in the background, so a dashboard edit reaches the next turns.

FAIL CLOSED
-----------
`SavedPrompt.system_prompt is None` means one thing only: the agent exists and
has no prompt set. It MUST NOT also be reachable from a 404, a timeout, a missing
key, or a response shape this SDK did not expect. An agent that silently runs
with no instructions at all is worse than one that refuses to start.

That is why `_from_payload` checks for the KEY and never reaches for
`payload.get("system_prompt")`: `.get()` turns every wrong-shaped 200 — an
older backend, a captive-portal login page, a proxy's error envelope — into a
plausible-looking empty prompt.
"""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass, replace
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from ._agent_binding import AgentBinding

logger = logging.getLogger("decimalai")


@dataclass(frozen=True, repr=False)
class SavedPrompt:
    """What the platform knows about an agent's prompt, as of one read.

    A frozen dataclass, not a dict, for two reasons. Against a backend older
    than a field, `config.system_prompt` fails at the attribute — at startup,
    on the line that reads it — rather than as a `KeyError` deep inside a
    request handler days later. And frozen means a value read once at import
    cannot be mutated halfway through a process into something the traces no
    longer describe.

    Attributes:
        agent_name: The agent's CANONICAL name, as the server resolved it. Not
            necessarily the string that was passed: an agent renamed in the
            dashboard keeps answering to its old name (there is no
            server→client push, so deployed code goes on sending the old one),
            and `resolved_from` records that it happened.
        system_prompt: The effective prompt, or None when none is set. Never
            None for any other reason — see the module docstring.
        version_number: Which version `system_prompt` came from.
        content_hash: Its content hash. Pass it back as `if_none_match` to poll
            for edits cheaply; it is also the right thing to log next to a run.
        label: The optional note whoever saved that version left on it.
        version_mode: "latest" (track edits) or "pinned" (hold one version).
        pinned_version_number: The pinned version when pinned, else None.
        agent_id: The agent's platform id. None for an agent that predates the
            identity table and has never been written to since — which is why
            it is not the thing anything here keys on.
        resolved_from: The name that was ASKED for, when it differs from
            `agent_name` — i.e. this agent has been renamed. None normally.
    """

    agent_name: str
    system_prompt: Optional[str]
    version_number: Optional[int] = None
    content_hash: Optional[str] = None
    label: Optional[str] = None
    version_mode: str = "latest"
    pinned_version_number: Optional[int] = None
    agent_id: Optional[str] = None
    resolved_from: Optional[str] = None
    #: True when this config did NOT come from the platform on this call — it is
    #: either a cached value served because the platform was unreachable, or the
    #: caller's own `fallback=`. Log it. An agent silently running on a stale or
    #: substitute prompt is exactly the thing that must not be invisible.
    is_fallback: bool = False
    #: Set only when `is_fallback` — how old the served value is, in seconds,
    #: or None for a caller-supplied fallback that was never fetched.
    stale_age_seconds: Optional[float] = None

    def __repr__(self) -> str:
        """Never dumps the prompt body — a prompt is up to 100,000 characters.

        The default dataclass repr would put all of it into every traceback,
        log line and `print(config)`. The distinction that actually matters at
        a glance — "is there a prompt, and which version is it?" — survives:
        `system_prompt=<412 chars>` and `system_prompt=None` do not look alike.
        """
        body = (
            "None" if self.system_prompt is None
            else f"<{len(self.system_prompt)} chars>"
        )
        return (
            f"SavedPrompt(agent_name={self.agent_name!r}, system_prompt={body}, "
            f"version_number={self.version_number!r}, "
            f"content_hash={self.content_hash!r})"
        )

    @classmethod
    def _from_payload(
        cls, payload: Any, *, requested_name: str
    ) -> "SavedPrompt":
        """Build from `GET /api/v1/agents/{name}/prompt`, or raise.

        The membership check on `system_prompt` is the load-bearing line of
        this module; the rest of the fields use `.get()` because a backend that
        stops sending `label` is a cosmetic gap, while one that stops sending
        the prompt is an agent about to run with no instructions.
        """
        if not isinstance(payload, dict) or "system_prompt" not in payload:
            raise ValueError(
                "Unexpected response reading the system prompt for "
                f"{requested_name!r}: no 'system_prompt' field. Refusing to "
                "treat this as 'no prompt set' — check that base_url points "
                "at a DecimalAI backend and that it is current. "
                f"Got: {_summarize(payload)}"
            )

        prompt = payload["system_prompt"]
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError(
                "Unexpected response reading the system prompt for "
                f"{requested_name!r}: 'system_prompt' is "
                f"{type(prompt).__name__}, not a string."
            )

        return cls(
            # `or requested_name`: canonicalization is a nicety (it makes a
            # renamed agent's config self-describing), so a backend that omits
            # the name degrades to the name the caller already holds rather
            # than failing the load.
            agent_name=str(payload.get("agent_name") or requested_name),
            system_prompt=prompt,
            version_number=payload.get("version_number"),
            content_hash=payload.get("content_hash"),
            label=payload.get("label"),
            version_mode=str(payload.get("version_mode") or "latest"),
            pinned_version_number=payload.get("pinned_version_number"),
            agent_id=payload.get("agent_id"),
            resolved_from=payload.get("resolved_from"),
        )


def _summarize(payload: Any) -> str:
    """A short, safe description of a response we could not read.

    Keys for a dict (never values — one of them could be a secret), and a
    clipped repr for anything else. The point is to tell an HTML login page
    apart from an old backend without pasting either into a log.
    """
    if isinstance(payload, dict):
        keys = sorted(str(k) for k in payload)[:8]
        return ("a JSON object with keys " + ", ".join(keys)) if keys else "{}"
    text = repr(payload)
    return text[:120] + "…" if len(text) > 120 else text


#: `(name_sent, canonical_name)` pairs already warned about, so a rename is
#: reported once per process rather than on every `load_agent()` call. Same
#: shape as `_warn_on_near_miss_agent_name` in `__init__.py`: a warning that
#: repeats is a warning people filter out.
_WARNED_RENAMES: Set[Tuple[str, str]] = set()


def _warn_if_renamed(config: SavedPrompt) -> None:
    """Say so when the server canonicalized the name — traces do NOT follow.

    A rename is handled asymmetrically on the server, and the halves are easy
    to mistake for each other. `canonical_agent_name` is called from the agents
    router and from the skill resolver, so the PROMPT and the SKILLS follow a
    rename. It is called from nowhere in the trace ingest path, so traces keep
    landing under whatever string `instrument(agent_name=...)` was given.

    The result is an agent that looks entirely healthy and is quietly split in
    two: the dashboard page for the new name shows the prompt and the skills,
    and the runs show up on the old one. Nothing errors. So the one place that
    can see both names says it, once, and names the exact edit that fixes it.

    Deliberately does NOT rewrite anything. The name bound at `instrument()` is
    the caller's, and silently re-pointing their traces mid-process is the
    class of behaviour this whole module exists to avoid.
    """
    if not config.resolved_from:
        return
    key = (config.resolved_from, config.agent_name)
    if key in _WARNED_RENAMES:
        return
    _WARNED_RENAMES.add(key)
    logger.warning(
        "decimalai: agent %r was renamed to %r. Its prompt and "
        "skills still resolve under the old name, but trace ingest does NOT "
        "follow a rename — your instrument(agent_name=%r) is still filing "
        "traces under the old name. Update it to %r.",
        config.resolved_from, config.agent_name,
        config.resolved_from, config.agent_name,
    )


#: agent_name -> (config, fetched_at_monotonic). Process-local; a cached entry is
#: served only while it is younger than `cache_ttl_seconds`, or — regardless of
#: age — when the platform cannot be reached at all. See load_agent().
_prompt_cache: Dict[str, "tuple[SavedPrompt, float]"] = {}
_prompt_cache_lock = Lock()

#: Default: DO NOT serve a cached prompt in place of a live read.
#:
#: The cache exists for one job — surviving an outage — so by default it is only
#: consulted when the platform cannot be reached. Serving a fresh read from cache
#: would quietly cost the property Phase 3c exists for (edit the prompt in the
#: dashboard, the next run picks it up) and would blunt fail-closed: a revoked key
#: or a deleted agent would keep "working" until the TTL expired.
#: `load_agent()` is documented as a once-per-process call, so there is nothing to
#: optimise here. A caller who really does read per turn can opt in.
_DEFAULT_PROMPT_CACHE_TTL_S = 0.0


def _reset_prompt_cache() -> None:
    """Drop every cached prompt.

    Called by `init()`. Re-initialising points the SDK at a different key,
    workspace or base_url, and serving the previous one's prompt for the same
    agent name would be a cross-workspace read. Also what test suites use to
    keep the process-local cache from leaking between cases.
    """
    with _prompt_cache_lock:
        _prompt_cache.clear()


def read_prompt(
    agent_name: str,
    *,
    version: Optional[int] = None,
    fallback: Optional[str] = None,
    cache_ttl_seconds: float = _DEFAULT_PROMPT_CACHE_TTL_S,
) -> SavedPrompt:
    """Read an agent's saved prompt from DecimalAI: one HTTP request.

    The read every ``load_agent()`` starts with and re-runs to refresh its copy.

    Args:
        agent_name: The agent as it exists on DecimalAI. A renamed agent still
            answers to its old name — `agent_name` comes back canonical, and a
            one-time warning says so.
        version: Read one historical version instead of the effective one.
        fallback: A prompt to use if the platform cannot be reached AND nothing
            is cached. The result then carries `is_fallback=True`.
        cache_ttl_seconds: Serve a previously-fetched prompt without re-asking.
            Default 0. Regardless of it, the last good value IS served when the
            platform is unreachable: stale beats down.

    Raises:
        DecimalConfigError: `decimalai.init()` has not been called.
        AgentNotFoundError: no such agent in this workspace.
        ValueError: `agent_name` is empty, or the response was not a prompt
            payload this SDK can read.
        DecimalAPIError: any other HTTP failure (bad key, unresolvable pin).
    """
    if not str(agent_name or "").strip():
        raise ValueError(
            "load_agent() needs the agent's name, e.g. "
            'decimalai.load_agent("refund-bot").'
        )

    from ._config import _get_client

    key = f"{agent_name}\x00{version if version is not None else ''}"

    # A fresh cached value short-circuits the network entirely.
    if cache_ttl_seconds > 0:
        with _prompt_cache_lock:
            entry = _prompt_cache.get(key)
        if entry is not None:
            cached, fetched_at = entry
            if time.monotonic() - fetched_at < cache_ttl_seconds:
                return cached

    client = _get_client()
    try:
        payload = client.get_agent_prompt(agent_name, version=version)
    except Exception as exc:
        # Only availability failures may be softened. A 404 (no such agent) or a
        # 409 (unresolvable pin) is a real answer from a reachable platform and
        # must still raise — serving a stale prompt for a deleted agent would be
        # worse than stopping.
        if not _is_availability_failure(exc):
            raise
        with _prompt_cache_lock:
            entry = _prompt_cache.get(key)
        if entry is not None:
            cached, fetched_at = entry
            age = time.monotonic() - fetched_at
            logger.warning(
                "decimalai: platform unreachable reading %r's prompt (%s) — serving the "
                "cached copy, %.0fs old.",
                agent_name, exc, age,
            )
            return replace(cached, is_fallback=True, stale_age_seconds=age)
        if fallback is not None:
            logger.warning(
                "decimalai: platform unreachable reading %r's prompt (%s) and nothing "
                "cached — using the fallback prompt.",
                agent_name, exc,
            )
            return SavedPrompt(
                agent_name=str(agent_name),
                system_prompt=fallback,
                is_fallback=True,
            )
        raise

    # `get_agent_prompt` only returns None for a 304, which needs a conditional
    # request this call never makes.
    config = SavedPrompt._from_payload(payload, requested_name=str(agent_name))
    _warn_if_renamed(config)
    with _prompt_cache_lock:
        _prompt_cache[key] = (config, time.monotonic())
    return config


def _is_availability_failure(exc: BaseException) -> bool:
    """Whether this failure means "could not reach the platform" rather than
    "the platform answered, and the answer was no"."""
    import httpx

    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    return status in (500, 502, 503, 504)


# ── what decimalai.load_agent() returns ──────────────────────────────────────

#: The turn `AgentConfig.system()` last composed in this context. `load_skill()`
#: is served against that turn's routing decision, so call it in the same thread
#: or task as the `system()` it follows.
_TURN: "ContextVar[Optional[str]]" = ContextVar("decimalai_agent_turn", default=None)

#: The SkillRouter behind every `load_agent()` config. `init()` drops it (see
#: `_drop_adapter_skill_routers`) when a re-init changes the key or workspace.
_skill_router_singleton: Any = None


def _get_skill_router() -> Any:
    """The SkillRouter for `AgentConfig`, built on first use; None when switched off."""
    global _skill_router_singleton
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
            api_key=config.api_key, base_url=config.base_url,
            # Every routing call names its agent; no router default.
            agent_name=None,
            # Skill text reaches the model through load_skill, not the prompt.
            inject_body=False,
        )
        return _skill_router_singleton
    except Exception:
        logger.debug("SkillRouter init failed", exc_info=True)
        return None


def _ensure_initialized() -> None:
    """`decimalai.init()` from the environment when nothing initialized the SDK yet."""
    from . import _config

    if _config._client is None:
        import decimalai

        decimalai.init()


#: The span attributes a prompt record becomes (see `_prompt_attributes`).
_PROMPT_ATTRIBUTES = (
    ("decimal.agent", "decimal_agent"),
    ("decimal.prompt.source", "source"),
    ("decimal.prompt.version", "version"),
    ("decimal.prompt.content_hash", "content_hash"),
    ("decimal.prompt.variables", "variables"),
)


def _prompt_attributes(record: Dict[str, Any]) -> Dict[str, Any]:
    """A run's prompt record as OpenTelemetry span attributes (names, never values)."""
    attributes: Dict[str, Any] = {}
    for attribute, key in _PROMPT_ATTRIBUTES:
        value = record.get(key)
        if value is None:
            continue
        attributes[attribute] = list(value) if isinstance(value, (list, tuple)) else value
    if record.get("skills_unavailable"):
        attributes["decimal.skills.unavailable"] = record["skills_unavailable"]
    return attributes


def _annotate_current_span(attributes: Dict[str, Any]) -> None:
    """Set attributes on the OpenTelemetry span that is current, if one is recording."""
    try:
        from opentelemetry import trace as trace_api
    except ImportError:
        return
    try:
        span = trace_api.get_current_span()
        if span is None or not span.is_recording():
            return
        for key, value in attributes.items():
            span.set_attribute(key, value)
    except Exception:  # a record must never break a turn
        logger.debug("could not annotate the current span", exc_info=True)


class AgentConfig(AgentBinding):
    """One agent's prompt and skills from DecimalAI, for code that calls a model itself.

    Made by :func:`load_agent`::

        support = decimalai.load_agent("support", fallback=BASE_PROMPT)
        system = support.system(question, context=customer)
        response = client.messages.create(
            model=MODEL, system=system, tools=[*my_tools, support.tool_spec], ...)
        # when the model calls load_skill:
        text = support.load_skill(tool_input["name"])

    Each `system()` call is one turn: it routes that message (the first turn
    before it returns, later ones in the background unless the agent's list is
    cut to fit), fills the prompt's ``{{placeholders}}`` from ``context``, and
    returns the prompt, then the skill list, then a sentence naming the tool.
    With an OpenTelemetry run live (``decimalai.init(openai=True)``,
    ``decimalai.agent_run()``), the turn's record goes on that run's trace.

    Attributes:
        name: The agent's canonical DecimalAI name.
    """

    _VALUES_HINT = "Pass them as system(..., context=...), as attributes or dict keys."

    def __init__(self, prompt: Any, *, fallback: Optional[str], on_unavailable: str) -> None:
        super().__init__(prompt, fallback=fallback, on_unavailable=on_unavailable)
        self._tool_names: List[str] = []

    def __repr__(self) -> str:
        return (
            f"AgentConfig(name={self.name!r}, prompt_version={self.prompt_version!r}, "
            f"skills={self.skill_names!r})"
        )

    # ── the pieces ───────────────────────────────────────────
    @property
    def prompt(self) -> Optional[str]:
        """The saved prompt as written (placeholders unfilled), else the fallback."""
        with self._lock:
            text = getattr(self._prompt, "system_prompt", None)
        return text if text is not None else self._fallback

    def system(self, message: Optional[str] = None, *, context: Any = None) -> str:
        """This turn's system text: the prompt, filled from ``context``, then the skill list.

        Args:
            message: The user's latest message, which the skills are routed on.
                Without one, the skills come in the platform's fixed order.
            context: Values for the prompt's ``{{placeholders}}``, as attributes
                or dict keys (``{{customer.name}}`` reads ``context.customer.name``
                or ``context["customer"]["name"]``).

        Raises:
            MissingPromptVariable: the prompt needs a value ``context`` lacks.
            RuntimeError: ``on_unavailable="fail"`` and DecimalAI never returned
                this agent's skills.
        """
        turn = f"turn_{uuid4().hex}"
        _TURN.set(turn)
        context_of_turn = {"message": message, "values": context}
        state = self._prepare_run_sync(context_of_turn, None, turn)
        return self._compose(context_of_turn, None, turn, state)

    @property
    def tool_spec(self) -> Dict[str, Any]:
        """The ``load_skill`` tool as ``{name, description, parameters}`` (JSON Schema).

        Translate it to your provider's tool type. Its ``name`` argument is an enum
        of the agent's skills when the whole list fits; otherwise free text.
        """
        from .skill_router import LOAD_SKILL_TOOL_DESCRIPTION, LOAD_SKILL_TOOL_NAME

        with self._lock:
            names = list(self._tool_names)
        return {
            "name": LOAD_SKILL_TOOL_NAME,
            "description": LOAD_SKILL_TOOL_DESCRIPTION,
            "parameters": self._tool_schema(names),
        }

    def load_skill(self, name: str) -> str:
        """One listed skill's full text, for the model's ``load_skill`` call.

        Served against the routing decision of the latest `system()` in this
        thread or task. Always returns text the model can act on: a skill already
        loaded this turn, an unknown name or an outage come back as a sentence.
        """
        return self._load_sync(str(name or "").strip())

    # ── the seams AgentBinding asks for ─────────────────────
    def _router(self) -> Any:
        return _get_skill_router()

    def _run_key(self) -> Optional[str]:
        return _TURN.get()

    def _message(self, context: Any, agent: Any) -> Optional[str]:
        message = context.get("message")
        return message.strip() or None if isinstance(message, str) else None

    def _values(self, context: Any) -> Any:
        return context.get("values")

    def _agent_label(self, agent: Any) -> Optional[str]:
        return None

    def _on_menu(self, names: List[str]) -> None:
        with self._lock:
            self._tool_names = list(names)

    def _record(self, **fields: Any) -> bool:
        from .otel import record_skill_rail

        if isinstance(fields.get("prompt"), dict):
            _annotate_current_span(_prompt_attributes(fields["prompt"]))
        return record_skill_rail(
            routing_id=fields.get("routing_id"),
            offered=fields.get("offered"),
            delivered=fields.get("delivered"),
            delivered_versions=fields.get("delivered_versions"),
            prompt_text=fields.get("system_prompt"),
        )

    def _serve_load(self, name: str) -> str:
        router = self._router()
        if router is None:
            return "load_skill error: skill router unavailable."
        scope = self._run_key()
        try:
            body = router.load_skill(name, scope=scope, agent_name=self.name)
        except Exception:
            logger.debug("load_skill failed (non-fatal)", exc_info=True)
            return f"load_skill error: could not load {name!r} (transient error)."
        if scope is not None:
            # Only names whose body was actually served reach the record.
            try:
                served = router.consume_loaded_names(scope=scope)
                hashes = router.consume_loaded_hashes(scope=scope) or {}
                versions = router.consume_delivered_versions(scope=scope)
                if served:
                    from .otel import record_skill_rail

                    record_skill_rail(loaded=served, loaded_hashes=hashes, delivered_versions=versions)
            except Exception:
                logger.debug("load_skill record failed (non-fatal)", exc_info=True)
        return body


def load_agent(
    agent_name: str,
    *,
    fallback: Optional[str] = None,
    on_unavailable: str = "continue",
) -> AgentConfig:
    """Pull an agent's prompt and skills from DecimalAI, for code that calls a model itself.

    ::

        import decimalai

        support = decimalai.load_agent("support", fallback=BASE_PROMPT)
        system = support.system(question, context=customer)
        tools = [*my_tools, support.tool_spec]
        # ... when the model calls load_skill(name): support.load_skill(name)

    Reads the agent's saved prompt here (so a misspelled name fails now) and
    keeps a local copy, re-read in the background at most once a minute. Each
    `AgentConfig.system()` call routes its message: the first turn before it
    returns, later ones in the background, and every turn of an agent with more
    skills than fit the list. Initializes the SDK from ``DECIMAL_API_KEY`` when
    ``decimalai.init()`` has not been called.

    A framework with its own adapter has its own ``load_agent()``, which hands
    the same prompt and skills over in that framework's shape:
    ``decimalai.openai_agents.load_agent``.

    Args:
        agent_name: The agent's name in DecimalAI.
        fallback: The prompt to use when the agent has no saved prompt, or when
            DecimalAI cannot be reached now. Without one, an unreachable
            DecimalAI raises here rather than starting an agent with no prompt.
        on_unavailable: ``"continue"`` (default) composes turns without skills
            when DecimalAI never returned them, and the record says so;
            ``"fail"`` raises from `AgentConfig.system()` instead.

    Raises:
        AgentNotFoundError: DecimalAI has no agent by this name.
        ValueError: ``on_unavailable`` is neither ``"continue"`` nor ``"fail"``.
    """
    if on_unavailable not in ("continue", "fail"):
        raise ValueError("on_unavailable must be 'continue' or 'fail'")
    _ensure_initialized()
    prompt = read_prompt(agent_name, fallback=fallback)
    return AgentConfig(prompt, fallback=fallback, on_unavailable=on_unavailable)
