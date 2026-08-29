"""`decimalai.load_agent()` — the agent's configuration, read at run time.

An agent's system prompt is typed in the dashboard and stored as a versioned
object on the platform. This is how a running process reads it.

EXPLICIT, NEVER AUTO-INJECTED
-----------------------------
The prompt comes back as data and the caller passes it to their framework
themselves::

    config = decimalai.load_agent("refund-bot")
    agent = Agent(name="refund-bot", instructions=config.system_prompt)

Skills ARE auto-injected, and that asymmetry is the deliberate part. A skill
menu is ADDITIVE — worst case it costs a few hundred tokens the model ignores.
A system prompt is the agent's core instruction, and injecting it silently
means their repo reads "Never issue refunds over $500" while the model was
handed "You are a helpful assistant". There is no way for the customer to see
that from their own source, which makes it unfixable from their side.

Explicit does NOT cost the no-redeploy property. `load_agent()` fetches at run
time, so editing the prompt in the dashboard changes what the next process
start sends. The line in their file stays the same; only the value moves.

FAIL CLOSED
-----------
`system_prompt is None` means one thing and only one thing: the agent exists
and has no prompt set. That is a real state — `system_prompt` is optional at
creation — so it MUST NOT also be reachable from a 404, a timeout, a missing
key, or a response shape this SDK did not expect. An agent that silently runs
with no instructions at all is worse than one that refuses to start, and it
looks identical to a working one until someone reads the output closely.

That is why `_from_payload` checks for the KEY and never reaches for
`payload.get("system_prompt")`: `.get()` turns every wrong-shaped 200 — an
older backend, a captive-portal login page, a proxy's error envelope — into a
plausible-looking empty prompt. Same shape as the `?? 'free'` default that told
paying customers they were on the free plan.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from threading import Lock
from typing import Any, Dict, Optional, Set, Tuple

logger = logging.getLogger("decimalai")


@dataclass(frozen=True, repr=False)
class AgentConfig:
    """What the platform knows about an agent, as of this call.

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
            f"AgentConfig(agent_name={self.agent_name!r}, system_prompt={body}, "
            f"version_number={self.version_number!r}, "
            f"content_hash={self.content_hash!r})"
        )

    @classmethod
    def _from_payload(
        cls, payload: Any, *, requested_name: str
    ) -> "AgentConfig":
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


def _warn_if_renamed(config: AgentConfig) -> None:
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
        "decimalai.load_agent: agent %r was renamed to %r. Its prompt and "
        "skills still resolve under the old name, but trace ingest does NOT "
        "follow a rename — your instrument(agent_name=%r) is still filing "
        "traces under the old name. Update it to %r.",
        config.resolved_from, config.agent_name,
        config.resolved_from, config.agent_name,
    )


#: agent_name -> (config, fetched_at_monotonic). Process-local; a cached entry is
#: served only while it is younger than `cache_ttl_seconds`, or — regardless of
#: age — when the platform cannot be reached at all. See load_agent().
_prompt_cache: Dict[str, "tuple[AgentConfig, float]"] = {}
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


def load_agent(
    agent_name: str,
    *,
    version: Optional[int] = None,
    fallback: Optional[str] = None,
    cache_ttl_seconds: float = _DEFAULT_PROMPT_CACHE_TTL_S,
) -> AgentConfig:
    """Read an agent's configuration from DecimalAI.

    Call it once, where the agent is built — it is one HTTP request, and it is
    meant to run at process start, not per turn::

        import decimalai
        decimalai.init()

        config = decimalai.load_agent("refund-bot")
        agent = Agent(name="refund-bot", instructions=config.system_prompt)

    `config.system_prompt` is `None` when the agent has no prompt set. Send no
    system message in that case; do not substitute one, or the agent silently
    starts following instructions nobody wrote.

    Args:
        agent_name: The agent as it exists on DecimalAI. A renamed agent still
            answers to its old name — `config.agent_name` comes back
            canonical, and a one-time warning says so.
        version: Read one historical version instead of the effective one.
            Rarely what you want: the effective prompt already honours a pin.
        fallback: A prompt to use if the platform cannot be reached AND nothing
            is cached. Supply it in anything that has to boot during an outage —
            it is the difference between an agent that degrades and one that
            does not start. The returned config carries `is_fallback=True`.
        cache_ttl_seconds: Opt in to serving a previously-fetched config without
            re-asking. Default 0 — every call reads the platform, so a dashboard
            edit reaches the next run and a revoked key still fails. Regardless
            of this setting, the last good value IS served when the platform is
            unreachable: stale beats down.

    Returns:
        A frozen :class:`AgentConfig`. Check `is_fallback` if you care whether
        it came from the platform on this call.

    Raises:
        DecimalConfigError: `decimalai.init()` has not been called.
        AgentNotFoundError: no such agent in this workspace.
        ValueError: `agent_name` is empty, or the response was not a prompt
            payload this SDK can read.
        DecimalAPIError: any other HTTP failure (bad key, unresolvable pin).

    A prompt that cannot be read still stops the process — fail-closed is right
    for a prompt, because substituting one silently makes the agent follow
    instructions nobody wrote. What changed on 2026-08-28 is that there is now
    something to fall back TO. Before, `load_agent()` had no cache, no retry and
    no fallback, so an unreachable platform meant a generated agent could not
    boot at all — which made the documented promise "turn DecimalAI off and your
    agent still works" false for every agent `decimalai init` produces.
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
                "decimalai.load_agent(%r): platform unreachable (%s) — serving the "
                "cached prompt, %.0fs old. config.is_fallback is True.",
                agent_name, exc, age,
            )
            return replace(cached, is_fallback=True, stale_age_seconds=age)
        if fallback is not None:
            logger.warning(
                "decimalai.load_agent(%r): platform unreachable (%s) and nothing "
                "cached — using the fallback prompt. config.is_fallback is True.",
                agent_name, exc,
            )
            return AgentConfig(
                agent_name=str(agent_name),
                system_prompt=fallback,
                is_fallback=True,
            )
        raise

    # `get_agent_prompt` only returns None for a 304, which needs a conditional
    # request this call never makes.
    config = AgentConfig._from_payload(payload, requested_name=str(agent_name))
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
