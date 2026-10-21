"""Skill Router — SDK-side skill loading and prompt-fragment assembly.

The Router is the runtime piece that loads the right skills into your
prompt at the right moment. It talks to the DecimalAI platform's
`/api/v1/skills/...` endpoints and exposes:

- Full menu — every active skill, name+description only
- Smart routing — semantic search + performance reranking
- Progressive disclosure — menu → body → attachments
- Sync of local SKILL.md files to the platform

`build_prompt_fragment(query)` is the load-bearing primitive every
framework adapter calls — it returns `(prompt_fragment, routing_id)`,
where the `routing_id` must be threaded into the resulting trace so
the `routing_decision × trace_skill_activation` join can close.
"""

from __future__ import annotations

import hashlib
import json as json_module
import logging
import os
import re
import time
from collections import OrderedDict
from contextvars import ContextVar
from copy import deepcopy
from http.cookiejar import CookieJar, DefaultCookiePolicy
from threading import Lock
from threading import local as _thread_local
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlsplit

import httpx

logger = logging.getLogger("decimalai.skill_router")


# Skill Rater discovery telemetry. The Router knows which names
# it offered on each `build_prompt_fragment` call. Adapters read this
# contextvar after the call and stamp the names onto the active trace's
# `skills_offered_in_prompt` field. ContextVar (not threading.local) so
# asyncio tasks get isolated copies — same reasoning as the routing_id
# contextvars in each adapter.
_last_offered_names_ctx: ContextVar[Optional[List[str]]] = ContextVar(
    "decimalai_skill_router_last_offered_names", default=None,
)

# The prefix/tail split's two pre-rendered strings, handed from
# `build_prompt_fragment` (which does the fetch and the caching) up to
# `build_prompt_parts` (which is the public shape). A ContextVar rather than an
# instance attribute for the same reason as the rails above: one Router
# singleton serves concurrent runs, and an attribute would let one run read the
# other's prefix.
_last_split_parts_ctx: ContextVar[Optional[tuple]] = ContextVar(
    "decimalai_skill_router_last_split_parts", default=None,
)

# Whether the most recent `build_prompt_fragment` result was a menu the platform
# CUT to fit its caps: more skills are eligible than it lists (``truncated`` on the
# whole-menu answer, or a ``smart_routing`` answer, which is the over-the-caps
# path). An adapter that keeps a local copy of an agent's menu re-routes such an
# agent on every message, since which rows make the cut depends on the message.
_last_menu_truncated_ctx: ContextVar[Optional[bool]] = ContextVar(
    "decimalai_skill_router_last_menu_truncated", default=None,
)


def consume_last_menu_truncated() -> Optional[bool]:
    """Read + clear whether the last routed menu was cut to fit (None: unknown)."""
    value = _last_menu_truncated_ctx.get()
    _last_menu_truncated_ctx.set(None)
    return value


def consume_last_offered_names() -> List[str]:
    """Read + clear the names from the most recent `build_prompt_fragment`.

    Adapters call this immediately after invoking the Router so the offered
    set is captured against the current trace, not leaked into the next.
    Returns an empty list when nothing has been offered (no Router call
    yet, or the prior call had no skills).
    """
    names = _last_offered_names_ctx.get()
    if names:
        _last_offered_names_ctx.set(None)
        return list(names)
    return []


# 'delivered' = the full skill BODY reached the model (the
# inject_skill_body path). Distinct from 'offered' (menu row only) — a bare
# fragment injection without a body is offered-only, never delivered.
# Same contextvar mirror rail as offered names above.
_last_delivered_names_ctx: ContextVar[Optional[List[str]]] = ContextVar(
    "decimalai_skill_router_last_delivered_names", default=None,
)


def consume_last_delivered_names() -> List[str]:
    """Read + clear the names whose BODY the most recent
    `build_prompt_fragment` injected. Mirrors
    `consume_last_offered_names` — adapters drain both after the call.
    """
    names = _last_delivered_names_ctx.get()
    if names:
        _last_delivered_names_ctx.set(None)
        return list(names)
    return []


_last_delivered_versions_ctx: ContextVar[Optional[List[Dict[str, str]]]] = ContextVar(
    "decimalai_skill_router_last_delivered_versions", default=None,
)


def consume_last_delivered_versions() -> List[Dict[str, str]]:
    """Read + clear immutable identities of this call's injected skill bodies.

    These are delivery witnesses, never model-initiated activations. Each hash
    comes from the response that supplied that exact body, including on a
    fragment-cache hit; a later fetch must not replace the witnessed version.
    """
    versions = _last_delivered_versions_ctx.get()
    _last_delivered_versions_ctx.set(None)
    return [dict(version) for version in versions or []]


def _release_scoped_routing_rail(router: Any, scope: Optional[str]) -> None:
    """Give back the routing slot a scoped routing call filed under ``scope``.

    For the adapters that pass ``scope=`` to the router but read the decision
    back from the per-call rails above, never from the router: every
    ``agent_config()`` (ADK, Anthropic, OpenAI Agents, Pydantic AI and
    ``decimalai.agent_config()``, through ``_agent_binding``) and CrewAI. The router still files every scoped
    decision a second time under the run's scope, for the adapter that does read
    it back from there (LangChain), and holds it until a scoped ``consume_*``
    drains it. Left in place, each run keeps one of the router's
    ``_MAX_SCOPED_RAILS`` slots for the life of the process: from run 4,097 on,
    every new run evicts an older one and warns, and a LangChain run still in
    flight in the same process loses its decision to that churn. Until
    2026-10-08 only CrewAI released it.

    Call it once the routing call returns or raises — an empty route files a
    slot too. A router stand-in without the scoped drains has nothing to
    release.
    """
    if scope is None:
        return
    try:
        router.consume_routing_id(scope=scope)
        router.consume_offered_names(scope=scope)
        router.consume_delivered_names(scope=scope)
        versions = getattr(router, "consume_delivered_versions", None)
        if callable(versions):
            versions(scope=scope)
    except Exception:
        logger.debug("router keeps no scoped routing rail to release", exc_info=True)


def _stamp_active_trace(
    routing_id: Optional[str],
    offered_names: Optional[List[str]],
    delivered_names: Optional[List[str]],
    delivered_versions: Optional[List[Dict[str, str]]] = None,
) -> None:
    """Auto-stamp the active generic trace (`decimalai.start_trace`)
    with the routing decision — routing_id + offered/delivered names — so
    the raw-loop quickstart closes the offered-vs-activated join without
    manual `set_routing_id` / `log_skill_offered` calls. No-op without an
    active trace. Known limit: when an adapter-instrumented run is wrapped
    in a generic trace, the generic trace AND the adapter's own trace are
    distinct objects, so both carry the same routing telemetry (set-dedup
    only protects within one trace). Harmless today — ladder offered
    counts come from server-side RoutingDecision rows, not trace fields.
    Lazy import — generic.py imports nothing from this module at module
    level, so no cycle.
    """
    try:
        from .generic import _get_current_trace
        ctx = _get_current_trace()
        if ctx is None:
            return
        if routing_id:
            ctx.set_routing_id(routing_id)
        if offered_names:
            ctx.log_skill_offered(names=list(offered_names))
        if delivered_names:
            ctx.log_skill_delivered(names=list(delivered_names))
        if delivered_versions:
            ctx.log_skill_delivery_versions(versions=delivered_versions)
    except Exception:
        logger.debug("auto-stamp of active trace failed (non-fatal)", exc_info=True)


def estimate_tokens(text: str) -> int:
    """Cheap chars/4 token estimate — mirrors the backend's estimator.
    ±20% on English markdown is fine: the body budgets are soft
    context-hygiene caps, not hard limits."""
    return (len(text) + 3) // 4


def _body_is_known_partial(body: Any, record: Optional[Dict[str, Any]] = None) -> bool:
    """A partial body cannot witness delivery of its full immutable version.

    New servers declare truncation. Recognize the old terminal markers too,
    without mistaking a complete record's literal example for truncation.
    """
    record = record or {}
    if record.get("truncated") is True or record.get("content_hash_matches_body") is False:
        return True
    total = record.get("total_chars")
    if isinstance(body, str) and isinstance(total, int) and total > len(body):
        return True
    if record.get("content_hash_matches_body") is True:
        return False
    return isinstance(body, str) and bool(re.search(
        r"\n\n\[\.\.\. truncated (?:by the per-body limit|\d+ of \d+ chars"
        r" — request without max_chars for the full body)\]\s*$",
        body,
    ))


# ── Progressive disclosure: the load_skill tool ────────────────────────
# The description tier is always-on; bodies load on demand when the model
# calls load_skill(name). Every agent_config() (openai_agents, pydantic_ai, adk,
# anthropic) hands the model this tool, for the framework's or the SDK's own
# tool loop; adapters that patch a layer with no tool loop (langchain, crewai)
# can't route a tool result and stay on prompt injection.

LOAD_SKILL_TOOL_NAME = "load_skill"

LOAD_SKILL_TOOL_DESCRIPTION = (
    "Load the full instructions (body) of a skill listed under '## Recommended "
    "Skills' or '## Available Skills'. Call this BEFORE using a skill — the menu "
    "row is only a description; the body contains the actual instructions. "
    "Parallel calls for multiple skills are allowed. Budgeted: at most a few "
    "bodies per turn, so load only the skills you will actually use."
)

LOAD_SKILL_TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Exact skill name as shown in the skills menu.",
        },
    },
    "required": ["name"],
}


def load_skill_tool_spec() -> Dict[str, Any]:
    """Provider-agnostic tool spec adapters translate to their framework's
    tool type. Shape: ``{name, description, parameters}`` (JSON Schema)."""
    return {
        "name": LOAD_SKILL_TOOL_NAME,
        "description": LOAD_SKILL_TOOL_DESCRIPTION,
        "parameters": LOAD_SKILL_TOOL_PARAMETERS,
    }


# Appended to the prompt fragment by adapters when the load_skill tool is
# registered — the server-side fragment keeps the activation-statement
# instruction unchanged so evaluation stays on-distribution with production.
LOAD_SKILL_PROMPT_HINT = (
    "If a skill applies, call the load_skill tool with its exact name to read "
    "its full instructions before using it. Load each skill at most once, then "
    "answer the user directly using what you loaded."
)


# ── hot-path availability ───────────────────────────────────
#
# "Hot path" = a call that happens INSIDE a user's turn, so its latency is the
# customer's latency. Everything else (CLI, publish, scans) can afford to wait.
#
# The body fetch is one too (2026-10-14). `build_prompt_fragment` fetches the
# injected body and `load_skill` fetches a body mid-turn, and both used to get
# the 30 s cold budget with no breaker: a brownout that the 5 s route budget
# had already given up on still held the turn for up to 30 s per body. Disk
# export also reads `/body`, from the CLI, and opts out with `hot=False`.
_HOT_PATHS = ("/api/v1/skills/route", "/api/v1/skills/menu")
_SKILLS_PREFIX = "/api/v1/skills/"
_BODY_SUFFIX = "/body"
_ADMISSION_MAX_RETRIES = 2
_ADMISSION_MAX_WAIT_S = 5.0

# WHY 5 SECONDS AND NOT 2 (changed 2026-09-02, and the old value was measured
# doing real harm).
#
# Failing fast is right. The value was not. Shipped at read=2.0 in 0.12.0, and
# on 2026-09-01 the DecimalAI fleet — 93 synthetic users on the public SDK —
# went from delivering a skill body in 69.4% of sessions to 3.1%, in the same
# hour it restarted onto that release. `no_skill_offered` went from 7.9% of
# sessions to 77.6%. Nothing about the platform changed at that moment; the
# budget did.
#
# The platform cannot meet 2 s and never could. Measured on prod:
#   * healthy (backend at cpu=2, same fleet load): /skills/route p95 1.39 s,
#     max 1.71 s. A 2 s budget is 1.4x that — inside normal jitter.
#   * degraded (cpu=1, where it runs today): p95 7.35 s, and a live probe on
#     2026-09-02 returned 16.9 s / 29.8 s / 8.3 s.
# So 2 s failed on a bad day AND had no margin on a good one.
#
# 5 s is 3.6x the healthy p95, so ordinary variance stops costing the customer
# their skills, and it is still a small share of an agent turn (commonly
# 5-30 s) — which is the thing fail-fast exists to protect. The 30 s cold-path
# budget this replaced was measured at 30.3 s per turn; 5 s keeps ~6x of that
# improvement while actually completing.
#
# ⚠ THE FAILURE MODE THIS GUARDS IS SILENT. A timeout here does not raise —
# `smart_route` returns an empty menu and the agent answers without skills. The
# customer sees a worse answer and no error. That is why the budget must be
# generous enough to succeed, why exceeding it is now counted rather than
# shrugged at (`routing_status()`), and why the breaker opening logs at ERROR.
_DEFAULT_HOT_PATH_READ_S = 5.0
#: Env override, so an operator can tune without a release and a test can pin a
#: value. Deliberately NOT read by the fleet: the fleet runs the customer
#: default on purpose, because a budget only the fleet can meet tests nothing.
_HOT_PATH_READ_ENV = "DECIMALAI_SKILL_ROUTE_TIMEOUT_S"
#: Clamp. Below the floor the call cannot succeed even against a warm backend
#: (healthy p95 is 1.39 s); above the ceiling we are back to holding a turn open
#: for the length of a provider outage.
_HOT_PATH_READ_FLOOR_S = 0.5
_HOT_PATH_READ_CEILING_S = 30.0
_COLD_PATH_TIMEOUT = 30.0


def _hot_path_read_budget() -> float:
    """The read budget for a hot-path call, honouring the env override.

    Read per call rather than captured at import: a module-level constant is
    fixed by whatever the environment looked like when the first import
    happened, which in a test suite is whichever test ran first. The cost is a
    dict lookup and a float parse against an HTTP request.

    Garbage falls back to the default rather than raising — this sits inside a
    customer's turn, and a malformed env var must not be the thing that breaks
    their agent.
    """
    raw = os.environ.get(_HOT_PATH_READ_ENV)
    if not raw:
        return _DEFAULT_HOT_PATH_READ_S
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "%s=%r is not a number — using the %.1fs default.",
            _HOT_PATH_READ_ENV, raw, _DEFAULT_HOT_PATH_READ_S,
        )
        return _DEFAULT_HOT_PATH_READ_S
    if value <= 0:
        return _DEFAULT_HOT_PATH_READ_S
    return max(_HOT_PATH_READ_FLOOR_S, min(_HOT_PATH_READ_CEILING_S, value))


# WHY CONNECT GETS 2 SECONDS, NOT 1 (changed 2026-10-20).
#
# httpcore applies the connect budget to the TCP connection and again to the
# TLS handshake. A healthy handshake from the OpenAI tutorial's laptop
# (Indonesia, Cloudflare's Singapore edge) takes ~0.08 s, so 1 s looked
# generous. But the OS re-sends a lost SYN after ~1 s (the initial retransmit
# timeout), so ONE lost packet used the whole budget: on 2026-10-20 trace
# a3f2c0ab's load_skill failed in 1.21 s with `(timeout)` while the backend
# served other requests normally, and the model answered without its skill.
# 2 s survives a re-sent SYN with up to ~1 s of round trip, and the connect
# retry in `_request` gives a second, fresh connection on top.
#
# The cost is the unreachable case: two attempts x 2 s = 4 s per call (was
# 1 s), still under the 5 s a hung backend already costs, and the breaker
# opens after three such calls.
_HOT_PATH_CONNECT_S = 2.0
#: The read budget for the one retry of a hot read whose first attempt timed
#: out. `/body` and `/menu` answer in well under 1 s at the 95th percentile on
#: prod (1.01 s and 0.34 s on 2026-10-20), so a 5 s read timeout means that
#: request stalled, not that the platform is slow; a second try usually lands.
#: Short, so a stalled hot read costs a turn at most this much more.
_HOT_PATH_READ_RETRY_S = 2.0


def _hot_path_timeout(read_s: Optional[float] = None) -> httpx.Timeout:
    """connect/read/write/pool. Read is the one that matters: a hung backend
    must not hold a turn open. The caller degrades to an empty menu, which is
    correct — the agent answers without skills rather than not answering.
    ``read_s`` overrides the read budget for one call (the menu fallback's)."""
    return httpx.Timeout(
        connect=_HOT_PATH_CONNECT_S,
        read=_hot_path_read_budget() if read_s is None else read_s,
        write=2.0, pool=1.0,
    )


# ── Transport: one retry per failure kind, and a shared connection pool ───
#
# Before 2026-10-20 a request got exactly one attempt: any transport error was
# final. In a turn that is silent — `load_skill` tells the model the skill is
# unavailable and not to retry — so one dropped packet cost the answer its
# skill. `_request` now retries once when it is safe:
#
#   * The connection could not be opened (`_CONNECT_FAILURES`): nothing was
#     sent, so the platform never saw the request. Retried for every method.
#   * The connection dropped before an answer (`_DROPPED_FAILURES`, e.g. a
#     reset or a pooled connection the far end had closed): retried when the
#     request is safe to replay (a GET, the routing request, an endpoint
#     documented idempotent).
#   * The read timed out: retried for those same requests except the routing
#     request. A slow route is the platform's own latency, a retry would add to
#     the load that made it slow, and the menu fallback already covers it.
#
# Each kind is retried at most once per call, however many admission (429)
# retries the call also makes, and a call that fails both attempts counts as
# ONE failure for the breaker.
_CONNECT_FAILURES = (httpx.ConnectError, httpx.ConnectTimeout)
_DROPPED_FAILURES = (httpx.ReadError, httpx.RemoteProtocolError)
_ROUTE_PATH = "/api/v1/skills/route"

# One `httpx.Client` per process instead of a fresh one per call
# (`httpx.request` builds a client, an SSL context and a TCP+TLS connection
# every time). Successive calls in a turn — menu, then `load_skill` a model call
# later — reuse the connection, so they skip the handshake the 2026-10-20
# failure happened in, and the CPU of building an SSL context.
#
# Rebuilt in a forked child: a pool inherited across `fork` (a pre-fork server
# that imported a router) shares sockets with the parent. Idle connections
# expire after httpx's default 5 s, below common servers' keep-alive timeout
# (uvicorn's is 5 s), so a reused connection has rarely been closed under it;
# when one has, the request fails as a dropped connection, retried above.
_platform_http: Optional[httpx.Client] = None
_platform_http_pid: Optional[int] = None
_platform_http_lock = Lock()


def _platform_client() -> httpx.Client:
    global _platform_http, _platform_http_pid
    pid = os.getpid()
    with _platform_http_lock:
        if _platform_http is None or _platform_http_pid != pid:
            # Same defaults as `httpx.request` (verify, trust_env, no redirects),
            # and like it keeps no cookies: one pool serves every router and API
            # key in the process, so a stored cookie would ride along on another
            # key's requests.
            _platform_http = httpx.Client(
                cookies=CookieJar(policy=DefaultCookiePolicy(allowed_domains=[])),
            )
            _platform_http_pid = pid
        return _platform_http


def _send(method: str, url: str, **kwargs: Any) -> httpx.Response:
    """One HTTP attempt, through the shared pool.

    Test suites here and in other DecimalAI repos stand in for the platform by
    patching ``httpx.request``. Sending through the pool unconditionally would
    route around every such stub to a real network, so a patched
    ``httpx.request`` is used instead — the pre-pool behaviour, one fresh
    connection per call. Unpatched, ``httpx.request`` is ``httpx._api.request``.
    """
    library_request = getattr(getattr(httpx, "_api", None), "request", None)
    if httpx.request is not library_request:
        return httpx.request(method, url, **kwargs)
    return _platform_client().request(method, url, **kwargs)


# ── The menu fallback (2026-10-20) ────────────────────────────────────────
#
# A route that times out or errors used to leave the turn with NO skills: on
# 2026-10-18 the OpenAI tutorial's agent answered without its procedures after
# a route timeout, and nothing but the trace said so. The agent's own menu
# (`GET /skills/menu`) is the cheap endpoint — 0.08 s on prod while routes took
# 1.4-7.8 s — so after such a failure one request for it serves the agent's
# skills, unranked, which beats none.
#
#: Route failures after which that one request is worth making: the route was
#: slow (`timeout`, `http_504`) or failed in the handler (`http_500`). Not a 429
#: or 503 (the platform is shedding load), a transport error (it is unreachable)
#: or an open breaker: a second request then only adds to the load that made the
#: first one fail.
_MENU_FALLBACK_REASONS = frozenset({"timeout", "http_500", "http_504"})
#: The fallback's own read budget. It takes no admission retries either, so a
#: failed route costs a turn at most this much more.
_MENU_FALLBACK_READ_S = 2.0
#: After a fallback served a menu, routes from the same host and key are skipped
#: for this long and turns go straight to the menu. Otherwise every turn would
#: pay the route's timeout again: the menu's answer proves the host reachable,
#: so the breaker never opens on a route that alone is slow.
_ROUTE_PAUSE_S = 30.0
_route_pauses: Dict[Tuple[str, str], float] = {}
_route_pauses_lock = Lock()


def _route_pause_key(base_url: str, api_key: Optional[str]) -> Tuple[str, str]:
    return (_without_userinfo(base_url), _key_digest(api_key))


def _route_paused(base_url: str, api_key: Optional[str]) -> bool:
    with _route_pauses_lock:
        until = _route_pauses.get(_route_pause_key(base_url, api_key))
    return until is not None and time.monotonic() < until


def _pause_route(base_url: str, api_key: Optional[str]) -> None:
    now = time.monotonic()
    with _route_pauses_lock:
        for key in [k for k, until in _route_pauses.items() if until <= now]:
            del _route_pauses[key]
        _route_pauses[_route_pause_key(base_url, api_key)] = now + _ROUTE_PAUSE_S


def _clear_route_pauses() -> None:
    """Forget every pause. For tests, and for a process that wants routes back now."""
    with _route_pauses_lock:
        _route_pauses.clear()


def _empty_menu_reason(e: "SkillRouterError") -> str:
    """Why smart_route returned nothing, as one word the caller can count.

    The fleet filed every empty menu as `no_skill_offered` and could not tell a
    Cloud Run edge abort (a 0 s 429) from a slow read from an open breaker —
    on 2026-09-03..04, 30% of its router-calling sessions, 58% of them 429s, all
    with an empty detail. `http_<status>`, `timeout`, `circuit_open`, `transport`.
    """
    code = getattr(e, "status_code", None)
    if code:
        return f"http_{code}"
    if isinstance(e.__cause__, httpx.TimeoutException):
        return "timeout"
    if "circuit open" in str(e):
        return "circuit_open"
    return "transport"


def _is_body_path(path: str) -> bool:
    return path.startswith(_SKILLS_PREFIX) and path.endswith(_BODY_SUFFIX)


def _is_hot_path(path: str) -> bool:
    # Exact match for route/menu: a prefix test also caught every per-skill
    # path whose NAME starts with "route" or "menu" (`/skills/routes-planner/
    # publish` got the 5 s budget and tripped the breaker agents depend on).
    return path in _HOT_PATHS or _is_body_path(path)


_REGISTRY_SKILLS_PREFIX = "/api/v1/registry/skills/"


def _path_part(value: Any) -> str:
    """One caller-supplied value (a skill name, an id, a file path) as URL path text.

    Raw, a ``#`` or ``?`` ended the path: ``weird#name`` requested
    ``/skills/weird`` with a fragment, and a space or ``%`` was sent as
    whatever the transport made of it. ``/`` stays literal: the body route is
    ``{skill_name:path}`` for namespaced registry names (``owner/name``, about
    2/3 of the registry) and attachment paths are nested, while the other
    per-skill routes are one segment, where an encoded ``%2F`` is decoded back
    to ``/`` before routing and would not match either. Keeping it literal
    means every name that worked before produces the same bytes.
    """
    return quote(str(value), safe="/")


def _skill_path(ref: Any, suffix: str = "") -> str:
    """``/api/v1/skills/{ref}{suffix}`` with ``ref`` encoded (see ``_path_part``)."""
    return f"{_SKILLS_PREFIX}{_path_part(ref)}{suffix}"


def _registry_skill_path(ref: Any, suffix: str = "") -> str:
    """``/api/v1/registry/skills/{ref}{suffix}`` with ``ref`` encoded."""
    return f"{_REGISTRY_SKILLS_PREFIX}{_path_part(ref)}{suffix}"


def _agent_skill_path(agent_name: Any, skill_id: Any) -> str:
    """``/api/v1/agents/{agent_name}/skills/{skill_id}``, each value one encoded
    segment: the route takes a single segment for each, so unlike a skill name
    a ``/`` here is encoded rather than kept."""
    return f"/api/v1/agents/{quote(str(agent_name), safe='')}/skills/{quote(str(skill_id), safe='')}"


def _skill_body_path(name: str) -> str:
    """``/api/v1/skills/{name}/body`` with the name percent-encoded."""
    return _skill_path(name, _BODY_SUFFIX)


def _is_health_failure(path: str, status: int) -> bool:
    """Does this HTTP status mean agents are running without skills?

    Until 2026-10-14 the breaker called ANY response a success, so a revoked
    key (401) or a 503 storm left ``routing_status().healthy`` True while every
    turn ran with an empty menu. Auth, plan gate (402), admission (429) and
    server errors are failures. A 404 from ``/body`` is the platform answering
    "no skill by that name" — an answer, not an outage — while a 404 from
    ``/route`` or ``/menu`` means the endpoint itself is missing (wrong base URL,
    a proxy eating the path). Other 4xx are about one request, not the platform.
    """
    if status >= 500 or status in (401, 402, 403, 429):
        return True
    return status == 404 and not _is_body_path(path)


def _routing_call_failed(result: Dict[str, Any]) -> bool:
    """Is this routing result the SDK's own empty stand-in for a failed call?

    ``get_menu`` / ``smart_route`` swallow their errors into
    ``{"skills": [], "prompt_fragment": "", "strategy": "none", ...}``. The
    platform never sends ``strategy: "none"``; it answers ``full_menu`` or
    ``smart_routing``, even for a workspace with no skills.
    """
    return result.get("strategy") == "none" and not result.get("routing_id")


def _ranked_by_this_query(result: Dict[str, Any]) -> bool:
    """Did the platform rank this result by the query it was asked?

    Only then is its top row a body worth injecting. The platform answers a
    query with its effectiveness-ordered menu when the workspace is small
    enough to show whole and on every retrieval fallback (``strategy:
    "full_menu"``). When no query term matched a row, that order ignores the
    query, and injecting its top row put the workspace's best-rated skill, up
    to 6k tokens of it, into turns like "thanks!" (audit R2).

    ``query_ranked`` is the platform's answer, and it decides whenever it is
    sent. ``degraded`` does NOT: it says how the result was retrieved (embedder
    down, a retrieval leg failed), not whether its order is about the query.
    A degraded result can be either kind — lexical-only hybrid retrieval, in
    which every row matched a query term, or a fallback menu whose top row a
    term matched, are both ranked (injecting there is how the SDK behaved
    before embeddings existed); a fallback menu nothing matched is not.

    Without ``query_ranked`` (a platform that predates it on hybrid results):
    ``full_menu`` is unranked, and hybrid retrieval (``smart_routing``) ranks
    by the query by construction, degraded or not.
    """
    ranked = result.get("query_ranked")
    if ranked is not None:
        return ranked is True
    return result.get("strategy") != "full_menu"


#: A failed routing call that took at least this long is cached for the turn.
#: Edge aborts, auth and plan refusals and most 5xx come back in milliseconds;
#: a timeout or an admission wait costs seconds.
_SLOW_FAILURE_S = 1.0

#: How long a slow failure is reused for a caller that names no run (no
#: ``scope=``): LangChain, whose ambient run id names each model call rather
#: than the turn, and a hand-written loop. For them the cache slot is one per
#: (query, agent) for the whole process, so "the next model call of this turn"
#: and "another conversation asking the same thing" are the same lookup, and
#: the router cannot tell them apart. When the failure was kept for the 30 s a
#: success is, one timeout during a brownout left every conversation sending
#: that message (a chat UI's suggested-prompt button) without skills for half
#: a minute.
#:
#: 5 s covers the gap the cache exists for: the model call that follows the
#: failed route (it is the call that route was for, plus any tool it runs). It
#: is also the breaker's first pause, and does not grow with a raised read
#: budget: the gap between a turn's model calls does not either. A turn that
#: outlasts it retries once per window, which also lets a recovered platform
#: back in; three failures in a row open the breaker. The price: a
#: conversation that asks the same thing within those 5 s is served the
#: failure too. Only a run id closes that, and a caller that names its run
#: keeps the failure for the turn (the cache TTL) where no other run sees it.
_UNSCOPED_FAILURE_TTL_S = 5.0


def _failure_worth_caching(result: Dict[str, Any], elapsed_s: float) -> bool:
    """Should this failed routing call be cached for the rest of the turn?

    Only when retrying it would cost the turn time again: a timeout (the 5 s
    read budget, every time) or any failure that took ``_SLOW_FAILURE_S`` or
    more (a 429 whose admission retries waited out their ``Retry-After``).

    Not a fast failure — an edge 429 with no ``Retry-After``, a 503, a 401 —
    which costs nothing to repeat and is usually gone on the next call. A run
    of them is the circuit breaker's job: it counts them, and three in a row
    open it. Not an open breaker either: that skipped the call, so it cost
    nothing, and caching it would keep the run skill-less after it closes.
    """
    reason = result.get("degraded_reason")
    if reason == "circuit_open":
        return False
    return reason == "timeout" or elapsed_s >= _SLOW_FAILURE_S


def _bodies_without_ranking(
    result: Dict[str, Any], routed_names: List[str], prioritized: List[str],
) -> List[str]:
    """The bodies worth injecting from a result NOT ranked by this query.

    The top row of such a result is the workspace's best-rated skill, not the
    one this turn is about, so it is not injected on that basis (audit R2).
    Two bodies are the right ones regardless of the query, because there is no
    other to pick:

    * ``prioritized`` — the developer's ``priority_skills`` the platform
      offered this agent: shared guidance (reply format, house style) that
      applies to every reply. A name it did not offer is never here.
    * the agent's ONLY skill — a ``full_menu`` whose ``rows_total`` is 1 is the
      platform saying this menu is the agent's whole eligible set, and it is
      one skill. Any turn that needs a body needs this one. The SDK cannot
      tell "thanks!" from a reworded question, so small talk gets it too.

    A single row that is a SLICE of a larger set (a capped fallback menu, a
    top-k retrieval, a platform that does not say how big the set is) is not
    the agent's only skill, and stays menu-only.
    """
    names = list(prioritized)
    only = (
        result.get("strategy") == "full_menu"
        and result.get("rows_total") == 1
        and len(routed_names) == 1
    )
    if only and routed_names[0] not in names:
        names.append(routed_names[0])
    return names


class _RoutingTotals:
    """The process-wide counters behind ``routing_status()``.

    Monotonic for the life of the process: a health check wants "has this ever
    degraded", not a gauge that a recovery quietly resets to zero. They live
    apart from the breakers because there is one breaker per host and per API
    key, in a bounded store (see ``_HotPathBreakers``): dropping the least
    recently used breaker must not make a total go down.

    ``timeouts`` counts transport failures (a read timeout, a refused
    connection); ``http_errors`` counts responses that arrived but mean no
    skills (see ``_is_health_failure``); ``menu_fallbacks`` counts turns that
    got the agent's menu, unranked, in place of a failed or paused route.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self.timeouts = 0
        self.http_errors = 0
        self.opens = 0
        self.suppressed = 0
        self.menu_fallbacks = 0
        self.last_error: Optional[str] = None
        self.last_error_at: Optional[float] = None
        self.last_status_code: Optional[int] = None
        self.last_success_at: Optional[float] = None

    def failure(self, error_text: Optional[str], status_code: Optional[int]) -> None:
        with self._lock:
            if status_code is None:
                self.timeouts += 1
            else:
                self.http_errors += 1
                self.last_status_code = status_code
            if error_text is not None:
                self.last_error = error_text
                self.last_error_at = time.time()

    def opened(self) -> None:
        with self._lock:
            self.opens += 1

    def skipped(self) -> None:
        with self._lock:
            self.suppressed += 1

    def menu_fallback(self) -> None:
        with self._lock:
            self.menu_fallbacks += 1

    def success(self) -> None:
        with self._lock:
            self.last_success_at = time.time()


def _status_hint(status_code: Optional[int]) -> str:
    """What an operator can do about a key whose responses opened its breaker."""
    if status_code == 401:
        return "Check that this API key is valid and has not been revoked."
    if status_code == 403:
        return "This API key may not route skills here; check its organization and scopes."
    if status_code == 402:
        return "This key's plan does not cover the request; check its billing."
    if status_code == 429:
        return "This key is rate limited or over its quota; routing resumes when the limit resets."
    if status_code == 404:
        return "The routing endpoint was not found at this base URL; check it."
    if status_code is not None and status_code >= 500:
        return "The platform answered this key's requests with a server error."
    if status_code is not None and status_code < 400:
        return "The response was not JSON (a proxy or login page?); check the base URL."
    return ""


class _CircuitBreaker:
    """Stop paying the timeout once the platform is clearly unreachable.

    Failing open is the right call for skill routing, but before this the SDK
    failed open only AFTER the full timeout, and re-attempted the same dead
    endpoint on the next turn, and the one after that. With a short timeout the
    per-turn cost is small; with a breaker it goes to zero for a cooldown window
    and the agent stops making a doomed call on every request it serves.

    THE COOLDOWN BACKS OFF (2026-09-02). It was a flat 30 s, which meant three
    slow calls in a row bought a full half-minute of guaranteed skill-less
    answers — on a platform whose latency is bursty, so a single slow minute
    suppressed routing through the next one. It now starts at 5 s and doubles
    per consecutive open to a 30 s ceiling: a blip costs 5 s, a real outage
    still backs all the way off, and one success resets the ladder.

    ONE BREAKER PER CAUSE (2026-10-14). A breaker pauses only the routing its
    failures are about: ``kind="transport"`` for a host that did not answer,
    ``kind="credential"`` for one API key the platform answered with a status
    that leaves its turns without skills. ``_HotPathBreakers`` holds them.
    """

    def __init__(
        self,
        threshold: int = 3,
        cooldown_s: float = 5.0,
        max_cooldown_s: float = 30.0,
        *,
        kind: str = "transport",
        target: str = "",
        totals: Optional[_RoutingTotals] = None,
    ):
        self.threshold = threshold
        self.base_cooldown_s = cooldown_s
        self.max_cooldown_s = max_cooldown_s
        # What a failure here is about, for the ERROR line: the host at
        # `target` did not answer ("transport"), or `target` answered one API
        # key with a failure ("credential").
        self.kind = kind
        self.target = target
        self.totals = totals if totals is not None else _RoutingTotals()
        self._consecutive_failures = 0
        self._consecutive_opens = 0
        self._opened_at: Optional[float] = None
        self._lock = Lock()
        self.last_error: Optional[str] = None
        self.last_status_code: Optional[int] = None
        # When a call last consulted or fed this breaker (monotonic). Lets
        # `routing_status()` stop counting one the process no longer routes with.
        self.last_used_at = time.monotonic()

    def _current_cooldown(self) -> float:
        """Caller holds the lock."""
        step = max(0, self._consecutive_opens - 1)
        return min(self.max_cooldown_s, self.base_cooldown_s * (2 ** step))

    def is_open(self) -> bool:
        with self._lock:
            self.last_used_at = time.monotonic()
            if self._opened_at is None:
                return False
            if time.monotonic() - self._opened_at >= self._current_cooldown():
                # Half-open: let one call through to test the water. The open
                # COUNT is deliberately not reset here — only a success clears
                # it, so a platform that fails every probe keeps backing off
                # instead of retrying every 5 s forever.
                self._opened_at = None
                self._consecutive_failures = 0
                return False
            self.totals.skipped()
            return True

    def record_failure(
        self,
        error: Optional[BaseException] = None,
        *,
        status_code: Optional[int] = None,
    ) -> None:
        """Count one failed hot-path call.

        ``status_code`` set means the platform answered with a status that
        leaves agents without skills (401/403/429/5xx…); unset means the call
        never got an answer (timeout, refused connection).
        """
        with self._lock:
            self.last_used_at = time.monotonic()
            self._consecutive_failures += 1
            if status_code is not None:
                self.last_status_code = status_code
            error_text = f"{type(error).__name__}: {error}" if error is not None else None
            if error_text is not None:
                self.last_error = error_text
            self.totals.failure(error_text, status_code)
            if self._consecutive_failures >= self.threshold and self._opened_at is None:
                self._opened_at = time.monotonic()
                self._consecutive_opens += 1
                self.totals.opened()
                # ERROR, not warning. While this is open every agent routing
                # through it answers without skills and nothing else says so.
                logger.error("%s", self._open_message())

    def _open_message(self) -> str:
        """The ERROR line for an open, naming what can be done. Caller holds the lock.

        Never the key: a credential breaker is named by its base URL alone.
        """
        n, cooldown = self._consecutive_failures, self._current_cooldown()
        if self.kind == "credential":
            return (
                f"SkillRouter: {n} consecutive failed responses from {self.target} "
                f"for one API key — pausing skill routing with that key for "
                f"{cooldown:.0f}s; other keys are not paused. Its agents will run "
                f"WITHOUT SKILLS meanwhile. Last error: {self.last_error}. "
                f"{_status_hint(self.last_status_code)}"
            ).rstrip()
        where = f" to {self.target}" if self.target else ""
        return (
            f"SkillRouter: {n} consecutive failures reaching the platform{where} — "
            f"pausing skill routing{where} for {cooldown:.0f}s, for every API key. "
            f"Agents will run WITHOUT SKILLS until it answers. Last error: "
            f"{self.last_error}. If this is slowness rather than an outage, raise "
            f"{_HOT_PATH_READ_ENV} (currently {_hot_path_read_budget():.1f}s)."
        )

    def record_reachable(self) -> None:
        """The host answered, whatever it said: its transport failures are over.

        Not a routing success — the answer may be a 401 — so the totals' last
        success is left alone.
        """
        with self._lock:
            self.last_used_at = time.monotonic()
            self._consecutive_failures = 0
            self._consecutive_opens = 0
            self._opened_at = None

    def record_success(self) -> None:
        self.record_reachable()
        self.totals.success()


#: A breaker no call has consulted for this long, while the process kept routing
#: through others, stops counting toward `routing_status()`: the old key after
#: `init(api_key=...)` replaced it, a base URL that was corrected. Long enough
#: that a tenant with traffic every few seconds is never dropped between turns.
_BREAKER_SUPERSEDED_AFTER_S = 60.0


def _without_userinfo(url: str) -> str:
    """``url`` minus any ``user:password@``, so it can be logged."""
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    return parts._replace(netloc=parts.netloc.rsplit("@", 1)[1]).geturl()


def _host_of(base_url: str) -> str:
    """``scheme://host[:port]`` of a base URL: what a transport failure is about."""
    parts = urlsplit(_without_userinfo(base_url))
    if parts.scheme and parts.netloc:
        return f"{parts.scheme}://{parts.netloc}".lower()
    return _without_userinfo(base_url)


def _key_digest(api_key: Optional[str]) -> str:
    """Names a key in the breaker store without holding the key there."""
    return hashlib.sha256((api_key or "").encode("utf-8")).hexdigest()[:16]


class _HotPathBreakers:
    """Every hot-path circuit breaker in the process, one per cause.

    It was ONE breaker for the whole process. That was right while only
    transport failures fed it, and wrong from the moment 401/402/403/429/5xx
    did (Phase 1, 2026-10-14, so that a revoked key stops reading as healthy):
    three 401s from one router's revoked key, or a 429 from one tenant's quota,
    paused skill routing for every other router in the process, healthy keys
    included, and the fix-the-key path waited out the cooldown.

    Two kinds now:

    * per HOST (``for_host``), fed by transport failures — no answer at all.
      Shared by every key on that host, because the host not answering is not
      something any key changes, and sharing it is what keeps N routers from
      each paying the read timeout during an outage. Any answer, even a 401,
      proves the host reachable and resets it. Another host is not paused.
    * per (base URL, API key) (``for_credential``), fed by responses that
      leave a turn without skills. Two routers holding one key share it (a key
      that is revoked is revoked everywhere); no other key is paused by it.

    Bounded LRU: a process that serves many tenants' keys does not grow without
    limit. Dropping a breaker costs at most ``threshold`` more fast failures for
    that key, and the totals live in ``totals``, not in the breakers.
    """

    def __init__(
        self,
        threshold: int = 3,
        cooldown_s: float = 5.0,
        max_cooldown_s: float = 30.0,
        *,
        max_breakers: int = 512,
    ):
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.max_cooldown_s = max_cooldown_s
        self.max_breakers = max(2, int(max_breakers))
        self.totals = _RoutingTotals()
        self._breakers: "OrderedDict[Tuple[str, ...], _CircuitBreaker]" = OrderedDict()
        self._lock = Lock()

    def for_host(self, base_url: str) -> _CircuitBreaker:
        host = _host_of(base_url)
        return self._get(("transport", host), "transport", host)

    def for_credential(self, base_url: str, api_key: Optional[str]) -> _CircuitBreaker:
        return self._get(
            ("credential", base_url, _key_digest(api_key)), "credential",
            _without_userinfo(base_url),
        )

    def _get(self, key: Tuple[str, ...], kind: str, target: str) -> _CircuitBreaker:
        with self._lock:
            breaker = self._breakers.get(key)
            if breaker is not None:
                self._breakers.move_to_end(key)
                return breaker
            breaker = _CircuitBreaker(
                self.threshold, self.cooldown_s, self.max_cooldown_s,
                kind=kind, target=target, totals=self.totals,
            )
            self._breakers[key] = breaker
            while len(self._breakers) > self.max_breakers:
                self._breakers.popitem(last=False)
            return breaker

    def status(self) -> "RoutingStatus":
        with self._lock:
            breakers = list(self._breakers.values())
        states = []
        for b in breakers:
            with b._lock:
                states.append((b._opened_at is not None, b._consecutive_failures, b.last_used_at))
        # A breaker counts unless the process has gone on routing through
        # others for `_BREAKER_SUPERSEDED_AFTER_S` without consulting it.
        # Measured against the latest use, not the clock, so an idle process
        # keeps reporting what its last calls found.
        latest = max((used for _, _, used in states), default=0.0)
        current = [
            (is_open, failures) for is_open, failures, used in states
            if latest - used <= _BREAKER_SUPERSEDED_AFTER_S
        ]
        open_now = any(is_open for is_open, _ in current)
        consecutive = max((failures for _, failures in current), default=0)
        t = self.totals
        with t._lock:
            return RoutingStatus(
                healthy=not open_now and consecutive == 0,
                breaker_open=open_now,
                consecutive_failures=consecutive,
                timeouts=t.timeouts,
                http_errors=t.http_errors,
                opens=t.opens,
                suppressed=t.suppressed,
                menu_fallbacks=t.menu_fallbacks,
                read_budget_s=_hot_path_read_budget(),
                last_error=t.last_error,
                last_error_at=t.last_error_at,
                last_status_code=t.last_status_code,
                last_success_at=t.last_success_at,
            )


_hot_path_breakers = _HotPathBreakers()


class RoutingStatus:
    """Snapshot of skill routing's health. See ``decimalai.routing_status()``.

    Exists because the failure this reports is otherwise invisible: a hot-path
    timeout degrades to an empty menu, so the agent answers without skills and
    returns 200. On 2026-09-01 that state ran for 21 hours across 93 agents
    with every other health signal green.

    ``menu_fallbacks`` counts turns that got the agent's menu, unranked, because
    the route failed or was paused after failing (see ``_MENU_FALLBACK_REASONS``).
    Those turns had skills, so they do not make ``healthy`` False; a count that
    keeps rising says routing itself is not working.

    Example::

        st = decimalai.routing_status()
        if not st.healthy:
            alert_oncall(f"agents running without skills: {st.last_error!r}")
    """

    __slots__ = (
        "healthy", "breaker_open", "consecutive_failures", "timeouts",
        "http_errors", "opens", "suppressed", "read_budget_s", "last_error",
        "last_error_at", "last_status_code", "last_success_at", "menu_fallbacks",
    )

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"RoutingStatus(healthy={self.healthy}, breaker_open={self.breaker_open}, "
            f"timeouts={self.timeouts}, http_errors={self.http_errors}, opens={self.opens}, "
            f"read_budget_s={self.read_budget_s})"
        )


def routing_status() -> RoutingStatus:
    """Is skill routing actually working in this process?

    `healthy` is False while any breaker is open OR after any hot-path failure
    that has not been followed by a success for the same cause — a timeout
    reaching a host, or a response such as 401, 403, 429 or 5xx for one API
    key — because each leaves turns without skills. It answers "are my agents
    getting their skills right now", which is the question a 200 from the
    agent cannot. A 404 for an unknown skill name is not a failure.

    Every host and key the process routes with counts, so a process with one
    failing key reads unhealthy while another key succeeds: that key's agents
    are the ones running without skills. One the process stopped using (no call
    for `_BREAKER_SUPERSEDED_AFTER_S` while others were used) stops counting;
    the totals (`timeouts`, `http_errors`, `opens`, `suppressed`) keep its
    failures. `consecutive_failures` is the longest current run of any one
    breaker, i.e. how close the nearest one is to opening.
    """
    return _hot_path_breakers.status()


class _BodyLoadBudget:
    """Per-turn accounting for on-demand body loads (progressive-disclosure guardrail).

    Bounds what load_skill can add to one turn's context:
    ``max_bodies`` distinct bodies and ``token_budget`` estimated tokens,
    within ``deadline_s`` of the first load (extra round-trips must not
    stall a turn indefinitely — there is no per-turn SLA elsewhere).
    Re-loading an already-loaded skill costs nothing (dedup, LRU-refreshed).
    In tool mode a body that entered context cannot be evicted, so on
    budget exhaustion the tool REFUSES with an explanatory message rather
    than silently dropping content.
    """

    def __init__(self, max_bodies: int, token_budget: int, deadline_s: float):
        self.max_bodies = max_bodies
        self.token_budget = token_budget
        self.deadline_s = deadline_s
        self.loaded: "OrderedDict[str, int]" = OrderedDict()  # name → est tokens
        self.tokens_used = 0
        self._first_load_at: Optional[float] = None

    def check(self, name: str) -> Optional[str]:
        """None if the load may proceed, else a message to return to the model instead.

        A repeat load still costs no budget, but it no longer returns the body a
        second time. Handing back the identical text with no signal invited an
        unbounded loop: observed 4/4 runs of the shipped openai-agents scaffold
        re-loading the same skill on turns 3..10 and dying with MaxTurnsExceeded
        without ever producing an answer. Because the repeat was free, nothing
        bounded the loop — at max_turns=40 it burned all 40. The refusal path
        below only fires on DISTINCT skills, so it never saw this at all.
        """
        if name in self.loaded:
            self.loaded.move_to_end(name)  # LRU refresh, unchanged
            return (
                f"'{name}' is already loaded in this turn's context — its full "
                "instructions are above. Do not call load_skill for it again. "
                "Answer the user now with what you have."
            )
        if self._first_load_at is not None and (
            time.monotonic() - self._first_load_at > self.deadline_s
        ):
            return (
                f"load_skill budget exhausted: the {self.deadline_s:.0f}s per-turn "
                "body-load deadline has passed. Proceed with what is already loaded."
            )
        if len(self.loaded) >= self.max_bodies:
            return (
                f"load_skill budget exhausted: {self.max_bodies} bodies already "
                "loaded this turn "
                f"({', '.join(self.loaded)}). Proceed with what is already loaded."
            )
        return None

    def would_exceed(self, tokens: int) -> bool:
        return self.tokens_used + tokens > self.token_budget

    def record(self, name: str, tokens: int) -> None:
        if self._first_load_at is None:
            self._first_load_at = time.monotonic()
        # Re-load of an already-loaded name replaces its charge (dedup must
        # not double-count the same body against the budget).
        if name in self.loaded:
            self.tokens_used -= self.loaded[name]
        self.loaded[name] = tokens
        self.loaded.move_to_end(name)  # LRU order — most recent last
        self.tokens_used += tokens


# ContextVar (not instance state): adapters share one router singleton, so
# concurrent agent runs must not share a turn budget. build_prompt_fragment
# resets it at each prompt build (= turn start); load_skill lazily creates
# one if the tool fires without a prior fragment build in this context.
#: Handoff slot for the `content_hash` of the body `get_skill_body` just
#: fetched, read by `load_skill` on the SAME thread one line later.
#:
#: Thread-local and not a ContextVar, deliberately: a ContextVar set inside a
#: framework's tool executor may be written into a COPIED context that never
#: propagates back to the reader (the reality documented all over this module's
#: rails), while `get_skill_body` and `load_skill` are plain synchronous calls on
#: one thread and a thread-local always survives between them. Not state — it is
#: only ever read on the line after it is written, and is cleared before the
#: write it describes.
_body_hash_tls = _thread_local()


def _served_version_identity(record: Dict[str, Any]) -> Dict[str, str]:
    """Copy an exact server-served UUID pair, never reconstruct one from a name."""
    from uuid import UUID

    pair = {}
    for key in ("skill_id", "version_id"):
        value = record.get(key)
        if not isinstance(value, str):
            return {}
        try:
            if str(UUID(value)) != value:
                return {}
        except ValueError:
            return {}
        pair[key] = value
    return pair

_body_budget_ctx: ContextVar[Optional[_BodyLoadBudget]] = ContextVar(
    "decimalai_skill_router_body_budget", default=None,
)

#: The routing decision one load_skill call names, when its caller knows it better
#: than the run's latest: a later message of a conversation, loading a skill an
#: earlier message showed. Set around that one call, so loads cannot change
#: each other's (see `load_skill`).
_load_routing_id_ctx: ContextVar[Optional[str]] = ContextVar(
    "decimalai_skill_router_load_routing_id", default=None,
)


# ── ambient run scope ───────────────────────────────────────
#
# An adapter that OWNS the load_skill tool names its run at the call site —
# `pydantic_ai` passes `scope=` straight into `load_skill`/`consume_loaded_names`,
# and that stays the most precise answer available. LangChain cannot: it
# registers no native load_skill tool, so the call comes from arbitrary
# user-supplied tool code with no `scope=` to pass, and the trace-send path
# runs under `copy_context()` where a ContextVar the tool wrote is invisible.
#
# A resolver closes that gap without the Router importing any framework: the
# adapter registers a callable that answers "which run is executing right
# now?" using the framework's OWN run identity (LangChain reads the run id off
# `var_child_runnable_config`), and the Router files its per-run rails under
# that key. Every resolver must be cheap, must never raise, and must answer
# None when no run of its framework is on the stack — None simply means "no
# scoped rail", which is exactly today's behaviour.
_ambient_scope_resolvers: List[Any] = []
_ambient_scope_lock = Lock()


def register_ambient_scope_resolver(resolver: Any) -> None:
    """Register a callable that names the run currently executing, or None.

    Idempotent per callable, so an adapter can call it from module import
    and from `instrument()` without stacking duplicates.
    """
    if not callable(resolver):
        return
    with _ambient_scope_lock:
        if resolver not in _ambient_scope_resolvers:
            _ambient_scope_resolvers.append(resolver)


def _ambient_scope() -> Optional[str]:
    """The scope key of the run executing right now, or None.

    First non-None answer wins. A resolver that raises is skipped rather
    than allowed to break a routing call or a body load — rail bookkeeping
    must never be able to fail a user's model call.
    """
    # Read without the lock: this runs on every routing call and every body
    # load, registration happens once at adapter import, and a list append is
    # atomic — the worst a race can do is miss a resolver that is being
    # registered at this instant, which reads as "no scope" for one call.
    for resolver in _ambient_scope_resolvers:
        try:
            scope = resolver()
        except Exception:  # pragma: no cover - defensive
            logger.debug("ambient scope resolver failed (non-fatal)", exc_info=True)
            continue
        if isinstance(scope, str) and scope:
            return scope
    return None


# In-process LRU cache for `build_prompt_fragment` results.
# Bounded size so a chat session running many distinct queries can't
# blow memory; TTL short so a freshly-published skill shows up in the
# next turn without an explicit refresh.
class _FragmentCache:
    """Tiny thread-safe TTL+LRU cache.

    Why we don't pull cachetools: avoids a new SDK dependency for ~30
    lines of logic. The semantics mirror `cachetools.TTLCache` with
    LRU eviction; if we ever need richer cache behavior we can swap to
    that library trivially.
    """

    def __init__(self, maxsize: int = 64, ttl_seconds: float = 30.0):
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._data: "OrderedDict[Any, Tuple[Any, float]]" = OrderedDict()
        self._lock = Lock()

    def get(self, key: Any) -> Optional[Any]:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            value, expire_ts = entry
            if expire_ts < time.time():
                # Expired — evict; treat as miss so the caller refetches.
                self._data.pop(key, None)
                return None
            # LRU bump.
            self._data.move_to_end(key)
            return value

    def set(self, key: Any, value: Any, *, ttl_seconds: Optional[float] = None) -> None:
        """Store ``value``; ``ttl_seconds`` shortens (never lengthens) this entry's life."""
        ttl = self._ttl if ttl_seconds is None else min(self._ttl, ttl_seconds)
        with self._lock:
            expire_ts = time.time() + ttl
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (value, expire_ts)
            while len(self._data) > self._maxsize:
                # Evict oldest. ``last=False`` pops the LRU end.
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


# ── Disk-runtime detection ─────────────────────────────────
# The Router injects skills into the system prompt via the platform.
# Some runtimes (Claude Code, Cursor) ALSO inject skills by auto-discovering
# SKILL.md files from disk — they're outside the SDK's control. When both
# fire, the same skill appears twice in the prompt. There's no clean way
# to dedup at our layer (we don't see the runtime's injection), so we warn
# at install time when the environment looks like one of those runtimes.

_DISK_RUNTIME_ENV_VARS = {
    # Claude Code sets CLAUDECODE=1 when its shell launches.
    "CLAUDECODE": "claude-code",
    # Alternative seen in some Claude Code builds.
    "CLAUDE_CODE_ENTRYPOINT": "claude-code",
    # Cursor's agent terminal sets this when the editor wraps a process.
    "CURSOR_AGENT": "cursor",
}

_disk_runtime_warned = False


def _detect_disk_runtime() -> Optional[str]:
    """Return a short runtime identifier if we appear to be running inside
    a disk-loading runtime (Claude Code, Cursor, etc.), else None.

    Heuristic — checks well-known environment variables. Returns the first
    match; order isn't load-bearing because a process is realistically inside
    one runtime at a time.
    """
    for env_var, label in _DISK_RUNTIME_ENV_VARS.items():
        if os.environ.get(env_var):
            return label
    return None


def _warn_if_disk_runtime_detected(framework: str) -> None:
    """Log a one-shot warning when the Router loader is enabled inside a
    runtime that auto-loads SKILL.md from disk.

    No-op when not in a disk runtime, or after the first call (so a process
    using the SDK across many threads / agents doesn't log the same line
    repeatedly).
    """
    global _disk_runtime_warned
    if _disk_runtime_warned:
        return
    runtime = _detect_disk_runtime()
    if not runtime:
        return
    _disk_runtime_warned = True
    logger.warning(
        "SkillRouter loader (decimalai.%s) enabled inside %s, which "
        "loads SKILL.md files from disk itself: a skill that is both routed and "
        "on disk reaches the prompt twice. The SDK never writes skills to disk "
        "on its own; registry skills get there only through `decimalai skills add` "
        "and `decimalai skills install`. To avoid the duplicate, either keep the "
        "router as the only source (do not also install the routed skills, and "
        "remove local copies %s loads), or skip enable_skill_loader=True here and "
        "let %s load the skills you install with `decimalai skills add`. Set "
        "DECIMALAI_SUPPRESS_DISK_RUNTIME_WARNING=1 to silence intentionally.",
        framework, runtime, runtime, runtime,
    )


# Support a one-shot environment override so users who've consciously chosen
# the duplicate-injection setup (some testing scenarios) can silence the
# warning without changing code. Read once at import time; the bool above
# also guards re-firing.
if os.environ.get("DECIMALAI_SUPPRESS_DISK_RUNTIME_WARNING"):
    _disk_runtime_warned = True


# ── The disk verbs are the package manager's ───────────────
# SkillRouter's disk methods (export, update_skills, status) are thin wrappers
# over `decimalai.skillpkg`, the engine the `decimalai skills` commands run too,
# so there is one way a registry skill reaches disk: a manifest entry, a folder
# under `.agents/skills` and a lock entry pinning its bytes.

#: Change actions that wrote a folder.
_FOLDER_WRITES = ("add", "update", "fetch", "restore")


def _disk_project(project_root: Optional[str]) -> Any:
    """The project the engine will use for ``project_root`` (found from the working
    directory when it is None), for reading back what it wrote."""
    from .skillpkg.project import Project

    return Project.find() if project_root is None else Project.at(project_root)


def _locked_entries(project: Any) -> Dict[str, Any]:
    """The project's version 2 lock entries; none when there is no readable lock."""
    try:
        from .skillpkg import lock as _lock

        return dict(_lock.load(project, missing_ok=True, repair=True).skills)
    except Exception:
        return {}


def _refusal(result: Any) -> str:
    """Why an engine call failed: its errors, each naming the skill and the fix."""
    from .skillpkg.errors import SkillpkgError

    errors = [str(p) for p in result.problems if isinstance(p, SkillpkgError)]
    return "; ".join(errors) or f"exit code {int(result.exit_code)}"


def _written(result: Any, project: Any, locked: Dict[str, Any]) -> Dict[str, Any]:
    """The summary keys :meth:`SkillRouter.export` returns, for one engine result."""
    paths: List[str] = []
    attachments = 0
    for change in result.changes:
        if getattr(change, "action", None) not in _FOLDER_WRITES:
            continue
        paths.append(str(project.skills_root / change.install_name / "SKILL.md"))
        files = (locked.get(change.install_name) or {}).get("files") or {}
        attachments += max(len(files) - 1, 0)
    return {"skills_written": len(paths), "attachments_written": attachments, "paths": paths}


#: The keys an adapter's sync-up sends for one skill: what ``POST /skills/sync``
#: reads, plus the taxonomy labels ``skills.sync_to_platform`` sends. Never the
#: descriptor's local path, its hashes or the registry's stamp.
SYNC_FIELDS = ("name", "display_name", "description", "category", "trigger_phrases",
               "skill_type", "skill_scope", "invocation", "local_updated_at")


def _sync_up_items(skills: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """``(items, managed)``: the ``POST /skills/sync`` items for the skills you wrote.

    ``skills`` are descriptors from ``discover_skills()`` (or dicts a caller passed).
    A folder ``decimalai skills`` installed (a ``managed`` descriptor, which carries
    the registry's stamp) is never sent: it is a registry skill, and syncing it would
    create a private copy of it in your org. Its name is returned in ``managed``. An
    item without a body is skipped, since the platform refuses the whole batch for one.
    """
    items: List[Dict[str, Any]] = []
    managed: List[str] = []
    for skill in skills or ():
        if not isinstance(skill, dict) or not isinstance(skill.get("name"), str):
            continue
        if skill.get("managed"):
            managed.append(skill.get("registry_name") or skill["name"])
            continue
        body = skill.get("body_markdown") or skill.get("body")
        if not isinstance(body, str) or not body.strip():
            continue
        item = {key: skill[key] for key in SYNC_FIELDS if skill.get(key) is not None}
        item["body_markdown"] = body
        items.append(item)
    return items, managed


class SkillRouterError(Exception):
    """Raised when a SkillRouter HTTP request fails.

    Surface this to callers who need to distinguish "no skills found" from
    "the platform is unreachable." Best-effort callers (e.g. ``get_menu``,
    ``smart_route``) catch this and return safe defaults so a transient
    network failure doesn't break prompt assembly.
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        detail: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        result: Optional[Any] = None,
    ):
        self.status_code = status_code
        #: Parsed JSON error body (the FastAPI ``detail`` value) when the server
        #: sent one — e.g. a publish refusal's structured payload with ``reason``
        #: (``safety_blocked`` / ``intent_rejected`` / ``content_blocked`` …),
        #: ``findings``, ``flags``, or ``categories``. ``None`` for transport
        #: errors and non-JSON bodies.
        self.detail = detail
        #: Response headers on non-2xx responses — the fork 409 carries the
        #: existing fork's name in ``X-Installed-As`` (install idempotency).
        #: ``None`` for transport errors.
        self.headers = headers
        #: The engine's ``Result`` when a disk method failed (``export``,
        #: ``update_skills``): every refusal, with its exit code.
        self.result = result
        super().__init__(message)


class _InitAgentName:
    """The default of ``SkillRouter(agent_name=...)``: the agent ``init()`` named."""

    def __repr__(self) -> str:
        return "<the agent decimalai.init() named>"


#: ``SkillRouter(agent_name=...)`` left out. Distinct from an explicit None,
#: which keeps the meaning it has always had — no default agent, so a call that
#: names none routes against the org-wide skill set. The adapters build their
#: routers with that explicit None: they name the agent on every call (the one
#: the trace is filed under), and a router default would route a run they could
#: not name as an agent its trace never mentions.
_AGENT_FROM_INIT: Any = _InitAgentName()


class SkillRouter:
    """SDK-side skill routing client.

    Usage::

        from decimalai.skill_router import SkillRouter

        router = SkillRouter(
            api_key="dai_sk_...",
            agent_name="my-agent",
        )
        # base_url defaults to the same resolution decimalai.init() uses:
        # explicit argument, else DECIMAL_BASE_URL, else https://api.decimal.ai.
        # After decimalai.init(agent_name="my-agent"), SkillRouter() is the same
        # router: api_key defaults to init()'s key (then DECIMAL_API_KEY /
        # DECIMALAI_API_KEY), and agent_name to init()'s agent (then
        # DECIMAL_AGENT_NAME). Explicit arguments win.

        # Get the skill menu prompt fragment
        menu = router.get_menu()
        system_prompt += menu["prompt_fragment"]

        # On-demand body loading
        body = router.get_skill_body("code-review")

        # Smart routing
        routed = router.smart_route("Review this PR for security issues")
        system_prompt += routed["prompt_fragment"]
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        agent_name: Optional[str] = _AGENT_FROM_INIT,
        strategy: str = "auto",
        max_menu_size: int = 20,
        fragment_cache_ttl: float = 30.0,
        fragment_cache_size: int = 64,
        inject_body: bool = False,
        inject_body_top_k: int = 1,
        max_loaded_bodies: int = 3,
        body_token_budget: int = 6000,
        per_body_char_limit: Optional[int] = None,
        body_load_deadline_s: float = 20.0,
        priority_skills: Optional[List[str]] = None,
    ):
        from . import _config
        from ._config import DecimalConfigError, default_agent_name, default_api_key

        # Required until 2026-10-18, with no fallback: a router built next to
        # `decimalai.init()` had to be handed the same key and agent again, and
        # an agent left out routed against the org-wide skill set while the
        # traces were filed under the agent init() named — silently.
        init_url = ""
        if api_key is None:
            api_key = default_api_key()
            initialized = _config._config
            if initialized is not None and api_key and api_key == initialized.api_key:
                # init()'s key goes to the host init() was pointed at, which it
                # may have been given explicitly (`init(base_url=...)`).
                init_url = initialized.base_url
            if not api_key:
                raise DecimalConfigError(
                    "SkillRouter has no API key. Pass api_key=, call "
                    "decimalai.init(), or set the DECIMAL_API_KEY environment "
                    "variable."
                )
        if agent_name is _AGENT_FROM_INIT:
            agent_name = default_agent_name()
        self.api_key = api_key
        # Same resolution decimalai.init() uses (explicit → DECIMAL_BASE_URL
        # → default) so a directly-constructed router doesn't silently split
        # a non-default deployment onto the public host.
        self.base_url = (
            base_url
            or init_url
            or os.environ.get("DECIMAL_BASE_URL", "")
            or "https://api.decimal.ai"
        ).rstrip("/")
        self.agent_name = agent_name
        self.strategy = strategy
        self.max_menu_size = max_menu_size
        # When True, build_prompt_fragment delivers the routed skill's BODY (the actual
        # knowledge K), not just a menu row — so the skill's runtime value (front-loaded K →
        # fewer turns / better answers) actually reaches the agent. ``inject_body_top_k`` controls
        # how many of the top routed skills get their body injected. Default off (menu-only) so
        # existing callers are unchanged.
        self.inject_body = inject_body
        self.inject_body_top_k = max(1, int(inject_body_top_k))
        # Reserve body slots for shared guidance such as reply format, but
        # only when the server offered it to this agent. Never widen access.
        self.priority_skills = list(dict.fromkeys(priority_skills or []))
        # Body guardrail (the progressive-disclosure path): caps for on-demand load_skill AND
        # the body-inject path (bodies injected un-trimmed before top-k). Defaults
        # Full bodies fit or are omitted, never trimmed into a partial version.
        # Derive the default character cap from the same chars/4 estimate as
        # the total token budget; an explicit caller cap remains authoritative.
        self.max_loaded_bodies = max(1, int(max_loaded_bodies))
        self.body_token_budget = max(1, int(body_token_budget))
        self.per_body_char_limit = max(256, int(
            per_body_char_limit if per_body_char_limit is not None else 4 * self.body_token_budget
        ))
        self.body_load_deadline_s = float(body_load_deadline_s)

        # Fallback body-load budget for tool-execution contexts that did not
        # inherit the prompt-build ContextVar (see load_skill).
        self._last_budget: Optional[_BodyLoadBudget] = None
        # Per-run budgets, for callers that can name the run they are in
        # (``scope=``). `_last_budget` is a single slot on a singleton every
        # concurrent run shares: two parallel runs overwrite each other's
        # turn budget, so one run's caps get charged against the other's
        # loads. A scoped caller gets its own. Bounded LRU so a run that
        # never finishes cannot pin memory.
        self._scoped_budgets: "OrderedDict[str, _BodyLoadBudget]" = OrderedDict()
        # 'loaded' = the model pulled a skill's BODY on demand through the
        # load_skill tool (progressive disclosure, step 3). Instance state,
        # NOT a contextvar like the offered/delivered rails: loads fire
        # mid-run inside the framework's tool executor, whose copied context
        # never propagates back to the adapter's trace-send path — the same
        # reality behind the `_last_budget` fallback above. Adapters drain
        # this off their router singleton via `consume_loaded_names()`.
        self._loaded_names: List[str] = []
        # name -> content_hash for every body this router actually loaded.
        # Parallel to `_loaded_names` rather than folded into it: the rails are
        # drained by five adapters and a shape change there is a much wider blast
        # radius than capturing the hash. Read it via `loaded_skill_hash()`.
        self._loaded_hashes: Dict[str, Optional[str]] = {}
        # Per-run twin of `_loaded_names`. `load_skill`'s docstring has always
        # promised "the loaded-names rail [is] kept per scope"; until this
        # existed, `scope` reached only the turn budget and the rail itself was
        # one shared list, so a scoped caller asking "what did THIS run load"
        # got every concurrent run's loads. Bounded LRU, same discipline and
        # same cap as `_scoped_budgets`.
        self._scoped_loaded_names: "OrderedDict[str, List[str]]" = OrderedDict()
        # name -> content_hash for the loads on THIS run's rail, drained by
        # `consume_loaded_hashes(scope=...)` alongside the names. Separate from
        # `_loaded_hashes` above, which is a persistent last-seen map serving
        # `loaded_skill_hash()` and is never cleared: a persistent map cannot
        # answer "what did THIS run read" once a concurrent run has loaded the
        # same skill at another version. Same LRU cap and same lock as
        # `_scoped_loaded_names`, so the two stay evictable together.
        self._scoped_loaded_hashes: "OrderedDict[str, Dict[str, Optional[str]]]" = (
            OrderedDict()
        )
        # The unscoped rail's twin of the above: hashes for the loads sitting on
        # `_loaded_names` right now, cleared when those names are drained. The
        # unscoped rail is already documented as cross-contaminating concurrent
        # runs' NAMES; this at least keeps it from inventing a VERSION for a name
        # it got right — see `_record_window_hash` for the ambiguity rule.
        self._loaded_hashes_window: Dict[str, Optional[str]] = {}
        # Instance mirror of the routing decision — routing_id + offered +
        # delivered names — for the same reason `_loaded_names` exists, one
        # step earlier in the run: prompt assembly also happens in a copied
        # context (LangChain runs its callbacks under `copy_context()`, the
        # OpenAI Agents runner copies around the instructions callable), so
        # the contextvar rails above are invisible to the adapter's
        # trace-send path. The contextvars stay authoritative where they DO
        # propagate (the generic tracer; every agent_config(), which reads them
        # right after its own routing call); adapters
        # union this rail in via `consume_routing_id()` /
        # `consume_offered_names()` / `consume_delivered_names()`.
        self._routing_id_rail: Optional[str] = None
        self._offered_names_rail: List[str] = []
        self._delivered_names_rail: List[str] = []
        self._delivered_versions_rail: List[Dict[str, str]] = []
        # Per-run twin of the three rails above, the same shape and the same
        # discipline as `_scoped_loaded_names`. Without it a scoped adapter
        # could keep its LOADS apart from a concurrent run's but not its
        # ROUTING DECISION, and a run that captured nothing of its own still
        # claimed whichever routing_id and offered set were lying on the
        # shared slot — measured: a LangChain trace with zero LLM calls
        # reporting another run's routing_id and its full offered set.
        # scope -> {"routing_id": str|None, "offered": [...], "delivered": [...]}
        self._scoped_routing_rails: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        # Provenance for the UNSCOPED rails above: which runs wrote the
        # content sitting on them right now. The rails themselves carry none —
        # that IS the defect they are known for — so a reader had no way to
        # tell "my own names, written from a context that could not reach me"
        # apart from "somebody else's names". With this it can: an adapter
        # asks `unscoped_rail_owners()` and drops the values if any owner is
        # not one of its own scopes. Cleared automatically the moment all four
        # unscoped rails are empty, so it describes only undrained content.
        self._unscoped_rail_owners: "OrderedDict[str, None]" = OrderedDict()
        # The routing decision each run's `load_skill` calls belong to, sent as
        # `routing_id` on the body request. The platform serves owned,
        # installed and assigned skills whatever it says, and serves a skill
        # reachable ONLY through the public registry just when that decision
        # offered it. The SDK cannot make that call itself: the menu caps at
        # about eight rows, so an owned skill below the cut looks exactly like
        # a planted public name from here.
        #
        # Value: the run's latest `routing_id`, or "" when its routing call
        # returned none (it failed — the run was offered nothing). Per run
        # under `scope=`, plus the latest for the whole router, which an
        # unscoped load (or a scope this router never routed for) sends. That
        # may be a concurrent run's decision, which is safe for the skills that
        # matter: an owned or installed skill, or one assigned to the agent the
        # load names, is served whichever decision is named. Not drained by
        # `_release_scoped_routing_rail` — a run's loads come after its route.
        # Bounded LRU like the rails. `_last_run_routing_id` None means this
        # router never routed (a hand-built menu, a direct `load_skill`), and
        # those loads send no `routing_id` at all.
        self._run_routing_ids: "OrderedDict[str, str]" = OrderedDict()
        self._last_run_routing_id: Optional[str] = None
        # One lock for every per-scope store. `otel._skill_rails` and
        # `openai_agents._run_rails` both take one; these did not, and the
        # `get` -> `move_to_end` -> `popitem` sequences below are not atomic.
        # Never held across a network call or a resolver callback.
        self._rail_lock = Lock()
        # Full-menu cache (single slot, force-refresh to invalidate).
        # Cache the menu per (category, project_id, effective_agent) so a
        # second get_menu() with different args doesn't return the first call's
        # menu. force_refresh still bypasses/refreshes the matching key.
        # Bounded + TTL'd, not a plain dict. It holds a per-RUN key (see
        # get_menu), so an unbounded dict would grow once per run forever in a
        # long-lived server. It was previously unbounded AND never expired,
        # which also meant every full-menu run in a process shared the FIRST
        # run's routing_id for the life of the process.
        self._menu_cache = _FragmentCache(
            maxsize=fragment_cache_size,
            ttl_seconds=fragment_cache_ttl,
        )
        # `build_prompt_fragment` cache. Keyed by the input
        # tuple so repeat calls within the same turn (multi-LLM-call
        # agent loops) reuse a single routing decision instead of
        # racking up N RoutingDecision rows + N Gemini embedding calls.
        self._fragment_cache = _FragmentCache(
            maxsize=fragment_cache_size,
            ttl_seconds=fragment_cache_ttl,
        )

    def _headers(self) -> Dict[str, str]:
        # Shared builder — see `_config.sdk_headers`. Routing requests carry the
        # SDK version for the same reason ingest does: "which SDK version asked
        # for this menu?" should be answerable from the request itself.
        from ._config import sdk_headers

        return sdk_headers(self.api_key)

    def _request(
        self,
        method: str,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        *,
        hot: Optional[bool] = None,
        read_budget_s: Optional[float] = None,
        admission_wait_s: Optional[float] = None,
        replay_safe: Optional[bool] = None,
        read_retry_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request to the platform API.

        Args:
            hot: whether this call sits inside a user's turn (5 s read budget,
                circuit breaker, counted by ``routing_status()``). Decided by the
                path when omitted; disk export passes ``False`` for ``/body``.
            read_budget_s: a hot call's read budget in place of the default.
            admission_wait_s: the most this call may wait out ``Retry-After``
                on a 429 (default ``_ADMISSION_MAX_WAIT_S``); 0 never waits.
            replay_safe: whether the request may be sent again after the
                platform may have received it (admission waits, a dropped
                connection, a read timeout). Defaults to GETs and the routing
                request; pass ``True`` for an endpoint documented idempotent.
            read_retry_s: the read budget of the one retry after a read
                timeout. Defaults to ``_HOT_PATH_READ_RETRY_S`` for a hot call
                (capped at its own read budget) and the cold budget otherwise;
                0 never retries a read timeout.

        Raises:
            SkillRouterError: on transport failure, non-2xx response, or
                non-JSON response body. Callers that want graceful
                degradation (empty menu when API is down, etc.) should
                catch ``SkillRouterError`` themselves — this surface
                deliberately does not swallow errors.
        """
        url = f"{self.base_url}{path}"
        clean_params = (
            {k: v for k, v in params.items() if v is not None} if params else None
        )

        hot = _is_hot_path(path) if hot is None else hot
        # Two breakers, one per cause (see `_HotPathBreakers`): the host's,
        # which every key on it shares, and this key's own. Looked up per call,
        # so a router whose `base_url` or `api_key` is reassigned follows it.
        host_breaker = key_breaker = None
        if hot:
            breakers = _hot_path_breakers
            host_breaker = breakers.for_host(self.base_url)
            key_breaker = breakers.for_credential(self.base_url, self.api_key)
            # Short-circuit rather than pay the timeout again. See _CircuitBreaker.
            if host_breaker.is_open():
                raise SkillRouterError(
                    f"SkillRouter circuit open for {method} {path} — skipping the call "
                    f"while the platform at {host_breaker.target} is unreachable. "
                    "Skills are unavailable this turn."
                )
            if key_breaker.is_open():
                raise SkillRouterError(
                    f"SkillRouter circuit open for {method} {path} — skipping the call "
                    "while this API key's requests keep failing (last: HTTP "
                    f"{key_breaker.last_status_code}). Skills are unavailable this turn."
                )

        # Freeze the caller's auth/scope and request inputs across retries.
        # Only reads, the routing request and endpoints the caller marks
        # `replay_safe` are sent twice after the platform may have seen them
        # (admission waits, dropped connections, read timeouts); publishing and
        # other writes are not replayed. A positive Retry-After is required:
        # an unmarked edge abort or plan quota must remain fail-fast.
        if replay_safe is None:
            replay_safe = method.upper() == "GET" or (
                method.upper() == "POST" and path == _ROUTE_PATH
            )
        admission_safe = replay_safe
        if read_retry_s is None:
            read_retry_s = (
                min(_HOT_PATH_READ_RETRY_S, read_budget_s or _hot_path_read_budget())
                if hot else _COLD_PATH_TIMEOUT
            )
        retry_read_timeout = replay_safe and path != _ROUTE_PATH and read_retry_s > 0
        headers = dict(self._headers())
        request_json = deepcopy(json)
        request_params = deepcopy(clean_params)
        started = time.monotonic()
        waited = 0.0
        max_wait = _ADMISSION_MAX_WAIT_S if admission_wait_s is None else admission_wait_s
        # One spare attempt per transport failure kind, for the whole call (see
        # "Transport" above `_platform_client`).
        spare = {"connect": True, "dropped": replay_safe, "read": retry_read_timeout}
        base_timeout: Any = _hot_path_timeout(read_budget_s) if hot else _COLD_PATH_TIMEOUT
        retry_timeout: Any = (
            _hot_path_timeout(read_retry_s) if hot
            else httpx.Timeout(_COLD_PATH_TIMEOUT, read=read_retry_s)
        )

        def attempt_once() -> httpx.Response:
            timeout = base_timeout
            while True:
                err: httpx.HTTPError
                try:
                    return _send(
                        method, url, params=request_params, json=request_json,
                        headers=headers, timeout=timeout,
                    )
                except _CONNECT_FAILURES as e:
                    kind, err = "connect", e
                except _DROPPED_FAILURES as e:
                    kind, err = "dropped", e
                except httpx.ReadTimeout as e:
                    kind, err = "read", e
                if not spare[kind]:
                    raise err
                spare[kind] = False
                if kind == "read":
                    timeout = retry_timeout
                logger.info(
                    "SkillRouter %s on %s %s (%s: %s); retrying once",
                    {"connect": "could not connect", "dropped": "connection dropped",
                     "read": "read timed out"}[kind],
                    method, path, type(err).__name__, err,
                )

        try:
            for attempt in range(_ADMISSION_MAX_RETRIES + 1):
                # Admission recovery never retries generic 5xx; transport
                # failures get their one spare attempt inside `attempt_once`.
                resp = attempt_once()
                if not admission_safe or resp.status_code != 429 or attempt >= _ADMISSION_MAX_RETRIES:
                    break
                if resp.headers.get("X-Quota-Exceeded"):
                    break
                from ._client import _parse_retry_after
                delay = _parse_retry_after(resp.headers.get("Retry-After"))
                if (delay <= 0 or waited + delay > max_wait
                        or time.monotonic() - started + delay > max_wait):
                    break
                logger.warning(
                    "SkillRouter admission limited on %s %s; retrying in %.1fs (%d/%d)",
                    method, path, delay, attempt + 1, _ADMISSION_MAX_RETRIES,
                )
                time.sleep(delay)
                waited += delay
        except httpx.HTTPError as e:
            if host_breaker is not None:
                host_breaker.record_failure(e)
            raise SkillRouterError(
                f"SkillRouter transport error on {method} {path}: {e}"
            ) from e

        if resp.status_code >= 400:
            # Surface the server's structured error body (FastAPI ``detail``) —
            # publish-gate refusals carry actionable findings/flags/categories
            # that callers need for remediation, not just the status code.
            detail: Optional[Any] = None
            try:
                body = resp.json()
                detail = body.get("detail", body) if isinstance(body, dict) else body
            except (json_module.JSONDecodeError, ValueError):
                detail = None
            message = f"SkillRouter request failed ({resp.status_code}): {method} {path}"
            if isinstance(detail, dict):
                reason = detail.get("reason")
                gate_msg = detail.get("message")
                if reason or gate_msg:
                    message += f" — {reason or 'error'}: {gate_msg or ''}".rstrip(": ")
            elif isinstance(detail, str) and detail:
                message += f" — {detail}"
            # Header keys lowercased so callers can look them up without
            # caring whether the transport preserved case (httpx normalizes
            # to lowercase already; mocks in tests may not).
            try:
                resp_headers = {
                    str(k).lower(): v for k, v in dict(resp.headers).items()
                }
            except Exception:
                resp_headers = None
            error = SkillRouterError(
                message,
                status_code=resp.status_code,
                detail=detail,
                headers=resp_headers,
            )
            if host_breaker is not None and key_breaker is not None:
                # Any answer proves the host reachable. Whether it delivers
                # skills is this key's matter, not every key's.
                host_breaker.record_reachable()
                if _is_health_failure(path, resp.status_code):
                    key_breaker.record_failure(error, status_code=resp.status_code)
                else:
                    # The platform answered (an unknown skill name, a request
                    # it rejected) — it is up, and that is what the breaker
                    # tracks.
                    key_breaker.record_success()
            raise error

        try:
            data = resp.json()
        except (json_module.JSONDecodeError, ValueError) as e:
            error = SkillRouterError(
                f"SkillRouter got non-JSON response from {method} {path}: {e}"
            )
            if host_breaker is not None and key_breaker is not None:
                # A 200 that is not JSON (an HTML login or challenge page from
                # a proxy) delivers no skills either.
                host_breaker.record_reachable()
                key_breaker.record_failure(error, status_code=resp.status_code)
            raise error from e
        if host_breaker is not None and key_breaker is not None:
            host_breaker.record_reachable()
            key_breaker.record_success()
        return data

    def get_menu(
        self,
        *,
        category: Optional[str] = None,
        project_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        force_refresh: bool = False,
        scope: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch the full skill menu from the platform.

        Returns a dict with 'skills', 'prompt_fragment', 'strategy'.
        Results are cached until force_refresh=True.

        ``agent_name`` (falls back to the instance default) lets the platform's
        Use/Fork resolver scope the menu to one agent when enabled server-side;
        it's a no-op against older backends / when the resolver is off.
        """
        effective_agent = agent_name or self.agent_name
        # `scope` is the run this menu is for, and it is part of the key because
        # the cached result carries a `routing_id` — the platform's record of ONE
        # routing decision. Handing a second run the first run's routing_id makes
        # two runs claim one decision, which is precisely what the effectiveness
        # join must not see: offers get undercounted by however many runs shared
        # the slot. This cache had no TTL either, so on the full-menu path every
        # run in a process shared the first one's decision indefinitely.
        #
        # The cost is real and is the honest price of per-run attribution: on the
        # full-menu path (query=None) each run now fetches its own menu instead
        # of reusing a process-wide one. Unscoped callers pass None and keep
        # exactly today's key and call volume.
        cache_key = (category, project_id, effective_agent, scope or "")
        if not force_refresh:
            cached = self._menu_cache.get(cache_key)
            if cached is not None:
                return cached

        params: Dict[str, Any] = {}
        if category:
            params["category"] = category
        if project_id:
            params["project_id"] = project_id
        if effective_agent:
            params["agent_name"] = effective_agent

        try:
            result = self._request("GET", "/api/v1/skills/menu", params=params)
        except SkillRouterError as e:
            # Graceful degradation: prompt assembly shouldn't break if the
            # platform is unreachable. Log loudly so callers can see *why*.
            logger.warning("get_menu failed, returning empty menu: %s", e)
            return {"skills": [], "prompt_fragment": "", "strategy": "none",
                    "degraded_reason": _empty_menu_reason(e)}

        if result:
            self._menu_cache.set(cache_key, result)
        return result or {"skills": [], "prompt_fragment": "", "strategy": "none"}

    def smart_route(
        self,
        query: str,
        *,
        top_k: int = 10,
        category: Optional[str] = None,
        include_attachments: bool = True,
        agent_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Smart routing: semantic search + performance re-ranking.

        Available on every plan, including Free.

        Sends the query to the platform for embedding and skill matching.

        Args:
            query: The user message to route to a matching skill.
            top_k: Maximum number of skill matches to return.
            category: Optional category filter.
            include_attachments: If True (default), includes script content
                inline for skills that have multi-file bundles. Set to False
                to reduce response size when you don't need bundled scripts.
        """
        payload: Dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "include_attachments": include_attachments,
        }
        # Honor a per-call agent_name override (falling back to the instance
        # default) so the routing decision is recorded against the agent the
        # caller named, not always self.agent_name.
        effective_agent = agent_name or self.agent_name
        if effective_agent:
            payload["agent_name"] = effective_agent
        if category:
            payload["category"] = category

        try:
            return self._request("POST", "/api/v1/skills/route", json=payload)
        except SkillRouterError as e:
            logger.warning("smart_route failed, returning empty: %s", e)
            return {"skills": [], "prompt_fragment": "", "strategy": "none",
                    "degraded_reason": _empty_menu_reason(e)}

    def _fallback_menu(
        self, route_failure: str, *, category: Optional[str], agent_name: Optional[str],
    ) -> Dict[str, Any]:
        """This agent's menu, unranked, in place of a route that cannot be used.

        One fresh ``GET /skills/menu`` (a fresh decision, so the run's
        ``routing_id`` is its own), on a ``_MENU_FALLBACK_READ_S`` read budget
        and with no admission retries. It goes through the same breakers as any
        hot call, so an open one skips it. Returns the menu marked ``degraded``,
        or the SDK's empty stand-in when the menu failed too.
        """
        params: Dict[str, Any] = {}
        if category:
            params["category"] = category
        if agent_name:
            params["agent_name"] = agent_name
        try:
            menu = self._request(
                "GET", "/api/v1/skills/menu", params=params,
                read_budget_s=min(_MENU_FALLBACK_READ_S, _hot_path_read_budget()),
                # No second read either: a failed route may cost a turn at most
                # `_MENU_FALLBACK_READ_S` more.
                admission_wait_s=0.0, read_retry_s=0.0,
            )
        except SkillRouterError as e:
            logger.warning("skill routing failed (%s) and so did the menu fallback: %s", route_failure, e)
            return {"skills": [], "prompt_fragment": "", "strategy": "none",
                    "degraded_reason": _empty_menu_reason(e)}
        if not isinstance(menu, dict) or not menu:
            return {"skills": [], "prompt_fragment": "", "strategy": "none",
                    "degraded_reason": route_failure}
        _hot_path_breakers.totals.menu_fallback()
        return {**menu, "degraded": True, "degraded_reason": f"menu_fallback:{route_failure}"}

    def get_skill_body(
        self,
        skill_name: str,
        version: Optional[int] = None,
        *,
        max_chars: Optional[int] = None,
        agent_name: Optional[str] = None,
    ) -> Optional[str]:
        """Fetch the body markdown of a skill for prompt injection.

        ``max_chars`` asks the server to trim (body guardrail); ``agent_name``
        resolves the exact version that agent was offered (Use pins).
        Returns the body text or None if not found. Direct callers may request
        partial text, but a partial record never hands off a full-version hash
        and is refused by prompt injection and ``load_skill``.

        Prefer :meth:`get_skill_body_record` when you also want the version the
        body came from — this wrapper drops it.
        """
        # Cleared FIRST, so a miss or a raise below can never leave the previous
        # body's hash on this thread for `load_skill` to pick up.
        _body_hash_tls.value = None
        _body_hash_tls.partial = None
        _body_hash_tls.identity = None
        result = self.get_skill_body_record(
            skill_name, version, max_chars=max_chars, agent_name=agent_name,
        )
        if not result:
            return None
        body = result.get("body")
        _body_hash_tls.partial = _body_is_known_partial(body, result)
        content_hash = result.get("content_hash")
        if _body_hash_tls.partial:
            with self._rail_lock:
                self._loaded_hashes.pop(skill_name, None)
        elif isinstance(content_hash, str) and content_hash:
            # Two readers, on purpose. `_loaded_hashes` is the persistent
            # last-seen map behind `loaded_skill_hash()`; the thread-local is
            # the race-free handoff to `load_skill`, which needs the hash of the
            # body IT just fetched and not whichever version another thread
            # stored last. See the comment at that call site.
            _body_hash_tls.value = content_hash
            _body_hash_tls.identity = _served_version_identity(result)
            with self._rail_lock:
                self._loaded_hashes[skill_name] = content_hash
        return body

    def get_skill_body_record(
        self,
        skill_name: str,
        version: Optional[int] = None,
        *,
        max_chars: Optional[int] = None,
        agent_name: Optional[str] = None,
        routing_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Like :meth:`get_skill_body` but returns the whole server record.

        The platform hands back ``content_hash`` and ``version`` alongside ``body``
        precisely so no client has to recompute the hash. Until 2026-08-28 this
        method did not exist and ``get_skill_body`` discarded both one line after
        receiving them, so every activation the SDK reported carried
        ``skill_hash: null`` — breaking the join that attributes a measured lift to
        the skill VERSION the model actually read.

        ``routing_id`` names the routing decision this fetch belongs to, for a
        loop that loads skills inside an agent run: the platform then serves a
        skill reachable only through the public registry just when that
        decision offered it (owned, installed and assigned skills are served
        either way), and ``""`` means a run that was offered nothing.
        :meth:`load_skill` sends it for you. Omitted, nothing changes.
        """
        params: Dict[str, Any] = {}
        if version is not None:
            params["version"] = version
        if max_chars is not None:
            params["max_chars"] = max_chars
        if agent_name is not None:
            params["agent_name"] = agent_name
        if routing_id is None:
            routing_id = getattr(_body_hash_tls, "load_routing_id", None)
        if routing_id is not None:
            params["routing_id"] = routing_id

        # Why the last fetch on this thread returned nothing, for `load_skill`:
        # None means "no such skill" (a 404) or success; a reason string means
        # the platform could not answer, which the model must not be told is a
        # missing skill. Same thread-local handoff as the hash, for the same
        # reason (see `_body_hash_tls`).
        _body_hash_tls.unavailable = None
        try:
            result = self._request(
                "GET",
                _skill_body_path(skill_name),
                params=params,
            )
        except SkillRouterError as e:
            logger.warning("get_skill_body(%s) failed: %s", skill_name, e)
            if getattr(e, "status_code", None) != 404:
                _body_hash_tls.unavailable = _empty_menu_reason(e)
            return None
        return result if isinstance(result, dict) else None

    def loaded_skill_hash(self, name: str) -> Optional[str]:
        """The ``content_hash`` of a body this router loaded, if it saw one.

        Populated by :meth:`load_skill`. Adapters that report activations can
        pass this as ``log_skill_activation(hash=...)`` so the activation is
        attributable to a specific skill version.
        """
        with self._rail_lock:
            return self._loaded_hashes.get(name)

    def load_skill(
        self,
        name: str,
        *,
        agent_name: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> str:
        """The load_skill tool handler — fetch a surfaced skill's body on demand.

        Always returns a STRING (never raises): either the trimmed body block
        the adapter feeds back as the tool result, or an explanatory message
        (budget exhausted / not found / platform unavailable) the
        model can act on. Enforces the per-turn body guardrail:
        ``max_loaded_bodies``, ``body_token_budget``, ``per_body_char_limit``
        cap, ``body_load_deadline_s``. Records the load on the active trace
        (``skills_loaded_by_agent``) so the offered-vs-loaded join closes
        server-side.

        Names the run's routing decision on the request (``routing_id``, see
        :meth:`_run_routing_id`). The platform then serves any skill the org
        owns, has installed or has assigned, and serves a skill reachable only
        through the public registry just when that decision offered it — so a
        prompt injection naming a planted public skill gets "no skill named",
        while a skill of the org's own that the menu cap pushed off this
        turn's menu still loads.

        Args:
            scope: an opaque key naming the RUN this load belongs to (the
                adapter's trace id). Budget accounting, the routing decision
                sent with the load and the loaded-names rail are kept per
                scope, so concurrent runs sharing one router singleton stop
                consuming each other's turn budget. Omit it and the
                single-slot fallback applies.
        """
        name = (name or "").strip()
        if not name:
            return "load_skill error: a skill name is required."

        # A caller that names its run is the most precise answer available,
        # so it wins outright: falling back to the ContextVar there would
        # hand this run whichever budget another run happened to leave in
        # the current context. Unscoped callers keep the old order —
        # ContextVar (isolated per agent run) first, then the single-slot
        # fallback for frameworks that execute each tool call in a context
        # that did not inherit the prompt-build one.
        if scope is not None:
            budget = self._budget_for(scope)
        else:
            budget = _body_budget_ctx.get() or self._last_budget
        if budget is None:
            budget = _BodyLoadBudget(
                self.max_loaded_bodies, self.body_token_budget,
                self.body_load_deadline_s,
            )
            _body_budget_ctx.set(budget)
            self._remember_budget(budget, scope)

        refusal = budget.check(name)
        if refusal is not None:
            return refusal

        # Still `get_skill_body`, deliberately. It is a public method, and
        # tests, subclasses and conformance drivers all stub it; switching this
        # call to `get_skill_body_record` to get at the hash would leave every
        # one of those stubs uncalled — 19 tests went red proving it.
        #
        # So the hash comes back on a thread-local instead. `get_skill_body`
        # runs on THIS thread, synchronously, between the two lines below, which
        # makes the handoff race-free without changing any signature. Re-reading
        # `_loaded_hashes[name]` after the call would NOT be: it is a persistent
        # last-seen map, and a concurrent run loading the same skill at another
        # version overwrites it in between — attributing this activation to a
        # version the model never read. A wrong version is worse than none. NULL
        # already means "no known hash" everywhere downstream; a wrong hash
        # resolves to a real, wrong row.
        #
        # A stub leaves the thread-local at the None set here, so a stubbed
        # `get_skill_body` reports the load with no hash — exactly what every
        # load reported before this rail existed.
        _body_hash_tls.value = None
        _body_hash_tls.partial = None
        _body_hash_tls.identity = None
        _body_hash_tls.unavailable = None
        # The run's routing decision rides the same thread-local into
        # `get_skill_body_record`, for the same reason as the hash: a new
        # keyword on `get_skill_body` would break every stub of it. Set for
        # exactly this call and cleared even if it raises, so no other fetch
        # on this thread (an injected body, an export) ever carries it.
        _body_hash_tls.load_routing_id = (
            _load_routing_id_ctx.get() or self._run_routing_id(scope))
        try:
            body = self.get_skill_body(
                name,
                max_chars=self.per_body_char_limit,
                agent_name=agent_name or self.agent_name,
            )
        finally:
            _body_hash_tls.load_routing_id = None
        content_hash = getattr(_body_hash_tls, "value", None)
        identity = getattr(_body_hash_tls, "identity", None) or {}
        partial = getattr(_body_hash_tls, "partial", None)
        unavailable = getattr(_body_hash_tls, "unavailable", None)
        if partial is True or (partial is None and _body_is_known_partial(body)):
            return (
                f"load_skill budget exhausted: {name!r} was returned as a partial body. "
                "Its complete instructions were not loaded. Proceed with what is already loaded."
            )
        if body is None and unavailable:
            # Not "no skill named X": the skill exists, the platform did not
            # answer. Telling the model it is missing invites it to guess
            # another name, or to claim the skill does not exist.
            return (
                f"load_skill: {name!r} could not be loaded right now — the skills "
                f"platform is unavailable ({unavailable}). Its instructions were not "
                "loaded. Proceed without it and do not retry this turn."
            )
        if body is None or not body.strip():
            return (
                f"load_skill: no skill named {name!r} is available. Use the exact "
                "name from the skills menu."
            )
        body = body.strip()
        # Older backends may ignore max_chars. Refuse instead of delivering a
        # clipped body with the immutable full-version hash.
        if len(body) > self.per_body_char_limit:
            return (
                f"load_skill budget exhausted: the complete body of {name!r} exceeds "
                f"the {self.per_body_char_limit}-character per-body limit. "
                "Its instructions were not loaded. Proceed with what is already loaded."
            )

        tokens = estimate_tokens(body)
        # Dedup re-loads skip the token check — their charge is replaced,
        # not added (see _BodyLoadBudget.record).
        if name not in budget.loaded and budget.would_exceed(tokens):
            return (
                f"load_skill budget exhausted: loading {name!r} (~{tokens} tokens) "
                f"would exceed the {budget.token_budget}-token body budget for this "
                "turn. Proceed with what is already loaded."
            )
        budget.record(name, tokens)

        # Body-load activation signal — trace ingest persists
        # skills_loaded_by_agent; joined with RoutingDecision.offered_skill_names
        # this is the server-side multi-skill activation record (step 3 of progressive disclosure).
        # The rail is what adapter traces drain (`consume_loaded_names` at
        # trace-send); the generic call below only reaches a native
        # @decimalai.trace context, which is absent under the adapters —
        # there it raises and the load would otherwise go unrecorded.
        with self._rail_lock:
            if name not in self._loaded_names:
                self._loaded_names.append(name)
        # An explicit `scope=` wins; failing that, ask the adapters which run
        # is executing. LangChain registers no load_skill tool of its own, so
        # its loads arrive from user tool code that has no scope to pass —
        # without the ambient answer the load lands ONLY on the shared slot,
        # and whichever concurrent trace sends first takes it. The BUDGET
        # deliberately does not consult the resolver: budgets are per TURN and
        # keyed by the caller's own run key, while a resolver answers with the
        # framework's finest-grained live run id, so charging against it would
        # silently reset the per-turn caps mid-run.
        rail_scope = scope if scope is not None else _ambient_scope()
        self._note_unscoped_writer(rail_scope)
        # The hash goes onto BOTH rails, for the same reason the name does: an
        # adapter drains whichever one it can reach, and the unscoped rail is
        # the only one a framework with no run identity ever writes to.
        self._record_window_hash(name, content_hash)
        if rail_scope is not None:
            self._record_scoped_load(rail_scope, name, content_hash)
        if isinstance(content_hash, str) and content_hash:
            witness = {"name": name, "hash": content_hash, **identity}
            self._record_delivery_versions([witness], scope=rail_scope)
            try:
                from .generic import _get_current_trace
                enclosing = _get_current_trace()
                if enclosing is not None:
                    enclosing.log_skill_delivery_versions(versions=[witness])
            except Exception:
                logger.debug("load_skill: no active trace for immutable delivery", exc_info=True)
        try:
            from .generic import log_skill_loaded
            try:
                log_skill_loaded(name=name, hash=content_hash)
            except TypeError:
                # A caller that monkey-patched this function with the old
                # one-argument signature. The LOAD still has to be recorded —
                # losing the activation to gain a hash would be a bad trade.
                log_skill_loaded(name=name)
        except Exception:
            logger.debug("load_skill: no active trace to record the load", exc_info=True)

        return f"## Skill: {name}\n\n{body}"

    # Cap on how many concurrent runs keep their own turn budget. Runs are
    # short; anything past this is a leak, so evict oldest-first.
    _MAX_SCOPED_BUDGETS = 64
    # Separate, larger cap for the per-scope RAILS. A budget key is minted
    # once per turn by an adapter that names its own run, so 64 keys is 64
    # concurrent runs. A rail key can be finer: LangChain scopes by the
    # LangChain run id visible where the write happens, and one graph run
    # writes under several of them (the agent node routes, the tools node
    # loads). Sized to match `otel._SKILL_RAIL_MAX` so an eviction here means
    # the same thing it means there.
    # 4096, not 256. One RUN holds one rail scope per node that writes, so a
    # graph that loads from several tool nodes multiplies keys per run — 256
    # was reachable by a few dozen concurrent runs, and eviction here drops a
    # TRUE activation from a run still in flight. These entries are a handful of
    # short strings each; the memory is not the constraint, and losing a real
    # observation is much more expensive than holding it.
    #
    # Eviction is still possible, so it is no longer SILENT: _evict_rail_overflow
    # warns. A skills rail that quietly empties under load is exactly the failure
    # this rung was rebuilt to remove.
    _MAX_SCOPED_RAILS = 4096
    _MAX_DELIVERED_VERSIONS = 512

    def _evict_rail_overflow(self, store: Any, what: str) -> None:
        """Drop the oldest entry and say so. Callers hold ``_rail_lock``."""
        scope, _ = store.popitem(last=False)
        logger.warning(
            "skill rail overflow: evicted the %s rail for scope %r after %d live "
            "scopes. A run still in flight loses its recorded skill activity. "
            "Raise SkillRouter._MAX_SCOPED_RAILS if this recurs.",
            what, scope, self._MAX_SCOPED_RAILS,
        )

    def _remember_run_routing(self, scope: Optional[str], routing_id: Optional[str]) -> None:
        """Record the decision one routing call made, for this run's load_skill.

        Called on every routing call, a cache hit and a failed one included: a
        run whose route failed was offered nothing, recorded as ``""``.
        """
        value = routing_id if isinstance(routing_id, str) else ""
        with self._rail_lock:
            self._last_run_routing_id = value
            if scope is None:
                return
            self._run_routing_ids[scope] = value
            self._run_routing_ids.move_to_end(scope)
            while len(self._run_routing_ids) > self._MAX_SCOPED_RAILS:
                self._run_routing_ids.popitem(last=False)

    def _run_routing_id(self, scope: Optional[str]) -> Optional[str]:
        """The ``routing_id`` a load in ``scope`` sends, or None to send none.

        This run's latest decision; for an unscoped load, or a scope this
        router never routed for, the router's latest. None only when this
        router has never routed.
        """
        with self._rail_lock:
            if scope is not None and scope in self._run_routing_ids:
                return self._run_routing_ids[scope]
            return self._last_run_routing_id

    def _budget_for(self, scope: Optional[str]) -> Optional[_BodyLoadBudget]:
        """This run's turn budget, or the legacy single slot when unscoped."""
        if scope is None:
            return self._last_budget
        with self._rail_lock:
            budget = self._scoped_budgets.get(scope)
            if budget is not None:
                self._scoped_budgets.move_to_end(scope)
        return budget

    def _remember_budget(
        self, budget: _BodyLoadBudget, scope: Optional[str],
    ) -> None:
        """Store a freshly-minted turn budget against its run."""
        self._last_budget = budget
        if scope is None:
            return
        with self._rail_lock:
            self._scoped_budgets[scope] = budget
            self._scoped_budgets.move_to_end(scope)
            while len(self._scoped_budgets) > self._MAX_SCOPED_BUDGETS:
                self._scoped_budgets.popitem(last=False)

    @staticmethod
    def _merge_hash(
        store: Dict[str, Optional[str]], name: str, content_hash: Optional[str],
    ) -> None:
        """Record one load's hash, degrading a CONFLICT to None.

        Callers hold ``_rail_lock``.

        The conflict is real: the same skill name loaded twice against one rail
        with two different hashes means two versions were read and the rail can
        name only one. Whichever we picked would be a claim we cannot support
        for the other, so we keep neither — ``None`` is exactly "activation with
        no known hash", which every downstream reader already handles
        (`_resolve_managed_skill` skips the version lookup, the analytics
        service skips falsy hashes). A gap is not a lie; a wrong version is.
        """
        if name in store and store[name] != content_hash:
            store[name] = None
            return
        store[name] = content_hash

    def _record_window_hash(self, name: str, content_hash: Optional[str]) -> None:
        """Record a hash on the UNSCOPED rail's current drain window."""
        with self._rail_lock:
            self._merge_hash(self._loaded_hashes_window, name, content_hash)

    def _record_scoped_load(
        self, scope: str, name: str, content_hash: Optional[str] = None,
    ) -> None:
        """Note that THIS run loaded a body. Only reached from `load_skill`,
        and only after the body was actually served — a budget refusal or a
        not-found returns before here, so the rail never names a skill whose
        body did not reach the model.

        ``content_hash`` defaults to None so an older in-tree caller that passes
        two arguments still records the name; the hash rail then reports this
        load as version-unknown rather than dropping it.
        """
        with self._rail_lock:
            bucket = self._scoped_loaded_names.get(scope)
            if bucket is None:
                bucket = []
                self._scoped_loaded_names[scope] = bucket
            if name not in bucket:
                bucket.append(name)
            self._scoped_loaded_names.move_to_end(scope)
            hashes = self._scoped_loaded_hashes.get(scope)
            if hashes is None:
                hashes = {}
                self._scoped_loaded_hashes[scope] = hashes
            self._merge_hash(hashes, name, content_hash)
            self._scoped_loaded_hashes.move_to_end(scope)
            while len(self._scoped_loaded_names) > self._MAX_SCOPED_RAILS:
                self._evict_rail_overflow(self._scoped_loaded_names, "loaded")
            # Evicted on its own key order rather than following the names
            # store: the two are written together and so evict together, but
            # tying one's eviction to the other's popitem would drop a
            # DIFFERENT scope's hashes if they ever fell out of step.
            while len(self._scoped_loaded_hashes) > self._MAX_SCOPED_RAILS:
                self._scoped_loaded_hashes.popitem(last=False)

    def consume_loaded_names(self, scope: Optional[str] = None) -> List[str]:
        """Read + clear the names whose body `load_skill` served since the
        last drain. Adapter trace-send paths call this on their router
        singleton — the same instance whose load_skill tool served the
        bodies — and stamp the names onto ``skills_loaded_by_agent``.

        Args:
            scope: the RUN whose loads to drain — the same key that run passed
                to `load_skill`, or that an ambient resolver answered with on
                its behalf. Scoped drains see ONLY that run's loads, which is
                what a concurrent fanout needs. Omitting it keeps the legacy
                shared slot, whose known cost is that concurrent runs sharing
                one router all drain into whichever trace sends first.
        """
        if scope is not None:
            with self._rail_lock:
                drained = self._scoped_loaded_names.pop(scope, None)
            return list(drained) if drained else []
        with self._rail_lock:
            drained = list(self._loaded_names)
            self._loaded_names.clear()
        self._forget_unscoped_owners_if_drained()
        return drained

    def consume_loaded_hashes(
        self, scope: Optional[str] = None,
    ) -> Dict[str, Optional[str]]:
        """Read + clear the ``name -> content_hash`` map for the same loads
        :meth:`consume_loaded_names` drains, so an adapter can report the skill
        VERSION the model actually read.

        Deliberately a SEPARATE method rather than a wider return from
        `consume_loaded_names`: that one is drained by three adapters at five
        call sites and its list return is part of the shape older callers were
        written against. A new method leaves every one of them working
        untouched, and an adapter that never calls this simply keeps reporting
        `skill_hash: null` — which is what all of them did before.

        Pair it with the matching `consume_loaded_names(scope=...)` call. A
        scope whose names are drained but whose hashes are not leaves one small
        dict behind until the LRU reclaims it; it is bounded, never unbounded.

        A name maps to ``None`` when no hash was available, or when the same
        name was loaded twice at different versions against one rail — see
        `_merge_hash`.
        """
        if scope is not None:
            with self._rail_lock:
                drained_map = self._scoped_loaded_hashes.pop(scope, None)
            return dict(drained_map) if drained_map else {}
        with self._rail_lock:
            drained_map = dict(self._loaded_hashes_window)
            self._loaded_hashes_window.clear()
        return drained_map

    def _note_unscoped_writer(self, scope: Optional[str]) -> None:
        """Record who just wrote the unscoped rails. Takes `_rail_lock`.

        A write with no ambient scope records no owner: it came from outside
        any run the adapters can name, so it belongs to nobody in particular
        and stays claimable by whoever drains — the pre-scope behaviour, kept
        deliberately because there is no better answer available for it.
        """
        if scope is None:
            return
        with self._rail_lock:
            self._unscoped_rail_owners[scope] = None
            self._unscoped_rail_owners.move_to_end(scope)
            # Only reachable if nothing ever drains the unscoped rails; evict
            # oldest-first rather than grow. An evicted owner reads as
            # "unowned", which is the pre-scope behaviour for that entry.
            while len(self._unscoped_rail_owners) > self._MAX_SCOPED_RAILS:
                self._unscoped_rail_owners.popitem(last=False)  # provenance only, not data

    def _forget_unscoped_owners_if_drained(self) -> None:
        """Drop the provenance once the content it describes is gone.

        Self-maintaining: no adapter has to remember to clear it, and it can
        never describe rails that have already been handed to a trace.
        """
        with self._rail_lock:
            if not (
                self._loaded_names
                or self._offered_names_rail
                or self._delivered_names_rail
                or self._delivered_versions_rail
                or self._routing_id_rail
            ):
                self._unscoped_rail_owners.clear()

    def drain_unscoped_rails_for(
        self, scopes: Any, *, include_versions: bool = False,
    ) -> Any:
        """Take the unscoped rails ONLY if every writer is one of ``scopes``.

        Replaces a peek-then-drain pair that had two defects, both of which put
        one run's data on another run's trace — the thing the scoped rails exist
        to stop.

        * NOT ATOMIC. The caller asked who wrote the rails, then drained them in
          separate lock round-trips. A write landing in that window was taken by
          the draining trace while the ownership snapshot still said "clear".
        * UNOWNED READ AS OWNED. The check was ``owners - scopes``, which is
          empty both when every writer is ours AND when there is NO writer at
          all. A ``load_skill`` issued from a plain thread — outside any run the
          resolver can name — records no owner, so the next trace to drain
          claimed it as its own activation.

        Ownership is now REQUIRED, not merely un-contradicted: no owner means no
        claim. And when the rails are not ours they are LEFT IN PLACE rather
        than drained-and-discarded, because discarding is how the original bug
        lost the true activation from the run that earned it.
        """
        wanted = set(scopes or ())
        with self._rail_lock:
            owners = set(self._unscoped_rail_owners)
            # UNOWNED rails are still claimable, and that is a deliberate
            # trade rather than an oversight. Requiring an owner was tried and
            # reverted: the documented "assemble the fragment yourself, then
            # invoke" pattern calls build_prompt_fragment OUTSIDE any runnable,
            # so no resolver can name a run for it, and four regression tests —
            # guarding a real reported failure where the trace shipped a NULL
            # routing_id — went red.
            #
            # The residual is narrow and worth stating: a load_skill issued from
            # a plain thread, concurrent with a run, is unowned and the run may
            # claim it. Closing that means giving the outside-a-run pattern an
            # owner of its own, which is the real fix and is not this change.
            # Breaking a documented path to close a narrow race is the worse
            # trade.
            if owners - wanted:
                return (None, [], [], [], []) if include_versions else (None, [], [], [])
            routing_id = self._routing_id_rail
            offered = list(self._offered_names_rail or [])
            delivered = list(self._delivered_names_rail or [])
            loaded = list(self._loaded_names or [])
            # Legacy unowned name rails support assemble-then-invoke callers.
            # Immutable identities require an actual writer owned by this run;
            # a later invocation must not borrow another thread's body metadata.
            versions = [dict(v) for v in self._delivered_versions_rail] if owners else []
            self._routing_id_rail = None
            if self._offered_names_rail:
                self._offered_names_rail.clear()
            if self._delivered_names_rail:
                self._delivered_names_rail.clear()
            self._loaded_names.clear()
            self._delivered_versions_rail.clear()
            self._unscoped_rail_owners.clear()
        result = (routing_id, offered, delivered, loaded)
        return (*result, versions) if include_versions else result

    def unscoped_rail_owners(self) -> List[str]:
        """The scopes that wrote whatever is currently on the unscoped rails.

        A PEEK, not a drain — the caller reads it to decide whether the
        unscoped values are safe to use, and the answer must survive until it
        has actually drained them.
        """
        with self._rail_lock:
            return list(self._unscoped_rail_owners)

    def _scoped_rail(self, scope: str) -> Dict[str, Any]:
        """This run's routing-rail bucket, created on first write.
        Caller holds `_rail_lock`."""
        bucket = self._scoped_routing_rails.get(scope)
        if bucket is None:
            bucket = {"routing_id": None, "offered": [], "delivered": [], "versions": []}
            self._scoped_routing_rails[scope] = bucket
        self._scoped_routing_rails.move_to_end(scope)
        while len(self._scoped_routing_rails) > self._MAX_SCOPED_RAILS:
            self._evict_rail_overflow(self._scoped_routing_rails, "routing")
        return bucket

    def _drop_scoped_rail_if_spent(self, scope: str) -> None:
        """Forget a run's routing rail once all three fields are drained.
        Caller holds `_rail_lock`. Without this a run whose routing_id was
        read but whose (empty) name lists never were would keep its entry
        forever, which is a per-run leak in a long-lived server."""
        bucket = self._scoped_routing_rails.get(scope)
        if bucket is not None and not any(bucket.values()):
            self._scoped_routing_rails.pop(scope, None)

    def _record_routing_rails(
        self,
        routing_id: Optional[str],
        offered_names: Optional[List[str]],
        delivered_names: Optional[List[str]],
        scope: Optional[str] = None,
        delivered_versions: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """Mirror one routing decision onto the instance rails.

        Names accumulate (dedup-append, like `_loaded_names`) because a
        multi-LLM-call turn routes more than once and the adapter drains
        only at trace-send; `routing_id` is last-write-wins, matching the
        contextvar rails the adapters set per call.

        Written twice: once onto the process-wide slot every existing
        adapter drains, and — when the run can be named — once more under
        that run's own scope. The scoped copy is what lets a concurrent
        adapter answer "what did THIS run route" instead of "what did the
        singleton see most recently".
        """
        if routing_id:
            self._routing_id_rail = routing_id
        for name in offered_names or ():
            if name not in self._offered_names_rail:
                self._offered_names_rail.append(name)
        for name in delivered_names or ():
            if name not in self._delivered_names_rail:
                self._delivered_names_rail.append(name)

        # Resolve OUTSIDE the lock: a resolver is adapter code.
        rail_scope = scope if scope is not None else _ambient_scope()
        if routing_id or offered_names or delivered_names:
            # Only a write earns an owner — a routing call that surfaced
            # nothing put nothing on the unscoped rails to be claimed.
            self._note_unscoped_writer(rail_scope)
        if rail_scope is None:
            self._record_delivery_versions(delivered_versions or [], scope=None)
            return
        with self._rail_lock:
            bucket = self._scoped_rail(rail_scope)
            if routing_id:
                bucket["routing_id"] = routing_id
            for name in offered_names or ():
                if name not in bucket["offered"]:
                    bucket["offered"].append(name)
            for name in delivered_names or ():
                if name not in bucket["delivered"]:
                    bucket["delivered"].append(name)
        self._record_delivery_versions(delivered_versions or [], scope=rail_scope)

    def _record_delivery_versions(self, versions: List[Dict[str, str]], *, scope: Optional[str]) -> None:
        if not versions:
            return
        self._note_unscoped_writer(scope)
        with self._rail_lock:
            for version in versions:
                if version not in self._delivered_versions_rail:
                    self._delivered_versions_rail.append(dict(version))
                    if len(self._delivered_versions_rail) > self._MAX_DELIVERED_VERSIONS:
                        self._delivered_versions_rail.pop(0)
                        logger.warning("skill version rail overflow: oldest unscoped witness evicted")
                if scope is not None:
                    bucket = self._scoped_rail(scope)
                    if version not in bucket["versions"]:
                        bucket["versions"].append(dict(version))
                        if len(bucket["versions"]) > self._MAX_DELIVERED_VERSIONS:
                            bucket["versions"].pop(0)
                            logger.warning("skill version rail overflow: oldest scoped witness evicted")

    def consume_delivered_versions(self, scope: Optional[str] = None) -> List[Dict[str, str]]:
        """Drain immutable bodies for this run, including actual load_skill serves.

        Keep the same scope as the names drain. The per-call contextvar remains
        preferable when it propagates; this mirror survives copied callbacks.
        """
        with self._rail_lock:
            if scope is not None:
                bucket = self._scoped_routing_rails.get(scope)
                if bucket is None:
                    return []
                result = [dict(v) for v in bucket["versions"]]
                bucket["versions"] = []
                # Scoped copies remain authoritative for every other live run.
                # Release this mirror too: adapters that consume per-call rails
                # otherwise retain every completed body in a singleton forever.
                self._delivered_versions_rail[:] = [
                    v for v in self._delivered_versions_rail if v not in result
                ]
                self._drop_scoped_rail_if_spent(scope)
            else:
                result = [dict(v) for v in self._delivered_versions_rail]
                self._delivered_versions_rail.clear()
        self._forget_unscoped_owners_if_drained()
        return result

    def consume_routing_id(self, scope: Optional[str] = None) -> Optional[str]:
        """Read + clear the most recent routing id minted since the last
        drain. Adapters fall back to this when their own routing-id
        contextvar comes back empty because prompt assembly ran in a
        copied context.

        Args:
            scope: the RUN whose decision to drain. Unscoped keeps the legacy
                shared slot and its concurrency cost — the first drainer takes
                it, whoever minted it.
        """
        if scope is not None:
            with self._rail_lock:
                bucket = self._scoped_routing_rails.get(scope)
                if bucket is None:
                    return None
                rid = bucket["routing_id"]
                bucket["routing_id"] = None
                self._drop_scoped_rail_if_spent(scope)
            return rid
        rid = self._routing_id_rail
        self._routing_id_rail = None
        self._forget_unscoped_owners_if_drained()
        return rid

    def consume_offered_names(self, scope: Optional[str] = None) -> List[str]:
        """Read + clear the names offered (menu rows) since the last drain.
        The instance-state twin of `consume_last_offered_names`. See
        `consume_routing_id` for what `scope` buys.
        """
        if scope is not None:
            with self._rail_lock:
                bucket = self._scoped_routing_rails.get(scope)
                if bucket is None:
                    return []
                drained = list(bucket["offered"])
                bucket["offered"] = []
                self._drop_scoped_rail_if_spent(scope)
            return drained
        drained = list(self._offered_names_rail)
        self._offered_names_rail.clear()
        self._forget_unscoped_owners_if_drained()
        return drained

    def consume_delivered_names(self, scope: Optional[str] = None) -> List[str]:
        """Read + clear the names whose BODY the prompt fragment carried
        since the last drain. The instance-state twin of
        `consume_last_delivered_names`. See `consume_routing_id` for what
        `scope` buys.
        """
        if scope is not None:
            with self._rail_lock:
                bucket = self._scoped_routing_rails.get(scope)
                if bucket is None:
                    return []
                drained = list(bucket["delivered"])
                bucket["delivered"] = []
                self._drop_scoped_rail_if_spent(scope)
            return drained
        drained = list(self._delivered_names_rail)
        self._delivered_names_rail.clear()
        self._forget_unscoped_owners_if_drained()
        return drained

    def get_menu_prompt(
        self,
        query: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Convenience: get just the prompt fragment.

        If query is provided and strategy is 'auto' or 'semantic',
        uses smart routing. Otherwise returns full menu.
        """
        if query and self.strategy in ("auto", "semantic"):
            result = self.smart_route(query, **kwargs)
        else:
            result = self.get_menu(**kwargs)
        return result.get("prompt_fragment", "")

    def build_prompt_fragment(
        self,
        query: Optional[str] = None,
        *,
        agent_name: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = 10,
        bypass_cache: bool = False,
        inject_body: Optional[bool] = None,
        scope: Optional[str] = None,
        _want_parts: bool = False,
    ) -> tuple[str, Optional[str]]:
        """Return ``(prompt_fragment, routing_id)`` — the primitive every adapter calls.

        This is the single entry point framework adapters (every
        ``agent_config()`` through :meth:`build_prompt_parts`, langchain, crewai)
        use to load skills into a prompt.
        Returns a tuple instead of just the string so the adapter can
        thread ``routing_id`` into the resulting trace — without that, the
        ``routing_decision × trace_skill_activation`` join can't close,
        and you lose the offered-vs-activated signal that drives skill
        effectiveness scoring.

        Strategy resolution:
          - ``query`` is None or ``self.strategy == "menu"`` → full menu
            (every active skill, name+description only)
          - ``query`` provided and ``self.strategy`` is ``"auto"`` or
            ``"semantic"`` → smart routing (semantic search + perf rerank)

        Caching:
          - Results are cached for ~30s keyed by
            ``(query, agent_name, category, top_k, strategy)``.
          - A multi-LLM-call agent loop within a single user turn calls
            this method N times with the same arguments; the cache makes
            only the first call hit the network. Critically, every
            cached return carries the **same** ``routing_id``, so the
            platform sees exactly one ``RoutingDecision`` per turn (not
            N) and the offered-vs-activated join stays accurate.
          - Pass ``bypass_cache=True`` for replay / regression scenarios
            where you want a fresh routing decision regardless.

        Errors are swallowed: a failed platform call returns
        ``("", None)`` so prompt assembly never blocks on a transient
        network blip. The warning is logged and ``routing_status()`` turns
        unhealthy. A SLOW failure (a timeout, an admission wait) is cached so
        the other LLM calls of the turn do not each pay that time again
        (until 2026-10-14 every call retried). Under ``scope=`` it is kept
        for the turn (the cache TTL) and a new run retries at once. Without
        a scope the slot is shared by every caller asking the same query as
        the same agent, so it is kept only ``_UNSCOPED_FAILURE_TTL_S`` (5 s),
        long enough for the turn's next model call and not long enough to
        strip skills from other conversations. Another query retries; so
        does ``bypass_cache=True``. A fast failure (an edge 429, a 5xx, a
        401) is not cached: retrying it costs nothing, and the circuit
        breaker stops a run of them. Nor is an open breaker: it costs nothing
        and should stop answering for the run the moment the breaker closes.

        Bodies are injected for a result the platform ranked by THIS query
        (see ``_ranked_by_this_query``) — degraded retrieval included, when
        its rows still matched the query. From any other result only the
        bodies no ranking could change are injected: offered
        ``priority_skills``, and the agent's only skill (see
        ``_bodies_without_ranking``). Otherwise the menu goes in alone.

        Args:
            query: The user message — used for semantic routing when
                strategy allows. None forces full-menu mode.
            agent_name: Overrides the instance ``agent_name`` for this
                call only (used by adapters that know the agent at
                runtime but not at construction).
            category: Optional category filter passed through.
            top_k: Max skills to surface (smart_route only).
            bypass_cache: When True, ignore the cached value and refetch.
                The fresh result still populates the cache for
                subsequent calls.
            scope: an opaque key naming the RUN this fragment is being
                built for (the adapter's trace id). The per-turn body-load
                budget this call resets is then kept per run, so two
                concurrent runs sharing one router singleton no longer
                overwrite each other's budget. See `load_skill`.

        Returns:
            ``(prompt_fragment, routing_id)``. ``routing_id`` is the
            ``rt_<24-hex>`` ID the platform assigned to this routing
            decision, or None if the call failed.
        """
        effective_agent = agent_name or self.agent_name
        effective_inject = self.inject_body if inject_body is None else inject_body
        cache_key = (
            query or "",
            effective_agent or "",
            category or "",
            top_k,
            self.strategy,
            bool(effective_inject),  # body-injected vs menu-only must not share a cache slot
            # Two concurrent runs must not share a cache slot, because sharing
            # one means sharing its routing_id — one routing decision claimed by
            # two runs, which double-counts it in the effectiveness join. The
            # case is real for full-menu adapters: they route with query=None
            # under one agent name, so without this every concurrent run of that
            # agent collides on a single key. Unscoped callers pass None -> ""
            # -> today's key, so their behaviour and routing-call volume are
            # unchanged.
            scope or "",
            # The split path sets `skills_offered_in_prompt` to the whole menu
            # (contract §2: what the model COULD see), while the fragment path
            # sets it to the router's pick. Same query, different telemetry —
            # so they cannot share a cache slot without one silently
            # overwriting the other's rails.
            bool(_want_parts),
        )

        if not bypass_cache:
            cached = self._fragment_cache.get(cache_key)
            if cached is not None:
                (fragment, routing_id, offered_names, delivered_names, split_parts,
                 delivered_versions, menu_truncated) = cached
                # The hit re-offers the same decision, so this run's loads name
                # it — the cache key carries the scope, so the miss filed it
                # under this run already unless the bounded store evicted it.
                self._remember_run_routing(scope, routing_id)
                # A cache hit must re-emit the split exactly as the miss did.
                # Without this, the second LLM call of a turn loses the prefix
                # and the adapter falls back to the fragment — the injected
                # text would change mid-turn, which is the one thing a
                # "byte-identical prefix" must never do.
                _last_split_parts_ctx.set(split_parts)
                _last_menu_truncated_ctx.set(menu_truncated)
                # A cache hit must re-emit the telemetry the miss path set:
                # a second turn inside the 30s window would otherwise lose
                # offered/delivered names on every rail (the first turn's
                # consume_* drained the contextvars). Unconditional set —
                # an empty rail must CLEAR any stale unconsumed names from
                # a previous call — the rails are per-call, never sticky.
                _last_offered_names_ctx.set(
                    list(offered_names) if offered_names else None
                )
                _last_delivered_names_ctx.set(
                    list(delivered_names) if delivered_names else None
                )
                _last_delivered_versions_ctx.set(
                    [dict(version) for version in delivered_versions] or None
                )
                self._record_routing_rails(
                    routing_id, offered_names, delivered_names, scope=scope,
                    delivered_versions=delivered_versions,
                )
                _stamp_active_trace(routing_id, offered_names, delivered_names, delivered_versions)
                return fragment, routing_id

        # Reset both rails up front so a call that offers or
        # delivers nothing can't inherit a previous call's unconsumed
        # names (producers without a consumer exist — e.g. the generic
        # quickstart never drains the rails).
        _last_offered_names_ctx.set(None)
        _last_delivered_names_ctx.set(None)
        _last_delivered_versions_ctx.set(None)
        _last_split_parts_ctx.set(None)
        _last_menu_truncated_ctx.set(None)

        # How long the routing call took: a failure is cached only when
        # retrying it would cost the turn that time again (see below).
        route_started = time.monotonic()
        if query and self.strategy in ("auto", "semantic") and _route_paused(self.base_url, self.api_key):
            # A route failed moments ago and the menu answered: go straight to
            # the menu until the pause ends (see `_ROUTE_PAUSE_S`).
            result = self._fallback_menu("route_paused", category=category, agent_name=effective_agent)
        elif query and self.strategy in ("auto", "semantic"):
            try:
                result = self.smart_route(
                    query,
                    top_k=top_k,
                    category=category,
                    # The route inlines up to 100 KB of scripts per skill when
                    # asked, and nothing on this path reads them — they cost
                    # transfer inside the 5 s budget and were thrown away.
                    # Disk export fetches attachments through its own routes.
                    include_attachments=False,
                    agent_name=effective_agent,  # pass the per-call override through
                )
            except SkillRouterError as e:
                logger.warning("build_prompt_fragment smart_route failed: %s", e)
                result = {"skills": [], "prompt_fragment": "", "strategy": "none",
                          "degraded_reason": _empty_menu_reason(e)}
            failure = result.get("degraded_reason") if _routing_call_failed(result) else None
            if failure in _MENU_FALLBACK_REASONS:
                fallback = self._fallback_menu(failure, category=category, agent_name=effective_agent)
                if not _routing_call_failed(fallback):
                    logger.warning(
                        "Skill routing failed (%s); this turn gets the agent's skills unranked, "
                        "and routes from this key are skipped for %.0f s.", failure, _ROUTE_PAUSE_S,
                    )
                    _pause_route(self.base_url, self.api_key)
                    result = fallback
        else:
            try:
                result = self.get_menu(category=category, agent_name=effective_agent, scope=scope)
            except SkillRouterError as e:
                logger.warning("build_prompt_fragment get_menu failed: %s", e)
                result = {"skills": [], "prompt_fragment": "", "strategy": "none",
                          "degraded_reason": _empty_menu_reason(e)}
        routing_failed = _routing_call_failed(result)
        route_elapsed = time.monotonic() - route_started

        fragment = result.get("prompt_fragment", "") or ""
        routing_id = result.get("routing_id")
        # Expose the offered skill names via a contextvar so
        # framework adapters can stamp them onto RunTrace.skills_offered_in_prompt
        # without having to extend this method's return signature.
        offered_names = [
            name for s in (result.get("skills") or [])
            if isinstance(s, dict) and isinstance(name := s.get("name"), str) and name
        ]

        # The router's own ranked pick, kept separate from `offered_names`
        # because the split widens that to the whole menu. Bodies are chosen by
        # RELEVANCE and must keep coming from the ranked list — iterating the
        # menu instead would fetch bodies in menu order, i.e. arbitrarily.
        routed_names = list(offered_names)

        # The split's two pre-rendered strings. Computed before the body block
        # because bodies ride the PREFIX (contract invariant 5).
        split_prefix, split_tail = ("", "")
        if _want_parts:
            split_prefix, split_tail = self._split_parts(result)

        # Contract §2: on the split path the model can reach for anything in
        # the menu, so that is what gets reported as offered in the prompt.
        # `routing_decision.offered_skill_names` on the SERVER is untouched and
        # still records the router's own pick — `acceptance_rate` divides by
        # that one, so widening here does not move it.
        if _want_parts and split_prefix:
            menu_names = [
                n for n in (result.get("stable_menu_skills") or [])
                if isinstance(n, str) and n
            ]
            if menu_names:
                offered_names = menu_names

        if offered_names:
            _last_offered_names_ctx.set(offered_names)

        # Deliver the actual skill BODY (the knowledge K), not just the menu row, so the
        # skill's runtime value reaches the agent. We fetch the body of the top-k routed skills and
        # append it. This rides the same 30s fragment cache, so within one user turn the body is
        # fetched once (one get_skill_body call), not per LLM call. A missing body degrades
        # gracefully to the menu-only fragment.
        # Only inject when the PLATFORM says the result is ranked by this
        # query. Asking for a route was never enough: the platform answers a
        # route with its effectiveness-ordered menu for small workspaces and on
        # every retrieval fallback, and when no query term matched, the top row
        # of that is not about this query — "thanks!" got the best-rated
        # skill's body. A degraded result whose rows did match is ranked.
        # Fetch with a bounded character cap, but inject only complete bodies.
        # Partial server responses and oversized old-server bodies are omitted;
        # the same total token budget applies to the first and later bodies.
        #
        # Two kinds of body need no ranking, because there is no wrong one to
        # pick (see `_bodies_without_ranking`): the agent's ONLY skill, and the
        # `priority_skills` the developer reserved a slot for. Without them a
        # small workspace's reworded question ("I want my money back" against
        # "refund-policy: Refund rules" shares no word, so the platform says
        # unranked) left the injection-only adapters a menu row the model
        # cannot open.
        routed_for_query = bool(query) and self.strategy in ("auto", "semantic")
        prioritized = [name for name in self.priority_skills if name in routed_names]
        if routed_for_query and _ranked_by_this_query(result):
            body_names = prioritized + [name for name in routed_names if name not in prioritized]
        elif routed_for_query:
            body_names = _bodies_without_ranking(result, routed_names, prioritized)
        else:
            body_names = []
        delivered_names: List[str] = []
        delivered_versions: List[Dict[str, str]] = []
        if effective_inject and routed_names and body_names:
            bodies = []
            body_tokens = 0
            count_cap = min(self.inject_body_top_k, self.max_loaded_bodies)
            for name in body_names[:count_cap]:
                body_options: Dict[str, Any] = {"max_chars": self.per_body_char_limit}
                if effective_agent is not None:
                    body_options["agent_name"] = effective_agent
                # A stand-in may override get_skill_body without clearing its
                # thread-local handoff. Never inherit a previous fetch's hash.
                _body_hash_tls.value = None
                _body_hash_tls.partial = None
                _body_hash_tls.identity = None
                body = self.get_skill_body(name, **body_options)
                content_hash = getattr(_body_hash_tls, "value", None)
                partial = getattr(_body_hash_tls, "partial", None)
                if partial is True or (partial is None and _body_is_known_partial(body)):
                    continue
                if not (body and body.strip()):
                    continue
                body = body.strip()
                if len(body) > self.per_body_char_limit:
                    continue
                tokens = estimate_tokens(body)
                if body_tokens + tokens > self.body_token_budget:
                    break
                body_tokens += tokens
                bodies.append(f"## Skill: {name}\n\n{body}")
                # The body actually reached the prompt → delivered.
                delivered_names.append(name)
                if isinstance(content_hash, str) and content_hash:
                    witness = {"name": name, "hash": content_hash}
                    witness.update(getattr(_body_hash_tls, "identity", None) or {})
                    if isinstance(routing_id, str) and routing_id:
                        witness["routing_id"] = routing_id
                    delivered_versions.append(witness)
            if bodies:
                body_block = "\n\n".join(bodies)
                fragment = f"{fragment}\n\n{body_block}" if fragment else body_block
                # Invariant 5: bodies ride the PREFIX. The tail does not exist
                # on openai_agents / pydantic_ai, and a body placed there is
                # dropped silently while still counted as delivered.
                if split_prefix:
                    split_prefix = f"{split_prefix}\n\n{body_block}"
                _last_delivered_names_ctx.set(list(delivered_names))
                _last_delivered_versions_ctx.set(
                    [dict(version) for version in delivered_versions] or None
                )

        # Invariant 6: delivered ⊆ offered ⊆ the injected text.
        #
        # A skill whose BODY was delivered can legitimately sit outside the
        # menu: bodies come from the router's ranked pick, and the menu is
        # capped and unions back only the router's top few picks (contract
        # §3.3b; an older server, only the HINTED names). Its `## Skill: <name>`
        # header is in the prefix either way, so the name genuinely is in the
        # injected text — it is `offered` that would be understating what the
        # model can see. Unioning every routed row on the server instead would
        # make the prefix vary with the whole routed list, not its top few
        # (invariant 2).
        if _want_parts and split_prefix and delivered_names:
            for name in delivered_names:
                if name not in offered_names:
                    offered_names.append(name)
            _last_offered_names_ctx.set(list(offered_names))
            self._record_routing_rails(
                routing_id, offered_names, delivered_names, scope=scope,
            )

        # A fresh fragment marks a new turn: reset the on-demand body-load
        # budget so load_skill's caps apply per turn, not per process.
        # Cache hits (multi-LLM-call loops within one turn) keep the
        # existing budget — that is exactly the window the caps guard.
        fresh_budget = _BodyLoadBudget(
            self.max_loaded_bodies, self.body_token_budget, self.body_load_deadline_s,
        )
        _body_budget_ctx.set(fresh_budget)
        self._remember_budget(fresh_budget, scope)

        # The decision this run's load_skill calls name, so the platform can
        # tell a skill this run was shown from a public name it never offered.
        # A failed call files "" — the run was offered nothing.
        self._remember_run_routing(scope, routing_id)

        # Mirror onto the instance rails before stamping — the adapters
        # whose contextvars never reach trace-send read the routing
        # telemetry off the router itself.
        self._record_routing_rails(
            routing_id, offered_names, delivered_names, scope=scope,
            delivered_versions=delivered_versions,
        )

        # Stamp the active generic trace (raw-loop quickstart) —
        # adapter paths stamp their own trace objects, so this is a no-op
        # or an idempotent double-set there.
        _stamp_active_trace(routing_id, offered_names, delivered_names, delivered_versions)

        # Cache the result, and a failed call when it was SLOW. A slow failure
        # used to be retried by every LLM call of the turn — each one paying
        # the full 5 s timeout (or an admission wait) until three of them
        # opened the breaker. Cached, the turn pays it once; the next run
        # (another scope) or query asks again. A FAST failure is not cached
        # (see `_failure_worth_caching`): retrying it costs nothing. Without a
        # scope the slot belongs to nobody in particular — every conversation
        # asking this query as this agent reads it, LangChain's included — so
        # a slow failure there is held only for `_UNSCOPED_FAILURE_TTL_S`, not
        # the 30 s that took skills from every other conversation. The cached
        # value carries offered/delivered so cache hits can re-emit them; the
        # public return stays a 2-tuple.
        split_parts = (split_prefix, split_tail) if split_prefix else None
        _last_split_parts_ctx.set(split_parts)
        menu_truncated = result.get("truncated") if isinstance(result, dict) else None
        if not isinstance(menu_truncated, bool):
            # Over the menu caps the platform answers by retrieval for THIS
            # message ("smart_routing"): a fixed head of the eligible set plus the
            # router's top picks for the message, with no `truncated` field. Which rows
            # appear depends on the message, which is what the flag reports.
            menu_truncated = (
                True if isinstance(result, dict) and result.get("strategy") == "smart_routing"
                else None
            )
        _last_menu_truncated_ctx.set(menu_truncated)

        cache_failure = routing_failed and _failure_worth_caching(result, route_elapsed)
        if fragment or routing_id or cache_failure:
            failure_only = cache_failure and not (fragment or routing_id)
            self._fragment_cache.set(
                cache_key,
                (fragment, routing_id, offered_names, delivered_names, split_parts,
                 delivered_versions, menu_truncated),
                ttl_seconds=_UNSCOPED_FAILURE_TTL_S if failure_only and not scope else None,
            )
        return fragment, routing_id

    # ── The prefix/tail split ────────────────────────────────────────
    #
    # See `platform/docs/contracts/skill_route_split.md`. `stable_menu`,
    # `menu_instruction` and `routing_hint` are PRE-RENDERED STRINGS. They are
    # injected verbatim: this method never re-renders, re-sorts, or re-derives
    # them, and in particular never re-writes the hint sentence — the server
    # owns that wording, and two copies of it drift.

    def _split_parts(self, result: Dict[str, Any]) -> tuple[str, str]:
        """``(prefix, tail)`` from a routing result, or ``("", "")``.

        A server that does not send `stable_menu` has not been upgraded. The
        caller falls back to `prompt_fragment`, which is unchanged and still
        correct — degrading to today's behaviour, never to an empty prompt.
        That distinction is the whole lesson of the second attempt, where a
        shape mismatch normalised to an empty list and no-oped in silence.
        """
        menu = result.get("stable_menu")
        hint = result.get("routing_hint")
        instruction = result.get("menu_instruction")
        # Strings, or this server does not speak the split. A list here is an
        # older/newer shape, NOT something to coerce.
        if not isinstance(menu, str) or not menu.strip():
            return "", ""
        if not isinstance(hint, str):
            hint = ""
        if not isinstance(instruction, str):
            instruction = ""
        prefix = menu.rstrip("\n")
        if instruction.strip():
            prefix = f"{prefix}\n\n{instruction.strip()}"
        return prefix, hint.strip()

    def build_prompt_parts(
        self,
        query: Optional[str] = None,
        *,
        agent_name: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = 10,
        bypass_cache: bool = False,
        inject_body: Optional[bool] = None,
        scope: Optional[str] = None,
    ) -> tuple[str, str, Optional[str]]:
        """Return ``(prefix, tail, routing_id)`` — the cacheable split.

        `prefix` is byte-identical turn to turn for a given eligible set, so an
        adapter can put it where the provider will cache it. `tail` is the one
        sentence that depends on THIS query, and goes next to the query.

        Skill BODIES ride the prefix, never the tail. Adapters differ in
        whether they have a tail position at all (openai_agents and
        pydantic_ai do not), and a body placed in a tail that does not exist is
        dropped while still being counted as delivered — reported reach that
        never happened.

        Falls back to ``(fragment, "", routing_id)`` against a server that does
        not send the split, so an older platform degrades to today's behaviour
        rather than to an empty prompt.
        """
        fragment, routing_id = self.build_prompt_fragment(
            query, agent_name=agent_name, category=category, top_k=top_k,
            bypass_cache=bypass_cache, inject_body=inject_body, scope=scope,
            _want_parts=True,
        )
        parts = _last_split_parts_ctx.get()
        if parts is None:
            return fragment, "", routing_id
        prefix, tail = parts
        return prefix, tail, routing_id

    def sync_skills(
        self,
        skills: List[Dict[str, Any]],
        author: Optional[str] = None,
        conflict_policy: str = "local_wins",
        response_mode: str = "summary",
        install_id: Optional[str] = None,
        install_label: Optional[str] = None,
        *,
        offer_scope: Optional[str] = None,
        agent_name: Optional[str] = None,
        add_to_agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Push local skills to the platform (bulk upsert).

        Args:
            skills: List of dicts with name, body_markdown, and optionally
                description, category, trigger_phrases, frontmatter,
                content_hash, local_updated_at.
            author: Display name recorded on every SkillVersion created
                by this sync (e.g. CI runner identity).
            conflict_policy: How to resolve hash mismatches.
                - ``"local_wins"`` (default for the SDK): always overwrite
                  remote. Right for CI / build artifacts where the repo
                  is the source of truth.
                - ``"newer_wins"``: compare timestamps (needs
                  ``local_updated_at`` on each skill).
                - ``"remote_wins"``: never overwrite; conflicts surface
                  as a ``pulled`` count.
            response_mode: ``"summary"`` (default; returns counts) or
                ``"diff"`` (returns a per-skill ``actions`` array with
                bodies for any ``pulled`` entries).
            offer_scope: Which agents get the skills this sync CREATES
                (existing skills keep theirs): ``"restricted"``, only the agents
                you add them to; ``"workspace"``, every agent in the org. Leave
                it out and they reach every agent for now; the platform is
                making new skills opt-in, so pass it to keep a behavior.
            agent_name: Add the synced skills to this agent. New ones are then
                created ``"restricted"`` (only this agent) unless
                ``offer_scope`` says otherwise.
            add_to_agent: With ``agent_name``: ``"all"`` (the server's default)
                adds every synced skill; ``"created"`` adds only the ones this
                sync created, so a skill removed from the agent on the dashboard
                is not added back. The adapters' boot sync uses ``"created"``.

        Returns:
            Dict with ``created`` / ``updated`` / ``unchanged`` / ``pulled`` /
            ``failures`` counts, plus ``actions`` if ``response_mode="diff"``.
        """
        payload: Dict[str, Any] = {
            "skills": skills,
            "conflict_policy": conflict_policy,
            "response_mode": response_mode,
        }
        if author:
            payload["author"] = author
        # Per-install divergence: when an install_id is supplied the
        # backend records this checkout's synced baseline. Omitting it keeps
        # the legacy stateless behavior.
        if install_id:
            payload["install_id"] = install_id
            if install_label:
                payload["install_label"] = install_label
        # A backend that predates these fields ignores them (the skills still sync).
        if offer_scope is not None:
            payload["offer_scope"] = offer_scope
        if agent_name:
            payload["agent_name"] = agent_name
            if add_to_agent:
                payload["add_to_agent"] = add_to_agent

        return self._request("POST", "/api/v1/skills/sync", json=payload)

    # ── CRUD ─────────────────────────────────────────────

    def list_skills(
        self,
        *,
        category: Optional[str] = None,
        stability: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all skills in the org.

        Args:
            category: Filter by category.
            stability: Filter by stability level.

        Returns:
            List of skill descriptors.
        """
        params: Dict[str, Any] = {}
        if category:
            params["category"] = category
        if stability:
            params["stability"] = stability
        result = self._request("GET", "/api/v1/skills", params=params)
        return result.get("skills", [])

    def get_skill(self, name: str) -> Dict[str, Any]:
        """Get a single skill by name.

        Returns:
            Skill descriptor with all metadata.
        """
        return self._request("GET", _skill_path(name))

    def create_skill(
        self,
        name: str,
        description: str,
        body_markdown: str,
        *,
        display_name: Optional[str] = None,
        category: Optional[str] = None,
        stability: str = "stable",
        trigger_phrases: Optional[List[str]] = None,
        offer_scope: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new skill on the platform.

        Args:
            name: Unique skill name (slug format).
            description: Short description.
            body_markdown: Full skill body in markdown.
            display_name: Optional human title shown in the registry instead of
                the slug `name`. When omitted the registry humanizes the slug.
            category: Optional category grouping.
            stability: One of 'stable', 'experimental', 'deprecated'.
            trigger_phrases: Optional list of trigger phrases.
            offer_scope: Which agents get the skill. ``"restricted"``: only
                the agents you add it to (:meth:`assign_skill`, or
                ``agent_name`` here). ``"workspace"``: every agent in the org.
                Leave it out and an API key still gets ``"workspace"`` today;
                the platform is making new skills opt-in (``"restricted"``), so
                pass it to keep a behavior. Set here, the skill is never offered
                org-wide, not even until a later :meth:`update_skill`.
            agent_name: Also add the new skill to this agent. Implies
                ``offer_scope="restricted"`` unless you say otherwise.

        Returns:
            Created skill descriptor: ``skill_id``, ``name``, ``version``,
            ``offer_scope`` and, with ``agent_name``, ``added_to_agent``.
        """
        payload: Dict[str, Any] = {
            "name": name,
            "description": description,
            "body_markdown": body_markdown,
            "stability": stability,
        }
        if display_name:
            payload["display_name"] = display_name
        if category:
            payload["category"] = category
        if trigger_phrases:
            payload["trigger_phrases"] = trigger_phrases
        if offer_scope is not None:
            payload["offer_scope"] = offer_scope
        if agent_name:
            payload["agent_name"] = agent_name
        result = self._request("POST", "/api/v1/skills", json=payload)
        wanted = offer_scope if offer_scope is not None else ("restricted" if agent_name else None)
        if wanted is not None and result.get("offer_scope") != wanted:
            # An older backend ignores the field and does not echo it: set it
            # the way callers had to before.
            self.update_skill(result["skill_id"], offer_scope=wanted)
            result = {**result, "offer_scope": wanted}
        if agent_name and not result.get("added_to_agent"):
            # An older backend ignores `agent_name` too: add it the same way.
            self.assign_skill(agent_name, result["skill_id"])
            result = {**result, "added_to_agent": agent_name}
        return result

    def update_skill(
        self,
        skill_id: str,
        *,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        body_markdown: Optional[str] = None,
        category: Optional[str] = None,
        stability: Optional[str] = None,
        trigger_phrases: Optional[List[str]] = None,
        is_active: Optional[bool] = None,
        offer_scope: Optional[str] = None,
        change_summary: Optional[str] = None,
        author: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update an existing skill.

        Args:
            skill_id: The skill UUID.
            description: New description.
            body_markdown: New body (creates a new version if changed).
            category: New category.
            stability: New stability level.
            trigger_phrases: New trigger phrases.
            is_active: Set active/inactive.
            offer_scope: Which agents get the skill. ``"workspace"`` is every
                agent in the org; ``"restricted"`` is only the agents it has
                been assigned to (:meth:`assign_skill`). Metadata only: it does
                not create a new version, but each agent whose list changes
                gets a new setup version on the platform.
            change_summary: Description of body changes (for version history).
            author: Author of the change.

        Returns:
            Updated skill descriptor.
        """
        payload: Dict[str, Any] = {}
        if display_name is not None:
            payload["display_name"] = display_name
        if description is not None:
            payload["description"] = description
        if body_markdown is not None:
            payload["body_markdown"] = body_markdown
        if category is not None:
            payload["category"] = category
        if stability is not None:
            payload["stability"] = stability
        if trigger_phrases is not None:
            payload["trigger_phrases"] = trigger_phrases
        if is_active is not None:
            payload["is_active"] = is_active
        if offer_scope is not None:
            payload["offer_scope"] = offer_scope
        if change_summary is not None:
            payload["change_summary"] = change_summary
        if author is not None:
            payload["author"] = author
        return self._request("PUT", _skill_path(skill_id), json=payload)

    def delete_skill(self, skill_id: str) -> Dict[str, Any]:
        """Deactivate (soft-delete) a skill.

        Args:
            skill_id: The skill UUID.
        """
        return self._request("DELETE", _skill_path(skill_id))

    # ── Assignment ───────────────────────────────────────

    def assign_skill(
        self,
        agent_name: str,
        skill_id: str,
        *,
        pin_version_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assign a skill to one agent, so the router offers it to that agent.

        This is how an ``offer_scope="restricted"`` skill reaches an agent. A
        ``"workspace"`` skill is offered to every agent already; assigning it
        ranks it ahead of equally relevant unassigned skills for this agent and
        lets you pin a version. Safe to repeat: assigning again only updates
        the pin.

        Args:
            agent_name: The name your code runs the agent under (``agent_config(name)``,
                ``Agent(name=...)``). An agent DecimalAI has not seen yet is created.
            skill_id: The skill's UUID: one of your organization's skills, or a
                public registry skill, which is then linked for this agent only.
            pin_version_id: Serve this skill version instead of the latest.

        Returns:
            ``{"status": "ok", "action": "assigned" | "updated",
            "subscription_id": ..., "pinned_version_id": ...}``.
        """
        return self._request(
            "POST", _agent_skill_path(agent_name, skill_id),
            params={"pin_version_id": pin_version_id},
            # Idempotent (see above), so a dropped connection or a read timeout
            # is retried once, like a read.
            replay_safe=True,
        )

    def unassign_skill(self, agent_name: str, skill_id: str) -> Dict[str, Any]:
        """Remove a skill assignment from one agent (needs a key with admin permission).

        A ``"restricted"`` skill is no longer offered to the agent; a
        ``"workspace"`` one still is, as to every agent.

        Returns:
            ``{"status": "ok", "action": "unassigned"}``, or ``"noop"`` when
            there was no assignment to remove.
        """
        return self._request(
            # Retried once like a read: a repeat after the first one landed
            # answers "noop", which leaves the same state.
            "DELETE", _agent_skill_path(agent_name, skill_id), replay_safe=True,
        )

    # ── Versions ─────────────────────────────────────────

    def list_versions(self, name: str) -> List[Dict[str, Any]]:
        """List all versions of a skill.

        Args:
            name: The skill name.

        Returns:
            List of version descriptors, newest first.
        """
        result = self._request("GET", _skill_path(name, "/versions"))
        return result.get("versions", [])

    def get_version(
        self, name: str, version_number: int
    ) -> Dict[str, Any]:
        """Get a specific version of a skill.

        Args:
            name: The skill name.
            version_number: The version number.

        Returns:
            Version descriptor with body_markdown.
        """
        return self._request(
            "GET", _skill_path(name, f"/versions/{_path_part(version_number)}")
        )

    # ── Analytics ────────────────────────────────────────

    def get_metrics(
        self,
        agent_name: str,
        *,
        skill_name: Optional[str] = None,
        window_days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get skill performance metrics for an agent.

        Args:
            agent_name: The agent to get metrics for.
            skill_name: Optional — filter to a single skill.
            window_days: Time window in days (default 30).

        Returns:
            List of metric descriptors per skill.
        """
        params: Dict[str, Any] = {
            "agent_name": agent_name,
            "window_days": window_days,
        }
        if skill_name:
            params["skill_name"] = skill_name
        result = self._request(
            "GET", "/api/v1/skills/analytics/metrics", params=params
        )
        return result.get("skills", [])

    def compare_versions(
        self,
        skill_name: str,
        baseline_hash: str,
        candidate_hash: str,
        *,
        agent_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare two skill versions statistically.

        Args:
            skill_name: The skill name.
            baseline_hash: Content hash of the baseline version.
            candidate_hash: Content hash of the candidate version.
            agent_name: Optional — scope comparison to one agent.

        Returns:
            Comparison result with delta, p-value, verdict.
        """
        params: Dict[str, Any] = {
            "skill_name": skill_name,
            "baseline_hash": baseline_hash,
            "candidate_hash": candidate_hash,
        }
        if agent_name:
            params["agent_name"] = agent_name
        return self._request(
            "GET", "/api/v1/skills/analytics/compare", params=params
        )

    def get_leaderboard(
        self,
        agent_name: str,
        *,
        window_days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get skill effectiveness leaderboard for an agent.

        Args:
            agent_name: The agent to rank skills for.
            window_days: Time window in days (default 30).

        Returns:
            List of leaderboard entries ranked by effectiveness.
        """
        params: Dict[str, Any] = {
            "agent_name": agent_name,
            "window_days": window_days,
        }
        result = self._request(
            "GET", "/api/v1/skills/analytics/leaderboard", params=params
        )
        return result.get("leaderboard", [])

    def reembed(
        self, *, target_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Re-embed all skills with the specified model.

        Args:
            target_model: Target embedding model name.

        Returns:
            Result with reembedded/skipped counts.
        """
        payload: Dict[str, Any] = {}
        if target_model:
            payload["target_model"] = target_model
        return self._request("POST", "/api/v1/skills/reembed", json=payload)

    # ── Attachments ─────────────────────────────────────

    def list_attachments(
        self,
        skill_id: str,
        *,
        directory: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List attachments for a skill (scripts, references, templates, assets).

        Args:
            skill_id: The skill UUID.
            directory: Optional filter — one of 'scripts', 'references',
                       'templates', 'assets'.

        Returns:
            List of attachment descriptors (without content_text).
        """
        params: Dict[str, Any] = {}
        if directory:
            params["directory"] = directory
        result = self._request(
            "GET", _skill_path(skill_id, "/attachments"), params=params
        )
        return result.get("attachments", [])

    def get_attachment(
        self,
        skill_id: str,
        attachment_id: str,
    ) -> Dict[str, Any]:
        """Get a single attachment with its full content.

        Args:
            skill_id: The skill UUID.
            attachment_id: The attachment UUID.

        Returns:
            Attachment descriptor including content_text.
        """
        return self._request(
            "GET",
            _skill_path(skill_id, f"/attachments/{_path_part(attachment_id)}"),
        )

    def get_attachment_by_path(
        self,
        skill_id: str,
        file_path: str,
    ) -> Dict[str, Any]:
        """Get an attachment by its relative file path.

        Args:
            skill_id: The skill UUID.
            file_path: Relative path, e.g. "scripts/validate.py".

        Returns:
            Attachment descriptor including content_text.
        """
        return self._request(
            "GET",
            _skill_path(skill_id, f"/attachments/by-path/{_path_part(file_path)}"),
        )

    def upload_attachment(
        self,
        skill_id: str,
        file_path: str,
        directory: str,
        content_text: str,
        *,
        content_type: Optional[str] = None,
        skill_version_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload a single attachment to a skill.

        Args:
            skill_id: The skill UUID.
            file_path: Relative path, e.g. "scripts/validate.py".
            directory: One of 'scripts', 'references', 'templates', 'assets'.
            content_text: File content as text.
            content_type: MIME type (auto-detected if omitted).
            skill_version_id: Optional version scope.

        Returns:
            Created attachment info (id, file_path, size_bytes).
        """
        payload: Dict[str, Any] = {
            "file_path": file_path,
            "directory": directory,
            "content_text": content_text,
        }
        if content_type:
            payload["content_type"] = content_type
        if skill_version_id:
            payload["skill_version_id"] = skill_version_id

        return self._request(
            "POST", _skill_path(skill_id, "/attachments"), json=payload
        )

    def upload_attachments_bulk(
        self,
        skill_id: str,
        files: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Upload multiple attachments at once.

        Args:
            skill_id: The skill UUID.
            files: List of dicts with file_path, directory, content_text.

        Returns:
            Bulk creation result with created count and attachment IDs.
        """
        return self._request(
            "POST",
            _skill_path(skill_id, "/attachments/bulk"),
            json={"files": files},
        )

    def delete_attachment(
        self,
        skill_id: str,
        attachment_id: str,
    ) -> Dict[str, Any]:
        """Delete a single attachment.

        Args:
            skill_id: The skill UUID.
            attachment_id: The attachment UUID.

        Returns:
            Deletion result.
        """
        return self._request(
            "DELETE",
            _skill_path(skill_id, f"/attachments/{_path_part(attachment_id)}"),
        )

    # ── Disk: the package manager's verbs ──────────────────
    #
    # Every disk method is a wrapper over `decimalai.skillpkg`; see the notes above
    # `SkillRouterError`. None of them reads `/body`, forks or writes a skills
    # folder by itself.

    def _engine(self) -> Dict[str, Any]:
        """The registry and key the engine uses: this router's."""
        return {"api_key": self.api_key, "base_url": self.base_url, "channel": "sdk"}

    def _add_for_export(
        self, name: str, *, project_root: Optional[str], yes: bool, force: bool,
    ) -> Dict[str, Any]:
        """``skillpkg.add(name)``, summarised the way :meth:`export` returns it.

        Raises:
            SkillRouterError: the engine refused; ``.result`` has every refusal.
        """
        from .skillpkg import ops

        result = ops.add(name, yes=yes, force=force, directory=project_root, **self._engine())
        if not result.ok:
            raise SkillRouterError(f"could not add {name!r}: {_refusal(result)}", result=result)
        project = _disk_project(project_root)
        locked = _locked_entries(project)
        summary = _written(result, project, locked)
        change = next((c for c in result.changes if getattr(c, "install_name", None)), None)
        install_name = getattr(change, "install_name", None)
        summary.update({
            "written_paths": list(summary["paths"]),
            "skill_name": getattr(change, "skill", None) or name,
            "skill_dirname": install_name,
            "errors": [],
            "lock_entry": locked.get(install_name) if install_name else None,
            "result": result,
        })
        return summary

    # ── Registry lifecycle: publish / unpublish / merge upstream ───
    #
    # These wrap the org-side skill routes, NOT the registry routes,
    # because publish/unpublish/merge are *org-side* operations that flip
    # visibility or pull upstream changes — the registry routes
    # (`/api/v1/registry/...`) are for browsing and forking *from* the
    # registry. Sibling to `fork()` below.

    def publish_skill(
        self,
        name: str,
        *,
        category: str,
        tags: Optional[List[str]] = None,
        author_display_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Publish one of your org's skills to the public registry.

        The server enforces four gates before flipping ``visibility`` to
        ``public``:

        1. Caller must be the skill creator (or org admin).
        2. The version being published must have eval cases AND at least
           one completed benchmark run — a run that *fails* is fine, a
           missing or error-dominated one is not.
        3. The safety scan must pass.
        4. Skill name must not collide with any other public skill
           (registry org imports OR another community publisher).

        On success the skill stays in your org (no row migration) but
        gains ``visibility='public'``, ``skill_badge='community'``, and
        the supplied registry metadata.

        Args:
            name: Name of the skill in your org.
            category: Registry category (e.g. ``"code-review"``,
                ``"testing"``). Required — there is no sensible default.
            tags: Optional list of tags for registry browse filtering.
            author_display_name: Public author name shown on the registry
                page. Defaults to your user id if omitted.

        Returns:
            Dict with ``status``, ``skill_id``, ``category``,
            ``skill_badge``.

        Raises:
            SkillRouterError: 404 (skill not found), 403 (not owner),
                400 (gates failed), 409 (name collision).

        Example::

            router.publish_skill(
                "refund-policy",
                category="support",
                tags=["billing", "refunds"],
                author_display_name="Acme Support Team",
            )
        """
        params: Dict[str, Any] = {"category": category}
        if tags:
            # Server expects comma-separated string, not array.
            params["tags"] = ",".join(tags)
        if author_display_name:
            params["author_display_name"] = author_display_name
        return self._request(
            "POST", _skill_path(name, "/publish"), params=params,
        )

    def unpublish_skill(self, name: str) -> Dict[str, Any]:
        """Remove a published skill from the public registry.

        Existing forks in other orgs are unaffected — they are
        independent copies. The server clears each fork's cached
        ``has_upstream_update`` flag so consumers don't see a phantom
        "merge available" banner pointing at a now-private source.

        Returns:
            Dict with ``status``, ``skill_id``, and ``forks_cleared``
            (the number of consumer forks whose upgrade banner was
            cleared).
        """
        return self._request(
            "POST", _skill_path(name, "/unpublish"),
        )

    def merge_upstream(
        self,
        name: str,
        *,
        mode: str = "preview",
    ) -> Dict[str, Any]:
        """Pull upstream registry changes into a forked skill.

        Distinct from ``update_skills()``, which only writes
        platform-state to disk — this method touches the platform itself,
        creating a new SkillVersion on the fork with the upstream body
        and advancing ``forked_at_version_id``.

        Two modes:

        - ``"preview"`` (default, safe): returns ``upstream_body``,
          ``upstream_version_number``, ``upstream_change_summary``, and
          ``current_body`` for a side-by-side comparison. No write.
        - ``"replace"``: creates a new version on the fork with the
          upstream body, advances ``forked_at_version_id``, clears
          ``has_upstream_update``.

        The server rejects the call with 404 if the upstream has been
        unpublished or deleted — pulling a now-private body into your
        fork would leak content the author has revoked.

        Args:
            name: Name of the forked skill in your org.
            mode: ``"preview"`` or ``"replace"``.

        Raises:
            SkillRouterError: 400 (not a fork), 404 (upstream gone),
                403 (no edit access).
        """
        if mode not in ("preview", "replace"):
            raise ValueError(
                f"mode must be 'preview' or 'replace', got {mode!r}"
            )
        return self._request(
            "POST",
            _skill_path(name, "/merge-upstream"),
            params={"mode": mode},
        )

    def fork(
        self,
        name: str,
        *,
        new_name: Optional[str] = None,
        offer_scope: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fork a registry skill into your workspace — DB only, no disk write.

        This is the workspace-side copy: it creates a ``Skill`` row you own
        (provenance kept via ``forked_from_skill_id``), so the Router can
        serve it and the platform can measure it. Nothing is written to
        disk — for that, use :meth:`install` (fork + write ``SKILL.md``),
        which is what file-loading runtimes (Claude Code, Cursor) need.

        Args:
            name: Registry skill name to fork.
            new_name: Optional custom name for the fork in your workspace.
            offer_scope: Which agents get the copy: ``"restricted"``, only the
                agents you add it to; ``"workspace"``, every agent. Leave it out
                and it reaches every agent for now; the platform is making new
                skills opt-in, so pass it to keep a behavior.

        Returns:
            The platform's fork response (``status``, ``skill`` with id/name,
            ``forked_from_skill_id``, ``forked_at_version_id``).

        Raises:
            ValueError: the name doesn't match any registry skill.
            RuntimeError: the registry search or fork call failed.

        Example::

            router = SkillRouter(api_key="dai_sk_...")
            router.fork("pdf")            # now in your workspace; not on disk
        """
        # Step 1: resolve the registry skill id by name. `q=` is a semantic
        # search that always ranks something, so match the name EXACTLY —
        # items[0] used to fork an unrelated skill on any typo or rename.
        from ._registry_resolve import RESOLVE_LIMIT, find_exact, not_found_message

        try:
            registry_result = self._request(
                "GET", "/api/v1/registry/skills",
                params={"q": name, "limit": RESOLVE_LIMIT},
            )
            items = registry_result.get("items", [])
            match = find_exact(items, name)
            if match is None:
                raise ValueError(not_found_message(name, items))
            registry_id = match["id"]
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to search registry: {e}")

        params: Optional[Dict[str, Any]] = {
            **({"new_name": new_name} if new_name else {}),
            **({"offer_scope": offer_scope} if offer_scope is not None else {}),
        } or None
        # Step 2: fork. ``/fork`` is the only route now — the ``/install`` alias
        # was RETIRED server-side on 2026-08-11 and answers 410, because
        # "install" came to mean a LINK everywhere else and a route that
        # silently forked under that name was doing the opposite of what it
        # promised. The 404 fallback that used to sit here is gone with it: it
        # could only ever have fired against a backend older than /fork, and it
        # also fired pointlessly on every genuine skill-not-found 404, turning
        # one wrong id into two round trips.
        try:
            return self._request(
                "POST", _registry_skill_path(registry_id, "/fork"),
                params=params,
            )
        except SkillRouterError as e:
            # `from e` so callers (install) can read the SkillRouterError's
            # status_code/headers off __cause__ — the fork 409 carries the
            # existing fork's name in X-Installed-As.
            raise RuntimeError(f"Failed to fork from registry: {e}") from e

    def export(
        self,
        name: str,
        *,
        project_root: Optional[str] = None,
        yes: bool = False,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Install one registry skill into the project: ``decimalai skills add <name>``.

        The same engine as the command (:func:`decimalai.skillpkg.add`): the skill is
        recorded in ``.decimal/skills.toml``, its folder is written to
        ``.agents/skills/<install-name>/`` with the registry's own bytes, and
        ``.decimal/skills.lock`` pins exactly those bytes. **No copy is taken** (no
        fork), and nothing is read through ``/body``. Re-running it is safe: a skill
        already installed as asked is left alone.

        Works for anything the registry lets your key install: a skill you own, one
        you have linked with :meth:`use`, or a public skill. A skill whose safety
        review is ``caution`` or ``unreviewed`` needs ``yes=True`` when there is no
        terminal to ask on.

        A project a 0.14 SDK wrote (a version 1 ``.decimal/skills.lock``) is refused
        until ``decimalai skills migrate`` converts it.

        There is one skills root, ``.agents/skills``: Claude Code loads it with
        ``claude --plugin-dir "$(git rev-parse --show-toplevel)/.agents"`` (an absolute
        path: a relative one fails silently from a subdirectory), and other runtimes
        read it through :func:`decimalai.skills_dir`.

        Args:
            name: a registry name, url_slug, install name or skill id, optionally
                with ``@N`` or ``@latest``.
            project_root: the project (default: found from the working directory).
            yes: accept a ``caution`` or ``unreviewed`` skill without asking.
            force: replace an unmanaged folder of that name (backed up first).

        Returns:
            ``skills_written`` (1 when a folder was written, 0 when it was already
            installed), ``attachments_written``, ``paths`` and ``written_paths`` (the
            SKILL.md written), ``skill_name`` (the registry name), ``skill_dirname``
            (the install name), ``errors``, ``lock_entry`` (the skill's lock entry)
            and ``result`` (the engine's ``Result``).

        Raises:
            SkillRouterError: the engine refused (not found, no access, blocked, a
                local edit in the way...); ``.result`` holds every refusal.

        Example::

            router = SkillRouter(api_key="dai_sk_...")
            router.export("acme/refund-policy")   # .agents/skills/acme-refund-policy/
        """
        return self._add_for_export(name, project_root=project_root, yes=yes, force=force)

    def use(
        self,
        name: str,
        *,
        scope: str = "workspace",
        agents: Optional[List[str]] = None,
        mode: str = "latest",
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Use a registry skill *live* — a linked pointer, not a copy (see Use vs Fork).

        Unlike :meth:`fork` (which makes an editable copy you own), ``use`` keeps the
        public skill's identity and tracks the author's updates. It creates a ``Use``
        row in your workspace; the platform's resolver then serves the skill to your
        agents at the resolved version.

        Args:
            name: Registry skill name to use.
            scope: ``"workspace"`` (every agent) or ``"agent"`` (only ``agents``).
            agents: Agent names when ``scope="agent"``.
            mode: ``"latest"`` (auto-track the newest *safe* version) or ``"pinned"``.
            version: Version id/number to pin when ``mode="pinned"``.

        Returns:
            The platform's Use response (``status``, ``uses``, ``shadowed_by_owned``).

        Raises:
            ValueError: the name doesn't match any registry skill.
            SkillRouterError: the ``/use`` route is unavailable (older backend) or failed.
        """
        if scope not in ("workspace", "agent"):
            raise ValueError(f"scope must be 'workspace' or 'agent', got {scope!r}")
        if mode not in ("latest", "pinned"):
            raise ValueError(f"mode must be 'latest' or 'pinned', got {mode!r}")
        # Exact-name resolution only — `q=` is a semantic search that ranks the
        # whole corpus, so items[0] is not a match.
        from ._registry_resolve import RESOLVE_LIMIT, find_exact, not_found_message

        try:
            search = self._request(
                "GET", "/api/v1/registry/skills", params={"q": name, "limit": RESOLVE_LIMIT},
            )
            items = search.get("items") or []
            match = find_exact(items, name)
            if match is None:
                raise ValueError(not_found_message(name, items))
            registry_id = match["id"]
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to search registry: {e}")

        params: Dict[str, Any] = {"scope": scope, "mode": mode}
        if version is not None:
            params["version"] = version
        if agents:
            params["agents"] = ",".join(agents)
        return self._request(
            "POST", _registry_skill_path(registry_id, "/use"), params=params,
        )

    def preview(self, name: str) -> Optional[Dict[str, Any]]:
        """Fetch a public registry skill as an ephemeral snapshot — no fork, no disk write.

        Sibling to `fork()` for read-only consumption. `fork()` copies the
        registry skill into your workspace; `preview` returns just the body +
        metadata + attachments without any persistent side effects. Suitable for:

        - Ephemeral evaluation runs ("does this skill help on my dataset?")
        - Sandboxed agent invocations that shouldn't pollute the user's org
        - "Try before you fork" preview UX
        - Pure read-only registry consumers (CLI tools, dashboards)

        Args:
            name: Registry skill name (case-sensitive slug).

        Returns:
            Dict with keys: ``name``, ``body_markdown``, ``description``,
            ``category``, ``tags``, ``skill_badge``,
            ``author_display_name``, ``source_type``, ``source_url``,
            ``install_count``, ``effectiveness``, ``attachments``,
            ``skill_id`` (the upstream registry id), ``latest_version_number``.
            Returns ``None`` if no matching public skill exists.

        Note:
            Hits the public registry endpoints (``GET /api/v1/registry/skills``
            + ``GET /api/v1/registry/skills/{id}``); no auth required, but the
            SDK's API key is sent anyway for rate-limit accounting.

        Example::

            router = SkillRouter(api_key="dai_sk_...")
            snap = router.preview("pdf")
            if snap:
                print(snap["body_markdown"][:200])
        """
        # Exact-name resolution only — previewing whatever `q=` ranked first is
        # how a caller ends up reading (and trusting) a different skill's body.
        from ._registry_resolve import RESOLVE_LIMIT, find_exact

        try:
            search = self._request(
                "GET", "/api/v1/registry/skills",
                params={"q": name, "limit": RESOLVE_LIMIT},
            )
        except SkillRouterError as e:
            logger.warning("preview(%s) search failed: %s", name, e)
            return None

        items = search.get("items") or []
        match = find_exact(items, name)
        if match is None:
            return None

        registry_id = match["id"]
        try:
            detail = self._request("GET", _registry_skill_path(registry_id))
        except SkillRouterError as e:
            logger.warning("preview(%s) detail failed: %s", name, e)
            return None

        return {
            "name": detail.get("name"),
            "skill_id": detail.get("id"),
            "body_markdown": detail.get("body_markdown"),
            "description": detail.get("description"),
            "category": detail.get("category"),
            "tags": detail.get("tags") or [],
            "skill_badge": detail.get("skill_badge"),
            "author_display_name": detail.get("author_display_name"),
            "source_type": detail.get("source_type"),
            "source_url": detail.get("source_url"),
            "install_count": detail.get("install_count") or 0,
            "effectiveness": detail.get("effectiveness"),
            "attachments": detail.get("attachments") or [],
            "latest_version_number": detail.get("latest_version_number"),
        }

    def status(
        self,
        *,
        project_root: Optional[str] = None,
        remote: bool = False,
    ) -> Dict[str, Any]:
        """What the project has installed: ``decimalai skills status``.

        The same engine as the command (:func:`decimalai.skillpkg.status`), for the
        project at ``project_root`` (default: found from the working directory). Read
        only; with ``remote=True`` the registry is asked which skills are outdated or
        revoked, and no other call reaches the network.

        Returns:
            What the package manager has installed:

            - ``synced``: installed folders that match the lock;
            - ``modified_locally``: installed folders edited by hand;
            - ``missing_locally``: skills the manifest or lock lists with no folder;
            - ``untracked``: skills on disk the package manager does not manage — folders
              in ``.agents/skills`` the lock does not list, and the skills you wrote in
              the project's other skill folders;
            - ``total_on_disk``: folders under ``.agents/skills``;
            - ``total_on_platform``: the registry skills the project installs (manifest
              and lock);

            plus ``problems`` (each finding as text) and ``result`` (the engine's
            ``Result``, whose ``changes`` are one ``SkillStatus`` per skill).

        Example::

            status = SkillRouter(api_key="dai_sk_...").status()
            for name in status["modified_locally"]:
                print(f"  {name}: edited by hand")
        """
        from .skillpkg import ops
        from .skills import discover_skills

        result = ops.status(remote=remote, directory=project_root, **self._engine())
        rows = [c for c in result.changes if isinstance(c, ops.SkillStatus)]

        def named(*states: str) -> List[str]:
            return sorted({str(r.install_name or r.skill) for r in rows if r.state in states})

        project = _disk_project(project_root)
        try:
            authored = [s["name"] for s in discover_skills(project_root=str(project.directory))
                        if not s.get("managed")]
        except Exception:
            authored = []
        return {
            "synced": named("clean"),
            "modified_locally": named("modified"),
            "missing_locally": named("missing", "not_installed"),
            "untracked": sorted(set(named("unmanaged", "unmanaged_stamped")) | set(authored)),
            "total_on_disk": len([r for r in rows if r.state not in ("missing", "not_installed")]),
            "total_on_platform": len([r for r in rows if r.state not in ("unmanaged",
                                                                         "unmanaged_stamped",
                                                                         "illegal")]),
            "problems": [str(p) for p in result.problems],
            "result": result,
        }

    def update_skills(
        self,
        *,
        project_root: Optional[str] = None,
        force: bool = False,
        yes: bool = False,
    ) -> Dict[str, Any]:
        """Move installed skills to their newest version: ``decimalai skills update``.

        The same engine as the command (:func:`decimalai.skillpkg.update`): every
        ``latest`` entry of ``.decimal/skills.toml`` moves to the current version, exact
        pins stay, and each changed folder is replaced whole. A folder edited by hand
        stops the update unless ``force=True``, which backs it up to
        ``.decimal/backup/<txid>/`` first and then replaces it. If any skill cannot be
        updated, none is.

        A project a 0.14 SDK wrote (a version 1 lock) is refused until
        ``decimalai skills migrate`` converts it.

        Args:
            project_root: the project (default: found from the working directory).
            force: replace folders edited by hand (after backing them up).
            yes: accept a new version whose safety review is ``caution`` or
                ``unreviewed`` without asking.

        Returns:
            ``updated`` and ``updated_skills`` (the folders written), ``skipped`` (the
            skills already current), ``failed``, ``error`` when the update was refused,
            and ``result`` (the engine's ``Result``).
        """
        from .skillpkg import ops

        result = ops.update(force=force, yes=yes, directory=project_root, **self._engine())
        updated = [c.install_name for c in result.changes
                   if getattr(c, "action", None) in _FOLDER_WRITES]
        locked = _locked_entries(_disk_project(project_root))
        out: Dict[str, Any] = {
            "updated": len(updated),
            # The engine reports only what it changed; the rest of the lock stayed.
            "skipped": len([name for name in locked if name not in updated]),
            "failed": 0,
            "updated_skills": updated,
            "result": result,
        }
        if not result.ok:
            out["failed"] = 1
            out["error"] = _refusal(result)
            logger.warning("update_skills: %s", out["error"])
        return out

    # ── Registry Search ────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        badge: Optional[str] = None,
        sort: str = "effectiveness",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search the public skills registry.

        Uses the platform's hybrid search (keyword for short queries,
        semantic embedding for longer queries). Results include
        effectiveness data, install counts, and ratings.

        Args:
            query: Search query (e.g. "pdf conversion", "code review security").
            category: Filter by registry category (e.g. "code-review", "testing").
            tags: Filter by tags (e.g. ["python", "security"]).
            badge: Filter by badge ("featured", "community", "imported").
            sort: Sort by "effectiveness" (alias of "recommended" — SkillScore
                  v2 ranking: evidence-tiered quality composite, cold-start
                  rows relegated), "popular" (recent activations, installs as
                  tiebreak), "rating" (live pass rate), "recent" (created_at),
                  "biggest_improvement" (benchmark lift), "efficiency"
                  (token savings), or "top_rated" (live ratings).
            limit: Maximum results to return (1-100).

        Returns:
            List of registry skill dicts, each containing:
            - name, description, category, tags
            - effectiveness: {avg_effectiveness, avg_pass_rate, trend, ...}
            - install_count, avg_rating, rating_count
            - skill_badge, author_display_name
            - installed_as (if you already have a fork)

        Example::

            results = router.search("code review security")
            for skill in results:
                eff = skill.get("effectiveness") or {}
                print(f"{skill['name']}: {eff.get('avg_effectiveness', 'N/A')}% effective")
        """
        params: Dict[str, Any] = {
            "q": query,
            "sort": sort,
            "limit": min(limit, 100),
        }
        if category:
            params["category"] = category
        if tags:
            params["tags"] = ",".join(tags)
        if badge:
            params["badge"] = badge

        try:
            resp = self._request("GET", "/api/v1/registry/skills", params=params)
            items = resp.get("items", [])
            logger.info(
                "Registry search '%s': %d results",
                query[:50], len(items),
            )
            return items
        except Exception as e:
            logger.error("Registry search failed: %s", e)
            return []
