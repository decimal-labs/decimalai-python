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

import json as json_module
import logging
import os
import time
import warnings
from collections import OrderedDict
from contextvars import ContextVar
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

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


def _stamp_active_trace(
    routing_id: Optional[str],
    offered_names: Optional[List[str]],
    delivered_names: Optional[List[str]],
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
    except Exception:
        logger.debug("auto-stamp of active trace failed (non-fatal)", exc_info=True)


def estimate_tokens(text: str) -> int:
    """Cheap chars/4 token estimate — mirrors the backend's estimator.
    ±20% on English markdown is fine: the body budgets are soft
    context-hygiene caps, not hard limits."""
    return (len(text) + 3) // 4


# ── Progressive disclosure: the load_skill tool ────────────────────────
# The description tier is always-on; bodies load on demand when the model
# calls load_skill(name). Adapters that own their tool loop (openai_agents,
# pydantic_ai) register this as a native tool; adapters that patch a
# non-loop layer (anthropic, langchain) can't route a tool result and stay
# on prompt injection until a loop wrapper exists.

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
    "its full instructions before using it."
)


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
        """None if the load may proceed, else a refusal message for the model."""
        if name in self.loaded:
            return None  # dedup — repeat load is free
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
        return bool(self.loaded) and self.tokens_used + tokens > self.token_budget

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
_body_budget_ctx: ContextVar[Optional[_BodyLoadBudget]] = ContextVar(
    "decimalai_skill_router_body_budget", default=None,
)


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

    def set(self, key: Any, value: Any) -> None:
        with self._lock:
            expire_ts = time.time() + self._ttl
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
        "auto-discovers SKILL.md files from disk. Both will inject skills "
        "into the system prompt, producing duplicates. With "
        "skill_authority='router' (or 'auto' + this loader) the SDK already "
        "stops mirroring skills to disk (disk_sync=False); to fully avoid the "
        "duplicate, also remove the local SKILL.md files %s loads so the "
        "Router is the only source. Otherwise skip enable_skill_loader=True "
        "here and let %s do the loading. Set "
        "DECIMALAI_SUPPRESS_DISK_RUNTIME_WARNING=1 to silence intentionally.",
        framework, runtime, runtime, runtime,
    )


# Support a one-shot environment override so users who've consciously chosen
# the duplicate-injection setup (some testing scenarios) can silence the
# warning without changing code. Read once at import time; the bool above
# also guards re-firing.
if os.environ.get("DECIMALAI_SUPPRESS_DISK_RUNTIME_WARNING"):
    _disk_runtime_warned = True


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
        super().__init__(message)


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
        api_key: str,
        base_url: Optional[str] = None,
        agent_name: Optional[str] = None,
        strategy: str = "auto",
        max_menu_size: int = 20,
        fragment_cache_ttl: float = 30.0,
        fragment_cache_size: int = 64,
        inject_body: bool = False,
        inject_body_top_k: int = 1,
        max_loaded_bodies: int = 3,
        body_token_budget: int = 6000,
        per_body_char_limit: int = 8192,
        body_load_deadline_s: float = 20.0,
    ):
        self.api_key = api_key
        # Same resolution decimalai.init() uses (explicit → DECIMAL_BASE_URL
        # → default) so a directly-constructed router doesn't silently split
        # a non-default deployment onto the public host.
        self.base_url = (
            base_url
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
        # Body guardrail (the progressive-disclosure path): caps for on-demand load_skill AND
        # the body-inject path (bodies injected un-trimmed before top-k). Defaults
        # mirror the backend's config (max_loaded_bodies=3, body_token_budget
        # ≈6k tok, per-body trim 8KB).
        self.max_loaded_bodies = max(1, int(max_loaded_bodies))
        self.body_token_budget = max(1, int(body_token_budget))
        self.per_body_char_limit = max(256, int(per_body_char_limit))
        self.body_load_deadline_s = float(body_load_deadline_s)

        # Fallback body-load budget for tool-execution contexts that did not
        # inherit the prompt-build ContextVar (see load_skill).
        self._last_budget: Optional[_BodyLoadBudget] = None
        # 'loaded' = the model pulled a skill's BODY on demand through the
        # load_skill tool (progressive disclosure, step 3). Instance state,
        # NOT a contextvar like the offered/delivered rails: loads fire
        # mid-run inside the framework's tool executor, whose copied context
        # never propagates back to the adapter's trace-send path — the same
        # reality behind the `_last_budget` fallback above. Adapters drain
        # this off their router singleton via `consume_loaded_names()`.
        self._loaded_names: List[str] = []
        # Instance mirror of the routing decision — routing_id + offered +
        # delivered names — for the same reason `_loaded_names` exists, one
        # step earlier in the run: prompt assembly also happens in a copied
        # context (LangChain runs its callbacks under `copy_context()`, the
        # OpenAI Agents runner copies around the instructions callable), so
        # the contextvar rails above are invisible to the adapter's
        # trace-send path. The contextvars stay authoritative where they DO
        # propagate (the generic tracer, the anthropic adapter); adapters
        # union this rail in via `consume_routing_id()` /
        # `consume_offered_names()` / `consume_delivered_names()`.
        self._routing_id_rail: Optional[str] = None
        self._offered_names_rail: List[str] = []
        self._delivered_names_rail: List[str] = []
        # Full-menu cache (single slot, force-refresh to invalidate).
        # Cache the menu per (category, project_id, effective_agent) so a
        # second get_menu() with different args doesn't return the first call's
        # menu. force_refresh still bypasses/refreshes the matching key.
        self._menu_cache: Dict[Tuple[Optional[str], Optional[str], Optional[str]], Dict[str, Any]] = {}
        # `build_prompt_fragment` cache. Keyed by the input
        # tuple so repeat calls within the same turn (multi-LLM-call
        # agent loops) reuse a single routing decision instead of
        # racking up N RoutingDecision rows + N Gemini embedding calls.
        self._fragment_cache = _FragmentCache(
            maxsize=fragment_cache_size,
            ttl_seconds=fragment_cache_ttl,
        )

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request to the platform API.

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

        try:
            resp = httpx.request(
                method,
                url,
                params=clean_params,
                json=json,
                headers=self._headers(),
                timeout=30.0,
            )
        except httpx.HTTPError as e:
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
            raise SkillRouterError(
                message,
                status_code=resp.status_code,
                detail=detail,
                headers=resp_headers,
            )

        try:
            return resp.json()
        except (json_module.JSONDecodeError, ValueError) as e:
            raise SkillRouterError(
                f"SkillRouter got non-JSON response from {method} {path}: {e}"
            ) from e

    def get_menu(
        self,
        *,
        category: Optional[str] = None,
        project_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Fetch the full skill menu from the platform.

        Returns a dict with 'skills', 'prompt_fragment', 'strategy'.
        Results are cached until force_refresh=True.

        ``agent_name`` (falls back to the instance default) lets the platform's
        Use/Fork resolver scope the menu to one agent when enabled server-side;
        it's a no-op against older backends / when the resolver is off.
        """
        effective_agent = agent_name or self.agent_name
        cache_key = (category, project_id, effective_agent)
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
            return {"skills": [], "prompt_fragment": "", "strategy": "none"}

        if result:
            self._menu_cache[cache_key] = result
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
            return {"skills": [], "prompt_fragment": "", "strategy": "none"}

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
        Returns the body text or None if not found.
        """
        params: Dict[str, Any] = {}
        if version is not None:
            params["version"] = version
        if max_chars is not None:
            params["max_chars"] = max_chars
        if agent_name is not None:
            params["agent_name"] = agent_name

        try:
            result = self._request(
                "GET",
                f"/api/v1/skills/{skill_name}/body",
                params=params,
            )
        except SkillRouterError as e:
            logger.warning("get_skill_body(%s) failed: %s", skill_name, e)
            return None
        return result.get("body") if result else None

    def load_skill(self, name: str, *, agent_name: Optional[str] = None) -> str:
        """The load_skill tool handler — fetch a surfaced skill's body on demand.

        Always returns a STRING (never raises): either the trimmed body block
        the adapter feeds back as the tool result, or an explanatory message
        (budget exhausted / not found) the model can act on. Enforces the
        per-turn body guardrail: ``max_loaded_bodies``, ``body_token_budget``,
        ``per_body_char_limit`` trim, ``body_load_deadline_s``. Records the
        load on the active trace (``skills_loaded_by_agent``) so the
        offered-vs-loaded join closes server-side.
        """
        name = (name or "").strip()
        if not name:
            return "load_skill error: a skill name is required."

        # ContextVar first (isolated per agent run); instance fallback for
        # frameworks that execute each tool call in a context that did not
        # inherit the prompt-build context; else start a fresh budget.
        budget = _body_budget_ctx.get() or self._last_budget
        if budget is None:
            budget = _BodyLoadBudget(
                self.max_loaded_bodies, self.body_token_budget,
                self.body_load_deadline_s,
            )
            _body_budget_ctx.set(budget)
            self._last_budget = budget

        refusal = budget.check(name)
        if refusal is not None:
            return refusal

        body = self.get_skill_body(
            name,
            max_chars=self.per_body_char_limit,
            agent_name=agent_name or self.agent_name,
        )
        if body is None or not body.strip():
            return (
                f"load_skill: no skill named {name!r} is available. Use the exact "
                "name from the skills menu."
            )
        body = body.strip()
        # Client-side trim as defense in depth — older backends ignore max_chars.
        if len(body) > self.per_body_char_limit:
            body = (
                body[: self.per_body_char_limit]
                + "\n\n[... truncated by the per-body limit]"
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
        if name not in self._loaded_names:
            self._loaded_names.append(name)
        try:
            from .generic import log_skill_loaded
            log_skill_loaded(name=name)
        except Exception:
            logger.debug("load_skill: no active trace to record the load", exc_info=True)

        return f"## Skill: {name}\n\n{body}"

    def consume_loaded_names(self) -> List[str]:
        """Read + clear the names whose body `load_skill` served since the
        last drain. Adapter trace-send paths call this on their router
        singleton — the same instance whose load_skill tool served the
        bodies — and stamp the names onto ``skills_loaded_by_agent``.
        Known cost (same as the `_last_budget` fallback): concurrent runs
        sharing one router drain into whichever trace sends first.
        """
        if not self._loaded_names:
            return []
        drained = list(self._loaded_names)
        self._loaded_names.clear()
        return drained

    def _record_routing_rails(
        self,
        routing_id: Optional[str],
        offered_names: Optional[List[str]],
        delivered_names: Optional[List[str]],
    ) -> None:
        """Mirror one routing decision onto the instance rails.

        Names accumulate (dedup-append, like `_loaded_names`) because a
        multi-LLM-call turn routes more than once and the adapter drains
        only at trace-send; `routing_id` is last-write-wins, matching the
        contextvar rails the adapters set per call.
        """
        if routing_id:
            self._routing_id_rail = routing_id
        for name in offered_names or ():
            if name not in self._offered_names_rail:
                self._offered_names_rail.append(name)
        for name in delivered_names or ():
            if name not in self._delivered_names_rail:
                self._delivered_names_rail.append(name)

    def consume_routing_id(self) -> Optional[str]:
        """Read + clear the most recent routing id minted since the last
        drain. Adapters fall back to this when their own routing-id
        contextvar comes back empty because prompt assembly ran in a
        copied context. Same concurrency cost as `consume_loaded_names`.
        """
        rid = self._routing_id_rail
        self._routing_id_rail = None
        return rid

    def consume_offered_names(self) -> List[str]:
        """Read + clear the names offered (menu rows) since the last drain.
        The instance-state twin of `consume_last_offered_names`.
        """
        if not self._offered_names_rail:
            return []
        drained = list(self._offered_names_rail)
        self._offered_names_rail.clear()
        return drained

    def consume_delivered_names(self) -> List[str]:
        """Read + clear the names whose BODY the prompt fragment carried
        since the last drain. The instance-state twin of
        `consume_last_delivered_names`.
        """
        if not self._delivered_names_rail:
            return []
        drained = list(self._delivered_names_rail)
        self._delivered_names_rail.clear()
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
    ) -> tuple[str, Optional[str]]:
        """Return ``(prompt_fragment, routing_id)`` — the primitive every adapter calls.

        This is the single entry point framework adapters (openai_agents,
        langchain, pydantic_ai, anthropic) use to load skills into a prompt.
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
        network blip. The warning is logged so ops can see it. Errors
        are NOT cached — the next call after a failure retries.

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
        )

        if not bypass_cache:
            cached = self._fragment_cache.get(cache_key)
            if cached is not None:
                fragment, routing_id, offered_names, delivered_names = cached
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
                self._record_routing_rails(routing_id, offered_names, delivered_names)
                _stamp_active_trace(routing_id, offered_names, delivered_names)
                return fragment, routing_id

        # Reset both rails up front so a call that offers or
        # delivers nothing can't inherit a previous call's unconsumed
        # names (producers without a consumer exist — e.g. the anthropic
        # adapter and the generic quickstart never drain the rails).
        _last_offered_names_ctx.set(None)
        _last_delivered_names_ctx.set(None)

        if query and self.strategy in ("auto", "semantic"):
            try:
                result = self.smart_route(
                    query,
                    top_k=top_k,
                    category=category,
                    agent_name=effective_agent,  # pass the per-call override through
                )
            except SkillRouterError as e:
                logger.warning("build_prompt_fragment smart_route failed: %s", e)
                return "", None
        else:
            try:
                result = self.get_menu(category=category)
            except SkillRouterError as e:
                logger.warning("build_prompt_fragment get_menu failed: %s", e)
                return "", None

        fragment = result.get("prompt_fragment", "") or ""
        routing_id = result.get("routing_id")
        # Expose the offered skill names via a contextvar so
        # framework adapters can stamp them onto RunTrace.skills_offered_in_prompt
        # without having to extend this method's return signature.
        offered_names = [
            name for s in (result.get("skills") or [])
            if isinstance(s, dict) and isinstance(name := s.get("name"), str) and name
        ]
        if offered_names:
            _last_offered_names_ctx.set(offered_names)

        # Deliver the actual skill BODY (the knowledge K), not just the menu row, so the
        # skill's runtime value reaches the agent. We fetch the body of the top-k routed skills and
        # append it. This rides the same 30s fragment cache, so within one user turn the body is
        # fetched once (one get_skill_body call), not per LLM call. A missing body degrades
        # gracefully to the menu-only fragment.
        # Only inject when the result is RELEVANCE-RANKED (smart routing). In full-menu mode the
        # "top" skill isn't ranked by the query, so injecting its body would be arbitrary — there
        # we keep the menu only.
        # Body guardrail (the progressive-disclosure path): bodies injected un-trimmed pre-topk.
        # Now each body is server-trimmed (max_chars) + client-trimmed as
        # defense, count-capped, and the total respects body_token_budget.
        smart_routed = bool(query) and self.strategy in ("auto", "semantic")
        delivered_names: List[str] = []
        if effective_inject and offered_names and smart_routed:
            bodies = []
            body_tokens = 0
            count_cap = min(self.inject_body_top_k, self.max_loaded_bodies)
            for name in offered_names[:count_cap]:
                body = self.get_skill_body(name, max_chars=self.per_body_char_limit)
                if not (body and body.strip()):
                    continue
                body = body.strip()
                if len(body) > self.per_body_char_limit:
                    body = (
                        body[: self.per_body_char_limit]
                        + "\n\n[... truncated by the per-body limit]"
                    )
                tokens = estimate_tokens(body)
                if bodies and body_tokens + tokens > self.body_token_budget:
                    break
                body_tokens += tokens
                bodies.append(f"## Skill: {name}\n\n{body}")
                # The body actually reached the prompt → delivered.
                delivered_names.append(name)
            if bodies:
                body_block = "\n\n".join(bodies)
                fragment = f"{fragment}\n\n{body_block}" if fragment else body_block
                _last_delivered_names_ctx.set(list(delivered_names))

        # A fresh fragment marks a new turn: reset the on-demand body-load
        # budget so load_skill's caps apply per turn, not per process.
        # Cache hits (multi-LLM-call loops within one turn) keep the
        # existing budget — that is exactly the window the caps guard.
        fresh_budget = _BodyLoadBudget(
            self.max_loaded_bodies, self.body_token_budget, self.body_load_deadline_s,
        )
        _body_budget_ctx.set(fresh_budget)
        self._last_budget = fresh_budget

        # Mirror onto the instance rails before stamping — the adapters
        # whose contextvars never reach trace-send read the routing
        # telemetry off the router itself.
        self._record_routing_rails(routing_id, offered_names, delivered_names)

        # Stamp the active generic trace (raw-loop quickstart) —
        # adapter paths stamp their own trace objects, so this is a no-op
        # or an idempotent double-set there.
        _stamp_active_trace(routing_id, offered_names, delivered_names)

        # Cache the success path. We don't cache errors — next call
        # retries on the assumption the failure was transient. The cached
        # value carries offered/delivered so cache hits can re-emit them;
        # the public return stays a 2-tuple.
        if fragment or routing_id:
            self._fragment_cache.set(
                cache_key, (fragment, routing_id, offered_names, delivered_names),
            )
        return fragment, routing_id

    def sync_skills(
        self,
        skills: List[Dict[str, Any]],
        author: Optional[str] = None,
        conflict_policy: str = "local_wins",
        response_mode: str = "summary",
        install_id: Optional[str] = None,
        install_label: Optional[str] = None,
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
        return self._request("GET", f"/api/v1/skills/{name}")

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

        Returns:
            Created skill descriptor.
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
        return self._request("POST", "/api/v1/skills", json=payload)

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
        if change_summary is not None:
            payload["change_summary"] = change_summary
        if author is not None:
            payload["author"] = author
        return self._request("PUT", f"/api/v1/skills/{skill_id}", json=payload)

    def delete_skill(self, skill_id: str) -> Dict[str, Any]:
        """Deactivate (soft-delete) a skill.

        Args:
            skill_id: The skill UUID.
        """
        return self._request("DELETE", f"/api/v1/skills/{skill_id}")

    # ── Versions ─────────────────────────────────────────

    def list_versions(self, name: str) -> List[Dict[str, Any]]:
        """List all versions of a skill.

        Args:
            name: The skill name.

        Returns:
            List of version descriptors, newest first.
        """
        result = self._request("GET", f"/api/v1/skills/{name}/versions")
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
            "GET", f"/api/v1/skills/{name}/versions/{version_number}"
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
            "GET", f"/api/v1/skills/{skill_id}/attachments", params=params
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
            f"/api/v1/skills/{skill_id}/attachments/{attachment_id}",
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
            f"/api/v1/skills/{skill_id}/attachments/by-path/{file_path}",
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
            "POST", f"/api/v1/skills/{skill_id}/attachments", json=payload
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
            f"/api/v1/skills/{skill_id}/attachments/bulk",
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
            f"/api/v1/skills/{skill_id}/attachments/{attachment_id}",
        )

    # ── Disk Export ─────────────────────────────────────────

    def _resolved_export_source(self, name: str) -> Optional[Dict[str, Any]]:
        """The skill as the RESOLVER sees it, or None if this backend can't say.

        ``GET /skills/{name}/body`` runs the Use/Fork resolver, so it answers
        for a **linked** skill — a Use pointer at a public skill owned by the
        publisher's org — as well as one you own. Every other read path here is
        scoped to skills your org owns, which is why :meth:`install` forks
        before it writes files.

        Returns None when the response lacks the export fields, which means an
        older backend — the caller then falls back to the org-scoped path and
        behaves exactly as before.
        """
        try:
            resp = self._request("GET", f"/api/v1/skills/{name}/body")
        except Exception:
            return None
        if not isinstance(resp, dict):
            return None
        body = (resp.get("body") or "").strip()
        # `skill_id` and `description` landed together; either missing means
        # this backend predates the change and cannot build a file on its own.
        if not body or not resp.get("skill_id") or resp.get("description") is None:
            return None
        if resp.get("truncated"):
            # The route truncates for prompt injection. A truncated body is
            # fine in a menu and is not a file — fall back rather than write a
            # SKILL.md that silently ends mid-sentence.
            return None
        return {
            "id": resp["skill_id"],
            "name": resp.get("name") or name,
            "description": resp.get("description") or "",
            "body_markdown": resp.get("body") or "",
        }

    def _fetch_skill_for_export(self, name: str) -> Dict[str, Any]:
        """Fetch one skill with a RESOLVED body + attachments for disk export.

        Tries the resolver-backed body route first so a **linked** skill can be
        exported without forking it (see :meth:`_resolved_export_source`), then
        falls back to the org-scoped path this has always used.

        ``GET /api/v1/skills/{name}`` historically serialized no
        ``body_markdown`` at all, so ``export_to_disk`` wrote SKILL.md files
        with an EMPTY body — install looked successful but delivered
        nothing. Body resolution order on that path:

        1. ``body_markdown`` on the skill response itself (newer backends).
        2. The current version via ``GET /skills/{name}/versions/{n}``
           (``latest_version.version_number``, else newest from
           ``list_versions``) — that endpoint has always returned the body.

        Raises:
            SkillRouterError: the body could not be resolved. Failing loudly
                is the contract — an empty SKILL.md must never be written.
        """
        skill = self._resolved_export_source(name)
        if skill is None:
            skill = self.get_skill(name)
        if not (skill.get("body_markdown") or "").strip():
            version_number = (skill.get("latest_version") or {}).get("version_number")
            if version_number is None:
                versions = self.list_versions(name)
                version_number = (versions[0] or {}).get("version_number") if versions else None
            body = ""
            if version_number is not None:
                body = self.get_version(name, version_number).get("body_markdown") or ""
            if not body.strip():
                raise SkillRouterError(
                    f"Could not resolve a body for skill '{name}': the skill "
                    "response carries no body_markdown and no version body was "
                    "found — refusing to write an empty SKILL.md."
                )
            skill["body_markdown"] = body

        skill_id = skill.get("id", "")
        if skill_id:
            skill["attachments"] = self._attachments_for_export(skill_id)
        return skill

    def _attachments_for_export(self, skill_id: str) -> List[Dict[str, Any]]:
        """Attachment bodies for a skill you own OR one you only link to.

        ``/api/v1/skills/{id}/attachments`` is scoped to skills your org owns,
        so it cannot see a linked skill. The public registry serves the same
        files — ``/api/v1/registry/skills/{id}/attachments`` to list and
        ``.../{attachment_id}`` for one WITH content — and a Use target is by
        definition public, so that path is always available for exactly the
        skills the org-scoped one refuses.

        Best-effort throughout, as before: a skill whose scripts cannot be
        fetched still exports its SKILL.md rather than failing the whole run.
        """
        def _pull(lister, getter) -> Optional[List[Dict[str, Any]]]:
            try:
                atts = lister()
            except Exception:
                return None
            out: List[Dict[str, Any]] = []
            for att in atts or []:
                try:
                    out.append(getter(att["id"]))
                except Exception:
                    logger.debug("Failed to fetch attachment %s", att.get("file_path"))
            return out

        owned = _pull(
            lambda: self.list_attachments(skill_id),
            lambda aid: self.get_attachment(skill_id, aid),
        )
        if owned:
            return owned

        # Empty is a legitimate answer for an owned skill with no attachments,
        # so only a FAILED listing (None) falls through to the public route —
        # otherwise every attachment-less owned skill would pay a second
        # round-trip to be told the same thing.
        if owned == []:
            return []

        public = _pull(
            lambda: self._request(
                "GET", f"/api/v1/registry/skills/{skill_id}/attachments",
            ).get("attachments", []),
            lambda aid: self._request(
                "GET", f"/api/v1/registry/skills/{skill_id}/attachments/{aid}",
            ),
        )
        return public or []

    def export_to_disk(
        self,
        *,
        skills: Optional[List[str]] = None,
        agents: Optional[List[str]] = None,
        scope: str = "project",
        project_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Export skills from the platform to disk for agent runtimes.

        Fetches skill bodies + attachments from the platform and writes
        SKILL.md files + scripts to the correct directories for each
        specified agent runtime.

        Args:
            skills: List of skill names to export. If None, exports ALL
                    skills in the user's org.
            agents: Agent runtimes to write for (e.g., ['claude-code', 'cursor']).
                    Defaults to ['universal'].
            scope: 'project' (writes to .agents/skills/ etc.) or
                   'global' (writes to ~/.claude/skills/ etc.).
            project_root: Project root directory (for scope='project').

        Returns:
            Summary with skills_written, attachments_written, paths, and
            errors (per-skill fetch/body-resolution failures — empty on
            full success).

        Raises:
            SkillRouterError: every explicitly-requested skill failed to
                fetch or resolve a body (an empty SKILL.md is never
                written silently).

        Example::

            router = SkillRouter(api_key="dai_sk_...")

            # Export all org skills for Claude Code and Cursor
            router.export_to_disk(agents=["claude-code", "cursor"])

            # Export specific skills globally
            router.export_to_disk(
                skills=["code-review", "pdf"],
                agents=["claude-code"],
                scope="global",
            )
        """
        from .disk_export import export_skills_to_disk

        # Resolve the target names, then fetch each skill with a RESOLVED
        # body (see _fetch_skill_for_export — never export an empty body).
        if skills:
            names = list(skills)
        else:
            names = [s["name"] for s in self.list_skills()]

        skill_data_list = []
        errors: List[Dict[str, str]] = []
        for name in names:
            try:
                skill_data_list.append(self._fetch_skill_for_export(name))
            except Exception as e:
                logger.warning("Failed to fetch skill '%s' for export: %s", name, e)
                errors.append({"skill": name, "error": str(e)})

        # Explicitly-requested skills that ALL failed → raise, don't return a
        # zero-count summary a caller could mistake for "nothing to do".
        if skills and errors and not skill_data_list:
            raise SkillRouterError(
                "export_to_disk failed for every requested skill: "
                + "; ".join(f"{e['skill']}: {e['error']}" for e in errors)
            )

        if not skill_data_list:
            logger.info("No skills to export")
            return {
                "skills_written": 0,
                "attachments_written": 0,
                "paths": [],
                "errors": errors,
            }

        summary = export_skills_to_disk(
            skill_data_list,
            agents=agents,
            scope=scope,
            project_root=project_root,
        )
        # Per-skill fetch failures surface in the summary — partial success
        # must not read as total success.
        summary["errors"] = errors
        return summary

    # ── Registry lifecycle: publish / unpublish / merge upstream ───
    #
    # These wrap the org-side skill routes, NOT the registry routes,
    # because publish/unpublish/merge are *org-side* operations that flip
    # visibility or pull upstream changes — the registry routes
    # (`/api/v1/registry/...`) are for browsing and forking *from* the
    # registry. Sibling to `fork()` / `install()` below.

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
            "POST", f"/api/v1/skills/{name}/publish", params=params,
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
            "POST", f"/api/v1/skills/{name}/unpublish",
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
            f"/api/v1/skills/{name}/merge-upstream",
            params={"mode": mode},
        )

    def fork(
        self,
        name: str,
        *,
        new_name: Optional[str] = None,
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

        params = {"new_name": new_name} if new_name else None
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
                "POST", f"/api/v1/registry/skills/{registry_id}/fork",
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
        agents: Optional[List[str]] = None,
        scope: str = "project",
        project_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Write one skill to disk. **No copy is taken.**

        The file half of adoption, on its own. `install()` below does this too
        but forks first, so asking for a file used to mean also taking an
        editable copy you never wanted. Nothing about writing a SKILL.md
        requires owning the skill — the writer wants a name, a description and
        a body — so this is the verb for "I want the files", and Install (the
        link) and Fork (the copy) are the verbs for "I want the skill".

        Works for anything the resolver grants you: a skill you own, or one you
        have linked with :meth:`use`. See :meth:`_resolved_export_source`.

        Args:
            name: Skill name.
            agents: Agent runtimes to write for (e.g. ``['claude-code']``).
            scope: ``'project'`` or ``'global'``.
            project_root: Project root directory.

        Returns:
            The disk-write summary — ``skills_written``, ``attachments_written``,
            ``paths``, ``errors``.

        Example::

            router = SkillRouter(api_key="dai_sk_...")
            router.use("pdf")                        # link it
            router.export("pdf", agents=["claude-code"])   # and put it on disk
        """
        return self.export_to_disk(
            skills=[name], agents=agents, scope=scope, project_root=project_root,
        )

    def install(
        self,
        name: str,
        *,
        agents: Optional[List[str]] = None,
        scope: str = "project",
        project_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fork a registry skill into your workspace AND write it to disk.

        .. deprecated::
            The two halves are separate verbs now, because bundling them meant
            asking for a file also took an editable copy. Use :meth:`export` for
            the files, :meth:`use` to link the skill, :meth:`fork` to copy it.
            This keeps working and keeps forking, so nothing breaks — it is just
            no longer the shape to reach for.

        The ergonomic on-ramp for file-loading runtimes: it does
        :meth:`fork` (the workspace copy) and then writes ``SKILL.md`` +
        attachments into each agent runtime's skills directory so Claude
        Code / Cursor / etc. load it. SDK-only runtimes that consume skills
        through the Router don't need this — :meth:`fork` is enough.

        Args:
            name: Registry skill name to install.
            agents: Agent runtimes to write for (e.g. ['claude-code']).
            scope: 'project' or 'global'.
            project_root: Project root directory.

        Returns:
            Dict with ``fork`` (the fork response), ``export`` (disk-write
            summary), and ``skill_name``. The legacy ``install`` key mirrors
            ``fork`` for back-compat.

        Idempotency: a fork 409 ("already forked in this org") is NOT an
        error — the backend names the existing fork in ``X-Installed-As``,
        and install proceeds to the disk export under that name, so
        re-running install always converges to files-on-disk.

        Example::

            router = SkillRouter(api_key="dai_sk_...")
            router.install("pdf", agents=["claude-code", "cursor"])
        """
        from .disk_export import SkillExportFileExistsError

        already_forked = False
        try:
            fork_result = self.fork(name)
            installed_name = fork_result.get("skill", {}).get("name", name)
        except (RuntimeError, SkillRouterError) as e:
            # fork() wraps the transport error in RuntimeError (`from e`);
            # the /install alias fallback can surface it un-wrapped.
            cause = e if isinstance(e, SkillRouterError) else e.__cause__
            if not (isinstance(cause, SkillRouterError) and cause.status_code == 409):
                raise
            already_forked = True
            installed_name = (cause.headers or {}).get("x-installed-as") or name
            logger.info(
                "install(%r): already in your org as %r — refreshing files on disk",
                name, installed_name,
            )
            fork_result = {
                "status": "already_installed",
                "installed_as": installed_name,
                "skill": {"name": installed_name},
            }

        try:
            export_result = self.export_to_disk(
                skills=[installed_name],
                agents=agents,
                scope=scope,
                project_root=project_root,
            )
        except SkillExportFileExistsError:
            if not already_forked:
                raise
            # Re-install with the files already on disk IS the converged
            # state — don't fail the idempotent path over it.
            logger.info(
                "install(%r): files for %r already on disk — nothing to write",
                name, installed_name,
            )
            export_result = {
                "skills_written": 0,
                "attachments_written": 0,
                "paths": [],
                "errors": [],
                "already_on_disk": True,
            }
        return {
            "fork": fork_result,
            "install": fork_result,  # legacy alias key
            "export": export_result,
            "skill_name": installed_name,
        }

    def install_skill(
        self,
        name: str,
        *,
        agents: Optional[List[str]] = None,
        scope: str = "project",
        project_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Deprecated alias for :meth:`install` (fork + write to disk).

        Renamed in the fork/install/router terminology cleanup: ``fork()``
        is the workspace copy, ``install()`` is fork + disk. Use those.
        """
        warnings.warn(
            "SkillRouter.install_skill() is deprecated; use install() "
            "(fork + write to disk) or fork() (workspace copy only).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.install(
            name, agents=agents, scope=scope, project_root=project_root,
        )

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
            "POST", f"/api/v1/registry/skills/{registry_id}/use", params=params,
        )

    def preview(self, name: str) -> Optional[Dict[str, Any]]:
        """Fetch a public registry skill as an ephemeral snapshot — no fork, no disk write.

        Sibling to `fork()` for read-only consumption. `fork()` / `install()`
        copy the registry skill into your workspace (and, for `install()`,
        write SKILL.md to disk); `preview` returns just the body + metadata +
        attachments without any persistent side effects. Suitable for:

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
            detail = self._request("GET", f"/api/v1/registry/skills/{registry_id}")
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

    def pull_skill(self, name: str) -> Optional[Dict[str, Any]]:
        """Deprecated alias for :meth:`preview` (ephemeral snapshot, no fork)."""
        warnings.warn(
            "SkillRouter.pull_skill() is deprecated; use preview().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.preview(name)

    def status(
        self,
        *,
        project_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare local skill state vs platform state.

        Reads the lockfile and compares content hashes against what
        the platform has. Reports which skills are:
        - synced: local and platform match
        - modified_locally: local hash differs from platform
        - missing_locally: on platform but not installed on disk
        - untracked: on disk but not yet synced to platform

        Args:
            project_root: Project root directory.

        Returns:
            Status report dict.

        Example::

            router = SkillRouter(api_key="dai_sk_...")
            status = router.status()
            for name in status["modified_locally"]:
                print(f"  {name}: local changes not synced")
        """
        from .disk_export import _read_lockfile
        from .skills import discover_skills

        root = project_root or os.getcwd()

        # Read lockfile
        lockdata = _read_lockfile(root)
        locked_skills = lockdata.get("skills", {})

        # Discover skills on disk
        discovered = discover_skills() or []
        disk_names = {s["name"] for s in discovered}
        disk_hashes = {}
        for s in discovered:
            h = s.get("hash", "")
            # Normalize: strip sha256: prefix if present
            if h.startswith("sha256:"):
                h = h[7:]
            disk_hashes[s["name"]] = h[:12] if h else ""

        # Fetch platform skills
        try:
            platform_skills = self.list_skills()
            platform_names = {s["name"] for s in platform_skills}
            # The backend serializes content_hash
            # ONLY under the nested ``latest_version`` object (never as a
            # top-level key), and it is the FULL 64-char SHA-256 while the disk
            # hash above is truncated to 12 chars. Read the nested key and
            # compare by prefix — exactly as pull_missing() does
            # (skill_router.py:1774). The old code read a non-existent top-level
            # key (always "") and compared full-vs-12-char with ==, so every
            # synced skill was mislabelled "modified_locally".
            platform_hashes = {
                s["name"]: (s.get("latest_version") or {}).get("content_hash", "")
                for s in platform_skills
            }
        except Exception:
            logger.warning("Could not fetch platform skills for status check")
            platform_names = set()
            platform_hashes = {}

        # Classify
        synced = []
        modified_locally = []
        missing_locally = []
        untracked = []

        for name in platform_names:
            if name in disk_hashes:
                disk_h = disk_hashes[name]
                plat_h = platform_hashes.get(name, "")
                # Drift only when both hashes are known and the disk prefix is
                # NOT a prefix of the platform hash. If either side is unknown we
                # can't prove drift, so treat as synced (mirrors pull_missing).
                if disk_h and plat_h and not plat_h.startswith(disk_h):
                    modified_locally.append(name)
                else:
                    synced.append(name)
            else:
                missing_locally.append(name)

        for name in disk_names:
            if name not in platform_names:
                untracked.append(name)

        return {
            "synced": sorted(synced),
            "modified_locally": sorted(modified_locally),
            "missing_locally": sorted(missing_locally),
            "untracked": sorted(untracked),
            "total_on_disk": len(disk_names),
            "total_on_platform": len(platform_names),
        }

    def update_skills(
        self,
        *,
        agents: Optional[List[str]] = None,
        scope: str = "project",
        project_root: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Pull upstream changes from the platform and update local files.

        Compares local content hashes against the platform. If the
        platform version is newer (different hash), re-exports the
        skill to disk. Local changes are overwritten.

        Args:
            agents: Agent runtimes to update for. Uses lockfile agents
                    if not specified.
            scope: 'project' or 'global'.
            project_root: Project root directory.
            force: If True, re-export ALL skills regardless of hash match.

        Returns:
            Update summary with updated, skipped, and failed counts.

        Example::

            router = SkillRouter(api_key="dai_sk_...")
            result = router.update_skills()
            print(f"Updated {result['updated']} skills")
        """
        from .disk_export import _read_lockfile

        root = project_root or os.getcwd()
        lockdata = _read_lockfile(root)
        locked = lockdata.get("skills", {})

        # Get platform hashes
        try:
            hash_result = self._request("GET", "/api/v1/skills/hashes")
        except Exception as e:
            logger.error("Failed to fetch platform hashes: %s", e)
            return {"updated": 0, "skipped": 0, "failed": 0, "error": str(e)}

        platform_hashes = hash_result.get("hashes", {})

        # Determine which skills need updating
        to_update = []
        skipped = 0

        for name, info in platform_hashes.items():
            platform_hash = info.get("hash", "")
            local_entry = locked.get(name, {})
            # `body_hash` (full sha256 of body_markdown — the same axis the
            # platform serves) is the comparable value; `content_hash` now
            # pins the written FILE (frontmatter+body) so it never matches a
            # platform body hash. Legacy lockfiles stored sha256(body)[:12]
            # as content_hash — the prefix check keeps those in-sync entries
            # from re-exporting on every update.
            local_hash = local_entry.get("body_hash") or local_entry.get("content_hash") or ""

            up_to_date = bool(
                platform_hash and local_hash
                and (platform_hash == local_hash or platform_hash.startswith(local_hash))
            )
            if force or not up_to_date:
                to_update.append(name)
            else:
                skipped += 1

        if not to_update:
            logger.info("All skills are up to date")
            return {"updated": 0, "skipped": skipped, "failed": 0}

        # Resolve agents from lockfile if not specified
        if not agents and locked:
            first_entry = next(iter(locked.values()), {})
            agents = first_entry.get("agents", ["universal"])

        # Fetch and re-export outdated skills
        try:
            result = self.export_to_disk(
                skills=to_update,
                agents=agents,
                scope=scope,
                project_root=root,
            )
            return {
                "updated": result.get("skills_written", 0),
                "skipped": skipped,
                "failed": 0,
                "updated_skills": to_update,
            }
        except Exception as e:
            logger.error("Failed to update skills: %s", e)
            return {
                "updated": 0,
                "skipped": skipped,
                "failed": len(to_update),
                "error": str(e),
            }

    # ── Bidirectional Sync ─────────────────────────────────

    def pull_missing(
        self,
        *,
        local_skill_names: Optional[set] = None,
        agents: Optional[List[str]] = None,
        scope: str = "project",
        project_root: Optional[str] = None,
        disk_wins: bool = False,
    ) -> Dict[str, Any]:
        """Pull skills from platform that are missing locally or out-of-date.

        Compares the set of platform skills against local SKILL.md files
        and writes any missing/outdated skills to disk.

        By default, platform content wins on conflicts (the platform version
        overwrites the local version when hashes differ). Set ``disk_wins=True``
        to keep local edits and only pull truly missing skills.

        Args:
            local_skill_names: Skill names already on disk (to avoid re-scan).
                If None, will discover from disk.
            agents: Target agent runtimes for export (e.g. ["claude-code"]).
                Defaults to ["universal"].
            scope: 'project' or 'global'.
            project_root: Project root directory.
            disk_wins: If True, only pull skills that are completely missing
                on disk (don't overwrite local edits). If False (default),
                platform version wins — local files are updated to match
                the platform.

        Returns:
            Summary dict with pulled, updated, skipped counts.

        Example::

            router = SkillRouter(api_key="dai_sk_...")

            # Pull all platform skills to disk (platform wins on conflict)
            result = router.pull_missing(agents=["claude-code"])
            print(f"Pulled {result['pulled']} new, updated {result['updated']}")
        """
        from .skills import discover_skills

        root = project_root or os.getcwd()
        target_agents = agents or ["universal"]

        # Get platform skill hashes
        try:
            hash_result = self._request("GET", "/api/v1/skills/hashes")
            platform_hashes = hash_result.get("hashes", {})
        except Exception as e:
            logger.warning("pull_missing: failed to fetch platform hashes: %s", e)
            return {"pulled": 0, "updated": 0, "skipped": 0, "error": str(e)}

        if not platform_hashes:
            return {"pulled": 0, "updated": 0, "skipped": 0}

        # Get local disk state
        if local_skill_names is None:
            discovered = discover_skills() or []
            local_skill_names = {s["name"] for s in discovered}
            # Build local hash map for conflict detection
            local_hashes: Dict[str, str] = {}
            for s in discovered:
                h = s.get("hash", "")
                # Normalize: strip sha256: prefix if present
                if h.startswith("sha256:"):
                    h = h[7:]
                local_hashes[s["name"]] = h[:12] if h else ""
        else:
            local_hashes = {}

        # Classify skills
        to_pull: List[str] = []  # Missing from disk entirely
        to_update: List[str] = []  # On disk but hash differs
        skipped = 0

        for name, info in platform_hashes.items():
            platform_hash = info.get("hash", "")

            if name not in local_skill_names:
                # Missing locally — always pull
                to_pull.append(name)
            elif not disk_wins:
                # Conflict check: platform wins by default
                local_h = local_hashes.get(name, "")
                if local_h and platform_hash and not platform_hash.startswith(local_h):
                    to_update.append(name)
                else:
                    skipped += 1
            else:
                skipped += 1

        all_to_write = to_pull + to_update
        if not all_to_write:
            return {"pulled": 0, "updated": 0, "skipped": skipped}

        # Fetch full content and write to disk
        try:
            result = self.export_to_disk(
                skills=all_to_write,
                agents=target_agents,
                scope=scope,
                project_root=root,
            )
            written = result.get("skills_written", 0)
            logger.info(
                "pull_missing: wrote %d skills (%d new, %d updated)",
                written, len(to_pull), len(to_update),
            )
            return {
                "pulled": len(to_pull),
                "updated": len(to_update),
                "skipped": skipped,
                "pulled_skills": to_pull,
                "updated_skills": to_update,
            }
        except Exception as e:
            # warning, not error: pull_missing runs as a best-effort background
            # sync, so a failed export must not take the caller down — the
            # result dict already carries the error for callers that need to
            # act on it.
            logger.warning("pull_missing: export failed: %s", e)
            return {
                "pulled": 0,
                "updated": 0,
                "skipped": skipped,
                "error": str(e),
            }

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
            badge: Filter by badge ("verified", "featured", "community", "imported").
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

    # Alias for discoverability
    search_registry = search
