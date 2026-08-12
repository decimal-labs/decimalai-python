"""Shared infrastructure for the live-LLM integration test layers.

Used by every `test_framework_live_*.py` file in this directory. Keeping
config, polling, assertions, and tool primitives in one place so each layer
file stays focused on its own scenario.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from uuid import uuid4

import pytest


# ─── Backend / SDK config ────────────────────────────────────────────

BACKEND_URL = os.environ.get("DECIMAL_BACKEND_URL", "http://localhost:8000")
API_KEY = os.environ.get("DECIMAL_API_KEY", "dai_sk_test_key_001")

# ─── Model selection by *tier*, never a hardcoded id ─────────────────
#
# Hardcoding a model id rots. The previous values here — gemini-2.5-flash and
# gpt-5-mini — had both been superseded (gemini-3.5-flash shipped GA at I/O
# 2026; the current routed-production mini is gpt-5.4-mini). So select by tier
# and let the concrete id float to whatever the current mainstream model is:
#   - budget   = the routed-production model most users actually run (gate default)
#   - frontier = the flagship, for the occasional high-bar cell
# Override the whole column with LIVE_LLM_TIER=frontier, or pin one id with
# LIVE_LLM_GEMINI_MODEL / LIVE_LLM_OPENAI_MODEL (the release gate exports these
# so the matrix has a single source of truth). The
# latest-model canary resolves newer ids from each provider's models API and
# files drift — bump this map deliberately when it does, like a lockfile.
MODEL_TIERS = {
    # google/openai ids verified listed+callable against the live models API on
    # 2026-05-30. The anthropic ids are from the current model card but NOT yet
    # live-verified (no ANTHROPIC_API_KEY at authoring) — canary + bump like a
    # lockfile once a key is available. budget = the cheap tier (Haiku, the
    # cost analog of flash/mini); frontier = the flagship (Sonnet for anthropic —
    # Opus is hard-blocked by _guard_no_opus to stop accidental high-cost spend).
    "budget":   {"google": "gemini-3.5-flash", "openai": "gpt-5.4-mini", "anthropic": "claude-haiku-4-5-20251001"},
    # Flagship tier. google stays on stable 2.5-pro: the 3.x pro line is preview-only.
    "frontier": {"google": "gemini-2.5-pro",   "openai": "gpt-5.5",      "anthropic": "claude-sonnet-4-6"},  # NOT opus
}
LIVE_LLM_TIER = os.environ.get("LIVE_LLM_TIER", "budget")


def _guard_no_opus(model: str) -> str:
    """Hard stop on Opus: it is the priciest model and has drained the API budget
    when selected by accident (a stray ``LIVE_LLM_TIER=frontier`` or ``*_MODEL``
    override). Refuse any opus id unless ``DECIMAL_ALLOW_OPUS=1`` opts in."""
    if "opus" in model.lower() and os.environ.get("DECIMAL_ALLOW_OPUS") != "1":
        raise RuntimeError(
            f"Opus model {model!r} is blocked to prevent accidental high-cost spend. "
            f"Set DECIMAL_ALLOW_OPUS=1 to use it deliberately."
        )
    return model


def resolve_model(provider: str) -> str:
    """Concrete model id for a provider: an explicit per-provider env override
    wins, else the LIVE_LLM_TIER column of MODEL_TIERS. Opus is blocked unless
    DECIMAL_ALLOW_OPUS=1."""
    env_var = {
        "google": "LIVE_LLM_GEMINI_MODEL",
        "openai": "LIVE_LLM_OPENAI_MODEL",
        "anthropic": "LIVE_LLM_ANTHROPIC_MODEL",
    }[provider]
    return _guard_no_opus(os.environ.get(env_var) or MODEL_TIERS[LIVE_LLM_TIER][provider])


GEMINI_MODEL = resolve_model("google")
OPENAI_MODEL = resolve_model("openai")
ANTHROPIC_MODEL = resolve_model("anthropic")
POLL_TIMEOUT_S = 15
POLL_INTERVAL_S = 0.5
LIVE_TESTS_ENABLED = os.environ.get("RUN_LIVE_LLM_TESTS") == "1"


# ─── Framework × provider matrix (single source of truth) ────────────
#
# Every live test parametrizes its (provider, model) cells through `matrix()`
# rather than copy-pasting a param list, so the framework↔provider pairing
# lives in exactly one place. Mirrors the release gate's own
# FRAMEWORK_PROVIDERS table — kept in sync by hand because the SDK suite is
# standalone and can't import the gate package.
#
# Provider-native frameworks pair only with their own vendor; absence here
# means "all providers". This is a deliberate pairing, not a workaround:
# the OpenAI Agents SDK emits strict tool schemas Gemini's OpenAI-compat
# endpoint rejects, and running it on Gemini would test a combination nobody
# ships — so it stays OpenAI-only even though it can be coaxed to pass. ADK
# is Gemini-native for the mirror reason.
#
# anthropic is a first-class generic provider: Claude leads the agent-builder
# audience (enterprise API $ + coding agents), so every generic framework runs
# on it too. Unlike the Gemini-on-OpenAI-compat shim, the anthropic cells drive
# the native `anthropic` SDK / `langchain_anthropic`, so Anthropic's own usage
# fields (usage.input_tokens / output_tokens) are exercised — the exact token
# divergence this lane exists to close.
ALL_PROVIDERS = ("google", "openai", "anthropic")
FRAMEWORK_PROVIDERS = {
    "openai_agents": ("openai",),
    "adk": ("google",),
    # The Claude Agent SDK is Anthropic's own agent framework (it drives the
    # Claude Code engine), so it pairs only with the anthropic provider — the
    # mirror of adk↔google / openai_agents↔openai.
    "claude_agent_sdk": ("anthropic",),
}


def matrix(framework: str, *, only: tuple[str, ...] | None = None) -> list:
    """pytest.param list of (provider, model) cells for a framework lane.

    Providers come from `FRAMEWORK_PROVIDERS` (or all providers if the
    framework isn't listed), optionally narrowed by `only` for a scenario
    that further restricts on *setup* grounds rather than pairing — e.g. a
    test hard-wired to one vendor's client, or a native extra that installs
    a single provider. The model is resolved per provider for the active
    tier; the param id is the provider name, so the release gate's
    per-provider `-k` lane selectors match every cell.
    """
    providers = FRAMEWORK_PROVIDERS.get(framework, ALL_PROVIDERS)
    if only is not None:
        providers = tuple(p for p in providers if p in only)
    return [pytest.param(p, resolve_model(p), id=p) for p in providers]


def make_react_agent(model, tools, prompt=None):
    """Build a ReAct agent across langchain / langgraph versions.

    langgraph >=1.0 moved `create_react_agent` to `langchain.agents.create_agent`
    (renaming the `prompt=` kwarg to `system_prompt=`) and deprecated the old
    symbol. The `latest` clean-room lane runs `-W error::DeprecationWarning`, so
    calling the deprecated `langgraph.prebuilt.create_react_agent` *raises* there.
    Prefer the new `create_agent`; fall back to the old symbol on older stacks
    (the floor / pinned lanes) where `langchain.agents.create_agent` doesn't
    exist yet. Both return a `CompiledStateGraph` that decimalai's introspection
    (`langchain_introspect`) handles identically, so this is transparent to every
    matrix assertion.
    """
    try:
        from langchain.agents import create_agent
    except ImportError:
        from langgraph.prebuilt import create_react_agent
        if prompt is not None:
            return create_react_agent(model, tools, prompt=prompt)
        return create_react_agent(model, tools)
    if prompt is not None:
        return create_agent(model, tools, system_prompt=prompt)
    return create_agent(model, tools)


# ─── Domain — shopping cart (used by simple-agent layer) ────────────

PRICES = {"widget": 10, "gadget": 25}
SHOPPING_QUERY = (
    "I'd like 3 widgets and 2 gadgets. Use your tools to look up the unit "
    "price of each item, then use your tools to compute the total cost. "
    "Reply with the total cost in dollars."
)
SHOPPING_EXPECTED_TOTAL = 3 * PRICES["widget"] + 2 * PRICES["gadget"]  # 80


def lookup_price(item: str) -> int:
    key = item.lower().strip().rstrip("s")
    return PRICES.get(key, 0)


def safe_calculate(expression: str) -> float:
    allowed = set("0123456789 +-*/().")
    if not all(c in allowed for c in expression):
        raise ValueError(f"unsafe expression: {expression!r}")
    return eval(expression, {"__builtins__": {}})  # noqa: S307


# ─── Domain — customer support (used by complex/error/multi-agent) ──
#
# Five tools, deliberately giving the agent multiple ways to combine them.
# get_order_details raises ValueError for unknown IDs so the error-recovery
# layer can exercise the failing-tool path with the same fixtures.

CUSTOMERS = {
    1234: {"name": "Stanley", "tier": "Pro", "joined": "2024-08-01"},
    5678: {"name": "Alex", "tier": "Free", "joined": "2025-01-15"},
}

ORDERS_BY_CUSTOMER = {
    1234: [("ORD-9001", "2026-05-01"), ("ORD-8003", "2025-12-12")],
    5678: [("ORD-7711", "2026-04-22")],
}

ORDER_DETAILS = {
    "ORD-9001": {"status": "delivered", "items": ["Pro Laptop"], "total": 1200},
    "ORD-8003": {"status": "delivered", "items": ["USB Hub"], "total": 35},
    "ORD-7711": {"status": "in_transit", "items": ["Cable"], "total": 12},
}

FAQ_INDEX = {
    "damaged delivery": (
        "Damaged-on-arrival orders are eligible for a full refund plus 10% "
        "store credit when reported within 30 days."
    ),
    "return policy": "Returns are accepted within 30 days for unopened items.",
    "shipping": "Standard shipping is 3-5 business days.",
}

SUPPORT_QUERY = (
    "I'm customer 1234. My order ORD-9001 arrived broken. "
    "What can you do for me?"
)
SUPPORT_QUERY_BAD_ORDER = (
    "I'm customer 1234. My order ORD-INVALID-999 is broken. "
    "What can you do for me?"
)


def get_customer(customer_id: int) -> dict:
    """Return customer profile by ID. Raises if not found."""
    if customer_id not in CUSTOMERS:
        raise ValueError(f"Customer {customer_id} not found")
    return CUSTOMERS[customer_id]


def get_orders(customer_id: int) -> list[dict]:
    """Return list of orders for a customer (may be empty)."""
    return [
        {"order_id": oid, "date": d}
        for oid, d in ORDERS_BY_CUSTOMER.get(customer_id, [])
    ]


def get_order_details(order_id: str) -> dict:
    """Return order details. Raises ValueError for unknown order IDs —
    used by the error-recovery layer to trigger a failing tool call."""
    if order_id not in ORDER_DETAILS:
        raise ValueError(f"Order {order_id} not found in our system")
    return ORDER_DETAILS[order_id]


def search_faq(query: str) -> str:
    """Best-match FAQ lookup. Returns 'no result' string if no match."""
    q = query.lower()
    for key, text in FAQ_INDEX.items():
        if any(word in q for word in key.split()):
            return text
    return "No matching FAQ entry."


def calculate_refund(order_total: float, condition: str) -> dict:
    """Compute refund based on item condition."""
    cond = condition.lower()
    if cond in ("damaged", "broken", "defective"):
        return {"refund": order_total, "credit": round(order_total * 0.10, 2)}
    if cond == "unopened":
        return {"refund": order_total, "credit": 0}
    return {"refund": 0, "credit": 0}


# ─── Backend polling + assertions ────────────────────────────────────

def unique_agent(prefix: str) -> str:
    """Generate a unique agent name so re-runs don't collide."""
    return f"live-{prefix}-{datetime.now().strftime('%H%M%S')}-{uuid4().hex[:6]}"


def backend_alive() -> bool:
    try:
        with urllib.request.urlopen(f"{BACKEND_URL}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def flush_sdk_sender() -> None:
    from decimalai._config import _sender
    _sender.flush(timeout=POLL_TIMEOUT_S)


def list_agent_traces(agent_name: str) -> list[dict]:
    url = f"{BACKEND_URL}/api/v1/traces?agent_name={agent_name}&limit=20"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())["traces"]


def list_manifests(agent_name: str) -> list[dict]:
    """List manifests registered for an agent (newest first)."""
    url = f"{BACKEND_URL}/api/v1/manifests?agent_name={agent_name}&limit=20"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())["manifests"]


def post_regression_check(agent_name: str, candidate_manifest_id: str) -> dict:
    """Run a regression check against a candidate manifest. Returns the impact report."""
    url = f"{BACKEND_URL}/api/v1/regression-check"
    body = json.dumps({
        "agent_name": agent_name,
        "candidate_manifest_id": candidate_manifest_id,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def get_trace_detail(trace_id: str) -> dict:
    url = f"{BACKEND_URL}/api/v1/traces/{trace_id}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def poll_for_trace(agent_name: str, expected_count: int = 1) -> list[dict]:
    deadline = time.time() + POLL_TIMEOUT_S
    last = []
    while time.time() < deadline:
        last = list_agent_traces(agent_name)
        if len(last) >= expected_count:
            return last
        time.sleep(POLL_INTERVAL_S)
    raise AssertionError(
        f"Timed out waiting for {expected_count} trace(s) on agent={agent_name}; "
        f"last saw {len(last)}."
    )


_LLM_CALL_SPAN_NAMES = {"response", "generation", "llm", "chat_model"}


def assert_rich_agent_trace(
    detail: dict,
    *,
    min_llm_calls: int,
    min_tool_calls: int,
    min_distinct_tools: int = 0,
) -> None:
    """Verify a trace looks like a real multi-step agent run.

    Different adapters record LLM and tool calls in different places:
      - LangChain: LLM calls in `llm_calls[]`, tool calls nested as
        `tool_calls` on each llm_call.
      - Generic (`decimalai.log_llm_call` / `log_tool_call`): LLM calls in
        `llm_calls[]`, tool calls as top-level spans with `span_type=tool`.
      - OpenAI Agents SDK: LLM calls as spans named `response`/`generation`,
        tool calls as spans with `span_type=tool`.
    Count both so the assertion is adapter-agnostic.
    """
    llm_calls = detail.get("llm_calls", [])
    spans = detail.get("spans", [])
    llm_spans = [s for s in spans if (s.get("name") or "").lower() in _LLM_CALL_SPAN_NAMES]
    llm_call_count = len(llm_calls) + len(llm_spans)

    nested_tool_calls_per_llm = [c.get("tool_calls") or [] for c in llm_calls]
    nested_tool_calls = sum(len(tcs) for tcs in nested_tool_calls_per_llm)
    tool_spans = [s for s in spans if s.get("span_type") == "tool"]
    tool_call_count = nested_tool_calls + len(tool_spans)

    # Distinct tool names — across both nested tool_calls and tool spans.
    distinct_tool_names = set()
    for tcs in nested_tool_calls_per_llm:
        for tc in tcs:
            name = tc.get("name") or tc.get("tool_name") or tc.get("function", {}).get("name")
            if name:
                distinct_tool_names.add(name)
    for s in tool_spans:
        n = s.get("name")
        if n:
            distinct_tool_names.add(n)

    assert llm_call_count >= min_llm_calls, (
        f"Expected ≥ {min_llm_calls} LLM calls, "
        f"got llm_calls={len(llm_calls)} llm-named-spans={len(llm_spans)}. "
        f"Trace id={detail['id']}"
    )
    assert tool_call_count >= min_tool_calls, (
        f"Expected ≥ {min_tool_calls} tool invocations, "
        f"got nested={nested_tool_calls} spans={len(tool_spans)}. "
        f"Trace id={detail['id']}"
    )
    if min_distinct_tools:
        assert len(distinct_tool_names) >= min_distinct_tools, (
            f"Expected ≥ {min_distinct_tools} DISTINCT tool names, "
            f"got {len(distinct_tool_names)}: {sorted(distinct_tool_names)}. "
            f"Trace id={detail['id']}"
        )
    assert detail.get("manifest_id"), "manifest_id missing — auto-detection failed"


def require_key_for(provider: str) -> None:
    """Skip the current test cell if the provider's API key is missing."""
    if provider == "google" and not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")


# Lowercased substrings that mark a provider-side availability problem — out of
# quota, rate-limited, billing-disabled, or the provider's own fleet being
# overloaded — rather than a defect in our SDK or backend. OpenAI raises 429
# insufficient_quota / RateLimitError; Gemini raises 429 RESOURCE_EXHAUSTED when
# rate-limited and 503 UNAVAILABLE ("high demand… try again later") when its
# fleet is saturated. All are environmental and TRANSIENT, so the right outcome
# is SKIP, not FAIL: a real regression must stay distinguishable from "the key
# ran out" or "the provider was briefly overloaded." A momentary 503 must not
# turn the whole live matrix (or the release gate's live tier) red.
_PROVIDER_UNAVAILABLE_MARKERS = (
    # ── 429: quota / rate limit ──
    "insufficient_quota",
    "exceeded your current quota",
    "ratelimiterror",
    "rate limit",
    "rate_limit",
    "too many requests",
    "resource_exhausted",
    "error code: 429",
    "status code: 429",
    "code: 429",
    # ── 5xx / overload: provider fleet saturated or transiently down ──
    "error code: 503",
    "status code: 503",
    "code: 503",
    "503 unavailable",
    "service unavailable",
    "temporarily unavailable",
    "overloaded",
    "high demand",
    "try again later",
    "internalservererror",
    "error code: 500",
    # ── 529: Anthropic-specific "overloaded_error" ──
    "error code: 529",
    "status code: 529",
    "code: 529",
    "overloaded_error",
    # ── billing wall: account out of credits / billing not set up. Anthropic
    # returns this as a 400 invalid_request_error (NOT a 429), and OpenAI's
    # analog ("insufficient_quota") is already above — both are environmental
    # "the account can't pay", not an SDK/backend defect, so SKIP not FAIL.
    # Keep these phrases SPECIFIC (not a bare "billing") so a real error that
    # merely mentions billing isn't silently downgraded to a skip.
    "credit balance is too low",
    "credit balance too low",
    # ── client-side timeout: a model/HTTP call exceeded our timeout (a hung
    # socket or a slow provider), or a 504 gateway timeout. Environmental/
    # transient → retry then SKIP, never hang. Catches genai DeadlineExceeded,
    # httpx Read/ConnectTimeout (all contain "timeout"), and 504s.
    "deadline",
    "timed out",
    "timeout",
    "error code: 504",
    "status code: 504",
)


def is_provider_unavailable_error(exc: BaseException) -> bool:
    """True if ``exc`` (or anything in its cause/context chain) is a provider
    quota / rate-limit / billing wall, or a transient 5xx overload, so the
    caller can SKIP rather than FAIL.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        text = f"{type(cur).__name__}: {cur}".lower()
        if any(m in text for m in _PROVIDER_UNAVAILABLE_MARKERS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


# Gemini's OpenAI-compatible endpoint — lets OpenAI-shaped clients (incl. the
# OpenAI Agents SDK) call Gemini with a GEMINI_API_KEY and no OpenAI key.
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def openai_agents_model(provider: str, model: str):
    """Return a model spec for an `agents.Agent` for the given provider.

    - 'openai': the bare model string → SDK's default OpenAI client (OPENAI_API_KEY).
    - 'google': an OpenAIChatCompletionsModel pointed at Gemini's OpenAI-compatible
      endpoint and authed with GEMINI_API_KEY, so the Agents SDK runs on Gemini with
      no OpenAI key. Tracing is unaffected — it flows through whatever processor is
      installed via set_trace_processors(), independent of the model backend.
    """
    if provider == "openai":
        return model
    if provider == "google":
        from openai import AsyncOpenAI
        from agents import OpenAIChatCompletionsModel
        client = AsyncOpenAI(
            base_url=GEMINI_OPENAI_BASE_URL,
            api_key=os.environ["GEMINI_API_KEY"],
        )
        return OpenAIChatCompletionsModel(model=model, openai_client=client)
    raise ValueError(f"unknown provider {provider!r}")


def _loggable_turns(contents: list) -> list:
    """Serialize a google.genai ``contents`` history into the ``log_llm`` input
    shape. Model turns are appended as the SDK's own ``types.Content`` so the
    Gemini-3.x ``thought_signature`` survives the round-trip, so this must accept
    both plain dicts and ``Content`` objects."""
    turns = []
    for p in contents:
        if isinstance(p, dict):
            role, parts = p.get("role", ""), p.get("parts", [])
        else:
            role, parts = getattr(p, "role", "") or "", getattr(p, "parts", []) or []
        turns.append({"role": str(role), "content": json.dumps(parts, default=str)})
    return turns


# ── Gemini call resilience: client-side timeout + optional throttle ──────────
# The gate once hung ~10h on a native Gemini call that had NO client-side timeout
# (a stuck socket). And the free-tier key rate-limits (429) under load. These give
# every native-genai call a hard timeout (a hung call raises → retried → ultimately
# SKIPped, never hangs) and an optional proactive throttle (min seconds between
# calls) to stay under a low-RPM key's cap. Both are env-tunable.
GEMINI_CALL_TIMEOUT_S = float(os.environ.get("LIVE_LLM_GEMINI_TIMEOUT_S", "90"))
_GEMINI_MIN_INTERVAL_S = float(os.environ.get("LIVE_LLM_GEMINI_MIN_INTERVAL_S", "0"))
_gemini_last_call_at = [0.0]


def gemini_throttle() -> None:
    """Sleep so consecutive native-genai calls are ≥ LIVE_LLM_GEMINI_MIN_INTERVAL_S
    apart (proactive 429 avoidance; no-op when the interval is 0). Test cells run
    sequentially, so a plain timestamp suffices — no lock needed."""
    if _GEMINI_MIN_INTERVAL_S <= 0:
        return
    wait = _GEMINI_MIN_INTERVAL_S - (time.monotonic() - _gemini_last_call_at[0])
    if wait > 0:
        time.sleep(wait)
    _gemini_last_call_at[0] = time.monotonic()


def gemini_client(api_key: str | None = None):
    """A ``google.genai`` client with a client-side request timeout, so a hung HTTP
    call raises (→ retried/skipped) instead of hanging the run. Use everywhere
    instead of a bare ``genai.Client()``."""
    from google import genai
    from google.genai import types
    return genai.Client(
        api_key=api_key or os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(timeout=int(GEMINI_CALL_TIMEOUT_S * 1000)),
    )


def chat_google_genai(model: str, **kwargs):
    """``ChatGoogleGenerativeAI`` with a request timeout + bounded retries baked in,
    so the langchain Gemini cells fail fast on a hung call and ride out 429s. Use
    instead of constructing ``ChatGoogleGenerativeAI`` directly."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    kwargs.setdefault("temperature", 0)
    kwargs.setdefault("timeout", GEMINI_CALL_TIMEOUT_S)
    kwargs.setdefault("max_retries", 6)
    return ChatGoogleGenerativeAI(model=model, **kwargs)


def _generate_with_retry(client, *, model, contents, config, attempts: int = 4,
                         base_delay: float = 2.0):
    """``generate_content`` with bounded backoff on transient provider 5xx/overload.

    Gemini's budget tier intermittently returns 503 UNAVAILABLE ("high demand…")
    under load; retrying the single call in place (vs. re-running the whole agent
    loop) rides out the spike and keeps the run to ONE trace. A persistent failure
    re-raises so the caller still SKIPs rather than hangs. Only the provider-
    unavailable class is retried — a real error propagates on the first hit.
    """
    last: BaseException | None = None
    for i in range(attempts):
        try:
            gemini_throttle()
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:  # noqa: BLE001 — re-raised below unless transient
            if not is_provider_unavailable_error(e):
                raise
            last = e
            if i < attempts - 1:
                time.sleep(base_delay * (2 ** i))
    assert last is not None
    raise last


def gemini_tool_loop(
    model: str,
    query: str,
    *,
    tool_declarations: list,
    handlers: dict,
    log_llm,
    log_tool,
    system: str | None = None,
    max_iters: int = 8,
) -> str:
    """Drive a native ``google.genai`` function-calling loop, logging each LLM
    and tool call via the supplied callables.

    The generic-decorator path deliberately exercises the *native* provider SDK
    (see ``_gemini_generic_loop`` in test_framework_live_llm.py) rather than the
    OpenAI-compatible shim, so the decorator is proven against real Gemini
    request/response shapes and ``usage_metadata`` token fields. This shared
    driver lets the richer scenarios (complex, tool-error) reuse that path
    instead of re-deriving the google.genai boilerplate.

    Args:
        tool_declarations: google.genai function-declaration dicts (note the
            uppercase JSON-schema types: ``OBJECT``/``STRING``/``INTEGER``/…).
        handlers: ``{tool_name: callable}``. A handler may raise — the exception
            text is fed back to the model as the ``function_response`` and the
            tool call is logged with ``status="error"`` (the error-recovery path).
        log_llm / log_tool: callables with the ``decimalai.log_llm_call`` /
            ``decimalai.log_tool_call`` signatures (module-level or a
            ``start_trace`` ctx's bound methods).
        system: optional system instruction.
    """
    from google import genai
    from google.genai import types

    client = gemini_client()
    tool_obj = types.Tool(function_declarations=tool_declarations)
    cfg_kwargs: dict = {"tools": [tool_obj]}
    if system:
        cfg_kwargs["system_instruction"] = system
    config = types.GenerateContentConfig(**cfg_kwargs)

    contents: list = [{"role": "user", "parts": [{"text": query}]}]
    for _ in range(max_iters):
        resp = _generate_with_retry(
            client, model=model, contents=contents, config=config,
        )
        log_llm(
            model=model,
            input=_loggable_turns(contents),
            output={"content": resp.text or ""},
            input_tokens=getattr(resp.usage_metadata, "prompt_token_count", None),
            output_tokens=getattr(resp.usage_metadata, "candidates_token_count", None),
        )
        cand = resp.candidates[0]
        func_calls = [p for p in (cand.content.parts or []) if getattr(p, "function_call", None)]
        if not func_calls:
            return resp.text or ""
        # Append the model turn verbatim — reconstructing it from function_call alone
        # drops the thought_signature that Gemini 3.x requires on the next turn.
        contents.append(cand.content)
        for fc in func_calls:
            name = fc.function_call.name
            args = dict(fc.function_call.args or {})
            try:
                result = handlers[name](**args)
                status = "success"
            except Exception as e:
                result = f"ERROR: {e}"
                status = "error"
            log_tool(name=name, input=args, output={"result": result}, status=status)
            contents.append({
                "role": "user",
                "parts": [{"function_response": {"name": name, "response": {"result": result}}}],
            })
    raise RuntimeError("Agent loop exceeded safety bound")


# ─── Anthropic-native driver (analog of gemini_tool_loop) ────────────

def _anthropic_loggable_turns(messages: list) -> list:
    """Serialize an anthropic ``messages`` history into the ``log_llm`` input
    shape. Content may be a plain string or a list of content blocks (text /
    tool_use / tool_result), so non-str content is JSON-encoded."""
    turns = []
    for m in messages:
        content = m.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, default=str)
        turns.append({"role": str(m.get("role", "")), "content": content})
    return turns


def _anthropic_create_with_retry(client, *, attempts: int = 4, base_delay: float = 2.0, **kwargs):
    """``messages.create`` with bounded backoff on transient provider 5xx/overload.

    Mirrors ``_generate_with_retry``: Anthropic returns 429 (rate/quota) and 529
    ``overloaded_error`` under load; retrying the single call in place rides out
    the spike and keeps the run to ONE trace. Only the provider-unavailable class
    is retried — a real error propagates on the first hit.
    """
    _guard_no_opus(str(kwargs.get("model", "")))  # last-line block: no opus call ships, however the id arrived
    last: BaseException | None = None
    for i in range(attempts):
        try:
            return client.messages.create(**kwargs)
        except Exception as e:  # noqa: BLE001 — re-raised below unless transient
            if not is_provider_unavailable_error(e):
                raise
            last = e
            if i < attempts - 1:
                time.sleep(base_delay * (2 ** i))
    assert last is not None
    raise last


def anthropic_tool_loop(
    model: str,
    query: str,
    *,
    tools: list,
    handlers: dict,
    log_llm,
    log_tool,
    system: str | None = None,
    max_iters: int = 8,
    max_tokens: int = 1024,
) -> str:
    """Drive a native ``anthropic`` Messages tool-use loop, logging each LLM and
    tool call via the supplied callables. The Anthropic analog of
    ``gemini_tool_loop``.

    Deliberately exercises the *native* anthropic SDK rather than an
    OpenAI-compat shim, so the generic-decorator path is proven against real
    Claude request/response shapes and — the point of this lane — Anthropic's
    token field names ``usage.input_tokens`` / ``usage.output_tokens`` (NOT the
    OpenAI ``prompt_tokens`` / ``completion_tokens``).

    Args:
        tools: Anthropic tool dicts — ``{"name", "description", "input_schema"}``
            where input_schema is a JSON schema (lowercase types:
            ``object`` / ``string`` / ``integer`` / …).
        handlers: ``{tool_name: callable}``. A handler may raise — the error text
            is returned as the ``tool_result`` with ``is_error=True`` and the
            tool call is logged ``status="error"`` (the error-recovery path).
        log_llm / log_tool: ``decimalai.log_llm_call`` / ``log_tool_call``-shaped
            callables.
        system: optional system prompt (Anthropic takes it as a top-level arg,
            not a message).
    """
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    create_kwargs: dict = {"model": model, "max_tokens": max_tokens, "tools": tools}
    if system:
        create_kwargs["system"] = system

    messages: list = [{"role": "user", "content": query}]
    for _ in range(max_iters):
        resp = _anthropic_create_with_retry(client, messages=messages, **create_kwargs)
        text_out = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        )
        log_llm(
            model=model,
            input=_anthropic_loggable_turns(messages),
            output={"content": text_out},
            input_tokens=getattr(resp.usage, "input_tokens", None),
            output_tokens=getattr(resp.usage, "output_tokens", None),
        )
        tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        if resp.stop_reason != "tool_use" or not tool_uses:
            return text_out
        # Append the assistant turn verbatim so tool_use ids thread into the
        # matching tool_result blocks on the next user turn.
        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for tu in tool_uses:
            args = dict(tu.input or {})
            try:
                result = handlers[tu.name](**args)
                status, is_error = "success", False
            except Exception as e:
                result, status, is_error = f"ERROR: {e}", "error", True
            log_tool(name=tu.name, input=args, output={"result": result}, status=status)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps({"result": result}),
                "is_error": is_error,
            })
        messages.append({"role": "user", "content": tool_results})
    raise RuntimeError("Agent loop exceeded safety bound")


def get_topology(orchestrator_name: str) -> dict:
    """Fetch the multi-agent topology view that the SubagentHealthDashboard +
    DelegationAnalytics + AgentTopologyGraph components consume."""
    url = f"{BACKEND_URL}/api/v1/agents/{orchestrator_name}/topology"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def get_agent_summary(agent_name: str) -> dict:
    """Fetch a single agent's entry from the workspace agents list.

    Must pass ``include_fixtures=true``: every agent these live tests create is
    named ``live-*``, and the backend filters agents matching ``live-%`` out of
    the DEFAULT agents list so test fixtures never reach a real user's
    dashboard. Without the opt-in, the agent we just created is never in the
    list and this raced as a timeout forever — the agent is being hidden, which
    looks exactly like eventual consistency that never converges.

    Polling is kept (mirrors ``poll_for_trace``) for the genuine ingest lag: a
    just-created sub-agent's trace lands slightly before the list materializes.
    """
    url = f"{BACKEND_URL}/api/v1/agents?limit=200&include_fixtures=true"
    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {API_KEY}"})
        with urllib.request.urlopen(req, timeout=5) as r:
            for a in json.loads(r.read()).get("agents", []):
                if a.get("agent_name") == agent_name:
                    return a
        time.sleep(POLL_INTERVAL_S)
    raise AssertionError(f"Agent {agent_name!r} not found in workspace list")


def assert_topology_declared(orchestrator_name: str, specialist_name: str) -> None:
    """Verify the orchestrator's manifest declares the specialist as a subagent.

    This is the part of the multi-agent product surface that depends only on
    the *orchestrator* having been registered correctly — independent of
    whether the specialist actually ran on this particular invocation. The
    AgentTopologyGraph component reads from this endpoint.

    The backend splits declared subagents into two buckets: ``subagents``
    (the name also exists as an independently-registered agent, i.e. it produced
    its own traces/llm_calls → clickable in the UI) and ``unregistered_subagents``
    (declared in the manifest but never independently registered → rendered as a
    non-clickable "ghost" chip). Both mean the orchestrator *declared* the
    specialist, which is what this assertion is about — so we accept either.
    An OpenAI-Agents handoff runs the specialist inside the orchestrator's single
    trace, so the specialist legitimately lands in ``unregistered_subagents``;
    a LangChain handoff runs it as its own agent, so it lands in ``subagents``.
    """
    topo = get_topology(orchestrator_name)
    assert topo.get("has_topology") is True, (
        f"Topology endpoint reports has_topology=False for {orchestrator_name}. "
        f"Subagent UI would render empty. Full response: {topo}"
    )
    declared_names = (
        {s.get("name") for s in (topo.get("subagents") or [])}
        | {s.get("name") for s in (topo.get("unregistered_subagents") or [])}
    )
    assert specialist_name in declared_names, (
        f"Specialist {specialist_name!r} not declared in topology — neither in "
        f"subagents nor unregistered_subagents. declared={sorted(declared_names)}"
    )


def assert_parallel_tool_calls(
    detail: dict,
    *,
    min_parallel: int = 2,
    window_ms: int = 200,
) -> None:
    """Verify ≥N tool spans started within `window_ms` of each other.

    A real parallel fan-out from the LLM (e.g. `[get_price(a), get_price(b)]`)
    results in tool spans whose start times are clustered within milliseconds.
    Serial calls would be separated by the latency of the previous tool. We
    use a tight clustering window (default 200ms) to distinguish the two.

    Adapter-agnostic: works for LangChain (llm_calls[].tool_calls is empty
    on this backend shape — tool calls live in spans only) and OpenAI Agents
    (zero llm_calls; tool spans only).
    """
    spans = detail.get("spans", [])
    tool_spans = [
        s for s in spans
        if s.get("span_type") == "tool" and s.get("started_at")
    ]

    def parse(ts: Any):
        if isinstance(ts, str):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return ts

    starts = sorted(parse(s["started_at"]) for s in tool_spans)
    if len(starts) < min_parallel:
        raise AssertionError(
            f"Expected ≥ {min_parallel} tool spans, got {len(starts)}. "
            f"Trace id={detail['id']}"
        )

    # Sliding window: find min_parallel consecutive starts within window_ms.
    for i in range(len(starts) - min_parallel + 1):
        span_ms = (starts[i + min_parallel - 1] - starts[i]).total_seconds() * 1000
        if span_ms <= window_ms:
            return

    pairwise_ms = [
        round((starts[i+1] - starts[i]).total_seconds() * 1000, 2)
        for i in range(len(starts) - 1)
    ]
    raise AssertionError(
        f"Expected ≥ {min_parallel} tool spans clustered within "
        f"{window_ms}ms (parallel fan-out). "
        f"Inter-start gaps (ms): {pairwise_ms}. "
        f"Trace id={detail['id']}"
    )


def assert_subagent_resolved(specialist_name: str, orchestrator_name: str) -> None:
    """Verify the specialist appears in the agents list with `is_subagent=True`
    and the correct parent. Requires the specialist to have actually run on
    this invocation — use this only when the test deterministically triggers
    the sub-agent (e.g., orchestrator has a single tool that wraps it).
    """
    specialist = get_agent_summary(specialist_name)
    assert specialist.get("is_subagent") is True, (
        f"Specialist {specialist_name!r} not marked is_subagent=True "
        f"in agents list. Entry: {specialist}"
    )
    assert specialist.get("parent_agent_name") == orchestrator_name, (
        f"Specialist's parent_agent_name is "
        f"{specialist.get('parent_agent_name')!r}, expected {orchestrator_name!r}"
    )


# ─── Shared autouse fixtures (re-exported via conftest.py) ───────────

def require_gates_fixture():
    """Pytest fixture body — call this from each file's autouse fixture."""
    if not LIVE_TESTS_ENABLED:
        pytest.skip("Set RUN_LIVE_LLM_TESTS=1 to run live LLM tests.")
    if not backend_alive():
        pytest.skip(f"Backend at {BACKEND_URL} unreachable.")


def reset_sdk_fixture():
    """Pytest fixture body — call this from each file's autouse fixture."""
    import decimalai
    decimalai.init(api_key=API_KEY, base_url=BACKEND_URL, enabled=True)
