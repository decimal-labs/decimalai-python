"""HTTP client for communicating with the Decimal platform."""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
from typing import Any, Dict, List, Optional, cast
from uuid import UUID

import httpx

from ._config import sdk_headers
from ._responses import (
    AgentListResponse,
    IngestionResult,
    IngestionSkipped,
    ManifestDiffResponse,
    ManifestListResponse,
    ManifestRegistrationResponse,
    RegressionCheckListResponse,
    RegressionCheckResponse,
    TraceDetailResponse,
    TraceListResponse,
    VerifyAuthResponse,
)
from .schema.trace import RunTrace

logger = logging.getLogger("decimalai")

_MAX_RETRIES = 3
_DEFAULT_RETRY_DELAY = 1.0  # seconds

# How many traces `buffer_trace` accumulates before auto-flushing — and, because
# `flush()` now hangs on to a batch the backend was merely too sick to accept,
# the ceiling on how many traces may be carried across failed flushes. One
# number on purpose: the buffer's steady state is a single batch, so "keep one
# batch" is the whole retention policy and there is nothing to tune.
_AUTO_FLUSH_THRESHOLD = 50

# Statuses that say "the request never reached the application, or the
# application is transiently unhealthy" — the payload is not implicated, so the
# same bytes can succeed a second later. These are what a Google Frontend / load
# balancer returns while a Cloud Run revision is restarting or has no healthy
# instance.
#
# 500 is deliberately NOT in this set. It means the application DID run and blew
# up part-way, so replaying a non-idempotent POST can double-write. Call sites
# that are safe to replay opt in with `idempotent=True`.
_RETRYABLE_STATUSES = frozenset({502, 503, 504})

# How long `buffer_trace` stops triggering an AUTOMATIC flush after one failed
# retryably. Without it, a preserved buffer sits exactly AT
# `_AUTO_FLUSH_THRESHOLD`, so every single trace the caller records afterwards
# kicks off another full retry ladder (up to 1+2+4s of sleeping) on the caller's
# own thread — a backend outage would stall the user's agent instead of just
# delaying its traces. This is not a new backoff scheme: it is one more step of
# the existing one, i.e. the wait the next attempt would have started from.
_FLUSH_RETRY_COOLDOWN = _DEFAULT_RETRY_DELAY * (2 ** _MAX_RETRIES)  # 8.0s


def _parse_retry_after(value: Optional[str]) -> float:
    """Parse a ``Retry-After`` header into delta-seconds.

    Per RFC 7231 the value may be delta-seconds ("5")
    OR an HTTP-date ("Wed, 21 Oct 2025 07:28:00 GMT") — many proxies/CDNs/LBs
    (Cloudflare, nginx, GCP LB) emit the date form on 429. The old ``float(value)``
    raised an uncaught ValueError on the date form, defeating retry/backoff on
    every write path. Returns 0.0 on absent/unparseable (caller falls back to
    exponential backoff).
    """
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        from datetime import datetime, timezone
        from email.utils import parsedate_to_datetime
        when = parsedate_to_datetime(value)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return 0.0

# Warn-once latch. ``init(verify=True)`` caches the backend's
# ``require_manifest_on_ingest`` flag onto the global config; honoring the
# init() docstring, we surface an actionable warning at the source when a
# manual ingest omits ``manifest_id`` against a strict backend — instead of
# letting the caller hit a bare 400. Latched to once-per-process so a stream
# of mis-shaped traces can't spam the log.
_STRICT_MANIFEST_WARNED = False


def _warn_if_strict_manifest_missing(payload_has_manifest: bool) -> None:
    """Emit a one-time warning when the backend requires manifest_id and it's absent.

    No-op unless ``init(verify=True)`` cached ``require_manifest_on_ingest=True``
    (the default for prod backends is unset/False, so this stays silent there).
    """
    global _STRICT_MANIFEST_WARNED
    if payload_has_manifest or _STRICT_MANIFEST_WARNED:
        return
    from ._config import _config as _global_config

    if _global_config is None or not _global_config.backend_require_manifest_on_ingest:
        return
    _STRICT_MANIFEST_WARNED = True
    logger.warning(
        "Ingesting a trace without a manifest_id, but this backend requires one "
        "(require_manifest_on_ingest=True) — the server will reject it with a 400 "
        "'manifest_id is required'. Register a manifest first "
        "(decimalai.register_manifest(...)) and attach its id to the trace, or use "
        "a framework integration / the generic tracer, which register one "
        "automatically."
    )


def _scrub_surrogates(value: Any) -> Any:
    """Make a serialized payload safe to UTF-8 encode for upload.

    A lone UTF-16 surrogate (e.g. ``"\\ud800"``) is a valid Python ``str`` but
    cannot be encoded to UTF-8, so httpx's ``json=`` encoder
    (``ensure_ascii=False``) raises ``UnicodeEncodeError`` and crashes
    ``ingest_trace`` / ``ingest_traces_batch`` / ``register_manifest`` in the
    caller's process *before any request is sent*. A compliant client cannot put
    a lone surrogate on the wire, so we scrub the serialized copy — replacing
    un-encodable code points with the UTF-8 ``replace`` substitute — and proceed,
    rather than failing the whole upload. Operates on the serialized dict/list (never the caller's
    model), recurses into nested containers, and is a no-op for clean strings.
    Pairs with the server-side global backstop.
    """
    if isinstance(value, str):
        try:
            value.encode("utf-8")
            return value
        except UnicodeEncodeError:
            return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, dict):
        return {_scrub_surrogates(k): _scrub_surrogates(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_surrogates(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_scrub_surrogates(v) for v in value)
    return value


class DecimalRateLimitError(Exception):
    """Raised when the Decimal platform returns 429 after all retries are exhausted."""

    def __init__(self, retry_after: float = 0, message: str = ""):
        self.retry_after = retry_after
        super().__init__(message or f"Rate limit exceeded. Retry after {retry_after}s")


class DecimalQuotaExceededError(Exception):
    """Raised immediately when a plan quota is exhausted — never retried.

    A quota 429 and a rate-limit 429 share a status code but nothing else: a rate limit
    clears in seconds, a quota does not clear until the billing period rolls over. Retrying
    it cannot succeed, so the old behaviour (3 attempts with backoff, then a rate-limit
    error) burned wall-clock and then DROPPED the payload with a misleading message.

    The server marks the difference with an ``X-Quota-Exceeded`` header naming the
    dimension; ``resets_in_seconds`` on the body says when capacity returns. Deliberately
    NOT a subclass of DecimalRateLimitError — code that catches a rate limit to sleep and
    retry must not swallow this one.
    """

    def __init__(self, dimension: str = "", resets_in_seconds: int = 0,
                 plan: str = "", message: str = ""):
        self.dimension = dimension
        self.resets_in_seconds = resets_in_seconds
        self.plan = plan
        super().__init__(message or (
            f"Plan quota exhausted for {dimension!r}"
            + (f" on the {plan!r} plan" if plan else "")
            + (f"; resets in {resets_in_seconds}s" if resets_in_seconds else "")
            + ". Retrying will not help — upgrade the plan or wait for the period to roll over."
        ))


class DecimalAPIError(httpx.HTTPStatusError):
    """An HTTP error from the Decimal platform, enriched with the server's message.

    ``httpx.Response.raise_for_status()`` produces a generic message
    like ``"Client error '400 Bad Request' for url ..."`` and throws away the
    JSON body the server sent — so callers never see *why* a request failed
    (e.g. ``"manifest_id is required"``). This subclass extracts the server's
    ``detail`` / ``message`` and ``request_id`` and folds them into the error
    text, while remaining a :class:`httpx.HTTPStatusError` so existing
    ``except httpx.HTTPStatusError`` handlers keep working unchanged.
    """

    def __init__(self, response: "httpx.Response") -> None:
        self.status_code: int = response.status_code
        self.server_detail: Optional[str] = None
        self.server_code: Optional[str] = None
        self.request_id: Optional[str] = None

        # Pull a human message out of the JSON body, tolerating non-JSON
        # bodies (HTML error pages, plain text, empty) without masking the
        # original failure.
        try:
            body = response.json()
        except Exception:
            body = None

        if isinstance(body, dict):
            detail = body.get("detail")
            # FastAPI 422 detail is a list of validation errors — keep it readable.
            if isinstance(detail, (list, dict)):
                detail = json.dumps(detail)
            self.server_detail = (
                detail
                or body.get("message")
                or body.get("error")
            )
            # Server error envelope is {detail, code, request_id}: `code` is a
            # machine-readable error code (e.g. "validation_error"), NOT a
            # request id. Keep them in separate fields — folding `code` into
            # `request_id` (the old bug) mislabels the error code as a trace.
            self.server_code = body.get("code")
            self.request_id = body.get("request_id")

        # request_id is also exposed as a response header on this backend.
        if not self.request_id:
            self.request_id = response.headers.get("x-request-id")

        reason = (response.reason_phrase or "").strip()
        parts = [f"HTTP {self.status_code}"]
        if reason:
            parts[0] += f" {reason}"
        if self.server_detail:
            parts.append(str(self.server_detail))
        else:
            # Fall back to the response text so something is always surfaced.
            text = (response.text or "").strip()
            if text:
                parts.append(text[:500])
        message = ": ".join(parts)
        suffix = []
        if self.server_code:
            suffix.append(f"code={self.server_code}")
        if self.request_id:
            suffix.append(f"request_id={self.request_id}")
        if suffix:
            message += f" ({', '.join(suffix)})"

        super().__init__(message, request=response.request, response=response)


class AgentNotFoundError(DecimalAPIError):
    """A 404 from an agent-scoped route: this workspace has no such agent.

    Its own class because it is the ONE failure of ``load_agent()`` the caller
    can fix without leaving their editor — a typo, or a name that belongs to a
    different workspace than the key. Everything else (network, key, 5xx) is
    environmental, and a caller who wants to special-case the fixable one
    should not have to string-match a message. Still a
    :class:`DecimalAPIError`, so ``except httpx.HTTPStatusError`` keeps working.

    NOT raised for "the agent exists but has no prompt". That is a real state
    and comes back as ``system_prompt=None``; conflating the two is exactly the
    confusion the null contract on the read route exists to prevent.
    """

    def __init__(self, response: "httpx.Response", *, agent_name: str = "") -> None:
        super().__init__(response)
        self.agent_name = agent_name

        # An unmatched ROUTE and a missing AGENT are both 404, and they have
        # opposite fixes. FastAPI's unmatched-route body is exactly
        # `{"detail": "Not Found"}`, while every agent 404 on this router names
        # the agent — so the two are distinguishable, and guessing wrong sends
        # someone to rename an agent that was never the problem.
        detail = (self.server_detail or "").strip()
        if detail in ("", "Not Found"):
            hint = (
                "The server matched no route. This backend may predate agent "
                "prompts (it needs GET /api/v1/agents/{name}/prompt) — check "
                "the base URL, then the backend version. If both are current, "
                f"check that {agent_name!r} is the name in the dashboard."
            )
        else:
            hint = (
                f"Check the spelling of {agent_name!r} in the dashboard, and "
                "that your API key belongs to the same workspace. "
                "`decimalai init <name>` lists this workspace's agents."
            )
        self.args = (f"{self.args[0]}\n{hint}",)


def _raise_for_status(resp: "httpx.Response") -> None:
    """Like ``resp.raise_for_status()`` but surfaces the server's message.

    On a 4xx/5xx, re-raises a :class:`DecimalAPIError` (a subclass of
    ``httpx.HTTPStatusError``) carrying the server's ``detail``/``message`` and
    ``request_id``. Success responses pass through untouched.
    """
    if resp.is_success:
        return
    raise DecimalAPIError(resp)


class DecimalAIClient:
    """Client for the Decimal platform API.

    Handles authentication, trace ingestion, and manifest registration.
    Can be used standalone or created automatically via ``decimalai.init()``.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.decimal.ai",
        project: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.project = project

        # DEPRECATED (0.10.0): `project` no longer emits an `X-Decimal-Project`
        # header — the platform never read it. See DecimalConfig.api_headers.
        #
        # Built by the shared builder so trace ingest identifies the SDK. Before
        # this change the dict set no User-Agent, so httpx supplied its own default
        # and every ingested trace arrived stamped `python-httpx/<x.y.z>` — the
        # transport's version, not the SDK's. That is why 285,660 production
        # traces could not be attributed to an SDK version.
        headers = sdk_headers(api_key)

        self._http = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )
        self._trace_buffer: List[RunTrace] = []
        # Monotonic deadline before which `buffer_trace` will not auto-flush.
        # Set only when a flush fails in a way a later flush could still fix;
        # an explicit flush()/close()/atexit ignores it. See
        # `_preserve_buffer_after_failed_flush`.
        self._flush_cooldown_until: float = 0.0
        # Traces destroyed by the buffer cap since the last successful flush.
        # Counted rather than logged per trace, and reported once per flush.
        self._dropped_while_buffer_full: int = 0

    # ── Auth ────────────────────────────────────────────────────

    def verify_auth(self) -> VerifyAuthResponse:
        """Verify the API key and return project configuration."""
        resp = self._http.get("/api/v1/auth/verify")
        _raise_for_status(resp)
        return cast(VerifyAuthResponse, resp.json())

    # ── Retry logic ────────────────────────────────────────────

    def _request_with_retry(
        self, method: str, url: str, *, idempotent: bool = False, **kwargs: Any
    ) -> httpx.Response:
        """Make an HTTP request, retrying the failures a retry can actually fix.

        Retries up to ``_MAX_RETRIES`` times on HTTP 429 and on the transient
        server-side statuses in ``_RETRYABLE_STATUSES`` (502/504 from a proxy,
        503 from a load balancer with no healthy instance). Uses the
        ``Retry-After`` header if present, otherwise exponential backoff
        (1s, 2s, 4s) — one ladder, shared by both.

        ``idempotent=True`` adds 500 to that set. It is opt-in per call site
        because a 500 means the application ran and failed part-way, so blindly
        replaying a POST that writes can double-write. Trace ingest opts in:
        every trace carries a client-generated ``id``, so a replay either stores
        a trace that was never stored or is rejected as a duplicate (409
        ``trace_id_conflict`` on ``POST /traces``, a per-trace ``errors`` entry
        on ``POST /traces/batch``) — never a second copy.

        Until this change the loop returned or raised on EVERY non-429 status,
        so a 503 got zero retries and ``flush()`` then cleared the buffer: a
        single blip destroyed up to a full batch of traces. The DecimalAI
        fleet's own HTTP client had been retrying 5xx for months, which is why
        the internal numbers never showed the loss the SDK was taking.
        """
        last_exc: Optional[httpx.HTTPStatusError] = None

        for attempt in range(_MAX_RETRIES + 1):  # 0, 1, 2, 3
            resp = self._http.request(method, url, **kwargs)

            if resp.status_code == 429:
                # A plan quota is TERMINAL — it does not clear until the billing period rolls
                # over, so the retry loop below can only burn wall-clock: each 429 costs 3
                # attempts before the payload is dropped. Fail fast with an error that names
                # the exhausted dimension.
                quota_dimension = resp.headers.get("X-Quota-Exceeded")
                if quota_dimension:
                    body = {}
                    try:
                        parsed = resp.json()
                        detail = parsed.get("detail") if isinstance(parsed, dict) else None
                        body = detail if isinstance(detail, dict) else {}
                    except Exception:  # noqa: BLE001 — a malformed body must not mask the quota
                        body = {}
                    raise DecimalQuotaExceededError(
                        dimension=quota_dimension,
                        resets_in_seconds=int(body.get("resets_in_seconds") or 0),
                        plan=str(body.get("plan") or resp.headers.get("X-RateLimit-Plan") or ""),
                    )

                # 429 — parse Retry-After (delta-seconds OR HTTP-date) and maybe retry
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                delay = max(retry_after, _DEFAULT_RETRY_DELAY * (2 ** attempt))

                last_exc = httpx.HTTPStatusError(
                    "429 Too Many Requests",
                    request=resp.request,
                    response=resp,
                )

                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "Rate limited (429). Retrying in %.1fs (attempt %d/%d)",
                        delay, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(delay)
                    continue

                raise DecimalRateLimitError(
                    retry_after=retry_after,
                    message=(
                        f"Rate limit exceeded after {_MAX_RETRIES} retries. "
                        f"Server says retry after {retry_after}s."
                    ),
                )

            retryable = resp.status_code in _RETRYABLE_STATUSES or (
                idempotent and resp.status_code == 500
            )
            if retryable and attempt < _MAX_RETRIES:
                # A load-balancer 503 rarely carries Retry-After, but honour it
                # when it does — same precedence and same ladder as the 429 path
                # above, so there is only ever one backoff scheme to reason about.
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                delay = max(retry_after, _DEFAULT_RETRY_DELAY * (2 ** attempt))
                logger.warning(
                    "Server error (HTTP %d) on %s %s. Retrying in %.1fs (attempt %d/%d)",
                    resp.status_code, method, url, delay, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(delay)
                continue

            # Success, a 4xx, or a retryable 5xx whose attempts are spent. In the
            # failure cases _raise_for_status raises a DecimalAPIError carrying
            # the real status code — which is what lets flush() below tell a
            # transient 5xx apart from a payload the server will never accept.
            _raise_for_status(resp)
            return resp

        # Should never reach here, but satisfy type checker
        raise last_exc  # type: ignore[misc]

    # ── Trace ingestion ────────────────────────────────────────
    #
    # All trace ingestion methods consult `_should_send_traces()` first.
    # When the SDK is in `manifest_only` mode (DECIMALAI_MODE=manifest_only),
    # they no-op. This is the bouncer that makes the mode actually mean
    # something — framework integrations (LangChain/OpenAI Agents/etc.) can
    # fire callbacks freely without polluting the production trace store
    # with CI/test data.
    #
    # In short: manifest_only mode registers manifests but sends no traces.

    def _should_send_traces(self, method_name: str) -> bool:
        """Return False (and log once) when the SDK is in manifest_only mode."""
        from ._config import _is_manifest_only

        if _is_manifest_only():
            logger.debug(
                "Skipping %s — SDK is in manifest_only mode (DECIMALAI_MODE=manifest_only). "
                "Traces are deliberately not sent during CI manifest extraction.",
                method_name,
            )
            return False
        return True

    def ingest_trace(self, trace: RunTrace) -> IngestionResult:
        """Send a single trace to the platform.

        No-op when the SDK is in manifest_only mode (returns
        ``{"status": "skipped", ...}`` — see :class:`IngestionSkipped`).
        """
        if not self._should_send_traces("ingest_trace"):
            skipped: IngestionSkipped = {"status": "skipped", "reason": "manifest_only_mode"}
            return skipped
        payload = _scrub_surrogates(trace.model_dump(mode="json"))
        _warn_if_strict_manifest_missing(bool(payload.get("manifest_id")))
        resp = self._request_with_retry(
            "POST", "/api/v1/traces", json=payload, idempotent=True
        )
        logger.debug("Ingested trace %s", trace.id)
        return cast(IngestionResult, resp.json())

    def ingest_traces_batch(self, traces: List[RunTrace]) -> IngestionResult:
        """Send a batch of traces to the platform.

        No-op when the SDK is in manifest_only mode.
        """
        if not self._should_send_traces("ingest_traces_batch"):
            skipped: IngestionSkipped = {
                "status": "skipped",
                "reason": "manifest_only_mode",
                "skipped_count": len(traces),
            }
            return skipped
        payload = [_scrub_surrogates(t.model_dump(mode="json")) for t in traces]
        _warn_if_strict_manifest_missing(
            all(p.get("manifest_id") for p in payload) if payload else True
        )
        resp = self._request_with_retry(
            "POST", "/api/v1/traces/batch", json=payload, idempotent=True
        )
        logger.debug("Ingested %d traces", len(traces))
        return cast(IngestionResult, resp.json())

    def ingest_raw_trace(self, payload: Dict[str, Any]) -> IngestionResult:
        """Send a raw trace dict directly to the platform.

        No-op when the SDK is in manifest_only mode.

        No Pydantic validation — the backend validates the payload.
        Useful for custom pipelines, batch imports, and non-Python sources.
        """
        if not self._should_send_traces("ingest_raw_trace"):
            skipped: IngestionSkipped = {"status": "skipped", "reason": "manifest_only_mode"}
            return skipped
        # Raw payloads come from custom pipelines / non-Python sources, exactly
        # where lone UTF-16 surrogates are most likely — scrub them or httpx's
        # JSON encoder raises UnicodeEncodeError before the request is built.
        payload = _scrub_surrogates(payload)
        _warn_if_strict_manifest_missing(bool(payload.get("manifest_id")))
        resp = self._request_with_retry(
            "POST", "/api/v1/traces", json=payload, idempotent=True
        )
        logger.debug("Ingested raw trace")
        return cast(IngestionResult, resp.json())

    def ingest_raw_traces_batch(
        self, payloads: List[Dict[str, Any]]
    ) -> IngestionResult:
        """Send a batch of raw trace dicts to the platform.

        No-op when the SDK is in manifest_only mode.
        """
        if not self._should_send_traces("ingest_raw_traces_batch"):
            skipped: IngestionSkipped = {
                "status": "skipped",
                "reason": "manifest_only_mode",
                "skipped_count": len(payloads),
            }
            return skipped
        payloads = [_scrub_surrogates(p) for p in payloads]
        _warn_if_strict_manifest_missing(
            all(p.get("manifest_id") for p in payloads) if payloads else True
        )
        resp = self._request_with_retry(
            "POST", "/api/v1/traces/batch", json=payloads, idempotent=True
        )
        logger.debug("Ingested %d raw traces", len(payloads))
        return cast(IngestionResult, resp.json())

    def buffer_trace(self, trace: RunTrace) -> None:
        """Buffer a trace for batched sending.

        No-op when the SDK is in manifest_only mode. Buffered traces would
        eventually be flushed, but in manifest_only mode we never want to
        send — short-circuiting here also avoids unbounded buffer growth
        from framework integrations that fire repeatedly during CI runs.

        The auto-flush is also suppressed for ``_FLUSH_RETRY_COOLDOWN`` after a
        flush failed retryably: a preserved buffer sits AT the threshold, so
        without the cooldown every subsequent trace would drag the caller's
        thread through another full retry ladder. An explicit ``flush()`` (or
        ``close()``, or the atexit hook) still tries immediately.

        That cooldown is the ONLY window in which the buffer can exceed
        ``_AUTO_FLUSH_THRESHOLD``, so the cap is enforced here too — a busy agent
        would otherwise grow it without limit for the length of an outage, and
        an unbounded buffer is its own bug.
        """
        if not self._should_send_traces("buffer_trace"):
            return
        self._trace_buffer.append(trace)
        if len(self._trace_buffer) < _AUTO_FLUSH_THRESHOLD:
            return
        if time.monotonic() >= self._flush_cooldown_until:
            self.flush()
        if len(self._trace_buffer) > _AUTO_FLUSH_THRESHOLD:
            # Only reachable when the cooldown above suppressed the flush — a
            # flush that ran and preserved has already trimmed to the cap. The
            # buffer is a full batch the backend just refused, so drop the
            # OLDEST to hold the bound and COUNT it rather than logging once per
            # trace: a per-trace warning through an outage is its own kind of
            # damage. The running total is reported on the next flush, whichever
            # way that one goes.
            del self._trace_buffer[0]
            self._dropped_while_buffer_full += 1

    def _preserve_buffer_after_failed_flush(self, reason: str) -> None:
        """Keep the buffered traces for a later flush — bounded, and logged.

        For the failures a later flush could still fix. The bound is the other
        half of that promise: ``buffer_trace`` auto-flushes at
        ``_AUTO_FLUSH_THRESHOLD``, so a preserved buffer would otherwise grow
        without limit while a backend stayed down. Keep the
        ``_AUTO_FLUSH_THRESHOLD`` most RECENT traces — during an outage the ones
        someone is about to go looking for sit next to the symptom, not at the
        start of the incident — and say plainly how many were lost, because a
        dropped trace never comes back.
        """
        overflow = max(0, len(self._trace_buffer) - _AUTO_FLUSH_THRESHOLD)
        if overflow:
            del self._trace_buffer[:overflow]
            self._dropped_while_buffer_full += overflow
        self._flush_cooldown_until = time.monotonic() + _FLUSH_RETRY_COOLDOWN
        if self._dropped_while_buffer_full:
            logger.warning(
                "%s — preserving the %d most recent trace(s) for the next flush. "
                "The buffer is at its %d-trace cap, so %d trace(s) have been "
                "DROPPED since the last successful flush. They are gone.",
                reason, len(self._trace_buffer), _AUTO_FLUSH_THRESHOLD,
                self._dropped_while_buffer_full,
            )
        else:
            logger.warning(
                "%s — preserving %d buffered trace(s) for the next flush",
                reason, len(self._trace_buffer),
            )

    def _drop_buffer_after_failed_flush(self, exc: BaseException) -> None:
        """Clear the buffer for a failure no retry can fix — and make it visible.

        This is a permanent trace loss, so it is also recorded on the background
        sender: ``export_status().last_error`` / ``last_send_error()`` is where a
        production health check looks, and the batch path used to drop traces
        without ever touching either — the same silence that made the original
        bug take weeks to notice. Mirrors ``otel._submit_or_send_inline``, which
        already records failures from a foreground send.
        """
        self._trace_buffer.clear()
        self._flush_cooldown_until = 0.0
        self._dropped_while_buffer_full = 0
        try:
            from . import _config

            _config._sender._record_failure(exc)
        except Exception:  # noqa: BLE001 — observability must never break flush
            logger.debug("Could not record the flush failure on the sender", exc_info=True)

    def flush(self) -> None:
        """Flush all buffered traces to the platform.

        The buffer is **preserved** for every failure a later flush could still
        fix: a rate limit (429, once ``_request_with_retry`` has exhausted its
        backoff), a server-side 5xx, and a transport error (DNS, connect, read
        timeout). None of those are a verdict on the payload. Clearing on them
        is what made a single 503 destroy traces permanently — and because
        ``buffer_trace`` auto-flushes at ``_AUTO_FLUSH_THRESHOLD``, the batch in
        hand when a blip arrived was usually a full one.

        The buffer is cleared when the failure is terminal: a 4xx (the server
        has judged these bytes and will judge them the same way forever), a plan
        quota (it does not clear until the billing period rolls over), and a
        local serialization failure. Holding a permanently-rejected batch would
        wedge every later trace behind it — an unbounded buffer is its own bug.

        Never raises: a failed flush must not take down the caller's agent.
        """
        if not self._trace_buffer:
            return
        try:
            self.ingest_traces_batch(self._trace_buffer)
            self._trace_buffer.clear()
            self._flush_cooldown_until = 0.0
            if self._dropped_while_buffer_full:
                # Close the loop on the outage: the recovery log is the last
                # chance to say how much never made it, and it is the line
                # someone reads when the trace count looks short.
                logger.warning(
                    "Flush recovered, but %d trace(s) were dropped while the "
                    "buffer sat at its %d-trace cap during the failure.",
                    self._dropped_while_buffer_full, _AUTO_FLUSH_THRESHOLD,
                )
                self._dropped_while_buffer_full = 0
        except DecimalRateLimitError:
            self._preserve_buffer_after_failed_flush("Rate limited")
        except DecimalAPIError as exc:
            # The one branch this whole change exists for. `status_code` is the
            # server's, not a guess: 5xx means "try again", 4xx means "these
            # bytes are the problem".
            if exc.status_code >= 500:
                self._preserve_buffer_after_failed_flush(
                    f"Server error (HTTP {exc.status_code}) after "
                    f"{_MAX_RETRIES} retries — {exc}"
                )
            else:
                logger.exception(
                    "Failed to flush %d traces — the server rejected them with "
                    "HTTP %d, which a retry cannot fix. Dropping the batch.",
                    len(self._trace_buffer), exc.status_code,
                )
                self._drop_buffer_after_failed_flush(exc)
        except httpx.RequestError as exc:
            # Never reached the server at all (DNS, refused connection, read
            # timeout). Same reasoning as a 5xx: the payload is not implicated.
            self._preserve_buffer_after_failed_flush(
                f"Transport error ({type(exc).__name__}: {exc})"
            )
        except Exception as exc:
            # Everything else — a quota (terminal until the billing period rolls
            # over), a serialization failure, a bug. Dropping is the honest
            # outcome; the log and export_status() carry the cause.
            logger.exception("Failed to flush %d traces", len(self._trace_buffer))
            self._drop_buffer_after_failed_flush(exc)

    # ── Trace queries ──────────────────────────────────────────

    def list_traces(
        self,
        limit: int = 20,
        offset: int = 0,
        status: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> TraceListResponse:
        """List traces for the current project."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if agent_name:
            params["agent_name"] = agent_name
        resp = self._http.get("/api/v1/traces", params=params)
        _raise_for_status(resp)
        return cast(TraceListResponse, resp.json())

    def get_trace(self, trace_id: str | UUID) -> TraceDetailResponse:
        """Get a single trace with its full span tree."""
        resp = self._http.get(f"/api/v1/traces/{trace_id}")
        _raise_for_status(resp)
        return cast(TraceDetailResponse, resp.json())

    # ── Agents ──────────────────────────────────────────────────

    def list_agents(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> AgentListResponse:
        """List all agents in the workspace."""
        resp = self._http.get(
            "/api/v1/agents", params={"limit": limit, "offset": offset}
        )
        _raise_for_status(resp)
        return cast(AgentListResponse, resp.json())

    def get_agent_prompt(
        self,
        agent_name: str,
        *,
        version: Optional[int] = None,
        if_none_match: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """The agent's configured system prompt. Backs :func:`decimalai.load_agent`.

        Deliberately its OWN endpoint rather than a scan of ``list_agents()``:
        passing a ``limit`` there activates a truncating pagination path whose
        ordering puts manifest-only agents LAST, so the never-traced,
        UI-created agent this feature exists to serve is the first one dropped —
        and reported as "no such agent".

        Args:
            agent_name: URL-quoted here, so a name needing escaping still
                addresses the right agent instead of a mangled path.
            version: Read one historical version instead of the effective one.
            if_none_match: A previously seen ``content_hash``. The route
                answers 304 when it still matches; this method then returns
                ``None`` — "unchanged", distinct from a payload whose
                ``system_prompt`` is ``None`` ("no prompt set"). Only useful to
                a caller polling for edits; ``load_agent()`` never sends it,
                because a cache is what would break the no-redeploy property it
                is sold on.

        Raises:
            AgentNotFoundError: 404 — no such agent in this workspace (or a
                backend with no prompt route at all; the message says which).
            DecimalAPIError: any other 4xx/5xx, including the 409 an
                unresolvable pin produces.
        """
        quoted = urllib.parse.quote(str(agent_name), safe="")
        params: Dict[str, Any] = {}
        if version is not None:
            params["version"] = version
        headers = {"If-None-Match": if_none_match} if if_none_match else None

        resp = self._http.get(
            f"/api/v1/agents/{quoted}/prompt", params=params, headers=headers,
        )
        # 304 is not `is_success`, so it has to be answered before
        # `_raise_for_status` turns a correct conditional response into an error.
        if resp.status_code == 304:
            return None
        if resp.status_code == 404:
            raise AgentNotFoundError(resp, agent_name=str(agent_name))
        _raise_for_status(resp)
        return cast(Dict[str, Any], resp.json())

    # ── Manifest registration ─────────────────────────────────

    def register_manifest(self, manifest: Any) -> ManifestRegistrationResponse:
        """Register a manifest snapshot with the platform.

        Args:
            manifest: A ManifestSnapshot (from decimalai.schema.manifest).

        Returns:
            Registration response with manifest_id and compatibility info.
        """
        payload = _scrub_surrogates(manifest.model_dump(mode="json"))
        resp = self._http.post("/api/v1/manifests", json=payload)
        _raise_for_status(resp)
        logger.debug("Registered manifest %s (hash=%s)", manifest.id, manifest.manifest_hash)
        return cast(ManifestRegistrationResponse, resp.json())

    def list_manifests(
        self,
        limit: int = 20,
        offset: int = 0,
        agent_name: Optional[str] = None,
    ) -> ManifestListResponse:
        """List manifests from the platform."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if agent_name:
            params["agent_name"] = agent_name
        resp = self._http.get("/api/v1/manifests", params=params)
        _raise_for_status(resp)
        return cast(ManifestListResponse, resp.json())

    def get_manifest(
        self, manifest_id: str,
    ) -> Dict[str, Any]:
        """Get a single manifest with full component details."""
        resp = self._http.get(f"/api/v1/manifests/{manifest_id}")
        _raise_for_status(resp)
        return resp.json()

    def diff_manifest(
        self, manifest_id: str,
    ) -> ManifestDiffResponse:
        """Get a diff between this manifest and its predecessor (parent).

        Returns:
            An envelope ``{"diff": <ManifestDiff | None>}``. The structural diff
            lives under the ``"diff"`` key (with ``changed_surfaces`` and
            ``summary``); ``diff`` is ``None`` with a ``"message"`` when there is
            no parent to compare against, and ``None`` with
            ``"verdict": "self_comparison"`` on a self-diff. Read
            ``result["diff"]["changed_surfaces"]``, not ``result[...]`` directly.
        """
        resp = self._http.get(f"/api/v1/manifests/{manifest_id}/diff")
        _raise_for_status(resp)
        return cast(ManifestDiffResponse, resp.json())

    # ── Regression Check ─────────────────────────────────────────

    def run_regression_check(
        self,
        agent_name: str,
        candidate_manifest_id: str,
        pr_context: Optional[Dict[str, Any]] = None,
        trace_window_days: int = 30,
        dry_run: bool = False,
        source: Optional[str] = None,
    ) -> RegressionCheckResponse:
        """Run a manifest impact analysis for a candidate manifest.

        The backend computes the structural diff vs the agent's baseline
        manifest and returns a severity-classified impact report (which
        historical traces will break, may behave differently, or are
        unaffected).

        Args:
            agent_name: Agent name (matches the value passed to decimalai.init()).
            candidate_manifest_id: Manifest ID returned by flush_manifest_for_ci().
            pr_context: Optional PR metadata dict (repo, pr_number, branch,
                commit_sha). Persisted with the regression check for traceability.
            trace_window_days: How far back to look for affected traces.

        Returns:
            Impact report dict with verdict, severity counts, and per-surface
            impact entries. See ``RegressionCheckResponse`` in
            ``decimalai._responses`` for the full schema.
        """
        payload: Dict[str, Any] = {
            "agent_name": agent_name,
            "candidate_manifest_id": candidate_manifest_id,
            "trace_window_days": trace_window_days,
        }
        if pr_context:
            payload["pr_context"] = pr_context
        # 048: explicit source attribution. The CLI passes source="cli";
        # direct SDK callers can override, otherwise the backend defaults
        # to "api" (or "github_action" when pr_context is set).
        if source is not None:
            payload["source"] = source

        params: Dict[str, Any] = {}
        if dry_run:
            params["dry_run"] = "true"

        resp = self._http.post("/api/v1/regression-check", json=payload, params=params)
        _raise_for_status(resp)
        return cast(RegressionCheckResponse, resp.json())

    def get_regression_check(self, regression_check_id: str) -> RegressionCheckResponse:
        """Fetch a previously-run regression check by ID."""
        resp = self._http.get(f"/api/v1/regression-check/{regression_check_id}")
        _raise_for_status(resp)
        return cast(RegressionCheckResponse, resp.json())

    def list_regression_checks(
        self,
        agent_name: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> RegressionCheckListResponse:
        """List regression checks for an agent, most recent first."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if agent_name:
            params["agent_name"] = agent_name
        resp = self._http.get("/api/v1/regression-check", params=params)
        _raise_for_status(resp)
        return cast(RegressionCheckListResponse, resp.json())

    # ── Eval Scores ──────────────────────────────────────────────

    def push_eval_scores(
        self,
        trace_id: str | UUID,
        source: str,
        scores: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Push external evaluation scores to a trace.

        Args:
            trace_id: The trace to attach scores to.
            source: Origin of the scores (e.g., "deepeval", "langsmith", "custom").
            scores: List of score dicts, each with at least "name" and "score".
                    Optional fields: "passed", "reason", "category".
            metadata: Optional source-specific metadata (e.g., a ``source_label``
                display name, run_id, metric version). Forwarded to the backend
                ``metadata`` field, which reads ``source_label`` to override the
                source's display name.

        Returns:
            Ingestion response with stored score count.

        Example::

            client.push_eval_scores(
                trace_id="abc123",
                source="deepeval",
                scores=[
                    {"name": "correctness", "score": 0.92, "reason": "Accurate"},
                    {"name": "faithfulness", "score": 0.85},
                ],
            )
        """
        payload: Dict[str, Any] = {"source": source, "scores": scores}
        if metadata:
            payload["metadata"] = metadata
        resp = self._request_with_retry(
            "POST", f"/api/v1/traces/{trace_id}/eval-scores", json=payload,
        )
        logger.debug("Pushed %d eval scores to trace %s", len(scores), str(trace_id)[:8])
        return resp.json()

    def register_evals(
        self,
        evals: List[Any],
        agent_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register SDK-defined @eval functions with the platform.

        Tells the backend "these named evaluators exist in user code" so the
        UI can surface them in the evaluator list before any scores arrive.
        Idempotent — safe to call repeatedly. Fail-open: returns an empty
        result on transport errors instead of raising.

        Args:
            evals: List of DecimalEval (or compatible) instances. Each must
                expose ``to_registration_dict(agent_name=...)``.
            agent_name: Optional agent scope. If None, the registration is
                org-wide (visible across all agents).

        Returns:
            The backend's registration response, or {"registered": [], "total": 0}
            on failure.
        """
        items = []
        for ev in evals:
            try:
                items.append(ev.to_registration_dict(agent_name=agent_name))
            except AttributeError:
                logger.debug(
                    "register_evals: skipping %r — no to_registration_dict()",
                    ev,
                )

        if not items:
            return {"registered": [], "total": 0}

        try:
            resp = self._request_with_retry(
                "POST",
                "/api/v1/evaluators/register",
                json={"evaluators": items},
            )
            logger.debug(
                "Registered %d SDK evaluator(s) with platform", len(items),
            )
            return resp.json()
        except Exception as e:
            logger.warning(
                "register_evals failed (%s: %s) — evals will still emit "
                "scores but may not appear in UI until first trace lands",
                type(e).__name__,
                e,
            )
            return {"registered": [], "total": 0}

    # ── Evaluator config (deterministic + LLM judge templates) ──

    def list_evaluators(
        self,
        agent_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List configured evaluators (server-managed) for an agent or workspace.

        Args:
            agent_name: If provided, list only evaluators attached to this agent.

        Returns:
            ``{"evaluators": [...]}`` — each entry has id, name, eval_type,
            category, enabled, etc.
        """
        params: Dict[str, Any] = {}
        if agent_name:
            params["agent_name"] = agent_name
        resp = self._http.get("/api/v1/evaluators", params=params)
        _raise_for_status(resp)
        return resp.json()

    def list_evaluator_templates(self) -> Dict[str, Any]:
        """List available evaluator templates (pre-built deterministic / LLM judges).

        Returns:
            ``{"templates": [...], "categories": {...}}``.
        """
        resp = self._http.get("/api/v1/evaluators/templates")
        _raise_for_status(resp)
        return resp.json()

    def add_evaluator(
        self,
        *,
        agent_name: Optional[str] = None,
        template_id: Optional[str] = None,
        name: Optional[str] = None,
        eval_type: Optional[str] = None,
        category: Optional[str] = None,
        prompt_template: Optional[str] = None,
        threshold: Optional[float] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Configure a new evaluator for an agent (or workspace-wide).

        Two modes:

        - **Template-based** — pass ``template_id`` (e.g. ``"skill_output_check"``
          or ``"helpfulness_judge"``) to instantiate a pre-built evaluator.
        - **Custom** — pass ``name``, ``eval_type`` (``"deterministic"`` |
          ``"llm_judge"``), and for LLM judges, a ``prompt_template`` containing
          ``{input}`` / ``{output}`` placeholders.

        Args:
            agent_name: Agent to attach this evaluator to. Omit for workspace-wide.
            template_id: Pre-built template ID — overrides custom fields.
            name: Evaluator name (required if no template_id).
            eval_type: ``"deterministic"`` or ``"llm_judge"``.
            category: ``"quality"``, ``"safety"``, ``"rag"``, ``"agentic"``, ``"custom"``.
            prompt_template: Rubric prompt for LLM judges.
            threshold: Pass threshold (default 0.5).
            display_name: Human-readable name shown in the dashboard.
            description: One-line explanation of what this evaluator checks.
        """
        payload: Dict[str, Any] = {}
        if agent_name:
            payload["agent_name"] = agent_name
        if template_id:
            payload["template_id"] = template_id
        if name:
            payload["name"] = name
        if eval_type:
            payload["eval_type"] = eval_type
        if category:
            payload["category"] = category
        if prompt_template:
            payload["prompt_template"] = prompt_template
        if threshold is not None:
            payload["threshold"] = threshold
        if display_name:
            payload["display_name"] = display_name
        if description:
            payload["description"] = description

        resp = self._request_with_retry("POST", "/api/v1/evaluators", json=payload)
        return resp.json()

    def remove_evaluator(self, evaluator_id: str) -> Dict[str, Any]:
        """Remove an evaluator by id."""
        resp = self._http.delete(f"/api/v1/evaluators/{evaluator_id}")
        _raise_for_status(resp)
        return resp.json()

    def get_eval_scores(self, trace_id: str | UUID) -> Dict[str, Any]:
        """Get all evaluation scores (quality + compatibility) for a trace.

        Returns:
            Dict with quality_scores, compatibility_scores, and aggregates.
        """
        resp = self._http.get(f"/api/v1/traces/{trace_id}/eval-scores")
        _raise_for_status(resp)
        return resp.json()

    def get_eval_breakdown(self, trace_id: str | UUID) -> Dict[str, Any]:
        """Get the full eval breakdown with provenance for a trace.

        Returns scores grouped by source (Manifest Diff, DeepEval, LangSmith,
        Custom, etc.) with icons, labels, badge colors, and decision reasons
        explaining how the final verdict was computed.

        Returns:
            Dict with eval_verdict, quality_avg, compat_avg, source_groups,
            and decision_reasons.

        Example::

            breakdown = client.get_eval_breakdown("trace-123")
            print(breakdown["eval_verdict"])  # "keep" / "drop" / ...
            for group in breakdown["source_groups"]:
                print(f"{group['source_label']}: {group['source_avg']}")
                for score in group["scores"]:
                    print(f"  {score['name']}: {score['score']}")
        """
        resp = self._http.get(f"/api/v1/traces/{trace_id}/eval-breakdown")
        _raise_for_status(resp)
        return resp.json()

    def get_decision(self, trace_id: str | UUID) -> Dict[str, Any]:
        """Compute and get the unified verdict for a trace.

        Returns:
            Dict with verdict (keep/repair/replay/drop), quality_avg,
            compat_avg, and per-score breakdowns.
        """
        resp = self._request_with_retry(
            "POST", f"/api/v1/traces/{trace_id}/decision",
        )
        return resp.json()

    def batch_decision(
        self,
        trace_ids: Optional[List[str]] = None,
        manifest_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Batch compute unified verdicts for multiple traces.

        Args:
            trace_ids: Specific trace IDs to score.
            manifest_id: Score all traces from this manifest (alternative to trace_ids).

        Returns:
            Dict with decisions list, total count, and verdict_counts breakdown.
        """
        payload: Dict[str, Any] = {}
        if trace_ids:
            payload["trace_ids"] = trace_ids
        if manifest_id:
            payload["manifest_id"] = manifest_id

        resp = self._request_with_retry(
            "POST", "/api/v1/traces/batch-decision", json=payload,
        )
        return resp.json()

    def get_eval_stats(
        self,
        agent_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get aggregate evaluation statistics.

        Returns:
            Dict with total_evaluated, pass_rate, verdict_breakdown, avg_quality.
        """
        params: Dict[str, Any] = {}
        if agent_name:
            params["agent_name"] = agent_name
        resp = self._http.get("/api/v1/traces/eval/stats", params=params)
        _raise_for_status(resp)
        return resp.json()

    def annotate_trace(
        self,
        trace_id: str | UUID,
        notes: Optional[str] = None,
        *,
        label: Optional[str] = None,
        rating: Optional[int] = None,
        correctness: Optional[str] = None,
        error_categories: Optional[List[str]] = None,
        corrected_output: Optional[str] = None,
        tags: Optional[List[str]] = None,
        score: Optional[float] = None,
        flagged_for_review: Optional[bool] = None,
        add_to_dataset: Optional[bool] = None,
        text: Optional[str] = None,
        annotation_type: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add a human annotation to a trace.

        Mirrors the backend ``CreateAnnotationRequest``. Annotations are
        trace-scoped. Provide at least one of: ``label``
        (``thumbs_up``/``thumbs_down``/``needs_correction``), ``rating`` (1-5),
        ``correctness`` (``correct``/``partial``/``incorrect``),
        ``error_categories``, ``corrected_output``, ``tags``, ``notes``, or
        ``score`` (0..1). ``flagged_for_review`` and ``add_to_dataset`` are
        side-flags that don't count as content on their own.

        Args:
            trace_id: The trace to annotate.
            notes: Free-text note (the old ``text`` argument maps here).
            label, rating, correctness, error_categories, corrected_output,
            tags, score, flagged_for_review, add_to_dataset: see above.

        Returns:
            The created annotation (id, timestamp, and the fields you set).
        """
        # Back-compat: the old signature was annotate_trace(trace_id, text, ...).
        if text is not None and notes is None:
            notes = text
        # annotation_type/span_id were never accepted by the backend (the
        # endpoint 422'd on them via extra="forbid"); accept-and-ignore for one
        # release so old call sites don't TypeError, but warn.
        if annotation_type is not None or span_id is not None:
            import warnings

            warnings.warn(
                "annotate_trace(): 'annotation_type' and 'span_id' are no longer "
                "supported (annotations are trace-scoped) and are ignored; they "
                "will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
        candidates: Dict[str, Any] = {
            "label": label,
            "rating": rating,
            "correctness": correctness,
            "error_categories": error_categories,
            "corrected_output": corrected_output,
            "tags": tags,
            "notes": notes,
            "score": score,
            "flagged_for_review": flagged_for_review,
            "add_to_dataset": add_to_dataset,
        }
        payload: Dict[str, Any] = {k: v for k, v in candidates.items() if v is not None}
        # The backend requires at least one substantive field (the two flags
        # don't count); fail locally with a clear error instead of a 422.
        _substantive = {
            "label", "rating", "correctness", "error_categories",
            "corrected_output", "tags", "notes", "score",
        }
        if not (_substantive & payload.keys()):
            raise ValueError(
                "annotate_trace() needs at least one of: label, rating, "
                "correctness, error_categories, corrected_output, tags, notes, score."
            )
        resp = self._request_with_retry(
            "POST", f"/api/v1/traces/{trace_id}/annotations", json=payload,
        )
        logger.debug("Annotated trace %s", str(trace_id)[:8])
        return resp.json()

    # ── Datasets ──────────────────────────────────────────────

    def list_datasets(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List all datasets in the workspace."""
        resp = self._http.get(
            "/api/v1/datasets", params={"limit": limit, "offset": offset}
        )
        _raise_for_status(resp)
        return resp.json()

    def get_dataset(
        self, dataset_id: str,
    ) -> Dict[str, Any]:
        """Get a dataset with its version history."""
        resp = self._http.get(f"/api/v1/datasets/{dataset_id}")
        _raise_for_status(resp)
        return resp.json()

    def resolve_version_id(
        self,
        dataset_id: str,
        version: Optional[str] = None,
    ) -> str:
        """Resolve a version specifier to a concrete version ID.

        Supports:
            - ``None`` or ``"latest"`` → resolves to the current (latest) version
            - ``"v3"`` or ``"3"`` → resolves by version number
            - A full version UUID → returned as-is

        Args:
            dataset_id: The dataset to look up.
            version: Version specifier (None, "latest", "v3", "3", or UUID).

        Returns:
            The resolved version ID string.

        Raises:
            ValueError: If the version specifier cannot be resolved.
        """
        # Full UUID — return as-is. Parse strictly so human-friendly labels
        # that merely look UUID-ish (e.g. "release-1", "prod-2026") fall
        # through to the "vN"/number resolution path instead of being sent to
        # the backend verbatim (which would 404 opaquely).
        if version:
            try:
                UUID(version)
                return version
            except (ValueError, AttributeError):
                pass

        ds = self.get_dataset(dataset_id)

        # Latest / default
        if not version or version.lower() == "latest":
            current = ds.get("current_version_id")
            if current:
                return current
            # Fallback: pick the highest version_number from versions list
            versions = ds.get("versions", [])
            if not versions:
                raise ValueError(
                    f"Dataset {dataset_id} has no versions. "
                    f"Build one first with client.build_dataset()."
                )
            latest = max(versions, key=lambda v: v.get("version_number", 0))
            return latest["id"]

        # Version number: "v3" or "3"
        version_num_str = version.lstrip("vV")
        try:
            version_num = int(version_num_str)
        except ValueError:
            raise ValueError(
                f"Unrecognized version specifier: '{version}'. "
                f"Use 'latest', 'v3', '3', or a full version UUID."
            )

        versions = ds.get("versions", [])
        for v in versions:
            if v.get("version_number") == version_num:
                return v["id"]

        available = sorted(v.get("version_number", 0) for v in versions)
        raise ValueError(
            f"Version v{version_num} not found for dataset {dataset_id}. "
            f"Available versions: {', '.join(f'v{n}' for n in available)}"
        )

    def build_dataset(
        self,
        dataset_id: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a new dataset version from traces.

        Args:
            dataset_id: The dataset to build.
            filters: Optional trace filters (eval_verdict, agent_name, etc.).

        Returns:
            Build result with version_id and row_count.
        """
        payload: Dict[str, Any] = {}
        if filters:
            payload["filters"] = filters
        resp = self._request_with_retry(
            "POST", f"/api/v1/datasets/{dataset_id}/build", json=payload,
        )
        logger.debug("Built dataset %s", dataset_id[:8])
        return resp.json()

    def export_dataset(
        self,
        dataset_id: str,
        version_id: Optional[str] = None,
        format: str = "jsonl",
    ) -> Any:
        """Export a dataset version in a training-ready format.

        Args:
            dataset_id: The dataset to export.
            version_id: The version to export. Accepts ``None``/``"latest"``
                for the most recent version, ``"v3"`` for version 3, or a
                full version UUID. Defaults to the latest version.
            format: Export format: ``"jsonl"`` (default) or ``"parquet"``.

        Returns:
            For JSONL: the response content as a string (newline-delimited JSON).
            For Parquet: raw bytes of the Parquet file.
            For other formats: parsed JSON response.
        """
        resolved = self.resolve_version_id(dataset_id, version_id)
        resp = self._http.get(
            f"/api/v1/datasets/{dataset_id}/versions/{resolved}/export",
            params={"format": format},
        )
        _raise_for_status(resp)

        content_type = resp.headers.get("content-type", "")
        if "octet-stream" in content_type:
            return resp.content  # raw bytes (Parquet)
        if "jsonl" in content_type or "ndjson" in content_type:
            return resp.text  # JSONL string
        return resp.text  # default to text for JSONL

    def pull_dataset(
        self,
        dataset_id: str,
        path: str,
        *,
        version: Optional[str] = None,
        format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Download a dataset version to a local file.

        This is the high-level convenience method for getting training data
        onto disk. It resolves the version, downloads in the requested format,
        and writes to the specified path.

        Args:
            dataset_id: The dataset to download.
            path: Local file path to write to (e.g., ``"./data.jsonl"``).
                  Parent directories are created automatically.
            version: Version specifier: ``None``/``"latest"`` for the most
                     recent version, ``"v3"`` for version 3, or a full UUID.
            format: Export format: ``"jsonl"`` (default) or ``"parquet"``.
                    If not specified, inferred from the file extension.

        Returns:
            Summary dict with ``row_count``, ``file_path``, ``bytes_written``,
            ``format``, ``version_id``, and ``dataset_id``.

        Example::

            client = DecimalAIClient(api_key="dai_sk_...")

            # Pull the latest version as JSONL
            result = client.pull_dataset("ds_abc123", "./training_data.jsonl")

            # Pull a specific version as Parquet
            result = client.pull_dataset(
                "ds_abc123", "./data.parquet", version="v2"
            )
        """
        # Infer format from file extension if not specified
        if format is None:
            if path.endswith(".parquet") or path.endswith(".pq"):
                format = "parquet"
            else:
                format = "jsonl"

        # Resolve version
        resolved_version = self.resolve_version_id(dataset_id, version)

        # Download
        data = self.export_dataset(dataset_id, resolved_version, format=format)

        # Write to disk
        if format == "parquet":
            from .export.parquet import write_parquet
            result = write_parquet(data, path)
        else:
            from .export.jsonl import write_jsonl
            result = write_jsonl(data, path)

        result["version_id"] = resolved_version
        result["dataset_id"] = dataset_id
        return result

    # ── Replay ────────────────────────────────────────────────

    def create_replay_batch(
        self,
        source_manifest_id: str,
        target_manifest_id: str,
        trace_ids: List[str],
    ) -> Dict[str, Any]:
        """Create a replay batch from traces.

        Args:
            source_manifest_id: Manifest the traces were recorded against.
            target_manifest_id: New manifest version to replay against.
            trace_ids: List of trace IDs to include in the batch.

        Returns:
            Batch details including batch_id, status, and task count.
        """
        payload = {
            "source_manifest_id": source_manifest_id,
            "target_manifest_id": target_manifest_id,
            "trace_ids": trace_ids,
        }
        resp = self._request_with_retry("POST", "/api/v1/replay/batches", json=payload)
        logger.debug("Created replay batch with %d traces", len(trace_ids))
        return resp.json()

    def get_replay_batch(self, batch_id: str) -> Dict[str, Any]:
        """Get a replay batch with task details.

        Returns:
            Batch details including status, progress, and task list.
        """
        resp = self._http.get(f"/api/v1/replay/batches/{batch_id}")
        _raise_for_status(resp)
        return resp.json()

    def list_replay_batches(self, limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """List replay batches.

        Returns:
            Dict with batches list and total count.
        """
        resp = self._http.get("/api/v1/replay/batches", params={"limit": limit, "offset": offset})
        _raise_for_status(resp)
        return resp.json()

    def submit_replay_result(
        self,
        task_id: str,
        replayed_trace_id: Optional[str] = None,
        eval_score: Optional[float] = None,
        eval_verdict: Optional[str] = None,
        status: str = "completed",
    ) -> Dict[str, Any]:
        """Submit the result of a replay task.

        Args:
            task_id: The replay task ID.
            replayed_trace_id: ID of the new trace from the replay.
            eval_score: Optional quality score (0.0 to 1.0).
            eval_verdict: Optional verdict (pass/fail).
            status: Task status (completed/failed/skipped).

        Returns:
            Updated task details.
        """
        payload: Dict[str, Any] = {"status": status}
        if replayed_trace_id:
            payload["replayed_trace_id"] = replayed_trace_id
        if eval_score is not None:
            payload["eval_score"] = eval_score
        if eval_verdict is not None:
            payload["eval_verdict"] = eval_verdict

        resp = self._request_with_retry(
            "POST", f"/api/v1/replay/tasks/{task_id}/submit", json=payload,
        )
        logger.debug("Submitted replay result for task %s", task_id[:8])
        return resp.json()

    def get_replay_prompts(
        self,
        agent_name: str,
        verdict: Optional[str] = None,
        limit: int = 500,
    ) -> Dict[str, Any]:
        """Get stale prompts that need to be replayed.

        Downloads prompts from traces classified as needing replay
        based on the compatibility report. Users should re-run these
        prompts through their agent and submit the results back.

        Args:
            agent_name: Agent name to get replay prompts for.
            verdict: Filter by verdict (replay, drop, repair).
                     Defaults to replay + drop.
            limit: Maximum number of prompts (default 500, max 5000).

        Returns:
            Dict with agent_name, total count, and prompts list.
            Each prompt has: trace_id, user_input, original_output,
            verdict, agent_name, manifest_id, created_at.
        """
        params: Dict[str, Any] = {
            "agent_name": agent_name,
            "format": "json",
            "limit": limit,
        }
        if verdict:
            params["verdict"] = verdict

        resp = self._http.get("/api/v1/replay/export", params=params)
        _raise_for_status(resp)
        return resp.json()

    def get_replay_task(self, task_id: str) -> Dict[str, Any]:
        """Get a single replay task with its input for execution.

        Args:
            task_id: The replay task ID.

        Returns:
            Task details including task_input and replayability.
        """
        resp = self._http.get(f"/api/v1/replay/tasks/{task_id}")
        _raise_for_status(resp)
        return resp.json()

    def link_replay(
        self,
        original_trace_id: str,
        replayed_trace_id: str,
    ) -> Dict[str, Any]:
        """Link a replayed trace to its original and auto-score.

        Creates a replay task connecting the two traces without requiring
        an explicit batch. The backend auto-scores the comparison.

        Args:
            original_trace_id: ID of the original trace.
            replayed_trace_id: ID of the replayed trace.

        Returns:
            Dict with task_id, eval_score, eval_verdict, batch_id.
        """
        payload = {
            "original_trace_id": original_trace_id,
            "replayed_trace_id": replayed_trace_id,
        }
        resp = self._request_with_retry("POST", "/api/v1/replay/link", json=payload)
        logger.debug(
            "Linked replay: %s → %s",
            original_trace_id[:8], replayed_trace_id[:8],
        )
        return resp.json()

    def get_replay_sessions(
        self,
        agent_name: str,
        verdict: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Get session-grouped replay prompts for multi-turn replay.

        Groups replay-eligible traces by session_id and includes all
        turns in each session (including context-only turns) in
        chronological order.

        Args:
            agent_name: Agent name to get replay sessions for.
            verdict: Filter trigger turns by verdict.
            limit: Maximum number of sessions (default 50, max 500).

        Returns:
            Dict with agent_name, total, and sessions list.
            Each session has: session_id, turns[], verdict, agent_name.
        """
        params: Dict[str, Any] = {
            "agent_name": agent_name,
            "format": "json",
            "limit": limit,
        }
        if verdict:
            params["verdict"] = verdict

        resp = self._http.get("/api/v1/replay/export/sessions", params=params)
        _raise_for_status(resp)
        return resp.json()


    # ── Compatibility ──────────────────────────────────────────

    def compat_check(
        self,
        agent_name: str,
        recompute: bool = False,
    ) -> Dict[str, Any]:
        """Check training data compatibility for an agent's latest manifest transition.

        Compares the two most recent manifests and returns how many traces
        are affected (keep/repair/replay/drop). If a cached report exists
        it is returned; pass ``recompute=True`` to force fresh analysis.

        Args:
            agent_name: Agent to check compatibility for.
            recompute: Force a fresh analysis (ignore cached report).

        Returns:
            Dict with status, version info, verdict counts, and component impact.
            If the agent has fewer than 2 manifests, returns status="no_transition".

        Example::

            result = client.compat_check("my-agent")
            if result["status"] == "ok":
                print(f"Keep: {result['keep']}, Repair: {result['repair']}")
                print(f"Replay: {result['replay']}, Drop: {result['drop']}")
            else:
                print(result["message"])
        """
        params: Dict[str, Any] = {}
        if recompute:
            params["recompute"] = True
        resp = self._http.get(
            f"/api/v1/agents/{agent_name}/compat-summary", params=params
        )
        _raise_for_status(resp)
        return resp.json()

    def impact_report(
        self,
        agent_name: str,
        manifest_id: Optional[str] = None,
        baseline_manifest_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Aggregate Impact Report for an agent's manifest transition.

        One call returns everything the dashboard's Impact Report shows:
        ``surface_changes`` (the manifest diff with per-change severity and
        sample trace IDs), ``affected_trace_count`` (the canonical "N traces
        affected" number), ``compat_summary`` (keep/repair/replay/drop),
        ``severity`` + ``severity_reason``, and ``human_summary``.

        Defaults to the agent's latest transition (newest manifest vs its
        parent); pass ``manifest_id`` and/or ``baseline_manifest_id`` to pin
        either side of the comparison.

        Args:
            agent_name: Agent to report on.
            manifest_id: Candidate manifest (default: newest).
            baseline_manifest_id: Baseline manifest (default: candidate's parent).

        Returns:
            Dict with status ("ok" / "no_transition"), surface_changes,
            affected_trace_count, compat_summary, severity, human_summary.

        Example::

            report = client.impact_report("my-agent")
            if report["status"] == "ok":
                print(report["human_summary"])
                print(f"{report['affected_trace_count']} traces affected")
        """
        params: Dict[str, Any] = {}
        if manifest_id:
            params["manifest_id"] = manifest_id
        if baseline_manifest_id:
            params["baseline_manifest_id"] = baseline_manifest_id
        resp = self._http.get(
            f"/api/v1/agents/{agent_name}/impact-report", params=params
        )
        _raise_for_status(resp)
        return resp.json()

    # ── Repair ────────────────────────────────────────────────────
    # Closes the detect→impact→REPAIR→export loop without the dashboard.

    def repair_preview(
        self, old_manifest_id: str, new_manifest_id: str, sample_size: int = 5
    ) -> Dict[str, Any]:
        """Preview mechanical repair rules for an old→new manifest transition.

        Returns ``{"rules": [...], "previews": [...], "total_eligible": int}``,
        or ``{"rules": [], "previews": [], "message"/"error": ...}`` when there
        is nothing to repair / a manifest is not found. Each rule in ``rules`` is
        positionally indexed; pass those 0-based indices to
        ``repair_apply(..., approved_rule_indices=...)`` to apply a subset.
        ``sample_size`` must be 1..50 (server-enforced).
        """
        payload: Dict[str, Any] = {
            "old_manifest_id": old_manifest_id,
            "new_manifest_id": new_manifest_id,
            "sample_size": sample_size,
        }
        resp = self._http.post("/api/v1/repair/preview", json=payload)
        _raise_for_status(resp)
        return resp.json()

    def repair_apply(
        self,
        old_manifest_id: str,
        new_manifest_id: str,
        approved_rule_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Apply repairs for an old→new manifest transition.

        With ``approved_rule_indices=None`` applies ALL eligible rules
        (``POST /repair/apply`` → ``{batch_id, status, total_episodes,
        repaired_count, failed_count}``). With a non-empty list applies only
        those rules (``POST /repair/apply-selective`` → ``{batch_id, status,
        total_episodes, repaired_count, rules_applied}``). The indices are
        0-based positions into the ``rules`` array returned by
        :meth:`repair_preview`.
        """
        if approved_rule_indices:
            payload: Dict[str, Any] = {
                "old_manifest_id": old_manifest_id,
                "new_manifest_id": new_manifest_id,
                "approved_rule_indices": approved_rule_indices,
            }
            resp = self._http.post("/api/v1/repair/apply-selective", json=payload)
        else:
            payload = {
                "old_manifest_id": old_manifest_id,
                "new_manifest_id": new_manifest_id,
            }
            resp = self._http.post("/api/v1/repair/apply", json=payload)
        _raise_for_status(resp)
        return resp.json()

    def get_repair_batch(self, batch_id: str) -> Dict[str, Any]:
        """Fetch a repair batch's status + per-trace results (GET /repair/{id})."""
        resp = self._http.get(f"/api/v1/repair/{batch_id}")
        _raise_for_status(resp)
        return resp.json()

    def get_dataset_examples(
        self, dataset_id: str, version_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get the materialized examples of a dataset version.

        Args:
            dataset_id: The dataset to read.
            version_id: The version to read. Accepts ``None``/``"latest"`` for
                the most recent version, ``"v3"`` for version 3, or a full
                version UUID. Defaults to the latest version.

        Returns:
            ``{"dataset_id", "version_id", "count", "examples": [...]}`` where
            each example is the per-row JSON from the export (typically
            ``{"messages": [...], ...}``).

        Notes:
            There is no dedicated ``/examples`` endpoint on the backend —
            example rows are served by the version ``export`` route
            (``GET /api/v1/datasets/{id}/versions/{vid}/export``) as JSONL.
            This method resolves the version, pulls that JSONL, and parses it
            into a list so callers get structured rows without a 404.
        """
        resolved = self.resolve_version_id(dataset_id, version_id)
        resp = self._http.get(
            f"/api/v1/datasets/{dataset_id}/versions/{resolved}/export",
            params={"format": "jsonl"},
        )
        _raise_for_status(resp)
        examples = [
            json.loads(line) for line in resp.text.splitlines() if line.strip()
        ]
        return {
            "dataset_id": dataset_id,
            "version_id": resolved,
            "count": len(examples),
            "examples": examples,
        }

    # ── Lifecycle ──────────────────────────────────────────────

    def discard_undelivered(self, when: str = "Closing") -> None:
        """Report and drop traces that ``flush()`` kept when nothing will retry.

        ``flush()`` deliberately hangs on to a batch a *later* flush could still
        deliver. ``close()`` and interpreter shutdown ARE the last attempt, so
        leaving the caller with "preserving N trace(s) for the next flush" would
        be a promise nobody is left to keep. Say the traces were lost, plainly,
        at the one moment it becomes true.
        """
        if not self._trace_buffer:
            return
        logger.warning(
            "%s with %d buffered trace(s) that never reached the platform — "
            "the last flush failed and there is no later attempt.%s Call "
            "decimalai.export_status() for the cause.",
            when,
            len(self._trace_buffer),
            (
                f" A further {self._dropped_while_buffer_full} trace(s) were "
                "dropped earlier at the buffer cap."
                if self._dropped_while_buffer_full
                else ""
            ),
        )
        self._trace_buffer.clear()
        self._dropped_while_buffer_full = 0

    def close(self) -> None:
        """Flush remaining traces and close the HTTP client."""
        self.flush()
        self.discard_undelivered("Closing the client")
        self._http.close()

    def __enter__(self) -> "DecimalAIClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
