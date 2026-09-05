"""Global configuration for the DecimalAI SDK.

Manages a singleton config + client that all integrations share.
Populated by ``decimalai.init()``.
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional

logger = logging.getLogger("decimalai")

# Try to load .env if python-dotenv is available.
#
# `find_dotenv(usecwd=True)` is load-bearing. Bare `load_dotenv()` resolves from the CALLING
# FRAME's file — this module, inside site-packages — and walks UP from there. With a project-local
# venv (`myapp/.venv/`) that walk escapes site-packages and happens to land on `myapp/.env`, so it
# works on a laptop. With the interpreter anywhere else — a container's
# /usr/local/lib/python3.x/site-packages with the app at /app, a Lambda layer, a shared venv — the
# walk never reaches the project and returns '', so the .env is silently ignored and init() dies
# with "No API key provided". Same code, same .env, opposite outcome, decided by venv placement.
# `usecwd=True` walks up from the working directory instead, i.e. from next to the user's agent.py.
try:
    from dotenv import find_dotenv, load_dotenv

    _dotenv_path = find_dotenv(usecwd=True)
    if _dotenv_path:
        load_dotenv(_dotenv_path, override=False)
    else:
        load_dotenv(override=False)
except Exception:
    # ImportError when python-dotenv is absent; anything else means a malformed .env,
    # which must not stop the SDK from importing.
    pass


def _tristate_env(name: str) -> Optional[bool]:
    """Read a boolean env var that distinguishes "unset" from "explicitly false".

    Returns None when the variable is absent or empty, so a caller can tell
    "the user did not say" from "the user said no".
    """
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip().lower()
    if not raw:
        return None
    return raw in ("1", "true", "yes", "on")


# ── SDK identity ───────────────────────────────────────────────
#
# Every request the SDK makes must say which SDK version made it.
#
# Why this exists: before this change the ONLY request that identified itself
# was the one-off ``init()`` auth-verify probe. Trace ingest — the path that
# matters — went out over an httpx.Client whose default User-Agent is
# ``python-httpx/<x.y.z>``, so 285,660 production traces recorded the
# TRANSPORT's version and nothing about the SDK. The consequence was
# concrete: the synthetic-user fleet ran 0.8.0 against production for six
# weeks while everyone believed it was current, and the only way to notice
# was to grep Cloud Run logs for that single startup probe.
#
# Header choice — ``User-Agent``, and ONLY ``User-Agent``:
#
#   * Cloud Run already extracts it into the indexed ``httpRequest.userAgent``
#     field. That is the exact field the 0.8.0 discovery was made in, so this
#     change pays off with zero backend work:
#         gcloud logging read '... httpRequest.userAgent:"decimalai-sdk"'
#   * A custom ``X-Decimal-SDK-Version`` would parse more cleanly, but it is
#     invisible to that field — Cloud Run does not log arbitrary request
#     headers — so it would buy nothing until the backend is changed to read
#     it. This module already carries the scar of shipping a header the
#     platform silently ignored (``X-Decimal-Project``, see ``api_headers``);
#     a second unread header repeats that mistake and creates the same false
#     impression that something is being captured.
#   * Sending BOTH was considered and rejected. The only argument for it is
#     to avoid SDK adoption lag — but both headers would ship in the SAME
#     release, so the population of clients sending one is identical to the
#     population sending the other. The custom header's sole benefit over
#     ``decimalai-sdk/(\S+)`` against the User-Agent is parse convenience,
#     which does not justify a second wire format. When the backend actually
#     persists this (see RELEASING/handoff notes), THAT is the moment to add
#     a structured header, coupled to a reader.
#
# Privacy: the comment section carries the Python version and ``sys.platform``
# ("linux" / "darwin" / "win32") and nothing else. Those two answer the
# immediate follow-up question ("is this a container or someone's laptop, and
# on which interpreter?") at zero cost. Deliberately absent: hostname,
# username, filesystem paths, machine architecture, OS build — none of which
# triage anything and all of which identify a user or a machine.

# Static across the life of the process; computed once.
_PY_VERSION = (
    f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
)
_PLATFORM = sys.platform


def sdk_user_agent(context: Optional[str] = None) -> str:
    """Build the SDK's ``User-Agent`` string.

    Shape follows the conventional ``product/version (comment)`` form::

        decimalai-sdk/<version> (python/<x.y.z>; <sys.platform>)
        decimalai-sdk/<version> (python/<x.y.z>; <sys.platform>; init-verify)

    ``context`` appends a label to the comment so a caller can distinguish
    its traffic without restating the product token. ``init()``'s auth-verify
    probe passes ``"init-verify"``, which preserves the previous ability to tell
    a one-off startup probe apart from steady-state ingest — the substring
    ``init-verify`` is still present, so log filters written against the old
    ``decimalai-sdk/0.8.0 (init-verify)`` format keep matching.

    The version is read from ``decimalai.__version__`` at call time rather
    than copied into a constant here, so it can never drift from the package
    (``TestVersion.test_version_matches_pyproject`` pins that to pyproject in
    turn, making one source of truth for the whole chain).
    """
    # Lazy: this module is imported *by* ``decimalai/__init__.py``, so a
    # module-level ``from . import __version__`` would be a circular import.
    from . import __version__

    comment = f"python/{_PY_VERSION}; {_PLATFORM}"
    if context:
        comment = f"{comment}; {context}"
    return f"decimalai-sdk/{__version__} ({comment})"


def sdk_headers(api_key: str, context: Optional[str] = None) -> dict[str, str]:
    """The headers EVERY DecimalAI request sends.

    This is the single shared builder. ``DecimalConfig.api_headers``,
    ``DecimalAIClient``'s httpx client, ``SkillRouter._headers`` and the
    ``init()`` verify probe all route through here so that adding a header
    once adds it everywhere — the property that was missing when the version
    lived only on the verify probe.
    """
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": sdk_user_agent(context),
    }


@dataclass
class DecimalConfig:
    """Configuration for the DecimalAI SDK."""

    api_key: str = ""
    base_url: str = "https://api.decimal.ai"
    project: Optional[str] = None
    enabled: bool = True
    # When True, the SDK runs in CI manifest-extraction mode:
    # - Background trace sender does NOT start
    # - Framework integrations capture manifest data but do not send traces
    # - The user's init script is expected to call decimalai.flush_manifest_for_ci()
    #   to upload the captured manifest as a candidate for regression check.
    # Set automatically when DECIMALAI_MODE=manifest_only env var is present.
    manifest_only: bool = False
    # Populated by the init-time verify probe (None if verify=False).
    # When True, the backend requires every ingested trace to carry a
    # manifest_id; the SDK uses this to surface a clearer error if its
    # own auto-manifest registration fails.
    backend_require_manifest_on_ingest: Optional[bool] = None
    # When True, framework adapters inject the routed skill's BODY (the knowledge K) into the
    # prompt at runtime — not just a menu row — so a benchmarked skill's value actually reaches the
    # agent.
    #
    # TRI-STATE, and the None is load-bearing. Unset (None) means "let the adapter decide", which
    # `resolve_inject_body()` below answers from whether that adapter owns a tool loop. An explicit
    # True/False — from init(inject_skill_body=...) or DECIMALAI_INJECT_SKILL_BODY=0/1 — always wins.
    #
    # This was a plain `bool` defaulting False until 2026-08-28, and that default silently broke the
    # whole point of the skill rail on every adapter with no tool loop: langchain and anthropic
    # register no `load_skill` tool, so prompt injection is their ONLY body channel, and False meant
    # the model got a menu of titles it could never read. Verified end to end — a skill body saying
    # "opened boxes carry a 23.5% restocking fee" against a real model produced 15% (confabulated)
    # with the old default and 23.5% with injection on. Do not flatten this back to a bool.
    inject_skill_body: Optional[bool] = field(
        default_factory=lambda: _tristate_env("DECIMALAI_INJECT_SKILL_BODY")
    )
    # Progressive disclosure: register the native load_skill tool on
    # adapters that own their tool loop (openai_agents, pydantic_ai) whenever
    # the skill loader is enabled — so a surfaced description is always
    # executable. On by default; kill switch DECIMALAI_LOAD_SKILL_TOOL=0 or
    # init(load_skill_tool=False).
    load_skill_tool: bool = field(
        default_factory=lambda: os.environ.get("DECIMALAI_LOAD_SKILL_TOOL", "1").strip().lower()
        not in ("0", "false", "no", "off")
    )
    # Which channel is authoritative for delivering skills to the model:
    #   "harness" — a native skill-loading runtime (Claude Code/Cursor) loads
    #               skills from disk; the SDK keeps mirroring them there
    #               (disk_sync on). Today's behavior.
    #   "router"  — the SDK router is the single injection channel; the SDK stops
    #               writing skills to disk (disk_sync off) so it never duplicates.
    #   "auto"    — router when an adapter's skill loader is active, else harness.
    # Only affects an adapter once its router loader is enabled — a no-op otherwise.
    # Set via init(skill_authority=...) or DECIMALAI_SKILL_AUTHORITY.
    skill_authority: str = field(
        default_factory=lambda: (
            os.environ.get("DECIMALAI_SKILL_AUTHORITY", "").strip().lower() or "auto"
        )
    )
    # `_max_batch_size = 50` and `_flush_interval_seconds = 5.0` used to sit
    # here. Both were DELETED on 2026-08-29 rather than wired up, because
    # neither described anything this SDK does, and a config field that reads
    # like a knob but turns nothing is worse than no field: someone reasoning
    # about trace latency would have concluded there was a 5-second periodic
    # flush, and there never was one.
    #
    # There is no periodic flush because there is nothing for it to drain.
    # `BackgroundSender.submit` hands each trace straight to a
    # ThreadPoolExecutor, which starts the send immediately — the send path
    # holds no queue that time could bound. The ONE buffer in the SDK is
    # `DecimalAIClient._trace_buffer`, and it is bounded by COUNT
    # (`_AUTO_FLUSH_THRESHOLD` in _client.py) plus the explicit
    # `flush()`/`close()`/atexit drains, not by a clock. A daemon timer would
    # be a thread guarding a queue that does not exist.
    #
    # `_max_batch_size` was dead the same way: the real batch size is
    # `_AUTO_FLUSH_THRESHOLD`, which this value never fed and never agreed with.
    # If a time-based flush is ever wanted, it belongs on that buffer in
    # _client.py, next to the threshold it would sit beside.

    @property
    def api_headers(self) -> dict[str, str]:
        # DEPRECATED (0.10.0): `project` no longer emits an `X-Decimal-Project`
        # header. The platform never read it — trace scoping is resolved from
        # the API key alone, and a trace's project_id is set only for a
        # project-scoped key — so every value sent here was discarded on
        # arrival. Sending a header the server ignores made `project=` look
        # like it grouped traces when it did nothing at all.
        # Grouping that actually works: workspaces (resolved from the key, or
        # the X-Workspace-Id header the dashboard sends).
        return sdk_headers(self.api_key)

    def resolve_disk_sync(self, loader_active: bool) -> bool:
        """Whether an adapter should mirror platform skills to disk (disk_sync).

        Router authority => False (the router is the sole injector; writing to
        disk only creates duplicates for a native skill-loading runtime).
        Harness authority => True (today's behavior). "auto" resolves to router
        only when the adapter's router skill loader is active.
        """
        authority = (self.skill_authority or "auto").lower()
        if authority == "router":
            return False
        if authority == "harness":
            return True
        return not loader_active  # auto

    def resolve_inject_body(self, *, has_tool_loop: bool) -> bool:
        """Whether this adapter should inject the routed skill's BODY into the prompt.

        An explicit setting always wins — init(inject_skill_body=...) or
        DECIMALAI_INJECT_SKILL_BODY=0/1. Unset, the answer comes from the adapter:

          no tool loop  (langchain, anthropic)      => True
              Prompt injection is the ONLY body channel these have. Without it the model
              is handed a menu of skill titles it has no mechanism to read, which is not a
              degraded rail — it is a rail that cannot work.

          has tool loop (openai_agents, pydantic_ai) => False
              These register a real `load_skill` tool, so the model fetches on demand.
              Injecting as well would double-deliver and double the token cost.

        Only reached once an adapter's skill loader is enabled (`enable_skill_loader=True`),
        which is opt-in and off by default — so nobody who never asked for skills is affected.
        """
        if self.inject_skill_body is not None:
            return bool(self.inject_skill_body)
        return not has_tool_loop


class DecimalConfigError(Exception):
    """Raised when the SDK is misconfigured."""

    pass


# ── Global singleton ───────────────────────────────────────────

_config: Optional[DecimalConfig] = None
_client: Optional["DecimalAIClient"] = None  # type: ignore[name-defined]


def _get_config() -> DecimalConfig:
    """Return the global config, raising if init() hasn't been called."""
    if _config is None:
        raise DecimalConfigError(
            "DecimalAI SDK not initialized. Call decimalai.init(api_key=...) first."
        )
    return _config


def _get_client() -> "DecimalAIClient":  # type: ignore[name-defined]
    """Return the global HTTP client, raising if init() hasn't been called."""
    if _client is None:
        raise DecimalConfigError(
            "DecimalAI SDK not initialized. Call decimalai.init(api_key=...) first."
        )
    return _client


def _is_enabled() -> bool:
    """Check if tracing is enabled (False = no-op mode)."""
    return _config is not None and _config.enabled


def _is_manifest_only() -> bool:
    """True if the SDK is in CI manifest-extraction mode.

    Framework integrations should call this to decide whether to skip
    trace emission while still allowing manifest capture. Set via the
    DECIMALAI_MODE=manifest_only env var (handled in init()).
    """
    return _config is not None and _config.manifest_only


# ── Export observability ──────────────────────────────────────

@dataclass(frozen=True)
class ExportStatus:
    """Snapshot of the background trace exporter's state.

    Returned by ``decimalai.export_status()``. Use this in production
    health checks, CI assertions, or any monitoring code that needs
    to answer "are my traces actually arriving at the backend?"

    Example::

        st = decimalai.export_status()
        if st.consecutive_failures >= 3:
            alert_oncall(f"DecimalAI traces failing: {st.last_error!r}")
        if st.sent == 0 and st.failed > 0:
            raise RuntimeError("All traces are failing — check API key.")

    ``last_manifest_error`` is reported separately from ``last_error``
    so callers can distinguish "manifest registration failed (then the
    trace was rejected for missing manifest_id)" from "the trace POST
    itself failed for an unrelated reason like 401 or timeout".
    """

    sent: int
    failed: int
    queue_depth: int
    consecutive_failures: int
    last_error: Optional[str]
    last_error_at: Optional[datetime]
    last_success_at: Optional[datetime]
    last_manifest_error: Optional[str] = None
    last_manifest_error_at: Optional[datetime] = None


# Type alias for the on_export_error callback. Receives (exception,
# trace_id_or_None). Called from the background sender thread; user
# code must be thread-safe.
ExportErrorCallback = Callable[[BaseException, Optional[str]], None]


# ── Background Sender ─────────────────────────────────────────

class BackgroundSender:
    """Non-blocking trace sender using a single daemon thread.

    Traces are submitted to a ThreadPoolExecutor so the calling thread
    (the user's agent) is never blocked on HTTP I/O.

    Tracks send-side observability (sent/failed/streaks/timestamps) so
    callers can introspect via ``decimalai.export_status()`` or react
    via ``decimalai.on_export_error(cb)`` without polling logs. Pre-2026
    the only signal was a per-failure WARNING log + a single
    ``_last_send_error`` field — enough to debug but not enough to
    monitor in production.
    """

    def __init__(self) -> None:
        self._executor: Optional[ThreadPoolExecutor] = None
        self._pending: list[Any] = []  # Futures for flush()
        # submit() runs on arbitrary caller
        # threads; without a lock the read-modify-write on self._pending (append
        # + reassign-on-prune) can lose futures, so flush() may exit before some
        # work is awaited (silent trace drop on a fast exit). Guard every
        # _pending mutation/snapshot with this dedicated lock — held only around
        # list ops, never across future.result().
        self._pending_lock = threading.Lock()

        # Observability — read/written from the sender thread on
        # completion AND from the caller thread via export_status().
        # All numeric fields go behind _state_lock; the lock is taken
        # only at completion + read time, so it's not on the hot path.
        self._state_lock = threading.Lock()
        self._sent_count: int = 0
        self._failed_count: int = 0
        self._consecutive_failures: int = 0
        self._last_send_error: Optional[BaseException] = None
        self._last_send_error_at: Optional[datetime] = None
        self._last_success_at: Optional[datetime] = None

        # Manifest-registration errors are tracked separately from
        # trace-send errors so callers can tell the two failure modes
        # apart (the trace POST then rejects with a confusingly
        # different error — "manifest_id required" — when manifest
        # registration silently fell back to a synthetic UUID). Set by
        # ``record_manifest_error`` (called from generic.py /
        # langchain.py / openai_agents.py after retries are exhausted).
        self._last_manifest_error: Optional[BaseException] = None
        self._last_manifest_error_at: Optional[datetime] = None

        # User callbacks for failures. Called from the sender thread.
        self._error_callbacks: List[ExportErrorCallback] = []

        # Has the consecutive-failure escalation banner already fired
        # in this streak? Reset on success. Prevents log spam when the
        # backend goes down and 50 traces all fail with the same error.
        self._escalation_fired: bool = False

    # ---- executor lifecycle ----

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="decimal-sender"
            )
        return self._executor

    # ---- public observability ----

    def register_error_callback(self, cb: ExportErrorCallback) -> None:
        """Register a callback fired on each background-send failure.

        Called from the sender thread, so user code must be thread-safe.
        Use this to route export failures into Sentry/Datadog/PagerDuty
        or to raise a custom exception in a background monitor thread.
        Multiple callbacks are supported and called in registration order.
        """
        self._error_callbacks.append(cb)

    def status(self, queue_depth_hint: Optional[int] = None) -> ExportStatus:
        """Snapshot the current send-side state.

        ``queue_depth_hint`` is the count of in-flight futures, computed
        outside the lock to avoid contending with the sender thread.
        Pass None to have status() count under the lock.
        """
        if queue_depth_hint is not None:
            queue_depth = queue_depth_hint
        else:
            with self._pending_lock:
                queue_depth = len([f for f in self._pending if not f.done()])
        with self._state_lock:
            return ExportStatus(
                sent=self._sent_count,
                failed=self._failed_count,
                queue_depth=queue_depth,
                consecutive_failures=self._consecutive_failures,
                last_error=(
                    f"{type(self._last_send_error).__name__}: "
                    f"{str(self._last_send_error)[:200]}"
                    if self._last_send_error is not None
                    else None
                ),
                last_error_at=self._last_send_error_at,
                last_success_at=self._last_success_at,
                last_manifest_error=(
                    f"{type(self._last_manifest_error).__name__}: "
                    f"{str(self._last_manifest_error)[:200]}"
                    if self._last_manifest_error is not None
                    else None
                ),
                last_manifest_error_at=self._last_manifest_error_at,
            )

    def record_manifest_error(self, exc: BaseException) -> None:
        """Record a manifest-registration failure (synchronous, caller-thread).

        Called by ``_maybe_register_manifest`` paths after exhausting
        retries. Surfaces in ``ExportStatus.last_manifest_error`` so
        callers can tell "manifest registration failed → trace then
        rejected as missing manifest_id" apart from "trace POST itself
        failed".
        """
        with self._state_lock:
            self._last_manifest_error = exc
            self._last_manifest_error_at = datetime.now(timezone.utc)

    # ---- internal completion handlers ----

    def _record_success(self) -> None:
        with self._state_lock:
            self._sent_count += 1
            self._last_success_at = datetime.now(timezone.utc)
            self._consecutive_failures = 0
            self._escalation_fired = False

    def _record_failure(self, exc: BaseException, trace_id: Optional[str] = None) -> None:
        with self._state_lock:
            self._failed_count += 1
            self._last_send_error = exc
            self._last_send_error_at = datetime.now(timezone.utc)
            self._consecutive_failures += 1
            consec = self._consecutive_failures
            fired = self._escalation_fired
            if consec >= 3 and not fired:
                self._escalation_fired = True
                escalate = True
            else:
                escalate = False

        # Per-failure log (current behavior) — kept WARNING so the line
        # remains greppable. Escalation banner only fires once per streak.
        logger.warning(
            "decimalai: Background send failed — %s: %s. "
            "The trace was NOT ingested. "
            "Call decimalai.export_status() for a summary.",
            type(exc).__name__,
            str(exc)[:200],
        )
        logger.debug("Background send failed", exc_info=(type(exc), exc, exc.__traceback__))

        if escalate:
            logger.error(
                "decimalai: %d consecutive trace exports have failed. "
                "Last error: %s: %s. "
                "Check your API key, base_url, and backend logs. "
                "Use decimalai.on_export_error(cb) for programmatic alerting.",
                consec,
                type(exc).__name__,
                str(exc)[:200],
            )

        # Fire user callbacks. Don't let one bad callback break the chain.
        for cb in list(self._error_callbacks):
            try:
                cb(exc, trace_id)
            except Exception:
                logger.debug("on_export_error callback raised", exc_info=True)

    # ---- submit / flush / shutdown ----

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> None:
        """Submit work to the background thread.

        Wraps the user-supplied callable so success/failure both update
        the observability counters even when the caller never calls
        flush() — e.g., long-running daemons that just submit traces
        and rely on the sender thread to drain them.
        """
        executor = self._ensure_executor()

        def _runner() -> Any:
            try:
                result = fn(*args, **kwargs)
                self._record_success()
                return result
            except Exception as exc:
                # Try to dig out a trace_id for the callback. The
                # caller's args may be a Trace object or dict; best-effort.
                tid = _extract_trace_id(args)
                self._record_failure(exc, tid)
                raise

        future = executor.submit(_runner)
        # Tag so flush() knows _runner has already recorded the outcome
        # (success counter or failure callback). Futures injected by tests
        # bypass _runner and need flush() to record on their behalf.
        try:
            setattr(future, "_dai_recorded_in_runner", True)
        except (AttributeError, TypeError):
            pass
        with self._pending_lock:
            self._pending.append(future)
            # Prune completed futures to avoid unbounded growth.
            self._pending = [f for f in self._pending if not f.done()]

    def flush(self, timeout: float = 5.0) -> None:
        """Wait for all pending work to complete.

        Counters and last_send_error are normally updated inside the
        ``_runner`` wrapper added by ``submit``. For futures that were
        injected into ``_pending`` directly (test fixtures, advanced
        callers wiring their own queues), flush() records the outcome
        here as a safety net.
        """
        # Snapshot-and-clear under the lock, then await OUTSIDE it so submit()
        # and status() never block on the (potentially multi-second) drain.
        with self._pending_lock:
            pending = self._pending
            self._pending = []
        for future in pending:
            try:
                future.result(timeout=timeout)
                # Future succeeded — only record if it wasn't already
                # counted by _runner.
                if not getattr(future, "_dai_recorded_in_runner", False):
                    self._record_success()
            except Exception as exc:
                # Recorded by _runner unless the future was injected
                # directly into _pending (tests do this); record here
                # in that case so observability stays accurate.
                if not getattr(future, "_dai_recorded_in_runner", False):
                    self._record_failure(exc)

    def shutdown(self) -> None:
        """Flush pending work and shut down the executor."""
        self.flush()
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None


def _extract_trace_id(args: tuple) -> Optional[str]:
    """Best-effort: pull a trace_id off the first positional arg.

    The sender is called with (client.ingest_trace, trace) where trace
    is a pydantic model with an `.id` attr. Return that id as a string
    if available, else None. Never raise — this is callback metadata.
    """
    if not args:
        return None
    first = args[0]
    tid = getattr(first, "id", None)
    if tid is None and isinstance(first, dict):
        tid = first.get("id") or first.get("trace_id")
    if tid is None:
        return None
    try:
        return str(tid)
    except Exception:
        return None


# Global sender instance
_sender = BackgroundSender()


# ---- deferred manifest registration ------------------------------------
#
# A trace may only carry a manifest_id the platform actually STORED:
# `require_manifest_on_ingest` is true on the hosted API, so an id it has never
# seen is answered 400 "Trace validation failed: manifest_id '...' does not
# exist" and the trace is LOST. Every adapter registers on the CALLER's thread
# at root-run end, where the budget has to stay in the hundreds of milliseconds
# or the customer's agent waits on us -- far too short to outlast a Cloud Run
# revision with no available instance, which answers 429 for tens of seconds.
# Measured on production: 1,315 of exactly those 400s in the 48 h to
# 2026-09-04T09:33Z, every one from this SDK.
#
# So the long ladder lives HERE, on the background sender, where waiting costs
# the customer's agent nothing. Five attempts, each jittered +/-50% so a fleet
# of processes that lost their instance together does not re-attempt in
# lockstep. `_request_with_retry` spends up to 7 s of its own 429 ladder inside
# each attempt, so the whole thing is bounded at roughly 30-60 s: long enough to
# bridge a restart, short enough that one trace cannot own the sender thread.
# The tuple IS the bound -- there is no wall-clock cap to drift out of test.
_MANIFEST_RETRY_BACKOFFS_S = (0.0, 1.0, 2.0, 4.0, 8.0)


def submit_trace_pending_manifest(
    client: Any, trace: Any, reregister: Callable[[], Optional[str]]
) -> None:
    """Queue one trace whose manifest registration was refused.

    ``reregister`` re-attempts the adapter's own registration and returns the
    id the platform stored, or None. The trace is HELD -- never posted under a
    local id -- until it returns one.
    """
    _sender.submit(_ingest_when_manifest_lands, client, trace, reregister)


def _ingest_when_manifest_lands(
    client: Any, trace: Any, reregister: Callable[[], Optional[str]]
) -> Any:
    import random
    import time as _time

    for delay in _MANIFEST_RETRY_BACKOFFS_S:
        if delay:
            _time.sleep(delay * (0.5 + random.random()))
        manifest_id = reregister()
        if manifest_id:
            trace.manifest_id = manifest_id
            return client.ingest_trace(trace)

    # Still refused. Do NOT post: the platform's answer to a local id is a
    # certainty, and a certain 400 is only noise on top of a loss that has
    # already happened. Raising here is what routes it through the sender's
    # `_record_failure`, so `export_status().last_error` names the real cause
    # next to `last_manifest_error` instead of the platform's confusing
    # "manifest_id does not exist".
    raise RuntimeError(
        "decimalai: manifest registration for agent %r is still refused after "
        "%d attempts; trace %s was NOT sent rather than shipped under an id the "
        "platform never stored. See decimalai.export_status()."
        % (
            getattr(trace, "agent_name", "?"),
            len(_MANIFEST_RETRY_BACKOFFS_S),
            getattr(trace, "id", "?"),
        )
    )


# ---- atexit summary ----------------------------------------------------

def _emit_shutdown_summary() -> None:
    """Print a one-line stderr summary on process exit IF anything failed.

    Silent on the happy path (sent_count > 0 and failed_count == 0).
    Loud at the right moment — users notice startup banners more than
    mid-session warnings, so flushing the diagnostic at shutdown catches
    "wait, where were my traces?" cases without spamming healthy runs.
    """
    try:
        st = _sender.status()
        if st.failed == 0 and st.sent > 0:
            return  # happy path — stay silent
        if st.failed == 0 and st.sent == 0:
            # User never sent anything; could be a no-op script.
            return
        # Anything failed → emit summary.
        sys.stderr.write(
            "decimalai: shutdown summary — "
            f"{st.sent} trace(s) sent, {st.failed} failed"
            + (f" (last error: {st.last_error})" if st.last_error else "")
            + ". Call decimalai.export_status() for details.\n"
        )
    except Exception:
        # Never crash on the way out.
        pass


def _shutdown() -> None:
    """Flush pending traces, emit summary, clean up. Called via atexit."""
    global _client
    try:
        _sender.shutdown()
    except Exception:
        pass
    try:
        _emit_shutdown_summary()
    except Exception:
        pass
    if _client is not None:
        try:
            if hasattr(_client, "close"):
                _client.close()
        except Exception:
            pass


atexit.register(_shutdown)
