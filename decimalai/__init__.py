"""DecimalAI SDK — Agent dataset lifecycle platform.

Quick start::

    import decimalai
    decimalai.init()  # reads DECIMAL_API_KEY from env

    # LangChain — one-liner
    decimalai.init(langchain=True)

    # Or turn on tracing yourself
    from decimalai.langchain import instrument
    instrument()

    # Generic / any framework
    @decimalai.trace(agent_name="my-agent")
    def run_agent(query):
        msgs = [{"role": "user", "content": query}]
        resp = openai.chat.completions.create(model="gpt-4o", messages=msgs)
        decimalai.log_llm_call(
            model="gpt-4o",
            input=msgs,
            output={"content": resp.choices[0].message.content},
        )
        return resp.choices[0].message.content
"""

__version__ = "0.10.0"

import atexit
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .schema.manifest import ManifestSnapshot

logger = logging.getLogger("decimalai")

# Ensures the atexit handler is only registered once even if init() is
# called multiple times (e.g., re-init in a test or after a config change).
_atexit_registered = False


def _atexit_flush() -> None:
    """Flush any buffered traces on interpreter shutdown.

    The DecimalAI client buffers up to 50 traces before auto-flushing.
    Scripts that exit before reaching the threshold would silently drop
    their traces — this handler is the safety net.

    Must never raise: atexit handlers run during shutdown when the
    interpreter may already be in a degraded state.
    """
    try:
        from . import _config as _cfg
        client = getattr(_cfg, "_client", None)
        if client is not None:
            client.flush()
    except Exception:
        # Swallow — atexit shouldn't crash a process that's already exiting.
        pass


def last_send_error() -> Optional[BaseException]:
    """Return the most recent background-send failure, if any.

    Returns the exception captured by ``BackgroundSender`` the last time
    a queued trace failed to reach the backend, or None if no failure
    has been recorded yet.

    Added alongside the change that raised failed-send logging from
    DEBUG to WARNING, so a rejected trace is never silent. The log line
    tells the operator something failed; this helper lets a program
    assert it in tests / CI / production health checks::

        decimalai.flush()
        if (err := decimalai.last_send_error()) is not None:
            raise RuntimeError(f"Trace ingest failed: {err}")

    Prefer ``decimalai.export_status()`` for richer programmatic state
    (counts, streaks, timestamps). This function is kept for backward
    compatibility with code that pre-dates the structured surface.
    """
    try:
        from . import _config as _cfg
        sender = getattr(_cfg, "_sender", None)
        if sender is None:
            return None
        return getattr(sender, "_last_send_error", None)
    except Exception:
        return None


def export_status():
    """Return the current export-side state as an ``ExportStatus`` snapshot.

    Use this for production health checks, CI assertions, or any
    monitoring code that needs to answer "are my traces actually
    arriving at the backend?" without grepping logs.

    Example::

        st = decimalai.export_status()
        if st.consecutive_failures >= 3:
            alert_oncall(f"DecimalAI traces failing: {st.last_error}")

    Returns an ``ExportStatus`` dataclass with ``sent``, ``failed``,
    ``queue_depth``, ``consecutive_failures``, ``last_error``,
    ``last_error_at``, and ``last_success_at`` fields.

    No-op safe: returns a zero-filled ``ExportStatus`` if ``init()``
    hasn't been called.
    """
    from . import _config as _cfg
    from ._config import ExportStatus

    sender = getattr(_cfg, "_sender", None)
    if sender is None:
        return ExportStatus(
            sent=0,
            failed=0,
            queue_depth=0,
            consecutive_failures=0,
            last_error=None,
            last_error_at=None,
            last_success_at=None,
        )
    return sender.status()


def on_export_error(callback) -> None:
    """Register a callback fired on each background trace-export failure.

    The callback is called from the background sender thread, so user
    code must be thread-safe. Signature::

        def cb(exc: BaseException, trace_id: Optional[str]) -> None: ...

    Multiple callbacks are supported (called in registration order).
    Exceptions raised by the callback are swallowed (logged at DEBUG)
    so one bad alerter doesn't break the chain.

    Use this to route export failures to Sentry, Datadog, PagerDuty, or
    any production alerting surface::

        decimalai.on_export_error(lambda exc, tid: sentry_sdk.capture_exception(exc))

    No-op safe: if ``init()`` hasn't been called yet, the callback is
    discarded (registration is not buffered). Call ``init()`` first.
    """
    from . import _config as _cfg
    sender = getattr(_cfg, "_sender", None)
    if sender is None:
        logger.warning(
            "decimalai.on_export_error called before init(); callback discarded."
        )
        return
    sender.register_error_callback(callback)


def flush() -> None:
    """Synchronously flush any buffered traces.

    The atexit handler at `_atexit_flush` is the safety net for normal
    interpreter shutdown, but it doesn't fire in environments where the
    process is killed before atexit runs: CI runners that SIGKILL on
    timeout, async event loops that exit before the handler completes,
    daemonized workers, Jupyter kernels restarted via interrupt. Calling
    this explicitly before exit/teardown guarantees the buffered traces
    reach the backend.

    No-op if `init()` hasn't been called.

    Before this addition, `decimalai.flush()` raised AttributeError —
    users had to reach into the private `_config._client.flush()` path.

    It also drains the background-sender queue (`_config._sender.flush()`),
    not just the batch buffer. Earlier, `decimalai.flush()` only drained
    the `_client._trace_buffer` batch path — the per-trace
    `auto_send=True` route (used by `start_trace`) went through
    `_sender.submit()` and was NOT awaited. Closing the gap means
    `last_send_error()` returns a meaningful value right after a
    `flush()` call, instead of only after atexit-shutdown.
    """
    try:
        from . import _config as _cfg
        client = getattr(_cfg, "_client", None)
        if client is not None:
            client.flush()
        sender = getattr(_cfg, "_sender", None)
        if sender is not None:
            sender.flush()
            # Emit a one-line summary if anything failed in this batch.
            # Silent on the happy path. The point is to catch "wait, my
            # traces aren't arriving" cases at the moment the user is
            # actively looking (just called flush()) — not buried in a
            # DEBUG log they'd only find in postmortem.
            try:
                st = sender.status()
                if st.failed > 0:
                    logger.warning(
                        "decimalai.flush(): %d trace(s) sent, %d failed%s. "
                        "Call decimalai.export_status() for details.",
                        st.sent,
                        st.failed,
                        f" (last error: {st.last_error})" if st.last_error else "",
                    )
            except Exception:
                pass
    except Exception as e:
        logger.warning("decimalai.flush() failed: %s", e)


def _verify_backend_at_init(
    *,
    base_url: str,
    api_key: str,
    timeout: float,
) -> None:
    """Probe the backend with the configured key to fail loudly on bad setup.

    Hits ``GET {base_url}/api/v1/auth/verify``:
      - 200          → cache ``require_manifest_on_ingest`` on the global
                       config and return.
      - 401 / 403    → raise ``DecimalConfigError`` with a clear message —
                       the user almost certainly has a typo or revoked key.
      - other 4xx    → log warning, do not raise (the SDK can still try
                       to send; the backend might just be missing the
                       /auth/verify route, which is the case for older
                       backends).
      - 5xx          → log warning, do not raise (backend is up but
                       degraded; let the sender retry).
      - timeout      → log warning, do not raise (transient network).
      - connection   → raise ``DecimalConfigError`` (base_url is wrong
                       or backend is down; failing fast lets the caller
                       see the URL they actually configured).

    Designed to be safe to call at every init; it does one GET and
    populates global config. No state otherwise.
    """
    import urllib.error
    import urllib.request

    from . import _config as _cfg
    from ._config import DecimalConfigError

    url = f"{base_url.rstrip('/')}/api/v1/auth/verify"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": f"decimalai-sdk/{__version__} (init-verify)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            import json
            body = json.loads(resp.read().decode("utf-8"))
            if _cfg._config is not None:
                _cfg._config.backend_require_manifest_on_ingest = bool(
                    body.get("require_manifest_on_ingest", False)
                )
            logger.debug(
                "decimalai.init: verify ok (scope=%s, strict_mode=%s)",
                body.get("scope"),
                body.get("require_manifest_on_ingest"),
            )
            return
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise DecimalConfigError(
                f"Invalid API key — the server rejected it ({url}, HTTP {exc.code}). "
                "The key is invalid, revoked, or scoped to a different "
                "workspace. Get a valid key at "
                "https://app.decimal.ai/settings (Settings → API Key). "
                "Pass init(verify=False) to skip this probe."
            ) from exc
        # Older backends may not have /api/v1/auth/verify at all.
        # 404 should not crash init; the sender will surface real
        # failures via export_status() if traces don't land.
        logger.warning(
            "decimalai.init: verify probe got HTTP %d from %s — continuing. "
            "Trace failures will be surfaced via decimalai.export_status().",
            exc.code,
            url,
        )
        return
    except urllib.error.URLError as exc:
        # Connection refused, DNS failure, etc. The base_url is wrong
        # or the backend is down. Fail at init so the caller can fix it
        # before launching a long-running agent.
        raise DecimalConfigError(
            f"Could not reach DecimalAI backend at {base_url!r}: {exc.reason!s}. "
            "Check DECIMAL_BASE_URL and your network. "
            "Pass init(verify=False) to skip this probe."
        ) from exc
    except TimeoutError:
        # Transient network slowness — don't crash. The sender will
        # surface real failures later.
        logger.warning(
            "decimalai.init: verify probe timed out after %.1fs against %s — "
            "continuing. Trace failures will be surfaced via "
            "decimalai.export_status().",
            timeout,
            url,
        )
        return
    except Exception as exc:
        # Belt-and-suspenders. Init should never crash from the probe
        # except for the explicit raise branches above.
        logger.warning(
            "decimalai.init: verify probe failed unexpectedly (%s: %s) — "
            "continuing without a backend probe. "
            "Use decimalai.export_status() if traces don't appear.",
            type(exc).__name__,
            exc,
        )


def init(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    project: Optional[str] = None,
    enabled: bool = True,
    langchain: bool = False,
    openai_agents: bool = False,
    adk: bool = False,
    llamaindex: bool = False,
    claude_agent_sdk: bool = False,
    crewai: bool = False,
    autogen: bool = False,
    otel: bool = False,
    openai: bool = False,
    anthropic: bool = False,
    google: bool = False,
    agent_name: Optional[str] = None,
    verify: bool = True,
    verify_timeout: float = 3.0,
    skill_authority: Optional[str] = None,
    inject_skill_body: Optional[bool] = None,
    load_skill_tool: Optional[bool] = None,
) -> None:
    """Initialize the DecimalAI SDK.

    Must be called once before using any integration. Configuration is
    resolved in order: explicit parameter → environment variable → default.

    Supported frameworks::

        decimalai.init(langchain=True)       # LangChain / LangGraph
        decimalai.init(openai_agents=True)   # OpenAI Agents SDK
        decimalai.init(adk=True)             # Google ADK (Agent Development Kit)
        decimalai.init(llamaindex=True)      # LlamaIndex
        decimalai.init(claude_agent_sdk=True)  # Anthropic Claude Agent SDK
        decimalai.init(crewai=True)          # CrewAI
        decimalai.init(autogen=True)         # AutoGen / AG2
        decimalai.init(otel=True)            # Any OpenTelemetry framework

    No-framework direct provider-SDK calls (the one-liner for raw
    ``openai`` / ``anthropic`` / ``google.genai`` usage with no agent
    framework in between)::

        decimalai.init(openai=True)          # trace raw OpenAI SDK calls
        decimalai.init(anthropic=True)       # ...or Anthropic
        decimalai.init(google=True)          # ...or Google GenAI

    Args:
        api_key: API key. Falls back to ``DECIMAL_API_KEY`` env var.
        base_url: Backend URL. Falls back to ``DECIMAL_BASE_URL``, then
            ``https://api.decimal.ai``.
        project: DEPRECATED and inert — the platform never read it, so it has
            never grouped anything. Accepted for back-compat; emits a
            ``DeprecationWarning`` and will be removed. Use workspaces (scoped
            by the API key) to group traces.
        enabled: Set ``False`` to disable all tracing (integrations become no-ops).
        langchain: If ``True``, instrument LangChain / LangGraph.
        openai_agents: If ``True``, instrument the OpenAI Agents SDK.
        adk: If ``True``, instrument Google ADK (Agent Development Kit) via a
            native ADK plugin. ADK is Gemini-native.
        llamaindex: If ``True``, instrument LlamaIndex (v0.10.20+).
        claude_agent_sdk: If ``True``, instrument Anthropic's Claude Agent SDK
            (``claude-agent-sdk``) by wrapping its ``query()`` stream. Anthropic-native.
        crewai: If ``True``, instrument CrewAI (via OpenTelemetry).
        autogen: If ``True``, instrument AutoGen / AG2 (via OpenTelemetry).
        otel: If ``True``, install a generic OpenTelemetry exporter.
            Use this for any OTEL-compatible framework not listed above.
        openai: If ``True``, auto-trace direct OpenAI SDK calls (no
            framework) via the OpenAI OpenInference instrumentor. Don't
            combine with a framework flag that already traces OpenAI
            (e.g. ``openai_agents``/``langchain``) or the call is captured
            twice — see :mod:`decimalai.providers`.
        anthropic: If ``True``, auto-trace direct Anthropic SDK calls.
        google: If ``True``, auto-trace direct Google GenAI
            (``google.genai``) SDK calls.
        agent_name: Default agent name for auto-install integrations.
        verify: If ``True`` (default), synchronously probe the backend
            during init to (a) validate the API key, (b) detect a
            wrong/unreachable ``base_url``, and (c) cache the backend's
            strict-mode flag so the SDK can warn about
            ``manifest_id``-required misconfigurations at the source.
            Raises ``DecimalConfigError`` on 401/403 or connection
            failure — failing loud at init beats a silent day of
            background-send 401s. Pass ``verify=False`` in CI / cold-
            start sensitive code paths where the ~50-200ms probe is
            unacceptable.
        verify_timeout: Seconds before the verify probe gives up.
            On timeout, init() logs a warning and continues (does not
            raise) so a transiently slow backend doesn't break startup.

    Raises:
        DecimalConfigError: If ``api_key`` is not provided and not in env,
            OR (when ``verify=True``) if the backend rejects the key
            (401/403) or is unreachable.
    """
    import decimalai._config as _cfg

    from ._client import DecimalAIClient
    from ._config import DecimalConfig, DecimalConfigError

    # Resolve API key
    resolved_key = api_key or os.environ.get("DECIMAL_API_KEY", "")
    if isinstance(resolved_key, str):
        resolved_key = resolved_key.strip()
    if not resolved_key and enabled:
        raise DecimalConfigError(
            "No API key provided. Pass api_key= to decimalai.init() "
            "or set the DECIMAL_API_KEY environment variable."
        )
    if enabled and resolved_key:
        # Reject keys that contain control characters / non-printable bytes —
        # they cause obscure 400/401 failures at the backend rather than a
        # clear auth error the caller can fix.
        if any(ord(c) < 0x20 or ord(c) == 0x7F for c in resolved_key):
            raise DecimalConfigError(
                "API key contains non-printable characters. "
                "Check for stray \\n, \\t, or binary bytes in DECIMAL_API_KEY."
            )
        # Warn (don't raise) on unrecognized key shape — keeps custom-deployment
        # keys working while flagging obvious typos like 'sk-...' (OpenAI key).
        if not (resolved_key.startswith("dai_sk_") or resolved_key.startswith("dai_pk_")):
            logger.warning(
                "API key does not look like a DecimalAI key (expected prefix "
                "'dai_sk_' or 'dai_pk_'). Traces will fail at the backend if "
                "the key is invalid. Got prefix: %r",
                resolved_key[:8],
            )

    # Resolve base URL
    resolved_url = (
        base_url
        or os.environ.get("DECIMAL_BASE_URL", "")
        or "https://api.decimal.ai"
    )

    # Detect CI manifest-extraction mode from env var.
    # When set, framework integrations skip trace emission and the
    # background sender doesn't start. The user's init script is expected
    # to call decimalai.flush_manifest_for_ci() to upload the captured
    # manifest as a regression-check candidate.
    manifest_only = (
        os.environ.get("DECIMALAI_MODE", "").strip().lower() == "manifest_only"
    )

    # `project` has never done anything. The SDK sent it as an X-Decimal-Project
    # header; the platform reads no such header, and a trace's project_id is set
    # only for a project-scoped API key. So a caller who passed project= got
    # silent no-grouping — the exact "silent no-op" failure the docs warn about
    # elsewhere. The header is gone as of 0.10.0; warn rather than raise so
    # existing code keeps running while the kwarg is retired.
    if project is not None:
        import warnings

        warnings.warn(
            "decimalai.init(project=...) is deprecated and has no effect — the "
            "platform never read it, so it has never grouped traces. Remove the "
            "argument; use workspaces (scoped by your API key) to group traces. "
            "The parameter will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )

    # explicit param → env var → default. Omit when None so DecimalConfig's
    # env-reading default_factory runs (DECIMALAI_SKILL_AUTHORITY /
    # DECIMALAI_INJECT_SKILL_BODY).
    _config_kwargs = dict(
        api_key=resolved_key,
        base_url=resolved_url.rstrip("/"),
        project=project,
        enabled=enabled,
        manifest_only=manifest_only,
    )
    if skill_authority is not None:
        _config_kwargs["skill_authority"] = skill_authority.strip().lower()
    if inject_skill_body is not None:
        _config_kwargs["inject_skill_body"] = inject_skill_body
    if load_skill_tool is not None:
        _config_kwargs["load_skill_tool"] = load_skill_tool
    config = DecimalConfig(**_config_kwargs)
    _cfg._config = config

    if enabled:
        _cfg._client = DecimalAIClient(
            api_key=config.api_key,
            base_url=config.base_url,
            project=config.project,
        )
        # Register the atexit flush handler exactly once. This prevents
        # silent trace loss when a script ingests <50 traces and exits.
        global _atexit_registered
        if not _atexit_registered:
            atexit.register(_atexit_flush)
            _atexit_registered = True

        # Init-time health probe. Default-on because the failure mode
        # it prevents (a whole day of traces silently 401'd because of
        # a typo in DECIMAL_API_KEY) is dramatically worse than the
        # 50-200ms one-time cost. Opt-out via verify=False for CI or
        # cold-start sensitive paths, or via the env var
        # ``DECIMALAI_SKIP_VERIFY=1`` for unit-test suites that don't
        # have a real backend (set once in conftest, no per-test churn).
        skip_verify_env = os.environ.get("DECIMALAI_SKIP_VERIFY", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        if verify and not manifest_only and not skip_verify_env:
            _verify_backend_at_init(
                base_url=config.base_url,
                api_key=config.api_key,
                timeout=verify_timeout,
            )

        if manifest_only:
            logger.info(
                "DecimalAI SDK initialized in manifest_only mode "
                "(CI manifest extraction; trace emission suppressed)"
            )
        else:
            logger.info(
                "DecimalAI SDK initialized: base_url=%s project=%s",
                config.base_url,
                config.project,
            )
    else:
        _cfg._client = None
        logger.info("DecimalAI SDK initialized in disabled mode (no-op)")

    # Update module-level client ref
    import decimalai as _self
    _self._global_client = _cfg._client

    # Auto-install langchain tracing if requested
    if langchain:
        from .langchain import instrument as _lc_install
        _lc_install(agent_name=agent_name)

    # Auto-install OpenAI Agents tracing if requested
    if openai_agents:
        from .openai_agents import instrument as _oai_install
        _oai_install(agent_name=agent_name)

    # Auto-install Google ADK tracing if requested
    if adk:
        from .adk import instrument as _adk_install
        _adk_install(agent_name=agent_name)

    # Auto-install LlamaIndex span handler if requested
    if llamaindex:
        from .llamaindex import instrument as _li_install
        _li_install(agent_name=agent_name)

    # Auto-install Claude Agent SDK stream tracing if requested
    if claude_agent_sdk:
        from .claude_agent_sdk import instrument as _cas_install
        _cas_install(agent_name=agent_name)

    # Auto-install OTEL exporter for CrewAI, AutoGen, or generic OTEL
    # CrewAI and AutoGen emit standard OpenTelemetry GenAI spans,
    # so they use the same exporter — the named flags are just
    # convenience aliases for discoverability.
    if otel or crewai or autogen:
        # Use the manifest-capable exporter (decimalai.otel) — it buffers spans
        # by root span (no per-batch fragmentation) AND registers a manifest
        # from the captured model/tools/prompt, so the versioning moat engages
        # for CrewAI/AutoGen/generic-OTel. (The older integrations.otel exporter
        # did neither.)
        from .otel import instrument as _otel_install
        _otel_install(agent_name=agent_name)

    # Auto-trace direct provider-SDK calls (no framework) if requested.
    # Each flag enables the matching provider's OpenInference instrumentor,
    # routed through decimalai's OTEL exporter. See decimalai.providers.
    if openai or anthropic or google:
        from .providers import instrument as _provider_instrument
        _provider_instrument(
            openai=openai,
            anthropic=anthropic,
            google=google,
            agent_name=agent_name,
        )


def send(trace) -> None:
    """Manually send a trace to the backend.

    For advanced usage when ``auto_send=False``.
    """
    from ._config import _get_client

    client = _get_client()
    client.ingest_trace(trace)


def ingest_raw(payload: dict) -> dict:
    """Send a raw trace dict directly to the DecimalAI backend.

    This is the lowest-level interface — pass any dict that conforms
    to the RunTrace JSON schema. No Pydantic model, no SDK overhead.
    Useful for custom pipelines, batch imports, and non-Python sources.

    Args:
        payload: A dict matching the RunTrace JSON schema. Must include
            at minimum ``agent_name``, ``started_at``, ``ended_at``.

    Returns:
        The server response as a dict.

    Raises:
        DecimalConfigError: If the SDK has not been initialized.

    Example::

        import decimalai
        decimalai.init(api_key="...")

        decimalai.ingest_raw({
            "agent_name": "my-agent",
            "status": "success",
            "started_at": "2025-01-01T00:00:00Z",
            "ended_at": "2025-01-01T00:00:01Z",
            "llm_calls": [{"model_name": "gpt-4o", ...}],
        })
    """
    from ._config import _get_client

    client = _get_client()
    return client.ingest_raw_trace(payload)


def ingest_raw_batch(payloads: list) -> dict:
    """Send a batch of raw trace dicts to the DecimalAI backend.

    Args:
        payloads: A list of dicts, each matching the RunTrace JSON schema.

    Returns:
        The server response as a dict.

    Example::

        from datetime import datetime, timezone

        decimalai.ingest_raw_batch([
            {
                "agent_name": "support-agent",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "status": "success",
                "user_input_preview": "How do I reset my password?",
            },
            {
                "agent_name": "support-agent",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "status": "success",
                "user_input_preview": "Where is my receipt?",
            },
        ])
    """
    from ._config import _get_client

    client = _get_client()
    return client.ingest_raw_traces_batch(payloads)


def register_manifest(
    agent_name: str,
    tools: Optional[list] = None,
    prompts: Optional[dict] = None,
    models: Optional[dict] = None,
    subagents: Optional[list] = None,
    output_schema: Optional[dict] = None,
    guardrails: Optional[list] = None,
    context_config: Optional[dict] = None,
    skills: Optional[list] = None,
    workflow: Optional[dict] = None,
    behavioral_policy: Optional[dict] = None,
    environment: Optional[dict] = None,
    version_label: Optional[str] = None,
    contract_mode: str = "closed",
) -> dict:
    """Register an agent manifest for version tracking.

    Call this once at app startup to declare your agent's contract.
    DecimalAI will hash the manifest and auto-detect changes across
    deployments, triggering compatibility reports.

    This is the **explicit** manifest registration method — for full
    control over what gets tracked. For automatic detection, use
    ``decimalai.init(langchain=True)`` or ``@decimalai.tool``.

    Args:
        agent_name: Name of the agent.
        tools: List of tool descriptors ``[{"name": ..., "schema": ...}]``.
        prompts: Dict of prompt templates ``{"system": "..."}``.
        models: Dict of model configs ``{"default": {"model": "gpt-4o", ...}}``.
        subagents: List of subagent descriptors ``[{"name": ...}]``.
        output_schema: Output schema dict.
        guardrails: List of guardrail descriptors ``[{"name": ..., "type": ...}]``
            (declares the safety/validation surface; ``"closed"`` mode flags
            missing/failed guardrails in production).
        context_config: Retrieval/memory context descriptor
            ``{"sources": [...], ...}`` (declares the context surface; ``"closed"``
            mode flags undeclared context sources).
        skills: List of skill descriptors ``[{"name": ..., "hash": ...}]``
            (declares the skill registry — pillar 1).
        workflow: Workflow/graph topology descriptor ``{"name": ..., "hash": ...}``
            (declares the agent's graph for the versioning moat).
        behavioral_policy: Versioned policy-document surface ``{"policy_id",
            "policy_hash", "rules"?}`` — binds the agent to a named policy
            artifact (refund rules, a safety guardrail set, an escalation SOP…)
            by hash. The dict is opaque (hashed whole, or by ``policy_hash``), so
            any change to the bound policy diffs as breaking and a policy flip
            doesn't hide in the prompt hash.
        environment: Deployment/infra surface ``{"deployment_id", "region",
            "infra_image_hash", "runtime_versions", ...}``.
        version_label: Human-readable version label (e.g., ``"v2.1"``).
        contract_mode: ``"closed"`` (default) or ``"descriptive"``.

            * ``"closed"`` — the registered manifest is treated as a complete
              contract. The backend flags production traces that exercise
              components outside the manifest as contract violations
              (undeclared tools, out-of-scope models, off-schema outputs,
              undeclared context sources, missing/failed guardrails). This
              is the default because explicit declaration IS the act of
              declaring a contract.
            * ``"descriptive"`` — the manifest records what the agent has,
              but new components in production are not flagged. Useful during
              prototyping when you're still enumerating components.

            Override via ``decimalai.register_manifest(..., contract_mode="descriptive")``
            if you want strict capture without violation detection.

    Returns:
        API response with ``manifest_id`` and compatibility info.

    Raises:
        ValueError: If ``contract_mode`` is not ``"closed"`` or ``"descriptive"``.

    Example::

        import decimalai
        decimalai.init()

        decimalai.register_manifest(
            agent_name="my-agent",
            tools=[{"name": "search", "schema": {"type": "object"}}],
            prompts={"system": "You are a helpful assistant."},
            models={"default": {"provider": "openai", "model": "gpt-4o"}},
        )
    """
    if contract_mode not in ("closed", "descriptive"):
        raise ValueError(
            f"contract_mode must be 'closed' or 'descriptive', got {contract_mode!r}"
        )

    from ._config import _get_client
    from .schema.manifest import extract_from_config

    snapshot = extract_from_config(
        agent_name=agent_name,
        tools=tools,
        prompts=prompts,
        models=models,
        subagents=subagents,
        output_schema=output_schema,
        guardrails=guardrails,
        context_config=context_config,
        skills=skills,
        workflow=workflow,
        behavioral_policy=behavioral_policy,
        environment=environment,
        version_label=version_label,
        is_closed_world=(contract_mode == "closed"),
    )

    client = _get_client()
    return client.register_manifest(snapshot)


def export_manifest(
    snapshot: "ManifestSnapshot",
    format: str = "agentversion",
    *,
    version_label: Optional[str] = None,
    created_by: Optional[dict] = None,
    parent_manifest_id: Optional[str] = None,
    description: Optional[str] = None,
) -> dict:
    """Export an SDK manifest snapshot to an external manifest format.

    The SDK stores a manifest in component-list shape (:class:`ManifestSnapshot`,
    e.g. from :func:`decimalai.schema.manifest.extract_from_config`). The OSS
    ``agentversion`` tool reads the contract-keyed shape its ``diff`` /
    ``validate`` commands consume. This is the seam between the two: a pure,
    offline conversion that mints a canonical ``amf_<ULID>`` id and emits a dict
    that validates against ``agentversion.AgentManifest``.

    ``agentversion`` is an **optional** dependency — it is never imported unless
    present. When it (and its ``jcs`` dep) is installed, the canonical
    ``jcs-sha256`` hash is computed so the export validates with zero warnings;
    otherwise the SDK's surface hash is carried under an honest algorithm label.

    Args:
        snapshot: The :class:`ManifestSnapshot` to convert.
        format: Target format. Only ``"agentversion"`` is supported today.
        version_label, created_by, parent_manifest_id, description:
            Forwarded to the agentversion exporter (all optional).

    Returns:
        A manifest dict in the requested format.

    Raises:
        ValueError: If ``format`` is not a supported target.

    Example::

        import decimalai
        from decimalai.schema.manifest import extract_from_config

        snap = extract_from_config(
            agent_name="support-agent",
            prompts={"system": "You are a helpful assistant."},
            models={"default": {"provider": "openai", "model": "gpt-4o"}},
        )
        manifest = decimalai.export_manifest(snap)  # -> agentversion dict
    """
    if format == "agentversion":
        return snapshot.to_agentversion(
            version_label=version_label,
            created_by=created_by,
            parent_manifest_id=parent_manifest_id,
            description=description,
        )
    raise ValueError(
        f"Unsupported manifest export format {format!r}; supported: 'agentversion'"
    )


def repair_preview(
    old_manifest_id: str,
    new_manifest_id: str,
    sample_size: int = 5,
) -> dict:
    """Preview mechanical repair rules for a manifest transition.

    Lets you complete the detect→impact→**repair**→export loop from CI/SDK
    without the dashboard. Returns the preview dict from the platform; each rule
    in ``result["rules"]`` is positionally indexed for :func:`repair_apply`.
    Requires :func:`init`.
    """
    from ._config import _get_client

    client = _get_client()
    return client.repair_preview(
        old_manifest_id, new_manifest_id, sample_size=sample_size
    )


def repair_apply(
    old_manifest_id: str,
    new_manifest_id: str,
    approved_rule_indices: Optional[list] = None,
) -> dict:
    """Apply repairs for a manifest transition.

    Applies all eligible rules, or only the given 0-based indices into the
    :func:`repair_preview` ``rules`` array. Returns the repair-batch summary.
    Requires :func:`init`.
    """
    from ._config import _get_client

    client = _get_client()
    return client.repair_apply(
        old_manifest_id, new_manifest_id, approved_rule_indices=approved_rule_indices
    )


def flush_manifest_for_ci(
    agent_name: str,
    *,
    chain: Optional[Any] = None,
    tools: Optional[list] = None,
    prompts: Optional[dict] = None,
    models: Optional[dict] = None,
    subagents: Optional[list] = None,
    output_schema: Optional[dict] = None,
    guardrails: Optional[list] = None,
    context_config: Optional[dict] = None,
    skills: Optional[list] = None,
    workflow: Optional[dict] = None,
    behavioral_policy: Optional[dict] = None,
    environment: Optional[dict] = None,
    version_label: Optional[str] = None,
    output_path: Optional[str] = None,
) -> dict:
    """Upload a manifest as a CI regression-check candidate and write the ID.

    This is the helper called by the customer's init_for_decimal.py script
    when running under DECIMALAI_MODE=manifest_only. It:

    1. Registers the manifest via the standard /api/v1/manifests endpoint
       (the backend stores it; the regression-check service will treat it
       as a candidate when invoked with this manifest_id).
    2. Reads PR context from GitHub Actions environment variables
       (GITHUB_REPOSITORY, GITHUB_HEAD_REF, GITHUB_REF, GITHUB_SHA).
    3. Writes the resulting manifest_id to one of:
       - $GITHUB_OUTPUT (the standard GitHub Actions output mechanism)
       - The path passed in `output_path`, if provided
       - decimal_manifest_id.txt in the current directory (fallback)

       The next step in the GitHub Action workflow reads this ID and
       passes it to the regression-check API.

    Args:
        agent_name: Name of the agent.
        tools, prompts, models, subagents, output_schema, guardrails,
        context_config, skills, workflow, behavioral_policy, environment,
        version_label: Same as register_manifest().
        output_path: Optional explicit file path to write the manifest_id.
            If not provided, falls back to $GITHUB_OUTPUT or
            ./decimal_manifest_id.txt.

    Returns:
        Dict with at minimum:
            - manifest_id (str)
            - pr_context (dict): the PR context derived from env vars
            - output_path (str): where the manifest_id was written

    Raises:
        DecimalConfigError: if SDK is not initialized.

    Example::

        # In scripts/init_for_decimal.py, called by the GitHub Action
        import decimalai
        decimalai.init()  # picks up DECIMALAI_MODE=manifest_only from env

        from myapp.agent import build_agent
        agent = build_agent()  # captures into framework integration state

        decimalai.flush_manifest_for_ci(
            agent_name="support-agent",
            tools=[...],
            prompts={...},
            models={...},
        )

        # Or via LangChain introspection (no explicit dicts needed):
        from langchain.agents import create_react_agent
        agent = create_react_agent(llm, tools, prompt)
        decimalai.flush_manifest_for_ci(
            agent_name="support-agent",
            chain=agent,  # introspect tools/prompts/models from the agent object
        )

        # langgraph `create_react_agent` users: tools auto-detect, but
        # model + prompt are closure-captured and not extractable.
        # Pass them explicitly:
        from langgraph.prebuilt import create_react_agent
        agent = create_react_agent(model=llm, tools=tools, prompt=prompt_text)
        decimalai.flush_manifest_for_ci(
            agent_name="support-agent",
            chain=agent,
            prompts={"system": prompt_text},
            models={"default": {"provider": "openai", "model": "gpt-4o"}},
        )
    """
    # If a LangChain chain is provided, introspect it for tools/prompts/models.
    # Explicit args override anything introspection finds — that's the escape
    # hatch when introspection picks up the wrong thing.
    if chain is not None:
        from .integrations.langchain_introspect import introspect_chain
        i_tools, i_prompts, i_models = introspect_chain(chain)
        tools = tools or i_tools
        prompts = prompts or i_prompts
        models = models or i_models

    response = register_manifest(
        agent_name=agent_name,
        tools=tools,
        prompts=prompts,
        models=models,
        subagents=subagents,
        output_schema=output_schema,
        guardrails=guardrails,
        context_config=context_config,
        skills=skills,
        workflow=workflow,
        behavioral_policy=behavioral_policy,
        environment=environment,
        version_label=version_label,
    )

    manifest_id = response.get("manifest_id")
    if not manifest_id:
        raise RuntimeError(
            f"Manifest registration did not return a manifest_id. Response: {response!r}"
        )

    pr_context = _read_github_pr_context()
    written_to = _write_manifest_id_for_ci(manifest_id, output_path)

    logger.info(
        "Manifest %s registered as CI candidate; id written to %s. "
        "Next: run `decimalai regression-check` to compute the impact report.",
        manifest_id,
        written_to,
    )

    return {
        "manifest_id": manifest_id,
        "pr_context": pr_context,
        "output_path": written_to,
        "registration_response": response,
    }


def _read_github_pr_context() -> dict:
    """Read GitHub Actions PR context from environment variables.

    Returns an empty dict if not running under GitHub Actions or if the
    relevant env vars are missing. Best-effort — callers should treat
    missing fields as expected (e.g., when running via CLI locally).
    """
    repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if not repo:
        return {}

    pr_number = None
    ref_name = os.environ.get("GITHUB_REF_NAME", "").strip()
    # GITHUB_REF for PRs is "refs/pull/<number>/merge"
    github_ref = os.environ.get("GITHUB_REF", "")
    if github_ref.startswith("refs/pull/"):
        try:
            pr_number = int(github_ref.split("/")[2])
        except (IndexError, ValueError):
            pr_number = None

    return {
        "repo": repo,
        "pr_number": pr_number,
        "branch": os.environ.get("GITHUB_HEAD_REF") or ref_name or None,
        "commit_sha": os.environ.get("GITHUB_SHA", "").strip() or None,
    }


def _write_manifest_id_for_ci(manifest_id: str, output_path: Optional[str]) -> str:
    """Write the manifest_id to a path the next CI step can read.

    Resolution order:
        1. Explicit `output_path` argument
        2. $GITHUB_OUTPUT (standard GitHub Actions output mechanism)
        3. ./decimal_manifest_id.txt (fallback for local CLI use)

    For $GITHUB_OUTPUT, writes in the `key=value` format expected by Actions:
        decimal_manifest_id=<id>

    Returns the path written to.
    """
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(manifest_id)
        return output_path

    github_output = os.environ.get("GITHUB_OUTPUT", "").strip()
    if github_output:
        # Append in key=value format so the next step can `${{ steps.x.outputs.decimal_manifest_id }}`
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"decimal_manifest_id={manifest_id}\n")
        return github_output

    fallback = "decimal_manifest_id.txt"
    with open(fallback, "w", encoding="utf-8") as f:
        f.write(manifest_id)
    return fallback


def get_replay_prompts(
    agent_name: str,
    verdict: Optional[str] = None,
    limit: int = 500,
) -> dict:
    """Get stale prompts that need to be re-run through your agent.

    Returns prompts from traces that were classified as needing replay
    after a manifest change. Download these, run them through your
    updated agent, and send the new traces back via the SDK.

    Args:
        agent_name: Agent name to get replay prompts for.
        verdict: Filter by verdict (replay, drop, repair).
                 Defaults to replay + drop.
        limit: Maximum number of prompts (default 500, max 5000).

    Returns:
        Dict with ``agent_name``, ``total``, and ``prompts`` list.

    Example::

        import decimalai
        decimalai.init()

        result = decimalai.get_replay_prompts("my-agent")
        for prompt in result["prompts"]:
            print(prompt["user_input"])
    """
    from ._config import _get_client

    client = _get_client()
    return client.get_replay_prompts(agent_name, verdict=verdict, limit=limit)


def create_replay_batch(
    source_manifest_id: str,
    target_manifest_id: str,
    trace_ids: list,
) -> dict:
    """Create a replay batch from stale traces.

    Groups traces into tasks that can be pulled, replayed externally,
    and submitted back.

    Args:
        source_manifest_id: Manifest the traces were recorded against.
        target_manifest_id: New manifest version to replay against.
        trace_ids: List of trace IDs to include in the batch.

    Returns:
        Dict with ``batch_id``, ``total_tasks``, ``batch_status``.
    """
    from ._config import _get_client

    client = _get_client()
    return client.create_replay_batch(source_manifest_id, target_manifest_id, trace_ids)


def get_replay_batch(batch_id: str) -> dict:
    """Get replay batch progress, including per-task status.

    Args:
        batch_id: The batch ID returned from ``create_replay_batch()``.

    Returns:
        Dict with batch status, progress counts, and task list.
    """
    from ._config import _get_client

    client = _get_client()
    return client.get_replay_batch(batch_id)


def submit_replay_result(
    task_id: str,
    replayed_trace_id: Optional[str] = None,
    eval_score: Optional[float] = None,
    eval_verdict: Optional[str] = None,
    status: str = "completed",
) -> dict:
    """Submit the result of a replay task.

    After re-running a prompt through your updated agent,
    submit the result back so DecimalAI can score it.

    Args:
        task_id: The replay task ID.
        replayed_trace_id: ID of the new trace (auto-captured by SDK).
        eval_score: Optional quality score (0.0 to 1.0).
        eval_verdict: Optional verdict (pass/fail).
        status: Task status (completed/failed/skipped).

    Returns:
        Updated task details with eval results.
    """
    from ._config import _get_client

    client = _get_client()
    return client.submit_replay_result(
        task_id,
        replayed_trace_id=replayed_trace_id,
        eval_score=eval_score,
        eval_verdict=eval_verdict,
        status=status,
    )


def eval(
    trace_id: str,
    name: str,
    score: float,
    *,
    source: str = "custom",
    source_label: Optional[str] = None,
    passed: Optional[bool] = None,
    reason: Optional[str] = None,
    category: str = "quality",
) -> dict:
    """Push a single eval score to a trace.

    This is the simplest way to attach evaluation results to a trace.
    All scores are visible in the dashboard's Evaluation Breakdown card
    grouped by source.

    Args:
        trace_id: The trace to attach the score to.
        name: Metric name (e.g., "factual_accuracy", "coherence").
        score: Score value between 0.0 and 1.0.
        source: Eval source identifier. Defaults to "custom".
        source_label: Human-readable display name (e.g., "My RAG Eval").
        passed: Binary pass/fail. Defaults to score >= 0.5.
        reason: Human-readable explanation of the score.
        category: "quality" (default) or "compatibility".

    Returns:
        API response with stored score details and recomputed verdict.

    Example::

        import decimalai
        decimalai.init()

        decimalai.eval(
            trace_id="abc123",
            name="factual_accuracy",
            score=0.75,
            reason="3/4 facts verified against source docs",
        )
    """
    from ._config import _get_client

    client = _get_client()
    score_entry = {
        "name": name,
        "score": score,
        "passed": passed if passed is not None else score >= 0.5,
    }
    if reason:
        score_entry["reason"] = reason

    metadata = {}
    if source_label:
        metadata["source_label"] = source_label

    return client.push_eval_scores(
        trace_id=trace_id,
        source=source,
        scores=[score_entry],
        metadata=metadata or None,
    )


def score(
    trace_id: str,
    name: str,
    value: float,
    reason: Optional[str] = None,
) -> dict:
    """Shorthand for pushing a single eval score.

    Args:
        trace_id: The trace to attach the score to.
        name: Metric name.
        value: Score between 0.0 and 1.0.
        reason: Optional explanation.

    Returns:
        API response.

    Example::

        decimalai.score("abc123", "factual_accuracy", 0.75)
    """
    return eval(
        trace_id=trace_id,
        name=name,
        score=value,
        reason=reason,
    )


def get_eval_breakdown(trace_id: str) -> dict:
    """Get the full eval breakdown for a trace with provenance info.

    Returns scores grouped by source (Manifest Diff, DeepEval, LangSmith,
    Custom, etc.) with icons, labels, and decision reasons.

    Returns:
        Dict with eval_verdict, quality_avg, compat_avg, source_groups,
        and decision_reasons.

    Example::

        import decimalai
        decimalai.init()

        bd = decimalai.get_eval_breakdown("abc123")
        print(f"Verdict: {bd['eval_verdict']}")
    """
    from ._config import _get_client

    client = _get_client()
    return client.get_eval_breakdown(trace_id)


def pull_dataset(
    dataset_id: str,
    path: str,
    *,
    version: Optional[str] = None,
    format: Optional[str] = None,
) -> dict:
    """Download a dataset to a local file for training.

    This is the primary way to get training data from DecimalAI onto disk.
    It handles version resolution, download, and file writing in a single call.

    Args:
        dataset_id: The dataset to download.
        path: Local file path to write to (e.g., ``"./training_data.jsonl"``).
              Parent directories are created automatically.
        version: Which version to pull:

            - ``None`` or ``"latest"`` → the most recent version (default)
            - ``"v3"`` or ``"3"`` → version 3 specifically
            - A full version UUID → that exact version

        format: Export format: ``"jsonl"`` (default) or ``"parquet"``.
                If not specified, inferred from the file extension.

    Returns:
        Summary dict with ``row_count``, ``file_path``, ``bytes_written``,
        ``format``, ``version_id``, and ``dataset_id``.

    Example::

        import decimalai
        decimalai.init()

        # Pull the latest version
        result = decimalai.pull_dataset("ds_abc123", "./training_data.jsonl")
        print(f"Wrote {result['row_count']} rows to {result['file_path']}")

        # Pull a specific version as Parquet
        result = decimalai.pull_dataset(
            "ds_abc123",
            "./data.parquet",
            version="v2",
        )

        # Pull by version number
        result = decimalai.pull_dataset(
            "ds_abc123", "./data.jsonl", version="3"
        )
    """
    from ._config import _get_client

    client = _get_client()
    return client.pull_dataset(
        dataset_id, path, version=version, format=format,
    )


def push_to_hub(
    dataset_id: str,
    repo_id: str,
    *,
    version: Optional[str] = None,
    token: Optional[str] = None,
    private: bool = True,
    commit_message: Optional[str] = None,
    split: str = "train",
) -> dict:
    """Push a DecimalAI dataset to HuggingFace Hub.

    Once pushed, the dataset is immediately loadable by Axolotl, Unsloth,
    TRL, and any tool that supports ``load_dataset()``.

    Args:
        dataset_id: The DecimalAI dataset ID.
        repo_id: HuggingFace repo in ``"org/dataset-name"`` format.
        version: Version to push (``None``/``"latest"``, ``"v3"``, or UUID).
        token: HuggingFace API token. Falls back to ``HF_TOKEN`` env var.
        private: Create a private repo (default ``True``).
        commit_message: Custom commit message.
        split: Dataset split name (default ``"train"``).

    Returns:
        Dict with ``repo_url``, ``repo_id``, ``row_count``,
        ``version_id``, and ``dataset_id``.

    Example::

        import decimalai
        decimalai.init()

        result = decimalai.push_to_hub(
            "ds_abc123", "my-org/support-agent-sft"
        )
        print(f"Pushed to {result['repo_url']}")

        # Now usable in Axolotl, Unsloth, TRL:
        # from datasets import load_dataset
        # ds = load_dataset("my-org/support-agent-sft")
    """
    from .integrations.huggingface import push_to_hub as _push

    return _push(
        dataset_id, repo_id,
        version=version,
        token=token,
        private=private,
        commit_message=commit_message,
        split=split,
    )


def load_hf_dataset(
    dataset_id: str,
    *,
    version: Optional[str] = None,
) -> "Any":
    """Load a DecimalAI dataset as a HuggingFace Dataset object.

    Returns a ``datasets.Dataset`` that plugs directly into TRL,
    Axolotl, Unsloth, or any HuggingFace-compatible trainer — no
    intermediate file needed.

    Args:
        dataset_id: The DecimalAI dataset ID.
        version: Version to load (``None``/``"latest"``, ``"v3"``, or UUID).

    Returns:
        A ``datasets.Dataset`` object.

    Example::

        import decimalai
        decimalai.init()

        ds = decimalai.load_hf_dataset("ds_abc123")
        print(ds)  # Dataset({features: ['messages'], num_rows: 500})

        # Plug into TRL
        from trl import SFTTrainer
        trainer = SFTTrainer(model=model, train_dataset=ds, ...)
    """
    from .integrations.huggingface import load_dataset as _load

    return _load(dataset_id, version=version)


# ── Re-export generic tracing API ──────────────────────────────

# Module-level client reference
import decimalai._config as _cfg  # noqa: E402

from .decorators import tool  # noqa: E402, F401

# Re-export eval adapters from new location — but wrap them so the
# user-facing API auto-fetches the client instead of leaking it as a
# required parameter.
# Before: `decimalai.push_custom_scores(trace_id, scores, source)` →
#   `TypeError: missing 1 required positional argument: 'client'`.
# Now: works exactly like that. The advanced raw-adapter form is
# still available as `decimalai.evals.adapters.push_custom_scores` for
# callers that want to pass their own client.
from .evals import adapters as _adapters  # noqa: E402
from .evals import batch_eval  # noqa: E402, F401


def push_custom_scores(
    trace_id: str,
    scores: list,
    source: str = "custom",
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Push custom evaluation scores to a DecimalAI trace.

    `client` is fetched from the SDK's initialized state if not passed.
    Most callers should just do `decimalai.init(...)` then call this
    without thinking about the client.
    """
    if client is None:
        from . import _config as _cfg
        client = _cfg._get_client()
    return _adapters.push_custom_scores(client, trace_id, source, scores)


def push_deepeval_results(
    test_results: Any,
    trace_id_field: str = "input",
    client: Optional[Any] = None,
) -> list:
    """Push DeepEval test results as scores. Auto-fetches client if not passed."""
    if client is None:
        from . import _config as _cfg
        client = _cfg._get_client()
    return _adapters.push_deepeval_results(client, test_results, trace_id_field)


def push_langsmith_scores(
    trace_id: str,
    run_scores: list,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Push LangSmith run scores. Auto-fetches client if not passed."""
    if client is None:
        from . import _config as _cfg
        client = _cfg._get_client()
    return _adapters.push_langsmith_scores(client, trace_id, run_scores)
from .dataset import Dataset  # noqa: E402, F401
from .generic import (  # noqa: E402, F401
    log_llm_call,
    log_skill_activation,
    log_skill_delivered,
    log_skill_loaded,
    log_skill_offered,
    log_tool_call,
    set_routing_id,
    start_trace,
    trace,
)

_global_client = None

def _refresh_global_client():
    """Update _global_client from config (called after init)."""
    global _global_client
    _global_client = getattr(_cfg, '_client', None)


def __getattr__(name: str):
    """Resolve top-level names that are imported lazily on first access.

    `from decimalai import SkillRouter` is the spelling most people reach for
    (and the one the published docs use), but eagerly importing
    `decimalai.skill_router` at module level would pull httpx into every
    `import decimalai`, including the many that never touch skill routing.
    Defining this module-level `__getattr__` (PEP 562) keeps the name working
    while deferring the import — and its cost — to the first attribute access.
    """
    if name == "SkillRouter":
        from .skill_router import SkillRouter
        return SkillRouter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SkillRouter",
    "__version__",
    "init",
    "flush",
    "last_send_error",
    "export_status",
    "on_export_error",
    "send",
    "eval",
    "score",
    "get_eval_breakdown",
    "register_manifest",
    "export_manifest",
    "flush_manifest_for_ci",
    "repair_preview",
    "repair_apply",
    "get_replay_prompts",
    "create_replay_batch",
    "get_replay_batch",
    "submit_replay_result",
    "pull_dataset",
    "push_to_hub",
    "load_hf_dataset",
    "trace",
    "start_trace",
    "log_llm_call",
    "log_tool_call",
    "log_skill_activation",
    "log_skill_offered",
    "log_skill_delivered",
    "log_skill_loaded",
    "set_routing_id",
    "tool",
    "push_deepeval_results",
    "push_langsmith_scores",
    "push_custom_scores",
    "batch_eval",
]


# ── Auto-init from environment variable ────────────────────────
# Setting DECIMAL_AUTO_TRACE=langchain will auto-init and install tracing.

def _auto_init_from_env() -> None:
    """Auto-initialize from environment variables if configured.

    Two paths:
    1. `DECIMAL_AUTO_TRACE=<framework>` — full framework auto-install.
    2. `DECIMAL_API_KEY` present AND `DECIMAL_AUTOINIT != "false"` — bare
       init() so users get ingest + atexit flush without a line of
       boilerplate. Opt out with `DECIMAL_AUTOINIT=false`.
    """
    auto_trace = os.environ.get("DECIMAL_AUTO_TRACE", "").strip().lower()
    api_key = os.environ.get("DECIMAL_API_KEY", "")

    if auto_trace:
        if not api_key:
            logger.warning(
                "DecimalAI auto-init: DECIMAL_AUTO_TRACE=%s set but DECIMAL_API_KEY is "
                "missing — no auto-tracing active. Set DECIMAL_API_KEY to enable.",
                auto_trace,
            )
            return
        try:
            init(
                langchain=(auto_trace == "langchain"),
                openai_agents=(auto_trace == "openai-agents"),
                adk=(auto_trace == "adk"),
                llamaindex=(auto_trace == "llamaindex"),
                otel=(auto_trace in ("otel", "autogen", "crewai")),
                # Direct/no-framework provider SDKs. "openai" is the raw SDK;
                # the OpenAI Agents framework is the distinct "openai-agents".
                openai=(auto_trace == "openai"),
                anthropic=(auto_trace == "anthropic"),
                google=(auto_trace == "google"),
            )
            logger.info("DecimalAI auto-initialized via DECIMAL_AUTO_TRACE=%s", auto_trace)
        except Exception:
            logger.warning(
                "DecimalAI auto-init from DECIMAL_AUTO_TRACE=%s failed — no auto-tracing "
                "active. Set logging to DEBUG for the full traceback.",
                auto_trace,
            )
            logger.debug("Auto-init failed", exc_info=True)
        return

    # Bare auto-init: API key present, no framework hint, opt-out not set.
    # Env-var presence is the consent signal; the explicit `=false` flag is
    # for side-effect-sensitive contexts (testing, library callers that
    # want to control init themselves).
    if api_key and os.environ.get("DECIMAL_AUTOINIT", "").strip().lower() != "false":
        try:
            init()
            logger.debug("DecimalAI auto-init from DECIMAL_API_KEY (bare mode)")
        except Exception:
            logger.warning(
                "DecimalAI bare auto-init failed — call decimalai.init() manually. "
                "Set logging to DEBUG for the traceback."
            )
            logger.debug("Bare auto-init failed", exc_info=True)


_auto_init_from_env()
