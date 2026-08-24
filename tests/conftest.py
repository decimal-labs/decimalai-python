"""Top-level test conftest for the DecimalAI Python SDK.

Four cross-cutting concerns are handled here so individual test modules
don't have to:

1. **Init-time verify probe is bypassed** via the ``DECIMALAI_SKIP_VERIFY``
   env var, set at ``pytest_configure``. Unit tests don't have a real
   backend (or use fake keys against a real one); the probe would either
   DNS-fail or 401, neither of which is what the unit tests are exercising.
   Integration tests under ``tests/integration/`` can re-enable verify by
   unsetting this var.

2. **Module-instance stability for ``decimalai`` + submodules.**
   ``test_auto_init_logging.py`` legitimately reimports ``decimalai`` (and
   sometimes ``decimalai._config``) to test the auto-init-on-import path.
   But other test modules import names like ``from decimalai._config
   import _is_manifest_only`` at module load — those bindings stay attached
   to the *original* module instance. After a reimport, ``init()`` mutates
   the NEW ``decimalai._config`` global while the test's bound name still
   reads the OLD one, and the assertion fails with no obvious cause.

   The ``_restore_decimalai_modules`` autouse fixture below snapshots the
   live ``decimalai*`` entries in ``sys.modules`` before each test and
   restores them after. A test that reimports the package gets to see its
   reimported view during the test body; the next test sees the original
   instances again. The fix is invisible to tests — no per-test changes.

3. **LangChain global instrumentation is torn down after each test.**
   ``decimalai.langchain.instrument()`` publishes a handler into a module
   ContextVar and hands that var to langchain-core's
   ``register_configure_hook``. Both are process-global and the hook list has
   no unregister, so the ContextVar is the only half a test can put back —
   and the adapter offers no uninstall to do it with. Any test that installs
   (directly, or indirectly through a ``DECIMAL_AUTO_TRACE=langchain``
   module-load auto-init) therefore leaves a live handler behind for the rest
   of the session, and every LATER test that runs a real chain with its own
   per-call handler is traced TWICE — it asserts one trace and gets two.
   The ``_clear_langchain_global_handler`` autouse fixture below is that
   missing teardown.
"""
from __future__ import annotations

import os
import sys
import weakref

import pytest


def pytest_configure(config):
    """Set the skip-verify env var as early as possible.

    Runs before any test module is imported, so even tests that import
    decimalai at module-level get the right behavior.
    """
    os.environ.setdefault("DECIMALAI_SKIP_VERIFY", "1")


@pytest.fixture(autouse=True)
def _restore_decimalai_modules():
    """Snapshot ``decimalai*`` modules in sys.modules; restore after test.

    Prevents test-isolation breakage when one test deletes
    ``sys.modules["decimalai"]`` (or ``decimalai._config``) and re-imports
    — subsequent tests with module-level ``from decimalai._config import X``
    bindings would otherwise read stale globals from an orphaned module
    instance. See module docstring for the failure mode this catches.
    """
    snapshot = {
        name: mod
        for name, mod in sys.modules.items()
        if name == "decimalai" or name.startswith("decimalai.")
    }
    try:
        yield
    finally:
        # Drop any modules the test added that weren't in the snapshot
        # (e.g. test imported decimalai.langchain for the first time —
        # safe to keep, but consistency matters more than cache).
        current = [
            name
            for name in list(sys.modules)
            if name == "decimalai" or name.startswith("decimalai.")
        ]
        for name in current:
            if name not in snapshot:
                # Newly imported during the test — leave it alone; not
                # an isolation hazard, and re-importing it costs cycles.
                continue
        # Restore the original module instances so module-level bindings
        # in OTHER test files keep pointing at live globals.
        for name, mod in snapshot.items():
            sys.modules[name] = mod


@pytest.fixture(autouse=True)
def _clear_langchain_global_handler():
    """Drop the process-global LangChain handler an install leaves behind.

    See concern 3 in the module docstring for the double-tracing failure this
    prevents. Only reaches for the adapter when a test already imported it, so
    it never drags langchain-core into a run that didn't ask for it.
    """
    try:
        yield
    finally:
        lc_mod = sys.modules.get("decimalai.langchain")
        if lc_mod is not None:
            lc_mod._decimal_callback_var.set(None)


# ── 4. Span exports are drained between tests ──────────────────────────────────
#
# The SDK exports through a `BatchSpanProcessor`, whose whole job is to hand spans
# to a worker thread and return. Nothing tied that worker to a test boundary, so a
# span opened in one test could be exported while the NEXT test was running — and
# land in that test's mock.
#
# That is not hypothetical. Three intermittent CI failures on 2026-08-22..23 had
# this shape, each in a different file, each passing on rerun and unreproducible
# locally:
#
#   test_skill_rail_per_run.py    TestAnthropicClaimsOnlyWhatItInjected
#   test_pydantic_ai_run_scope.py TestRunScopeInstall
#   test_manifest_only_mode.py    assert captured["agent_name"] == "support-agent"
#                                 AssertionError: 'doomed' == 'support-agent'
#
# `doomed` is the agent name from test_llamaindex_run_identity.py:431 — a different
# FILE. Demonstrated directly: three spans opened in one function were exported by
# the batch worker while the next function was running, every time.
#
# Note what this is NOT. `_synchronous_sender` below already makes the *send*
# inline, and `agent_run` already resets its ContextVar in a `finally` — both were
# checked and both are correct. The gap is one layer up: the send is only reached
# when the batch processor decides to export, and that decision was on its own
# timer. Fixing the send half and leaving the export half async is why the flake
# survived a fixture that looks like it should have caught it.
#
# Draining after every test closes it: a test's spans are exported before the next
# one starts, so nothing can arrive late. `force_flush` is exactly this operation.
# Providers built BY tests are covered too — six test files construct their own
# `TracerProvider`, three of them the files that flaked — so `__init__` is wrapped
# to record every instance weakly rather than relying on a hand-maintained list a
# new test would silently fall out of.
_LIVE_PROVIDERS: "weakref.WeakSet" = weakref.WeakSet()


def pytest_sessionstart(session):
    """Record every TracerProvider built during the session, weakly."""
    try:
        from opentelemetry.sdk.trace import TracerProvider
    except ImportError:  # pragma: no cover - OTel is a core dep
        return
    if getattr(TracerProvider, "_decimalai_tracked", False):
        return
    original_init = TracerProvider.__init__

    def _tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            _LIVE_PROVIDERS.add(self)
        except TypeError:  # pragma: no cover - unhashable provider
            pass

    TracerProvider.__init__ = _tracking_init
    TracerProvider._decimalai_tracked = True


@pytest.fixture(autouse=True)
def _drain_span_exports():
    """Flush every live TracerProvider after each test, so no export lands late."""
    yield
    providers = list(_LIVE_PROVIDERS)
    try:
        from opentelemetry import trace as trace_api

        providers.append(trace_api.get_tracer_provider())
    except Exception:  # pragma: no cover - no OTel, nothing to drain
        pass
    try:
        from decimalai import providers as _dp

        if _dp._last_provider is not None:
            providers.append(_dp._last_provider)
    except Exception:  # pragma: no cover - SDK not importable here
        pass
    for provider in providers:
        flush = getattr(provider, "force_flush", None)
        if not callable(flush):
            continue  # ProxyTracerProvider and friends have nothing to drain
        try:
            # Short on purpose: this runs after EVERY test, and a provider that
            # cannot drain in a second is a finding, not something to wait out.
            flush(timeout_millis=1000)
        except Exception:  # noqa: BLE001 - a shut-down provider raises; not this
            pass            # fixture's business, and never a reason to fail a test


@pytest.fixture(autouse=True)
def _synchronous_sender(request, monkeypatch):
    """Run the background trace-sender INLINE for unit tests.

    The SDK ships traces on a background thread (``_config._sender.submit``); the
    handler tests then wait on it with a 5s ``flush()`` timeout before asserting
    the trace reached the (mock) client. Under heavy CPU load — e.g. right after
    the live suite has hammered the machine — that worker thread can be starved
    and miss the window, so ``ingest_trace`` hasn't been called when the assert
    runs: an intermittent "red only when the box is busy" flake. (Unit tests do
    NOT hit a real backend; the client is always mocked or HTTP-stubbed — so the
    only thing async dispatch buys here is that flake.)

    Patching the *global sender instance's* submit to run inline removes the
    thread and the timeout race, so unit tests are deterministic regardless of
    load. We patch the instance — not ``BackgroundSender`` itself — so the
    sender's own tests (which build their own ``BackgroundSender()`` to assert
    async behavior) and ``test_silent_drop_visibility`` (drives ``flush()`` /
    ``_pending`` directly) are untouched. Live/integration tests exercise the
    REAL async sender against a real backend, so they opt out.
    """
    if request.node.get_closest_marker("integration") or request.node.get_closest_marker("live_llm"):
        yield
        return

    from decimalai import _config

    sender = _config._sender

    def _sync_submit(fn, *args, **kwargs):
        # Mirror BackgroundSender.submit's bookkeeping, minus the executor: run
        # the send inline, record the outcome, and never propagate — the async
        # path records the failure inside the worker thread and the submit caller
        # never sees it, so neither should we.
        try:
            result = fn(*args, **kwargs)
            sender._record_success()
            return result
        except Exception as exc:  # noqa: BLE001 — match async: record + swallow
            sender._record_failure(exc, _config._extract_trace_id(args))
            return None

    monkeypatch.setattr(sender, "submit", _sync_submit)
    yield

