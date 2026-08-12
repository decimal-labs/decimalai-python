"""Top-level test conftest for the DecimalAI Python SDK.

Two cross-cutting concerns are handled here so individual test modules
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
"""
from __future__ import annotations

import os
import sys

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

