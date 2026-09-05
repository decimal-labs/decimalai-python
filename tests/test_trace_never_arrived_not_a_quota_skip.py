"""A missing trace must never be laundered into a provider-quota SKIP.

`tests/integration/conftest.py`'s `pytest_runtest_makereport` downgrades any
failure that `is_provider_unavailable_error` matches into a SKIP, and its
marker list contains "timed out". So every poller that gives up on a trace
must raise `TraceNeverArrived` — the one type the classifier refuses — and
never a bare AssertionError whose message happens to say "Timed out".

On 2026-09-03 that exact collision reported a real dropped sub-agent trace as
"SKIPPED - provider unavailable (quota/rate-limit)" and the live matrix read
21 passed / 0 failed straight through the defect.

Hermetic on purpose: lives OUTSIDE tests/integration/ so it runs in the
default suite. Inside that directory the autouse `_require_gates` fixture
would skip it unless RUN_LIVE_LLM_TESTS=1 and a backend were up.
"""
from __future__ import annotations

import pytest

from integration import test_framework_e2e as e2e
from integration._live_helpers import TraceNeverArrived, is_provider_unavailable_error


def test_e2e_poll_for_trace_timeout_is_not_a_provider_skip(monkeypatch):
    """The e2e poller's give-up path must raise the type the classifier refuses."""
    # No backend: the probe reports "no traces, ever".
    monkeypatch.setattr(e2e, "_list_agent_traces", lambda agent_name: [])
    monkeypatch.setattr(e2e, "POLL_TIMEOUT_S", 0.05)
    monkeypatch.setattr(e2e, "POLL_INTERVAL_S", 0.01)

    with pytest.raises(AssertionError) as ei:
        e2e._poll_for_trace("agent-that-never-traced")

    assert is_provider_unavailable_error(ei.value) is False, (
        f"_poll_for_trace raised {type(ei.value).__name__}({ei.value!s:.60}...) "
        "which tests/integration/conftest.py downgrades to a provider-quota SKIP"
    )
    assert isinstance(ei.value, TraceNeverArrived)


def test_trace_never_arrived_is_never_a_provider_skip():
    """Bare, and wrapped in another exception that re-prints its message."""
    exc = TraceNeverArrived("Timed out waiting for 1 trace(s) on agent=x; last saw 0.")
    assert is_provider_unavailable_error(exc) is False

    # The usual `raise Other(f"...: {e}") from e` copies the inner message into
    # the wrapper, so the wrapper's own text also contains "timed out".
    try:
        try:
            raise exc
        except TraceNeverArrived as inner:
            raise RuntimeError(f"live cell failed: {inner}") from inner
    except RuntimeError as wrapped:
        assert is_provider_unavailable_error(wrapped) is False


def test_real_provider_timeout_still_skips():
    """The guard must not blunt the marker list it sits in front of."""
    assert is_provider_unavailable_error(TimeoutError("Read timed out.")) is True
    assert is_provider_unavailable_error(
        RuntimeError("429 RESOURCE_EXHAUSTED: rate limit exceeded")
    ) is True
