"""Lock in: SDK rejects oversized agent_name at client-time.

Without this check, a 500-char agent_name passes SDK validation, is sent
to the backend, gets rejected by the VARCHAR(255) constraint, and the
trace is lost. The SDK raises ValueError client-side BEFORE any backend
round-trip instead.
"""

import pytest


def test_oversized_agent_name_raises_at_sdk_time():
    """500-char agent_name → ValueError before any HTTP call."""
    from decimalai.generic import TraceContext

    with pytest.raises(ValueError) as exc:
        TraceContext(agent_name="x" * 500)

    assert "255" in str(exc.value)
    assert "got 500" in str(exc.value)


def test_max_length_agent_name_accepted():
    """Exactly 255 chars should be accepted (DB column boundary)."""
    from decimalai.generic import TraceContext

    # No exception — boundary value works
    ctx = TraceContext(agent_name="x" * 255)
    assert ctx.agent_name == "x" * 255


def test_none_agent_name_still_accepted():
    """The None default still works (no validation when omitted)."""
    from decimalai.generic import TraceContext

    ctx = TraceContext(agent_name=None)
    assert ctx.agent_name is None


def test_unicode_agent_name_works():
    """Unicode characters under the 255-char limit pass validation."""
    from decimalai.generic import TraceContext

    # Multi-byte unicode — count by Python str length, not byte length
    ctx = TraceContext(agent_name="支持-中文-🎯-emoji")
    assert "中文" in ctx.agent_name


def test_start_trace_propagates_validation_error():
    """The validation applies through start_trace() too."""
    import decimalai

    with pytest.raises(ValueError):
        with decimalai.start_trace(agent_name="x" * 1000) as tr:
            pass
