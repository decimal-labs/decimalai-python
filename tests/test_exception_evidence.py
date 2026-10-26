"""A failed invocation retains a cause, including message-less cancellation/timeouts."""
import asyncio

import pytest

from decimalai._error_message import exception_message
from decimalai.generic import start_trace


@pytest.mark.parametrize("error,expected", [
    (TimeoutError(), "TimeoutError"),
    (asyncio.CancelledError(), "CancelledError"),
    (RuntimeError("  \n"), "RuntimeError"),
    (ValueError("existing detail"), "existing detail"),
])
def test_failed_generic_invocation_preserves_original_exception_and_cause(error, expected, monkeypatch):
    monkeypatch.setattr("decimalai.skills.discover_skills", lambda *args, **kwargs: [])
    with pytest.raises(type(error)) as raised:
        with start_trace(agent_name="error-evidence", auto_send=False) as ctx:
            raise error
    assert raised.value is error
    trace = ctx.build_trace()
    assert trace.status == "error"
    assert trace.error_message == expected


def test_broken_exception_string_does_not_replace_the_original_failure(monkeypatch):
    class BadString(Exception):
        def __str__(self):
            raise RuntimeError("broken formatter")

    monkeypatch.setattr("decimalai.skills.discover_skills", lambda *args, **kwargs: [])
    original = BadString()
    with pytest.raises(BadString) as raised:
        with start_trace(agent_name="error-evidence", auto_send=False) as ctx:
            raise original
    assert raised.value is original
    assert ctx.build_trace().error_message == "BadString"


def test_exception_evidence_remains_bounded():
    assert exception_message(ValueError("x" * 600), 500) == "x" * 500
