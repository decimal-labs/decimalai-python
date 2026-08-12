"""Regression for the dead DeepEval `trace-` prefix branch in
``decimalai.evals.adapters._extract_trace_id``.

DecimalAI trace ids are uuid4 — never `trace-` prefixed — so the old
last-resort branch that returned ``input_val`` when it started with
``"trace-"`` was unreachable for real data and could mis-extract a user's
input string as a trace id. These tests pin: (1) the real extraction paths
(``trace_id`` attribute, ``additional_metadata``) still work, and (2) a
``trace-``-prefixed input is NO LONGER treated as a trace id.
"""

from types import SimpleNamespace

from decimalai.evals.adapters import _extract_trace_id


def test_extracts_from_trace_id_attribute():
    tr = SimpleNamespace(trace_id="3f2c9a10-0000-4000-8000-000000000001")
    assert _extract_trace_id(tr) == "3f2c9a10-0000-4000-8000-000000000001"


def test_extracts_from_additional_metadata():
    tr = SimpleNamespace(
        trace_id=None,
        additional_metadata={"decimal_trace_id": "uuid-from-metadata"},
    )
    assert _extract_trace_id(tr) == "uuid-from-metadata"


def test_trace_prefixed_input_is_not_treated_as_trace_id():
    """The dead branch is gone: an input string starting with 'trace-' must
    NOT be returned as a trace id (it's user content, not a uuid4)."""
    tr = SimpleNamespace(
        trace_id=None,
        additional_metadata={},
        input="trace-this-request-please",
    )
    assert _extract_trace_id(tr) is None


def test_returns_none_when_no_trace_id_present():
    tr = SimpleNamespace(trace_id=None, additional_metadata={}, input="hello")
    assert _extract_trace_id(tr) is None
