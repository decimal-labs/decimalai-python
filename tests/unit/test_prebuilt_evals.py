"""Tests for pre-built deterministic evaluators (no LLM required).

Covers all 5 evaluators from ``decimalai.evals.prebuilt``:
  - json_valid
  - contains / not_contains
  - regex_match
  - length_check
"""

import pytest
from decimalai.evals import TraceData, EvalResult
from decimalai.evals.prebuilt import (
    json_valid,
    contains,
    not_contains,
    regex_match,
    length_check,
)


def _make_trace(output: str = "", input: str = "test question") -> TraceData:
    """Helper to create a minimal TraceData for testing."""
    return TraceData(
        id="test-trace-001",
        input=input,
        output=output,
        status="success",
    )


# ── json_valid ───────────────────────────────────────────────────


class TestJsonValid:
    def test_valid_json_object(self):
        trace = _make_trace('{"key": "value", "count": 42}')
        result = json_valid(trace)
        assert result.passed is True
        assert result.score == 1.0

    def test_valid_json_array(self):
        trace = _make_trace('[1, 2, 3]')
        result = json_valid(trace)
        assert result.passed is True

    def test_valid_json_string(self):
        trace = _make_trace('"hello"')
        result = json_valid(trace)
        assert result.passed is True

    def test_invalid_json(self):
        trace = _make_trace("This is not JSON at all")
        result = json_valid(trace)
        assert result.passed is False
        assert result.score == 0.0
        assert "Invalid JSON" in result.reason

    def test_empty_output(self):
        trace = _make_trace("")
        result = json_valid(trace)
        assert result.passed is False
        assert result.score == 0.0

    def test_json_with_whitespace(self):
        trace = _make_trace('  \n  {"valid": true}  \n  ')
        result = json_valid(trace)
        assert result.passed is True


# ── contains ─────────────────────────────────────────────────────


class TestContains:
    def test_all_patterns_found(self):
        ev = contains(["source:", "http"])
        trace = _make_trace("Check source: http://example.com")
        result = ev(trace)
        assert result.passed is True
        assert result.score == 1.0

    def test_one_pattern_missing_require_all(self):
        ev = contains(["source:", "ftp://"], require_all=True)
        trace = _make_trace("Check source: http://example.com")
        result = ev(trace)
        assert result.passed is False
        assert result.score == 0.5  # 1 of 2 found

    def test_one_found_require_any(self):
        ev = contains(["source:", "ftp://"], require_all=False)
        trace = _make_trace("Check source: http://example.com")
        result = ev(trace)
        assert result.passed is True
        assert result.score == 1.0

    def test_none_found_require_any(self):
        ev = contains(["xyz", "abc"], require_all=False)
        trace = _make_trace("No matches here")
        result = ev(trace)
        assert result.passed is False
        assert result.score == 0.0

    def test_case_insensitive_default(self):
        ev = contains(["SOURCE:"])
        trace = _make_trace("check source: data")
        result = ev(trace)
        assert result.passed is True

    def test_case_sensitive(self):
        ev = contains(["SOURCE:"], case_sensitive=True)
        trace = _make_trace("check source: data")
        result = ev(trace)
        assert result.passed is False

    def test_custom_name(self):
        ev = contains(["test"], name="my_check")
        assert ev.name == "my_check"


# ── not_contains ─────────────────────────────────────────────────


class TestNotContains:
    def test_no_violations(self):
        ev = not_contains(["TODO", "FIXME", "HACK"])
        trace = _make_trace("This is a clean output with no issues")
        result = ev(trace)
        assert result.passed is True
        assert result.score == 1.0

    def test_violation_found(self):
        ev = not_contains(["TODO", "FIXME"])
        trace = _make_trace("This has a TODO that needs fixing")
        result = ev(trace)
        assert result.passed is False
        assert "TODO" in result.reason

    def test_case_insensitive_default(self):
        ev = not_contains(["todo"])
        trace = _make_trace("Fix this TODO later")
        result = ev(trace)
        assert result.passed is False

    def test_case_sensitive(self):
        ev = not_contains(["TODO"], case_sensitive=True)
        trace = _make_trace("here is a todo item")
        result = ev(trace)
        assert result.passed is True  # "todo" != "TODO"


# ── regex_match ──────────────────────────────────────────────────


class TestRegexMatch:
    def test_pattern_found(self):
        ev = regex_match(r"\d{4}-\d{2}-\d{2}")
        trace = _make_trace("The date is 2024-01-15 for reference")
        result = ev(trace)
        assert result.passed is True
        assert "2024-01-15" in result.reason

    def test_pattern_not_found(self):
        ev = regex_match(r"\d{4}-\d{2}-\d{2}")
        trace = _make_trace("No date here")
        result = ev(trace)
        assert result.passed is False
        assert result.score == 0.0

    def test_full_match_pass(self):
        ev = regex_match(r"\d+", full_match=True)
        trace = _make_trace("42")
        result = ev(trace)
        assert result.passed is True

    def test_full_match_fail(self):
        ev = regex_match(r"\d+", full_match=True)
        trace = _make_trace("42 items")
        result = ev(trace)
        assert result.passed is False

    def test_custom_name(self):
        ev = regex_match(r"\d+", name="has_number")
        assert ev.name == "has_number"

    def test_ignorecase_flag(self):
        import re
        ev = regex_match(r"error", flags=re.IGNORECASE)
        trace = _make_trace("ERROR detected")
        result = ev(trace)
        assert result.passed is True


# ── length_check ─────────────────────────────────────────────────


class TestLengthCheck:
    def test_within_bounds(self):
        ev = length_check(min_words=5, max_words=20)
        trace = _make_trace("This is a ten word sentence with some extra padding words")
        result = ev(trace)
        assert result.passed is True

    def test_too_short(self):
        ev = length_check(min_words=10)
        trace = _make_trace("Too short")
        result = ev(trace)
        assert result.passed is False
        assert "too short" in result.reason

    def test_too_long(self):
        ev = length_check(max_words=3)
        trace = _make_trace("This output has way too many words for the limit")
        result = ev(trace)
        assert result.passed is False
        assert "too long" in result.reason

    def test_char_limits(self):
        ev = length_check(min_chars=10, max_chars=50)
        trace = _make_trace("Just right length text")
        result = ev(trace)
        assert result.passed is True

    def test_char_too_short(self):
        ev = length_check(min_chars=100)
        trace = _make_trace("Short")
        result = ev(trace)
        assert result.passed is False

    def test_no_limits(self):
        ev = length_check()
        trace = _make_trace("Anything goes")
        result = ev(trace)
        assert result.passed is True
