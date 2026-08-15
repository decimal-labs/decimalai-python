"""Lock in: init(autogen=True) auto-instruments AG2 agents, or warns.

Deep-audit finding (community tier): AG2 (the pip-resolved ``autogen``
distribution) emits NO OpenTelemetry spans unless
``autogen.opentelemetry.instrument_agent(agent, tracer_provider=...)`` is
called per agent — so init(autogen=True) installed an exporter that
received nothing: chat ran, zero traces, no warning. The fix hooks
``ConversableAgent.__init__`` so agents constructed after init() are
instrumented automatically (plus AG2's global LLM wrapper), and warns
loudly whenever that wiring isn't possible.

AG2 itself is faked in sys.modules — no autogen install needed.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import TracerProvider

import decimalai.autogen as da


@pytest.fixture(autouse=True)
def _fresh_hook_state(monkeypatch):
    """Each test starts with the ConversableAgent hook not yet installed."""
    monkeypatch.setattr(da, "_ag2_hook_installed", False)
    yield


def _fake_ag2(monkeypatch):
    """Inject a fake `autogen` + `autogen.opentelemetry` into sys.modules.

    Returns (ConversableAgent, instrument_agent mock, instrument_llm_wrapper
    mock). The fake ConversableAgent is a fresh class per call, so the
    __init__ hook the SDK installs never leaks across tests.
    """

    class ConversableAgent:
        def __init__(self, name="agent"):
            self.name = name

    instrument_agent = MagicMock(side_effect=lambda agent, *, tracer_provider: agent)
    instrument_llm_wrapper = MagicMock()

    autogen_mod = types.ModuleType("autogen")
    autogen_mod.ConversableAgent = ConversableAgent
    otel_mod = types.ModuleType("autogen.opentelemetry")
    otel_mod.instrument_agent = instrument_agent
    otel_mod.instrument_llm_wrapper = instrument_llm_wrapper
    autogen_mod.opentelemetry = otel_mod
    monkeypatch.setitem(sys.modules, "autogen", autogen_mod)
    monkeypatch.setitem(sys.modules, "autogen.opentelemetry", otel_mod)
    return ConversableAgent, instrument_agent, instrument_llm_wrapper


def test_agents_constructed_after_activation_are_instrumented(monkeypatch):
    """The load-bearing behavior: after activation, constructing an agent
    runs it through instrument_agent against the exporter's provider."""
    ConversableAgent, instrument_agent, instrument_llm_wrapper = _fake_ag2(monkeypatch)
    provider = TracerProvider()

    da._activate_ag2_instrumentation(provider)

    instrument_llm_wrapper.assert_called_once_with(
        tracer_provider=provider, capture_messages=True
    )
    instrument_agent.assert_not_called()

    agent = ConversableAgent(name="assistant")
    instrument_agent.assert_called_once_with(agent, tracer_provider=provider)
    # The sentinel marks the instance so it is never double-instrumented.
    assert agent._decimalai_ag2_instrumented is True


def test_activation_is_idempotent(monkeypatch):
    """Running activation twice (init() called again) must not stack a
    second __init__ hook — one agent, one instrument_agent call."""
    ConversableAgent, instrument_agent, instrument_llm_wrapper = _fake_ag2(monkeypatch)
    provider = TracerProvider()

    da._activate_ag2_instrumentation(provider)
    da._activate_ag2_instrumentation(provider)

    ConversableAgent(name="solo")
    assert instrument_agent.call_count == 1
    assert instrument_llm_wrapper.call_count == 1


def test_instrument_agent_failure_never_breaks_construction(monkeypatch, caplog):
    """A blowing-up instrument_agent warns (with the manual one-liner) and
    still returns a usable agent — tracing setup must not take the app down."""
    ConversableAgent, instrument_agent, _ = _fake_ag2(monkeypatch)
    instrument_agent.side_effect = RuntimeError("wrapping failed")
    da._activate_ag2_instrumentation(TracerProvider())

    with caplog.at_level("WARNING", logger="decimalai.autogen"):
        agent = ConversableAgent(name="fragile")

    assert agent.name == "fragile"
    warning = "\n".join(
        r.getMessage() for r in caplog.records if r.levelname == "WARNING"
    )
    assert "instrument_agent" in warning


def test_warns_loudly_when_ag2_has_no_otel_module(monkeypatch, caplog):
    """An `autogen` distribution without autogen.opentelemetry (old AG2 /
    pyautogen) → loud warning: zero traces otherwise."""
    autogen_mod = types.ModuleType("autogen")  # no ConversableAgent/otel

    class ConversableAgent:
        pass

    autogen_mod.ConversableAgent = ConversableAgent
    monkeypatch.setitem(sys.modules, "autogen", autogen_mod)
    monkeypatch.setitem(sys.modules, "autogen.opentelemetry", None)

    import importlib.util

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: object() if name == "autogen" else None,
    )

    with caplog.at_level("WARNING", logger="decimalai.autogen"):
        da._activate_ag2_instrumentation(TracerProvider())

    warning = "\n".join(
        r.getMessage() for r in caplog.records if r.levelname == "WARNING"
    )
    assert "NO traces will be captured" in warning
    assert "autogen.opentelemetry" in warning


def test_warns_loudly_when_no_autogen_installed(monkeypatch, caplog):
    """No AutoGen distribution importable at all → loud warning with the
    install options, not a silent exporter-only init."""
    monkeypatch.setitem(sys.modules, "autogen", None)

    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None)

    with caplog.at_level("WARNING", logger="decimalai.autogen"):
        da._activate_ag2_instrumentation(TracerProvider())

    warning = "\n".join(
        r.getMessage() for r in caplog.records if r.levelname == "WARNING"
    )
    assert "NO traces will be captured" in warning
    assert "pip install ag2" in warning


def test_module_docstring_no_longer_claims_native_spans():
    """The old docstring claimed AutoGen/AG2 'emit standard OpenTelemetry
    spans' natively — the exact fiction behind the silent zero-trace P0."""
    assert "both emit standard" not in (da.__doc__ or "")
    assert "instrument_agent" in (da.__doc__ or "")
