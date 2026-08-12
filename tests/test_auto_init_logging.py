"""Test that DECIMAL_AUTO_TRACE failures surface at WARNING (not DEBUG).

Regression coverage against a whole class of errors collapsing into
silence. Previously `_auto_init_from_env()` swallowed both the
missing-API-key branch and the init() exception at logger.debug,
leaving opt-in users with zero signal when auto-tracing silently
no-op'd. Both now log at WARNING; the exception path additionally
keeps a debug-level traceback for diagnostics.
"""

from __future__ import annotations

import importlib
import logging
import sys

import pytest


def _reimport_decimalai(monkeypatch, env: dict[str, str]) -> None:
    """Reload decimalai with the given env so module-load auto-init runs."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    if "decimalai" in sys.modules:
        del sys.modules["decimalai"]
    importlib.import_module("decimalai")


def test_auto_trace_without_api_key_logs_warning(monkeypatch, caplog):
    monkeypatch.delenv("DECIMAL_API_KEY", raising=False)
    with caplog.at_level(logging.WARNING, logger="decimalai"):
        _reimport_decimalai(monkeypatch, {"DECIMAL_AUTO_TRACE": "langchain"})
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("DECIMAL_API_KEY" in r.getMessage() for r in warnings), (
        "Expected WARNING about missing DECIMAL_API_KEY when DECIMAL_AUTO_TRACE is set; "
        f"got: {[r.getMessage() for r in caplog.records]}"
    )


def test_auto_trace_init_failure_logs_warning_plus_debug(monkeypatch, caplog):
    monkeypatch.setenv("DECIMAL_API_KEY", "test-key-not-real")
    monkeypatch.setenv("DECIMAL_AUTO_TRACE", "langchain")

    if "decimalai" in sys.modules:
        del sys.modules["decimalai"]

    import decimalai as _d

    def _boom(**_kwargs):
        raise RuntimeError("simulated init failure")

    monkeypatch.setattr(_d, "init", _boom)

    with caplog.at_level(logging.DEBUG, logger="decimalai"):
        _d._auto_init_from_env()

    msgs = [(r.levelno, r.getMessage()) for r in caplog.records]
    warn_msgs = [m for lvl, m in msgs if lvl >= logging.WARNING]
    debug_msgs = [m for lvl, m in msgs if lvl == logging.DEBUG]

    assert any("auto-init" in m.lower() and "failed" in m.lower() for m in warn_msgs), (
        f"Expected a WARNING about auto-init failure; got warn={warn_msgs} debug={debug_msgs}"
    )
    assert any("Auto-init failed" in m for m in debug_msgs), (
        f"Expected a DEBUG line carrying the traceback; got debug={debug_msgs}"
    )


def test_auto_trace_unset_no_api_key_is_silent(monkeypatch, caplog):
    monkeypatch.delenv("DECIMAL_AUTO_TRACE", raising=False)
    monkeypatch.delenv("DECIMAL_API_KEY", raising=False)
    if "decimalai" in sys.modules:
        del sys.modules["decimalai"]
    with caplog.at_level(logging.WARNING, logger="decimalai"):
        importlib.import_module("decimalai")
    warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warns, (
        f"Auto-init should be a no-op when DECIMAL_AUTO_TRACE is unset; got warnings: "
        f"{[r.getMessage() for r in warns]}"
    )


def test_bare_auto_init_when_api_key_set(monkeypatch, caplog):
    """DECIMAL_API_KEY alone triggers bare init() (no framework)."""
    monkeypatch.delenv("DECIMAL_AUTO_TRACE", raising=False)
    monkeypatch.delenv("DECIMAL_AUTOINIT", raising=False)
    monkeypatch.setenv("DECIMAL_API_KEY", "test-bare-init-key")
    if "decimalai" in sys.modules:
        del sys.modules["decimalai"]
    with caplog.at_level(logging.INFO, logger="decimalai"):
        importlib.import_module("decimalai")
    import decimalai._config as _cfg
    assert _cfg._client is not None, "Bare auto-init should have created a client"
    assert _cfg._config is not None and _cfg._config.api_key == "test-bare-init-key"


def test_bare_auto_init_opt_out_with_false(monkeypatch):
    """DECIMAL_AUTOINIT=false suppresses bare auto-init even with API key."""
    monkeypatch.delenv("DECIMAL_AUTO_TRACE", raising=False)
    monkeypatch.setenv("DECIMAL_API_KEY", "test-key-should-be-ignored")
    monkeypatch.setenv("DECIMAL_AUTOINIT", "false")
    # Drop both the package and its _config submodule so prior-test state
    # (a populated _cfg._client) doesn't leak in.
    for mod in [m for m in list(sys.modules) if m == "decimalai" or m.startswith("decimalai.")]:
        del sys.modules[mod]
    importlib.import_module("decimalai")
    import decimalai._config as _cfg
    assert _cfg._client is None, "Opt-out should leave _client unset"
