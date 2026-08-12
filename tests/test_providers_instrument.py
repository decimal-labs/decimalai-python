"""Unit tests for direct provider-SDK tracing (decimalai.providers).

No OpenInference instrumentors, no provider SDKs, no network. The three seams
the module reaches through are patched so the tri-state flag logic can be
asserted in isolation:

* ``_sdk_present(module)``    — is the provider SDK importable?
* ``_load_instrumentor(spec)``— import the OpenInference instrumentor class.
* ``_ensure_pipeline(...)``   — attach the DecimalAI exporter to a TracerProvider.

Patching these lets us prove exactly which providers get ``.instrument()``-ed
under each flag combination without any of those packages installed. One class
(``TestEnsurePipeline``) exercises the *real* ``_ensure_pipeline`` against a
caller-owned ``TracerProvider`` — the pure branch that touches no OTEL global
state (opentelemetry-sdk is a core dep, so it's always importable here).

The ``init()`` wiring (``decimalai.init(openai=True)`` → ``instrument(...)``) is
covered in ``TestInitWiring`` by spying on the dispatched call.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

import decimalai.providers as providers
from decimalai.providers import instrument


# Stand-in for a TracerProvider; identity is all we assert on.
SENTINEL_PROVIDER = object()


@pytest.fixture(autouse=True)
def _reset_providers_state():
    """Clear the process-global instrument/pipeline caches around each test."""
    providers._instrumented.clear()
    providers._pipeline_provider = None
    yield
    providers._instrumented.clear()
    providers._pipeline_provider = None


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Stub ``_ensure_pipeline`` so no real OTEL provider is built.

    Returns the patched MagicMock (records ``(agent_name, tracer_provider)``
    calls). Hands back the explicit provider when one is passed, else the
    module-level ``SENTINEL_PROVIDER`` — mirroring the real function's contract.
    """
    def _ensure(agent_name, tracer_provider=None):
        return tracer_provider if tracer_provider is not None else SENTINEL_PROVIDER

    m = MagicMock(side_effect=_ensure)
    monkeypatch.setattr(providers, "_ensure_pipeline", m)
    return m


@pytest.fixture
def loaded(monkeypatch):
    """Patch ``_load_instrumentor`` to return a per-provider instrumentor mock.

    Returns ``{provider_name: instrumentor_class_mock}``. Each class mock's
    ``return_value`` is the instance whose ``.instrument(**kwargs)`` we assert
    on. Every provider's instrumentor 'exists' by default; a test can
    ``loaded.pop(name)`` to simulate a missing OpenInference package.
    """
    classes = {name: MagicMock(name=f"{name}Instrumentor") for name in providers._PROVIDERS}
    module_to_name = {
        spec.instrumentor_module: name for name, spec in providers._PROVIDERS.items()
    }

    def _load(spec):
        return classes.get(module_to_name[spec.instrumentor_module])

    monkeypatch.setattr(providers, "_load_instrumentor", _load)
    return classes


def _all_sdks_present(monkeypatch):
    monkeypatch.setattr(providers, "_sdk_present", lambda module: True)


# ── Forced-on (explicit True) ─────────────────────────────────────────

class TestForceOn:
    def test_single_provider_forced(self, fake_pipeline, loaded, monkeypatch):
        _all_sdks_present(monkeypatch)
        result = instrument(openai=True)

        assert result is SENTINEL_PROVIDER
        loaded["openai"].return_value.instrument.assert_called_once_with(
            tracer_provider=SENTINEL_PROVIDER
        )
        loaded["anthropic"].return_value.instrument.assert_not_called()
        loaded["google"].return_value.instrument.assert_not_called()
        assert providers._instrumented == {"openai"}

    def test_explicit_does_not_enable_present_siblings(self, fake_pipeline, loaded, monkeypatch):
        """``instrument(openai=True)`` must NOT auto-enable the other SDKs even
        when they're installed — auto only applies to the bare call."""
        _all_sdks_present(monkeypatch)
        instrument(openai=True)

        loaded["openai"].return_value.instrument.assert_called_once()
        loaded["anthropic"].return_value.instrument.assert_not_called()
        loaded["google"].return_value.instrument.assert_not_called()

    def test_agent_name_flows_to_pipeline(self, fake_pipeline, loaded, monkeypatch):
        _all_sdks_present(monkeypatch)
        instrument(anthropic=True, agent_name="my-agent")
        fake_pipeline.assert_called_once_with("my-agent", None)


# ── Auto (bare instrument(), all flags None) ──────────────────────────

class TestAuto:
    def test_bare_instruments_only_present_sdks(self, fake_pipeline, loaded, monkeypatch):
        present = {"openai", "google.genai"}  # anthropic SDK absent
        monkeypatch.setattr(providers, "_sdk_present", lambda module: module in present)

        instrument()

        loaded["openai"].return_value.instrument.assert_called_once()
        loaded["google"].return_value.instrument.assert_called_once()
        loaded["anthropic"].return_value.instrument.assert_not_called()
        assert providers._instrumented == {"openai", "google"}

    def test_bare_with_nothing_present_is_noop(self, fake_pipeline, loaded, monkeypatch):
        monkeypatch.setattr(providers, "_sdk_present", lambda module: False)

        result = instrument()

        assert result is None  # nothing requested → no pipeline built
        fake_pipeline.assert_not_called()
        assert providers._instrumented == set()


# ── Skip (False, or None alongside an explicit sibling) ───────────────

class TestSkip:
    def test_false_flag_skips(self, fake_pipeline, loaded, monkeypatch):
        _all_sdks_present(monkeypatch)
        instrument(openai=True, anthropic=False)
        loaded["openai"].return_value.instrument.assert_called_once()
        loaded["anthropic"].return_value.instrument.assert_not_called()

    def test_none_sibling_with_explicit_is_skipped(self, fake_pipeline, loaded, monkeypatch):
        """openai=True with anthropic/google left None → auto_all False → the
        None siblings are skipped, not auto-enabled."""
        _all_sdks_present(monkeypatch)
        instrument(openai=True)
        loaded["anthropic"].return_value.instrument.assert_not_called()
        loaded["google"].return_value.instrument.assert_not_called()


# ── Missing OpenInference instrumentor → soft skip ────────────────────

class TestMissingInstrumentor:
    def test_forced_missing_warns_without_raising(self, fake_pipeline, loaded, monkeypatch, caplog):
        _all_sdks_present(monkeypatch)
        loaded.pop("openai")  # its instrumentor import returns None

        with caplog.at_level(logging.WARNING, logger="decimalai.providers"):
            result = instrument(openai=True)

        assert result is SENTINEL_PROVIDER  # pipeline built; no exception
        assert any(
            "pip install openinference-instrumentation-openai" in r.getMessage()
            for r in caplog.records
        )
        assert "openai" not in providers._instrumented  # never actually instrumented

    def test_auto_missing_is_info_not_warning(self, fake_pipeline, loaded, monkeypatch, caplog):
        """In auto mode a missing instrumentor is INFO (it's a best-effort
        attempt), not a WARNING (which is reserved for an explicit request)."""
        monkeypatch.setattr(providers, "_sdk_present", lambda module: module == "openai")
        loaded.pop("openai")

        with caplog.at_level(logging.INFO, logger="decimalai.providers"):
            instrument()

        msgs = [r.getMessage() for r in caplog.records]
        assert any("pip install openinference-instrumentation-openai" in m for m in msgs)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# ── Idempotency + additive instrumentation (global path) ──────────────

class TestIdempotency:
    def test_second_identical_call_is_noop(self, fake_pipeline, loaded, monkeypatch):
        _all_sdks_present(monkeypatch)
        instrument(openai=True)
        instrument(openai=True)
        loaded["openai"].return_value.instrument.assert_called_once()
        assert providers._instrumented == {"openai"}

    def test_later_call_adds_a_new_provider(self, fake_pipeline, loaded, monkeypatch):
        _all_sdks_present(monkeypatch)
        instrument(openai=True)
        instrument(anthropic=True)
        loaded["openai"].return_value.instrument.assert_called_once()
        loaded["anthropic"].return_value.instrument.assert_called_once()
        assert providers._instrumented == {"openai", "anthropic"}


# ── Resilience: one provider failing doesn't break the others ─────────

class TestInstrumentFailure:
    def test_instrumentor_exception_is_swallowed(self, fake_pipeline, loaded, monkeypatch, caplog):
        _all_sdks_present(monkeypatch)
        loaded["openai"].return_value.instrument.side_effect = RuntimeError("boom")

        with caplog.at_level(logging.WARNING, logger="decimalai.providers"):
            result = instrument(openai=True, anthropic=True)

        assert result is SENTINEL_PROVIDER  # no raise
        loaded["anthropic"].return_value.instrument.assert_called_once()
        assert "openai" not in providers._instrumented  # failed → retryable
        assert "anthropic" in providers._instrumented
        assert any("failed to instrument" in r.getMessage() for r in caplog.records)


# ── OTEL SDK absent → graceful None ───────────────────────────────────

class TestOtelMissing:
    def test_pipeline_import_error_returns_none(self, loaded, monkeypatch, caplog):
        _all_sdks_present(monkeypatch)

        def boom(agent_name, tracer_provider=None):
            raise ImportError("opentelemetry not installed")

        monkeypatch.setattr(providers, "_ensure_pipeline", MagicMock(side_effect=boom))

        with caplog.at_level(logging.WARNING, logger="decimalai.providers"):
            result = instrument(openai=True)

        assert result is None
        assert any("pip install decimalai" in r.getMessage() for r in caplog.records)
        loaded["openai"].return_value.instrument.assert_not_called()


# ── Explicit tracer_provider escape hatch ─────────────────────────────

class TestExplicitProviderEscapeHatch:
    def test_uses_passed_provider_and_bypasses_idempotency(self, fake_pipeline, loaded, monkeypatch):
        _all_sdks_present(monkeypatch)
        custom = object()

        instrument(openai=True, tracer_provider=custom)
        instrument(openai=True, tracer_provider=custom)

        # Instrumented BOTH times: the global idempotency guard is bypassed when
        # the caller owns the provider's lifecycle.
        assert loaded["openai"].return_value.instrument.call_count == 2
        loaded["openai"].return_value.instrument.assert_called_with(tracer_provider=custom)
        assert "openai" not in providers._instrumented  # global set untouched
        assert providers._pipeline_provider is None      # global cache untouched

    def test_pipeline_receives_explicit_provider(self, fake_pipeline, loaded, monkeypatch):
        _all_sdks_present(monkeypatch)
        custom = object()
        instrument(google=True, tracer_provider=custom)
        fake_pipeline.assert_called_once_with(None, custom)


# ── Real _ensure_pipeline, caller-owned provider (no OTEL globals) ────

class TestEnsurePipeline:
    def test_explicit_provider_gets_processor_and_no_global_mutation(self):
        from opentelemetry.sdk.trace import TracerProvider

        tp = TracerProvider()
        before = len(getattr(tp, "_active_span_processor")._span_processors)

        out = providers._ensure_pipeline("agent-x", tracer_provider=tp)

        assert out is tp
        after = len(getattr(tp, "_active_span_processor")._span_processors)
        assert after == before + 1  # our SimpleSpanProcessor was attached
        assert providers._pipeline_provider is None  # global path not taken


# ── init() → providers.instrument wiring ──────────────────────────────

class TestInitWiring:
    def test_init_openai_dispatches(self, monkeypatch):
        import decimalai
        spy = MagicMock()
        monkeypatch.setattr("decimalai.providers.instrument", spy)

        decimalai.init(openai=True, enabled=False)

        spy.assert_called_once_with(
            openai=True, anthropic=False, google=False, agent_name=None,
        )

    def test_init_multiple_providers_with_agent_name(self, monkeypatch):
        import decimalai
        spy = MagicMock()
        monkeypatch.setattr("decimalai.providers.instrument", spy)

        decimalai.init(anthropic=True, google=True, agent_name="svc", enabled=False)

        spy.assert_called_once_with(
            openai=False, anthropic=True, google=True, agent_name="svc",
        )

    def test_init_without_provider_flag_does_not_dispatch(self, monkeypatch):
        import decimalai
        spy = MagicMock()
        monkeypatch.setattr("decimalai.providers.instrument", spy)

        decimalai.init(enabled=False)

        spy.assert_not_called()


# ── DECIMAL_AUTO_TRACE env one-liner → init() provider flags ──────────

class TestAutoTraceEnv:
    @pytest.mark.parametrize("value", ["openai", "anthropic", "google"])
    def test_provider_value_maps_to_init_flag(self, monkeypatch, value):
        import decimalai
        spy = MagicMock()
        monkeypatch.setattr(decimalai, "init", spy)
        monkeypatch.setenv("DECIMAL_API_KEY", "dai_sk_test")
        monkeypatch.setenv("DECIMAL_AUTO_TRACE", value)

        decimalai._auto_init_from_env()

        spy.assert_called_once()
        kwargs = spy.call_args.kwargs
        assert kwargs[value] is True
        for other in ("openai", "anthropic", "google"):
            if other != value:
                assert kwargs[other] is False

    def test_openai_agents_does_not_enable_raw_openai(self, monkeypatch):
        """The raw-SDK flag ``openai`` must stay distinct from the OpenAI Agents
        framework value ``openai-agents`` — they're different code paths."""
        import decimalai
        spy = MagicMock()
        monkeypatch.setattr(decimalai, "init", spy)
        monkeypatch.setenv("DECIMAL_API_KEY", "dai_sk_test")
        monkeypatch.setenv("DECIMAL_AUTO_TRACE", "openai-agents")

        decimalai._auto_init_from_env()

        kwargs = spy.call_args.kwargs
        assert kwargs["openai_agents"] is True
        assert kwargs["openai"] is False
