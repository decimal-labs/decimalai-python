"""Lock in: init(crewai=True) ACTIVATES CrewAI instrumentation, or warns.

Deep-audit finding (community tier): init(crewai=True) only installed the
OTEL exporter — but current CrewAI emits no spans to the global
TracerProvider on its own, so a crew.kickoff() produced ZERO traces with
no warning. The fix activates the OpenInference CrewAI instrumentor
against the exporter's provider when importable, and warns loudly when it
isn't (or when activation fails, e.g. an opentelemetry version conflict).

No CrewAI / no openinference install needed — the instrumentor module is
faked in sys.modules; the exporter wiring uses the real opentelemetry-sdk.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

import decimalai
from decimalai.otel import DecimalSpanExporter


@pytest.fixture(autouse=True)
def _reset_sdk(monkeypatch):
    """Known SDK config + a quiet provider-instrumentor loop.

    The provider loop asks decimalai.providers whether the openai /
    anthropic / google SDKs are importable — force "no" so these tests
    don't depend on what happens to be installed in the venv.
    """
    import decimalai._config as cfg
    import decimalai.providers as providers
    from decimalai._config import DecimalConfig

    saved_config = cfg._config
    saved_client = cfg._client

    cfg._config = DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {"manifest_id": "m1"}
    monkeypatch.setattr(providers, "_sdk_present", lambda _mod: False)
    yield
    cfg._config = saved_config
    cfg._client = saved_client


def _fake_crewai_instrumentor(monkeypatch, instrument_side_effect=None):
    """Inject a fake openinference.instrumentation.crewai into sys.modules."""
    instrumentor = MagicMock(name="CrewAIInstrumentor_instance")
    if instrument_side_effect is not None:
        instrumentor.instrument.side_effect = instrument_side_effect
    instrumentor_cls = MagicMock(return_value=instrumentor)

    mod = types.ModuleType("openinference.instrumentation.crewai")
    mod.CrewAIInstrumentor = instrumentor_cls
    for name in ("openinference", "openinference.instrumentation"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "openinference.instrumentation.crewai", mod)
    return instrumentor


def _decimal_exporters(provider):
    """Pull every DecimalSpanExporter wired onto a provider's processors."""
    active = provider._active_span_processor
    children = getattr(active, "_span_processors", ()) or ()
    return [
        proc.span_exporter
        for proc in children
        if isinstance(getattr(proc, "span_exporter", None), DecimalSpanExporter)
    ]


def test_init_crewai_activates_instrumentor_on_exporter_provider(monkeypatch):
    """init(crewai=True) must call CrewAIInstrumentor().instrument() with the
    SAME TracerProvider the DecimalAI exporter sits on — not just install
    the exporter and hope CrewAI finds it."""
    instrumentor = _fake_crewai_instrumentor(monkeypatch)

    decimalai.init(api_key="dai_sk_test", base_url="http://localhost:8000", crewai=True)

    instrumentor.instrument.assert_called_once()
    provider = instrumentor.instrument.call_args.kwargs["tracer_provider"]
    exporters = _decimal_exporters(provider)
    assert len(exporters) == 1, (
        "The instrumentor was not bound to the provider carrying the "
        f"DecimalSpanExporter (found {len(exporters)} exporters)"
    )


def test_init_crewai_warns_when_instrumentor_missing(monkeypatch, caplog):
    """No openinference-instrumentation-crewai → a loud warning naming the
    consequence (no traces) and the pip install — never a silent no-op."""
    monkeypatch.setitem(sys.modules, "openinference.instrumentation.crewai", None)

    with caplog.at_level("WARNING", logger="decimalai"):
        decimalai.init(
            api_key="dai_sk_test", base_url="http://localhost:8000", crewai=True
        )

    warning = "\n".join(
        r.getMessage() for r in caplog.records if r.levelname == "WARNING"
    )
    assert "NO CrewAI traces" in warning
    assert "pip install openinference-instrumentation-crewai" in warning


def test_init_crewai_warns_when_activation_fails(monkeypatch, caplog):
    """Activation blowing up (the known opentelemetry version-conflict
    wrinkle) warns and continues — init() must never crash on it."""
    _fake_crewai_instrumentor(
        monkeypatch, instrument_side_effect=RuntimeError("semconv mismatch")
    )

    with caplog.at_level("WARNING", logger="decimalai"):
        decimalai.init(
            api_key="dai_sk_test", base_url="http://localhost:8000", crewai=True
        )

    warning = "\n".join(
        r.getMessage() for r in caplog.records if r.levelname == "WARNING"
    )
    assert "NO CrewAI traces" in warning
    assert "version conflict" in warning
