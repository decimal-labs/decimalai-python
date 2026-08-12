"""Tests for the AutoGen / AG2 adapter (decimalai.autogen).

AutoGen emits standard OpenTelemetry GenAI spans, so the adapter has no
dedicated handler — ``install()`` wires the manifest-capable
``decimalai.otel.DecimalSpanExporter`` onto a real ``TracerProvider`` via a
``BatchSpanProcessor``.

These tests use the REAL opentelemetry-sdk (a hard dependency of this
package, so always present in PR CI) — no AutoGen install needed. They
assert the actual observable wiring: install() registers a span
processor whose exporter is our ``DecimalSpanExporter``, not that a
delegate function was merely invoked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import TracerProvider

from decimalai.autogen import install
from decimalai.otel import DecimalSpanExporter


@pytest.fixture(autouse=True)
def _reset_sdk():
    """Set a known SDK config so install() / the exporter run cleanly."""
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    saved_config = cfg._config
    saved_client = cfg._client

    cfg._config = DecimalConfig(
        api_key="dai_sk_test",
        base_url="http://localhost:8000",
        enabled=True,
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {"manifest_id": "m1"}
    yield
    cfg._config = saved_config
    cfg._client = saved_client


def _decimal_exporters(provider: TracerProvider):
    """Pull every DecimalSpanExporter wired onto a provider's processors."""
    active = provider._active_span_processor
    children = getattr(active, "_span_processors", ()) or ()
    found = []
    for proc in children:
        exporter = getattr(proc, "span_exporter", None)
        if isinstance(exporter, DecimalSpanExporter):
            found.append(exporter)
    return found


class TestAutogenInstall:
    def test_install_returns_tracer_provider(self):
        provider = install(agent_name="my-autogen-agent")
        assert isinstance(provider, TracerProvider)

    def test_install_registers_decimal_exporter(self):
        """install() must wire a DecimalSpanExporter onto the provider —
        not a no-op delegation."""
        provider = install(agent_name="my-autogen-agent")

        exporters = _decimal_exporters(provider)
        assert len(exporters) == 1, (
            "install() did not register exactly one DecimalSpanExporter; "
            f"found {len(exporters)}"
        )
        # The agent_name flows through to the exporter (and thus onto traces).
        assert exporters[0].default_agent_name == "my-autogen-agent"

    def test_install_adds_to_existing_provider(self):
        """When passed an existing provider, install() adds the exporter to
        it and returns that same provider (no new global provider)."""
        provider = TracerProvider()
        returned = install(agent_name="x", provider=provider)

        assert returned is provider
        exporters = _decimal_exporters(provider)
        assert len(exporters) == 1
        assert exporters[0].default_agent_name == "x"

    def test_exported_span_reaches_backend(self):
        """End-to-end through the real OTEL processor: a GenAI span emitted
        on the installed provider is converted to a RunTrace and submitted
        to the client — proving the exporter is genuinely wired in, not
        just present."""
        import decimalai._config as cfg

        provider = TracerProvider()
        install(agent_name="autogen-e2e", provider=provider)

        tracer = provider.get_tracer("autogen-test")
        with tracer.start_as_current_span("chat") as span:
            span.set_attribute("gen_ai.request.model", "gpt-4o")
            span.set_attribute("gen_ai.system", "openai")
            span.set_attribute("gen_ai.usage.input_tokens", 11)
            span.set_attribute("gen_ai.usage.output_tokens", 22)

        # BatchSpanProcessor exports on a timer, not on span end — force it
        # through; the exporter then submits via the background sender.
        provider.force_flush()
        cfg._sender.flush()

        cfg._client.ingest_trace.assert_called_once()
        run_trace = cfg._client.ingest_trace.call_args[0][0]
        assert run_trace.agent_name == "autogen-e2e"
        assert len(run_trace.llm_calls) == 1
        llm = run_trace.llm_calls[0]
        assert llm.model_name == "gpt-4o"
        assert llm.provider == "openai"
        assert llm.input_tokens == 11
        assert llm.output_tokens == 22

        # The manifest-capable exporter registers a manifest for
        # the AutoGen GenAI span and stamps it on the trace.
        cfg._client.register_manifest.assert_called()
        assert run_trace.manifest_id == "m1"
