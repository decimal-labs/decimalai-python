"""Live-LLM — direct provider-SDK calls through the one-liner tracer.

The single biggest slice of real-world LLM usage is code that calls a provider
SDK *directly* — ``openai.OpenAI().chat.completions.create(...)``,
``anthropic.Anthropic().messages.create(...)``,
``google.genai.Client().models.generate_content(...)`` — with no agent
framework in between. ``decimalai.providers.instrument()`` (and the
``decimalai.init(openai=/anthropic=/google=True)`` aliases) auto-trace exactly
that: each provider's OpenInference instrumentor is enabled and its spans are
routed through DecimalAI's OTEL exporter (:class:`decimalai.otel.DecimalSpanExporter`).

This test proves that path end-to-end, once per provider, with a real call:

  * Enable tracing via the **product function** ``providers.instrument(<provider>=
    True, tracer_provider=local)`` — passing a local ``TracerProvider`` is the
    documented escape hatch, and it's mandatory here because sibling live files
    (crewai/adk) may already have claimed the process-global provider (OTEL
    honors ``set_tracer_provider`` only once).
  * Make a *raw* SDK call — no framework — on a deterministic arithmetic prompt.
  * Assert the backend trace captured the model turn as an ``llm_call`` carrying
    the provider's model id, plus an auto-detected manifest (the exporter mints
    one from the model alone — see ``_maybe_register_manifest``).

Direct provider calls are inherently per-provider (each cell drives its own
vendor SDK), so this lane runs on all three providers — it is *not* a
provider-native framework and is absent from ``FRAMEWORK_PROVIDERS``.

Marker: live_llm + providers.
Install the extra with ``pip install -e ".[providers-tests]"`` (provider SDKs +
their OpenInference instrumentors) and set the provider's API key.
"""

from __future__ import annotations

import pytest

from . import _live_helpers as h


# Pure-reasoning prompt — no tools, deterministic answer. A system prompt nudges
# the model to emit only the number (and exercises a fuller manifest shape).
PROMPT = "What is 17 multiplied by 4? Reply with only the number."
SYSTEM = "You are a calculator. Reply with only the number, nothing else."
EXPECTED = "68"

# Provider -> substring that must appear in the recorded model id.
_MODEL_SUBSTR = {"openai": "gpt", "anthropic": "claude", "google": "gemini"}


def _raw_provider_call(provider: str, model: str) -> str:
    """Make a direct, no-framework SDK call for ``provider`` and return its text.

    These are the canonical one-call shapes the providers lane exists to trace —
    the OpenInference instrumentor (enabled by ``instrument()``) patches each of
    these methods.

    Each client is bound to a local before the call — never the throwaway
    ``Client().method(...)`` shape. google-genai 1.x ties its underlying
    ``SyncHttpxClient`` to the ``Client``'s lifetime, so a temporary client is
    GC-closed mid-request → "Cannot send a request, as the client has been
    closed." Holding the local keeps all three providers GC-safe and uniform.
    """
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content or ""

    if provider == "anthropic":
        from anthropic import Anthropic

        client = Anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=64,
            system=SYSTEM,
            messages=[{"role": "user", "content": PROMPT}],
        )
        # content is a list of blocks; text blocks carry .text.
        return "".join(getattr(b, "text", "") for b in resp.content)

    if provider == "google":
        from google import genai
        from google.genai import types

        client = h.gemini_client()
        resp = client.models.generate_content(
            model=model,
            contents=PROMPT,
            config=types.GenerateContentConfig(system_instruction=SYSTEM),
        )
        return resp.text or ""

    raise AssertionError(f"unknown provider {provider!r}")


@pytest.mark.live_llm
@pytest.mark.providers
@pytest.mark.parametrize("provider, model", h.matrix("providers"))
def test_direct_provider_sdk_call_traced(provider, model):
    """A raw provider-SDK call → ``providers.instrument()`` → one backend trace
    whose model turn is captured as an llm_call with the right model id, plus an
    auto-detected manifest."""
    h.require_key_for(provider)
    pytest.importorskip("opentelemetry.sdk")

    from decimalai.providers import _PROVIDERS, _load_instrumentor, instrument

    spec = _PROVIDERS[provider]
    pytest.importorskip(spec.sdk_module)            # provider SDK
    pytest.importorskip(spec.instrumentor_module)   # its OpenInference instrumentor

    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider

    agent_name = h.unique_agent(f"providers-{provider}")

    # Local provider so we drive the real instrument() escape hatch without
    # fighting OTEL's once-per-process set_tracer_provider guard.
    provider_otel = TracerProvider(
        resource=Resource.create({SERVICE_NAME: "decimal-agent"})
    )

    # THE product call: one line enables direct-SDK tracing for this provider.
    instrument(
        **{provider: True},
        agent_name=agent_name,
        tracer_provider=provider_otel,
    )
    try:
        answer = _raw_provider_call(provider, model)
        provider_otel.force_flush()
    finally:
        # Undo the global SDK patch so it can't leak into sibling live cells.
        instrumentor_cls = _load_instrumentor(spec)
        if instrumentor_cls is not None:
            instrumentor_cls().uninstrument()
        provider_otel.shutdown()

    assert EXPECTED in answer.replace(",", ""), (
        f"Direct {provider} call didn't surface {EXPECTED!r}: {answer!r}"
    )

    h.flush_sdk_sender()
    traces = h.poll_for_trace(agent_name)
    detail = h.get_trace_detail(traces[0]["id"])

    llm_calls = detail.get("llm_calls", [])
    assert llm_calls, (
        f"Trace {detail['id']} has no llm_calls — the direct {provider} call "
        f"wasn't captured by the instrumentor. spans={detail.get('spans')}"
    )
    models = " ".join(
        str(c.get("model_name") or c.get("model") or "") for c in llm_calls
    ).lower()
    assert _MODEL_SUBSTR[provider] in models, (
        f"Expected {_MODEL_SUBSTR[provider]!r} in recorded llm_calls models "
        f"{models!r}. Trace id={detail['id']}"
    )
    assert detail.get("manifest_id"), "manifest_id missing — auto-detection failed"
