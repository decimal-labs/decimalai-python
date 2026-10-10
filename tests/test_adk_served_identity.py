"""Real ADK request, observer and serialized export retain served UUIDs."""

import asyncio
import json
from unittest.mock import MagicMock

import httpx
import pytest


def test_real_adk_preserves_exact_served_identity_through_observer_and_http(monkeypatch):
    pytest.importorskip("google.adk.runners")
    from google.adk.agents import LlmAgent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    import decimalai._config as cfg
    import decimalai.adk as adk
    from decimalai._client import DecimalAIClient
    from decimalai._config import DecimalConfig
    from decimalai.skill_router import SkillRouter, consume_last_delivered_versions

    monkeypatch.setattr(Runner, "run", Runner.run)
    monkeypatch.setattr(adk, "_manifest_ids", {})
    monkeypatch.setattr(adk, "_manifest_trackers", {})
    monkeypatch.setattr(adk, "_pending_manifests", {})
    monkeypatch.setattr(cfg, "_config", DecimalConfig(api_key="dai_sk_test", enabled=True))
    client = MagicMock()
    client.register_manifest.return_value = {"manifest_id": "offline-manifest"}
    monkeypatch.setattr(cfg, "_client", client)
    router = SkillRouter(api_key="dai_sk_test", base_url="http://localhost:8000", inject_body=True)
    name, body = "python-docstring-conventions", "Preserve the complete Python function contract."
    pair = {"skill_id": "f82a8b7a-8dc4-470d-b568-d48cd6c5e62e",
            "version_id": "4f757989-d063-4134-b71b-dfe9b5c12893"}
    route_id = "rt_" + "1" * 24
    monkeypatch.setattr(router, "smart_route", lambda *a, **k: {
        "prompt_fragment": name, "skills": [{"name": name}], "routing_id": route_id,
    })
    monkeypatch.setattr(router, "get_skill_body_record", lambda *a, **k: {
        "body": body, "content_hash": "a" * 64, "version": 4, **pair,
    })
    monkeypatch.setattr(adk, "_skill_router_singleton", router)
    observed, requests = [], []

    class StubLlm(BaseLlm):
        async def generate_content_async(self, llm_request, stream=False):
            requests.append(llm_request.config.system_instruction)
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="Documented.")]))

    service = InMemorySessionService()
    runner = Runner(agent=LlmAgent(name="documentation", model=StubLlm(model="offline-stub")),
                    app_name="identity-proof", session_service=service,
                    plugins=[adk.DecimalaiPlugin(agent_name="documentation", enable_skill_loader=True,
                                                 on_trace=observed.append)])

    async def run():
        await service.create_session(app_name="identity-proof", user_id="u", session_id="s")
        async for _ in runner.run_async(user_id="u", session_id="s", new_message=types.Content(
                role="user", parts=[types.Part(text="Document this supplied Python function.")])):
            pass

    asyncio.run(run())
    cfg._sender.flush()
    consume_last_delivered_versions()
    assert len(requests) == len(observed) == client.ingest_trace.call_count == 1
    assert body in requests[0]
    sent = client.ingest_trace.call_args.args[0]
    expected = [{"name": name, "hash": "a" * 64, "routing_id": route_id, **pair}]
    assert sent.skills_delivered_versions == observed[0].skills_delivered_versions == expected
    systems = [m['content'] for m in sent.llm_calls[0].rendered_input if m['role'] == 'system']
    assert systems == requests
    captured = []

    def capture(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "accepted"}, request=request)

    http_client = DecimalAIClient(api_key="dai_sk_test", base_url="http://identity-proof.test")
    http_client._http.close()
    http_client._http = httpx.Client(base_url="http://identity-proof.test", transport=httpx.MockTransport(capture))
    try:
        http_client.ingest_trace(sent)
    finally:
        http_client._http.close()
    assert captured[0]['skills_delivered_versions'] == expected
    observed[0].skills_delivered_versions[0]['version_id'] = "observer mutation"
    assert sent.skills_delivered_versions == expected
    assert not router._scoped_routing_rails
