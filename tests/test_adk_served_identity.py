"""A skill the model loads keeps its exact served identity, through the observer and HTTP.

``load_skill`` serves a body with the skill and version ids the platform served it
under, against the invocation's routing decision (the trace's ``routing_id``). That
identity has to reach ``skills_delivered_versions`` whole: in the copy the plugin's
observer gets, in the trace that is exported, and in the payload a real
``DecimalAIClient`` POSTs.
"""

import hashlib
import json

import httpx
import pytest

pytest.importorskip("google.adk")

import decimalai.adk as adk  # noqa: E402
from tests.test_adk_load_agent import model, platform, run  # noqa: E402,F401
from tests.test_agent_binding import BODIES  # noqa: E402


def test_a_loaded_skill_keeps_its_served_identity_through_observer_and_http(platform):  # noqa: F811
    from decimalai._client import DecimalAIClient

    observed: list = []
    support = adk.agent_config("support")
    run(support, model(("call", "damaged-item"), "answer"), "My keyboard arrived cracked.",
        traces=observed)
    sent = platform.client.ingest_trace.call_args.args[0]
    (routing_id,) = platform.queries
    body = BODIES["damaged-item"]
    # The load is served against the invocation's decision, which the trace names.
    assert sent.routing_id == observed[0].routing_id == routing_id
    expected = [{
        "name": "damaged-item", "hash": hashlib.sha256(body.encode()).hexdigest(),
        "skill_id": f"11111111-1111-4111-8111-{len('damaged-item'):012d}",
        "version_id": "22222222-2222-4222-8222-222222222222",
    }]
    assert sent.skills_delivered_versions == observed[0].skills_delivered_versions == expected
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
    assert captured[0]["skills_delivered_versions"] == expected
    observed[0].skills_delivered_versions[0]["version_id"] = "observer mutation"
    assert sent.skills_delivered_versions == expected
    assert not adk._skill_router_singleton._scoped_routing_rails
