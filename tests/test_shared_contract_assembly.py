"""Cross-implementation conformance: the SDK exporter and the agentversion shared
assembly must produce the SAME contract — and therefore the SAME canonical
jcs-sha256 identity hash the platform stores.

Two hand-written SDK→agentversion translators once existed with no shared
code and no shared test, and they drifted. This file is the guard: if ``to_agentversion``
ever diverges from ``agentversion.contract.contract_from_components``, or the SDK's
``_surface_for_type`` from ``agentversion.surface_key_for_component``, these fail.
"""

from agentversion.contract import contract_from_components
from agentversion.diff import surface_key_for_component
from agentversion.hasher import hash_manifest

from decimalai.schema.manifest import _surface_for_type, extract_from_config

_CASES = {
    "full": dict(
        agent_name="a",
        prompts={"system": "You help.", "developer": "Be terse."},
        models={"default": {"provider": "openai", "model": "gpt-4o",
                            "temperature": 0.71, "top_p": 0.93,
                            "max_tokens": 1024, "response_format": "json",
                            "tool_calling_mode": "native", "runtime_version": "1.2.0"}},
        tools=[{"name": "lookup", "description": "d", "version": "2",
                "input_schema": {"x": 1}, "output_schema": {"y": 2},
                "stability": "stable", "annotations": {"a": 1}}],
        subagents=[{"name": "billing", "version": "1"}],
        output_schema={"format": "json", "strict": True, "modalities": ["text"]},
        workflow={"name": "g", "version": "3"},
        skills=[{"name": "sk", "hash": "sha256:sk", "description": "s"}],
        guardrails=[{"name": "pii", "kind": "pii", "scope": "output"}],
        context_config={"retrieval": {"source": "pinecone"}},
        behavioral_policy={"policy_id": "refund", "policy_hash": "sha256:p",
                           "objection_threshold": 3, "concede_events": ["refund_issued"],
                           "always_forbidden": ["admit_liability"]},
        environment={"deployment_id": "prod-1", "region": "us-east",
                     "runtime_versions": {"python": "3.12"},
                     "secret_refs": ["prod/openai-key"]},
    ),
    "minimal": dict(agent_name="b", models={"default": {"provider": "google", "model": "gemini-2.0-flash"}}),
    "extra_prompts": dict(agent_name="c", prompts={"foo": "F", "bar": "B", "baz": "Z"},
                          models={"m": {"model": "claude-haiku-4-5"}}),
    "policy_only": dict(agent_name="d",
                        models={"default": {"model": "gpt-4o"}},
                        behavioral_policy={"policy_id": "escalation", "policy_hash": "sha256:e",
                                           "objection_threshold": 1}),
    "env_only": dict(agent_name="e",
                     models={"default": {"model": "gpt-4o"}},
                     environment={"deployment_id": "d2", "region": "eu-west"}),
}


def _norm(snap):
    return [{"component_type": c.component_type, "component_name": c.component_name,
             "component_version": c.component_version, "content_hash": c.content_hash,
             "schema_json": c.schema_json} for c in snap.components]


def test_surface_map_matches_agentversion():
    for ctype in ("tool", "skill", "prompt", "model", "subagent",
                  "output_schema", "workflow", "guardrail", "context_config", "unknown_x"):
        assert _surface_for_type(ctype) == surface_key_for_component(ctype)


def test_to_agentversion_contract_matches_shared_assembly():
    for name, kw in _CASES.items():
        snap = extract_from_config(**kw)
        assert snap.to_agentversion()["contract"] == contract_from_components(_norm(snap)), name


def test_exported_hash_equals_platform_canonical_hash():
    """The hash the SDK stamps == hash_manifest(shared contract) == what the
    platform stores as the canonical identity hash."""
    for name, kw in _CASES.items():
        snap = extract_from_config(**kw)
        exported = snap.to_agentversion()
        platform_hash = hash_manifest({"contract": contract_from_components(_norm(snap))})
        assert exported["identity"]["overall_hash"] == platform_hash, name
        assert platform_hash.startswith("sha256:")


def test_new_surfaces_present_and_top_level_model_fields():
    """The behavioral_policy / environment surfaces are emitted, and
    tool_calling_mode / runtime_version sit at model_runtime top level rather
    than buried under generation_config."""
    contract = extract_from_config(**_CASES["full"]).to_agentversion()["contract"]
    assert contract["behavioral_policy"]["objection_threshold"] == 3
    assert contract["environment"]["region"] == "us-east"
    mr = contract["model_runtime"]
    assert mr["tool_calling_mode"] == "native"
    assert mr["runtime_version"] == "1.2.0"
    assert "tool_calling_mode" not in mr.get("generation_config", {})


def test_new_surfaces_absent_when_undeclared():
    """Additive-only: a model-only manifest carries neither new surface, so
    existing manifests keep their exact prior hash."""
    contract = extract_from_config(**_CASES["minimal"]).to_agentversion()["contract"]
    assert "behavioral_policy" not in contract
    assert "environment" not in contract
