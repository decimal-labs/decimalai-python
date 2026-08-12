"""SDK → ``agentversion`` manifest export (the converter seam).

The SDK stores a manifest as a flat component list (:class:`ManifestSnapshot`);
the OSS ``agentversion`` tool reads the contract-keyed shape its ``diff`` /
``validate`` commands consume. :func:`decimalai.export_manifest` /
:meth:`ManifestSnapshot.to_agentversion` is the seam between the two.

These tests assert it *actually works* — not just that it returns a dict:

  * The exported shape is the agentversion contract (required + optional
    surfaces, canonical ``amf_<ULID>`` id), with **no** ``agentversion``
    dependency — the converter is pure and offline.
  * With ``agentversion`` installed, the export validates with zero errors
    **and zero warnings**, round-trips through the ``AgentManifest`` model,
    and is diffable: a prompt change surfaces as a ``prompt_stack`` change,
    while re-exporting the same snapshot diffs to nothing (determinism).
"""

from __future__ import annotations

import re
import sys

import pytest

import decimalai
from decimalai.schema.manifest import extract_from_config

# agentversion's canonical object-id grammar: <lowercase prefix>_<26-char
# Crockford-base32 ULID> (alphabet excludes I/L/O/U).
_AMF_RE = re.compile(r"^amf_[0-9A-HJKMNP-TV-Z]{26}$")

_REQUIRED_SURFACES = {
    "prompt_stack", "model_runtime", "tool_registry", "workflow", "output_contract",
}


def _full_snapshot(system_prompt: str = "You are a helpful assistant."):
    """A snapshot exercising every component type the converter maps."""
    return extract_from_config(
        agent_name="support-agent",
        tools=[
            {"name": "search", "schema": {"type": "object"},
             "description": "Search the KB", "stability": "stable"},
            {"name": "get_order", "schema": {"type": "object"},
             "output_schema": {"type": "object"}},
        ],
        prompts={"system": system_prompt, "developer": "Be terse."},
        models={"default": {"provider": "openai", "model": "gpt-4o",
                            "temperature": 0.7, "top_p": 0.9}},
        subagents=[{"name": "billing", "version": "2"}],
        output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        workflow={"name": "react", "nodes": ["plan", "act"]},
        skills=[{"name": "refund-policy", "hash": "abc", "description": "Policy",
                 "stability": "experimental"}],
        guardrails=[{"name": "pii_filter", "kind": "pii", "scope": "output"}],
        context_config={"retrieval": {"source": "pinecone", "top_k": 5}},
        version_label="v1",
    )


# ── shape: pure converter, no agentversion needed ──────────────────────────

def test_export_shape_is_agentversion_contract():
    m = decimalai.export_manifest(_full_snapshot())

    assert m["kind"] == "agent_manifest"
    assert _AMF_RE.match(m["manifest_id"]), m["manifest_id"]
    assert m["agent_name"] == "support-agent"
    assert m["version_label"] == "v1"
    assert "created_at" in m and "identity" in m

    contract = m["contract"]
    # Every required surface present; every declared optional surface present.
    assert _REQUIRED_SURFACES <= set(contract)
    for optional in ("skill_registry", "subagents", "guardrails", "context_config"):
        assert optional in contract, f"missing optional surface {optional!r}"

    # prompt_stack: named prompts claim their slots.
    assert contract["prompt_stack"]["system_prompt"]["id"] == "system"
    assert contract["prompt_stack"]["developer_prompt"]["id"] == "developer"

    # model_runtime: provider/model + quantizable generation config.
    mr = contract["model_runtime"]
    assert mr["provider"] == "openai" and mr["model"] == "gpt-4o"
    assert mr["generation_config"]["temperature"] == 0.7

    # tool_registry: descriptors carry name+hash; stability passed through.
    tr = contract["tool_registry"]
    assert {t["name"] for t in tr["tools"]} == {"search", "get_order"}
    assert all(t["hash"] for t in tr["tools"])
    search = next(t for t in tr["tools"] if t["name"] == "search")
    assert search["stability"] == "stable"
    assert tr["registry_hash"]

    # output_contract derived from the declared schema (not the "none" sentinel).
    assert contract["output_contract"]["format"] == "json"
    assert contract["output_contract"]["schema_hash"]

    # subagent required fields (name/version/hash) all populated.
    sa = contract["subagents"][0]
    assert sa["name"] == "billing" and sa["version"] == "2" and sa["hash"]


def test_output_contract_none_sentinel_when_no_schema():
    """A manifest with no output schema still gets a valid output_contract."""
    snap = extract_from_config(
        agent_name="a",
        prompts={"system": "hi"},
        models={"default": {"provider": "google", "model": "gemini-2.5-pro"}},
    )
    oc = decimalai.export_manifest(snap)["contract"]["output_contract"]
    assert oc == {"version": "0", "schema_hash": "", "format": "none",
                  "strict": False, "modalities": []}


def test_provider_inferred_from_model_id_when_absent():
    """No explicit provider → inferred from the model id."""
    snap = extract_from_config(
        agent_name="a", models={"default": {"model": "gemini-3.5-flash"}},
    )
    assert decimalai.export_manifest(snap)["contract"]["model_runtime"]["provider"] == "google"


def test_unsupported_format_raises():
    with pytest.raises(ValueError, match="Unsupported manifest export format"):
        decimalai.export_manifest(_full_snapshot(), format="not-a-format")


def test_export_works_without_agentversion(monkeypatch):
    """The converter must not require agentversion: block its import entirely
    and confirm export still produces a valid shape, falling back to the SDK's
    own surface hash under an honest algorithm label (no spec_version)."""
    for name in ("agentversion", "agentversion.hasher", "agentversion.constants"):
        monkeypatch.setitem(sys.modules, name, None)

    m = decimalai.export_manifest(_full_snapshot())

    assert _AMF_RE.match(m["manifest_id"])
    assert _REQUIRED_SURFACES <= set(m["contract"])
    assert "spec_version" not in m
    assert m["identity"]["hash_algorithm"] == "decimalai-surface-sha256"
    assert m["identity"]["overall_hash"]  # the SDK manifest hash, carried as-is


# ── validation + diff: require agentversion ────────────────────────────────

def test_export_validates_clean():
    pytest.importorskip("agentversion")
    from agentversion import AgentManifest, validate_manifest

    m = decimalai.export_manifest(_full_snapshot())

    # jcs present → canonical hash recomputed → zero warnings, zero errors.
    res = validate_manifest(m)
    assert res.valid is True, res.errors
    assert res.errors == [], res.errors
    assert res.warnings == [], res.warnings
    assert m["identity"]["hash_algorithm"] == "jcs-sha256"
    assert m["identity"]["overall_hash"].startswith("sha256:")

    # And it round-trips through the strict pydantic model.
    AgentManifest(**m)


def test_prompt_change_surfaces_as_prompt_stack_diff():
    """Journey A — ship a prompt change safely. Two manifests differing only in
    the system prompt must diff to exactly one changed surface: prompt_stack."""
    pytest.importorskip("agentversion")
    from agentversion.diff import diff_manifests

    old = decimalai.export_manifest(_full_snapshot("You are a helpful assistant."))
    new = decimalai.export_manifest(_full_snapshot("You are a terse assistant."))

    diff = diff_manifests(old, new)
    changed = {c.surface for c in diff.changed_surfaces}
    assert changed == {"prompt_stack"}, changed


def test_identical_snapshot_diffs_to_nothing():
    """Determinism: the contract block carries no per-export randomness (only
    manifest_id/created_at do), so re-exporting the same snapshot diffs clean."""
    pytest.importorskip("agentversion")
    from agentversion.diff import diff_manifests

    snap = _full_snapshot()
    diff = diff_manifests(decimalai.export_manifest(snap),
                          decimalai.export_manifest(snap))
    assert diff.changed_surfaces == []
    assert diff.summary.max_severity == "none"


# ── cross-stack parity: the SDK + platform must agree on component→surface routing ──────────────

def test_sdk_routing_matches_agentversion_canonical_map():
    """The SDK exporter and the platform's diff translators each map a producer's flat
    `component_type` to an agentversion surface. They previously shared NO code/test, so the
    guardrail singular→plural rename had to be hand-applied in every copy. The routing now lives in
    agentversion (`surface_key_for_component`), and this guards that the SDK export agrees with it."""
    pytest.importorskip("agentversion")
    from agentversion import surface_key_for_component

    contract = decimalai.export_manifest(_full_snapshot())["contract"]
    for ctype in ("tool", "prompt", "model", "subagent", "output_schema", "workflow", "skill", "guardrail"):
        surface = surface_key_for_component(ctype)
        assert surface in contract, (
            f"SDK export routes component_type {ctype!r} away from agentversion surface {surface!r}"
        )
    # The rename specifically: the surface is the PLURAL 'guardrails', never a singular 'guardrail' key.
    assert "guardrails" in contract and "guardrail" not in contract


# ── New agentversion-0.2.0 surfaces ────────────────────────────────────────

def test_behavioral_policy_change_surfaces_as_breaking():
    """A multi-turn policy rule change diffs as exactly behavioral_policy,
    breaking/major — it no longer hides in the prompt hash."""
    pytest.importorskip("agentversion")
    from agentversion.diff import diff_manifests

    def snap(threshold, phash):
        return decimalai.export_manifest(extract_from_config(
            agent_name="pa",
            models={"default": {"provider": "openai", "model": "gpt-4o"}},
            behavioral_policy={"policy_id": "refund", "policy_hash": phash,
                               "objection_threshold": threshold},
        ))

    diff = diff_manifests(snap(3, "sha256:p3"), snap(1, "sha256:p1"))
    changed = {c.surface for c in diff.changed_surfaces}
    assert changed == {"behavioral_policy"}, changed
    bp = next(c for c in diff.changed_surfaces if c.surface == "behavioral_policy")
    assert bp.change_type == "breaking"
    assert bp.severity == "major"


def test_environment_change_is_non_breaking():
    """An environment change diffs as environment, non_breaking."""
    pytest.importorskip("agentversion")
    from agentversion.diff import diff_manifests

    def snap(region):
        return decimalai.export_manifest(extract_from_config(
            agent_name="ea", models={"default": {"model": "gpt-4o"}},
            environment={"deployment_id": "d", "region": region},
        ))

    diff = diff_manifests(snap("us-east"), snap("eu-west"))
    changed = {c.surface for c in diff.changed_surfaces}
    assert changed == {"environment"}, changed
    env = next(c for c in diff.changed_surfaces if c.surface == "environment")
    assert env.change_type == "non_breaking"


def test_tool_calling_mode_change_is_breaking():
    """tool_calling_mode lifted to model_runtime top level drives the
    breaking model classification (not buried in model_config_hash)."""
    pytest.importorskip("agentversion")
    from agentversion.diff import diff_manifests

    def snap(mode):
        return decimalai.export_manifest(extract_from_config(
            agent_name="ta",
            models={"default": {"provider": "openai", "model": "gpt-4o",
                                "tool_calling_mode": mode}},
        ))

    diff = diff_manifests(snap("native"), snap("none"))
    changed = {c.surface for c in diff.changed_surfaces}
    assert changed == {"model_runtime"}, changed
    mr = next(c for c in diff.changed_surfaces if c.surface == "model_runtime")
    assert mr.change_type == "breaking"


def test_new_surfaces_validate_and_export_top_level():
    """The new surfaces round-trip through the strict AgentManifest model and
    tool_calling_mode/runtime_version export at model_runtime top level."""
    pytest.importorskip("agentversion")
    from agentversion.manifest import AgentManifest

    m = decimalai.export_manifest(extract_from_config(
        agent_name="va",
        models={"default": {"provider": "openai", "model": "gpt-4o",
                            "tool_calling_mode": "native", "runtime_version": "9.9"}},
        behavioral_policy={"policy_id": "p", "policy_hash": "sha256:p"},
        environment={"deployment_id": "d", "region": "us"},
    ))
    AgentManifest(**m)  # validates clean under the strict model
    mr = m["contract"]["model_runtime"]
    assert mr["tool_calling_mode"] == "native"
    assert mr["runtime_version"] == "9.9"
    assert "behavioral_policy" in m["contract"]
    assert "environment" in m["contract"]
