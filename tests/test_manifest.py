"""Tests for SDK manifest extraction, change detection, and registration."""

import hashlib
import json
from uuid import uuid4

import pytest

from decimalai.schema.manifest import (
    ComponentSnapshot,
    ManifestSnapshot,
    ManifestTracker,
    extract_from_config,
    _hash_content,
)


class TestHashContent:
    def test_deterministic(self):
        """Same content → same hash."""
        assert _hash_content({"a": 1, "b": 2}) == _hash_content({"b": 2, "a": 1})

    def test_different_content(self):
        """Different content → different hash."""
        assert _hash_content({"a": 1}) != _hash_content({"a": 2})

    def test_sha256_format(self):
        """Hash is a 64-char hex string."""
        h = _hash_content("hello")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestExtractFromConfig:
    def test_minimal_extraction(self):
        """Extract with just agent_name."""
        snap = extract_from_config(agent_name="my-agent")
        assert snap.agent_name == "my-agent"
        assert snap.manifest_hash != ""
        assert len(snap.components) == 0

    def test_tools_extraction(self):
        """Tools are extracted as component snapshots."""
        tools = [
            {"name": "search", "schema": {"type": "object"}},
            {"name": "calculator", "schema": {"properties": {"x": {"type": "int"}}}},
        ]
        snap = extract_from_config(agent_name="agent", tools=tools)
        assert len(snap.components) == 2
        assert snap.components[0].component_type == "tool"
        assert snap.components[0].component_name == "search"
        assert snap.components[0].content_hash is not None
        assert snap.component_summary_json["tools"] == 2

    def test_prompts_extraction(self):
        """Prompts are extracted with text preview."""
        prompts = {"system": "You are a helpful assistant.", "user_template": "Help me with {query}"}
        snap = extract_from_config(agent_name="agent", prompts=prompts)
        assert len(snap.components) == 2
        assert snap.components[0].component_type == "prompt"
        assert snap.component_summary_json["prompts"] == 2

    def test_models_extraction(self):
        """Models are extracted with provider info."""
        models = {
            "router": {"provider": "openai", "model": "gpt-4o-mini"},
            "worker": {"provider": "google", "model": "gemini-2.0-flash"},
        }
        snap = extract_from_config(agent_name="agent", models=models)
        assert len(snap.components) == 2
        assert snap.components[0].component_type == "model"
        assert snap.agent_models_json is not None
        assert snap.agent_models_json["router"]["model"] == "gpt-4o-mini"

    def test_hash_changes_on_tool_change(self):
        """Changing a tool schema changes the manifest hash."""
        tools_v1 = [{"name": "search", "schema": {"type": "object", "properties": {"query": {"type": "string"}}}}]
        tools_v2 = [{"name": "search", "schema": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "int"}}}}]

        snap1 = extract_from_config(agent_name="agent", tools=tools_v1)
        snap2 = extract_from_config(agent_name="agent", tools=tools_v2)
        assert snap1.manifest_hash != snap2.manifest_hash

    def test_hash_stable_with_same_config(self):
        """Same config → same hash."""
        tools = [{"name": "calc", "schema": {"type": "object"}}]
        snap1 = extract_from_config(agent_name="agent", tools=tools)
        snap2 = extract_from_config(agent_name="agent", tools=tools)
        assert snap1.manifest_hash == snap2.manifest_hash

    def test_full_extraction(self):
        """Extract with all component types."""
        snap = extract_from_config(
            agent_name="full-agent",
            version_label="v3",
            tools=[{"name": "search"}],
            prompts={"system": "You are helpful."},
            models={"main": {"provider": "openai", "model": "gpt-4o"}},
            subagents=[{"name": "researcher", "version": "1"}],
            output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
            workflow={"name": "research-graph", "nodes": ["router", "researcher"]},
        )
        assert len(snap.components) == 6  # tool + prompt + model + subagent + output + workflow
        assert snap.version_label == "v3"
        assert snap.graph_topology_hash is not None
        assert all(
            k in snap.component_summary_json
            for k in ["tools", "prompts", "models", "subagents", "output_schemas", "workflows"]
        )


class TestManifestTracker:
    def test_first_snapshot_is_new(self):
        """First snapshot is always new."""
        tracker = ManifestTracker()
        snap = extract_from_config(agent_name="agent", tools=[{"name": "search"}])
        assert tracker.check_and_update(snap) is True

    def test_same_snapshot_is_not_new(self):
        """Same snapshot is not new."""
        tracker = ManifestTracker()
        snap = extract_from_config(agent_name="agent", tools=[{"name": "search"}])
        tracker.check_and_update(snap)
        assert tracker.check_and_update(snap) is False

    def test_changed_snapshot_is_new(self):
        """Changed config → new snapshot."""
        tracker = ManifestTracker()
        snap1 = extract_from_config(agent_name="agent", tools=[{"name": "search"}])
        snap2 = extract_from_config(agent_name="agent", tools=[{"name": "search"}, {"name": "calc"}])
        
        tracker.check_and_update(snap1)
        assert tracker.check_and_update(snap2) is True

    def test_reset(self):
        """Reset clears state."""
        tracker = ManifestTracker()
        snap = extract_from_config(agent_name="agent", tools=[{"name": "search"}])
        tracker.check_and_update(snap)
        tracker.reset()
        assert tracker.last_hash is None
        # After reset, same snap is "new" again
        assert tracker.check_and_update(snap) is True

    def test_tracks_last_manifest(self):
        """Tracker stores the last manifest."""
        tracker = ManifestTracker()
        snap = extract_from_config(agent_name="agent", tools=[{"name": "search"}])
        tracker.check_and_update(snap)
        assert tracker.last_manifest is not None
        assert tracker.last_manifest.agent_name == "agent"


# ── Phase 1: Quantized float hashing ─────────────────────

class TestQuantizeFloat:
    def test_basic_quantization(self):
        from decimalai.schema.manifest import quantize_float
        assert quantize_float(0.71, 0.1) == 0.7
        assert quantize_float(0.75, 0.1) == 0.8
        assert quantize_float(0.0, 0.1) == 0.0
        assert quantize_float(1.0, 0.1) == 1.0

    def test_top_p_quantization(self):
        from decimalai.schema.manifest import quantize_float
        assert quantize_float(0.92, 0.05) == 0.9
        assert quantize_float(0.93, 0.05) == 0.95

    def test_none_passthrough(self):
        from decimalai.schema.manifest import quantize_float
        assert quantize_float(None, 0.1) is None


class TestTemperatureHashing:
    def test_small_temp_change_same_hash(self):
        """0.7 → 0.71 should produce the same manifest hash."""
        models_v1 = {"default": {"provider": "openai", "model": "gpt-4o", "temperature": 0.7}}
        models_v2 = {"default": {"provider": "openai", "model": "gpt-4o", "temperature": 0.71}}
        snap1 = extract_from_config(agent_name="agent", models=models_v1)
        snap2 = extract_from_config(agent_name="agent", models=models_v2)
        assert snap1.manifest_hash == snap2.manifest_hash

    def test_large_temp_change_different_hash(self):
        """0.7 → 0.8 should produce different hashes."""
        models_v1 = {"default": {"provider": "openai", "model": "gpt-4o", "temperature": 0.7}}
        models_v2 = {"default": {"provider": "openai", "model": "gpt-4o", "temperature": 0.8}}
        snap1 = extract_from_config(agent_name="agent", models=models_v1)
        snap2 = extract_from_config(agent_name="agent", models=models_v2)
        assert snap1.manifest_hash != snap2.manifest_hash

    def test_runtime_settings_excluded(self):
        """max_retries and timeout should not affect the hash."""
        models_v1 = {"default": {"provider": "openai", "model": "gpt-4o", "max_retries": 3, "timeout": 30}}
        models_v2 = {"default": {"provider": "openai", "model": "gpt-4o", "max_retries": 5, "timeout": 60}}
        snap1 = extract_from_config(agent_name="agent", models=models_v1)
        snap2 = extract_from_config(agent_name="agent", models=models_v2)
        assert snap1.manifest_hash == snap2.manifest_hash

    def test_schema_json_preserves_original(self):
        """schema_json should contain original values, not quantized."""
        models = {"default": {"provider": "openai", "model": "gpt-4o", "temperature": 0.71}}
        snap = extract_from_config(agent_name="agent", models=models)
        model_comp = [c for c in snap.components if c.component_type == "model"][0]
        assert model_comp.schema_json["temperature"] == 0.71  # Original, not quantized


class TestPerSurfaceHashing:
    def test_tool_order_invariant(self):
        """Tools in different order should produce the same hash (sorted by name)."""
        tools_v1 = [{"name": "alpha"}, {"name": "beta"}]
        tools_v2 = [{"name": "beta"}, {"name": "alpha"}]
        snap1 = extract_from_config(agent_name="agent", tools=tools_v1)
        snap2 = extract_from_config(agent_name="agent", tools=tools_v2)
        assert snap1.manifest_hash == snap2.manifest_hash

    def test_multi_instance_convergence(self):
        """Same config from two calls → same hash (simulating two pods)."""
        config = {
            "agent_name": "finance-agent",
            "tools": [{"name": "get_stock_price", "schema": {"type": "object"}}],
            "prompts": {"system": "You are a financial advisor."},
            "models": {"default": {"provider": "openai", "model": "gpt-4o"}},
        }
        snap1 = extract_from_config(**config)
        snap2 = extract_from_config(**config)
        assert snap1.manifest_hash == snap2.manifest_hash

    def test_detection_source_set_to_sdk(self):
        """Manifests from extract_from_config should have detection_source='sdk'."""
        snap = extract_from_config(agent_name="agent", tools=[{"name": "search"}])
        assert snap.detection_source == "sdk"

