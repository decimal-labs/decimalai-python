"""Lock in: exported generation_config agrees with model_config_hash.

Deep-audit finding (sdk-data): to_agentversion read temperature/top_p
un-quantized (the display values) while model_config_hash carries the
quantized hash. So two manifests with temp 0.71 vs 0.79 hash identically
(both quantize to 0.7) but exported DIFFERENT generation_config.temperature
— confusing diff tools that treat the hash as identity.

The fix exports the quantized values (same steps used for the hash: 0.1
for temperature, 0.05 for top_p) so the exported config and the hash agree.
"""

from __future__ import annotations

import decimalai
from decimalai.schema.manifest import extract_from_config


def _export(temp, top_p):
    snap = extract_from_config(
        agent_name="agent",
        models={"default": {"provider": "openai", "model": "gpt-4o",
                            "temperature": temp, "top_p": top_p}},
    )
    return snap, decimalai.export_manifest(snap)


def test_same_hash_implies_same_exported_generation_config():
    """temp 0.71 and 0.79 both quantize to 0.7 → same hash AND same
    exported generation_config.temperature."""
    # temp 0.71 & 0.74 both quantize to 0.7 (step 0.1); top_p 0.91 & 0.92
    # both quantize to 0.90 (step 0.05) — so both manifests share a hash.
    snap_a, m_a = _export(0.71, 0.91)
    snap_b, m_b = _export(0.74, 0.92)

    # Same identity (model_config_hash) ...
    assert snap_a.manifest_hash == snap_b.manifest_hash

    mr_a = m_a["contract"]["model_runtime"]
    mr_b = m_b["contract"]["model_runtime"]
    assert mr_a.get("model_config_hash") == mr_b.get("model_config_hash")

    # ... so the exported generation params must also agree.
    assert mr_a["generation_config"]["temperature"] == mr_b["generation_config"]["temperature"], (
        "two manifests with the same model_config_hash exported different "
        "temperatures — generation_config must be quantized to match the hash."
    )
    assert mr_a["generation_config"]["top_p"] == mr_b["generation_config"]["top_p"]


def test_exported_temperature_is_quantized_value():
    """0.71 must export as the quantized 0.7, not the raw 0.71."""
    _snap, m = _export(0.71, 0.9)
    assert m["contract"]["model_runtime"]["generation_config"]["temperature"] == 0.7


def test_on_grid_value_unchanged():
    """A value already on the quantization grid exports unchanged."""
    _snap, m = _export(0.7, 0.9)
    gc = m["contract"]["model_runtime"]["generation_config"]
    assert gc["temperature"] == 0.7
    assert gc["top_p"] == 0.9
