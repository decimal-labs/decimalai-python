"""Lock in: a version-only bump with an explicit hash changes the manifest hash.

Deep-audit finding (sdk-data): _compute_surface_hash hashed only
{name, hash} per component. For skills/guardrails ``content_hash`` is
``entry.get('hash') or _hash_content(entry)``, so a caller supplying an
EXPLICIT hash and bumping ONLY the version changed neither the per-skill
content_hash nor the surface dict — the manifest_hash was identical and
check_and_update returned False, so the new version was never registered.

The fix folds component_version into the per-component dict in
_compute_surface_hash.
"""

from decimalai.schema.manifest import ManifestTracker, extract_from_config


def test_skill_version_only_bump_with_explicit_hash_changes_manifest_hash():
    """Same explicit content hash, version 1.0 → 2.0 → different manifest hash."""
    skill_v1 = [{"name": "code-review", "hash": "sha256:deadbeef", "version": "1.0"}]
    skill_v2 = [{"name": "code-review", "hash": "sha256:deadbeef", "version": "2.0"}]

    snap1 = extract_from_config(agent_name="agent", skills=skill_v1)
    snap2 = extract_from_config(agent_name="agent", skills=skill_v2)

    assert snap1.manifest_hash != snap2.manifest_hash, (
        "a version-only bump (with an explicit, unchanged hash) must change "
        "the manifest hash — otherwise the new version is never registered."
    )


def test_guardrail_version_only_bump_changes_manifest_hash():
    g_v1 = [{"name": "pii-filter", "hash": "sha256:cafe", "version": "1.0"}]
    g_v2 = [{"name": "pii-filter", "hash": "sha256:cafe", "version": "2.0"}]

    snap1 = extract_from_config(agent_name="agent", guardrails=g_v1)
    snap2 = extract_from_config(agent_name="agent", guardrails=g_v2)

    assert snap1.manifest_hash != snap2.manifest_hash


def test_tracker_reregisters_after_version_bump():
    """End-to-end through the tracker: the bumped version must register."""
    tracker = ManifestTracker()
    skill_v1 = [{"name": "code-review", "hash": "sha256:deadbeef", "version": "1.0"}]
    skill_v2 = [{"name": "code-review", "hash": "sha256:deadbeef", "version": "2.0"}]

    assert tracker.check_and_update(extract_from_config(agent_name="a", skills=skill_v1)) is True
    # Same manifest again — should dedup.
    assert tracker.check_and_update(extract_from_config(agent_name="a", skills=skill_v1)) is False
    # Version bump — should be treated as a NEW manifest worth registering.
    assert tracker.check_and_update(extract_from_config(agent_name="a", skills=skill_v2)) is True


def test_identical_version_still_dedups():
    """Guard against over-correcting: identical config still produces the
    same hash (no spurious re-registration)."""
    skill = [{"name": "code-review", "hash": "sha256:deadbeef", "version": "1.0"}]
    snap1 = extract_from_config(agent_name="agent", skills=skill)
    snap2 = extract_from_config(agent_name="agent", skills=skill)
    assert snap1.manifest_hash == snap2.manifest_hash
