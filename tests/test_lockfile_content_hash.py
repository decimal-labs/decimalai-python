"""Lockfile hash contract.

Pre-fix, ``.decimal/skills.lock`` stored ``sha256(body)[:12]`` — and because
the export path delivered an EMPTY body, every entry pinned
``e3b0c44298fc`` (the hash of ""), i.e. nothing. The lock must pin the FULL
sha256 hex of the exact SKILL.md content actually written to disk
(frontmatter + body as written), plus ``body_hash`` (full sha256 of
body_markdown — the axis ``GET /skills/hashes`` serves) for update
comparisons.
"""

from __future__ import annotations

import hashlib
import json

from decimalai.disk_export import export_skills_to_disk


def _skill():
    return {
        "id": "sk1",
        "name": "pdf",
        "description": "Extract text from PDFs",
        "body_markdown": "# PDF\n\nUse when the user hands you a PDF.",
        "category": "documents",
        "latest_version": {"version_number": 4},
    }


def _lock(root) -> dict:
    with open(root / ".decimal" / "skills.lock", encoding="utf-8") as f:
        return json.load(f)


def test_content_hash_is_full_sha256_of_written_file(tmp_path):
    summary = export_skills_to_disk([_skill()], project_root=str(tmp_path))
    assert summary["skills_written"] == 1

    skill_md = tmp_path / ".agents" / "skills" / "pdf" / "SKILL.md"
    file_hash = hashlib.sha256(skill_md.read_bytes()).hexdigest()

    entry = _lock(tmp_path)["skills"]["pdf"]
    assert entry["content_hash"] == file_hash
    assert len(entry["content_hash"]) == 64  # full hex, not a 12-char prefix


def test_body_hash_matches_platform_axis_and_version_is_real(tmp_path):
    export_skills_to_disk([_skill()], project_root=str(tmp_path))
    entry = _lock(tmp_path)["skills"]["pdf"]
    assert entry["body_hash"] == hashlib.sha256(
        _skill()["body_markdown"].encode("utf-8")
    ).hexdigest()
    # Pinned to the platform's version number, not a hardcoded 1.
    assert entry["version"] == 4


def test_hash_never_pins_the_empty_string(tmp_path):
    """The regression this file exists for: every lock entry pinned
    e3b0c44298fc, the sha256 of the empty string, i.e. nothing."""
    export_skills_to_disk([_skill()], project_root=str(tmp_path))
    entry = _lock(tmp_path)["skills"]["pdf"]
    empty_hash = hashlib.sha256(b"").hexdigest()
    assert entry["content_hash"] != empty_hash
    assert not entry["content_hash"].startswith("e3b0c44298fc")
    assert entry["body_hash"] != empty_hash
