"""The lock the deprecated ``export_skills_to_disk`` writes, and the ones it never touches.

A project ``decimalai skills`` manages has a version 2 ``.decimal/skills.lock``. The
deprecated writer must never write a version 1 lock into it (0.13.9 did, merging v1
entries into the v2 file), so there it writes no lock at all and the folders it wrote
stay unmanaged. A project with no lock gets none either: a fresh project never starts
with legacy state. Only where a version 1 lock already exists does the writer keep
updating it, so ``decimalai skills migrate`` can convert it. There the entry pins the
FULL sha256 of the exact SKILL.md written (never ``sha256("")[:12]``,
``e3b0c44298fc``, which an empty body once pinned), ``body_hash`` (the axis
``GET /skills/hashes`` serves) and the platform's version number.
"""

from __future__ import annotations

import hashlib
import json

import pytest

import decimalai.disk_export as disk_export
from decimalai.disk_export import _write_lockfile, export_skills_to_disk
from decimalai.skillpkg import fs
from decimalai.skillpkg import lock as L
from decimalai.skillpkg.project import Project


@pytest.fixture(autouse=True)
def _fresh_once_state(monkeypatch):
    monkeypatch.setattr(disk_export, "_warned_once", set())


def _skill():
    return {
        "id": "00000000-0000-4000-8000-000000000001",
        "latest_version_id": "00000000-0000-4000-8000-000000000002",
        "name": "pdf",
        "description": "Extract text from PDFs",
        "body_markdown": "# PDF\n\nUse when the user hands you a PDF.",
        "category": "documents",
        "latest_version": {"version_number": 4},
    }


def _lock(root) -> dict:
    with open(root / ".decimal" / "skills.lock", encoding="utf-8") as f:
        return json.load(f)


# ── no lock: none is written ──────────────────────────────────────────


def test_a_project_without_a_lock_gets_none(tmp_path):
    summary = export_skills_to_disk([_skill()], project_root=str(tmp_path))

    assert summary["skills_written"] == 1
    assert summary["lockfile"] is None
    assert not (tmp_path / ".decimal" / "skills.lock").exists()
    assert (tmp_path / ".agents" / "skills" / "pdf" / "SKILL.md").is_file()


# ── a version 1 lock: it is kept up to date ──────────────────────────


@pytest.fixture
def v1_project(tmp_path):
    """A project an older SDK wrote: a version 1 lock with one entry of its own."""
    (tmp_path / ".decimal").mkdir()
    (tmp_path / ".decimal" / "skills.lock").write_text(json.dumps(
        {"version": 1, "skills": {"older": {"version": 1, "content_hash": "ab" * 32}}}),
        encoding="utf-8")
    return tmp_path


def test_content_hash_is_full_sha256_of_written_file(v1_project):
    tmp_path = v1_project
    summary = export_skills_to_disk([_skill()], project_root=str(tmp_path))
    assert summary["skills_written"] == 1
    assert summary["lockfile"] == "v1"

    skill_md = tmp_path / ".agents" / "skills" / "pdf" / "SKILL.md"
    file_hash = hashlib.sha256(skill_md.read_bytes()).hexdigest()

    lock = _lock(tmp_path)
    assert lock["version"] == 1
    entry = lock["skills"]["pdf"]
    assert entry["content_hash"] == file_hash
    assert len(entry["content_hash"]) == 64  # full hex, not a 12-char prefix


def test_body_hash_matches_platform_axis_and_version_is_real(v1_project):
    tmp_path = v1_project
    export_skills_to_disk([_skill()], project_root=str(tmp_path))
    entry = _lock(tmp_path)["skills"]["pdf"]
    assert entry["body_hash"] == hashlib.sha256(
        _skill()["body_markdown"].encode("utf-8")
    ).hexdigest()
    # Pinned to the platform's version number, not a hardcoded 1.
    assert entry["version"] == 4


def test_hash_never_pins_the_empty_string(v1_project):
    tmp_path = v1_project
    export_skills_to_disk([_skill()], project_root=str(tmp_path))
    entry = _lock(tmp_path)["skills"]["pdf"]
    empty_hash = hashlib.sha256(b"").hexdigest()
    assert entry["content_hash"] != empty_hash
    assert not entry["content_hash"].startswith("e3b0c44298fc")
    assert entry["body_hash"] != empty_hash


def test_an_existing_version_1_lock_keeps_its_other_entries(v1_project):
    export_skills_to_disk([_skill()], project_root=str(v1_project))
    assert sorted(_lock(v1_project)["skills"]) == ["older", "pdf"]


def test_a_version_1_lock_with_no_version_key_counts_as_one(tmp_path):
    """The oldest SDKs wrote no ``version``; that is a version 1 lock too."""
    (tmp_path / ".decimal").mkdir()
    (tmp_path / ".decimal" / "skills.lock").write_text('{"skills": {}}', encoding="utf-8")
    summary = export_skills_to_disk([_skill()], project_root=str(tmp_path))
    assert summary["lockfile"] == "v1"
    assert "pdf" in _lock(tmp_path)["skills"]


# ── a version 2 lock: never written into ─────────────────────────────


def _v2_project(tmp_path) -> bytes:
    project = Project.at(tmp_path)
    L.save(project, L.empty())
    return project.lock_path.read_bytes()


def test_a_version_2_lock_is_never_written_into(tmp_path):
    before = _v2_project(tmp_path)

    summary = export_skills_to_disk([_skill()], project_root=str(tmp_path))

    assert (tmp_path / ".decimal" / "skills.lock").read_bytes() == before
    assert summary["lockfile"] is None
    assert (tmp_path / ".agents" / "skills" / "pdf" / "SKILL.md").is_file()


def test_the_folder_it_wrote_is_reported_unmanaged(tmp_path):
    _v2_project(tmp_path)
    summary = export_skills_to_disk([_skill()], project_root=str(tmp_path))
    project = Project.at(tmp_path)

    assert summary["unmanaged"] == [str(project.skills_root / "pdf")]
    state = fs.scan_root(project, L.load(project).skills)["pdf"]
    assert state.state is fs.State.UNMANAGED_STAMPED


def test_the_v1_writer_itself_refuses_a_version_2_lock(tmp_path):
    before = _v2_project(tmp_path)
    with pytest.raises(ValueError):
        _write_lockfile(str(tmp_path), {"version": 1, "skills": {"pdf": {}}})
    assert (tmp_path / ".decimal" / "skills.lock").read_bytes() == before


def test_an_unreadable_lock_is_left_alone(tmp_path):
    (tmp_path / ".decimal").mkdir()
    (tmp_path / ".decimal" / "skills.lock").write_text("{not json", encoding="utf-8")
    summary = export_skills_to_disk([_skill()], project_root=str(tmp_path))
    assert summary["lockfile"] is None
    assert (tmp_path / ".decimal" / "skills.lock").read_text(encoding="utf-8") == "{not json"
