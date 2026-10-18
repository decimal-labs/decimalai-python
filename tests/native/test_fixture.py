"""The shared native-mode fixture is what the CLI writes, and its update is a real one."""
from __future__ import annotations

from decimalai import _native
from decimalai._native import BY_LOCK, BY_SEEN, BY_STAMP, Load

from .conftest import V2_SENTINEL, V2_VERSION_ID


def test_the_installed_target_resolves_from_its_own_bytes(native_project):
    text = (native_project.skills_dir / native_project.target / "SKILL.md").read_text()
    got = _native.resolve(Load(name=native_project.target, text=text))
    assert got.entry() == native_project.expected() and got.how == BY_STAMP


def test_update_to_v2_moves_the_folder_and_the_lock(native_project):
    v1_text = (native_project.skills_dir / native_project.target / "SKILL.md").read_text()
    v1 = native_project.expected()
    _native.resolve(Load(name=native_project.target, text=v1_text))  # this process saw v1
    v2 = native_project.update_to_v2()
    v2_text = (native_project.skills_dir / native_project.target / "SKILL.md").read_text()
    assert v2["version_id"] == V2_VERSION_ID and V2_SENTINEL in v2_text
    assert v2["body_sha256"] != v1["body_sha256"]
    body_only = v1_text.split("---", 2)[2]
    stale = _native.resolve(Load(name=native_project.target, text=body_only))
    assert (stale.version_id, stale.how) == (v1["version_id"], BY_SEEN)
    fresh = _native.resolve(Load(name=native_project.target, text=v2_text.split("---", 2)[2]))
    assert (fresh.version_id, fresh.how) == (V2_VERSION_ID, BY_LOCK)
