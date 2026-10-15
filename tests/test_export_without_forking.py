"""Exporting a skill to disk never requires owning a copy of it.

`export()` once had to fork a linked skill (a Use pointer at a public skill another
org owns) because the routes it read were scoped to your org. Then it read `/body`,
which runs the resolver. Now it reads neither: it is `decimalai skills add`, whose
resolve and bundle routes answer for whatever the key may install, and whose bundle
carries the attachments with their hashes, checked before anything is written.
"""
from __future__ import annotations

import os

import pytest

from decimalai.skill_router import SkillRouterError
from decimalai.skillpkg.errors import ExitCode
from tests.fakes.fake_registry import OTHER_ORG, OWNER_ORG, FakeRegistry
from tests.test_router_skillpkg_shims import _fresh_once_state, engine_on, no_router_http, router

__all__ = ["_fresh_once_state"]

BODY = "# Migration review\n\nCheck for ACCESS EXCLUSIVE locks.\n"
FILES = ({"path": "scripts/check.py", "content": "print('lock check')\n", "executable": True},)


def _skill_md(root, dirname: str) -> str:
    path = os.path.join(str(root), ".agents", "skills", dirname, "SKILL.md")
    assert os.path.exists(path), f"expected {path} to exist"
    with open(path, encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def registry(monkeypatch):
    reg = FakeRegistry()
    engine_on(reg, monkeypatch)
    return reg


def test_a_linked_skill_exports_without_a_copy_or_a_body_call(tmp_path, registry, monkeypatch):
    made = no_router_http(monkeypatch)
    skill = registry.add_skill("sql-review", org=OTHER_ORG, body=BODY, files=FILES)
    registry.use(skill, OWNER_ORG)

    summary = router().export("sql-review", project_root=str(tmp_path))

    assert summary["skills_written"] == 1 and summary["errors"] == []
    assert BODY.strip() in _skill_md(tmp_path, "sql-review")
    assert made == [], "forked, or read /body, to export"
    assert registry.calls("body") == []


def test_attachments_come_in_the_bundle_with_their_mode(tmp_path, registry, monkeypatch):
    no_router_http(monkeypatch)
    registry.add_skill("sql-review", org=OTHER_ORG, body=BODY, files=FILES)

    summary = router().export("sql-review", project_root=str(tmp_path))

    written = tmp_path / ".agents" / "skills" / "sql-review" / "scripts" / "check.py"
    assert summary["attachments_written"] == 1
    assert written.read_text(encoding="utf-8") == "print('lock check')\n"
    if os.name == "posix":
        assert os.stat(written).st_mode & 0o777 == 0o755
    assert [c.route for c in registry.calls()] == ["resolve", "bundle"]


def test_an_older_registry_is_refused_never_a_fallback_to_body(tmp_path, monkeypatch):
    no_router_http(monkeypatch)
    old = FakeRegistry(old=True)
    engine_on(old, monkeypatch)
    old.add_skill("sql-review", body=BODY)

    with pytest.raises(SkillRouterError) as ei:
        router().export("sql-review", project_root=str(tmp_path))

    assert ei.value.result.exit_code == ExitCode.UNAVAILABLE
    assert old.calls("body") == []
    assert not (tmp_path / ".agents").exists()


def test_bytes_that_differ_from_the_declared_hashes_are_never_written(
        tmp_path, registry, monkeypatch):
    """Where `/body` once served a truncated body that could end up in a file, every
    byte is now checked against the resolve answer before the folder moves in."""
    no_router_http(monkeypatch)
    registry.add_skill("sql-review", body=BODY)

    def cut(bundle):
        bundle["files"][0]["content"] = bundle["files"][0]["content"][:40]

    registry.tamper = cut
    with pytest.raises(SkillRouterError) as ei:
        router().export("sql-review", project_root=str(tmp_path))
    assert ei.value.result.exit_code == ExitCode.INTEGRITY
    assert not (tmp_path / ".agents" / "skills" / "sql-review").exists()
