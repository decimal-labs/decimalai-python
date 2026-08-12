"""Exporting a skill to disk no longer requires owning a copy of it.

`install()` forks and then writes files, so wanting a skill on disk forced you
to take a copy. The disk writer never cared who owned the skill -- it wants a
name, a description and a body. It was the FETCH that forced the fork:
`GET /skills/{name}` resolves via `resolve_org_skill(org_id=...)`, and a linked
skill (a Use pointer at a public skill owned by the publisher's org) is not in
your org.

`GET /skills/{name}/body` runs the Use/Fork resolver instead, so it answers for
both. It now carries `skill_id` and `description` too, which is what makes it
sufficient on its own.

Attachments have the same split: `/api/v1/skills/{id}/attachments` is org-scoped
and 404s for a linked skill, while the public registry serves the same files --
and a Use target is public by definition, so that route covers exactly the cases
the org-scoped one refuses.
"""
from __future__ import annotations

import os
from unittest.mock import patch

from decimalai.skill_router import SkillRouter

BODY = "# Migration review\n\nCheck for ACCESS EXCLUSIVE locks."


def _router():
    return SkillRouter(api_key="dai_sk_test", base_url="http://localhost:8000")


def _skill_md(root: str, dirname: str) -> str:
    path = os.path.join(root, ".agents", "skills", dirname, "SKILL.md")
    assert os.path.exists(path), f"expected {path} to exist"
    with open(path, encoding="utf-8") as f:
        return f.read()


def _body_response(**over):
    base = {
        "name": "sql-review", "version": 3, "body": BODY,
        "truncated": False, "total_chars": len(BODY),
        "skill_id": "sk-linked", "description": "Reviews SQL migrations",
    }
    base.update(over)
    return base


def test_a_linked_skill_exports_without_touching_the_org_scoped_route(tmp_path):
    """The regression this file exists for. No fork, and no GET /skills/{name}:
    for a linked skill that route 404s, which is what used to force the copy."""
    router = _router()
    calls = []

    def fake_request(method, path, **kwargs):
        calls.append(path)
        if path == "/api/v1/skills/sql-review/body":
            return _body_response()
        if path == "/api/v1/skills/sk-linked/attachments":
            raise RuntimeError("404 — org-scoped, cannot see a linked skill")
        if path == "/api/v1/registry/skills/sk-linked/attachments":
            return {"attachments": []}
        raise AssertionError(f"unexpected request: {method} {path}")

    with patch.object(router, "_request", side_effect=fake_request):
        summary = router.export_to_disk(skills=["sql-review"], project_root=str(tmp_path))

    assert summary["skills_written"] == 1
    assert summary["errors"] == []
    assert BODY in _skill_md(str(tmp_path), "sql-review")
    assert "/api/v1/skills/sql-review" not in calls, "fell back to the org-scoped route"
    assert not any("/fork" in p or "/install" in p for p in calls), "forked to export"


def test_attachments_come_from_the_public_registry_when_the_org_route_refuses(tmp_path):
    router = _router()

    def fake_request(method, path, **kwargs):
        if path == "/api/v1/skills/sql-review/body":
            return _body_response()
        if path == "/api/v1/skills/sk-linked/attachments":
            raise RuntimeError("404")
        if path == "/api/v1/registry/skills/sk-linked/attachments":
            return {"attachments": [{"id": "a1", "file_path": "scripts/check.py"}]}
        if path == "/api/v1/registry/skills/sk-linked/attachments/a1":
            return {
                "id": "a1", "file_path": "scripts/check.py",
                "content_text": "print('lock check')",
            }
        raise AssertionError(f"unexpected request: {method} {path}")

    with patch.object(router, "_request", side_effect=fake_request):
        summary = router.export_to_disk(skills=["sql-review"], project_root=str(tmp_path))

    assert summary["attachments_written"] == 1
    written = os.path.join(
        str(tmp_path), ".agents", "skills", "sql-review", "scripts", "check.py",
    )
    assert os.path.exists(written)


def test_an_owned_skill_with_no_attachments_does_not_ask_the_registry(tmp_path):
    """An empty list is a real answer. Only a FAILED listing falls through, or
    every attachment-less skill pays a second round-trip to hear the same
    thing."""
    router = _router()
    calls = []

    def fake_request(method, path, **kwargs):
        calls.append(path)
        if path == "/api/v1/skills/sql-review/body":
            return _body_response(skill_id="sk-owned")
        if path == "/api/v1/skills/sk-owned/attachments":
            return {"attachments": []}
        raise AssertionError(f"unexpected request: {method} {path}")

    with patch.object(router, "_request", side_effect=fake_request):
        router.export_to_disk(skills=["sql-review"], project_root=str(tmp_path))

    assert not any(p.startswith("/api/v1/registry/") for p in calls)


# ── falling back, so an older backend behaves exactly as before ──────


def test_an_older_backend_without_the_export_fields_falls_back(tmp_path):
    router = _router()
    calls = []

    def fake_request(method, path, **kwargs):
        calls.append(path)
        if path == "/api/v1/skills/sql-review/body":
            # Pre-change shape: no skill_id, no description.
            return {"name": "sql-review", "version": 3, "body": BODY}
        if path == "/api/v1/skills/sql-review":
            return {
                "id": "sk-owned", "name": "sql-review", "description": "d",
                "body_markdown": BODY, "latest_version": {"version_number": 3},
            }
        if path == "/api/v1/skills/sk-owned/attachments":
            return {"attachments": []}
        raise AssertionError(f"unexpected request: {method} {path}")

    with patch.object(router, "_request", side_effect=fake_request):
        summary = router.export_to_disk(skills=["sql-review"], project_root=str(tmp_path))

    assert summary["skills_written"] == 1
    assert "/api/v1/skills/sql-review" in calls, "should have fallen back"
    assert BODY in _skill_md(str(tmp_path), "sql-review")


def test_a_truncated_body_is_never_written_to_disk(tmp_path):
    """That route truncates for prompt injection. Fine in a menu, not a file --
    a SKILL.md that ends mid-sentence is worse than a fallback round-trip."""
    router = _router()
    calls = []
    full = BODY + "\n\nAnd the rest of the guidance that got cut."

    def fake_request(method, path, **kwargs):
        calls.append(path)
        if path == "/api/v1/skills/sql-review/body":
            return _body_response(body=BODY, truncated=True, total_chars=len(full))
        if path == "/api/v1/skills/sql-review":
            return {
                "id": "sk-owned", "name": "sql-review", "description": "d",
                "body_markdown": full, "latest_version": {"version_number": 3},
            }
        if path == "/api/v1/skills/sk-owned/attachments":
            return {"attachments": []}
        raise AssertionError(f"unexpected request: {method} {path}")

    with patch.object(router, "_request", side_effect=fake_request):
        router.export_to_disk(skills=["sql-review"], project_root=str(tmp_path))

    assert "the rest of the guidance" in _skill_md(str(tmp_path), "sql-review")
