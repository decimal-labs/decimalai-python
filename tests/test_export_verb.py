"""`export()` writes the files and takes no copy.

`install()` does a fork and then a disk write, so asking for a file also took an
editable copy of the skill. Nothing about writing a SKILL.md needs ownership —
the writer wants a name, a description and a body. So the file half is its own
verb now, and Install (link) / Fork (copy) are the verbs for acquiring the skill
itself.

`install()` is unchanged and still forks. Deprecating a shape is not the same as
breaking it.
"""
from __future__ import annotations

import os
from unittest.mock import patch

from decimalai.skill_router import SkillRouter

BODY = "# Review\n\nCheck the locks."


def _router():
    return SkillRouter(api_key="dai_sk_test", base_url="http://localhost:8000")


def _stub(calls):
    def fake_request(method, path, **kwargs):
        calls.append((method, path))
        if path == "/api/v1/skills/pdf/body":
            return {
                "name": "pdf", "version": 1, "body": BODY, "truncated": False,
                "total_chars": len(BODY), "skill_id": "sk1", "description": "d",
            }
        if path == "/api/v1/skills/sk1/attachments":
            return {"attachments": []}
        if path.endswith("/fork"):
            return {"status": "installed", "skill": {"id": "sk1", "name": "pdf"}}
        if path == "/api/v1/registry/skills":
            return {"items": [{"id": "sk1", "name": "pdf"}]}
        raise AssertionError(f"unexpected request: {method} {path}")
    return fake_request


def test_export_writes_the_file_and_forks_nothing(tmp_path):
    router, calls = _router(), []
    with patch.object(router, "_request", side_effect=_stub(calls)):
        summary = router.export("pdf", project_root=str(tmp_path))

    assert summary["skills_written"] == 1
    path = os.path.join(str(tmp_path), ".agents", "skills", "pdf", "SKILL.md")
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        assert BODY in f.read()

    assert not any(p.endswith(("/fork", "/install")) for _, p in calls), (
        "export took a copy — that is the whole thing it exists not to do"
    )


def test_install_still_forks(tmp_path):
    """The deprecated shape keeps its behaviour. A deprecation that silently
    changes what a published method DOES is worse than no deprecation."""
    router, calls = _router(), []
    with patch.object(router, "_request", side_effect=_stub(calls)):
        router.install("pdf", project_root=str(tmp_path))

    assert any(p.endswith("/fork") for _, p in calls), "install stopped forking"


def test_export_and_install_write_the_same_bytes(tmp_path):
    """The only difference is the copy, not the file."""
    a, b = tmp_path / "via-export", tmp_path / "via-install"
    a.mkdir(); b.mkdir()

    router = _router()
    with patch.object(router, "_request", side_effect=_stub([])):
        router.export("pdf", project_root=str(a))
    with patch.object(router, "_request", side_effect=_stub([])):
        router.install("pdf", project_root=str(b))

    def _read(root):
        with open(os.path.join(str(root), ".agents", "skills", "pdf", "SKILL.md"),
                  encoding="utf-8") as f:
            return f.read()

    assert _read(a) == _read(b)
