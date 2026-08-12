"""Tests for export body resolution and install idempotency.

Two ways a skill install could report success and leave the user with nothing:

* Empty body: ``GET /api/v1/skills/{name}`` serializes no ``body_markdown``, so
  ``export_to_disk`` reconstructed SKILL.md around ``skill.get("body_markdown",
  "")`` — an EMPTY body. Install looked successful and delivered nothing.
  Fix: use ``body_markdown`` when the backend sends it, else resolve the
  current version body via the versions endpoint, else FAIL LOUDLY.
* Skipped export: the fork 409 ("already forked in this org") carries the existing fork's
  name in ``X-Installed-As`` — the SDK raised before reading it and skipped
  the disk export, so "already installed" meant "nothing on disk". Fix:
  install() reads the header and proceeds to export, always converging to
  files-on-disk.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from decimalai.disk_export import SkillExportFileExistsError
from decimalai.skill_router import SkillRouter, SkillRouterError


def _router():
    return SkillRouter(api_key="dai_sk_test", base_url="http://localhost:8000")


def _read_skill_md(root: str, dirname: str) -> str:
    path = os.path.join(root, ".agents", "skills", dirname, "SKILL.md")
    assert os.path.exists(path), f"expected {path} to exist"
    with open(path, encoding="utf-8") as f:
        return f.read()


# ── body resolution ─────────────────────────────────────────


class TestExportBodyResolution:
    def test_body_markdown_from_skill_response_when_present(self, tmp_path):
        """Newer backends include body_markdown on GET /skills/{name} — use it
        directly, no versions round-trip."""
        router = _router()
        calls = []

        def fake_request(method, path, **kwargs):
            calls.append(path)
            if path == "/api/v1/skills/pdf":
                return {
                    "id": "sk1", "name": "pdf", "description": "d",
                    "body_markdown": "# Body from skill response",
                    "latest_version": {"version_number": 3},
                }
            if path == "/api/v1/skills/sk1/attachments":
                return {"attachments": []}
            raise AssertionError(f"unexpected request: {method} {path}")

        with patch.object(router, "_request", side_effect=fake_request):
            summary = router.export_to_disk(skills=["pdf"], project_root=str(tmp_path))

        assert summary["skills_written"] == 1
        assert summary["errors"] == []
        content = _read_skill_md(str(tmp_path), "pdf")
        assert "# Body from skill response" in content
        assert not any("/versions/" in p for p in calls)

    def test_body_falls_back_to_version_endpoint(self, tmp_path):
        """Backends without body_markdown on the skill response (the shape seen
        in production): resolve via latest_version → GET /versions/{n}."""
        router = _router()

        def fake_request(method, path, **kwargs):
            if path == "/api/v1/skills/pdf":
                return {
                    "id": "sk1", "name": "pdf", "description": "d",
                    "latest_version": {"version_number": 2},
                }
            if path == "/api/v1/skills/pdf/versions/2":
                return {"body_markdown": "# Body from version 2", "version_number": 2}
            if path == "/api/v1/skills/sk1/attachments":
                return {"attachments": []}
            raise AssertionError(f"unexpected request: {method} {path}")

        with patch.object(router, "_request", side_effect=fake_request):
            summary = router.export_to_disk(skills=["pdf"], project_root=str(tmp_path))

        assert summary["skills_written"] == 1
        content = _read_skill_md(str(tmp_path), "pdf")
        assert "# Body from version 2" in content
        # The lockfile pins the version the body actually came from.
        with open(tmp_path / ".decimal" / "skills.lock", encoding="utf-8") as f:
            lock = json.load(f)
        assert lock["skills"]["pdf"]["version"] == 2

    def test_body_falls_back_via_list_versions(self, tmp_path):
        """No latest_version on the skill response either — newest from the
        versions listing wins."""
        router = _router()

        def fake_request(method, path, **kwargs):
            if path == "/api/v1/skills/pdf":
                return {"id": "sk1", "name": "pdf", "description": "d"}
            if path == "/api/v1/skills/pdf/versions":
                return {"versions": [{"version_number": 5}, {"version_number": 4}]}
            if path == "/api/v1/skills/pdf/versions/5":
                return {"body_markdown": "# Newest body"}
            if path == "/api/v1/skills/sk1/attachments":
                return {"attachments": []}
            raise AssertionError(f"unexpected request: {method} {path}")

        with patch.object(router, "_request", side_effect=fake_request):
            summary = router.export_to_disk(skills=["pdf"], project_root=str(tmp_path))

        assert summary["skills_written"] == 1
        assert "# Newest body" in _read_skill_md(str(tmp_path), "pdf")

    def test_unresolvable_body_fails_loudly_and_writes_nothing(self, tmp_path):
        """Never silently write an empty SKILL.md — an install that reports
        success and delivers an empty file is worse than a loud failure."""
        router = _router()

        def fake_request(method, path, **kwargs):
            if path == "/api/v1/skills/pdf":
                return {
                    "id": "sk1", "name": "pdf", "description": "d",
                    "latest_version": {"version_number": 1},
                }
            if path == "/api/v1/skills/pdf/versions/1":
                return {"body_markdown": ""}  # version body ALSO empty
            raise AssertionError(f"unexpected request: {method} {path}")

        with patch.object(router, "_request", side_effect=fake_request):
            with pytest.raises(SkillRouterError) as ei:
                router.export_to_disk(skills=["pdf"], project_root=str(tmp_path))

        assert "pdf" in str(ei.value)
        assert "empty SKILL.md" in str(ei.value)
        assert not os.path.exists(tmp_path / ".agents" / "skills" / "pdf" / "SKILL.md")

    def test_partial_failure_writes_good_and_reports_bad(self, tmp_path):
        router = _router()

        def fake_request(method, path, **kwargs):
            if path == "/api/v1/skills/good":
                return {
                    "id": "skg", "name": "good", "description": "d",
                    "body_markdown": "# Good body",
                }
            if path == "/api/v1/skills/skg/attachments":
                return {"attachments": []}
            if path == "/api/v1/skills/bad":
                return {"id": "skb", "name": "bad", "description": "d"}
            if path == "/api/v1/skills/bad/versions":
                return {"versions": []}
            raise AssertionError(f"unexpected request: {method} {path}")

        with patch.object(router, "_request", side_effect=fake_request):
            summary = router.export_to_disk(
                skills=["good", "bad"], project_root=str(tmp_path),
            )

        assert summary["skills_written"] == 1
        assert [e["skill"] for e in summary["errors"]] == ["bad"]
        assert "# Good body" in _read_skill_md(str(tmp_path), "good")


# ── install idempotency on the fork 409 ─────────────────────


def _search_hit():
    return {"items": [{"id": "reg1", "name": "pdf"}]}


def _conflict(headers):
    return SkillRouterError(
        "SkillRouter request failed (409): POST /api/v1/registry/skills/reg1/fork "
        "— Already forked as 'pdf-2'",
        status_code=409,
        headers=headers,
    )


class TestInstallIdempotency:
    def test_fork_409_reads_x_installed_as_and_exports(self):
        router = _router()

        def fake_request(method, path, **kwargs):
            if path == "/api/v1/registry/skills":
                return _search_hit()
            if path == "/api/v1/registry/skills/reg1/fork":
                raise _conflict({"x-installed-as": "pdf-2"})
            raise AssertionError(f"unexpected request: {method} {path}")

        export = {"skills_written": 1, "attachments_written": 0, "paths": ["p"], "errors": []}
        with patch.object(router, "_request", side_effect=fake_request):
            with patch.object(router, "export_to_disk", return_value=export) as mock_export:
                result = router.install("pdf", agents=["claude-code"])

        # Converged to the EXISTING fork's name, and the disk export ran.
        assert result["skill_name"] == "pdf-2"
        assert mock_export.call_args.kwargs["skills"] == ["pdf-2"]
        assert result["fork"]["status"] == "already_installed"
        assert result["fork"]["installed_as"] == "pdf-2"
        assert result["export"] == export

    def test_fork_409_without_header_falls_back_to_requested_name(self):
        router = _router()

        def fake_request(method, path, **kwargs):
            if path == "/api/v1/registry/skills":
                return _search_hit()
            if path == "/api/v1/registry/skills/reg1/fork":
                raise _conflict(None)
            raise AssertionError(f"unexpected request: {method} {path}")

        with patch.object(router, "_request", side_effect=fake_request):
            with patch.object(
                router, "export_to_disk",
                return_value={"skills_written": 1, "paths": [], "errors": []},
            ) as mock_export:
                result = router.install("pdf")

        assert result["skill_name"] == "pdf"
        assert mock_export.call_args.kwargs["skills"] == ["pdf"]

    def test_fork_409_with_files_already_on_disk_is_converged(self):
        """already-forked + already-on-disk = the state install promises."""
        router = _router()

        def fake_request(method, path, **kwargs):
            if path == "/api/v1/registry/skills":
                return _search_hit()
            if path == "/api/v1/registry/skills/reg1/fork":
                raise _conflict({"x-installed-as": "pdf-2"})
            raise AssertionError(f"unexpected request: {method} {path}")

        with patch.object(router, "_request", side_effect=fake_request):
            with patch.object(
                router, "export_to_disk",
                side_effect=SkillExportFileExistsError("/x/SKILL.md"),
            ):
                result = router.install("pdf")

        assert result["skill_name"] == "pdf-2"
        assert result["export"]["already_on_disk"] is True
        assert result["export"]["skills_written"] == 0

    def test_non_409_fork_failure_still_raises(self):
        router = _router()

        def fake_request(method, path, **kwargs):
            if path == "/api/v1/registry/skills":
                return _search_hit()
            if path == "/api/v1/registry/skills/reg1/fork":
                raise SkillRouterError("boom", status_code=500)
            raise AssertionError(f"unexpected request: {method} {path}")

        with patch.object(router, "_request", side_effect=fake_request):
            with patch.object(router, "export_to_disk") as mock_export:
                with pytest.raises(RuntimeError):
                    router.install("pdf")
        mock_export.assert_not_called()

    def test_request_exposes_response_headers_on_error(self):
        """_request must carry the 409's headers (lowercased) onto the
        exception — install() reads X-Installed-As off them."""
        router = _router()
        resp = MagicMock()
        resp.status_code = 409
        resp.json.return_value = {"detail": "Already forked as 'pdf-2'"}
        resp.headers = {"X-Installed-As": "pdf-2", "Content-Type": "application/json"}
        with patch("httpx.request", return_value=resp):
            with pytest.raises(SkillRouterError) as ei:
                router._request("POST", "/api/v1/registry/skills/reg1/fork")
        assert ei.value.status_code == 409
        assert ei.value.headers is not None
        assert ei.value.headers["x-installed-as"] == "pdf-2"
