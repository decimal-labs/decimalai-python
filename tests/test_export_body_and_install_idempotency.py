"""An install that reports success leaves the skill on disk, and re-running it converges.

Two ways an install once reported success and left the user with nothing:

* an empty body: the export read a route that served no ``body_markdown`` and wrote a
  SKILL.md around nothing. Now the registry renders the folder and refuses a skill
  with no body (``render_invalid``), so the export fails loudly and writes nothing;
* a skipped export: ``install()`` forked first, and a fork 409 ("already forked")
  skipped the disk write. ``install()`` no longer forks, so there is no 409 to read;
  re-running it, or ``export()``, leaves the installed folder as it is.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from decimalai.skill_router import SkillRouter, SkillRouterError
from decimalai.skillpkg.errors import ExitCode
from tests.fakes.fake_registry import OWNER_ORG, FakeRegistry
from tests.test_router_skillpkg_shims import _fresh_once_state, engine_on, no_router_http, router

__all__ = ["_fresh_once_state"]


def _read_skill_md(root, dirname: str) -> str:
    path = os.path.join(str(root), ".agents", "skills", dirname, "SKILL.md")
    assert os.path.exists(path), f"expected {path} to exist"
    with open(path, encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def registry(monkeypatch):
    reg = FakeRegistry()
    engine_on(reg, monkeypatch)
    no_router_http(monkeypatch)
    return reg


# ── the body ────────────────────────────────────────────────


class TestTheBodyIsTheRegistrys:
    def test_the_folder_holds_the_body_the_registry_serves(self, tmp_path, registry):
        registry.add_skill("pdf", org=OWNER_ORG, public=False, body="# Body from the registry\n")
        summary = router().export("pdf", project_root=str(tmp_path))
        assert summary["skills_written"] == 1
        assert "# Body from the registry" in _read_skill_md(tmp_path, "pdf")

    def test_it_is_the_newest_version_and_the_lock_pins_it(self, tmp_path, registry):
        skill = registry.add_skill("pdf", versions=3)
        summary = router().export("pdf", project_root=str(tmp_path))
        assert summary["lock_entry"]["version"] == 3
        assert summary["lock_entry"]["version_id"] == skill.newest.version_id
        assert "Step 3" in _read_skill_md(tmp_path, "pdf")

    def test_a_skill_with_no_body_fails_loudly_and_writes_nothing(self, tmp_path, registry):
        """Never write an empty SKILL.md: a loud failure beats a success that delivers
        nothing."""
        registry.add_skill("pdf", body="   \n")
        with pytest.raises(SkillRouterError) as ei:
            router().export("pdf", project_root=str(tmp_path))
        assert "pdf" in str(ei.value)
        assert ei.value.result.exit_code == ExitCode.UNAVAILABLE
        assert not os.path.exists(tmp_path / ".agents" / "skills" / "pdf" / "SKILL.md")

    def test_partial_failure_writes_good_and_reports_bad(self, tmp_path, registry):
        registry.add_skill("good", body="# Good body\n")
        summary = router().export_to_disk(skills=["good", "bad"], project_root=str(tmp_path))
        assert summary["skills_written"] == 1
        assert [e["skill"] for e in summary["errors"]] == ["bad"]
        assert "# Good body" in _read_skill_md(tmp_path, "good")


# ── re-running converges ────────────────────────────────────


class TestInstallConverges:
    def test_a_second_install_leaves_the_folder_and_reports_it_installed(self, tmp_path, registry):
        registry.add_skill("pdf")
        r = router()
        first = r.install("pdf", project_root=str(tmp_path))
        before = _read_skill_md(tmp_path, "pdf")

        second = r.install("pdf", project_root=str(tmp_path))

        assert first["export"]["skills_written"] == 1
        assert second["export"]["skills_written"] == 0
        assert second["skill_name"] == "pdf" and second["export"]["errors"] == []
        assert _read_skill_md(tmp_path, "pdf") == before

    def test_a_hand_written_folder_of_that_name_is_not_overwritten(self, tmp_path, registry):
        registry.add_skill("pdf")
        mine = tmp_path / ".agents" / "skills" / "pdf"
        mine.mkdir(parents=True)
        (mine / "SKILL.md").write_text("---\nname: pdf\ndescription: mine\n---\n# Mine\n",
                                       encoding="utf-8")
        summary = router().export("pdf", project_root=str(tmp_path))
        # The registry chose the other form of the name, so both folders exist.
        assert summary["skill_dirname"] != "pdf"
        assert "# Mine" in (mine / "SKILL.md").read_text(encoding="utf-8")

    def test_request_exposes_response_headers_on_error(self):
        """_request carries a non-2xx response's headers (lowercased) onto the exception."""
        r = SkillRouter(api_key="dai_sk_test", base_url="http://localhost:8000")
        resp = MagicMock()
        resp.status_code = 409
        resp.json.return_value = {"detail": "Already forked as 'pdf-2'"}
        resp.headers = {"X-Installed-As": "pdf-2", "Content-Type": "application/json"}
        with patch("httpx.request", return_value=resp):
            with pytest.raises(SkillRouterError) as ei:
                r._request("POST", "/api/v1/registry/skills/reg1/fork")
        assert ei.value.status_code == 409
        assert ei.value.headers is not None
        assert ei.value.headers["x-installed-as"] == "pdf-2"
