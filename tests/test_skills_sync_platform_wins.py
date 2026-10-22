"""``decimalai skills sync``: DecimalAI's copy wins a conflict, and your edit is kept.

The CLI sends ``conflict_policy="remote_wins_on_conflict"`` with the checkout's
``install_id``: the platform pushes a body only this checkout changed and pulls anything
DecimalAI changed, marking a pull ``conflict`` when the file holds edits it never stored.
Before such a file is overwritten, it is copied to ``.decimal/backup/``. A platform from
before the rule answers 422 and is asked again with ``newer_wins``; every pull over a
different file then keeps a copy.

Without SKILLS_DIR the sync covers ./skills and the skills you wrote that
``.decimal/skills.toml`` lists (in ``.agents/skills``), never a folder it does not list.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from decimalai import _authored, _keyless
from decimalai.cli.main import cli
from decimalai.skillpkg.project import Project

TONE = ("---\nname: refund-tone\ndescription: How Acme words a refund reply.\n---\n\n"
        "# Refund tone\n\nApologize once, state the refund.\n")
THEIRS = "# Refund tone\n\nEdited on DecimalAI: apologize once, then give the date."


def response(payload, status=200, text=""):
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture
def project(tmp_path, monkeypatch):
    for name in ("DECIMAL_API_KEY", "DECIMALAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / ".agents" / "skills" / "refund-tone"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(TONE, encoding="utf-8")
    project = Project.at(tmp_path)
    _authored.add(project, "refund-tone")
    return project


def sync(client, *args):
    with patch("decimalai.cli.main._make_client", return_value=client):
        return CliRunner().invoke(cli, ["skills", "sync", *args, "--api-key", "k",
                                        "--base-url", "http://localhost:8000"])


def posted(client, call=-1):
    return client._http.post.call_args_list[call].kwargs["json"]


def tone(project) -> Path:
    return project.skills_root / "refund-tone" / "SKILL.md"


def test_the_platform_is_asked_to_let_its_copy_win_a_conflict(project):
    client = MagicMock()
    client._http.post.return_value = response({"actions": [{"name": "refund-tone", "action": "no_change"}]})
    result = sync(client)
    assert result.exit_code == 0, result.output
    body = posted(client)
    assert body["conflict_policy"] == "remote_wins_on_conflict"
    assert body["install_id"], "the platform decides from this checkout's last sync"


def test_without_skills_dir_the_listed_skills_you_wrote_are_synced_and_nothing_else(project):
    other = project.skills_root / "run-tests"
    other.mkdir()
    (other / "SKILL.md").write_text("---\nname: run-tests\ndescription: d\n---\n\n# Run tests\n\nmake test\n")
    client = MagicMock()
    client._http.post.return_value = response({"actions": []})
    result = sync(client)
    assert result.exit_code == 0, result.output
    assert [s["name"] for s in posted(client)["skills"]] == ["refund-tone"]
    assert "Discovered 1 skill(s) under .agents/skills (refund-tone)" in result.output


def test_nothing_listed_and_no_skills_folder_is_nothing_to_sync(project):
    _authored.remove(project, "refund-tone")
    with patch("decimalai.cli.main._make_client") as make:
        result = CliRunner().invoke(cli, ["skills", "sync", "--api-key", "k"])
    make.assert_not_called()
    assert result.exit_code == 0, result.output
    assert "Nothing to sync" in result.output and "decimalai skills add <name>" in result.output


def test_a_conflict_keeps_a_copy_of_your_file_before_decimalais_copy_replaces_it(project):
    client = MagicMock()
    client._http.post.return_value = response({"actions": [{
        "name": "refund-tone", "action": "pulled", "body_markdown": THEIRS,
        "version_number": 3, "conflict": True}]})
    result = sync(client)
    assert result.exit_code == 0, result.output
    assert tone(project).read_text().endswith(THEIRS)
    assert "name: refund-tone" in tone(project).read_text(), "the frontmatter stays"
    (kept,) = list(project.backup_dir.glob("sync-*/refund-tone/SKILL.md"))
    assert kept.read_text() == TONE
    assert "DecimalAI's copy replaced your edits, which are saved at" in result.output
    assert "copy it back and sync again" in result.output
    assert "backup/" in (project.state_dir / ".gitignore").read_text()


def test_a_pull_of_a_file_you_did_not_edit_keeps_no_copy(project):
    client = MagicMock()
    client._http.post.return_value = response({"actions": [{
        "name": "refund-tone", "action": "pulled", "body_markdown": THEIRS,
        "version_number": 3, "conflict": False}]})
    result = sync(client)
    assert result.exit_code == 0, result.output
    assert tone(project).read_text().endswith(THEIRS)
    assert not project.backup_dir.exists()


def test_a_platform_from_before_the_rule_is_asked_again_with_newer_wins(project):
    client = MagicMock()
    client._http.post.side_effect = [
        response({"details": {"errors": [{"field": "conflict_policy", "message": "Input should be"}]}},
                 status=422, text='{"error":"conflict_policy: Input should be ..."}'),
        response({"actions": [{"name": "refund-tone", "action": "pulled", "body_markdown": THEIRS,
                               "version_number": 3}]}),
    ]
    result = sync(client)
    assert result.exit_code == 0, result.output
    assert [posted(client, i)["conflict_policy"] for i in (0, 1)] == [
        "remote_wins_on_conflict", "newer_wins"]
    (kept,) = list(project.backup_dir.glob("sync-*/refund-tone/SKILL.md"))
    assert kept.read_text() == TONE, "with no conflict flag, any different file is kept"


def test_another_validation_error_is_shown_not_retried(project):
    client = MagicMock()
    client._http.post.return_value = response(
        {"details": {"errors": [{"field": "skills.0.name", "message": "bad name"}]}},
        status=422, text='{"error":"skills.0.name: bad name"}')
    result = sync(client)
    assert result.exit_code == 1
    assert client._http.post.call_count == 1
    assert "skills.0.name: bad name" in result.output


def test_in_a_connected_project_the_link_records_the_synced_text(project):
    (project.state_dir / "link.toml").write_text(
        'base_url = "https://ws.test"\norg_id = "org-1"\nagents = ["order-support"]\n'
        '[assigned."order-support"]\n"refund-tone" = "authored"\n'
        '[uploaded]\n"refund-tone" = "0000"\n')
    client = MagicMock()
    client._http.post.return_value = response({"actions": [{
        "name": "refund-tone", "action": "pushed", "new_version_number": 2}]})
    result = sync(client)
    assert result.exit_code == 0, result.output
    link = _keyless.read_link(project)
    assert link["uploaded"] == {"refund-tone": _authored.body_hash(project, "refund-tone")}
    assert link["assigned"] == {"order-support": {"refund-tone": "authored"}}, "the rest is kept"
