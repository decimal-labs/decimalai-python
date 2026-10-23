"""Every client says which agents a skill it creates reaches (owner decision D1, 2026-10-22).

A new skill on DecimalAI reaches only the agents it is added to. Until the platform's contract
step, a create, sync or fork that says nothing still gets "every agent" from an API key; after
it, nothing. So the SDK never says nothing: `create_skill`, `sync_skills` and `fork` send
``offer_scope="restricted"`` unless told otherwise (tests/test_skill_router.py,
tests/test_skills_sync_export.py), and this file covers the rest:

* ``decimalai skills sync`` takes ``--to-agent NAME`` (the name ``skills add`` uses) and ``--offer-scope``, always sends a scope,
  and says where the skills it created went;
* ``sync_to_platform()`` takes both;
* an adapter's boot sync that names no agent says once that its new skills are on none.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

import decimalai.skill_router as sr_mod
from decimalai.cli.main import cli
from decimalai.skills import sync_to_platform

TONE = ("---\nname: refund-tone\ndescription: How Acme words a refund reply.\n---\n\n"
        "# Refund tone\n\nApologize once, state the refund.\n")


def response(payload):
    resp = MagicMock()
    resp.status_code = 200
    resp.text = ""
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture
def skills_dir(tmp_path, monkeypatch):
    for name in ("DECIMAL_API_KEY", "DECIMALAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "skills" / "refund-tone"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(TONE, encoding="utf-8")
    return tmp_path / "skills"


def sync(client, *args):
    with patch("decimalai.cli.main._make_client", return_value=client):
        return CliRunner().invoke(cli, ["skills", "sync", *args, "--api-key", "k",
                                        "--base-url", "http://localhost:8000"])


def posted(client):
    return client._http.post.call_args.kwargs["json"]


CREATED = {"actions": [{"name": "refund-tone", "action": "created", "new_version_number": 1}]}


def test_a_sync_that_names_no_agent_creates_opt_in_skills_and_says_they_are_on_none(skills_dir):
    client = MagicMock()
    client._http.post.return_value = response(CREATED)
    result = sync(client, str(skills_dir))
    assert result.exit_code == 0, result.output
    body = posted(client)
    assert body["offer_scope"] == "restricted"
    assert "agent_name" not in body
    assert "these 1 are on none yet" in result.output
    assert "decimalai skills sync --to-agent NAME" in result.output


def test_a_sync_that_creates_nothing_has_nothing_to_say_about_agents(skills_dir):
    client = MagicMock()
    client._http.post.return_value = response({"actions": [{"name": "refund-tone", "action": "no_change"}]})
    result = sync(client, str(skills_dir))
    assert result.exit_code == 0, result.output
    assert "on none yet" not in result.output


def test_to_agent_puts_the_synced_skills_on_that_agent(skills_dir):
    client = MagicMock()
    client._http.post.return_value = response({**CREATED, "added_to_agent": "support-agent",
                                               "added": ["refund-tone"], "kept_off": []})
    result = sync(client, str(skills_dir), "--to-agent", "support-agent")
    assert result.exit_code == 0, result.output
    body = posted(client)
    assert (body["agent_name"], body["offer_scope"]) == ("support-agent", "restricted")
    # A skill someone took off the agent stays off: a sync run again must not undo that.
    assert body["add_to_agent"] == "unless_removed"
    assert "On support-agent: refund-tone" in result.output
    assert "on none yet" not in result.output


def test_to_agent_names_a_skill_a_person_removed_from_it(skills_dir):
    client = MagicMock()
    client._http.post.return_value = response({
        "actions": [{"name": "refund-tone", "action": "no_change"}],
        "added_to_agent": "support-agent", "added": [], "kept_off": ["refund-tone"],
    })
    result = sync(client, str(skills_dir), "--to-agent", "support-agent")
    assert result.exit_code == 0, result.output
    assert "Left off support-agent (removed there; add it back on the agent's page): refund-tone" in result.output


def test_to_agent_says_when_no_agent_could_be_made(skills_dir):
    client = MagicMock()
    client._http.post.return_value = response({**CREATED, "agent_not_found": "Support Agent",
                                               "agent_problem": "Agent names are slugs."})
    result = sync(client, str(skills_dir), "--to-agent", "Support Agent")
    assert result.exit_code == 0, result.output
    assert "Not added to Support Agent: Agent names are slugs." in result.output


def test_offer_scope_workspace_offers_new_skills_to_every_agent(skills_dir):
    client = MagicMock()
    client._http.post.return_value = response(CREATED)
    result = sync(client, str(skills_dir), "--offer-scope", "workspace")
    assert result.exit_code == 0, result.output
    assert posted(client)["offer_scope"] == "workspace"
    assert "on none yet" not in result.output


def test_offer_scope_takes_only_the_two_values(skills_dir):
    result = sync(MagicMock(), str(skills_dir), "--offer-scope", "org")
    assert result.exit_code != 0
    assert "restricted" in result.output and "workspace" in result.output


def test_a_dry_run_says_which_agents_new_skills_would_reach(skills_dir):
    client = MagicMock()
    result = sync(client, str(skills_dir), "--dry-run", "--to-agent", "support-agent")
    assert result.exit_code == 0, result.output
    client._http.post.assert_not_called()
    assert "New skills would reach: support-agent" in result.output
    result = sync(client, str(skills_dir), "--dry-run")
    assert "New skills would reach: no agent until you add them to one" in result.output


# ── sync_to_platform() ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("kwargs, expected", [
    ({}, (None, "restricted")),
    ({"agent_name": "support-agent"}, ("support-agent", "restricted")),
    ({"offer_scope": "workspace"}, (None, "workspace")),
])
def test_sync_to_platform_says_which_agents_get_new_skills(skills_dir, monkeypatch, kwargs, expected):
    seen = []

    class _Router:
        def __init__(self, **kw):
            pass

        def sync_skills(self, skills, **kw):
            seen.append(kw)
            return {"created": len(skills), "updated": 0, "unchanged": 0}

    monkeypatch.setattr(sr_mod, "SkillRouter", _Router)
    sync_to_platform("dai_sk_test", search_paths=[str(skills_dir)], **kwargs)
    assert [(kw["agent_name"], kw["offer_scope"]) for kw in seen] == [expected]
    assert seen[0]["add_to_agent"] == ("unless_removed" if kwargs.get("agent_name") else None)


# ── an adapter's boot sync with no agent named ──────────────────────────────────


@pytest.fixture
def fresh_warning(monkeypatch):
    monkeypatch.setattr(sr_mod, "_warned_boot_sync_without_agent", False)


def test_a_boot_sync_with_no_agent_says_once_that_its_new_skills_are_on_none(fresh_warning, caplog):
    with caplog.at_level(logging.WARNING, logger="decimalai.skill_router"):
        sr_mod._note_boot_sync_without_agent({"created": 2})
        sr_mod._note_boot_sync_without_agent({"created": 1})
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "2 new skill(s)" in warnings[0].getMessage()
    assert "reach no agent" in warnings[0].getMessage()


@pytest.mark.parametrize("result", [{"created": 0}, {"synced": 3}, None, {"created": "1"}])
def test_a_boot_sync_that_created_nothing_says_nothing(fresh_warning, caplog, result):
    with caplog.at_level(logging.WARNING, logger="decimalai.skill_router"):
        sr_mod._note_boot_sync_without_agent(result)
    assert not caplog.records


def test_the_retired_runtime_flag_is_not_an_agent(skills_dir):
    """`--agent` named a RUNTIME on the 0.14 disk verbs (`--agent claude-code`), retired in 0.15.
    Sync makes the agent it is given, so taking `--agent` would quietly make an agent named
    after a runtime: it is refused, and the flag is `--to-agent`, as on `skills add`."""
    client = MagicMock()
    result = sync(client, str(skills_dir), "--agent", "claude-code")
    assert result.exit_code != 0
    client._http.post.assert_not_called()
