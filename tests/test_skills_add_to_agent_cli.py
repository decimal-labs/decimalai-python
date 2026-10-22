"""``decimalai skills add`` / ``remove``: one command for the project's list and the agent's.

Before a project is connected to DecimalAI, ``add`` puts a skill on the project's list in
``.decimal/skills.toml``: a registry skill (the package manager installs it) or a skill
you wrote in ``.agents/skills`` (listed under ``[authored]``). Once the project is
connected (``.decimal/link.toml``), an agent's skills live on DecimalAI and ``add`` puts
the skill on the agent there: a registry skill is assigned, a skill you wrote is uploaded
and added. The project's file does not change then, and ``link.toml`` records what was
given, so the agent's runs do not name it as missing.
"""

from __future__ import annotations

import socket

import pytest
from click.testing import CliRunner

from decimalai import _authored, _keyless
from decimalai.cli.main import cli
from decimalai.skill_router import SkillRouter
from decimalai.skillpkg.project import Project
from tests.fakes.fake_registry import OWNER_KEY, FakeRegistry

TONE = ("---\nname: refund-tone\ndescription: How Acme words a refund reply.\n---\n\n"
        "# Refund tone\n\nApologize once, state the refund, give the date it lands.\n")
LINK = ('base_url = "{url}"\norg_id = "org-1"\norg_slug = "acme"\nagents = [{agents}]\n')


@pytest.fixture
def project(tmp_path, monkeypatch):
    for name in ("DECIMAL_API_KEY", "DECIMALAI_API_KEY", "DECIMALAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    tone = tmp_path / ".agents" / "skills" / "refund-tone"
    tone.mkdir(parents=True)
    (tone / "SKILL.md").write_text(TONE, encoding="utf-8")
    return Project.at(tmp_path)


@pytest.fixture
def registry(monkeypatch):
    reg = FakeRegistry()
    reg.add_skill("acme/order-tracking", versions=2, description="Track a parcel.")
    reg.add_skill("risky/refund-shortcut", band="caution", description="Refund anything.")
    with reg.serve() as url:
        real = socket.socket.connect

        def connect(sock, address):
            if not (isinstance(address, tuple) and address[0] == "127.0.0.1"):
                raise OSError(f"test: no network to {address!r}")
            return real(sock, address)

        monkeypatch.setattr(socket.socket, "connect", connect)
        reg.url = url
        yield reg


@pytest.fixture
def platform(monkeypatch):
    """What the SkillRouter asked the platform: assignments and syncs."""
    asked = {"assigned": [], "synced": []}

    def request(self, method, path, json=None, params=None, **kwargs):
        body = json
        if (method, path) == ("POST", "/api/v1/skills/sync"):
            asked["synced"].append(body)
            names = [s["name"] for s in body["skills"]]
            return {"status": "ok", "created": len(names), "failures": [],
                    "actions": [{"name": n, "action": "created", "new_version_number": 1} for n in names],
                    "added_to_agent": body.get("agent_name"), "added": names, "kept_off": []}
        assert method == "POST" and "/skills/" in path, (method, path)
        asked["assigned"].append((path, dict(params or {})))
        return {"status": "ok", "action": "assigned"}

    monkeypatch.setattr(SkillRouter, "_request", request)
    return asked


def run(*args, key=None, url=None):
    argv = ["skills", *args]
    if key:
        argv += ["--api-key", key]
    if url:
        argv += ["--base-url", url]
    return CliRunner().invoke(cli, argv, catch_exceptions=False)


def connect(project, url="https://ws.test", agents=("order-support",)):
    project.state_dir.mkdir(exist_ok=True)
    (project.state_dir / "link.toml").write_text(
        LINK.format(url=url, agents=", ".join(f'"{a}"' for a in agents)))


def manifest(project) -> str:
    return project.manifest_path.read_text() if project.manifest_path.exists() else ""


# ── before the project is connected ───────────────────────────────────────────


def test_a_skill_you_wrote_is_listed_for_the_projects_agents(project):
    result = run("add", "refund-tone")
    assert result.exit_code == 0, result.output
    assert "listed refund-tone (a skill you wrote) under [authored] in .decimal/skills.toml" in result.output
    assert _authored.listed(project) == ["refund-tone"]
    assert not (project.state_dir / "skills.lock").exists(), "nothing was installed"
    again = run("add", "refund-tone")
    assert "refund-tone is already listed" in again.output


def test_remove_takes_a_skill_you_wrote_off_the_list_and_keeps_its_folder(project):
    run("add", "refund-tone")
    result = run("remove", "refund-tone")
    assert result.exit_code == 0, result.output
    assert "took refund-tone off the list" in result.output
    assert _authored.listed(project) == []
    assert (project.skills_root / "refund-tone" / "SKILL.md").is_file()


def test_a_registry_skill_is_installed_on_the_projects_list(project, registry):
    result = run("add", "acme/order-tracking", url=registry.url)
    assert result.exit_code == 0, result.output
    assert '"acme/order-tracking" = "latest"' in manifest(project)


# ── once the project is connected ─────────────────────────────────────────────


def test_a_registry_skill_goes_on_the_connected_agent_not_the_file(project, registry, platform):
    connect(project, url=registry.url)
    result = run("add", "acme/order-tracking", key=OWNER_KEY, url=registry.url)
    assert result.exit_code == 0, result.output
    skill = next(s for s in registry.skills if s.name == "acme/order-tracking")
    assert platform["assigned"] == [(f"/api/v1/agents/order-support/skills/{skill.skill_id}",
                                     {"pin_version_id": None})]
    assert "added acme/order-tracking to agent 'order-support' on DecimalAI (v2, following the latest)" \
        in result.output
    assert ".decimal/skills.toml is unchanged: agent 'order-support' gets its skills from DecimalAI" \
        in result.output
    assert manifest(project) == "" and not (project.skills_root / "acme-order-tracking").exists()
    assert _keyless.read_link(project)["assigned"]["order-support"] == {"acme/order-tracking": "latest"}


def test_a_pinned_registry_skill_is_assigned_at_that_version(project, registry, platform):
    connect(project, url=registry.url)
    result = run("add", "acme/order-tracking@1", key=OWNER_KEY, url=registry.url)
    assert result.exit_code == 0, result.output
    skill = next(s for s in registry.skills if s.name == "acme/order-tracking")
    (path, params), = platform["assigned"]
    assert params["pin_version_id"] == skill.versions[0].version_id
    assert "(pinned to v1)" in result.output
    assert _keyless.read_link(project)["assigned"]["order-support"] == {"acme/order-tracking": "1"}


def test_a_skill_you_wrote_is_uploaded_and_added_to_the_connected_agent(project, platform):
    connect(project)
    result = run("add", "refund-tone", key="dai_sk_test")
    assert result.exit_code == 0, result.output
    (sync,) = platform["synced"]
    assert [s["name"] for s in sync["skills"]] == ["refund-tone"]
    assert (sync["agent_name"], sync["add_to_agent"], sync["conflict_policy"]) == (
        "order-support", "all", "remote_wins_on_conflict")
    assert "uploaded refund-tone to your organization (not published) and added it to agent " \
           "'order-support' on DecimalAI" in result.output
    link = _keyless.read_link(project)
    assert link["assigned"]["order-support"] == {"refund-tone": "authored"}
    assert link["uploaded"] == {"refund-tone": _authored.body_hash(project, "refund-tone")}
    assert manifest(project) == "", "the project's list is not where the agent's skills live now"


def test_without_a_key_a_connected_project_says_where_the_skills_live(project):
    connect(project)
    result = run("add", "refund-tone")
    assert result.exit_code == 2
    assert "agent 'order-support' gets its skills from DecimalAI" in result.output
    assert "set DECIMAL_API_KEY to add a skill to it" in result.output
    assert manifest(project) == ""


def test_a_project_connected_for_several_agents_asks_which(project, platform):
    connect(project, agents=("billing", "order-support"))
    result = run("add", "refund-tone", key="dai_sk_test")
    assert result.exit_code == 2
    assert "connected to DecimalAI for agents billing, order-support" in result.output
    assert "--to-agent" in result.output
    chosen = run("add", "refund-tone", "--to-agent", "billing", key="dai_sk_test")
    assert chosen.exit_code == 0, chosen.output
    assert platform["synced"][0]["agent_name"] == "billing"


def test_to_agent_names_an_agent_in_a_project_that_is_not_connected(project, registry, platform):
    result = run("add", "acme/order-tracking", "--to-agent", "support", key=OWNER_KEY, url=registry.url)
    assert result.exit_code == 0, result.output
    assert platform["assigned"][0][0].startswith("/api/v1/agents/support/skills/")
    assert not (project.state_dir / "link.toml").exists(), "connecting is agent_config()'s job"
    assert manifest(project) == ""


def test_a_skill_whose_review_is_not_clean_needs_yes(project, registry, platform):
    connect(project, url=registry.url)
    refused = run("add", "risky/refund-shortcut", key=OWNER_KEY, url=registry.url)
    assert refused.exit_code == 7 and "pass --yes" in refused.output, "the trust gate's exit code"
    assert platform["assigned"] == []
    accepted = run("add", "risky/refund-shortcut", "--yes", key=OWNER_KEY, url=registry.url)
    assert accepted.exit_code == 0, accepted.output
    assert len(platform["assigned"]) == 1


def test_remove_in_a_connected_project_says_the_agents_keep_the_skill(project):
    run("add", "refund-tone")
    connect(project)
    result = run("remove", "refund-tone")
    assert result.exit_code == 0, result.output
    assert "agent order-support keep it on DecimalAI: remove it from an agent in the dashboard" \
        in result.output
