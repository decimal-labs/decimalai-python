"""`decimalai agents create` — the create form's POST, from a terminal."""

import json

import httpx
import pytest
from click.testing import CliRunner

from decimalai.cli.main import cli

PACK = {
    "slug": "support",
    "starter_prompt": "You are a support agent.",
    "starter_skills": [{"id": "s1"}, {"id": "s2"}],
    "starter_skills_missing": [],
}


@pytest.fixture
def api(monkeypatch):
    """Route the CLI's real client through a fake backend and record every call."""
    calls = []
    routes = {}

    def handler(request):
        calls.append(request)
        key = (request.method, request.url.path)
        status, body = routes.get(key, (404, {"detail": "Not Found"}))
        if callable(body):
            body = body(request)
        return httpx.Response(status, json=body)

    original = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )
    monkeypatch.setenv("DECIMAL_API_KEY", "dai_sk_test")
    monkeypatch.setattr("time.sleep", lambda *_: None)

    def created(request):
        body = json.loads(request.content)
        return {"agent_name": body["agent_name"], "agent_id": "a1",
                "skills_assigned": len(body["skill_ids"]), "skills_failed": []}

    routes[("POST", "/api/v1/agents")] = (201, created)
    routes[("GET", "/api/v1/registry/packs/support")] = (200, PACK)
    routes[("GET", "/api/v1/skills")] = (200, {"skills": []})

    def posted():
        return [json.loads(c.content) for c in calls if c.method == "POST"]

    return routes, calls, posted


def run(*args):
    return CliRunner().invoke(cli, ["agents", "create", *args])


def test_a_pack_brings_its_starter_skills_and_leaves_the_prompt_to_the_server(api):
    routes, _, posted = api
    result = run("triage", "--pack", "support")
    assert result.exit_code == 0, result.output
    # No system_prompt: the server then records the pack's starter as v1 with
    # provenance='pack', which is what "kept the starter?" reads later.
    assert posted() == [{"agent_name": "triage", "skill_ids": ["s1", "s2"], "pack": "support"}]
    assert "Created agent triage" in result.output
    assert "Skills attached: 2" in result.output
    assert "decimalai init triage" in result.output


def test_skill_slugs_resolve_to_your_own_skill_before_the_registry(api):
    routes, calls, posted = api
    routes[("GET", "/api/v1/skills")] = (200, lambda r: {"skills": [{"id": "own1"}]}
                                         if r.url.params.get("name") == "my-runbook" else {"skills": []})
    routes[("GET", "/api/v1/registry/skills/first-fluke-oma-pdf")] = (200, {"id": "r1"})
    result = run("reader", "--skill", "my-runbook", "--skill", "first-fluke-oma-pdf")
    assert result.exit_code == 0, result.output
    assert posted() == [{"agent_name": "reader", "skill_ids": ["own1", "r1"]}]
    assert not any(c.url.path == "/api/v1/registry/skills/my-runbook" for c in calls)


def test_a_namespaced_registry_name_resolves_through_an_exact_search_match(api):
    """The detail route can't take `owner/name` in its path; search can, and only
    the exact name counts."""
    routes, _, posted = api
    routes[("GET", "/api/v1/registry/skills")] = (200, {"items": [
        {"name": "borghei/design-auditor", "id": "wrong"},
        {"name": "borghei/data-quality-auditor", "id": "r2"},
    ]})
    result = run("dq", "--skill", "borghei/data-quality-auditor")
    assert result.exit_code == 0, result.output
    assert posted() == [{"agent_name": "dq", "skill_ids": ["r2"]}]


def test_an_unknown_skill_creates_nothing(api):
    """A typo must not leave an agent quietly missing the skill it was asked for,
    and must never be swapped for whatever a search ranks first."""
    routes, _, posted = api
    routes[("GET", "/api/v1/registry/skills")] = (200, {"items": [
        {"name": "minimax-pdf", "id": "near-miss"},
    ]})
    result = run("reader", "--skill", "pdf-procesing")
    assert result.exit_code == 1
    assert "No skill named 'pdf-procesing'" in result.output
    assert "Nothing was created" in result.output
    assert posted() == []


def test_a_prompt_file_replaces_the_starter_and_no_starters_skips_its_skills(api, tmp_path):
    _, calls, posted = api
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Answer refunds only.\n")
    result = run("refunds", "--pack", "support", "--no-starters", "--prompt-file", str(prompt))
    assert result.exit_code == 0, result.output
    assert posted() == [{"agent_name": "refunds", "skill_ids": [],
                         "system_prompt": "Answer refunds only.\n", "pack": "support"}]
    assert not any(c.url.path.startswith("/api/v1/registry/packs") for c in calls)


def test_missing_starters_are_named_before_the_create(api):
    routes, _, posted = api
    routes[("GET", "/api/v1/registry/packs/support")] = (
        200, {**PACK, "starter_skills": [{"id": "s1"}], "starter_skills_missing": ["gone-skill"]})
    result = run("triage", "--pack", "support")
    assert result.exit_code == 0, result.output
    assert "gone-skill" in result.output
    assert posted()[0]["skill_ids"] == ["s1"]


def test_an_unknown_pack_shows_the_valid_list_and_creates_nothing(api):
    routes, _, posted = api
    routes[("GET", "/api/v1/registry/packs/suport")] = (
        404, {"detail": "Unknown pack 'suport'. Valid packs: support, doc-ops"})
    result = run("triage", "--pack", "suport")
    assert result.exit_code == 1
    assert "Valid packs: support, doc-ops" in result.output
    assert posted() == []


def test_a_taken_name_is_one_line_not_a_traceback(api):
    routes, _, _ = api
    routes[("POST", "/api/v1/agents")] = (409, {"detail": "An agent named 'triage' already exists."})
    result = run("triage")
    assert result.exit_code == 1
    assert "already exists" in result.output
    assert "Traceback" not in result.output


def test_a_skill_the_server_refused_is_reported_but_the_agent_stands(api):
    routes, _, _ = api
    routes[("GET", "/api/v1/registry/skills/big")] = (200, {"id": "r9"})
    routes[("POST", "/api/v1/agents")] = (201, {
        "agent_name": "triage", "skills_assigned": 0, "skills_failed": ["r9"],
        "skills_failed_detail": [{"skill_id": "r9", "error": "Linked-skill limit reached",
                                  "error_code": "linked_skills_limit_exceeded"}]})
    result = run("triage", "--skill", "big")
    assert result.exit_code == 0, result.output
    assert "Created agent triage" in result.output
    assert "r9: Linked-skill limit reached" in result.output


def test_no_starters_without_a_pack_is_a_usage_error(api):
    _, calls, _ = api
    result = run("triage", "--no-starters")
    assert result.exit_code == 2
    assert calls == []
