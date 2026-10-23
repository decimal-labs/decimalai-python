"""`decimalai agents apply-pack` — a role pack on an agent that already exists.

The command reviews first (the dashboard's /setup/preview), shows what changes, asks, then
applies the same changes with the fingerprint it reviewed (/setup/apply). The server owns the
starter text: the CLI sends the pack's slug, never a copy of its prompt.
"""

import json

import httpx
import pytest
from click.testing import CliRunner

from decimalai.cli.main import cli

STARTER = "You are the front-line support agent for <product>.\nTools: <only what your code attaches>."
PACK = {
    "slug": "support",
    "name": "Support",
    "starter_prompt": STARTER,
    "starter_skills": [{"id": "s1"}, {"id": "s2"}, {"id": "s3"}],
    "starter_skills_missing": [],
}


def _review(**over):
    review = {
        "base_fingerprint": "fp1",
        "noop": False,
        "next_version_number": 4,
        "problems": [],
        "notes": [],
        "changes": [
            {"op": "set_prompt_text", "pack": "support", "problem": None},
            {"op": "add_skill", "skill_id": "s1", "skill": "reply-format", "was_on_list": False, "problem": None},
            {"op": "add_skill", "skill_id": "s2", "skill": "refund-policy", "was_on_list": False, "problem": None},
            {"op": "add_skill", "skill_id": "s3", "skill": "tone", "was_on_list": True, "problem": None},
        ],
        "diff": [{"type": "prompt_version", "from": {"version_number": 2}, "to": {"version_number": 3}}],
        "prompt_text": {"pack": "support", "label": "Applied the Support pack", "before": "You help.",
                        "after": STARTER},
    }
    review.update(over)
    return review


APPLIED = {
    "noop": False,
    "version": {"number": 4},
    "applied": [{"op": "set_prompt_text", "pack": "support", "action": "created", "version_number": 3,
                 "served": True}],
    "skipped": [],
    "notes": [],
}


@pytest.fixture
def api(monkeypatch):
    """Route the CLI's real client through a fake backend and record every call."""
    calls = []
    routes = {}

    def handler(request):
        calls.append(request)
        status, body = routes.get((request.method, request.url.path), (404, {"detail": "Not Found"}))
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
    routes[("GET", "/api/v1/registry/packs/support")] = (200, PACK)
    routes[("GET", "/api/v1/skills")] = (200, {"skills": []})
    routes[("POST", "/api/v1/agents/order-support/setup/preview")] = (200, _review())
    routes[("POST", "/api/v1/agents/order-support/setup/apply")] = (200, APPLIED)

    def posted(path_end):
        return [json.loads(c.content) for c in calls if c.method == "POST" and c.url.path.endswith(path_end)]

    return routes, calls, posted


def run(*args, input=None):
    return CliRunner().invoke(cli, ["agents", "apply-pack", *args], input=input)


def test_it_reviews_then_applies_the_pack_by_slug_with_the_reviewed_fingerprint(api):
    routes, _, posted = api
    result = run("order-support", "support", "--yes")
    assert result.exit_code == 0, result.output
    changes = [{"op": "set_prompt_text", "pack": "support"},
               {"op": "add_skill", "skill_id": "s1"}, {"op": "add_skill", "skill_id": "s2"},
               {"op": "add_skill", "skill_id": "s3"}]
    assert posted("/preview") == [{"changes": changes}]
    assert posted("/apply") == [{"changes": changes, "base_fingerprint": "fp1",
                                 "note": "Applied the Support pack"}]
    out = result.output
    assert "Prompt: v2 → v3, the Support starter prompt (2 lines)" in out
    assert "Skills: 2 added, 1 already on the agent" in out and "+ reply-format" in out
    assert "Writes setup v4 on the agent's Versions page." in out
    assert "✓ Applied the Support pack to order-support (setup v4)" in out
    assert "Prompt: saved as v3" in out
    # The blanks the reader fills next, and where.
    assert "<product>" in out and "<only what your code attaches>" in out
    assert "/agents/order-support/prompt" in out


def test_a_skill_the_server_refuses_is_left_out_and_named(api):
    routes, _, posted = api
    review = _review()
    review["changes"][2]["problem"] = "refund-policy: a safety check blocked it."
    routes[("POST", "/api/v1/agents/order-support/setup/preview")] = (200, review)
    result = run("order-support", "support", "--yes")
    assert result.exit_code == 0, result.output
    (applied,) = posted("/apply")
    assert {"op": "add_skill", "skill_id": "s2"} not in applied["changes"]
    assert "✗ Left out: refund-policy: a safety check blocked it." in result.output


def test_a_dry_run_changes_nothing(api):
    _, _, posted = api
    result = run("order-support", "support", "--dry-run")
    assert result.exit_code == 0, result.output
    assert posted("/apply") == []
    assert "Dry run: nothing was changed." in result.output


def test_it_asks_first_and_a_no_changes_nothing(api):
    _, _, posted = api
    result = run("order-support", "support", input="n\n")
    assert result.exit_code == 1
    assert "Apply it?" in result.output and "Nothing was changed." in result.output
    assert posted("/apply") == []
    result = run("order-support", "support", input="y\n")
    assert result.exit_code == 0, result.output
    assert len(posted("/apply")) == 1


def test_an_unknown_agent_points_at_creating_one_from_the_pack(api):
    routes, _, posted = api
    routes[("POST", "/api/v1/agents/order-support/setup/preview")] = (404, {"detail": "No agent named"})
    result = run("order-support", "support", "--yes")
    assert result.exit_code == 1
    assert "decimalai agents create order-support --pack support" in result.output
    assert posted("/apply") == []


def test_an_unknown_pack_lists_the_valid_ones(api):
    routes, _, posted = api
    routes[("GET", "/api/v1/registry/packs/suport")] = (
        404, {"detail": "Unknown pack 'suport'. Valid packs: coding, support"})
    result = run("order-support", "suport", "--yes")
    assert result.exit_code == 1
    assert "✗ Unknown pack 'suport'. Valid packs: coding, support" in result.output
    assert posted("/preview") == []


def test_nothing_to_apply_when_the_agent_already_has_it(api):
    routes, _, posted = api
    review = _review(noop=True, next_version_number=None, diff=[])
    review["prompt_text"]["before"] = STARTER
    for c in review["changes"][1:]:
        c["was_on_list"] = True
    routes[("POST", "/api/v1/agents/order-support/setup/preview")] = (200, review)
    result = run("order-support", "support", "--yes")
    assert result.exit_code == 0, result.output
    assert "already the Support starter, unchanged" in result.output
    assert "Nothing to apply: order-support already has this." in result.output
    assert posted("/apply") == []


def test_new_text_on_a_pinned_agent_is_still_saved_and_says_it_is_not_served(api):
    routes, _, posted = api
    note = "order-support is pinned to prompt v1, so the new text is saved as the next version but not served."
    routes[("POST", "/api/v1/agents/order-support/setup/preview")] = (
        200, _review(noop=True, next_version_number=None, diff=[], notes=[note]))
    routes[("POST", "/api/v1/agents/order-support/setup/apply")] = (200, {
        **APPLIED, "noop": True, "version": None,
        "applied": [{**APPLIED["applied"][0], "served": False}], "notes": [note]})
    result = run("order-support", "support", "--yes")
    assert result.exit_code == 0, result.output
    assert len(posted("/apply")) == 1
    assert f"⚠ {note}" in result.output
    assert "saved as v3 — not served: the agent is pinned to an older version" in result.output


def test_keep_prompt_adds_only_the_skills(api):
    _, _, posted = api
    result = run("order-support", "support", "--keep-prompt", "--yes")
    assert result.exit_code == 0, result.output
    (changes,) = [p["changes"] for p in posted("/apply")]
    assert all(c["op"] == "add_skill" for c in changes)
    assert "Prompt: unchanged (--keep-prompt)" in result.output


def test_no_starters_takes_the_prompt_alone(api):
    _, _, posted = api
    result = run("order-support", "support", "--no-starters", "--yes")
    assert result.exit_code == 0, result.output
    assert posted("/preview") == [{"changes": [{"op": "set_prompt_text", "pack": "support"}]}]


def test_keep_prompt_and_no_starters_together_leave_nothing(api):
    result = run("order-support", "support", "--keep-prompt", "--no-starters")
    assert result.exit_code == 2
    assert "leaves nothing to apply" in result.output


def test_a_change_made_since_the_review_is_reported_not_retried(api):
    routes, _, posted = api
    routes[("POST", "/api/v1/agents/order-support/setup/apply")] = (409, {"detail": {
        "message": "order-support changed since you reviewed it. Review the changes again.",
        "current_version": 5}})
    result = run("order-support", "support", "--yes")
    assert result.exit_code == 1
    assert "changed since you reviewed it" in result.output
    assert "decimalai agents apply-pack order-support support" in result.output
    assert len(posted("/apply")) == 1


def test_a_viewer_is_told_why(api):
    routes, _, _ = api
    routes[("POST", "/api/v1/agents/order-support/setup/apply")] = (
        403, {"detail": "This action requires the editor role."})
    result = run("order-support", "support", "--yes")
    assert result.exit_code == 1
    assert "requires the editor role" in result.output
