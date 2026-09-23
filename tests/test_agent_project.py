"""Generated Support project acceptance and failures that must never be green."""
import json
from unittest.mock import patch

import pytest

from decimalai.agent_checks import (
    SUPPORT_SKILLS, check_definition, delivery_result, grade_answer, run_project_checks,
)
from decimalai.cli.project import render_project, write_project
from tests.test_cli_init_scaffold import AGENT, _client, _run_cli


SKILLS = [{"skill_name": name} for name in SUPPORT_SKILLS]


def test_cli_generates_a_complete_project(tmp_path):
    dest = tmp_path / "support"
    result = _run_cli([AGENT, "--api-key", "secret-never-write", "--project", str(dest)], _client(skills=SKILLS))
    assert result.exit_code == 0, result.output
    assert set(p.name for p in dest.iterdir()) == {
        "agent.py", "check_agent.py", "checks.json", "project.json", "requirements.txt", ".env.example", ".gitignore", "README.md",
    }
    for name in ("agent.py", "check_agent.py"):
        compile((dest / name).read_text(), name, "exec")
    assert "secret-never-write" not in "".join(p.read_text() for p in dest.iterdir())
    assert json.loads((dest / "checks.json").read_text())["suite"] == "support-draft/v1"


def test_renamed_fork_uses_source_identity():
    skills = [dict(s) for s in SKILLS]
    skills[0] = {"skill_name": "our-billing-policy", "source_skill_slug": SUPPORT_SKILLS[0]}
    definition = json.loads(render_project(AGENT, skills, None, "http://localhost:8000")["checks.json"])
    assert definition["cases"][0]["required_skills"][0] == "our-billing-policy"


@pytest.mark.parametrize("flags", [["--force"], ["--out", "x.py"], ["--framework", "adk"]])
def test_project_rejects_conflicting_flags_before_http(tmp_path, flags):
    client = _client()
    result = _run_cli([AGENT, "--project", str(tmp_path / "new"), *flags], client)
    assert result.exit_code != 0
    client._http.get.assert_not_called()


def test_project_requires_name():
    assert _run_cli(["--project", "new"]).exit_code != 0


def test_partial_pack_does_not_write_project(tmp_path):
    dest = tmp_path / "new"
    result = _run_cli([AGENT, "--api-key", "test", "--project", str(dest)], _client(skills=SKILLS[:1]))
    assert result.exit_code != 0
    assert not dest.exists()


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_existing_paths_never_clobbered(tmp_path, kind):
    dest = tmp_path / "new"
    if kind == "directory":
        dest.mkdir()
    elif kind == "file":
        dest.write_text("mine")
    else:
        dest.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError):
        write_project(dest, {"agent.py": "new"})
    assert dest.is_symlink() if kind == "symlink" else dest.exists()


def test_dry_run_writes_nothing(tmp_path):
    dest = tmp_path / "new"
    result = _run_cli([AGENT, "--api-key", "test", "--project", str(dest), "--dry-run"], _client(skills=SKILLS))
    assert result.exit_code == 0
    assert "check_agent.py" in result.output
    assert not dest.exists()


CASE = check_definition(dict(zip(SUPPORT_SKILLS, SUPPORT_SKILLS)))["cases"][0]
ANSWER = "A substantive answer must still be judged against every criterion, even when fluent."


def valid_verdict(passed=True):
    return {"criteria": {key: {"passed": passed, "reason": "Specific assessment"} for key in CASE["criteria"]}}


@pytest.mark.parametrize("verdict", [
    {}, {"passed": True}, {"error": "quota_exceeded"}, {"criteria": {}},
    {"criteria": {k: {"passed": "true", "reason": "yes"} for k in CASE["criteria"]}},
    {"criteria": {k: {"passed": True, "reason": ""} for k in CASE["criteria"]}},
    valid_verdict(False),
])
def test_judge_fails_closed(verdict):
    with patch("decimalai.evals.llm_evaluators._call_llm", return_value=verdict):
        assert not grade_answer(CASE, ANSWER, "model")["passed"]


def test_one_failed_criterion_fails_the_case():
    verdict = valid_verdict()
    verdict["criteria"]["honest_actions"]["passed"] = False
    with patch("decimalai.evals.llm_evaluators._call_llm", return_value=verdict):
        assert not grade_answer(CASE, ANSWER, "model")["passed"]


def test_valid_judge_and_empty_answer():
    with patch("decimalai.evals.llm_evaluators._call_llm", return_value=valid_verdict()) as judge:
        assert grade_answer(CASE, ANSWER, "model")["passed"]
        judge.reset_mock()
        assert not grade_answer(CASE, "", "model")["passed"]
        judge.assert_not_called()


def test_delivery_requires_actual_body_and_prompt_not_menu_names():
    snapshot = {"prompt": {"system_prompt": "System rules"}, "skills": [
        {"name": "billing", "body": "\nActual billing knowledge\n", "version": 2, "content_hash": "abc", "body_hash": "abc"},
    ]}
    trace = {"skills_delivered": ["billing"], "llm_calls": [{"rendered_input": [
        {"content": "System rules\nMenu: billing"},
    ]}]}
    assert not delivery_result(trace, snapshot, ["billing"])["passed"]
    trace["llm_calls"][0]["rendered_input"].append({"content": "Actual billing knowledge"})
    assert delivery_result(trace, snapshot, ["billing"])["passed"]
    trace["skills_delivered"] = []
    assert not delivery_result(trace, snapshot, ["billing"])["passed"]


def test_missing_key_overwrites_previous_success(tmp_path, monkeypatch):
    write_project(tmp_path / "project", render_project(AGENT, SKILLS, None, "http://localhost:8000"))
    directory = tmp_path / "project"
    (directory / "check-results.json").write_text('{"passed": true}')
    monkeypatch.delenv("DECIMAL_API_KEY", raising=False)
    monkeypatch.delenv("DECIMALAI_API_KEY", raising=False)
    with patch("dotenv.load_dotenv"):
        assert run_project_checks(directory) == 1
    receipt = json.loads((directory / "check-results.json").read_text())
    assert receipt["passed"] is False
    assert "Missing DECIMAL_API_KEY" in receipt["error"]


def test_modified_check_cannot_weaken_acceptance(tmp_path):
    directory = tmp_path / "project"
    write_project(directory, render_project(AGENT, SKILLS, None, "http://localhost:8000"))
    checks = json.loads((directory / "checks.json").read_text())
    checks["cases"] = []
    (directory / "checks.json").write_text(json.dumps(checks))
    assert run_project_checks(directory) == 1
    receipt = json.loads((directory / "check-results.json").read_text())
    assert not receipt["passed"] and "modified" in receipt["error"]


def test_resume_reuses_exact_id_and_never_imports_agent(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from decimalai.agent_checks import digest
    (tmp_path / 'agent.py').write_text('raise AssertionError("resume must not run the agent")')
    receipt = {'check_id': 'exact-check', 'agent_name': 'support', 'base_url': 'http://localhost:8000',
               'dashboard_url': 'http://localhost/setup', 'remote_result': {'suite': 'support-draft/v1'},
               'files': {'agent.py': digest((tmp_path / 'agent.py').read_text())}, 'passed': False}
    (tmp_path / 'check-results.json').write_text(json.dumps(receipt))
    client = MagicMock()
    client._http.put.return_value.json.return_value = {'status': 'pending_trace'}
    client._http.get.return_value.json.return_value = {'status': 'passed'}
    monkeypatch.setenv('DECIMAL_API_KEY', 'fixture')
    with patch('decimalai._client.DecimalAIClient', return_value=client), patch('decimalai.agent_checks.time.sleep'):
        assert run_project_checks(tmp_path, resume=True, wait_seconds=5) == 0
    client._http.put.assert_called_once_with('/api/v1/agents/support/setup/checks/exact-check/result', json=receipt['remote_result'])
    assert json.loads((tmp_path / 'check-results.json').read_text())['passed']
    (tmp_path / 'agent.py').write_text('changed')
    with patch('decimalai._client.DecimalAIClient') as factory:
        assert run_project_checks(tmp_path, resume=True) == 1
        factory.assert_not_called()


def test_pending_confirmation_is_not_a_pass(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    receipt = {'check_id': 'exact-check', 'agent_name': 'support', 'base_url': 'http://localhost:8000',
               'dashboard_url': 'http://localhost/setup', 'remote_result': {'suite': 'support-draft/v1'},
               'files': {}, 'passed': False}
    (tmp_path / 'check-results.json').write_text(json.dumps(receipt))
    client = MagicMock()
    client._http.put.return_value.json.return_value = {'status': 'pending_trace'}
    monkeypatch.setenv('DECIMAL_API_KEY', 'fixture')
    with patch('decimalai._client.DecimalAIClient', return_value=client):
        assert run_project_checks(tmp_path, resume=True, wait_seconds=0) == 1
    client._http.get.assert_not_called()
    assert not json.loads((tmp_path / 'check-results.json').read_text())['passed']


@pytest.mark.parametrize('promise', ["we’ll review it right away", "I'll have our billing team review it", "we can escalate this case", "I will send the deletion link", "we can request a billing review", "I have refunded the payment", "I can begin account recovery", "we can take another look"] )
def test_draft_action_promises_cannot_be_overruled_by_a_lenient_judge(promise):
    with patch('decimalai.evals.llm_evaluators._call_llm', return_value=valid_verdict()) as judge:
        result = grade_answer(CASE, f'I understand the duplicate-charge concern. If you reply, {promise}.', 'model')
    assert not result['passed']
    assert not result['criteria']['honest_actions']['passed']
    judge.assert_not_called()


def test_customer_request_and_text_guidance_are_not_action_promises():
    answer = "I understand the two $29 entries. You can request a billing review. I can explain the steps, but I cannot initiate it."
    with patch('decimalai.evals.llm_evaluators._call_llm', return_value=valid_verdict()) as judge:
        assert grade_answer(CASE, answer, 'model')['passed']
    judge.assert_called_once()
