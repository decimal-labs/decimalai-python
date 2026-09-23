"""Run the actual generated check with a real SDK, graph and HTTP trace ingest.

Only the model and semantic judge are stubbed. No changes to generated files.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from decimalai.agent_checks import SUPPORT_SKILLS
from tests.conformance.probe import Probe

ROOT = Path(__file__).resolve().parents[1]

CHILD = '''
import os, sys
from pathlib import Path
from unittest.mock import patch
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from decimalai.agent_checks import run_project_checks
class Model(FakeMessagesListChatModel):
    model_name: str = "fixture-model"
    @property
    def _identifying_params(self):
        return {"model_name": self.model_name}
    def bind_tools(self, *args, **kwargs):
        return self
    def _generate(self, *args, **kwargs):
        if os.environ.get("CHECK_FAILURE") == "quota":
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return super()._generate(*args, **kwargs)
model = Model(responses=[AIMessage(content="A useful sample response from the actual generated graph, deliberately assessed by the stub judge.")])
failure = os.environ.get("CHECK_FAILURE")
if failure in ("repair", "exhausted", "tool_call", "generic_opening"):
    bad = AIMessage(content="I understand the billing concern. We can request a billing review for you.")
    if failure == "tool_call":
        bad = AIMessage(content="", tool_calls=[{"name": "refund", "args": {}, "id": "unavailable"}])
    if failure == "generic_opening":
        bad = AIMessage(content="Thanks for reaching out. Here is a generic draft with enough length to be meaningful.")
    model.responses = [bad] if failure == "exhausted" else [bad, model.responses[0]]
def judge(case, answer, model_name):
    return {"passed": os.environ.get("CHECK_FAILURE") != "wrong_answer"}
with patch("langchain.chat_models.init_chat_model", return_value=model), patch("decimalai.agent_checks.grade_answer", side_effect=judge):
    raise SystemExit(run_project_checks(Path(sys.argv[1])))
'''


@pytest.mark.parametrize("failure", ["none", "quota", "wrong_answer", "missing_body", "missing_delivery", "missing_provider", "trace_rejected", "drift", "repair", "exhausted", "tool_call", "generic_opening"])
def test_actual_generated_project_http(tmp_path, failure):
    pytest.importorskip("langchain")
    class CheckProbe(Probe):
        prompt_reads = 0
        check_results = {}
        ingested = []
        def route(self, method, path, query, body):
            if "/setup/checks" in path:
                if method == "POST":
                    self.check_results[body["id"]] = None
                    return 200, {"id": body["id"], "onboarding_id": "onboarding-fixture", "dashboard_url": "http://localhost/setup"}, []
                if method == "PUT":
                    self.check_results[path.split("/")[-2]] = body
                    passed = (body["configuration_stable"] and body["files_stable"] and len(body["cases"]) == 2
                              and all(c["behavior_passed"] and c["delivery_passed"] and c["export_passed"] for c in body["cases"]))
                    return 200, {"status": "passed" if passed else "failed"}, []
            status, result, errors = super().route(method, path, query, body)
            if failure == "missing_delivery" and path.endswith("/body") and "max_chars" in query:
                result["body"] = ""
            return status, result, errors
        def _ingest_one(self, body):
            self.ingested.append(body)
            if failure == "trace_rejected":
                return 400, {"detail": "deliberate rejection"}, []
            return super()._ingest_one(body)
        def _agent_prompt(self, agent_name, query):
            self.prompt_reads += 1
            status, body, errors = super()._agent_prompt(agent_name, query)
            if failure == "drift" and self.prompt_reads >= 4:
                body["content_hash"] = "changed-after-running"
            return status, body, errors
    probe = CheckProbe().start()
    try:
        probe.register_agent("support", system_prompt="Draft useful support replies. No action tools.", skills=[
            {"name": n, "description": "Support guidance", "body": "" if failure == "missing_body" else f"# {n}\nSpecific instruction for {n}."}
            for n in SUPPORT_SKILLS
        ])
        env = dict(os.environ, PYTHONPATH=str(ROOT), DECIMAL_API_KEY=probe.api_key,
                   OPENAI_API_KEY="fixture-not-a-real-key", CHECK_FAILURE=failure)
        if failure == "missing_provider":
            env.pop("OPENAI_API_KEY")
        dest = tmp_path / "project"
        scaffold = subprocess.run([sys.executable, "-m", "decimalai.cli.main", "init", "support", "--project", str(dest), "--base-url", probe.base_url],
                                  env=env, cwd=tmp_path, capture_output=True, text=True, timeout=30)
        assert scaffold.returncode == 0, scaffold.stdout + scaffold.stderr
        source = (dest / "agent.py").read_bytes()
        check = subprocess.run([sys.executable, "-c", CHILD, str(dest)], env=env, cwd=tmp_path,
                               capture_output=True, text=True, timeout=45)
        receipt = json.loads((dest / "check-results.json").read_text())
        success = failure in ("none", "repair", "tool_call", "generic_opening")
        assert check.returncode == (0 if success else 1), check.stdout + check.stderr + str(receipt)
        assert receipt["passed"] is success
        assert (dest / "agent.py").read_bytes() == source
        if success:
            assert len(receipt["cases"]) == 2
            assert {r["run_id"] for r in receipt["cases"]} == probe.trace_ids
            assert all(r["skill_delivery"]["passed"] for r in receipt["cases"])
            assert receipt["check_id"] in probe.check_results
            assert all(c["input_hashes"] for c in probe.check_results[receipt["check_id"]]["cases"])
            assert "answer" not in json.dumps(probe.check_results[receipt["check_id"]])
            assert all(len(r["input_hashes"]) == (1 if failure == "none" else 2) for r in receipt["cases"])
            assert all("We can request" not in r["answer"] for r in receipt["cases"])
            if failure == "repair":
                assert all("Revise the previous draft" in json.dumps(t["llm_calls"][-1]["rendered_input"])
                           for t in probe.ingested)
        elif failure == "exhausted":
            assert len(probe.ingested) == 2
            assert all(t["status"] == "error" and len(t["llm_calls"]) == 3 for t in probe.ingested)
            assert all("answer" not in r and "bounded revisions" in r["error"] for r in receipt["cases"])
        elif failure == "quota":
            assert all("provider_quota" in r["error"] for r in receipt["cases"])
            assert all(len(r["run_ids"]) == 1 for r in receipt["cases"])
        elif failure == "drift":
            assert not receipt["configuration_stable"]
        elif failure == "missing_provider":
            assert "Missing OPENAI_API_KEY" in receipt["error"]
        elif failure == "missing_delivery":
            assert all(not r["skill_delivery"]["passed"] for r in receipt["cases"])
            assert all(r["behavior"]["passed"] for r in receipt["cases"])
        elif failure == "wrong_answer":
            assert all(not r["behavior"]["passed"] for r in receipt["cases"])
            assert all(r["skill_delivery"]["passed"] for r in receipt["cases"])
        elif failure == "trace_rejected":
            assert all(not r["trace_export"]["passed"] for r in receipt["cases"])
    finally:
        probe.stop()
