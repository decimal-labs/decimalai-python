"""The one translation point: a trace the SDK built reaches the platform with skill ids.

``DecimalAIClient.ingest_trace`` and ``ingest_traces_batch`` (which ``flush`` uses) run
every payload through ``decimalai.skillpkg.identity.translate_payload`` just before the
POST, so a skill a native loader reported by its install name, its Claude Code plugin name
or its path arrives as the registry skill and version it is. A payload the caller built
(``ingest_raw_trace``, ``ingest_raw_traces_batch``) is sent exactly as given. No other
module posts to the trace endpoints, so nothing can go around the translation.

Requests are captured with ``httpx.MockTransport``: these assertions are on the bytes that
would go over the wire.
"""
from __future__ import annotations

import ast
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pytest
from skillevaluation import skillfmt

import decimalai
from decimalai._client import DecimalAIClient
from decimalai.schema.trace import RunTrace
from decimalai.skillpkg import identity
from decimalai.skillpkg import lock as L
from decimalai.skillpkg import project as P

PACKAGE_DIR = Path(decimalai.__file__).resolve().parent

SKILL_ID = str(uuid.UUID(int=1))
VERSION_ID = str(uuid.UUID(int=101))
BODY = "# Refunds\n\nThe refund window is thirty days from delivery.\n"


def _install(directory: Path) -> None:
    """One managed skill, as `decimalai skills add acme/refund-policy` leaves it."""
    fields = skillfmt.RenderFields(
        skill_id=SKILL_ID, version_id=VERSION_ID, version=3,
        registry_name="acme/refund-policy", body=BODY, description="Apply Acme refund rules.")
    bundle = skillfmt.render_bundle(fields, [], install_name="acme-refund-policy",
                                    variant="foreign")
    project = P.Project.at(directory)
    folder = project.skill_dir(bundle.install_name)
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_bytes(bundle.skill_md.encode("utf-8"))
    entry = L.make_entry(
        bundle.install_name, name="acme/refund-policy", skill_id=SKILL_ID,
        version_id=VERSION_ID, version=3,
        render={"format": bundle.format, "hash": bundle.render_hash, "variant": bundle.variant},
        body_sha256=bundle.body_sha256, description_sha256=bundle.description_sha256,
        files={f.path: f.sha256 for f in bundle.files},
        source={"base_url": "https://api.example.test", "org": None},
        invocation=bundle.invocation, installed_at="2026-10-20T10:00:00Z")
    lock = L.empty().with_entry(bundle.install_name, entry)
    L.save(project, lock)
    L.write_plugin_json(project, lock.lock_hash)


@pytest.fixture(autouse=True)
def _fresh_index_cache():
    identity.clear_cache()
    yield
    identity.clear_cache()


@pytest.fixture
def managed_project(tmp_path, monkeypatch) -> Path:
    (tmp_path / ".git").mkdir()
    _install(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


class _Wire:
    """A client whose transport records every request instead of sending it."""

    def __init__(self) -> None:
        self.requests: List[httpx.Request] = []
        self.client = DecimalAIClient(api_key="dai_sk_test", base_url="http://wire.test")
        self.client._http = httpx.Client(base_url="http://wire.test",
                                         transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/api/v1/traces/batch":
            return httpx.Response(200, json={"count": len(json.loads(request.content))})
        return httpx.Response(200, json={"id": "t", "spans": 0, "llm_calls": 0})

    def bodies(self, path: str) -> List[Any]:
        return [json.loads(r.content) for r in self.requests if r.url.path == path]


@pytest.fixture
def wire() -> _Wire:
    return _Wire()


def _trace(**fields) -> RunTrace:
    return RunTrace(agent_name="support-bot", **fields)


def _reported(**extra) -> Dict[str, Any]:
    """The four name lists, as a native loader reports the managed skill."""
    fields: Dict[str, Any] = {
        "active_skills": [{"name": "acme-refund-policy"}],
        "skills_offered_in_prompt": ["acme-refund-policy"],
        "skills_loaded_by_agent": ["decimalai-skills:acme-refund-policy"],
        "skills_delivered": [".agents/skills/acme-refund-policy/SKILL.md"],
    }
    fields.update(extra)
    return fields


IDENTIFIED = {
    "name": "acme/refund-policy",
    "skill_id": SKILL_ID,
    "version_id": VERSION_ID,
    "body_sha256": skillfmt.body_sha256(BODY),
}


def _assert_translated(body: Dict[str, Any]) -> None:
    assert body["active_skills"] == [IDENTIFIED]
    assert body["skills_offered_in_prompt"] == ["acme/refund-policy"]
    assert body["skills_loaded_by_agent"] == ["acme/refund-policy"]
    assert body["skills_delivered"] == ["acme/refund-policy"]


class TestTheSdksOwnTracesAreTranslated:
    def test_ingest_trace(self, managed_project, wire):
        wire.client.ingest_trace(_trace(**_reported()))
        [body] = wire.bodies("/api/v1/traces")
        _assert_translated(body)

    def test_ingest_traces_batch(self, managed_project, wire):
        wire.client.ingest_traces_batch([_trace(**_reported()), _trace(**_reported())])
        [batch] = wire.bodies("/api/v1/traces/batch")
        assert len(batch) == 2
        for body in batch:
            _assert_translated(body)

    def test_flush_sends_the_buffer_translated(self, managed_project, wire):
        wire.client.buffer_trace(_trace(**_reported()))
        wire.client.flush()
        [batch] = wire.bodies("/api/v1/traces/batch")
        _assert_translated(batch[0])

    def test_the_rest_of_the_payload_is_what_model_dump_gives(self, managed_project, wire):
        trace = _trace(**_reported())
        wire.client.ingest_trace(trace)
        [body] = wire.bodies("/api/v1/traces")
        expected = trace.model_dump(mode="json")
        for key in identity.NAME_LISTS:
            body.pop(key)
            expected.pop(key)
        assert body == expected

    def test_outside_a_managed_project_the_payload_is_unchanged(self, tmp_path, monkeypatch,
                                                                wire):
        (tmp_path / ".git").mkdir()
        monkeypatch.chdir(tmp_path)
        trace = _trace(**_reported())
        wire.client.ingest_trace(trace)
        [body] = wire.bodies("/api/v1/traces")
        assert body == trace.model_dump(mode="json")

    def test_a_translation_failure_never_costs_the_trace(self, managed_project, wire,
                                                         monkeypatch):
        def broken(payload, *, index=None):
            raise RuntimeError("boom")

        monkeypatch.setattr(identity, "translate_payload", broken)
        trace = _trace(**_reported())
        wire.client.ingest_trace(trace)
        wire.client.ingest_traces_batch([trace])
        [body] = wire.bodies("/api/v1/traces")
        [batch] = wire.bodies("/api/v1/traces/batch")
        assert body == batch[0] == trace.model_dump(mode="json")

    def test_an_unreadable_lock_never_costs_the_trace(self, managed_project, wire):
        (managed_project / ".decimal" / "skills.lock").write_text("{not json")
        trace = _trace(**_reported())
        wire.client.ingest_trace(trace)
        [body] = wire.bodies("/api/v1/traces")
        assert body == trace.model_dump(mode="json")


class TestACallersOwnPayloadIsSentAsGiven:
    def test_ingest_raw_trace(self, managed_project, wire):
        payload = {"agent_name": "support-bot", **_reported()}
        sent = json.loads(json.dumps(payload))
        wire.client.ingest_raw_trace(payload)
        [body] = wire.bodies("/api/v1/traces")
        assert body == sent

    def test_ingest_raw_traces_batch(self, managed_project, wire):
        payloads = [{"agent_name": "support-bot", **_reported()}]
        sent = json.loads(json.dumps(payloads))
        wire.client.ingest_raw_traces_batch(payloads)
        [batch] = wire.bodies("/api/v1/traces/batch")
        assert batch == sent


# ── nothing goes around the translation ──────────────────────────────────────

#: A trace ingest route, `/api/v1/traces` or `/api/v1/traces/batch`, ending the string it is
#: in: whole, or as the tail of an f-string or `+` (`f"{base}/api/v1/traces"`). A route that
#: goes on (`/api/v1/traces/{id}`, `/api/v1/traces/import`) is another endpoint.
_INGEST_PATH = re.compile(r"/api/v1/traces(?:/batch)?/?$")
_INGEST_METHODS = {"ingest_trace", "ingest_traces_batch", "ingest_raw_trace",
                   "ingest_raw_traces_batch"}


def _template(node: ast.AST) -> Optional[str]:
    """A string expression's text, with ``{}`` for each part only known at run time."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) and isinstance(v.value, str)
                       else "{}" for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _template(node.left), _template(node.right)
        if left is not None or right is not None:
            return (left if left is not None else "{}") + (right if right is not None else "{}")
    return None


def _ingest_path_literals(tree: ast.AST):
    """``(enclosing function, text)`` for every string expression naming a trace ingest route."""
    found = []

    def visit(node: ast.AST, func: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child.name)
                continue
            text = _template(child)
            if text is None:
                visit(child, func)
            elif _INGEST_PATH.search(text):
                found.append((func, text))

    visit(tree, "<module>")
    return found


def _posts(tree: ast.AST):
    """``(enclosing function, method, path)`` for each `_request_with_retry` / `_http.*` call."""
    out = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(func):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            consts = [a.value for a in node.args
                      if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            if node.func.attr == "_request_with_retry" and len(consts) >= 2:
                out.append((func.name, consts[0], consts[1]))
            elif node.func.attr in ("post", "get", "request", "put") and consts:
                out.append((func.name, node.func.attr.upper(), consts[-1]))
    return out


class TestOneTranslationPoint:
    def test_no_other_module_posts_to_the_trace_ingest_routes(self):
        offenders = []
        for path in sorted(PACKAGE_DIR.rglob("*.py")):
            if path.name == "_client.py" and path.parent == PACKAGE_DIR:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for func, value in _ingest_path_literals(tree):
                offenders.append(f"{path.relative_to(PACKAGE_DIR)}:{func}: {value!r}")
        assert offenders == [], (
            "Only DecimalAIClient may send a trace, so every trace the SDK builds goes "
            "through the skill identity translation in ingest_trace / ingest_traces_batch. "
            "Call the client instead of posting:\n  " + "\n  ".join(offenders))

    def test_the_client_posts_traces_only_from_its_ingest_methods(self):
        tree = ast.parse((PACKAGE_DIR / "_client.py").read_text(encoding="utf-8"))
        posting = {func for func, method, path in _posts(tree)
                   if method == "POST" and _INGEST_PATH.search(path)}
        assert posting == _INGEST_METHODS

    def test_the_guard_sees_a_post_it_should_refuse(self):
        # The scan above is only a guard if it can go red.
        tree = ast.parse(
            "import httpx\n"
            "def send(payload):\n"
            "    httpx.post(f'{BASE}/api/v1/traces', json=payload)\n"
            "def send_many(client, payloads):\n"
            "    client._http.post('/api/v1/traces/batch', json=payloads)\n"
            "def joined(client, payload):\n"
            "    client._http.post(PREFIX + '/api/v1/traces', json=payload)\n")
        assert [f for f, _ in _ingest_path_literals(tree)] == ["send", "send_many", "joined"]
        assert _ingest_path_literals(ast.parse(
            "a = client._http.post('/api/v1/traces/import-bulk')\n"
            "b = client._http.get(f'/api/v1/traces/{tid}')\n"
            "c = f'Open dashboard: {url}/traces'\n"
            "d = '/api/v1/traces/stats'\n")) == []
