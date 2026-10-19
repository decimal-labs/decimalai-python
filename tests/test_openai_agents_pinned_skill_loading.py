"""A skill load must fetch the loading agent's saved skill version.

These tests use `load_agent()` tools, a real SkillRouter and real local HTTP.
Two agents share one router and concurrently load the same skill at different
pins. No provider or remote backend is contacted.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

agents = pytest.importorskip("agents")

from decimalai import openai_agents as oa  # noqa: E402
from decimalai import skill_router as sr  # noqa: E402
from decimalai.skill_router import SkillRouter  # noqa: E402
from tests import test_openai_agents_phase1 as phase1  # noqa: E402

#: Phase 1's fixture: a configured SDK whose client serves a saved prompt.
sdk = phase1.sdk

NAME = "file-watcher"
SKILL_ID = "cb105bfb-b108-4e39-9304-ed0b41600ab7"
VERSION_IDS = {"pinned-agent": "cf4a7d84-1bd2-4a27-82bb-b957d8f1c0ca", "other-agent": "af4b28a6-2a52-4318-a498-ac3641b781d5"}
OTHER_NAME = "failure-review"
OTHER_SKILL_ID = "6078cce3-1d1e-4ab5-bda5-c689ec4267d1"
OTHER_VERSION_IDS = {"pinned-agent": "d2d8c5a2-4db1-41d4-a337-a30d29e61409", "other-agent": "4b40e45b-fd55-4022-96c6-52636105b590"}
RUN_IDS = {name: f"test-{name}" for name in VERSION_IDS}
run_key = contextvars.ContextVar("native_pin_test_run", default=None)


@pytest.fixture
def scoped(sdk, monkeypatch):
    """Runs keyed by this test's own run key, with fresh body budgets."""
    monkeypatch.setattr(oa, "_current_run_key", lambda: run_key.get())
    sr._body_budget_ctx.set(None)
    yield
    sr._body_budget_ctx.set(None)


@pytest.fixture
def body_server():
    requests = []
    lock = threading.Lock()
    concurrent = threading.Event()
    both_requests = threading.Barrier(2)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            parsed = urlsplit(self.path)
            agent = parse_qs(parsed.query).get("agent_name", [None])[0]
            with lock:
                requests.append((parsed.path, agent))
            if parsed.path == "/api/v1/skills/menu":
                payload = {"skills": [{"name": NAME}, {"name": OTHER_NAME}],
                           "prompt_fragment": f"SKILLS FOR {agent or 'UNSCOPED LATEST'}",
                           "strategy": "menu"}
            else:
                if concurrent.is_set():
                    both_requests.wait(timeout=5)
                name = parsed.path.split("/")[-2]
                versions = VERSION_IDS if name == NAME else OTHER_VERSION_IDS
                skill_id = SKILL_ID if name == NAME else OTHER_SKILL_ID
                version = versions.get(agent, "00000000-0000-4000-8000-000000000003")
                body = f"FULL INSTRUCTIONS FOR {agent or 'UNSCOPED LATEST'} ({name})"
                payload = {"body": body, "skill_id": skill_id, "version_id": version,
                           "content_hash": hashlib.sha256(body.encode()).hexdigest()}
            payload = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    router = SkillRouter(api_key="local-test-key", base_url=f"http://127.0.0.1:{server.server_port}")
    try:
        yield router, requests, concurrent
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


async def invoke(config, *, scope, name=NAME):
    """Call ``config``'s load_skill tool the way the Agents SDK does, in run ``scope``."""
    from agents.tool_context import ToolContext

    token = run_key.set(scope)
    try:
        context = agents.RunContextWrapper(context=None)
        arguments = json.dumps({"name": name})
        tool_context = ToolContext.from_agent_context(
            context, "call-test", tool_name="load_skill", tool_arguments=arguments,
        )
        return await config.load_skill.on_invoke_tool(tool_context, arguments)
    finally:
        run_key.reset(token)


def test_concurrent_loads_fetch_each_agents_own_pin(scoped, body_server):
    router, requests, concurrent = body_server
    oa._skill_router_singleton = router
    pinned, other = oa.load_agent("pinned-agent"), oa.load_agent("other-agent")
    concurrent.set()

    async def run():
        return await asyncio.gather(invoke(pinned, scope=RUN_IDS[pinned.name]),
                                    invoke(other, scope=RUN_IDS[other.name]))

    bodies = asyncio.run(run())
    assert "FULL INSTRUCTIONS FOR pinned-agent" in bodies[0]
    assert "FULL INSTRUCTIONS FOR other-agent" in bodies[1]
    assert sorted(requests) == sorted([(f"/api/v1/skills/{NAME}/body", name) for name in VERSION_IDS])
    assert router.agent_name is None  # the shared router was never repointed
    for config in (pinned, other):
        rail = oa._pop_run_rail(RUN_IDS[config.name])
        assert rail["loaded"] == [NAME]
        assert rail["delivered_versions"][0]["version_id"] == VERSION_IDS[config.name]


def test_parallel_distinct_skills_in_one_run_keep_every_body_witness(scoped, body_server, monkeypatch):
    router, requests, concurrent = body_server
    oa._skill_router_singleton = router
    config = oa.load_agent("pinned-agent")
    concurrent.set()
    both_loaded = threading.Barrier(2)
    consume_versions = router.consume_delivered_versions

    def drain_after_both_bodies(*args, **kwargs):
        # Only scheduling is controlled: both real HTTP bodies are recorded
        # before one callback drains this run's shared version rail.
        both_loaded.wait(timeout=5)
        return consume_versions(*args, **kwargs)

    monkeypatch.setattr(router, "consume_delivered_versions", drain_after_both_bodies)
    scope = "parallel-distinct-skills"

    async def run():
        return await asyncio.gather(invoke(config, scope=scope), invoke(config, scope=scope, name=OTHER_NAME))

    bodies = asyncio.run(run())
    assert f"({NAME})" in bodies[0]
    assert f"({OTHER_NAME})" in bodies[1]
    assert sorted(requests) == sorted([(f"/api/v1/skills/{name}/body", config.name) for name in (NAME, OTHER_NAME)])
    rail = oa._pop_run_rail(scope)
    assert sorted(rail["loaded"]) == sorted([NAME, OTHER_NAME])
    assert {(v["name"], v["skill_id"], v["version_id"], v["hash"]) for v in rail["delivered_versions"]} == {
        (name, skill_id, version_id, hashlib.sha256(f"FULL INSTRUCTIONS FOR {config.name} ({name})".encode()).hexdigest())
        for name, skill_id, version_id in ((NAME, SKILL_ID, VERSION_IDS[config.name]), (OTHER_NAME, OTHER_SKILL_ID, OTHER_VERSION_IDS[config.name]))
    }
    assert sorted(rail["delivered"]) == sorted([NAME, OTHER_NAME])
    assert consume_versions(scope=scope) == []


def test_a_handoff_in_one_run_uses_each_agents_menu_and_keeps_both_pins(scoped, body_server):
    """Agent A hands off to agent B in the same run: each shows its own skill
    list, each load fetches its own agent's pin, and the run keeps both."""
    router, requests, _ = body_server
    oa._skill_router_singleton = router
    configs = [oa.load_agent("pinned-agent"), oa.load_agent("other-agent")]
    scope = "handoff-run"

    async def run():
        bodies = []
        for config in configs:
            agent = agents.Agent(name=config.name, instructions=config.instructions, tools=[*config.tools])
            token = run_key.set(scope)
            try:
                prompt = await config.instructions(agents.RunContextWrapper(context=None), agent)
                assert f"SKILLS FOR {config.name}" in prompt
            finally:
                run_key.reset(token)
            bodies.append(await invoke(config, scope=scope))
        return bodies

    bodies = asyncio.run(run())
    assert "FULL INSTRUCTIONS FOR pinned-agent" in bodies[0]
    assert "FULL INSTRUCTIONS FOR other-agent" in bodies[1]
    assert requests == [(path, agent) for agent in VERSION_IDS for path in ("/api/v1/skills/menu", f"/api/v1/skills/{NAME}/body")]
    rail = oa._pop_run_rail(scope)
    assert rail["loaded"] == [NAME]
    assert {v["version_id"] for v in rail["delivered_versions"]} == set(VERSION_IDS.values())
    assert set(rail["decimal_prompts"]) == set(VERSION_IDS)


def test_legacy_router_fallback_preserves_agent_and_never_fetches_unscoped(scoped, monkeypatch):
    class OldScopedRouter:
        def load_skill(self, name, *, agent_name):
            return f"## Skill: {name}\n\nBODY FOR {agent_name}"

    monkeypatch.setattr(oa, "_skill_router_singleton", OldScopedRouter())
    token = run_key.set("legacy-run")
    try:
        assert "BODY FOR pinned-agent" in oa._handle_load_skill(NAME, agent_name="pinned-agent")
    finally:
        run_key.reset(token)

    calls = []

    class UnscopedRouter:
        def load_skill(self, name):
            calls.append(name)
            return f"## Skill: {name}\n\nLATEST BODY"

    monkeypatch.setattr(oa, "_skill_router_singleton", UnscopedRouter())
    assert "cannot honor" in oa._handle_load_skill(NAME, agent_name="pinned-agent")
    assert calls == []
