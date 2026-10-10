"""Native tool invocation must fetch the current agent's saved skill version.

These tests use the real Agents function-tool wrapper and real local HTTP. Two
agents share one router and concurrently load the same skill at different pins.
No provider or remote backend is contacted.
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

from decimalai import openai_agents as oa
from decimalai import skill_router as sr
from decimalai.skill_router import SkillRouter

agents = pytest.importorskip("agents")
NAME = "file-watcher"
SKILL_ID = "cb105bfb-b108-4e39-9304-ed0b41600ab7"
VERSION_IDS = {"pinned-agent": "cf4a7d84-1bd2-4a27-82bb-b957d8f1c0ca", "other-agent": "af4b28a6-2a52-4318-a498-ac3641b781d5"}
OTHER_NAME = "failure-review"
OTHER_SKILL_ID = "6078cce3-1d1e-4ab5-bda5-c689ec4267d1"
OTHER_VERSION_IDS = {"pinned-agent": "d2d8c5a2-4db1-41d4-a337-a30d29e61409", "other-agent": "4b40e45b-fd55-4022-96c6-52636105b590"}
RUN_IDS = {name: f"test-{name}" for name in VERSION_IDS}
run_key = contextvars.ContextVar("native_pin_test_run", default=None)


@pytest.fixture
def native_tools(monkeypatch):
    """Restore all patched Agent methods and router rails after each test."""
    for method in ("__init__", "get_system_prompt", "get_all_tools"):
        monkeypatch.setattr(agents.Agent, method, getattr(agents.Agent, method))
    monkeypatch.setattr(oa, "_skill_loader_installed", False)
    monkeypatch.setattr(oa, "_agent_hooks_installed", False)
    monkeypatch.setattr(oa, "_skill_router_singleton", None)
    monkeypatch.setattr(oa, "_run_rails", oa.OrderedDict())
    monkeypatch.setattr(oa, "_current_run_key", lambda: run_key.get())
    monkeypatch.setattr(oa, "_load_skill_tool_registration_failed", False)
    monkeypatch.setattr(oa, "_load_skill_tool_enabled", lambda: True)
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


async def invoke(agent, *, scope=None, name=NAME):
    from agents.tool_context import ToolContext

    token = run_key.set(scope)
    try:
        context = agents.RunContextWrapper(context=None)
        tools = await agent.get_all_tools(context)
        tool = next(tool for tool in tools if tool.name == "load_skill")
        tool_context = ToolContext.from_agent_context(context, "call-test", tool_name="load_skill", tool_arguments=json.dumps({"name": name}))
        return await tool.on_invoke_tool(tool_context, json.dumps({"name": name}))
    finally:
        run_key.reset(token)


def test_native_tool_concurrently_fetches_exact_agent_pins(native_tools, body_server, monkeypatch):
    router, requests, concurrent = body_server
    monkeypatch.setattr(oa, "_skill_router_singleton", router)
    oa._install_skill_loader()
    pinned = agents.Agent(name="pinned-agent", instructions="Use a skill")
    other = agents.Agent(name="other-agent", instructions="Use a skill")
    concurrent.set()

    async def run():
        return await asyncio.gather(invoke(pinned, scope=RUN_IDS[pinned.name]), invoke(other, scope=RUN_IDS[other.name]))

    bodies = asyncio.run(run())
    assert "FULL INSTRUCTIONS FOR pinned-agent" in bodies[0]
    assert "FULL INSTRUCTIONS FOR other-agent" in bodies[1]
    assert sorted(requests) == sorted([(f"/api/v1/skills/{NAME}/body", name) for name in VERSION_IDS])
    assert router.agent_name is None  # singleton was never repointed
    for agent in (pinned, other):
        rail = oa._pop_run_rail(RUN_IDS[agent.name])
        assert rail["loaded"] == [NAME]
        assert rail["delivered_versions"][0]["version_id"] == VERSION_IDS[agent.name]


def test_parallel_distinct_skills_in_one_run_keep_every_body_witness(native_tools, body_server, monkeypatch):
    router, requests, concurrent = body_server
    monkeypatch.setattr(oa, "_skill_router_singleton", router)
    oa._install_skill_loader()
    agent = agents.Agent(name="pinned-agent", instructions="Use both skills")
    concurrent.set()
    both_loaded = threading.Barrier(2)
    consume_versions = router.consume_delivered_versions

    def drain_after_both_bodies(*args, **kwargs):
        # Only scheduling is controlled: both real HTTP bodies are recorded
        # before one native callback drains this run's shared version rail.
        both_loaded.wait(timeout=5)
        return consume_versions(*args, **kwargs)

    monkeypatch.setattr(router, "consume_delivered_versions", drain_after_both_bodies)
    scope = "parallel-distinct-skills"

    async def run():
        return await asyncio.gather(invoke(agent, scope=scope), invoke(agent, scope=scope, name=OTHER_NAME))

    bodies = asyncio.run(run())
    assert f"({NAME})" in bodies[0]
    assert f"({OTHER_NAME})" in bodies[1]
    assert sorted(requests) == sorted([(f"/api/v1/skills/{name}/body", agent.name) for name in (NAME, OTHER_NAME)])
    rail = oa._pop_run_rail(scope)
    assert sorted(rail["loaded"]) == sorted([NAME, OTHER_NAME])
    assert {(v["name"], v["skill_id"], v["version_id"], v["hash"]) for v in rail["delivered_versions"]} == {
        (name, skill_id, version_id, hashlib.sha256(f"FULL INSTRUCTIONS FOR {agent.name} ({name})".encode()).hexdigest())
        for name, skill_id, version_id in ((NAME, SKILL_ID, VERSION_IDS[agent.name]), (OTHER_NAME, OTHER_SKILL_ID, OTHER_VERSION_IDS[agent.name]))
    }
    assert sorted(rail["delivered"]) == sorted([NAME, OTHER_NAME])
    assert consume_versions(scope=scope) == []


def test_same_run_handoff_uses_each_agents_menu_and_retains_both_pins(native_tools, body_server, monkeypatch):
    router, requests, _ = body_server
    monkeypatch.setattr(oa, "_skill_router_singleton", router)
    oa._install_skill_loader()
    original = agents.Agent(name="pinned-agent", instructions="Use a skill")
    handoff = original.clone(name="other-agent")
    scope = "handoff-run"

    async def run():
        bodies = []
        for agent in (original, handoff):
            token = run_key.set(scope)
            try:
                # The actual framework prompt hook gives each handoff agent
                # its own offered menu and a fresh per-turn body budget.
                prompt = await agent.get_system_prompt(agents.RunContextWrapper(context=None))
                assert f"SKILLS FOR {agent.name}" in prompt
            finally:
                run_key.reset(token)
            bodies.append(await invoke(agent, scope=scope))
        return bodies

    bodies = asyncio.run(run())
    assert "FULL INSTRUCTIONS FOR pinned-agent" in bodies[0]
    assert "FULL INSTRUCTIONS FOR other-agent" in bodies[1]
    assert requests == [(path, agent) for agent in VERSION_IDS for path in ("/api/v1/skills/menu", f"/api/v1/skills/{NAME}/body")]
    rail = oa._pop_run_rail(scope)
    assert rail["loaded"] == [NAME]
    assert {v["version_id"] for v in rail["delivered_versions"]} == set(VERSION_IDS.values())


def test_clone_rebinds_the_sdk_tool_without_changing_the_original(native_tools, body_server, monkeypatch):
    router, requests, _ = body_server
    monkeypatch.setattr(oa, "_skill_router_singleton", router)
    oa._install_skill_loader()
    original = agents.Agent(name="pinned-agent", instructions="Use a skill")
    clone = original.clone(name="other-agent")
    assert clone.tools[0] is original.tools[0]  # the real SDK shares tools
    assert "FULL INSTRUCTIONS FOR other-agent" in asyncio.run(invoke(clone, scope="clone-run"))
    assert "FULL INSTRUCTIONS FOR pinned-agent" in asyncio.run(invoke(original, scope="original-run"))
    assert [agent for _, agent in requests] == ["other-agent", "pinned-agent"]


def test_prebuilt_agent_is_bound_when_its_tools_are_retrofitted(native_tools, body_server, monkeypatch):
    router, requests, _ = body_server
    monkeypatch.setattr(oa, "_skill_router_singleton", router)
    early = agents.Agent(name="pinned-agent", instructions="Use a skill")
    assert early.tools == []
    oa._install_skill_loader()
    assert "FULL INSTRUCTIONS FOR pinned-agent" in asyncio.run(invoke(early, scope="early-run"))
    assert requests == [(f"/api/v1/skills/{NAME}/body", "pinned-agent")]
    assert early.tools == []  # retrofit never mutates declared user tools


def test_user_declared_loader_is_not_replaced(native_tools):
    @agents.function_tool
    def load_skill(name: str) -> str:
        """A user-owned loader."""
        return f"USER BODY {name}"

    oa._install_skill_loader()
    agent = agents.Agent(name="pinned-agent", tools=[load_skill])
    tools = asyncio.run(agent.get_all_tools(agents.RunContextWrapper(context=None)))
    assert tools == [load_skill]
    assert "USER BODY file-watcher" in asyncio.run(invoke(agent, scope="user-run"))


def test_legacy_router_fallback_preserves_agent_and_never_fetches_unscoped(native_tools, monkeypatch):
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
