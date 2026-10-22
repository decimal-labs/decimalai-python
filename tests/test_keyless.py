"""Keyless ``agent_config()``: no DecimalAI key, and the switch to one (owner decisions
2026-10-21, and the skill list's one-home rule of 2026-10-22).

With no key set and ``init()`` never called, every entry point (``decimalai.agent_config``
and the OpenAI Agents, Pydantic AI, Google ADK and Anthropic modules) runs on its
``fallback=`` with the project's skills. Pinned here:

- the list is what ``.decimal/skills.toml`` lists: registry skills ``decimalai skills add``
  installed, and skills you wrote listed under ``[authored]``. A folder in
  ``.agents/skills`` it does not list is left out (and named once); nothing is searched,
  nothing is added on the agent's behalf, and a keyless run makes no request at all;
- ``.decimal/link.toml`` makes a keyless run in a connected project raise;
- the first keyed run creates the agent (with no prompt) and copies the project's list
  onto it once: registry skills assigned at their pinned versions, skills you wrote
  uploaded (``remote_wins_on_conflict``) and added. The link file records what was
  copied;
- every later keyed start reads local files only: no request, no write. It says once per
  process what the project lists that the agent was not given, and which skills you wrote
  no longer match DecimalAI's copy. From the first connect on, the agent's skills live on
  DecimalAI.

The registry is ``tests/fakes/fake_registry.py`` on a real socket, so the install engine
goes through the HTTP client it uses against the platform.
"""

from __future__ import annotations

import socket
import urllib.parse
from types import SimpleNamespace

import httpx
import pytest

import decimalai
import decimalai._agent as agent_mod
import decimalai._agent_binding as binding
import decimalai._config as cfg
import decimalai._keyless as keyless
from decimalai import _authored, skillpkg
from decimalai._client import AgentNotFoundError
from decimalai._config import DecimalConfig, DecimalConfigError
from decimalai.skill_router import LOAD_SKILL_PROMPT_HINT, SkillRouter, SkillRouterError
from tests.fakes.fake_registry import FakeRegistry

MESSAGE = "My package arrived damaged, can I get a refund?"
FALLBACK = "You help customers with their orders."
POLICY_BODY = "# Refund policy\n\nRefund a damaged item in full within 30 days."
TONE = ("---\nname: refund-tone\ndescription: How Acme words a refund reply.\n---\n\n"
        "# Refund tone\n\nApologize once, state the refund, give the date it lands.\n")


# ── the keyless world ─────────────────────────────────────────────────────────


@pytest.fixture
def world(monkeypatch, tmp_path):
    """No key, no init(), a git project of its own, the registry on 127.0.0.1, and every
    other connection refused. ``world.attempts`` records every destination tried."""
    for name in ("DECIMAL_API_KEY", "DECIMALAI_API_KEY", "DECIMALAI_BASE_URL", "DECIMAL_AGENT_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cfg, "_config", None)
    monkeypatch.setattr(cfg, "_client", None)
    monkeypatch.setattr(binding, "_in_background", lambda fn, *args: fn(*args))
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    keyless._reset()

    registry = FakeRegistry()
    policy = registry.add_skill("acme/refund-policy", versions=2, body=POLICY_BODY,
                                description="Refund rules for damaged items: when to refund, and how much.")
    tracking = registry.add_skill("acme/order-tracking", description="Track a parcel's shipping status.")

    attempts: list = []
    with registry.serve() as url:
        port = urllib.parse.urlsplit(url).port
        real_connect = socket.socket.connect
        real_getaddrinfo = socket.getaddrinfo

        def connect(sock, address):
            attempts.append(address)
            if not (isinstance(address, tuple) and tuple(address[:2]) == ("127.0.0.1", port)):
                raise OSError(f"test: no network to {address!r}")
            return real_connect(sock, address)

        def getaddrinfo(host, *args, **kwargs):
            if host not in ("127.0.0.1", b"127.0.0.1"):
                attempts.append(host)
                raise socket.gaierror(f"test: no lookup of {host!r}")
            return real_getaddrinfo(host, *args, **kwargs)

        monkeypatch.setattr(socket.socket, "connect", connect)
        monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
        monkeypatch.setenv("DECIMAL_BASE_URL", url)
        yield SimpleNamespace(registry=registry, url=url, port=port, attempts=attempts, project=tmp_path,
                              policy=policy, tracking=tracking)
    keyless._reset()


def offered(system: str) -> list:
    return [line.split("|")[1].strip() for line in system.splitlines()
            if line.startswith("| ") and not line.startswith("| Skill")]


def manifest(world) -> str:
    path = world.project / ".decimal" / "skills.toml"
    return path.read_text() if path.exists() else ""


def add(ref: str) -> None:
    """`decimalai skills add <registry ref>` with no key: the anonymous install routes."""
    result = skillpkg.add(ref, interactive=False)
    assert result.ok, result


def write_own(world, name="refund-tone", text=TONE):
    path = world.project / ".agents" / "skills" / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "SKILL.md").write_text(text, encoding="utf-8")
    return path / "SKILL.md"


def own(world, name="refund-tone", text=TONE):
    """A skill you wrote, listed: `decimalai skills add refund-tone`."""
    path = write_own(world, name, text)
    _authored.add(_project(world), name)
    return path


def _project(world):
    from decimalai.skillpkg.project import Project

    return Project.at(world.project)


class Quiet:
    """Nothing reached the network from here on."""

    def __init__(self, world):
        self.world = world
        self.requests = len(world.registry.requests)
        self.attempts = len(world.attempts)

    def check(self):
        assert len(self.world.registry.requests) == self.requests, self.world.registry.requests[self.requests:]
        assert len(self.world.attempts) == self.attempts, self.world.attempts[self.attempts:]
        assert cfg._config is None and cfg._client is None, "nothing initialized the SDK"


# ── the neutral agent_config() ────────────────────────────────────────────────


def test_a_keyless_agent_runs_on_its_fallback_and_says_so(world, capsys):
    support = decimalai.agent_config("order-support", fallback=FALLBACK)
    assert (support.name, support.prompt, support.prompt_source) == (
        "order-support", FALLBACK, "fallback (no key)")
    err = capsys.readouterr().err
    assert "agent 'order-support' runs keyless" in err
    assert "the ones .decimal/skills.toml lists (`decimalai skills add <name>` adds one)" in err
    add("acme/refund-policy")
    run = support.run(MESSAGE)
    assert run.system.startswith(FALLBACK + "\n\n## Available Skills")
    assert run.system.endswith(LOAD_SKILL_PROMPT_HINT)
    decimalai.agent_config("order-support", fallback=FALLBACK)
    assert "keyless" not in capsys.readouterr().err, "said once per agent"


def test_without_a_fallback_the_instructions_are_the_skill_list(world):
    add("acme/refund-policy")
    support = decimalai.agent_config("order-support")
    assert (support.prompt, support.prompt_source) == (None, "fallback (no key)")
    assert support.run(MESSAGE).system.startswith("## Available Skills")


def test_a_run_lists_the_projects_skills_reads_them_from_disk_and_asks_nothing(world):
    add("acme/refund-policy@2")
    own(world)
    quiet = Quiet(world)
    run = decimalai.agent_config("order-support", fallback=FALLBACK).run(MESSAGE)
    assert offered(run.system) == ["acme/refund-policy", "refund-tone"]
    assert "enum" not in run.tool_spec["parameters"]["properties"]["name"], "any listed name loads"
    assert run.load_skill("acme/refund-policy") == f"## Skill: acme/refund-policy\n\n{POLICY_BODY}"
    assert run.load_skill("refund-tone").startswith("## Skill: refund-tone\n\n# Refund tone")
    assert run.load_skill("acme/order-tracking").startswith(
        "load_skill: no skill named 'acme/order-tracking'"), "the registry is not reached for it"
    quiet.check()
    assert "order-tracking" not in manifest(world), "nothing is added on the agent's behalf"


def test_a_folder_the_project_does_not_list_is_left_out_and_named_once(world, capsys):
    write_own(world, "run-tests", "---\nname: run-tests\ndescription: Run this repo's tests.\n---\n\n"
                                  "# Run tests\n\nRun make test from the root.\n")
    own(world)
    support = decimalai.agent_config("order-support")
    err = capsys.readouterr().err
    assert (".agents/skills has skills this project's agents do not use: run-tests. To give one "
            "to agent 'order-support': decimalai skills add <name>") in err
    run = support.run(MESSAGE)
    assert offered(run.system) == ["refund-tone"]
    assert run.load_skill("run-tests").startswith("load_skill: no skill named 'run-tests'")
    decimalai.agent_config("billing")
    assert "do not use" not in capsys.readouterr().err, "once per project per process"


def test_a_skill_listed_while_the_process_runs_is_offered_from_its_next_run(world):
    support = decimalai.agent_config("order-support")
    assert offered(support.run(MESSAGE).system) == []
    own(world)
    assert offered(support.run(MESSAGE).system) == ["refund-tone"]


def test_a_listed_skill_too_long_to_serve_is_refused_when_loaded(world):
    own(world, "refund-manual", "---\nname: refund-manual\ndescription: Every refund rule.\n---\n\n"
                                "# Refund manual\n\n" + "x" * 25_000 + "\n")
    run = decimalai.agent_config("order-support").run(MESSAGE)
    assert "exceeds the 24000-character per-body limit" in run.load_skill("refund-manual")


def test_a_keyless_process_never_initializes_or_traces(world):
    add("acme/refund-policy")
    quiet = Quiet(world)
    before = decimalai.export_status()  # process-wide: earlier tests' traces count too
    support = decimalai.agent_config("order-support", fallback=FALLBACK)
    support.run(MESSAGE).load_skill("acme/refund-policy")
    decimalai.flush()
    after = decimalai.export_status()
    assert (after.sent, after.failed) == (before.sent, before.failed)
    quiet.check()


# ── every entry point ─────────────────────────────────────────────────────────


def test_openai_agents_keyless(world, monkeypatch):
    pytest.importorskip("agents")
    from agents import Agent, Runner
    from agents.tracing import get_trace_provider, set_trace_processors

    import decimalai.openai_agents as oa
    from tests.test_openai_agents_load_agent import ScriptedModel, lookup_order

    add("acme/refund-policy")
    quiet = Quiet(world)
    monkeypatch.setattr(oa, "_ensure_tracing", lambda: pytest.fail("a keyless run is traced nowhere"))
    saved = list(get_trace_provider()._multi_processor._processors)
    set_trace_processors([])  # OpenAI's own exporter would call api.openai.com
    try:
        support = oa.agent_config("order-support", fallback=FALLBACK)
        assert support.prompt_source == "fallback (no key)"
        model = ScriptedModel([("call", "load_skill", {"name": "acme/refund-policy"}), "answer"])
        agent = Agent(name=support.name, instructions=support.instructions,
                      tools=[lookup_order, *support.tools], model=model)
        Runner.run_sync(agent, MESSAGE)
    finally:
        set_trace_processors(saved)
    assert model.instructions[0].startswith(FALLBACK + "\n\n## Available Skills")
    assert "| acme/refund-policy |" in model.instructions[0]
    result = next(i for i in model.inputs[1] if isinstance(i, dict) and i.get("type") == "function_call_output")
    assert result["output"].startswith("## Skill: acme/refund-policy")
    quiet.check()


def test_pydantic_ai_keyless(world):
    pytest.importorskip("pydantic_ai")
    import decimalai.pydantic_ai as pai
    from tests.test_pydantic_ai_load_agent import Model, agent_with

    add("acme/refund-policy")
    quiet = Quiet(world)
    support = pai.agent_config("order-support", fallback=FALLBACK)
    assert support.prompt_source == "fallback (no key)"
    model = Model(("call", "acme/refund-policy"), "answer")
    assert agent_with(support, model).run_sync(MESSAGE, deps={}).output == "done"
    first, second = model.requests
    assert first["instructions"].startswith(FALLBACK + "\n\n## Available Skills")
    assert second["returns"][0].startswith("## Skill: acme/refund-policy")
    quiet.check()


def test_adk_keyless(world):
    pytest.importorskip("google.adk")
    import decimalai.adk as adk
    from tests.test_adk_load_agent import model, ours, run

    add("acme/refund-policy")
    quiet = Quiet(world)
    support = adk.agent_config("order-support", fallback=FALLBACK)
    assert support.prompt_source == "fallback (no key)"
    llm = model(("call", "acme/refund-policy"), "answer")
    answer, _ = run(support, llm, MESSAGE, state={})
    assert answer == "done"
    first, second = llm.requests
    assert ours(first).startswith(FALLBACK + "\n\n## Available Skills")
    assert second["returns"][0]["result"].startswith("## Skill: acme/refund-policy")
    quiet.check()


def test_anthropic_keyless(world, monkeypatch):
    pytest.importorskip("anthropic")
    pytest.importorskip("httpx2")
    import decimalai.anthropic as da
    from tests.test_anthropic_load_agent import Messages, tool_runner

    add("acme/refund-policy")
    quiet = Quiet(world)
    monkeypatch.setattr(da, "_turn_on_tracing", lambda: pytest.fail("a keyless run is traced nowhere"))
    support = da.agent_config("order-support", fallback=FALLBACK)
    assert support.prompt_source == "fallback (no key)"
    api = Messages(("call", "acme/refund-policy"), "answer")
    with support.run(MESSAGE) as run:
        assert tool_runner(api.client(), run, MESSAGE).stop_reason == "end_turn"
    prompt, menu = api.requests[0]["system"]
    assert prompt["text"] == FALLBACK and "| acme/refund-policy |" in menu["text"]
    (result,) = api.tool_results(1)
    assert result["content"].startswith("## Skill: acme/refund-policy")
    quiet.check()


# ── the guard ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("module,extra", [
    ("decimalai", None), ("openai_agents", "agents"), ("pydantic_ai", "pydantic_ai"),
    ("adk", "google.adk"), ("anthropic", "anthropic"),
])
def test_a_connected_project_without_a_key_raises_naming_the_workspace(world, module, extra):
    if extra:
        pytest.importorskip(extra)
    import importlib

    entry = decimalai if module == "decimalai" else importlib.import_module(f"decimalai.{module}")
    link = world.project / ".decimal" / "link.toml"
    link.parent.mkdir()
    link.write_text('base_url = "https://api.decimal.ai"\norg_id = "org-1"\norg_slug = "acme"\n'
                    'workspace_id = "ws-prod"\nagents = ["order-support"]\n')
    with pytest.raises(DecimalConfigError) as exc:
        entry.agent_config("order-support", fallback=FALLBACK)
    assert "organization acme (workspace ws-prod)" in str(exc.value)
    assert "DECIMAL_API_KEY" in str(exc.value)
    assert world.attempts == [], "refused before any request"


def test_a_damaged_link_file_still_refuses(world):
    link = world.project / ".decimal" / "link.toml"
    link.parent.mkdir()
    link.write_text("this is [not toml\n")
    with pytest.raises(DecimalConfigError, match="DECIMAL_API_KEY"):
        decimalai.agent_config("order-support", fallback=FALLBACK)


# ── adding a key ──────────────────────────────────────────────────────────────


class Workspace:
    """The platform behind a key: agents, the verify route, assignments and the sync."""

    def __init__(self, *agents, workspace_id="ws-1", org_id="org-1"):
        self.agents = set(agents)
        self.org_id = org_id
        self.created: list = []
        self.verified = 0
        self.workspace_id = workspace_id
        self.verify_error = None
        self.assigned: list = []
        self.assign_error = None
        self.prompt_writes: list = []
        self.synced: list = []
        #: conflict policies this platform takes (one from before 2026-10-22 lacks the new one)
        self.policies = {"newer_wins", "local_wins", "remote_wins", "remote_wins_on_conflict"}
        #: skill name -> the body DecimalAI has, changed there: the sync pulls it
        self.changed_on_decimalai: dict = {}
        #: refuse every sync (a 422 that is not about the policy)
        self.sync_refused = False
        #: skills deleted on DecimalAI: a sync finds them by name and leaves them off the agent
        self.deleted: set = set()

    # the DecimalAIClient methods agent_config() calls
    def get_agent_prompt(self, agent_name, version=None, **kwargs):
        if agent_name not in self.agents:
            raise AgentNotFoundError(
                httpx.Response(404, json={"detail": "Agent not found"},
                               request=httpx.Request("GET", f"http://ws.test/api/v1/agents/{agent_name}/prompt")),
                agent_name=agent_name)
        return {"agent_name": agent_name, "system_prompt": None, "version_number": None, "content_hash": None}

    def create_agent(self, agent_name, **fields):
        self.created.append((agent_name, fields))
        self.agents.add(agent_name)

    def _request_with_retry(self, method, url, **kwargs):
        request = httpx.Request(method, "http://ws.test" + url)
        if (method, url) == ("GET", "/api/v1/org"):
            return httpx.Response(200, request=request, json={
                "id": self.org_id, "name": "Acme", "slug": "acme", "plan": "free",
                "billing_email": "billing@example.invalid"})
        assert (method, url) == ("GET", "/api/v1/auth/verify")
        self.verified += 1
        if self.verify_error is not None:
            raise self.verify_error
        return httpx.Response(200, json={"api_key_valid": True, "workspace_id": self.workspace_id},
                              request=request)

    def __getattr__(self, name):  # any other client call fails the test
        raise AttributeError(f"agent_config() called client.{name}")

    # the SkillRouter's requests
    def request(self, method, path, json=None, params=None, **kwargs):
        if (method, path) == ("POST", "/api/v1/skills/sync"):
            return self.sync(json)
        if "/prompt" in path and method != "GET":
            self.prompt_writes.append(path)
        assert method == "POST" and path.startswith("/api/v1/agents/") and "/skills/" in path, path
        if self.assign_error is not None:
            raise self.assign_error
        self.assigned.append((path, dict(params or {})))
        return {"status": "ok", "action": "assigned"}

    def sync(self, body):
        self.synced.append(body)
        if body["conflict_policy"] not in self.policies or self.sync_refused:
            # What SkillRouter._request raises for the platform's 422 envelope: the
            # message only ("detail"), not the field list ("details") that names the cause.
            raise SkillRouterError(
                "SkillRouter request failed (422): POST /api/v1/skills/sync — Request validation failed",
                status_code=422, detail="Request validation failed")
        actions = []
        for item in body["skills"]:
            theirs = self.changed_on_decimalai.get(item["name"])
            if theirs is not None and body["conflict_policy"] == "remote_wins_on_conflict":
                actions.append({"name": item["name"], "action": "pulled", "body_markdown": theirs,
                                "version_number": 2, "conflict": True})
            else:
                actions.append({"name": item["name"], "action": "created", "new_version_number": 1})
        names = [a["name"] for a in actions]
        return {"status": "ok", "created": len(names), "updated": 0, "unchanged": 0, "pulled": 0,
                "failures": [], "actions": actions, "added_to_agent": body.get("agent_name"),
                "added": [n for n in names if n not in self.deleted], "kept_off": []}


def with_key(monkeypatch, workspace):
    """The same process, with a key: init() has run, and the router answers from ``workspace``."""
    monkeypatch.setenv("DECIMAL_API_KEY", "dai_sk_test")
    monkeypatch.setattr(cfg, "_config", DecimalConfig(api_key="dai_sk_test", base_url="https://ws.test"))
    monkeypatch.setattr(cfg, "_client", workspace)
    router = SkillRouter(api_key="dai_sk_test", base_url="https://ws.test", inject_body=False)
    monkeypatch.setattr(router, "_request", workspace.request)
    monkeypatch.setattr(agent_mod, "_skill_router_singleton", router)


def keyless_phase(world):
    """What the tutorial's first steps leave: one registry skill added, pinned."""
    add("acme/refund-policy@2")
    run = decimalai.agent_config("order-support", fallback=FALLBACK).run(MESSAGE)
    assert run.load_skill("acme/refund-policy").startswith("## Skill:")


def lock_entry(world, name):
    import json

    skills = json.loads((world.project / ".decimal/skills.lock").read_text())["skills"]
    return next(e for e in skills.values() if e["name"] == name)


def test_adding_a_key_creates_the_agent_copies_the_projects_skills_and_links_the_project(
        world, monkeypatch, capsys):
    keyless_phase(world)
    own(world)
    entry = lock_entry(world, "acme/refund-policy")
    capsys.readouterr()

    workspace = Workspace()
    with_key(monkeypatch, workspace)
    support = decimalai.agent_config("order-support", fallback=FALLBACK)
    assert workspace.created == [("order-support", {})], "created with no prompt: the code's is not saved"
    assert support.prompt_source == "fallback (none saved)" and support.prompt == FALLBACK
    assert workspace.assigned == [(f"/api/v1/agents/order-support/skills/{entry['skill_id']}",
                                   {"pin_version_id": entry["version_id"]})]
    (sync,) = workspace.synced
    assert [s["name"] for s in sync["skills"]] == ["refund-tone"]
    assert (sync["conflict_policy"], sync["agent_name"], sync["add_to_agent"]) == (
        "remote_wins_on_conflict", "order-support", "all")
    assert sync["install_id"], "the checkout's baseline is recorded, so a later conflict can be told"
    assert workspace.prompt_writes == []
    link = keyless.read_link(None)
    assert (link["base_url"], link["workspace_id"], link["org_id"], link["org_slug"]) == (
        "https://ws.test", "ws-1", "org-1", "acme")
    assert link["agents"] == ["order-support"]
    assert link["assigned"] == {"order-support": {"acme/refund-policy": "2", "refund-tone": "authored"}}
    assert link["uploaded"] == {"refund-tone": _authored.body_hash(_project(world), "refund-tone")}
    err = capsys.readouterr().err
    assert "created agent 'order-support'" in err
    assert ("connected agent 'order-support' to your DecimalAI workspace: copied acme/refund-policy v2, "
            "refund-tone (uploaded) onto it. From now on its skills live on DecimalAI: change them in "
            "the dashboard or with `decimalai skills add`; editing .decimal/skills.toml no longer "
            "changes them. Wrote .decimal/link.toml: commit it.") in err

    decimalai.agent_config("order-support", fallback=FALLBACK)
    assert (workspace.verified, len(workspace.assigned), len(workspace.synced)) == (1, 1, 1), "copied once"
    assert capsys.readouterr().err == ""

    # The deploy that lost its key: the committed link file refuses to run keyless.
    monkeypatch.delenv("DECIMAL_API_KEY")
    monkeypatch.setattr(cfg, "_config", None)
    monkeypatch.setattr(cfg, "_client", None)
    with pytest.raises(DecimalConfigError, match=r"organization acme \(workspace ws-1\)"):
        decimalai.agent_config("order-support", fallback=FALLBACK)


def test_an_organization_wide_key_links_the_organization(world, monkeypatch):
    keyless_phase(world)
    with_key(monkeypatch, Workspace(workspace_id=None))
    decimalai.agent_config("order-support")
    link = (world.project / ".decimal/link.toml").read_text()
    assert 'org_slug = "acme"' in link and "workspace_id" not in link


def test_a_latest_entry_is_assigned_to_follow_the_latest(world, monkeypatch):
    add("acme/order-tracking")
    workspace = Workspace()
    with_key(monkeypatch, workspace)
    decimalai.agent_config("order-support")
    assert [params for _path, params in workspace.assigned] == [{"pin_version_id": None}]


def test_an_existing_agent_with_nothing_to_connect_is_left_alone(world, monkeypatch):
    workspace = Workspace("order-support")
    with_key(monkeypatch, workspace)
    decimalai.agent_config("order-support")
    assert (workspace.created, workspace.verified) == ([], 0)
    assert not (world.project / ".decimal").exists()


def test_outside_a_git_checkout_nothing_is_written(world, monkeypatch):
    keyless_phase(world)
    (world.project / ".git").rmdir()
    workspace = Workspace()
    with_key(monkeypatch, workspace)
    decimalai.agent_config("order-support")
    assert workspace.created and workspace.verified == 0
    assert not (world.project / ".decimal/link.toml").exists()


def test_a_platform_that_does_not_answer_connects_nothing_and_the_next_run_tries_again(world, monkeypatch):
    keyless_phase(world)
    workspace = Workspace("order-support")
    workspace.verify_error = httpx.ConnectError("refused")
    with_key(monkeypatch, workspace)
    decimalai.agent_config("order-support")
    assert not (world.project / ".decimal/link.toml").exists() and workspace.assigned == []
    workspace.verify_error = None
    decimalai.agent_config("order-support")
    assert (world.project / ".decimal/link.toml").exists() and len(workspace.assigned) == 1


def test_a_link_to_another_workspace_is_left_as_it_is(world, monkeypatch, caplog):
    keyless_phase(world)
    link = world.project / ".decimal/link.toml"
    link.write_text('base_url = "https://ws.test"\norg_id = "org-other"\norg_slug = "other"\n'
                    'agents = ["billing"]\n')
    workspace = Workspace("order-support")
    with_key(monkeypatch, workspace)
    with caplog.at_level("WARNING", logger="decimalai"):
        decimalai.agent_config("order-support")
    assert "organization other" in caplog.text and "delete the file" in caplog.text
    assert workspace.assigned == [] and "org-other" in link.read_text()


def test_a_skill_that_cannot_be_copied_is_said_and_left_to_the_person(world, monkeypatch, capsys):
    """The copy happens once. What failed is named then, and on later starts the project
    is seen to list a skill the agent was not given; nothing is retried behind anyone's back."""
    keyless_phase(world)
    capsys.readouterr()
    workspace = Workspace("order-support")
    workspace.assign_error = RuntimeError("402 linked-skill cap reached")
    with_key(monkeypatch, workspace)
    decimalai.agent_config("order-support")
    err = capsys.readouterr().err
    assert ("agent 'order-support': could not copy acme/refund-policy (402 linked-skill cap "
            "reached) onto it. Add it with `decimalai skills add <name>` or on the agent's page "
            "in the dashboard.") in err
    link = world.project / ".decimal/link.toml"
    assert link.exists() and '"acme/refund-policy"' not in link.read_text(), "only what was copied"
    workspace.assign_error = AssertionError("a start never assigns")
    decimalai.agent_config("order-support")
    assert ("agent 'order-support' gets its skills from DecimalAI, and .decimal/skills.toml lists "
            "acme/refund-policy, which it was not given. Add one with `decimalai skills add <name>` "
            "or in the dashboard; editing the file does not.") in capsys.readouterr().err


def test_a_platform_from_before_the_new_conflict_rule_is_synced_with_newer_wins(world, monkeypatch, capsys):
    own(world)
    workspace = Workspace()
    workspace.policies.discard("remote_wins_on_conflict")
    with_key(monkeypatch, workspace)
    decimalai.agent_config("order-support")
    assert [s["conflict_policy"] for s in workspace.synced] == ["remote_wins_on_conflict", "newer_wins"]
    assert keyless.read_link(None)["assigned"]["order-support"] == {"refund-tone": "authored"}


def test_a_sync_refused_for_another_reason_is_said_after_one_retry(world, monkeypatch, capsys):
    own(world)
    capsys.readouterr()
    workspace = Workspace()
    workspace.sync_refused = True
    with_key(monkeypatch, workspace)
    decimalai.agent_config("order-support")
    assert [s["conflict_policy"] for s in workspace.synced] == ["remote_wins_on_conflict", "newer_wins"]
    err = capsys.readouterr().err
    assert "could not copy refund-tone (SkillRouter request failed (422)" in err
    assert keyless.read_link(None)["assigned"]["order-support"] == {}


def test_a_skill_you_wrote_that_decimalai_changed_is_left_as_decimalai_has_it(world, monkeypatch, capsys):
    own(world)
    capsys.readouterr()
    workspace = Workspace()
    theirs = "# Refund tone\n\nEdited on DecimalAI: apologize once, then the refund."
    workspace.changed_on_decimalai["refund-tone"] = theirs
    with_key(monkeypatch, workspace)
    decimalai.agent_config("order-support")
    err = capsys.readouterr().err
    assert "copied refund-tone (DecimalAI's copy, which differs from yours) onto it" in err
    assert "Apologize once, state the refund" in (world.project / ".agents/skills/refund-tone/SKILL.md").read_text(), \
        "a run never writes your file"
    decimalai.agent_config("order-support")
    assert ("refund-tone in .agents/skills does not match DecimalAI's copy, which agent "
            "'order-support' uses: run `decimalai skills sync` to sync it (a change made on "
            "DecimalAI wins).") in capsys.readouterr().err


def test_a_skill_you_wrote_that_is_deleted_on_decimalai_is_not_claimed_as_copied(world, monkeypatch, capsys):
    """A sync finds a deleted skill by its name and never brings it back, so it stays off the
    agent: the copy says so, and link.toml does not record it as given (found live 2026-10-22)."""
    own(world)
    capsys.readouterr()
    workspace = Workspace()
    workspace.deleted.add("refund-tone")
    with_key(monkeypatch, workspace)
    decimalai.agent_config("order-support")
    err = capsys.readouterr().err
    assert "copied refund-tone" not in err
    assert ("could not copy refund-tone (uploaded, but DecimalAI did not add it to the agent: a "
            "skill named refund-tone may be deleted there) onto it") in err
    link = keyless.read_link(None)
    assert link["assigned"]["order-support"] == {}
    assert "uploaded" not in link


def test_a_registry_skill_edited_on_disk_is_named_when_the_agent_connects(world, monkeypatch, capsys):
    keyless_phase(world)
    installed = world.project / ".agents/skills/acme-refund-policy/SKILL.md"
    installed.write_text(installed.read_text() + "\nAlso refund shipping.\n")
    capsys.readouterr()
    with_key(monkeypatch, Workspace())
    decimalai.agent_config("order-support")
    assert ("acme/refund-policy was edited in .agents/skills/acme-refund-policy, but agent "
            "'order-support' uses the registry's v2: the edit does not carry over. To keep it, "
            "fork the skill on DecimalAI") in capsys.readouterr().err


# ── after connecting: a start reads, says, and changes nothing ───────────────


def connected(world, monkeypatch, capsys, *agents, with_own=False):
    """The tutorial's project after its first keyed runs: refund-policy v2 (and, with
    ``with_own``, refund-tone) copied onto each of ``agents``, ``link.toml`` written.
    Returns the workspace, its requests forgotten."""
    keyless_phase(world)
    if with_own:
        own(world)
    workspace = Workspace()
    with_key(monkeypatch, workspace)
    for agent in agents or ("order-support",):
        decimalai.agent_config(agent, fallback=FALLBACK)
    workspace.assigned.clear()
    workspace.synced.clear()
    workspace.verified = 0
    workspace.assign_error = AssertionError("a keyed start never assigns")
    capsys.readouterr()
    keyless._told.clear()
    return workspace


def add_without_a_key(monkeypatch, ref):
    """`decimalai skills add <ref>` through the engine, with no key in the environment."""
    with monkeypatch.context() as m:
        m.delenv("DECIMAL_API_KEY", raising=False)
        result = skillpkg.add(ref, interactive=False)
    assert result.ok, result


def untouched(world, workspace, before: bytes) -> None:
    assert (workspace.assigned, workspace.synced, workspace.verified) == ([], [], 0)
    assert (world.project / ".decimal/link.toml").read_bytes() == before, "a start writes nothing"


def test_the_link_file_records_what_was_copied_and_reads_back(world, monkeypatch, capsys):
    connected(world, monkeypatch, capsys)
    link = keyless.read_link(None)
    assert link["agents"] == ["order-support"]
    assert link["assigned"] == {"order-support": {"acme/refund-policy": "2"}}


def test_a_skill_added_to_the_file_after_connecting_is_named_not_assigned(world, monkeypatch, capsys):
    workspace = connected(world, monkeypatch, capsys)
    before = (world.project / ".decimal/link.toml").read_bytes()
    add_without_a_key(monkeypatch, "acme/order-tracking")
    decimalai.agent_config("order-support", fallback=FALLBACK)
    assert ("agent 'order-support' gets its skills from DecimalAI, and .decimal/skills.toml lists "
            "acme/order-tracking, which it was not given.") in capsys.readouterr().err
    decimalai.agent_config("order-support", fallback=FALLBACK)
    assert capsys.readouterr().err == "", "said once per process"
    untouched(world, workspace, before)


def test_a_version_changed_in_the_file_is_named_not_assigned(world, monkeypatch, capsys):
    workspace = connected(world, monkeypatch, capsys)
    before = (world.project / ".decimal/link.toml").read_bytes()
    add_without_a_key(monkeypatch, "acme/refund-policy@1")
    decimalai.agent_config("order-support")
    assert (".decimal/skills.toml asks for acme/refund-policy at v1; agent 'order-support''s "
            "version of it is set on DecimalAI: `decimalai skills add acme/refund-policy@1` or the "
            "dashboard.") in capsys.readouterr().err
    untouched(world, workspace, before)


def test_a_skill_removed_from_the_file_leaves_the_agent_and_the_output_alone(world, monkeypatch, capsys):
    workspace = connected(world, monkeypatch, capsys)
    before = (world.project / ".decimal/link.toml").read_bytes()
    assert skillpkg.remove("acme/refund-policy").ok
    decimalai.agent_config("order-support")
    assert capsys.readouterr().err == ""
    untouched(world, workspace, before)


def test_a_skill_removed_from_the_agent_in_the_dashboard_stays_removed(world, monkeypatch, capsys):
    """A keyed start never asks the workspace, so it cannot put back what someone took
    off the agent there."""
    workspace = connected(world, monkeypatch, capsys)
    before = (world.project / ".decimal/link.toml").read_bytes()
    for _ in range(3):
        decimalai.agent_config("order-support")
    assert capsys.readouterr().err == ""
    untouched(world, workspace, before)


def test_a_link_written_by_0_24_reads_as_what_the_file_lists(world, monkeypatch, capsys):
    """A 0.24 link lists the agent and records nothing: 0.24 assigned everything the
    file listed when it connected, so nothing in the file is news."""
    workspace = connected(world, monkeypatch, capsys)
    link = world.project / ".decimal/link.toml"
    link.write_text('base_url = "https://ws.test"\norg_id = "org-1"\norg_slug = "acme"\n'
                    'workspace_id = "ws-1"\nagents = ["order-support"]\n')
    before = link.read_bytes()
    decimalai.agent_config("order-support")
    assert capsys.readouterr().err == ""
    untouched(world, workspace, before)


def test_an_edit_to_a_skill_you_wrote_is_named_until_it_is_synced(world, monkeypatch, capsys):
    workspace = connected(world, monkeypatch, capsys, with_own=True)
    before = (world.project / ".decimal/link.toml").read_bytes()
    tone = world.project / ".agents/skills/refund-tone/SKILL.md"
    tone.write_text(tone.read_text() + "\nSign off with the agent's first name.\n")
    decimalai.agent_config("order-support")
    assert "refund-tone in .agents/skills does not match DecimalAI's copy" in capsys.readouterr().err
    untouched(world, workspace, before)

    # What `decimalai skills sync` records once the two match again.
    keyless.record(_project(world), uploaded={"refund-tone": _authored.body_hash(_project(world), "refund-tone")})
    keyless._told.clear()
    decimalai.agent_config("order-support")
    assert capsys.readouterr().err == ""


def test_each_connected_agent_is_named_on_its_own_start(world, monkeypatch, capsys):
    connected(world, monkeypatch, capsys, "order-support", "billing")
    assert keyless.read_link(None)["agents"] == ["billing", "order-support"]
    add_without_a_key(monkeypatch, "acme/order-tracking")
    decimalai.agent_config("billing")
    decimalai.agent_config("order-support")
    err = capsys.readouterr().err
    assert "agent 'billing' gets its skills from DecimalAI" in err
    assert "agent 'order-support' gets its skills from DecimalAI" in err
