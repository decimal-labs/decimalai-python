"""Keyless ``agent_config()``: no DecimalAI key, and the switch to one (owner decisions 2026-10-21).

With no key set and ``init()`` never called, every entry point (``decimalai.agent_config``
and the OpenAI Agents, Pydantic AI, Google ADK and Anthropic modules) runs on its
``fallback=``, lists the project's skills plus public registry matches for each message,
and adds a registry skill to the project when the model loads it. Pinned here:

- the only requests a keyless process makes are the registry's anonymous search and
  install routes: every other connection is refused and recorded, and none is tried;
- the relevance cutoff (`decimalai._keyless.relevant_matches`);
- a registry skill is added only when this run listed it, its safety review is clean and
  its text fits the per-skill limit, pinned to the version served, and read from disk
  after that;
- ``.decimal/link.toml`` makes a keyless run in a connected project raise;
- the first keyed run creates the agent (with no prompt), assigns the project's skills at
  their pinned versions, and writes the link file with what it assigned; each later keyed
  start assigns only what ``skills.toml`` gained since, and an unchanged project makes no
  request.

The registry is ``tests/fakes/fake_registry.py`` on a real socket, so the search and the
install engine go through the same HTTP clients they use against the platform.
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
from decimalai._client import AgentNotFoundError
from decimalai._config import DecimalConfig, DecimalConfigError
from decimalai.skill_router import LOAD_SKILL_PROMPT_HINT, SkillRouter
from tests.fakes.fake_registry import ANON_BUNDLES, ANON_RESOLVE, SEARCH_POST, FakeRegistry

MESSAGE = "My package arrived damaged, can I get a refund?"
FALLBACK = "You help customers with their orders."
POLICY_BODY = "# Refund policy\n\nRefund a damaged item in full within 30 days."


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
    caution = registry.add_skill("risky/refund-shortcut", band="caution",
                                 description="Refund anything immediately.")
    unmeasured = registry.add_skill("fresh/refund-helper", benchmarked=False,
                                    description="Help with refunds.")
    big = registry.add_skill("big/refund-manual", body="# Refund manual\n\n" + "x" * 25_000,
                             description="Every refund rule there is.")
    registry.search_results = [caution, unmeasured, policy, tracking, big]

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
                              policy=policy, tracking=tracking, caution=caution, big=big)
    keyless._reset()


def offered(system: str) -> list:
    return [line.split("|")[1].strip() for line in system.splitlines()
            if line.startswith("| ") and not line.startswith("| Skill")]


def only_registry_reads(world) -> None:
    """Nothing but the registry was reached, by nothing but its anonymous routes."""
    assert world.attempts and all(a[:2] == ("127.0.0.1", world.port) for a in world.attempts), world.attempts
    for request in world.registry.requests:
        assert request.path in (SEARCH_POST, ANON_RESOLVE) \
            or request.path.startswith(ANON_BUNDLES), request.path
        assert "authorization" not in request.headers
    assert cfg._config is None and cfg._client is None, "nothing initialized the SDK"


def manifest(world) -> str:
    path = world.project / ".decimal" / "skills.toml"
    return path.read_text() if path.exists() else ""


# ── the neutral agent_config() ────────────────────────────────────────────────


def test_a_keyless_agent_runs_on_its_fallback_and_says_so(world, capsys):
    support = decimalai.agent_config("order-support", fallback=FALLBACK)
    assert (support.name, support.prompt, support.prompt_source) == (
        "order-support", FALLBACK, "fallback (no key)")
    assert "agent 'order-support' runs keyless" in capsys.readouterr().err
    run = support.run(MESSAGE)
    assert run.system.startswith(FALLBACK + "\n\n## Available Skills")
    assert run.system.endswith(LOAD_SKILL_PROMPT_HINT)
    decimalai.agent_config("order-support", fallback=FALLBACK)
    assert "keyless" not in capsys.readouterr().err, "said once per agent"


def test_without_a_fallback_the_instructions_are_the_skill_list(world):
    support = decimalai.agent_config("order-support")
    assert (support.prompt, support.prompt_source) == (None, "fallback (no key)")
    assert support.run(MESSAGE).system.startswith("## Available Skills")


def test_a_run_lists_safe_benchmarked_matches_that_share_a_word_with_its_message(world):
    run = decimalai.agent_config("order-support", fallback=FALLBACK).run(MESSAGE)
    # caution and unbenchmarked are not offered; order-tracking shares no word; at most two.
    assert offered(run.system) == ["acme/refund-policy", "big/refund-manual"]
    assert "enum" not in run.tool_spec["parameters"]["properties"]["name"], "any listed name loads"
    [search] = world.registry.calls("search")
    assert (search.method, search.path, search.json["q"]) == ("POST", SEARCH_POST, MESSAGE)
    assert search.query == {}, "the message is in the body, never the URL"
    assert world.project.joinpath(".decimal").exists() is False, "listing writes nothing"
    only_registry_reads(world)


def test_loading_a_listed_registry_skill_adds_it_pinned_and_later_runs_read_it_from_disk(world, capsys):
    support = decimalai.agent_config("order-support", fallback=FALLBACK)
    run = support.run(MESSAGE)
    text = run.load_skill("acme/refund-policy")
    assert text == f"## Skill: acme/refund-policy\n\n{POLICY_BODY}"
    assert '"acme/refund-policy" = "2"' in manifest(world), "pinned to the version served"
    assert (world.project / ".agents/skills/acme-refund-policy/SKILL.md").is_file()
    err = capsys.readouterr().err
    assert err.count("added skill acme/refund-policy v2 to .decimal/skills.toml") == 1
    assert len(world.registry.calls("bundle")) == 1, "the bundle checked is the bundle installed"
    assert "already loaded" in run.load_skill("acme/refund-policy")

    before = len(world.registry.requests)
    later = support.run("Thanks, that helps.")
    assert offered(later.system)[0] == "acme/refund-policy", "the project's own skill, listed first"
    assert later.load_skill("acme/refund-policy").startswith("## Skill: acme/refund-policy")
    assert [r.route for r in world.registry.requests[before:]] == ["search"], "served from disk"
    only_registry_reads(world)


def test_a_name_this_run_did_not_list_adds_nothing(world):
    run = decimalai.agent_config("order-support").run(MESSAGE)
    assert run.load_skill("acme/order-tracking").startswith("load_skill: no skill named 'acme/order-tracking'")
    assert run.load_skill("risky/refund-shortcut").startswith("load_skill: no skill named")
    assert not world.registry.calls("resolve") and manifest(world) == ""


def test_a_skill_whose_review_is_no_longer_clean_is_not_added_or_listed_again(world):
    support = decimalai.agent_config("order-support")
    run = support.run(MESSAGE)
    world.policy.band = "caution"   # reviewed again after the search answered
    text = run.load_skill("acme/refund-policy")
    assert text.startswith("load_skill: 'acme/refund-policy' cannot be used here (its safety review is caution)")
    assert manifest(world) == "" and not world.registry.calls("bundle")
    world.policy.band = "clean"
    assert "acme/refund-policy" not in offered(support.run(MESSAGE).system)


def test_a_skill_too_long_to_serve_is_not_added_or_listed_again(world):
    support = decimalai.agent_config("order-support")
    run = support.run(MESSAGE)
    text = run.load_skill("big/refund-manual")
    assert "over the 24000-character limit for one skill" in text
    assert manifest(world) == "" and not (world.project / ".agents").exists()
    assert "big/refund-manual" not in offered(support.run(MESSAGE).system)


def test_a_registry_without_the_post_route_is_not_searched(world, caplog):
    """Its GET search would carry each message in a URL. The registry that happens to
    is api.decimal.ai itself, rolled back to before the POST route."""
    world.registry.search_post = False
    support = decimalai.agent_config("order-support")
    with caplog.at_level("WARNING", logger="decimalai"):
        assert offered(support.run(MESSAGE).system) == []
        assert offered(support.run("My refund never arrived.").system) == []
    asked = [(r.method, r.path) for r in world.registry.calls("search")]
    assert asked == [("POST", SEARCH_POST)], "asked once, and never with the GET"
    assert sum("has no POST /api/v1/registry/skills/search" in r.getMessage()
               for r in caplog.records) == 1


def test_the_message_is_in_no_url_a_keyless_run_requests(world):
    """Not the search's, and not the install routes' a load makes after it. Read off
    each request as it travelled, percent-encoding intact, and decoded."""
    marker = "Zanzibar-4417"
    message = f"My package {marker} arrived damaged, can I get a refund?"
    support = decimalai.agent_config("order-support")
    run = support.run(message)
    assert run.load_skill("acme/refund-policy").startswith("## Skill:")
    support.run(message)
    requests = world.registry.requests
    assert {r.route for r in requests} >= {"search", "resolve", "bundle"}, [r.route for r in requests]
    for r in requests:
        travelled = r.raw_path + "?" + urllib.parse.urlencode(r.query)
        for text in (travelled, urllib.parse.unquote_plus(travelled)):
            assert marker not in text and "damaged" not in text, (r.method, travelled)
    searched = world.registry.calls("search")
    assert searched and all(r.method == "POST" and marker in r.json["q"] for r in searched), \
        "the message reached the registry, in the body"


def test_a_failed_search_lists_the_projects_skills_alone(world, caplog):
    support = decimalai.agent_config("order-support")
    support.run(MESSAGE).load_skill("acme/refund-policy")
    world.registry.fault("search", 503, 503)
    with caplog.at_level("WARNING", logger="decimalai"):
        assert offered(support.run(MESSAGE).system) == ["acme/refund-policy"]
        assert offered(support.run(MESSAGE).system) == ["acme/refund-policy"]
    assert sum("registry search failed" in r.getMessage() for r in caplog.records) == 1


def test_a_keyless_process_never_initializes_or_traces(world):
    before = decimalai.export_status()  # process-wide: earlier tests' traces count too
    support = decimalai.agent_config("order-support", fallback=FALLBACK)
    run = support.run(MESSAGE)
    run.load_skill("acme/refund-policy")
    decimalai.flush()
    after = decimalai.export_status()
    assert (after.sent, after.failed) == (before.sent, before.failed)
    only_registry_reads(world)


# ── every entry point ─────────────────────────────────────────────────────────


def test_openai_agents_keyless(world, monkeypatch):
    pytest.importorskip("agents")
    from agents import Agent, Runner
    from agents.tracing import get_trace_provider, set_trace_processors

    import decimalai.openai_agents as oa
    from tests.test_openai_agents_load_agent import ScriptedModel, lookup_order

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
    assert '"acme/refund-policy" = "2"' in manifest(world)
    only_registry_reads(world)


def test_pydantic_ai_keyless(world):
    pytest.importorskip("pydantic_ai")
    import decimalai.pydantic_ai as pai
    from tests.test_pydantic_ai_load_agent import Model, agent_with

    support = pai.agent_config("order-support", fallback=FALLBACK)
    assert support.prompt_source == "fallback (no key)"
    model = Model(("call", "acme/refund-policy"), "answer")
    assert agent_with(support, model).run_sync(MESSAGE, deps={}).output == "done"
    first, second = model.requests
    assert first["instructions"].startswith(FALLBACK + "\n\n## Available Skills")
    assert second["returns"][0].startswith("## Skill: acme/refund-policy")
    assert '"acme/refund-policy" = "2"' in manifest(world)
    only_registry_reads(world)


def test_adk_keyless(world):
    pytest.importorskip("google.adk")
    import decimalai.adk as adk
    from tests.test_adk_load_agent import model, ours, run

    support = adk.agent_config("order-support", fallback=FALLBACK)
    assert support.prompt_source == "fallback (no key)"
    llm = model(("call", "acme/refund-policy"), "answer")
    answer, _ = run(support, llm, MESSAGE, state={})
    assert answer == "done"
    first, second = llm.requests
    assert ours(first).startswith(FALLBACK + "\n\n## Available Skills")
    assert second["returns"][0]["result"].startswith("## Skill: acme/refund-policy")
    assert '"acme/refund-policy" = "2"' in manifest(world)
    only_registry_reads(world)


def test_anthropic_keyless(world, monkeypatch):
    pytest.importorskip("anthropic")
    pytest.importorskip("httpx2")
    import decimalai.anthropic as da
    from tests.test_anthropic_load_agent import Messages, tool_runner

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
    assert '"acme/refund-policy" = "2"' in manifest(world)
    only_registry_reads(world)


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


# ── the relevance cutoff ──────────────────────────────────────────────────────


def card(name, description="", *, safety="passed", bench=True):
    return {"name": name, "description": description, "skill_safety": safety,
            "benchmark_summary": {"verdict": "lift"} if bench else None}


def test_the_cutoff_reads_the_first_five_results_and_offers_two():
    items = [card("a/one", "nothing in common"), card("b/refunds", "refunds"),
             card("c/refund-rules", "rules"), card("d/refund-faq", "faq"),
             card("e/whatever", "no"), card("f/refund-late", "too late")]
    names = [i["name"] for i in keyless.relevant_matches("I want a refund", items)]
    assert names == ["b/refunds", "c/refund-rules"]
    late = [card("a/x", "x")] * 5 + [card("f/refund-late", "refund")]
    assert keyless.relevant_matches("I want a refund", late) == [], "past the first five"


def test_the_cutoff_needs_a_review_a_benchmark_and_a_shared_word():
    items = [card("a/refund", safety="caution"), card("b/refund", bench=False),
             card("c/refund", safety="unreviewed"), card("d/track", "parcel status")]
    assert keyless.relevant_matches("refund my order", items) == []
    assert keyless.relevant_matches("Hi, thanks for your help!", [card("tolz/do-stuff", "helper")]) == []


def test_the_cutoff_offers_one_skill_per_slug_and_none_the_project_has():
    items = [card("x/returns-logistics", "returns"), card("returns-logistics", "returns"),
             card("y/return-policy", "returns")]
    assert [i["name"] for i in keyless.relevant_matches("returns", items)] == [
        "x/returns-logistics", "y/return-policy"]
    assert [i["name"] for i in keyless.relevant_matches(
        "returns", items, exclude=["acme/return-policy"])] == ["x/returns-logistics"]


# ── adding a key ──────────────────────────────────────────────────────────────


class Workspace:
    """The platform behind a key: agents, the verify route, and assignments."""

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
        if "/prompt" in path and method != "GET":
            self.prompt_writes.append(path)
        assert method == "POST" and path.startswith("/api/v1/agents/") and "/skills/" in path, path
        if self.assign_error is not None:
            raise self.assign_error
        self.assigned.append((path, dict(params or {})))
        return {"status": "ok", "action": "assigned"}


def with_key(monkeypatch, workspace):
    """The same process, with a key: init() has run, and the router answers from ``workspace``."""
    monkeypatch.setenv("DECIMAL_API_KEY", "dai_sk_test")
    monkeypatch.setattr(cfg, "_config", DecimalConfig(api_key="dai_sk_test", base_url="https://ws.test"))
    monkeypatch.setattr(cfg, "_client", workspace)
    router = SkillRouter(api_key="dai_sk_test", base_url="https://ws.test", inject_body=False)
    monkeypatch.setattr(router, "_request", workspace.request)
    monkeypatch.setattr(agent_mod, "_skill_router_singleton", router)


def keyless_phase(world):
    """What the tutorial's first runs leave: one registry skill added on use, pinned."""
    run = decimalai.agent_config("order-support", fallback=FALLBACK).run(MESSAGE)
    assert run.load_skill("acme/refund-policy").startswith("## Skill:")


def test_adding_a_key_creates_the_agent_assigns_the_projects_skills_and_links_the_project(
        world, monkeypatch, capsys):
    keyless_phase(world)
    lock_entry = next(iter(__import__("json").loads(
        (world.project / ".decimal/skills.lock").read_text())["skills"].values()))
    capsys.readouterr()

    workspace = Workspace()
    with_key(monkeypatch, workspace)
    support = decimalai.agent_config("order-support", fallback=FALLBACK)
    assert workspace.created == [("order-support", {})], "created with no prompt: the code's is not saved"
    assert support.prompt_source == "fallback (none saved)" and support.prompt == FALLBACK
    assert workspace.assigned == [(f"/api/v1/agents/order-support/skills/{lock_entry['skill_id']}",
                                   {"pin_version_id": lock_entry["version_id"]})]
    assert workspace.prompt_writes == []
    link = (world.project / ".decimal/link.toml").read_text()
    assert 'base_url = "https://ws.test"' in link and 'workspace_id = "ws-1"' in link
    assert 'org_id = "org-1"' in link and 'org_slug = "acme"' in link
    assert 'agents = ["order-support"]' in link and "billing" not in link
    err = capsys.readouterr().err
    assert "created agent 'order-support'" in err
    assert ("connected agent 'order-support' to your workspace: assigned acme/refund-policy v2; "
            "wrote .decimal/link.toml. Commit it.") in err

    decimalai.agent_config("order-support", fallback=FALLBACK)
    assert (workspace.verified, len(workspace.assigned), capsys.readouterr().err) == (1, 1, ""), \
        "connected once"

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
    from decimalai import skillpkg

    skillpkg.add("acme/order-tracking", interactive=False, api_key="")
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


def test_a_skill_that_cannot_be_assigned_is_said_and_tried_again_on_the_next_start(
        world, monkeypatch, capsys):
    keyless_phase(world)
    capsys.readouterr()
    workspace = Workspace("order-support")
    workspace.assign_error = RuntimeError("402 linked-skill cap reached")
    with_key(monkeypatch, workspace)
    decimalai.agent_config("order-support")
    err = capsys.readouterr().err
    assert "could not assign acme/refund-policy (402 linked-skill cap reached)" in err
    link = world.project / ".decimal/link.toml"
    assert link.exists() and '"acme/refund-policy"' not in link.read_text(), "only what was assigned"
    decimalai.agent_config("order-support")
    assert "could not assign acme/refund-policy" in capsys.readouterr().err
    workspace.assign_error = None
    decimalai.agent_config("order-support")
    assert "assigned acme/refund-policy v2" in capsys.readouterr().err
    assert '"acme/refund-policy" = "2"' in link.read_text()
    decimalai.agent_config("order-support")
    assert capsys.readouterr().err == "" and len(workspace.assigned) == 1


# ── after connecting: .decimal/skills.toml and the workspace converge ─────────


def connected(world, monkeypatch, capsys, *agents):
    """The tutorial's project after its first keyed runs: refund-policy v2 assigned to each
    of ``agents``, ``link.toml`` written. Returns the workspace, its requests forgotten."""
    keyless_phase(world)
    workspace = Workspace()
    with_key(monkeypatch, workspace)
    for agent in agents or ("order-support",):
        decimalai.agent_config(agent, fallback=FALLBACK)
    workspace.assigned.clear()
    workspace.verified = 0
    capsys.readouterr()
    return workspace


def add_without_a_key(monkeypatch, ref):
    """`decimalai skills add <ref>` run with no key in the environment."""
    from decimalai import skillpkg

    with monkeypatch.context() as m:
        m.delenv("DECIMAL_API_KEY", raising=False)
        result = skillpkg.add(ref, interactive=False)
    assert result.ok, result


def lock_entry(world, name):
    import json

    skills = json.loads((world.project / ".decimal/skills.lock").read_text())["skills"]
    return next(e for e in skills.values() if e["name"] == name)


def test_the_link_file_records_what_was_assigned_and_reads_back(world, monkeypatch, capsys):
    connected(world, monkeypatch, capsys)
    link = keyless.read_link(None)
    assert link["agents"] == ["order-support"]
    assert link["assigned"] == {"order-support": {"acme/refund-policy": "2"}}


def test_a_skill_added_after_connecting_reaches_the_agent_on_its_next_keyed_start(
        world, monkeypatch, capsys):
    """`decimalai skills add` only writes the project: without a key it cannot reach the
    workspace, and with one the install routes resolve and fetch but link nothing (checked
    against the platform live). So the next keyed start assigns it."""
    workspace = connected(world, monkeypatch, capsys)
    add_without_a_key(monkeypatch, "acme/order-tracking")
    assert workspace.assigned == [], "adding a skill makes no workspace request"

    decimalai.agent_config("order-support", fallback=FALLBACK)
    tracking = lock_entry(world, "acme/order-tracking")
    assert workspace.assigned == [(f"/api/v1/agents/order-support/skills/{tracking['skill_id']}",
                                   {"pin_version_id": None})], "only the new skill"
    assert workspace.verified == 1, "the key is checked against the link before assigning"
    err = capsys.readouterr().err
    assert ("agent 'order-support': assigned acme/order-tracking (latest) from .decimal/skills.toml; "
            "updated .decimal/link.toml. Commit it.") in err
    assert keyless.read_link(None)["assigned"]["order-support"] == {
        "acme/order-tracking": "latest", "acme/refund-policy": "2"}

    decimalai.agent_config("order-support", fallback=FALLBACK)
    assert (len(workspace.assigned), workspace.verified, capsys.readouterr().err) == (1, 1, ""), \
        "an unchanged project makes no request"


def test_a_skill_moved_to_another_version_is_assigned_at_it(world, monkeypatch, capsys):

    workspace = connected(world, monkeypatch, capsys)
    add_without_a_key(monkeypatch, "acme/refund-policy@1")
    decimalai.agent_config("order-support")
    entry = lock_entry(world, "acme/refund-policy")
    assert entry["version"] == 1
    assert workspace.assigned == [(f"/api/v1/agents/order-support/skills/{entry['skill_id']}",
                                   {"pin_version_id": entry["version_id"]})]
    assert "assigned acme/refund-policy v1" in capsys.readouterr().err
    assert keyless.read_link(None)["assigned"]["order-support"] == {"acme/refund-policy": "1"}


def test_a_removed_skill_is_dropped_from_the_record_and_left_assigned(world, monkeypatch, capsys):
    """Unassigning needs an admin key, and a skill may be on the agent for other reasons:
    a run never unassigns. It says so once, and makes no request."""
    from decimalai import skillpkg

    workspace = connected(world, monkeypatch, capsys)
    assert skillpkg.remove("acme/refund-policy").ok
    decimalai.agent_config("order-support")
    assert (workspace.assigned, workspace.verified) == ([], 0)
    err = capsys.readouterr().err
    assert "acme/refund-policy left .decimal/skills.toml and stays assigned in your workspace" in err
    assert keyless.read_link(None)["assigned"]["order-support"] == {}
    decimalai.agent_config("order-support")
    assert capsys.readouterr().err == ""


def test_a_skill_removed_from_the_agent_in_the_dashboard_stays_removed(world, monkeypatch, capsys):
    """The record, not the workspace, says what was assigned: a keyed start never asks the
    workspace, so it cannot put back what someone took off the agent there."""
    workspace = connected(world, monkeypatch, capsys)
    workspace.assign_error = AssertionError("nothing may be assigned")
    for _ in range(3):
        decimalai.agent_config("order-support")
    assert (workspace.assigned, workspace.verified, capsys.readouterr().err) == ([], 0, "")


def test_a_link_written_before_the_record_assigns_the_projects_skills_once(world, monkeypatch, capsys):
    """A 0.24 link file lists the agent and records no skills."""
    workspace = connected(world, monkeypatch, capsys)
    link = world.project / ".decimal/link.toml"
    link.write_text('base_url = "https://ws.test"\norg_id = "org-1"\norg_slug = "acme"\n'
                    'workspace_id = "ws-1"\nagents = ["order-support"]\n')
    decimalai.agent_config("order-support")
    assert len(workspace.assigned) == 1
    assert keyless.read_link(None)["assigned"]["order-support"] == {"acme/refund-policy": "2"}
    decimalai.agent_config("order-support")
    assert len(workspace.assigned) == 1


def test_each_connected_agent_gets_the_new_skill_on_its_own_start(world, monkeypatch, capsys):

    workspace = connected(world, monkeypatch, capsys, "order-support", "billing")
    assert keyless.read_link(None)["agents"] == ["billing", "order-support"]
    add_without_a_key(monkeypatch, "acme/order-tracking")
    decimalai.agent_config("billing")
    assert [path.split("/")[4] for path, _ in workspace.assigned] == ["billing"]
    assigned = keyless.read_link(None)["assigned"]
    assert assigned["billing"] == {"acme/order-tracking": "latest", "acme/refund-policy": "2"}
    assert assigned["order-support"] == {"acme/refund-policy": "2"}, "not assigned for it yet"
    decimalai.agent_config("order-support")
    assert [path.split("/")[4] for path, _ in workspace.assigned] == ["billing", "order-support"]


def test_a_key_of_another_organization_assigns_nothing_to_its_workspace(world, monkeypatch, capsys, caplog):

    connected(world, monkeypatch, capsys)
    add_without_a_key(monkeypatch, "acme/order-tracking")
    other = Workspace("order-support", org_id="org-2", workspace_id="ws-9")
    with_key(monkeypatch, other)
    with caplog.at_level("WARNING", logger="decimalai"):
        decimalai.agent_config("order-support")
    assert other.assigned == [] and "delete the file" in caplog.text
    assert "acme/order-tracking" not in keyless.read_link(None)["assigned"]["order-support"]


def test_a_platform_that_does_not_answer_assigns_the_new_skill_on_a_later_start(world, monkeypatch, capsys):

    workspace = connected(world, monkeypatch, capsys)
    add_without_a_key(monkeypatch, "acme/order-tracking")
    workspace.verify_error = httpx.ConnectError("refused")
    decimalai.agent_config("order-support")
    assert workspace.assigned == []
    assert "acme/order-tracking" not in keyless.read_link(None)["assigned"]["order-support"]
    workspace.verify_error = None
    decimalai.agent_config("order-support")
    assert len(workspace.assigned) == 1
