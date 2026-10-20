"""``init(enabled=False)`` must be a real kill switch.

The regression this locks down: ``init(enabled=False, langchain=True)`` used to
skip only the client, then fall through to the framework auto-install block
anyway. ``decimalai.langchain.instrument()`` is not a passive no-op — it calls
the DecimalAI API (it uploads the skills you wrote, ``POST /api/v1/skills/sync``,
and until 0.15 it also pulled every org skill: ``GET /api/v1/skills/hashes``, then
the per-skill body routes, then SKILL.md files in the caller's working tree). So a
user who explicitly turned the SDK off still got network traffic and new files.

Both tests drive the REAL adapter against a fake httpx transport in a temp cwd,
so they observe the two effects a user actually cares about:

* ``test_disabled_*`` — nothing attempted, nothing written.
* ``test_enabled_*`` — the control. It proves the gate is conditional, not a
  blanket off-switch: with ``enabled=True`` the same call still runs the adapter,
  which still syncs the skill the project wrote.

The control asks for disk sync explicitly (``skill_authority="harness"``), and it
now also pins what changed in 0.15: even asked, and even with the fake platform
serving a skill the project does not have, no adapter writes a skill to disk.
Registry skills reach disk only when ``decimalai skills`` runs. What this file
tests about ``enabled`` is unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

import httpx
import pytest


# The fake platform serves one skill that is NOT on local disk. A background
# pull (0.14 and earlier) classified it as "missing" and exported it, a real file
# write; the control asserts that no such write happens any more.
_SKILL_NAME = "kill-switch-demo-skill"
_SKILL_BODY = {
    "skill_id": "skl_kill_switch_demo",
    "name": _SKILL_NAME,
    "description": "Demo skill served by the fake platform.",
    "body": "# Kill Switch Demo\n\nThis body should only ever reach disk when enabled.\n",
}


class _FakePlatform:
    """Records every HTTP attempt and answers the skill-pull endpoints."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def __call__(self, method: str, url: Any, **kwargs: Any) -> httpx.Response:
        url = str(url)
        self.calls.append(f"{method} {url}")
        if "/skills/hashes" in url:
            return httpx.Response(
                200, json={"hashes": {_SKILL_NAME: {"hash": "deadbeefcafe00"}}}
            )
        if url.endswith("/body"):
            return httpx.Response(200, json=_SKILL_BODY)
        return httpx.Response(200, json={})


@pytest.fixture
def platform(monkeypatch) -> _FakePlatform:
    """Intercept every httpx entry point the SDK can reach."""
    fake = _FakePlatform()
    monkeypatch.setattr(httpx, "request", fake)
    monkeypatch.setattr(
        httpx.Client, "request",
        lambda self, method, url, **kw: fake(method, url, **kw),
    )
    monkeypatch.setattr(
        httpx.Client, "send",
        lambda self, request, **kw: fake(request.method, request.url),
    )
    return fake


@pytest.fixture
def fresh_sdk(monkeypatch, tmp_path):
    """A clean SDK + LangChain adapter, rooted in an empty temp project.

    ``instrument()`` is idempotent via a module global, and it registers a
    process-global LangChain hook — both are saved and restored so this file
    can't leak state into the rest of the suite.
    """
    import decimalai._config as cfg
    import decimalai.langchain as dlc
    import langchain_core.tracers.context as lc_ctx

    monkeypatch.chdir(tmp_path)
    # Don't let a real ~/.claude/skills checkout leak into the run.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()

    monkeypatch.setattr(cfg, "_config", None, raising=False)
    monkeypatch.setattr(cfg, "_client", None, raising=False)
    monkeypatch.setattr(dlc, "_installed", False)
    # Keep the real adapter's skill sync/pull intact — only neutralize the
    # global callback registration, which has no bearing on this behaviour.
    monkeypatch.setattr(lc_ctx, "register_configure_hook", lambda *a, **k: None)
    # `init(langchain=True)` also installs the skill loader, which rebinds
    # BaseChatModel's four entry points for the whole process. Put them back,
    # or every later test in the session runs on a patched chat model.
    from langchain_core.language_models.chat_models import BaseChatModel

    for entry_point in ("invoke", "ainvoke", "stream", "astream"):
        monkeypatch.setattr(
            BaseChatModel, entry_point, getattr(BaseChatModel, entry_point)
        )
    monkeypatch.setattr(dlc, "_skill_loader_installed", dlc._skill_loader_installed)
    return tmp_path


def _files_under(root: Path) -> List[str]:
    return sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and not str(p.relative_to(root)).startswith("home/")
    )


def test_disabled_langchain_makes_no_http_call_and_writes_no_files(
    platform, fresh_sdk
):
    """enabled=False + langchain=True: no adapter, no API call, no disk write."""
    import decimalai
    import decimalai._config as cfg
    import decimalai.langchain as dlc

    decimalai.init(
        api_key="dai_sk_test",
        base_url="http://localhost:9",
        enabled=False,
        langchain=True,
    )

    assert platform.calls == [], (
        "init(enabled=False) attempted HTTP calls: " + repr(platform.calls)
    )
    assert _files_under(fresh_sdk) == [], (
        "init(enabled=False) wrote files into the project: "
        + repr(_files_under(fresh_sdk))
    )
    assert dlc._installed is False, "the LangChain adapter was installed anyway"
    assert cfg._client is None


def test_enabled_langchain_still_syncs_and_writes_no_skill(platform, fresh_sdk):
    """The control: enabled=True still runs the adapter, which uploads the skill the
    project wrote — and, asked for disk sync or not, writes no skill to disk."""
    import decimalai

    mine = fresh_sdk / ".claude" / "skills" / "my-own"
    mine.mkdir(parents=True)
    (mine / "SKILL.md").write_text(
        "---\nname: my-own\ndescription: Mine.\n---\n# Mine\n\nDo it my way.\n",
        encoding="utf-8",
    )
    before = _files_under(fresh_sdk)

    decimalai.init(
        api_key="dai_sk_test",
        base_url="http://localhost:9",
        enabled=True,
        langchain=True,
        # Ask for disk sync explicitly: without it the loader turns sync off, and
        # the control would prove nothing about `enabled`.
        skill_authority="harness",
    )
    decimalai.flush()

    assert any("/skills/sync" in c for c in platform.calls), (
        "enabled=True no longer syncs the project's own skills: " + repr(platform.calls)
    )
    assert not any("/skills/hashes" in c for c in platform.calls), (
        "instrument() pulled platform skills again: " + repr(platform.calls)
    )
    written = [p for p in _files_under(fresh_sdk) if p not in before]
    assert not any(p.endswith("SKILL.md") for p in written), (
        "instrument() wrote a skill to disk: " + repr(written)
    )


# ── A later init(enabled=False) stops a skill loader an earlier init installed ──
#
# `init(langchain=True)` (and the openai_agents / adk / crewai flags) install
# the skill loader, which patches the framework for the life of the process. A
# later `init(enabled=False)` cleared the trace client, so traces stopped, but
# every loader kept routing: each user message still went to /skills/route and
# skills were still injected, after the caller had turned the SDK off.

_ROUTE = {
    "strategy": "smart_routing",
    "skills": [{"name": "refund-policy", "description": "Refund rules"}],
    "prompt_fragment": "## Recommended Skills\n| refund-policy | Refund rules |",
    "routing_id": "rt_" + "a" * 24,
}
_ADAPTERS = ("langchain", "openai_agents", "adk", "crewai", "anthropic", "pydantic_ai")


class _RoutingPlatform(_FakePlatform):
    def __call__(self, method: str, url: Any, **kwargs: Any) -> httpx.Response:
        url = str(url)
        if url.endswith("/skills/route") or url.endswith("/skills/menu"):
            self.calls.append(f"{method} {url}")
            return httpx.Response(200, json=_ROUTE)
        return super().__call__(method, url, **kwargs)

    def skill_calls(self) -> List[str]:
        return [c for c in self.calls if "/skills/route" in c or "/skills/menu" in c
                or c.endswith("/body")]


@pytest.mark.parametrize("adapter", _ADAPTERS)
def test_every_adapters_router_is_withheld_while_disabled(monkeypatch, adapter):
    """The one door every loader's routing, body and load_skill calls go
    through. Disabled, it opens to nothing; enabled again, the same router."""
    import importlib

    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    module = importlib.import_module(f"decimalai.{adapter}")
    router = object()
    monkeypatch.setattr(module, "_skill_router_singleton", router)

    monkeypatch.setattr(cfg, "_config", DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:9", enabled=False,
    ))
    assert module._get_skill_router() is None, (
        f"decimalai.{adapter} still hands out its router after init(enabled=False)"
    )
    assert module._skill_router_singleton is router, "kept for when it is re-enabled"

    monkeypatch.setattr(cfg, "_config", DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:9", enabled=True,
    ))
    assert module._get_skill_router() is router


@pytest.mark.parametrize("adapter", _ADAPTERS)
def test_a_disabled_config_builds_no_router(monkeypatch, adapter):
    import importlib

    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    module = importlib.import_module(f"decimalai.{adapter}")
    monkeypatch.setattr(module, "_skill_router_singleton", None)
    monkeypatch.setattr(cfg, "_config", DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:9", enabled=False,
    ))
    assert module._get_skill_router() is None
    assert module._skill_router_singleton is None


def test_langchain_loader_stops_routing_after_a_disabled_reinit(monkeypatch, fresh_sdk):
    """The real loader on the real BaseChatModel: routes while enabled, makes
    no call to the platform once disabled, and routes again when re-enabled."""
    import decimalai
    import decimalai.langchain as dlc
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    fake = _RoutingPlatform()
    monkeypatch.setattr(httpx, "request", fake)
    monkeypatch.setattr(dlc, "_skill_router_singleton", None)
    seen: List[Any] = []
    real_generate = FakeListChatModel._generate

    def recording_generate(self, messages, *args, **kwargs):
        seen.append(" ".join(str(m.content) for m in messages))
        return real_generate(self, messages, *args, **kwargs)

    monkeypatch.setattr(FakeListChatModel, "_generate", recording_generate)
    llm = FakeListChatModel(responses=["ok"] * 3)

    decimalai.init(api_key="dai_sk_test", base_url="http://localhost:9", langchain=True)
    llm.invoke("how do refunds work?")
    assert fake.skill_calls(), "control: the enabled loader did not route at all"
    assert "Recommended Skills" in seen[-1]

    decimalai.init(api_key="dai_sk_test", base_url="http://localhost:9", enabled=False)
    before = list(fake.skill_calls())
    llm.invoke("can I return an opened box?")
    assert fake.skill_calls() == before, (
        "init(enabled=False) did not stop the skill loader: "
        + repr(fake.skill_calls()[len(before):])
    )
    assert "Recommended Skills" not in seen[-1]

    decimalai.init(api_key="dai_sk_test", base_url="http://localhost:9", enabled=True)
    llm.invoke("what is the restocking fee?")
    assert len(fake.skill_calls()) > len(before), "re-enabling did not resume routing"
    assert "Recommended Skills" in seen[-1]


def test_openai_agents_load_agent_skills_stop_after_a_disabled_reinit(monkeypatch, tmp_path):
    """An agent config built while enabled keeps a copy of its skill
    list. After `init(enabled=False)` it neither routes nor shows that copy;
    the prompt it read still runs."""
    agents = pytest.importorskip("agents")
    import asyncio

    import decimalai
    import decimalai._config as cfg
    import decimalai.openai_agents as oa
    from agents.tracing import get_trace_provider, set_trace_processors

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DECIMALAI_SUPPRESS_DISK_RUNTIME_WARNING", "1")
    monkeypatch.setattr(cfg, "_config", None, raising=False)
    monkeypatch.setattr(cfg, "_client", None, raising=False)
    monkeypatch.setattr(oa, "_skill_router_singleton", None)
    monkeypatch.setattr(oa, "_tracing_processor", None, raising=False)
    provider = get_trace_provider()
    monkeypatch.setattr(provider._multi_processor, "_processors",
                        tuple(provider._multi_processor._processors))
    set_trace_processors([])
    fake = _RoutingPlatform()
    monkeypatch.setattr(httpx, "request", fake)

    decimalai.init(api_key="dai_sk_test", base_url="http://localhost:9", openai_agents=True)
    monkeypatch.setattr(cfg._client, "get_agent_prompt", lambda name, version=None: {
        "agent_name": name, "system_prompt": "You help.", "version_number": 1,
    })
    support = oa.agent_config("support")
    agent = agents.Agent(name="support", instructions=support.instructions, tools=[*support.tools])

    def prompt(text):
        ctx = agents.RunContextWrapper(context=None, turn_input=[{"role": "user", "content": text}])
        return asyncio.run(support.instructions(ctx, agent))

    assert "Recommended Skills" in prompt("how do refunds work?"), "control"

    decimalai.init(api_key="dai_sk_test", base_url="http://localhost:9", enabled=False)
    before = list(fake.skill_calls())
    shown = prompt("can I return an opened box?")
    assert "Recommended Skills" not in shown and shown.startswith("You help.")
    assert fake.skill_calls() == before


@pytest.mark.parametrize("module", ["decimalai", "decimalai.openai_agents", "decimalai.pydantic_ai",
                                    "decimalai.adk", "decimalai.anthropic"])
def test_no_agent_config_turns_the_kill_switch_back_off(monkeypatch, module):
    """`init(enabled=False)` leaves no client; an `agent_config()` that re-initialized from the
    environment on seeing none turned DecimalAI back on for the whole process."""
    import importlib

    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    try:
        binding_module = importlib.import_module(module)
        binding_module.agent_config  # noqa: B018
    except (ImportError, AttributeError):
        pytest.skip(f"{module} has no agent_config() here")
    monkeypatch.setattr(cfg, "_config", DecimalConfig(api_key="dai_sk_test", base_url="http://127.0.0.1:9",
                                                      enabled=False))
    monkeypatch.setattr(cfg, "_client", None)
    if hasattr(binding_module, "_ensure_tracing"):
        # It registers a process-wide trace processor (OpenAI Agents): not this test's subject,
        # and left behind it makes later tests' runs export twice.
        monkeypatch.setattr(binding_module, "_ensure_tracing", lambda: None)
    called = []
    monkeypatch.setattr("decimalai.init", lambda *a, **k: called.append(k))
    try:
        binding_module.agent_config("support", fallback="You help.")
    except ImportError:
        pytest.skip(f"{module}'s framework is not installed here")
    assert called == [], f"{module}.agent_config() switched DecimalAI back on"
    assert cfg._config.enabled is False and cfg._client is None


def test_under_the_kill_switch_agent_config_runs_on_its_fallback_without_asking(monkeypatch):
    import decimalai
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    monkeypatch.setattr(cfg, "_config", DecimalConfig(api_key="dai_sk_test", base_url="http://127.0.0.1:9",
                                                      enabled=False))
    monkeypatch.setattr(cfg, "_client", None)

    def no_client():
        raise AssertionError("DecimalAI was asked")

    monkeypatch.setattr(cfg, "_get_client", no_client)
    support = decimalai.agent_config("support", fallback="You help.")
    assert support.prompt_source == "fallback (unreachable)"
    assert support.run("Hi").system == "You help."
    with pytest.raises(RuntimeError, match="switched off"):
        decimalai.agent_config("support")


def test_anthropic_load_agent_skills_stop_after_a_disabled_reinit(monkeypatch, tmp_path):
    """The same for an Anthropic `agent_config()` config, through a real re-init: built
    while enabled, it neither routes nor shows its copy of the skill list in
    `run.system` after `init(enabled=False)`; the prompt it read still runs."""
    pytest.importorskip("anthropic")
    import decimalai
    import decimalai._config as cfg
    import decimalai.anthropic as da

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DECIMALAI_SUPPRESS_DISK_RUNTIME_WARNING", "1")
    monkeypatch.setattr(cfg, "_config", None, raising=False)
    monkeypatch.setattr(cfg, "_client", None, raising=False)
    monkeypatch.setattr(da, "_skill_router_singleton", None)
    # agent_config() turns on Anthropic tracing for the process; not from a unit test.
    monkeypatch.setattr(da, "_tracing_turned_on", True)
    fake = _RoutingPlatform()
    monkeypatch.setattr(httpx, "request", fake)

    decimalai.init(api_key="dai_sk_test", base_url="http://localhost:9")
    monkeypatch.setattr(cfg._client, "get_agent_prompt", lambda name, version=None: {
        "agent_name": name, "system_prompt": "You help.", "version_number": 1,
    })
    support = da.agent_config("support")

    def system(text):
        with support.run(text) as run:
            return "\n".join(block["text"] for block in run.system)

    assert "Recommended Skills" in system("how do refunds work?"), "control"

    decimalai.init(api_key="dai_sk_test", base_url="http://localhost:9", enabled=False)
    before = list(fake.skill_calls())
    shown = system("can I return an opened box?")
    assert "Recommended Skills" not in shown and shown.startswith("You help.")
    assert fake.skill_calls() == before
