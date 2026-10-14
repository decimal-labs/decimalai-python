"""``init(enabled=False)`` must be a real kill switch.

The regression this locks down: ``init(enabled=False, langchain=True)`` used to
skip only the client, then fall through to the framework auto-install block
anyway. ``decimalai.langchain.instrument()`` is not a passive no-op — it calls
the DecimalAI API (``GET /api/v1/skills/hashes``, then the per-skill body
routes) and writes SKILL.md files into the caller's working tree. So a user who
explicitly turned the SDK off still got network traffic and new files on disk.

Both tests drive the REAL adapter against a fake httpx transport in a temp cwd,
so they observe the two effects a user actually cares about:

* ``test_disabled_*`` — nothing attempted, nothing written (fails without the
  gate: the old code records ``GET .../skills/hashes`` and writes a SKILL.md).
* ``test_enabled_*`` — the control. It proves the gate is conditional, not a
  blanket off-switch: with ``enabled=True`` the same call still syncs and still
  writes.

The control asks for disk mirroring explicitly (``skill_authority="harness"``).
It did not have to when this file was written, because writing to the caller's
working tree was the unconditional default — the very default that turned out to
be the bug in ``tests/test_disk_mirror_consent.py``: instrumenting an agent
created ``.agents/skills/`` in whatever directory the user ran from, and with no
disk-loading runtime in the process nothing read those files back. Consent is
now required, so the control states it. What this file tests is unchanged:
``enabled`` gates the adapter, and nothing else about it moved.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

import httpx
import pytest


# The fake platform serves one skill that is NOT on local disk, so the pull
# path classifies it as "missing" and exports it — i.e. a real file write.
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


def test_enabled_langchain_still_syncs_and_writes(platform, fresh_sdk):
    """The control: enabled=True behaviour is unchanged by the gate."""
    import decimalai

    decimalai.init(
        api_key="dai_sk_test",
        base_url="http://localhost:9",
        enabled=True,
        langchain=True,
        # Opt in to disk mirroring: writing is no longer implicit, so without
        # this the control would prove nothing about `enabled`.
        skill_authority="harness",
    )

    assert any("/skills/hashes" in c for c in platform.calls), (
        "enabled=True no longer pulls skills from the platform: "
        + repr(platform.calls)
    )
    written = _files_under(fresh_sdk)
    assert any(p.endswith("SKILL.md") for p in written), (
        "enabled=True no longer writes skills to disk: " + repr(written)
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


def test_openai_agents_loader_stops_routing_after_a_disabled_reinit(monkeypatch, tmp_path):
    agents = pytest.importorskip("agents")
    import asyncio

    import decimalai
    import decimalai._config as cfg
    import decimalai.openai_agents as oa
    from agents.run_internal import turn_resolution
    from agents.tracing import get_trace_provider, set_trace_processors

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DECIMALAI_SUPPRESS_DISK_RUNTIME_WARNING", "1")
    monkeypatch.setattr(cfg, "_config", None, raising=False)
    monkeypatch.setattr(cfg, "_client", None, raising=False)
    for method in ("__init__", "get_system_prompt", "get_all_tools"):
        monkeypatch.setattr(agents.Agent, method, getattr(agents.Agent, method))
    monkeypatch.setattr(turn_resolution, "check_for_final_output_from_tools",
                        turn_resolution.check_for_final_output_from_tools)
    for name in ("_skill_loader_installed", "_agent_hooks_installed",
                 "_final_output_guard_installed", "_installed"):
        if hasattr(oa, name):
            monkeypatch.setattr(oa, name, False)
    monkeypatch.setattr(oa, "_skill_router_singleton", None)
    monkeypatch.setattr(oa, "_tracing_processor", None, raising=False)
    provider = get_trace_provider()
    monkeypatch.setattr(provider._multi_processor, "_processors",
                        tuple(provider._multi_processor._processors))
    set_trace_processors([])
    fake = _RoutingPlatform()
    monkeypatch.setattr(httpx, "request", fake)

    def prompt(agent, text):
        ctx = agents.RunContextWrapper(context=None, turn_input=[{"role": "user", "content": text}])
        return asyncio.run(agent.get_system_prompt(ctx)) or ""

    decimalai.init(api_key="dai_sk_test", base_url="http://localhost:9", openai_agents=True)
    agent = agents.Agent(name="support", instructions="You help.")
    assert "Recommended Skills" in prompt(agent, "how do refunds work?"), "control"

    decimalai.init(api_key="dai_sk_test", base_url="http://localhost:9", enabled=False)
    before = list(fake.skill_calls())
    assert "Recommended Skills" not in prompt(agent, "can I return an opened box?")
    assert fake.skill_calls() == before
