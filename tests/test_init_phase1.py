"""`decimalai.init()` — the flag form delivers skills, and re-init changes org.

Two defects from the 2026-10-13 framework-integration audit, both silent:

1. **The documented one-liner delivered no skills.** The quickstart teaches
   `decimalai.init(langchain=True)` (and `openai_agents=True`, `crewai=True`,
   `adk=True`). That form called the adapter's `instrument()` with the skill
   loader OFF, so a user who followed the docs got good traces and zero skills,
   with nothing anywhere saying so. The flag form now turns the loader on;
   `skills=False` keeps the tracing-only install for anyone who wants it.

2. **Re-init kept the previous org's router.** `init()` rebuilds the trace
   client and drops the prompt cache, because a re-init means a different key,
   workspace or base URL. Each adapter keeps its own `SkillRouter` singleton,
   built from the config that was live when it was first used, with a 30 s
   fragment cache inside it. Those were never dropped, so in a key-rotating or
   multi-tenant process org A's skills kept arriving in org B's prompts. A
   re-init that changes the router's identity now drops them.

Everything here is offline: adapter internals with process-wide side effects
(class patches, global trace-processor lists) are stubbed, and the real adapter
code decides everything else.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, List
from unittest.mock import MagicMock

import pytest

import decimalai
import decimalai._config as cfg


KEY_A = "dai_sk_org_a"
KEY_B = "dai_sk_org_b"
URL = "http://localhost:9"


@pytest.fixture(autouse=True)
def _restore_sdk_globals(monkeypatch):
    """Put the SDK's config and client back after each test."""
    monkeypatch.setattr(cfg, "_config", cfg._config)
    monkeypatch.setattr(cfg, "_client", cfg._client)
    monkeypatch.setattr(decimalai, "_global_client", decimalai._global_client)
    yield


def _init(**kwargs: Any) -> None:
    kwargs.setdefault("api_key", KEY_A)
    kwargs.setdefault("base_url", URL)
    kwargs.setdefault("verify", False)
    decimalai.init(**kwargs)


def _keep(monkeypatch, module: Any, *names: str) -> None:
    """Restore these module globals after the test, whatever it set them to."""
    for name in names:
        monkeypatch.setattr(module, name, getattr(module, name))


def _counter(monkeypatch, module: Any, name: str) -> List[int]:
    """Replace `module.name` with a no-op that counts its calls."""
    calls: List[int] = []
    monkeypatch.setattr(module, name, lambda *a, **k: calls.append(1))
    return calls


# ── 1. the flag form delivers skills ────────────────────────────────


@pytest.fixture
def langchain_adapter(monkeypatch):
    """The real LangChain adapter, minus its process-wide side effects.

    `register_configure_hook` has no unregister and `_install_skill_loader`
    rebinds `BaseChatModel.invoke` for the whole process, so both are stubbed;
    everything `instrument()` decides — including whether to install the
    loader — is the real code.
    """
    pytest.importorskip("langchain_core")
    import langchain_core.tracers.context as lc_ctx

    import decimalai.langchain as lc
    import decimalai.skills as skills_mod

    _keep(
        monkeypatch, lc,
        "_installed", "_skill_loader_installed", "_install_agent_name",
        "_explicit_manifest_config", "_evals", "_builtin_evals_enabled",
        "_skill_body_top_k", "_priority_skills", "_runtime_policy",
    )
    monkeypatch.setattr(lc, "_installed", False)
    monkeypatch.setattr(lc_ctx, "register_configure_hook", lambda *a, **k: None)
    # This machine's own ~/.claude/skills must not be uploaded by a unit test.
    monkeypatch.setattr(skills_mod, "discover_skills", lambda *a, **k: [])
    loader = _counter(monkeypatch, lc, "_install_skill_loader")
    return lc, loader


class TestFlagFormDeliversSkills:
    def test_langchain_flag_installs_the_skill_loader(self, langchain_adapter):
        lc, loader = langchain_adapter
        _init(langchain=True, agent_name="support")
        assert loader == [1], (
            "init(langchain=True) installed tracing without the skill loader — "
            "the documented one-liner delivers zero skills"
        )
        assert lc._install_agent_name == "support"

    def test_langchain_flag_with_skills_false_is_tracing_only(
        self, langchain_adapter
    ):
        lc, loader = langchain_adapter
        _init(langchain=True, agent_name="support", skills=False)
        assert loader == []
        assert lc._installed is True, "skills=False must still install tracing"

    def test_openai_agents_flag_installs_the_loader_and_one_processor(
        self, monkeypatch
    ):
        pytest.importorskip("agents")
        import agents.tracing as agents_tracing

        import decimalai.openai_agents as oa

        processors: List[Any] = []
        monkeypatch.setattr(agents_tracing, "add_trace_processor", processors.append)
        loader = _counter(monkeypatch, oa, "_install_skill_loader")
        hooks = _counter(monkeypatch, oa, "_install_agent_hooks")

        _init(openai_agents=True, agent_name="support")

        assert loader == [1], (
            "init(openai_agents=True) installed tracing without the skill loader"
        )
        assert len(processors) == 1, (
            f"one init() registered {len(processors)} trace processors"
        )
        assert hooks == []  # the loader installs its own hooks

    def test_openai_agents_flag_with_skills_false_is_tracing_only(
        self, monkeypatch
    ):
        pytest.importorskip("agents")
        import agents.tracing as agents_tracing

        import decimalai.openai_agents as oa

        processors: List[Any] = []
        monkeypatch.setattr(agents_tracing, "add_trace_processor", processors.append)
        loader = _counter(monkeypatch, oa, "_install_skill_loader")
        hooks = _counter(monkeypatch, oa, "_install_agent_hooks")

        _init(openai_agents=True, agent_name="support", skills=False)

        assert loader == []
        assert hooks == [1]
        assert len(processors) == 1

    @pytest.mark.parametrize("skills, expected", [(True, True), (False, False)])
    def test_adk_flag_follows_skills(self, monkeypatch, skills, expected):
        import decimalai.adk as adk

        _keep(
            monkeypatch, adk,
            "_install_agent_name", "_skill_loader_enabled", "_instrument_config",
        )
        # Pretend Runner is already patched: the patch is process-wide and
        # irreversible, and the loader decision is made before that point.
        monkeypatch.setattr(adk, "_runner_patched", True)
        monkeypatch.setattr(adk, "_skill_loader_enabled", False)
        monkeypatch.setattr(adk, "_instrument_config", adk._InstrumentationConfig())

        _init(adk=True, agent_name="support", skills=skills)

        assert adk._instrument_config.enable_skill_loader is expected
        assert adk._instrument_config.agent_name == "support"

    @pytest.mark.parametrize("skills, expected", [(True, [1]), (False, [])])
    def test_crewai_flag_follows_skills(self, monkeypatch, skills, expected):
        import decimalai.crewai as dc
        import decimalai.otel as otel
        import decimalai.providers as providers

        _keep(monkeypatch, dc, "_tracing_installed", "_install_agent_name")
        # The exporter and the OpenInference activation are what init(crewai=True)
        # already did and are covered in test_crewai_init_activation.py; here
        # they only need to not touch the process's global tracer provider.
        monkeypatch.setattr(otel, "instrument", lambda **k: MagicMock(name="provider"))
        monkeypatch.setitem(sys.modules, "openinference.instrumentation.crewai", None)
        monkeypatch.setattr(providers, "_sdk_present", lambda _mod: False)
        loader = _counter(monkeypatch, dc, "_install_skill_loader")

        _init(crewai=True, agent_name="support", skills=skills)

        assert loader == expected
        assert dc._install_agent_name == "support"

    def test_crewai_flag_keeps_an_observer_set_before_it(self, monkeypatch):
        """Turning the loader on from init() must not reset what
        `crewai.instrument(on_trace=...)` already published."""
        import decimalai.crewai as dc
        import decimalai.otel as otel
        import decimalai.providers as providers

        _keep(
            monkeypatch, dc,
            "_tracing_installed", "_install_agent_name", "_instrument_config",
        )
        monkeypatch.setattr(otel, "instrument", lambda **k: MagicMock(name="provider"))
        monkeypatch.setitem(sys.modules, "openinference.instrumentation.crewai", None)
        monkeypatch.setattr(providers, "_sdk_present", lambda _mod: False)
        _counter(monkeypatch, dc, "_install_skill_loader")
        observer = dc._InstrumentationConfig(on_trace=lambda trace: None)
        monkeypatch.setattr(dc, "_instrument_config", observer)

        _init(crewai=True, agent_name="support")

        assert dc._instrument_config is observer

    def test_anthropic_provider_flag_stays_tracing_only(self, monkeypatch):
        """`anthropic=True` is the provider TRACING rail, and the docs pair it
        with framework adapters that deliver skills themselves (Pydantic AI on
        an Anthropic model). Patching `messages.create` from it would inject a
        second copy of the skills under that adapter. Raw-Messages skills stay
        `decimalai.anthropic.instrument(enable_skill_loader=True)`."""
        import decimalai.anthropic as da
        import decimalai.providers as providers

        monkeypatch.setattr(providers, "instrument", lambda **k: None)
        loader = _counter(monkeypatch, da, "_install_skill_loader")

        _init(anthropic=True, agent_name="support")

        assert loader == []

    @pytest.mark.parametrize("skills, authority, reads_disk", [
        (True, None, False),       # router is the skills channel: no disk sync
        (True, "harness", True),   # asked for disk explicitly: kept
        (False, None, True),       # tracing only: the pre-change behaviour
    ])
    def test_the_loader_decides_disk_sync_the_way_instrument_does(
        self, langchain_adapter, monkeypatch, tmp_path,
        skills, authority, reads_disk,
    ):
        """Turning the loader on moves the flag form onto the same disk_sync
        default `instrument(enable_skill_loader=True)` has always had."""
        import decimalai.skills as skills_mod

        # The harness case is allowed to pull skills down; never into the repo.
        monkeypatch.chdir(tmp_path)

        reads: List[int] = []
        monkeypatch.setattr(
            skills_mod, "discover_skills", lambda *a, **k: reads.append(1) or []
        )
        kwargs = {"skill_authority": authority} if authority else {}
        _init(langchain=True, agent_name="support", skills=skills, **kwargs)
        assert bool(reads) is reads_disk

    def test_auto_trace_env_is_the_flag_form(self, langchain_adapter, monkeypatch):
        """`DECIMAL_AUTO_TRACE=langchain` is documented as the env spelling of
        `init(langchain=True)`, so it delivers skills too."""
        lc, loader = langchain_adapter
        monkeypatch.setenv("DECIMAL_API_KEY", KEY_A)
        monkeypatch.setenv("DECIMAL_BASE_URL", URL)
        monkeypatch.setenv("DECIMAL_AUTO_TRACE", "langchain")
        decimalai._auto_init_from_env()
        assert loader == [1]

    def test_disabled_installs_nothing_whatever_skills_says(
        self, langchain_adapter
    ):
        lc, loader = langchain_adapter
        _init(langchain=True, enabled=False, skills=True)
        assert loader == []
        assert lc._installed is False

    @pytest.mark.parametrize("bad", [[{"name": "x"}], "yes", 1, None])
    def test_skills_takes_a_bool(self, bad):
        """`instrument(skills=[...])` is a list of skill dicts for the manifest;
        `init(skills=...)` is an on/off switch. Mixing them up must not
        silently mean "on"."""
        with pytest.raises(TypeError, match="skills"):
            _init(langchain=True, skills=bad)


# ── 1b. the flag form and openai_agents idempotency ─────────────────
#
# One init() registers one processor (asserted above). A re-init, and the docs'
# old two-line form (the flag, then an explicit instrument()), must not add a
# second: before the idempotent instrument() each one did, and every run was
# sent twice. Counted on the Agents SDK's LIVE processor list, because that is
# what instrument() itself consults — a stand-in add_trace_processor that only
# records calls is never on that list, so it would read as "not registered yet"
# and count a re-registration the real provider never sees.


class TestOpenAIAgentsRepeatInstall:
    @pytest.fixture
    def processors(self, monkeypatch):
        pytest.importorskip("agents")
        import agents.tracing as agents_tracing

        import decimalai.openai_agents as oa

        provider = agents_tracing.get_trace_provider()
        saved = list(provider._multi_processor._processors)
        agents_tracing.set_trace_processors([])
        # The process's one processor, as if no test had built it yet.
        monkeypatch.setattr(oa, "_tracing_processor", None, raising=False)
        monkeypatch.setattr(oa, "_tracing_processor_added", False, raising=False)
        _counter(monkeypatch, oa, "_install_skill_loader")
        _counter(monkeypatch, oa, "_install_agent_hooks")

        def ours() -> List[Any]:
            return [
                p for p in provider._multi_processor._processors
                if isinstance(p, oa.DecimalTracingProcessor)
            ]

        yield ours
        agents_tracing.set_trace_processors(saved)

    def test_reinit_registers_one_processor(self, processors):
        _init(openai_agents=True, agent_name="support")
        _init(openai_agents=True, agent_name="support", api_key=KEY_B)
        assert len(processors()) == 1

    def test_flag_then_explicit_instrument_registers_one_processor(
        self, processors
    ):
        """The docs' old two-line form: `init(openai_agents=True)` and then
        `instrument(enable_skill_loader=True)`."""
        from decimalai.openai_agents import instrument

        _init(openai_agents=True, agent_name="support")
        instrument(agent_name="support", enable_skill_loader=True)
        assert len(processors()) == 1


# ── 2. re-init drops routers built for the previous org ─────────────

ROUTER_ADAPTERS = (
    "langchain", "openai_agents", "adk", "anthropic", "crewai", "pydantic_ai",
)


def _adapter(name: str, monkeypatch) -> Any:
    mod = importlib.import_module(f"decimalai.{name}")
    monkeypatch.setattr(mod, "_skill_router_singleton", None)
    return mod


class TestReinitDropsAdapterRouters:
    @pytest.mark.parametrize("name", ROUTER_ADAPTERS)
    def test_a_different_key_rebuilds_the_router(self, name, monkeypatch):
        mod = _adapter(name, monkeypatch)
        _init(api_key=KEY_A)
        first = mod._get_skill_router()
        assert first is not None and first.api_key == KEY_A

        _init(api_key=KEY_B)
        second = mod._get_skill_router()

        assert second is not first, (
            f"decimalai.{name} kept the router built for the previous key — "
            "org A's skills keep arriving in org B's prompts"
        )
        assert second.api_key == KEY_B

    @pytest.mark.parametrize("name", ROUTER_ADAPTERS)
    def test_a_different_base_url_rebuilds_the_router(self, name, monkeypatch):
        mod = _adapter(name, monkeypatch)
        _init(base_url="http://localhost:9")
        first = mod._get_skill_router()

        _init(base_url="http://127.0.0.1:9")
        second = mod._get_skill_router()

        assert second is not first
        assert second.base_url == "http://127.0.0.1:9"

    @pytest.mark.parametrize("name", ROUTER_ADAPTERS)
    def test_the_same_identity_keeps_the_router(self, name, monkeypatch):
        """A repeat init() with the same key and URL — e.g. the bare auto-init
        at import followed by the user's own init() — keeps the router and
        its cache. Only a change of identity is a reason to drop it."""
        mod = _adapter(name, monkeypatch)
        _init()
        first = mod._get_skill_router()
        _init()
        assert mod._get_skill_router() is first

    def test_a_different_body_setting_rebuilds_the_router(self, monkeypatch):
        """The router bakes `inject_body` in at construction, from
        `init(inject_skill_body=...)`; a re-init that changes it must not keep
        serving the old setting."""
        mod = _adapter("langchain", monkeypatch)
        _init(inject_skill_body=True)
        first = mod._get_skill_router()
        _init(inject_skill_body=False)
        assert mod._get_skill_router() is not first

    def test_a_router_the_adapter_has_not_built_is_left_alone(self, monkeypatch):
        mod = _adapter("langchain", monkeypatch)
        _init(api_key=KEY_A)
        _init(api_key=KEY_B)
        assert mod._skill_router_singleton is None

    def test_reinit_imports_no_adapter(self, monkeypatch):
        """Adapters are looked up, never imported: importing one would pull its
        framework into a process that never asked for it."""
        monkeypatch.delitem(sys.modules, "decimalai.crewai", raising=False)
        _init(api_key=KEY_A)
        _init(api_key=KEY_B)
        assert "decimalai.crewai" not in sys.modules
