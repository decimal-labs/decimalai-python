"""The installed agent name must reach the Router on the auto-inject paths.

Why this file exists. ``resolve_agent_skills`` on the platform builds an
agent's offered set from two sources: org-OWNED skills whose ``offer_scope``
is ``workspace`` (ambient — every agent sees them), and ``SkillSubscription``
rows, where an agent-scope row matches on ``agent_name``.

A skill *pulled from the registry* belongs to another org, so it is never in
the ambient set. It reaches one agent only through an agent-scope row — which
the resolver can only match when the router is told which agent it is serving.

``langchain`` and ``anthropic`` both auto-injected skills without sending the
name, so every registry skill attached to a single agent was silently never
offered: no error, no empty-menu warning, just a prompt with none of the
skills the user picked. ``openai_agents`` and ``pydantic_ai`` already sent it.
Since 0.21 Anthropic's skills come from ``agent_config(name)``, which names that
agent on every routing call and every load.

These tests pin the argument at the call boundary rather than asserting on a
rendered fragment, because the bug was never in what the Router returned — it
was in what the adapter asked for.
"""

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest


class _SpyRouter:
    """Records what the adapter asked the Router for."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def build_prompt_fragment(
        self, query: Optional[str] = None, **kwargs: Any
    ) -> Tuple[str, Optional[str]]:
        self.calls.append({"query": query, **kwargs})
        return "", None

    # The adapters drain these rails off the router after a call (a agent_config()
    # run by its scope). They tolerate them being absent, but a spy that answers
    # them keeps the test exercising the real code path instead of an exception
    # handler.
    def consume_routing_id(self, **kwargs: Any) -> None:
        return None

    def consume_offered_names(self, **kwargs: Any) -> List[str]:
        return []

    def consume_delivered_names(self, **kwargs: Any) -> List[str]:
        return []

    def consume_loaded_names(self, **kwargs: Any) -> List[str]:
        return []

    def consume_loaded_hashes(self, **kwargs: Any) -> Dict[str, str]:
        return {}

    def consume_delivered_versions(self, **kwargs: Any) -> List[Dict[str, str]]:
        return []


class TestLangChainSendsAgentName:
    def _inject(self, monkeypatch, agent_name: Optional[str]):
        from langchain_core.messages import HumanMessage

        import decimalai.langchain as lc

        spy = _SpyRouter()
        monkeypatch.setattr(lc, "_get_skill_router", lambda: spy)
        monkeypatch.setattr(lc, "_install_agent_name", agent_name)
        lc._inject_skills_into_input([HumanMessage(content="refund my order")])
        return spy

    def test_installed_agent_name_reaches_the_router(self, monkeypatch):
        spy = self._inject(monkeypatch, "refund-bot")

        assert spy.calls, "the adapter never consulted the Router"
        assert spy.calls[0]["agent_name"] == "refund-bot", (
            "agent_name did not reach build_prompt_fragment — agent-scope "
            "skills (every registry skill pulled onto one agent) resolve to "
            "nothing without it"
        )

    def test_query_is_still_extracted(self, monkeypatch):
        """Scoping must not cost the semantic-routing signal."""
        spy = self._inject(monkeypatch, "refund-bot")
        assert spy.calls[0]["query"] == "refund my order"

    def test_no_installed_name_sends_none_not_a_placeholder(self, monkeypatch):
        """Un-named installs keep the old org-wide behaviour.

        A placeholder here would be worse than None: it would match no
        agent-scope row and silently scope the menu to an agent that does
        not exist, rather than falling back to the ambient set.
        """
        spy = self._inject(monkeypatch, None)
        assert spy.calls[0]["agent_name"] is None

    def test_router_failure_still_returns_input_unchanged(self, monkeypatch):
        """Fail-open is the property that lets this run in a request path."""
        from langchain_core.messages import HumanMessage

        import decimalai.langchain as lc

        class _Boom:
            def build_prompt_fragment(self, *a: Any, **k: Any):
                raise RuntimeError("router down")

        monkeypatch.setattr(lc, "_get_skill_router", lambda: _Boom())
        monkeypatch.setattr(lc, "_install_agent_name", "refund-bot")

        original = [HumanMessage(content="hi")]
        assert lc._inject_skills_into_input(original) is original


class TestAnthropicSendsAgentName:
    """`decimalai.anthropic.agent_config(name)`: the name reaches the routing call
    and the load, the two requests a registry skill's agent-scope row is matched on."""

    def _run(self, monkeypatch) -> "_SpyRouter":
        pytest.importorskip("anthropic")
        import decimalai._agent as agent_mod
        import decimalai._agent_binding as binding
        import decimalai._config as cfg
        import decimalai.anthropic as an
        from decimalai import skill_router as sr
        from decimalai._agent import SavedPrompt
        from decimalai._config import DecimalConfig

        class _LoadingSpy(_SpyRouter):
            def build_prompt_parts(self, query=None, **kwargs: Any):
                self.calls.append({"query": query, **kwargs})
                sr._last_offered_names_ctx.set(["refund-policy"])
                return "## Available Skills\n| refund-policy | Refund rules |", "", "rt_scope"

            def load_skill(self, name: str, **kwargs: Any) -> str:
                self.calls.append({"load": name, **kwargs})
                return f"## Skill: {name}\n\nRefund within 30 days."

        spy = _LoadingSpy()
        monkeypatch.setattr(cfg, "_config", DecimalConfig(api_key="dai_sk_test", enabled=True))
        monkeypatch.setattr(cfg, "_client", MagicMock())
        monkeypatch.setattr(an, "_get_skill_router", lambda: spy)
        monkeypatch.setattr(an, "_tracing_turned_on", True)
        monkeypatch.setattr(binding, "_in_background", lambda fn, *args: fn(*args))
        monkeypatch.setattr(agent_mod, "read_prompt", lambda name, **kw: SavedPrompt(
            agent_name=name, system_prompt="You are a refunds agent."))
        try:
            with an.agent_config("refund-bot").run("refund my order") as run:
                run.tool_result({"type": "tool_use", "id": "toolu_1", "name": "load_skill",
                          "input": {"name": "refund-policy"}})
        finally:
            sr._last_offered_names_ctx.set(None)
        return spy

    def test_the_load_agent_name_reaches_the_router_and_the_load(self, monkeypatch):
        route, load = self._run(monkeypatch).calls

        assert route["agent_name"] == "refund-bot", (
            "agent_name did not reach build_prompt_parts — agent-scope skills "
            "(every registry skill pulled onto one agent) resolve to nothing without it"
        )
        assert load["load"] == "refund-policy" and load["agent_name"] == "refund-bot"
        assert load["scope"] == route["scope"] is not None, "the load is filed under its run"

    def test_query_is_still_extracted(self, monkeypatch):
        """Scoping must not cost the semantic-routing signal."""
        route, _load = self._run(monkeypatch).calls
        assert route["query"] == "refund my order"


def test_openai_agents_still_passes_agent_name_at_call_time():
    """Guards the adapter this fix was modelled on against regression."""
    import inspect

    import decimalai.openai_agents as oa

    # Was `agent_name=getattr(agent, "name", None)`: `Agent(name=...)` won
    # routing outright. Since 0.18 the name is the one `agent_config(name)` was
    # given, on every routing call and every load, whatever the Agent's own
    # `name=` — graded end to end in tests/test_init_agent_name_defaults.py.
    src = inspect.getsource(oa.OpenAIAgentConfig)
    assert "agent_name=self.name" in src, (
        "openai_agents stopped passing agent_name — the reference "
        "implementation for per-agent skill scoping"
    )


# ── Streaming delivery ───────────────────────────────────────────────────────
#
# Skills were delivered on `invoke`/`ainvoke` only. A production chat agent
# calls `.stream()`, which never touches either, so the most common shape of
# agent in the wild received no skills at all — with no error to notice.
#
# The double-injection test is the other half and matters just as much:
# `invoke` calls `generate` internally, and so does `stream` on a model with no
# native `_stream`. Patching `generate` as well — the obvious way to "close the
# gap" — would inject the menu twice on both paths.

MENU = "## Recommended Skills\n| plant-care | waters things |"


@pytest.fixture
def loader_installed(monkeypatch):
    """Install the REAL monkey-patch, then put BaseChatModel back."""
    from langchain_core.language_models.chat_models import BaseChatModel

    import decimalai.langchain as lc

    patched = ("invoke", "ainvoke", "stream", "astream")
    originals = {name: getattr(BaseChatModel, name) for name in patched}

    monkeypatch.setattr(lc, "_skill_loader_installed", False)
    monkeypatch.setattr(lc, "_install_agent_name", "refund-bot")
    lc._install_skill_loader()
    try:
        yield
    finally:
        for name, fn in originals.items():
            setattr(BaseChatModel, name, fn)
        lc._skill_loader_installed = False


@pytest.fixture
def spy(monkeypatch):
    import decimalai.langchain as lc

    class _Router(_SpyRouter):
        def build_prompt_fragment(self, query=None, **kwargs):
            self.calls.append({"query": query, **kwargs})
            return MENU, None

    r = _Router()
    monkeypatch.setattr(lc, "_get_skill_router", lambda: r)
    return r


def _fake_models():
    from typing import Iterator

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, AIMessageChunk
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

    seen: List[Any] = []

    class NoNativeStream(BaseChatModel):
        """No `_stream` — LangChain falls back through `generate`."""

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            seen.append(list(messages))
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="ok"))]
            )

        @property
        def _llm_type(self) -> str:
            return "fake"

    class NativeStream(NoNativeStream):
        """Has `_stream`, like every production provider — bypasses `generate`."""

        def _stream(self, messages, stop=None, run_manager=None, **kw) -> Iterator[Any]:
            seen.append(list(messages))
            yield ChatGenerationChunk(message=AIMessageChunk(content="ok"))

        @property
        def _llm_type(self) -> str:
            return "native"

    return NoNativeStream, NativeStream, seen


def _menu_reached_model(seen: List[Any]) -> bool:
    return any(
        MENU in getattr(m, "content", "")
        for msgs in seen
        for m in msgs
        if m.__class__.__name__ == "SystemMessage"
    )


class TestStreamingGetsSkills:
    def test_sync_stream_delivers_skills(self, loader_installed, spy):
        from langchain_core.messages import HumanMessage

        _, Native, seen = _fake_models()
        list(Native().stream([HumanMessage(content="water my fern?")]))

        assert spy.calls, ".stream() never consulted the Router"
        assert _menu_reached_model(seen), "the menu never reached the model on .stream()"

    @pytest.mark.asyncio
    async def test_async_stream_delivers_skills(self, loader_installed, spy):
        from langchain_core.messages import HumanMessage

        _, Native, seen = _fake_models()
        async for _ in Native().astream([HumanMessage(content="water my fern?")]):
            pass

        assert spy.calls, ".astream() never consulted the Router"
        assert _menu_reached_model(seen)

    def test_invoke_injects_exactly_once(self, loader_installed, spy):
        """`invoke` routes through `generate`; a `generate` patch would double it."""
        from langchain_core.messages import HumanMessage

        NoNative, _, seen = _fake_models()
        NoNative().invoke([HumanMessage(content="water my fern?")])

        assert len(spy.calls) == 1, f"expected 1 injection, got {len(spy.calls)}"
        systems = [
            m
            for msgs in seen
            for m in msgs
            if m.__class__.__name__ == "SystemMessage"
        ]
        assert len(systems) == 1, f"menu injected {len(systems)}× into one call"

    def test_stream_fallback_path_injects_exactly_once(self, loader_installed, spy):
        """A model with no `_stream` reaches `generate` via `stream` — still once."""
        from langchain_core.messages import HumanMessage

        NoNative, _, seen = _fake_models()
        list(NoNative().stream([HumanMessage(content="water my fern?")]))

        assert len(spy.calls) == 1, f"expected 1 injection, got {len(spy.calls)}"

    def test_generate_is_left_unpatched_on_purpose(self, loader_installed):
        """Guards the comment: patching it would double-inject."""
        from langchain_core.language_models.chat_models import BaseChatModel

        assert BaseChatModel.generate.__name__ != "patched_generate"
        assert BaseChatModel.agenerate.__name__ != "patched_agenerate"

    def test_stream_stays_lazy(self, loader_installed, spy):
        """Wrapping must not turn a lazy generator into eager work."""
        from langchain_core.messages import HumanMessage

        _, Native, seen = _fake_models()
        gen = Native().stream([HumanMessage(content="hi")])
        assert not spy.calls, "calling .stream() did work before iteration began"
        list(gen)
        assert spy.calls

    @pytest.mark.asyncio
    async def test_astream_fallback_path_injects_exactly_once(
        self, loader_installed, spy
    ):
        """`astream` on a non-streaming model delegates to `ainvoke`."""
        from langchain_core.messages import HumanMessage

        NoNative, _, seen = _fake_models()
        async for _ in NoNative().astream([HumanMessage(content="water my fern?")]):
            pass

        assert len(spy.calls) == 1, f"expected 1 injection, got {len(spy.calls)}"
