"""The Pydantic AI run scope (``decimalai.pydantic_ai.instrument()``).

Pydantic AI emits no spans of its own, so every span in one of its traces comes
from the provider instrumentor underneath — one unparented root span per
provider call. A single ``agent.run_sync()`` that asks for a tool and then
answers therefore arrived as two unrelated one-span traces, each its own OTel
trace_id, all of them filed under whichever agent name the exporter was built
with. ``instrument()`` now brackets each run so the calls have a real parent and
the Agent's own name travels with them.

Driven with a *synthetic* ``pydantic_ai.Agent`` (no pydantic-ai install
required), matching the existing skill-loader tests next door. The one thing a
fake cannot vouch for — that ``run``/``run_sync`` really do funnel through
``Agent.iter``, which is why that is the patch point — is checked against the
real package in ``TestRealPackageShape``, and skipped when it is absent.
"""

from __future__ import annotations

import asyncio
import sys
import types
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

from decimalai import providers  # noqa: E402
from decimalai.otel import _active_agent_name  # noqa: E402

# ── synthetic pydantic_ai ────────────────────────────────────────────────────

#: What Pydantic AI's own name inference produces for an unnamed Agent: the
#: local variable it was assigned to. Never a DecimalAI agent, always plausible.
INFERRED_NAME = "agent"


def _make_fake_pydantic_ai(seen):
    """A ``pydantic_ai`` whose Agent has the real funnel shape: run→iter."""
    mod = types.ModuleType("pydantic_ai")

    class Agent:
        def __init__(self, *, name=None):
            # Keyword-only, like the real one: `name` is behind the `*` in
            # pydantic_ai.Agent.__init__, which is what lets the adapter read the
            # caller's answer out of **kwargs and know it was given.
            self.name = name

        @asynccontextmanager
        async def iter(self, prompt):
            # Whatever the adapter published is visible from inside the run.
            seen.append(_active_agent_name.get())
            yield f"agent-run:{prompt}"

        async def run(self, prompt):
            async with self.iter(prompt) as run:
                return run

        def run_sync(self, prompt):
            # Pydantic AI's `_infer_name`: an Agent with no name gets one from
            # the local VARIABLE it was assigned to, filled in before iter()
            # opens. Modelled here because it is the behaviour
            # `instrument(agent_name=…)` exists to sit in front of — without it
            # the fake would make an unnamed Agent look like it has no name at
            # all, which is the one state the real package never leaves it in.
            if self.name is None:
                self.name = INFERRED_NAME
            return asyncio.run(self.run(prompt))

    mod.Agent = Agent
    return mod, Agent


@pytest.fixture(autouse=True)
def _clean_context():
    """Start from "no agent is running".

    ``_active_agent_name`` is a process-wide ContextVar that several adapters
    publish to, so an earlier test in the same session leaves its agent name
    behind — and these tests assert on exactly that value.
    """
    token = _active_agent_name.set(None)
    yield
    _active_agent_name.reset(token)


@pytest.fixture
def fake_pydantic_ai(monkeypatch):
    """Install a synthetic pydantic_ai and reset the adapter's install guard."""
    import decimalai.pydantic_ai as pa

    seen: list = []
    mod, Agent = _make_fake_pydantic_ai(seen)
    monkeypatch.setitem(sys.modules, "pydantic_ai", mod)
    monkeypatch.setattr(pa, "_run_scope_installed", False, raising=False)
    # Reset with the run scope, because both are installed by the same call and
    # both are module-global. `_name_capture_installed` left True would let a
    # later test's Agent class go uncaptured — and `_default_agent_name` left
    # set would answer for a test that never asked for it, which is the shape of
    # a test that passes for the previous test's reason.
    monkeypatch.setattr(pa, "_name_capture_installed", False, raising=False)
    monkeypatch.setattr(pa, "_default_agent_name", None, raising=False)
    yield types.SimpleNamespace(module=mod, Agent=Agent, seen=seen)
    pa._run_scope_installed = False
    pa._name_capture_installed = False
    pa._default_agent_name = None


@pytest.fixture
def pipeline():
    """A caller-owned provider with the real DecimalAI exporter on it."""
    from opentelemetry.sdk.trace import TracerProvider

    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    prev = (cfg._config, cfg._client, providers._last_provider)
    cfg._config = DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {
        "manifest_id": "test-manifest-id", "status": "active",
    }
    tp = TracerProvider()
    providers._ensure_pipeline(None, tp)  # also points _last_provider at tp

    def traces():
        from decimalai._config import _sender

        _sender.flush()
        return [c[0][0] for c in cfg._client.ingest_trace.call_args_list]

    yield types.SimpleNamespace(provider=tp, traces=traces)
    cfg._config, cfg._client, providers._last_provider = prev


# ── the fix ──────────────────────────────────────────────────────────────────


class TestRunScopeInstall:
    def test_a_run_is_scoped_to_the_agents_own_name(self, fake_pydantic_ai, pipeline):
        import decimalai.pydantic_ai as pa

        pa.instrument()
        fake_pydantic_ai.Agent(name="support-bot").run_sync("hello")

        assert fake_pydantic_ai.seen == ["support-bot"]

    def test_two_agents_in_one_process_are_told_apart(self, fake_pydantic_ai, pipeline):
        """The exporter's name is fixed when it is built; the Agent's is not."""
        import decimalai.pydantic_ai as pa

        pa.instrument()
        fake_pydantic_ai.Agent(name="agent-a").run_sync("hi")
        fake_pydantic_ai.Agent(name="agent-b").run_sync("hi")

        assert fake_pydantic_ai.seen == ["agent-a", "agent-b"]
        assert sorted(t.agent_name for t in pipeline.traces()) == ["agent-a", "agent-b"]

    def test_the_run_emits_one_trace_with_a_real_root(self, fake_pydantic_ai, pipeline):
        import decimalai.pydantic_ai as pa

        pa.instrument()
        fake_pydantic_ai.Agent(name="agent-a").run_sync("hi")

        (trace,) = pipeline.traces()
        roots = [s for s in trace.spans if s.parent_span_id is None]
        assert [s.name for s in roots] == ["agent.run"]

    def test_the_agents_return_value_is_untouched(self, fake_pydantic_ai, pipeline):
        import decimalai.pydantic_ai as pa

        pa.instrument()
        result = fake_pydantic_ai.Agent(name="agent-a").run_sync("hello")

        assert result == "agent-run:hello"

    def test_trace_runs_false_leaves_the_agent_unpatched(
        self, fake_pydantic_ai, pipeline
    ):
        import decimalai.pydantic_ai as pa

        pa.instrument(trace_runs=False)
        fake_pydantic_ai.Agent(name="agent-a").run_sync("hi")

        assert fake_pydantic_ai.seen == [None]
        assert pipeline.traces() == []

    def test_install_is_idempotent(self, fake_pydantic_ai, pipeline):
        """A second instrument() must not wrap the wrapper — that would put a
        second, redundant ``agent.run`` layer in every waterfall."""
        import decimalai.pydantic_ai as pa

        pa.instrument()
        pa.instrument()
        fake_pydantic_ai.Agent(name="agent-a").run_sync("hi")

        (trace,) = pipeline.traces()
        assert [s.name for s in trace.spans] == ["agent.run"]

    def test_a_raising_run_releases_the_name(self, fake_pydantic_ai, pipeline):
        """A run that blows up must not leave its agent name published — the
        next run in this thread would then be filed under the wrong agent."""
        import decimalai.pydantic_ai as pa

        class Boom(Exception):
            pass

        @asynccontextmanager
        async def exploding_iter(self, prompt):
            raise Boom("the provider refused this on purpose")
            yield  # pragma: no cover - unreachable, keeps this a generator

        # Replaced BEFORE instrument(), so the run scope wraps the failure
        # rather than being bypassed by it.
        fake_pydantic_ai.Agent.iter = exploding_iter
        pa.instrument()

        with pytest.raises(Boom):
            fake_pydantic_ai.Agent(name="agent-a").run_sync("hi")

        assert _active_agent_name.get() is None


class TestBoundAgentName:
    """`instrument(agent_name=...)` — the fallback for an Agent nobody named.

    Added 2026-08-29 with the pydantic-ai scaffold, and this is the failure it
    answers: Pydantic AI does not leave an unnamed Agent nameless, it INVENTS a
    name from the local variable. So `agent = Agent("openai:gpt-4o-mini")` runs
    as the agent `"agent"`, the DecimalAI run scope stamps that on the trace, and
    the dashboard page for the agent the user actually configured stays empty
    forever while traces pile up under a name nobody chose. Nothing errors.

    Three precedence rules, one per test, most specific first.
    """

    def test_an_unnamed_agent_runs_under_the_bound_name(
        self, fake_pydantic_ai, pipeline
    ):
        import decimalai.pydantic_ai as pa

        pa.instrument(agent_name="refund-bot")
        fake_pydantic_ai.Agent().run_sync("hi")

        assert fake_pydantic_ai.seen == ["refund-bot"]
        assert [t.agent_name for t in pipeline.traces()] == ["refund-bot"]

    def test_the_agents_own_name_still_wins(self, fake_pydantic_ai, pipeline):
        """A process running two agents must not collapse to one.

        The bound name is a FALLBACK. If it overrode an explicit
        `Agent(name=...)` it would recreate the very bug it was added for, just
        pointing the other way.
        """
        import decimalai.pydantic_ai as pa

        pa.instrument(agent_name="refund-bot")
        fake_pydantic_ai.Agent(name="order-bot").run_sync("hi")

        assert fake_pydantic_ai.seen == ["order-bot"]

    def test_without_a_bound_name_nothing_changes(self, fake_pydantic_ai, pipeline):
        """The pre-existing behaviour, kept: no `agent_name=`, no new opinion.

        An adapter that started overriding inferred names for everyone would
        move the traces of every existing caller on upgrade.
        """
        import decimalai.pydantic_ai as pa

        pa.instrument()
        fake_pydantic_ai.Agent().run_sync("hi")

        assert fake_pydantic_ai.seen == [INFERRED_NAME]

    def test_a_named_agent_built_before_the_bind_keeps_its_name(
        self, fake_pydantic_ai, pipeline
    ):
        """The capture rides on EVERY instrument(), not only a named one.

        The capture patch is the only thing that tells "named by the caller"
        apart from "named by inference", and by run time both look identical on
        `Agent.name`. So it has to go on `Agent.__init__` from the first
        instrument() call rather than from whichever call happens to pass
        `agent_name` — otherwise an Agent constructed in between is recorded as
        never having been named, and the process-wide fallback silently takes a
        name its owner chose explicitly.

        The unnamed case does NOT test this: it reaches the same answer with or
        without the capture, because a missing capture and a missing name both
        fall through to the bound name. Only an explicitly named Agent
        distinguishes them, which is why this test names one.
        """
        import decimalai.pydantic_ai as pa

        pa.instrument()
        agent = fake_pydantic_ai.Agent(name="order-bot")  # named, before the bind
        pa.instrument(agent_name="refund-bot")            # bound afterwards
        agent.run_sync("hi")

        assert fake_pydantic_ai.seen == ["order-bot"]


class TestRealPackageShape:
    """The assumption the patch point rests on, checked against the real package."""

    def test_every_entry_point_funnels_through_agent_iter(self):
        import inspect

        pytest.importorskip("pydantic_ai")
        from pydantic_ai import Agent

        assert callable(getattr(Agent, "iter", None)), (
            "Agent.iter is gone — decimalai.pydantic_ai patches it because it is "
            "the one place every run funnels through. Re-find the funnel before "
            "moving the patch, or runs go back to one trace per provider call."
        )
        for entry in ("run", "run_stream"):
            fn = getattr(Agent, entry, None)
            if fn is None:
                continue
            assert "self.iter(" in inspect.getsource(fn), (
                f"Agent.{entry} no longer opens self.iter(), so the DecimalAI run "
                f"scope no longer brackets it"
            )

    def test_agent_name_is_keyword_only(self):
        """`_install_name_capture` reads the caller's name out of **kwargs.

        That is complete only while `name` is keyword-only. If it ever became
        positional, `Agent(model, "refund-bot")` would be recorded as an UNNAMED
        Agent — so an explicit name would silently lose to the bound fallback,
        and two agents in one process would collapse onto one page.
        """
        import inspect

        pytest.importorskip("pydantic_ai")
        from pydantic_ai import Agent

        param = inspect.signature(Agent.__init__).parameters["name"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"pydantic_ai.Agent.__init__ now takes `name` as {param.kind}. "
            "decimalai/pydantic_ai.py::_install_name_capture reads it from "
            "**kwargs, which no longer sees every spelling."
        )

    def test_the_name_pydantic_ai_infers_is_the_variable_name(self):
        """The fake's inference matches the real one — checked, not assumed.

        `TestBoundAgentName` is only meaningful if an unnamed real Agent really
        does end up named after its variable. If Pydantic AI ever stopped doing
        that, those tests would keep passing against a fake that models a
        behaviour the product no longer has.
        """
        pytest.importorskip("pydantic_ai")
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        agent = Agent(TestModel())
        assert agent.name is None
        agent.run_sync("hello")
        assert agent.name == "agent", (
            f"an unnamed Agent assigned to `agent` ran as {agent.name!r}. The "
            "fake in this module models `_infer_name` as 'the variable name'; "
            "if that changed, INFERRED_NAME and the fake must change with it."
        )
