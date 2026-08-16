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


def _make_fake_pydantic_ai(seen):
    """A ``pydantic_ai`` whose Agent has the real funnel shape: run→iter."""
    mod = types.ModuleType("pydantic_ai")

    class Agent:
        def __init__(self, name=None):
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
    yield types.SimpleNamespace(module=mod, Agent=Agent, seen=seen)
    pa._run_scope_installed = False


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
