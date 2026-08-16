"""Tests for ADK error-path trace finalization (decimalai.adk).

An invocation that raises out of ``Runner.run_async`` (e.g. a model 429)
never reaches ``after_agent``/``after_run`` — ADK's node runtime only fires
those on success. The plugin's ``on_run_error_callback`` must flush the
otherwise-orphaned run state as an ERROR trace, without ever double-sending
a run that already finalized on the success path.

Drives the plugin's callbacks directly with synthetic contexts — no
google-adk install, no network. ``google.adk.plugins.base_plugin`` is
stubbed in sys.modules so ``_plugin_class()`` builds against a local
BasePlugin. Mirrors test_openai_agents.py: a mock client on the global
config, read the RunTrace from ingest_trace.call_args.
"""

from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from decimalai.schema.common import FinishReason, Status
from decimalai.schema.manifest import ManifestTracker


# ── Setup/Teardown ──────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_sdk(monkeypatch):
    """Reset global SDK state and stub google-adk before each test."""
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    cfg._config = DecimalConfig(
        api_key="dai_sk_test",
        base_url="http://localhost:8000",
        enabled=True,
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {"manifest_id": "test-manifest-id", "status": "active"}

    # Stub the one google-adk import _plugin_class() performs, so the tests
    # run without google-adk installed (and deterministically with it).
    base_plugin_mod = types.ModuleType("google.adk.plugins.base_plugin")

    class BasePlugin:  # minimal stand-in: real one just stores the name
        def __init__(self, name):
            self.name = name

    base_plugin_mod.BasePlugin = BasePlugin
    plugins_mod = types.ModuleType("google.adk.plugins")
    plugins_mod.base_plugin = base_plugin_mod
    adk_pkg_mod = types.ModuleType("google.adk")
    adk_pkg_mod.plugins = plugins_mod
    google_mod = types.ModuleType("google")
    google_mod.adk = adk_pkg_mod
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.adk", adk_pkg_mod)
    monkeypatch.setitem(sys.modules, "google.adk.plugins", plugins_mod)
    monkeypatch.setitem(sys.modules, "google.adk.plugins.base_plugin", base_plugin_mod)

    # Reset decimalai.adk module state; restore after so a cached class
    # built against the stub never leaks into other tests.
    import decimalai.adk as adk

    # `_manifest_ids` is a per-agent dict, not a single module global: the
    # single-global form filed a second agent's traces under the first agent's
    # manifest, which conformance C6 caught. Copy it so a mutation here cannot
    # leak, and restore the original object.
    saved = (adk._PluginClass, dict(adk._manifest_ids), dict(adk._manifest_trackers))
    adk._PluginClass = None
    adk._manifest_ids = {}
    adk._manifest_trackers = {}
    yield
    adk._PluginClass, adk._manifest_ids, adk._manifest_trackers = saved


# ── Synthetic ADK objects (plugin reads them via getattr) ───


def _make_agent(name: str = "root_agent") -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        model="gemini-2.0-flash",
        instruction="Answer briefly.",
        tools=[],
        sub_agents=[],
    )


def _invocation_context(agent: SimpleNamespace, inv_id: str = "inv-1") -> SimpleNamespace:
    return SimpleNamespace(agent=agent, invocation_id=inv_id, user_content="hello")


def _callback_context(inv_id: str = "inv-1") -> SimpleNamespace:
    return SimpleNamespace(invocation_id=inv_id, agent_name=None)


def _flushed_traces():
    import decimalai._config as cfg
    from decimalai._config import _sender

    _sender.flush()
    return [call.args[0] for call in cfg._client.ingest_trace.call_args_list]


# ── Tests ───────────────────────────────────────────────────


class TestAdkErrorPathFinalization:
    def test_success_sends_one_trace_and_late_run_error_does_not_resend(self):
        from decimalai.adk import DecimalaiPlugin

        plugin = DecimalaiPlugin(agent_name="test-agent")
        agent = _make_agent()
        ic = _invocation_context(agent)

        async def _drive():
            await plugin.before_run_callback(invocation_context=ic)
            await plugin.after_agent_callback(agent=agent, callback_context=_callback_context())

        asyncio.run(_drive())
        traces = _flushed_traces()
        assert len(traces) == 1
        assert traces[0].status == Status.SUCCESS

        # A run error surfaced after the success finalize (e.g. an after_run
        # plugin failure) must not produce a second trace: the state is gone.
        asyncio.run(
            plugin.on_run_error_callback(invocation_context=ic, error=RuntimeError("late"))
        )
        assert len(_flushed_traces()) == 1

    def test_escaped_model_error_sends_one_error_trace(self):
        from decimalai.adk import DecimalaiPlugin

        plugin = DecimalaiPlugin(agent_name="test-agent")
        agent = _make_agent()
        ic = _invocation_context(agent)
        err = RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")

        async def _drive():
            await plugin.before_run_callback(invocation_context=ic)
            await plugin.before_model_callback(
                callback_context=_callback_context(),
                llm_request=SimpleNamespace(model="gemini-2.0-flash", contents=[]),
            )
            # The model call fails, then the exception escapes the runner:
            # ADK notifies on_model_error, then on_run_error, then re-raises.
            await plugin.on_model_error_callback(
                callback_context=_callback_context(),
                llm_request=SimpleNamespace(model="gemini-2.0-flash", contents=[]),
                error=err,
            )
            await plugin.on_run_error_callback(invocation_context=ic, error=err)

        asyncio.run(_drive())
        traces = _flushed_traces()
        assert len(traces) == 1
        trace = traces[0]
        assert trace.status == Status.ERROR
        assert "quota exceeded" in (trace.error_message or "")
        assert len(trace.llm_calls) == 1
        assert trace.llm_calls[0].status == Status.ERROR
        assert trace.llm_calls[0].finish_reason == FinishReason.ERROR

    def test_run_error_before_any_model_call_sends_error_trace(self):
        from decimalai.adk import DecimalaiPlugin

        plugin = DecimalaiPlugin(agent_name="test-agent")
        agent = _make_agent()
        ic = _invocation_context(agent)

        async def _drive():
            await plugin.before_run_callback(invocation_context=ic)
            await plugin.on_run_error_callback(
                invocation_context=ic, error=RuntimeError("setup hook failed")
            )

        asyncio.run(_drive())
        traces = _flushed_traces()
        assert len(traces) == 1
        assert traces[0].status == Status.ERROR
        assert traces[0].error_message == "setup hook failed"

    def test_run_error_for_unknown_invocation_is_a_noop(self):
        from decimalai.adk import DecimalaiPlugin

        plugin = DecimalaiPlugin(agent_name="test-agent")
        ic = _invocation_context(_make_agent(), inv_id="never-started")

        asyncio.run(
            plugin.on_run_error_callback(invocation_context=ic, error=RuntimeError("boom"))
        )
        assert _flushed_traces() == []
