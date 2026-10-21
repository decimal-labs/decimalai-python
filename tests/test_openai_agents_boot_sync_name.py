"""The OpenAI adapter's boot sync names the agent its traces are filed under.

Review S2 (2026-10-22): the boot sync ran before the processor was registered and named the
Agent object's own `name=` ("Assistant"), while every trace went under its DecimalAI name, for
example `init(agent_name="support")`. The sync then made a phantom "Assistant" agent that got the
skills, and the agent the traces named got none.
"""
from __future__ import annotations

import pytest

agents = pytest.importorskip("agents")

import decimalai._config as cfg  # noqa: E402
import decimalai.openai_agents as oa  # noqa: E402
import decimalai.skill_router as sr_mod  # noqa: E402
from decimalai._config import DecimalConfig  # noqa: E402
from tests.test_openai_agents_phase1 import sdk  # noqa: E402,F401 — the fixture

SKILL = {"name": "refund-steps", "description": "Refund steps", "body_markdown": "# Refunds\n\nCheck the window first."}


@pytest.fixture
def synced(sdk, monkeypatch, tmp_path):  # noqa: F811 — the imported fixture
    # The boot sync records this checkout's install identity (.decimal/install.json): a project
    # of the test's own, never this repository.
    monkeypatch.chdir(tmp_path)
    calls = []

    class _Router:
        def __init__(self, **kwargs):
            pass

        def sync_skills(self, items, **kwargs):
            calls.append(kwargs)
            return {"synced": len(items), "created": len(items)}

    monkeypatch.setattr(sr_mod, "SkillRouter", _Router)
    monkeypatch.setattr(cfg._sender, "submit", lambda fn, *a, **k: fn(*a, **k))   # inline
    return calls


def _named(monkeypatch, name):
    monkeypatch.setattr(cfg, "_config", DecimalConfig(
        api_key="dai_sk_test", base_url="http://127.0.0.1:9", enabled=True, agent_name=name,
    ))


def test_an_agent_object_syncs_under_the_name_its_traces_carry(synced, monkeypatch):
    _named(monkeypatch, "support")
    oa.instrument(agent=agents.Agent(name="Assistant", instructions="Help."), skills=[SKILL], disk_sync=True)
    assert [c["agent_name"] for c in synced] == ["support"]
    assert oa._tracing_processor.decimal_name_for("Assistant") == "support"   # where its traces go
    assert synced[0]["add_to_agent"] == "unless_removed"


def test_an_agent_given_its_own_decimalai_name_syncs_under_it(synced, monkeypatch):
    _named(monkeypatch, "support")
    oa.instrument(agent=agents.Agent(name="Assistant", instructions="Help."), agent_name="billing",
                  skills=[SKILL], disk_sync=True)
    assert [c["agent_name"] for c in synced] == ["billing"]


def test_with_no_other_name_the_agents_own_name_is_used(synced, monkeypatch):
    _named(monkeypatch, None)
    oa.instrument(agent=agents.Agent(name="assistant", instructions="Help."), skills=[SKILL], disk_sync=True)
    assert [c["agent_name"] for c in synced] == ["assistant"]
