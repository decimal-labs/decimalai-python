"""An agent's always-include skills (D-C): every ``agent_config()`` run gets their full text.

The platform lists them on the menu and route answers (``always_include``: name and version),
from the agent's own skill rows. These hold the shared core (``decimalai._agent_binding``) to:

- the text goes after the skill list, as part of DecimalAI's skill section, so a conversation
  keeps it with the list (the Anthropic module's system blocks are rebuilt from that section);
- it is fetched once and kept until the list or a version changes, not per run;
- the run's record says it was delivered, with its version, and ``load_skill`` answers for it
  that it is already loaded rather than sending it again;
- a skill this run's own text already pasted (``load_skill`` off) is not pasted twice;
- the kill switch stops it, and an older platform that lists none changes nothing.

The LangChain binding's tagged message is held in ``test_langchain_agent_config.py``.
"""

from __future__ import annotations

import logging

import pytest

import decimalai._agent_binding as binding
from decimalai.skill_router import SkillRouter
from tests.test_agent_binding import BODIES, Binding, ChatBinding, FakePlatform, prompt


@pytest.fixture
def platform(monkeypatch):
    """As ``test_agent_binding.py``'s: the real router against the fake platform."""
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    monkeypatch.setattr(cfg, "_config", DecimalConfig(
        api_key="dai_sk_test", base_url="http://always-include.test", enabled=True))
    monkeypatch.setattr(binding, "_in_background", lambda fn, *args: fn(*args))
    fake = FakePlatform()
    router = SkillRouter(api_key="dai_sk_test", base_url="http://always-include.test", inject_body=False)
    monkeypatch.setattr(router, "_request", fake.request)
    fake.router = router
    return fake


def section(name: str) -> str:
    return f"## Skill: {name}\n\n{BODIES[name]}"


def test_every_run_gets_the_text_after_the_list_and_the_record_says_so(platform):
    platform.always = {"order-support": 1}
    support = Binding(prompt(), platform.router)
    text = support.run("My keyboard arrived cracked.")

    head, _, tail = text.partition("## Available Skills")
    assert head.startswith("You help customers with orders.")
    assert tail.endswith(section("order-support")), "after the list and its usage line"
    record = support.records[0]
    assert record["delivered"] == ["order-support"]
    assert [v["name"] for v in record["delivered_versions"]] == ["order-support"]
    assert record["delivered_versions"][0]["hash"]


def test_the_text_is_fetched_once_until_the_list_or_a_version_changes(platform):
    platform.always = {"order-support": 1}
    support = Binding(prompt(), platform.router)
    for message in ("cracked", "where is it", "thanks"):
        assert support.run(message).endswith(section("order-support"))
    assert len(platform.body_calls("order-support")) == 1

    # Distinct messages: the router keeps an answer for 30 s per message.
    platform.always = {"order-support": 2}           # a new version
    support._route_run(None, [], "a new version")    # the next answer lists it
    support.run("again")
    assert len(platform.body_calls("order-support")) == 2

    platform.always = {}                             # taken off the list
    support._route_run(None, [], "taken off")
    assert section("order-support") not in support.run("once more")


def test_a_load_of_it_says_it_is_already_there(platform):
    platform.always = {"order-support": 1}
    support = Binding(prompt(), platform.router)
    support.run("Where is my order?")
    assert "already loaded" in support._load_sync("order-support")
    assert len(platform.body_calls("order-support")) == 1, "not fetched again for the load"


def test_a_skill_the_run_already_pasted_is_not_pasted_twice(platform, monkeypatch):
    import decimalai._config as cfg

    monkeypatch.setattr(cfg._config, "load_skill_tool", False)   # skill text pasted per message
    platform.always = {"damaged-item": 1}
    support = Binding(prompt(), platform.router)
    text = support.run("My keyboard arrived cracked.")
    assert text.count("## Skill: damaged-item") == 1
    assert support.records[0]["delivered"].count("damaged-item") == 1


def test_a_conversation_keeps_the_text_with_its_list(platform):
    platform.always = {"order-support": 1}
    support = ChatBinding(prompt(), platform.router)
    first = support.chat_run("a", "My keyboard arrived cracked.")
    record = first["conversation"]
    assert record.menu.endswith(section("order-support")) and record.text.endswith(record.menu)
    assert record.delivered == ["order-support"]

    platform.always = {"damaged-item": 1}            # the agent's list changes mid-chat
    support._route_run(None, [], "anything")
    later = support.chat_run("a", "Where is my order?")
    assert later["text"] == first["text"], "the system text a conversation keeps does not change"
    assert "order-support" in later["loaded"]


def test_the_kill_switch_stops_it(platform, monkeypatch):
    import decimalai._config as cfg

    platform.always = {"order-support": 1}
    support = Binding(prompt(), platform.router)
    monkeypatch.setattr(cfg, "_disabled_by_init", lambda: True)
    assert support.run("Where is my order?") == "You help customers with orders."
    assert platform.body_calls("order-support") == []


def test_an_older_platform_lists_none_and_nothing_changes(platform):
    support = Binding(prompt(), platform.router)
    text = support.run("My keyboard arrived cracked.")
    assert "## Skill:" not in text
    assert support.records[0]["delivered"] == []
    assert [r for r in platform.requests if r[1].endswith("/body")] == []


def test_a_text_over_the_skill_text_budget_is_left_out_and_said(platform, caplog):
    platform.always = {"order-support": 1}
    platform.router.body_token_budget = 3
    support = Binding(prompt(), platform.router)
    with caplog.at_level(logging.WARNING, logger="decimalai.skill_router"):
        text = support.run("Where is my order?")
    assert "## Skill: order-support" not in text and support.records[0]["delivered"] == []
    assert "always-include skill 'order-support'" in caplog.text and "does not fit" in caplog.text


def test_set_always_include_puts_the_flag_on_the_agents_skill(platform):
    sent = []
    platform.router._request = lambda method, path, json=None, params=None, **kw: (
        sent.append((method, path, json)) or {"status": "ok", "changed": True})
    platform.router.set_always_include("support bot", "reply/format")
    platform.router.set_always_include("support bot", "reply/format", always_include=False)
    assert sent == [
        ("PUT", "/api/v1/agents/support%20bot/skills/reply%2Fformat/always-include", {"always_include": True}),
        ("PUT", "/api/v1/agents/support%20bot/skills/reply%2Fformat/always-include", {"always_include": False}),
    ]
