"""The split as the LangChain adapter places it.

The reason this file exists separately from `test_skill_route_split.py`: the
wire contract can be perfect and the PLACEMENT still break every Anthropic
caller. `langchain_anthropic` raises

    ValueError: Received multiple non-consecutive system messages

for a system message that appears after any human or AI turn — so putting the
tail "down beside the query", which is the obvious reading of the word tail,
crashes ChatAnthropic on single-turn and multi-turn alike. Verified against
langchain_anthropic 1.6.1.

Both parts therefore go in as system messages CONSECUTIVE with the caller's
own. The stable half still leads, which is the whole cache property; the
varying half sits behind it rather than beside the query.
"""
from __future__ import annotations

from typing import Any, List, Optional

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402

PREFIX_MARKER = "== STABLE MENU =="
TAIL_MARKER = "Most relevant for this request:"
CALLER = "You are Ada, a careful assistant."


class _SplitRouter:
    """Stand-in whose PREFIX is constant and whose TAIL varies per query —
    the shape the real server produces, and the shape these tests are about."""

    def __init__(self) -> None:
        self.queries: List[Optional[str]] = []

    def build_prompt_parts(self, query=None, **kwargs):
        self.queries.append(query)
        prefix = f"{PREFIX_MARKER}\n| alpha | ... |\n| beta | ... |"
        tail = f"{TAIL_MARKER} skill-{abs(hash(query)) % 97}."
        return prefix, tail, "rt_" + "a" * 24

    def consume_routing_id(self):
        return None

    def consume_offered_names(self):
        return []

    def consume_delivered_names(self):
        return []

    def consume_loaded_names(self, **kwargs):
        return []


class _LegacyRouter(_SplitRouter):
    """A router object that predates the split: only the old primitive."""

    build_prompt_parts = None  # not callable → the adapter must degrade

    def build_prompt_fragment(self, query=None, **kwargs):
        self.queries.append(query)
        return f"{PREFIX_MARKER}\nlegacy fragment", "rt_" + "b" * 24


@pytest.fixture
def adapter(monkeypatch):
    import decimalai.skill_router as sr
    import decimalai.langchain as lc_mod

    monkeypatch.setattr(sr, "consume_last_offered_names", lambda *a, **k: [])
    monkeypatch.setattr(sr, "consume_last_delivered_names", lambda *a, **k: [])

    def _use(router):
        monkeypatch.setattr(lc_mod, "_skill_router_singleton", router)
        return lc_mod

    return _use


def _roles(msgs: List[Any]) -> List[str]:
    return [type(m).__name__ for m in msgs]


def _text(m: Any) -> str:
    c = getattr(m, "content", m)
    return c if isinstance(c, str) else str(c)


class TestPlacement:
    def test_prefix_and_tail_are_both_injected_in_that_order(self, adapter):
        lc = adapter(_SplitRouter())
        out = lc._inject_skills_into_input(
            [SystemMessage(content=CALLER), HumanMessage(content="refund my laptop")]
        )
        blob = [_text(m) for m in out]
        pre = next(i for i, t in enumerate(blob) if PREFIX_MARKER in t)
        tail = next(i for i, t in enumerate(blob) if TAIL_MARKER in t)
        assert pre < tail, "the varying half must sit BEHIND the stable half"

    def test_the_callers_own_prompt_still_leads(self, adapter):
        lc = adapter(_SplitRouter())
        out = lc._inject_skills_into_input(
            [SystemMessage(content=CALLER), HumanMessage(content="q")]
        )
        assert CALLER in _text(out[0])

    def test_no_system_message_ever_follows_a_non_system_message(self, adapter):
        """The ChatAnthropic crash guard, stated structurally so it runs
        everywhere — not only where langchain_anthropic is installed."""
        lc = adapter(_SplitRouter())
        cases = [
            [SystemMessage(content=CALLER), HumanMessage(content="q")],
            [HumanMessage(content="q")],
            [SystemMessage(content=CALLER), HumanMessage(content="hi"),
             AIMessage(content="hello"), HumanMessage(content="and now?")],
        ]
        for msgs in cases:
            out = lc._inject_skills_into_input(msgs)
            names = _roles(out)
            seen_non_system = False
            for n in names:
                if n == "SystemMessage":
                    assert not seen_non_system, (
                        "a system message lands after a human/AI turn — "
                        f"langchain_anthropic raises on this: {names}"
                    )
                else:
                    seen_non_system = True

    def test_multi_turn_keeps_the_prefix_byte_identical(self, adapter):
        lc = adapter(_SplitRouter())
        prefixes, tails = set(), set()
        for q in ["refund a laptop", "book a flight", "cancel an order"]:
            out = lc._inject_skills_into_input(
                [SystemMessage(content=CALLER), HumanMessage(content=q)]
            )
            for t in (_text(m) for m in out):
                if PREFIX_MARKER in t:
                    prefixes.add(t)
                if TAIL_MARKER in t:
                    tails.add(t)
        assert len(prefixes) == 1, "the cacheable half moved between queries"
        assert len(tails) > 1, "the query-dependent half did not move — no routing"


class TestDegradation:
    def test_a_router_without_the_split_still_delivers_skills(self, adapter):
        """The failure this guards is silent: the adapter used to swallow the
        AttributeError and return the input UNCHANGED — perfect tracing, zero
        skills. 21 tests degraded exactly that way before the fallback existed."""
        lc = adapter(_LegacyRouter())
        out = lc._inject_skills_into_input(
            [SystemMessage(content=CALLER), HumanMessage(content="q")]
        )
        assert any(PREFIX_MARKER in _text(m) for m in out), (
            "an older router object delivered NOTHING instead of degrading"
        )

    def test_an_empty_tail_injects_nothing_extra(self, adapter):
        class _NoHint(_SplitRouter):
            def build_prompt_parts(self, query=None, **kwargs):
                return f"{PREFIX_MARKER}\nmenu", "", "rt_" + "c" * 24

        lc = adapter(_NoHint())
        out = lc._inject_skills_into_input(
            [SystemMessage(content=CALLER), HumanMessage(content="q")]
        )
        assert sum(1 for m in out if type(m).__name__ == "SystemMessage") == 2
        assert not any(TAIL_MARKER in _text(m) for m in out)


class TestAgainstRealChatAnthropic:
    """The assertion above is structural; this one is the real thing."""

    def test_the_payload_builds(self, adapter):
        pytest.importorskip("langchain_anthropic")
        from langchain_anthropic import ChatAnthropic

        lc = adapter(_SplitRouter())
        model = ChatAnthropic(model="claude-sonnet-4-5", api_key="sk-ant-stub")
        for msgs in (
            [SystemMessage(content=CALLER), HumanMessage(content="refund?")],
            [SystemMessage(content=CALLER), HumanMessage(content="hi"),
             AIMessage(content="hello"), HumanMessage(content="refund?")],
        ):
            out = lc._inject_skills_into_input(msgs)
            payload = model._get_request_payload(out)
            blocks = [b["text"] for b in (payload.get("system") or [])]
            assert any(PREFIX_MARKER in b for b in blocks)
            assert any(TAIL_MARKER in b for b in blocks)
            # stable content strictly before varying content
            assert (next(i for i, b in enumerate(blocks) if PREFIX_MARKER in b)
                    < next(i for i, b in enumerate(blocks) if TAIL_MARKER in b))


class TestInjectedMessagesAreNotMistakenForTheCallersPrompt:
    """The manifest's auto-detection keeps the first system message it sees and
    warns when a later one differs — sound for the caller's own prompts, wrong
    for ours. The split injects two system messages BY DESIGN, so every call
    tripped the check and advised a fix (`install(prompts=...)`) that cannot
    work, because the message that "changed" is one the SDK added.

    Measured before the mark: a bare invoke went 0 -> 1 warnings and a 3-turn
    conversation 3 -> 6.
    """

    def _capture(self, adapter, msgs):
        import uuid
        import decimalai.langchain as lc

        lc_mod = adapter(_SplitRouter())
        h = lc_mod.CallbackHandler()
        rid = uuid.uuid4()
        h.on_chat_model_start(
            {"name": "x"}, [lc_mod._inject_skills_into_input(msgs)], run_id=rid
        )
        return (h._state_for(rid, None).seen_prompts or {}).get("system", "")

    def test_the_callers_own_prompt_is_still_captured(self, adapter):
        got = self._capture(
            adapter, [SystemMessage(content=CALLER), HumanMessage(content="q")]
        )
        assert got.strip() == CALLER
        assert PREFIX_MARKER not in got

    def test_with_no_caller_prompt_nothing_is_captured(self, adapter):
        """Not the skill menu. Before the mark, an agent whose code passes no
        system message had its manifest record the routed menu AS its system
        prompt — a prompt the user never wrote."""
        got = self._capture(adapter, [HumanMessage(content="q")])
        assert got == "", got

    def test_no_drift_warning_is_emitted(self, adapter, caplog):
        import logging
        import uuid
        import decimalai.langchain as lc_mod

        lc = adapter(_SplitRouter())
        convo = [SystemMessage(content=CALLER)]
        with caplog.at_level(logging.WARNING, logger="decimalai.langchain"):
            h = lc.CallbackHandler()
            rid = uuid.uuid4()
            for i in range(3):
                convo = convo + [HumanMessage(content=f"q{i}")]
                h.on_chat_model_start(
                    {"name": "x"},
                    [lc._inject_skills_into_input(list(convo))],
                    run_id=rid,
                )
                convo = convo + [AIMessage(content=f"a{i}")]
        drift = [r for r in caplog.records if "changed within trace" in r.getMessage()]
        assert not drift, [r.getMessage() for r in drift]

    def test_the_mark_never_reaches_the_model(self, adapter):
        """It rides in additional_kwargs, which providers ignore — the content
        the model sees must be exactly the prefix and the tail."""
        lc = adapter(_SplitRouter())
        out = lc._inject_skills_into_input(
            [SystemMessage(content=CALLER), HumanMessage(content="q")]
        )
        for m in out:
            assert "decimalai_injected" not in _text(m)
