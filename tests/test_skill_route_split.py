"""The prefix/tail split — the SDK half of the wire contract.

Two attempts at this change were reverted. Both failed on things this file
exists to catch, so every test here names the invariant it guards from
`platform/docs/contracts/skill_route_split.md`.

The canonical response lives at `tests/fixtures/skill_route_split.json` and is
byte-identical to the platform's copy. Asserting against that file rather than
against a hand-written dict is the point: attempt two shipped an SDK expecting
`{"skills": [...]}` against a server sending a pre-rendered string, and each
side's tests were green against its own idea of the shape.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from decimalai.skill_router import (
    SkillRouter,
    consume_last_offered_names,
    consume_last_delivered_names,
)

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "skill_route_split.json").read_text()
)


def _router(**kw):
    kw.setdefault("api_key", "test")
    kw.setdefault("strategy", "auto")
    return SkillRouter(**kw)


def _route(result):
    """Patch smart_route to return `result`, leaving everything else real."""
    return patch.object(SkillRouter, "smart_route", return_value=result)


class TestWireShape:
    """§1 — both new fields are PRE-RENDERED STRINGS."""

    def test_fixture_fields_are_strings(self):
        for key in ("stable_menu", "routing_hint", "menu_instruction"):
            assert isinstance(FIXTURE[key], str), key

    def test_stable_menu_skills_agrees_with_the_rendered_menu(self):
        """The list and the string are twins. They drifted once already, in the
        same session that wrote this contract: the server grew the field and
        the pinned fixture did not, so the SDK read a key that was not there."""
        import re
        rows = re.findall(r"^\| ([^|]+?) \|", FIXTURE["stable_menu"], re.M)
        rendered = [r for r in rows if r != "Skill" and not set(r) <= set("- ")]
        assert rendered == FIXTURE["stable_menu_skills"]

    def test_prefix_and_tail_come_back_verbatim(self):
        r = _router()
        with _route(FIXTURE):
            prefix, tail, _ = r.build_prompt_parts("refund my laptop")
        # The tail is the server's sentence, not one the SDK re-derived.
        assert tail == FIXTURE["routing_hint"]
        # The prefix is the menu followed by the activation protocol.
        assert prefix.startswith(FIXTURE["stable_menu"].rstrip("\n"))
        assert prefix.endswith(FIXTURE["menu_instruction"])

    def test_the_sdk_never_rewrites_the_hint(self):
        """The server owns the wording. Attempt two shipped two copies of this
        sentence that already disagreed."""
        r = _router()
        odd = dict(FIXTURE, routing_hint="Try these first: alpha.")
        with _route(odd):
            _, tail, _ = r.build_prompt_parts("q")
        assert tail == "Try these first: alpha."

    @pytest.mark.parametrize("bad", [
        {"skills": [{"name": "a"}]},          # attempt two's expected shape
        [],
        None,
        123,
    ])
    def test_a_non_string_menu_degrades_to_the_fragment(self, bad):
        """THE attempt-two failure: a shape the SDK does not recognise must
        fall back to `prompt_fragment`, which is unchanged and still correct —
        never normalise to empty and inject nothing."""
        r = _router()
        with _route(dict(FIXTURE, stable_menu=bad)):
            prefix, tail, _ = r.build_prompt_parts("q")
        assert prefix == FIXTURE["prompt_fragment"]
        assert tail == ""

    def test_an_old_server_without_the_fields_still_works(self):
        r = _router()
        legacy = {k: v for k, v in FIXTURE.items()
                  if k not in ("stable_menu", "routing_hint", "menu_instruction")}
        with _route(legacy):
            prefix, tail, _ = r.build_prompt_parts("q")
        assert prefix == FIXTURE["prompt_fragment"]
        assert tail == ""


class TestInvariant1FragmentUnchanged:
    """§3.1 — `prompt_fragment` bytes unchanged for existing callers."""

    def test_build_prompt_fragment_is_untouched_by_the_split(self):
        r = _router()
        with _route(FIXTURE):
            fragment, _ = r.build_prompt_fragment("refund my laptop")
        assert fragment == FIXTURE["prompt_fragment"]

    def test_the_two_methods_do_not_share_a_cache_slot(self):
        """They report different offered sets, so a shared slot would let one
        silently overwrite the other's rails."""
        r = _router()
        with _route(FIXTURE):
            fragment, _ = r.build_prompt_fragment("same query")
            prefix, _, _ = r.build_prompt_parts("same query")
        assert fragment == FIXTURE["prompt_fragment"]
        assert prefix != fragment


class TestInvariant2StablePrefix:
    """§3.2 — byte-identical across turns for a given (eligible set, hinted set)."""

    def test_prefix_is_identical_across_turns(self):
        r = _router()
        seen = set()
        for turn, q in enumerate(["refund a laptop", "book a flight", "refund again"]):
            hint = f"Most relevant for this request: skill-{turn}."
            with _route(dict(FIXTURE, routing_hint=hint)):
                prefix, tail, _ = r.build_prompt_parts(q, bypass_cache=True)
            seen.add(prefix)
            assert tail == hint          # the tail is what moves...
        assert len(seen) == 1            # ...and the prefix is what does not

    def test_a_cache_hit_re_emits_the_same_prefix(self):
        """Within one turn a multi-call agent loop hits the cache. If the hit
        lost the parts, the injected text would change mid-turn — the one thing
        a byte-stable prefix must never do."""
        r = _router()
        with _route(FIXTURE):
            first, first_tail, _ = r.build_prompt_parts("q")
            second, second_tail, _ = r.build_prompt_parts("q")
        assert first == second
        assert first_tail == second_tail


class TestInvariant3HintNamesAreVisible:
    """§3.3 — every name in the hint appears in the menu."""

    def test_every_hinted_name_is_in_the_stable_menu(self):
        hint = FIXTURE["routing_hint"]
        names = hint.split(":", 1)[1].strip().rstrip(".").split(", ")
        assert names, "fixture should name at least one skill"
        for n in names:
            assert n in FIXTURE["stable_menu"], n

    def test_a_hint_is_useless_without_the_menu_beside_it(self):
        """Both halves reach the model, or the hint points at nothing."""
        r = _router()
        with _route(FIXTURE):
            prefix, tail, _ = r.build_prompt_parts("q")
        injected = f"{prefix}\n{tail}"
        names = tail.split(":", 1)[1].strip().rstrip(".").split(", ")
        for n in names:
            assert n in prefix, f"{n} hinted but absent from the prefix"
            assert injected.count(n) >= 2


class TestInvariant5BodiesRideThePrefix:
    """§3.5 — a body in the tail is dropped on adapters that have no tail."""

    def test_bodies_land_in_the_prefix_never_the_tail(self):
        r = _router(inject_body=True)
        with _route(FIXTURE), \
             patch.object(SkillRouter, "get_skill_body", return_value="BODY_SENTINEL"):
            prefix, tail, _ = r.build_prompt_parts("refund my laptop")
        assert "BODY_SENTINEL" in prefix
        assert "BODY_SENTINEL" not in tail

    def test_bodies_are_chosen_by_relevance_not_menu_order(self):
        """The split widens `offered` to the whole menu. If the body loop read
        that widened list it would fetch bodies in MENU order — arbitrary."""
        r = _router(inject_body=True)
        asked = []

        def _body(self, name, **kw):  # patched onto the class → takes self
            asked.append(name)
            return f"body of {name}"

        with _route(FIXTURE), patch.object(SkillRouter, "get_skill_body", _body):
            r.build_prompt_parts("refund my laptop")

        routed = [s["name"] for s in FIXTURE["skills"]]
        assert asked, "no body was fetched"
        assert asked == routed[:len(asked)], (asked, routed)


class TestInvariant6DeliveredSubsetOfferedSubsetText:
    """§3.6 — delivered ⊆ offered ⊆ the injected text."""

    def test_the_chain_holds(self):
        r = _router(inject_body=True)
        with _route(FIXTURE), \
             patch.object(SkillRouter, "get_skill_body", return_value="BODY"):
            prefix, tail, _ = r.build_prompt_parts("refund my laptop")
        offered = set(consume_last_offered_names())
        delivered = set(consume_last_delivered_names())
        injected = f"{prefix}\n{tail}"

        assert delivered, "nothing delivered — test proves nothing"
        assert delivered <= offered, delivered - offered
        for name in offered:
            assert name in injected, f"{name} reported offered but absent from the prompt"

    def test_a_body_delivered_from_outside_the_menu_is_still_reported_offered(self):
        """The edge the menu cap creates: bodies come from the router's ranked
        pick, the menu is capped, and only HINTED names are unioned back. A
        body-delivered skill below the cap would otherwise be `delivered` but
        not `offered` — while its `## Skill:` header sits in the prefix, so the
        model can plainly see it."""
        r = _router(inject_body=True)
        narrow = dict(
            FIXTURE,
            stable_menu="## Available Skills\n\n| Skill | Description |\n"
                        "|-------|-------------|\n| aaa-flight-search | x. |\n",
            stable_menu_skills=["aaa-flight-search"],
        )
        with _route(narrow), \
             patch.object(SkillRouter, "get_skill_body", return_value="BODY"):
            prefix, tail, _ = r.build_prompt_parts("refund my laptop")
        offered = set(consume_last_offered_names())
        delivered = set(consume_last_delivered_names())

        assert delivered, "no body delivered — the test proves nothing"
        assert delivered <= offered, delivered - offered
        for n in offered:
            assert n in f"{prefix}\n{tail}", n

    def test_offered_is_the_whole_menu_on_the_split_path(self):
        """Contract §2: the SDK's field answers 'what could the model see?'"""
        r = _router()
        with _route(FIXTURE):
            r.build_prompt_parts("refund my laptop")
        offered = consume_last_offered_names()
        assert sorted(offered) == sorted(FIXTURE["stable_menu_skills"])
        # strictly wider than the router's own pick
        assert len(offered) > len(FIXTURE["skills"])

    def test_offered_stays_the_routers_pick_on_the_fragment_path(self):
        """The unsplit path is unchanged — `acceptance_rate` divides by the
        router's list, and widening it there would move a chart that measures
        something else."""
        r = _router()
        with _route(FIXTURE):
            r.build_prompt_fragment("refund my laptop")
        offered = consume_last_offered_names()
        assert sorted(offered) == sorted(s["name"] for s in FIXTURE["skills"])
