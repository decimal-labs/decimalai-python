"""``decimalai._claude_skill_render``: Claude Code's rewrite of a skill body, redone.

The expectations are the bundled CLI's own output (claude-agent-sdk 0.2.166, Claude Code
2.1.299), recorded with the local Messages stub into
``tests/fixtures/claude_skill_args_2_1_299.json``; nothing here is what the module was
believed to do. ``tests/native/test_claude_agent_sdk_native.py`` runs the real CLI on every
suite run and fails when a later CLI rewrites differently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from skillevaluation import skillfmt

from decimalai import _claude_skill_render as render

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "claude_skill_args_2_1_299.json")
                     .read_text())
PLACES = {"skill_dir": "<SKILL_DIR>", "plugin_root": "<PLUGIN_ROOT>",
          "project_dirs": ["<PROJECT_DIR>"], "session_id": "<SESSION_ID>"}


def _filled(args: str) -> list:
    """What the CLI fills $0, $1, ... with: the words, a word-starting '!' escaped."""
    return [render._escape_bang(w) for w in render.words(args)]


@pytest.mark.parametrize("case", FIXTURE["words"], ids=lambda c: repr(c["args"])[:40])
def test_args_split_into_the_words_the_cli_fills_in(case):
    assert _filled(case["args"]) == case["words"]


@pytest.mark.parametrize("case", FIXTURE["renders"], ids=lambda c: c["case"])
def test_a_body_renders_to_what_the_cli_delivered(case):
    frontmatter = yaml.safe_load(case["frontmatter"]) if case["frontmatter"] else {}
    got = list(render.renderings("\n" + case["body"], case["args"] or "",
                                 names=render.argument_names(frontmatter), **PLACES))
    if "${CLAUDE_EFFORT}" in case["body"]:
        # The effort level is the CLI's to know: such a body is never matched.
        assert got == []
        return
    want = skillfmt.body_sha256(case["delivered"])
    assert want in {skillfmt.body_sha256(out) for out in got}


def _fill(body: str, args: str, names=()) -> str:
    return render.fill(body, args, render.words(args), names)


class TestFill:
    def test_words_are_numbered_from_zero(self):
        assert _fill("$0 then $1", "alpha beta") == "alpha then beta"

    def test_a_number_past_the_last_word_is_left_as_written(self):
        # Nothing was filled, so the args are appended, as to a body with no placeholder.
        assert _fill("Late fee: $12.", "a b c") == "Late fee: $12.\n\nARGUMENTS: a b c"

    def test_a_price_is_filled_when_there_are_enough_words(self):
        words = " ".join(f"w{i}" for i in range(12)) + " a b"
        assert _fill("Late fee: $12.", words) == "Late fee: a."

    def test_a_call_without_args_still_rewrites_the_body(self):
        assert _fill("[$ARGUMENTS] \\$1 $1", "") == "[] $1 $1"

    def test_args_are_appended_only_to_a_body_that_took_none(self):
        assert _fill("No placeholder.", "7 days") == "No placeholder.\n\nARGUMENTS: 7 days"
        assert _fill("Days: $0.", "7 days") == "Days: 7."
        assert _fill("No placeholder.", "") == "No placeholder."

    def test_a_digit_or_letter_after_the_number_is_not_a_placeholder(self):
        assert _fill("$1st $0_x $10 $1", "a b") == "$1st $0_x $10 b"

    def test_an_escaped_placeholder_loses_its_backslash_and_stays(self):
        assert _fill("\\$0 \\$ARGUMENTS \\\\$0", "x") == "$0 $ARGUMENTS \\\\x"

    def test_argument_names_from_the_frontmatter(self):
        names = render.argument_names({"arguments": ["city", "date", "7", " "]})
        assert names == ["city", "date"]
        assert render.argument_names({"arguments": "city date"}) == ["city", "date"]
        assert _fill("$city on $date, $cityx", "Paris", names) == "Paris on , $cityx"

    def test_a_value_cannot_start_a_shell_command(self):
        assert _fill("$0 / $ARGUMENTS", "!x `!y`") == "\\!x / \\!x ` \\!y`"
        assert render.appended("!urgent now") == "ARGUMENTS: \\!urgent now"

    def test_a_dollar_inside_a_value_is_never_read_as_a_placeholder(self):
        assert _fill("$0 and $ARGUMENTS", "$1") == "$1 and $1"


class TestRender:
    def test_the_variables_after_the_arguments(self):
        out = render.render("${CLAUDE_SKILL_DIR}/x $0 ${CLAUDE_SESSION_ID}", "a", ["a"],
                            skill_dir="/s", session_id="sid")
        assert out == "/s/x a sid"

    def test_a_variable_with_no_known_value_is_no_rendering(self):
        assert render.render("${CLAUDE_PLUGIN_ROOT}", "", []) is None
        assert render.render("${CLAUDE_PLUGIN_DATA}", "", []) is None

    def test_inline_shell_is_no_rendering(self):
        assert render.render("Today: !`date`", "", []) is None
        assert render.render("```!\nls\n```", "", []) is None


class TestWords:
    def test_quotes_group_and_an_unclosed_quote_runs_to_the_end(self):
        assert render.words('say "hello world" ok') == ["say", "hello world", "ok"]
        assert render.words("the customer's order is late") == [
            "the", "customer's order is late"]

    def test_the_first_command_only(self):
        assert render.words("a b; c d") == ["a", "b"]
        assert render.words("order #12 is late") == ["order"]

    def test_a_parser_with_no_command_falls_back_to_whitespace(self):
        assert render.words("#12 is late") == ["#12", "is", "late"]
        assert render.words("for x") == ["for", "x"]

    def test_both_splits_are_candidates(self):
        assert render.candidate_words("a $5 b") == [["a", "b"], ["a", "$5", "b"]]
        assert render.candidate_words("a b") == [["a", "b"]]
