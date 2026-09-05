"""The conformance README's framework counts are generated from the code, not typed.

Both sentences went stale the day ADK and pydantic-ai were scaffolded: the README said
"SUPPORTED_FRAMEWORKS today is langchain and openai-agents" and "two frameworks today"
while `decimalai init` wrote files for four. A doc that miscounts the thing it is
explaining teaches the wrong shape of the matrix to the next person who reads it,
which is how an exemption ledger drifts.

Asserted against `scaffold.py` rather than against a number, so the next framework
fails this test instead of quietly outdating the prose.
"""
from __future__ import annotations

import re
from pathlib import Path

from decimalai.cli.scaffold import SUPPORTED_FRAMEWORKS

README = Path(__file__).resolve().parent / "conformance" / "README.md"

_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
          6: "six", 7: "seven", 8: "eight", 9: "nine"}

#: Drivers the matrix runs. The N/A count is this minus the scaffolded ones.
_DRIVER_COUNT = 9


def test_the_readme_names_every_scaffolded_framework():
    body = README.read_text(encoding="utf-8")
    sentence = re.search(r"`SUPPORTED_FRAMEWORKS` today is ([^.]+)\.", body)
    assert sentence, "the README no longer states which frameworks are scaffolded"
    named = sentence.group(1)
    for fw in SUPPORTED_FRAMEWORKS:
        assert fw in named, (
            f"{fw!r} is scaffolded but the README's list does not name it: {named!r}"
        )


def test_the_readme_counts_the_journey_cells_correctly():
    body = README.read_text(encoding="utf-8")
    want = _WORDS[len(SUPPORTED_FRAMEWORKS)]
    assert f"{want} frameworks today" in body, (
        f"`decimalai init` writes a file for {len(SUPPORTED_FRAMEWORKS)} frameworks, so "
        f"the README should say '{want} frameworks today'"
    )
    na = _WORDS[_DRIVER_COUNT - len(SUPPORTED_FRAMEWORKS)]
    assert f"The other {na} are a" in body, (
        f"{_DRIVER_COUNT} drivers minus {len(SUPPORTED_FRAMEWORKS)} scaffolded leaves "
        f"{na} declared N/A"
    )
