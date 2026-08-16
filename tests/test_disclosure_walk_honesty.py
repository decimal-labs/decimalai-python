"""The disclosure walk may report a short count — it may never call one exact.

scripts/check_notebook_manifest.py recomputes `registry_disclosure` — the block
of numbers saying how much per-case evidence this registry withholds — and
prints a YAML paste block for a human to copy into manifest.yaml, which the
Tier-0 notebook then prints verbatim to readers.

Two of those numbers, `measured_public_skills` and `graded_cases`, are a census:
the listing serves `total_hint: null`, so the only way to know them is to walk
every page. The walk can stop short three ways — its own page cap, the
registry's `truncated: true` depth limit, or a cursor that stops advancing — and
in every one of them the counts are floors. The paste block used to stamp
`# exact` on them unconditionally, so a truncated run produced a plausible wrong
number that a human would paste into the manifest and the notebook would then
print as fact.

THE TRUNCATED CASE IS THE ONE THAT WAS LYING, so it is the one most heavily
covered here: every stop-short path is asserted to reach the paste block as
"AT LEAST", and the complete path is asserted to still earn "# exact" (a fix
that just deleted the word would pass a one-sided test and lose the census).

Hermetic: `fetch` is replaced, no network, no key.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_notebook_manifest.py"


@pytest.fixture()
def checker():
    """The script as a module.

    Loaded by path rather than imported: `scripts/` is not a package and the
    file is a CLI, not an installed module. Freshly loaded per test so the
    `fetch` monkeypatch below cannot leak between cases.
    """
    spec = importlib.util.spec_from_file_location("check_notebook_manifest", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _listing(total_pages, *, truncate_at=None, frozen_cursor=False, page_size=100, cases=10):
    """A fake registry listing, in the shape `/registry/skills` really answers.

    `truncate_at` makes page N come back with `truncated: true` — and, exactly as
    the real endpoint does, ALSO with `has_next: false` and no cursor, because
    the registry contains the walk by withholding the cursor rather than by
    erroring (platform registry.py: "withholding the cursor reads as 'end of
    results' to every existing consumer"). That is what makes this case
    dangerous: on the wire a truncated page is byte-for-byte shaped like an
    exhausted one apart from the flag.

    `frozen_cursor` reproduces the live 2026-08-12 defect: the cursor
    fingerprint omitted `measured`, so every page decoded as page 1 and
    `has_next` stayed true forever.
    """

    def fake_fetch(url, params=None, **kwargs):
        if "/registry/skills" in url and (params or {}).get("measured") == "only":
            index = 0 if frozen_cursor else int((params or {}).get("cursor") or 0)
            served = index
            items = [
                {
                    "url_slug": f"skill-{served}-{i}",
                    "source_type": "platform",
                    "benchmark_summary": {"total_cases": cases},
                }
                for i in range(page_size)
            ]
            if truncate_at is not None and index + 1 == truncate_at:
                return ({"items": items, "has_next": False, "next_cursor": None,
                         "truncated": True}, 200)
            has_next = frozen_cursor or index + 1 < total_pages
            cursor = "stuck" if frozen_cursor else str(index + 1)
            return ({"items": items, "has_next": has_next,
                     "next_cursor": cursor if has_next else None}, 200)
        # Per-skill benchmark payload for the sampled percentages.
        return ({"latest_run": {"results": [{"case_prompt": "p"} for _ in range(cases)]}}, 200)

    return fake_fetch


def _figures(checker, walk):
    return checker.measure_disclosure(walk, sample_per_source=5, seed=1, pace=0.0, full=False)


def _paste(checker, walk):
    return checker.paste_block({}, _figures(checker, walk), "2026-01-01")


# ── the case that was lying ──────────────────────────────────────────────────

def test_a_registry_truncated_walk_is_never_called_exact(checker):
    """`truncated: true` — the registry's own depth limit — must reach the paste
    block as a floor.

    This is the case with no other tell: `has_next` is false and there is no
    cursor, so a walk that does not read the flag terminates believing it saw
    the whole registry, and every downstream number inherits that belief.
    """
    checker.fetch = _listing(500, truncate_at=3)
    walk = checker.walk_measured(pace=0.0)

    assert walk.complete is False
    assert "truncated" in walk.reason
    assert len(walk.rows) == 300  # 3 pages of 100 out of 50,000 real rows

    block = _paste(checker, walk)
    assert "# exact" not in block, (
        "the paste block called a registry-truncated count exact:\n" + block
    )
    assert "measured_public_skills: 300   # AT LEAST this many — NOT exact" in block
    assert "DO NOT PASTE THE TWO COUNTS" in block
    assert walk.reason in block, "the paste block must say WHY the walk stopped"


def test_a_truncated_walk_marks_the_drift_lines_as_floors(checker):
    """A short count is below the manifest by an unknown amount, so the drift
    percentage next to it is not a measurement. It is still gated — a floor
    already past tolerance is a real finding — but it is labelled."""
    checker.fetch = _listing(500, truncate_at=3)
    walk = checker.walk_measured(pace=0.0)

    declared = {
        "measured_public_skills": 1400,
        "graded_cases": 14000,
        "cases_withheld_pct": 0.0,
        "platform_cases_withheld_pct": 0.0,
        "github_import_cases_withheld_pct": 0.0,
    }
    problems, _notes = checker.compare_disclosure(
        declared, _figures(checker, walk),
        {"counts_drift_pct": 15.0, "withheld_drift_pts": 3.0},
    )
    counts = [p for p in problems if "measured_public_skills" in p or "graded_cases" in p]
    assert counts, "a 300-vs-1400 count gap must still be reported"
    for line in counts:
        assert "FLOOR" in line, f"an unfinished walk's count read as a measurement: {line}"
        assert "live ≥" in line, line


# ── the other two ways the walk stops short ──────────────────────────────────

def test_the_page_cap_is_reported_rather_than_stamped_exact(checker):
    """The original defect: 60 pages x 100 = 6,000, silently, with `# exact`."""
    checker.fetch = _listing(500)
    walk = checker.walk_measured(pace=0.0, max_pages=3)

    assert walk.complete is False
    assert len(walk.rows) == 300
    block = _paste(checker, walk)
    assert "# exact" not in block
    assert "AT LEAST" in block


def test_a_frozen_cursor_stops_the_walk_without_inflating_the_floor(checker):
    """The live 2026-08-12 registry bug served page 1 forever with has_next true.

    Two things have to hold. It must TERMINATE without leaning on the page cap —
    otherwise raising the cap turns a one-minute wrong answer into an hours-long
    one. And the floor it reports must be true: re-counting the same 100 skills
    on every lap would push `measured_public_skills` ABOVE the real total, which
    is "at least N" lying in the other direction.
    """
    checker.fetch = _listing(500, frozen_cursor=True)
    walk = checker.walk_measured(pace=0.0)

    assert walk.complete is False
    assert "looping" in walk.reason
    assert len(walk.rows) == 100, (
        "the walk counted the same skills twice — its floor is now above the truth"
    )
    assert "AT LEAST" in _paste(checker, walk)


# ── and the census that still has to be a census ─────────────────────────────

def test_a_finished_walk_still_earns_the_word_exact(checker):
    """The counts ARE a census when the walk reaches the end, and the manifest
    declares them as one. Deleting the word rather than conditioning it would
    pass every test above and quietly demote a real number."""
    checker.fetch = _listing(4)
    walk = checker.walk_measured(pace=0.0)

    assert walk.complete is True
    assert walk.reason is None
    assert len(walk.rows) == 400

    block = _paste(checker, walk)
    assert "measured_public_skills: 400   # exact" in block
    assert "graded_cases: 4000   # exact" in block
    assert "AT LEAST" not in block
    assert "DO NOT PASTE" not in block


def test_the_page_cap_clears_the_registry_it_has_to_walk(checker):
    """The cap has to be a runaway backstop, not a ceiling the corpus can reach.

    The measured subset is ~1,400 skills today and the whole registry is ~57k;
    at the endpoint's maximum `limit=100` the cap must clear the latter, so that
    every skill in the registry becoming measured still leaves the walk able to
    finish. It also has to stay finite — the loop is bounded by it.
    """
    assert checker.MEASURED_WALK_MAX_PAGES * 100 >= 57_000
    assert checker.MEASURED_WALK_MAX_PAGES < 100_000  # still a bound, not "forever"


# ── the counts cannot be separated from their completeness ───────────────────

def test_the_counts_cannot_travel_without_their_completeness_bit(checker):
    """The structural half of the fix.

    `complete` used to be a second return value that died at the call site as a
    printed note, while the rows travelled on alone into the figures and the
    paste block. `measure_disclosure` now takes the walk itself, so there is no
    signature that accepts the counts and leaves the bit behind — and a figures
    dict assembled without it raises instead of defaulting to "exact".
    """
    checker.fetch = _listing(4)
    walk = checker.walk_measured(pace=0.0)

    with pytest.raises(AttributeError):
        checker.measure_disclosure(walk.rows, sample_per_source=5, seed=1, pace=0.0, full=False)

    figures = _figures(checker, walk)
    assert figures["counts_complete"] is True
    del figures["counts_complete"]
    with pytest.raises(KeyError):
        checker.paste_block({}, figures, "2026-01-01")
    with pytest.raises(KeyError):
        checker.compare_disclosure({}, figures, {"counts_drift_pct": 15.0,
                                                 "withheld_drift_pts": 3.0})


# ── --disclosure-full removes the SAMPLING error, not the incompleteness ──────
#
# Both sites below were missed the first time this file was written: the counts
# learned to say "AT LEAST", but the RATES kept the word "censused" whatever the
# walk did. Same defect, one field over — `--disclosure-full` only means no
# sampling was involved, which says nothing about whether the listing it read
# was the whole registry.


def _full_mode_live(complete):
    """What `--disclosure-full` produces: real rates, and no sampling margin."""
    return {
        "measured_public_skills": 300,
        "graded_cases": 3000,
        "counts_complete": complete,
        "counts_incomplete_reason": None if complete else "the registry truncated the listing",
        "coverage_pct": 50.0,
        "strata": {
            "platform": {"rate": 0.57, "margin": None},
            "github_import": {"rate": 0.10, "margin": None},
        },
        "overall_rate": 0.42,
        "overall_margin": None,
    }


_DECLARED = {
    "cases_withheld_pct": 42.0,
    "platform_cases_withheld_pct": 57.0,
    "github_import_cases_withheld_pct": 10.0,
    "measured_public_skills": 300,
    "graded_cases": 3000,
}
_TOL = {"withheld_drift_pts": 3.0, "counts_drift_pct": 15.0}


def test_full_mode_over_a_truncated_walk_is_not_a_census(checker):
    block = checker.paste_block(_DECLARED, _full_mode_live(False), "2026-08-16")
    assert "exact (censused)" not in block, block
    assert "INCOMPLETE listing" in block, block

    _, notes = checker.compare_disclosure(_DECLARED, _full_mode_live(False), _TOL)
    assert not any("censused, exact" in n for n in notes), notes


def test_a_finished_full_walk_still_earns_the_word_censused(checker):
    """The other direction: a fix that merely deleted the word would pass above."""
    block = checker.paste_block(_DECLARED, _full_mode_live(True), "2026-08-16")
    assert "exact (censused)" in block, block

    _, notes = checker.compare_disclosure(_DECLARED, _full_mode_live(True), _TOL)
    assert any("censused, exact" in n for n in notes), notes


def test_rows_with_no_identity_cannot_inflate_the_floor(checker, monkeypatch):
    """A row with neither url_slug nor id is indistinguishable from a repeat.

    De-dup used to be skipped entirely for such rows (`if slug:`), so a looping
    registry counted the same skill many times — "at least N" lying in the other
    direction, which is worse than a short count because it reads as reassuring.
    """
    pages = [
        {
            "items": [{"benchmark_summary": {"total_cases": 1}} for _ in range(100)],
            "has_next": True,
            "next_cursor": f"cursor-{i}",
        }
        for i in range(3)
    ]
    pages.append({"items": [], "has_next": False, "next_cursor": None})
    it = iter(pages)
    monkeypatch.setattr(checker, "fetch", lambda *a, **k: (next(it), None))

    walk = checker.walk_measured(pace=0.0)
    assert walk.complete is False, "an un-dedupable walk reported itself a census"
    assert "neither url_slug nor id" in (walk.reason or ""), walk.reason


def test_id_is_accepted_as_the_dedup_key_when_url_slug_is_absent(checker, monkeypatch):
    """Falling back to `id` keeps a legitimate payload shape de-duplicating."""
    page = {
        "items": [{"id": f"skill_{i}", "benchmark_summary": {"total_cases": 2}} for i in range(10)],
        "has_next": False,
        "next_cursor": None,
    }
    monkeypatch.setattr(checker, "fetch", lambda *a, **k: (page, None))

    walk = checker.walk_measured(pace=0.0)
    assert walk.complete is True, walk.reason
    assert len(walk.rows) == 10
