"""The anti-drift guard: advertise a framework, and you must conformance-test it.

Every other file in this directory grades adapters that somebody remembered to
write a driver for. This one grades the *set* — it is the test that fails when a
framework is added to the product and quietly left out of the matrix.

Why that is the highest-value test here
---------------------------------------
The defect this whole suite exists to prevent — the LlamaIndex adapter shipping
for months emitting zero traces against a fully green suite — has a sibling that
no wire assertion can catch: an adapter that is never exercised at all. A
contract applied to ten of eleven frameworks says nothing about the eleventh,
and the gap is invisible precisely because nothing is red.

Three independent sources, so the guard cannot be defeated by editing one file
-------------------------------------------------------------------------------
1. **The docs capability table** (``https://docs.decimal.ai/sdk/python/frameworks``)
   — what the product tells users it supports. Vendored into
   ``drivers.ADVERTISED_SNAPSHOT`` so the guard still bites in a checkout that
   has only this repo, and re-derived whenever the docs repo IS on disk, so the
   snapshot cannot go stale in either direction.
2. **``decimalai.init()``'s framework flags** — the SDK's own advertised
   surface, and the one source that lives in THIS repo. This is what makes the
   guard real in GitHub Actions, where the docs repo is not checked out and
   source 1 can only skip.
3. **The framework extras in ``pyproject.toml``** — what ``pip install
   "decimalai[x]"`` promises to make work.

Adding a framework touches at least one of the three. Whichever one it is, this
file fails until ``tests/conformance/drivers/<name>.py`` exists, and the failure
message spells out what to create.

Nothing here is skippable, and there is deliberately **no debt ledger**. An
earlier draft had a ``KNOWN_MISSING`` set that suppressed the failure for
frameworks whose driver "wasn't written yet"; that turns a hard guard into a
formality — add the docs row, add the ledger line, stay green, which is the
exact outcome the guard exists to make impossible. A framework that genuinely
cannot satisfy part of the contract declares those ITEMS N/A in its driver's
``Capabilities``, with a reason that gets printed. It never opts out of having a
driver.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from typing import Dict, List, Set

import pytest

from . import contract
from .delivery import DELIVERY_MODES
from .drivers import ADVERTISED_SNAPSHOT, CAPABILITY_ITEMS, DRIVER_MODULES, all_drivers
from .journey import (
    SCAFFOLD_KEYS,
    journey_framework,
    journey_na_ledger,
    journey_requirements,
)
from .na_ledger import (
    DECLARED_DELIVERY_NA,
    DECLARED_JOURNEY_NA,
    DECLARED_NA,
    DELIVERY_NA_BUDGET,
    ENVIRONMENT_SKIPS,
    JOURNEY_NA_BUDGET,
    NA_BUDGET,
)

#: Tier A — this file needs no framework installed and no network. Every driver
#: module imports its framework lazily inside ``run()``, so ``all_drivers()``
#: works in a bare checkout, which is what lets the guard run in the same job as
#: the hermetic matrix (and in any job at all).
pytestmark = pytest.mark.conformance

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
DRIVERS_DIR = Path(__file__).parent / "drivers"
# Sibling docs checkout; absent in most clones, so the guard falls back to the
# vendored snapshot rather than depending on anyone's directory layout.
DOCS_FRAMEWORKS = (
    Path(__file__).resolve().parents[3]
    / "decimalai-docs"
    / "sdk"
    / "python"
    / "frameworks.mdx"
)


# ── source 1: the docs capability table ──────────────────────────────────────


def advertised_from_docs(text: str) -> Set[str]:
    """Slugs from the first column of the docs capability-comparison table.

    A row like ``| Claude Agent SDK / Claude Code | First-class | …`` yields two
    slugs, because the docs advertise both names and a reader may look for
    either. Parentheticals and inline code are qualifiers on the row, not
    separate frameworks, so they are dropped.
    """
    rows: List[str] = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("| Framework |"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not cells or set(cells[0]) <= set("-: "):
                continue
            rows.append(cells[0])
    slugs: Set[str] = set()
    for row in rows:
        row = re.sub(r"\(.*?\)", " ", row)      # drop parentheticals
        row = re.sub(r"`[^`]*`", " ", row)      # drop inline code
        for part in row.split("/"):
            slug = re.sub(r"[^a-z0-9]+", "-", part.strip().lower()).strip("-")
            if slug:
                slugs.add(slug)
    return slugs


# ── source 2: decimalai.init()'s framework flags ─────────────────────────────

#: ``init(<flag>=True)`` → the docs capability-table slug(s) that flag turns on.
#: One flag can serve several slugs: the LangChain callback handler is how
#: LangGraph is traced too, and one Claude Agent SDK adapter covers both names
#: the docs use for it.
FLAG_SLUGS: Dict[str, Set[str]] = {
    "langchain": {"langchain", "langgraph"},
    "openai_agents": {"openai-agents"},
    "adk": {"google-adk"},
    "llamaindex": {"llamaindex"},
    "claude_agent_sdk": {"claude-agent-sdk", "claude-code"},
    "crewai": {"crewai"},
    # AutoGen / AG2 is retired as an integration, but `init(autogen=True)` is
    # public API and still runs: it installs the generic OTel exporter and warns.
    # So it maps to the generic-otel slug, which drivers/generic_otel.py drives —
    # the flag IS that rail now, and grading it there is the honest reading, not
    # an exemption. If the flag ever regains framework-specific behaviour it
    # needs its own slug, its own docs row, and its own driver again.
    "autogen": {"generic-otel"},
    "otel": {"generic-otel"},
}

#: Flags that instrument a raw provider SDK rather than a framework. The docs
#: advertise these in the "No framework at all" prose, not in the capability
#: table, so they carry no slug — but they are still a rail somebody can ship a
#: no-op on, so each records which driver exercises it. A flag added here with
#: no driver named, or naming a driver that does not exist, fails below.
PROVIDER_FLAG_DRIVERS: Dict[str, str] = {
    # init(anthropic=True) is providers.instrument(anthropic=True) — the
    # OpenInference Anthropic instrumentor routed through DecimalSpanExporter.
    "anthropic": "anthropic",
    # init(openai=True) is the same rail for the `openai` SDK. The Pydantic AI
    # driver's documented setup IS `instrument() + init(openai=True)` (Pydantic
    # AI does no tracing of its own), so that driver exercises this flag end to
    # end on every run.
    "openai": "pydantic-ai",
    # init(google=True) has no driver: the hermetic tier needs a stub that
    # speaks the provider's wire format, and google.genai's client has no
    # base_url seam to point at one the way `openai` and `anthropic` do. This is
    # a real gap, recorded rather than hidden — see README, "what v1 does not
    # cover".
    "google": "",
}


def init_framework_flags() -> Set[str]:
    """Every opt-in boolean flag on ``decimalai.init()``.

    Opt-in means ``bool`` annotated with a ``False`` default, which is exactly
    the "turn this integration on" shape — ``enabled``/``verify`` default True
    and are configuration, not integrations, so they never appear here and never
    need a ledger entry.
    """
    import decimalai

    return {
        name
        for name, p in inspect.signature(decimalai.init).parameters.items()
        if p.annotation is bool and p.default is False
    }


# ── source 3: the framework extras in pyproject.toml ─────────────────────────

#: ``pip install "decimalai[<extra>]"`` → the slug(s) that extra makes work.
EXTRA_SLUGS: Dict[str, Set[str]] = {
    "langchain": {"langchain"},
    "langgraph": {"langgraph"},
    "openai-agents": {"openai-agents"},
    "claude-agent-sdk": {"claude-agent-sdk", "claude-code"},
    "pydantic-ai": {"pydantic-ai"},
    "adk": {"google-adk"},
    "llamaindex": {"llamaindex"},
}

#: Extras that install no framework adapter, each with the reason. Anything
#: ending in ``-tests`` is a test-only install set and is classified
#: automatically (it pulls the same frameworks for a live lane, so requiring a
#: separate driver for it would double-count).
NON_FRAMEWORK_EXTRAS: Dict[str, str] = {
    "openai": "the provider SDK for the init(openai=True) rail, not a framework "
              "— see PROVIDER_FLAG_DRIVERS",
    "evals": "litellm, for the eval runner; instruments nothing",
    "all": "an alias for every runtime extra above; adds no framework of its own",
    "dev": "the unit-test toolchain",
}


def declared_extras(pyproject_text: str) -> List[str]:
    """Extra names from ``[project.optional-dependencies]``.

    Hand-parsed rather than via ``tomllib`` because this repo's CI matrix still
    includes Python 3.10, where ``tomllib`` does not exist and no TOML parser is
    a declared test dependency. Only the extras' NAMES are needed, and they are
    the only ``name = [`` lines in that section.
    """
    lines = pyproject_text.splitlines()
    try:
        start = lines.index("[project.optional-dependencies]") + 1
    except ValueError:
        raise AssertionError(
            "pyproject.toml has no [project.optional-dependencies] section — "
            "this guard reads the extras from there"
        )
    names: List[str] = []
    for line in lines[start:]:
        if line.startswith("["):
            break
        m = re.match(r"^([A-Za-z0-9._-]+)\s*=\s*\[", line)
        if m:
            names.append(m.group(1))
    return names


# ── what the drivers actually cover ──────────────────────────────────────────


def covered_slugs() -> Set[str]:
    drivers = all_drivers()
    return set().union(*(d.covers for d in drivers)) if drivers else set()


def _how_to_add(slug: str, source: str) -> str:
    """The failure text. Its whole job is to leave nothing to work out.

    A guard that says "uncovered: {'haystack'}" makes the reader reverse-engineer
    the suite before they can obey it; the point of the guard is that obeying it
    is cheap.
    """
    module = slug.replace("-", "_")
    return f"""
NO CONFORMANCE DRIVER FOR AN ADVERTISED FRAMEWORK: {slug!r}

  advertised by: {source}
  driver:        none of tests/conformance/drivers/*.py declares covers={{{slug!r}}}

An adapter nobody conformance-tests is an adapter nobody has proven emits a
trace. That is not hypothetical: the LlamaIndex adapter shipped for months
emitting NOTHING, with 626 adapter tests green, because every one of them drove
a mock instead of the wire.

To fix, in tests/conformance/ (about 40 lines of framework-specific code):

  1. Create drivers/{module}.py. Copy the closest existing driver —
     drivers/generic_otel.py is the shortest complete example, drivers/langchain.py
     the most representative. It needs exactly three things:

       * a stub model that answers the SHARED script: for turn in stub_script(ctx),
         emit turn.tool_call / turn.content / turn.input_tokens / turn.output_tokens
         as whatever message object this framework expects. This is the only
         genuinely framework-specific code in the file.
       * def run(ctx): the snippet the docs page for {slug} actually shows, with
         the model swapped for the stub and base_url=ctx.base_url. Put
         ctx.prompt_sentinel in the prompt via user_message(ctx), and register
         ctx.tool_name as the tool.
       * DRIVER = Driver(
             name={slug!r},
             covers=frozenset({{{slug!r}}}),
             requires=("<the import(s) it needs>",),
             entrypoint="<the SDK surface it exercises>",
             run=run,
             run_concurrent=fanout_threads(run),
             run_error=..., run_degenerate=..., run_skills=...,
         )
         Every optional hook you leave out must be declared N/A in
         capabilities=Capabilities(...) with a reason — the reason is printed in
         the matrix. Nothing is ever silently skipped.

  2. Add {module!r} to DRIVER_MODULES in drivers/__init__.py.

  3. If the docs capability table gained a row, add {slug!r} to
     ADVERTISED_SNAPSHOT in drivers/__init__.py too (the vendored copy of that
     table, so this guard works without the docs repo on disk).

  4. Run `pytest tests/conformance`. Whatever the adapter gets wrong now FAILS.
     That is the correct outcome, not a broken test — record it and fix the
     adapter.

Write NO assertions in the driver: `test_drivers_contain_no_assertions` enforces
that structurally. If this framework seems to need its own assertion, the
contract is wrong — fix contract.py so every framework gets the fix.
"""


# ── the guards ───────────────────────────────────────────────────────────────


def test_advertised_snapshot_matches_docs() -> None:
    """The vendored advertised list is what the docs actually advertise.

    Skips when the docs repo is not beside this one — which is the normal case
    in GitHub Actions. The two guards below do not skip, and between them cover
    every way a framework can be added from inside this repo.
    """
    if not DOCS_FRAMEWORKS.exists():
        pytest.skip(f"docs repo not on disk at {DOCS_FRAMEWORKS}")
    from_docs = advertised_from_docs(DOCS_FRAMEWORKS.read_text())
    assert from_docs, "could not parse the capability table out of frameworks.mdx"
    only_docs = sorted(from_docs - set(ADVERTISED_SNAPSHOT))
    only_snap = sorted(set(ADVERTISED_SNAPSHOT) - from_docs)
    assert from_docs == set(ADVERTISED_SNAPSHOT), (
        "the docs capability table and drivers.ADVERTISED_SNAPSHOT disagree.\n"
        f"  only in the docs table:  {only_docs}\n"
        f"  only in the snapshot:    {only_snap}\n"
        "\n"
        "If the docs gained a row, that framework now needs a conformance driver: "
        "add it to ADVERTISED_SNAPSHOT and create tests/conformance/drivers/<name>.py "
        "(the next guard prints the full recipe).\n"
        "If a row was REMOVED from the docs, drop the slug from ADVERTISED_SNAPSHOT "
        "and from the driver's covers= — a driver for something the product no "
        "longer advertises is dead weight."
    )


def test_every_advertised_framework_has_a_driver() -> None:
    """Every framework in the capability table is exercised by a driver."""
    uncovered = sorted(set(ADVERTISED_SNAPSHOT) - covered_slugs())
    assert not uncovered, "\n".join(
        _how_to_add(slug, "the docs capability table (drivers.ADVERTISED_SNAPSHOT)")
        for slug in uncovered
    )


def test_drivers_cover_only_advertised_frameworks() -> None:
    """A driver may not claim a slug the product does not advertise.

    Backwards drift, and it matters because ``covers`` is what the guard above
    counts: a driver claiming ``langchain-classic`` would silently satisfy
    nothing while looking like coverage.
    """
    orphan = sorted(covered_slugs() - set(ADVERTISED_SNAPSHOT))
    assert not orphan, (
        f"drivers claim covers= slugs the docs do not advertise: {orphan}.\n"
        "Either the slug is misspelled (compare against ADVERTISED_SNAPSHOT in "
        "drivers/__init__.py), or the docs table needs that row. A driver for a "
        "rail the docs advertise only in prose — the raw-provider one-liners — "
        "declares covers=frozenset() and is recorded in PROVIDER_FLAG_DRIVERS "
        "here instead."
    )


def test_every_init_framework_flag_has_a_driver() -> None:
    """Every ``init(<flag>=True)`` integration is classified and exercised.

    This is the guard that works in a checkout of this repo alone: adding
    ``init(haystack=True)`` fails here on the same commit that adds it, whether
    or not the docs repo is anywhere nearby.
    """
    flags = init_framework_flags()
    known = set(FLAG_SLUGS) | set(PROVIDER_FLAG_DRIVERS)

    unclassified = sorted(flags - known)
    assert not unclassified, (
        f"decimalai.init() gained integration flag(s) with no conformance "
        f"classification: {unclassified}.\n"
        "\n"
        "Every opt-in flag on init() turns on an adapter, and an adapter with no "
        "driver is one nobody has proven emits a trace. In "
        "tests/conformance/test_coverage.py either:\n"
        "  * add it to FLAG_SLUGS mapping it to its docs capability-table slug(s), "
        "and create tests/conformance/drivers/<name>.py for it; or\n"
        "  * add it to PROVIDER_FLAG_DRIVERS if it instruments a raw provider SDK "
        "rather than a framework, naming the driver that exercises that rail."
    )

    stale = sorted(known - flags)
    assert not stale, (
        f"test_coverage.py classifies init() flag(s) that no longer exist: {stale}. "
        f"Remove them from FLAG_SLUGS / PROVIDER_FLAG_DRIVERS — and if the "
        f"integration is gone for good, retire its driver and its "
        f"ADVERTISED_SNAPSHOT slug with it."
    )

    covered = covered_slugs()
    for flag in sorted(FLAG_SLUGS):
        missing = sorted(FLAG_SLUGS[flag] - covered)
        assert not missing, "\n".join(
            _how_to_add(slug, f"decimalai.init({flag}=True)") for slug in missing
        )

    driver_names = {d.name for d in all_drivers()}
    for flag, driver in sorted(PROVIDER_FLAG_DRIVERS.items()):
        if not driver:
            continue  # a recorded gap, printed by test_recorded_provider_gaps
        assert driver in driver_names, (
            f"PROVIDER_FLAG_DRIVERS says init({flag}=True) is exercised by driver "
            f"{driver!r}, and no such driver exists (have: {sorted(driver_names)}). "
            f"Either the driver was renamed — update the mapping — or the rail lost "
            f"its coverage, in which case write the driver."
        )


def test_every_framework_extra_has_a_driver() -> None:
    """Every ``pip install "decimalai[x]"`` framework extra is exercised.

    The extras are a promise made on PyPI: install this and the adapter works.
    A framework can arrive here first — an extra can land before the docs page
    does — so this is the third independent way in.
    """
    extras = declared_extras(PYPROJECT.read_text())
    assert extras, "parsed no extras out of pyproject.toml — the parser has drifted"

    classified = set(EXTRA_SLUGS) | set(NON_FRAMEWORK_EXTRAS)
    unclassified = sorted(
        e for e in extras if e not in classified and not e.endswith("-tests")
    )
    assert not unclassified, (
        f"pyproject.toml gained extra(s) with no conformance classification: "
        f"{unclassified}.\n"
        "\n"
        "In tests/conformance/test_coverage.py either add it to EXTRA_SLUGS "
        "(mapping it to the docs capability-table slug it makes work, and "
        "creating a driver for that slug), or to NON_FRAMEWORK_EXTRAS with the "
        "reason it instruments nothing. `<name>-tests` extras classify "
        "themselves — they install frameworks for a live lane, and the runtime "
        "extra beside them already carries the driver."
    )

    stale = sorted(set(EXTRA_SLUGS) - set(extras))
    assert not stale, (
        f"EXTRA_SLUGS maps extras that pyproject.toml no longer declares: {stale}"
    )

    covered = covered_slugs()
    for extra in sorted(EXTRA_SLUGS):
        missing = sorted(EXTRA_SLUGS[extra] - covered)
        assert not missing, "\n".join(
            _how_to_add(slug, f'pip install "decimalai[{extra}]"') for slug in missing
        )


def test_every_driver_module_on_disk_is_registered() -> None:
    """A driver file nobody runs is worse than no driver file.

    It reads like coverage in a directory listing and grades nothing. Catches
    the half-landed change: driver written, ``DRIVER_MODULES`` not updated.
    """
    on_disk = {
        p.stem for p in DRIVERS_DIR.glob("*.py")
        if not p.stem.startswith("_")
    }
    unregistered = sorted(on_disk - set(DRIVER_MODULES))
    assert not unregistered, (
        f"driver module(s) on disk but not in DRIVER_MODULES: {unregistered}. "
        f"Add them to drivers/__init__.py so they are actually graded — or, if "
        f"the file is a shared helper rather than a driver, prefix its name with "
        f"an underscore (like drivers/_openai_wire.py)."
    )
    ghosts = sorted(set(DRIVER_MODULES) - on_disk)
    assert not ghosts, (
        f"DRIVER_MODULES lists module(s) with no file: {ghosts}"
    )


def test_no_driver_is_silently_unavailable() -> None:
    """In a full-install environment, every driver must actually run.

    This closes the one way the hermetic tier can be green and prove nothing. A
    driver whose imports are absent is recorded ``NOT RUN`` in the matrix and its
    fifteen items SKIP — so a CI job that forgot half the framework installs is
    indistinguishable, at the exit code, from one where every adapter passed.

    Opt-in, via ``DECIMAL_CONFORMANCE_REQUIRE_ALL=1``, because the honest answer
    differs by environment: a developer with only ``[dev]`` installed SHOULD get
    skips rather than a red suite. CI sets it after installing
    ``.[conformance-tests]``, which is the environment where a missing import is
    a build problem rather than a fact of life.
    """
    import os

    if os.environ.get("DECIMAL_CONFORMANCE_REQUIRE_ALL") != "1":
        pytest.skip(
            "set DECIMAL_CONFORMANCE_REQUIRE_ALL=1 (after installing "
            '`.[conformance-tests]`) to require every driver to be runnable'
        )
    unavailable = {
        d.name: d.missing_requirements for d in all_drivers() if not d.available
    }
    assert not unavailable, (
        "DECIMAL_CONFORMANCE_REQUIRE_ALL=1 is set, and these drivers cannot run "
        "because their framework is not installed:\n"
        + "\n".join(f"  {name}: missing {mods}" for name, mods in sorted(unavailable.items()))
        + "\n\nThese would have been reported NOT RUN and skipped, leaving the job "
        "green while proving nothing about those adapters. Add the missing "
        "distribution(s) to the `conformance-tests` extra in pyproject.toml — that "
        "extra is the single source for what this tier needs installed."
    )


def test_every_baseline_line_names_something_real() -> None:
    """A ledger line for a driver or item that no longer exists grades nothing.

    ``known_failures.txt`` is self-cleaning in one direction only: a listed item
    that starts passing fails the build. The other direction is silent — delete
    a driver, and its lines stop matching any test at all, so they sit in the
    ledger forever reading like known debt while documenting a defect that can
    no longer occur. That is how a ledger stops being a ledger: the honest count
    of what is red drifts above the real one, and nobody trusts it enough to
    shrink it.

    This is not hypothetical. Retiring the AG2 driver left ``ag2:C6`` and
    ``ag2:C9`` behind, and the matrix stayed green with two orphans in the file.
    """
    from .test_conformance import BASELINE, BASELINE_PATH
    from . import contract

    # Driver NAMES, not module names — the matrix keys on `driver.name`, and the
    # two differ wherever a distribution has a hyphen in it (module
    # ``pydantic_ai`` → driver ``pydantic-ai``). Comparing against
    # DRIVER_MODULES here reads correct and quietly flags every hyphenated
    # driver as an orphan.
    known_drivers = {d.name for d in all_drivers()}
    known_items = set(contract.ITEMS)

    orphans: List[str] = []
    for key in sorted(BASELINE):
        driver, _, item = key.partition(":")
        if driver not in known_drivers:
            orphans.append(
                f"  {key}: no driver named {driver!r} — was it retired? "
                f"Delete the line."
            )
        elif item not in known_items:
            orphans.append(
                f"  {key}: {item!r} is not a contract item. "
                f"Known items: {', '.join(sorted(known_items))}"
            )

    assert not orphans, (
        f"{BASELINE_PATH.name} has line(s) that match no test, so they can "
        f"never go green and never fail:\n" + "\n".join(orphans)
    )


def test_recorded_provider_gaps_are_visible(record_property) -> None:
    """Print the raw-provider rails that have no driver, rather than hiding them.

    ``PROVIDER_FLAG_DRIVERS`` allows an empty driver name for a rail the
    hermetic tier genuinely cannot stub today. That is a real gap in coverage,
    and the one thing it must never be is invisible — so it is named, in the
    test output, on every run. It does not fail the suite: the rail is out of
    the capability table, and failing here would make the only honest way to
    record a gap a red build, which is how gaps stop being recorded.
    """
    gaps = sorted(f for f, d in PROVIDER_FLAG_DRIVERS.items() if not d)
    record_property("conformance_provider_gaps", ",".join(gaps))
    if gaps:
        print(
            "\nCONFORMANCE GAP — raw-provider rails with no driver: "
            + ", ".join(f"decimalai.init({f}=True)" for f in gaps)
            + "\n  Advertised in the docs prose, not the capability table, so no "
              "guard fails on them. See tests/conformance/README.md, "
              "'What v1 does not cover'."
        )


# ── the N/A ledger: bound the holes, and count them ──────────────────────────
#
# Everything above answers "is this framework exercised at all?". These answer
# the question one level in: "and how much of the contract did it opt OUT of?".
# `Capabilities.__post_init__` already refuses a SILENT exemption — every False
# flag needs a reason, and the reason is printed. What nothing checked was the
# SET. Each exemption arrives alone, with its own perfectly good sentence, and
# nothing ever looks at the total again.
#
# That is not hypothetical. C13b was declared N/A on langchain by a reason that
# was accurate, careful, and WAS the defect: "the model has no way to ASK for a
# body, so the strongest rung observable here is DELIVERED" describes, precisely,
# a rail that then delivered nothing. Prose cannot tell a framework's limit from
# an adapter's choice, because the two read identically. A ledger can at least
# make the set move in a diff.


def computed_na() -> Dict[str, str]:
    """Every ``driver:item`` the drivers would grade N/A, and the flag granting it."""
    out: Dict[str, str] = {}
    for driver in all_drivers():
        caps = driver.capabilities
        for item in contract.ITEM_ORDER:
            if caps.na_reason(item) is None:
                continue
            flag = next(
                f for f, items in CAPABILITY_ITEMS.items()
                if item in items and not getattr(caps, f)
            )
            out[f"{driver.name}:{item}"] = flag
    return out


def test_every_capability_na_is_declared_and_counted() -> None:
    """The N/A set is exactly the declared one, and the total is written down."""
    actual = computed_na()
    added = sorted(set(actual) - set(DECLARED_NA))
    removed = sorted(set(DECLARED_NA) - set(actual))
    regranted = sorted(
        f"{k} (declared under {DECLARED_NA[k]}, now granted by {actual[k]})"
        for k in set(actual) & set(DECLARED_NA)
        if actual[k] != DECLARED_NA[k]
    )
    assert not (added or removed or regranted), (
        "the set of contract items declared N/A has moved, and "
        "tests/conformance/na_ledger.py::DECLARED_NA has not.\n"
        f"  newly N/A (a hole nobody declared): {added}\n"
        f"  no longer N/A (a stale exemption):  {removed}\n"
        f"  granted by a different flag:        {regranted}\n"
        "\n"
        "An N/A reads like coverage in the matrix and like a pass at the exit "
        "code. Adding one is a deliberate act: put the line in the ledger with "
        "the flag that grants it, and move NA_BUDGET. Removing one is better — "
        "delete the line. Never widen an existing flag to swallow another item: "
        "that is how C14 nearly ended up hidden behind "
        "model_can_load_skill_bodies."
    )
    assert len(actual) == NA_BUDGET, (
        f"{len(actual)} contract items are N/A but na_ledger.NA_BUDGET says "
        f"{NA_BUDGET}. The number is stated separately from the ledger on purpose "
        f"— it is the one line a reviewer skimming a diff sees move."
    )


def test_every_delivery_na_is_declared_and_counted() -> None:
    """Same, for the delivery axis — where an N/A hides a whole body channel."""
    actual = {
        f"{d.name}:{mode}": d.capabilities.delivery_limits[mode].adapter_module
        for d in all_drivers()
        for mode in DELIVERY_MODES
        if d.capabilities.delivery_limit(mode) is not None
    }
    assert actual == dict(DECLARED_DELIVERY_NA), (
        "the set of delivery channels declared a FRAMEWORK limit has moved.\n"
        f"  computed from the drivers: {sorted(actual)}\n"
        f"  declared in the ledger:    {sorted(DECLARED_DELIVERY_NA)}\n"
        "\n"
        "A channel a framework cannot carry is a limit; a channel an adapter has "
        "not implemented is a defect. Only the first may be declared, only with a "
        "FrameworkLimit proof, and only with a line here."
    )
    assert len(actual) == DELIVERY_NA_BUDGET, (
        f"{len(actual)} delivery cells are N/A but "
        f"na_ledger.DELIVERY_NA_BUDGET says {DELIVERY_NA_BUDGET}."
    )


def test_every_journey_na_is_declared_and_counted() -> None:
    """The journey exemptions are exactly the SDK's own unscaffoldable set.

    Nothing here is this suite's opinion. ``decimalai/cli/scaffold.py`` decides,
    in shipped product code, which frameworks ``decimalai init`` writes a file
    for; this recomputes the exemption set from that and fails on any difference
    in either direction.

    Which makes the ledger self-clearing in the direction that matters: the day
    somebody writes a template for the Anthropic Messages adapter, the framework
    moves out of ``UNSCAFFOLDED_WITH_SEAM``, ``anthropic:J1`` stops being N/A and
    starts being GRADED — and this test fails until the stale line is deleted.
    The alternative, a list of names typed here, would have kept skipping the
    cell for a journey that had quietly started working.
    """
    actual = {
        f"{d.name}:{contract.JOURNEY_ITEM}": journey_na_ledger(d.name)
        for d in all_drivers()
        if journey_framework(d.name) is None
    }
    assert actual == dict(DECLARED_JOURNEY_NA), (
        "the set of frameworks with no `decimalai init` journey has moved.\n"
        f"  computed from decimalai/cli/scaffold.py: {sorted(actual)}\n"
        f"  declared in the ledger:                  {sorted(DECLARED_JOURNEY_NA)}\n"
        "\n"
        "A framework the CLI refuses to scaffold has no journey to walk, and that "
        "is a fact about the product, not a choice this suite gets to make. If a "
        "framework gained a template, delete its line — its journey is now graded. "
        "If one lost its template, that is a product regression before it is a "
        "ledger update."
    )
    assert len(actual) == JOURNEY_NA_BUDGET, (
        f"{len(actual)} frameworks have no journey but "
        f"na_ledger.JOURNEY_NA_BUDGET says {JOURNEY_NA_BUDGET}. This number should "
        f"go DOWN — every entry is a framework whose users get no scaffold."
    )


def test_the_journey_requirement_set_is_derivable_for_every_scaffoldable_framework() -> None:
    """Every scaffoldable framework can say what its generated file needs.

    ``journey.journey_requirements`` derives that from two product sources — the
    imports of the file ``render_agent_file`` emits, and the packages
    ``install_command()`` tells the user to install. Both can be computed with no
    framework installed, which is what lets the journey cell SKIP honestly on a
    slim checkout instead of erroring inside a fixture.

    It also catches the quiet one: a template whose provider binding is named
    nowhere. ``langchain-openai`` appears in no import line — the file calls
    ``init_chat_model("gpt-4o-mini")`` and the binding is chosen by the model
    string — so a requirement set built from imports alone would have declared
    the journey runnable on a machine that cannot run it.
    """
    from decimalai.cli.scaffold import SUPPORTED_FRAMEWORKS

    for framework in SUPPORTED_FRAMEWORKS:
        required = journey_requirements(framework)
        assert "decimalai" in required, (
            f"the file `decimalai init --framework {framework}` writes does not "
            f"import decimalai — every template must, and the journey's "
            f"requirement derivation reads the imports"
        )
        assert len(required) > 1, (
            f"{framework}: journey_requirements returned only {required}. A "
            f"generated agent file needs its framework too; a one-element answer "
            f"means the derivation stopped working and the cell would 'run' on a "
            f"machine with nothing installed"
        )


def test_no_rail_may_declare_it_cannot_deliver() -> None:
    """A rail that cannot get a body to the model is not a rail. It is the defect.

    ``rail_can_deliver_bodies`` exists so that declaring C14 N/A is a move
    somebody has to make in a named place — and this guard is why that place is
    watched. There is no legitimate False here while ``has_skills_rail`` is True:
    the answer to "this adapter has no way to deliver a body" is a fix in the
    adapter, not an exemption in ``Capabilities``.

    The distinction it protects is the one the 2026-08-28 defect turned on.
    ``model_can_load_skill_bodies=False`` is a TRUE statement about langchain: the
    model genuinely cannot ask for a body through an invoke-layer patch, and C13b
    genuinely has nothing to find. But the same sentence, one word wider, would
    have exempted C14 as well — and C14 is exactly the item that catches an
    adapter with no body channel at all. One flag, two questions, and the second
    answer is never "N/A".
    """
    offenders = sorted(
        d.name for d in all_drivers()
        if d.capabilities.has_skills_rail and not d.capabilities.rail_can_deliver_bodies
    )
    assert not offenders, (
        f"driver(s) {offenders} declare has_skills_rail=True and "
        f"rail_can_deliver_bodies=False.\n"
        "\n"
        "That combination says: this framework is offered a menu of skills and "
        "has no way to put any of their bodies in front of the model. That is not "
        "a capability the contract should skip — it is the exact defect C14 was "
        "written for, after a shipped `decimalai init` agent answered a real "
        "customer ticket with one sentence about its own tooling because it had "
        "been handed titles it could not read.\n"
        "\n"
        "Fix the adapter: give it a body channel (a registered load_skill tool, or "
        "inject_skill_body resolving True — see DecimalConfig.resolve_inject_body). "
        "If the framework genuinely cannot carry ONE of the two channels, that is "
        "a delivery_limits entry with a FrameworkLimit proof, which exempts that "
        "CHANNEL and leaves C14 graded on the other."
    )


# ── cross-checks against the SDK's own ledgers ───────────────────────────────

#: ``SCAFFOLD_KEYS`` (driver name -> the key(s) ``decimalai/cli/scaffold.py``
#: files that framework under) moved to ``journey.py`` on 2026-08-29, when the
#: journey tier became its second consumer. It answers a question about
#: ``decimalai init``, which is what that module is about, and two copies would
#: have drifted the first time a framework was renamed.


def test_rail_declarations_match_the_scaffold_seam_ledger() -> None:
    """``has_skills_rail`` must agree with the SDK's own list of seam-carrying
    frameworks — a SECOND source, in shipped product code, for the same fact.

    ``decimalai/cli/scaffold.py`` splits every framework three ways:
    ``SUPPORTED_FRAMEWORKS`` (a seam and a template), ``UNSCAFFOLDED_WITH_SEAM``
    (a seam, no template yet) and ``NO_PROMPT_SEAM`` (no seam at all — "a
    generated file would trace correctly and deliver none of the agent's skills,
    silently", which is why `decimalai init` refuses to write one).

    That is the same claim ``has_skills_rail`` makes, decided independently, by
    the code users actually run. If they ever disagree, one of them is wrong and
    somebody must say which: a driver claiming no rail on a framework the CLI
    will happily scaffold means an adapter is being graded on nothing, and the
    reverse means the CLI is refusing to scaffold a framework that works.

    This file is read, never edited — it belongs to the SDK, not to the suite.
    """
    from decimalai.cli.scaffold import (
        NO_PROMPT_SEAM,
        SUPPORTED_FRAMEWORKS,
        UNSCAFFOLDED_WITH_SEAM,
    )

    with_seam = set(SUPPORTED_FRAMEWORKS) | set(UNSCAFFOLDED_WITH_SEAM)
    without = set(NO_PROMPT_SEAM)

    unclassified = sorted(
        f"{name} (scaffold keys {sorted(keys)})"
        for name, keys in SCAFFOLD_KEYS.items()
        if not (keys & (with_seam | without))
    )
    assert not unclassified, (
        f"driver(s) {unclassified} name no framework the SDK's scaffold ledger "
        f"classifies. Either SCAFFOLD_KEYS is stale, or `decimalai init` has "
        f"never been taught this framework exists."
    )

    missing = sorted(set(d.name for d in all_drivers()) - set(SCAFFOLD_KEYS))
    assert not missing, (
        f"driver(s) {missing} have no SCAFFOLD_KEYS entry, so this cross-check "
        f"silently skips them — which is exactly the shape of hole it exists to "
        f"close. Map each to the key(s) decimalai/cli/scaffold.py files it under."
    )

    disagreements: List[str] = []
    for driver in all_drivers():
        keys = SCAFFOLD_KEYS[driver.name]
        seam = bool(keys & with_seam)
        no_seam = bool(keys & without)
        if seam and not driver.capabilities.has_skills_rail:
            disagreements.append(
                f"  {driver.name}: the SDK scaffolds/lists it as seam-carrying "
                f"({sorted(keys & with_seam)}), but the driver declares "
                f"has_skills_rail=False — so C8/C13/C13b/C14 are all N/A on an "
                f"adapter the product says can deliver skills"
            )
        if no_seam and driver.capabilities.has_skills_rail:
            disagreements.append(
                f"  {driver.name}: decimalai/cli/scaffold.py::NO_PROMPT_SEAM says "
                f"this framework has no prompt seam ({sorted(keys & without)}) and "
                f"`decimalai init` refuses to scaffold it, but the driver declares "
                f"has_skills_rail=True"
            )
    assert not disagreements, (
        "the conformance drivers and decimalai/cli/scaffold.py disagree about "
        "which frameworks can deliver a skill:\n" + "\n".join(disagreements)
    )


def test_every_framework_limit_is_still_in_the_adapter() -> None:
    """A FrameworkLimit cites a refusal in adapter source. It must still be there.

    The runtime half of the proof lives in ``contract._grade_framework_limit``,
    which will not grant the N/A unless the adapter EMITS the refusal on the run.
    This is the static half, and it runs with no framework installed: it catches
    the case where the adapter was rewritten, the warning deleted, and the
    delivery cell then skipped in every environment that cannot run that driver.
    """
    problems: List[str] = []
    for driver in all_drivers():
        for mode, limit in sorted(driver.capabilities.delivery_limits.items()):
            if not limit.adapter_path.exists():
                problems.append(
                    f"  {driver.name}:{mode} cites {limit.adapter_module}, which "
                    f"does not exist at {limit.adapter_path}"
                )
            elif not limit.marker_in_source():
                problems.append(
                    f"  {driver.name}:{mode} cites {limit.adapter_module} for the "
                    f"refusal {limit.refusal_marker!r}, and that text is no longer "
                    f"in the file — either the adapter grew the capability (delete "
                    f"the limit and let the cell be graded) or it stopped saying so"
                )
    assert not problems, (
        "framework-limit proof(s) no longer hold:\n" + "\n".join(problems)
    )


# ── skips ────────────────────────────────────────────────────────────────────


def _skip_call_sites(path: Path) -> List[tuple]:
    """``(function name, lineno, call)`` for every ``pytest.skip`` in ``path``."""
    tree = ast.parse(path.read_text())
    out: List[tuple] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            func = inner.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "pytest"
                and func.attr in ("skip", "importorskip")
            ):
                out.append((node.name, inner.lineno, f"pytest.{func.attr}"))
    return out


def test_no_undeclared_skip_call_sites() -> None:
    """A skip is how an ungraded cell reaches the exit code as a success.

    ``pytest -q`` prints ``102 skipped`` next to a green bar and nobody reads
    102. So every skip in this package is either

    * a declared N/A or a missing framework, taken through
      ``na_ledger.skip_declared``, which FAILS when the key is not in the ledger; or
    * one of the ``ENVIRONMENT_SKIPS`` — a guard whose subject genuinely is not
      on this machine (the docs repo, the platform repo, an opt-in env var), each
      named with the condition.

    Anything else is a new way to be quietly ungraded, so it fails here, on the
    commit that adds it.
    """
    allowed_functions = set(ENVIRONMENT_SKIPS) | {"skip_declared"}
    offenders: List[str] = []
    here = Path(__file__).parent
    for path in sorted(here.glob("*.py")):
        for func_name, lineno, call in _skip_call_sites(path):
            if func_name not in allowed_functions:
                offenders.append(f"  {path.name}:{lineno} {call} in {func_name}()")
    assert not offenders, (
        "undeclared skip call site(s):\n" + "\n".join(offenders) + "\n\n"
        "Route the skip through na_ledger.skip_declared(kind, key, reason) so it "
        "is checked against the ledger — or, if it is an environment condition "
        "rather than an ungraded cell, add the function to "
        "na_ledger.ENVIRONMENT_SKIPS with the condition spelled out.\n"
        f"Declared environment skips today: {sorted(ENVIRONMENT_SKIPS)}"
    )


def test_every_environment_skip_names_a_real_test() -> None:
    """A declared environment skip for a test that no longer exists grants an
    exemption nobody can see, to a function nobody can find."""
    here = Path(__file__).parent
    defined: Set[str] = set()
    for path in sorted(here.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined.add(node.name)
    ghosts = sorted(set(ENVIRONMENT_SKIPS) - defined)
    assert not ghosts, (
        f"na_ledger.ENVIRONMENT_SKIPS names function(s) that do not exist: "
        f"{ghosts}. Delete them — a stale allowance is an allowance nobody is "
        f"reading."
    )


def test_every_delivery_baseline_line_names_something_real() -> None:
    """The delivery ledger's orphan guard, for the same reason as the item one.

    ``known_delivery_failures.txt`` is self-cleaning in one direction: a listed
    cell that starts passing fails the build. The other direction is silent —
    rename a driver or retire a mode and the line matches no test at all, so it
    sits there forever reading like known debt while documenting a defect that
    can no longer occur.
    """
    from .test_conformance import DELIVERY_BASELINE, DELIVERY_BASELINE_PATH

    known_drivers = {d.name for d in all_drivers() if d.capabilities.has_skills_rail}
    orphans: List[str] = []
    for key in sorted(DELIVERY_BASELINE):
        driver, _, mode = key.partition(":")
        if driver not in known_drivers:
            orphans.append(
                f"  {key}: no rail driver named {driver!r} — was it retired, or did "
                f"it lose has_skills_rail? Delete the line."
            )
        elif mode not in DELIVERY_MODES:
            orphans.append(
                f"  {key}: {mode!r} is not a delivery mode "
                f"({', '.join(DELIVERY_MODES)})"
            )
    assert not orphans, (
        f"{DELIVERY_BASELINE_PATH.name} has line(s) that match no cell, so they "
        f"can never go green and never fail:\n" + "\n".join(orphans)
    )
