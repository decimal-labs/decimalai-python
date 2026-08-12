"""Shared autouse fixtures for every live-LLM test file in this directory.

By living in conftest.py, these fixtures auto-apply to every test under
`tests/integration/` without each file having to re-declare them.
"""

import os

import pytest

from . import _live_helpers as _h

# Marks a TestReport whose failure the quota hook below downgraded to a skip,
# so the coverage floor can tell a quota/availability skip apart from a benign
# config skip (provider key not set).
_QUOTA_SKIP_ATTR = "_decimalai_quota_skip"


@pytest.fixture(autouse=True)
def _require_gates():
    _h.require_gates_fixture()


@pytest.fixture(autouse=True)
def _reset_sdk():
    _h.reset_sdk_fixture()
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Convert provider quota/rate-limit failures into skips.

    A 429/insufficient_quota from a provider is an environmental condition (the
    key's billing quota), not a code defect — failing on it reds the whole
    live sweep for a billing problem and buries real regressions. Scoped to
    tests/integration via this conftest, so unit tests are unaffected.
    """
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call" and call.excinfo is not None:
        if _h.is_provider_unavailable_error(call.excinfo.value):
            rep.outcome = "skipped"
            setattr(rep, _QUOTA_SKIP_ATTR, True)
            rep.longrepr = (
                str(item.fspath),
                (item.location[1] or 0) + 1,
                f"Skipped: provider unavailable (quota/rate-limit): {call.excinfo.value}",
            )


def pytest_sessionfinish(session, exitstatus):
    """Coverage floor for the live-LLM tier — defeat the quota masquerade.

    The quota hook above turns 429 / overload failures into skips so a billing
    wall can't red the suite. The cost: a run where the provider is *fully*
    quota-starved (only 2 of 31 cells actually executed) still exits 0 and looks
    like a clean pass. That's the dangerous case — a green check that verified
    almost nothing.

    So at session end we tally, *for live_llm tests only*:
      - executed       = real verdicts (passed + failed)
      - quota_skipped  = failures the hook downgraded (provider unavailable)
      - config_skipped = benign skips (provider key not set) — NOT counted
        against coverage, so a single-funded-provider run stays green.

    `attempted = executed + quota_skipped`. If the fraction that produced a real
    verdict falls below LIVE_MIN_RAN_FRACTION (default 0.5), we fail the session.
    config_skipped is excluded from `attempted` on purpose: not having an OpenAI
    key is a deliberate run shape, not a starved one.
    """
    tr = session.config.pluginmanager.get_plugin("terminalreporter")
    if tr is None:
        return

    def _live(reports):
        return [r for r in reports if "live_llm" in getattr(r, "keywords", {})]

    stats = tr.stats
    passed = len(_live(stats.get("passed", [])))
    failed = len(_live(stats.get("failed", [])))
    live_skips = _live(stats.get("skipped", []))
    quota_skipped = sum(1 for r in live_skips if getattr(r, _QUOTA_SKIP_ATTR, False))
    config_skipped = len(live_skips) - quota_skipped

    executed = passed + failed
    attempted = executed + quota_skipped

    # No live cell was even attempted (no keys, or none collected) — nothing to
    # police; the suite simply had no live work to do.
    if attempted == 0:
        return

    floor = float(os.environ.get("LIVE_MIN_RAN_FRACTION", "0.5"))
    fraction = executed / attempted
    breached = fraction < floor

    tr.write_sep("=", "LIVE-LLM COVERAGE FLOOR", red=breached, green=not breached)
    tr.write_line(f"  executed (real verdict): {executed}  (passed={passed} failed={failed})")
    tr.write_line(f"  quota/availability skips: {quota_skipped}")
    tr.write_line(f"  config skips (no key):    {config_skipped}  (excluded from coverage)")
    tr.write_line(
        f"  coverage: {executed}/{attempted} attempted = {fraction:.0%}  (floor {floor:.0%})"
    )
    if breached:
        tr.write_line(
            "  FAIL: too few live tests produced a real verdict — this run is "
            "quota-starved and is NOT a valid pass."
        )
        tr.write_line("        (tune with LIVE_MIN_RAN_FRACTION; set 0 to disable)")
        session.exitstatus = 1
    else:
        tr.write_line("  OK: live coverage above floor.")
    tr.write_sep("=", "", red=breached, green=not breached)
