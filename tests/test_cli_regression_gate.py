"""The CLI gate must not pass an `unverified` verdict silently.

2026-07-28. The backend gained an `unverified` verdict — structural changes found,
no traffic to measure them against. The GitHub Action learned to rank it; this CLI
did not, and `rank.get(verdict, 0)` returned 0, so `decimalai regression-check`
exited 0 at EVERY --fail-on setting. That CLI is the documented path for non-GitHub
CI, and the docs show `--fail-on medium`.

Ranked by the DIFF's severity rather than a fixed value: a fixed rank is wrong in
both directions — too low and a deleted tool merges green, too high and a one-line
prompt tweak on an untrafficked agent reds the build.
"""

from decimalai.cli.main import _should_fail


class TestUnverifiedVerdict:
    def test_high_severity_unverified_fails_the_default_gate(self):
        assert _should_fail("unverified", "high", "high") is True

    def test_low_severity_unverified_does_not_fail_a_high_gate(self):
        # A prompt tweak nobody exercised must not red the build.
        assert _should_fail("unverified", "high", "low") is False

    def test_medium_gate_catches_medium_severity(self):
        assert _should_fail("unverified", "medium", "medium") is True

    def test_fail_on_none_still_never_fails(self):
        for sev in ("high", "medium", "low", None):
            assert _should_fail("unverified", "none", sev) is False

    def test_missing_severity_falls_back_to_warn_not_fail(self):
        """An older backend that does not send the field must not start reding builds."""
        assert _should_fail("unverified", "high", None) is False

    def test_known_verdicts_are_unchanged(self):
        assert _should_fail("high_risk", "high") is True
        assert _should_fail("medium_risk", "high") is False
        assert _should_fail("medium_risk", "medium") is True
        assert _should_fail("no_change", "high") is False
