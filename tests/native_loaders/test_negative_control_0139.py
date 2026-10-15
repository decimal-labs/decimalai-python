"""The negative control: the contract must FAIL on what SDK 0.13.9 writes.

``fixtures/v0139_project`` is a frozen copy of 0.13.9's exporter output for the same
three skills the oracle installs (written by v0.13.9's own ``disk_export.py``,
through its most favourable route: ``export_skills_to_disk`` handed every field the
platform knows, ``url_slug`` included). It is graded against the oracle's
expectation: the same skills, ids and bodies a correct install would produce.

This stays in the suite for good. A ratchet nobody has seen go red protects
nothing; this one is red on every run, in the cells it predicts, and turns the
build red the day a contract item is weakened so far that 0.13.9's folders pass it.
"""

from __future__ import annotations

import pytest

from . import contract, ledger, source
from .loaders import LOADERS, items_for


def test_the_fixture_survived_its_trip_into_git():
    """``.gitignore`` drops ``.agents/`` and ``.decimal/``; the fixture stores them renamed."""
    files = source.negative_control_files()
    skill_mds = [f for f in files if f.startswith(".agents/skills/") and f.endswith("/SKILL.md")]
    assert len(skill_mds) == 3, files
    assert ".decimal/skills.lock" in files, files
    assert any(f.endswith("/scripts/compute_refund.py") for f in files), files


@pytest.mark.parametrize("loader", [l.name for l in LOADERS])
def test_the_predicted_cells_fail_on_0139_output(negative, loader):
    obs = negative.observation(loader)
    predicted = sorted(k.split(":", 1)[1] for k in ledger.NEGATIVE_CONTROL_PREDICTED
                       if k.split(":", 1)[0] == loader)
    failing = contract.failing_items(loader, items_for(loader), obs, negative.expect)
    survived = [item for item in predicted if item not in failing]
    assert not survived, (
        f"{loader}: the contract PASSES {survived} on SDK 0.13.9's folders, which it was "
        f"predicted (and once seen) to fail. A contract item went soft, or the observation "
        f"stopped carrying what it grades.")


@pytest.mark.parametrize("loader", [l.name for l in LOADERS])
def test_every_red_cell_on_0139_output_is_recorded(negative, loader):
    obs = negative.observation(loader)
    failing = contract.failing_items(loader, items_for(loader), obs, negative.expect)
    recorded = ledger.NEGATIVE_CONTROL_RED.get(loader)
    assert recorded is not None and set(failing) == set(recorded), (
        f"{loader}: on 0.13.9's output the contract fails {sorted(failing)}; "
        f"ledger.NEGATIVE_CONTROL_RED records {sorted(recorded or ())}.\n"
        + "\n".join(f"  {k}: {v[:240]}" for k, v in sorted(failing.items()))
        + "\nA cell that stopped failing is a softer item or a looser loader: find which "
        "before re-recording.")
