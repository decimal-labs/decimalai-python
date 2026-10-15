"""N1-N10 for every native loader, on the project a correct install leaves on disk.

One cell per (loader, item), graded by ``contract.py`` from the loader's child
observation. A loader graded in the other test environment is skipped with a reason
naming that environment; a declared N/A skips only after its proof held on this run;
a known upstream gap is an xfail that fails the day the gap closes.
"""

from __future__ import annotations

import re

import pytest

from . import contract, ledger
from .loaders import LOADERS, items_for

CELLS = [(loader.name, item) for loader in LOADERS for item in items_for(loader.name)]


@pytest.mark.parametrize(("loader", "item"), CELLS, ids=[f"{l}:{i}" for l, i in CELLS])
def test_native_loader_cell(graded, loader, item):
    obs = graded.observation(loader)
    cell = contract.grade(loader, item, obs, graded.expect)
    if cell.kind == "na":
        ledger.skip_declared("na", f"{loader}:{item}", cell.reason)
    if cell.kind == "gap":
        pytest.xfail(f"known upstream gap: {cell.reason}")


@pytest.mark.parametrize("loader", sorted(ledger.NOISE_ALLOWLIST))
def test_every_noise_allowance_still_occurs(graded, loader):
    """An allowlisted warning that no longer appears is an exemption nobody needs."""
    channel = graded.observation(loader).get("channel") or []
    stale = [(pattern, reason) for pattern, reason in ledger.NOISE_ALLOWLIST[loader]
             if not any(re.search(pattern, line) for line in channel)]
    assert not stale, (
        f"{loader}: these N2 allowances matched nothing on a healthy run, so the warning "
        f"they excuse is gone. Delete them from ledger.NOISE_ALLOWLIST: {stale}")


def test_the_graded_project_comes_from_the_declared_producer(graded):
    """The suite says what wrote the folder it graded; today that is the oracle."""
    from . import oracle, source

    assert source.PRODUCER == "oracle"
    written = {p.relative_to(graded.project).as_posix(): p.read_bytes()
               for p in graded.project.rglob("*") if p.is_file()}
    assert written == oracle.build().files, "the graded project is not the oracle's bytes"


def test_no_skill_folder_holds_a_second_skill_md(graded):
    contract.assert_one_skill_md_per_folder(graded.expect)
