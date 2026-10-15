"""``decimalai skills verify`` rejects what the native loaders reject: measured, not assumed.

``ledger_matrix.json`` records what every loader does with a folder carrying one defect
(``matrix.py``). This file holds verify to that record, on the producer's project with
the defect applied (the same edit the matrix makes to the oracle's bytes, which the
producer's equal):

* every defect is rejected in a managed project: exit 1, and an error names the rule the
  defect breaks (``RULE``), not only the hash mismatch any edit causes;
* a defect some loader does not read cleanly (any cell but ``OK``: it raised, dropped,
  renamed or warned) is an error even in a folder verify does not manage
  (``verify PATH``) at default strictness, so its rule is a ``loader`` rule: a
  hand-authored skill a loader would mishandle fails verify, it does not warn;
* a defect every loader reads cleanly may be a warning there; ``--strict`` makes every
  defect an error;
* where verify is stricter than every loader measured, ``STRICTER_THAN_THE_LOADERS`` says
  why, and the entry cleans itself the day a loader is measured mishandling it;
* the clean project passes in both modes, and every defect of the matrix has its rule
  here, so a defect cannot join the ledger without verify being held to it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict

import pytest

from decimalai import skillpkg
from decimalai.skillpkg.verifier import RULES

from . import matrix, oracle, producer

#: The rule each single defect breaks.
RULE: Dict[str, str] = {
    "allowed-tools as a YAML list": "F6a",
    "compatibility as a YAML list": "F5",
    "nested map under metadata": "F8",
    "unquoted number in metadata": "F8",
    "namespaced name": "F3",
    "folder name differs from name": "F3",
    "skill folder is a symlink": "V1",
    "unknown top-level key": "F2",
    "uppercase name": "F3",
    "name over 64 characters": "F3",
    "double hyphen in name": "F3",
    "symlinked resource file": "V1",
    "a second SKILL.md inside the folder": "F15",
}

#: Defects every loader in the matrix reads cleanly, that verify still makes an error in
#: a folder it does not manage, and why.
STRICTER_THAN_THE_LOADERS: Dict[str, str] = {
    "a second SKILL.md inside the folder": (
        "no loader measured lists references/SKILL.md as a skill, but one that walks a "
        "skill folder deeper would load it as another skill, and the registry never "
        "writes one (it is listed unavailable): a folder holding one was edited by hand"),
}

DEFECTS = [d for d in matrix.DEFECTS if d != "baseline"]


def _cells() -> Dict[str, Dict[str, str]]:
    return json.loads(matrix.MATRIX_PATH.read_text())["cells"]


def _mishandled(defect: str) -> Dict[str, str]:
    """The loaders that do not read a folder with ``defect`` cleanly, and what they do."""
    return {loader: cell for loader, cell in sorted(_cells()[defect].items()) if cell != "OK"}


@pytest.fixture(scope="module")
def made() -> producer.Production:
    return producer.template()


def _project(made: producer.Production, tmp_path: Path, defect: str) -> Path:
    project = tmp_path / "project"
    shutil.copytree(made.project, project, symlinks=True)
    mutate = matrix.DEFECTS[defect]
    if mutate is not None:
        mutate(project)
    return project


def _errors(result) -> Dict[str, str]:
    return {f.rule: str(f) for f in result.problems if f.is_error}


def test_every_defect_of_the_matrix_has_its_rule():
    assert set(RULE) == set(DEFECTS), (
        f"defects without a rule here: {sorted(set(DEFECTS) - set(RULE))}; rules for no "
        f"defect: {sorted(set(RULE) - set(DEFECTS))}")
    assert set(_cells()) == set(matrix.DEFECTS)
    assert set(RULE.values()) <= set(RULES)


def test_the_clean_project_passes_in_both_modes(made, tmp_path):
    project = _project(made, tmp_path, "baseline")
    for result in (skillpkg.verify(directory=project),
                   skillpkg.verify(str(project / oracle.ROOT), strict=True)):
        assert int(result.exit_code) == 0 and not result.problems, [str(f) for f in result.problems]


@pytest.mark.parametrize("defect", DEFECTS)
def test_verify_rejects_the_defect_in_a_managed_project(made, tmp_path, defect):
    result = skillpkg.verify(directory=_project(made, tmp_path, defect))
    errors = _errors(result)
    assert int(result.exit_code) == 1 and RULE[defect] in errors, (
        f"{defect!r}: verify's errors are {sorted(errors)}, not {RULE[defect]}: "
        f"{[str(f) for f in result.problems]}")


@pytest.mark.parametrize("defect", DEFECTS)
def test_what_a_loader_mishandles_is_an_error_in_any_folder(made, tmp_path, defect):
    """``verify PATH`` grades by the rules for folders it does not manage."""
    project = _project(made, tmp_path, defect)
    default = skillpkg.verify(str(project / oracle.ROOT))
    strict = skillpkg.verify(str(project / oracle.ROOT), strict=True)
    rule = RULE[defect]
    assert rule in {f.rule for f in default.problems}, (
        f"{defect!r}: verify PATH does not report {rule} at all: "
        f"{[str(f) for f in default.problems]}")
    assert rule in _errors(strict) and int(strict.exit_code) == 1, (
        f"{defect!r}: --strict does not make {rule} an error")
    mishandled = _mishandled(defect)
    if mishandled:
        assert rule in _errors(default) and RULES[rule][0] == "loader", (
            f"{defect!r}: {mishandled} per ledger_matrix.json, but verify PATH reports "
            f"{rule} ({RULES[rule][0]} scope) only as a warning: a hand-authored skill "
            "these loaders mishandle would pass verify")


@pytest.mark.parametrize("defect", sorted(STRICTER_THAN_THE_LOADERS))
def test_each_stricter_rule_is_still_stricter_than_every_loader(made, tmp_path, defect):
    """Self-cleaning: once a loader is measured mishandling the defect, the measurement
    justifies the rule and the entry goes."""
    assert not _mishandled(defect), (
        f"{defect!r}: ledger_matrix.json now records {_mishandled(defect)}; delete it from "
        "STRICTER_THAN_THE_LOADERS (the measurement holds verify to it now)")
    project = _project(made, tmp_path, defect)
    assert RULE[defect] in _errors(skillpkg.verify(str(project / oracle.ROOT)))


def test_every_rule_verify_makes_an_error_where_no_loader_objects_is_declared(made, tmp_path):
    for defect in DEFECTS:
        if _mishandled(defect):
            continue
        project = _project(made, tmp_path / defect.replace(" ", "-"), defect)
        if RULE[defect] in _errors(skillpkg.verify(str(project / oracle.ROOT))):
            assert defect in STRICTER_THAN_THE_LOADERS, (
                f"{defect!r}: every loader reads it cleanly, yet verify PATH makes it an "
                "error; say why in STRICTER_THAN_THE_LOADERS, or make its rule an sdk rule")
