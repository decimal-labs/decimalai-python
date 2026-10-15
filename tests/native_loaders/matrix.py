"""The ledger matrix: every loader against every single defect, through its own discovery.

    python -m tests.native_loaders.matrix                  # print this environment's table
    python -m tests.native_loaders.matrix --check-ledger   # diff it against ledger_matrix.json
    python -m tests.native_loaders.matrix --update-ledger  # re-record this environment's cells

Run from the checkout root, once per test environment (each grades only its own
loaders, ``LOADER_ENVS``). One defect is applied to ``acme-refund-policy`` in a fresh
oracle project, the other two skills stay clean, and each loader's ``discover()``
runs in its own child. Cells:

``OK``        listed under the install name, nothing on the loader's channel
``OK (+w)``   listed, but the loader's channel said something
``LOUD``      the loader raised
``WARNED``    dropped, and the loader's own channel said so
``SILENT``    dropped, with no signal at all
``RENAMED``   dropped from its install name but listed under another (``(+w)``: and warned)
``EXTRA``     listed, plus a skill the lock does not hold (``(+w)``: and warned)
``CRASH``     the child returned nothing (a finding in itself)

Not per commit: the cells move with upstream releases (the floors take the newest
release), so a per-commit gate would turn red on someone else's release. The release
gate and the floating lane run it; a changed cell is a finding, a loader that got
stricter or looser, and is re-recorded deliberately with ``--update-ledger``. The
matrix is also the bar ``decimalai skills verify`` must clear: it must reject every
defect that any column does not mark ``OK``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import oracle
from .loaders import LOADERS, LOADERS_BY_NAME
from .runner import LoaderProcessError, available, run_many

MATRIX_PATH = Path(__file__).resolve().parent / "ledger_matrix.json"

T = oracle.SCENARIO.target
ROOT = oracle.ROOT
CLASSES = ("OK", "OK (+w)", "LOUD", "WARNED", "SILENT", "RENAMED", "RENAMED (+w)",
           "EXTRA", "EXTRA (+w)", "CRASH")


class MutationError(RuntimeError):
    """A defect no longer applies to the oracle's bytes (the oracle changed)."""


def _edit(md: Path, old: str, new: str) -> None:
    text = md.read_text()
    if old not in text:
        raise MutationError(f"{md}: {old!r} not found; the oracle changed under this defect")
    md.write_text(text.replace(old, new, 1))


def _rename(project: Path, new: str) -> Dict[str, str]:
    """Rename the target's folder, ``name`` and lock key together."""
    skills = project / ROOT
    (skills / T).rename(skills / new)
    _edit(skills / new / "SKILL.md", f"name: {T}\n", f"name: {new}\n")
    return {T: new}


def _md(project: Path) -> Path:
    return project / ROOT / T / "SKILL.md"


Mutation = Callable[[Path], Optional[Dict[str, str]]]

DEFECTS: Dict[str, Optional[Mutation]] = {
    "baseline": None,
    "allowed-tools as a YAML list": lambda p: _edit(
        _md(p), 'allowed-tools: "Read Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/compute_refund.py *)"\n',
        "allowed-tools:\n  - Read\n  - Bash\n"),
    "compatibility as a YAML list": lambda p: _edit(
        _md(p), 'compatibility: "Requires python3"\n', 'compatibility:\n  - "python>=3.10"\n'),
    "nested map under metadata": lambda p: _edit(
        _md(p), "metadata:\n", 'metadata:\n  decimalai:\n    version: "3"\n'),
    "unquoted number in metadata": lambda p: _edit(
        _md(p), 'decimalai-version: "3"', "decimalai-version: 3"),
    "namespaced name": lambda p: _edit(_md(p), f"name: {T}\n", "name: acme/refund-policy\n"),
    "folder name differs from name": lambda p: (p / ROOT / T).rename(p / ROOT / f"{T}-v3"),
    "skill folder is a symlink": lambda p: _symlink_folder(p),
    "unknown top-level key": lambda p: _edit(_md(p), "metadata:\n", 'decimalai-version-id: "x"\nmetadata:\n'),
    "uppercase name": lambda p: _edit(_md(p), f"name: {T}\n", "name: Acme-Refund-Policy\n"),
    "name over 64 characters": lambda p: _rename(p, "acme-refund-policy-" + "x" * 50),
    "double hyphen in name": lambda p: _rename(p, "acme--refund-policy"),
    "symlinked resource file": lambda p: _symlink_resource(p),
    "a second SKILL.md inside the folder": lambda p: _nested_skill_md(p),
}


def _symlink_folder(project: Path) -> None:
    store = project / ".decimal" / "store" / T
    store.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(project / ROOT / T), str(store))
    os.symlink(store, project / ROOT / T, target_is_directory=True)


def _symlink_resource(project: Path) -> None:
    res = project / ROOT / T / oracle.SCENARIO.resource
    store = project / ".decimal" / "store" / "fees.md"
    store.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(res), str(store))
    os.symlink(store, res)


def _nested_skill_md(project: Path) -> None:
    # An imported skill can carry a SKILL.md of its own among its attachments; a
    # loader that walks the folder may then list it as one more skill.
    # An attachment path is <dir>/<file>, so this is where an imported one would land.
    nested = project / ROOT / T / "references" / "SKILL.md"
    nested.write_text('---\nname: references\ndescription: "A skill inside a skill."\n---\nbody\n')


def project_for(defect: str, dest: Path) -> Tuple[Path, Dict[str, dict], str]:
    """A fresh oracle project with ``defect`` applied: the project, the lock entries it is
    graded against, and the mutated skill's install name."""
    dest = Path(dest).resolve()
    lock = oracle.write(dest)
    expect = dict(lock["skills"])
    mutate = DEFECTS[defect]
    result = mutate(dest) if mutate else None
    # Only a dict means the lock key moved with the folder; anything else left the lock alone.
    renamed = result if isinstance(result, dict) else {}
    for old, new in renamed.items():
        expect[new] = expect.pop(old)
    return dest, expect, renamed.get(T, T)


def classify(result: object, expect: Dict[str, dict], target: str) -> str:
    if isinstance(result, LoaderProcessError) or not isinstance(result, dict):
        return "CRASH"
    if result.get("raised"):
        return "LOUD"
    names = set(result.get("listed") or [])
    want = set(expect)
    warned = " (+w)" if result.get("channel") else ""
    extra = names - want
    if target in names:
        return ("EXTRA" + warned) if extra else ("OK" + warned)
    if extra:
        return "RENAMED" + warned
    return "WARNED" if warned else "SILENT"


def graded_here() -> List[str]:
    from tests.conformance.environments import loader_graded_elsewhere

    return [l.name for l in LOADERS if loader_graded_elsewhere(l.name) is None and not available(l)]


def run(defects: Sequence[str], loaders: Sequence[str], workdir: Path) -> Dict[str, Dict[str, str]]:
    plan, cells = [], []
    for defect in defects:
        project, expect, target = project_for(defect, workdir / _slug(defect) / "project")
        for loader in loaders:
            plan.append((defect, loader, expect, target))
            cells.append((loader, "discover", project, expect, oracle.PLUGIN_NAME))
    results = run_many(cells, workdir=workdir)
    grid: Dict[str, Dict[str, str]] = {}
    for (defect, loader, expect, target), result in zip(plan, results):
        grid.setdefault(defect, {})[loader] = classify(result, expect, target)
    return grid


def _slug(defect: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in defect)[:40]


def versions(loaders: Sequence[str]) -> Dict[str, Dict[str, Optional[str]]]:
    out = {}
    for name in loaders:
        dists = {}
        for d in LOADERS_BY_NAME[name].distributions:
            try:
                dists[d] = metadata.version(d)
            except metadata.PackageNotFoundError:
                dists[d] = None
        out[name] = dists
    return out


def diff(grid: Dict[str, Dict[str, str]], recorded: Dict[str, dict]) -> List[str]:
    cells = recorded.get("cells", {})
    out = []
    for defect in DEFECTS:
        for loader, cell in sorted(grid.get(defect, {}).items()):
            was = cells.get(defect, {}).get(loader)
            if was != cell:
                out.append(f"  {defect} x {loader}: ledger {was!r}, measured {cell!r}")
    return out


def table(grid: Dict[str, Dict[str, str]], loaders: Sequence[str]) -> str:
    lines = ["| defect | " + " | ".join(loaders) + " |", "|---|" + "---|" * len(loaders)]
    for defect in DEFECTS:
        lines.append(f"| {defect} | " + " | ".join(grid.get(defect, {}).get(l, "") for l in loaders) + " |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tests.native_loaders.matrix")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check-ledger", action="store_true")
    mode.add_argument("--update-ledger", action="store_true")
    ap.add_argument("--loaders", help="comma-separated subset of this environment's loaders")
    ap.add_argument("--defects", help="comma-separated subset of the defects")
    args = ap.parse_args(argv)

    from tests.conformance.environments import current_env

    here = graded_here()
    loaders = [l for l in (args.loaders.split(",") if args.loaders else here)]
    unknown = sorted(set(loaders) - set(here))
    if unknown:
        print(f"not graded in {current_env()!r}: {unknown} (graded here: {here})", file=sys.stderr)
        return 2
    defects = args.defects.split(",") if args.defects else list(DEFECTS)
    with tempfile.TemporaryDirectory(prefix="native-matrix-") as tmp:
        grid = run(defects, loaders, Path(tmp).resolve())
    print(f"environment {current_env()!r}, loaders {loaders}\n")
    print(table(grid, loaders))
    recorded = json.loads(MATRIX_PATH.read_text()) if MATRIX_PATH.is_file() else {}
    if args.update_ledger:
        cells = recorded.setdefault("cells", {})
        for defect, row in grid.items():
            cells.setdefault(defect, {}).update(row)
        recorded["cells"] = {d: dict(sorted(cells[d].items())) for d in DEFECTS if d in cells}
        recorded.setdefault("measured_at", {}).update(versions(loaders))
        recorded["measured_at"] = dict(sorted(recorded["measured_at"].items()))
        MATRIX_PATH.write_text(json.dumps(recorded, indent=2, sort_keys=False) + "\n")
        print(f"\nrecorded {len(defects) * len(loaders)} cells in {MATRIX_PATH.name}")
        return 0
    if args.check_ledger:
        changed = diff(grid, recorded)
        moved = [f"  {n}: ledger {recorded.get('measured_at', {}).get(n)}, installed {v}"
                 for n, v in versions(loaders).items()
                 if recorded.get("measured_at", {}).get(n) != v]
        if moved:
            print("\nversions differ from the ones the ledger was measured at:\n" + "\n".join(moved))
        if changed:
            print("\nLEDGER MISMATCH (a loader got stricter or looser):\n" + "\n".join(changed)
                  + "\nInvestigate, then re-record with --update-ledger.")
            return 1
        print(f"\nledger matches: {len(defects) * len(loaders)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
