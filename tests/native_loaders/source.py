"""Where the folders this suite grades come from.

Two projects are graded on every run:

* the **graded project**: what the eight loaders must read cleanly. It is written by
  the real CLI (``producer.py``): ``decimalai skills add`` for each of the three
  skills, then ``decimalai skills install --frozen`` into a checkout of the manifest
  and the lock alone, as subprocesses of this interpreter's ``decimalai`` script,
  against the hermetic registry in ``tests/conformance/probe.py``. The commands run
  once per process; every graded project is a copy of that output (bytes and modes).
  The oracle (``oracle.py``), an independent rendering of the on-disk contract, is
  the byte-for-byte reference that output is compared with.
  ``DECIMAL_NATIVE_LOADER_PRODUCER=oracle`` grades the oracle's own folder instead,
  which tells a producer regression from a loader one.
* the **negative control**: a frozen copy of what SDK 0.13.9's exporter writes for
  the same three skills (``fixtures/v0139_project``). The contract must FAIL on it,
  in the cells ``ledger.NEGATIVE_CONTROL_PREDICTED`` names, for as long as this
  suite exists. If a contract item is weakened, that test turns red.

The fixture is stored with its two dot-directories renamed (``agents/``,
``decimal/``) because this repository's ``.gitignore`` excludes ``.agents/`` and
``.decimal/`` at every depth; ``negative_control_project`` restores the names when
it copies the fixture out, and checks nothing was lost on the way into git.

``skills_dir`` stands in for ``decimalai.skills_dir()`` in the loader modules: the
real one refuses a version 1 lock, and the negative control is one, so a loader there
would never get as far as reading the folders it is graded on. On the graded project
the two agree (``test_producer.py``). Every loader module starts from it, exactly as
each documented loader snippet starts from ``decimalai.skills_dir()``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from . import oracle

HERE = Path(__file__).resolve().parent
NEGATIVE_CONTROL = HERE / "fixtures" / "v0139_project"
#: Stored name -> the name 0.13.9 wrote (see the module docstring).
_RENAMED = {"agents": ".agents", "decimal": ".decimal"}

#: What may write the graded project: the real CLI (``producer.py``), or the oracle.
PRODUCERS = ("cli", "oracle")
PRODUCER_ENV = "DECIMAL_NATIVE_LOADER_PRODUCER"


def _producer() -> str:
    chosen = (os.environ.get(PRODUCER_ENV) or "cli").strip()
    if chosen not in PRODUCERS:
        raise ValueError(f"{PRODUCER_ENV}={chosen!r}: one of {', '.join(PRODUCERS)}")
    return chosen


#: What ``graded_project`` writes on this run.
PRODUCER = _producer()


def graded_project(dest: Path) -> Path:
    """Write the project the eight loaders must read cleanly into ``dest``."""
    dest = Path(dest).resolve()
    if PRODUCER == "oracle":
        oracle.write(dest)
        return dest
    from . import producer

    return producer.produce_copy(dest)


def negative_control_files() -> list:
    """The fixture's files, as 0.13.9 named them (project-relative)."""
    out = []
    for path in sorted(NEGATIVE_CONTROL.rglob("*")):
        if path.is_file():
            parts = path.relative_to(NEGATIVE_CONTROL).parts
            out.append("/".join((_RENAMED.get(parts[0], parts[0]),) + parts[1:]))
    return out


def negative_control_project(dest: Path) -> Path:
    """Copy the frozen 0.13.9 export into ``dest``, restoring ``.agents`` and ``.decimal``."""
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    for stored, real in _RENAMED.items():
        src = NEGATIVE_CONTROL / stored
        if not src.is_dir():
            raise FileNotFoundError(
                f"{src} is missing: the negative-control fixture lost a directory on its "
                "way into git (check `git ls-files tests/native_loaders/fixtures`)")
        shutil.copytree(src, dest / real, symlinks=True)
    return dest


class SkillsNotInstalledError(RuntimeError):
    pass


def _project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".decimal" / "skills.lock").is_file() or (
                candidate / ".decimal" / "skills.toml").is_file():
            return candidate
    raise SkillsNotInstalledError(
        f"no .decimal/skills.lock above {start}: run `decimalai skills install`")


def skills_dir(directory: Optional[os.PathLike] = None, *,
               relative_to: Optional[os.PathLike] = None) -> Path:
    """The managed root, ``<project>/.agents/skills``: absolute and symlink-resolved.

    With ``relative_to`` it is returned relative to that directory instead (the
    openai-agents sandbox lists nothing for a source outside its cwd, and the
    pydantic-ai harness reads paths relative to its workspace); a root outside it
    raises.
    """
    project = _project_root(Path(directory or os.getcwd()).resolve())
    root = project / oracle.ROOT
    if not root.is_dir():
        raise SkillsNotInstalledError(f"{root} does not exist: run `decimalai skills install`")
    root = root.resolve()
    if relative_to is None:
        return root
    base = Path(relative_to).resolve()
    try:
        return root.relative_to(base)
    except ValueError:
        raise ValueError(f"the skills root {root} is not under {base}") from None
