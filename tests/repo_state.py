"""This repository's own project state: the files no test may change.

The SDK keeps a project's state in ``.decimal/`` (``install.json``, the skills manifest
and lock, a ``.gitignore``) and the skills it installs in ``.agents/``, and finds the
project by walking up from the working directory. This repository has a ``.decimal/``
of its own (its ``.gitignore`` is tracked), so a test that syncs, installs or runs an
engine call with the repository as its working directory takes the repository for its
project and writes there. A checkout that already holds ``.decimal/install.json`` hides
it: the identity is read, not written. A fresh clone does not.

``repo_state`` fingerprints both directories. ``tests/conftest.py`` compares one taken
when the session starts with one taken when it ends, and fails the run on a difference.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

SDK_ROOT = Path(__file__).resolve().parent.parent

#: Where the SDK writes a project's state, relative to the project.
GUARDED = (".decimal", ".agents")

State = Dict[str, Tuple]


def _entry(path: Path) -> Tuple:
    if path.is_symlink():
        return ("link", os.readlink(path))
    if path.is_dir():
        return ("dir",)
    data = path.read_bytes()
    return ("file", len(data), hashlib.sha256(data).hexdigest())


def repo_state(root: Path = SDK_ROOT) -> State:
    """Every directory, file and symlink under ``root``'s ``.decimal/`` and ``.agents/``.

    A file is its size and sha256, a symlink its target (never followed), a directory
    only that it exists. An absent directory has no entries.
    """
    root = Path(root)
    out: State = {}
    for top in GUARDED:
        base = root / top
        if not (base.is_symlink() or base.exists()):
            continue
        out[top] = _entry(base)
        if out[top][0] != "dir":
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            for name in dirnames + filenames:
                path = Path(dirpath) / name
                out[path.relative_to(root).as_posix()] = _entry(path)
    return dict(sorted(out.items()))


def repo_changes(before: Mapping[str, Tuple], after: Mapping[str, Tuple]) -> List[str]:
    """The paths added, removed or changed between two ``repo_state`` fingerprints."""
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
