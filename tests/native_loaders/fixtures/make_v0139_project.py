"""Rewrite ``v0139_project`` from SDK 0.13.9's own exporter (provenance, not a test).

    python -m tests.native_loaders.fixtures.make_v0139_project      # from the checkout root

Loads ``decimalai/disk_export.py`` as tagged ``v0.13.9`` (``git show``), and hands its
``export_skills_to_disk`` the oracle's three skills with every field the registry
knows, ``url_slug`` included: the most favourable route 0.13.9 has (its CLI's
``/body`` route cannot export a namespaced skill at all). ``compatibility`` is the
registry's string, as stored; 0.13.9 writes it as a list of characters.

The output is stored with ``.agents`` and ``.decimal`` renamed to ``agents`` and
``decimal`` (this repository's ``.gitignore`` excludes the dot names). A rerun differs
from the committed copy only in the lock's timestamps.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .. import oracle

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEST = HERE / "v0139_project"


def exporter():
    source = subprocess.run(["git", "show", "v0.13.9:decimalai/disk_export.py"], cwd=REPO,
                            capture_output=True, text=True, check=True).stdout
    tmp = Path(tempfile.mkdtemp(prefix="disk-export-0139-")) / "disk_export_0139.py"
    tmp.write_text(source)
    spec = importlib.util.spec_from_file_location("disk_export_0139", tmp)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registry_dict(row: oracle.SkillRow) -> dict:
    return {
        "id": row.skill_id, "name": row.name, "url_slug": oracle.install_name_base(row.name),
        "description": row.description, "body_markdown": row.stored_body.strip(),
        "license": row.license, "compatibility": row.compatibility,
        "allowed_tools": list(row.allowed_tools), "invocation_mode": row.invocation_mode,
        "category": row.category, "skill_scope": "public" if row.public else "private",
        "version": row.version, "version_id": row.version_id,
        "latest_version": {"version_number": row.version},
        "attachments": [{"file_path": p, "directory": p.split("/")[0], "content_text": text,
                         "is_executable": ex} for p, text, ex in row.files],
    }


def main() -> int:
    out = Path(tempfile.mkdtemp(prefix="v0139-project-")).resolve()
    exporter().export_skills_to_disk([registry_dict(r) for r in oracle.ROWS],
                                     agents=["universal"], project_root=str(out))
    shutil.rmtree(DEST, ignore_errors=True)
    DEST.mkdir()
    shutil.copytree(out / ".agents", DEST / "agents")
    shutil.copytree(out / ".decimal", DEST / "decimal")
    print(f"wrote {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
