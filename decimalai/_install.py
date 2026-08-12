"""Project-local install identity for per-checkout skill sync divergence.

The platform tracks skill drift per *install* — one checkout / CI workspace.
This module persists a stable, project-local ``install_id`` in
``.decimal/install.json`` (next to the skills lockfile) so the SDK and CLI can
stamp ``POST /skills/sync`` and ``POST /skills/installs/report`` with it. The
backend then knows which install reconciled which version and can flag local /
remote drift.

The id is per-checkout, NOT per-user, so it must not be committed: creating it
also drops a ``.decimal/.gitignore`` that excludes ``install.json``.
"""

from __future__ import annotations

import json
import os
import socket
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Colocated with disk_export.LOCKFILE_NAME (".decimal/skills.lock") so all
# project-local SDK state lives under one dotdir users gitignore once.
STATE_DIR = ".decimal"
INSTALL_FILE = "install.json"


def find_project_root(start: Optional[str] = None) -> str:
    """Best-effort project root for install identity.

    Prefers an existing ``.decimal/`` marker (so a run from a subdirectory
    shares the repo's install_id with one at the root), then the nearest
    ``.git`` ancestor, else ``start`` / the current working directory.
    """
    base = Path(start or os.getcwd()).resolve()
    candidates = [base, *base.parents]
    for d in candidates:
        if (d / STATE_DIR).is_dir():
            return str(d)
    for d in candidates:
        if (d / ".git").exists():
            return str(d)
    return str(base)


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _ensure_gitignore(state_dir: Path) -> None:
    """Keep install.json out of version control (per-checkout identity)."""
    gi = state_dir / ".gitignore"
    if gi.exists():
        return
    try:
        gi.write_text(f"{INSTALL_FILE}\n", encoding="utf-8")
    except OSError:
        pass


def _default_label() -> Optional[str]:
    try:
        host = socket.gethostname().strip()
    except OSError:
        return None
    return host or None


def get_install_identity(
    project_root: Optional[str] = None,
    *,
    label: Optional[str] = None,
    create: bool = True,
) -> Dict[str, Any]:
    """Load (or create) this checkout's install identity.

    Returns a dict with ``install_id``, ``install_label`` and ``created_at``.
    A pre-existing file wins (the id is stable across runs); a corrupt file is
    regenerated. With ``create=False`` a missing file yields an in-memory
    identity that is NOT written — used by ``--dry-run`` so a preview never
    leaves a footprint.
    """
    root = Path(project_root) if project_root else Path(find_project_root())
    state_dir = root / STATE_DIR
    path = state_dir / INSTALL_FILE

    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if isinstance(data, dict) and data.get("install_id"):
            if label and not data.get("install_label"):
                data["install_label"] = label
                if create:
                    try:
                        _atomic_write_json(path, data)
                    except OSError:
                        pass
            return data

    data: Dict[str, Any] = {
        "install_id": str(uuid.uuid4()),
        "install_label": label or _default_label(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if create:
        try:
            _atomic_write_json(path, data)
            _ensure_gitignore(state_dir)
        except OSError:
            pass
    return data
