"""Copy immutable body witnesses without consulting mutable router state."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
from uuid import UUID


def copy_delivered_versions(
    versions: Any, *, names: Optional[Iterable[str]] = None,
) -> List[Dict[str, str]]:
    """Keep served name/hash records and their optional atomic UUID pair.

    Older routers carry only a hash. Missing or malformed UUID metadata does
    not invent an identity, and dictionaries never share ownership with a
    cache, another invocation, or an observer.
    """
    if not isinstance(versions, (list, tuple)):
        return []
    allowed = set(names) if names is not None else None
    copied: List[Dict[str, str]] = []
    for version in versions:
        if not isinstance(version, dict):
            continue
        name, digest = version.get("name"), version.get("hash")
        if not isinstance(name, str) or not name or not isinstance(digest, str) or not digest:
            continue
        if allowed is not None and name not in allowed:
            continue
        witness = {"name": name, "hash": digest}
        routing_id = version.get("routing_id")
        if isinstance(routing_id, str) and routing_id:
            witness["routing_id"] = routing_id
        skill_id, version_id = version.get("skill_id"), version.get("version_id")
        if isinstance(skill_id, str) and isinstance(version_id, str):
            try:
                canonical = str(UUID(skill_id)) == skill_id and str(UUID(version_id)) == version_id
            except ValueError:
                pass
            else:
                if canonical:
                    witness.update(skill_id=skill_id, version_id=version_id)
        if witness not in copied:
            copied.append(witness)
    return copied


def merge_delivered_versions(existing: Any, incoming: Any) -> List[Dict[str, str]]:
    """Accumulate every distinct witnessed body version, retaining copies."""
    return copy_delivered_versions([
        *copy_delivered_versions(existing), *copy_delivered_versions(incoming),
    ])
