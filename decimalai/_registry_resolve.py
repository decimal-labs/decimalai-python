"""Resolve a registry skill NAME to its record — exactly, never by ranked guess.

The public registry's ``q=`` parameter is a *semantic* search: it ranks the whole
corpus and essentially always returns something. Four call sites (CLI ``skills
pull``; ``SkillRouter.fork`` / ``.use`` / ``.preview``) used to search for the
caller's slug with ``limit=1`` and take ``items[0]`` — so a typo, a renamed
skill, or a retired one resolved silently to an *unrelated* skill:

    $ decimalai skills pull totally-bogus-skill-xyz123
      ✓ Pulled agent-skill-format v1        # not what was asked for, reported as success
    $ decimalai skills pull pdf-procesing   # one-character typo
      ✓ Pulled minimax-pdf v1

A SKILL.md is instructions the agent loads and follows, so silently substituting
a different one is both a correctness bug and a supply-chain one — install-time
typosquatting performed by our own tooling. Resolution here is therefore
exact-match only; a miss is an error that *names* what the search did rank, so
the caller can see the near-miss instead of unknowingly running it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

# Fetch a handful rather than 1: the exact match is not always rank 1 (semantic
# search may float a longer, more "relevant" name above the literal one), and the
# extras become the "did you mean" list on a miss.
RESOLVE_LIMIT = 10


def find_exact(items: Sequence[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    """Return the item whose registry name is exactly ``name``, else ``None``.

    Falls back to a *unique* case-insensitive match, because registry names are
    not all lowercase slugs (``Excel / XLSX``, ``StatsPAI_skill``, ``Capability
    Evolver``) and a caller typing the lowercase form means the obvious thing.
    An ambiguous case-fold (two names differing only in case) is treated as no
    match — guessing between them is the bug this module exists to prevent.
    """
    if not name:
        return None
    for item in items:
        if item.get("name") == name:
            return item
    folded = [
        item for item in items
        if str(item.get("name") or "").casefold() == name.casefold()
    ]
    return folded[0] if len(folded) == 1 else None


def near_misses(items: Sequence[Dict[str, Any]], limit: int = 5) -> List[str]:
    """The names the search *did* rank, for a 'did you mean' hint."""
    out: List[str] = []
    for item in items:
        nm = item.get("name")
        if nm and nm not in out:
            out.append(str(nm))
        if len(out) >= limit:
            break
    return out


def not_found_message(name: str, items: Sequence[Dict[str, Any]]) -> str:
    """Error text for a name that has no exact match in the registry."""
    msg = f"Skill '{name}' not found in the public registry."
    hints = near_misses(items)
    if hints:
        msg += " Closest matches: " + ", ".join(hints)
    return msg
