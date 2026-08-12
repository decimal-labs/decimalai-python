"""Lock in: Tier-2 fuzzy body matching works for auto-discovered skills.

Deep-audit finding (sdk-core): detect_skill_activations Tier-2 reads
``skill.get('body') or skill.get('body_markdown')`` and only fuzzy-matches
when non-empty — but parse_skill_md / discover_skills descriptors carried
NO body, so Tier-2 could never fire for the SDK's own auto-discovery.
Only Tier-1 name matching was ever active.

The fix retains the markdown body (already read to compute the hash) in
the descriptor, so Tier-2 can fuzzy-match on body content.
"""

import textwrap
from pathlib import Path

import pytest

from decimalai.skills import detect_skill_activations, parse_skill_md


_SKILL_MD = textwrap.dedent(
    """\
    ---
    name: codebase-conventions
    description: Project-specific code conventions for this repository.
    ---

    Always wrap currency amounts using the format_money helper function.
    Never expose raw decimal values directly in any user-facing response.
    Database identifiers must be lowercase snake_case with a tenant prefix.
    """
)


def _write_skill(tmp_path: Path) -> str:
    skill_dir = tmp_path / "codebase-conventions"
    skill_dir.mkdir()
    md = skill_dir / "SKILL.md"
    md.write_text(_SKILL_MD, encoding="utf-8")
    return str(md)


def test_parse_skill_md_retains_body(tmp_path):
    """The descriptor must include the markdown body so Tier-2 can use it."""
    descriptor = parse_skill_md(_write_skill(tmp_path))
    assert descriptor is not None
    assert descriptor.get("body"), "descriptor must retain the skill body"
    assert "format_money helper" in descriptor["body"]


def test_tier2_fuzzy_activation_fires_for_discovered_skill(tmp_path):
    """A prompt that contains the skill's body content but NOT a name header
    must still activate the skill via Tier-2 fuzzy matching.
    """
    descriptor = parse_skill_md(_write_skill(tmp_path))
    registry = [descriptor]

    # Prompt embeds the skill BODY lines verbatim but never mentions the
    # skill NAME ("codebase-conventions") or a "## Skill:" header — so only
    # Tier-2 can match it.
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant.\n"
                "Always wrap currency amounts using the format_money helper function.\n"
                "Never expose raw decimal values directly in any user-facing response.\n"
                "Database identifiers must be lowercase snake_case with a tenant prefix.\n"
            ),
        }
    ]

    activated = detect_skill_activations(prompt, registry)
    assert "codebase-conventions" in activated, (
        "Tier-2 fuzzy body matching must activate the auto-discovered skill; "
        "pre-fix the descriptor had no body so Tier-2 was dead code."
    )


def test_tier2_disabled_when_fuzzy_match_off(tmp_path):
    """Sanity: with fuzzy_match=False, body-only prompts do NOT activate
    (only Tier-1 name matching, which this prompt avoids)."""
    descriptor = parse_skill_md(_write_skill(tmp_path))
    prompt = [
        {
            "role": "system",
            "content": (
                "Always wrap currency amounts using the format_money helper function.\n"
                "Never expose raw decimal values directly in any user-facing response.\n"
                "Database identifiers must be lowercase snake_case with a tenant prefix.\n"
            ),
        }
    ]
    activated = detect_skill_activations(prompt, [descriptor], fuzzy_match=False)
    assert "codebase-conventions" not in activated
