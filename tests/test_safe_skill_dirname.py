"""Coverage for the install-path traversal guard.

`grep -r "traversal\\|UnsafePath" tests/` returned nothing before this file — the
guard shipped untested, which is part of why `skills sync` was found in 2026-08-04
writing a server-supplied name to disk without calling it at all.

Threat model: the backend supplies the skill name. A malicious or compromised
registry entry must not be able to write into ~/.bashrc, .git/hooks, cron.d, or an
SSH config on the installing machine.
"""

import pytest

from decimalai.disk_export import SkillExportUnsafePathError, _safe_skill_dirname


@pytest.mark.parametrize(
    "name",
    [
        "code-review",
        "playwright-cli",
        "skill.with.dots",
        "Mixed_Case-Name",
        "a",
    ],
)
def test_accepts_a_plain_single_component(name):
    assert _safe_skill_dirname(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "../evil",
        "../../etc/passwd",
        "nested/child",
        "microsoft/azure-kusto",  # a namespaced registry name is NOT a dirname
        "back\\slash",
        "..\\windows",
        "/absolute",
        "/etc/passwd",
        "has\x00null",
    ],
)
def test_rejects_traversal_and_separators(name):
    with pytest.raises(SkillExportUnsafePathError):
        _safe_skill_dirname(name)


def test_error_names_the_offending_value_and_kind():
    with pytest.raises(SkillExportUnsafePathError) as ei:
        _safe_skill_dirname("../evil")
    msg = str(ei.value)
    assert "../evil" in msg
    assert "skill name" in msg
    # Subclasses ValueError so existing `except ValueError` handlers still catch it.
    assert isinstance(ei.value, ValueError)


def test_refuses_rather_than_sanitizing():
    """The guard must never quietly rewrite a hostile name into a safe-looking one —
    a silently sanitized install is an install the user cannot audit."""
    with pytest.raises(SkillExportUnsafePathError):
        _safe_skill_dirname("../../.ssh/authorized_keys")


# ── the split that makes `decimalai skills pull` work again ─────────

def test_export_uses_url_slug_for_the_directory_and_name_for_frontmatter(tmp_path):
    """decimalai skills pull <namespaced-skill> used to fail outright.

    A registry name may be `owner/skill`. That is the skill's IDENTITY and must
    stay in the SKILL.md frontmatter, but it cannot be a single path component —
    so `_safe_skill_dirname` refused it and the CLI printed "unsafe skill name
    ... The skill source may be malicious." for 72.6% of the registry.

    The guard was never wrong; it was fed the wrong field. `url_slug` is the
    slash-free identifier the platform serves alongside the name. These pin the
    SPLIT: directory from url_slug, declared name from name.
    """
    from decimalai.disk_export import export_skill_to_disk

    result = export_skill_to_disk(
        {
            "name": "microsoft/azure-kusto",
            "url_slug": "microsoft-azure-kusto",
            "description": "Query Azure Data Explorer.",
            "body_markdown": "# Azure Kusto\n\nQuery things.",
        },
        agents=["claude-code"],
        project_root=str(tmp_path),
    )

    assert result["skill_dirname"] == "microsoft-azure-kusto"
    assert result["skill_name"] == "microsoft/azure-kusto"

    written = result["written_paths"][0]
    # The directory is the slug — one level, where agents actually look.
    assert "microsoft-azure-kusto" in written
    assert "microsoft/azure-kusto" not in written

    # …and the file still DECLARES the real, namespaced identity.
    content = open(written, encoding="utf-8").read()
    assert "name: microsoft/azure-kusto" in content


def test_export_falls_back_to_name_when_url_slug_is_absent(tmp_path):
    """Older backends don't send url_slug; behaviour must be unchanged there."""
    from decimalai.disk_export import export_skill_to_disk

    result = export_skill_to_disk(
        {
            "name": "commit-conventions",
            "description": "d",
            "body_markdown": "# body",
        },
        agents=["claude-code"],
        project_root=str(tmp_path),
    )
    assert result["skill_dirname"] == "commit-conventions"
    assert result["skill_name"] == "commit-conventions"


def test_a_traversing_url_slug_is_still_refused(tmp_path):
    """The guard must still fire — url_slug is server-supplied too."""
    import pytest as _pytest

    from decimalai.disk_export import SkillExportUnsafePathError, export_skill_to_disk

    with _pytest.raises(SkillExportUnsafePathError):
        export_skill_to_disk(
            {
                "name": "innocent",
                "url_slug": "../../../etc/evil",
                "description": "d",
                "body_markdown": "# b",
            },
            agents=["claude-code"],
            project_root=str(tmp_path),
        )
