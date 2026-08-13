"""Coverage for the install-path traversal guard.

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
        "acme/example-tool",  # a namespaced registry name is NOT a dirname
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


# ── the split between the on-disk directory and the declared name ─────────

def test_export_uses_url_slug_for_the_directory_and_name_for_frontmatter(tmp_path):
    """A server-supplied skill name must never escape its own directory.

    INVARIANT: the on-disk directory comes from `url_slug` and the declared
    identity comes from `name`. A registry name may be `owner/skill` — that is
    the skill's IDENTITY and belongs in the SKILL.md frontmatter, but it is not
    a single path component, so it can never be used as a directory name.
    `url_slug` is the slash-free identifier the platform serves alongside the
    name, and it is the only field the export path may turn into a directory.
    """
    from decimalai.disk_export import export_skill_to_disk

    result = export_skill_to_disk(
        {
            "name": "acme/example-tool",
            "url_slug": "acme-example-tool",
            "description": "Query the example data store.",
            "body_markdown": "# Example Tool\n\nQuery things.",
        },
        agents=["claude-code"],
        project_root=str(tmp_path),
    )

    assert result["skill_dirname"] == "acme-example-tool"
    assert result["skill_name"] == "acme/example-tool"

    written = result["written_paths"][0]
    # The directory is the slug — one level, where agents actually look.
    assert "acme-example-tool" in written
    assert "acme/example-tool" not in written

    # …and the file still DECLARES the real, namespaced identity.
    content = open(written, encoding="utf-8").read()
    assert "name: acme/example-tool" in content


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
