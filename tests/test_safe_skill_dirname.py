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


# ── the folder, the frontmatter name and the registry name ─────────

SKILL_ID = "00000000-0000-4000-8000-000000000001"
VERSION_ID = "00000000-0000-4000-8000-000000000002"


@pytest.fixture(autouse=True)
def _fresh_once_state(monkeypatch):
    import decimalai.disk_export as disk_export

    monkeypatch.setattr(disk_export, "_warned_once", set())


def _export(tmp_path, **skill):
    from decimalai.disk_export import export_skill_to_disk

    skill = {"id": SKILL_ID, "version_id": VERSION_ID, "description": "d",
             "body_markdown": "# Example Tool\n\nQuery things.", **skill}
    return export_skill_to_disk(skill, agents=["claude-code"], project_root=str(tmp_path))


def _frontmatter(path):
    from skillevaluation import skillfmt

    with open(path, encoding="utf-8") as f:
        return skillfmt.parse_skill_md_text(f.read())[0]


def test_a_namespaced_skill_gets_its_install_name_and_keeps_its_identity(tmp_path):
    """A registry name may be `owner/skill` — that is the skill's IDENTITY, but not a
    single path component. The folder, and the `name:` it declares (loaders require the
    two to match), is the install name every door uses; the registry name is kept in
    the stamp, `metadata.decimalai-name`."""
    result = _export(tmp_path, name="acme/example-tool", url_slug="acme-example-tool")

    assert result["skill_dirname"] == "acme-example-tool"
    assert result["skill_name"] == "acme/example-tool"
    written = result["written_paths"][0]
    assert "acme-example-tool" in written
    assert "acme/example-tool" not in written
    fm = _frontmatter(written)
    assert fm["name"] == "acme-example-tool"
    assert fm["metadata"]["decimalai-name"] == "acme/example-tool"


def test_the_install_name_does_not_depend_on_url_slug(tmp_path):
    """Older backends don't send url_slug; the folder is the same either way."""
    with_slug = _export(tmp_path / "a", name="acme/example-tool", url_slug="acme-example-tool")
    without = _export(tmp_path / "b", name="acme/example-tool")
    assert with_slug["skill_dirname"] == without["skill_dirname"] == "acme-example-tool"


def test_a_plain_name_is_its_own_folder(tmp_path):
    result = _export(tmp_path, name="commit-conventions")
    assert result["skill_dirname"] == "commit-conventions"
    assert result["skill_name"] == "commit-conventions"


@pytest.mark.parametrize("field,value", [
    ("url_slug", "../../../etc/evil"),
    ("url_slug", "/etc/evil"),
    ("name", "../evil"),
    ("name", "acme/../../evil"),
    ("name", "/abs/evil"),
    ("name", "back\\slash"),
])
def test_a_traversing_name_or_url_slug_is_still_refused(tmp_path, field, value):
    """The guard still fires: both values are server-supplied, and a hostile one is
    refused, never quietly sanitized into a safe-looking folder."""
    from decimalai.disk_export import SkillExportUnsafePathError

    skill = {"name": "innocent", field: value}
    with pytest.raises(SkillExportUnsafePathError):
        _export(tmp_path, **skill)
    assert not (tmp_path / ".claude").exists()


@pytest.mark.parametrize("path", ["../escape.md", "/etc/passwd", "scripts/../../x.py"])
def test_a_traversing_attachment_path_is_refused(tmp_path, path):
    from decimalai.disk_export import SkillExportUnsafePathError, export_skill_to_disk

    with pytest.raises(SkillExportUnsafePathError):
        export_skill_to_disk(
            {"id": SKILL_ID, "version_id": VERSION_ID, "name": "pdf", "description": "d",
             "body_markdown": "b"},
            project_root=str(tmp_path),
            attachments=[{"file_path": path, "content_text": "x"}])
    assert not (tmp_path / ".agents").exists()
