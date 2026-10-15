"""The deprecated ``disk_export`` writers: what they write, and what they never overwrite.

``export_skill_to_disk`` never replaces a folder that already exists unless ``force=True``,
and then the old folder is moved to ``.decimal/backup/`` first, never deleted. A folder
``decimalai skills`` manages is never replaced at all. What it writes is the registry's
format: ``SKILL.md`` format 1, rendered by ``skillevaluation.skillfmt`` from the skill
dict, with ``allowed-tools`` and ``compatibility`` as strings (never a list, and a string
is never split into one entry per character).
"""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path

import pytest
import yaml
from skillevaluation import skillfmt

import decimalai.disk_export as disk_export
from decimalai.disk_export import (
    SkillExportFileExistsError,
    export_skill_to_disk,
)

SKILL_ID = "00000000-0000-4000-8000-000000000001"
VERSION_ID = "00000000-0000-4000-8000-000000000002"


@pytest.fixture(autouse=True)
def _fresh_once_state(monkeypatch):
    monkeypatch.setattr(disk_export, "_warned_once", set())


def _skill(name: str = "pdf", **over) -> dict:
    skill = {
        "id": SKILL_ID,
        "latest_version": {"id": VERSION_ID, "version_number": 2},
        "name": name,
        "description": "test skill",
        "body_markdown": "skill body content",
    }
    skill.update(over)
    return skill


def _frontmatter(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        fm, _ = skillfmt.parse_skill_md_text(f.read())
    return fm


def test_first_install_writes_skill_md_when_no_existing_file():
    with tempfile.TemporaryDirectory() as root:
        result = export_skill_to_disk(_skill(), agents=["universal"], project_root=root)
        assert len(result["written_paths"]) == 1
        assert os.path.exists(result["written_paths"][0])
        assert result["unmanaged"] == [os.path.dirname(result["written_paths"][0])]


def test_second_install_raises_without_force():
    with tempfile.TemporaryDirectory() as root:
        export_skill_to_disk(_skill(), agents=["universal"], project_root=root)
        with pytest.raises(SkillExportFileExistsError) as exc_info:
            export_skill_to_disk(_skill(), agents=["universal"], project_root=root)
        assert "Refusing to overwrite" in str(exc_info.value)
        assert "SKILL.md" in exc_info.value.path


def test_second_install_with_force_replaces_the_folder_and_keeps_the_old_one(tmp_path):
    first = export_skill_to_disk(_skill(), agents=["universal"], project_root=str(tmp_path))
    folder = Path(first["written_paths"][0]).parent
    (folder / "notes.txt").write_text("mine", encoding="utf-8")

    result = export_skill_to_disk(_skill(body_markdown="a new body"), agents=["universal"],
                                  project_root=str(tmp_path), force=True)

    assert "a new body" in (folder / "SKILL.md").read_text(encoding="utf-8")
    assert not (folder / "notes.txt").exists()          # the folder is replaced whole
    (backup,) = result["backups"]
    assert (Path(backup) / "notes.txt").read_text(encoding="utf-8") == "mine"


def test_an_existing_folder_without_skill_md_also_blocks(tmp_path):
    folder = tmp_path / ".agents" / "skills" / "pdf" / "references"
    folder.mkdir(parents=True)
    (folder / "data.md").write_text("first", encoding="utf-8")
    with pytest.raises(SkillExportFileExistsError) as exc_info:
        export_skill_to_disk(
            _skill(), agents=["universal"], project_root=str(tmp_path),
            attachments=[{"file_path": "references/data.md", "content_text": "second"}],
        )
    assert exc_info.value.path.endswith("pdf")
    assert (folder / "data.md").read_text(encoding="utf-8") == "first"


def test_user_authored_skill_md_is_protected():
    """A SKILL.md the user authored at the target path is preserved: the export raises
    before touching the file."""
    with tempfile.TemporaryDirectory() as root:
        user_dir = os.path.join(root, ".claude", "skills", "pdf")
        os.makedirs(user_dir, exist_ok=True)
        user_path = os.path.join(user_dir, "SKILL.md")
        with open(user_path, "w") as f:
            f.write("USER AUTHORED CONTENT — DO NOT LOSE")

        with pytest.raises(SkillExportFileExistsError):
            export_skill_to_disk(_skill(), agents=["claude-code"], project_root=root)

        with open(user_path) as f:
            assert "USER AUTHORED" in f.read()


def test_a_managed_folder_is_never_replaced_even_with_force(tmp_path):
    from decimalai.skillpkg import ops
    from decimalai.skillpkg.client import RegistryClient
    from tests.fakes.fake_registry import BASE_URL, OWNER_KEY, FakeRegistry

    registry = FakeRegistry()
    skill = registry.add_skill("pdf")
    client = RegistryClient(api_key=OWNER_KEY, base_url=BASE_URL,
                            transport=registry.transport(), sleep=lambda _s: None)
    assert ops.add("pdf", directory=tmp_path, client=client).ok
    before = (tmp_path / ".agents" / "skills" / "pdf" / "SKILL.md").read_bytes()

    with pytest.raises(SkillExportFileExistsError) as exc_info:
        export_skill_to_disk(_skill(id=skill.skill_id), project_root=str(tmp_path), force=True)

    assert "decimalai skills update" in str(exc_info.value)
    assert (tmp_path / ".agents" / "skills" / "pdf" / "SKILL.md").read_bytes() == before


def test_a_symlinked_skills_directory_is_refused(tmp_path):
    if os.name != "posix":
        pytest.skip("symlinks")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "skills").symlink_to(elsewhere)
    with pytest.raises(disk_export.SkillExportUnsafePathError):
        export_skill_to_disk(_skill(), agents=["claude-code"], project_root=str(tmp_path))
    assert list(elsewhere.iterdir()) == []


# ── what it writes: the registry's format ────────────────────────────


class TestFormatOne:
    def test_the_file_is_format_one_and_passes_every_loader_rule(self, tmp_path):
        result = export_skill_to_disk(
            _skill("acme/refund-policy", license="MIT", compatibility=["python3", "node18+"],
                   allowed_tools=["Read", "Bash", "WebFetch"], category="support"),
            project_root=str(tmp_path))
        path = result["written_paths"][0]
        text = Path(path).read_text(encoding="utf-8")

        assert result["skill_dirname"] == "acme-refund-policy"
        assert Path(path).parent.name == "acme-refund-policy"
        problems = skillfmt.validate_skill_md(text, folder_name="acme-refund-policy",
                                              managed=False)
        assert [p for p in problems if p.scope == "loader"] == [], problems
        fm = _frontmatter(path)
        assert fm["name"] == "acme-refund-policy"
        assert fm["metadata"]["decimalai-format"] == "1"
        assert fm["metadata"]["decimalai-name"] == "acme/refund-policy"
        assert fm["metadata"]["decimalai-skill-id"] == SKILL_ID
        assert fm["metadata"]["decimalai-version-id"] == VERSION_ID
        assert fm["metadata"]["decimalai-version"] == "2"
        assert fm["metadata"]["decimalai-category"] == "support"
        assert fm["license"] == "MIT"
        assert fm["compatibility"] == "python3; node18+"
        assert fm["allowed-tools"] == "Read"   # what the registry writes for a foreign skill
        assert "category" not in fm             # no top-level keys outside the format
        assert text.endswith("skill body content\n")

    def test_a_string_is_never_split_into_characters(self, tmp_path):
        result = export_skill_to_disk(
            _skill(allowed_tools="Read Grep", compatibility="Requires python3"),
            project_root=str(tmp_path))
        fm = _frontmatter(result["written_paths"][0])
        assert fm["allowed-tools"] == "Read Grep"
        assert fm["compatibility"] == "Requires python3"

    @pytest.mark.parametrize("field,value", [
        ("allowed_tools", {"Read": True}),
        ("compatibility", ["python3", 3]),
        ("description", ["not", "a", "string"]),
        ("invocation", "sometimes"),
    ])
    def test_a_field_of_the_wrong_type_is_refused(self, tmp_path, field, value):
        with pytest.raises(ValueError):
            export_skill_to_disk(_skill(**{field: value}), project_root=str(tmp_path))
        assert not (tmp_path / ".agents").exists()

    def test_user_only_invocation_is_written_the_way_claude_reads_it(self, tmp_path):
        result = export_skill_to_disk(_skill(invocation="User"), project_root=str(tmp_path))
        fm = _frontmatter(result["written_paths"][0])
        assert fm["disable-model-invocation"] is True
        assert fm["metadata"]["decimalai-invocation"] == "user"

    def test_a_dict_without_registry_ids_is_refused_with_the_command(self, tmp_path):
        skill = _skill()
        del skill["id"]
        with pytest.raises(ValueError) as ei:
            export_skill_to_disk(skill, project_root=str(tmp_path))
        assert "decimalai skills add pdf" in str(ei.value)

    def test_an_empty_body_is_never_written(self, tmp_path):
        with pytest.raises(ValueError):
            export_skill_to_disk(_skill(body_markdown="  \n"), project_root=str(tmp_path))
        assert not (tmp_path / ".agents").exists()

    def test_a_global_scope_is_refused_with_the_command_that_writes_one(self, tmp_path):
        with pytest.raises(ValueError) as ei:
            export_skill_to_disk(_skill(), scope="global", project_root=str(tmp_path))
        assert "decimalai skills pull pdf --out ~/.claude/skills" in str(ei.value)

    def test_it_warns_once_that_it_is_deprecated(self, tmp_path):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            export_skill_to_disk(_skill("a-one"), project_root=str(tmp_path))
            export_skill_to_disk(_skill("a-two"), project_root=str(tmp_path))
        messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(messages) == 1 and "decimalai skills add" in messages[0]


# ── Frontmatter round-trip (a body with no frontmatter of its own) ────


class TestFrontmatterRoundTrip:
    """``_reconstruct_skill_md`` must survive its own parser — an unquoted
    `description:` containing a colon used to produce invalid YAML, so the
    file was silently skipped on re-discovery."""

    def test_colon_description_round_trips(self):
        from decimalai.disk_export import _reconstruct_skill_md

        md = _reconstruct_skill_md(
            "gws-classroom",
            "Google Classroom: Manage classes, rosters, and assignments via the API.",
            "# Body\n",
        )
        fm = yaml.safe_load(md.split("---")[1])
        assert fm["name"] == "gws-classroom"
        assert fm["description"].startswith("Google Classroom: Manage classes")

    def test_hostile_values_round_trip(self):
        from decimalai.disk_export import _reconstruct_skill_md

        hostile = [
            "plain words are fine",
            "колонки: unicode",
            '"already quoted"',
            "# looks like a comment",
            "[brackets, like, a, list]",
            "yes",          # YAML 1.1 boolean trap
            "  leading space",
            "trailing space ",
            "newline\nin the middle",
        ]
        for desc in hostile:
            md = _reconstruct_skill_md("name-x", desc, "# B\n", license_info="MIT: custom")
            fm = yaml.safe_load(md.split("---")[1])
            assert fm["description"] == desc, desc
            assert fm["license"] == "MIT: custom"

    def test_lists_are_written_as_strings(self):
        from decimalai.disk_export import _reconstruct_skill_md

        md = _reconstruct_skill_md(
            "name-x", "d", "# B\n",
            compatibility=["python: >=3.10", "needs: network"],
            allowed_tools=["Bash", "Read"],
        )
        fm = yaml.safe_load(md.split("---")[1])
        assert fm["compatibility"] == "python: >=3.10; needs: network"
        assert fm["allowed-tools"] == "Bash Read"

    def test_a_string_stays_one_string(self):
        from decimalai.disk_export import _reconstruct_skill_md

        md = _reconstruct_skill_md("name-x", "d", "# B\n", allowed_tools="Read Grep",
                                   compatibility="python3")
        fm = yaml.safe_load(md.split("---")[1])
        assert fm["allowed-tools"] == "Read Grep"
        assert fm["compatibility"] == "python3"
