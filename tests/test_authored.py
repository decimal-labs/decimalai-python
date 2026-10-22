"""``[authored]`` in ``.decimal/skills.toml``: the skills you wrote that a project's agents use.

The package manager owns ``[skills]`` and keeps every other byte of the file; this table
is edited beside it the same way: only its one line changes, the rest is kept, and an
edit that does not read back is refused. Released SDKs read ``root`` and ``[skills]``
only, so the table must never change what they read.
"""

from __future__ import annotations

import pytest

from decimalai import _authored
from decimalai.skillpkg import manifest as _manifest
from decimalai.skillpkg.project import Project

SKILL = "---\nname: {name}\ndescription: {name} rules.\n---\n\n# {name}\n\nDo the thing.\n"


@pytest.fixture
def project(tmp_path):
    (tmp_path / ".git").mkdir()
    return Project.at(tmp_path)


def folder(project, name, text=None):
    path = project.skills_root / name
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(text or SKILL.format(name=name), encoding="utf-8")
    return path


def test_adding_to_a_project_with_no_manifest_writes_one_the_package_manager_reads(project):
    assert _authored.add(project, "refund-tone") is True
    text = project.manifest_path.read_text()
    assert '[authored]\nskills = ["refund-tone"]\n' in text
    assert dict(_manifest.loads(text).entries) == {}, "no registry entry appears"
    assert _authored.listed(project) == ["refund-tone"]
    assert _authored.add(project, "refund-tone") is False, "listed once"


def test_an_edit_keeps_every_other_byte(project):
    original = (
        "# my notes\r\n"
        'root = ".agents/skills"\r\n'
        "\r\n"
        "[skills]\r\n"
        '"acme/refund-policy" = "2"   # pinned on purpose\r\n'
        "\r\n"
        "[authored]\r\n"
        'skills = ["one"]  # mine\r\n'
        "\r\n"
        "[tool.other]\r\n"
        "x = 1\r\n"
    )
    project.state_dir.mkdir()
    project.manifest_path.write_bytes(original.encode())
    _authored.add(project, "two")
    edited = project.manifest_path.read_bytes().decode()
    assert edited == original.replace('skills = ["one"]  # mine\r\n', 'skills = ["one", "two"]\r\n')
    assert dict(_manifest.loads(edited).entries) == dict(_manifest.loads(original).entries)


def test_a_table_without_the_key_and_a_multi_line_list_are_both_rewritten(project):
    project.state_dir.mkdir()
    project.manifest_path.write_text('[skills]\n\n[authored]\n# nothing yet\n')
    _authored.add(project, "one")
    assert project.manifest_path.read_text() == '[skills]\n\n[authored]\nskills = ["one"]\n# nothing yet\n'
    project.manifest_path.write_text('[authored]\nskills = [\n  "one",\n  "two",\n]\n')
    _authored.remove(project, "one")
    assert project.manifest_path.read_text() == '[authored]\nskills = ["two"]\n'


def test_the_package_manager_keeps_the_table_when_it_saves(project):
    _authored.add(project, "refund-tone")
    m = _manifest.load(project)
    m.set("acme/refund-policy", "2")
    _manifest.save(project, m)
    assert _authored.listed(project) == ["refund-tone"]
    assert dict(_manifest.load(project).entries)["acme/refund-policy"].spec == "2"


def test_names_that_are_not_one_folder_are_ignored_when_read(project):
    project.state_dir.mkdir()
    project.manifest_path.write_text(
        '[authored]\nskills = ["ok", "a/b", "..", "", 3, "ok", " spaced "]\n')
    assert _authored.listed(project) == ["ok", "spaced"]


def test_a_broken_manifest_lists_nothing_and_is_not_edited(project):
    project.state_dir.mkdir()
    project.manifest_path.write_text("[skills\n")
    assert _authored.listed(project) == []
    with pytest.raises(Exception):
        _authored.add(project, "one")
    assert project.manifest_path.read_text() == "[skills\n"


def test_unlisted_reports_folders_that_are_neither_installed_nor_listed(project):
    folder(project, "refund-tone")
    folder(project, "run-tests")
    (project.skills_root / "no-skill-md").mkdir()
    _authored.add(project, "refund-tone")
    assert _authored.unlisted(project) == ["run-tests"]
    assert _authored.is_own_folder(project, "refund-tone")
    assert not _authored.is_own_folder(project, "missing")
    assert not _authored.is_own_folder(project, "acme/refund-policy"), "not a folder name"


def test_body_hash_reads_the_instructions_only(project):
    folder(project, "refund-tone")
    before = _authored.body_hash(project, "refund-tone")
    path = project.skills_root / "refund-tone" / "SKILL.md"
    path.write_text(path.read_text().replace("refund-tone rules.", "Other words."))
    assert _authored.body_hash(project, "refund-tone") == before, "frontmatter is not the instructions"
    path.write_text(path.read_text() + "\nOne more step.\n")
    assert _authored.body_hash(project, "refund-tone") != before
