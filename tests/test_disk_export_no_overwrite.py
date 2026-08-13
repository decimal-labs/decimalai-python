"""Tests for the no-silent-overwrite invariant in disk_export.

export_skill_to_disk never overwrites a file that already exists on disk; an
explicit `force=True` is required. These tests mechanize that guarantee.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from decimalai.disk_export import (
    SkillExportFileExistsError,
    export_skill_to_disk,
)


def _skill(name: str = "pdf") -> dict:
    return {
        "name": name,
        "description": "test skill",
        "body_markdown": "skill body content",
    }


def test_first_install_writes_skill_md_when_no_existing_file():
    with tempfile.TemporaryDirectory() as root:
        result = export_skill_to_disk(_skill(), agents=["universal"], project_root=root)
        assert len(result["written_paths"]) == 1
        assert os.path.exists(result["written_paths"][0])


def test_second_install_raises_without_force():
    with tempfile.TemporaryDirectory() as root:
        export_skill_to_disk(_skill(), agents=["universal"], project_root=root)
        with pytest.raises(SkillExportFileExistsError) as exc_info:
            export_skill_to_disk(_skill(), agents=["universal"], project_root=root)
        assert "Refusing to overwrite" in str(exc_info.value)
        assert "SKILL.md" in exc_info.value.path


def test_second_install_with_force_overwrites():
    with tempfile.TemporaryDirectory() as root:
        export_skill_to_disk(_skill(), agents=["universal"], project_root=root)
        result = export_skill_to_disk(
            _skill(), agents=["universal"], project_root=root, force=True
        )
        assert len(result["written_paths"]) == 1


def test_attachment_collision_also_raises():
    skill_with_att = {
        **_skill(),
        # attachment passed via kwarg; export uses skill_with_att directly
    }
    attachments = [{"file_path": "data.txt", "content_text": "first"}]
    with tempfile.TemporaryDirectory() as root:
        export_skill_to_disk(
            skill_with_att,
            agents=["universal"],
            project_root=root,
            attachments=attachments,
        )
        # Second install with same attachment must raise
        with pytest.raises(SkillExportFileExistsError) as exc_info:
            export_skill_to_disk(
                skill_with_att,
                agents=["universal"],
                project_root=root,
                attachments=[{"file_path": "data.txt", "content_text": "second"}],
                force=False,
            )
        assert exc_info.value.path.endswith("data.txt") or exc_info.value.path.endswith("SKILL.md")


def test_user_authored_skill_md_is_protected():
    """A SKILL.md the user authored at the target path is preserved: running
    `decimalai skills install pdf` over it raises before touching the file."""
    with tempfile.TemporaryDirectory() as root:
        # Simulate user-authored file
        user_dir = os.path.join(root, ".claude", "skills", "pdf")
        os.makedirs(user_dir, exist_ok=True)
        user_path = os.path.join(user_dir, "SKILL.md")
        with open(user_path, "w") as f:
            f.write("USER AUTHORED CONTENT — DO NOT LOSE")

        with pytest.raises(SkillExportFileExistsError):
            export_skill_to_disk(_skill(), agents=["claude-code"], project_root=root)

        # File preserved
        with open(user_path) as f:
            assert "USER AUTHORED" in f.read()


# ── Frontmatter round-trip ───────────────────────────────────────────


class TestFrontmatterRoundTrip:
    """Reconstructed SKILL.md must survive its own parser — an unquoted
    `description:` containing a colon used to produce invalid YAML, so the
    just-exported skill was silently skipped on re-discovery."""

    def test_colon_description_round_trips(self):
        import yaml
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
        import yaml
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

    def test_list_values_round_trip(self):
        import yaml
        from decimalai.disk_export import _reconstruct_skill_md

        md = _reconstruct_skill_md(
            "name-x", "d", "# B\n",
            compatibility=["python: >=3.10", "needs: network"],
            allowed_tools=["Bash", "Read: files"],
        )
        fm = yaml.safe_load(md.split("---")[1])
        assert fm["compatibility"] == ["python: >=3.10", "needs: network"]
        assert fm["allowed-tools"] == ["Bash", "Read: files"]
