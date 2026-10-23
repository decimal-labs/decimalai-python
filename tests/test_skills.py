"""Tests for skill discovery and activation detection."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from decimalai.skills import (
    SkillRegistry,
    detect_skills_present_in_prompt,
    discover_skills,
    parse_skill_md,
    _split_frontmatter,
    _skill_appears_in_text,
    _hash_content,
    _parse_simple_yaml,
)


def detect_present_names(rendered_input, registry, **kwargs):
    """The skills present in the prompt, by name, each once, in first-seen order."""
    return list(dict.fromkeys(
        name for name, _ in detect_skills_present_in_prompt(rendered_input, registry, **kwargs)))


# ── Fixtures ──────────────────────────────────────────────


SAMPLE_SKILL_MD = """---
name: code-review
description: Reviews code for security and style. Use when reviewing PRs.
metadata:
  version: "1.0"
allowed-tools: Bash(git:*) Read
---
# Code Review

## Overview
Perform a security and best-practices audit on the provided code.

## Instructions
1. Check for common security vulnerabilities
2. Review code style and naming conventions
3. Suggest improvements
"""


MINIMAL_SKILL_MD = """---
name: minimal-skill
description: A minimal skill
---
# Minimal Skill Instructions

Do the thing.
"""


@pytest.fixture
def skill_dir(tmp_path):
    """Create a temp .claude/skills/ directory with sample skills."""
    base = tmp_path / ".claude" / "skills"

    # code-review skill
    cr = base / "code-review"
    cr.mkdir(parents=True)
    (cr / "SKILL.md").write_text(SAMPLE_SKILL_MD)

    # minimal skill
    ms = base / "minimal-skill"
    ms.mkdir(parents=True)
    (ms / "SKILL.md").write_text(MINIMAL_SKILL_MD)

    # Invalid skill (no frontmatter)
    inv = base / "invalid"
    inv.mkdir(parents=True)
    (inv / "SKILL.md").write_text("Just some text, no frontmatter\n")

    return str(base)


# ── parse_skill_md ────────────────────────────────────────


class TestParseSkillMd:
    def test_parses_full_skill(self, skill_dir):
        result = parse_skill_md(os.path.join(skill_dir, "code-review", "SKILL.md"))
        assert result is not None
        assert result["name"] == "code-review"
        assert result["description"] == "Reviews code for security and style. Use when reviewing PRs."
        assert result["version"] == "1.0"
        assert result["hash"].startswith("sha256:")
        assert result["stability"] == "stable"
        assert "code-review" in result["source_path"]

    def test_parses_minimal_skill(self, skill_dir):
        result = parse_skill_md(os.path.join(skill_dir, "minimal-skill", "SKILL.md"))
        assert result is not None
        assert result["name"] == "minimal-skill"
        assert result["version"] is None  # No metadata.version

    def test_invalid_skill_returns_none(self, skill_dir):
        result = parse_skill_md(os.path.join(skill_dir, "invalid", "SKILL.md"))
        assert result is None  # No frontmatter

    def test_parses_taxonomy_labels(self, tmp_path):
        """skill-type / skill-scope / invocation frontmatter land in the
        descriptor."""
        p = tmp_path / "labeled"
        p.mkdir()
        (p / "SKILL.md").write_text(
            "---\nname: labeled\ndescription: Labeled\n"
            "skill-type: capability\nskill-scope: private\ninvocation: model\n---\n"
            "# Labeled\nDo the labeled thing.\n"
        )
        result = parse_skill_md(str(p / "SKILL.md"))
        assert result["skill_type"] == "capability"
        assert result["skill_scope"] == "private"
        assert result["invocation"] == "model"

    def test_new_preference_public_parses(self, tmp_path):
        """The other new-taxonomy pairing round-trips too."""
        p = tmp_path / "pref"
        p.mkdir()
        (p / "SKILL.md").write_text(
            "---\nname: pref\ndescription: Pref\n"
            "skill-type: preference\nskill-scope: public\n---\n"
            "# Pref\nHouse style.\n"
        )
        result = parse_skill_md(str(p / "SKILL.md"))
        assert result["skill_type"] == "preference"
        assert result["skill_scope"] == "public"

    def test_legacy_convention_maps_to_preference_public(self, tmp_path):
        """A pre-split SKILL.md spelling `skill-type: convention` still parses,
        mapping to the new (preference, public) pair (skills redesign
        2026-07-04 legacy compat)."""
        p = tmp_path / "legacy"
        p.mkdir()
        (p / "SKILL.md").write_text(
            "---\nname: legacy\ndescription: Legacy\n"
            "skill-type: convention\n---\n"
            "# Legacy\nStill works.\n"
        )
        result = parse_skill_md(str(p / "SKILL.md"))
        assert result["skill_type"] == "preference"
        assert result["skill_scope"] == "public"

    def test_legacy_proprietary_maps_to_capability_private(self, tmp_path):
        """Legacy `proprietary` → (capability, private)."""
        p = tmp_path / "legacy2"
        p.mkdir()
        (p / "SKILL.md").write_text(
            "---\nname: legacy2\ndescription: Legacy2\n"
            "skill-type: proprietary\n---\n"
            "# Legacy2\nStill works.\n"
        )
        result = parse_skill_md(str(p / "SKILL.md"))
        assert result["skill_type"] == "capability"
        assert result["skill_scope"] == "private"

    def test_legacy_model_gap_maps_to_capability_public(self, tmp_path):
        """Legacy kebab `model-gap` → (capability, public)."""
        p = tmp_path / "legacy3"
        p.mkdir()
        (p / "SKILL.md").write_text(
            "---\nname: legacy3\ndescription: Legacy3\n"
            "skill-type: model-gap\n---\n"
            "# Legacy3\nStill works.\n"
        )
        result = parse_skill_md(str(p / "SKILL.md"))
        assert result["skill_type"] == "capability"
        assert result["skill_scope"] == "public"

    def test_explicit_scope_overrides_legacy_implied_scope(self, tmp_path):
        """An explicit `skill-scope` wins over the scope a legacy `skill-type`
        implies — `convention` implies public, but private is honored."""
        p = tmp_path / "override"
        p.mkdir()
        (p / "SKILL.md").write_text(
            "---\nname: override\ndescription: Override\n"
            "skill-type: convention\nskill-scope: private\n---\n"
            "# Override\nScope wins.\n"
        )
        result = parse_skill_md(str(p / "SKILL.md"))
        assert result["skill_type"] == "preference"
        assert result["skill_scope"] == "private"

    def test_invalid_taxonomy_values_ignored(self, tmp_path):
        """Unrecognized taxonomy values are dropped, never raised."""
        p = tmp_path / "bogus"
        p.mkdir()
        (p / "SKILL.md").write_text(
            "---\nname: bogus\ndescription: Bogus\n"
            "skill-type: not-a-thing\nskill-scope: nowhere\n---\n"
            "# Bogus\nInvalid labels.\n"
        )
        result = parse_skill_md(str(p / "SKILL.md"))
        assert result["skill_type"] is None
        assert result["skill_scope"] is None

    def test_disable_model_invocation_maps_to_user(self, tmp_path):
        """Claude Code's `disable-model-invocation: true` spelling is accepted
        as `invocation: user` on import, and re-exports with both spellings
        so the file round-trips through either authoring tool."""
        p = tmp_path / "on-demand"
        p.mkdir()
        (p / "SKILL.md").write_text(
            "---\nname: on-demand\ndescription: On demand\n"
            "disable-model-invocation: true\n---\n"
            "# On Demand\nOnly when asked.\n"
        )
        result = parse_skill_md(str(p / "SKILL.md"))
        assert result["invocation"] == "user"

    def test_explicit_invocation_wins_over_disable_flag(self, tmp_path):
        p = tmp_path / "both"
        p.mkdir()
        (p / "SKILL.md").write_text(
            "---\nname: both\ndescription: Both spellings\n"
            "invocation: any\ndisable-model-invocation: true\n---\n"
            "# Both\nExplicit invocation wins.\n"
        )
        result = parse_skill_md(str(p / "SKILL.md"))
        assert result["invocation"] == "any"

    def test_labels_absent_default_to_none(self, skill_dir):
        """No default is stamped — legacy skills carry no labels."""
        result = parse_skill_md(os.path.join(skill_dir, "minimal-skill", "SKILL.md"))
        assert result["category"] is None
        assert result["skill_type"] is None
        assert result["skill_scope"] is None
        assert result["invocation"] is None


def _write_skill(tmp_path, folder, frontmatter, body="# Body\nDo the thing carefully.\n"):
    d = tmp_path / folder
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n{body}", encoding="utf-8")
    return d / "SKILL.md"


class TestNameGrammar:
    """The folder-name grammar every native loader accepts: lowercase letters, digits and
    single hyphens, at most 64 characters. A leading digit is allowed."""

    def test_a_leading_digit_is_a_valid_name(self, tmp_path):
        path = _write_skill(
            tmp_path, "2slides-ppt-generator",
            "name: 2slides-ppt-generator\ndescription: Make slides",
        )
        result = parse_skill_md(str(path))
        assert result is not None
        assert result["name"] == "2slides-ppt-generator"
        names = {s["name"] for s in discover_skills([str(tmp_path)])}
        assert "2slides-ppt-generator" in names

    @pytest.mark.parametrize("bad", ["-x", "x--y", "x-", "Upper", "under_score", "a" * 65])
    def test_names_outside_the_grammar_are_still_skipped(self, tmp_path, bad):
        path = _write_skill(tmp_path, "folder", f'name: "{bad}"\ndescription: d')
        assert parse_skill_md(str(path)) is None

    def test_a_name_that_differs_from_its_folder_is_warned_about_and_read(
        self, tmp_path, caplog
    ):
        path = _write_skill(tmp_path, "folder-name", "name: frontmatter-name\ndescription: d")
        with caplog.at_level("WARNING", logger="decimalai.skills"):
            result = parse_skill_md(str(path))
        assert result is not None and result["name"] == "frontmatter-name"
        assert any(
            "frontmatter-name" in r.getMessage() and "folder-name" in r.getMessage()
            for r in caplog.records
        ), [r.getMessage() for r in caplog.records]


class TestYamlValues:
    """Frontmatter is YAML: quoted escapes decode, and YAML 1.1's typed scalars do not
    change what a name, a description or a metadata value says."""

    def test_an_escaped_description_is_decoded(self, tmp_path):
        path = _write_skill(
            tmp_path, "shipper",
            'name: shipper\ndescription: "Use when the user says \\"ship it\\": deploy\\tnow"',
        )
        result = parse_skill_md(str(path))
        assert result["description"] == 'Use when the user says "ship it": deploy\tnow'

    def test_yes_no_and_dates_stay_the_strings_that_were_written(self, tmp_path):
        path = _write_skill(
            tmp_path, "yes",
            "name: yes\ndescription: no\nmetadata:\n  version: 1.10\n  updated: 2024-01-01",
        )
        result = parse_skill_md(str(path))
        assert result["name"] == "yes"
        assert result["description"] == "no"
        assert result["version"] == "1.10"
        fm, _ = _split_frontmatter(path.read_text(encoding="utf-8"))
        assert fm["metadata"] == {"version": "1.10", "updated": "2024-01-01"}

    @pytest.mark.parametrize("written", [
        "Triage GitHub issue #123 and label it",
        "Idiomatic C# for .NET projects #dotnet tips",
        "Format markdown headings like # Title and ## Sub",
        "&escape sequences in shell",
    ])
    def test_an_unquoted_description_yaml_would_cut_keeps_its_text(self, tmp_path, written):
        """In YAML a ` #` starts a comment and a leading `&` is an anchor, so these read
        short. The reader before YAML kept the line as written, and so does this one."""
        path = _write_skill(tmp_path, "cut", f"name: cut\ndescription: {written}")
        assert parse_skill_md(str(path))["description"] == written
        fm, _ = _split_frontmatter(path.read_text(encoding="utf-8"))
        assert fm["description"] == written  # what `skills sync` posts

    def test_an_alias_to_an_earlier_anchor_keeps_its_text(self, tmp_path):
        path = _write_skill(
            tmp_path, "alias", "name: alias\ncategory: &ops devops\ndescription: *ops"
        )
        result = parse_skill_md(str(path))
        assert result["description"] == "*ops"
        assert result["category"] == "&ops devops"

    def test_a_category_and_a_metadata_value_keep_their_text(self, tmp_path):
        path = _write_skill(
            tmp_path, "meta",
            "name: meta\ndescription: d\ncategory: devops # ci tools\n"
            "metadata:\n  version: &v 2.0\n  owner: platform #infra",
        )
        result = parse_skill_md(str(path))
        assert result["category"] == "devops # ci tools"
        assert result["version"] == "&v 2.0"
        fm, _ = _split_frontmatter(path.read_text(encoding="utf-8"))
        assert fm["metadata"] == {"version": "&v 2.0", "owner": "platform #infra"}

    def test_comments_on_their_own_lines_and_after_quotes_stay_comments(self, tmp_path):
        path = _write_skill(
            tmp_path, "commented",
            "name: commented\n# description: the old one\n"
            'description: "Fix issue #123" # quoted, so this is a comment\n'
            "metadata:\n  # stamped by hand\n  version: '1.0' # pinned",
        )
        result = parse_skill_md(str(path))
        assert result["description"] == "Fix issue #123"
        fm, _ = _split_frontmatter(path.read_text(encoding="utf-8"))
        assert fm["metadata"] == {"version": "1.0"}

    def test_a_name_is_read_as_yaml_reads_it(self, tmp_path):
        """A name is a slug, which can never hold ` #` or `&`: the text as written would
        only make the skill unreadable, so a trailing comment stays a comment there."""
        path = _write_skill(tmp_path, "deploy", "name: deploy  # the slug\ndescription: d")
        assert parse_skill_md(str(path))["name"] == "deploy"

    def test_a_date_valued_key_leaves_the_frontmatter_json_serializable(self):
        """`skills sync` posts the frontmatter as JSON; a YAML date must not break that."""
        fm, _ = _split_frontmatter(
            "---\nname: dated\ndescription: d\ndate: 2025-01-15\n"
            "tags:\n  - 2025-01-15\n---\n# Body"
        )
        assert fm["date"] == "2025-01-15"
        json.dumps(fm)

    def test_invalid_yaml_falls_back_to_the_simple_reader_with_a_warning(
        self, tmp_path, caplog
    ):
        """`key: value: more` is not YAML, but hand-authored trees have it and discovery
        has always read it. It is still read, and it is no longer read silently."""
        path = _write_skill(
            tmp_path, "colon", "name: colon\ndescription: Use when: the user asks"
        )
        with caplog.at_level("WARNING", logger="decimalai.skills"):
            result = parse_skill_md(str(path))
        assert result is not None
        assert result["description"] == "Use when: the user asks"
        assert any("YAML" in r.getMessage() for r in caplog.records), [
            r.getMessage() for r in caplog.records
        ]


class TestBlockScalarsInTheSimpleReader:
    """`description: >-` holds its text on the indented lines below. The simple reader (the
    fallback for frontmatter YAML rejects, and the reader without skillevaluation) used to
    keep the header `>-` as the description and drop the text: 17 skills on the platform
    showed `>-` in their agents' menus (found 2026-10-23). It now reads them as YAML does,
    and as the platform's own parser does."""

    @pytest.mark.parametrize("frontmatter, expected", [
        ("description: >-\n  Use when the user\n  asks about tickets.", "Use when the user asks about tickets."),
        ("description: >\n  Use when the user\n  asks about tickets.", "Use when the user asks about tickets."),
        ("description: |\n  Line one.\n  Line two.", "Line one.\nLine two."),
        ("description: |-\n  Line one.\n\n  Line three.", "Line one.\n\nLine three."),
        ("description: >\n  First paragraph\n  goes on.\n\n  Second one.", "First paragraph goes on.\nSecond one."),
        ("description: |+\n  Kept.\n", "Kept.\n"),
        ("description: >2-\n    Indented by four,\n    read from two.", "  Indented by four,   read from two."),
        ("description: >-  # a comment\n  Text.", "Text."),
    ])
    def test_a_block_scalar_is_its_text(self, frontmatter, expected):
        assert _parse_simple_yaml(frontmatter)["description"] == expected

    def test_the_keys_after_a_block_scalar_still_read(self):
        result = _parse_simple_yaml(
            "name: triage\ndescription: >-\n  Use when: a ticket arrives\n  and needs a queue.\n"
            "category: ops\nmetadata:\n  version: 1.2"
        )
        assert result == {
            "name": "triage",
            "description": "Use when: a ticket arrives and needs a queue.",
            "category": "ops",
            "metadata": {"version": "1.2"},
        }

    def test_an_empty_block_scalar_is_empty_not_its_header(self):
        assert _parse_simple_yaml("description: >-\nname: x") == {"description": "", "name": "x"}

    def test_a_skill_whose_yaml_falls_back_keeps_its_folded_description(self, tmp_path):
        """`key: value: more` elsewhere is not YAML, so the whole frontmatter takes the
        simple reader: that is how a `>-` reached the platform."""
        path = _write_skill(
            tmp_path, "triage",
            "name: triage\ndescription: >-\n  Pick the queue a ticket\n  belongs to.\n"
            "category: Use when: support",
        )
        result = parse_skill_md(str(path))
        assert result["description"] == "Pick the queue a ticket belongs to."
        fm, _ = _split_frontmatter(path.read_text(encoding="utf-8"))
        assert fm["description"] == "Pick the queue a ticket belongs to."  # what `skills sync` posts


# ── discover_skills ───────────────────────────────────────


class TestDiscoverSkills:
    def test_discovers_skills_from_directory(self, skill_dir):
        skills = discover_skills([skill_dir], include_global=False)
        assert len(skills) == 2  # code-review + minimal-skill
        names = {s["name"] for s in skills}
        assert "code-review" in names
        assert "minimal-skill" in names

    def test_ignores_nonexistent_paths(self):
        skills = discover_skills(["/nonexistent/path"], include_global=False)
        assert skills == []

    def test_deduplicates_by_name(self, skill_dir):
        # Passing same path twice should not duplicate
        skills = discover_skills([skill_dir, skill_dir], include_global=False)
        names = [s["name"] for s in skills]
        assert len(names) == len(set(names))

    def test_include_global_defaults_to_false(self, skill_dir, monkeypatch, tmp_path):
        """A developer running `discover_skills()` from a project
        with no local skills must NOT sweep up ~/.claude/skills/* into
        the upload. Calling with default args + only an empty
        search_paths must return [] even when ~/.claude/skills has
        entries. Use HOME redirect to simulate this safely."""
        # Plant a fake global skill that WOULD be picked up if the
        # default flipped back to True.
        fake_home = tmp_path / "home"
        global_skill_dir = fake_home / ".claude" / "skills" / "fake-leak"
        global_skill_dir.mkdir(parents=True)
        (global_skill_dir / "SKILL.md").write_text(
            "---\nname: fake-leak\ndescription: a global skill that "
            "must not leak into the org registry\n---\n\n"
            "body content longer than the minimum length threshold.\n"
        )
        monkeypatch.setenv("HOME", str(fake_home))

        # Call discover_skills() with NO `include_global` arg — the
        # default behavior. Pass a non-matching search_paths so the
        # only candidate would be the global one.
        skills = discover_skills(search_paths=[str(tmp_path / "no-such-project")])
        names = {s["name"] for s in skills}
        assert "fake-leak" not in names, (
            "Regression: default discover_skills() leaked a global "
            "~/.claude/skills entry into the org registry. include_global "
            "must default to False so a user's personal skills never sync."
        )

    def test_include_global_true_still_works_as_opt_in(self, monkeypatch, tmp_path):
        """The opt-in path should still pick up global skills when the
        user explicitly asks for them."""
        fake_home = tmp_path / "home"
        global_skill_dir = fake_home / ".claude" / "skills" / "explicit-global"
        global_skill_dir.mkdir(parents=True)
        (global_skill_dir / "SKILL.md").write_text(
            "---\nname: explicit-global\ndescription: user explicitly "
            "asked for this one\n---\n\n"
            "body content longer than the minimum length threshold.\n"
        )
        monkeypatch.setenv("HOME", str(fake_home))

        skills = discover_skills(
            search_paths=[str(tmp_path / "no-such-project")],
            include_global=True,
        )
        names = {s["name"] for s in skills}
        assert "explicit-global" in names


# ── SkillRegistry ──────────────────────────────────────────


class TestSkillRegistry:
    def test_empty_registry(self):
        registry = SkillRegistry()
        assert len(registry) == 0
        assert not registry

    def test_auto_discovered(self):
        registry = SkillRegistry(auto_discovered=[
            {"name": "a", "hash": "sha256:aaa"},
            {"name": "b", "hash": "sha256:bbb"},
        ])
        assert len(registry) == 2
        assert registry.names == ["a", "b"]

    def test_explicit_overrides_auto(self):
        registry = SkillRegistry(
            auto_discovered=[{"name": "a", "hash": "sha256:auto"}],
            explicit=[{"name": "a", "hash": "sha256:explicit"}],
        )
        assert len(registry) == 1
        assert registry.get("a")["hash"] == "sha256:explicit"

    def test_merge(self):
        registry = SkillRegistry(
            auto_discovered=[{"name": "a", "hash": "sha256:aaa"}],
            explicit=[{"name": "b", "hash": "sha256:bbb"}],
        )
        assert len(registry) == 2
        assert registry.names == ["a", "b"]


# ── the skills present in a prompt, by name ─────────────


class TestPresentNames:
    def test_detects_skill_header(self):
        rendered_input = [
            {"role": "system", "content": "You are helpful.\n\n## Skill: code-review\nReview code for security..."},
        ]
        registry = [{"name": "code-review", "hash": "sha256:abc"}]
        activated = detect_present_names(rendered_input, registry)
        assert "code-review" in activated

    def test_detects_active_skill_header(self):
        rendered_input = [
            {"role": "system", "content": "Base prompt.\n\n## Active Skill: sql-optimizer\nOptimize queries..."},
        ]
        registry = [
            {"name": "code-review", "hash": "sha256:abc"},
            {"name": "sql-optimizer", "hash": "sha256:def"},
        ]
        activated = detect_present_names(rendered_input, registry)
        assert "sql-optimizer" in activated
        assert "code-review" not in activated

    def test_no_activation_when_no_match(self):
        rendered_input = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        registry = [{"name": "code-review", "hash": "sha256:abc"}]
        activated = detect_present_names(rendered_input, registry)
        assert activated == []

    def test_string_input(self):
        activated = detect_present_names(
            "## Skill: code-review\nDo review.",
            [{"name": "code-review", "hash": "sha256:abc"}],
        )
        assert "code-review" in activated

    def test_empty_registry(self):
        assert detect_present_names("some input", []) == []

    def test_empty_input(self):
        assert detect_present_names(None, [{"name": "x", "hash": "y"}]) == []


# ── Internal helpers ──────────────────────────────────────


class TestSplitFrontmatter:
    def test_valid_frontmatter(self):
        fm, body = _split_frontmatter("---\nname: test\n---\n# Body content")
        assert fm["name"] == "test"
        assert body == "# Body content"

    def test_no_frontmatter(self):
        fm, body = _split_frontmatter("Just markdown content")
        assert fm == {}
        assert body == "Just markdown content"

    def test_value_containing_triple_dash(self):
        """A '---' inside a frontmatter scalar must not be mistaken for
        the closing fence. The closing fence is a line that is exactly '---'."""
        fm, body = _split_frontmatter(
            '---\nname: test\ndescription: "Before --- After"\n---\n# Body'
        )
        assert fm["name"] == "test"
        assert fm["description"] == "Before --- After"
        assert body == "# Body"

    def test_horizontal_rule_in_body_preserved(self):
        """A markdown horizontal rule ('---' on its own line) in the BODY is
        harmless: the first such line after the frontmatter is the closing
        fence; later ones stay in the body."""
        fm, body = _split_frontmatter("---\nname: test\n---\nIntro\n\n---\n\nMore")
        assert fm["name"] == "test"
        assert "---" in body

    def test_lists_are_read_then_joined_into_strings(self):
        """YAML lists in `allowed-tools` and `compatibility` are read (the old reader
        returned None for both), then joined, because every caller of these two keys
        takes a string: `allowed-tools` is space-separated, `compatibility` joins with
        '; '."""
        fm, _ = _split_frontmatter(
            "---\nname: x\ndescription: d\n"
            "compatibility:\n  - \"python>=3.10\"\n  - git\n"
            "allowed-tools:\n  - \"Bash(git log:*)\"\n  - Read\n"
            "---\n# b"
        )
        assert fm["allowed-tools"] == "Bash(git log:*) Read"
        assert fm["compatibility"] == "python>=3.10; git"

    def test_a_flow_list_is_joined_too(self):
        fm, _ = _split_frontmatter("---\nname: x\nallowed-tools: [Read, Grep]\n---\n# b")
        assert fm["allowed-tools"] == "Read Grep"


class TestSkillAppearsInText:
    def test_h2_skill_header(self):
        assert _skill_appears_in_text("code-review", "## Skill: code-review")

    def test_h1_skill_name(self):
        assert _skill_appears_in_text("code-review", "# code-review")

    def test_bracketed_name(self):
        assert _skill_appears_in_text("code-review", "Loading [code-review] skill...")

    def test_no_match(self):
        assert not _skill_appears_in_text("code-review", "You are a helpful assistant")

    def test_case_insensitive(self):
        assert _skill_appears_in_text("code-review", "## SKILL: Code-Review")
