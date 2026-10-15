"""``.decimal/skills.toml``: what a project wants.

People edit this file by hand and commit it, so the engine reads it strictly and
writes it gently: a bad value is refused with the line it is on, and a save rewrites
only the ``[skills]`` lines that changed. Comments, blank lines, key spelling and other
tables come back byte for byte.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from decimalai.skillpkg import errors as E
from decimalai.skillpkg import manifest as M
from decimalai.skillpkg import project as P

FRESH = (
    "# Managed by `decimalai skills`. Commit this file. Hand edits are fine;\n"
    "# run `decimalai skills install` afterwards.\n"
    'root = ".agents/skills"\n'
    "\n"
    "[skills]\n"
)

# A hand-edited manifest: comments everywhere, a bare key, a trailing comment, an
# inline table, a blank line inside [skills], and another tool's table after it.
EDITED = """\
# Our agent's skills. Ask #platform before adding one.
root = ".agents/skills"   # the only root

[skills]
# support desk
"acme/refund-policy" = "latest"
pdf-tools = "3"  # pinned: v4 broke the table parser

"anthropic/pdf" = { version = "latest", id = "5eed0000-0000-0000-0000-000000000001" }

[tool.other]
keep = ["this", "exactly"]   # and this comment
"""


def _project(tmp_path, text=None) -> P.Project:
    project = P.Project.at(tmp_path)
    if text is not None:
        project.state_dir.mkdir(parents=True, exist_ok=True)
        project.manifest_path.write_bytes(text.encode("utf-8") if isinstance(text, str) else text)
    return project


# ── reading ──────────────────────────────────────────────────────────────────


class TestLoad:
    def test_the_documented_example(self):
        m = M.loads(EDITED)
        assert m.root == ".agents/skills"
        assert list(m.entries) == ["acme/refund-policy", "pdf-tools", "anthropic/pdf"]
        assert m["acme/refund-policy"] == M.ManifestEntry("acme/refund-policy", "latest")
        assert m["pdf-tools"] == M.ManifestEntry("pdf-tools", "3")
        assert m["anthropic/pdf"] == M.ManifestEntry(
            "anthropic/pdf", "latest", id="5eed0000-0000-0000-0000-000000000001")

    def test_entry_properties(self):
        latest, exact = M.ManifestEntry("a", "latest"), M.ManifestEntry("b", "12")
        assert latest.latest and latest.version is None and latest.pin == "latest"
        assert not exact.latest and exact.version == 12 and exact.pin == "exact"

    def test_the_oracle_manifest_loads(self):
        from tests.native_loaders import oracle

        text = oracle.build().files[".decimal/skills.toml"].decode("utf-8")
        m = M.loads(text)
        assert set(m.entries) == {row.name for row in oracle.ROWS}
        for row in oracle.ROWS:
            assert m[row.name].spec == ("latest" if row.pin == "latest" else str(row.version))

    def test_root_and_skills_may_be_absent(self):
        m = M.loads("# nothing yet\n")
        assert m.root == ".agents/skills"
        assert len(m) == 0

    @pytest.mark.parametrize("root", [".agents/skills/", "./.agents/skills"])
    def test_spellings_of_the_default_root_are_the_default_root(self, root):
        assert M.loads(f'root = "{root}"\n').root == ".agents/skills"

    def test_other_tables_and_keys_are_ignored(self):
        m = M.loads('extra = 1\n[skills]\n"a" = "latest"\n[tool.x]\ny = 2\n')
        assert list(m.entries) == ["a"]

    def test_load_from_the_project(self, tmp_path):
        m = M.load(_project(tmp_path, EDITED))
        assert len(m) == 3

    def test_a_missing_manifest(self, tmp_path):
        project = _project(tmp_path)
        with pytest.raises(E.SkillsNotInstalledError):
            M.load(project)
        empty = M.load(project, missing_ok=True)
        assert len(empty) == 0 and empty.root == ".agents/skills"
        assert empty.dumps() == FRESH

    def test_a_byte_order_mark_is_read_and_kept(self, tmp_path):
        project = _project(tmp_path, b"\xef\xbb\xbf" + EDITED.encode())
        m = M.load(project)
        assert len(m) == 3
        m.set("new/skill", "latest")
        M.save(project, m)
        assert project.manifest_path.read_bytes().startswith(b"\xef\xbb\xbf# Our agent")


class TestRefusals:
    """Every refusal is a ManifestError (exit 3) that names the line."""

    @pytest.mark.parametrize("text, line, fragment", [
        ('root = ".claude/skills"\n', 1, "root"),
        ('root = 3\n', 1, "root"),
        ('# a\n\n[skills]\n"a" = "latest"\n"pdf" = 3\n', 5, "string"),
        ('[skills]\n"pdf" = "v3"\n', 2, "latest"),
        ('[skills]\n"pdf" = "03"\n', 2, "positive integer"),
        ('[skills]\n"pdf" = "0"\n', 2, "positive integer"),
        ('[skills]\n"pdf" = "-1"\n', 2, "positive integer"),
        ('[skills]\n"pdf" = "^1.2"\n', 2, "positive integer"),
        ('[skills]\n"pdf" = "Latest"\n', 2, "latest"),
        ('[skills]\n"pdf" = ""\n', 2, "latest"),
        ('[skills]\n"a" = "1"\n"b" = { id = "x" }\n', 3, "version"),
        ('[skills]\n"b" = { version = "1", ref = "x" }\n', 2, "ref"),
        ('[skills]\n"b" = { version = "1", id = 7 }\n', 2, "id"),
        ('[skills]\n"b" = { version = "1", id = "" }\n', 2, "id"),
        ('[skills]\n"" = "latest"\n', 2, "name"),
        ('skills = "pdf"\n', 1, "table"),
    ])
    def test_a_bad_value_names_its_line(self, text, line, fragment):
        with pytest.raises(E.ManifestError) as exc:
            M.loads(text)
        err = exc.value
        assert err.exit_code == 3
        assert f"line {line}" in err.message, err.message
        assert fragment in err.message, err.message
        assert err.details.get("line") == line
        assert err.path == ".decimal/skills.toml"

    def test_invalid_toml_names_its_line(self):
        with pytest.raises(E.ManifestError) as exc:
            M.loads('[skills]\n"a" = "latest"\n"b" = \n')
        assert "line 3" in exc.value.message
        assert exc.value.details.get("line") == 3

    def test_a_duplicate_key_is_invalid_toml(self):
        with pytest.raises(E.ManifestError, match="line 3"):
            M.loads('[skills]\n"a" = "1"\n"a" = "2"\n')

    def test_two_entries_with_one_id_are_refused(self):
        with pytest.raises(E.ManifestError) as exc:
            M.loads('[skills]\n"a" = { version = "1", id = "x" }\n'
                    '"b" = { version = "latest", id = "x" }\n')
        assert "line 3" in exc.value.message

    def test_not_utf8(self, tmp_path):
        with pytest.raises(E.ManifestError, match="UTF-8"):
            M.load(_project(tmp_path, b'[skills]\n"caf\xe9" = "1"\n'))

    def test_the_fix_names_the_file(self):
        with pytest.raises(E.ManifestError) as exc:
            M.loads('root = "elsewhere"\n')
        assert "`.decimal/skills.toml`" in str(exc.value)


# ── specs ────────────────────────────────────────────────────────────────────


class TestSourceHygiene:
    def test_invisible_characters_are_written_as_escapes(self):
        # A literal BOM, NEL or line separator in source is invisible in review and
        # easily lost in transit; the engine's modules spell them as escapes.
        import decimalai.skillpkg as pkg

        bad = {chr(c) for c in (0xFEFF, 0x85, 0x2028, 0x2029, 0x200B, 0xA0)}
        for path in sorted(Path(pkg.__file__).parent.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            found = sorted({f"U+{ord(ch):04X}" for ch in text if ch in bad})
            assert not found, f"{path.name} has literal {', '.join(found)}"


class TestParseSpec:
    @pytest.mark.parametrize("value, spec", [
        ("latest", "latest"), ("1", "1"), ("42", "42"), (3, "3"),
    ])
    def test_accepted(self, value, spec):
        assert M.parse_spec(value) == spec

    @pytest.mark.parametrize("value", ["", "0", "03", "-1", "1.0", "v1", "LATEST", " 1",
                                       0, -2, True, None, 1.0])
    def test_refused_as_usage_errors(self, value):
        with pytest.raises(E.UsageError):
            M.parse_spec(value)


# ── writing ──────────────────────────────────────────────────────────────────


class TestFreshFile:
    def test_layout(self):
        m = M.new()
        m.set("acme/refund-policy")
        m.set("pdf-tools", "3")
        m.set("anthropic/pdf", "latest", id="5eed")
        assert m.dumps() == FRESH + (
            '"acme/refund-policy" = "latest"\n'
            '"pdf-tools" = "3"\n'
            '"anthropic/pdf" = { version = "latest", id = "5eed" }\n'
        )

    def test_matches_the_oracle_byte_for_byte(self):
        from tests.native_loaders import oracle

        expected = oracle.build().files[".decimal/skills.toml"].decode("utf-8")
        m = M.new()
        for row in oracle.ROWS:
            m.set(row.name, "latest" if row.pin == "latest" else str(row.version))
        assert m.dumps() == expected

    @pytest.mark.parametrize("name", [
        'acme/say "hi"', "back\\slash", "tab\there", "日本語スキル", "line sep",
        "composiohq/Docker Hub Automation", "o'brien/skill", "ctl\x01x", "del\x7f",
    ])
    def test_any_registry_name_round_trips(self, name):
        m = M.new()
        m.set(name, "2")
        back = M.loads(m.dumps())
        assert list(back.entries) == [name]
        assert back[name].spec == "2"

    def test_save_writes_only_on_change(self, tmp_path):
        project = _project(tmp_path)
        m = M.new()
        m.set("a")
        assert M.save(project, m) is True
        assert project.manifest_path.read_text() == FRESH + '"a" = "latest"\n'
        before = project.manifest_path.stat()
        assert M.save(project, M.load(project)) is False
        after = project.manifest_path.stat()
        assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)


class TestLinePreservingEdits:
    def test_an_unchanged_manifest_is_byte_identical(self):
        assert M.loads(EDITED).dumps() == EDITED

    def test_changing_a_spec_rewrites_only_that_line(self):
        m = M.loads(EDITED)
        m.set("pdf-tools", "4")
        out = m.dumps()
        # The bare key, the spacing and the trailing comment survive.
        assert out == EDITED.replace(
            'pdf-tools = "3"  # pinned', 'pdf-tools = "4"  # pinned')

    def test_adding_appends_after_the_last_skills_entry(self):
        m = M.loads(EDITED)
        m.set("new/skill", "latest")
        assert m.dumps() == EDITED.replace(
            '"5eed0000-0000-0000-0000-000000000001" }\n',
            '"5eed0000-0000-0000-0000-000000000001" }\n"new/skill" = "latest"\n')

    def test_removing_deletes_only_that_line(self):
        m = M.loads(EDITED)
        assert m.remove("acme/refund-policy") is True
        assert m.remove("never-there") is False
        assert m.dumps() == EDITED.replace('"acme/refund-policy" = "latest"\n', "")

    def test_adding_an_id_rewrites_the_line_as_an_inline_table(self):
        m = M.loads(EDITED)
        m.set("acme/refund-policy", "latest", id="abc")
        assert m.dumps() == EDITED.replace(
            '"acme/refund-policy" = "latest"\n',
            '"acme/refund-policy" = { version = "latest", id = "abc" }\n')

    def test_dropping_an_id_rewrites_the_inline_table_as_a_string(self):
        m = M.loads(EDITED)
        m.set("anthropic/pdf", "7")
        assert m.dumps() == EDITED.replace(
            '"anthropic/pdf" = { version = "latest", id = "5eed0000-0000-0000-0000-000000000001" }',
            '"anthropic/pdf" = "7"')

    def test_a_manifest_without_a_skills_table_gains_one_at_the_end(self):
        text = '# mine\nroot = ".agents/skills"\n'
        m = M.loads(text)
        m.set("a", "1")
        assert m.dumps() == text + '\n[skills]\n"a" = "1"\n'

    def test_a_file_without_a_final_newline(self):
        text = '[skills]\n"a" = "1"'
        m = M.loads(text)
        m.set("b", "2")
        assert m.dumps() == '[skills]\n"a" = "1"\n"b" = "2"\n'

    def test_an_empty_skills_table_gets_entries_under_its_header(self):
        text = "[skills]  # none yet\n\n[tool.x]\ny = 1\n"
        m = M.loads(text)
        m.set("a", "1")
        assert m.dumps() == '[skills]  # none yet\n"a" = "1"\n\n[tool.x]\ny = 1\n'

    def test_crlf_files_stay_crlf(self):
        text = EDITED.replace("\n", "\r\n")
        m = M.loads(text)
        assert m.dumps() == text
        m.set("new/skill", "1")
        out = m.dumps()
        assert '"new/skill" = "1"\r\n' in out
        assert "\n" not in out.replace("\r\n", "")

    def test_quoted_keys_with_equals_and_hashes_are_rewritten_correctly(self):
        text = '[skills]\n"a=b#c" = "1" # note = x\n'
        m = M.loads(text)
        m.set("a=b#c", "2")
        assert m.dumps() == '[skills]\n"a=b#c" = "2" # note = x\n'

    def test_a_literal_quoted_key_keeps_its_spelling(self):
        text = "[skills]\n'acme/x' = '1'\n"
        m = M.loads(text)
        m.set("acme/x", "2")
        assert m.dumps() == "[skills]\n'acme/x' = \"2\"\n"

    def test_a_multi_line_value_is_replaced_whole(self):
        text = '[skills]\n"a" = """\n1"""\n"b" = { version = "2", id = "x" }\n'
        m = M.loads(text)
        m.set("a", "3")
        assert M.loads(m.dumps())["a"].spec == "3"
        assert m.dumps().endswith('"b" = { version = "2", id = "x" }\n')

    def test_an_entry_written_as_its_own_table_is_read_but_not_rewritten(self):
        text = '[skills]\n"a" = "1"\n\n[skills."b"]\nversion = "2"\nid = "x"\n'
        m = M.loads(text)
        assert m["b"] == M.ManifestEntry("b", "2", id="x")
        m.set("a", "5")  # an entry the editor can rewrite
        assert M.loads(m.dumps())["a"].spec == "5"
        m.set("b", "3")  # one it cannot: refused, never written wrong
        with pytest.raises(E.ManifestError, match="line 4"):
            m.dumps()

    def test_every_edit_reads_back_as_intended(self):
        m = M.loads(EDITED)
        m.set("pdf-tools", "9")
        m.remove("anthropic/pdf")
        m.set("x/y", "latest", id="i")
        back = M.loads(m.dumps())
        assert dict(back.entries) == dict(m.entries)
        assert "[tool.other]\nkeep = [\"this\", \"exactly\"]   # and this comment\n" in m.dumps()

    def test_entries_cannot_be_mutated_behind_the_editor(self):
        m = M.loads(EDITED)
        with pytest.raises(TypeError):
            m.entries["z"] = M.ManifestEntry("z")  # type: ignore[index]

    @pytest.mark.parametrize("bad", ["", "\ud800"])
    def test_set_refuses_a_name_toml_cannot_hold(self, bad):
        with pytest.raises(E.UsageError):
            M.new().set(bad)

    def test_set_refuses_a_bad_spec_or_id(self):
        with pytest.raises(E.UsageError):
            M.new().set("a", "v2")
        with pytest.raises(E.UsageError):
            M.new().set("a", "latest", id="")
