"""``decimalai skills verify``: what a native skill loader would reject, said before one does.

The fixtures (``fixtures/defects/``, written by ``make_defects.py`` beside them) hold one
skill folder per folder-level rule inside a managed project, and one project per
project-level rule. The properties tested here:

* each rule is reported, with its id, on the folder that breaks it, and on no other;
* in a managed folder every rule is an error, the SDK's own rules included;
* in a folder the lock does not list, and in any tree given as ``verify PATH``, the
  loader rules are errors and the SDK's rules are warnings, unless ``strict``;
* ``plugin.json``, ``lock_hash``, the manifest and an interrupted run are checked
  against the lock;
* the files an author keeps beside SKILL.md (reference.md, LICENSE.txt, eval.yaml) are
  never a finding in a folder the lock does not list; only an unsafe path is;
* a project with hand-authored skills and no lock is graded as an unmanaged root, with
  one notice; with neither a lock nor a root there is nothing to verify (exit code 3);
* a lock the verifier cannot read (version 1, npx, corrupt) is a refusal with exit
  code 3, never a reset and never a pass;
* verify only reads: it writes nothing, takes no lock, and settles no journal.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import pytest
from skillevaluation import skillfmt

from decimalai import skillpkg
from decimalai.skillpkg import errors as E
from decimalai.skillpkg import lock as L
from decimalai.skillpkg import verifier as V

HERE = Path(__file__).resolve().parent
DEFECTS = HERE / "fixtures" / "defects"
REPO_ROOT = HERE.parents[1]
POSIX = os.name == "posix"
needs_posix = pytest.mark.skipif(not POSIX, reason="POSIX modes and symlinks")
#: chmod 000 hides a file from its owner on POSIX; root reads it anyway.
needs_unreadable = pytest.mark.skipif(
    not POSIX or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs POSIX permissions and a non-root user")

ROOT = ".agents/skills"
ERROR, WARNING = "error", "warning"
PROJECT_RULES = ("V5", "V6", "V7", "V9")


# ── what each fixture must report ────────────────────────────────────────────

#: managed/: each folder the lock lists -> the rules verify reports on it. Every one is
#: an error, whatever its scope.
MANAGED = {
    "acme-refund-policy": set(),
    "f1-duplicate-key": {"F1"},
    "f2-unknown-key": {"F2"},
    "f3-namespaced-name": {"F3"},
    "f4-empty-description": {"F4"},
    "f5-compatibility-list": {"F5"},
    "f6a-allowed-tools-list": {"F6a"},
    "f6b-bare-bash": {"F6b"},
    "f7-disable-model-invocation-false": {"F7"},
    "f8-nested-metadata": {"F8"},
    "f9-license-list": {"F9"},
    "f10-empty-body": {"F10"},
    "f11-stamp-differs": {"F11"},
    "f12-body-differs": {"F12"},
    "f13-two-trailing-newlines": {"F13"},
    # An attachment rule breaks only on a file the lock cannot list, so V3 comes with it.
    "f14-top-level-file": {"F14", "V3"},
    "f15-nested-skill-md": {"F15", "V3"},
    "v1-symlinked-folder": {"V1"},
    "v1-symlinked-resource": {"V1"},
    "v3-edited-attachment": {"V3"},
    "v4-exec-bit": {"V4"},
}

#: managed/: each folder the lock does not list -> {rule: severity} without ``strict``.
UNMANAGED = {
    "v2-nested-folder": {"V2": ERROR},
    "v8-stamped-orphan": {"V8": WARNING},
    "hand-authored-clean": {},
    "hand-authored-unknown-key": {"F2": WARNING},
    "hand-authored-namespaced-name": {"F3": ERROR},
    "hand-authored-companion-files": {},
}

#: ``verify PATH`` on one folder of managed/ -> {rule: severity} without ``strict``: the
#: unmanaged rules, whatever the lock says. F11-F13, V3 and V4 compare a folder with its
#: lock entry, so a loose copy has nothing to break; v2-nested-folder holds a skill
#: folder one level down, which is a skill of its own when the folder is the PATH.
AS_PATH = {
    "acme-refund-policy": {},
    "f1-duplicate-key": {"F1": ERROR},
    "f2-unknown-key": {"F2": WARNING},
    "f3-namespaced-name": {"F3": ERROR},
    "f4-empty-description": {"F4": ERROR},
    "f5-compatibility-list": {"F5": ERROR},
    "f6a-allowed-tools-list": {"F6a": ERROR},
    "f6b-bare-bash": {"F6b": WARNING},
    "f7-disable-model-invocation-false": {"F7": WARNING},
    "f8-nested-metadata": {"F8": ERROR},
    "f9-license-list": {"F9": ERROR},
    "f10-empty-body": {"F10": ERROR},
    "f11-stamp-differs": {},
    "f12-body-differs": {},
    "f13-two-trailing-newlines": {},
    "f14-top-level-file": {},  # a file beside SKILL.md is the author's business
    "f15-nested-skill-md": {"F15": ERROR},
    "v1-symlinked-folder": {"V1": ERROR},
    "v1-symlinked-resource": {"V1": ERROR},
    "v2-nested-folder": {},
    "v3-edited-attachment": {},
    "v4-exec-bit": {},
    "v8-stamped-orphan": {},
    "hand-authored-clean": {},
    "hand-authored-unknown-key": {"F2": WARNING},
    "hand-authored-namespaced-name": {"F3": ERROR},
    "hand-authored-companion-files": {},
}

#: Each project -> the project-level rules it reports, and its exit code.
PROJECTS = {
    "clean": (set(), E.ExitCode.OK),
    "plugin-json-stale": ({"V5"}, E.ExitCode.PROBLEMS),
    "lock-hash-edited": ({"V6"}, E.ExitCode.PROBLEMS),
    "foreign-entry": ({"V6"}, E.ExitCode.PROBLEMS),
    "manifest-disagrees": ({"V7"}, E.ExitCode.PROBLEMS),
    "interrupted": ({"V9"}, E.ExitCode.PROBLEMS),
}


# ── fixtures ─────────────────────────────────────────────────────────────────


def _generator():
    spec = importlib.util.spec_from_file_location("make_defects", DEFECTS / "make_defects.py")
    module = importlib.util.module_from_spec(spec)
    dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # no __pycache__ inside the fixtures
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = dont_write
    return module


def _set_modes(project: Path) -> None:
    """0644 everywhere, 0755 where the lock says: git keeps modes only where fileMode is on."""
    for path in project.rglob("*"):
        if path.is_file() and not path.is_symlink():
            path.chmod(0o644)
    try:
        lock = json.loads((project / ".decimal" / "skills.lock").read_text())
        entries = lock["skills"].items() if lock.get("version") == 2 else ()
    except (OSError, ValueError, KeyError):
        entries = ()
    for name, entry in entries:
        for rel in entry.get("executable") or ():
            path = project / ROOT / name / rel
            if path.is_file():
                path.chmod(0o755)


def _move_out(path: Path, elsewhere: Path) -> None:
    """Replace ``path`` with a symlink to the same thing, kept outside the project."""
    elsewhere.mkdir(parents=True, exist_ok=True)
    target = elsewhere / path.name
    shutil.move(str(path), str(target))
    os.symlink(target, path, target_is_directory=target.is_dir())


def _setup_managed(project: Path, elsewhere: Path) -> None:
    """The defects git cannot hold: two symlinks and a lost exec bit."""
    skills = project / ROOT
    _move_out(skills / "v1-symlinked-folder", elsewhere)
    _move_out(skills / "v1-symlinked-resource" / "references" / "fees.md", elsewhere)
    (skills / "v4-exec-bit" / "scripts" / "compute_refund.py").chmod(0o644)


def materialize(tmp_path: Path, name: str) -> Path:
    """A working copy of fixture project ``name`` under ``tmp_path``, with its dot names."""
    dest = tmp_path / name
    shutil.copytree(DEFECTS / name, dest, ignore=shutil.ignore_patterns("__pycache__"))
    for stored, real in (("agents", ".agents"), ("decimal", ".decimal")):
        if (dest / stored).is_dir():
            (dest / stored).rename(dest / real)
    if POSIX:
        _set_modes(dest)
    if name == "managed" and POSIX:
        _setup_managed(dest, tmp_path / "elsewhere")
    return dest


@pytest.fixture
def clean(tmp_path) -> Path:
    return materialize(tmp_path, "clean")


@pytest.fixture(scope="module")
def managed_dir(tmp_path_factory) -> Path:
    return materialize(tmp_path_factory.mktemp("verify"), "managed")


@pytest.fixture(scope="module")
def managed(managed_dir) -> skillpkg.Result:
    return V.verify(directory=managed_dir)


@pytest.fixture(scope="module")
def managed_strict(managed_dir) -> skillpkg.Result:
    return V.verify(directory=managed_dir, strict=True)


# ── helpers ──────────────────────────────────────────────────────────────────


def _findings(result) -> list:
    """The findings of ``result``, notices left out."""
    return [p for p in result.problems if isinstance(p, V.Finding) and p.severity != V.NOTICE]


def _notices(result) -> list:
    return [p for p in result.problems if isinstance(p, V.Finding) and p.severity == V.NOTICE]


def by_folder(result) -> Dict[Optional[str], Dict[str, str]]:
    """skill -> {rule: severity}; every finding about one folder under that folder's name."""
    out: Dict[Optional[str], Dict[str, str]] = defaultdict(dict)
    for f in _findings(result):
        severities = out[f.skill]
        # One rule never has two severities on one folder.
        assert severities.get(f.rule, f.severity) == f.severity, f
        severities[f.rule] = f.severity
    return dict(out)


def rules(result) -> set:
    return {f.rule for f in _findings(result)}


def _tree(path: Path) -> dict:
    out = {}
    for p in sorted(path.rglob("*")):
        st = os.lstat(p)
        if stat.S_ISLNK(st.st_mode):
            out[p.relative_to(path).as_posix()] = ("link", os.readlink(p))
        elif p.is_file():
            out[p.relative_to(path).as_posix()] = (p.read_bytes(), stat.S_IMODE(st.st_mode))
        else:
            out[p.relative_to(path).as_posix()] = ("dir",)
    return out


def _relock(project: Path, edit) -> dict:
    """Apply ``edit`` to the lock's entries as the engine would: ``lock_hash`` recomputed,
    ``plugin.json`` rewritten to match."""
    path = project / ".decimal" / "skills.lock"
    data = json.loads(path.read_text())
    edit(data["skills"])
    data["lock_hash"] = skillfmt.lock_hash(ROOT, data["skills"])
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    (project / ".agents" / ".claude-plugin" / "plugin.json").write_bytes(
        L.plugin_json_bytes(data["lock_hash"]))
    return data


# ── the rule table ───────────────────────────────────────────────────────────


class TestRules:
    def test_the_format_rules_are_skillfmts_own(self):
        # verify calls skillfmt; it never restates the format rules.
        for rule, (scope, what) in skillfmt.RULES.items():
            assert V.RULES[rule] == (scope, what)

    def test_every_rule_has_a_scope_and_a_description(self):
        for rule, (scope, what) in V.RULES.items():
            assert scope in ("loader", "sdk"), rule
            assert isinstance(what, str) and what.strip(), rule

    def test_the_folder_and_project_rules(self):
        assert {r: s for r, (s, _) in V.RULES.items() if r.startswith("V")} == {
            "V1": "loader", "V2": "loader", "V3": "sdk", "V4": "sdk", "V5": "loader",
            "V6": "sdk", "V7": "sdk", "V8": "sdk", "V9": "sdk",
        }

    def test_every_rule_is_reported_by_some_fixture(self):
        reported = set().union(*MANAGED.values(), *(set(r) for r in UNMANAGED.values()),
                               *(set(r) for r in AS_PATH.values()),
                               *(r for r, _ in PROJECTS.values()))
        assert reported == set(V.RULES), (
            f"rules without a fixture: {sorted(set(V.RULES) - reported)}")


# ── managed folders: every rule is an error ──────────────────────────────────


@needs_posix
class TestManagedFolders:
    @pytest.mark.parametrize("folder", sorted(MANAGED))
    def test_each_folder_reports_exactly_its_rules(self, managed, folder):
        assert set(by_folder(managed).get(folder, {})) == MANAGED[folder]

    @pytest.mark.parametrize("folder", sorted(set(MANAGED) - {"acme-refund-policy"}))
    def test_every_finding_is_an_error_whatever_its_scope(self, managed, folder):
        found = [f for f in _findings(managed) if f.skill == folder]
        assert found
        for f in found:
            assert f.severity == ERROR and f.managed, f
            assert f.scope == V.RULES[f.rule][0]

    def test_the_sdk_rules_are_errors_here(self, managed):
        sdk = {f.rule for f in _findings(managed) if f.managed and f.scope == "sdk"}
        assert {"F2", "F6b", "F7", "F11", "F12", "F13", "F14", "V3", "V4"} <= sdk
        assert all(f.severity == ERROR for f in _findings(managed) if f.managed)

    def test_strict_changes_nothing_for_a_managed_folder(self, managed, managed_strict):
        def only_managed(result):
            return [f.to_dict() for f in _findings(result) if f.managed]
        assert only_managed(managed_strict) == only_managed(managed)

    def test_the_project_fails_and_says_where(self, managed):
        assert managed.exit_code == E.ExitCode.PROBLEMS
        assert not managed.ok
        f4 = next(f for f in _findings(managed) if f.rule == "F4")
        assert f4.path == f"{ROOT}/f4-empty-description/SKILL.md"
        assert "description" in f4.message

    def test_the_project_level_rules_hold_there(self, managed):
        assert not rules(managed) & set(PROJECT_RULES)

    def test_v3_names_each_file(self, managed):
        v3 = {f.path for f in _findings(managed) if f.rule == "V3"}
        assert v3 == {
            f"{ROOT}/v3-edited-attachment/references/fees.md",
            f"{ROOT}/f14-top-level-file/notes.md",
            f"{ROOT}/f15-nested-skill-md/references/SKILL.md",
        }

    def test_v4_names_the_script(self, managed):
        (v4,) = [f for f in _findings(managed) if f.rule == "V4"]
        assert v4.path == f"{ROOT}/v4-exec-bit/scripts/compute_refund.py"
        assert "executable" in v4.message

    def test_v1_names_each_link(self, managed):
        v1 = {f.path for f in _findings(managed) if f.rule == "V1"}
        assert v1 == {f"{ROOT}/v1-symlinked-folder",
                      f"{ROOT}/v1-symlinked-resource/references/fees.md"}

    def test_a_restore_command_is_offered(self, managed):
        v3 = next(f for f in _findings(managed) if f.rule == "V3")
        assert "decimalai skills install" in v3.fix


# ── unmanaged folders: loader rules are errors, SDK rules warnings ───────────


@needs_posix
class TestUnmanagedFolders:
    @pytest.mark.parametrize("folder", sorted(UNMANAGED))
    def test_each_folder_reports_its_rules_at_its_severity(self, managed, folder):
        assert by_folder(managed).get(folder, {}) == UNMANAGED[folder]

    @pytest.mark.parametrize("folder", sorted(UNMANAGED))
    def test_strict_makes_every_rule_an_error(self, managed_strict, folder):
        assert by_folder(managed_strict).get(folder, {}) == {
            rule: ERROR for rule in UNMANAGED[folder]}

    def test_findings_on_unmanaged_folders_say_so(self, managed):
        for f in _findings(managed):
            assert f.managed == (f.skill in MANAGED), f

    def test_warnings_alone_pass_and_strict_fails(self, tmp_path, clean):
        shutil.copytree(materialize(tmp_path / "m", "managed") / ROOT / "hand-authored-unknown-key",
                        clean / ROOT / "hand-authored-unknown-key")
        result = V.verify(directory=clean)
        assert by_folder(result) == {"hand-authored-unknown-key": {"F2": WARNING}}
        assert result.exit_code == E.ExitCode.OK and result.ok
        assert V.verify(directory=clean, strict=True).exit_code == E.ExitCode.PROBLEMS

    def test_companion_files_are_never_a_finding(self, managed_dir):
        folder = managed_dir / ROOT / "hand-authored-companion-files"
        for strict in (False, True):
            assert V.verify(folder, strict=strict).problems == ()
            assert by_folder(V.verify(directory=managed_dir, strict=strict)).get(
                "hand-authored-companion-files") is None

    @needs_posix
    def test_only_an_unsafe_path_is_a_finding(self, tmp_path):
        folder = tmp_path / "my-skill"
        (folder / "references").mkdir(parents=True)
        (folder / "SKILL.md").write_text("---\nname: my-skill\ndescription: Do it.\n---\nBody.\n")
        (folder / "top\\level.md").write_text("x\n")          # beside SKILL.md: not graded
        (folder / "references" / "a\\b.md").write_text("x\n")  # a backslash: unsafe
        (folder / "notes").mkdir()
        (folder / "notes" / "SKILL.md").write_text("x\n")       # a second skill to a loader
        found = {(f.rule, f.path.rsplit("/my-skill/", 1)[1], f.severity)
                 for f in _findings(V.verify(folder))}
        assert found == {("F14", "references/a\\b.md", WARNING),
                         ("F15", "notes/SKILL.md", ERROR)}

    def test_a_managed_folder_keeps_the_whole_format(self, managed):
        assert by_folder(managed)["f14-top-level-file"] == {"F14": ERROR, "V3": ERROR}

    def test_the_stamp_names_the_skill_it_claims(self, managed):
        (v8,) = [f for f in _findings(managed) if f.rule == "V8"]
        assert v8.skill == "v8-stamped-orphan"
        assert "decimalai-skill-id" in v8.message

    def test_a_skill_nested_too_deep_is_named(self, managed):
        (v2,) = [f for f in _findings(managed) if f.rule == "V2"]
        assert v2.skill == "v2-nested-folder"
        assert "refund-policy/SKILL.md" in v2.message


# ── verify PATH: any tree, by the unmanaged rules ────────────────────────────


@needs_posix
class TestVerifyPath:
    @pytest.mark.parametrize("folder", sorted(AS_PATH))
    def test_one_folder(self, managed_dir, folder):
        result = V.verify(managed_dir / ROOT / folder)
        found = {rule: sev for sevs in by_folder(result).values() for rule, sev in sevs.items()}
        assert found == AS_PATH[folder]
        assert not any(f.managed for f in _findings(result))
        errors = any(sev == ERROR for sev in found.values())
        assert result.exit_code == (E.ExitCode.PROBLEMS if errors else E.ExitCode.OK)

    @pytest.mark.parametrize("folder", sorted(f for f, r in AS_PATH.items() if r))
    def test_one_folder_strict(self, managed_dir, folder):
        result = V.verify(managed_dir / ROOT / folder, strict=True)
        found = {rule: sev for sevs in by_folder(result).values() for rule, sev in sevs.items()}
        assert found == {rule: ERROR for rule in AS_PATH[folder]}
        assert result.exit_code == E.ExitCode.PROBLEMS

    def test_the_skill_md_file_itself(self, managed_dir):
        folder = managed_dir / ROOT / "f4-empty-description"
        assert rules(V.verify(folder / "SKILL.md")) == {"F4"}

    def test_a_root_grades_every_folder_in_it(self, managed_dir):
        result = V.verify(managed_dir / ROOT)
        found = by_folder(result)
        want = {folder: sevs for folder, sevs in AS_PATH.items() if sevs}
        want["v2-nested-folder"] = {"V2": ERROR}  # its skill is one level too deep here
        assert found == want
        assert not rules(result) & {"F11", "F12", "F13", "V3", "V4", "V8", *PROJECT_RULES}

    def test_a_path_inside_a_project_ignores_its_lock(self, clean):
        folder = clean / ROOT / "acme-refund-policy"
        (folder / "references" / "fees.md").write_text("edited\n")
        (folder / "scripts" / "compute_refund.py").chmod(0o644)
        assert V.verify(folder).problems == ()
        assert rules(V.verify(directory=clean)) == {"V3", "V4"}

    def test_a_relative_path_is_taken_from_the_directory(self, managed_dir):
        assert ([f.to_dict() for f in V.verify(ROOT, directory=managed_dir).problems]
                == [f.to_dict() for f in V.verify(managed_dir / ROOT).problems])

    def test_paths_are_reported_under_the_path_given(self, managed_dir):
        given = managed_dir / ROOT
        f4 = next(f for f in _findings(V.verify(given)) if f.rule == "F4")
        assert f4.path == (given / "f4-empty-description" / "SKILL.md").as_posix()

    def test_hidden_folders_and_clutter_are_skipped(self, tmp_path, clean):
        root = clean / ROOT
        (root / ".DS_Store").write_bytes(b"\x00\x01")
        (root / "acme-refund-policy" / "__pycache__").mkdir()
        (root / "acme-refund-policy" / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"\0")
        (root / "acme-refund-policy" / "Thumbs.db").write_bytes(b"\0")
        hidden = root / ".git" / "broken"
        hidden.mkdir(parents=True)
        (hidden / "SKILL.md").write_text("no frontmatter\n")
        assert V.verify(root).problems == ()

    def test_files_beside_the_skill_folders_are_fine(self, clean):
        (clean / ROOT / "README.md").write_text("# Our skills\n")
        assert V.verify(clean / ROOT).problems == ()

    def test_a_tree_with_no_skill_fails(self, tmp_path):
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "README.md").write_text("hi\n")
        result = V.verify(tmp_path)
        assert rules(result) == {"V2"}
        assert result.exit_code == E.ExitCode.PROBLEMS

    def test_a_missing_path_is_a_usage_error(self, tmp_path):
        result = V.verify(tmp_path / "nope")
        assert result.exit_code == E.ExitCode.USAGE
        assert isinstance(result.problems[0], E.UsageError)

    def test_a_file_that_is_not_skill_md_is_a_usage_error(self, tmp_path):
        (tmp_path / "notes.md").write_text("x\n")
        assert V.verify(tmp_path / "notes.md").exit_code == E.ExitCode.USAGE

    def test_a_symlinked_path_is_not_followed(self, managed_dir):
        result = V.verify(managed_dir / ROOT / "v1-symlinked-folder")
        assert rules(result) == {"V1"}

    def test_the_folder_name_is_the_name_to_match(self, tmp_path):
        folder = tmp_path / "my-skill"
        folder.mkdir()
        (folder / "SKILL.md").write_text(textwrap.dedent("""\
            ---
            name: other-name
            description: Do the thing.
            ---
            Body.
            """))
        (f3,) = _findings(V.verify(folder))
        assert f3.rule == "F3" and "differs from its folder" in f3.message


# ── project-level checks ─────────────────────────────────────────────────────


class TestProjects:
    @pytest.mark.parametrize("name", sorted(PROJECTS))
    def test_each_project_reports_its_rule(self, tmp_path, name):
        want, exit_code = PROJECTS[name]
        result = V.verify(directory=materialize(tmp_path, name))
        assert rules(result) == want
        assert result.exit_code == exit_code
        assert all(f.severity == ERROR for f in _findings(result))

    def test_clean_is_clean_even_strict(self, clean):
        for strict in (False, True):
            result = V.verify(directory=clean, strict=strict)
            assert result.problems == () and result.ok
            assert result.changes == ()

    def test_v5_says_what_plugin_json_should_be(self, tmp_path):
        project = materialize(tmp_path, "plugin-json-stale")
        lock = json.loads((project / ".decimal" / "skills.lock").read_text())
        (v5,) = _findings(V.verify(directory=project))
        assert v5.path == ".agents/.claude-plugin/plugin.json"
        assert f"lock-{lock['lock_hash'][:12]}" in v5.message
        assert "decimalai skills install" in v5.fix

    @pytest.mark.parametrize("content", [
        None,                                                    # missing
        b"{not json\n",
        b'{"name":"decimalai","version":"lock-000000000000"}\n',  # the marketplace name
        b"[]\n",
    ])
    def test_v5_other_shapes(self, clean, content):
        plugin = clean / ".agents" / ".claude-plugin" / "plugin.json"
        lock = json.loads((clean / ".decimal" / "skills.lock").read_text())
        if content is None:
            plugin.unlink()
        else:
            plugin.write_bytes(content.replace(b"000000000000", lock["lock_hash"][:12].encode()))
        assert rules(V.verify(directory=clean)) == {"V5"}

    def test_plugin_json_is_optional_when_nothing_is_installed(self, clean):
        shutil.rmtree(clean / ROOT / "acme-refund-policy")
        _relock(clean, lambda skills: skills.clear())
        (clean / ".agents" / ".claude-plugin" / "plugin.json").unlink()
        (clean / ".decimal" / "skills.toml").write_text('root = ".agents/skills"\n\n[skills]\n')
        assert V.verify(directory=clean).problems == ()

    def test_v6_lock_hash_names_both_hashes(self, tmp_path):
        project = materialize(tmp_path, "lock-hash-edited")
        (v6,) = _findings(V.verify(directory=project))
        assert v6.path == ".decimal/skills.lock"
        assert "lock_hash" in v6.message and v6.fix == E.LockHashMismatch.fix

    def test_v6_foreign_entries_leave_their_folder_unmanaged(self, tmp_path):
        project = materialize(tmp_path, "foreign-entry")
        desk = project / ROOT / "acme-returns-desk" / "SKILL.md"
        desk.write_text(desk.read_text().replace("name: acme-returns-desk", "name: Acme"))
        result = V.verify(directory=project)
        (v6,) = [f for f in _findings(result) if f.rule == "V6"]
        assert "acme/returns-desk" in v6.message and v6.fix == E.ForeignLockEntriesError.fix
        f3 = [f for f in _findings(result) if f.rule == "F3"]
        assert f3 and all(f.skill == "acme-returns-desk" and not f.managed for f in f3)

    def test_v7_names_each_disagreement(self, tmp_path):
        project = materialize(tmp_path, "manifest-disagrees")
        v7 = sorted((f.skill, f.message) for f in _findings(V.verify(directory=project)))
        assert [skill for skill, _ in v7] == ["acme/refund-policy", "acme/returns-desk"]
        assert "4" in v7[0][1] and "3" in v7[0][1]
        assert "not installed" in v7[1][1]

    def test_v7_a_locked_skill_the_manifest_does_not_want(self, clean):
        (clean / ".decimal" / "skills.toml").write_text('root = ".agents/skills"\n\n[skills]\n')
        (v7,) = _findings(V.verify(directory=clean))
        assert v7.rule == "V7" and v7.skill == "acme/refund-policy"

    def test_v7_an_exact_pin_must_match(self, clean):
        manifest = clean / ".decimal" / "skills.toml"
        manifest.write_text(manifest.read_text().replace('"latest"', '"3"'))
        assert rules(V.verify(directory=clean)) == {"V7"}  # exact in the manifest, latest locked
        _relock(clean, lambda skills: skills["acme-refund-policy"].update(pin="exact"))
        assert V.verify(directory=clean).problems == ()
        manifest.write_text(manifest.read_text().replace('"3"', '"2"'))
        assert rules(V.verify(directory=clean)) == {"V7"}

    def test_v7_a_manifest_matched_by_id(self, clean):
        manifest = clean / ".decimal" / "skills.toml"
        lock = json.loads((clean / ".decimal" / "skills.lock").read_text())
        sid = lock["skills"]["acme-refund-policy"]["skill_id"]
        manifest.write_text(manifest.read_text().replace(
            '"acme/refund-policy" = "latest"',
            f'"acme/renamed-policy" = {{ version = "latest", id = "{sid}" }}'))
        assert V.verify(directory=clean).problems == ()

    def test_v7_an_id_is_matched_by_id_alone(self, clean):
        # As `install --frozen` matches: an id the lock does not hold is not installed,
        # even when the name is, and the locked skill is then not listed.
        manifest = clean / ".decimal" / "skills.toml"
        manifest.write_text(manifest.read_text().replace(
            '"acme/refund-policy" = "latest"',
            '"acme/refund-policy" = { version = "latest", id = "another-id" }'))
        v7 = [f.message for f in _findings(V.verify(directory=clean)) if f.rule == "V7"]
        assert len(v7) == 2
        assert any("not installed" in m for m in v7) and any("does not list it" in m for m in v7)

    @pytest.mark.parametrize("content", [None, b"root = [\n", b'root = ".claude/skills"\n'])
    def test_v7_a_missing_or_unreadable_manifest(self, clean, content):
        manifest = clean / ".decimal" / "skills.toml"
        if content is None:
            manifest.unlink()
        else:
            manifest.write_bytes(content)
        result = V.verify(directory=clean)
        assert rules(result) == {"V7"}
        assert result.exit_code == E.ExitCode.PROBLEMS

    def test_v9_names_the_interrupted_skill(self, tmp_path):
        project = materialize(tmp_path, "interrupted")
        (v9,) = _findings(V.verify(directory=project))
        assert v9.skill == "acme-refund-policy"
        assert v9.path == ".decimal/skills.lock.journal"

    def test_v9_an_unreadable_journal(self, clean):
        (clean / ".decimal" / "skills.lock.journal").write_text("{")
        assert rules(V.verify(directory=clean)) == {"V9"}

    def test_a_missing_folder(self, clean):
        shutil.rmtree(clean / ROOT / "acme-refund-policy")
        (v3,) = _findings(V.verify(directory=clean))
        assert v3.rule == "V3" and v3.skill == "acme-refund-policy"
        assert "decimalai skills install" in v3.fix

    def test_a_missing_root(self, clean):
        shutil.rmtree(clean / ROOT)
        assert {(f.rule, f.skill) for f in _findings(V.verify(directory=clean))} == {
            ("V3", "acme-refund-policy")}

    def test_a_missing_and_an_extra_file(self, clean):
        folder = clean / ROOT / "acme-refund-policy"
        (folder / "references" / "fees.md").unlink()
        (folder / "references" / "faq.md").write_text("Q\n")
        assert {(f.rule, f.path) for f in _findings(V.verify(directory=clean))} == {
            ("V3", f"{ROOT}/acme-refund-policy/references/fees.md"),
            ("V3", f"{ROOT}/acme-refund-policy/references/faq.md"),
        }

    def test_an_unavailable_file_may_be_present_or_absent(self, clean):
        binary = {"path": "assets/logo.png", "reason": "binary attachments are not stored"}
        _relock(clean, lambda skills: skills["acme-refund-policy"].update(unavailable=[binary]))
        assert V.verify(directory=clean).problems == ()
        (clean / ROOT / "acme-refund-policy" / "assets").mkdir()
        (clean / ROOT / "acme-refund-policy" / "assets" / "logo.png").write_bytes(b"\x89PNG")
        assert V.verify(directory=clean).problems == ()

    def test_os_clutter_and_partial_copies_are_ignored(self, clean):
        root = clean / ROOT
        (root / ".DS_Store").write_bytes(b"\0")
        (root / "acme-refund-policy" / ".DS_Store").write_bytes(b"\0")
        (root / "acme-refund-policy" / "scripts" / "__pycache__").mkdir()
        (root / "acme-refund-policy" / "scripts" / "__pycache__" / "c.pyc").write_bytes(b"\0")
        (root / "acme-refund-policy" / "desktop.ini").write_bytes(b"\0")
        (root / ".acme-refund-policy.partial").mkdir()
        assert V.verify(directory=clean).problems == ()

    def test_a_file_in_the_root(self, clean):
        (clean / ROOT / "README.md").write_text("# skills\n")
        (v2,) = _findings(V.verify(directory=clean))
        assert v2.rule == "V2" and v2.severity == ERROR

    def test_a_folder_without_skill_md(self, clean):
        (clean / ROOT / "empty").mkdir()
        (v2,) = _findings(V.verify(directory=clean))
        assert v2.rule == "V2" and v2.skill == "empty"

    @needs_posix
    @pytest.mark.parametrize("link", [".agents", ROOT, ".decimal", ".decimal/skills.lock"])
    def test_a_symlinked_state_path_stops_the_check(self, tmp_path, clean, link):
        _move_out(clean / link, tmp_path / "elsewhere")
        result = V.verify(directory=clean)
        assert rules(result) == {"V1"}
        assert result.exit_code == E.ExitCode.PROBLEMS

    @needs_posix
    def test_a_symlink_anywhere_under_decimal(self, tmp_path, clean):
        (tmp_path / "elsewhere").mkdir()
        os.symlink(tmp_path / "elsewhere", clean / ".decimal" / "backup")
        (v1,) = _findings(V.verify(directory=clean))
        assert v1.rule == "V1" and v1.path == ".decimal/backup"

    @needs_posix
    def test_a_symlinked_plugin_json(self, tmp_path, clean):
        _move_out(clean / ".agents" / ".claude-plugin" / "plugin.json", tmp_path / "elsewhere")
        assert rules(V.verify(directory=clean)) == {"V1"}

    def test_the_root_is_a_file(self, clean):
        shutil.rmtree(clean / ROOT)
        (clean / ROOT).write_text("not a folder\n")
        result = V.verify(directory=clean)
        assert rules(result) == {"V2"}

    def test_a_folder_renamed_by_case(self, clean):
        (clean / ROOT / "acme-refund-policy").rename(clean / ROOT / "Acme-Refund-Policy")
        assert by_folder(V.verify(directory=clean)) == {
            "Acme-Refund-Policy": {"V2": ERROR}, "acme-refund-policy": {"V2": ERROR}}

    def test_names_that_differ_only_by_case(self, clean):
        if (clean / ROOT / "ACME-REFUND-POLICY").exists():
            pytest.skip("a case-insensitive filesystem holds one folder for both names")
        shutil.copytree(clean / ROOT / "acme-refund-policy", clean / ROOT / "Acme-Refund-Policy")
        found = by_folder(V.verify(directory=clean))
        # On a macOS or Windows checkout they are one folder: neither can be trusted.
        assert found == {"Acme-Refund-Policy": {"V2": ERROR}, "acme-refund-policy": {"V2": ERROR}}


# ── refusals: a lock verify cannot read ──────────────────────────────────────


class TestRefusals:
    def test_a_version_1_lock(self, tmp_path):
        result = V.verify(directory=materialize(tmp_path, "lock-v1"))
        assert result.exit_code == E.ExitCode.LOCK == 3
        (err,) = result.problems
        assert isinstance(err, E.LegacyLockError)
        assert "decimalai skills migrate" in err.fix

    def test_an_npx_lock(self, clean):
        (clean / ".decimal" / "skills.lock.json").write_text('{"version": 1, "skills": {}}\n')
        result = V.verify(directory=clean)
        assert result.exit_code == E.ExitCode.LOCK
        assert isinstance(result.problems[0], E.LegacyLockError)

    @pytest.mark.parametrize("content, error", [
        (b"{", E.SkillsLockError),
        (b'{"version": 3, "root": ".agents/skills", "skills": {}}', E.SkillsLockError),
        (b'{"version": 2, "root": ".claude/skills", "skills": {}}', E.SkillsLockError),
        (b'{"skills": {}}', E.LegacyLockError),
    ])
    def test_an_unreadable_lock(self, clean, content, error):
        (clean / ".decimal" / "skills.lock").write_bytes(content)
        result = V.verify(directory=clean)
        assert result.exit_code == E.ExitCode.LOCK
        assert type(result.problems[0]) is error

    def test_a_lock_entry_the_engine_cannot_use(self, clean):
        _relock(clean, lambda skills: skills["acme-refund-policy"].update(path="../outside"))
        result = V.verify(directory=clean)
        assert result.exit_code == E.ExitCode.LOCK
        assert isinstance(result.problems[0], E.SkillsLockError)

    def test_neither_a_lock_nor_a_root(self, clean):
        (clean / ".decimal" / "skills.lock").unlink()
        shutil.rmtree(clean / ".agents")
        result = V.verify(directory=clean)
        assert result.exit_code == E.ExitCode.LOCK
        (err,) = result.problems
        assert isinstance(err, E.SkillsNotInstalledError)
        assert "decimalai skills install" in err.fix
        # Skill folders kept elsewhere are checked with a PATH, no lock needed.
        assert "decimalai skills verify <path>" in err.fix

    def test_the_lock_is_never_rewritten(self, tmp_path):
        project = materialize(tmp_path, "lock-v1")
        before = _tree(project)
        V.verify(directory=project)
        assert _tree(project) == before


# ── no lock: hand-authored skills in the root ────────────────────────────────


class TestNoLock:
    def test_the_root_is_graded_as_unmanaged(self, tmp_path):
        result = V.verify(directory=materialize(tmp_path, "no-lock"))
        assert {k: v for k, v in by_folder(result).items()} == {
            "hand-authored-unknown-key": {"F2": WARNING}}
        assert result.exit_code == E.ExitCode.OK
        (notice,) = _notices(result)
        assert notice.rule == V.NO_LOCK and not notice.is_error
        assert "no lock" in notice.message.replace("`.decimal/skills.lock`", "lock")
        assert "decimalai skills add" in notice.message
        assert result.problems[0] is notice  # the notice comes first

    def test_strict_fails_on_the_findings_alone(self, tmp_path):
        result = V.verify(directory=materialize(tmp_path, "no-lock"), strict=True)
        assert by_folder(result) == {"hand-authored-unknown-key": {"F2": ERROR}}
        assert result.exit_code == E.ExitCode.PROBLEMS
        assert [n.severity for n in _notices(result)] == [V.NOTICE]

    def test_a_loader_error_fails_it(self, tmp_path):
        project = materialize(tmp_path, "no-lock")
        md = project / ROOT / "hand-authored-clean" / "SKILL.md"
        md.write_text(md.read_text().replace("name: hand-authored-clean", "name: Clean"))
        result = V.verify(directory=project)
        assert set(by_folder(result)["hand-authored-clean"]) == {"F3"}
        assert result.exit_code == E.ExitCode.PROBLEMS

    def test_a_stamped_copy_is_not_an_orphan_without_a_lock(self, tmp_path):
        result = V.verify(directory=materialize(tmp_path, "no-lock"))
        assert "pulled-skill" not in by_folder(result)  # no V8: the notice says it once

    def test_nothing_is_compared_with_a_lock_that_is_not_there(self, clean):
        # A lock deleted from a project: its folders, plugin.json and manifest stay.
        (clean / ".decimal" / "skills.lock").unlink()
        folder = clean / ROOT / "acme-refund-policy"
        (folder / "references" / "fees.md").write_text("edited\n")
        (clean / ".agents" / ".claude-plugin" / "plugin.json").write_text("{}\n")
        (clean / ".decimal" / "skills.lock.journal").write_text("{")
        result = V.verify(directory=clean)
        assert _findings(result) == [] and len(_notices(result)) == 1
        assert result.exit_code == E.ExitCode.OK

    def test_the_root_rules_still_hold(self, clean):
        (clean / ".decimal" / "skills.lock").unlink()
        (clean / ROOT / "README.md").write_text("# skills\n")
        (clean / ROOT / "empty").mkdir()
        assert {(f.rule, f.skill) for f in _findings(V.verify(directory=clean))} == {
            ("V2", "README.md"), ("V2", "empty")}

    @needs_posix
    def test_a_symlinked_root(self, tmp_path):
        project = materialize(tmp_path, "no-lock")
        _move_out(project / ROOT, tmp_path / "elsewhere")
        result = V.verify(directory=project)
        assert rules(result) == {"V1"} and len(_notices(result)) == 1
        assert result.exit_code == E.ExitCode.PROBLEMS

    def test_a_legacy_lock_still_refuses(self, tmp_path):
        assert V.verify(directory=materialize(tmp_path, "lock-v1")).exit_code == E.ExitCode.LOCK

    def test_the_notice_is_json(self, tmp_path):
        data = V.verify(directory=materialize(tmp_path, "no-lock")).to_dict()
        assert data["ok"] is True
        assert data["problems"][0]["severity"] == "notice"
        assert data["problems"][0]["rule"] == V.NO_LOCK


# ── unreadable files: a finding, never a crash ───────────────────────────────


@pytest.fixture
def unreadable():
    """``unreadable(path)`` chmods ``path`` to 000, and puts its mode back afterwards."""
    held = []

    def hide(path: Path) -> Path:
        held.append((path, stat.S_IMODE(os.lstat(path).st_mode)))
        path.chmod(0)
        return path

    yield hide
    for path, mode in reversed(held):
        path.chmod(mode)


@needs_unreadable
class TestUnreadable:
    def test_a_file_in_a_managed_folder(self, clean, unreadable):
        unreadable(clean / ROOT / "acme-refund-policy" / "references" / "fees.md")
        result = V.verify(directory=clean)
        (v3,) = _findings(result)
        assert (v3.rule, v3.severity, v3.skill) == ("V3", ERROR, "acme-refund-policy")
        assert v3.path == f"{ROOT}/acme-refund-policy/references/fees.md"
        assert v3.message.startswith("unreadable: ")
        assert result.exit_code == E.ExitCode.PROBLEMS

    def test_skill_md_of_a_managed_folder(self, clean, unreadable):
        unreadable(clean / ROOT / "acme-refund-policy" / "SKILL.md")
        (v3,) = _findings(V.verify(directory=clean))
        assert v3.rule == "V3" and v3.message.startswith("unreadable: ")
        assert v3.path == f"{ROOT}/acme-refund-policy/SKILL.md"

    def test_the_other_folders_are_still_graded(self, tmp_path, unreadable):
        project = materialize(tmp_path, "managed")
        unreadable(project / ROOT / "v3-edited-attachment" / "references" / "fees.md")
        unreadable(project / ROOT / "acme-refund-policy" / "scripts" / "compute_refund.py")
        found = by_folder(V.verify(directory=project))
        assert found.pop("v3-edited-attachment") == {"V3": ERROR}
        assert found.pop("acme-refund-policy") == {"V3": ERROR}
        want = {k: {r: ERROR for r in v} for k, v in MANAGED.items() if v}
        want.update({k: v for k, v in UNMANAGED.items() if v})
        del want["v3-edited-attachment"]
        assert found == want

    def test_the_rest_of_the_folder_is_still_compared(self, clean, unreadable):
        folder = clean / ROOT / "acme-refund-policy"
        unreadable(folder / "references" / "fees.md")
        (folder / "scripts" / "compute_refund.py").write_text("edited\n")
        assert {(f.rule, f.path.rsplit("/", 1)[1]) for f in _findings(V.verify(directory=clean))} \
            == {("V3", "fees.md"), ("V3", "compute_refund.py")}

    def test_a_file_that_turns_unreadable_after_the_probe(self, clean, unreadable,
                                                          monkeypatch):
        # Opened fine a moment ago, refused when it is hashed: still a finding, not a crash.
        monkeypatch.setattr(V, "_readable", lambda path: True)
        folder = clean / ROOT / "acme-refund-policy"
        unreadable(folder / "references" / "fees.md")
        entry = json.loads((clean / ".decimal" / "skills.lock").read_text())["skills"][
            "acme-refund-policy"]
        (v3,) = V.verify_folder(folder, entry=entry)
        assert v3.rule == "V3" and v3.message.startswith("unreadable: `references/fees.md`")

    def test_an_unreadable_folder_inside_a_managed_one(self, clean):
        refs = clean / ROOT / "acme-refund-policy" / "references"
        refs.chmod(0)
        try:
            result = V.verify(directory=clean)
        finally:
            refs.chmod(0o755)
        assert result.exit_code == E.ExitCode.PROBLEMS
        assert any("cannot be read" in f.message for f in _findings(result))

    def test_skill_md_of_an_unmanaged_folder(self, managed_dir, unreadable):
        md = unreadable(managed_dir / ROOT / "hand-authored-clean" / "SKILL.md")
        try:
            for result in (V.verify(md.parent), V.verify(directory=managed_dir)):
                f1 = [f for f in _findings(result) if f.skill == "hand-authored-clean"]
                assert [(f.rule, f.severity) for f in f1] == [("F1", ERROR)]
                assert "cannot be read" in f1[0].message
        finally:
            md.chmod(0o644)

    def test_an_attachment_of_an_unmanaged_folder_is_never_read(self, tmp_path, unreadable):
        project = materialize(tmp_path, "no-lock")
        unreadable(project / ROOT / "hand-authored-clean" / "reference.md")
        unreadable(project / ROOT / "pulled-skill" / "references" / "fees.md")
        result = V.verify(directory=project)
        assert by_folder(result) == {"hand-authored-unknown-key": {"F2": WARNING}}
        assert result.exit_code == E.ExitCode.OK

    def test_an_unreadable_lock_is_a_refusal(self, clean, unreadable):
        unreadable(clean / ".decimal" / "skills.lock")
        result = V.verify(directory=clean)
        assert result.exit_code == E.ExitCode.LOCK
        (err,) = result.problems
        assert isinstance(err, E.SkillsLockError) and "cannot be read" in err.message

    def test_an_unreadable_manifest(self, clean, unreadable):
        unreadable(clean / ".decimal" / "skills.toml")
        (v7,) = _findings(V.verify(directory=clean))
        assert v7.rule == "V7" and "cannot be read" in v7.message

    def test_an_unreadable_journal(self, clean, unreadable):
        journal = clean / ".decimal" / "skills.lock.journal"
        journal.write_text("{}\n")
        unreadable(journal)
        (v9,) = _findings(V.verify(directory=clean))
        assert v9.rule == "V9" and "cannot be read" in v9.message


# ── verify only reads ────────────────────────────────────────────────────────


class TestReadOnly:
    @pytest.mark.parametrize("name", ["managed", "interrupted", "foreign-entry",
                                      "lock-hash-edited", "plugin-json-stale", "no-lock"])
    def test_nothing_is_written(self, tmp_path, name):
        project = materialize(tmp_path, name)
        before = _tree(project)
        V.verify(directory=project)
        V.verify(directory=project, strict=True)
        V.verify(project / ROOT)
        assert _tree(project) == before  # no .skills.lck, no recovery, no repaired lock

    def test_importing_the_verifier_loads_no_http_client_or_framework(self):
        code = textwrap.dedent("""
            import json, sys
            import decimalai
            base = set(sys.modules)
            import decimalai.skillpkg.verifier
            new = set(sys.modules) - base
            bad = ("httpx", "requests", "urllib3", "aiohttp", "http.client", "urllib.request",
                   "langchain", "crewai", "agents", "google.adk", "pydantic_ai", "strands",
                   "decimalai.skillpkg.client")
            print(json.dumps(sorted(m for m in new
                                    if any(m == b or m.startswith(b + ".") for b in bad))))
        """)
        env = {k: v for k, v in os.environ.items() if not k.startswith("DECIMAL")}
        env.update({"DECIMAL_API_KEY": "", "DECIMAL_AUTO_TRACE": "", "DECIMAL_AUTOINIT": "false",
                    "DECIMALAI_SKIP_VERIFY": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               env=env, cwd=REPO_ROOT, timeout=120)
        assert child.returncode == 0, child.stderr
        assert json.loads(child.stdout.strip().splitlines()[-1]) == []


# ── the Python API ───────────────────────────────────────────────────────────


class TestApi:
    def test_the_export_is_the_function(self):
        assert skillpkg.verify is V.verify

    def test_the_default_project_is_found_from_the_working_directory(self, clean, monkeypatch):
        (clean / ".git").mkdir()
        monkeypatch.chdir(clean / ROOT / "acme-refund-policy")
        assert V.verify().ok
        (clean / ROOT / "acme-refund-policy" / "references" / "fees.md").write_text("x\n")
        assert rules(V.verify()) == {"V3"}

    def test_the_result_is_json(self, tmp_path):
        result = V.verify(directory=materialize(tmp_path, "manifest-disagrees"))
        data = json.loads(json.dumps(result.to_dict()))
        assert data["exit_code"] == 1 and data["ok"] is False and data["changes"] == []
        assert {tuple(sorted(p)) for p in data["problems"]} == {(
            "fix", "managed", "message", "path", "rule", "scope", "severity", "skill")}

    def test_a_refusal_is_json_too(self, tmp_path):
        result = V.verify(directory=materialize(tmp_path, "lock-v1"))
        (problem,) = result.to_dict()["problems"]
        assert problem["code"] == "legacy_lock" and problem["exit_code"] == 3

    def test_a_finding_reads_as_one_line(self):
        f = V.Finding(rule="F4", scope="loader", severity=ERROR, message="the description is empty",
                      skill="x", path=".agents/skills/x/SKILL.md")
        assert str(f) == "error F4 .agents/skills/x/SKILL.md: the description is empty"
        assert f.is_error
        assert not V.Finding("F2", "sdk", WARNING, "m").is_error

    def test_findings_come_in_a_stable_order(self, tmp_path, managed_dir):
        first = [f.to_dict() for f in V.verify(directory=managed_dir).problems]
        assert first == [f.to_dict() for f in V.verify(directory=managed_dir).problems]
        keys = [(f["skill"], f["path"]) for f in first]
        assert keys == sorted(keys)
        # Project-level findings first, then folder by folder.
        project = materialize(tmp_path, "manifest-disagrees")
        (project / ROOT / "acme-refund-policy" / "notes.md").write_text("x\n")
        order = [f.rule for f in V.verify(directory=project).problems]
        assert order == ["V7", "V7", "F14", "V3"]

    def test_verify_folder_grades_one_folder(self, managed_dir):
        lock = json.loads((managed_dir / ".decimal" / "skills.lock").read_text())
        folder = managed_dir / ROOT / "f11-stamp-differs"
        entry = lock["skills"]["f11-stamp-differs"]
        assert {f.rule for f in V.verify_folder(folder, entry=entry)} == {"F11"}
        assert V.verify_folder(folder) == []
        # A folder named as a skill must hold one.
        (v2,) = V.verify_folder(managed_dir / ROOT / "v2-nested-folder")
        assert v2.rule == "V2" and "refund-policy/SKILL.md" in v2.message


# ── the fixtures survive git ─────────────────────────────────────────────────


class TestFixtures:
    def _committed(self) -> Dict[str, bytes]:
        out = {}
        for path in sorted(DEFECTS.rglob("*")):
            rel = path.relative_to(DEFECTS).as_posix()
            if path.is_file() and rel != "make_defects.py" and "__pycache__" not in rel:
                out[rel] = path.read_bytes()
        return out

    def test_the_fixtures_are_what_the_generator_writes(self):
        committed, generated = self._committed(), _generator().build()
        lost = sorted(set(generated) - set(committed))
        assert not lost, (
            f"fixture files are missing: {lost}. A .gitignore pattern may have kept them out "
            "of the commit; rename them, then run make_defects.py")
        assert sorted(set(committed) - set(generated)) == [], "files the generator never writes"
        changed = sorted(p for p in generated if generated[p] != committed[p])
        assert not changed, f"fixtures differ from make_defects.py: {changed}"

    def test_every_expectation_names_a_fixture(self):
        folders = {p.name for p in (DEFECTS / "managed" / "agents" / "skills").iterdir()}
        assert folders == set(MANAGED) | set(UNMANAGED) == set(AS_PATH)
        projects = {p.name for p in DEFECTS.iterdir() if p.is_dir() and p.name != "__pycache__"}
        assert projects == set(PROJECTS) | {"managed", "lock-v1", "no-lock"}

    def test_no_fixture_path_is_one_git_ignores(self):
        paths = [f"tests/skillpkg/fixtures/defects/{rel}" for rel in self._committed()]
        try:
            child = subprocess.run(["git", "check-ignore", "--no-index", "--stdin"],
                                   input="\n".join(paths), capture_output=True, text=True,
                                   cwd=REPO_ROOT, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            child = None
        if child is None or child.returncode not in (0, 1):
            # No git, or not a checkout (an sdist): the names git is known to ignore.
            ignored = [p for p in paths
                       if any(part in (".agents", ".decimal", "__pycache__", ".DS_Store",
                                       ".claude", ".env", "build", "dist")
                              for part in p.split("/"))]
        else:
            ignored = child.stdout.split()
        assert ignored == [], f"git would not commit these fixture files: {ignored}"
