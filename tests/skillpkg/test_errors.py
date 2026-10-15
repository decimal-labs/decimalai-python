"""The skills engine's failure contract, and a package that costs nothing to import.

``decimalai skills`` commands report failures as exit codes that CI scripts and
Dockerfiles branch on (``install --frozen`` exiting 6 means "the registry served
different bytes", 8 means "network"). Those codes are documented, so they are pinned
here: changing one is a breaking change and has to change this file on purpose.

Every refusal also carries a ``fix``, the command that gets the user unstuck, and
names the skill it is about. The fixes are pinned word for word too, and the skill's
name is quoted in them, so a command pastes into a shell as it is printed.

``decimalai.skillpkg`` itself is a thin, lazy front door: importing it loads no
framework and no HTTP client, and each public function is imported from its own
module only when it is first used.
"""
from __future__ import annotations

import dataclasses
import doctest
import inspect
import json
import os
import pickle
import pkgutil
import shlex
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

import decimalai.skillpkg as skillpkg
from decimalai.skillpkg import errors as E

PACKAGE_DIR = Path(skillpkg.__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent

# ── The documented contract ──────────────────────────────────────────────────

EXIT_CODES = {
    "OK": 0,
    "PROBLEMS": 1,          # status --check / verify found problems; journal conflict
    "USAGE": 2,             # usage error (click's own on the CLI)
    "LOCK": 3,              # lock or manifest missing, unreadable, v1 or npx
    "LOCK_OUT_OF_DATE": 4,  # lock disagrees with the manifest (--frozen)
    "LOCAL_CHANGES": 5,     # local edits or an unmanaged folder block the operation
    "INTEGRITY": 6,         # served bytes differ from declared/locked hashes; lock_hash
    "UNAVAILABLE": 7,       # not found, ambiguous, withheld, blocked, unsafe, no access ...
    "NETWORK": 8,           # network failure, or --offline needed a fetch
}

# class name -> (exit code, stable code string, direct parent)
HIERARCHY = {
    "SkillpkgError": (1, "error", "Exception"),
    "JournalConflictError": (1, "journal_conflict", "SkillpkgError"),
    "SkillsBusyError": (1, "busy", "SkillpkgError"),
    "UsageError": (2, "usage", "SkillpkgError"),
    "SkillsLockError": (3, "lock_unreadable", "SkillpkgError"),
    "LegacyLockError": (3, "legacy_lock", "SkillsLockError"),
    "ForeignLockEntriesError": (3, "foreign_lock_entries", "SkillsLockError"),
    "ManifestError": (3, "manifest_invalid", "SkillpkgError"),
    "SkillsNotInstalledError": (3, "not_installed", "SkillpkgError"),
    "LockOutOfDateError": (4, "lock_out_of_date", "SkillpkgError"),
    "LocalChangesError": (5, "local_changes", "SkillpkgError"),
    "ModifiedFolderError": (5, "modified", "LocalChangesError"),
    "UnmanagedFolderError": (5, "unmanaged_folder", "LocalChangesError"),
    "IllegalEntryError": (5, "illegal_entry", "SkillpkgError"),
    "IntegrityError": (6, "integrity", "SkillpkgError"),
    "LockHashMismatch": (6, "lock_hash_mismatch", "IntegrityError"),
    "BundleContractError": (6, "bundle_invalid", "IntegrityError"),
    "UnavailableError": (7, "unavailable", "SkillpkgError"),
    "SkillNotFoundError": (7, "not_found", "UnavailableError"),
    "AmbiguousRefError": (7, "ambiguous", "UnavailableError"),
    "SkillWithheldError": (7, "withheld", "UnavailableError"),
    "SkillBlockedError": (7, "blocked", "UnavailableError"),
    "VersionUnsafeError": (7, "version_unsafe", "UnavailableError"),
    "InstallNameCollisionError": (7, "install_name_collision", "UnavailableError"),
    "RenderInvalidError": (7, "render_invalid", "UnavailableError"),
    "RenderGoneError": (7, "render_gone", "UnavailableError"),
    "LockedVersionRevokedError": (7, "revoked", "UnavailableError"),
    "AccessDeniedError": (7, "no_access", "UnavailableError"),
    "AlreadyInstalledError": (7, "already_installed", "UnavailableError"),
    "ConfirmationRequiredError": (7, "confirmation_required", "UnavailableError"),
    "SkillNotManagedError": (7, "not_managed", "UnavailableError"),
    "RegistryTooOldError": (7, "registry_too_old", "UnavailableError"),
    "NetworkError": (8, "network", "SkillpkgError"),
    "OfflineError": (8, "offline", "NetworkError"),
}

# class name -> its default fix, verbatim. The fix is the command a user pastes, so it is
# pinned like the exit code: a class that loses its own fix silently shows its parent's (a
# withheld skill would say "check your API key"), and a wrong command sends the user the
# wrong way. Changing a fix means changing its row here on purpose.
FIXES = {
    "SkillpkgError": "run `decimalai skills status` to see the state of the project",
    "JournalConflictError": (
        "move the skill's folder out of `.agents/skills/`, delete "
        "`.decimal/skills.lock.journal`, then run `decimalai skills install`"
    ),
    "SkillsBusyError": (
        "wait for the other `decimalai skills` command to finish; if none is running, "
        "delete `.decimal/.skills.lck`"
    ),
    "UsageError": "run `decimalai skills --help` for the accepted forms",
    "SkillsLockError": (
        "restore `.decimal/skills.lock` from version control, or delete it and run "
        "`decimalai skills install` to write a new one"
    ),
    "LegacyLockError": "run `decimalai skills migrate`",
    "ForeignLockEntriesError": (
        "run `decimalai skills install` to drop them (their folders stay, unmanaged), "
        "then commit `.decimal/skills.lock`"
    ),
    "ManifestError": (
        "correct the named line of `.decimal/skills.toml`, then run `decimalai skills install`"
    ),
    "SkillsNotInstalledError": (
        "run `decimalai skills install`, or `decimalai skills add <ref>` for a first skill"
    ),
    "LockOutOfDateError": (
        "run `decimalai skills install`, then commit `.decimal/skills.toml` and "
        "`.decimal/skills.lock`"
    ),
    "LocalChangesError": (
        "move your changes out of the way, or re-run with `--force` to replace them"
    ),
    "ModifiedFolderError": (
        "keep your edits by copying the folder to a new name (a skill of your own), "
        "or re-run with `--force` to replace them"
    ),
    "UnmanagedFolderError": (
        "rename or remove that folder (it is not in `.decimal/skills.lock`), "
        "or re-run with `--force` to replace it"
    ),
    "IllegalEntryError": (
        "replace it with a real directory, or move it out of `.agents/skills/`; "
        "`decimalai skills` never follows a symlink inside the project"
    ),
    "IntegrityError": (
        "nothing was written; if the change is expected, run `decimalai skills update "
        "{skill}` and review the new files before committing the lock"
    ),
    "LockHashMismatch": (
        "run `decimalai skills install` to recompute it, then review and commit "
        "`.decimal/skills.lock`"
    ),
    "BundleContractError": (
        "nothing was written; run `pip install -U decimalai` and retry, "
        "and report it if it persists"
    ),
    "UnavailableError": "check the reference and your API key (`DECIMAL_API_KEY`), then retry",
    "SkillNotFoundError": (
        "check the spelling, or pass the skill's url_slug or id; `decimalai skills list` "
        "shows your organization's skills"
    ),
    "AmbiguousRefError": "pass the full registry name (`owner/name`) or the skill's id",
    "SkillWithheldError": (
        "choose another skill, and run `decimalai skills remove {skill}` if it is installed"
    ),
    "SkillBlockedError": (
        "choose another skill, and run `decimalai skills remove {skill}` if it is installed"
    ),
    "VersionUnsafeError": (
        "install another version with `decimalai skills add {skill}@latest` or `@<N>`"
    ),
    "InstallNameCollisionError": (
        "rename or remove the folder that already uses the name, then retry; "
        "`decimalai skills status` lists the folders"
    ),
    "RenderInvalidError": (
        "try another version with `decimalai skills add {skill}@<N>`, "
        "or report it to the skill's owner"
    ),
    "RenderGoneError": (
        "run `decimalai skills update {skill}` to take the current render, "
        "then commit `.decimal/skills.lock`"
    ),
    "LockedVersionRevokedError": (
        "run `decimalai skills update {skill}` or `decimalai skills remove {skill}`"
    ),
    "AccessDeniedError": (
        "set `DECIMAL_API_KEY` to a key of an organization that can see this skill"
    ),
    "AlreadyInstalledError": (
        "run `decimalai skills status` to find its folder; `decimalai skills update "
        "<name>@<N>` changes its version"
    ),
    "ConfirmationRequiredError": "review the skill, then re-run with `--yes` to accept it",
    "SkillNotManagedError": "run `decimalai skills status` to list the installed skills",
    "RegistryTooOldError": (
        "check `--base-url` and `DECIMAL_BASE_URL`; the default registry is "
        "https://api.decimal.ai"
    ),
    "NetworkError": "check the connection and `--base-url`, then run the command again",
    "OfflineError": (
        "run it without `--offline`, or commit `.agents/skills` so the folders "
        "are there without a fetch"
    ),
}

# Registry names a shell would split, unbalance or expand if a fix pasted them bare.
# Registry names may contain spaces (thousands of public ones do) and quotes.
SHELL_HOSTILE_NAMES = (
    "composiohq/Docker Hub Automation",
    "o'brien/skill",
    'acme/say "hi"',
    "acme/$(touch pwned)",
    "acme/*",
    "acme/a;b",
    "日本語スキル",
)

# The per-item error codes the install-resolve and bundle routes answer with.
REGISTRY_CODES = (
    "not_found", "ambiguous", "withheld", "blocked", "version_unsafe",
    "install_name_collision", "render_invalid", "render_gone",
)

# Public name -> the module that defines it. A function never shares its name with
# a module of this package (see test_no_submodule_is_named_like_an_export).
EXPORTS = {
    "add": "ops",
    "install": "ops",
    "update": "ops",
    "remove": "ops",
    "status": "ops",
    "verify": "verifier",
    "migrate": "migration",
    "Result": "errors",
}

ERROR_CLASSES = [getattr(E, name) for name in HIERARCHY]

# The classes whose pinned fix names the skill in a command.
NAMED_FIX_CLASSES = [getattr(E, name) for name, fix in FIXES.items() if "{skill}" in fix]


# ── Exit codes ───────────────────────────────────────────────────────────────


class TestExitCodes:
    def test_values_are_pinned(self):
        assert {m.name: m.value for m in E.ExitCode} == EXIT_CODES

    def test_every_code_has_a_meaning(self):
        assert set(E.EXIT_CODE_MEANINGS) == set(E.ExitCode)
        for code, meaning in E.EXIT_CODE_MEANINGS.items():
            assert isinstance(meaning, str) and meaning.strip(), code

    def test_every_failure_code_is_raised_by_some_error(self):
        raised = {cls.exit_code for cls in ERROR_CLASSES}
        assert raised == set(E.ExitCode) - {E.ExitCode.OK}

    def test_the_code_reaches_the_process_exit_status(self):
        child = _run_child(
            "import sys\n"
            "from decimalai.skillpkg.errors import ExitCode, IntegrityError\n"
            "sys.exit(IntegrityError('served bytes differ', skill='acme').exit_code)\n"
        )
        assert child.returncode == 6, child.stderr


# ── The error hierarchy ──────────────────────────────────────────────────────


class TestHierarchy:
    @pytest.mark.parametrize("name", sorted(HIERARCHY))
    def test_exit_code_code_and_parent_are_pinned(self, name):
        cls = getattr(E, name)
        exit_code, code, parent = HIERARCHY[name]
        assert cls.exit_code == exit_code
        assert isinstance(cls.exit_code, E.ExitCode)
        assert cls.code == code
        assert cls.__bases__[0].__name__ == parent

    def test_every_error_class_is_pinned(self):
        defined = {
            name for name, obj in vars(E).items()
            if inspect.isclass(obj) and issubclass(obj, BaseException)
            and obj.__module__ == E.__name__
        }
        assert defined == set(HIERARCHY), (
            "a new error class needs a row in HIERARCHY: its exit code is part of "
            "the documented contract"
        )

    def test_codes_are_unique(self):
        codes = [cls.code for cls in ERROR_CLASSES]
        assert len(codes) == len(set(codes))

    def test_every_error_is_a_skillpkg_error(self):
        for cls in ERROR_CLASSES:
            assert issubclass(cls, E.SkillpkgError)

    def test_usage_error_is_also_a_value_error(self):
        assert issubclass(E.UsageError, ValueError)

    def test_exit_code_belongs_to_the_class_not_the_instance(self):
        with pytest.raises(TypeError):
            E.NetworkError("timed out", exit_code=3)  # type: ignore[call-arg]


class TestFixHints:
    @pytest.mark.parametrize("cls", ERROR_CLASSES, ids=lambda c: c.__name__)
    def test_each_class_has_a_fix_that_names_a_command(self, cls):
        fix = cls.fix
        assert isinstance(fix, str) and fix.strip()
        assert "`" in fix, f"{cls.__name__}.fix names no command or path: {fix!r}"
        assert fix[0].islower(), "a fix reads as the clause after the reason"
        assert not fix.endswith("."), fix

    def test_every_error_class_has_a_pinned_fix(self):
        assert set(FIXES) == set(HIERARCHY), (
            "a new error class needs a row in FIXES: its fix is the command users paste"
        )

    @pytest.mark.parametrize("name", sorted(HIERARCHY))
    def test_each_class_states_its_own_pinned_fix(self, name):
        cls = getattr(E, name)
        assert "fix" in vars(cls), (
            f"{name} has no fix of its own, so it shows its parent's: {cls.fix!r}"
        )
        assert cls.fix == FIXES[name]

    @pytest.mark.parametrize("cls", ERROR_CLASSES, ids=lambda c: c.__name__)
    def test_the_instance_fix_names_the_skill(self, cls):
        named = cls("something went wrong", skill="acme/refund-policy")
        anonymous = cls("something went wrong")
        assert "{skill}" not in named.fix and "{skill}" not in anonymous.fix
        if "{skill}" in cls.fix:
            assert "acme/refund-policy" in named.fix
            assert "<name>" in anonymous.fix
        # A plain name is printed as it is, and the placeholder is never quoted.
        assert named.fix == cls.fix.replace("{skill}", "acme/refund-policy")
        assert anonymous.fix == cls.fix.replace("{skill}", "<name>")

    @pytest.mark.parametrize("name", SHELL_HOSTILE_NAMES)
    @pytest.mark.parametrize("cls", NAMED_FIX_CLASSES, ids=lambda c: c.__name__)
    def test_a_fix_command_keeps_the_skill_one_shell_word(self, cls, name):
        # Each command a fix names must split into the same words as it does for a plain
        # name, with the whole registry name in the skill's place. Bare, a name with a
        # space gives the command extra arguments and a name with a quote does not parse.
        plain = _spans(cls("reason", skill=PLAIN_NAME).fix)
        hostile = _spans(cls("reason", skill=name).fix)
        assert len(hostile) == len(plain), hostile
        for plain_span, hostile_span in zip(plain, hostile):
            expected = [word.replace(PLAIN_NAME, name) for word in shlex.split(plain_span)]
            assert shlex.split(hostile_span) == expected, hostile_span

    @pytest.mark.skipif(os.name != "posix", reason="a fix quotes for a POSIX shell")
    @pytest.mark.parametrize("name", SHELL_HOSTILE_NAMES)
    @pytest.mark.parametrize("cls", NAMED_FIX_CLASSES, ids=lambda c: c.__name__)
    def test_a_pasted_fix_command_reaches_the_cli_intact(self, cls, name, tmp_path):
        # The real shell, with `decimalai` swapped for a printf of its arguments: nothing
        # in the name is run, globbed or split. `acme/` exists, so a bare `acme/*` expands.
        (tmp_path / "acme").mkdir()
        (tmp_path / "acme" / "file").write_text("")
        plain = _spans(cls("reason", skill=PLAIN_NAME).fix)
        hostile = _spans(cls("reason", skill=name).fix)
        pasted = 0
        for plain_span, hostile_span in zip(plain, hostile):
            if PLAIN_NAME not in plain_span:
                continue
            # The user fills in a `<N>` placeholder before pasting.
            plain_span, hostile_span = (s.replace("<N>", "3") for s in (plain_span, hostile_span))
            assert hostile_span.startswith("decimalai "), hostile_span
            shell = subprocess.run(
                ["/bin/sh", "-c", "printf '%s\\n' " + hostile_span[len("decimalai "):]],
                cwd=tmp_path, capture_output=True, timeout=30,
            )
            assert shell.returncode == 0, (hostile_span, shell.stderr)
            expected = [word.replace(PLAIN_NAME, name) for word in shlex.split(plain_span)]
            assert shell.stdout.decode("utf-8").splitlines() == expected[1:], hostile_span
            pasted += 1
        assert pasted, f"{cls.__name__}.fix has no pasteable command naming the skill"
        assert sorted(p.name for p in tmp_path.rglob("*")) == ["acme", "file"], (
            "the shell ran part of the name"
        )

    def test_the_module_example_holds(self):
        result = doctest.testmod(E)
        assert result.attempted and not result.failed

    def test_str_names_the_skill_the_reason_and_the_fix(self):
        err = E.LockedVersionRevokedError(
            "locked version 3 is no longer available (blocked)",
            skill="acme/refund-policy",
        )
        text = str(err)
        assert text.startswith("acme/refund-policy: locked version 3")
        assert "no longer available (blocked)" in text
        assert "decimalai skills update acme/refund-policy" in text
        assert "decimalai skills remove acme/refund-policy" in text

    def test_an_explicit_fix_replaces_the_default(self):
        err = E.ModifiedFolderError("2 files edited", skill="pdf", fix="run `x`")
        assert err.fix == "run `x`"
        assert str(err) == "pdf: 2 files edited; run `x`"

    def test_the_reason_stays_available_without_the_fix(self):
        err = E.LegacyLockError("`.decimal/skills.lock` is a version 1 lock")
        assert err.message == "`.decimal/skills.lock` is a version 1 lock"
        assert err.args == (err.message,)
        assert "decimalai skills migrate" in err.fix


class TestErrorData:
    def test_to_dict_is_json_ready(self):
        err = E.ForeignLockEntriesError(
            "2 entries were written by an older decimalai",
            path=Path(".decimal/skills.lock"),
            details={"entries": ["a", "b"]},
        )
        data = err.to_dict()
        assert json.loads(json.dumps(data)) == data
        assert data == {
            "error": "ForeignLockEntriesError",
            "code": "foreign_lock_entries",
            "exit_code": 3,
            "message": "2 entries were written by an older decimalai",
            "skill": None,
            "path": os.path.join(".decimal", "skills.lock"),
            "fix": err.fix,
            "details": {"entries": ["a", "b"]},
        }

    def test_details_are_copied(self):
        details = {"files": ["SKILL.md"]}
        err = E.ModifiedFolderError("edited", details=details)
        details["files"].append("x")
        details["other"] = 1
        assert "other" not in err.details

    @pytest.mark.parametrize("cls", ERROR_CLASSES, ids=lambda c: c.__name__)
    def test_errors_survive_pickling(self, cls):
        err = cls("reason", skill="acme", path="p", fix="run `y`", details={"k": [1]})
        back = pickle.loads(pickle.dumps(err))
        assert type(back) is cls
        assert back.to_dict() == err.to_dict()
        assert str(back) == str(err)


class TestRegistryCodes:
    @pytest.mark.parametrize("code", REGISTRY_CODES)
    def test_each_registry_code_has_its_own_class(self, code):
        err = E.UnavailableError.for_code(code, "the registry said no", skill="acme")
        assert type(err).code == code
        assert err.exit_code == E.ExitCode.UNAVAILABLE
        assert err.skill == "acme"
        assert err.message == "the registry said no"

    def test_an_unknown_registry_code_is_still_unavailable(self):
        err = E.UnavailableError.for_code(
            "brand_new_code", "new refusal", details={"near": ["acme"]},
        )
        assert type(err) is E.UnavailableError
        assert err.exit_code == 7
        assert err.details == {"near": ["acme"], "registry_code": "brand_new_code"}


# ── Result ───────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class _Problem:  # the shape of a verify finding
    rule: str
    scope: str
    message: str


class TestResult:
    def test_defaults_to_ok(self):
        result = E.Result()
        assert result.ok
        assert result.exit_code == 0 and isinstance(result.exit_code, E.ExitCode)
        assert result.changes == () and result.problems == ()

    def test_fields_are_positional_in_the_documented_order(self):
        result = E.Result(["added acme"], [], 0)
        assert result.changes == ("added acme",)

    def test_sequences_become_tuples_and_the_result_is_frozen(self):
        result = E.Result(changes=["a"], problems=[_Problem("F3", "loader", "bad")])
        assert isinstance(result.changes, tuple) and isinstance(result.problems, tuple)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.exit_code = 1  # type: ignore[misc]

    def test_an_undocumented_exit_code_is_refused(self):
        with pytest.raises(ValueError):
            E.Result(exit_code=9)

    def test_from_error_carries_the_error_and_its_code(self):
        err = E.LockOutOfDateError("acme is pinned to 3 but locked at 4", skill="acme")
        result = E.Result.from_error(err, changes=["kept pdf"])
        assert not result.ok
        assert result.exit_code == E.ExitCode.LOCK_OUT_OF_DATE
        assert result.problems == (err,)
        assert result.changes == ("kept pdf",)

    def test_to_dict_is_json_ready(self):
        err = E.SkillNotFoundError("no such skill", skill="acme")
        result = E.Result(
            changes=[{"name": "pdf", "op": "KEEP"}],
            problems=[_Problem("F3", "loader", "name differs from its folder"), err],
            exit_code=E.ExitCode.UNAVAILABLE,
        )
        data = result.to_dict()
        assert json.loads(json.dumps(data)) == data
        assert data["exit_code"] == 7 and data["ok"] is False
        assert data["changes"] == [{"name": "pdf", "op": "KEEP"}]
        assert data["problems"][0] == {
            "rule": "F3", "scope": "loader", "message": "name differs from its folder",
        }
        assert data["problems"][1] == err.to_dict()


# ── The lazy package ─────────────────────────────────────────────────────────


class TestLazyExports:
    def test_public_names(self):
        assert sorted(skillpkg.__all__) == sorted(EXPORTS)
        assert len(skillpkg.__all__) == len(set(skillpkg.__all__))

    def test_module_map_is_pinned(self):
        assert skillpkg._EXPORTS == EXPORTS

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match=r"decimalai\.skillpkg.*'no_such_name'"):
            skillpkg.no_such_name  # noqa: B018
        assert not hasattr(skillpkg, "no_such_name")
        with pytest.raises(ImportError):
            from decimalai.skillpkg import no_such_name  # noqa: F401

    def test_result_resolves_to_the_errors_module(self):
        assert skillpkg.Result is E.Result
        from decimalai.skillpkg import Result

        assert Result is E.Result

    def test_dir_lists_the_exports(self):
        assert set(EXPORTS) <= set(dir(skillpkg))

    def test_an_export_is_read_from_its_module_on_every_access(self, monkeypatch):
        stub = types.ModuleType("decimalai.skillpkg.ops")

        def first(*args, **kwargs):
            return "first"

        def second(*args, **kwargs):
            return "second"

        stub.add = first  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "decimalai.skillpkg.ops", stub)
        assert skillpkg.add is first
        stub.add = second  # type: ignore[attr-defined]
        assert skillpkg.add is second, "a monkeypatched owner module must win"

    @pytest.mark.parametrize("name", sorted(EXPORTS))
    def test_each_export_resolves_or_names_the_module_still_to_land(self, name):
        module = f"decimalai.skillpkg.{EXPORTS[name]}"
        if _module_file(EXPORTS[name]) is None:
            # The scaffold lands before the modules that define the operations.
            # Until each one exists, its names fail by naming exactly that module.
            with pytest.raises(ModuleNotFoundError) as excinfo:
                getattr(skillpkg, name)
            assert excinfo.value.name == module
            return
        obj = getattr(skillpkg, name)
        assert not inspect.ismodule(obj)
        assert obj is getattr(sys.modules[module], name)

    def test_no_submodule_is_named_like_an_export(self):
        submodules = {info.name for info in pkgutil.iter_modules([str(PACKAGE_DIR)])}
        clash = submodules & set(EXPORTS)
        homes = ", ".join(
            f"{name}() belongs in decimalai/skillpkg/{EXPORTS[name]}.py" for name in sorted(clash)
        )
        assert not clash, (
            f"decimalai.skillpkg has a module named like an export: {sorted(clash)}. "
            "Importing a submodule binds it on the package, so after the first import "
            "`skillpkg.<name>` would be the module, not the function, and calling it "
            f"raises TypeError. Move it: {homes} (the map in "
            "decimalai/skillpkg/__init__.py), and import it from there."
        )

    def test_exports_survive_importing_every_submodule(self):
        import importlib

        for info in pkgutil.iter_modules([str(PACKAGE_DIR)]):
            importlib.import_module(f"decimalai.skillpkg.{info.name}")
        # The package's own map, not EXPORTS: this must hold whatever the map says.
        for name, module in skillpkg._EXPORTS.items():
            if _module_file(module) is not None:
                assert not inspect.ismodule(getattr(skillpkg, name)), (
                    f"skillpkg.{name} is a module: a submodule shadows the export"
                )


# ── Import cost ──────────────────────────────────────────────────────────────

# Top-level packages of the frameworks the SDK adapts, and of HTTP clients.
FRAMEWORKS = (
    "agent_framework", "agents", "anthropic", "autogen", "autogen_agentchat",
    "autogen_core", "claude_agent_sdk", "crewai", "deepagents", "google.adk",
    "google.genai", "langchain", "langchain_core", "langgraph", "litellm",
    "llama_index", "openai", "pydantic_ai", "strands",
)
HTTP_CLIENTS = (
    "aiohttp", "h11", "h2", "http.client", "httpcore", "httpx", "requests",
    "urllib.request", "urllib3",
)


class TestImportCost:
    def test_importing_the_package_loads_only_itself(self):
        report = _child_json(f"""
            import json, sys
            import decimalai
            base = set(sys.modules)
            import decimalai.skillpkg
            after_package = set(sys.modules)
            decimalai.skillpkg.Result
            after_result = set(sys.modules)
            frameworks = {FRAMEWORKS!r}
            print(json.dumps({{
                "package": sorted(after_package - base),
                "result": sorted(after_result - after_package),
                "non_stdlib": sorted(
                    m for m in after_result - base
                    if not m.startswith("decimalai.")
                    and m.split(".")[0] not in sys.stdlib_module_names
                ),
                "http": sorted(
                    m for m in after_result - base
                    if any(m == h or m.startswith(h + ".") for h in {HTTP_CLIENTS!r})
                ),
                "frameworks": sorted(
                    m for m in after_result
                    if any(m == f or m.startswith(f + ".") for f in frameworks)
                ),
            }}))
        """)
        assert report["package"] == ["decimalai.skillpkg"], (
            "importing decimalai.skillpkg must not import the modules behind its exports"
        )
        assert report["result"] == ["decimalai.skillpkg.errors"]
        assert report["non_stdlib"] == []
        assert report["http"] == []
        assert report["frameworks"] == []

    def test_the_errors_module_needs_only_the_standard_library(self):
        # Loaded on its own, from its file, with the SDK's own dependencies (and the
        # SDK itself) made unimportable: it is the leaf every other module of the
        # engine imports, so it may depend on nothing but the standard library.
        blocked = FRAMEWORKS + HTTP_CLIENTS + (
            "decimalai", "pydantic", "skillevaluation", "yaml", "dotenv", "click",
        )
        report = _child_json(f"""
            import importlib.util, json, sys
            for name in {blocked!r}:
                sys.modules[name] = None
            before = set(sys.modules)
            spec = importlib.util.spec_from_file_location(
                "standalone_skillpkg_errors", {str(PACKAGE_DIR / "errors.py")!r})
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            new = set(sys.modules) - before - {{spec.name}}
            print(json.dumps({{
                "non_stdlib": sorted(
                    m for m in new if m.split(".")[0] not in sys.stdlib_module_names),
                "exit": int(module.ModifiedFolderError("x").exit_code),
            }}))
        """)
        assert report == {"non_stdlib": [], "exit": 5}


# ── helpers ──────────────────────────────────────────────────────────────────

# A name no fix text contains, so its occurrences in a fix are exactly the skill's.
PLAIN_NAME = "plain-skill-name"


def _spans(fix: str) -> list:
    """The backticked commands and paths of a fix, in order."""
    return fix.split("`")[1::2]


def _module_file(module: str) -> Path | None:
    for candidate in (PACKAGE_DIR / f"{module}.py", PACKAGE_DIR / module / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _child_env() -> dict:
    """A child environment in which `import decimalai` does nothing on its own.

    `import decimalai` reads a `.env` found above the working directory, without
    overriding variables that are already set. Setting the auto-init variables to
    empty here keeps a developer's `.env` (say `DECIMAL_AUTO_TRACE=langchain`) from
    importing a framework into the child and failing these tests for the wrong reason.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("DECIMAL")}
    env.update({
        "DECIMAL_API_KEY": "",
        "DECIMALAI_API_KEY": "",
        "DECIMAL_AUTO_TRACE": "",
        "DECIMAL_AUTOINIT": "false",
        "DECIMALAI_SKIP_VERIFY": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return env


def _run_child(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT, env=_child_env(), capture_output=True, text=True, timeout=120,
    )


def _child_json(code: str) -> dict:
    child = _run_child(textwrap.dedent(code))
    assert child.returncode == 0, child.stderr
    return json.loads(child.stdout.strip().splitlines()[-1])
