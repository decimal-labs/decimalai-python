"""``skills add``: resolve a reference, pass the trust gate, record it, install it.

Pinned here:

- a reference resolves to its canonical registry name in ``.decimal/skills.toml``, with
  ``latest`` or the version asked for, and the folder, lock and ``plugin.json`` follow;
- re-adding with another spec changes the spec and moves the folder, like ``cargo add``;
- a name the project already uses (a hand-authored folder) gets the disambiguated name;
- the trust gate: blocked and withheld are refused, caution and unreviewed prompt on a
  terminal, are refused without one unless ``yes=True``, and owned skills skip it;
- an unmanaged folder holding the skill blocks the add (exit 5) unless ``force=True``,
  which backs it up first;
- every refusal leaves the project exactly as it was.

The other ``test_ops_*`` files import the helpers at the top of this one.
"""
from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from skillevaluation import skillfmt

from decimalai.skillpkg import errors as E
from decimalai.skillpkg import fs
from decimalai.skillpkg import lock as L
from decimalai.skillpkg import manifest as M
from decimalai.skillpkg import ops
from decimalai.skillpkg import project as P
from decimalai.skillpkg.client import RegistryClient
from tests.fakes.fake_registry import (
    BASE_URL,
    OTHER_ORG,
    OWNER_KEY,
    OWNER_ORG,
    FakeRegistry,
)

POSIX = os.name == "posix"

REFUND_FILES = (
    {"path": "references/fees.md", "content": "# Fees\n\nA 5% restocking fee.\n"},
    {"path": "scripts/compute_refund.py", "content": "print('refund')\n", "executable": True},
)


# ── helpers shared by every test_ops_* file ─────────────────────────────────


class Env:
    """A project directory, a fake registry, and the engine's calls pointed at both.

    The caller's API key belongs to ``OWNER_ORG``; :meth:`skill` adds a public skill of
    another organization, so it installs as the foreign variant behind the trust gate.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.dir = tmp_path / "project"
        self.dir.mkdir()
        self.project = P.Project.at(self.dir)
        self.registry = FakeRegistry()

    # the registry

    def skill(self, name: str = "acme/refund-policy", **kwargs: Any):
        kwargs.setdefault("org", OTHER_ORG)
        kwargs.setdefault("files", REFUND_FILES)
        return self.registry.add_skill(name, **kwargs)

    def owned(self, name: str, **kwargs: Any):
        kwargs.setdefault("org", OWNER_ORG)
        kwargs.setdefault("public", False)
        return self.registry.add_skill(name, **kwargs)

    def client(self, key: Optional[str] = OWNER_KEY) -> RegistryClient:
        return RegistryClient(api_key=key, base_url=BASE_URL, channel="cli",
                              transport=self.registry.transport(), sleep=lambda _s: None)

    def calls(self, route: Optional[str] = None) -> list:
        return self.registry.calls(route)

    # the engine

    def _kw(self, kwargs: Dict[str, Any], *, network: bool = True) -> Dict[str, Any]:
        kwargs.setdefault("directory", self.dir)
        if network:
            kwargs.setdefault("client", self.client())
        return kwargs

    def add(self, ref: str, **kwargs: Any) -> E.Result:
        kwargs.setdefault("interactive", False)
        kwargs.setdefault("yes", True)
        return ops.add(ref, **self._kw(kwargs))

    def install(self, **kwargs: Any) -> E.Result:
        kwargs.setdefault("interactive", False)
        return ops.install(**self._kw(kwargs))

    def update(self, name: Optional[str] = None, **kwargs: Any) -> E.Result:
        kwargs.setdefault("interactive", False)
        kwargs.setdefault("yes", True)
        return ops.update(name, **self._kw(kwargs))

    def remove(self, name: str, **kwargs: Any) -> E.Result:
        return ops.remove(name, **self._kw(kwargs, network=False))

    def status(self, **kwargs: Any) -> E.Result:
        return ops.status(**self._kw(kwargs))

    # the project

    def folder(self, name: str) -> Path:
        return self.project.skill_dir(name)

    def lock(self) -> L.Lock:
        return L.load(self.project)

    def lock_bytes(self) -> Optional[bytes]:
        try:
            return self.project.lock_path.read_bytes()
        except FileNotFoundError:
            return None

    def manifest(self) -> M.Manifest:
        return M.load(self.project)

    def manifest_text(self) -> Optional[str]:
        try:
            return self.project.manifest_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    def write_manifest(self, entries: Dict[str, str]) -> None:
        manifest = M.load(self.project, missing_ok=True)
        for name in list(manifest):
            manifest.remove(name)
        for name, spec in entries.items():
            manifest.set(name, spec)
        M.save(self.project, manifest)

    def states(self) -> Dict[str, fs.State]:
        lock = L.load(self.project, missing_ok=True, repair=True)
        return {n: s.state for n, s in fs.scan_root(self.project, lock.skills).items()}

    def tree(self) -> Dict[str, Any]:
        """Every file of the project except the run lock: path -> (bytes, executable)."""
        return tree(self.dir)


def tree(root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != ".skills.lck":
            mode = path.stat().st_mode
            out[path.relative_to(root).as_posix()] = (
                path.read_bytes(), bool(mode & stat.S_IXUSR) if POSIX else None)
    return out


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def errors_of(result: E.Result) -> list:
    return [p for p in result.problems if isinstance(p, E.SkillpkgError)]


def error_of(result: E.Result) -> E.SkillpkgError:
    errs = errors_of(result)
    assert errs, f"expected a refusal, got {result}"
    return errs[0]


def notices(result: E.Result, code: Optional[str] = None) -> list:
    return [p for p in result.problems
            if isinstance(p, ops.Notice) and (code is None or p.code == code)]


def actions(result: E.Result) -> list:
    return [(c.action, c.install_name) for c in result.changes]


def skill_md(env: Env, name: str) -> str:
    return (env.folder(name) / "SKILL.md").read_text(encoding="utf-8")


def frontmatter(env: Env, name: str) -> dict:
    fm, _ = skillfmt.parse_skill_md_text(skill_md(env, name))
    return fm


@pytest.fixture
def env(tmp_path) -> Env:
    return Env(tmp_path)


# ── the add row ─────────────────────────────────────────────────────────────


class TestAdd:
    def test_adds_latest_and_installs_the_folder(self, env):
        skill = env.skill(versions=2)
        result = env.add("acme/refund-policy")
        assert result.exit_code == E.ExitCode.OK, result.problems
        assert actions(result) == [("add", "acme-refund-policy")]
        change = result.changes[0]
        assert (change.skill, change.version, change.pin) == ("acme/refund-policy", 2, "latest")

        assert dict(env.manifest().entries) == {
            "acme/refund-policy": M.ManifestEntry("acme/refund-policy", "latest")}
        lock = env.lock()
        entry = lock.skills["acme-refund-policy"]
        assert entry["skill_id"] == skill.skill_id
        assert entry["version"] == 2 and entry["pin"] == "latest"
        assert entry["source"] == {"base_url": BASE_URL, "org": OWNER_ORG}
        assert env.states() == {"acme-refund-policy": fs.State.CLEAN}
        assert frontmatter(env, "acme-refund-policy")["name"] == "acme-refund-policy"
        if POSIX:
            script = env.folder("acme-refund-policy") / "scripts" / "compute_refund.py"
            assert script.stat().st_mode & 0o777 == 0o755
        plugin = env.project.plugin_json_path.read_bytes()
        assert plugin == L.plugin_json_bytes(lock.lock_hash)

    def test_an_exact_version_is_recorded_as_a_string(self, env):
        env.skill(versions=3)
        result = env.add("acme/refund-policy@2")
        assert result.ok, result.problems
        assert env.manifest()["acme/refund-policy"].spec == "2"
        entry = env.lock().skills["acme-refund-policy"]
        assert (entry["version"], entry["pin"]) == (2, "exact")

    def test_version_keyword_is_the_same_as_the_suffix(self, env):
        env.skill(versions=3)
        assert env.add("acme/refund-policy", version=1).ok
        assert env.manifest()["acme/refund-policy"].spec == "1"

    def test_a_slug_or_install_name_records_the_canonical_name(self, env):
        env.skill(url_slug="refunds-by-acme")
        assert env.add("refunds-by-acme").ok
        assert list(env.manifest()) == ["acme/refund-policy"]

    def test_an_id_reference_records_the_id(self, env):
        skill = env.skill()
        assert env.add(skill.skill_id).ok
        entry = env.manifest()["acme/refund-policy"]
        assert entry.id == skill.skill_id

    def test_writes_the_gitignore_for_the_engine_files(self, env):
        env.skill()
        assert env.add("acme/refund-policy").ok
        lines = (env.project.state_dir / ".gitignore").read_text().splitlines()
        assert set(P.GITIGNORE_LINES) <= set(lines)

    def test_the_gitignore_is_appended_not_replaced(self, env):
        env.skill()
        env.project.state_dir.mkdir()
        (env.project.state_dir / ".gitignore").write_text("install.json\nmine.txt")
        assert env.add("acme/refund-policy").ok
        lines = (env.project.state_dir / ".gitignore").read_text().splitlines()
        assert lines[:2] == ["install.json", "mine.txt"]
        assert lines.count("install.json") == 1
        assert set(P.GITIGNORE_LINES) <= set(lines)

    def test_policy_notes_and_unavailable_files_are_reported(self, env):
        env.skill(allowed_tools=["Read", "Write"], files=REFUND_FILES + (
            {"path": "assets/logo.png", "content": None},))
        result = env.add("acme/refund-policy")
        assert result.ok
        assert any("Write" in n.message for n in notices(result, "policy"))
        assert [n.message for n in notices(result, "unavailable")] and \
            "assets/logo.png" in notices(result, "unavailable")[0].message
        (unavailable,) = env.lock().skills["acme-refund-policy"]["unavailable"]
        assert unavailable["path"] == "assets/logo.png"

    def test_no_install_records_the_manifest_only(self, env):
        env.skill()
        result = env.add("acme/refund-policy", no_install=True)
        assert result.ok
        assert actions(result) == [("listed", "acme-refund-policy")]
        assert list(env.manifest()) == ["acme/refund-policy"]
        assert env.lock_bytes() is None
        assert not env.folder("acme-refund-policy").exists()
        assert env.calls("bundle") == []

    def test_reports_the_fetch_purpose_and_channel(self, env):
        env.skill()
        assert env.add("acme/refund-policy").ok
        (resolve,) = env.calls("resolve")
        assert resolve.json["purpose"] == "add" and resolve.json["channel"] == "cli"
        (bundle,) = env.calls("bundle")
        assert bundle.query["purpose"] == "add"


class TestReAdd:
    def test_re_add_with_another_spec_changes_the_spec_and_the_folder(self, env):
        env.skill(versions=3)
        assert env.add("acme/refund-policy").ok
        assert env.lock().skills["acme-refund-policy"]["version"] == 3

        result = env.add("acme/refund-policy@1")
        assert result.ok, result.problems
        assert actions(result) == [("update", "acme-refund-policy")]
        assert (result.changes[0].old_version, result.changes[0].version) == (3, 1)
        assert env.manifest()["acme/refund-policy"].spec == "1"
        entry = env.lock().skills["acme-refund-policy"]
        assert (entry["version"], entry["pin"]) == (1, "exact")
        assert "Step 1:" in skill_md(env, "acme-refund-policy")
        assert env.states() == {"acme-refund-policy": fs.State.CLEAN}

    def test_re_add_to_the_installed_version_changes_only_the_pin(self, env):
        env.skill(versions=2)
        assert env.add("acme/refund-policy").ok
        before = env.folder("acme-refund-policy").stat().st_ino
        installed_at = env.lock().skills["acme-refund-policy"]["installed_at"]

        result = env.add("acme/refund-policy@2")
        assert result.ok
        assert actions(result) == [("pin", "acme-refund-policy")]
        assert len(env.calls("bundle")) == 1, "no second fetch for a pin change"
        entry = env.lock().skills["acme-refund-policy"]
        assert (entry["pin"], entry["installed_at"]) == ("exact", installed_at)
        assert env.folder("acme-refund-policy").stat().st_ino == before

    def test_re_add_of_the_same_spec_changes_nothing(self, env):
        env.skill()
        assert env.add("acme/refund-policy").ok
        before = env.tree()
        result = env.add("acme/refund-policy")
        assert result.ok
        assert actions(result) == [("unchanged", "acme-refund-policy")]
        assert env.tree() == before

    def test_re_add_restores_a_missing_folder(self, env):
        env.skill()
        assert env.add("acme/refund-policy").ok
        before = env.tree()
        import shutil
        shutil.rmtree(env.folder("acme-refund-policy"))
        result = env.add("acme/refund-policy")
        assert result.ok
        assert actions(result) == [("fetch", "acme-refund-policy")]
        assert env.tree() == before


class TestInstallNames:
    def test_a_hand_authored_folder_with_the_name_gets_the_disambiguated_name(self, env):
        skill = env.skill()
        mine = env.folder("acme-refund-policy")
        mine.mkdir(parents=True)
        (mine / "SKILL.md").write_text(
            "---\nname: acme-refund-policy\ndescription: mine\n---\nMine.\n")
        before = (mine / "SKILL.md").read_bytes()

        result = env.add("acme/refund-policy")
        assert result.ok, result.problems
        alt = skillfmt.install_name_disambiguated("acme/refund-policy", skill.skill_id)
        assert actions(result) == [("add", alt)]
        assert env.states() == {"acme-refund-policy": fs.State.UNMANAGED, alt: fs.State.CLEAN}
        assert (mine / "SKILL.md").read_bytes() == before
        resolve = env.calls("resolve")[0].json
        assert resolve["taken"] == {"acme-refund-policy": None}

    def test_the_server_sees_every_locked_name(self, env):
        a = env.skill("acme/refund-policy")
        env.skill("acme/returns")
        assert env.add("acme/refund-policy").ok
        assert env.add("acme/returns").ok
        assert env.calls("resolve")[-1].json["taken"] == {"acme-refund-policy": a.skill_id}

    def test_both_names_taken_is_a_collision(self, env):
        skill = env.skill()
        for name in ("acme-refund-policy",
                     skillfmt.install_name_disambiguated("acme/refund-policy", skill.skill_id)):
            (env.folder(name)).mkdir(parents=True)
            (env.folder(name) / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: x\n---\nx\n")
        before = env.tree()
        result = env.add("acme/refund-policy")
        assert result.exit_code == E.ExitCode.UNAVAILABLE
        assert isinstance(error_of(result), E.InstallNameCollisionError)
        assert env.tree() == before


class TestTrustGate:
    @pytest.mark.parametrize("band", ["caution", "unreviewed"])
    def test_non_tty_without_yes_is_refused(self, env, band):
        env.skill(band=band)
        result = env.add("acme/refund-policy", yes=False, interactive=False)
        assert result.exit_code == E.ExitCode.UNAVAILABLE
        err = error_of(result)
        assert isinstance(err, E.ConfirmationRequiredError)
        assert band in err.message and "--yes" in err.fix
        assert "no terminal" in err.message, "refused for having no terminal, not declined"
        assert env.tree() == {}
        assert env.calls("bundle") == []

    def test_non_tty_never_asks(self, env):
        env.skill(band="caution")
        result = env.add("acme/refund-policy", yes=False, interactive=False,
                         confirm=lambda q: pytest.fail("asked without a terminal"))
        assert isinstance(error_of(result), E.ConfirmationRequiredError)

    @pytest.mark.parametrize("band", ["caution", "unreviewed"])
    def test_non_tty_with_yes_installs(self, env, band):
        env.skill(band=band)
        result = env.add("acme/refund-policy", yes=True, interactive=False)
        assert result.ok, result.problems
        assert env.states() == {"acme-refund-policy": fs.State.CLEAN}

    def test_tty_asks_and_installs_on_yes(self, env):
        env.skill(band="caution")
        asked = []
        result = env.add("acme/refund-policy", yes=False, interactive=True,
                         confirm=lambda q: asked.append(q) or True)
        assert result.ok
        assert len(asked) == 1 and "acme/refund-policy" in asked[0] and "caution" in asked[0]

    def test_tty_declined_is_refused_and_changes_nothing(self, env):
        env.skill(band="unreviewed")
        result = env.add("acme/refund-policy", yes=False, interactive=True,
                         confirm=lambda q: False)
        assert result.exit_code == E.ExitCode.UNAVAILABLE
        assert isinstance(error_of(result), E.ConfirmationRequiredError)
        assert env.tree() == {}

    def test_yes_skips_the_prompt_on_a_tty(self, env):
        env.skill(band="caution")
        result = env.add("acme/refund-policy", yes=True, interactive=True,
                         confirm=lambda q: pytest.fail("prompted despite yes"))
        assert result.ok

    def test_a_clean_skill_never_prompts(self, env):
        env.skill(band="clean")
        result = env.add("acme/refund-policy", yes=False, interactive=True,
                         confirm=lambda q: pytest.fail("prompted for a clean skill"))
        assert result.ok

    def test_owned_skills_skip_the_gate(self, env):
        env.owned("my-skill", band="unreviewed")
        result = env.add("my-skill", yes=False, interactive=False)
        assert result.ok, result.problems
        assert env.lock().skills["my-skill"]["render"]["variant"] == "authored"

    @pytest.mark.parametrize("attrs,cls", [({"band": "blocked"}, E.SkillBlockedError),
                                           ({"withheld": True}, E.SkillWithheldError)])
    def test_blocked_and_withheld_are_refused(self, env, attrs, cls):
        env.skill(**attrs)
        result = env.add("acme/refund-policy", yes=True)
        assert result.exit_code == E.ExitCode.UNAVAILABLE
        assert isinstance(error_of(result), cls)
        assert env.tree() == {}

    def test_a_blocked_answer_is_refused_even_with_yes(self, env):
        # A registry that answers a blocked skill as resolvable is refused by the client.
        env.skill()

        def mark_blocked(data):
            data["items"][0]["safety"] = {"band": "blocked", "withheld": False}

        env.registry.tamper_resolve = mark_blocked
        result = env.add("acme/refund-policy", yes=True)
        assert isinstance(error_of(result), E.SkillBlockedError)
        assert env.tree() == {}

    def test_a_withheld_skill_the_org_uses_needs_confirmation(self, env):
        # Withheld skills stay installable for organizations that own or use them.
        env.skill()

        def withheld_but_used(data):
            data["items"][0]["safety"] = {"band": "clean", "withheld": True}
            data["items"][0]["relationship"] = "used"

        env.registry.tamper_resolve = withheld_but_used
        result = env.add("acme/refund-policy", yes=False, interactive=False)
        err = error_of(result)
        assert isinstance(err, E.ConfirmationRequiredError) and "withheld" in err.message
        assert env.add("acme/refund-policy", yes=True).ok

    def test_the_default_reads_the_terminal(self, env, monkeypatch):
        env.skill(band="caution")
        monkeypatch.setattr(ops, "_stdin_is_tty", lambda: False)
        result = ops.add("acme/refund-policy", directory=env.dir, client=env.client())
        assert isinstance(error_of(result), E.ConfirmationRequiredError)
        monkeypatch.setattr(ops, "_stdin_is_tty", lambda: True)
        result = ops.add("acme/refund-policy", directory=env.dir, client=env.client(),
                         confirm=lambda q: True)
        assert result.ok


class TestAddRefusals:
    def test_not_found_is_exit_7_and_changes_nothing(self, env):
        env.skill()
        result = env.add("acme/no-such-skill")
        assert result.exit_code == E.ExitCode.UNAVAILABLE
        err = error_of(result)
        assert isinstance(err, E.SkillNotFoundError) and err.skill == "acme/no-such-skill"
        assert env.tree() == {}

    def test_ambiguous_is_exit_7(self, env):
        env.skill("pdf", org="org-a", url_slug="pdf-a")
        env.skill("pdf", org="org-b", url_slug="pdf-b")
        result = env.add("pdf")
        assert isinstance(error_of(result), E.AmbiguousRefError)

    def test_an_unmanaged_copy_of_the_skill_blocks_without_force(self, env):
        env.skill()
        assert env.add("acme/refund-policy").ok
        # The lock is lost; the folder stays, stamped with its skill id.
        env.project.lock_path.unlink()
        env.project.manifest_path.unlink()
        before = env.tree()
        result = env.add("acme/refund-policy")
        assert result.exit_code == E.ExitCode.LOCAL_CHANGES
        err = error_of(result)
        assert isinstance(err, E.UnmanagedFolderError)
        assert "acme-refund-policy" in err.message and "--force" in err.fix
        assert env.tree() == before

    def test_force_replaces_an_unmanaged_copy_and_backs_it_up(self, env):
        env.skill()
        assert env.add("acme/refund-policy").ok
        env.project.lock_path.unlink()
        env.project.manifest_path.unlink()
        (env.folder("acme-refund-policy") / "mine.md").write_text("my notes\n")
        result = env.add("acme/refund-policy", force=True)
        assert result.ok, result.problems
        change = result.changes[0]
        assert change.backup and (env.dir / change.backup / "mine.md").read_text() == "my notes\n"
        assert any(change.backup in n.message for n in notices(result, "backup"))
        assert env.states() == {"acme-refund-policy": fs.State.CLEAN}

    def test_a_symlink_where_the_folder_goes_is_never_overridden(self, env, tmp_path):
        if not POSIX:
            pytest.skip("symlinks")
        env.skill()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        env.project.skills_root.mkdir(parents=True)
        os.symlink(elsewhere, env.folder("acme-refund-policy"))
        # The symlinked name is not the skill's, so the registry picks around it: the
        # symlink itself is never written through.
        result = env.add("acme/refund-policy", force=True)
        assert result.ok
        assert os.listdir(elsewhere) == []

    def test_a_network_failure_is_exit_8(self, env):
        env.skill()
        env.registry.fault("resolve", "connect")
        result = env.add("acme/refund-policy")
        assert result.exit_code == E.ExitCode.NETWORK
        assert env.tree() == {}

    def test_tampered_bytes_are_exit_6_and_nothing_is_written(self, env):
        env.skill()

        def tamper(data):
            data["files"][-1]["content"] += "rm -rf /\n"

        env.registry.tamper = tamper
        result = env.add("acme/refund-policy")
        assert result.exit_code == E.ExitCode.INTEGRITY
        assert env.tree() == {}

    def test_already_installed_under_another_name_is_exit_7(self, env):
        skill = env.skill()
        assert env.add("acme/refund-policy").ok
        before = env.tree()
        alt = skillfmt.install_name_disambiguated("acme/refund-policy", skill.skill_id)

        def another_name(data):
            data["items"][0]["install_name"] = alt

        env.registry.tamper_resolve = another_name
        result = env.add("acme/refund-policy@1")
        assert result.exit_code == E.ExitCode.UNAVAILABLE
        err = error_of(result)
        assert isinstance(err, E.AlreadyInstalledError)
        assert "acme-refund-policy" in err.message
        assert env.tree() == before

    @pytest.mark.parametrize("ref", ["acme/\x00refund", "acme/\udc80refund", "a" * 600])
    def test_a_reference_the_registry_cannot_read_is_exit_2(self, env, ref):
        result = env.add(ref)
        assert result.exit_code == E.ExitCode.USAGE
        err = error_of(result)
        assert isinstance(err, E.UsageError) and "reference #1" in err.fix
        assert env.tree() == {}

    def test_a_malformed_version_is_a_usage_error(self, env):
        result = env.add("acme/refund-policy@v3")
        assert result.exit_code == E.ExitCode.USAGE
        assert env.calls() == []

    def test_a_conflicting_version_keyword_is_a_usage_error(self, env):
        result = env.add("acme/refund-policy@2", version=3)
        assert result.exit_code == E.ExitCode.USAGE

    def test_no_staging_is_left_behind(self, env):
        env.skill()
        env.registry.fault("bundle", 500)
        result = env.add("acme/refund-policy")
        assert result.exit_code == E.ExitCode.NETWORK
        assert not env.project.staging_dir.exists()
        assert not env.project.process_lock_path.exists()


class TestAPluginJsonOfTheUsersOwn:
    MINE = b'{"name": "my-tools", "version": "1.0.0", "description": "mine"}\n'

    def _plant(self, env):
        env.project.plugin_json_path.parent.mkdir(parents=True)
        env.project.plugin_json_path.write_bytes(self.MINE)

    def test_is_refused_before_anything_changes(self, env):
        env.skill()
        self._plant(env)
        before = env.tree()
        result = env.add("acme/refund-policy")
        assert result.exit_code == E.ExitCode.LOCAL_CHANGES
        err = error_of(result)
        assert isinstance(err, E.LocalChangesError)
        assert "plugin.json" in err.message and "--force" in err.fix
        assert env.tree() == before
        assert env.calls() == [], "refused before any network call"

    def test_force_backs_it_up_and_says_where(self, env):
        env.skill()
        self._plant(env)
        result = env.add("acme/refund-policy", force=True)
        assert result.ok, result.problems
        (note,) = [n for n in notices(result, "backup") if "plugin.json" in n.message]
        backup = env.dir / note.path
        assert backup.read_bytes() == self.MINE
        assert note.path.startswith(".decimal/backup/") and note.path in note.message
        lock = env.lock()
        assert env.project.plugin_json_path.read_bytes() == L.plugin_json_bytes(lock.lock_hash)

    def test_one_this_engine_wrote_is_replaced(self, env):
        env.skill("acme/refund-policy")
        env.skill("acme/returns")
        assert env.add("acme/refund-policy").ok
        assert env.add("acme/returns").ok
        assert env.project.plugin_json_path.read_bytes() == \
            L.plugin_json_bytes(env.lock().lock_hash)


class TestAnonymous:
    def test_anonymous_add_of_a_public_skill(self, env):
        env.skill()
        result = env.add("acme/refund-policy", client=env.client(key=None))
        assert result.ok, result.problems
        assert env.calls("resolve")[0].path == "/api/v1/registry/install/resolve"
        entry = env.lock().skills["acme-refund-policy"]
        assert entry["render"]["variant"] == "foreign"
        assert entry["source"]["org"] is None

    def test_the_configured_registry_is_used_without_a_client(self, env, monkeypatch):
        env.skill()
        monkeypatch.setenv("DECIMAL_API_KEY", OWNER_KEY)
        with env.registry.serve() as url:
            monkeypatch.setenv("DECIMAL_BASE_URL", url)
            result = ops.add("acme/refund-policy", directory=env.dir, yes=True)
        assert result.ok, result.problems
        assert env.lock().skills["acme-refund-policy"]["source"]["base_url"] == url
        assert env.calls("resolve")[0].headers["authorization"] == f"Bearer {OWNER_KEY}"
