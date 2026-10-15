"""The old disk verbs, in a project the package manager (``decimalai.skillpkg``) manages.

``SkillRouter``'s disk methods used to fetch a skill through ``/body`` and write it with
``disk_export``; the adapters' ``instrument()`` mirrored every org skill into whatever
directory the process ran from. Now there is one engine and one way skills reach disk:

- ``export(name)`` is ``skillpkg.add(name)``; ``export_to_disk``, ``install`` and
  ``install_skill`` are ``add`` too (no fork), and warn once that they are deprecated;
- ``update_skills(force=...)`` is ``skillpkg.update(force=...)``;
- ``status(project_root)`` is ``skillpkg.status(directory=project_root)``;
- ``pull_missing()`` runs ``skillpkg.install()`` when the project has a manifest, and
  otherwise writes nothing and says, once, what to run instead;
- ``instrument()`` never writes a skills directory, and its sync-up never uploads a
  folder the package manager installed;
- the install identity finds its project the way the engine does, and its
  ``.decimal/.gitignore`` is the engine's;
- ``decimalai.skills_dir`` is the engine's ``skills_dir``, imported on first use.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import httpx
import pytest
from skillevaluation import skillfmt

import decimalai.disk_export as disk_export
import decimalai.skill_router as sr
from decimalai.skill_router import SkillRouter, SkillRouterError
from decimalai.skillpkg import lock as L
from decimalai.skillpkg import manifest as M
from decimalai.skillpkg import ops
from decimalai.skillpkg.client import RegistryClient
from decimalai.skillpkg.errors import ExitCode, Result
from decimalai.skillpkg.project import GITIGNORE_LINES, Project
from tests.fakes.fake_registry import BASE_URL, OTHER_ORG, OWNER_KEY, OWNER_ORG, FakeRegistry

REPO = Path(__file__).resolve().parent.parent


# ── helpers other test files import ───────────────────────────────────────────


@pytest.fixture(autouse=True)
def _fresh_once_state(monkeypatch):
    """Each shim warns once per process; every test starts with none warned yet."""
    monkeypatch.setattr(sr, "_warned_once", set())
    monkeypatch.setattr(disk_export, "_warned_once", set())


def engine_on(registry: FakeRegistry, monkeypatch) -> List[Dict[str, Any]]:
    """Point every registry client the engine builds at ``registry``.

    Returns the list of keyword arguments the engine built clients with, so a test can
    see which key, registry and channel the router handed over.
    """
    seen: List[Dict[str, Any]] = []

    def from_env(cls, *, api_key=None, base_url=None, anonymous=False, **kwargs):
        seen.append({"api_key": api_key, "base_url": base_url, **kwargs})
        return RegistryClient(api_key=api_key, base_url=BASE_URL,
                              channel=kwargs.get("channel", "sdk"),
                              transport=registry.transport(), sleep=lambda _s: None)

    monkeypatch.setattr(RegistryClient, "from_env", classmethod(from_env))
    return seen


def no_router_http(monkeypatch) -> List[str]:
    """Fail any request the router makes itself (``/body``, a fork): the engine's own
    client goes through its transport, never through ``httpx.request``."""
    made: List[str] = []

    def refuse(method, url, **kwargs):
        made.append(f"{method} {url}")
        raise AssertionError(f"the router made a request of its own: {method} {url}")

    monkeypatch.setattr(httpx, "request", refuse)
    return made


def router() -> SkillRouter:
    return SkillRouter(api_key=OWNER_KEY, base_url=BASE_URL)


def _deprecations(caught) -> List[str]:
    return [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]


@pytest.fixture
def registry(monkeypatch) -> FakeRegistry:
    reg = FakeRegistry()
    reg.from_env_calls = engine_on(reg, monkeypatch)
    return reg


@pytest.fixture
def project(tmp_path) -> Path:
    path = tmp_path / "project"
    path.mkdir()
    return path


def _refund(registry: FakeRegistry, **kw):
    kw.setdefault("org", OTHER_ORG)
    kw.setdefault("files", (
        {"path": "references/fees.md", "content": "# Fees\n\nA 5% restocking fee.\n"},
    ))
    return registry.add_skill("acme/refund-policy", **kw)


def _tree(root: Path) -> List[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


# ── every shim delegates ──────────────────────────────────────────────────────


class _Recorder:
    """Stands in for one ``skillpkg.ops`` function and records each call."""

    def __init__(self, result: Optional[Result] = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.result = result or Result([ops.Change("add", "pdf", "pdf", version=1)])

    def __call__(self, *args, **kwargs):
        self.calls.append({"args": args, **kwargs})
        return self.result


def _manifest(project: Path) -> None:
    (project / ".decimal").mkdir(exist_ok=True)
    (project / ".decimal" / "skills.toml").write_text(
        'root = ".agents/skills"\n\n[skills]\n"pdf" = "latest"\n', encoding="utf-8")


DEPRECATED = {
    "export_to_disk": ("add", lambda r, d: r.export_to_disk(skills=["pdf"], project_root=d)),
    "export_to_disk_all": ("add", lambda r, d: r.export_to_disk(project_root=d)),
    "install": ("add", lambda r, d: r.install("pdf", project_root=d)),
    "install_skill": ("add", lambda r, d: r.install_skill("pdf", project_root=d)),
    "pull_missing": ("install", lambda r, d: r.pull_missing(project_root=d)),
}


class TestEveryShimDelegatesAndWarnsOnce:
    @pytest.mark.parametrize("shim", sorted(DEPRECATED))
    def test_a_deprecated_shim_calls_the_engine_every_time_and_warns_once(
            self, shim, project, monkeypatch):
        op_name, call = DEPRECATED[shim]
        recorder = _Recorder()
        monkeypatch.setattr(ops, op_name, recorder)
        _manifest(project)
        r = router()
        monkeypatch.setattr(r, "list_skills", lambda **kw: [{"name": "pdf"}])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call(r, str(project))
            call(r, str(project))

        assert len(recorder.calls) == 2
        deprecations = _deprecations(caught)
        assert len(deprecations) == 1, deprecations
        assert "decimalai skills" in deprecations[0] or "skillpkg" in deprecations[0]
        for c in recorder.calls:
            assert c["directory"] == str(project)
            assert c["api_key"] == OWNER_KEY and c["base_url"] == BASE_URL
            assert c.get("channel", "sdk") == "sdk"

    @pytest.mark.parametrize("shim", ["install", "install_skill", "export_to_disk"])
    def test_the_add_shims_name_the_skill(self, shim, project, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(ops, "add", recorder)
        DEPRECATED[shim][1](router(), str(project))
        assert recorder.calls[0]["args"] == ("pdf",)

    def test_export_is_add_and_is_not_deprecated(self, project, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(ops, "add", recorder)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            router().export("pdf", project_root=str(project))
        assert recorder.calls[0]["args"] == ("pdf",)
        assert recorder.calls[0]["directory"] == str(project)
        assert _deprecations(caught) == []

    def test_update_skills_is_update_and_force_means_force(self, project, monkeypatch):
        recorder = _Recorder(Result([ops.Change("update", "pdf", "pdf", version=2,
                                                old_version=1)]))
        monkeypatch.setattr(ops, "update", recorder)
        out = router().update_skills(project_root=str(project), force=True)
        (c,) = recorder.calls
        assert c["force"] is True and c["directory"] == str(project)
        assert out["updated"] == 1 and out["updated_skills"] == ["pdf"]

    def test_status_reads_the_project_it_is_given(self, project, monkeypatch):
        recorder = _Recorder(Result([ops.SkillStatus("pdf", "clean", skill="pdf", version=1)]))
        monkeypatch.setattr(ops, "status", recorder)
        out = router().status(project_root=str(project))
        assert recorder.calls[0]["directory"] == str(project)
        assert out["synced"] == ["pdf"]


# ── against a registry ────────────────────────────────────────────────────────


class TestExportIsAdd:
    def test_export_installs_a_managed_folder_with_no_body_call_and_no_copy(
            self, registry, project, monkeypatch):
        made = no_router_http(monkeypatch)
        skill = _refund(registry)

        summary = router().export("acme/refund-policy", project_root=str(project))

        folder = project.resolve() / ".agents" / "skills" / "acme-refund-policy"
        assert summary["skills_written"] == 1
        assert summary["skill_dirname"] == "acme-refund-policy"
        assert summary["skill_name"] == "acme/refund-policy"
        assert summary["written_paths"] == [str(folder / "SKILL.md")]
        assert summary["attachments_written"] == 1
        assert summary["errors"] == []
        assert summary["lock_entry"]["version_id"] == skill.newest.version_id
        assert L.load(Project.at(project)).skills["acme-refund-policy"]["skill_id"] == skill.skill_id
        assert registry.calls("body") == [] and made == []
        assert registry.from_env_calls[0]["api_key"] == OWNER_KEY
        assert registry.from_env_calls[0]["channel"] == "sdk"

    def test_agents_are_ignored_with_one_warning(self, registry, project, monkeypatch, caplog):
        no_router_http(monkeypatch)
        _refund(registry)
        registry.add_skill("pdf-tools")
        r = router()
        with caplog.at_level(logging.WARNING, logger="decimalai.skill_router"):
            r.export("acme/refund-policy", agents=["claude-code"], project_root=str(project))
            r.export("pdf-tools", agents=["claude-code"], project_root=str(project))
        assert not (project / ".claude").exists()
        notes = [rec.getMessage() for rec in caplog.records if "agents" in rec.getMessage()]
        assert len(notes) == 1 and "--plugin-dir" in notes[0]

    def test_a_global_scope_is_refused_with_the_command_to_use(self, registry, project):
        with pytest.raises(ValueError) as ei:
            router().export("acme/refund-policy", scope="global", project_root=str(project))
        assert "skills pull" in str(ei.value) and "--out" in str(ei.value)
        assert not (project / ".decimal").exists()

    def test_a_refusal_raises_and_writes_nothing(self, registry, project, monkeypatch):
        no_router_http(monkeypatch)
        with pytest.raises(SkillRouterError) as ei:
            router().export("no-such-skill", project_root=str(project))
        assert "no-such-skill" in str(ei.value)
        assert ei.value.result.exit_code == ExitCode.UNAVAILABLE
        assert not (project / ".agents").exists()

    def test_install_takes_no_copy(self, registry, project, monkeypatch):
        made = no_router_http(monkeypatch)
        _refund(registry)
        out = router().install("acme/refund-policy", project_root=str(project))
        assert made == [], "install forked"
        assert out["skill_name"] == "acme/refund-policy"
        assert out["export"]["skills_written"] == 1
        assert out["fork"] is None

    def test_export_to_disk_lists_every_org_skill_in_the_manifest(
            self, registry, project, monkeypatch):
        no_router_http(monkeypatch)
        registry.add_skill("house-style", org=OWNER_ORG, public=False)
        registry.add_skill("deploy-checklist", org=OWNER_ORG, public=False)
        r = router()
        monkeypatch.setattr(r, "list_skills", lambda **kw: [
            {"name": "house-style"}, {"name": "deploy-checklist"}])

        summary = r.export_to_disk(project_root=str(project))

        assert summary["skills_written"] == 2 and summary["errors"] == []
        manifest = M.load(Project.at(project))
        assert sorted(manifest.entries) == ["deploy-checklist", "house-style"]

    def test_export_to_disk_raises_when_every_requested_skill_fails(
            self, registry, project, monkeypatch):
        no_router_http(monkeypatch)
        with pytest.raises(SkillRouterError) as ei:
            router().export_to_disk(skills=["gone"], project_root=str(project))
        assert "gone" in str(ei.value)

    def test_status_reads_the_named_project_not_the_working_directory(
            self, registry, project, tmp_path, monkeypatch):
        no_router_http(monkeypatch)
        _refund(registry)
        r = router()
        r.export("acme/refund-policy", project_root=str(project))
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        out = r.status(project_root=str(project))

        assert out["synced"] == ["acme-refund-policy"]
        assert out["modified_locally"] == [] and out["missing_locally"] == []
        assert out["result"].ok


class TestPullMissing:
    def test_with_a_manifest_it_installs_the_manifest(self, registry, project, monkeypatch):
        no_router_http(monkeypatch)
        _refund(registry)
        added = ops.add("acme/refund-policy", no_install=True, directory=project)
        assert added.ok, added.problems

        out = router().pull_missing(project_root=str(project))

        assert out["pulled"] == 1 and out["pulled_skills"] == ["acme-refund-policy"]
        assert (project / ".agents" / "skills" / "acme-refund-policy" / "SKILL.md").is_file()

    def test_without_a_manifest_it_writes_nothing_and_says_what_to_run_once(
            self, project, monkeypatch, caplog):
        made = no_router_http(monkeypatch)
        (project / ".agents" / "skills").mkdir(parents=True)  # once taken as consent
        before = _tree(project)
        r = router()
        with caplog.at_level(logging.WARNING, logger="decimalai.skill_router"):
            first = r.pull_missing(project_root=str(project), agents=["universal"])
            r.pull_missing(project_root=str(project))

        assert first == {"pulled": 0, "updated": 0, "skipped": 0}
        assert _tree(project) == before and made == []
        notes = [rec for rec in caplog.records if "pull_missing" in rec.getMessage()]
        assert len(notes) == 1 and notes[0].levelno == logging.WARNING
        assert "decimalai skills add" in notes[0].getMessage()


# ── instrument() never writes disk; sync-up skips managed folders ─────────────


SYNC_FIELDS = {"name", "display_name", "description", "body_markdown", "category",
               "trigger_phrases", "skill_type", "skill_scope", "invocation",
               "local_updated_at"}

AUTHORED = "---\nname: my-own\ndescription: A skill I wrote.\n---\n# Mine\n\nDo it my way.\n"


def _managed_folder(root: Path) -> Path:
    fields = skillfmt.RenderFields(
        skill_id="00000000-0000-4000-8000-0000000000aa",
        version_id="00000000-0000-4000-8000-0000000000bb", version=3,
        registry_name="acme/refund-policy", body="# Refunds\n\nA 5% restocking fee.\n",
        description="Apply Acme refund rules.")
    folder = root / ".agents" / "skills" / "acme-refund-policy"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        skillfmt.render_skill_md(fields, install_name="acme-refund-policy", variant="foreign"),
        encoding="utf-8")
    return folder


class _Platform:
    """What the platform answers the router: enough for a 0.14 background pull to write."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, _self, method, path, **kwargs):
        self.calls.append({"method": method, "path": path, **kwargs})
        if path == "/api/v1/skills/hashes":
            return {"hashes": {"refund-policy": {"hash": "abc123", "version": 2}}}
        if path == "/api/v1/skills/refund-policy/body":
            return {"name": "refund-policy", "version": 2, "body": "# Refunds\n\nBe fair.",
                    "truncated": False, "skill_id": "sk-1", "description": "Refund rules"}
        if path.endswith("/attachments"):
            return {"attachments": []}
        if path == "/api/v1/skills/sync":
            return {"created": 0, "updated": 0, "unchanged": 0}
        return {}

    def paths(self) -> List[str]:
        return [c["path"] for c in self.calls]

    def synced(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for c in self.calls:
            if c["path"] == "/api/v1/skills/sync":
                out.extend(c["json"]["skills"])
        return out


@pytest.fixture
def platform(monkeypatch):
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    monkeypatch.setattr(cfg, "_config", DecimalConfig(
        api_key="dai_sk_test", base_url="http://platform.test", enabled=True))
    client = MagicMock()
    client.register_manifest.return_value = {"manifest_id": "m-1", "status": "active"}
    monkeypatch.setattr(cfg, "_client", client)
    sender = MagicMock()
    sender.submit.side_effect = lambda fn, *a, **kw: fn(*a, **kw)
    monkeypatch.setattr(cfg, "_sender", sender)
    fake = _Platform()
    monkeypatch.setattr(SkillRouter, "_request", lambda self, method, path, **kw:
                        fake(self, method, path, **kw))
    return fake


def _instrument_langchain(**kwargs):
    import decimalai.langchain as lc

    with patch.dict("sys.modules", {
        "langchain_core": MagicMock(),
        "langchain_core.tracers": MagicMock(),
        "langchain_core.tracers.context": MagicMock(register_configure_hook=MagicMock()),
    }):
        lc._installed = False
        lc._explicit_manifest_config = None
        try:
            lc.instrument(**kwargs)
        finally:
            lc._installed = False
            lc._explicit_manifest_config = None


def _instrument_openai_agents(**kwargs):
    import decimalai.openai_agents as oa

    with patch.dict("sys.modules", {
        "agents": MagicMock(),
        "agents.tracing": MagicMock(add_trace_processor=MagicMock(),
                                    set_trace_processors=MagicMock()),
    }):
        oa._manifest_id = None
        oa.instrument(**kwargs)


ADAPTERS = {"langchain": _instrument_langchain, "openai_agents": _instrument_openai_agents}


class TestInstrumentNeverWritesDisk:
    @pytest.mark.parametrize("adapter", sorted(ADAPTERS))
    @pytest.mark.parametrize("agent_name", ["universal", "claude-code", "support-bot"])
    def test_no_skills_directory_is_written(self, adapter, agent_name, project, platform,
                                            monkeypatch):
        # The directory 0.14 read as consent to mirror every org skill into.
        (project / ".agents" / "skills").mkdir(parents=True)
        (project / ".claude" / "skills").mkdir(parents=True)
        monkeypatch.chdir(project)
        before = _tree(project)

        ADAPTERS[adapter](agent_name=agent_name, disk_sync=True)

        assert [p for p in _tree(project) if p not in before
                and not p.startswith(".decimal")] == []
        assert not (project / ".decimal" / "skills.lock").exists()
        assert "/api/v1/skills/hashes" not in platform.paths()

    @pytest.mark.parametrize("adapter", sorted(ADAPTERS))
    def test_instrument_never_calls_pull_missing(self, adapter, project, platform, monkeypatch):
        monkeypatch.chdir(project)
        calls = []
        monkeypatch.setattr(SkillRouter, "pull_missing",
                            lambda self, **kw: calls.append(kw) or {"pulled": 0})
        ADAPTERS[adapter](agent_name="claude-code", disk_sync=True)
        assert calls == []

    @pytest.mark.parametrize("adapter", sorted(ADAPTERS))
    def test_sync_up_sends_authored_skills_only_and_only_the_sync_fields(
            self, adapter, project, platform, monkeypatch):
        _managed_folder(project)
        mine = project / ".claude" / "skills" / "my-own"
        mine.mkdir(parents=True)
        (mine / "SKILL.md").write_text(AUTHORED, encoding="utf-8")
        monkeypatch.chdir(project)

        ADAPTERS[adapter](disk_sync=True)

        sent = platform.synced()
        assert [s["name"] for s in sent] == ["my-own"]
        assert set(sent[0]) <= SYNC_FIELDS, set(sent[0]) - SYNC_FIELDS
        assert "Do it my way." in sent[0]["body_markdown"]

    @pytest.mark.parametrize("adapter", sorted(ADAPTERS))
    def test_only_managed_folders_means_no_sync_call(self, adapter, project, platform,
                                                     monkeypatch):
        _managed_folder(project)
        monkeypatch.chdir(project)
        ADAPTERS[adapter](disk_sync=True)
        assert "/api/v1/skills/sync" not in platform.paths()


# ── the install identity's project and gitignore ──────────────────────────────


class TestInstallIdentityProject:
    def test_the_search_stops_at_the_repository(self, tmp_path):
        from decimalai import _install

        (tmp_path / ".decimal").mkdir()           # a folder of repositories, above the repo
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        sub = repo / "pkg" / "nested"
        sub.mkdir(parents=True)
        assert _install.find_project_root(str(sub)) == str(repo.resolve())

    def test_a_decimal_dir_inside_the_repository_still_wins(self, tmp_path):
        from decimalai import _install

        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        (repo / "pkg" / ".decimal").mkdir(parents=True)
        sub = repo / "pkg" / "nested"
        sub.mkdir()
        assert _install.find_project_root(str(sub)) == str((repo / "pkg").resolve())

    def test_the_search_never_climbs_above_home(self, tmp_path, monkeypatch):
        from decimalai import _install

        home = tmp_path / "home"
        work = home / "scratch"
        work.mkdir(parents=True)
        (tmp_path / ".decimal").mkdir()           # above home: never this project
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))
        assert _install.find_project_root(str(work)) == str(work.resolve())

    def test_the_identity_lands_in_the_bounded_project(self, tmp_path, monkeypatch):
        from decimalai import _install

        (tmp_path / ".decimal").mkdir()
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        monkeypatch.chdir(repo)
        _install.get_install_identity()
        assert (repo / ".decimal" / "install.json").is_file()
        assert not (tmp_path / ".decimal" / "install.json").exists()

    def test_the_gitignore_gains_every_missing_line_once(self, tmp_path):
        from decimalai import _install

        state = tmp_path / ".decimal"
        state.mkdir()
        (state / ".gitignore").write_text("install.json\nmy-notes/", encoding="utf-8")

        _install.get_install_identity(str(tmp_path))
        first = (state / ".gitignore").read_bytes()
        _install._ensure_gitignore(state)

        lines = first.decode("utf-8").splitlines()
        assert lines[:2] == ["install.json", "my-notes/"]
        assert sorted(lines) == sorted(set(lines))
        assert set(GITIGNORE_LINES) <= set(lines)
        assert (state / ".gitignore").read_bytes() == first

    def test_there_is_one_gitignore_writer(self, tmp_path, monkeypatch):
        from decimalai import _install

        seen = []
        monkeypatch.setattr(ops, "_ensure_gitignore",
                            lambda project: seen.append(project.directory) or True)
        _install._ensure_gitignore(tmp_path / ".decimal")
        assert seen == [tmp_path.resolve()]

    def test_a_symlinked_gitignore_is_left_alone(self, tmp_path):
        from decimalai import _install

        if os.name != "posix":
            pytest.skip("symlinks")
        state = tmp_path / ".decimal"
        state.mkdir()
        target = tmp_path / "elsewhere"
        target.write_text("x\n", encoding="utf-8")
        (state / ".gitignore").symlink_to(target)
        _install._ensure_gitignore(state)          # never raises from identity code
        assert target.read_text(encoding="utf-8") == "x\n"


# ── decimalai.skills_dir ──────────────────────────────────────────────────────


class TestSkillsDirExport:
    def test_it_is_the_engine_function(self):
        import decimalai
        from decimalai.skillpkg import project

        assert decimalai.skills_dir is project.skills_dir
        assert "skills_dir" in decimalai.__all__
        from decimalai import skills_dir  # noqa: F401  the spelling the docs use

    def test_it_is_imported_on_first_use_only(self):
        code = ("import sys, decimalai; "
                "print(any(m.startswith('decimalai.skillpkg') for m in sys.modules)); "
                "decimalai.skills_dir; "
                "print('decimalai.skillpkg.project' in sys.modules)")
        env = dict(os.environ, DECIMAL_AUTOINIT="false")
        env.pop("DECIMAL_AUTO_TRACE", None)
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                             capture_output=True, text=True, check=True).stdout.split()
        assert out == ["False", "True"]

    def test_it_finds_the_installed_root(self, registry, project, monkeypatch):
        import decimalai

        no_router_http(monkeypatch)
        _refund(registry)
        router().export("acme/refund-policy", project_root=str(project))
        monkeypatch.chdir(project)
        assert decimalai.skills_dir() == project.resolve() / ".agents" / "skills"
        assert json.loads((project / ".decimal" / "skills.lock").read_text())["version"] == 2


# ── install() says, once, that it no longer forks ─────────────────────────────


class TestInstallSaysItNoLongerForks:
    @pytest.mark.parametrize("verb", ["install", "install_skill"])
    def test_a_warning_once_naming_fork(self, verb, registry, project, monkeypatch, caplog):
        no_router_http(monkeypatch)
        _refund(registry)
        registry.add_skill("pdf-tools")
        r = router()
        with caplog.at_level(logging.WARNING, logger="decimalai.skill_router"):
            out = getattr(r, verb)("acme/refund-policy", project_root=str(project))
            getattr(r, verb)("pdf-tools", project_root=str(project))
        notes = [rec for rec in caplog.records if "no longer forks" in rec.getMessage()]
        assert len(notes) == 1 and notes[0].levelno == logging.WARNING
        assert "router.fork(" in notes[0].getMessage()
        assert "fork" in out and out["fork"] is None    # the key stays, for 0.14 readers


# ── instrument(disk_sync=True) says, once, that it writes nothing ─────────────


NOTHING_WRITTEN = "asked for skills on disk"


class TestAnExplicitDiskSyncIsToldNothingIsWritten:
    @pytest.fixture(autouse=True)
    def _no_disk_runtime(self, monkeypatch):
        # Outside Claude Code or Cursor, so their own warning stays out of the way.
        for var in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CURSOR_AGENT"):
            monkeypatch.delenv(var, raising=False)

    @pytest.mark.parametrize("adapter", sorted(ADAPTERS))
    def test_disk_sync_true_warns_once(self, adapter, project, platform, monkeypatch, caplog):
        monkeypatch.chdir(project)
        with caplog.at_level(logging.WARNING):
            ADAPTERS[adapter](disk_sync=True)
            ADAPTERS[adapter](disk_sync=True)
        notes = [rec for rec in caplog.records if NOTHING_WRITTEN in rec.getMessage()]
        assert len(notes) == 1 and notes[0].levelno == logging.WARNING
        message = notes[0].getMessage()
        assert "no longer writes skills to disk" in message
        assert "`decimalai skills add" in message and "`decimalai skills install`" in message

    @pytest.mark.parametrize("adapter", sorted(ADAPTERS))
    def test_harness_authority_warns_once(self, adapter, project, platform, monkeypatch, caplog):
        import decimalai._config as cfg

        monkeypatch.chdir(project)
        monkeypatch.setattr(cfg._config, "skill_authority", "harness")
        with caplog.at_level(logging.WARNING):
            ADAPTERS[adapter]()
        assert len([rec for rec in caplog.records if NOTHING_WRITTEN in rec.getMessage()]) == 1

    @pytest.mark.parametrize("adapter", sorted(ADAPTERS))
    @pytest.mark.parametrize("kwargs", [{}, {"disk_sync": False},
                                        {"enable_skill_loader": True}])
    def test_nobody_asked_means_no_warning(self, adapter, kwargs, project, platform,
                                           monkeypatch, caplog):
        monkeypatch.chdir(project)
        monkeypatch.setattr("decimalai.langchain._install_skill_loader", lambda: None)
        monkeypatch.setattr("decimalai.openai_agents._install_skill_loader", lambda: None)
        with caplog.at_level(logging.WARNING):
            ADAPTERS[adapter](**kwargs)
        assert [rec for rec in caplog.records if NOTHING_WRITTEN in rec.getMessage()] == []

    def test_the_old_mirroring_hint_is_gone(self):
        assert not hasattr(sr, "_hint_disk_mirror_available")
        assert not hasattr(sr, "_disk_mirror_hinted")


# ── a project 0.14 wrote: the version 1 lock fallback ─────────────────────────


V1_NOTICE = "uses a version 1 skills lock"


def _v1(project: Path, skills: Optional[Dict[str, Any]] = None) -> Path:
    """The lock every 0.14 export wrote."""
    state = project / ".decimal"
    state.mkdir(exist_ok=True)
    (state / "skills.lock").write_text(json.dumps({"version": 1, "skills": skills or {}}),
                                       encoding="utf-8")
    return state / "skills.lock"


def _v1_lock(project: Path) -> Dict[str, Any]:
    return json.loads((project / ".decimal" / "skills.lock").read_text(encoding="utf-8"))


def _served_skill_md(registry: FakeRegistry, ref: str, tmp_path: Path) -> bytes:
    """What the engine writes for ``ref`` in a fresh project: the registry's bytes."""
    fresh = tmp_path / "fresh"
    fresh.mkdir(exist_ok=True)
    assert ops.add(ref, directory=fresh).ok
    (folder,) = (fresh / ".agents" / "skills").iterdir()
    return (folder / "SKILL.md").read_bytes()


def _v1_notices(caplog) -> List[logging.LogRecord]:
    return [rec for rec in caplog.records if V1_NOTICE in rec.getMessage()]


class TestVersion1Projects:
    def test_export_writes_the_registry_bytes_and_updates_the_v1_lock(
            self, registry, project, tmp_path, monkeypatch, caplog):
        no_router_http(monkeypatch)
        skill = _refund(registry)
        lock = _v1(project, {"older": {"version": 1, "content_hash": "ab" * 32}})
        r = router()

        with caplog.at_level(logging.WARNING, logger="decimalai.skill_router"):
            summary = r.export("acme/refund-policy", project_root=str(project))

        folder = project / ".agents" / "skills" / "acme-refund-policy"
        written = (folder / "SKILL.md").read_bytes()
        assert written == _served_skill_md(registry, "acme/refund-policy", tmp_path)
        assert (folder / "references" / "fees.md").is_file()
        assert summary["skills_written"] == 1 and summary["errors"] == []
        assert summary["skill_dirname"] == "acme-refund-policy"
        data = json.loads(lock.read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert sorted(data["skills"]) == ["acme/refund-policy", "older"]
        entry = data["skills"]["acme/refund-policy"]
        assert entry["skill_id"] == skill.skill_id and entry["version"] == 1
        assert entry["content_hash"] == hashlib.sha256(written).hexdigest()
        assert not (project / ".decimal" / "skills.toml").exists()   # never migrated
        (notice,) = _v1_notices(caplog)
        assert notice.levelno == logging.WARNING
        assert "`decimalai skills migrate`" in notice.getMessage()
        assert "0.16" in notice.getMessage()

    def test_the_notice_is_logged_once_per_process(self, registry, project, monkeypatch,
                                                   caplog):
        no_router_http(monkeypatch)
        _refund(registry)
        registry.add_skill("pdf-tools")
        _v1(project)
        r = router()
        with caplog.at_level(logging.WARNING, logger="decimalai.skill_router"):
            r.export("acme/refund-policy", project_root=str(project))
            r.export_to_disk(skills=["pdf-tools"], project_root=str(project))
            r.update_skills(project_root=str(project))
        assert len(_v1_notices(caplog)) == 1

    def test_agents_still_pick_the_folders(self, registry, project, monkeypatch):
        no_router_http(monkeypatch)
        _refund(registry)
        _v1(project)
        summary = router().export("acme/refund-policy", agents=["claude-code", "cursor"],
                                  project_root=str(project))
        for base in (".claude/skills", ".agents/skills"):
            assert (project / base / "acme-refund-policy" / "SKILL.md").is_file()
        assert len(summary["written_paths"]) == 2
        assert _v1_lock(project)["skills"]["acme/refund-policy"]["agents"] == [
            "claude-code", "cursor"]

    def test_install_and_export_to_disk_take_the_same_path_and_never_fork(
            self, registry, project, monkeypatch):
        made = no_router_http(monkeypatch)
        _refund(registry)
        registry.add_skill("pdf-tools")
        _v1(project)
        r = router()
        out = r.install("acme/refund-policy", project_root=str(project))
        summary = r.export_to_disk(skills=["pdf-tools"], project_root=str(project))
        assert made == [] and out["fork"] is None
        assert out["export"]["skills_written"] == 1 and summary["skills_written"] == 1
        assert sorted(_v1_lock(project)["skills"]) == ["acme/refund-policy", "pdf-tools"]

    def test_exporting_again_refreshes_its_own_clean_folder(self, registry, project,
                                                            monkeypatch):
        no_router_http(monkeypatch)
        skill = _refund(registry)
        _v1(project)
        r = router()
        r.export("acme/refund-policy", project_root=str(project))
        registry.add_version(skill, "# Refunds\n\nVersion two.\n")

        again = r.export("acme/refund-policy", project_root=str(project))

        md = project / ".agents" / "skills" / "acme-refund-policy" / "SKILL.md"
        assert "Version two." in md.read_text(encoding="utf-8")
        assert again["skills_written"] == 1
        assert _v1_lock(project)["skills"]["acme/refund-policy"]["version"] == 2

    def test_a_folder_edited_by_hand_is_refused_unless_force(self, registry, project,
                                                             monkeypatch):
        no_router_http(monkeypatch)
        _refund(registry)
        _v1(project)
        r = router()
        r.export("acme/refund-policy", project_root=str(project))
        md = project / ".agents" / "skills" / "acme-refund-policy" / "SKILL.md"
        md.write_text(md.read_text(encoding="utf-8") + "\nMy edit.\n", encoding="utf-8")

        with pytest.raises(SkillRouterError):
            r.export("acme/refund-policy", project_root=str(project))
        assert "My edit." in md.read_text(encoding="utf-8")

        forced = r.export("acme/refund-policy", project_root=str(project), force=True)
        assert "My edit." not in md.read_text(encoding="utf-8")
        assert forced["skills_written"] == 1

    def test_a_blocked_skill_is_refused(self, registry, project, monkeypatch):
        no_router_http(monkeypatch)
        _refund(registry, band="blocked")
        _v1(project)
        with pytest.raises(SkillRouterError):
            router().export("acme/refund-policy", project_root=str(project))
        assert not (project / ".agents").exists()

    def test_update_skills_refreshes_what_the_v1_lock_lists(self, registry, project,
                                                            monkeypatch):
        no_router_http(monkeypatch)
        refund = _refund(registry)
        registry.add_skill("pdf-tools")
        _v1(project)
        r = router()
        r.export_to_disk(skills=["acme/refund-policy", "pdf-tools"], project_root=str(project))
        registry.add_version(refund, "# Refunds\n\nVersion two.\n")

        out = r.update_skills(project_root=str(project))

        assert out["updated_skills"] == ["acme/refund-policy"]
        assert out["updated"] == 1 and out["skipped"] == 1 and out["failed"] == 0
        md = project / ".agents" / "skills" / "acme-refund-policy" / "SKILL.md"
        assert "Version two." in md.read_text(encoding="utf-8")
        assert _v1_lock(project)["skills"]["acme/refund-policy"]["version"] == 2
        assert not (project / ".decimal" / "skills.toml").exists()

    def test_update_skills_leaves_a_hand_edit_unless_force(self, registry, project,
                                                           monkeypatch, caplog):
        no_router_http(monkeypatch)
        refund = _refund(registry)
        _v1(project)
        r = router()
        r.export("acme/refund-policy", project_root=str(project))
        md = project / ".agents" / "skills" / "acme-refund-policy" / "SKILL.md"
        md.write_text(md.read_text(encoding="utf-8") + "\nMy edit.\n", encoding="utf-8")
        registry.add_version(refund, "# Refunds\n\nVersion two.\n")

        out = r.update_skills(project_root=str(project))
        assert out["updated"] == 0 and "My edit." in md.read_text(encoding="utf-8")
        assert out["failed"] == 1 and "acme/refund-policy" in out["error"]

        forced = r.update_skills(project_root=str(project), force=True)
        assert forced["updated"] == 1
        assert "Version two." in md.read_text(encoding="utf-8")

    def test_pull_missing_puts_back_what_the_v1_lock_lists(self, registry, project,
                                                           monkeypatch):
        no_router_http(monkeypatch)
        _refund(registry)
        registry.add_skill("pdf-tools")
        _v1(project)
        r = router()
        r.export_to_disk(skills=["acme/refund-policy", "pdf-tools"], project_root=str(project))
        import shutil

        shutil.rmtree(project / ".agents" / "skills" / "pdf-tools")

        out = r.pull_missing(project_root=str(project))

        assert out["pulled"] == 1 and out["pulled_skills"] == ["pdf-tools"]
        assert out["updated"] == 0 and out["skipped"] == 1
        assert (project / ".agents" / "skills" / "pdf-tools" / "SKILL.md").is_file()

    def test_pull_missing_with_disk_wins_never_replaces_a_folder(self, registry, project,
                                                                 monkeypatch):
        no_router_http(monkeypatch)
        refund = _refund(registry)
        _v1(project)
        r = router()
        r.export("acme/refund-policy", project_root=str(project))
        before = (project / ".agents" / "skills" / "acme-refund-policy" / "SKILL.md").read_bytes()
        registry.add_version(refund, "# Refunds\n\nVersion two.\n")

        kept = r.pull_missing(project_root=str(project), disk_wins=True)
        assert kept["updated"] == 0 and kept["skipped"] == 1
        assert (project / ".agents" / "skills" / "acme-refund-policy" / "SKILL.md"
                ).read_bytes() == before

        moved = r.pull_missing(project_root=str(project))
        assert moved["updated"] == 1

    def test_a_v2_project_still_uses_the_engine(self, registry, project, monkeypatch, caplog):
        no_router_http(monkeypatch)
        _refund(registry)
        with caplog.at_level(logging.WARNING, logger="decimalai.skill_router"):
            router().export("acme/refund-policy", project_root=str(project))
        assert _v1_notices(caplog) == []
        assert json.loads((project / ".decimal" / "skills.lock").read_text())["version"] == 2
