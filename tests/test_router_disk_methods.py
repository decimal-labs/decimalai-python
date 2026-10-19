"""SkillRouter's disk methods, in a project the package manager (``decimalai.skillpkg``)
manages. There is one engine and one way a registry skill reaches disk:

- ``export(name)`` is ``skillpkg.add(name)``;
- ``update_skills(force=...)`` is ``skillpkg.update(force=...)``;
- ``status(project_root)`` is ``skillpkg.status(directory=project_root)``;
- a project with a version 1 lock (decimalai 0.14 and older) is refused with the
  command that upgrades it, ``decimalai skills migrate``;
- ``instrument()`` never writes a skills directory, and its sync-up never uploads a
  folder the package manager installed;
- the install identity finds its project the way the engine does, and its
  ``.decimal/.gitignore`` is the engine's;
- ``decimalai.skills_dir`` is the engine's ``skills_dir``, imported on first use.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import httpx
import pytest
from skillevaluation import skillfmt

from decimalai.skill_router import SkillRouter, SkillRouterError
from decimalai.skillpkg import lock as L
from decimalai.skillpkg import ops
from decimalai.skillpkg.client import RegistryClient
from decimalai.skillpkg.errors import ExitCode, Result
from decimalai.skillpkg.project import GITIGNORE_LINES, Project
from tests.fakes.fake_registry import BASE_URL, OTHER_ORG, OWNER_KEY, FakeRegistry

REPO = Path(__file__).resolve().parent.parent


# ── helpers other test files import ───────────────────────────────────────────


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


def _version_1_lock(project: Path) -> None:
    """The lock decimalai 0.14 wrote."""
    (project / ".decimal").mkdir()
    (project / ".decimal" / "skills.lock").write_text(
        json.dumps({"version": 1, "skills": {}}), encoding="utf-8")


# ── each disk method is one engine call ───────────────────────────────────────


class _Recorder:
    """Stands in for one ``skillpkg.ops`` function and records each call."""

    def __init__(self, result: Optional[Result] = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.result = result or Result([ops.Change("add", "pdf", "pdf", version=1)])

    def __call__(self, *args, **kwargs):
        self.calls.append({"args": args, **kwargs})
        return self.result


class TestEachDiskMethodIsOneEngineCall:
    def test_export_is_add(self, project, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(ops, "add", recorder)
        router().export("pdf", project_root=str(project))
        (c,) = recorder.calls
        assert c["args"] == ("pdf",) and c["directory"] == str(project)
        assert c["api_key"] == OWNER_KEY and c["base_url"] == BASE_URL
        assert c.get("channel", "sdk") == "sdk"

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

    def test_a_refusal_raises_and_writes_nothing(self, registry, project, monkeypatch):
        no_router_http(monkeypatch)
        with pytest.raises(SkillRouterError) as ei:
            router().export("no-such-skill", project_root=str(project))
        assert "no-such-skill" in str(ei.value)
        assert ei.value.result.exit_code == ExitCode.UNAVAILABLE
        assert not (project / ".agents").exists()

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

    def test_export_refuses_a_version_1_project_with_the_command_that_upgrades_it(
            self, registry, project, monkeypatch):
        made = no_router_http(monkeypatch)
        _refund(registry)
        _version_1_lock(project)
        before = _tree(project)

        with pytest.raises(SkillRouterError) as ei:
            router().export("acme/refund-policy", project_root=str(project))

        assert "decimalai skills migrate" in str(ei.value)
        assert _tree(project) == before and made == []

    def test_update_skills_refuses_a_version_1_project_the_same_way(
            self, registry, project, monkeypatch):
        made = no_router_http(monkeypatch)
        _refund(registry)
        _version_1_lock(project)
        before = _tree(project)

        out = router().update_skills(project_root=str(project))

        assert out["updated"] == 0 and out["failed"] == 1
        assert "decimalai skills migrate" in out["error"]
        assert _tree(project) == before and made == []


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
        # A skills directory that exists is not a request to fill it.
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
    def test_instrument_never_runs_the_engine(self, adapter, project, platform, monkeypatch):
        monkeypatch.chdir(project)
        calls = []
        for name in ("add", "install", "update", "status"):
            monkeypatch.setattr(ops, name, lambda *a, _n=name, **kw: calls.append(_n))
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
