"""`export()` installs the files and takes no copy; `install()` is now the same verb.

`install()` used to fork and then write to disk, so asking for a file also took an
editable copy of the skill. `export()` split the file half out; now both are
`decimalai skills add` underneath, and `install()` is a deprecated alias that no longer
forks (a fork is `fork()`, a link is `use()`).
"""
from __future__ import annotations

import os

import pytest

from tests.fakes.fake_registry import FakeRegistry
from tests.test_router_skillpkg_shims import _fresh_once_state, engine_on, no_router_http, router

__all__ = ["_fresh_once_state"]  # the autouse fixture, imported for this module too


@pytest.fixture
def registry(monkeypatch):
    reg = FakeRegistry()
    engine_on(reg, monkeypatch)
    reg.add_skill("pdf", body="# Review\n\nCheck the locks.\n")
    return reg


def _read(root) -> bytes:
    with open(os.path.join(str(root), ".agents", "skills", "pdf", "SKILL.md"), "rb") as f:
        return f.read()


def test_export_writes_the_file_and_forks_nothing(tmp_path, registry, monkeypatch):
    made = no_router_http(monkeypatch)
    summary = router().export("pdf", project_root=str(tmp_path))

    assert summary["skills_written"] == 1
    assert b"Check the locks." in _read(tmp_path)
    assert made == [], "export took a copy, or read /body"


def test_install_no_longer_forks(tmp_path, registry, monkeypatch):
    """The deprecated shape keeps its signature and its keys; the fork side effect is
    gone (it inflated install counts and handed out copies nobody asked for)."""
    made = no_router_http(monkeypatch)
    out = router().install("pdf", project_root=str(tmp_path))
    assert made == []
    assert out["fork"] is None and out["skill_name"] == "pdf"


def test_export_and_install_write_the_same_bytes(tmp_path, registry, monkeypatch):
    no_router_http(monkeypatch)
    a, b = tmp_path / "via-export", tmp_path / "via-install"
    a.mkdir()
    b.mkdir()
    router().export("pdf", project_root=str(a))
    router().install("pdf", project_root=str(b))
    assert _read(a) == _read(b)


def test_running_it_again_is_a_no_op(tmp_path, registry, monkeypatch):
    no_router_http(monkeypatch)
    r = router()
    r.export("pdf", project_root=str(tmp_path))
    before = _read(tmp_path)
    again = r.export("pdf", project_root=str(tmp_path))
    assert again["skills_written"] == 0 and again["errors"] == []
    assert again["skill_dirname"] == "pdf"
    assert _read(tmp_path) == before
