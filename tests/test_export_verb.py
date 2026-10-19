"""`export()` installs the files and takes no copy.

The old `install()` forked and then wrote to disk, so asking for a file also took an
editable copy of the skill. `export()` is `decimalai skills add` underneath and takes no
copy: a fork is `fork()`, a link is `use()`.
"""
from __future__ import annotations

import os

import pytest

from tests.fakes.fake_registry import FakeRegistry
from tests.test_router_disk_methods import engine_on, no_router_http, router


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


def test_running_it_again_is_a_no_op(tmp_path, registry, monkeypatch):
    no_router_http(monkeypatch)
    r = router()
    r.export("pdf", project_root=str(tmp_path))
    before = _read(tmp_path)
    again = r.export("pdf", project_root=str(tmp_path))
    assert again["skills_written"] == 0 and again["errors"] == []
    assert again["skill_dirname"] == "pdf"
    assert _read(tmp_path) == before
