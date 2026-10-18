"""Fixtures for the native-mode tests: one module per framework, each driving that
framework's own skill loader with a scripted model and grading the activation DecimalAI
records for it.

``native_project`` is the project ``decimalai skills add`` writes (the native-loader
suite's oracle, byte for byte), in a fresh directory that is also the working directory.
``update_to_v2()`` then does what ``decimalai skills update`` does to the target skill:
the folder is replaced whole by version 2 (a new version id, a new body sentinel) and the
lock moves with it. A loader that cached version 1 still delivers version 1, and the
activation must say so.
"""

from __future__ import annotations

import dataclasses
import shutil
import uuid
from pathlib import Path
from typing import Dict

import pytest
from skillevaluation import skillfmt

from decimalai import _native
from decimalai.skillpkg import identity
from tests.native_loaders import oracle

TARGET = oracle.SCENARIO.target
V2_VERSION_ID = str(uuid.UUID(int=0xF00D0F00))
V2_SENTINEL = "SENTINEL-NATIVE-REFUND-V2-9e1b"


def _rows_v2():
    first = dataclasses.replace(oracle.ROWS[0], version_id=V2_VERSION_ID, version=4,
                                sentinel=V2_SENTINEL)
    return (first,) + tuple(oracle.ROWS[1:])


@dataclasses.dataclass
class NativeProject:
    root: Path
    lock: dict

    @property
    def skills_dir(self) -> Path:
        return self.root / ".agents" / "skills"

    @property
    def target(self) -> str:
        return TARGET

    def entry(self, install_name: str = TARGET) -> dict:
        return self.lock["skills"][install_name]

    def expected(self, install_name: str = TARGET) -> Dict[str, str]:
        """The ``active_skills`` entry a load of the installed version must produce."""
        e = self.entry(install_name)
        skill = identity.ManagedSkill(
            install_name=install_name, registry_name=e["name"], skill_id=e["skill_id"],
            version_id=e["version_id"], body_sha256=e["body_sha256"], path="")
        return {"name": skill.wire_name, "skill_id": e["skill_id"],
                "version_id": e["version_id"], "body_sha256": e["body_sha256"]}

    def sentinel(self, install_name: str = TARGET) -> str:
        return oracle.sentinel_of(install_name)

    def update_to_v2(self) -> Dict[str, str]:
        """Replace the target skill with version 2, folder and lock, and return the
        ``active_skills`` entry a load of version 2 must produce."""
        built = oracle.build(_rows_v2())
        folder = f"{oracle.ROOT}/{TARGET}"
        shutil.rmtree(self.root / folder)
        for rel, data in built.files.items():
            if rel.startswith(folder + "/") or not rel.startswith(oracle.ROOT + "/"):
                path = self.root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                path.chmod(0o755 if rel in built.executable else 0o644)
        self.lock = built.lock
        identity.clear_cache()
        return self.expected()


@pytest.fixture
def native_project(tmp_path, monkeypatch) -> NativeProject:
    root = (tmp_path / "project").resolve()
    root.mkdir()
    (root / ".git").mkdir()
    lock = oracle.write(root)
    monkeypatch.chdir(root)
    identity.clear_cache()
    _native._reset_seen()
    _native._reset_warnings()
    yield NativeProject(root=root, lock=lock)
    identity.clear_cache()
    _native._reset_seen()


def body_of(skill_md: bytes) -> str:
    parts = skillfmt.split_frontmatter(skill_md.decode("utf-8"))
    return parts[1] if parts else skill_md.decode("utf-8")


#: Each framework's own test module -> the native loader it exercises (its LOADER_ENVS name).
_FRAMEWORK_OF_MODULE = {
    "test_claude_agent_sdk_native": "claude-agent-sdk",
    "test_adk_native": "google-adk",
    "test_crewai_native": "crewai",
    "test_openai_agents_native": "openai-agents",
    "test_deepagents_native": "deepagents",
    "test_agent_framework_native": "agent-framework",
    "test_pydantic_ai_harness_native": "pydantic-ai-harness",
    "test_strands_native": "strands-agents",
}


@pytest.fixture(autouse=True)
def _framework_present(request):
    """A framework's test module runs where its framework is graded and installed.

    Graded in the other test environment: a declared skip naming it. Not installed with no
    environment named (CI's ``.[dev]`` floors job, a bare local run): a declared skip.
    Not installed in an environment that says it holds the framework: a failure, as
    everywhere in these suites (``ledger.skip_declared``).
    """
    module = request.node.module.__name__.rsplit(".", 1)[-1]
    framework = _FRAMEWORK_OF_MODULE.get(module)
    if framework is None:
        return
    from tests.conformance.environments import loader_graded_elsewhere
    from tests.native_loaders import ledger

    if loader_graded_elsewhere(framework):
        ledger.skip_declared("elsewhere", f"{framework}:native", f"{framework} native mode")
    missing = missing_imports(framework)
    if missing:
        ledger.skip_declared("unavailable", f"{framework}:native",
                             f"{framework} is not installed (no {', '.join(missing)})")


def missing_imports(framework: str) -> list:
    """The top-level imports ``framework`` needs that are not installed. ``find_spec`` of a
    dotted name imports its parent, so an absent parent (``google`` for ``google.adk``)
    raises instead of answering None."""
    import importlib.util

    from tests.native_loaders.loaders import LOADERS_BY_NAME

    out = []
    for module in LOADERS_BY_NAME[framework].imports:
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            out.append(module)
    return out
