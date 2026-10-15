"""The guards that keep this suite honest. None of them needs a framework installed.

* every loader module only observes (no ``assert``, no test framework import),
  and the scan is seen to find a planted one;
* every module on disk is registered, and every registered loader has a module;
* every loader belongs to a test environment whose extras install it, and the
  loaders graded in the other environment are named on every run;
* both environments collect this directory (the openai 2 slice included);
* no skip in this directory is undeclared;
* every ledger line names a real cell, and its budget says how many there are;
* ``NATIVE_LOADERS_SNAPSHOT`` is a vendored literal that equals the registered
  loaders, and the docs loader table agrees with it (when the docs checkout and
  its page are on disk);
* the two test environments stay apart: neither pulls in the other's world.
"""

from __future__ import annotations

import ast
import fnmatch
import re
from pathlib import Path
from typing import List, Set

import pytest

from tests.conformance import environments as envs

from . import contract, ledger
from .loaders import (
    LOADER_MODULES,
    LOADERS,
    LOADERS_BY_NAME,
    NATIVE_LOADERS_SNAPSHOT,
    items_for,
)

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
REPO = TESTS.parent
DOCS_PAGE = REPO.parent / "decimalai-docs" / "guides" / "native-skill-loaders.mdx"


def _canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# ── observers only ───────────────────────────────────────────────────────────


def _observer_files(root: Path = HERE) -> List[Path]:
    return sorted((root / "loaders").glob("*.py")) + sorted((root / "stubs").glob("*.py"))


def _observer_offenders(root: Path = HERE) -> List[str]:
    """Every ``assert`` and test-framework import in ``root``'s loader modules and stubs."""
    offenders = []
    for path in _observer_files(root):
        rel = path.relative_to(root).as_posix()
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Assert):
                offenders.append(f"{rel}:{node.lineno} assert")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + (
                    [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                if any(n.split(".")[0] in {"pytest", "unittest"} for n in names):
                    offenders.append(f"{rel}:{node.lineno} imports a test framework")
    return sorted(offenders)


def test_loader_modules_contain_no_assertions():
    """A loader module that asserts has stolen the contract's job."""
    offenders = _observer_offenders()
    assert not offenders, ("loader modules and stubs only observe; move the check into "
                           f"contract.py so every loader gets it: {offenders}")


def test_the_guard_above_bites(tmp_path):
    """Proof the AST guard is not vacuous: run on planted modules, it finds each offence;
    run on this directory, it reads every loader module and every stub."""
    (tmp_path / "loaders").mkdir()
    (tmp_path / "stubs").mkdir()
    (tmp_path / "loaders" / "adk.py").write_text("def observe(ctx):\n    sc = ctx\n    assert sc\n")
    (tmp_path / "loaders" / "clean.py").write_text("def observe(ctx):\n    return {}\n")
    (tmp_path / "stubs" / "model.py").write_text("import pytest\nfrom unittest import mock\n")
    assert _observer_offenders(tmp_path) == [
        "loaders/adk.py:3 assert",
        "stubs/model.py:1 imports a test framework",
        "stubs/model.py:2 imports a test framework",
    ]
    scanned = _observer_files()
    on_disk = sorted(p for d in ("loaders", "stubs") for p in (HERE / d).iterdir()
                     if p.suffix == ".py")
    assert scanned == on_disk, f"the guard reads {scanned}, not every module on disk"
    loader_modules = {p.stem for p in scanned if p.parent.name == "loaders"}
    assert set(LOADER_MODULES) <= loader_modules and len(LOADER_MODULES) == 8, sorted(loader_modules)
    assert any(p.parent.name == "stubs" and p.stem != "__init__" for p in scanned), scanned


def test_every_loader_module_on_disk_is_registered():
    on_disk = {p.stem for p in (HERE / "loaders").glob("*.py") if not p.stem.startswith("_")}
    assert on_disk == set(LOADER_MODULES), (
        f"loaders/ holds {sorted(on_disk)}; LOADERS registers {sorted(LOADER_MODULES)}")
    assert len(LOADERS) == 8 and len({l.name for l in LOADERS}) == 8


def test_every_registered_loader_has_its_two_entry_points():
    for loader in LOADERS:
        tree = ast.parse((HERE / "loaders" / f"{loader.module}.py").read_text())
        defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        assert {"observe", "discover"} <= defined, f"{loader.module}.py defines {sorted(defined)}"
        names = [n for n in tree.body if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", None) == "NAME" for t in n.targets)]
        assert names and names[0].value.value == loader.name, f"{loader.module}.NAME != {loader.name}"


# ── environments ─────────────────────────────────────────────────────────────


def test_every_loader_belongs_to_an_environment():
    assert set(envs.LOADER_ENVS) == {l.name for l in LOADERS}, (
        "tests/conformance/environments.py:LOADER_ENVS must name every loader, and only "
        f"those: {sorted(envs.LOADER_ENVS)} vs {sorted(l.name for l in LOADERS)}")
    for name, homes in envs.LOADER_ENVS.items():
        assert homes and homes <= set(envs.TEST_ENVS), f"{name}: {sorted(homes)}"


@pytest.mark.parametrize("loader", [l.name for l in LOADERS])
def test_each_loader_is_installed_by_the_extras_of_its_environment(loader):
    """A loader whose environment does not install it would be graded nowhere."""
    for env in sorted(envs.LOADER_ENVS[loader]):
        declared = {_canon(r.split(";")[0].split("[")[0].split(">")[0].split("=")[0].split("<")[0])
                    for r in envs.environment_requirements(env)}
        missing = [d for d in LOADERS_BY_NAME[loader].distributions if _canon(d) not in declared]
        assert not missing, (
            f"{loader} is graded in {env}, whose extras ({envs.TEST_ENVS[env]}) do not install "
            f"{missing}. Add them to that environment's conformance extra in pyproject.toml.")


def test_loaders_graded_in_another_environment_are_named(record_property):
    env = envs.current_env()
    elsewhere = sorted(f"{name} (graded in {home})" for name in envs.LOADER_ENVS
                       if (home := envs.loader_graded_elsewhere(name, env)))
    record_property("native_loaders_graded_elsewhere", ",".join(elsewhere))
    print(f"\nNATIVE LOADERS in {env!r}: graded elsewhere: {'; '.join(elsewhere) or 'none'}")


def test_both_environments_collect_this_suite():
    """The openai 2 slice collects only listed files; this directory must be one of them."""
    import sys

    root_conftest = next(m for m in list(sys.modules.values())
                         if (getattr(m, "__file__", None) or "")
                         and Path(m.__file__).resolve() == TESTS / "conftest.py")
    for path in sorted(HERE.glob("test_*.py")):
        rel = path.relative_to(TESTS).as_posix()
        assert any(fnmatch.fnmatch(rel, pat) for pat in root_conftest.OPENAI2_UNIT_TESTS), (
            f"{rel} is not in tests/conftest.py:OPENAI2_UNIT_TESTS, so the openai 2 run never "
            "grades CrewAI or Strands")
    assert "strands" in root_conftest.OPENAI2_REQUIRED_MODULES, (
        "tests/conftest.py:OPENAI2_REQUIRED_MODULES must require strands, or the openai 2 "
        "run passes by skipping it")


def test_the_two_worlds_stay_apart_in_this_environment():
    """Neither environment's extras pull in the other's frameworks (read from what is installed)."""
    env = envs.current_env()
    closure = envs.installed_closure(envs.environment_requirements(env) + ["decimalai"])
    other = {"openai3": {"crewai", "litellm", "strands-agents"},
             "openai2": {"openai-agents", "pydantic-ai-slim", "langchain", "deepagents",
                         "agent-framework-core", "pydantic-ai-harness"}}[env]
    assert not closure & other, f"{env}'s extras pull in {sorted(closure & other)}"


# ── skips ────────────────────────────────────────────────────────────────────


def _skip_call_sites(path: Path) -> List[tuple]:
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            func = getattr(inner, "func", None)
            if (isinstance(inner, ast.Call) and isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name) and func.value.id == "pytest"
                    and func.attr in ("skip", "importorskip")):
                out.append((node.name, inner.lineno, f"pytest.{func.attr}"))
    return out


def test_no_undeclared_skip_call_sites():
    """Every skip here is a declared kind (``ledger.skip_declared``) or a named environment skip."""
    allowed = set(ledger.ENVIRONMENT_SKIPS) | {"skip_declared"}
    offenders = [f"{p.relative_to(HERE)}:{line} {call} in {fn}()"
                 for p in sorted(HERE.rglob("*.py")) for fn, line, call in _skip_call_sites(p)
                 if fn not in allowed]
    assert not offenders, ("undeclared skip call site(s):\n  " + "\n  ".join(offenders)
                           + "\nRoute it through ledger.skip_declared, or declare the function "
                           "in ledger.ENVIRONMENT_SKIPS with its condition.")


def test_every_environment_skip_names_a_real_test():
    defined: Set[str] = set()
    for p in HERE.rglob("*.py"):
        defined |= {n.name for n in ast.walk(ast.parse(p.read_text()))
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert not set(ledger.ENVIRONMENT_SKIPS) - defined


# ── the ledger ───────────────────────────────────────────────────────────────


def test_every_ledger_line_names_a_real_cell():
    lines = (list(ledger.DECLARED_NA) + list(ledger.UPSTREAM_GAPS) + list(ledger.PROOFS)
             + sorted(ledger.NEGATIVE_CONTROL_PREDICTED)
             + [f"{l}:{i}" for l, items in ledger.NEGATIVE_CONTROL_RED.items() for i in items])
    orphans = [k for k in lines if k.partition(":")[0] not in LOADERS_BY_NAME
               or k.partition(":")[2] not in items_for(k.partition(":")[0])]
    assert not orphans, f"ledger lines that name no cell: {orphans}"
    assert set(ledger.NOISE_ALLOWLIST) <= set(LOADERS_BY_NAME)
    assert not set(ledger.DECLARED_NA) & set(ledger.UPSTREAM_GAPS)
    assert ledger.NA_BUDGET == len(ledger.DECLARED_NA), "move NA_BUDGET deliberately"
    assert ledger.GAP_BUDGET == len(ledger.UPSTREAM_GAPS), "move GAP_BUDGET deliberately"
    assert set(contract.GRADERS) == set(contract.ITEMS)


def test_every_proof_key_is_produced_by_its_loader():
    """A ledger proof reading a key no loader module writes would never hold."""
    for key in list(ledger.DECLARED_NA) + list(ledger.PROOFS):
        loader = LOADERS_BY_NAME[key.partition(":")[0]]
        source = (HERE / "loaders" / f"{loader.module}.py").read_text()
        exemption = ledger.DECLARED_NA.get(key) or ledger.PROOFS[key]
        wanted = re.findall(r"proofs\[(?:'|\")([a-z_]+)", exemption.check({"proofs": {}})[1])
        for proof in wanted:
            assert f'proofs"]["{proof}"]' in source, f"{key}: {loader.module}.py never sets {proof}"


# ── docs ─────────────────────────────────────────────────────────────────────


def loaders_from_docs(text: str) -> Set[str]:
    """Loader ids from the docs page: the inline-code ids in its ``| Loader |`` table's first column."""
    ids: Set[str] = set()
    in_table = False
    for line in text.splitlines():
        if line.startswith("| Loader |"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            first = line.strip("|").split("|")[0]
            if set(first.strip()) <= set("-: "):
                continue
            ids |= set(re.findall(r"`([a-z0-9-]+)`", first))
    return ids


def test_the_docs_table_parser_reads_ids():
    page = ("intro\n\n| Loader | Snippet |\n|---|---|\n| Claude Code (`claude-agent-sdk`) | x |\n"
            "| Google ADK (`google-adk`) | y |\n\nafter | `not-a-loader` |\n")
    assert loaders_from_docs(page) == {"claude-agent-sdk", "google-adk"}


def test_the_snapshot_names_every_loader():
    assert NATIVE_LOADERS_SNAPSHOT == {l.name for l in LOADERS}, (
        "NATIVE_LOADERS_SNAPSHOT (what the docs advertise) and the registered loaders disagree")


def test_the_snapshot_is_vendored_not_derived():
    """Derived from ``LOADERS``, the equality above could never fail: the snapshot must be
    a literal set of ids, edited by hand when the docs table changes."""
    tree = ast.parse((HERE / "loaders" / "__init__.py").read_text())
    value = next(n.value for n in tree.body
                 if isinstance(n, (ast.Assign, ast.AnnAssign))
                 and "NATIVE_LOADERS_SNAPSHOT" in [getattr(t, "id", None) for t in
                                                   (n.targets if isinstance(n, ast.Assign) else [n.target])])
    assert (isinstance(value, ast.Call) and getattr(value.func, "id", None) == "frozenset"
            and len(value.args) == 1 and not value.keywords), ast.dump(value)
    ids = ast.literal_eval(value.args[0])
    assert isinstance(ids, (set, list, tuple)) and all(isinstance(i, str) for i in ids), ast.dump(value)
    assert frozenset(ids) == NATIVE_LOADERS_SNAPSHOT


def test_loader_table_matches_the_docs():
    if not DOCS_PAGE.is_file():
        pytest.skip(f"{DOCS_PAGE} is not on disk (no docs checkout, or the page is not "
                    "written yet); NATIVE_LOADERS_SNAPSHOT stands in for it")
    advertised = loaders_from_docs(DOCS_PAGE.read_text())
    assert advertised == set(NATIVE_LOADERS_SNAPSHOT), (
        f"the docs loader table lists {sorted(advertised)}; this suite grades "
        f"{sorted(NATIVE_LOADERS_SNAPSHOT)}. A loader the docs advertise needs a module "
        "here; a module for a loader the docs dropped is dead weight.")


# ── the ledger matrix (run in full by the release gate, smoke-tested here) ───


def test_every_defect_still_applies_to_the_oracle(tmp_path):
    """A defect whose edit no longer matches the oracle's bytes would measure the baseline."""
    from . import matrix

    _, base_expect, _ = matrix.project_for("baseline", tmp_path / "baseline")
    baseline = {p.relative_to(tmp_path / "baseline").as_posix(): p.read_bytes()
                for p in (tmp_path / "baseline").rglob("*") if p.is_file()}
    for i, defect in enumerate(matrix.DEFECTS):
        if defect == "baseline":
            continue
        project, expect, target = matrix.project_for(defect, tmp_path / f"d{i}")
        files = {p.relative_to(project).as_posix(): (p.read_bytes() if not p.is_symlink() else b"->")
                 for p in project.rglob("*") if p.is_file() or p.is_symlink()}
        assert files != baseline, f"{defect!r} changed nothing"
        assert target in expect and len(expect) == len(base_expect), defect


def test_the_classifier_names_each_outcome():
    from . import matrix

    want = {"a": {}, "b": {}, "t": {}}
    assert matrix.classify({"listed": ["a", "b", "t"], "channel": []}, want, "t") == "OK"
    assert matrix.classify({"listed": ["a", "b", "t"], "channel": ["x"]}, want, "t") == "OK (+w)"
    assert matrix.classify({"listed": None, "raised": "E"}, want, "t") == "LOUD"
    assert matrix.classify({"listed": ["a", "b"], "channel": ["x"]}, want, "t") == "WARNED"
    assert matrix.classify({"listed": ["a", "b"], "channel": []}, want, "t") == "SILENT"
    assert matrix.classify({"listed": ["a", "b", "T/x"], "channel": []}, want, "t") == "RENAMED"
    assert matrix.classify({"listed": ["a", "b", "t", "z"], "channel": []}, want, "t") == "EXTRA"
    assert matrix.classify(RuntimeError("x"), want, "t") == "CRASH"


def test_the_ledger_matrix_holds_every_cell():
    import json

    from . import matrix

    recorded = json.loads(matrix.MATRIX_PATH.read_text())
    assert list(recorded["cells"]) == list(matrix.DEFECTS), "defects and recorded rows disagree"
    names = {l.name for l in LOADERS}
    for defect, row in recorded["cells"].items():
        assert set(row) == names, f"{defect}: cells for {sorted(row)}"
        assert set(row.values()) <= set(matrix.CLASSES), f"{defect}: {row}"
    assert set(recorded["cells"]["baseline"].values()) == {"OK"}, "the oracle is not clean everywhere"
    assert set(recorded["measured_at"]) == names


def test_the_matrix_runs_here_and_bites_on_a_flipped_cell(tmp_path, monkeypatch):
    """One real cell per environment, end to end: matches the ledger, then fails on a lie."""
    import json

    from . import matrix

    here = matrix.graded_here()
    cheapest = next((n for n in ("agent-framework", "strands-agents") if n in here), None)
    if cheapest is None:
        ledger.skip_declared("unavailable", "matrix:*", "no cheap loader is runnable here")
    args = ["--check-ledger", "--defects", "baseline", "--loaders", cheapest]
    assert matrix.main(args) == 0
    lie = json.loads(matrix.MATRIX_PATH.read_text())
    lie["cells"]["baseline"][cheapest] = "SILENT"
    flipped = tmp_path / "ledger_matrix.json"
    flipped.write_text(json.dumps(lie))
    monkeypatch.setattr(matrix, "MATRIX_PATH", flipped)
    assert matrix.main(args) == 1


# ── isolation (the Claude Code subprocess above all) ─────────────────────────

_A12 = {
    "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",
    "GIT_SSH_COMMAND": "/usr/bin/false",
    "HTTP_PROXY": "http://127.0.0.1:9",
    "HTTPS_PROXY": "http://127.0.0.1:9",
}


def test_every_child_runs_isolated_from_the_parent_session(tmp_path):
    from . import runner

    polluted = {"PATH": "/usr/bin:/bin", "CLAUDECODE": "1", "ANTHROPIC_API_KEY": "sk-real",
                "CLAUDE_CODE_OAUTH_TOKEN": "x", "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
                "HOME": "/Users/someone"}
    env = runner.child_env(tmp_path, base=polluted)
    leaked = sorted(k for k in env if k.startswith(("CLAUDE", "ANTHROPIC")) and k not in (
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL",
        "CLAUDE_CONFIG_DIR"))
    assert not leaked, f"the parent session leaks into the child: {leaked}"
    assert env["HOME"] == str(tmp_path) and env["CLAUDE_CONFIG_DIR"] == str(tmp_path / ".claude")
    assert {k: env.get(k) for k in _A12} == _A12
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null" and env["NO_PROXY"] == "127.0.0.1,localhost"


def test_the_claude_session_sets_its_own_isolation():
    """The CLI gets the stub, a dummy key, its own HOME and the SSH block in its options too."""
    source = (HERE / "loaders" / "claude.py").read_text()
    wanted = [f'"{k}": "{v}"' for k, v in _A12.items()] + [
        '"ANTHROPIC_API_KEY": "dummy"', '"ANTHROPIC_BASE_URL": stub_url',
        '"HOME": str(ctx.home)', '"CLAUDE_CONFIG_DIR": str(ctx.home / ".claude")',
        'permission_mode="default"', "setting_sources=[]"]
    missing = [w for w in wanted if w not in source]
    assert not missing, f"loaders/claude.py no longer sets: {missing}"
