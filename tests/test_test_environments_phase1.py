"""Each test environment is exactly what its extras install, and every test runs in one.

The SDK is tested in two environments, because no single one can hold every
framework it supports (openai-agents / pydantic-ai need openai>=3; crewai,
litellm and llama-index-llms-openai pin openai<3). ``tests/conformance/
environments.py`` names each one by the extras that build it. These tests hold
four things to that definition:

* a run that NAMES its environment (``DECIMAL_TEST_ENV=openai3|openai2``, what
  CI, the publish gate and ``bin/wt land`` set) refuses to start when a
  distribution those extras require is missing or below its floor. Without that,
  a job that installed less than its extras reads green while the tests that
  needed the missing frameworks skip, and a stale venv fails on unrelated tests
  instead of saying it is stale;
* no unit test skips for want of a module that neither environment installs —
  it would run nowhere a clean environment is built, only on a developer's
  fuller venv;
* every CI step that names an environment has installed that environment's
  extras first, and the unit suite runs in both environments, in CI and in the
  publish gate;
* the expansion and the check themselves bite.
"""

from __future__ import annotations

import ast
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import pytest
from packaging.requirements import Requirement

from tests.conformance import environments as envs

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def _canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _names(requirements: List[str]) -> set:
    return {_canon(Requirement(r).name) for r in requirements}


# ── the definition ────────────────────────────────────────────────────────────


def test_each_environment_is_named_by_the_extras_that_build_it():
    assert envs.TEST_ENV_EXTRAS == {
        "openai3": ("dev", "conformance-tests"),
        "openai2": ("conformance-tests-openai2",),
    }
    for env, extras in envs.TEST_ENV_EXTRAS.items():
        assert envs.TEST_ENVS[env] == f'pip install -e ".[{",".join(extras)}]"'


def test_an_environment_requires_every_line_its_extras_expand_to():
    openai3 = _names(envs.environment_requirements("openai3"))
    openai2 = _names(envs.environment_requirements("openai2"))
    # Through `dev`, `conformance-tests` -> `decimalai[all]` / `[providers-tests]`.
    assert {"pytest", "anthropic", "pydantic-ai-slim", "google-adk", "claude-agent-sdk",
            "openai-agents", "langchain-anthropic", "llama-index-core"} <= openai3
    # Through `conformance-tests-openai2` -> `decimalai[test-toolchain,evals,adk,...]`.
    assert {"pytest", "crewai", "litellm", "llama-index-llms-openai", "google-adk"} <= openai2
    assert "decimalai" not in openai3 | openai2, "a self-reference was not expanded"
    # The native-loader suite (tests/native_loaders): three loaders with no DecimalAI
    # adapter ride in openai3; Strands rides in openai2, because its `mcp<2.2` cap would
    # move mcp under claude-agent-sdk and openai-agents in openai3.
    assert {"deepagents", "agent-framework-core", "pydantic-ai-harness"} <= openai3
    assert "strands-agents" in openai2
    assert "strands-agents" not in openai3
    assert not openai2 & {"deepagents", "agent-framework-core", "pydantic-ai-harness"}
    # The two worlds stay apart: neither environment asks for the other's openai.
    assert not openai3 & {"crewai", "litellm", "llama-index-llms-openai"}
    assert not openai2 & {"openai-agents", "pydantic-ai-slim", "langchain"}


_SAMPLE = """\
[project]
dependencies = ["not-an-extra>=1"]

[project.optional-dependencies]
base = ["fw-a>=2.0"]
toolchain = [
    "pytest>=9.0",  # a comment that quotes "not-a-req>=1"
]
envx = [
    "decimalai[base,toolchain]",
    "fw-b>=1.5.0",
    "fw-never>=1.0; sys_platform == 'no-such-platform'",
]

[tool.other]
x = ["not-a-req-either>=1"]
"""


def test_the_check_names_what_is_missing_or_below_its_floor():
    installed = {"fw-a": "1.9", "pytest": "9.1"}
    problems = envs.problems_with(("envx",), text=_SAMPLE, version_of=installed.get)
    assert len(problems) == 2, problems
    assert any("fw-a" in p and "1.9" in p and ">=2.0" in p for p in problems), problems
    assert any("fw-b" in p and "not installed" in p for p in problems), problems
    assert not any(n in " ".join(problems) for n in ("fw-never", "not-a-req", "not-an-extra"))

    satisfied = {"fw-a": "2.0", "fw-b": "1.5.0", "pytest": "9.1"}
    assert envs.problems_with(("envx",), text=_SAMPLE, version_of=satisfied.get) == []


def test_the_hand_parser_reads_pyproject_exactly_as_tomllib_does():
    """3.10 has no tomllib, so the extras are hand-parsed; on 3.11+ hold that to the real thing."""
    tomllib = pytest.importorskip("tomllib")
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        real = tomllib.load(fh)["project"]["optional-dependencies"]
    assert envs.optional_dependencies() == real


def test_an_extra_nobody_declared_is_an_error_not_an_empty_environment():
    with pytest.raises(ValueError, match="nope"):
        envs.requirements_of(("nope",), text=_SAMPLE)
    with pytest.raises(ValueError, match="missing"):
        envs.requirements_of(("broken",), text=_SAMPLE.replace(
            'base = ["fw-a>=2.0"]', 'broken = ["decimalai[missing]"]'))


def test_the_command_line_prints_the_extras_and_checks_an_environment(capsys):
    assert envs.main(["extras", "openai2"]) == 0
    assert capsys.readouterr().out.strip() == "conformance-tests-openai2"
    assert envs.main(["no-such-command"]) == 2


# ── a named environment refuses to start in less than its extras ──────────────


def _root_conftest():
    """The root conftest module pytest loaded (its functions, not a re-import)."""
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None) or ""
        if f and Path(f).resolve() == REPO_ROOT / "tests" / "conftest.py":
            return mod
    raise AssertionError("tests/conftest.py is not loaded")


def test_a_named_environment_missing_a_requirement_refuses_to_start(monkeypatch):
    conftest = _root_conftest()
    monkeypatch.setenv("DECIMAL_TEST_ENV", "openai3")
    monkeypatch.setattr(envs, "environment_problems",
                        lambda env: [f"anthropic is not installed ({env})"])
    with pytest.raises(pytest.UsageError) as raised:
        conftest._check_test_environment()
    message = str(raised.value)
    assert "anthropic is not installed (openai3)" in message
    assert "dev,conformance-tests" in message, "it must say how to build the environment"
    assert "uv venv" in message


def _refusal(monkeypatch, env: str, prefix: Path, base_prefix: Path) -> str:
    """The refusal a named ``env`` run prints on an interpreter at ``prefix``."""
    conftest = _root_conftest()
    monkeypatch.setenv("DECIMAL_TEST_ENV", env)
    monkeypatch.setattr(envs, "environment_problems",
                        lambda e: [f"crewai is not installed ({e})"])
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(sys, "base_prefix", str(base_prefix))
    monkeypatch.setattr(sys, "executable", str(prefix / "bin" / "python3.12"))
    with pytest.raises(pytest.UsageError) as raised:
        conftest._check_test_environment()
    return str(raised.value)


def _command_lines(message: str) -> List[str]:
    """The indented lines a person is told to run."""
    return [ln.strip() for ln in message.splitlines()
            if ln.startswith("  ") and ln.strip().startswith(("uv ", "rm ", "cd "))]


@pytest.mark.parametrize("env", sorted(envs.TEST_ENV_EXTRAS))
@pytest.mark.parametrize("shape", ["base-interpreter", "prefix-without-pyvenv-cfg"])
def test_the_fix_for_an_interpreter_that_is_not_a_venv_builds_a_new_one(
        monkeypatch, tmp_path, env, shape):
    """Following the advice must never remove the interpreter that printed it.

    A run on Homebrew's (or CI's) own python once printed ``uv venv --clear
    <that python's prefix>``: run as told, it deletes the Python installation.
    """
    install = tmp_path / "Python.framework" / "Versions" / "3.12"
    (install / "bin").mkdir(parents=True)
    (install / "lib" / "python3.12" / "os.py").parent.mkdir(parents=True)
    base = install if shape == "base-interpreter" else tmp_path / "elsewhere"
    message = _refusal(monkeypatch, env, install, base)

    commands = _command_lines(message)
    assert commands, message
    for line in commands:
        assert "--clear" not in line and not line.startswith("rm "), line
        assert str(install) not in line, f"a command touches the interpreter: {line}"
    target = REPO_ROOT / {"openai3": ".venv", "openai2": ".venv-openai2"}[env]
    extras = ",".join(envs.TEST_ENV_EXTRAS[env])
    build = next(ln for ln in commands if ln.startswith("uv venv"))
    assert build.startswith(f"uv venv -p {envs.BUILD_PYTHON} {target} && "), build
    assert f"uv pip install -p {target}/bin/python -e '{REPO_ROOT}[{extras}]'" in build
    assert f"DECIMAL_TEST_ENV={env} {target}/bin/python -m pytest" in message
    assert "not a virtualenv" in message


def test_the_fix_for_a_venv_removes_only_that_venv_as_a_step_of_its_own(
        monkeypatch, tmp_path):
    venv = tmp_path / "checkout" / ".venv-openai2"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (tmp_path / "checkout" / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    message = _refusal(monkeypatch, "openai2", venv, tmp_path / "base-python")

    commands = _command_lines(message)
    assert commands[0] == f"rm -rf {venv}", commands
    assert commands[1] == (
        f"uv venv -p {envs.BUILD_PYTHON} {venv} && uv pip install -p {venv}/bin/python "
        f"-e '{tmp_path / 'checkout'}[conformance-tests-openai2]'"
    ), commands
    assert not any("--clear" in ln for ln in commands), commands


def test_an_unnamed_run_is_not_held_to_an_environment(monkeypatch):
    """A plain `.[dev]` install (a contributor, CI's 3.10 leg) keeps its skips."""
    conftest = _root_conftest()
    monkeypatch.delenv("DECIMAL_TEST_ENV", raising=False)

    def boom(env):
        raise AssertionError("an unnamed run must not be checked")

    monkeypatch.setattr(envs, "environment_problems", boom)
    conftest._check_test_environment()


def test_this_environment_has_everything_its_name_promises():
    """The real check on the real environment, so a regression in it shows up here."""
    import os

    env = (os.environ.get("DECIMAL_TEST_ENV") or "").strip()
    if not env:
        pytest.skip("DECIMAL_TEST_ENV is unset; only a named environment is held to its extras")
    assert envs.environment_problems(env) == []


# ── no test runs in no clean environment ──────────────────────────────────────


def _importorskip_modules() -> Iterator[Tuple[str, str]]:
    """(file, module) for every literal ``pytest.importorskip("module")`` in the unit tests."""
    for path in sorted((REPO_ROOT / "tests").rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel.startswith("tests/integration/"):
            continue  # the live lanes: each runs on its own live extra, against real providers
        tree = ast.parse(path.read_text(), filename=rel)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "importorskip"
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                yield rel, node.args[0].value


def _top_level_to_distributions() -> Dict[str, List[str]]:
    """Importable top-level name -> installed distributions, read from each dist's RECORD.

    ``metadata.packages_distributions()`` does this from 3.11; on 3.10 it reads only
    ``top_level.txt``, which wheels built by modern backends do not ship.
    """
    out: Dict[str, List[str]] = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        for f in dist.files or []:
            top = f.parts[0] if f.parts else ""
            if top.endswith(".py"):
                top = top[:-3]
            if top and "." not in top and not top.endswith("-info"):
                out.setdefault(top, []).append(name)
    return out


def _distributions_for(module: str, installed: Dict[str, List[str]]) -> set:
    top = module.split(".", 1)[0]
    found = set(installed.get(top, []))
    # Not installed here (the other environment's framework): the conventional names.
    found |= {top, module.replace(".", "-")}
    return {_canon(n) for n in found}


def test_no_unit_test_skips_for_a_module_that_no_environment_installs():
    """A skip on a module only a live extra installs runs on a developer's venv and nowhere else.

    That was tests/test_skill_route_split_langchain.py's ChatAnthropic case:
    ``langchain_anthropic`` came only from ``live-llm-tests``, so in an
    environment built from either test extra it skipped. Checked here against
    what each environment's extras declare, plus everything those pull in.

    Read in the openai 3 environment, which holds every unit test's frameworks but
    the openai 2 world's (those are matched by their conventional names): mapping an
    import name to its distribution needs the distribution installed.
    """
    if envs.environment_problems("openai3"):
        pytest.skip(f"needs the openai 3 environment ({envs.TEST_ENVS['openai3']}) to map "
                    "import names to distributions")
    declared = set()
    for env in envs.TEST_ENV_EXTRAS:
        reqs = envs.environment_requirements(env)
        declared |= _names(reqs) | envs.installed_closure(reqs + ["decimalai"])
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    installed = _top_level_to_distributions()   # slow; read once
    nowhere = sorted(
        f"{rel}: importorskip({module!r})"
        for rel, module in _importorskip_modules()
        if module.split(".", 1)[0] not in stdlib | {"decimalai", "tests"}
        and not _distributions_for(module, installed) & declared
    )
    assert not nowhere, (
        "these unit tests skip unless a module is installed that no test environment "
        f"({', '.join(envs.TEST_ENVS.values())}) installs, so they run in no clean "
        "environment:\n  " + "\n  ".join(nowhere) + "\n\nAdd the distribution to the "
        "extra of the environment the test belongs to (pyproject.toml), or drop the test."
    )


# ── CI and the publish gate build what they name, and run both environments ──


def _matrix_legs(job: dict) -> List[Dict[str, str]]:
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    if "include" in matrix:
        return [{k: str(v) for k, v in leg.items()} for leg in matrix["include"]]
    legs: List[Dict[str, str]] = [{}]
    for key, values in matrix.items():
        if isinstance(values, list):
            legs = [{**leg, key: str(v)} for leg in legs for v in values]
    return legs


def _sub(text: str, leg: Dict[str, str]) -> str:
    return re.sub(r"\$\{\{\s*matrix\.([A-Za-z0-9_-]+)\s*\}\}",
                  lambda m: leg.get(m.group(1), m.group(0)), str(text))


_INSTALL = re.compile(r"pip install\b[^\n]*?\.\[([^\]]+)\]")


def _pytest_runs(workflow: str) -> Iterator[Tuple[str, str, str, frozenset, str]]:
    """(job, leg, DECIMAL_TEST_ENV, extras installed before it, command) per pytest step."""
    yaml = pytest.importorskip("yaml")
    jobs = yaml.safe_load((WORKFLOWS / workflow).read_text())["jobs"]
    for name, job in jobs.items():
        for leg in _matrix_legs(job):
            installed: set = set()
            dynamic = False
            for step in job.get("steps") or []:
                run = _sub(step.get("run") or "", leg)
                for m in _INSTALL.finditer(run):
                    if "$" in m.group(1):
                        dynamic = True     # computed in the shell; nothing to read here
                    installed |= {e.strip() for e in m.group(1).split(",")}
                if "pytest" not in run:
                    continue
                env = {**(job.get("env") or {}), **(step.get("env") or {})}
                test_env = _sub(env.get("DECIMAL_TEST_ENV", ""), leg).strip()
                if dynamic or "${{" in test_env:
                    continue
                yield name, str(leg), test_env, frozenset(installed), run


@pytest.mark.parametrize("workflow", sorted(p.name for p in WORKFLOWS.glob("*.yml")))
def test_no_ci_step_names_an_environment_it_did_not_build(workflow):
    """A step that sets DECIMAL_TEST_ENV must have installed that environment's extras.

    The named-environment check (above) refuses to start otherwise — so this
    failing is what a broken CI job looks like before it is pushed.
    """
    wrong = [
        f"{job} {leg}: DECIMAL_TEST_ENV={test_env} after installing only "
        f"{sorted(installed) or 'nothing'} (needs {list(envs.TEST_ENV_EXTRAS[test_env])})"
        for job, leg, test_env, installed, _ in _pytest_runs(workflow)
        if test_env and not set(envs.TEST_ENV_EXTRAS[test_env]) <= installed
    ]
    assert not wrong, "\n".join(wrong)


def _unit_runs(workflow: str) -> List[Tuple[str, str]]:
    """(job, DECIMAL_TEST_ENV) for each step that runs the unit suite (`pytest tests/`)."""
    return [
        (job, test_env)
        for job, _, test_env, _, run in _pytest_runs(workflow)
        if re.search(r"pytest\s+tests/?(\s|$)", run) and "-m conformance" not in run
    ]


def test_ci_runs_the_unit_suite_in_both_environments():
    named = {test_env for _, test_env in _unit_runs("ci.yml")}
    assert {"openai3", "openai2"} <= named, (
        "ci.yml must run the unit suite with DECIMAL_TEST_ENV=openai3 (after installing "
        f"{envs.TEST_ENVS['openai3']}) and =openai2 ({envs.TEST_ENVS['openai2']}); "
        f"it names only {sorted(named)}. With `.[dev]` alone the ADK, Pydantic AI and "
        "Anthropic cases skip, and nothing else runs them."
    )


def test_the_publish_gate_runs_the_openai3_unit_suite_and_requires_the_openai2_one():
    named = {test_env for _, test_env in _unit_runs("publish.yml")}
    assert "openai3" in named, (
        "publish.yml's own test job must run the unit suite in the openai 3 environment "
        f"({envs.TEST_ENVS['openai3']}, DECIMAL_TEST_ENV=openai3); it names {sorted(named)}"
    )
    text = (WORKFLOWS / "publish.yml").read_text()
    required = re.search(r"REQUIRED = \(([^)]*)\)", text)
    assert required and '"test-openai2"' in required.group(1), (
        "the publish gate must require ci.yml's `test-openai2` run on the released commit"
    )
