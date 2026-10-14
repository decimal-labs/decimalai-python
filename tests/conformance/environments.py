"""Which hermetic test environment this is, and which drivers it must run.

No single environment can install every framework the SDK supports. Upstream has
split on openai 3: openai-agents and pydantic-ai-slim[openai] need ``openai>=3``,
while crewai, litellm and llama-index-llms-openai still pin ``openai<3``. So the
suite is run in two environments, each built from extras in ``pyproject.toml``
(``TEST_ENV_EXTRAS`` below) and nothing else:

* ``openai3`` — ``pip install -e ".[dev,conformance-tests]"``. The whole unit
  suite, and every conformance driver except crewai and adk.
* ``openai2`` — ``pip install -e ".[conformance-tests-openai2]"``. The unit files
  that need crewai / litellm / llama-index-llms-openai (``tests/conftest.py``),
  and the crewai and adk drivers.
* llamaindex and generic-otel need neither openai, and are graded in both.

A plain ``pip install -e ".[dev]"`` is NEITHER: it holds a few of the openai 3
frameworks (anthropic among them, which langchain-anthropic requires), and the unit
cases that need the rest (ADK, Pydantic AI, Claude Agent SDK, the OpenInference
instrumentors) skip there.

``DECIMAL_TEST_ENV=openai3|openai2`` says which one a run is in. Unset means
``openai3``, leniently: a contributor's ``.[dev]`` venv keeps its skips. Set, it
is a claim the run is held to:

* the session refuses to start when a distribution those extras require is not
  installed, or is installed below its floor (``environment_problems``, called
  from ``tests/conftest.py``) — so a job that installed less than its extras
  cannot pass by skipping, and a stale venv says it is stale instead of failing
  on tests unrelated to the change being gated;
* with ``DECIMAL_CONFORMANCE_REQUIRE_ALL=1``, every driver that BELONGS to this
  environment must be runnable
  (``test_coverage.test_no_driver_is_silently_unavailable``), and the drivers that
  belong only to the other one are named on every run
  (``test_coverage.test_drivers_graded_in_another_environment_are_named``);
* ``openai2`` also narrows the UNIT suite to the files that need the openai 2
  packages (``tests/conftest.py``), because the rest of the unit suite imports
  openai 3 frameworks that environment cannot hold.

Every driver belongs to at least one environment; ``test_coverage`` fails on a
driver with none, so a new framework cannot land graded nowhere.

Run as a script (no third-party import beyond ``packaging``) for the tools that
build environments — ``bin/wt``, the release gate::

    python tests/conformance/environments.py extras openai2   # -> conformance-tests-openai2
    python tests/conformance/environments.py check openai3    # exit 1 + what is missing/stale
"""

from __future__ import annotations

import os
import re
import shlex
import sys
from pathlib import Path
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

#: The variable that names the environment.
ENV_VAR = "DECIMAL_TEST_ENV"

#: The environment a run is in when ``DECIMAL_TEST_ENV`` is unset.
DEFAULT_ENV = "openai3"

#: Environment name -> the extras that build it, and nothing else. Each extra is
#: the single source for what that environment needs; never retype its list
#: elsewhere (CI, ``bin/wt`` and the release gate all read it from here).
TEST_ENV_EXTRAS: Dict[str, Tuple[str, ...]] = {
    "openai3": ("dev", "conformance-tests"),
    "openai2": ("conformance-tests-openai2",),
}

#: Environment name -> the install that builds it (from ``TEST_ENV_EXTRAS``).
TEST_ENVS: Dict[str, str] = {
    env: f'pip install -e ".[{",".join(extras)}]"' for env, extras in TEST_ENV_EXTRAS.items()
}

#: The interpreter the gates build these environments on. google-adk (in both)
#: needs 3.11+, and crewai supports <3.14; CI's environment jobs run 3.12.
BUILD_PYTHON = "3.12"

#: The checkout's pyproject.toml — the single source for every extra.
PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

_OPENAI3 = frozenset({"openai3"})
_OPENAI2 = frozenset({"openai2"})
_BOTH = frozenset(TEST_ENVS)

#: Driver name -> the environment(s) where its column is GUARANTEED to be graded.
#: A driver may also run elsewhere when its imports happen to be installed there;
#: this is where a missing import is a build failure rather than a fact of life.
DRIVER_ENVS: Dict[str, FrozenSet[str]] = {
    "langchain": _OPENAI3,
    "openai-agents": _OPENAI3,      # openai-agents needs openai>=3
    "pydantic-ai": _OPENAI3,        # pydantic-ai-slim[openai] needs openai>=3
    "anthropic": _OPENAI3,
    "claude-agent-sdk": _OPENAI3,
    "crewai": _OPENAI2,             # crewai pins openai<3
    # google-adk itself installs in either environment, but the driver's stub
    # model is LiteLLM's custom provider reached through ADK's LiteLlm, and
    # litellm pins openai<3.
    "adk": _OPENAI2,
    # A stub CustomLLM on llama-index-core, which both install sets carry.
    "llamaindex": _BOTH,
    # Needs only opentelemetry-sdk, a core dependency.
    "generic-otel": _BOTH,
}


def current_env() -> str:
    """The environment this process is in. Raises on a value nobody defined."""
    value = (os.environ.get(ENV_VAR) or "").strip() or DEFAULT_ENV
    if value not in TEST_ENVS:
        raise ValueError(
            f"{ENV_VAR}={value!r} is not a test environment. Known: "
            f"{', '.join(sorted(TEST_ENVS))} (unset means {DEFAULT_ENV!r})."
        )
    return value


def graded_elsewhere(driver_name: str, env: Optional[str] = None) -> Optional[str]:
    """The environment(s) that grade ``driver_name``, if this one does not.

    ``None`` means the driver belongs HERE (or is unmapped, which
    ``test_coverage`` fails on separately) — so a missing import is a failure
    under ``DECIMAL_CONFORMANCE_REQUIRE_ALL=1``.
    """
    env = env or current_env()
    homes = DRIVER_ENVS.get(driver_name)
    if not homes or env in homes:
        return None
    return ", ".join(sorted(homes))


# ── what an environment's extras require, and whether this venv has it ───────


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def optional_dependencies(text: Optional[str] = None) -> Dict[str, List[str]]:
    """``[project.optional-dependencies]`` as {extra: [requirement strings]}.

    Hand-parsed, like ``floors.py`` and ``test_coverage.declared_extras``: Python
    3.10 has no ``tomllib`` and no TOML parser is a test dependency. The section
    is a flat run of ``name = [ "...", ... ]`` lists; comments are dropped before
    quotes are read, so a requirement quoted in prose is not mistaken for one.
    """
    if text is None:
        text = PYPROJECT.read_text()
    lines = text.splitlines()
    try:
        start = lines.index("[project.optional-dependencies]") + 1
    except ValueError:
        return {}
    out: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in lines[start:]:
        if line.startswith("["):
            break
        code = line.split("#", 1)[0]
        m = re.match(r"^([A-Za-z0-9._-]+)\s*=\s*\[(.*)$", code)
        if m:
            current = m.group(1)
            out[current] = re.findall(r'"([^"]+)"', m.group(2))
            if "]" in m.group(2).replace("[", "").rsplit('"', 1)[-1]:
                current = None
            continue
        if current is None:
            continue
        out[current] += re.findall(r'"([^"]+)"', code)
        if code.strip().startswith("]"):
            current = None
    return out


def requirements_of(extras: Sequence[str], text: Optional[str] = None) -> List[str]:
    """Every requirement line ``extras`` install, ``decimalai[...]`` self-references expanded.

    A bare ``decimalai[all]`` names no framework; what it stands for does. Lines
    keep their own floor, extras and marker. Raises on an extra pyproject does
    not declare — ``pip install ".[typo]"`` warns and installs nothing, and an
    environment that quietly requires nothing is the failure this module exists
    to prevent.
    """
    from packaging.requirements import Requirement

    declared = optional_dependencies(text)
    out: List[str] = []
    seen: Set[str] = set()

    def walk(extra: str, via: str) -> None:
        if extra in seen:
            return
        if extra not in declared:
            raise ValueError(
                f"extra {extra!r} ({via}) is not declared in [project.optional-dependencies]"
            )
        seen.add(extra)
        for raw in declared[extra]:
            req = Requirement(raw)
            if _canonical(req.name) == "decimalai":
                for sub in sorted(req.extras):
                    walk(sub, f"decimalai[{sub}] in [{extra}]")
            elif raw not in out:
                out.append(raw)

    for extra in extras:
        walk(extra, "asked for")
    return out


def environment_requirements(env: str, text: Optional[str] = None) -> List[str]:
    """Every requirement line the extras that build ``env`` install."""
    return requirements_of(TEST_ENV_EXTRAS[env], text)


def _installed_version(name: str) -> Optional[str]:
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def problems_with(extras: Sequence[str], text: Optional[str] = None,
                  version_of: Optional[Callable[[str], Optional[str]]] = None) -> List[str]:
    """What this interpreter lacks of ``extras``: one line per missing or too-old distribution."""
    from packaging.requirements import Requirement

    version_of = version_of or _installed_version
    problems: List[str] = []
    for raw in requirements_of(extras, text):
        req = Requirement(raw)
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        have = version_of(req.name)
        if have is None:
            problems.append(f"{req.name} is not installed ({raw!r})")
        elif req.specifier and not req.specifier.contains(have, prereleases=True):
            below = any(s.operator in (">=", ">", "~=", "==") and not s.contains(
                have, prereleases=True) for s in req.specifier)
            problems.append(f"{req.name} {have} is {'below its floor' if below else 'outside'} "
                            f"({raw!r})")
    return problems


def environment_problems(env: str) -> List[str]:
    """What this interpreter lacks of the environment ``env`` names. Empty means it is that one."""
    return problems_with(TEST_ENV_EXTRAS[env])


def installed_closure(requirements: Iterable[str]) -> Set[str]:
    """Canonical names of the installed distributions ``requirements`` pull in, transitively.

    Read from installed metadata, so it follows what this venv actually holds.
    """
    from importlib import metadata

    from packaging.requirements import InvalidRequirement, Requirement

    stack: List[Tuple[str, FrozenSet[str]]] = []
    for raw in requirements:
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            continue
        stack.append((req.name, frozenset(req.extras)))
    seen: Set[Tuple[str, FrozenSet[str]]] = set()
    names: Set[str] = set()
    while stack:
        name, extras = stack.pop()
        key = (_canonical(name), extras)
        if key in seen:
            continue
        seen.add(key)
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        names.add(key[0])
        for raw in dist.requires or []:
            try:
                req = Requirement(raw)
            except InvalidRequirement:
                continue
            if req.marker is not None and not any(
                req.marker.evaluate({"extra": e}) for e in (sorted(extras) or [""])
            ):
                continue
            stack.append((req.name, frozenset(req.extras)))
    return names


def strays(env: str) -> List[str]:
    """Distributions only the OTHER environments ask for, installed here anyway.

    They do not break a run, but they make it grade more than an environment built
    from ``env``'s extras would — another world's conformance driver becomes
    runnable — so its counts stop matching CI's and the gate's. Reported, not
    refused.
    """
    mine = environment_requirements(env)
    keep = {_canonical(n) for n in installed_closure(mine + ["decimalai"])}
    from packaging.requirements import Requirement

    keep |= {_canonical(Requirement(r).name) for r in mine}
    others = {
        _canonical(Requirement(r).name)
        for other in TEST_ENV_EXTRAS if other != env
        for r in environment_requirements(other)
    }
    return sorted(n for n in others - keep if _installed_version(n) is not None)


#: Environment name -> the directory a NEW venv for it gets, in the checkout. They
#: are the names ``bin/wt`` and the landing gate look for.
VENV_DIRS: Dict[str, str] = {"openai3": ".venv", "openai2": ".venv-openai2"}


def is_venv(prefix: str, base_prefix: str) -> bool:
    """Whether ``prefix`` is a virtualenv (and so is safe to delete and rebuild).

    Both signs are required: a prefix that differs from the base interpreter's,
    and the ``pyvenv.cfg`` every venv carries. A system, Homebrew, conda or CI
    runner interpreter has neither, and removing it would remove Python itself.
    """
    return (os.path.realpath(prefix) != os.path.realpath(base_prefix)
            and os.path.isfile(os.path.join(prefix, "pyvenv.cfg")))


def rebuild_commands(env: str, venv: Optional[Path], checkout: Path) -> str:
    """The commands that give ``env`` an exact venv, for a person to run.

    ``venv`` is the venv to rebuild in place: removed as a step of its own, so the
    deletion is visible, then created again. ``None`` means the interpreter is not
    a venv. Then nothing is removed: a NEW venv is created in the checkout, and
    ``uv venv`` (without ``--clear``) refuses if that path is already taken.
    """
    extras = ",".join(TEST_ENV_EXTRAS[env])
    target = Path(venv) if venv is not None else Path(checkout) / VENV_DIRS[env]
    t = shlex.quote(str(target))
    build = (
        f"uv venv -p {BUILD_PYTHON} {t} && uv pip install -p "
        f"{shlex.quote(f'{target}/bin/python')} -e {shlex.quote(f'{checkout}[{extras}]')}"
    )
    if venv is None:
        return build
    return f"rm -rf {t}\n  {build}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``extras <env>`` prints its extras, comma-joined; ``check <env>`` checks this venv."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] not in ("extras", "check") or args[1] not in TEST_ENV_EXTRAS:
        print(
            "usage: environments.py extras|check <env>   (env: "
            + "|".join(TEST_ENV_EXTRAS) + ")",
            file=sys.stderr,
        )
        return 2
    command, env = args
    if command == "extras":
        print(",".join(TEST_ENV_EXTRAS[env]))
        return 0
    problems = environment_problems(env)
    for line in problems:
        print(f"{env}: {line}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
