"""The producer: the graded project, written by the real ``decimalai`` console script.

What a user, then CI, runs::

    decimalai skills add acme/refund-policy
    decimalai skills add acme/returns-desk
    decimalai skills add some-github-owner/an-extremely-long-...-the-limit@3
    rm -rf .agents                     # a checkout of the manifest and the lock alone
    decimalai skills install --frozen

each a subprocess of this interpreter's own ``decimalai`` script (``[project.scripts]``),
importing this checkout (``PYTHONPATH``), against the install routes of the hermetic
registry in ``tests/conformance/probe.py``. The registry holds the oracle's three skills
as registry rows (``install_row``): the skill and version columns, rendered by
``skillevaluation.skillfmt``, the renderer the platform serves with. The folder the eight
loaders grade is what ``install --frozen`` wrote; the oracle (``oracle.py``) stays the
byte-for-byte reference it is compared with (``differences_from_oracle``).

Every command gets a fresh environment: no ``DECIMAL*`` variable of the caller's (no real
key, no real registry), the key and the registry URL in ``DECIMAL_API_KEY`` and
``DECIMAL_BASE_URL`` (the CI route), every proxy variable on a closed port with only the
loopback exempt, so a request to anything but the probe fails, and ``HOME`` set to the
directory holding the project, so project discovery never looks above it and no user
configuration is read.

The same machinery installs the platform's own render vectors (``vectors/``) through the
CLI, each served verbatim (``produce_vectors``).

    python -m tests.native_loaders.producer <project-dir>      # write it by hand
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tests.conformance.probe import InstallRow, Probe
from tests.repo_state import repo_changes, repo_state

from . import oracle

HERE = Path(__file__).resolve().parent
SDK_ROOT = HERE.parent.parent
VECTORS = HERE / "vectors" / "platform_render_v1.json"

#: The project's API key; it belongs to the oracle's organization.
API_KEY = "dai_sk_native_loaders"
#: The organization the imported skill belongs to (never the project's).
UPSTREAM_ORG = str(uuid.UUID(int=0x0B5E0000))

#: The console script every command runs, as the distribution declares it.
SCRIPT_NAME = "decimalai"
ENTRY_POINT = "decimalai.cli.main:cli"

#: Where a registry that is not running points the CLI: any request is refused there.
CLOSED_PORT = "http://127.0.0.1:9"

#: Excluded from the commands' environment: the caller's key, registry and settings.
_DROP_PREFIXES = ("DECIMAL",)
_TIMEOUT = 120

Tree = Dict[str, Tuple[str, int, bytes]]   # path -> (kind, mode, bytes or link target)


class ProducerError(RuntimeError):
    """A command of the producer failed, or wrote something it must not."""


# ── the registry rows ────────────────────────────────────────────────────────


def install_row(row: oracle.SkillRow) -> InstallRow:
    """The registry row the oracle's skill is rendered from.

    The org's own skills are authored (the project's key belongs to their org); the
    import belongs to another org and is public, so the project gets the foreign
    variant. A public skill's url_slug is its install name, so the ``decimalai-source``
    line the platform writes is the one the oracle writes.
    """
    install_name = oracle.install_name_for(row.name, row.skill_id)
    authored = row.variant == "authored"
    return InstallRow(
        fields={
            "skill_id": row.skill_id,
            "version_id": row.version_id,
            "version": row.version,
            "registry_name": row.name,
            "body": row.stored_body,
            "description": row.description,
            "license": row.license or None,
            "compatibility": row.compatibility or None,
            "allowed_tools": list(row.allowed_tools),
            "invocation_mode": row.invocation_mode,
            "category": row.category or None,
            "source": (f"{oracle.APP_URL}/s/{install_name}@{row.version}/SKILL.md"
                       if row.public else None),
        },
        attachments=[{"path": p, "content": text, "executable": ex} for p, text, ex in row.files],
        org_id=oracle.ORG_ID if authored else UPSTREAM_ORG,
        public=row.public,
        imported=not authored,
        url_slug=install_name if row.public else None,
    )


def ref_for(row: oracle.SkillRow) -> str:
    """What the user types: the registry name, and ``@N`` for an exact pin."""
    return f"{row.name}@{row.version}" if row.pin == "exact" else row.name


# ── one command ──────────────────────────────────────────────────────────────


@dataclass
class Command:
    """One run of the console script, and what the registry saw during it."""

    argv: List[str]
    returncode: int
    stdout: str
    stderr: str
    #: (method, path, status) of every request the probe answered during the run.
    requests: List[Tuple[str, str, int]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def calls(self, route: str) -> int:
        """How many ``resolve`` or ``bundles/`` requests the run made."""
        return sum(1 for _, path, _ in self.requests if route in path)

    def explain(self) -> str:
        return (f"`{' '.join(self.argv)}` exited {self.returncode}\n--- stdout\n"
                f"{self.stdout[-3000:]}\n--- stderr\n{self.stderr[-3000:]}")


def console_script() -> Path:
    """This interpreter's ``decimalai`` script, after checking the distribution declares it."""
    declared = [e.value for e in metadata.entry_points(group="console_scripts")
                if e.name == SCRIPT_NAME]
    if declared != [ENTRY_POINT]:
        raise ProducerError(f"the installed distributions declare the console script "
                            f"{SCRIPT_NAME!r} as {declared}, not [{ENTRY_POINT!r}]: install "
                            "this checkout (`pip install -e .`)")
    script = Path(sysconfig.get_path("scripts")) / SCRIPT_NAME
    if not script.is_file():
        raise ProducerError(f"{script} does not exist: install this checkout into this "
                            f"interpreter's environment ({sys.executable})")
    return script


def script_interpreter(script: Path) -> str:
    """The interpreter the script's ``#!`` line names."""
    first = script.read_bytes().split(b"\n", 1)[0].decode()
    if not first.startswith("#!"):
        raise ProducerError(f"{script} has no #! line")
    return first[2:].strip().split()[0]


def tree(project: Path) -> Tree:
    """Every file and symlink under ``project``: kind, permission bits, bytes (or target)."""
    out: Tree = {}
    for dirpath, dirnames, filenames in os.walk(project):
        for name in sorted(dirnames + filenames):
            path = Path(dirpath) / name
            st = path.lstat()
            rel = path.relative_to(project).as_posix()
            if stat.S_ISLNK(st.st_mode):
                out[rel] = ("link", 0, os.readlink(path).encode())
            elif stat.S_ISREG(st.st_mode):
                out[rel] = ("file", stat.S_IMODE(st.st_mode), path.read_bytes())
    return dict(sorted(out.items()))


def stamps(project: Path) -> Dict[str, Tuple[int, int]]:
    """``(inode, mtime_ns)`` of every file: what changes when a file is rewritten at all."""
    return {rel: ((project / rel).lstat().st_ino, (project / rel).lstat().st_mtime_ns)
            for rel in tree(project)}


# ── the registry and the CLI pointed at it ───────────────────────────────────


class Registry:
    """The probe serving install rows, and the real CLI pointed at it.

        with Registry() as registry:                  # the oracle's three skills
            production = registry.produce(tmp / "project")
            registry.cli(production.project, "update")

        Registry().cli(project, "install")            # not running: offline, provably

    ``Command.requests`` is exact while one command runs at a time.
    """

    def __init__(self, rows: Optional[Sequence[InstallRow]] = None,
                 keys: Optional[Mapping[str, str]] = None) -> None:
        self.probe = Probe()
        self.probe.install_keys.update(keys if keys is not None else {API_KEY: oracle.ORG_ID})
        for row in (rows if rows is not None else [install_row(r) for r in oracle.ROWS]):
            self.probe.add_install_row(row)
        self.script = console_script()

    def __enter__(self) -> "Registry":
        self.probe.start()
        self._running = True
        return self

    def __exit__(self, *exc: Any) -> None:
        self._running = False
        self.probe.stop()

    _running = False

    @property
    def base_url(self) -> str:
        """The probe's URL; a closed port while it is not running, so every request fails."""
        return self.probe.base_url if self._running else CLOSED_PORT

    def env(self, project: Path, api_key: Optional[str]) -> Dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith(_DROP_PREFIXES)}
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SDK_ROOT), env.get("PYTHONPATH")]))
        env["DECIMAL_BASE_URL"] = self.base_url
        if api_key:
            env["DECIMAL_API_KEY"] = api_key
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy",
                     "all_proxy"):
            env[name] = CLOSED_PORT
        env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
        env["HOME"] = str(Path(project).resolve().parent)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env

    def cli(self, project: Path, *args: str, api_key: Optional[str] = API_KEY) -> Command:
        """``decimalai skills <args>`` in ``project``, as a subprocess. Never raises on failure.

        ``project`` is the working directory and, through ``HOME``, the top of project
        discovery; it may never be this repository or inside it.
        """
        project = Path(project).resolve()
        if project == SDK_ROOT or SDK_ROOT in project.parents:
            raise ProducerError(f"the producer runs the CLI in a project of its own, never in "
                                f"this repository ({project})")
        argv = [str(self.script), "skills", *args]
        cursor = self.probe.mark()
        try:
            done = subprocess.run(argv, cwd=str(project), env=self.env(project, api_key),
                                  capture_output=True, text=True, timeout=_TIMEOUT)
        except subprocess.TimeoutExpired as exc:
            raise ProducerError(f"`decimalai skills {' '.join(args)}` did not finish in "
                                f"{_TIMEOUT}s: {exc.stderr!r}") from None
        seen = [(r.method, r.path, r.status) for r in self.probe.since(cursor)]
        return Command([SCRIPT_NAME, "skills", *args], done.returncode, done.stdout,
                       done.stderr, seen)

    def must(self, project: Path, *args: str, api_key: Optional[str] = API_KEY) -> Command:
        command = self.cli(project, *args, api_key=api_key)
        if not command.ok:
            raise ProducerError(command.explain())
        return command

    def imports_from(self, project: Path) -> str:
        """Where the console script's interpreter imports ``decimalai`` from, in this env."""
        done = subprocess.run(
            [script_interpreter(self.script), "-c", "import decimalai; print(decimalai.__file__)"],
            cwd=str(project), env=self.env(project, None), capture_output=True, text=True,
            timeout=_TIMEOUT)
        if done.returncode != 0:
            raise ProducerError(f"the script's interpreter cannot import decimalai: {done.stderr}")
        return done.stdout.strip()

    def produce(self, dest: Path, refs: Optional[Sequence[str]] = None,
                api_key: Optional[str] = API_KEY) -> "Production":
        """``add`` each ref into a new project at ``dest``, then reinstall it ``--frozen``."""
        project = Path(dest).resolve()
        if project.exists() and any(project.iterdir()):
            raise ProducerError(f"the producer writes a new project; {project} is not empty")
        project.mkdir(parents=True, exist_ok=True)
        repo_before = repo_state()
        commands = [self.must(project, "add", ref, api_key=api_key)
                    for ref in (refs if refs is not None else [ref_for(r) for r in oracle.ROWS])]
        after_add = tree(project)
        if (project / ".agents").exists():
            shutil.rmtree(project / ".agents")
        commands.append(self.must(project, "install", "--frozen", api_key=api_key))
        production = Production(project=project, base_url=self.base_url, script=self.script,
                                commands=commands, after_add=after_add,
                                after_frozen=tree(project),
                                imports_from=self.imports_from(project))
        changed = repo_changes(repo_before, repo_state())
        if changed:
            raise ProducerError(f"producing {project} changed this repository's own files: "
                                f"{changed}")
        _PRODUCED[project] = production
        return production


@dataclass
class Production:
    """A project the producer wrote, and the record of how."""

    project: Path
    base_url: str
    script: Path
    commands: List[Command]
    #: The project after the ``add`` commands, and after ``install --frozen`` rewrote it.
    after_add: Tree
    after_frozen: Tree
    #: ``decimalai.__file__`` as the console script's interpreter imports it.
    imports_from: str


_PRODUCED: Dict[Path, Production] = {}
_TEMPLATE: List[Production] = []
_TEMPLATE_LOCK = threading.Lock()


def production_of(project: Path) -> Optional[Production]:
    """The record of the run that wrote ``project`` (or the project it is a copy of)."""
    return _PRODUCED.get(Path(project).resolve())


def produce(dest: Path) -> Production:
    """The graded project at ``dest``: the oracle's three skills, through the real CLI."""
    with Registry() as registry:
        return registry.produce(dest)


def template() -> Production:
    """One production per process, in a directory of its own, removed at exit."""
    with _TEMPLATE_LOCK:
        if not _TEMPLATE:
            scratch = Path(tempfile.mkdtemp(prefix="native-producer-")).resolve()
            atexit.register(shutil.rmtree, scratch, True)
            _TEMPLATE.append(produce(scratch / "project"))
        return _TEMPLATE[0]


def produce_copy(dest: Path) -> Path:
    """A copy of this process's production at ``dest``: the same bytes and modes.

    Each graded project (the session's, and every synthetic case that edits one) is a
    copy of one run of the CLI, so the commands run once per process, not per test.
    """
    made = template()
    dest = Path(dest).resolve()
    shutil.copytree(made.project, dest, symlinks=True)
    _PRODUCED[dest] = made
    return dest


# ── the oracle as the byte-for-byte reference ────────────────────────────────

_LOCK = ".decimal/skills.lock"
_PLUGIN = ".agents/.claude-plugin/plugin.json"
#: The lock fields that differ by machine and run, never by what was installed.
_PER_RUN = ("installed_at",)


def differences_from_oracle(project: Path, base_url: str) -> List[str]:
    """Every way ``project`` differs from the oracle's rendering of the same skills.

    Byte-for-byte and mode-for-mode for every file the oracle writes, except two that
    record where and when the install ran: the lock (equal entry by entry, apart from
    ``installed_at`` and ``source.base_url``, which must be ``base_url``) and
    ``plugin.json`` (equal, apart from its version, which follows the lock's hash). A
    file the oracle does not write may only be the engine's own bookkeeping under
    ``.decimal/``.
    """
    from skillevaluation import skillfmt

    want = oracle.build()
    got = tree(Path(project))
    problems: List[str] = []
    for rel in sorted(set(want.files) - set(got)):
        problems.append(f"{rel}: written by the oracle, not by the producer")
    for rel in sorted(set(got) - set(want.files)):
        if not rel.startswith(".decimal/"):
            problems.append(f"{rel}: written by the producer, not by the oracle")
    for rel in sorted(set(want.files) & set(got)):
        kind, mode, data = got[rel]
        wanted_mode = 0o755 if rel in want.executable else 0o644
        if kind != "file" or mode != wanted_mode:
            problems.append(f"{rel}: a {kind} with mode {oct(mode)}, not a file with "
                            f"{oct(wanted_mode)}")
        if rel in (_LOCK, _PLUGIN):
            continue
        if data != want.files[rel]:
            problems.append(f"{rel}: the bytes differ from the oracle's")
    if _LOCK in got:
        lock = json.loads(got[_LOCK][2])
        problems += _lock_differences(lock, want.lock, base_url)
        if _PLUGIN in got:
            plugin = json.loads(got[_PLUGIN][2])
            oracle_plugin = json.loads(want.files[_PLUGIN])
            recomputed = skillfmt.lock_hash(lock.get("root"), lock.get("skills") or {})
            if plugin.get("version") != f"lock-{recomputed[:12]}":
                problems.append(f"{_PLUGIN}: version {plugin.get('version')!r}, not "
                                f"lock-{recomputed[:12]}")
            plugin.pop("version", None)
            oracle_plugin.pop("version", None)
            if plugin != oracle_plugin:
                problems.append(f"{_PLUGIN}: {plugin} differs from the oracle's {oracle_plugin}")
    return problems


def _lock_differences(lock: dict, reference: dict, base_url: str) -> List[str]:
    problems = []
    for key in ("version", "root"):
        if lock.get(key) != reference[key]:
            problems.append(f"{_LOCK}: {key} {lock.get(key)!r}, not {reference[key]!r}")
    skills, ref_skills = lock.get("skills") or {}, reference["skills"]
    if sorted(skills) != sorted(ref_skills):
        problems.append(f"{_LOCK}: installs {sorted(skills)}, the oracle {sorted(ref_skills)}")
    for name in sorted(set(skills) & set(ref_skills)):
        entry, ref = dict(skills[name]), dict(ref_skills[name])
        source = dict(entry.get("source") or {})
        if source.get("base_url") != base_url:
            problems.append(f"{_LOCK} {name}: source.base_url {source.get('base_url')!r}, "
                            f"not the registry it came from ({base_url})")
        for e in (entry, ref):
            for k in _PER_RUN:
                e.pop(k, None)
            e["source"] = {k: v for k, v in (e.get("source") or {}).items() if k != "base_url"}
        if entry != ref:
            keys = sorted(k for k in set(entry) | set(ref) if entry.get(k) != ref.get(k))
            problems.append(f"{_LOCK} {name}: differs from the oracle's entry in {keys}: "
                            + "; ".join(f"{k}={entry.get(k)!r} vs {ref.get(k)!r}" for k in keys))
    return problems


# ── the platform's render vectors, through the CLI ───────────────────────────


def load_vectors() -> dict:
    return json.loads(VECTORS.read_text(encoding="utf-8"))


def _caller_key(org: Optional[str]) -> Optional[str]:
    return None if org is None else f"dai_sk_vector_{org.rsplit('-', 1)[-1]}"


def vector_groups(vectors: Optional[dict] = None) -> Dict[str, List[dict]]:
    """The vectors by caller, one project each: ``"anonymous"`` or the caller's org."""
    groups: Dict[str, List[dict]] = {}
    for case in (vectors or load_vectors())["cases"]:
        groups.setdefault(case.get("caller_org") or "anonymous", []).append(case)
    return groups


def produce_vectors(dest_root: Path, vectors: Optional[dict] = None) -> Dict[str, Production]:
    """Install every platform render vector with the real CLI, verbatim.

    One registry serves every vector (``InstallRow.from_vector``); each caller gets its
    own project, installed with its own key (none for an anonymous caller), each skill
    added by id at the version the platform rendered. Returns caller -> production.
    """
    vectors = vectors or load_vectors()
    groups = vector_groups(vectors)
    rows = [InstallRow.from_vector(case) for case in vectors["cases"]]
    keys = {_caller_key(org): org for org in groups if org != "anonymous"}
    dest_root = Path(dest_root).resolve()
    with Registry(rows, keys) as registry:
        def one(item: Tuple[int, str]) -> Tuple[str, Production]:
            i, caller = item
            refs = [f"{c['fields']['skill_id']}@{c['fields']['version']}" for c in groups[caller]]
            key = None if caller == "anonymous" else _caller_key(caller)
            return caller, registry.produce(dest_root / f"caller-{i}" / "project", refs, key)

        with ThreadPoolExecutor(len(groups)) as pool:
            return dict(pool.map(one, enumerate(sorted(groups))))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m tests.native_loaders.producer <project-dir>")
    made = produce(Path(sys.argv[1]))
    for command in made.commands:
        print(f"$ {' '.join(command.argv)}\n{command.stdout}")
