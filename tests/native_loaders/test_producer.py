"""P1-P3: the graded project is what the real CLI writes, and the CLI keeps it that way.

P1  the folder was written by this interpreter's ``decimalai`` console script, importing
    this checkout: ``skills add`` for each skill, then ``skills install --frozen`` into a
    checkout of the manifest and the lock alone, which wrote exactly what ``add`` had
    (lock and manifest untouched) and resolved nothing; and it equals the oracle's
    rendering byte for byte, mode for mode.
P2  ``decimalai skills verify`` passes on it, and fails on a ledger defect (every defect,
    rule by rule: ``test_verify_matches_loaders.py``).
P3  a second install writes nothing and asks the registry nothing; ``update`` leaves no
    stale file and keeps an exact pin; a local edit makes ``install --frozen`` and
    ``update`` exit 5 and keeps the edit; ``install --frozen --force`` restores the
    locked bytes and keeps the edit in a backup.

Every command is a subprocess against the hermetic registry (``producer.Registry``). A
registry that is not running points the CLI at a closed port, so a command that passes
there provably made no request.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import sysconfig
import uuid
from pathlib import Path

import pytest

from . import matrix, oracle, producer, source
from .producer import Registry

TARGET = oracle.SCENARIO.target
TARGET_DIR = f"{oracle.ROOT}/{TARGET}"
IMPORT = oracle.ROWS[2]
IMPORT_NAME = oracle.install_name_for(IMPORT.name, IMPORT.skill_id)


@pytest.fixture(scope="module")
def made() -> producer.Production:
    return producer.template()


def _copy(made: producer.Production, tmp_path: Path) -> Path:
    dest = tmp_path / "project"
    shutil.copytree(made.project, dest, symlinks=True)
    return dest


def _folder(tree: producer.Tree, folder: str) -> producer.Tree:
    return {k[len(folder) + 1:]: v for k, v in tree.items() if k.startswith(folder + "/")}


# ── P1: written by the real console entry point ──────────────────────────────


def test_p1_the_commands_ran_through_this_checkouts_console_script(made):
    assert made.script == Path(sysconfig.get_path("scripts")) / producer.SCRIPT_NAME
    assert made.script.is_file()
    imported = Path(made.imports_from).resolve()
    assert imported == (producer.SDK_ROOT / "decimalai" / "__init__.py").resolve(), (
        f"the console script's interpreter imports decimalai from {imported}, not from "
        f"this checkout ({producer.SDK_ROOT}): the commands would grade another SDK")
    assert [c.argv for c in made.commands] == (
        [["decimalai", "skills", "add", producer.ref_for(r)] for r in oracle.ROWS]
        + [["decimalai", "skills", "install", "--frozen"]])
    assert all(c.ok for c in made.commands), [c.explain() for c in made.commands if not c.ok]
    for row, command in zip(oracle.ROWS, made.commands):
        name = oracle.install_name_for(row.name, row.skill_id)
        assert f"added {row.name} v{row.version} as {oracle.ROOT}/{name}" in command.stdout, (
            command.explain())


def test_p1_install_frozen_wrote_exactly_what_add_wrote(made):
    """From a checkout of the manifest and the lock alone, ``--frozen`` rebuilt every
    folder byte for byte and mode for mode, and wrote neither the manifest nor the lock."""
    assert made.after_frozen == made.after_add
    assert {k.split("/", 1)[0] for k in made.after_add} == {".agents", ".decimal"}


def test_p1_install_frozen_resolved_nothing_and_fetched_every_skill(made):
    *adds, frozen = made.commands
    for add in adds:
        assert (add.calls("resolve"), add.calls("/bundles/")) == (1, 1), add.requests
    assert frozen.calls("resolve") == 0, f"--frozen resolved: {frozen.requests}"
    fetched = sorted(path.rsplit("/", 1)[1] for _, path, status in frozen.requests
                     if "/bundles/" in path and status == 200)
    assert fetched == sorted(r.version_id for r in oracle.ROWS), frozen.requests
    assert all(path.startswith("/api/v1/install/") for _, path, _ in
               [req for c in made.commands for req in c.requests]), "a request went unauthenticated"


def test_p1_the_producer_writes_the_oracles_bytes(made):
    assert producer.differences_from_oracle(made.project, made.base_url) == []


def test_p1_the_comparison_with_the_oracle_bites(made, tmp_path):
    """Run on planted differences, the comparison names each; a new timestamp is not one."""
    project = _copy(made, tmp_path)
    lock_path = project / ".decimal" / "skills.lock"
    lock = json.loads(lock_path.read_text())
    for entry in lock["skills"].values():
        entry["installed_at"] = "2030-01-01T00:00:00Z"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    assert producer.differences_from_oracle(project, made.base_url) == []

    lock["skills"][TARGET]["source"]["org"] = "someone-else"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    fees = project / TARGET_DIR / "references" / "fees.md"
    fees.write_bytes(fees.read_bytes() + b"\n")
    (project / TARGET_DIR / oracle.SCENARIO.script).chmod(0o644)
    (project / oracle.ROOT / "stray.md").write_text("x\n")
    problems = "\n".join(producer.differences_from_oracle(project, "http://elsewhere"))
    for needle in ("references/fees.md: the bytes differ", "compute_refund.py: a file with mode 0o644",
                   "stray.md: written by the producer, not by the oracle", "['source']",
                   "source.base_url"):
        assert needle in problems, problems


def test_p1_the_loaders_stand_in_for_skills_dir_agrees_with_the_sdk(made):
    """The loader modules start from ``source.skills_dir``; on the producer's project it
    is ``decimalai.skills_dir``, absolute and relative."""
    import decimalai

    project = made.project
    assert source.skills_dir(project) == decimalai.skills_dir(project) == (
        project / oracle.ROOT).resolve()
    assert (source.skills_dir(project, relative_to=project)
            == decimalai.skills_dir(project, relative_to=project) == Path(oracle.ROOT))


def test_p1_the_cli_never_runs_in_this_repository(tmp_path):
    for inside in (producer.SDK_ROOT, producer.SDK_ROOT / "tests"):
        with pytest.raises(producer.ProducerError, match="never in this repository"):
            Registry().cli(inside, "status")
    (tmp_path / ".decimal").mkdir()
    (tmp_path / ".decimal" / ".gitignore").write_text("install.json\n")
    before = producer.repo_state(tmp_path)
    (tmp_path / ".decimal" / ".gitignore").write_text("install.json\nstaging/\n")
    (tmp_path / ".decimal" / "install.json").write_text("{}\n")
    assert producer.repo_changes(before, producer.repo_state(tmp_path)) == [
        ".decimal/.gitignore", ".decimal/install.json"]


# ── P2: verify ───────────────────────────────────────────────────────────────


def test_p2_verify_passes_on_the_producers_project(made, tmp_path):
    project = _copy(made, tmp_path)
    done = Registry().cli(project, "verify", "--json")
    assert done.ok, done.explain()
    report = json.loads(done.stdout)
    assert report["exit_code"] == 0 and report["problems"] == [], report
    assert done.requests == []


def test_p2_verify_fails_on_a_ledger_defect(made, tmp_path):
    project = _copy(made, tmp_path)
    matrix.DEFECTS["namespaced name"](project)
    done = Registry().cli(project, "verify", "--json")
    assert done.returncode == 1, done.explain()
    rules = {(p["rule"], p["severity"]) for p in json.loads(done.stdout)["problems"]}
    assert ("F3", "error") in rules, rules


# ── P3: idempotent, no stale files, a local edit refused, --force restores ───


def test_p3_a_second_install_writes_nothing_and_asks_nothing(made, tmp_path):
    """Run offline (a closed port): any request would fail the command."""
    project = _copy(made, tmp_path)
    before, files = producer.stamps(project), producer.tree(project)
    offline = Registry()
    for args in (("install", "--frozen"), ("install",)):
        done = offline.cli(project, *args)
        assert done.ok, done.explain()
        assert "every skill matches the lock; up to date" in done.stdout, done.explain()
        assert producer.stamps(project) == before, f"`{' '.join(args)}` rewrote a file"
        assert producer.tree(project) == files


@pytest.fixture
def fresh(tmp_path):
    """A registry of its own (a test publishes to it) and a project installed from it."""
    with Registry() as registry:
        yield registry, registry.produce(tmp_path / "project")


def _publish_v4(registry: Registry, row: oracle.SkillRow, attachments) -> str:
    base = producer.install_row(row)
    version_id = str(uuid.UUID(int=0xF00D0400 + oracle.ROWS.index(row)))
    fields = {**base.fields, "version": 4, "version_id": version_id,
              "body": base.fields["body"].replace("store credit.", "store credit (v4).")}
    if base.fields["source"]:
        fields["source"] = base.fields["source"].replace(f"@{row.version}/", "@4/")
    registry.probe.add_install_row(dataclasses.replace(base, fields=fields,
                                                       attachments=list(attachments)))
    return version_id


FEES_2027 = "# Restocking fees, 2027\n\n| category | fee |\n|---|---|\n| electronics | 10% |\n"
V4_FILES = (
    {"path": "references/fees-2027.md", "content": FEES_2027, "executable": False},
    {"path": oracle.SCENARIO.script, "content": oracle.SCRIPT + "# v4\n", "executable": True},
)


def test_p3_update_leaves_no_stale_file_and_keeps_an_exact_pin(fresh):
    registry, made = fresh
    project = made.project
    v4 = _publish_v4(registry, oracle.ROWS[0], V4_FILES)
    _publish_v4(registry, IMPORT, V4_FILES)        # exact-pinned at 3: must not move
    untouched = {k: v for k, v in producer.stamps(project).items()
                 if not k.startswith(TARGET_DIR + "/") and k.startswith(oracle.ROOT)}

    done = registry.cli(project, "update")
    assert done.ok, done.explain()
    assert f"{oracle.ROWS[0].name}" in done.stdout and "v3 -> v4" in done.stdout, done.explain()

    after = producer.tree(project)
    folder = _folder(after, TARGET_DIR)
    assert sorted(folder) == ["SKILL.md", "references/fees-2027.md", oracle.SCENARIO.script], (
        f"the updated folder holds {sorted(folder)}: a file of v3 survived, or one is missing")
    assert folder[oracle.SCENARIO.script][1] == 0o755
    assert sorted(p.name for p in (project / oracle.ROOT).iterdir()) == sorted(
        oracle.install_name_for(r.name, r.skill_id) for r in oracle.ROWS), "staging debris"
    locked = json.loads((project / ".decimal" / "skills.lock").read_text())["skills"]
    entry = locked[TARGET]
    assert (entry["version"], entry["version_id"]) == (4, v4)
    assert (locked[IMPORT_NAME]["version"], locked[IMPORT_NAME]["pin"]) == (3, "exact")
    assert sorted(entry["files"]) == sorted(folder)
    stayed = {k: v for k, v in producer.stamps(project).items() if k in untouched}
    assert stayed == untouched, "update rewrote a folder it had no reason to touch"
    assert Registry().cli(project, "verify").ok


def test_p3_a_local_edit_is_refused_and_force_restores_the_locked_bytes(fresh):
    registry, made = fresh
    project = made.project
    md = project / TARGET_DIR / "SKILL.md"
    edited = md.read_bytes() + b"\nA local edit.\n"
    md.write_bytes(edited)
    lock_before = (project / ".decimal" / "skills.lock").read_bytes()

    done = registry.cli(project, "install", "--frozen")
    assert done.returncode == 5, done.explain()
    assert "local changes" in done.stderr and md.read_bytes() == edited, done.explain()

    _publish_v4(registry, oracle.ROWS[0], V4_FILES)
    done = registry.cli(project, "update")
    assert done.returncode == 5, done.explain()
    assert md.read_bytes() == edited, "update replaced a folder with local edits"
    assert (project / ".decimal" / "skills.lock").read_bytes() == lock_before

    done = registry.cli(project, "install", "--frozen", "--force")
    assert done.ok, done.explain()
    assert _folder(producer.tree(project), TARGET_DIR) == _folder(made.after_frozen, TARGET_DIR), (
        "--force did not restore the locked bytes and modes")
    backups = sorted((project / ".decimal" / "backup").glob(f"*/{TARGET}/SKILL.md"))
    assert [b.read_bytes() for b in backups] == [edited], backups
    assert backups[0].parent.relative_to(project).as_posix() in done.stdout, done.explain()
    assert Registry().cli(project, "verify").ok
