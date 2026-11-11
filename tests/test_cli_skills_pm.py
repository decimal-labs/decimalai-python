"""``decimalai skills`` as a package manager: the command table and its exit codes.

Each command is a thin layer over :mod:`decimalai.skillpkg`: it passes the project
(``--directory``), the registry (``--api-key``, ``--base-url``) and ``channel="cli"``,
prints every change and notice, and exits with the engine's documented code. The engine's
own semantics are pinned in ``tests/skillpkg/``; this file pins what the CLI adds:

- one test per command row, and one per exit code (0 to 8);
- ``--directory``, and the ``--json`` schema of ``status``, ``verify`` and ``update``;
- the deprecated verbs: positional ``install <ref>`` is ``add`` and makes NO fork call,
  ``export`` warns, ``--scope global`` is a usage error naming ``skills pull``;
- ``status`` prints the installed section and the authored one; ``status DIR`` prints
  only the authored one, as it always did;
- ``pull`` writes the registry's bytes to ``<out>/<install_name>/`` with ``eval.yaml``,
  and refuses to overwrite a folder that differs unless ``--force``;
- ``add`` ends with how to try the skill, and ``search`` needs no key;
- ``sync`` skips folders the registry stamped; ``scan`` reads the metadata category;
- an end-to-end run in subprocesses: add, install --frozen, verify, update, remove.

The registry is ``tests/fakes/fake_registry.py`` on a real socket (``serve()``).
"""
from __future__ import annotations

import ast
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import httpx
import pytest
from click.testing import CliRunner

import decimalai
from decimalai.cli import skills_pm
from decimalai.cli.main import cli
from decimalai.skillpkg import lock as L
from decimalai.skillpkg import manifest as M
from decimalai.skillpkg.project import Project
from tests.fakes.fake_registry import OWNER_KEY, OWNER_ORG, FakeRegistry

POSIX = os.name == "posix"
ROOT = ".agents/skills"
REFUND = "acme/refund-policy"
REFUND_DIR = "acme-refund-policy"
REFUND_FILES = (
    {"path": "references/fees.md", "content": "# Fees\n\nA 5% restocking fee.\n"},
    {"path": "scripts/compute_refund.py", "content": "print('refund')\n", "executable": True},
)
_ENV_KEYS = ("DECIMAL_API_KEY", "DECIMALAI_API_KEY", "DECIMAL_BASE_URL", "DECIMALAI_BASE_URL",
             "DECIMAL_APP_URL", "DECIMALAI_APP_URL")
_SDK_ROOT = Path(decimalai.__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_ambient_config(monkeypatch):
    """No key or registry from the developer's shell, and stdin is not a terminal."""
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(skills_pm, "_stdin_is_tty", lambda: False)


class Env:
    """A project directory (a git checkout), a fake registry on a socket, and the CLI."""

    def __init__(self, tmp_path: Path, registry: FakeRegistry, url: str) -> None:
        self.tmp = tmp_path
        self.dir = tmp_path / "project"
        (self.dir / ".git").mkdir(parents=True)
        self.project = Project.at(self.dir)
        self.registry = registry
        self.url = url

    def skill(self, name: str = REFUND, **kwargs: Any):
        kwargs.setdefault("files", REFUND_FILES)
        return self.registry.add_skill(name, **kwargs)

    def run(self, *args: str, key: Optional[str] = OWNER_KEY, net: bool = True,
            directory: bool = True, input: Optional[str] = None):
        argv = ["skills", *args]
        if directory:
            argv += ["--directory", str(self.dir)]
        if net:
            argv += ["--base-url", self.url]
            if key:
                argv += ["--api-key", key]
        return CliRunner().invoke(cli, argv, input=input)

    def folder(self, name: str = REFUND_DIR) -> Path:
        return self.dir / ".agents" / "skills" / name

    def skill_md(self, name: str = REFUND_DIR) -> str:
        return (self.folder(name) / "SKILL.md").read_text(encoding="utf-8")

    def lock(self) -> L.Lock:
        return L.load(self.project)

    def manifest(self) -> M.Manifest:
        return M.load(self.project)

    def state_bytes(self) -> Dict[str, Optional[bytes]]:
        out = {}
        for p in (self.project.manifest_path, self.project.lock_path):
            out[p.name] = p.read_bytes() if p.exists() else None
        return out

    def routes(self) -> list:
        return [r.route for r in self.registry.requests]


@pytest.fixture(scope="module")
def _socket():
    """One registry socket for the module: stopping a server costs half a second."""
    registry = FakeRegistry()
    with registry.serve() as url:
        yield registry, url


@pytest.fixture
def registry(_socket) -> FakeRegistry:
    """A fresh registry behind the module's socket: ``serve()`` answers from the same
    object, so re-initialising it clears every skill, request, fault and render."""
    registry, _ = _socket
    registry.__init__()
    return registry


@pytest.fixture
def env(tmp_path, registry, _socket):
    return Env(tmp_path, registry, _socket[1])


def _added(env: Env, *args: str) -> None:
    result = env.run("add", REFUND, *args)
    assert result.exit_code == 0, result.output
    assert env.folder().is_dir() or "--no-install" in args


# ── help ───────────────────────────────────────────────────────────────────


class TestHelp:
    def test_every_command_is_listed(self):
        result = CliRunner().invoke(cli, ["skills", "--help"])
        assert result.exit_code == 0
        for name in ("add", "search", "install", "update", "remove", "status", "verify",
                     "migrate", "pull", "sync", "scan"):
            assert f"  {name} " in result.output, name

    @pytest.mark.parametrize("command", ["add", "install", "update", "remove", "status",
                                         "verify", "migrate"])
    def test_the_exit_codes_are_documented(self, command):
        result = CliRunner().invoke(cli, ["skills", command, "--help"])
        assert result.exit_code == 0
        assert "Exit codes:" in result.output
        for code in range(9):
            assert f"  {code}  " in result.output, (command, code)
        assert "--directory" in result.output


# ── what add ends with ─────────────────────────────────────────────────────


def _case(prompt: str, **overrides: Any) -> Dict[str, Any]:
    return {"case_prompt": prompt, "outcome": "flip_to_pass", "authored_by": "platform",
            "mode": "single_shot", **overrides}


def _printed_python(output: str) -> str:
    """The program `add` prints, from its first import to its last line, unindented."""
    lines = output.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("from agents import"))
    end = next(i for i, line in enumerate(lines) if "Runner.run_sync" in line)
    return "\n".join(line[4:] for line in lines[start:end + 1])


class TestAddEndsWithHowToTry:
    QUESTION = "Which command refunds an order that shipped?"

    def _benchmark(self, env: Env, skill: Any, *results: Dict[str, Any]) -> None:
        env.registry.benchmarks[skill.skill_id] = {"latest_run": {"results": list(results)}}

    def test_one_line_for_a_skills_allowed_tools(self, env):
        env.skill(allowed_tools=["Bash", "Write", "mcp__x__y"])
        result = env.run("add", REFUND)
        assert result.exit_code == 0, result.output
        lines = [line for line in result.output.splitlines() if "allowed-tools" in line]
        assert lines == [f"  • {REFUND}: allowed-tools: dropped 'Bash', 'Write', 'mcp__x__y' "
                         "(not authored by your organization)"]

    def test_the_last_thing_it_prints_is_how_to_try_it(self, env):
        skill = env.skill()
        result = env.run("add", REFUND)
        assert result.exit_code == 0, result.output
        before, after = result.output.split("Now try it.")
        assert "install --frozen" in before
        assert "start a new chat in this project" in after
        # REFUND is namespaced ("acme/refund-policy"): /skills/acme/refund-policy is a 404 on
        # the web, so the link takes the skill's id, which the page resolves.
        assert after.rstrip().endswith(f"/skills/{skill.skill_id}")
        assert f"/skills/{REFUND}" not in after

    def test_a_plain_name_links_by_its_name(self, env):
        env.skill(name="refund-policy")
        result = env.run("add", "refund-policy")
        assert result.exit_code == 0, result.output
        assert result.output.rstrip().endswith("/skills/refund-policy")

    def test_it_asks_a_case_from_the_skills_tests(self, env):
        skill = env.skill()
        self._benchmark(env, skill, _case("A longer case that the model also failed without it."),
                        _case(self.QUESTION))
        result = env.run("add", REFUND)
        assert result.exit_code == 0, result.output
        assert f"    Use the {REFUND_DIR} skill. {self.QUESTION}\n" in result.output
        assert "from the skill's tests" in result.output
        assert self.QUESTION in _printed_python(result.output)

    @pytest.mark.parametrize("case", [
        _case("Passed both ways", outcome="pass_kept"),
        _case("The author wrote this, so it is the author's words", authored_by="publisher"),
        _case("A conversation, not one question", mode="multi_turn"),
        _case("x" * 301),
        _case(None),
    ])
    def test_it_never_asks_a_case_that_does_not_show_the_skill(self, env, case):
        skill = env.skill()
        self._benchmark(env, skill, case)
        result = env.run("add", REFUND)
        assert result.exit_code == 0, result.output
        assert f"Use the {REFUND_DIR} skill. What can it help me with?" in result.output
        assert "from the skill's tests" not in result.output

    def test_reading_the_test_results_never_sends_the_key(self, env):
        """The route is public, so a key has no business on it, and `add` here has one."""
        skill = env.skill()
        self._benchmark(env, skill, _case(self.QUESTION))
        result = env.run("add", REFUND, key=OWNER_KEY)
        assert result.exit_code == 0, result.output
        request = env.registry.calls("benchmark")[-1]
        assert request.path.endswith(f"/{skill.skill_id}/benchmark")
        assert "authorization" not in request.headers

    def test_a_question_reaches_the_terminal_with_no_control_characters(self, env):
        skill = env.skill()
        self._benchmark(env, skill, _case("Refund it\x1b[2J\u202e now\nplease"))
        result = env.run("add", REFUND)
        assert result.exit_code == 0, result.output
        assert f"    Use the {REFUND_DIR} skill. Refund it [2J now please\n" in result.output
        assert "\x1b" not in result.output and "\u202e" not in result.output

    def test_the_printed_python_is_valid_whatever_the_question_says(self, env):
        skill = env.skill()
        self._benchmark(env, skill, _case("""Is it "it's" or 'its'? Use a \\ here."""))
        result = env.run("add", REFUND)
        assert result.exit_code == 0, result.output
        literals = [node.value for node in ast.walk(ast.parse(_printed_python(result.output)))
                    if isinstance(node, ast.Constant)]
        assert f"""Use the {REFUND_DIR} skill. Is it "it's" or 'its'? Use a \\ here.""" in literals

    @pytest.mark.parametrize("fault", [500, "timeout", "connect"])
    def test_a_registry_that_will_not_answer_costs_only_the_question(self, env, fault):
        skill = env.skill()
        self._benchmark(env, skill, _case(self.QUESTION))
        env.registry.stall_seconds = 0.2
        env.registry.fault("benchmark", fault)
        with patch.object(skills_pm, "_TRY_IT_HTTP_TIMEOUT", 0.1):
            result = env.run("add", REFUND)
        assert result.exit_code == 0, result.output
        assert env.folder().is_dir()
        assert "What can it help me with?" in result.output and "Now try it." in result.output

    def test_the_claude_code_line_works_with_no_git_repository(self, env):
        """A relative --plugin-dir loads nothing from a subdirectory, and `git rev-parse`
        prints nothing outside a repository, so the line carries the absolute path."""
        env.skill()
        (env.dir / ".git").rmdir()
        result = env.run("add", REFUND)
        assert result.exit_code == 0, result.output
        assert f"claude --plugin-dir {env.dir / '.agents'}\n" in result.output.replace("'", "")
        assert "git rev-parse" not in result.output

    def test_only_a_skill_on_disk_gets_the_next_step(self, env):
        env.skill()
        assert "Now try it." not in env.run("add", REFUND, "--no-install").output

    def test_adding_it_again_does_not_repeat_it(self, env):
        env.skill()
        assert "Now try it." in env.run("add", REFUND).output
        assert "Now try it." not in env.run("add", REFUND).output

    def test_update_json_keeps_one_note_per_tool(self, env):
        skill = env.skill(allowed_tools=["Bash", "Write"])
        _added(env)
        env.registry.add_version(skill)
        result = env.run("update", "--json")
        assert result.exit_code == 0, result.output
        notes = [p["message"] for p in json.loads(result.output)["problems"]
                 if "allowed-tools" in p["message"]]
        assert len(notes) == 2


# ── search ─────────────────────────────────────────────────────────────────


def _card(slug: str = "refund-policy", **overrides: Any) -> Dict[str, Any]:
    return {"id": "11111111-2222-3333-4444-555555555555", "name": slug, "url_slug": slug,
            "description": "Handles refunds.", "author_display_name": "acme",
            "skill_safety": "passed",
            "benchmark_summary": {"lift_cases": 23, "lift_pass_rate_pct": 100.0,
                                  "pass_rate_delta_pts": 69.57, "lift_withheld_reason": None},
            **overrides}


class TestSearch:
    def _search(self, env: Env, *args: str, **kwargs: Any):
        return env.run("search", *args, directory=False, key=None, **kwargs)

    def test_a_result_says_what_it_changes_and_how_to_add_it(self, env):
        env.registry.search_cards = [_card()]
        result = self._search(env, "refunds")
        assert result.exit_code == 0, result.output
        assert "  refund-policy, by acme\n" in result.output
        assert "Handles refunds." in result.output
        assert "Passed 23 of 23 test cases with the skill, and 7 of 23 without it." in result.output
        assert "Safety review: passed." in result.output
        assert "    decimalai skills add refund-policy\n" in result.output

    def test_it_sends_no_key_even_with_one_in_the_environment(self, env, monkeypatch):
        env.registry.search_cards = [_card()]
        monkeypatch.setenv("DECIMAL_API_KEY", OWNER_KEY)
        assert self._search(env, "refunds").exit_code == 0
        request = env.registry.calls("search")[-1]
        assert "authorization" not in request.headers

    def test_the_words_are_one_query_on_the_registrys_ranking(self, env):
        env.registry.search_cards = [_card()]
        assert self._search(env, "refund", "policy", "--limit", "3").exit_code == 0
        query = env.registry.calls("search")[-1].query
        assert query == {"q": "refund policy", "sort": "recommended", "limit": "3"}

    def test_a_query_is_required(self, env):
        assert self._search(env).exit_code == 2

    @pytest.mark.parametrize("summary,sentence", [
        (None, "No with-and-without test result yet."),
        ({"lift_cases": 23, "lift_pass_rate_pct": 90.0, "pass_rate_delta_pts": 10.0,
          "lift_withheld_reason": "not_comparable"}, "No with-and-without test result yet."),
        ({"lift_cases": 0, "lift_pass_rate_pct": 0.0, "pass_rate_delta_pts": 0.0},
         "No with-and-without test result yet."),
        ({"lift_cases": 20, "lift_pass_rate_pct": 25.0, "pass_rate_delta_pts": -10.0},
         "Passed 5 of 20 test cases with the skill, and 7 of 20 without it."),
        # 50% of 25 is 12.5 passes, which no run had: a rate whose count is not whole gets rates.
        ({"lift_cases": 25, "lift_pass_rate_pct": 50.0, "pass_rate_delta_pts": 10.0},
         "Passed 50.0% of its test cases with the skill, and 40.0% without it."),
        # open-gsd-gsd-workspace as the registry sent it before 2026-11-11: 12 and 5 of 23, paired
        # with the 16 comparable cases. Multiplying printed "8 of 16 ... 3 of 16".
        ({"lift_cases": 16, "lift_pass_rate_pct": 52.2, "pass_rate_delta_pts": 30.43},
         "Passed 52.2% of its test cases with the skill, and 21.8% without it."),
        # The same run paired with the 23 cases its rate is counted over.
        ({"lift_cases": 23, "lift_pass_rate_pct": 52.2, "pass_rate_delta_pts": 30.43},
         "Passed 12 of 23 test cases with the skill, and 5 of 23 without it."),
    ])
    def test_the_test_result_is_the_pages_sentence_or_says_there_is_none(self, env, summary, sentence):
        env.registry.search_cards = [_card(benchmark_summary=summary)]
        assert sentence in self._search(env, "refunds").output

    @pytest.mark.parametrize("completed,counted", [
        ("2026-07-28T17:29:20.394975+00:00", False),
        ("2026-07-31T23:59:59+00:00", False),
        ("2026-08-01T00:00:00+00:00", True),
        ("2026-10-31T09:59:50+00:00", True),
        (None, True),
    ])
    def test_a_count_from_before_verdicts_came_from_the_checks_is_not_printed(self, env, completed, counted):
        env.registry.search_cards = [_card(benchmark_summary={
            "lift_cases": 23, "lift_pass_rate_pct": 8.7, "pass_rate_delta_pts": 4.35, "completed_at": completed,
        })]
        output = self._search(env, "refunds").output
        assert ("Passed 2 of 23 test cases" in output) is counted
        assert ("Tested before August 2026, when results were counted differently" in output) is not counted

    def test_json_keeps_the_registrys_answer_for_a_run_from_before_august(self, env):
        summary = {"lift_cases": 23, "lift_pass_rate_pct": 8.7, "pass_rate_delta_pts": 4.35,
                   "completed_at": "2026-07-28T17:29:20+00:00"}
        env.registry.search_cards = [_card(benchmark_summary=summary)]
        payload = json.loads(self._search(env, "refunds", "--json").output)
        assert payload["items"][0]["benchmark_summary"]["lift_cases"] == 23

    def test_registry_text_cannot_write_to_the_terminal(self, env):
        env.registry.search_cards = [_card(
            description="Evil\x1b[2J\u202e text", author_display_name="a\x07b", skill_safety="passed\x1b[0m",
        )]
        output = self._search(env, "refunds").output
        assert "\x1b" not in output and "\x07" not in output and "\u202e" not in output
        assert "Evil [2J text" in output

    def test_a_slug_that_cannot_be_typed_gives_the_id(self, env):
        env.registry.search_cards = [_card(url_slug="$(rm -rf ~)")]
        output = self._search(env, "refunds").output
        assert "decimalai skills add 11111111-2222-3333-4444-555555555555\n" in output
        assert "rm -rf" not in output

    def test_a_blocked_skill_is_named_apart_with_no_way_to_add_it(self, env):
        env.registry.search_cards = [_card(), _card("evil-skill", skill_safety="blocked")]
        output = self._search(env, "refunds").output
        assert "decimalai skills add evil-skill" not in output
        assert "Blocked, `skills add` refuses these: evil-skill" in output

    def test_more_results_are_mentioned(self, env):
        env.registry.search_cards = [_card(f"skill-{n}") for n in range(3)]
        output = self._search(env, "refunds", "--limit", "2").output
        assert "skill-1" in output and "skill-2" not in output
        assert "--limit 4" in output

    def test_no_results_is_not_a_failure_and_says_what_next(self, env):
        env.registry.search_cards = []
        result = self._search(env, "nothing")
        assert result.exit_code == 0
        assert "No skills matched 'nothing'" in result.output

    def test_json_is_the_registrys_answer(self, env):
        env.registry.search_cards = [_card()]
        result = self._search(env, "refunds", "--json")
        assert result.exit_code == 0
        assert json.loads(result.output)["items"][0]["url_slug"] == "refund-policy"

    def test_an_unreachable_registry_is_exit_8(self, env):
        result = CliRunner().invoke(cli, ["skills", "search", "x", "--base-url", "http://127.0.0.1:9"])
        assert result.exit_code == 8
        assert "could not search the registry" in result.output

    @pytest.mark.parametrize("fault", [500, 429])
    def test_a_registry_that_refuses_is_exit_8(self, env, fault):
        env.registry.fault("search", fault)
        assert self._search(env, "refunds").exit_code == 8

    def test_what_it_lists_can_be_added(self, env):
        env.skill("plain-skill")
        env.registry.search_cards = None
        listed = self._search(env, "plain").output
        command = next(line.strip() for line in listed.splitlines() if "skills add" in line)
        result = env.run("add", command.split()[-1], key=None)
        assert result.exit_code == 0, result.output


# ── add ────────────────────────────────────────────────────────────────────


class TestAdd:
    def test_add_installs_the_folder_records_the_spec_and_says_so(self, env):
        env.skill()
        result = env.run("add", REFUND)
        assert result.exit_code == 0, result.output
        assert f"added {REFUND} v1 as {ROOT}/{REFUND_DIR}" in result.output
        assert env.manifest()[REFUND].spec == "latest"
        assert REFUND_DIR in env.lock().skills
        assert (env.dir / ".agents" / ".claude-plugin" / "plugin.json").is_file()
        assert "install --frozen" in result.output  # the next step for CI
        # Every bundle fetch is counted as the CLI's.
        assert env.registry.fetches and {f.channel for f in env.registry.fetches} == {"cli"}

    def test_a_version_pin(self, env):
        skill = env.skill()
        env.registry.add_version(skill)
        result = env.run("add", f"{REFUND}@1")
        assert result.exit_code == 0, result.output
        assert env.manifest()[REFUND].spec == "1"
        assert env.lock().skills[REFUND_DIR]["version"] == 1

    def test_no_install_only_lists_it(self, env):
        env.skill()
        result = env.run("add", REFUND, "--no-install")
        assert result.exit_code == 0, result.output
        assert REFUND in env.manifest()
        assert not env.folder().exists()
        result = env.run("install")
        assert result.exit_code == 0, result.output
        assert env.folder().is_dir()

    def test_without_a_key_it_uses_the_anonymous_rails(self, env):
        env.skill()
        result = env.run("add", REFUND, key=None)
        assert result.exit_code == 0, result.output
        paths = {r.path for r in env.registry.requests}
        assert "/api/v1/registry/install/resolve" in paths
        assert not any(r.headers.get("authorization") for r in env.registry.requests)

    def test_directory_beats_the_working_directory(self, env, tmp_path, monkeypatch):
        env.skill()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        result = env.run("add", REFUND)
        assert result.exit_code == 0, result.output
        assert env.folder().is_dir()
        assert not (elsewhere / ".agents").exists() and not (elsewhere / ".decimal").exists()

    def test_without_directory_the_project_is_found_from_the_working_directory(self, env, monkeypatch):
        env.skill()
        sub = env.dir / "src" / "pkg"
        sub.mkdir(parents=True)
        monkeypatch.chdir(sub)
        result = env.run("add", REFUND, directory=False)
        assert result.exit_code == 0, result.output
        assert env.folder().is_dir()

    # the trust gate

    def test_caution_without_a_terminal_is_refused_with_7(self, env):
        env.skill(band="caution")
        result = env.run("add", REFUND)
        assert result.exit_code == 7, result.output
        assert "--yes" in result.output
        assert not env.folder().exists() and env.state_bytes() == {
            "skills.toml": None, "skills.lock": None}

    def test_caution_with_yes_installs(self, env):
        env.skill(band="caution")
        result = env.run("add", REFUND, "--yes")
        assert result.exit_code == 0, result.output
        assert env.folder().is_dir()

    def test_caution_on_a_terminal_asks_and_accepts(self, env, monkeypatch):
        monkeypatch.setattr(skills_pm, "_stdin_is_tty", lambda: True)
        env.skill(band="caution")
        result = env.run("add", REFUND, input="y\n")
        assert result.exit_code == 0, result.output
        assert "Install it?" in result.output
        assert env.folder().is_dir()

    def test_caution_on_a_terminal_declined_is_7(self, env, monkeypatch):
        monkeypatch.setattr(skills_pm, "_stdin_is_tty", lambda: True)
        env.skill(band="caution")
        result = env.run("add", REFUND, input="n\n")
        assert result.exit_code == 7, result.output
        assert not env.folder().exists()

    def test_blocked_is_7(self, env):
        env.skill(band="blocked")
        result = env.run("add", REFUND, "--yes")
        assert result.exit_code == 7, result.output
        assert REFUND in result.output

    def test_not_found_is_7(self, env):
        result = env.run("add", "acme/no-such-skill")
        assert result.exit_code == 7, result.output
        assert "acme/no-such-skill" in result.output

    def test_a_malformed_version_is_a_usage_error_2(self, env):
        env.skill()
        result = env.run("add", f"{REFUND}@abc")
        assert result.exit_code == 2, result.output

    def test_a_missing_reference_is_click_usage_2(self, env):
        result = env.run("add")
        assert result.exit_code == 2

    def test_an_unmanaged_copy_of_the_skill_is_5_and_force_backs_it_up(self, env):
        env.skill()
        _added(env)
        # The manifest and lock are lost; the folder stays, stamped with its skill id.
        env.project.lock_path.unlink()
        env.project.manifest_path.unlink()
        (env.folder() / "mine.md").write_text("my notes\n", encoding="utf-8")
        result = env.run("add", REFUND)
        assert result.exit_code == 5, result.output
        assert "--force" in result.output
        assert (env.folder() / "mine.md").exists()
        result = env.run("add", REFUND, "--force")
        assert result.exit_code == 0, result.output
        assert ".decimal/backup/" in result.output
        assert not (env.folder() / "mine.md").exists()

    def test_a_hand_authored_folder_with_the_name_keeps_it(self, env):
        env.skill()
        hand = env.folder()
        hand.mkdir(parents=True)
        (hand / "SKILL.md").write_text("---\nname: acme-refund-policy\ndescription: mine\n---\n\nMine.\n",
                                       encoding="utf-8")
        result = env.run("add", REFUND)
        assert result.exit_code == 0, result.output
        assert "Mine." in env.skill_md()
        (installed,) = env.lock().skills
        assert installed.startswith(f"{REFUND_DIR}-") and installed in result.output

    def test_served_bytes_that_differ_from_their_hashes_are_6(self, env):
        env.skill()

        def tamper(data):
            data["files"][0]["content"] += "injected\n"

        env.registry.tamper = tamper
        result = env.run("add", REFUND)
        assert result.exit_code == 6, result.output
        assert not env.folder().exists()

    def test_an_unreachable_registry_is_8(self, env):
        result = CliRunner().invoke(cli, [
            "skills", "add", REFUND, "--directory", str(env.dir),
            "--base-url", "http://127.0.0.1:9", "--api-key", OWNER_KEY])
        assert result.exit_code == 8, result.output
        assert "127.0.0.1:9" in result.output


# ── install ────────────────────────────────────────────────────────────────


class TestInstall:
    def test_nothing_to_do_makes_no_registry_call(self, env):
        env.skill()
        _added(env)
        before = len(env.registry.requests)
        result = env.run("install")
        assert result.exit_code == 0, result.output
        assert len(env.registry.requests) == before
        assert "up to date" in result.output

    def test_a_missing_folder_comes_back_at_its_locked_version(self, env):
        skill = env.skill()
        _added(env)
        env.registry.add_version(skill)
        _rmtree(env.folder())
        result = env.run("install")
        assert result.exit_code == 0, result.output
        assert f"fetched {REFUND} v1" in result.output
        assert env.lock().skills[REFUND_DIR]["version"] == 1

    def test_no_manifest_is_3(self, env):
        result = env.run("install")
        assert result.exit_code == 3, result.output
        assert "skills add" in result.output

    def test_a_version_1_lock_is_3_and_names_migrate(self, env):
        _write_v1_lock(env)
        result = env.run("install")
        assert result.exit_code == 3, result.output
        assert "decimalai skills migrate" in result.output

    def test_offline_needing_a_fetch_is_8(self, env):
        env.skill()
        _added(env, "--no-install")
        result = env.run("install", "--offline")
        assert result.exit_code == 8, result.output


class TestInstallFrozen:
    def test_clean_costs_no_network_and_writes_neither_manifest_nor_lock(self, env):
        env.skill()
        _added(env)
        before, calls = env.state_bytes(), len(env.registry.requests)
        result = env.run("install", "--frozen")
        assert result.exit_code == 0, result.output
        assert len(env.registry.requests) == calls
        assert env.state_bytes() == before

    def test_a_missing_folder_is_fetched_by_the_lock_key(self, env):
        env.skill()
        _added(env)
        locked = env.skill_md()
        before = env.state_bytes()
        _rmtree(env.folder())
        result = env.run("install", "--frozen")
        assert result.exit_code == 0, result.output
        assert env.skill_md() == locked
        assert env.state_bytes() == before
        assert {f.purpose for f in env.registry.fetches[-1:]} == {"frozen"}

    def test_no_lock_is_3(self, env):
        result = env.run("install", "--frozen")
        assert result.exit_code == 3, result.output

    def test_a_lock_out_of_date_with_the_manifest_is_4(self, env):
        env.skill()
        env.skill("acme/other")
        _added(env)
        manifest = env.manifest()
        manifest.set("acme/other", "latest")
        M.save(env.project, manifest)
        result = env.run("install", "--frozen")
        assert result.exit_code == 4, result.output
        assert "decimalai skills install" in result.output

    def test_a_local_edit_is_5_and_force_restores_it(self, env):
        env.skill()
        _added(env)
        locked = env.skill_md()
        (env.folder() / "SKILL.md").write_text(locked + "edited\n", encoding="utf-8")
        result = env.run("install", "--frozen")
        assert result.exit_code == 5, result.output
        result = env.run("install", "--frozen", "--force")
        assert result.exit_code == 0, result.output
        assert env.skill_md() == locked
        assert ".decimal/backup/" in result.output

    def test_bytes_that_differ_from_the_lock_are_6(self, env):
        env.skill()
        _added(env)
        _rmtree(env.folder())

        def tamper(data):
            data["files"][0]["content"] += "changed upstream\n"

        env.registry.tamper = tamper
        result = env.run("install", "--frozen")
        assert result.exit_code == 6, result.output
        assert not env.folder().exists()

    def test_a_locked_version_blocked_later_is_7(self, env):
        skill = env.skill()
        _added(env)
        _rmtree(env.folder())
        skill.band = "blocked"
        result = env.run("install", "--frozen")
        assert result.exit_code == 7, result.output
        assert "decimalai skills update" in result.output

    def test_offline_with_a_missing_folder_is_8(self, env):
        env.skill()
        _added(env)
        _rmtree(env.folder())
        result = env.run("install", "--frozen", "--offline")
        assert result.exit_code == 8, result.output

    def test_an_unmanaged_folder_is_reported_never_a_failure(self, env):
        env.skill()
        _added(env)
        mine = env.folder("my-own")
        mine.mkdir()
        (mine / "SKILL.md").write_text("---\nname: my-own\ndescription: d\n---\n\nBody.\n",
                                       encoding="utf-8")
        result = env.run("install", "--frozen")
        assert result.exit_code == 0, result.output
        assert "my-own" in result.output


# ── update ─────────────────────────────────────────────────────────────────


class TestUpdate:
    def test_latest_moves_and_says_from_what_to_what(self, env):
        skill = env.skill()
        _added(env)
        env.registry.add_version(skill)
        result = env.run("update")
        assert result.exit_code == 0, result.output
        assert f"updated {REFUND} v1 -> v2" in result.output
        assert env.lock().skills[REFUND_DIR]["version"] == 2

    def test_an_exact_pin_does_not_move(self, env):
        skill = env.skill()
        _added(env, "--no-install")
        manifest = env.manifest()
        manifest.set(REFUND, "1")
        M.save(env.project, manifest)
        assert env.run("install").exit_code == 0
        env.registry.add_version(skill)
        result = env.run("update")
        assert result.exit_code == 0, result.output
        assert env.lock().skills[REFUND_DIR]["version"] == 1

    def test_name_at_version_changes_the_spec(self, env):
        skill = env.skill()
        _added(env)
        env.registry.add_version(skill)
        env.registry.add_version(skill)
        result = env.run("update", f"{REFUND_DIR}@2")
        assert result.exit_code == 0, result.output
        assert env.manifest()[REFUND].spec == "2"
        assert env.lock().skills[REFUND_DIR]["version"] == 2

    def test_dry_run_changes_nothing(self, env):
        skill = env.skill()
        _added(env)
        env.registry.add_version(skill)
        before, md = env.state_bytes(), env.skill_md()
        result = env.run("update", "--dry-run")
        assert result.exit_code == 0, result.output
        assert "would have updated" in result.output
        assert env.state_bytes() == before and env.skill_md() == md

    def test_dry_run_json_schema(self, env):
        skill = env.skill()
        _added(env)
        env.registry.add_version(skill)
        result = env.run("update", "--dry-run", "--json")
        assert result.exit_code == 0, result.output
        data = json.loads(result.stdout)
        assert set(data) == {"ok", "exit_code", "changes", "problems"}
        assert data["ok"] is True and data["exit_code"] == 0
        (change,) = data["changes"]
        assert set(change) == {"action", "install_name", "skill", "version", "old_version",
                               "pin", "description_changed", "format", "old_format", "backup",
                               "dry_run"}
        assert change["action"] == "update" and change["dry_run"] is True
        assert (change["old_version"], change["version"]) == (1, 2)

    def test_a_modified_folder_is_5_and_force_prints_the_backup(self, env):
        skill = env.skill()
        _added(env)
        env.registry.add_version(skill)
        (env.folder() / "SKILL.md").write_text(env.skill_md() + "mine\n", encoding="utf-8")
        result = env.run("update")
        assert result.exit_code == 5, result.output
        assert env.lock().skills[REFUND_DIR]["version"] == 1
        result = env.run("update", "--force")
        assert result.exit_code == 0, result.output
        assert ".decimal/backup/" in result.output
        assert env.lock().skills[REFUND_DIR]["version"] == 2

    def test_no_project_is_3(self, env):
        result = env.run("update")
        assert result.exit_code == 3, result.output


# ── remove ─────────────────────────────────────────────────────────────────


class TestRemove:
    def test_remove_by_install_name(self, env):
        env.skill()
        _added(env)
        result = env.run("remove", REFUND_DIR, net=False)
        assert result.exit_code == 0, result.output
        assert f"removed {REFUND}" in result.output
        assert not env.folder().exists()
        assert REFUND not in env.manifest() and REFUND_DIR not in env.lock().skills

    def test_remove_by_registry_name_makes_no_registry_call(self, env):
        env.skill()
        _added(env)
        before = len(env.registry.requests)
        result = env.run("remove", REFUND, net=False)
        assert result.exit_code == 0, result.output
        assert len(env.registry.requests) == before
        assert not env.folder().exists()

    def test_a_modified_folder_is_5_unless_force(self, env):
        env.skill()
        _added(env)
        (env.folder() / "notes.txt").write_text("mine\n", encoding="utf-8")
        result = env.run("remove", REFUND_DIR, net=False)
        assert result.exit_code == 5, result.output
        assert env.folder().is_dir()
        result = env.run("remove", REFUND_DIR, "--force", net=False)
        assert result.exit_code == 0, result.output
        assert not env.folder().exists()

    def test_an_unknown_name_is_7(self, env):
        env.skill()
        _added(env)
        result = env.run("remove", "nope", net=False)
        assert result.exit_code == 7, result.output
        assert "nope" in result.output


# ── status ─────────────────────────────────────────────────────────────────


def _report_client(skills):
    client = MagicMock()
    resp = MagicMock()
    resp.json.return_value = {"skills": skills}
    resp.raise_for_status = MagicMock()
    client._http.post.return_value = resp
    return client


def _authored(directory: Path, name: str = "alpha") -> None:
    folder = directory / "skills" / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: d\n---\n\n# {name}\n\nBody.\n",
                                     encoding="utf-8")


def _synced(directory: Path) -> str:
    """This checkout's install identity, as a first `decimalai skills sync` leaves it."""
    from decimalai import _install

    return _install.get_install_identity(str(directory))["install_id"]


class TestStatus:
    def test_two_sections_installed_then_authored(self, env):
        env.skill()
        _added(env)
        _authored(env.dir)
        _synced(env.dir)
        client = _report_client([{"name": "alpha", "status": "local_ahead"}])
        with patch("decimalai.cli.main._make_client", return_value=client):
            result = env.run("status")
        assert result.exit_code == 0, result.output
        out = result.output
        assert "Installed skills" in out and "Authored skills" in out
        assert out.index("Installed skills") < out.index(REFUND_DIR) < out.index("Authored skills")
        assert out.index("Authored skills") < out.index("alpha")
        assert "clean" in out and "local_ahead" in out
        assert client._http.post.call_args[0][0] == "/api/v1/skills/installs/report"

    def test_without_a_key_the_authored_section_is_not_fetched(self, env):
        env.skill()
        _added(env)
        _authored(env.dir)
        with patch("decimalai.cli.main._make_client") as make:
            result = env.run("status", key=None)
        make.assert_not_called()
        assert result.exit_code == 0, result.output
        assert "Authored skills" in result.output and "DECIMAL_API_KEY" in result.output

    def test_no_authored_folder_means_no_authored_section(self, env):
        env.skill()
        _added(env)
        with patch("decimalai.cli.main._make_client") as make:
            result = env.run("status")
        make.assert_not_called()
        assert result.exit_code == 0, result.output
        assert "Authored skills" not in result.output

    def test_an_empty_project_says_how_to_add_one(self, env):
        result = env.run("status")
        assert result.exit_code == 0, result.output
        assert "decimalai skills add" in result.output
        assert not (env.dir / ".decimal").exists()  # status never writes

    def test_check_exits_1_on_drift_and_0_without(self, env):
        env.skill()
        _added(env)
        assert env.run("status", "--check").exit_code == 0
        (env.folder() / "SKILL.md").write_text(env.skill_md() + "x\n", encoding="utf-8")
        result = env.run("status", "--check")
        assert result.exit_code == 1, result.output
        assert "modified" in result.output and "install --force" in result.output
        assert env.run("status").exit_code == 0

    def test_remote_reports_outdated(self, env):
        skill = env.skill()
        _added(env)
        env.registry.add_version(skill)
        result = env.run("status", "--remote")
        assert result.exit_code == 0, result.output
        assert "outdated (v1 -> v2)" in result.output

    def test_json_schema(self, env):
        env.skill()
        _added(env)
        _authored(env.dir)
        with patch("decimalai.cli.main._make_client") as make:
            result = env.run("status", "--json")
        make.assert_not_called()  # JSON is the installed section alone
        assert result.exit_code == 0, result.output
        data = json.loads(result.stdout)
        assert set(data) == {"ok", "exit_code", "changes", "problems"}
        (row,) = data["changes"]
        assert set(row) == {"install_name", "state", "skill", "version", "pin", "spec", "files",
                            "reason", "remote"}
        assert (row["install_name"], row["state"], row["skill"], row["version"]) == (
            REFUND_DIR, "clean", REFUND, 1)

    def test_json_problem_schema(self, env):
        env.skill()
        _added(env)
        _rmtree(env.folder())
        result = env.run("status", "--json", "--check")
        assert result.exit_code == 1
        data = json.loads(result.stdout)
        problem = next(p for p in data["problems"] if p["code"] == "missing")
        assert set(problem) == {"code", "message", "skill", "path", "fix", "drift", "level"}
        assert problem["drift"] is True

    def test_status_dir_prints_only_the_authored_section(self, env, monkeypatch):
        env.skill()
        _added(env)
        _authored(env.dir)
        _synced(env.dir)
        monkeypatch.chdir(env.dir)
        client = _report_client([{"name": "alpha", "status": "in_sync"}])
        with patch("decimalai.cli.main._make_client", return_value=client):
            result = CliRunner().invoke(cli, [
                "skills", "status", "skills", "--api-key", "k",
                "--base-url", "http://localhost:8000"])
        assert result.exit_code == 0, result.output
        assert "Installed skills" not in result.stdout and REFUND_DIR not in result.stdout
        assert "alpha" in result.stdout and "in_sync=1" in result.stdout
        # Not deprecated: it is the one way to report an authored folder other than ./skills.
        assert "deprecated" not in result.output

    def test_status_dir_with_installed_only_flags_is_a_usage_error(self, env):
        _authored(env.dir)
        result = CliRunner().invoke(cli, ["skills", "status", str(env.dir / "skills"), "--check"])
        assert result.exit_code == 2, result.output

    # The authored section inside the combined view: a failure is a warning, drift counts
    # for --check, and status writes nothing.

    def _failing_client(self):
        client = MagicMock()
        client._http.post.side_effect = httpx.ConnectError("connection refused")
        return client

    def test_an_authored_failure_is_a_warning_and_keeps_the_exit_code(self, env):
        env.skill()
        _added(env)
        _authored(env.dir)
        _synced(env.dir)
        with patch("decimalai.cli.main._make_client", return_value=self._failing_client()):
            result = env.run("status")
        assert result.exit_code == 0, result.output
        assert REFUND_DIR in result.stdout          # the installed section still printed
        assert "connection refused" in result.stderr and "⚠" in result.stderr

    def test_check_with_a_failed_authored_check_is_8(self, env):
        env.skill()
        _added(env)
        _authored(env.dir)
        _synced(env.dir)
        with patch("decimalai.cli.main._make_client", return_value=self._failing_client()):
            result = env.run("status", "--check")
        assert result.exit_code == 8, result.output

    def test_check_with_installed_drift_and_a_failed_authored_check_is_1(self, env):
        # Drift is already the answer; the check that failed could not change it.
        env.skill()
        _added(env)
        _rmtree(env.folder())
        _authored(env.dir)
        _synced(env.dir)
        with patch("decimalai.cli.main._make_client", return_value=self._failing_client()):
            result = env.run("status", "--check")
        assert result.exit_code == 1, result.output

    @pytest.mark.parametrize("state,code", [
        ("local_ahead", 1), ("remote_ahead", 1), ("conflict", 1),
        ("in_sync", 0), ("local_only", 0), ("unknown", 0),
    ])
    def test_check_counts_authored_drift(self, env, state, code):
        env.skill()
        _added(env)
        _authored(env.dir)
        _synced(env.dir)
        client = _report_client([{"name": "alpha", "status": state}])
        with patch("decimalai.cli.main._make_client", return_value=client):
            result = env.run("status", "--check")
        assert result.exit_code == code, result.output
        with patch("decimalai.cli.main._make_client", return_value=client):
            assert env.run("status").exit_code == 0   # without --check, drift is reported only

    @staticmethod
    def _tree(directory: Path) -> Dict[str, bytes]:
        return {p.relative_to(directory).as_posix(): p.read_bytes()
                for p in sorted(directory.rglob("*")) if p.is_file()}

    def test_the_combined_view_writes_nothing(self, env):
        env.skill()
        _added(env)
        _authored(env.dir)
        _synced(env.dir)
        before = self._tree(env.dir)
        client = _report_client([{"name": "alpha", "status": "in_sync"}])
        with patch("decimalai.cli.main._make_client", return_value=client):
            result = env.run("status")
        assert result.exit_code == 0, result.output
        assert client._http.post.call_args.kwargs["json"]["install_id"]
        assert self._tree(env.dir) == before

    def test_status_dir_writes_nothing(self, env, monkeypatch):
        _authored(env.dir)
        _synced(env.dir)
        before = self._tree(env.dir)
        monkeypatch.chdir(env.dir)
        client = _report_client([{"name": "alpha", "status": "in_sync"}])
        with patch("decimalai.cli.main._make_client", return_value=client):
            result = CliRunner().invoke(cli, ["skills", "status", "skills", "--api-key", "k",
                                              "--base-url", "http://localhost:8000"])
        assert result.exit_code == 0, result.output
        assert client._http.post.called
        assert self._tree(env.dir) == before

    # A checkout that has never synced has no baseline to compare with: the platform is
    # not asked (each run would otherwise report under a new install id it keeps).

    def test_without_a_baseline_the_platform_is_not_asked(self, env):
        env.skill()
        _added(env)
        _authored(env.dir)
        for args in ((), ("--check",)):
            with patch("decimalai.cli.main._make_client") as make:
                result = env.run("status", *args)
            make.assert_not_called()
            assert result.exit_code == 0, result.output
            assert "no sync baseline yet; run `decimalai skills sync` once" in result.output
        assert not (env.dir / ".decimal" / "install.json").exists()

    def test_without_a_baseline_installed_drift_still_counts(self, env):
        env.skill()
        _added(env)
        _rmtree(env.folder())
        _authored(env.dir)
        with patch("decimalai.cli.main._make_client") as make:
            result = env.run("status", "--check")
        make.assert_not_called()
        assert result.exit_code == 1, result.output

    def test_an_existing_install_identity_is_reused(self, env):
        from decimalai import _install

        _authored(env.dir)
        known = _install.get_install_identity(str(env.dir))["install_id"]
        client = _report_client([{"name": "alpha", "status": "in_sync"}])
        with patch("decimalai.cli.main._make_client", return_value=client):
            assert env.run("status").exit_code == 0
        assert client._http.post.call_args.kwargs["json"]["install_id"] == known

    def test_remote_without_a_registry_is_8_and_lists_nothing(self, env):
        env.skill()
        _added(env)
        result = CliRunner().invoke(cli, ["skills", "status", "--remote", "--directory",
                                          str(env.dir), "--base-url", "http://127.0.0.1:9"])
        assert result.exit_code == 8, result.output
        assert "none;" not in result.output and "127.0.0.1:9" in result.output

    def test_a_directory_that_is_not_one_is_2(self, env):
        result = CliRunner().invoke(cli, ["skills", "status", "--directory",
                                          str(env.dir / "nope")])
        assert result.exit_code == 2, result.output

    def test_a_legacy_lock_names_migrate(self, env):
        _write_v1_lock(env)
        result = env.run("status", "--check")
        assert result.exit_code == 1, result.output
        assert "decimalai skills migrate" in result.output


# ── verify ─────────────────────────────────────────────────────────────────


class TestVerify:
    def test_a_clean_project_is_0(self, env):
        env.skill()
        _added(env)
        result = env.run("verify", net=False)
        assert result.exit_code == 0, result.output
        assert "no errors" in result.output

    def test_an_edited_managed_folder_is_1_with_the_rule_and_its_fix(self, env):
        env.skill()
        _added(env)
        text = env.skill_md().replace("Step 1", "Step one")
        (env.folder() / "SKILL.md").write_text(text, encoding="utf-8")
        result = env.run("verify", net=False)
        assert result.exit_code == 1, result.output
        assert "error F12" in result.output
        assert "fix: " in result.output

    def test_verify_path_grades_an_authored_tree_and_strict_makes_sdk_rules_errors(self, env):
        folder = env.dir / "skills" / "deploy"
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(
            "---\nname: deploy\ndescription: Deploy it.\nallowed-tools: Bash\n---\n\nDeploy.\n",
            encoding="utf-8")
        result = env.run("verify", "skills", net=False)
        assert result.exit_code == 0, result.output
        assert "warning F6b" in result.output
        result = env.run("verify", "skills", "--strict", net=False)
        assert result.exit_code == 1, result.output
        assert "error F6b" in result.output

    def test_a_notice_is_printed_and_never_an_error(self, env):
        mine = env.dir / ".agents" / "skills" / "mine"
        mine.mkdir(parents=True)
        (mine / "SKILL.md").write_text("---\nname: mine\ndescription: d\n---\n\nBody.\n",
                                       encoding="utf-8")
        result = env.run("verify", "--strict", net=False)
        assert result.exit_code == 0, result.output
        assert "notice no-lock" in result.output

    def test_json_schema(self, env):
        env.skill()
        _added(env)
        (env.folder() / "SKILL.md").write_text(env.skill_md() + "x\n", encoding="utf-8")
        result = env.run("verify", "--json", net=False)
        assert result.exit_code == 1
        data = json.loads(result.stdout)
        assert set(data) == {"ok", "exit_code", "changes", "problems"}
        assert data["ok"] is False and data["exit_code"] == 1
        finding = data["problems"][0]
        assert set(finding) == {"rule", "scope", "severity", "message", "skill", "path", "fix",
                                "managed"}

    def test_a_version_1_lock_is_3(self, env):
        _write_v1_lock(env)
        result = env.run("verify", net=False)
        assert result.exit_code == 3, result.output
        assert "decimalai skills migrate" in result.output

    def test_nothing_to_verify_is_3(self, env):
        result = env.run("verify", net=False)
        assert result.exit_code == 3, result.output

    def test_an_unusable_path_is_2(self, env):
        result = env.run("verify", "no/such/path", net=False)
        assert result.exit_code == 2, result.output


# ── migrate ────────────────────────────────────────────────────────────────


def _write_v1_lock(env: Env, *, skill_id: Optional[str] = None) -> None:
    lock = {"version": 1, "skills": {REFUND: {
        "version": 1, "content_hash": "e3b0c44298fc", "source": "platform",
        "skill_id": skill_id, "installed_at": "2026-06-17T05:44:40+00:00",
        "agents": ["universal"], "scope": "project"}}}
    env.project.lock_path.parent.mkdir(parents=True, exist_ok=True)
    env.project.lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    old = env.folder()
    old.mkdir(parents=True)
    (old / "SKILL.md").write_text("---\nname: acme-refund-policy\ndescription: old\n---\n\nOld.\n",
                                  encoding="utf-8")


class TestMigrate:
    def test_a_version_1_lock_becomes_a_manifest_and_a_version_2_lock(self, env):
        skill = env.skill()
        _write_v1_lock(env, skill_id=skill.skill_id)
        result = env.run("migrate")
        assert result.exit_code == 0, result.output
        assert "legacy claim" in result.output  # the table
        assert f"{REFUND_DIR}@1 (latest)" in result.output
        assert env.manifest()[REFUND].spec == "latest"
        assert env.lock().skills[REFUND_DIR]["version"] == 1
        assert "Old." not in env.skill_md()
        assert env.run("migrate").exit_code == 0  # a second run is a no-op

    def test_unresolvable_entries_are_7_unless_yes(self, env):
        _write_v1_lock(env)
        before = env.state_bytes()
        result = env.run("migrate")
        assert result.exit_code == 7, result.output
        assert "--yes" in result.output
        assert env.state_bytes() == before

    def test_nothing_to_migrate(self, env):
        result = env.run("migrate")
        assert result.exit_code == 0, result.output
        assert "nothing to migrate" in result.output


# ── pull ───────────────────────────────────────────────────────────────────


def _pull(env: Env, *args: str, base: Optional[str] = None):
    return CliRunner().invoke(cli, ["skills", "pull", *args, "--base-url", base or env.url])


def _served(env: Env, skill) -> Dict[str, bytes]:
    from skillevaluation import skillfmt

    bundle = env.registry.render(skill, install_name=skillfmt.install_name_for(skill.name, skill.skill_id))
    return {f.path: f.content.encode("utf-8") for f in bundle.files}


def _eval_answer(text: str = "cases:\n  - id: refund\n", count: int = 1):
    return httpx.Response(200, json={"eval_yaml_text": text, "case_count": count})


def _files_under(folder: Path) -> Dict[str, bytes]:
    return {p.relative_to(folder).as_posix(): p.read_bytes()
            for p in sorted(folder.rglob("*")) if p.is_file()}


class TestPull:
    def test_server_bytes_land_in_out_install_name_with_eval_yaml(self, env, tmp_path):
        skill = env.skill()
        env.registry.fault("other", _eval_answer())
        out = tmp_path / "out"
        result = _pull(env, REFUND, "--out", str(out))
        assert result.exit_code == 0, result.output
        folder = out / REFUND_DIR
        files = _files_under(folder)
        assert files.pop("eval.yaml") == b"cases:\n  - id: refund\n"
        assert files == _served(env, skill)
        if POSIX:
            assert (folder / "scripts" / "compute_refund.py").stat().st_mode & stat.S_IXUSR
        assert "1 test cases" in result.output or "1 test case" in result.output
        # No lock, no manifest: a loose copy.
        assert not (out / ".decimal").exists() and not (tmp_path / ".decimal").exists()
        assert f"decimalai skills add {REFUND}" in result.output
        assert {r.route for r in env.registry.requests} <= {"registry_bundle", "other"}
        assert not any(r.headers.get("authorization") for r in env.registry.requests)

    def test_default_out_is_the_working_directory(self, env, tmp_path, monkeypatch):
        env.skill()
        monkeypatch.chdir(tmp_path)
        result = _pull(env, REFUND, "--no-evals")
        assert result.exit_code == 0, result.output
        assert (tmp_path / REFUND_DIR / "SKILL.md").is_file()

    def test_no_evals_makes_no_eval_request(self, env, tmp_path):
        env.skill()
        result = _pull(env, REFUND, "--out", str(tmp_path), "--no-evals")
        assert result.exit_code == 0, result.output
        assert not (tmp_path / REFUND_DIR / "eval.yaml").exists()
        assert not any(r.path.endswith("/eval") for r in env.registry.requests)

    def test_a_folder_that_differs_is_5_and_untouched_without_force(self, env, tmp_path):
        env.skill()
        assert _pull(env, REFUND, "--out", str(tmp_path), "--no-evals").exit_code == 0
        md = tmp_path / REFUND_DIR / "SKILL.md"
        md.write_text(md.read_text(encoding="utf-8") + "my edit\n", encoding="utf-8")
        result = _pull(env, REFUND, "--out", str(tmp_path), "--no-evals")
        assert result.exit_code == 5, result.output
        assert "--force" in result.output and "SKILL.md" in result.output
        assert md.read_text(encoding="utf-8").endswith("my edit\n")

    def test_force_overwrites_and_leaves_other_files_in_place(self, env, tmp_path):
        skill = env.skill()
        assert _pull(env, REFUND, "--out", str(tmp_path), "--no-evals").exit_code == 0
        folder = tmp_path / REFUND_DIR
        (folder / "SKILL.md").write_text("edited\n", encoding="utf-8")
        (folder / "notes.txt").write_text("mine\n", encoding="utf-8")
        result = _pull(env, REFUND, "--out", str(tmp_path), "--no-evals", "--force")
        assert result.exit_code == 0, result.output
        files = _files_under(folder)
        assert files.pop("notes.txt") == b"mine\n"
        assert files == _served(env, skill)
        assert "notes.txt" in result.output

    def test_an_unrelated_folder_with_the_name_is_5(self, env, tmp_path):
        env.skill()
        (tmp_path / REFUND_DIR).mkdir()
        (tmp_path / REFUND_DIR / "README.md").write_text("mine\n", encoding="utf-8")
        result = _pull(env, REFUND, "--out", str(tmp_path), "--no-evals")
        assert result.exit_code == 5, result.output
        assert not (tmp_path / REFUND_DIR / "SKILL.md").exists()

    def test_the_same_files_again_is_0(self, env, tmp_path):
        env.skill()
        env.registry.fault("other", _eval_answer(), 404, _eval_answer())
        assert _pull(env, REFUND, "--out", str(tmp_path)).exit_code == 0
        result = _pull(env, REFUND, "--out", str(tmp_path))
        assert result.exit_code == 0, result.output
        assert "already" in result.output

    def test_stdout_prints_the_served_skill_md_and_writes_nothing(self, env, tmp_path, monkeypatch):
        skill = env.skill()
        here = tmp_path / "here"
        here.mkdir()
        monkeypatch.chdir(here)
        result = _pull(env, REFUND, "--stdout")
        assert result.exit_code == 0, result.output
        assert result.stdout == _served(env, skill)["SKILL.md"].decode("utf-8")
        assert list(here.iterdir()) == []

    def test_not_found_is_7(self, env, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        result = _pull(env, "acme/nope", "--out", str(out))
        assert result.exit_code == 7, result.output
        assert "acme/nope" in result.output
        assert list(out.iterdir()) == []

    def test_unreachable_is_8(self, env, tmp_path):
        result = _pull(env, REFUND, "--out", str(tmp_path), base="http://127.0.0.1:9")
        assert result.exit_code == 8, result.output

    def test_unavailable_and_policy_notes_are_named(self, env, tmp_path):
        env.skill("acme/tools", allowed_tools=["Bash", "Read"], files=(
            {"path": "scripts/run.sh", "content": "echo hi\n", "executable": True},
            {"path": "assets/logo.png", "content": None},
        ))
        result = _pull(env, "acme/tools", "--out", str(tmp_path), "--no-evals")
        assert result.exit_code == 0, result.output
        assert "assets/logo.png" in result.output
        assert "dropped 'Bash'" in result.output
        assert not (tmp_path / "acme-tools" / "assets").exists()

    def test_policy_notes_fold_into_one_line_per_reason(self, env, tmp_path):
        env.skill("acme/tools", allowed_tools=["Bash", "Write"])
        result = _pull(env, "acme/tools", "--out", str(tmp_path), "--no-evals")
        assert result.exit_code == 0, result.output
        assert ("note: allowed-tools: dropped 'Bash', 'Write' "
                "(not authored by your organization)") in result.output

    # A project's managed root belongs to its lock: a loose copy there would break
    # `install --frozen`, so pull refuses it, `--force` or not.

    def test_a_folder_the_lock_lists_is_refused_with_5(self, env):
        skill = env.skill()
        _added(env)
        _rmtree(env.folder())                     # a fresh checkout: locked, not yet installed
        env.registry.add_version(skill)
        root = env.dir / ".agents" / "skills"
        for force in ((), ("--force",)):
            result = _pull(env, REFUND, "--out", str(root), "--no-evals", *force)
            assert result.exit_code == 5, result.output
            assert "decimalai skills install" in result.output
            assert "decimalai skills update" in result.output
            assert not env.folder().exists()
        result = env.run("install", "--frozen")
        assert result.exit_code == 0, result.output

    def test_a_skill_new_to_the_project_is_refused_and_named_for_add(self, env):
        env.skill()
        env.skill("acme/other")
        _added(env)
        result = _pull(env, "acme/other", "--out", str(env.dir / ".agents" / "skills"),
                       "--no-evals")
        assert result.exit_code == 5, result.output
        assert "decimalai skills add acme/other" in result.output
        assert not env.folder("acme-other").exists()

    def test_inside_a_managed_folder_is_refused(self, env):
        env.skill()
        env.skill("acme/other")
        _added(env)
        locked = env.skill_md()
        result = _pull(env, "acme/other", "--out", str(env.folder()), "--no-evals")
        assert result.exit_code == 5, result.output
        assert env.skill_md() == locked and not (env.folder() / "acme-other").exists()

    def test_a_manifest_alone_marks_the_root_as_managed(self, env):
        env.skill()
        assert env.run("add", REFUND, "--no-install").exit_code == 0
        result = _pull(env, REFUND, "--out", str(env.dir / ".agents" / "skills"), "--no-evals")
        assert result.exit_code == 5, result.output

    def test_an_agents_skills_folder_with_no_project_state_is_just_a_folder(self, env, tmp_path):
        env.skill()
        out = tmp_path / "loose" / ".agents" / "skills"
        result = _pull(env, REFUND, "--out", str(out), "--no-evals")
        assert result.exit_code == 0, result.output
        assert (out / REFUND_DIR / "SKILL.md").is_file()

    def test_a_server_path_outside_the_folder_is_refused_and_nothing_is_written(self, env, tmp_path):
        env.skill()

        def tamper(data):
            data["files"][1]["path"] = "../../evil.sh"

        env.registry.tamper = tamper
        out = tmp_path / "a" / "b"
        out.mkdir(parents=True)
        result = _pull(env, REFUND, "--out", str(out), "--no-evals")
        assert result.exit_code == 6, result.output
        assert not (tmp_path / "evil.sh").exists() and not (tmp_path / "a" / "evil.sh").exists()
        assert list(out.iterdir()) == []


# ── sync, scan, benchmark ──────────────────────────────────────────────────


def _sync_client():
    client = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"actions": [{"action": "no_change", "name": "alpha"}]}
    resp.raise_for_status = MagicMock()
    client._http.post.return_value = resp
    return client


_STAMPED = """---
name: acme-refund-policy
description: "Use when the task involves refunds."
metadata:
  decimalai-skill-id: "00000000-0000-4000-8000-000000000001"
  decimalai-name: "acme/refund-policy"
---

# Refunds

Body.
"""


class TestSyncSkipsStampedFolders:
    def test_a_stamped_folder_is_not_sent_and_the_cli_says_so(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _authored(tmp_path, "alpha")
        stamped = tmp_path / "skills" / REFUND_DIR
        stamped.mkdir()
        (stamped / "SKILL.md").write_text(_STAMPED, encoding="utf-8")
        client = _sync_client()
        with patch("decimalai.cli.main._make_client", return_value=client):
            result = CliRunner().invoke(cli, ["skills", "sync", "--api-key", "k",
                                              "--base-url", "http://localhost:8000"])
        assert result.exit_code == 0, result.output
        sent = [s["name"] for s in client._http.post.call_args.kwargs["json"]["skills"]]
        assert sent == ["alpha"]
        assert REFUND_DIR in result.output and "registry" in result.output

    def test_only_stamped_folders_post_nothing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        stamped = tmp_path / "skills" / REFUND_DIR
        stamped.mkdir(parents=True)
        (stamped / "SKILL.md").write_text(_STAMPED, encoding="utf-8")
        with patch("decimalai.cli.main._make_client") as make:
            result = CliRunner().invoke(cli, ["skills", "sync", "--api-key", "k",
                                              "--base-url", "http://localhost:8000"])
        make.assert_not_called()
        assert result.exit_code == 0, result.output


class TestScanReadsTheMetadataCategory:
    def test_the_category_under_metadata_reaches_the_scanner(self, tmp_path):
        folder = tmp_path / "s"
        folder.mkdir()
        (folder / "SKILL.md").write_text(
            "---\nname: s\ndescription: d\nmetadata:\n  decimalai-category: security\n---\n\nBody.\n",
            encoding="utf-8")
        from skillevaluation import safety

        seen = {}
        real = safety.scan_skill_content

        def spy(body, **kwargs):
            seen.update(kwargs)
            return real(body, **kwargs)

        with patch.object(safety, "scan_skill_content", spy):
            result = CliRunner().invoke(cli, ["skills", "scan", str(folder)])
        assert result.exit_code == 0, result.output
        assert seen["category"] == "security"

    def test_a_list_category_is_not_passed_as_a_list(self, tmp_path):
        folder = tmp_path / "s"
        folder.mkdir()
        (folder / "SKILL.md").write_text(
            "---\nname: s\ndescription: d\ncategory: [a, b]\n---\n\nBody.\n", encoding="utf-8")
        from skillevaluation import safety

        seen = {}
        real = safety.scan_skill_content

        def spy(body, **kwargs):
            seen.update(kwargs)
            return real(body, **kwargs)

        with patch.object(safety, "scan_skill_content", spy):
            result = CliRunner().invoke(cli, ["skills", "scan", str(folder)])
        assert result.exit_code == 0, result.output
        assert seen["category"] is None


class TestBenchmarkCategory:
    def test_a_list_category_is_not_sent(self, tmp_path):
        folder = tmp_path / "s"
        folder.mkdir()
        (folder / "SKILL.md").write_text(
            "---\nname: s\ndescription: d\ncategory: [a, b]\n---\n\nBody.\n", encoding="utf-8")
        client = MagicMock()
        sync = MagicMock(status_code=200)
        sync.raise_for_status = MagicMock()
        run = MagicMock(status_code=200)
        run.raise_for_status = MagicMock()
        run.json.return_value = {"passed_cases": 0, "total_cases": 0, "overall_verdict": "pass",
                                 "aggregate_metrics": {}, "results": []}
        client._http.post.side_effect = [sync, run]
        with patch("decimalai.cli.main._make_client", return_value=client):
            result = CliRunner().invoke(cli, ["skills", "benchmark", str(folder), "--api-key", "k",
                                              "--base-url", "http://localhost:8000"])
        assert result.exit_code == 0, result.output
        payload = client._http.post.call_args_list[0].kwargs["json"]
        assert payload["skills"][0]["category"] is None


# ── end to end, in subprocesses ────────────────────────────────────────────


def _cli(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in _ENV_KEYS}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_SDK_ROOT), env.get("PYTHONPATH")]))
    return subprocess.run([sys.executable, "-m", "decimalai.cli", "skills", *args],
                          cwd=str(cwd), env=env, capture_output=True, text=True, timeout=120)


def test_end_to_end_add_frozen_verify_update_remove(tmp_path):
    registry = FakeRegistry()
    skill = registry.add_skill(REFUND, files=REFUND_FILES, org=OWNER_ORG, public=False)
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    with registry.serve() as url:
        net = ("--base-url", url, "--api-key", OWNER_KEY)

        done = _cli(project, "add", REFUND, *net)
        assert done.returncode == 0, done.stdout + done.stderr
        folder = project / ".agents" / "skills" / REFUND_DIR
        locked = (folder / "SKILL.md").read_bytes()

        # A fresh checkout of the committed manifest and lock, without the folder.
        _rmtree(folder)
        lock_before = (project / ".decimal" / "skills.lock").read_bytes()
        done = _cli(project, "install", "--frozen", *net)
        assert done.returncode == 0, done.stdout + done.stderr
        assert (folder / "SKILL.md").read_bytes() == locked
        assert (project / ".decimal" / "skills.lock").read_bytes() == lock_before

        done = _cli(project, "verify")
        assert done.returncode == 0, done.stdout + done.stderr

        registry.add_version(skill)
        done = _cli(project, "update", *net)
        assert done.returncode == 0, done.stdout + done.stderr
        assert "v1 -> v2" in done.stdout
        assert _cli(project, "verify").returncode == 0

        done = _cli(project, "remove", REFUND_DIR)
        assert done.returncode == 0, done.stdout + done.stderr
        assert not folder.exists()
        done = _cli(project, "status", "--json")
        assert done.returncode == 0, done.stdout + done.stderr
        assert json.loads(done.stdout)["changes"] == []
    assert {f.channel for f in registry.fetches} == {"cli"}


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path)
