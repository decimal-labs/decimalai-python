"""The package-manager verbs of ``decimalai skills``.

A project lists the registry skills it wants in ``.decimal/skills.toml``; each one is
installed as a real folder, ``.agents/skills/<install-name>/``, and ``.decimal/skills.lock``
records exactly what was written, so CI and Docker get the same bytes back
(``install --frozen``)::

    decimalai skills search refund policy         # find one, no API key
    decimalai skills add acme/refund-policy       # record it, install it
    decimalai skills install --frozen             # CI: exactly the lock, or a refusal
    decimalai skills update                       # move `latest` entries forward
    decimalai skills status / verify / remove / migrate
    decimalai skills pull <ref>                   # a loose copy, outside any project

Each command here is thin. It hands the project (``--directory``), the registry
(``--api-key``, ``--base-url``) and ``channel="cli"`` to :mod:`decimalai.skillpkg`, prints
every change and notice the engine returns, and exits with the engine's documented code
(:data:`decimalai.skillpkg.errors.EXIT_CODE_MEANINGS`). The engine is imported only when a
command runs, so ``decimalai --help`` loads none of it.

``skills status DIR`` prints the authored section for DIR alone.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import shlex
import stat
import sys
import unicodedata
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import click

from ..skillpkg.errors import EXIT_CODE_MEANINGS, ExitCode, SkillpkgError

__all__ = ["register"]

_DEFAULT_BASE_URL = "https://api.decimal.ai"
_ROOT = ".agents/skills"

#: The exit-code table, as the epilog of every command that returns one.
_EXIT_CODES = "\b\nExit codes:\n" + "\n".join(
    f"  {int(code)}  {meaning}" for code, meaning in EXIT_CODE_MEANINGS.items())

# The anonymous calls `pull` makes beside the bundle (eval.yaml, the quality line) build
# no client and reuse no connection, and a cold registry instance has served first
# requests in 9-28s: budget 30s per request and retry once on timeout, since the attempt
# that timed out warms the instance the retry then hits.
_PULL_HTTP_TIMEOUT = 30.0

# `skills add` has already done its job when it asks the registry for a question to try, so
# it waits this long, once, and does without. A first-run command stays quick.
_TRY_IT_HTTP_TIMEOUT = 6.0

#: skillfmt writes one of these per tool, so a skill with three tools printed three lines.
_TOOL_NOTE = re.compile(r"allowed-tools: dropped (?P<tool>'.*') (?P<why>\(.*\))")

#: A question longer than this does not read as something to paste at an agent.
_TRY_IT_MAX_QUESTION = 300

#: A registry `url_slug` that a person can type as it is.
_TYPABLE_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

#: OS and tool clutter, never counted as a difference in a folder `pull` writes to.
_CLUTTER_FILES = (".DS_Store", "Thumbs.db", "desktop.ini")
_CLUTTER_DIRS = ("__pycache__",)


def register(group: click.Group) -> None:
    """Add the package-manager commands to the ``skills`` group."""
    for command in (skills_add, skills_search, skills_install, skills_update, skills_remove,
                    skills_status, skills_verify, skills_migrate, skills_pull):
        group.add_command(command)


# ── options ─────────────────────────────────────────────────────────────────


def _directory_option(func):
    return click.option(
        "--directory", "directory", type=click.Path(file_okay=False), default=None,
        metavar="DIR",
        help="The project directory. Default: the project of the working directory, the "
             "nearest folder with .decimal/ up to the repository root. Relative paths "
             "given to the command are taken from it.",
    )(func)


def _registry_options(func):
    """``--api-key`` and ``--base-url``, read from the environment as every command does."""
    func = click.option(
        "--base-url", envvar=["DECIMAL_BASE_URL", "DECIMALAI_BASE_URL"],
        default=_DEFAULT_BASE_URL, show_envvar=True, help="Platform URL",
    )(func)
    func = click.option(
        "--api-key", envvar=["DECIMAL_API_KEY", "DECIMALAI_API_KEY"],
        help="API key. Without one, only public skills are available (anonymously).",
    )(func)
    return func


def _yes_option(func):
    return click.option(
        "--yes", "-y", is_flag=True, default=False,
        help="Accept a skill whose safety review has not cleared it (caution, "
             "unreviewed) without asking. Without a terminal it is refused otherwise.",
    )(func)


# ── the trust gate's question ───────────────────────────────────────────────


def _stdin_is_tty() -> bool:
    """Whether the engine may ask a question."""
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError, OSError):
        return False


def _confirm(question: str) -> bool:
    """Ask on the terminal; Ctrl-C or end of input is a no."""
    try:
        return click.confirm(question, default=False, err=True)
    except click.Abort:
        return False


def _gate() -> Dict[str, Any]:
    return {"channel": "cli", "interactive": _stdin_is_tty(), "confirm": _confirm}


# ── printing a result ───────────────────────────────────────────────────────


def _fold_tool_notes(problems: Iterable[Any]) -> List[Any]:
    """``problems`` with one allowed-tools note per skill and reason, naming every tool.

    Takes notices or plain strings. Only what is printed folds: ``--json`` and the engine
    keep one note per tool.
    """
    folded: List[Any] = []
    tools: Dict[Tuple[Optional[str], str], Tuple[int, List[str]]] = {}
    for problem in problems:
        message = getattr(problem, "message", problem)
        match = _TOOL_NOTE.fullmatch(message) if isinstance(message, str) else None
        if match is None:
            folded.append(problem)
            continue
        key = (getattr(problem, "skill", None), match["why"])
        if key not in tools:
            tools[key] = (len(folded), [])
            folded.append(problem)
        tools[key][1].append(match["tool"])
    for (_, why), (index, names) in tools.items():
        message = f"allowed-tools: dropped {', '.join(names)} {why}"
        first = folded[index]
        folded[index] = message if isinstance(first, str) else dataclasses.replace(first, message=message)
    return folded


def _print_problems(problems: Iterable[Any]) -> None:
    for problem in _fold_tool_notes(problems):
        if isinstance(problem, SkillpkgError):
            click.echo(f"  ✗ {problem}", err=True)
        elif getattr(problem, "level", None) == "info":
            click.echo(f"  • {problem}")
        else:
            click.echo(f"  ⚠ {problem}", err=True)


def _print_result(result: Any, *, nothing: Optional[str] = None) -> None:
    for change in result.changes:
        if getattr(change, "action", None) == "unchanged":
            glyph = "•"
        elif getattr(change, "dry_run", False):
            glyph = "→"
        else:
            glyph = "✓"
        click.echo(f"  {glyph} {change}")
    _print_problems(result.problems)
    if nothing and result.ok and not result.changes:
        click.echo(f"  ✓ {nothing}")


def _print_json(result: Any) -> None:
    click.echo(json.dumps(result.to_dict(), indent=2))


def _exit(result: Any) -> None:
    raise SystemExit(int(result.exit_code))


def _plain(text: Any, limit: int = 200) -> str:
    """Registry text on one line, safe for a terminal, cut to ``limit`` characters.

    Authors write these fields, and a registry skill can be imported from anywhere, so
    control and formatting characters go: an escape sequence could rewrite what the
    terminal shows, and a bidi override could reorder it.
    """
    if not isinstance(text, str):
        return ""
    kept = "".join(" " if unicodedata.category(c)[0] in "CZ" else c for c in text)
    kept = " ".join(kept.split())
    return kept if len(kept) <= limit else kept[:limit - 1].rstrip() + "…"


def _question_from_tests(base_url: str, skill_id: str) -> Optional[str]:
    """A task from the skill's tests that a model failed without it and passed with it.

    The shortest such case that DecimalAI wrote: a case its author wrote is the author's
    words, and this command tells someone to paste them at their agent. None when there
    is no such case or the registry does not answer in time.
    """
    import httpx

    url = f"{base_url.rstrip('/')}/api/v1/registry/skills/{urllib.parse.quote(skill_id, safe='')}/benchmark"
    try:
        resp = httpx.get(url, timeout=_TRY_IT_HTTP_TIMEOUT)
        run = (resp.json() if resp.status_code == 200 else {}).get("latest_run")
        results = run.get("results") if isinstance(run, dict) else None
    except (httpx.HTTPError, ValueError, AttributeError):
        return None
    questions = [
        _plain(row.get("case_prompt"), _TRY_IT_MAX_QUESTION + 1)
        for row in (results if isinstance(results, list) else [])
        if isinstance(row, dict) and row.get("outcome") == "flip_to_pass"
        and row.get("authored_by") == "platform" and row.get("mode") == "single_shot"
    ]
    fitting = [q for q in questions if q and len(q) <= _TRY_IT_MAX_QUESTION]
    return min(fitting, key=len) if fitting else None


def _skill_id(project: Any, install_name: str) -> Optional[str]:
    """The registry's id for an installed skill: a name can hold a slash, which the
    per-skill routes do not read."""
    from ..skillpkg import lock as L

    entry = L.load(project).skills.get(install_name)
    return entry.get("skill_id") if entry else None


def _next_steps(project: Any, change: Any, base_url: str) -> None:
    """What to commit, then how to try the skill, with nothing but this text to go on."""
    from .main import _dashboard_url

    click.echo("")
    click.echo("  Commit .decimal/skills.toml, .decimal/skills.lock and .agents/; CI and "
               "Docker run `decimalai skills install --frozen`.")

    name = change.skill or change.install_name
    skill_id = _skill_id(project, change.install_name)
    from_tests = _question_from_tests(base_url, skill_id) if skill_id else None
    # It names the skill because an agent that may choose often answers from memory instead.
    # Claude Code with playwright-cli on 2026-11-03, Haiku: the question alone loaded the skill
    # 1 time in 3, and this wording 3 in 3, all right. With no skill, 0 of 5 were right.
    question = (f"Use the {change.install_name} skill. "
                f"{from_tests or 'What can it help me with?'}")

    click.echo("")
    click.echo("  Now try it. Ask your agent:")
    click.echo("")
    click.echo(f"    {question}")
    if from_tests:
        click.echo("")
        click.echo("  This one is from the skill's tests: a model failed it without the skill "
                   "and passed it with the skill.")
    click.echo("")
    click.echo("  In Cursor, Copilot, Cline, Warp or Codex: start a new chat in this project.")
    click.echo(f"  In Claude Code: claude --plugin-dir {shlex.quote(str(project.agents_dir))}")
    click.echo('  In Python: pip install "decimalai[openai-agents]", set OPENAI_API_KEY, and '
               "run this from the project:")
    click.echo("")
    click.echo("    from agents import Agent, Runner")
    click.echo("    from decimalai.openai_agents import agent_config")
    click.echo("")
    click.echo('    support = agent_config("my-agent", fallback="You are a helpful assistant.")')
    click.echo("    agent = Agent(name=support.name, instructions=support.instructions, "
               "tools=[*support.tools])")
    click.echo(f"    print(Runner.run_sync(agent, {question!r}).final_output)")
    click.echo("")
    click.echo("  Steps for other frameworks, and what this skill changes:")
    click.echo(f"    {_dashboard_url(base_url)}/skills/{urllib.parse.quote(name, safe='/')}")


# ── add ─────────────────────────────────────────────────────────────────────


@click.command("add", epilog=_EXIT_CODES)
@click.argument("ref")
@_yes_option
@click.option("--force", is_flag=True, default=False,
              help="Replace an unmanaged folder that already uses the skill's name; it is "
                   "copied to .decimal/backup/ first.")
@click.option("--no-install", is_flag=True, default=False,
              help="Only record the skill in .decimal/skills.toml; "
                   "`decimalai skills install` fetches it.")
@click.option("--to-agent", "agent_name", default=None, metavar="NAME",
              help="Add the skill to this agent on DecimalAI (needs an API key). In a project "
                   "connected to DecimalAI (.decimal/link.toml) with one agent, that agent is "
                   "the default.")
@_directory_option
@_registry_options
def skills_add(ref, yes, force, no_install, agent_name, directory, api_key, base_url):
    """Add a skill: one from the registry, or one you wrote in .agents/skills.

    REF is a registry name, url_slug, install name or skill id, optionally with @N to pin
    version N or @latest to follow the newest; or the folder name of a skill you wrote,
    .agents/skills/<REF>/SKILL.md.

    Before the project is connected to DecimalAI, the skill goes on the project's list,
    .decimal/skills.toml, which agent_config() agents use. A registry skill is recorded
    there, written to .agents/skills/<install-name>/ and its exact bytes recorded in
    .decimal/skills.lock; adding it again with another version changes the version. A
    skill you wrote is listed under [authored]. A folder in .agents/skills that is not
    listed is not used by the agents (Codex, Cursor and other tools read that folder too).

    Once the project is connected (its first agent_config() run with an API key wrote
    .decimal/link.toml), an agent's skills live on DecimalAI, and this adds REF to the
    agent there: a registry skill is assigned to it, and a skill you wrote is uploaded to
    your organization (not published) and added. .decimal/skills.toml does not change.
    --to-agent does the same in any project.

    A skill whose safety review has not cleared it (caution, unreviewed) is asked about
    on a terminal and refused without one, unless --yes. Blocked skills are refused.

    \b
    Examples:
        $ decimalai skills add acme/refund-policy
        $ decimalai skills add acme/refund-policy@3
        $ decimalai skills add refund-policy          # a skill you wrote
        $ decimalai skills add pdf --yes --to-agent support
    """
    _add(ref, yes=yes, force=force, no_install=no_install, agent_name=agent_name,
         directory=directory, api_key=api_key, base_url=base_url)


def _add(ref: str, *, yes: bool, force: bool, no_install: bool, agent_name: Optional[str],
         directory: Optional[str], api_key: Optional[str], base_url: str) -> None:
    from .. import _authored, _keyless, skillpkg
    from ..skillpkg.project import Project

    project = Project.at(directory) if directory else Project.find()
    link = _keyless.read_link(project)
    target = agent_name
    if target is None and link:
        agents = [a for a in link.get("agents") or [] if isinstance(a, str)]
        if len(agents) > 1:
            raise click.UsageError(
                f"this project is connected to DecimalAI for agents {', '.join(sorted(agents))} "
                f"({project.rel(_keyless.link_path(project))}): pass --to-agent <name>")
        if agents:
            target = agents[0]
    if target is not None:
        _add_to_agent(project, ref, target, yes=yes, api_key=api_key, base_url=base_url,
                      connected=bool(link) and agent_name is None)
        return
    if not force and _authored.is_own_folder(project, ref.strip()):
        name = ref.strip()
        try:
            added = _authored.add(project, name)
        except (ValueError, SkillpkgError) as exc:
            click.echo(f"  ✗ {exc}", err=True)
            raise SystemExit(int(ExitCode.USAGE))
        where = project.rel(project.manifest_path)
        if added:
            click.echo(f"  ✓ listed {name} (a skill you wrote) under [authored] in {where}: "
                       "agent_config() agents in this project use it from "
                       f"{project.rel(_authored.skill_md(project, name))}")
        else:
            click.echo(f"  • {name} is already listed in {where}")
        return
    result = skillpkg.add(ref, yes=yes, force=force, no_install=no_install,
                          directory=directory, api_key=api_key or None, base_url=base_url,
                          **_gate())
    _print_result(result)
    added = next((c for c in result.changes if getattr(c, "action", None) == "add"), None)
    if result.ok and added is not None:
        _next_steps(project, added, base_url)
    _exit(result)


def _add_to_agent(project: Any, ref: str, agent: str, *, yes: bool, api_key: Optional[str],
                  base_url: str, connected: bool) -> None:
    """``decimalai skills add`` once the agent's skills live on DecimalAI: add REF there."""
    from .. import _authored, _keyless
    from ..skill_router import SkillRouter

    why = (f"agent {agent!r} gets its skills from DecimalAI (this project is connected: "
           f"{project.rel(_keyless.link_path(project))})" if connected
           else f"--to-agent adds the skill to agent {agent!r} on DecimalAI")
    if not api_key:
        raise click.UsageError(f"{why}; set DECIMAL_API_KEY to add a skill to it")
    router = SkillRouter(api_key=api_key, base_url=base_url)
    try:
        name = ref.strip()
        if _authored.is_own_folder(project, name):
            _upload_to_agent(project, router, name, agent)
        else:
            _assign_to_agent(project, router, ref, agent, yes=yes, api_key=api_key,
                             base_url=base_url)
    finally:
        close = getattr(router, "close", None)
        if callable(close):
            close()
    if connected:
        click.echo(f"  • {project.rel(project.manifest_path)} is unchanged: {why}.")


def _upload_to_agent(project: Any, router: Any, name: str, agent: str) -> None:
    from skillevaluation import skillfmt

    from .. import _authored, _keyless
    from .._install import get_install_identity
    from ..skill_router import _sync_up_items
    from ..skills import _with_local_timestamps, parse_skill_md

    descriptor = parse_skill_md(str(_authored.skill_md(project, name)))
    items, _managed = _sync_up_items(_with_local_timestamps([descriptor] if descriptor else []))
    if not items:
        click.echo(f"  ✗ {project.rel(_authored.skill_md(project, name))} is not a skill this "
                   "SDK can read (it needs frontmatter with a name and description, and "
                   "instructions)", err=True)
        raise SystemExit(int(ExitCode.USAGE))
    ident = get_install_identity(str(project.directory))
    try:
        result = _keyless.sync_authored(router, items, install_id=ident.get("install_id"),
                                        install_label=ident.get("install_label"),
                                        agent_name=agent, add_to_agent="all")
    except Exception as exc:
        click.echo(f"  ✗ could not upload {name}: {exc}", err=True)
        raise SystemExit(int(ExitCode.NETWORK))
    if result.get("agent_not_found"):
        click.echo(f"  ✗ {name} was uploaded, but not added to agent {agent!r}: "
                   f"{result.get('agent_problem') or 'no such agent'}", err=True)
        raise SystemExit(int(ExitCode.PROBLEMS))
    (action,) = [a for a in result.get("actions") or [{}] if isinstance(a, dict)] or [{}]
    if action.get("action") == "failed" or result.get("failures"):
        click.echo(f"  ✗ could not upload {name}: "
                   f"{action.get('error') or result.get('failures')}", err=True)
        raise SystemExit(int(ExitCode.PROBLEMS))
    if _keyless.kept_off_agent(result, [items[0]["name"]]):
        click.echo(f"  ✗ {name} was uploaded, but DecimalAI did not add it to agent {agent!r}: "
                   f"a skill named {items[0]['name']} may be deleted there. Restore it on "
                   "DecimalAI, or rename your skill's folder and name.", err=True)
        raise SystemExit(int(ExitCode.PROBLEMS))
    if action.get("action") == "pulled":
        body_hash = skillfmt.body_sha256(action.get("body_markdown") or "")
        click.echo(f"  ✓ added {name} to agent {agent!r} on DecimalAI, with DecimalAI's copy: "
                   "it was changed there, so your file was not uploaded. `decimalai skills "
                   "sync` brings that copy into the file.")
    else:
        body_hash = skillfmt.body_sha256(items[0]["body_markdown"])
        click.echo(f"  ✓ uploaded {name} to your organization (not published) and added it to "
                   f"agent {agent!r} on DecimalAI")
    _keyless.record(project, agent_name=agent, assigned={name: _keyless.AUTHORED},
                    uploaded={name: body_hash})


def _assign_to_agent(project: Any, router: Any, ref: str, agent: str, *, yes: bool,
                     api_key: str, base_url: str) -> None:
    from .. import _keyless
    from ..skillpkg.client import RegistryClient, ResolveRequest, trust_verdict
    from ..skillpkg.ops import _parse_ref

    try:
        base, spec = _parse_ref(ref, None)
    except SkillpkgError as exc:
        click.echo(f"  ✗ {exc}", err=True)
        raise SystemExit(int(ExitCode.USAGE))
    client = RegistryClient(api_key=api_key, base_url=base_url, channel="cli")
    try:
        resolved = client.resolve_one(ResolveRequest(ref=base, version=spec or "latest"),
                                      purpose="add")
    except SkillpkgError as exc:
        click.echo(f"  ✗ {exc}", err=True)
        raise SystemExit(int(exc.exit_code) if hasattr(exc, "exit_code") else 1)
    finally:
        client.close()
    refusal, caution = trust_verdict(resolved)
    if refusal:
        click.echo(f"  ✗ {resolved.registry_name} is {refusal}: it cannot be added", err=True)
        raise SystemExit(int(ExitCode.UNAVAILABLE))
    if caution and not yes:
        question = (f"{resolved.registry_name} v{resolved.version}: its safety review is "
                    f"{caution}. Add it to agent {agent!r} anyway?")
        if not (_stdin_is_tty() and _confirm(question)):
            click.echo(f"  ✗ {resolved.registry_name} was not added: its safety review is "
                       f"{caution} (pass --yes to add it anyway)", err=True)
            raise SystemExit(int(ExitCode.UNAVAILABLE))
    pinned = spec not in (None, "latest")
    try:
        router.assign_skill(agent, resolved.skill_id,
                            pin_version_id=resolved.version_id if pinned else None)
    except Exception as exc:
        click.echo(f"  ✗ could not add {resolved.registry_name} to agent {agent!r}: {exc}",
                   err=True)
        raise SystemExit(int(ExitCode.NETWORK))
    how = f"pinned to v{resolved.version}" if pinned else f"v{resolved.version}, following the latest"
    click.echo(f"  ✓ added {resolved.registry_name} to agent {agent!r} on DecimalAI ({how})")
    _keyless.record(project, agent_name=agent,
                    assigned={resolved.registry_name: spec or "latest"})


# ── search ──────────────────────────────────────────────────────────────────


# Until this date a benchmark stored a case as failed when no check separated the two runs, even
# when every check passed, and those counts were never repaired: `browser-automation` stored 2 of
# 23 where its own checks say 23 of 23. The page corrects them from the run's checks, which a
# search result doesn't carry. `casesPassedByChecks` in the platform's `metrics.ts` has this date.
_COUNTS_FROM_CHECKS_SINCE = "2026-08-01"


def _test_result(item: Dict[str, Any]) -> str:
    """The sentence a skill's page leads with, from the registry's own summary."""
    bench = item.get("benchmark_summary")
    if isinstance(bench, dict) and not bench.get("lift_withheld_reason"):
        cases, rate, lift = (bench.get(k) for k in ("lift_cases", "lift_pass_rate_pct", "pass_rate_delta_pts"))
        if isinstance(cases, int) and cases > 0 and all(isinstance(v, (int, float)) for v in (rate, lift)):
            completed = bench.get("completed_at")
            if isinstance(completed, str) and completed < _COUNTS_FROM_CHECKS_SINCE:
                return "Tested before August 2026, when results were counted differently, so no count is shown."
            # Rounds as the page does (half up), so the two never disagree on a count.
            with_skill = int(rate / 100 * cases + 0.5)
            without = int((rate - lift) / 100 * cases + 0.5)
            return (f"Passed {with_skill} of {cases} test cases with the skill, "
                    f"and {without} of {cases} without it.")
    return "No with-and-without test result yet."


def _add_ref(item: Dict[str, Any]) -> Optional[str]:
    """What to type after `skills add`: the slug, or the id when the slug cannot be typed."""
    slug, skill_id = item.get("url_slug"), item.get("id")
    if isinstance(slug, str) and _TYPABLE_REF.fullmatch(slug):
        return slug
    return _plain(skill_id, 80) or None


@click.command("search")
@click.argument("query", nargs=-1, required=True)
@click.option("--limit", type=click.IntRange(1, 100), default=10, show_default=True,
              help="How many skills to show.")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Print the registry's answer as JSON.")
@click.option("--base-url", envvar=["DECIMAL_BASE_URL", "DECIMALAI_BASE_URL"],
              default=_DEFAULT_BASE_URL, show_envvar=True, help="Platform URL")
def skills_search(query, limit, as_json, base_url):
    """Search the public skills registry: no signup, no API key.

    Each result says how an agent did with the skill and without it, and how to add it.
    Exit 8 means the registry could not be reached or did not answer.

    \b
    Examples:
        $ decimalai skills search browser automation
        $ decimalai skills search pdf --limit 25
    """
    import httpx

    from .main import _dashboard_url

    words = " ".join(query)
    url = base_url.rstrip("/") + "/api/v1/registry/skills"
    try:
        resp = _pull_get(url, params={"q": words, "sort": "recommended", "limit": limit})
        resp.raise_for_status()
        data = resp.json()
        items = [i for i in data.get("items") or [] if isinstance(i, dict)]
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        click.echo(f"  ✗ could not search the registry at {base_url}: {exc}", err=True)
        raise SystemExit(int(ExitCode.NETWORK))

    if as_json:
        click.echo(json.dumps(data, indent=2))
        return

    registry_url = f"{_dashboard_url(base_url)}/skills"
    if not items:
        click.echo(f"  No skills matched {_plain(words)!r}. Try fewer or broader words, or "
                   f"browse {registry_url}")
        return

    blocked = [i for i in items if i.get("skill_safety") == "blocked"]
    for item in items:
        if item in blocked:
            continue
        ref = _add_ref(item)
        author = _plain(item.get("author_display_name"), 60)
        click.echo(f"  {ref or _plain(item.get('name'), 80)}" + (f", by {author}" if author else ""))
        description = _plain(item.get("description"), 160)
        if description:
            click.echo(f"    {description}")
        click.echo(f"    {_test_result(item)} Safety review: "
                   f"{_plain(item.get('skill_safety'), 20) or 'unknown'}.")
        if ref:
            click.echo(f"    decimalai skills add {shlex.quote(ref)}")
        click.echo("")
    if blocked:
        click.echo("  Blocked, `skills add` refuses these: "
                   + ", ".join(_plain(i.get("url_slug") or i.get("name"), 80) for i in blocked))
        click.echo("")
    if data.get("has_next"):
        click.echo(f"  There are more: add --limit {min(limit * 2, 100)}, or use more words.")
    click.echo(f"  See a skill's test results: {registry_url}/<name>")


# ── install ─────────────────────────────────────────────────────────────────


@click.command("install", epilog=_EXIT_CODES)
@click.option("--frozen", is_flag=True, default=False,
              help="CI and Docker: install exactly what .decimal/skills.lock records. "
                   "Nothing is resolved, and the manifest and the lock are never written.")
@click.option("--offline", is_flag=True, default=False,
              help="Fail (exit 8) instead of fetching a folder that is missing.")
@click.option("--force", is_flag=True, default=False,
              help="Replace folders with local edits by their locked files; each is "
                   "copied to .decimal/backup/ first.")
@_yes_option
@_directory_option
@_registry_options
def skills_install(frozen, offline, force, yes, directory, api_key, base_url):
    """Make .agents/skills match the manifest and the lock.

    New entries of .decimal/skills.toml are resolved and installed, entries it no longer
    lists are removed, and missing folders are fetched at their locked version. A version
    is never moved: that is `decimalai skills update`. When every folder matches the lock
    nothing is fetched. To add a skill, run `decimalai skills add <ref>`.

    With --frozen (CI and Docker) the lock must agree with the manifest; missing folders
    are fetched by their exact lock entry and every byte is checked against it.

    \b
    Examples:
        $ decimalai skills install
        $ decimalai skills install --frozen          # in CI, or a Dockerfile
        $ decimalai skills install --frozen --offline
    """
    from .. import skillpkg

    result = skillpkg.install(frozen=frozen, force=force, offline=offline, yes=yes,
                              directory=directory, api_key=api_key or None,
                              base_url=base_url, **_gate())
    _print_result(result, nothing="every skill matches the lock; up to date")
    _exit(result)


# ── update ──────────────────────────────────────────────────────────────────


@click.command("update", epilog=_EXIT_CODES)
@click.argument("name", required=False)
@click.option("--force", is_flag=True, default=False,
              help="Replace folders with local edits; each is copied to .decimal/backup/ "
                   "first, and the path is printed.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Say what would change; change nothing.")
@_yes_option
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Print the result as JSON (with --dry-run: the plan, for scripts).")
@_directory_option
@_registry_options
def skills_update(name, force, dry_run, yes, as_json, directory, api_key, base_url):
    """Move skills to newer versions.

    With no NAME, every `latest` entry moves to the registry's current version, and pinned
    entries stay. NAME@N or NAME@latest changes that skill's version; NAME alone on a
    pinned skill refreshes its files at the same version. NAME is the folder name or the
    registry name. Each changed folder is replaced whole. If one skill cannot be updated,
    none is.

    \b
    Examples:
        $ decimalai skills update
        $ decimalai skills update acme-refund-policy@4
        $ decimalai skills update --dry-run --json
    """
    from .. import skillpkg

    result = skillpkg.update(name, force=force, dry_run=dry_run, yes=yes, directory=directory,
                             api_key=api_key or None, base_url=base_url, **_gate())
    if as_json:
        _print_json(result)
    else:
        _print_result(result, nothing="every skill is up to date")
    _exit(result)


# ── remove ──────────────────────────────────────────────────────────────────


@click.command("remove", epilog=_EXIT_CODES)
@click.argument("name")
@click.option("--force", is_flag=True, default=False,
              help="Remove a folder with local edits too; it is copied to .decimal/backup/ "
                   "first.")
@_directory_option
def skills_remove(name, force, directory):
    """Take a skill off the project's list.

    NAME is the folder name or the registry name. A registry skill's folder goes too,
    with its manifest and lock entries; nothing changes on the registry (Use links and
    forks stay). A skill you wrote is taken off the [authored] list; its folder stays.

    In a project connected to DecimalAI its agents keep the skill: remove it from an
    agent in the dashboard.

    \b
    Example:
        $ decimalai skills remove acme-refund-policy
    """
    from .. import _authored, _keyless, skillpkg
    from ..skillpkg.project import Project

    project = Project.at(directory) if directory else Project.find()
    agents = [a for a in (_keyless.read_link(project) or {}).get("agents") or []
              if isinstance(a, str)]
    still = (f"  • agent{'s' if len(agents) > 1 else ''} {', '.join(sorted(agents))} "
             f"{'keep' if len(agents) > 1 else 'keeps'} it on DecimalAI: remove it from an agent "
             "in the dashboard." if agents else None)
    if name in _authored.listed(project) and name not in _authored.managed(project):
        try:
            _authored.remove(project, name)
        except (ValueError, SkillpkgError) as exc:
            click.echo(f"  ✗ {exc}", err=True)
            raise SystemExit(int(ExitCode.USAGE))
        click.echo(f"  ✓ took {name} off the list in {project.rel(project.manifest_path)}; its "
                   f"folder {project.rel(project.skill_dir(name))} stays")
        if still:
            click.echo(still)
        return
    result = skillpkg.remove(name, force=force, directory=directory)
    _print_result(result)
    if still and result.ok and result.changes:
        click.echo(still)
    _exit(result)


# ── status ──────────────────────────────────────────────────────────────────

_STATE_GLYPH = {
    "clean": "✓",
    "modified": "✗",
    "missing": "↓",
    "not_installed": "↓",
    "unmanaged": "+",
    "unmanaged_stamped": "+",
    "illegal": "✗",
    "interrupted": "⚠",
}


@click.command("status", epilog=_EXIT_CODES)
@click.argument("skills_dir", required=False, type=click.Path(exists=True, file_okay=False))
@click.option("--remote", is_flag=True, default=False,
              help="Also ask the registry: outdated versions, descriptions changed upstream, "
                   "revoked versions.")
@click.option("--check", is_flag=True, default=False,
              help="For CI: exit 1 when anything has drifted (installed folders, or authored "
                   "skills local_ahead, remote_ahead or conflict); exit 8 when the authored "
                   "check could not reach the platform.")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Print the installed section as JSON.")
@_directory_option
@_registry_options
def skills_status(skills_dir, remote, check, as_json, directory, api_key, base_url):
    """Show the installed skills, and how your own skills compare with the platform.

    Installed: each folder of .agents/skills against the lock (clean, modified, missing,
    unmanaged), the manifest against the lock, an interrupted run, a version 1 lock that
    needs `decimalai skills migrate`. Read-only; only --remote reaches the network.

    Authored: when ./skills exists and an API key is set, each SKILL.md there against
    this checkout's last sync (in_sync, local_ahead, remote_ahead, conflict, local_only).
    If the platform cannot be asked, that is a warning; the exit code changes only with
    --check. With SKILLS_DIR, only the authored section, for that folder. Nothing is
    written.

    \b
    Examples:
        $ decimalai skills status
        $ decimalai skills status --remote --check   # in CI
        $ decimalai skills status ./agents/skills    # authored skills elsewhere
    """
    from .main import _authored_status

    if skills_dir is not None:
        if remote or check or as_json or directory:
            raise click.UsageError(
                "`decimalai skills status DIR` prints the authored skills in DIR only; drop "
                "DIR to use --remote, --check, --json or --directory")
        _authored_status(skills_dir, api_key, base_url)
        return

    from .. import skillpkg
    from ..skillpkg.project import Project

    result = skillpkg.status(remote=remote, check=check, directory=directory,
                             api_key=api_key or None, base_url=base_url, channel="cli")
    if as_json:
        _print_json(result)
        _exit(result)
    if not result.changes and any(isinstance(p, SkillpkgError) for p in result.problems):
        _print_problems(result.problems)   # a refusal (a bad --directory, the network)
        _exit(result)

    try:
        where = (Project.at(directory) if directory else Project.find()).directory
    except (OSError, SkillpkgError):
        where = Path(directory or os.getcwd())
    _print_installed(result, where)

    code = result.exit_code
    authored = Path(directory, "skills") if directory else Path("skills")
    if authored.is_dir():
        click.echo("")
        click.echo(f"  Authored skills ({authored if directory else './skills'}):")
        if api_key:
            counts = _authored_status(str(authored), api_key, base_url,
                                      project_root=directory, combined=True)
            code = _check_code(code, counts, check=check)
        else:
            click.echo("    set DECIMAL_API_KEY to compare them with the platform "
                       "(`decimalai skills sync` uploads them)")
    raise SystemExit(int(code))


def _check_code(code: ExitCode, counts: Optional[Dict[str, int]], *, check: bool) -> ExitCode:
    """The exit code of the combined view, given the authored section's counts (None: failed).

    The authored section never changes the code by itself. With ``check``, its drift
    counts like installed drift (1), and a check that failed is a network failure (8)
    unless drift was already found: then the answer is 1 whatever it would have said.
    """
    from .main import AUTHORED_DRIFT

    if not check or code != ExitCode.OK:
        return code
    if counts is None:
        return ExitCode.NETWORK
    if any(counts.get(state, 0) for state in AUTHORED_DRIFT):
        return ExitCode.PROBLEMS
    return code


def _print_installed(result: Any, where: Path) -> None:
    click.echo(f"  Installed skills ({_ROOT} in {where}):")
    rows = list(result.changes)
    if not rows:
        click.echo("    none; `decimalai skills add <ref>` installs a registry skill")
    for row in rows:
        name = row.install_name or row.skill or "?"
        what = ""
        if row.skill:
            what = row.skill if row.version is None else f"{row.skill} v{row.version}"
            if row.pin == "latest":
                what += " (latest)"
            elif row.pin:
                what += " (pinned)"
        state = row.state.replace("_", " ")
        if row.files:
            state += f": {', '.join(row.files)}"
        elif row.reason:
            state += f": {row.reason}"
        line = f"    {_STATE_GLYPH.get(row.state, '·')} {name:<32s} {what:<40s} {state}"
        if row.remote:
            line += f"  [{'; '.join(row.remote)}]"
        click.echo(line.rstrip())
    for problem in result.problems:
        if isinstance(problem, SkillpkgError):
            click.echo(f"  ✗ {problem}", err=True)
        else:
            glyph = "•" if getattr(problem, "level", None) == "info" else "⚠"
            click.echo(f"  {glyph} {problem}")


# ── verify ──────────────────────────────────────────────────────────────────


@click.command("verify", epilog=_EXIT_CODES)
@click.argument("path", required=False, type=click.Path())
@click.option("--strict", is_flag=True, default=False,
              help="Make the SDK's own rules errors in folders the lock does not list, and "
                   "in PATH.")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Print the findings as JSON.")
@_directory_option
def skills_verify(path, strict, as_json, directory):
    """Check skill folders the way the native skill loaders read them.

    With no PATH: the project, .agents/skills against .decimal/skills.lock. Every folder
    the lock lists must hold exactly its locked files; folders it does not list are
    checked like PATH.

    With PATH (a skill folder, its SKILL.md, or a folder of skill folders): the format
    rules every loader needs are errors; the SDK's own rules are warnings, or errors with
    --strict. Notices are information, never an error.

    \b
    Examples:
        $ decimalai skills verify
        $ decimalai skills verify ./skills --strict   # CI for a skills repository
    """
    from .. import skillpkg

    result = skillpkg.verify(path, directory=directory, strict=strict)
    if as_json:
        _print_json(result)
        _exit(result)
    counts = {"error": 0, "warning": 0, "notice": 0}
    for problem in result.problems:
        if isinstance(problem, SkillpkgError):
            click.echo(f"  ✗ {problem}", err=True)
            continue
        severity = getattr(problem, "severity", "error")
        counts[severity] = counts.get(severity, 0) + 1
        glyph = {"error": "✗", "warning": "⚠", "notice": "•"}.get(severity, "✗")
        click.echo(f"  {glyph} {problem}")
        fix = getattr(problem, "fix", None)
        if fix:
            click.echo(f"      fix: {fix}")
    if result.ok:
        warned = f"; {counts['warning']} warning(s)" if counts["warning"] else ""
        click.echo(f"  ✓ no errors{warned}")
    elif result.exit_code == ExitCode.PROBLEMS:
        click.echo(f"  {counts['error']} error(s), {counts['warning']} warning(s)")
    _exit(result)


# ── migrate ─────────────────────────────────────────────────────────────────


@click.command("migrate", epilog=_EXIT_CODES)
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Migrate the entries that resolve even when others do not.")
@_directory_option
@_registry_options
def skills_migrate(yes, directory, api_key, base_url):
    """Move a project from a version 1 lock (decimalai 0.14 and older) or an npx lock.

    Writes .decimal/skills.toml and a version 2 lock, and installs every skill again from
    the registry. Old folders inside .agents/skills move to .decimal/migrated-v1/; old
    folders anywhere else stay where they are and are listed. Nothing is deleted. A
    second run does nothing.

    \b
    Example:
        $ decimalai skills migrate
    """
    from .. import skillpkg
    from ..skillpkg.migration import format_table

    result = skillpkg.migrate(directory=directory, yes=yes, api_key=api_key or None,
                              base_url=base_url, channel="cli")
    if result.changes:
        for line in format_table(result.changes).splitlines():
            click.echo(f"  {line}".rstrip())
    elif result.ok and not result.problems:
        click.echo("  ✓ nothing to migrate: the project has no version 1 or npx lock")
    _print_problems(result.problems)
    _exit(result)


# ── pull ────────────────────────────────────────────────────────────────────


def _pull_get(url, **kwargs):
    """GET for the anonymous calls of ``pull`` — one retry on timeout."""
    import httpx

    try:
        return httpx.get(url, timeout=_PULL_HTTP_TIMEOUT, **kwargs)
    except httpx.TimeoutException:
        return httpx.get(url, timeout=_PULL_HTTP_TIMEOUT, **kwargs)


@click.command("pull")
@click.argument("ref")
@click.option("--out", "out_dir", default=None,
              help="Write the skill to OUT/<install-name>/. Default: the current directory.")
@click.option("--base-url", envvar=["DECIMAL_BASE_URL", "DECIMALAI_BASE_URL"],
              default=_DEFAULT_BASE_URL, show_envvar=True, help="Platform URL")
@click.option("--no-evals", "no_evals", is_flag=True, default=False,
              help="Skip the eval.yaml test suite (default: write it when the skill has one).")
@click.option("--stdout", "to_stdout", is_flag=True, default=False,
              help="Print SKILL.md to stdout instead of writing to disk.")
@click.option("--force", is_flag=True, default=False,
              help="Overwrite a folder that holds different files. Files that are not part "
                   "of the skill are left in place.")
def skills_pull(ref, out_dir, base_url, no_evals, to_stdout, force):
    """Copy a public registry skill to disk — no signup, no API key.

    Writes the registry's own files for the skill (SKILL.md and its bundled files) to
    OUT/<install-name>/, plus its eval.yaml test suite when it has one. It is a loose
    copy: no lock records it and nothing updates it. To manage a skill in a project
    (locked, updatable, installed in CI), use `decimalai skills add`.

    A folder that already holds different files is left alone (exit 5) unless --force.

    \b
    Examples:
        # Write ./playwright-cli/SKILL.md
        $ decimalai skills pull playwright-cli
    \b
        # Write to a specific path
        $ decimalai skills pull pdf --out ./agents/skills/
    \b
        # Just print the SKILL.md
        $ decimalai skills pull pdf --stdout
    """
    from skillevaluation import skillfmt

    from ..skillpkg.client import RegistryClient

    try:
        with RegistryClient.from_env(base_url=base_url, anonymous=True, channel="cli") as client:
            bundle = client.fetch_registry_bundle(ref, purpose="pull")
    except SkillpkgError as exc:
        click.echo(f"  ✗ {exc}", err=True)
        raise SystemExit(int(exc.exit_code))

    text = bundle.skill_md.decode("utf-8")
    if to_stdout:
        click.echo(text, nl=False)
        return

    try:
        frontmatter, _ = skillfmt.parse_skill_md_text(text)
    except skillfmt.SkillMdParseError:
        frontmatter = {}
    meta = frontmatter.get("metadata")
    meta = meta if isinstance(meta, dict) else {}
    name = str(meta.get("decimalai-name") or ref)
    version = meta.get("decimalai-version")
    skill_id = bundle.skill_id or meta.get("decimalai-skill-id")
    registry = base_url.rstrip("/") + "/api/v1/registry/skills"

    planned: Dict[str, Tuple[bytes, int]] = {f.path: (f.content, f.mode) for f in bundle.files}
    eval_cases = None
    if not no_evals and skill_id:
        evals = _fetch_eval(f"{registry}/{skill_id}/eval")
        if evals is not None:
            planned["eval.yaml"] = (evals[0].encode("utf-8"), 0o644)
            eval_cases = evals[1]

    target = Path(out_dir) if out_dir else Path.cwd()
    target = target / bundle.install_name
    _refuse_managed_root(target, bundle.install_name, ref)
    shown = f"v{version}" if version else "(version unknown)"
    try:
        differing, extra = _compare_folder(target, planned)
    except OSError as exc:
        click.echo(f"  ✗ could not read {target}: {exc}", err=True)
        raise SystemExit(int(ExitCode.PROBLEMS))
    complete = all(target.joinpath(*rel.split("/")).is_file() for rel in planned)
    if complete and not differing and not extra:
        click.echo(f"  ✓ {target} already holds {name} {shown}")
    else:
        if (differing or extra) and not force:
            listed = ", ".join((differing + extra)[:5])
            more = f" and {len(differing) + len(extra) - 5} more" if len(differing) + len(extra) > 5 else ""
            click.echo(f"  ✗ {target} already exists and differs from {name} {shown} "
                       f"({listed}{more}); re-run with --force to overwrite it, or pass "
                       "--out for another place", err=True)
            raise SystemExit(int(ExitCode.LOCAL_CHANGES))
        try:
            _write_folder(target, planned)
        except OSError as exc:
            click.echo(f"  ✗ could not write {target}: {exc}", err=True)
            raise SystemExit(int(ExitCode.PROBLEMS))
        click.echo(f"  ✓ Pulled {name} {shown}")
        click.echo(f"    → {target / 'SKILL.md'}")
        bundled = len(bundle.files) - 1
        if bundled:
            click.echo(f"    → {bundled} bundled file(s) (scripts/references/…)")
        if eval_cases is not None:
            click.echo(f"    → {target / 'eval.yaml'} ({eval_cases} test cases)")
        if extra:
            click.echo(f"    note: left in place, not part of the skill: {', '.join(extra)}")
    for missing in bundle.unavailable:
        click.echo(f"    note: not included: {missing.path} ({missing.reason})")
    for note in _fold_tool_notes(bundle.policy_notes):
        click.echo(f"    note: {note}")

    if skill_id:
        _print_quality(f"{registry}/{skill_id}")

    from .main import _dashboard_url

    click.echo("")
    click.echo("    A loose copy: no lock records it and nothing updates it. To manage it in a")
    click.echo("    project (locked, updatable, installed in CI):")
    click.echo(f"      $ decimalai skills add {shlex.quote(name)}")
    click.echo(f"      {_dashboard_url(base_url)}/skills/{urllib.parse.quote(name, safe='/')}")


def _refuse_managed_root(target: Path, install_name: str, ref: str) -> None:
    """Exit 5 when ``target`` is in the managed root of a project (``--force`` or not).

    A project with ``.decimal/skills.toml`` or ``.decimal/skills.lock`` owns its
    ``.agents/skills``: the lock records every byte there, and a loose copy in it is
    an unmanaged or modified folder that ``install --frozen`` refuses.
    """
    resolved = target.resolve()
    for root in resolved.parents:
        if root.name != "skills" or root.parent.name != ".agents":
            continue
        directory = root.parent.parent
        state = directory / ".decimal"
        if not ((state / "skills.toml").is_file() or (state / "skills.lock").is_file()):
            continue
        listed = False
        if resolved.parent == root:
            from ..skillpkg import lock as _lock
            from ..skillpkg.project import Project

            try:
                listed = install_name in _lock.load(Project.at(directory), missing_ok=True).skills
            except SkillpkgError:
                listed = False
        where = f"{_ROOT} of the project at {directory}"
        if listed:
            fix = ("run `decimalai skills install` there to restore the locked files, or "
                   f"`decimalai skills update {install_name}` for a newer version")
        else:
            fix = (f"run `decimalai skills add {shlex.quote(ref)}` there to install it as a "
                   "managed skill (`decimalai skills install` and `update` keep it current), "
                   "or pass --out somewhere else")
        click.echo(f"  ✗ {target} is inside {where}, which `decimalai skills` manages: its "
                   f"lock records every file there, and a loose copy makes "
                   f"`decimalai skills install --frozen` refuse; {fix}", err=True)
        raise SystemExit(int(ExitCode.LOCAL_CHANGES))


def _fetch_eval(url: str) -> Optional[Tuple[str, Any]]:
    """``(eval.yaml text, case count)`` from the registry, or None (no suite, or it failed)."""
    import httpx

    try:
        resp = _pull_get(url)
    except httpx.HTTPError as exc:
        click.echo(f"  ⚠ eval.yaml fetch failed: {exc} — skipping", err=True)
        return None
    if resp.status_code == 404:  # no eval.yaml authored: the SKILL.md stands on its own
        return None
    if resp.status_code != 200:
        click.echo(f"  ⚠ eval.yaml fetch returned HTTP {resp.status_code} — skipping", err=True)
        return None
    try:
        payload = resp.json() or {}
    except ValueError:
        return None
    text = payload.get("eval_yaml_text") if isinstance(payload, dict) else None
    if not isinstance(text, str) or not text:
        return None
    return text, payload.get("case_count", 0)


def _compare_folder(target: Path, planned: Dict[str, Tuple[bytes, int]]) -> Tuple[List[str], List[str]]:
    """``(files whose bytes differ, files that are not part of the skill)`` in ``target``.

    A missing file is no difference: writing it loses nothing. ``eval.yaml`` counts only
    when it is about to be written. A symlink, or a file where the folder belongs, is
    refused outright: writing through it would land somewhere else.
    """
    if not os.path.lexists(target):
        return [], []
    if target.is_symlink() or not target.is_dir():
        click.echo(f"  ✗ {target} is not a folder (a file or a symlink); pull writes a "
                   "folder there. Move it away, or pass --out", err=True)
        raise SystemExit(int(ExitCode.LOCAL_CHANGES))
    differing: List[str] = []
    extra: List[str] = []
    for dirpath, dirnames, filenames in os.walk(target):
        dirnames[:] = sorted(d for d in dirnames if d not in _CLUTTER_DIRS)
        base = Path(dirpath)
        for entry in [*dirnames, *filenames]:
            if (base / entry).is_symlink():
                rel = (base / entry).relative_to(target).as_posix()
                click.echo(f"  ✗ {target}/{rel} is a symlink; pull never writes through one. "
                           "Replace it with the real file, or pass --out", err=True)
                raise SystemExit(int(ExitCode.LOCAL_CHANGES))
        for filename in sorted(filenames):
            if filename in _CLUTTER_FILES or filename.endswith(".pyc"):
                continue
            rel = (base / filename).relative_to(target).as_posix()
            if rel in planned:
                if (base / filename).read_bytes() != planned[rel][0]:
                    differing.append(rel)
            elif rel != "eval.yaml":
                extra.append(rel)
    return differing, extra


def _write_folder(target: Path, planned: Dict[str, Tuple[bytes, int]]) -> None:
    for rel, (data, mode) in planned.items():
        dest = target.joinpath(*rel.split("/"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(data)
        if os.name == "posix":
            os.chmod(dest, stat.S_IMODE(mode))


def _print_quality(url: str) -> None:
    """The registry's quality and efficiency lines for the skill; nothing when unavailable.

    For example: ``SkillScore 87 (provisional) · +60 pts vs no skill · 96% live pass``.
    SkillScore is the quality composite (benchmark pass, live eval pass, rated trace
    quality); "provisional" means one evidence leg. The registry's detail route only
    reads: it counts no view, install or delivery, so this read changes no number.
    """
    import httpx

    try:
        resp = _pull_get(url)
        detail = resp.json() if resp.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return
    if not isinstance(detail, dict):
        return
    eff = detail.get("effectiveness") or {}
    bench = detail.get("benchmark_summary") or {}
    score = eff.get("skill_score")
    if score is None:
        score = eff.get("avg_effectiveness")  # frozen alias (older backends)
    quality = []
    if score is not None:
        early = " (provisional)" if (eff.get("score_components") or {}).get("provisional") else ""
        quality.append(f"SkillScore {score * 100:.0f}{early}")
    lift = bench.get("pass_rate_delta_pts")
    if lift is not None:
        quality.append(f"{'+' if lift >= 0 else ''}{lift:.0f} pts vs no skill")
    live_pass = eff.get("avg_pass_rate")
    if live_pass is not None:
        quality.append(f"{live_pass * 100:.0f}% live pass")
    if quality:
        click.echo(f"    {' · '.join(quality)}")
    efficiency = []
    for key, label in (("tokens_delta_pct", "tokens"), ("turns_delta_pct", "turns")):
        value = bench.get(key)
        if value is not None:
            efficiency.append(f"{'+' if value >= 0 else ''}{value:.0f}% {label}")
    if efficiency:
        click.echo(f"\n  Efficiency vs no skill: {' · '.join(efficiency)}")
