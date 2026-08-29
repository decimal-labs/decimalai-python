"""DecimalAI CLI — command-line interface."""

import logging
import urllib.parse

import click

from .. import __version__

logger = logging.getLogger("decimalai")


# Epilog rides every help render — including the bare `decimalai` invocation
# (click prints the group help there before exiting 2), so a keyless first
# run always sees where to start.
@click.group(epilog="Start with: decimalai init")
@click.version_option(version=__version__, prog_name="decimalai")
def cli():
    """DecimalAI — Agent dataset lifecycle platform."""
    pass


# ── Shared helper ──────────────────────────────────────────

def _make_client(api_key, base_url, project):
    """Create a client, raising a helpful error if api_key is missing."""
    from .._client import DecimalAIClient
    if not api_key:
        click.echo("")
        click.echo("  ✗ No API key found.", err=True)
        click.echo("")
        click.echo("  Set your key:")
        click.echo('    export DECIMAL_API_KEY="dai_sk_..."')
        click.echo("")
        click.echo(f"  Get one at {_dashboard_url(base_url)}/settings")
        click.echo("  Or run `decimalai init` to verify your setup.")
        raise SystemExit(1)
    return DecimalAIClient(api_key=api_key, project=project, base_url=base_url)


# ── Common options ─────────────────────────────────────────

_common_options = [
    click.option("--api-key", envvar=["DECIMAL_API_KEY", "DECIMALAI_API_KEY"], help="API key"),
    # The envvar spelling matters (fixed 2026-06-10): without it, pointing the
    # CLI at a self-hosted or local backend had no effect — `decimalai skills
    # pull` silently went to the hosted default and 404'd on skills that only
    # exist locally.
    click.option(
        "--base-url",
        envvar=["DECIMAL_BASE_URL", "DECIMALAI_BASE_URL"],
        default="https://api.decimal.ai",
        show_envvar=True,
        help="Platform URL",
    ),
    # Deprecated and inert: the platform never read the project label. Kept so
    # existing scripts passing --project don't start failing; it does nothing.
    click.option(
        "--project",
        default="default",
        help="(deprecated, no effect — traces are scoped by your API key)",
    ),
]


def _dashboard_url(base_url: str) -> str:
    """Frontend URL for follow-up links.

    Resolved in three steps because the bare `api.→app.` string-replace this
    used to do printed backend links (localhost:8000/skills/…) under local dev,
    which don't serve a UI. DECIMAL_APP_URL
    wins when set; localhost backends map to the conventional :3000 frontend;
    the replace stays as the hosted-domain fallback.
    """
    import os

    explicit = os.environ.get("DECIMAL_APP_URL") or os.environ.get("DECIMALAI_APP_URL")
    if explicit:
        return explicit.rstrip("/")
    if "localhost" in base_url or "127.0.0.1" in base_url:
        return "http://localhost:3000"
    return base_url.replace("api.", "app.").replace("/api", "").rstrip("/")

def common_options(func):
    for option in reversed(_common_options):
        func = option(func)
    return func


# ── Init command ───────────────────────────────────────────

_DEFAULT_BASE_URL = "https://api.decimal.ai"


def _http_die(exc, base_url):
    """Turn a failed platform call into the right one-line diagnosis.

    Same triage `init` uses for /auth/verify, and for the same reason: when
    the server ANSWERS, this is not a connection problem, and reporting a
    rejected key as a network fault sends people off debugging DNS.
    """
    import httpx

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (401, 403):
            click.echo("  ✗ Invalid API key — the server rejected it.", err=True)
            click.echo(f"    Get a valid key at {_dashboard_url(base_url)}/settings")
        else:
            click.echo(f"  ✗ Server returned HTTP {status} from {base_url}.", err=True)
            click.echo("    Check the base URL points at a DecimalAI backend, or try again.")
    else:
        click.echo(f"  ✗ Connection failed: {exc}", err=True)
        click.echo(f"    Check your base URL ({base_url}) and network.")
    raise SystemExit(1)


def _scaffold_agent_file(
    agent_name, api_key, base_url, framework, out_path, force, dry_run, model,
):
    """`decimalai init <agent-name>` — write a runnable file for a real agent.

    The gap this closes: the dashboard stores a name, a description and a set
    of skills, and the user still had to write the agent. Now the
    configuration they already made becomes a file that runs. We generate;
    they run — nothing executes on our side.

    The agent is RESOLVED against the API rather than assumed. Inventing one
    locally would produce a file that traces under a name the workspace does
    not have: the dashboard page stays empty, the attached skills never
    resolve, and nothing anywhere reports an error.
    """
    import os

    from .scaffold import (
        DEFAULT_OUTPUT,
        UnknownFramework,
        env_vars,
        install_command,
        normalize_framework,
        render_agent_file,
    )

    # Framework first: it needs no key and no network, so a typo fails in
    # milliseconds instead of after two round trips.
    try:
        framework = normalize_framework(framework)
    except UnknownFramework as e:
        import textwrap
        click.echo("")
        first = True
        # Wrapped per paragraph: the refusal explains WHY, which takes more
        # than one line, and an unwrapped 200-character sentence is a wall
        # nobody reads.
        for para in str(e).splitlines():
            for line in textwrap.wrap(para, width=74) or [""]:
                click.echo(("  ✗ " if first else "    ") + line, err=True)
                first = False
        click.echo("")
        raise SystemExit(1)

    resolved_key = (
        api_key
        or os.environ.get("DECIMAL_API_KEY")
        or os.environ.get("DECIMALAI_API_KEY")
    )
    if not resolved_key:
        click.echo("")
        click.echo("  ✗ No API key found.", err=True)
        click.echo("")
        click.echo("  Set your key:")
        click.echo('    export DECIMAL_API_KEY="dai_sk_..."')
        click.echo("")
        click.echo(f"  Get one at {_dashboard_url(base_url)}/settings")
        raise SystemExit(1)

    out_path = out_path or DEFAULT_OUTPUT

    from .._agent import AgentConfig
    from .._client import DecimalAIClient
    client = DecimalAIClient(api_key=resolved_key, base_url=base_url)

    # 1. Resolve the agent.
    #
    # This runs BEFORE the refuse-to-clobber check, and the order was chosen
    # by running it the other way round: with the file check first, someone
    # who typo'd the name in a directory that already had an agent.py was told
    # "agent.py already exists" and offered `--force` — advice that, if
    # followed, produces a completely different error. The name is the
    # question they got wrong and cannot fix locally, so it is diagnosed
    # first. The extra round trip costs ~0.1s.
    try:
        # No `limit`: the endpoint returns the full list when it is unset,
        # and passing one activates a pagination path that truncates. The
        # ordering puts manifest-only agents last, so a truncation would
        # drop the never-traced UI-created agent this command exists to
        # find, and report it as "no such agent".
        resp = client._http.get("/api/v1/agents")
        resp.raise_for_status()
        agents = resp.json().get("agents") or []
    except Exception as e:  # noqa: BLE001 — re-raised as a diagnosis
        client.close()
        _http_die(e, base_url)

    names = [a.get("agent_name") for a in agents if a.get("agent_name")]
    if agent_name not in names:
        client.close()
        click.echo("")
        click.echo(f"  ✗ No agent named {agent_name!r} in this workspace.", err=True)
        # A near miss is the common case (refund_bot vs refund-bot) and it is
        # worth naming: the two look identical at a glance and produce a
        # completely empty dashboard page.
        try:
            from .. import _edit_distance_within
            close = [n for n in names if _edit_distance_within(agent_name, n, 2)]
        except Exception:
            close = []
        click.echo("")
        if close:
            click.echo(f"  Did you mean: {', '.join(sorted(close)[:5])}")
            click.echo("")
        elif names:
            click.echo(f"  Agents in this workspace: {', '.join(sorted(names)[:8])}")
            click.echo("")
        click.echo("  Create it first, then run this again:")
        click.echo(f"    → {_dashboard_url(base_url)}/agents/new")
        click.echo("")
        raise SystemExit(1)

    # 2. Refuse to clobber — after the name is known good, but still before
    #    the second round trip and before anything is written.
    #    The parent-directory check rides along here for the same reason the
    #    name check moved ahead of the clobber check: a destination we cannot
    #    write is knowable now, and diagnosing it after the second round trip
    #    makes the user pay two requests for a stack trace.
    parent = os.path.dirname(out_path) or "."
    if not dry_run and not os.path.isdir(parent):
        client.close()
        click.echo("")
        click.echo(f"  ✗ No such directory: {parent}", err=True)
        click.echo("")
        click.echo(f"  Create it:       mkdir -p {parent}")
        click.echo("  Or write here:   decimalai init "
                   f"{agent_name}")
        click.echo("")
        raise SystemExit(1)
    if not dry_run and os.path.exists(out_path):
        if not force:
            client.close()
            click.echo("")
            click.echo(f"  ✗ {out_path} already exists.", err=True)
            click.echo("")
            click.echo("  Overwrite it:    decimalai init "
                       f"{agent_name} --force")
            click.echo("  Write elsewhere: decimalai init "
                       f"{agent_name} --out my_agent.py")
            click.echo("  See it first:    decimalai init "
                       f"{agent_name} --dry-run")
            click.echo("")
            raise SystemExit(1)
        if os.path.isdir(out_path):
            # --force overwrites a file; it must never try to unlink a
            # directory the user pointed at by accident.
            client.close()
            click.echo("")
            click.echo(f"  ✗ {out_path} is a directory.", err=True)
            click.echo("")
            raise SystemExit(1)

    # 3. Fetch its skills, so the file can name what it will use.
    try:
        quoted = urllib.parse.quote(agent_name, safe="")
        resp = client._http.get(f"/api/v1/agents/{quoted}/skills")
        resp.raise_for_status()
        skills = resp.json().get("skills") or []
    except Exception as e:  # noqa: BLE001
        client.close()
        _http_die(e, base_url)

    # 4. Read its prompt.
    #
    #    Two failures, two answers, and telling them apart is the whole point
    #    of this block. The generated file calls `decimalai.load_agent()` at
    #    module scope, so it needs the route to EXIST; but it reads the value
    #    itself at run time, so it does not need this read to have SUCCEEDED.
    from .._client import AgentNotFoundError
    prompt = None
    prompt_unreadable = False
    try:
        prompt = AgentConfig._from_payload(
            client.get_agent_prompt(agent_name), requested_name=agent_name,
        )
    except AgentNotFoundError:
        # Step 1 already proved this agent is in this workspace, so a 404
        # HERE can only mean the route is missing. Writing the file anyway
        # would hand someone a program whose first statement cannot succeed —
        # a scaffold that dies on line one, every run, generated in silence.
        # No string heuristic is needed: the existence check above is what
        # makes this unambiguous.
        client.close()
        click.echo("")
        click.echo("  ✗ This backend does not serve agent prompts yet.", err=True)
        click.echo("")
        click.echo("    It needs GET /api/v1/agents/{name}/prompt, which the")
        click.echo("    generated file calls at startup. Update the backend, or")
        click.echo(f"    point --base-url somewhere current ({base_url}).")
        click.echo("")
        raise SystemExit(1)
    except Exception:  # noqa: BLE001 — cosmetic; the run-time read is the loud one
        # A timeout or a 5xx does NOT prove the generated file cannot run: it
        # reads the prompt itself at run time, and `load_agent()` raises there
        # rather than returning an empty prompt. Cost a comment, not the
        # scaffold. Nothing is written INTO the file from this read except a
        # character count and a version number.
        prompt_unreadable = True
    client.close()

    source = render_agent_file(
        agent_name,
        framework=framework,
        skills=skills,
        model=model,
        prompt=prompt,
        # Only pinned when it is not the hosted default: it is not a secret,
        # and a file scaffolded against a local backend that silently points
        # at production is a worse default than a redundant line.
        base_url=None if base_url == _DEFAULT_BASE_URL else base_url,
    )

    if dry_run:
        click.echo(source, nl=False)
        return

    # The parent directory was checked above, but a race, a read-only mount,
    # or a permission bit still surfaces here — and every other failure in
    # this command is a one-line diagnosis, so this one is too.
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(source)
    except OSError as e:
        click.echo("")
        click.echo(f"  ✗ Could not write {out_path}: {e}", err=True)
        click.echo("")
        click.echo("  Write elsewhere: decimalai init "
                   f"{agent_name} --out ~/agent.py")
        click.echo("  See it first:    decimalai init "
                   f"{agent_name} --dry-run")
        click.echo("")
        raise SystemExit(1)

    dashboard = _dashboard_url(base_url)
    n = len(skills)
    # Says whether the prompt was picked up, at the moment someone would ask.
    # Never "no system prompt" for a read that FAILED: that is a claim, it is
    # the wrong one, and it sends someone to write a prompt they already wrote.
    # An unreadable read gets its own line below instead.
    prompt_note = ""
    if prompt is not None:
        prompt_note = (
            ", system prompt" if prompt.system_prompt else ", no system prompt"
        )
    click.echo("")
    click.echo(f"  ✓ Wrote {out_path} — {agent_name}, {framework}, "
               f"{n} skill{'' if n == 1 else 's'}{prompt_note}")
    if prompt_unreadable:
        click.echo("")
        click.echo("  ! Could not read this agent's system prompt just now — the "
                   "file reads it at")
        click.echo("    run time, so this does not affect what it will send.")
    click.echo("")
    click.echo("  Install:")
    click.echo(f"    {install_command(framework, model)}")
    click.echo("")
    click.echo("  Set:")
    for var in env_vars(framework, model):
        already = " (already set)" if os.environ.get(var) else ""
        click.echo(f'    export {var}="..."{already}')
    click.echo("")
    click.echo("  Run:")
    click.echo(f"    python {out_path}")
    click.echo("")
    click.echo("  The trace appears at:")
    click.echo(f"    → {dashboard}/agents/{urllib.parse.quote(agent_name, safe='')}")
    click.echo("")


@cli.command()
@click.argument("agent_name", required=False)
@click.option("--api-key", envvar=["DECIMAL_API_KEY", "DECIMALAI_API_KEY"], help="API key")
@click.option(
    "--base-url",
    envvar=["DECIMAL_BASE_URL", "DECIMALAI_BASE_URL"],
    default=_DEFAULT_BASE_URL,
    show_envvar=True,
    help="Platform URL",
)
@click.option(
    # default=None, not "langchain": a flag that carries a default cannot be
    # told apart from an unset one, which is how --framework came to be
    # silently discarded on the no-agent-name path while its neighbours were
    # loudly refused. The default is applied below instead.
    "--framework",
    default=None,
    help="Framework to generate for: langchain (default) or openai-agents",
)
@click.option("--out", "out_path", default=None,
              help="Where to write (default: ./agent.py)")
@click.option("--model", default=None,
              help="Model for the generated file's MODEL line")
@click.option("--force", is_flag=True, help="Overwrite an existing file")
@click.option("--dry-run", is_flag=True, help="Print the file instead of writing it")
@click.option("--test-trace/--no-test-trace", default=True, help="Send a test trace to verify connectivity")
def init(agent_name, api_key, base_url, framework, out_path, model, force, dry_run, test_trace):
    """Verify your setup, or scaffold a runnable agent.

    With no argument: checks API key validity, shows workspace info, and
    optionally sends a test trace to confirm end-to-end connectivity.

    With an AGENT_NAME: writes an `agent.py` wired to that agent — its name
    bound, its skills loaded, ready to run. The agent must already exist
    (create one in the dashboard first). Nothing runs on DecimalAI's side;
    the file is yours.

    \b
    Examples:
        $ decimalai init                          # verify the setup
        $ decimalai init refund-bot               # write ./agent.py
        $ decimalai init refund-bot --dry-run     # print it instead
        $ decimalai init refund-bot --framework openai-agents
    """
    import os

    if agent_name:
        return _scaffold_agent_file(
            agent_name, api_key, base_url, framework or "langchain", out_path,
            force, dry_run, model,
        )

    # Scaffold-only flags without a name would otherwise be silent no-ops,
    # and the user would sit waiting for a file that was never going to be
    # written.
    scaffold_only = [
        name for name, used in (
            ("--framework", framework is not None),
            ("--out", out_path is not None),
            ("--model", model is not None),
            ("--force", force),
            ("--dry-run", dry_run),
        ) if used
    ]
    if scaffold_only:
        click.echo("")
        click.echo(f"  ✗ {', '.join(scaffold_only)} needs an agent name.", err=True)
        click.echo("")
        click.echo("    decimalai init <agent-name>")
        click.echo("")
        raise SystemExit(1)

    # 1. Resolve API key
    resolved_key = api_key or os.environ.get("DECIMAL_API_KEY") or os.environ.get("DECIMALAI_API_KEY")
    if not resolved_key:
        click.echo("")
        click.echo("  ✗ No API key found.", err=True)
        click.echo("")
        click.echo("  Set your key:")
        click.echo('    export DECIMAL_API_KEY="dai_sk_..."')
        click.echo("")
        click.echo("  Or get one from: Settings → API Key in the dashboard")
        click.echo(f"    → {_dashboard_url(base_url)}/settings")
        raise SystemExit(1)

    key_preview = f"{resolved_key[:10]}...{resolved_key[-4:]}"
    click.echo("")
    click.echo(f"  ✓ API key: {key_preview}")

    # 2. Verify connectivity
    import httpx

    from .._client import DecimalAIClient
    client = DecimalAIClient(api_key=resolved_key, base_url=base_url)
    try:
        resp = client._http.get("/api/v1/auth/verify")
        resp.raise_for_status()
        data = resp.json()
        workspace_id = data.get("workspace_id")
        scope = data.get("scope") or "workspace"
        if workspace_id:
            click.echo(f"  ✓ Connected to workspace: {workspace_id} (scope: {scope})")
        else:
            click.echo(f"  ✓ Connected ({scope} scope — all workspaces)")
    except httpx.HTTPStatusError as e:
        # The server ANSWERED — this is not a connection problem, so don't
        # send the user off debugging DNS/URLs. The old "Connection failed"
        # + MDN-link message inverted the triage for the common case: a bad
        # API key was reported as a network fault.
        status = e.response.status_code
        if status in (401, 403):
            click.echo("  ✗ Invalid API key — the server rejected it.", err=True)
            click.echo(f"    Get a valid key at {_dashboard_url(base_url)}/settings")
        else:
            click.echo(f"  ✗ Server returned HTTP {status} from {base_url}.", err=True)
            click.echo("    Check the base URL points at a DecimalAI backend, or try again.")
        client.close()
        raise SystemExit(1)
    except Exception as e:
        # Transport errors only (DNS failure, connection refused, timeout).
        click.echo(f"  ✗ Connection failed: {e}", err=True)
        click.echo(f"    Check your base URL ({base_url}) and network.")
        client.close()
        raise SystemExit(1)

    # 3. Send test trace
    if test_trace:
        try:
            import decimalai as sdk
            sdk.init(api_key=resolved_key, base_url=base_url)

            @sdk.trace(agent_name="decimalai-init-test")
            def _test_agent(query: str) -> str:
                return f"Test response for: {query}"

            _test_agent("Hello from decimalai init!")
            click.echo("  ✓ Test trace sent successfully")
        except Exception as e:
            click.echo(f"  ⚠ Test trace failed: {e}")
            click.echo("    (Your API key is valid — trace sending may need a running server)")

    # 4. Show next steps
    dashboard_url = _dashboard_url(base_url)
    click.echo("")
    click.echo("  Already created an agent? Turn it into a runnable file:")
    click.echo("    decimalai init <agent-name>   # writes ./agent.py, skills wired")
    click.echo("")
    click.echo("  See it work in 2 minutes — seeds a demo into your workspace:")
    click.echo("    decimalai demo regression   # what your next agent change would break")
    click.echo("    decimalai demo skills       # the registry, ranked by real effectiveness")
    click.echo("")
    click.echo(f"  → Open dashboard: {dashboard_url}/traces")
    click.echo("  → Docs: https://docs.decimal.ai/quickstart")
    # Points at the support-agent notebook, not the quickstart one: quickstart.ipynb
    # calls decimalai.init() in its second code cell and raises DecimalConfigError on
    # its placeholder key, so it is the worst possible first click for someone who has
    # just run `init`. The support-agent notebook runs with no DecimalAI account at all.
    click.echo("  → Build an agent from a skill: https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/support-agent/support_agent.ipynb")
    click.echo("")
    client.close()

@cli.group()
def traces():
    """Trace commands."""
    pass


@traces.command("list")
@click.option("--limit", default=20, help="Max results to return")
@click.option("--status", default=None, help="Filter by status")
@common_options
def traces_list(limit, status, api_key, base_url, project):
    """List traces for a project."""
    client = _make_client(api_key, base_url, project)
    try:
        result = client.list_traces(limit=limit, status=status)
        traces_data = result.get("traces", [])
        click.echo(f"Found {len(traces_data)} traces:")
        for t in traces_data:
            click.echo(
                f"  {t.get('id', '?')[:8]}  "
                f"{t.get('status', '?'):8s}  "
                f"{t.get('agent_name', '?'):20s}  "
                f"{t.get('started_at', '?')}"
            )
    finally:
        client.close()


@traces.command("show")
@click.argument("trace_id")
@common_options
def traces_show(trace_id, api_key, base_url, project):
    """Show detail for a specific trace."""
    import json
    client = _make_client(api_key, base_url, project)
    try:
        result = client.get_trace(trace_id)
        click.echo(json.dumps(result, indent=2, default=str))
    finally:
        client.close()


@traces.command("stats")
@click.option("--agent-name", default=None, help="Filter by agent name")
@common_options
def traces_stats(agent_name, api_key, base_url, project):
    """Show trace statistics."""
    client = _make_client(api_key, base_url, project)
    try:
        params = {}
        if agent_name:
            params["agent_name"] = agent_name
        resp = client._http.get("/api/v1/traces/stats", params=params)
        resp.raise_for_status()
        import json
        click.echo(json.dumps(resp.json(), indent=2, default=str))
    finally:
        client.close()


@traces.command("import")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--format", "fmt", default="auto", help="Import format: auto, json, jsonl")
@common_options
def traces_import(file_path, fmt, api_key, base_url, project):
    """Import traces from a JSON or JSONL file."""
    import json as json_mod
    client = _make_client(api_key, base_url, project)

    # Auto-detect format
    if fmt == "auto":
        fmt = "jsonl" if file_path.endswith(".jsonl") else "json"

    try:
        with open(file_path) as f:
            if fmt == "jsonl":
                resp = client._http.post(
                    "/api/v1/import/jsonl",
                    content=f.read(),
                    headers={"Content-Type": "application/x-ndjson"},
                )
            else:
                data = json_mod.load(f)
                traces_data = data if isinstance(data, list) else data.get("traces", [data])
                resp = client._http.post("/api/v1/import/traces", json={"traces": traces_data})
            resp.raise_for_status()
            result = resp.json()
            click.echo(f"Imported {result.get('imported_count', 0)} traces ({result.get('error_count', 0)} failed)")
    finally:
        client.close()


# ── Eval commands ──────────────────────────────────────────

@cli.group()
def eval():
    """Evaluation commands."""
    pass


@eval.command("push")
@click.argument("trace_id")
@click.option("--score", multiple=True, help="Score in name=value format (e.g. quality=0.9)")
@click.option("--source", default="cli", help="Score source label")
@common_options
def eval_push(trace_id, score, source, api_key, base_url, project):
    """Push evaluation scores to a trace."""
    client = _make_client(api_key, base_url, project)
    try:
        scores = []
        for s in score:
            name, val = s.split("=", 1)
            scores.append({"name": name.strip(), "score": float(val)})
        if not scores:
            click.echo("Error: at least one --score name=value is required", err=True)
            raise SystemExit(1)
        client.push_eval_scores(trace_id, source=source, scores=scores)
        click.echo(f"Pushed {len(scores)} scores to trace {trace_id[:8]}")
    finally:
        client.close()


# ── Evaluator config (deterministic + LLM judge) ───────────

@cli.group()
def evaluators():
    """Manage evaluators — deterministic checks and LLM judges."""
    pass


@evaluators.command("list")
@click.option("--agent", "agent_name", help="List only evaluators for this agent")
@common_options
def evaluators_list(agent_name, api_key, base_url, project):
    """List configured evaluators."""
    client = _make_client(api_key, base_url, project)
    try:
        result = client.list_evaluators(agent_name=agent_name)
        evals = result.get("evaluators", [])
        if not evals:
            click.echo("No evaluators configured.")
            return
        for ev in evals:
            mark = "●" if ev.get("enabled") else "○"
            badge = ev.get("eval_type", "?")
            agent = ev.get("agent_name") or "(workspace)"
            click.echo(
                f"  {mark} {ev.get('display_name') or ev.get('name'):<30}  "
                f"{badge:<14}  {agent:<24}  id={ev.get('id', '')[:8]}"
            )
    finally:
        client.close()


@evaluators.command("templates")
@common_options
def evaluators_templates(api_key, base_url, project):
    """List available evaluator templates (pre-built deterministic + LLM judge)."""
    client = _make_client(api_key, base_url, project)
    try:
        result = client.list_evaluator_templates()
        for t in result.get("templates", []):
            click.echo(
                f"  {t['name']:<30}  {t['eval_type']:<14}  {t['category']:<10}  {t.get('description', '')}"
            )
    finally:
        client.close()


@evaluators.command("add")
@click.option("--agent", "agent_name", help="Attach to this agent (omit for workspace-wide)")
@click.option("--template", "template_id", help="Pre-built template id (e.g. 'helpfulness_judge', 'skill_output_check')")
@click.option("--name", help="Evaluator name (required if not using --template)")
@click.option("--type", "eval_type",
              type=click.Choice(["deterministic", "llm_judge"]),
              help="Evaluator type")
@click.option("--category", default="quality",
              type=click.Choice(["quality", "safety", "rag", "agentic", "custom"]),
              help="Category bucket")
@click.option("--rubric", "prompt_template",
              help="Rubric prompt for llm_judge type. Use {input} and {output} placeholders.")
@click.option("--threshold", type=float, default=None, help="Pass threshold (0.0-1.0)")
@click.option("--display-name", help="Human-readable name")
@click.option("--description", help="One-line description")
@common_options
def evaluators_add(agent_name, template_id, name, eval_type, category, prompt_template,
                   threshold, display_name, description, api_key, base_url, project):
    """Add an evaluator.

    \b
    Examples:
        # From a template
        decimalai evaluators add --template helpfulness_judge --agent support-agent

        # Custom LLM judge
        decimalai evaluators add \\
            --name tone_check \\
            --type llm_judge \\
            --rubric 'Rate the professionalism of: {output}' \\
            --agent support-agent

        # Custom deterministic (template-based; backend can't run arbitrary code)
        decimalai evaluators add --template skill_output_check --agent support-agent
    """
    if not template_id and not name:
        click.echo("Error: pass --template TEMPLATE_ID or --name NAME (+ --type)", err=True)
        raise SystemExit(1)
    if not template_id and not eval_type:
        click.echo("Error: --type is required when not using --template", err=True)
        raise SystemExit(1)
    if eval_type == "llm_judge" and not template_id and not prompt_template:
        click.echo("Error: --rubric is required for custom llm_judge evaluators", err=True)
        raise SystemExit(1)
    if eval_type == "deterministic" and not template_id:
        click.echo(
            "Note: custom deterministic evaluators must be Python @eval functions "
            "in your agent code. The HTTP/CLI path only supports pre-built templates "
            "(see `decimalai evaluators templates`).",
            err=True,
        )
        raise SystemExit(1)

    client = _make_client(api_key, base_url, project)
    try:
        result = client.add_evaluator(
            agent_name=agent_name,
            template_id=template_id,
            name=name,
            eval_type=eval_type,
            category=category,
            prompt_template=prompt_template,
            threshold=threshold,
            display_name=display_name,
            description=description,
        )
        label = result.get("display_name") or result.get("name") or "evaluator"
        click.echo(f"✓ Added {label} (id={result.get('id', '')[:8]})")
    finally:
        client.close()


@evaluators.command("remove")
@click.argument("evaluator_id")
@common_options
def evaluators_remove(evaluator_id, api_key, base_url, project):
    """Remove an evaluator by id."""
    client = _make_client(api_key, base_url, project)
    try:
        client.remove_evaluator(evaluator_id)
        click.echo(f"✓ Removed evaluator {evaluator_id[:8]}")
    finally:
        client.close()


# ── Skills commands ────────────────────────────────────────

@cli.group()
def skills():
    """Skill commands."""
    pass


@skills.command("list")
@click.option("--limit", default=50, help="Max results")
@common_options
def skills_list(limit, api_key, base_url, project):
    """List skills in your workspace."""
    client = _make_client(api_key, base_url, project)
    try:
        resp = client._http.get("/api/v1/skills", params={"limit": limit})
        resp.raise_for_status()
        data = resp.json()
        skills_data = data.get("skills", data) if isinstance(data, dict) else data
        if isinstance(skills_data, list):
            click.echo(f"Found {len(skills_data)} skills:")
            for s in skills_data:
                # The list endpoint nests the version under latest_version
                # (there is no top-level version_count) — reading the wrong
                # key printed "v?" for every row.
                version = (s.get("latest_version") or {}).get(
                    "version_number", s.get("version_count", "?")
                )
                click.echo(f"  {s.get('name', '?'):30s}  v{version}")
        else:
            import json
            click.echo(json.dumps(data, indent=2))
    finally:
        client.close()


@skills.command("sync")
@click.argument("skills_dir", type=click.Path(exists=True, file_okay=False), default="./skills")
@click.option(
    "--apply-pulls/--no-apply-pulls",
    default=True,
    help="Write pull_to_local results back to disk (default: yes).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Walk and hash but don't POST. Useful for previewing what would change.",
)
@common_options
def skills_sync(skills_dir, apply_pulls, dry_run, api_key, base_url, project):
    """Sync local SKILL.md files at SKILLS_DIR with the platform registry.

    Walks SKILLS_DIR (default ./skills) for SKILL.md files, hashes each
    one (SHA-256 of body_markdown), and POSTs to /api/v1/skills/sync
    with ``response_mode="diff"`` so the backend returns a per-skill
    ``actions`` array:

      \b
      created      — skill did not exist on backend, now does
      no_change    — local hash matches backend; nothing to do
      pushed       — local was newer; backend version bumped
      pulled       — backend was newer; CLI overwrites local file
      failed       — per-skill error (rest of the batch still processed)

    Conflict policy: ``newer_wins`` via ``local_updated_at``. Equal hashes
    are always ``no_change``.

    \b
    Examples:
        $ decimalai skills sync                # uses ./skills
        $ decimalai skills sync ./agents/skills
        $ decimalai skills sync --dry-run
    """
    import hashlib
    from pathlib import Path

    from ..skills import _split_frontmatter, _title_from_skill_md

    base = Path(skills_dir).resolve()

    # Walk for SKILL.md (one per skill directory). We tolerate both
    # layouts: <root>/<name>/SKILL.md and <root>/<name>.md (rare).
    skills_payload: list[dict] = []
    discovered_paths: dict[str, Path] = {}

    for skill_md in base.rglob("SKILL.md"):
        try:
            content = skill_md.read_text(encoding="utf-8")
        except OSError as e:
            click.echo(f"  ⚠ skipping {skill_md} ({e})", err=True)
            continue

        frontmatter, body = _split_frontmatter(content)
        if not body or not body.strip():
            click.echo(f"  ⚠ skipping {skill_md} (empty body)", err=True)
            continue

        name = (
            (frontmatter.get("name") or "").strip()
            or skill_md.parent.name
        )
        if not name:
            click.echo(f"  ⚠ skipping {skill_md} (no name)", err=True)
            continue

        # Hash is on the body specifically, not the whole file — that
        # way changing only frontmatter (e.g. tweaking description) does
        # not look like a content change to the backend.
        content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()

        # git-aware timestamp: prefer the commit time so a fresh checkout
        # (every file's mtime reset to "now") doesn't make the local copy always
        # "win" under newer_wins and clobber a more-recent dashboard edit.
        from ..skills import _local_updated_at_iso
        local_updated_at = _local_updated_at_iso(str(skill_md))

        entry = {
            "name": name,
            "display_name": _title_from_skill_md(frontmatter, body),
            "content_hash": content_hash,
            "body_markdown": body,
            "description": frontmatter.get("description") or "",
            "category": frontmatter.get("category"),
            "trigger_phrases": frontmatter.get("trigger_phrases"),
            "frontmatter": frontmatter or None,
            "local_updated_at": local_updated_at,
        }

        # An eval.yaml beside SKILL.md is part of the skill — sync it too,
        # so authored cases exist on the platform and a later
        # `skills push` can attach per-case results. Previously only
        # `skills benchmark` (metered) uploaded the suite, which
        # dead-ended the free local→push funnel: push skipped every case
        # and the hint said "run skills sync" — which didn't help.
        eval_yaml_path = skill_md.parent / "eval.yaml"
        if eval_yaml_path.is_file():
            try:
                eval_text = eval_yaml_path.read_text(encoding="utf-8")
                try:
                    from skillevaluation.parser import (
                        EvalYamlParseError,
                        parse_eval_yaml,
                    )
                except ImportError:
                    # Installs predating the skillevaluation dependency hit a
                    # bare ModuleNotFoundError here the moment an eval.yaml
                    # exists. Name the actual fix instead.
                    raise SystemExit(
                        "skills sync needs the `skillevaluation` package to validate "
                        f"{eval_yaml_path} — your decimalai install predates it. "
                        "Fix: pip install -U decimalai"
                    )
                try:
                    parse_eval_yaml(eval_text)
                    entry["eval_yaml_text"] = eval_text
                    entry["eval_yaml_hash"] = hashlib.sha256(
                        eval_text.encode("utf-8")
                    ).hexdigest()
                except EvalYamlParseError as exc:
                    click.echo(
                        f"  ⚠ {eval_yaml_path}: invalid eval.yaml not synced ({exc})",
                        err=True,
                    )
            except OSError as exc:
                click.echo(f"  ⚠ could not read {eval_yaml_path}: {exc}", err=True)

        skills_payload.append(entry)
        discovered_paths[name] = skill_md

    if not skills_payload:
        click.echo(f"  No SKILL.md files found under {base}")
        return

    click.echo(f"  Discovered {len(skills_payload)} skill(s) under {base}")

    if dry_run:
        click.echo("  --dry-run: skipping POST")
        for entry in skills_payload:
            click.echo(f"    • {entry['name']:<40s}  hash={entry['content_hash'][:12]}")
        return

    # Stamp this checkout's install identity so the backend records a
    # per-install synced baseline and `decimalai skills status` can show drift.
    from .._install import get_install_identity

    identity = get_install_identity()
    install_id = identity.get("install_id")
    if install_id:
        click.echo(f"  Install: {identity.get('install_label') or install_id[:8]}")

    client = _make_client(api_key, base_url, project)
    try:
        sync_body = {
            "skills": skills_payload,
            "conflict_policy": "newer_wins",
            "response_mode": "diff",
        }
        if install_id:
            sync_body["install_id"] = install_id
            if identity.get("install_label"):
                sync_body["install_label"] = identity["install_label"]
        resp = client._http.post("/api/v1/skills/sync", json=sync_body)
        if resp.status_code == 422:
            # Surface the backend's per-field validation errors — "422" alone
            # hides exactly the message that tells the user what to rename
            # (e.g. the agentskills.io name rules).
            click.echo("  ✗ Sync rejected by validation:", err=True)
            try:
                details = ((resp.json() or {}).get("details") or {}).get("errors") or []
            except ValueError:
                details = []
            for err in details or [{"field": "", "message": resp.text[:300]}]:
                field = err.get("field", "")
                click.echo(f"      {field}: {err.get('message', '')}", err=True)
            raise SystemExit(1)
        resp.raise_for_status()
        result = resp.json() or {}
    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"  ✗ Sync failed: {e}", err=True)
        raise SystemExit(1)
    finally:
        client.close()

    actions = result.get("actions") or []
    by_action: dict[str, list[dict]] = {
        "created": [], "pushed": [], "pulled": [], "no_change": [], "failed": [],
    }
    for a in actions:
        by_action.setdefault(a.get("action", "failed"), []).append(a)

    click.echo("")
    click.echo(f"  ✓ created     {len(by_action['created'])}")
    click.echo(f"  ✓ no_change   {len(by_action['no_change'])}")
    click.echo(f"  ✓ pushed      {len(by_action['pushed'])}")
    click.echo(f"  ✓ pulled      {len(by_action['pulled'])}")
    if by_action["failed"]:
        click.echo(f"  ✗ failed      {len(by_action['failed'])}")

    for item in by_action["created"]:
        click.echo(f"    + {item.get('name')} (v{item.get('new_version_number')})")
    for item in by_action["pushed"]:
        click.echo(f"    ↑ {item.get('name')} → v{item.get('new_version_number')}")
    for item in by_action["failed"]:
        click.echo(f"    ✗ {item.get('name')}: {item.get('error')}", err=True)

    pulled = by_action["pulled"]
    if pulled and apply_pulls:
        click.echo("")
        click.echo("  Writing pulled results to disk…")
        for item in pulled:
            name = item.get("name")
            body = item.get("body_markdown") or ""
            target = discovered_paths.get(name)
            if not target:
                # Backend pulled a skill we don't have locally; create
                # <skills_dir>/<name>/SKILL.md.
                #
                # `name` is SERVER-supplied, so it goes through the same guard the
                # pull path uses (see `_safe_skill_dirname` below, and the
                # path-traversal rationale in disk_export.py). Without it this line wrote
                # `base / name / SKILL.md` unchecked — a traversal or a namespaced
                # `owner/skill` name silently nested a directory outside the
                # intended layout. Skip the item rather than abort the whole sync;
                # one bad row must not strand the other pulled results.
                from ..disk_export import (
                    SkillExportUnsafePathError,
                    _safe_skill_dirname,
                )

                try:
                    # url_slug first — a namespaced name is a legitimate identity
                    # that simply cannot be a directory component (see the pull
                    # path and disk_export). Falls back to `name` for older
                    # backends, where it behaves exactly as before.
                    safe_name = _safe_skill_dirname(
                        item.get("url_slug") or name or ""
                    )
                except SkillExportUnsafePathError as e:
                    click.echo(f"    ✗ {name}: {e}", err=True)
                    continue
                target = base / safe_name / "SKILL.md"
                target.parent.mkdir(parents=True, exist_ok=True)
            # Preserve existing frontmatter if any; otherwise write body only.
            existing_fm = ""
            if target.exists():
                try:
                    existing = target.read_text(encoding="utf-8")
                    if existing.startswith("---"):
                        end = existing.find("---", 3)
                        if end != -1:
                            existing_fm = existing[: end + 3] + "\n\n"
                except OSError:
                    existing_fm = ""
            target.write_text(existing_fm + body, encoding="utf-8")
            click.echo(f"    ↓ {name} → {target} (v{item.get('version_number')})")
    elif pulled:
        click.echo("  (--no-apply-pulls set; skipping disk writes)")
        for item in pulled:
            click.echo(f"    ↓ {item.get('name')} (v{item.get('version_number')})")


# Drift status → display glyph and a stable display order (drift first so it
# can't hide below a wall of in_sync rows).
_STATUS_GLYPH = {
    "in_sync": "✓",
    "local_ahead": "↑",
    "remote_ahead": "↓",
    "conflict": "⚠",
    "local_only": "+",
    "unknown": "?",
}
_STATUS_ORDER = {
    "conflict": 0,
    "local_ahead": 1,
    "remote_ahead": 2,
    "local_only": 3,
    "unknown": 4,
    "in_sync": 5,
}


@skills.command("status")
@click.argument("skills_dir", type=click.Path(exists=True, file_okay=False), default="./skills")
@common_options
def skills_status(skills_dir, api_key, base_url, project):
    """Show per-install sync divergence for local SKILL.md files.

    Hashes each local skill (SHA-256 of body_markdown) and asks the platform
    how it compares to *this install's* last-synced baseline, reporting a
    status per skill:

      \b
      in_sync       local matches the platform
      local_ahead   you edited locally — `decimalai skills sync` to push
      remote_ahead  the platform moved — `decimalai skills sync` to pull
      conflict      both moved — reconcile by hand, then sync
      local_only    not on the platform yet (sync to create it)
      unknown       no baseline recorded — run `decimalai skills sync` once

    Read-only on skill content: it reports local hashes without pushing any
    bodies. The install identity comes from `.decimal/install.json`.

    \b
    Examples:
        $ decimalai skills status                # uses ./skills
        $ decimalai skills status ./agents/skills
    """
    import hashlib
    from pathlib import Path

    from ..skills import _split_frontmatter

    base = Path(skills_dir).resolve()

    # Same walk + body hash as `skills sync`, so a freshly-synced skill reads
    # as in_sync rather than spuriously diverged.
    items: list[dict] = []
    for skill_md in base.rglob("SKILL.md"):
        try:
            content = skill_md.read_text(encoding="utf-8")
        except OSError as e:
            click.echo(f"  ⚠ skipping {skill_md} ({e})", err=True)
            continue
        _frontmatter, body = _split_frontmatter(content)
        if not body or not body.strip():
            continue
        name = (_frontmatter.get("name") or "").strip() or skill_md.parent.name
        if not name:
            continue
        items.append({
            "name": name,
            "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        })

    if not items:
        click.echo(f"  No SKILL.md files found under {base}")
        return

    from .._install import get_install_identity

    identity = get_install_identity()
    install_id = identity.get("install_id")
    if not install_id:
        click.echo("  ✗ Could not determine install id", err=True)
        raise SystemExit(1)

    client = _make_client(api_key, base_url, project)
    try:
        report_body = {"install_id": install_id, "skills": items}
        if identity.get("install_label"):
            report_body["install_label"] = identity["install_label"]
        resp = client._http.post("/api/v1/skills/installs/report", json=report_body)
        resp.raise_for_status()
        result = resp.json() or {}
    except Exception as e:
        click.echo(f"  ✗ Status check failed: {e}", err=True)
        raise SystemExit(1)
    finally:
        client.close()

    reported = result.get("skills") or []
    reported.sort(key=lambda s: (_STATUS_ORDER.get(s.get("status"), 9), s.get("name", "")))

    counts: dict[str, int] = {}
    click.echo("")
    click.echo(f"  Install: {identity.get('install_label') or install_id[:8]}")
    click.echo("")
    for s in reported:
        st = s.get("status", "unknown")
        counts[st] = counts.get(st, 0) + 1
        glyph = _STATUS_GLYPH.get(st, "·")
        click.echo(f"    {glyph} {s.get('name', '?'):<40s}  {st}")

    click.echo("")
    click.echo("  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    drift = sum(counts.get(k, 0) for k in ("local_ahead", "remote_ahead", "conflict", "local_only"))
    if drift:
        click.echo("")
        click.echo("  → Run `decimalai skills sync` to reconcile.")


@skills.command("scan")
@click.argument("skills_dir", type=click.Path(exists=True), default="./skills")
@click.option("--format", "fmt", type=click.Choice(["text", "json", "github", "sarif"]), default="text",
              help="text (default), json, github (::annotations for CI), or sarif")
@click.option("--fail-on", type=click.Choice(["blocked", "flagged", "never"]), default="blocked",
              help="Exit non-zero when any skill reaches this status (default: blocked)")
def skills_scan(skills_dir, fmt, fail_on):
    """Static safety scan of local SKILL.md files — free, no LLM, no network, no API key.

    Runs the SAME deterministic Tier-1 scanner the registry publish gate runs (shipped in
    `skillevaluation.safety`), so you catch findings before you sync or publish. Nothing
    leaves your machine. Because it reads only frontmatter, results can be STRICTER than
    the server (a security skill without `category: security` may flag locally yet pass the
    gate) — advisory, never looser. Exit 1 when any skill reaches --fail-on.

    \b
    Examples:
        $ decimalai skills scan                      # scans ./skills
        $ decimalai skills scan ./myskill/SKILL.md
        $ decimalai skills scan --format github --fail-on blocked   # in CI
    """
    import json as _json
    import sys
    from pathlib import Path

    from ..skills import _split_frontmatter

    try:
        from skillevaluation import safety as _safety
    except ImportError:
        raise click.ClickException(
            "skills scan needs skillevaluation>=0.4.0 (the safety module). "
            "Upgrade with: pip install -U 'skillevaluation>=0.4.0'"
        )

    p = Path(skills_dir).resolve()
    if p.is_file():
        files = [p]
    elif (p / "SKILL.md").exists():
        files = [p / "SKILL.md"]
    else:
        files = sorted(p.rglob("SKILL.md"))
    files = [f for f in files if f.exists()]
    if not files:
        raise click.ClickException(f"no SKILL.md found under {skills_dir}")

    rank = {"clean": 0, "flagged": 1, "blocked": 2}
    worst = 0
    per: list[tuple] = []
    for f in files:
        fm, body = _split_frontmatter(f.read_text(encoding="utf-8"))
        fm = fm or {}
        name = str(fm.get("name") or f.parent.name)
        res = _safety.scan_skill_content(
            body,
            name=name,
            description=str(fm.get("description") or ""),
            category=fm.get("category"),
            allowed_tools=fm.get("allowed-tools") or fm.get("allowed_tools"),
            trigger_phrases=fm.get("trigger_phrases") or fm.get("triggers"),
        )
        per.append((f, name, res))
        worst = max(worst, rank.get(res["status"], 0))

    if fmt == "json":
        click.echo(_json.dumps({"skills": [{"file": str(f), "name": n, **r} for f, n, r in per]}, indent=2))
    elif fmt == "sarif":
        runs: list = []
        for f, n, r in per:
            runs.extend(_safety.to_sarif(r, skill_name=n, file_path=str(f))["runs"])
        click.echo(_json.dumps({"version": "2.1.0", "$schema": "https://json.schemastore.org/sarif-2.1.0.json", "runs": runs}, indent=2))
    elif fmt == "github":
        lvl = {"critical": "error", "warning": "warning", "info": "notice"}
        for f, n, r in per:
            for fnd in r.get("findings", []):
                msg = fnd["message"].replace("\n", " ")
                click.echo(f"::{lvl.get(fnd['severity'], 'notice')} file={f},line={fnd.get('line') or 1},"
                           f"title=SkillSafety {fnd['check']}::{msg}")
    else:
        icons = {"clean": "✓", "flagged": "⚠", "blocked": "⛔"}
        for f, n, r in per:
            click.echo(f"  {icons.get(r['status'], '?')} {n}: {r['status']} — {r['summary']}")
            for fnd in r.get("findings", []):
                loc = f":{fnd['line']}" if fnd.get("line") else ""
                click.echo(f"      [{fnd['severity']}] {fnd['check']}{loc} — {fnd['message']}")
                if fnd.get("remediation"):
                    click.echo(f"        fix: {fnd['remediation']}")

    if worst >= {"blocked": 2, "flagged": 1, "never": 99}[fail_on]:
        sys.exit(1)


@skills.command("review")
@click.argument("skill_name")
@common_options
def skills_review(skill_name, api_key, base_url, project):
    """Run a SkillSafety DEEP review (LLM Tier-2 intent + Tier-3 content) on a skill.

    Unlike `skills scan` (free, local, Tier-1 only), this runs the LLM tiers server-side
    to catch an intent/content rejection BEFORE you publish — metered against your
    plan's deep-review quota. Sync the skill first (`decimalai skills sync`), then:

        $ decimalai skills review my-skill
    """
    import sys

    client = _make_client(api_key, base_url, project)
    try:
        resp = client._http.post(f"/api/v1/skills/{skill_name}/checks/run?deep=true")
        resp.raise_for_status()
        r = resp.json() or {}
    except Exception as e:
        click.echo(f"  ✗ Deep review failed: {e}", err=True)
        raise SystemExit(1)
    finally:
        client.close()

    band = r.get("skill_safety", "?")
    icon = {"passed": "✓", "caution": "⚠", "blocked": "⛔", "unreviewed": "○"}.get(band, "?")
    click.echo(f"  {icon} {skill_name}: SkillSafety {band}")
    click.echo(f"      safety={r.get('safety_status')}  intent={r.get('intent_status')}  content={r.get('content_status')}")
    for fnd in r.get("findings", []):
        loc = f":{fnd['line']}" if fnd.get("line") else ""
        click.echo(f"      [{fnd['severity']}] {fnd['check']}{loc} — {fnd['message']}")
    iv = r.get("intent_review") or {}
    if iv.get("summary"):
        click.echo(f"      intent review: {iv['summary']}")
    if band == "blocked":
        sys.exit(1)


# The anonymous pull path builds no client and reuses no connection, and
# a cold prod instance has served first requests in 9-28s — a flat 20s
# timeout killed pulls the server went on to complete. Budget 30s per
# request and retry once on timeout: the attempt that timed out warms
# the instance the retry then hits.
_PULL_HTTP_TIMEOUT = 30.0


def _pull_get(url, **kwargs):
    """GET for the anonymous pull path — one retry on timeout."""
    import httpx

    try:
        return httpx.get(url, timeout=_PULL_HTTP_TIMEOUT, **kwargs)
    except httpx.TimeoutException:
        return httpx.get(url, timeout=_PULL_HTTP_TIMEOUT, **kwargs)


@skills.command("pull")
@click.argument("slug")
@click.option(
    "--out", "out_dir", default=None,
    help="Write SKILL.md to OUT/<slug>/SKILL.md. Default: current directory.",
)
@click.option(
    "--base-url",
    envvar=["DECIMAL_BASE_URL", "DECIMALAI_BASE_URL"],
    default="https://api.decimal.ai",
    show_envvar=True,
    help="Platform URL",
)
@click.option(
    "--no-evals", "no_evals", is_flag=True, default=False,
    help="Skip pulling the eval.yaml test suite (default: pull it).",
)
@click.option(
    "--stdout", "to_stdout", is_flag=True, default=False,
    help="Print body markdown to stdout instead of writing to disk.",
)
def skills_pull(slug, out_dir, base_url, no_evals, to_stdout):
    """Pull a public registry skill to disk — no signup required.

    Fetches the latest version of a public skill from the registry and writes
    SKILL.md (plus its bundled files and eval.yaml) to disk (or stdout).
    Read-only: no fork is created, no activations are tracked. Use
    `decimalai skills install` to fork + sync (which requires an API key).

    \b
    Examples:
        # Write ./playwright-cli/SKILL.md
        $ decimalai skills pull playwright-cli
    \b
        # Write to a specific path
        $ decimalai skills pull pdf --out ./agents/skills/
    \b
        # Just print the body
        $ decimalai skills pull pdf --stdout
    """
    import os

    import httpx

    # Public endpoints — no auth header, no api_key required.
    url = f"{base_url.rstrip('/')}/api/v1/registry/skills"
    from .._registry_resolve import RESOLVE_LIMIT, find_exact, not_found_message

    try:
        # `q=` is a semantic search that always ranks *something*; resolve by
        # exact name or fail loudly. Taking items[0] used to write an unrelated
        # skill to disk on any typo and report it as a success.
        search = _pull_get(url, params={"q": slug, "limit": RESOLVE_LIMIT})
        search.raise_for_status()
        items = (search.json() or {}).get("items") or []
        match = find_exact(items, slug)
        if match is None:
            click.echo(f"  ✗ {not_found_message(slug, items)}", err=True)
            raise SystemExit(1)
        skill_id = match["id"]
        # The registry's own spelling — `slug` may differ only in case.
        slug = match.get("name") or slug

        detail_resp = _pull_get(f"{url}/{skill_id}")
        detail_resp.raise_for_status()
        detail = detail_resp.json()
    except httpx.HTTPError as e:
        click.echo(f"  ✗ Registry lookup failed: {e}", err=True)
        raise SystemExit(1)

    body = detail.get("body_markdown") or ""
    # The skill name is server-controlled; reject a
    # traversal/absolute name before it reaches os.path.join (arbitrary file write).
    from ..disk_export import SkillExportUnsafePathError, _safe_skill_dirname
    try:
        # `url_slug`, not `name`. A registry name may be namespaced
        # (`owner/skill`) and a slash cannot be a single path component.
        # `url_slug` is the slash-free identifier minted for this; falling back
        # to `name` keeps older backends working, and for a plain name the two
        # are identical.
        name = _safe_skill_dirname(detail.get("url_slug") or detail.get("name") or slug)
    except SkillExportUnsafePathError as e:
        click.echo(f"  ✗ Refusing to pull: {e}", err=True)
        raise SystemExit(1)

    if to_stdout:
        click.echo(body, nl=False)
        return

    # Registry bodies don't carry frontmatter (the loader strips it on
    # import) — write a parseable SKILL.md, or discover_skills()/the
    # skillevaluation runner silently skip the file we just pulled.
    # Bodies that already have frontmatter (bring-your-own round-trips)
    # pass through verbatim.
    if not body.startswith("---"):
        from ..disk_export import _reconstruct_skill_md
        body = _reconstruct_skill_md(
            name, detail.get("description") or name, body,
        )

    target_dir = os.path.join(out_dir or os.getcwd(), name)
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, "SKILL.md")
    with open(target, "w", encoding="utf-8") as f:
        f.write(body)

    eff = detail.get("effectiveness") or {}
    bench = detail.get("benchmark_summary") or {}
    click.echo(f"  ✓ Pulled {name} v{detail.get('latest_version_number', '?')}")
    click.echo(f"    → {target}")

    # SkillScore v2 quality line, e.g.:
    #   SkillScore 87 (provisional) · +60 pts vs no skill · 96% live pass
    # SkillScore is the quality composite (benchmark pass / live eval
    # pass / AI-rated trace quality); "provisional" = one evidence leg.
    score = eff.get("skill_score")
    if score is None:
        score = eff.get("avg_effectiveness")  # frozen alias (older backends)
    quality_parts = []
    if score is not None:
        components = eff.get("score_components") or {}
        early = " (provisional)" if components.get("provisional") else ""
        quality_parts.append(f"SkillScore {score * 100:.0f}{early}")
    lift = bench.get("pass_rate_delta_pts")
    if lift is not None:
        quality_parts.append(f"{'+' if lift >= 0 else ''}{lift:.0f} pts vs no skill")
    live_pass = eff.get("avg_pass_rate")
    if live_pass is not None:
        quality_parts.append(f"{live_pass * 100:.0f}% live pass")
    if quality_parts:
        click.echo(f"    {' · '.join(quality_parts)}")

    # Bundled attachments: pulled bodies reference
    # scripts/* and references/* the old pull never delivered. The registry
    # exposes them publicly — GET /registry/skills/{id}/attachments (metadata)
    # then /attachments/{att_id} (content) — so deliver them next to SKILL.md.
    # Best-effort: a failed file warns and continues; the summary line names
    # anything the pull could not deliver.
    from ..disk_export import _safe_join_within

    attachment_count = detail.get("attachment_count") or 0
    written_attachments = 0
    attachments = []
    if attachment_count:  # skip the round-trip for the bundle-less majority
        try:
            atts_resp = _pull_get(f"{url}/{skill_id}/attachments")
            atts_resp.raise_for_status()
            attachments = (atts_resp.json() or {}).get("attachments") or []
        except httpx.HTTPError as e:
            click.echo(f"  ⚠ bundled-file list fetch failed: {e} — skipping", err=True)
    for att in attachments:
        file_path = att.get("file_path") or ""
        att_id = att.get("id") or ""
        if not file_path:
            continue
        content = att.get("content_text") or ""
        if not content and att_id:
            try:
                full_resp = _pull_get(f"{url}/{skill_id}/attachments/{att_id}")
                full_resp.raise_for_status()
                content = (full_resp.json() or {}).get("content_text") or ""
            except httpx.HTTPError as e:
                click.echo(f"  ⚠ bundled file {file_path} fetch failed: {e} — skipping", err=True)
                continue
        if not content:
            continue
        try:
            att_target = _safe_join_within(target_dir, file_path)
        except SkillExportUnsafePathError as e:
            # Server-supplied path — same traversal guard as the installer.
            click.echo(f"  ⚠ skipping bundled file: {e}", err=True)
            continue
        os.makedirs(os.path.dirname(att_target), exist_ok=True)
        with open(att_target, "w", encoding="utf-8") as f:
            f.write(content)
        written_attachments += 1
    if written_attachments:
        click.echo(f"    → {written_attachments} bundled file(s) (scripts/references/…)")
    if attachment_count > written_attachments:
        missing = attachment_count - written_attachments
        click.echo(
            f"    note: this skill references {missing} bundled file(s) not included in pull"
        )

    # Also write eval.yaml from the dedicated registry endpoint.
    # 404 means the skill has no publisher/community-authored eval.yaml
    # (typical for GitHub imports). --no-evals skips the fetch entirely.
    if not no_evals:
        try:
            eval_resp = _pull_get(f"{url}/{skill_id}/eval")
            if eval_resp.status_code == 200:
                eval_payload = eval_resp.json()
                eval_yaml = eval_payload.get("eval_yaml_text") or ""
                if eval_yaml:
                    eval_path = os.path.join(target_dir, "eval.yaml")
                    with open(eval_path, "w", encoding="utf-8") as f:
                        f.write(eval_yaml)
                    case_count = eval_payload.get("case_count", 0)
                    click.echo(f"    → {eval_path} ({case_count} test cases)")
            elif eval_resp.status_code == 404:
                # No eval.yaml authored yet — quiet skip, the SKILL.md is
                # still useful on its own.
                pass
            else:
                click.echo(
                    f"  ⚠ eval.yaml fetch returned HTTP {eval_resp.status_code} — skipping",
                    err=True,
                )
        except httpx.HTTPError as e:
            click.echo(f"  ⚠ eval.yaml fetch failed: {e} — skipping", err=True)

    # Efficiency headline — the cost story (tokens primary). The
    # pass-rate lift moved up into the SkillScore quality line, so this
    # block no longer repeats it.
    parts = []
    if bench.get("tokens_delta_pct") is not None:
        sign = "+" if bench["tokens_delta_pct"] >= 0 else ""
        parts.append(f"{sign}{bench['tokens_delta_pct']:.0f}% tokens")
    if bench.get("turns_delta_pct") is not None:
        sign = "+" if bench["turns_delta_pct"] >= 0 else ""
        parts.append(f"{sign}{bench['turns_delta_pct']:.0f}% turns")
    if parts:
        click.echo(f"\n  Efficiency vs no skill: {' · '.join(parts)}")

    click.echo("")
    click.echo("    Next — fork it into your workspace (adds telemetry + versioning):")
    click.echo(f"      $ decimalai skills install {slug}        # with an API key")
    click.echo(f"      {_dashboard_url(base_url)}/skills/{slug}   # or in the dashboard")


@skills.command("export")
@click.argument("slug")
@click.option(
    "--agent", "agents", multiple=True,
    help="Agent runtime to write SKILL.md for (repeatable): claude-code, cursor, "
         "github-copilot, … Default: universal (.agents/skills).",
)
@click.option(
    "--scope", type=click.Choice(["project", "global"]), default="project",
    show_default=True,
    help="project → .claude/skills, .agents/skills (cwd); global → ~/… dirs.",
)
@click.option(
    "--out", "project_root", default=None,
    help="Project root for --scope=project (default: current directory).",
)
@common_options
def skills_export(slug, agents, scope, project_root, api_key, base_url, project):
    """Write a skill to disk for file-loading runtimes. No copy is taken.

    Export is the FILE half of adoption, on its own. It writes SKILL.md (+
    scripts) into each agent runtime's skills dir so Claude Code / Cursor / …
    load it — and that is all it does. Nothing about writing a file requires
    owning the skill.

    Works on anything in your workspace: a skill you own, or one you installed
    (linked) from the registry. `decimalai skills install` used to be this
    command, but it forked first, so asking for a file also took an editable
    copy you never asked for.

    \b
    Examples:
        $ decimalai skills export pdf
        $ decimalai skills export pdf --agent claude-code --agent cursor
        $ decimalai skills export playwright-cli --scope global

    For a public skill you have NOT added to your workspace — no key, no
    account — use `decimalai skills pull`.
    """
    from ..skill_router import SkillRouter

    if not api_key:
        click.echo("  ✗ Export reads the skill from your workspace, so an API key is required.", err=True)
        click.echo('    Set it:   export DECIMAL_API_KEY="dai_sk_..."', err=True)
        click.echo(f"    Public skill you have not added yet:   decimalai skills pull {slug}", err=True)
        raise SystemExit(1)

    router = SkillRouter(api_key=api_key, base_url=base_url or "https://api.decimal.ai")
    try:
        summary = router.export(
            slug, agents=list(agents) or None, scope=scope, project_root=project_root,
        )
    except Exception as e:  # noqa: BLE001 — surfaced to the user, not swallowed
        click.echo(f"  ✗ {e}", err=True)
        raise SystemExit(1)

    written = summary.get("skills_written", 0)
    if not written:
        click.echo("  ✗ Nothing was written — is the skill in your workspace?", err=True)
        raise SystemExit(1)
    click.echo(f"  ✓ Wrote {slug} to disk")
    for p_ in (summary.get("paths") or [])[:8]:
        click.echo(f"    → {p_}")
    for err in summary.get("errors") or []:
        click.echo(f"    ⚠ {err.get('skill')}: {err.get('error')}", err=True)


@skills.command("install")
@click.argument("slug")
@click.option(
    "--agent", "agents", multiple=True,
    help="Agent runtime to write SKILL.md for (repeatable): claude-code, cursor, "
         "github-copilot, … Default: universal (.agents/skills).",
)
@click.option(
    "--scope", type=click.Choice(["project", "global"]), default="project",
    show_default=True,
    help="project → .claude/skills, .agents/skills (cwd); global → ~/… dirs.",
)
@click.option(
    "--out", "project_root", default=None,
    help="Project root for --scope=project (default: current directory).",
)
@common_options
def skills_install(slug, agents, scope, project_root, api_key, base_url, project):
    """Fork a registry skill into your workspace AND write it to disk.

    DEPRECATED — the two halves are separate commands now:
      `decimalai skills export <slug>`  writes the files (no copy)
      the Install button / `router.use()` links the skill (no copy)
      `router.fork()` takes the editable copy
    This still works and still forks; it is just no longer the shape to reach for.

    The full on-ramp (needs an API key). Two steps in one:

      \b
      1. Fork  — copy the registry skill into your workspace as a skill you
                 own, tracked + versioned (POST /registry/skills/{id}/install).
      2. Install to disk — write SKILL.md (+ scripts) into each agent
                 runtime's skills dir so Claude Code / Cursor / … load it.

    For just the file — no fork, no signup — use `decimalai skills pull`.

    \b
    Examples:
        $ decimalai skills install pdf
        $ decimalai skills install pdf --agent claude-code --agent cursor
        $ decimalai skills install playwright-cli --scope global
    """
    from ..skill_router import SkillRouter

    if not api_key:
        click.echo("  ✗ Installing forks the skill into your workspace, so an API key is required.", err=True)
        click.echo('    Set it:   export DECIMAL_API_KEY="dai_sk_..."', err=True)
        click.echo(f"    Just the file instead (no fork):   decimalai skills pull {slug}", err=True)
        raise SystemExit(1)

    router = SkillRouter(api_key=api_key, base_url=base_url or "https://api.decimal.ai")
    agent_list = list(agents) or None

    def _echo_paths(export):
        for p in (export.get("paths") or [])[:8]:
            click.echo(f"    → {p}")

    try:
        result = router.install(
            slug, agents=agent_list, scope=scope, project_root=project_root,
        )
    except Exception as e:  # SkillRouter raises RuntimeError / ValueError
        msg = str(e)
        if "409" in msg or "lready installed" in msg:
            # Already forked — not an error; just (re)write the files to disk.
            try:
                export = router.export_to_disk(
                    skills=[slug], agents=agent_list, scope=scope, project_root=project_root,
                )
            except Exception as e2:
                click.echo(f"  ✗ Already in your workspace, but writing to disk failed: {e2}", err=True)
                raise SystemExit(1)
            click.echo(f"  ✓ '{slug}' already in your workspace — refreshed {export.get('skills_written', 0)} file(s) on disk")
            _echo_paths(export)
            return
        click.echo(f"  ✗ Install failed: {e}", err=True)
        raise SystemExit(1)

    name = result.get("skill_name", slug)
    export = result.get("export") or {}
    written = export.get("skills_written", 0)
    click.echo(f"  ✓ Forked '{name}' into your workspace (tracked + versioned)")
    if written:
        click.echo(f"  ✓ Wrote SKILL.md to disk ({written} file(s))")
        _echo_paths(export)
    else:
        click.echo("  • No disk files written (pass --agent claude-code to materialize on disk).")
    click.echo("")


@skills.command("benchmark")
@click.argument("skill_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--model",
    default=None,
    metavar="MODEL",
    help=(
        "Benchmark against this model for the model-relative lane "
        "(e.g. gpt-4o-mini, claude-haiku-4-5). Lift is skill x model, so the "
        "result is tagged with the model and the registry headline is the best "
        "across the models you've run. Must be in the platform's supported "
        "models. Default: the platform's configured model."
    ),
)
@click.option(
    "--runs",
    default=None,
    type=int,
    metavar="N",
    help=(
        "Re-run the WHOLE suite N times, uniformly (1-10; default 1). Every "
        "(case, run) record enters the aggregate at equal weight, so the "
        "headline pass-rate is a MEAN over runs whose expected value does not "
        "depend on N — more runs only narrow the error bars. This is a "
        "run-level parameter on the hosted run endpoint; it does NOT touch "
        "your eval.yaml."
    ),
)
@click.option(
    "--trials",
    default=None,
    type=int,
    metavar="N",
    hidden=True,  # removed (ADR-0007) — kept only to emit a clear redirect to --runs
    help="Removed — use --runs N (run-level repetition, mean-averaged).",
)
@common_options
def skills_benchmark(skill_dir, api_key, base_url, project, model, runs, trials):
    """Run the A/B benchmark suite for a skill defined in ``skill_dir``.

    This is the HOSTED, **verified** path: the platform runner executes
    every case (metered against your plan's monthly case quota) and the
    result feeds the skill's registry numbers. For free unlimited local
    iteration on your own API key, use the open-source runner instead —
    ``pip install "skillevaluation[runner]"`` then ``skillevaluation run
    ./skills/<name>`` — and ``decimalai skills push results.json`` to
    attach the (unverified) result to the skill.

    ``skill_dir`` must contain a ``SKILL.md`` file. If an ``eval.yaml`` is
    present next to it, the suite is uploaded first via the sync endpoint
    so the platform has the latest test cases. Then the benchmark runs
    each case twice (with skill / without skill) and prints the deltas.

    ``--runs N`` re-runs the whole suite N times, uniformly, and averages
    the per-case results by MEAN (ADR-0007). The expected value of the
    headline is independent of N — more runs only narrow the error bars.
    It is a run-level parameter on the hosted run endpoint and does NOT
    modify your eval.yaml. (It replaces the removed per-case pass^k
    ``--trials`` flag.)

    \b
    Example:
        $ decimalai skills benchmark ./skills/gdpr-pii-classifier
        $ decimalai skills benchmark ./skills/gdpr-pii-classifier --model gpt-4o-mini
        $ decimalai skills benchmark ./skills/gdpr-pii-classifier --runs 3
    """
    import hashlib
    from pathlib import Path

    # --trials was removed (ADR-0007: pass^k retired for run-level MEAN). Fail
    # loud with the exact replacement rather than silently ignoring it — a stale
    # script that still passes --trials must not print numbers measured at a
    # different repetition count than it asked for.
    if trials is not None:
        click.echo(
            f"  ✗ --trials has been removed. Use --runs {trials} instead: it "
            "re-runs the whole suite N times and averages by MEAN (the old "
            "per-case pass^k reliability knob is retired — see ADR-0007).",
            err=True,
        )
        raise SystemExit(1)

    base = Path(skill_dir).resolve()
    skill_md = base / "SKILL.md"
    if not skill_md.exists():
        click.echo(f"  ✗ {skill_md} not found", err=True)
        raise SystemExit(1)

    # --runs sanity, before any network work. 10 mirrors the platform runner's
    # cap (_BENCHMARK_MAX_RUNS) — a bigger value is clamped server-side, so
    # reject it here for a clear error instead of a silent clamp.
    if runs is not None and (runs < 1 or runs > 10):
        click.echo(
            f"  ✗ --runs must be between 1 and 10 (got {runs}) — the hosted "
            "runner caps suite repetitions at 10.",
            err=True,
        )
        raise SystemExit(1)

    # Skill name preferably from frontmatter; fall back to dir name.
    from ..skills import _split_frontmatter
    content = skill_md.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(content)
    name = (frontmatter.get("name") or "").strip() or base.name
    if not body or not body.strip():
        click.echo(f"  ✗ {skill_md} has an empty body", err=True)
        raise SystemExit(1)

    client = _make_client(api_key, base_url, project)
    try:
        # 1. Sync this skill body + (optionally) eval.yaml so the backend
        #    has the latest test cases before running.
        body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        payload = {
            "skills": [
                {
                    "name": name,
                    "content_hash": body_hash,
                    "body_markdown": body,
                    "description": frontmatter.get("description") or name,
                    "category": frontmatter.get("category"),
                }
            ],
            "conflict_policy": "local_wins",  # benchmark always pushes local
        }
        eval_yaml_path = base / "eval.yaml"
        if eval_yaml_path.exists():
            eval_text = eval_yaml_path.read_text(encoding="utf-8")

            # Validate locally against the open skillevaluation spec BEFORE uploading,
            # so authors get actionable error messages without a server round-trip.
            try:
                from skillevaluation.parser import EvalYamlParseError, parse_eval_yaml
            except ImportError:
                # Stale install without the skillevaluation dependency — say so
                # rather than surfacing a bare ModuleNotFoundError.
                raise SystemExit(
                    "skills benchmark needs the `skillevaluation` package to validate "
                    f"{eval_yaml_path} — your decimalai install predates it. "
                    "Fix: pip install -U decimalai"
                )
            try:
                cases = parse_eval_yaml(eval_text)
                click.echo(f"  ✓ eval.yaml validated: {len(cases)} case(s)")
            except EvalYamlParseError as exc:
                click.echo(f"  ✗ {eval_yaml_path}: {exc}", err=True)
                raise SystemExit(1)

            payload["skills"][0]["eval_yaml_text"] = eval_text
            payload["skills"][0]["eval_yaml_hash"] = hashlib.sha256(
                eval_text.encode("utf-8")
            ).hexdigest()

        click.echo(f"  Syncing {name}…")
        try:
            sync_resp = client._http.post("/api/v1/skills/sync", json=payload)
            sync_resp.raise_for_status()
        except Exception as e:
            click.echo(f"  ⚠ sync failed (continuing with existing remote version): {e}")

        # 2. Trigger the benchmark. --runs is a run-level query param on the
        #    hosted endpoint (it does not depend on the synced eval.yaml).
        click.echo(f"  Running benchmark for {name}{f' on {model}' if model else ''}…")
        try:
            run_resp = client._http.post(
                f"/api/v1/skills/{name}/benchmark/run",
                params={
                    "triggered_by": "cli",
                    **({"model": model} if model else {}),
                    **({"runs": runs} if runs is not None else {}),
                },
            )
            if run_resp.status_code == 429:
                detail = {}
                try:
                    detail = (run_resp.json() or {}).get("detail") or {}
                except ValueError:
                    pass
                feature = detail.get("feature", "benchmark quota")
                click.echo(
                    f"  ✗ {feature} limit reached "
                    f"({detail.get('used', '?')}/{detail.get('limit', '?')} on the "
                    f"{detail.get('plan', '?')} plan).",
                    err=True,
                )
                click.echo(
                    "    Iterate free + locally instead: pip install 'skillevaluation[runner]'\n"
                    "    then: skillevaluation run <skill-dir> --model <your-model>\n"
                    f"    Upgrade for more verified cases: {detail.get('upgrade_url', '')}",
                    err=True,
                )
                raise SystemExit(1)
            run_resp.raise_for_status()
            run = run_resp.json()
        except SystemExit:
            raise
        except Exception as e:
            click.echo(f"  ✗ Benchmark failed: {e}", err=True)
            raise SystemExit(1)

        # 3. Print results table.
        m = run.get("aggregate_metrics") or {}
        click.echo("")
        click.echo(
            f"  ✓ {run['passed_cases']}/{run['total_cases']} passed "
            f"· verdict: {run['overall_verdict']}"
        )
        pr = m.get("pass_rate") or {}
        if pr:
            sign = "+" if pr.get("delta_pts", 0) >= 0 else ""
            click.echo(
                f"    Pass rate:    "
                f"{pr.get('with_skill', 0) * 100:.0f}% (with) "
                f"vs {pr.get('without_skill', 0) * 100:.0f}% (without) "
                f"= {sign}{pr.get('delta_pts', 0):.0f} pts"
            )
        for key, label in [
            ("duration_ms", "Avg duration"),
            ("turns",       "Avg turns   "),
            ("tokens",      "Avg tokens  "),
        ]:
            d = m.get(key) or {}
            dp = d.get("delta_pct")
            if dp is None:
                continue
            sign = "+" if dp >= 0 else ""
            click.echo(
                f"    {label}: "
                f"{d.get('with_skill_avg', 0):.0f} vs {d.get('without_skill_avg', 0):.0f} "
                f"= {sign}{dp:.0f}%"
            )

        # 4. List failed cases for actionable output.
        fails = [r for r in run.get("results", []) if r.get("outcome") in ("flip_to_fail", "fail_kept", "error")]
        if fails:
            click.echo("")
            click.echo(f"  Failing cases ({len(fails)}):")
            for r in fails[:10]:
                click.echo(f"    ✗ {r.get('test_case_id', '?')[:12]}  outcome={r['outcome']}")

        click.echo("")
        click.echo(
            f"  Full report: {_dashboard_url(base_url)}"
            f"/skills/{name}?tab=benchmark"
        )
    finally:
        client.close()


@skills.command("push")
@click.argument("results_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--skill", "skill_name", default=None,
    help="Skill name to attach the run to (default: skill.name from the results file).",
)
@common_options
def skills_push(results_file, skill_name, api_key, base_url, project):
    """Upload a local ``skillevaluation run`` result as an UNVERIFIED run.

    RESULTS_FILE is the ``results.json`` the open-source runner wrote
    (``pip install "skillevaluation[runner]"`` → ``skillevaluation run
    ./my-skill``). Pushing is free and consumes no benchmark quota — it
    is a JSON upload, not an execution.

    The imported run shows on the skill's Benchmark tab tagged
    *unverified*; it never feeds registry rankings. For a verified,
    rankable result run ``decimalai skills benchmark`` (metered) or
    publish the skill (publish triggers a free verification run).

    \b
    Example:
        $ skillevaluation run ./skills/gdpr-pii-classifier --model claude-haiku-4-5
        $ decimalai skills push results.json
    """
    import json as _json

    try:
        payload = _json.loads(open(results_file, encoding="utf-8").read())
    except (OSError, ValueError) as e:
        click.echo(f"  ✗ could not read {results_file}: {e}", err=True)
        raise SystemExit(1)

    name = skill_name or ((payload.get("skill") or {}).get("name") if isinstance(payload, dict) else None)
    if not name:
        click.echo(
            "  ✗ no skill name — the results file has no skill.name; pass --skill <name>",
            err=True,
        )
        raise SystemExit(1)

    client = _make_client(api_key, base_url, project)
    try:
        resp = client._http.post(
            f"/api/v1/skills/{name}/benchmark/import", json=payload
        )
        if resp.status_code == 404:
            # The natural first-touch mistake: pushing results for a skill
            # the platform has never seen. Point at the missing step
            # instead of dumping a traceback.
            click.echo(
                f"  ✗ skill {name!r} doesn't exist on the platform yet.\n"
                f"    Sync it first (creates the skill + eval cases), then push:\n"
                f"      decimalai skills sync <skills-dir>\n"
                f"      decimalai skills push {results_file}",
                err=True,
            )
            raise SystemExit(1)
        if resp.status_code == 422:
            detail = (resp.json() or {}).get("detail", resp.text)
            click.echo(f"  ✗ results document rejected: {detail}", err=True)
            raise SystemExit(1)
        resp.raise_for_status()
        data = resp.json()

        click.echo(
            f"  ✓ pushed {data.get('imported_cases', 0)}/{data.get('total_cases', 0)} case(s) "
            f"· verdict: {data.get('overall_verdict')} · UNVERIFIED"
        )
        skipped = data.get("skipped_cases") or []
        if skipped:
            click.echo(
                f"  ⚠ {len(skipped)} pushed case(s) had no authored counterpart and were "
                f"skipped: {', '.join(skipped[:5])}"
            )
            click.echo("    (run `decimalai skills sync` so the platform has your latest eval.yaml)")
        click.echo("")
        click.echo("  Unverified runs never feed rankings. To get a verified result:")
        click.echo("    decimalai skills benchmark <skill-dir>   # metered")
        click.echo("    …or publish the skill (verification run included, free).")
        click.echo(
            f"  View: {_dashboard_url(base_url)}"
            f"/skills/{name}?tab=benchmark"
        )
    finally:
        client.close()


# ── Manifests commands ─────────────────────────────────────

@cli.group()
def manifests():
    """Manifest commands."""
    pass


@manifests.command("list")
@click.option("--agent-name", default=None, help="Filter by agent name")
@click.option("--limit", default=20, help="Max results")
@common_options
def manifests_list(agent_name, limit, api_key, base_url, project):
    """List manifests."""
    client = _make_client(api_key, base_url, project)
    try:
        result = client.list_manifests(limit=limit, agent_name=agent_name)
        manifests_data = result.get("manifests", [])
        click.echo(f"Found {len(manifests_data)} manifests:")
        for m in manifests_data:
            click.echo(
                f"  {m.get('id', '?')[:12]}  "
                f"{m.get('agent_name', '?'):20s}  "
                f"{m.get('version_label', '')}"
            )
    finally:
        client.close()


# ── Datasets commands ──────────────────────────────────────

@cli.group()
def datasets():
    """Dataset commands — list, pull, and export training data."""
    pass


@datasets.command("list")
@click.option("--limit", default=20, help="Max results")
@click.option("--agent-name", default=None, help="Filter by target agent")
@common_options
def datasets_list(limit, agent_name, api_key, base_url, project):
    """List datasets in the workspace."""
    client = _make_client(api_key, base_url, project)
    try:
        result = client.list_datasets(limit=limit)
        ds_list = result.get("datasets", [])
        click.echo(f"Found {len(ds_list)} datasets:")
        click.echo("")
        click.echo(f"  {'ID':14s}  {'Name':25s}  {'Type':8s}  {'Rows':>6s}  {'Versions':>8s}")
        click.echo(f"  {'─' * 14}  {'─' * 25}  {'─' * 8}  {'─' * 6}  {'─' * 8}")
        for d in ds_list:
            click.echo(
                f"  {d.get('id', '?')[:12]:14s}  "
                f"{d.get('name', '?'):25s}  "
                f"{d.get('dataset_type', '?'):8s}  "
                f"{d.get('row_count', 0):>6}  "
                f"{'v' + str(d.get('version_count', 0)):>8s}"
            )
    finally:
        client.close()


@datasets.command("show")
@click.argument("dataset_id")
@common_options
def datasets_show(dataset_id, api_key, base_url, project):
    """Show dataset detail with version history.

    \b
    Example:
        $ decimalai datasets show ds_abc123
    """
    client = _make_client(api_key, base_url, project)
    try:
        ds = client.get_dataset(dataset_id)
        click.echo("")
        click.echo(f"  Dataset: {ds.get('name', '?')}")
        click.echo(f"  ID:      {ds.get('id', '?')}")
        click.echo(f"  Type:    {ds.get('dataset_type', '?')}")
        click.echo(f"  Agent:   {ds.get('training_target_agent', '—')}")
        click.echo(f"  Current: {ds.get('current_version_id', '—')}")
        click.echo("")

        versions = ds.get("versions", [])
        if versions:
            click.echo(f"  Versions ({len(versions)}):")
            click.echo(f"  {'#':>4s}  {'ID':14s}  {'Status':10s}  {'Rows':>6s}  {'Created':20s}")
            click.echo(f"  {'─' * 4}  {'─' * 14}  {'─' * 10}  {'─' * 6}  {'─' * 20}")
            for v in sorted(versions, key=lambda x: x.get("version_number", 0), reverse=True):
                marker = " ←" if v.get("id") == ds.get("current_version_id") else ""
                click.echo(
                    f"  v{v.get('version_number', '?'):>3}  "
                    f"{v.get('id', '?')[:12]:14s}  "
                    f"{v.get('status', '?'):10s}  "
                    f"{v.get('row_count', 0):>6}  "
                    f"{v.get('created_at', '?'):20s}{marker}"
                )
        else:
            click.echo("  No versions yet. Run `decimalai datasets build` to create one.")
        click.echo("")
    finally:
        client.close()


@datasets.command("build")
@click.argument("dataset_id")
@click.option("--verdict", default=None, help="Filter traces by verdict (keep, repair)")
@common_options
def datasets_build(dataset_id, verdict, api_key, base_url, project):
    """Build a new dataset version from traces."""
    client = _make_client(api_key, base_url, project)
    try:
        filters = {}
        if verdict:
            filters["eval_verdict"] = verdict
        result = client.build_dataset(dataset_id, filters=filters or None)
        click.echo(
            f"Built version {result.get('version_id', '?')}: "
            f"{result.get('row_count', 0)} rows"
        )
    finally:
        client.close()


@datasets.command("pull")
@click.argument("dataset_id")
@click.option("--output", "-o", required=True, help="Output file path (e.g., ./data.jsonl)")
@click.option(
    "--version", "-v", "version_spec", default=None,
    help="Version to pull: 'latest' (default), 'v3', '3', or a full version UUID",
)
@click.option(
    "--format", "fmt", default=None,
    type=click.Choice(["jsonl", "parquet"]),
    help="Export format (default: inferred from file extension)",
)
@common_options
def datasets_pull(dataset_id, output, version_spec, fmt, api_key, base_url, project):
    """Pull a dataset to a local file for training.

    Downloads a dataset version and writes it to disk in a training-ready format.
    By default, pulls the latest version and infers format from the file extension.

    \b
    Examples:
        # Pull latest version as JSONL
        $ decimalai datasets pull ds_abc123 -o ./training_data.jsonl

        # Pull a specific version
        $ decimalai datasets pull ds_abc123 -o ./data.jsonl --version v2

        # Pull as Parquet
        $ decimalai datasets pull ds_abc123 -o ./data.parquet
    """
    client = _make_client(api_key, base_url, project)
    try:
        version_label = version_spec or "latest"
        click.echo(f"  Pulling dataset {dataset_id} (version: {version_label})...")

        result = client.pull_dataset(
            dataset_id, output, version=version_spec, format=fmt,
        )

        click.echo("")
        click.echo("  ✓ Downloaded successfully")
        click.echo(f"    File:    {result['file_path']}")
        click.echo(f"    Format:  {result['format']}")
        click.echo(f"    Rows:    {result['row_count']}")
        click.echo(f"    Size:    {_format_bytes(result['bytes_written'])}")
        click.echo(f"    Version: {result['version_id'][:12]}...")
        click.echo("")
    except ValueError as e:
        click.echo(f"  ✗ {e}", err=True)
        raise SystemExit(1)
    finally:
        client.close()


@datasets.command("export")
@click.argument("dataset_id")
@click.option(
    "--version", "-v", "version_spec", default=None,
    help="Version to export: 'latest' (default), 'v3', '3', or a UUID",
)
@click.option(
    "--format", "fmt", default="jsonl",
    type=click.Choice(["jsonl", "parquet"]),
    help="Export format (default: jsonl)",
)
@click.option("--output", "-o", default=None, help="Output file path (prints to stdout if omitted)")
@common_options
def datasets_export(dataset_id, version_spec, fmt, output, api_key, base_url, project):
    """Export a dataset version for training.

    Prints to stdout by default (for piping), or writes to a file with -o.

    \b
    Examples:
        # Export latest version to stdout
        $ decimalai datasets export ds_abc123

        # Export specific version to file
        $ decimalai datasets export ds_abc123 --version v2 -o training.jsonl

        # Pipe to another tool
        $ decimalai datasets export ds_abc123 | head -5
    """
    client = _make_client(api_key, base_url, project)
    try:
        data = client.export_dataset(dataset_id, version_spec, format=fmt)

        if output:
            if fmt == "parquet" and isinstance(data, bytes):
                with open(output, "wb") as f:
                    f.write(data)
            else:
                with open(output, "w") as f:
                    f.write(data if isinstance(data, str) else str(data))
            click.echo(f"Exported to {output}", err=True)
        else:
            if isinstance(data, bytes):
                click.echo("Error: Parquet output requires --output/-o flag", err=True)
                raise SystemExit(1)
            click.echo(data)
    except ValueError as e:
        click.echo(f"  ✗ {e}", err=True)
        raise SystemExit(1)
    finally:
        client.close()


def _format_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    elif n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    else:
        return f"{n / (1024 * 1024 * 1024):.1f} GB"


@datasets.command("push-to-hub")
@click.argument("dataset_id")
@click.argument("repo_id")
@click.option(
    "--version", "-v", "version_spec", default=None,
    help="Version to push: 'latest' (default), 'v3', '3', or UUID",
)
@click.option("--token", default=None, help="HuggingFace API token (falls back to HF_TOKEN env var)")
@click.option("--public/--private", "is_public", default=False, help="Create as public or private (default: private)")
@click.option("--split", default="train", help="Dataset split name (default: train)")
@common_options
def datasets_push_to_hub(dataset_id, repo_id, version_spec, token, is_public, split, api_key, base_url, project):
    """Push a dataset to HuggingFace Hub.

    Makes the dataset loadable by Axolotl, Unsloth, TRL, and any tool
    that supports HuggingFace's load_dataset().

    \b
    Examples:
        # Push latest version
        $ decimalai datasets push-to-hub ds_abc123 my-org/support-agent-sft

        # Push a specific version as public
        $ decimalai datasets push-to-hub ds_abc123 my-org/my-dataset --version v2 --public

    \b
    After pushing, use it in training:
        # Axolotl YAML
        datasets:
          - path: my-org/support-agent-sft

        # Python (TRL, Unsloth)
        from datasets import load_dataset
        ds = load_dataset("my-org/support-agent-sft")
    """
    import decimalai as sdk
    sdk.init(api_key=api_key, base_url=base_url, project=project)

    version_label = version_spec or "latest"
    click.echo(f"  Pushing dataset {dataset_id} (version: {version_label}) to {repo_id}...")

    try:
        result = sdk.push_to_hub(
            dataset_id, repo_id,
            version=version_spec,
            token=token,
            private=not is_public,
            split=split,
        )

        click.echo("")
        click.echo("  ✓ Pushed successfully")
        click.echo(f"    Repo:    {result['repo_url']}")
        click.echo(f"    Rows:    {result['row_count']}")
        click.echo(f"    Split:   {result['split']}")
        click.echo(f"    Version: {result['version_id'][:12]}...")
        click.echo("")
        click.echo("  Load it in Python:")
        click.echo('    from datasets import load_dataset')
        click.echo(f'    ds = load_dataset("{repo_id}")')
        click.echo("")
    except ImportError as e:
        click.echo(f"  ✗ {e}", err=True)
        click.echo("    Install with: pip install huggingface_hub datasets", err=True)
        raise SystemExit(1)
    except ValueError as e:
        click.echo(f"  ✗ {e}", err=True)
        raise SystemExit(1)


# ── Replay commands ────────────────────────────────────────

@cli.group()
def replay():
    """Replay commands — re-run agent on stale traces."""
    pass


@replay.command("run")
@click.argument("batch_id")
@click.option("--agent-fn", required=True, help="Agent function path (module:function)")
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing")
@click.option("--no-skip-failed", is_flag=True, help="Stop on first failure")
@common_options
def replay_run(batch_id, agent_fn, api_key, base_url, project, dry_run, no_skip_failed):
    """Execute the pending tasks in a replay batch.

    BATCH_ID is the replay batch to execute. Each pending task is re-run
    through your agent function and its result is submitted back to the batch,
    which advances the batch's progress and auto-scores the replay.

    Example:
        decimalai replay run batch_abc123 --agent-fn my_app.agent:run
    """
    from .._config import _get_client
    from ..replay.tasks import (
        _call_agent,
        _latest_trace_id,
        _wait_for_new_trace,
        load_agent_fn,
    )

    if not api_key:
        click.echo("Error: --api-key or DECIMAL_API_KEY required", err=True)
        raise SystemExit(1)

    # Initialize the SDK
    import decimalai
    decimalai.init(api_key=api_key, project=project, base_url=base_url)

    # Load the agent function
    try:
        fn = load_agent_fn(agent_fn)
        click.echo(f"✓ Loaded agent function: {agent_fn}")
    except (ImportError, AttributeError) as e:
        click.echo(f"Error loading {agent_fn}: {e}", err=True)
        raise SystemExit(1)

    client = _get_client()
    batch = client.get_replay_batch(batch_id)
    tasks = [t for t in batch.get("tasks", []) if t.get("status") == "pending"]

    if dry_run:
        click.echo(
            f"[DRY RUN] Would replay {len(tasks)} pending task(s) in batch {batch_id}"
        )
        return

    if not tasks:
        click.echo(f"No pending tasks in batch {batch_id} (already complete).")
        return

    total = len(tasks)
    completed = passed = failed = skipped = 0

    for i, task in enumerate(tasks, start=1):
        task_id = task.get("id")
        task_input = task.get("task_input") or {}
        user_input = task_input.get("user_input")
        agent_name = task_input.get("agent_name") or batch.get("agent_name") or "unknown"

        # A task whose original trace was cross-tenant/deleted has no input.
        if not user_input:
            skipped += 1
            try:
                client.submit_replay_result(task_id=task_id, status="skipped")
            except Exception:
                logger.debug("Could not mark replay task %s skipped", task_id, exc_info=True)
            click.echo(f"  ⊘ [{i}/{total}] {str(task_id)[:8]}: skipped (no input)")
            continue

        baseline = _latest_trace_id(client, agent_name)
        try:
            _call_agent(fn, user_input)
        except Exception as e:
            failed += 1
            try:
                client.submit_replay_result(task_id=task_id, status="failed")
            except Exception:
                logger.debug("Could not mark replay task %s failed", task_id, exc_info=True)
            click.echo(f"  ✗ [{i}/{total}] {str(task_id)[:8]}: agent error: {e}", err=True)
            if no_skip_failed:
                raise SystemExit(1)
            continue

        # The backend auto-scores when eval_score is omitted and the replayed
        # trace exists, so we only need to submit the new trace id.
        new_trace_id = _wait_for_new_trace(client, agent_name, baseline)
        result = client.submit_replay_result(
            task_id=task_id,
            replayed_trace_id=new_trace_id,
            status="completed",
        )
        completed += 1
        verdict = (result or {}).get("eval_verdict")
        if verdict in ("keep", "pass"):
            passed += 1
        click.echo(f"  ✓ [{i}/{total}] {str(task_id)[:8]}: {verdict or 'completed'}")

    click.echo("")
    click.echo(f"{'=' * 40}")
    click.echo("Replay Summary")
    click.echo(f"{'=' * 40}")
    click.echo(f"  Total:      {total}")
    click.echo(f"  Completed:  {completed}")
    click.echo(f"  Passed:     {passed}")
    click.echo(f"  Failed:     {failed}")
    click.echo(f"  Skipped:    {skipped}")
    if total:
        click.echo(f"  Pass rate:  {passed / total:.0%}")

    if failed > 0:
        raise SystemExit(1)


# ── Compat-check command ───────────────────────────────────────

@cli.command("compat-check")
@click.option("--agent-name", required=True, help="Agent to check compatibility for")
@click.option(
    "--format", "fmt", default="table",
    type=click.Choice(["table", "json", "github"]),
    help="Output format: table (human), json (machine), github (PR annotations)",
)
@click.option("--recompute", is_flag=True, help="Force a fresh compatibility analysis")
@common_options
def compat_check(agent_name, fmt, recompute, api_key, base_url, project):
    """Check dataset compatibility after a manifest change.

    Compares the latest two manifests for an agent and reports how many
    training-data traces are affected (keep/repair/replay/drop). This
    command is informational — it always exits 0 so it won't block CI.

    Use in GitHub Actions to annotate PRs with compatibility impact:

    \b
    Example:
        $ decimalai compat-check --agent-name my-agent
        $ decimalai compat-check --agent-name my-agent --format github
        $ decimalai compat-check --agent-name my-agent --format json
    """
    import json as json_mod

    client = _make_client(api_key, base_url, project)
    try:
        data = client.compat_check(agent_name, recompute=recompute)
    except Exception as e:
        click.echo(f"  ✗ Failed to check compatibility: {e}", err=True)
        client.close()
        raise SystemExit(1)
    finally:
        client.close()

    # Handle case where agent has < 2 manifests
    if data.get("status") == "no_transition":
        if fmt == "json":
            click.echo(json_mod.dumps(data, indent=2))
        elif fmt == "github":
            click.echo(f"::notice title=DecimalAI Compat Check::{data.get('message', 'No manifest transition found.')}")
        else:
            click.echo("")
            click.echo(f"  ℹ {data.get('message', 'No manifest transition to check.')}")
            click.echo("")
        return  # Exit 0 — informational

    # Extract counts
    keep = data.get("keep", 0)
    repair = data.get("repair", 0)
    replay = data.get("replay", 0)
    drop = data.get("drop", 0)
    total = data.get("total_traces", 0)
    old_ver = data.get("old_version", "?")
    new_ver = data.get("new_version", "?")
    needs_attention = repair + replay + drop

    if fmt == "json":
        click.echo(json_mod.dumps(data, indent=2, default=str))
        return

    if fmt == "github":
        # Emit GitHub Actions annotations
        summary = (
            f"Agent '{agent_name}': {old_ver} → {new_ver} | "
            f"{total} traces scored: {keep} keep, {repair} repair, "
            f"{replay} replay, {drop} drop"
        )
        if needs_attention > 0:
            click.echo(f"::warning title=DecimalAI: {needs_attention} traces need attention::{summary}")
            # Emit per-component annotations for high-impact changes
            for comp, impact in (data.get("component_impact") or {}).items():
                comp_total = sum(impact.values())
                if comp_total > 0:
                    click.echo(
                        f"::notice title=Component: {comp}::"
                        f"repair={impact.get('repair', 0)}, "
                        f"replay={impact.get('replay', 0)}, "
                        f"drop={impact.get('drop', 0)}"
                    )
        else:
            click.echo(f"::notice title=DecimalAI: All traces compatible::{summary}")
        return

    # Table format (default)
    def _pct(n: int) -> str:
        return f"{round(n / total * 100)}%" if total > 0 else "—"

    click.echo("")
    click.echo(f"  Agent: {agent_name}")
    click.echo(f"  Manifest: {new_ver} (active) ← {old_ver} (superseded)")
    click.echo(f"  Traces scored: {total}")
    click.echo("")
    click.echo("  ┌──────────┬───────┬──────┐")
    click.echo("  │ Category │ Count │   %  │")
    click.echo("  ├──────────┼───────┼──────┤")
    click.echo(f"  │ ✅ Keep  │ {keep:>5} │ {_pct(keep):>4} │")
    click.echo(f"  │ ⚡ Repair│ {repair:>5} │ {_pct(repair):>4} │")
    click.echo(f"  │ ↻ Replay │ {replay:>5} │ {_pct(replay):>4} │")
    click.echo(f"  │ ✗ Drop   │ {drop:>5} │ {_pct(drop):>4} │")
    click.echo("  └──────────┴───────┴──────┘")
    click.echo("")

    if needs_attention > 0:
        click.echo(f"  ⚠ {needs_attention} trace(s) need attention (repair/replay/drop)")
        if repair > 0:
            click.echo(f"  → {repair} trace(s) are mechanically repairable — preview with: decimalai repair preview --old-manifest-id <old> --new-manifest-id <new>")
        dashboard_url = _dashboard_url(base_url)
        click.echo(
            "  → View full report: "
            f"{dashboard_url}/agents/{urllib.parse.quote(agent_name, safe='')}?tab=manifests"
        )
    else:
        click.echo(f"  ✅ All {total} traces are compatible with {new_ver}")

    click.echo("")
    # Always exit 0 — informational only, does not block CI


# ── Repair ────────────────────────────────────────────────

@cli.group()
def repair():
    """Preview and apply mechanical trace repairs.

    Operates on a manifest transition (old → new manifest id).
    """
    pass


@repair.command("preview")
@click.option("--old-manifest-id", required=True, help="Manifest the traces were captured against")
@click.option("--new-manifest-id", required=True, help="Manifest to migrate the traces to")
@click.option("--sample-size", default=5, type=int, help="How many traces to preview (1-50)")
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "json"]), help="Output format")
@common_options
def repair_preview_cmd(old_manifest_id, new_manifest_id, sample_size, fmt, api_key, base_url, project):
    """Preview the repair rules generated for an old→new manifest transition."""
    import json as json_mod

    client = _make_client(api_key, base_url, project)
    try:
        data = client.repair_preview(old_manifest_id, new_manifest_id, sample_size=sample_size)
    except Exception as e:
        click.echo(f"  ✗ Repair preview failed: {e}", err=True)
        client.close()
        raise SystemExit(1)
    finally:
        client.close()

    if fmt == "json":
        click.echo(json_mod.dumps(data, indent=2, default=str))
        return
    rules = data.get("rules", [])
    if not rules:
        click.echo("")
        click.echo(f"  ℹ {data.get('message') or data.get('error') or 'No repairable changes found.'}")
        click.echo("")
        return
    click.echo("")
    click.echo(f"  {len(rules)} repair rule(s); {data.get('total_eligible', 0)} eligible trace(s):")
    for i, r in enumerate(rules):
        click.echo(f"    [{i}] {r.get('rule_type')} — {r.get('component_name')} ({r.get('confidence')})  {r.get('details')}")
    click.echo("")
    click.echo("  Apply all:      decimalai repair apply --old-manifest-id ... --new-manifest-id ...")
    click.echo("  Apply a subset: decimalai repair apply ... --rule-index 0 --rule-index 2")
    click.echo("")


@repair.command("apply")
@click.option("--old-manifest-id", required=True, help="Manifest the traces were captured against")
@click.option("--new-manifest-id", required=True, help="Manifest to migrate the traces to")
@click.option("--rule-index", "rule_indices", multiple=True, type=int, help="Apply only this preview rule index (repeatable). Omit to apply all.")
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "json"]), help="Output format")
@common_options
def repair_apply_cmd(old_manifest_id, new_manifest_id, rule_indices, fmt, api_key, base_url, project):
    """Apply repairs (all eligible rules, or only the --rule-index ones)."""
    import json as json_mod

    indices = list(rule_indices) or None
    client = _make_client(api_key, base_url, project)
    try:
        data = client.repair_apply(old_manifest_id, new_manifest_id, approved_rule_indices=indices)
    except Exception as e:
        click.echo(f"  ✗ Repair apply failed: {e}", err=True)
        client.close()
        raise SystemExit(1)
    finally:
        client.close()

    if fmt == "json":
        click.echo(json_mod.dumps(data, indent=2, default=str))
        return
    click.echo("")
    click.echo(f"  Repair batch {str(data.get('batch_id', ''))[:12]} — {data.get('status')}")
    click.echo(f"    Episodes: {data.get('total_episodes', 0)}   Repaired: {data.get('repaired_count', 0)}")
    dash = _dashboard_url(base_url)
    click.echo(f"    → {dash}/repair/{data.get('batch_id', '')}")
    click.echo("")


# ── Regression Check ──────────────────────────────────────

@cli.command("regression-check")
@click.option("--agent-name", required=True, help="Agent name (matches decimalai.init())")
@click.option(
    "--candidate-manifest-id",
    default=None,
    help=(
        "Candidate manifest ID. If omitted, reads from $GITHUB_OUTPUT "
        "(decimal_manifest_id key) or ./decimal_manifest_id.txt — the "
        "files written by `decimalai.flush_manifest_for_ci()`."
    ),
)
@click.option(
    "--fail-on",
    default="high",
    type=click.Choice(["high", "medium", "none"]),
    help="When to fail the command (nonzero exit). Default: high.",
)
@click.option(
    "--format", "fmt",
    default="terminal",
    type=click.Choice(["terminal", "json", "github"]),
    help="Output format: terminal (human), json (machine), github (PR annotations)",
)
@click.option(
    "--trace-window-days",
    default=30,
    type=int,
    help="How far back to look for affected traces.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Compute the report without persisting it or consuming the metered quota. Useful for local exploration.",
)
@common_options
def regression_check(
    agent_name,
    candidate_manifest_id,
    fail_on,
    fmt,
    trace_window_days,
    dry_run,
    api_key,
    base_url,
    project,
):
    """Run a manifest impact analysis for a candidate manifest.

    Computes the structural diff between a candidate manifest and the
    agent's current baseline, then identifies which historical production
    traces will break, may behave differently, or are unaffected.

    \b
    Used in two contexts:
        1. CI (GitHub Action): after `flush_manifest_for_ci()` has written
           the candidate manifest_id, this command consumes it and posts
           the impact report.
        2. Local exploratory: pass --candidate-manifest-id explicitly to
           preview impact for any manifest.

    \b
    Exit codes:
        0  — verdict is below the --fail-on threshold (or --fail-on=none)
        1  — verdict meets/exceeds --fail-on threshold
        2  — error invoking the API or reading the candidate manifest ID

    \b
    Examples:
        $ decimalai regression-check --agent-name support-agent
        $ decimalai regression-check --agent-name support-agent --format github
        $ decimalai regression-check --agent-name support-agent \\
            --candidate-manifest-id mfst_xyz --fail-on medium
    """
    import json as json_mod

    # 1. Resolve the candidate manifest ID
    if not candidate_manifest_id:
        candidate_manifest_id = _resolve_candidate_manifest_id()
    if not candidate_manifest_id:
        click.echo("", err=True)
        click.echo("  ✗ No candidate manifest ID provided or discoverable.", err=True)
        click.echo("", err=True)
        click.echo("  Pass --candidate-manifest-id explicitly, OR run", err=True)
        click.echo("  `decimalai.flush_manifest_for_ci()` first to write one.", err=True)
        click.echo("", err=True)
        raise SystemExit(2)

    # 2. Build PR context (best-effort; backend stores it for traceability)
    import decimalai
    pr_context = decimalai._read_github_pr_context() or None

    # 3. Call the API
    client = _make_client(api_key, base_url, project)
    try:
        report = client.run_regression_check(
            agent_name=agent_name,
            candidate_manifest_id=candidate_manifest_id,
            pr_context=pr_context,
            trace_window_days=trace_window_days,
            dry_run=dry_run,
            # 048: identify the CLI as the source. The dashboard renders
            # a 💻 CLI badge so users can tell where each check came from.
            source="cli",
        )
    except Exception as e:
        click.echo(f"  ✗ Regression check failed: {e}", err=True)
        raise SystemExit(2)
    finally:
        client.close()

    # 4. Render
    if fmt == "json":
        click.echo(json_mod.dumps(report, indent=2, default=str))
    elif fmt == "github":
        _render_regression_github_annotations(report, agent_name)
    else:
        _render_regression_terminal(report, agent_name, base_url)

    # 5. Exit code per --fail-on
    verdict = report.get("verdict", "no_change")
    if _should_fail(verdict, fail_on, report.get("structural_severity")):
        raise SystemExit(1)


def _resolve_candidate_manifest_id() -> str | None:
    """Best-effort resolution of the candidate manifest ID, in priority order:

        1. $GITHUB_OUTPUT (parsed for `decimal_manifest_id=...` key)
        2. ./decimal_manifest_id.txt (whole file is the ID)

    Returns None if nothing is found.
    """
    import os
    from pathlib import Path

    gh_out = os.environ.get("GITHUB_OUTPUT", "").strip()
    if gh_out and Path(gh_out).exists():
        for line in Path(gh_out).read_text(encoding="utf-8").splitlines():
            if line.startswith("decimal_manifest_id="):
                return line.split("=", 1)[1].strip()

    fallback = Path("decimal_manifest_id.txt")
    if fallback.exists():
        text = fallback.read_text(encoding="utf-8").strip()
        if text:
            return text

    return None


def _should_fail(verdict: str, fail_on: str, structural_severity: str | None = None) -> bool:
    """Return True if the verdict meets/exceeds the fail-on threshold.

    ``unverified`` (2026-07-28) means the backend found structural changes but had
    NO traffic to measure them against. It is ranked by the DIFF's severity, not by
    a fixed value: a fixed rank just moves the problem — too low and a deleted tool
    still merges green, too high and a one-line prompt tweak on an untrafficked
    agent reds the build.

    This mirrors the GitHub Action's shouldFail. Both gates must agree, because this
    CLI is the documented path for non-GitHub CI. An unknown verdict must never rank
    0 — a gate that silently passes what it does not recognize is worse than one that
    errors.

    A backend that does not send ``structural_severity`` falls back to low/warn
    rather than inventing a failure, so an older server cannot start reding builds.
    """
    if fail_on == "none":
        return False
    rank = {"no_change": 0, "first_run": 0, "low_risk": 1, "medium_risk": 2, "high_risk": 3}
    threshold = {"high": 3, "medium": 2}.get(fail_on, 3)
    if verdict == "unverified":
        return rank.get(f"{(structural_severity or 'low').lower()}_risk", 1) >= threshold
    return rank.get(verdict, 0) >= threshold


def _render_regression_terminal(report: dict, agent_name: str, base_url: str) -> None:
    """Render an impact report in terminal/human format."""
    verdict = report.get("verdict", "?")
    msg = report.get("verdict_message", "")
    high = report.get("high_risk_count", 0)
    medium = report.get("medium_risk_count", 0)
    low = report.get("low_risk_count", 0)
    total = report.get("total_traces_analyzed", 0)
    diff = (report.get("diff_summary") or {}).get("changes", []) or []

    click.echo("")
    click.echo(f"  🔍 Agent Regression Check — {agent_name}")
    click.echo("")

    if verdict == "first_run":
        click.echo(f"  ✓ {msg}")
        click.echo("")
        return

    if diff:
        click.echo("  Manifest changes:")
        for c in diff[:8]:
            sev_marker = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(c.get("severity"), "•")
            # Grade annotation (parity with the PR comment + dashboard). Present
            # only for model/prompt changes on graded backends; empty otherwise.
            d = c.get("detail") or {}
            grade = d.get("grade")
            if grade and c.get("type") == "model_changed":
                kind = (d.get("change_kind") or "").replace("_", " ")
                old = d.get("old_model") or d.get("old_provider")
                new = d.get("new_model") or d.get("new_provider")
                trans = f" ({old} → {new})" if old and new else ""
                suffix = f"  [{grade}{(': ' + kind) if kind else ''}]{trans}"
            elif grade and c.get("type") == "prompt_section_rewritten":
                pct = d.get("diff_pct")
                suffix = f"  [{grade}{f', {pct}% changed' if pct is not None else ''}]"
            elif grade:
                suffix = f"  [{grade}]"
            else:
                suffix = ""
            click.echo(f"    {sev_marker} {c.get('type')} — {c.get('name', '')}{suffix}")
        if len(diff) > 8:
            click.echo(f"    … and {len(diff) - 8} more")
        click.echo("")

    click.echo(f"  Impact on last 30 days of traces ({total} total):")
    click.echo("")
    click.echo(f"    🔴 HIGH RISK    — {high} traces will break")
    click.echo(f"    🟡 MEDIUM RISK  — {medium} traces may behave differently")
    click.echo(f"    🟢 LOW RISK     — {low} traces affected with low risk")
    click.echo("")

    sev_emoji = {
        "high_risk": "🔴", "medium_risk": "🟡", "low_risk": "🟢",
        "no_change": "✓", "first_run": "✓",
    }.get(verdict, "•")
    click.echo(f"  Verdict: {sev_emoji} {verdict.replace('_', ' ').upper()} — {msg}")
    click.echo("")

    # Training-data policy implication (parity with the PR comment + dashboard).
    # Display-only: the gate above is the verdict + fail-on. Only model/prompt
    # changes carry detail.policy; older backends omit it (block stays silent).
    _policy_changes = [c for c in diff if (c.get("detail") or {}).get("policy")]
    if _policy_changes:
        _gloss = {
            "drop": "excluded from training", "replay": "need re-running first",
            "flag": "flagged for review", "repair": "auto-repaired", "keep": "retained",
        }
        _name = (_policy_changes[0]["detail"]["policy"] or {}).get("name", "default")
        click.echo(f"  Training-data policy ({_name}):")
        for c in _policy_changes:
            _pol = c["detail"]["policy"] or {}
            _disp = _pol.get("disposition", "")
            _g = (c.get("detail") or {}).get("grade", "")
            click.echo(f"    {c.get('type')} ({_g}) → {_disp} — {_gloss.get(_disp, '')}")
        click.echo("")

    rc_id = report.get("id")
    if rc_id:
        dashboard = _dashboard_url(base_url)
        click.echo(
            "  → View full report: "
            f"{dashboard}/agents/{urllib.parse.quote(agent_name, safe='')}/regression/{rc_id}"
        )
        click.echo("")


def _render_regression_github_annotations(report: dict, agent_name: str) -> None:
    """Render impact report as GitHub Actions annotations."""
    verdict = report.get("verdict", "?")
    msg = report.get("verdict_message", "")
    high = report.get("high_risk_count", 0)
    medium = report.get("medium_risk_count", 0)
    low = report.get("low_risk_count", 0)
    summary = (
        f"Agent '{agent_name}': {high} HIGH, {medium} MEDIUM, {low} LOW. {msg}"
    )
    if verdict == "high_risk":
        click.echo(f"::error title=DecimalAI: HIGH RISK regression::{summary}")
    elif verdict == "medium_risk":
        click.echo(f"::warning title=DecimalAI: MEDIUM RISK regression::{summary}")
    elif verdict == "first_run":
        click.echo(f"::notice title=DecimalAI: First run::{msg}")
    else:
        click.echo(f"::notice title=DecimalAI: No regression::{summary}")


# ── Demo sandbox (the one-command "wow") ───────────────────


def _demo_web_url(base_url: str, web: str | None) -> str:
    """Resolve the frontend base URL for the printed deep links.

    Explicit --web wins. Otherwise derive from the API base URL: a local
    backend pairs with the :3000 dev server, while a hosted ``api.`` host
    maps to ``app.`` (matching the dashboard links the rest of the CLI
    prints).
    """
    if web:
        return web.rstrip("/")
    # Same resolution as every other printed dashboard link.
    return _dashboard_url(base_url)


# Demo seeds/teardown run a server-side stats recompute + multi-table
# cleanup that can outrun the client's 30s default on a populated DB.
# Matches the standalone demo scripts' 120s budget.
_DEMO_HTTP_TIMEOUT = 120.0


@cli.group()
def demo():
    """Seed a guided demo sandbox — and tear it down.

    Two one-command demos against your workspace, sharing one teardown:

    \b
        decimalai demo regression   # "Your agent changed" (CI gate + interactive repair)
        decimalai demo skills       # "Find skills that work"
        decimalai demo reset        # remove ALL demo data

    Each seed is reset-on-run by default, so you always land in the same
    clean state. Pass --no-reset for an idempotent seed that leaves any
    existing demo data untouched.

    \b
    Local sandbox:
        decimalai demo skills --base-url http://localhost:8000 \\
            --api-key dai_sk_test_key_001
    """
    pass


@demo.command("regression")
@click.option(
    "--reset/--no-reset", default=True,
    help="Wipe existing demo data before seeding (default: reset).",
)
@click.option("--web", default=None, help="Frontend base URL for the printed link.")
@common_options
def demo_regression(reset, web, api_key, base_url, project):
    """Seed the "Your agent changed" demo and open the impact report.

    Seeds a v1→v2 agent (model swap, tool rename/removal, prompt rewrite)
    plus a real trace corpus, runs the regression check, and prints the
    URL straight to the impact report — the keep/repair/replay/drop
    fan-out you get both in CI / pre-deploy and in the interactive
    repair→build→export flow.
    """

    client = _make_client(api_key, base_url, project)
    web_url = _demo_web_url(base_url, web)
    report = None
    try:
        click.echo("")
        click.echo('  Demo A — "Your agent changed"')
        click.echo("  Seeding the demo agent (v1 → v2 + traces)…")
        resp = client._http.post(
            "/api/v1/demo/seed-impact", params={"force": reset},
            timeout=_DEMO_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        seed = resp.json()

        if seed.get("status") == "already_exists":
            agent = seed.get("agent_name", "[Demo] support-agent")
            quoted = urllib.parse.quote(agent, safe="")
            click.echo(f"  ℹ {seed.get('message', 'Demo data already exists.')}")
            click.echo("    Re-run with --reset to regenerate the impact report.")
            click.echo(f"    → {web_url}/agents/{quoted}")
            click.echo("")
            return

        agent_name = seed.get("agent_name", "[Demo] support-agent")
        v2_id = seed.get("v2_manifest_id")
        if not v2_id:
            click.echo(f"  ✗ Seed did not return a candidate manifest: {seed}", err=True)
            raise SystemExit(1)
        click.echo(
            f"    agent: {agent_name}  ·  traces: {seed.get('traces')}  ·  "
            f"v1 {str(seed.get('v1_manifest_id'))[:8]} → v2 {str(v2_id)[:8]}"
        )

        click.echo("  Running the regression check (v2 vs auto-resolved v1)…")
        report = client.run_regression_check(
            agent_name=agent_name,
            candidate_manifest_id=v2_id,
            source="cli",
        )
    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"  ✗ Demo regression seed failed: {e}", err=True)
        client.close()
        raise SystemExit(1)
    finally:
        client.close()

    run_id = report.get("id")
    quoted = urllib.parse.quote(agent_name, safe="")
    click.echo("")
    click.echo(f"  Verdict: {report.get('verdict')} — {report.get('verdict_message', '')}")
    click.echo(
        f"  Traces analyzed: {report.get('total_traces_analyzed')} "
        f"(high {report.get('high_risk_count')} / "
        f"med {report.get('medium_risk_count')} / "
        f"low {report.get('low_risk_count')})"
    )
    click.echo("")
    click.echo("  Open the impact report (keep / repair / replay / drop fan-out):")
    click.echo(f"    {web_url}/agents/{quoted}/impact-reports/{run_id}")
    click.echo("")


@demo.command("skills")
@click.option(
    "--reset/--no-reset", default=True,
    help="Wipe existing demo skills before seeding (default: reset).",
)
@click.option("--web", default=None, help="Frontend base URL for the printed links.")
@common_options
def demo_skills(reset, web, api_key, base_url, project):
    """Seed the "Find skills that work" demo and open the ranked registry.

    Seeds three demo skills into your workspace (org-scoped — other orgs
    never see them) with deliberately varied effectiveness, runs the
    registry stats recompute, and prints links to the registry and the
    top skill's detail page, which shows per-model pass rates and the
    cross-org router activation rate.
    """
    client = _make_client(api_key, base_url, project)
    web_url = _demo_web_url(base_url, web)
    try:
        click.echo("")
        click.echo('  Demo B — "Find skills that work"')
        click.echo("  Seeding 3 demo skills into your workspace + stats + traces (runs the recompute)…")
        resp = client._http.post(
            "/api/v1/demo/seed-skills", params={"force": reset},
            timeout=_DEMO_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        seed = resp.json()
    except Exception as e:
        click.echo(f"  ✗ Demo skills seed failed: {e}", err=True)
        client.close()
        raise SystemExit(1)
    finally:
        client.close()

    if seed.get("status") == "already_exists":
        click.echo(f"  ℹ {seed.get('message', 'Demo skills already exist.')}")
        click.echo("    Re-run with --reset to regenerate the ranked demo data.")
        click.echo(f"    → {web_url}/skills")
        click.echo("")
        return

    top_id = seed.get("top_skill_id")
    top_slug = seed.get("top_skill_slug")
    names = seed.get("skill_names") or []
    click.echo(
        f"    skills: {', '.join(names)}  ·  traces: {seed.get('traces')}  ·  "
        f"daily-stats rows: {seed.get('daily_stats_rows')}"
    )
    click.echo("")
    click.echo("  Open the Skills Registry (ranked by SkillScore — measured quality, not installs):")
    click.echo(f"    {web_url}/skills")
    if top_id:
        click.echo("")
        click.echo("  Open the top skill — per-model pass rates + 'across all orgs'")
        click.echo("  router activation rate:")
        # The canonical detail URL is /skills/<url_slug>. The seed payload's
        # top_skill_name is the DISPLAY name ("[Demo] code-reviewer") —
        # interpolating it produced a dead link with a literal space in it.
        # Older backends without top_skill_slug fall back to
        # /skills/<id>; the dispatcher resolves ids too.
        click.echo(f"    {web_url}/skills/{top_slug or top_id}")
    click.echo("")


@demo.command("reset")
@common_options
def demo_reset(api_key, base_url, project):
    """Remove ALL demo data (both demos) for this workspace.

    Deletes the seeded demo agent, its manifests, traces, and datasets AND
    the demo skills, their daily stats, effectiveness rollups, and usage
    traces. Exact-name matched, so your own skills/agents that merely start
    with "[Demo] " are left untouched.
    """
    client = _make_client(api_key, base_url, project)
    try:
        resp = client._http.delete(
            "/api/v1/demo/cleanup", timeout=_DEMO_HTTP_TIMEOUT
        )
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        click.echo(f"  ✗ Demo cleanup failed: {e}", err=True)
        client.close()
        raise SystemExit(1)
    finally:
        client.close()

    skills = result.get("skills") or {}
    click.echo("")
    click.echo("  ✓ Demo data removed.")
    click.echo(
        f"    agent demo:  {result.get('manifests', 0)} manifest(s), "
        f"{result.get('traces', 0)} trace(s)"
    )
    click.echo(
        f"    skills demo: {skills.get('skills', 0)} skill(s), "
        f"{skills.get('traces', 0)} trace(s)"
    )
    click.echo("")


if __name__ == "__main__":
    cli()

