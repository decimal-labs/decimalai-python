"""The JOURNEY axis: an agent on the platform → a file → a skill the model can read.

Why this file exists
--------------------
Everything else in this suite grades ONE ADAPTER, handed a ``Ctx`` by a driver
that already knows the agent's name, already holds the skills, and never once
runs the product's own entry point. C14 asks "did a body reach the model"; D1
asks "by which channel". Both are true and both were green on 2026-08-28 while
the thing a user actually does was broken end to end:

    they make an agent in the dashboard, type a prompt, attach skills;
    they run ``decimalai init <name>``;
    they run the file it wrote;
    and the skill's KNOWLEDGE has to be in front of the model.

That path was covered only by the live end-to-end tier — operator-run, needing a
real backend, a real key and a real model, and appearing in no CI workflow at
all. So the journey was graded on the days somebody remembered to walk it.

What makes it hermetic
----------------------
The probe already stands in for ``api.decimal.ai``. Teaching it the three routes
``decimalai init`` calls (``GET /api/v1/agents``, ``…/{name}/skills``,
``…/{name}/prompt`` — see ``probe.py``) means the whole journey runs with no
backend, no provider key, no network and no cost. The MODEL half is stood in for
the same way, by :class:`JourneyModel` — a real HTTP server speaking the OpenAI
wire format, which the generated file reaches through its real provider SDK over
a real socket. Nothing is monkeypatched into the SDK under test.

What runs where
---------------
Three processes, and the split is the point:

* **this one** (the pytest parent) starts the two servers and grades. It imports
  no framework — every framework import happens in a grandchild — so unlike the
  driver tier it needs no child of its own to stay clean.
* ``decimalai init`` runs as a subprocess: the REAL console entry point, over
  real HTTP, against the probe.
* the generated file runs as a subprocess too, through :mod:`_journey_run`, which
  executes it with ``runpy`` under ``__main__`` exactly as ``python agent.py``
  would.

What it does NOT cover
----------------------
The model's ANSWER. The stub model is scripted, so this tier can prove the
skill's body was in the context the model was handed and can prove nothing at all
about whether a real model then USED it. That question needs a real model and is
owned by the live end-to-end tier. Stated here because a journey tier that quietly
implied otherwise would be worse than none.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from decimalai.cli import scaffold

# The wire FORMAT is the drivers' — one source for what an OpenAI response
# looks like, so the journey's stub and the driver tier's stub cannot drift into
# disagreeing about the endpoint they both stand in for. Underscore-prefixed
# because they are private to that module's own callers; imported here
# deliberately rather than copied, which is the alternative.
from .drivers._openai_wire import (
    STUB_API_KEY,
    _chat_payload,
    _responses_payload,
    _turns_taken,
)
from .drivers import StubTurn, _importable
from .harness import Phase
from .isolation import CONFORMANCE_SKILLS, body_sentinel, jobs
from .probe import Probe, Recorded

# ── the fixture ──────────────────────────────────────────────────────────────

#: The agent the journey registers on the probe. One agent, because that is what
#: a user has when they run ``decimalai init`` for the first time.
JOURNEY_AGENT_NAME = "conformance-journey-agent"

#: The system prompt stored ON THE PLATFORM for that agent, carrying a sentinel.
#:
#: Load-bearing in a way that is easy to miss: the scaffold deliberately never
#: writes the prompt TEXT into the generated file (``scaffold._prompt_comment_lines``
#: — "a copy pasted into the file is a second source of truth that goes stale"),
#: so this sentence can only reach the model if the generated file went and READ
#: it at run time. That is why "the prompt reached the model" is a journey clause
#: no adapter item can stand in for.
#:
#: Shaped as a checkable FACT rather than a random token, for the same reason
#: ``CONFORMANCE_SKILLS`` is: one fixture serves both tiers — this tier asserts
#: the sentence was in the context, and a live tier can assert the number comes
#: back in the model's answer.
JOURNEY_PROMPT_SENTINEL = (
    "JOURNEY-PROMPT-ALPHA: this desk issues at most 15 replacement labels a month."
)
JOURNEY_SYSTEM_PROMPT = (
    "You are the returns desk for the conformance fixture.\n"
    f"{JOURNEY_PROMPT_SENTINEL}\n"
    "Answer from your skills."
)

#: What the stub model replies. The generated file prints ``run(...)``'s return
#: value, so finding this on stdout is how the tier knows the file did not merely
#: import cleanly — it ran a turn and got the answer back out. Not decoration:
#: the langchain template used to emit a bare chat model with no loop, and
#: ``run()`` returned ``""`` on every call with no error anywhere.
JOURNEY_ANSWER_SENTINEL = "JOURNEY-ANSWER-ALPHA: replacement labels are capped at 15."

#: How long one framework's journey may take (init + a full agent run).
DEFAULT_TIMEOUT_SECONDS = 600
TIMEOUT_ENV = "DECIMAL_CONFORMANCE_JOURNEY_TIMEOUT"

#: Fields of an OpenAI request the provider does NOT show the model. Dropped
#: before the sentinel search, so "it reached the model" cannot be satisfied by a
#: caller stuffing a body into a field the model never sees. Everything else in
#: the request — instructions, input/messages, tools and their descriptions — is
#: part of the context the provider builds.
_NOT_SHOWN_TO_THE_MODEL = frozenset({"metadata", "model", "stream", "store", "user"})

# ── which driver maps onto which scaffold key ────────────────────────────────

#: driver name -> the key(s) ``decimalai/cli/scaffold.py`` files that framework
#: under. Two spellings differ from the driver names, and both are deliberate:
#: the generic OTel rail is ``otel`` there (with ``autogen`` aliased onto the
#: same rail since 2026-08-16), and there is no scaffold key for a driver the
#: docs advertise only in prose.
#:
#: Lives here rather than in ``test_coverage.py`` (where it started) because it
#: answers a question about ``decimalai init``, which is what this module is
#: about, and because two consumers now need it: the journey's own N/A
#: resolution and the rail/seam cross-check. One copy, or they drift.
SCAFFOLD_KEYS: Dict[str, Set[str]] = {
    "langchain": {"langchain"},
    "openai-agents": {"openai-agents"},
    "anthropic": {"anthropic"},
    "pydantic-ai": {"pydantic-ai"},
    "llamaindex": {"llamaindex"},
    "claude-agent-sdk": {"claude-agent-sdk"},
    "crewai": {"crewai"},
    "adk": {"adk"},
    "generic-otel": {"otel", "autogen"},
}


def journey_framework(driver_name: str) -> Optional[str]:
    """The scaffold key this driver's journey runs, or None if it has none.

    Read straight off ``scaffold.SUPPORTED_FRAMEWORKS`` — never a list of names
    typed here. A framework that gains a template gains a journey cell on the
    same commit, with nothing in this suite to remember to update.
    """
    for key in sorted(SCAFFOLD_KEYS.get(driver_name, set())):
        if key in scaffold.SUPPORTED_FRAMEWORKS:
            return key
    return None


def journey_na_ledger(driver_name: str) -> Optional[str]:
    """Which of the SDK's OWN ledgers explains why this driver has no journey.

    ``"UNSCAFFOLDED_WITH_SEAM"`` — the adapter can deliver skills, there is just
    no template yet. ``"NO_PROMPT_SEAM"`` — ``decimalai init`` REFUSES to
    scaffold it, because a generated file would trace correctly and deliver none
    of the agent's skills.

    Returns None for a driver that IS scaffoldable (so no exemption is owed), and
    raises for one the SDK classifies nowhere — an unclassified driver must not
    be able to buy itself a quiet skip by being unknown.
    """
    keys = SCAFFOLD_KEYS.get(driver_name, set())
    if any(k in scaffold.SUPPORTED_FRAMEWORKS for k in keys):
        return None
    if any(k in scaffold.UNSCAFFOLDED_WITH_SEAM for k in keys):
        return "UNSCAFFOLDED_WITH_SEAM"
    if any(k in scaffold.NO_PROMPT_SEAM for k in keys):
        return "NO_PROMPT_SEAM"
    raise KeyError(
        f"driver {driver_name!r} names no framework decimalai/cli/scaffold.py "
        f"classifies (keys: {sorted(keys)}). Either SCAFFOLD_KEYS is stale or "
        f"`decimalai init` has never been taught this framework exists — and "
        f"until one of those is true, this driver cannot be granted a journey "
        f"exemption, because nobody can say what the exemption is FOR."
    )


def journey_requirements(framework: str) -> Tuple[str, ...]:
    """Root modules the generated file needs, derived from the SDK's own answers.

    Two sources, both the product's:

    * the IMPORTS of the file ``render_agent_file`` actually emits, and
    * the packages ``install_command()`` tells the user to pip install — which is
      where ``langchain-openai`` comes from. It appears in no import line: the
      template calls ``init_chat_model("gpt-4o-mini")``, and the provider binding
      is chosen by the model STRING at run time.

    Deriving it beats a list typed here, which would go stale the first time a
    template changed and would then skip the cell instead of failing it.
    """
    import ast

    source = scaffold.render_agent_file(
        JOURNEY_AGENT_NAME, framework=framework, skills=[]
    )
    modules: Set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            modules.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    for token in scaffold.install_command(framework).split():
        token = token.strip('"').strip("'")
        if token in ("pip", "install") or "[" in token:
            continue
        # Distribution name -> import name. True for every package these
        # templates name; a distribution whose import name differs by more than
        # the hyphen would need a mapping, and would announce itself as a cell
        # that skips on a machine that has the package.
        modules.add(token.replace("-", "_"))
    return tuple(sorted(modules))


def missing_requirements(framework: str) -> List[str]:
    """Which of :func:`journey_requirements` this machine does not have."""
    return [m for m in journey_requirements(framework) if not _importable(m)]


# ── the stub model (the witness) ─────────────────────────────────────────────


class JourneyModel:
    """A real HTTP server speaking the OpenAI wire format — and the witness.

    Two differences from ``drivers/_openai_wire.py``, both forced by what the
    journey is:

    * **No lane registration.** The generated file asks its own hardcoded
      question ("What can you help me with?"), so there is no sentinel to key a
      lane on. One agent, one conversation, one server.
    * **The script is derived from the REQUEST, not from the framework.** If the
      request offers a ``load_skill`` tool and no skill body has come back yet,
      the model asks for one; otherwise it answers. That is what keeps this cell
      channel-agnostic: langchain's adapter registers no ``load_skill`` tool and
      must deliver by injection, openai-agents' does and delivers through the
      loop, and NEITHER is scripted for by name here. A hand-written per-
      framework script would be the driver artifact the whole tier exists to
      avoid — and would have to be updated by whoever broke the channel.

    Every request body is kept verbatim. That is the evidence the journey is
    graded on: what the model was actually handed.
    """

    def __init__(self, skill_name: str) -> None:
        self.skill_name = skill_name
        self.requests: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ────────────────────────────────────────────

    def start(self) -> "JourneyModel":
        model = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # noqa: A003
                pass

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                status, payload = model.answer(self.path, raw)
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("conformance journey model stub was not started")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    # ── answering ────────────────────────────────────────────

    @staticmethod
    def _offered_tools(body: Dict[str, Any]) -> Set[str]:
        """Tool names in this request — both wire spellings.

        Responses puts ``{"type": "function", "name": …}`` at the top level;
        Chat Completions nests it under ``function``.
        """
        out: Set[str] = set()
        for tool in body.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            if isinstance(tool.get("name"), str):
                out.add(tool["name"])
            fn = tool.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                out.add(fn["name"])
        return out

    def answer(self, path: str, raw: bytes) -> Tuple[int, Any]:
        text = raw.decode("utf-8", "replace")
        try:
            body = json.loads(text) if text else {}
        except ValueError:
            return 400, {
                "error": {
                    "message": "conformance journey stub: body was not JSON",
                    "type": "invalid_request_error",
                }
            }
        if not isinstance(body, dict):
            body = {}
        with self._lock:
            self.requests.append({"path": path, "body": body})

        if "load_skill" in self._offered_tools(body) and _turns_taken(body) == 0:
            turn = StubTurn(("load_skill", {"name": self.skill_name}), "", 17, 5)
        else:
            turn = StubTurn(None, JOURNEY_ANSWER_SENTINEL, 23, 7)

        model_name = body.get("model") or "conformance-journey-stub"
        if path.endswith("/responses"):
            return 200, _responses_payload(model_name, turn, body)
        if path.endswith("/chat/completions"):
            return 200, _chat_payload(model_name, turn)
        return 404, {
            "error": {
                "message": f"conformance journey stub has no route for POST {path}",
                "type": "invalid_request_error",
            }
        }

    # ── the view the contract reads ──────────────────────────

    def shown_to_the_model(self) -> str:
        """Everything the model was handed, across every request, as one string.

        Non-visible fields are dropped first (see ``_NOT_SHOWN_TO_THE_MODEL``),
        so a sentinel found here really was in the context the provider built.
        """
        with self._lock:
            bodies = [dict(r["body"]) for r in self.requests]
        for body in bodies:
            for key in _NOT_SHOWN_TO_THE_MODEL:
                body.pop(key, None)
        return json.dumps(bodies)


# ── the capture ──────────────────────────────────────────────────────────────


@dataclass
class JourneyCapture:
    """One framework's walk through the journey. The contract's input.

    Everything here is an OBSERVATION. No verdicts — those live in
    ``contract.grade_journey``, for the same reason a driver contains no
    assertions.
    """

    driver: str
    framework: str
    agent_name: str
    #: What the platform stores for this agent, and what the rail offers.
    prompt_sentinel: str
    answer_sentinel: str
    skill_sentinels: Dict[str, str]

    #: `decimalai init` — the real console entry point, as a subprocess.
    init_command: List[str] = field(default_factory=list)
    init_returncode: int = -1
    init_stdout: str = ""
    init_stderr: str = ""
    init_phase: Phase = field(
        default_factory=lambda: Phase(name="journey:init", ctxs=[])
    )

    #: What it wrote.
    out_path: str = ""
    file_written: bool = False
    file_source: str = ""

    #: Running that file.
    run_command: List[str] = field(default_factory=list)
    run_returncode: Optional[int] = None
    run_stdout: str = ""
    run_stderr: str = ""
    run_phase: Phase = field(default_factory=lambda: Phase(name="journey:run", ctxs=[]))

    #: What the model was handed, verbatim, and the flattened view of it.
    model_requests: int = 0
    model_context: str = ""


# ── running one journey ──────────────────────────────────────────────────────


def _timeout_seconds() -> float:
    raw = os.environ.get(TIMEOUT_ENV)
    try:
        return float(raw) if raw else float(DEFAULT_TIMEOUT_SECONDS)
    except ValueError:
        return float(DEFAULT_TIMEOUT_SECONDS)


def _tail(text: Any, limit: int = 4000) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    text = (text or "").strip()
    return text if len(text) <= limit else "...\n" + text[-limit:]


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_RUNNER = _HERE / "_journey_run.py"


def _base_env(probe: Probe) -> Dict[str, str]:
    """The environment both subprocesses inherit.

    Two jobs. It points the SDK at the probe with the key the CLI asks for. And
    it STRIPS the settings a developer's shell could be carrying that would
    change what the cell means — the delivery knobs (which would decide the body
    channel for us) and the two LangSmith switches (which would send this run's
    prompts to a third party). Same argument ``isolation._child_env`` makes: "it
    passes if you run it this way" is not a result.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT)] + ([existing] if existing else [])
    )
    env["DECIMAL_API_KEY"] = probe.api_key
    for key in (
        "DECIMALAI_API_KEY",
        "DECIMAL_BASE_URL",
        "DECIMALAI_BASE_URL",
        "DECIMALAI_INJECT_SKILL_BODY",
        "DECIMALAI_LOAD_SKILL_TOOL",
    ):
        env.pop(key, None)
    # Off, explicitly. Both default off, but an inherited `true` would ship every
    # prompt in this run to LangSmith — a hermetic tier does not get to be
    # hermetic only on machines that happen not to have it set.
    env["LANGCHAIN_TRACING_V2"] = "false"
    env["LANGSMITH_TRACING"] = "false"
    return env


def run_journey(driver_name: str, framework: str) -> JourneyCapture:
    """Walk the whole journey once, for one framework, and capture it.

    Runs in the CALLER's process, which is safe here in a way it is not for a
    driver: this function imports no framework and executes no adapter code.
    Both halves that do — the CLI and the generated file — are subprocesses.
    """
    skills = [dict(s) for s in CONFORMANCE_SKILLS]
    capture = JourneyCapture(
        driver=driver_name,
        framework=framework,
        agent_name=JOURNEY_AGENT_NAME,
        prompt_sentinel=JOURNEY_PROMPT_SENTINEL,
        answer_sentinel=JOURNEY_ANSWER_SENTINEL,
        skill_sentinels={
            s["name"]: body_sentinel(s.get("body", "")) for s in skills
        },
    )

    workdir = tempfile.mkdtemp(prefix=f"conformance-journey-{framework}-")
    # `require_manifest_on_ingest=False`: the generated file registers its own
    # manifest before its first trace, but the journey is not grading ingest —
    # C2 owns that, per driver, and a manifest race here would fail this cell
    # for a reason it does not measure.
    probe = Probe(require_manifest_on_ingest=False).start()
    model = JourneyModel(skills[0]["name"]).start()
    try:
        probe.register_agent(
            JOURNEY_AGENT_NAME, system_prompt=JOURNEY_SYSTEM_PROMPT, skills=skills
        )
        env = _base_env(probe)
        out_path = Path(workdir) / scaffold.DEFAULT_OUTPUT
        capture.out_path = str(out_path)

        # ── 1. the real `decimalai init`, over real HTTP ─────────────
        cursor = probe.mark()
        capture.init_command = [
            sys.executable, "-m", "decimalai.cli", "init", JOURNEY_AGENT_NAME,
            "--framework", framework,
            "--base-url", probe.base_url,
            "--out", str(out_path),
        ]
        init = _run(capture.init_command, workdir, env)
        capture.init_returncode = init.returncode
        capture.init_stdout = _tail(init.stdout)
        capture.init_stderr = _tail(init.stderr)
        capture.init_phase.requests = probe.since(cursor)

        capture.file_written = out_path.is_file()
        if capture.file_written:
            capture.file_source = out_path.read_text(encoding="utf-8", errors="replace")
        else:
            # Nothing to run. The capture says so and the contract fails the
            # cell — never a skip: a journey that did not happen is not a
            # journey that passed.
            return capture

        # ── 2. run what it wrote ─────────────────────────────────────
        cursor = probe.mark()
        run_env = dict(env)
        # The provider key(s) the CLI itself tells the user to export, set to the
        # stub server's non-credential. Read from `scaffold.env_vars` so a
        # template that changes provider does not silently run keyless.
        for var in scaffold.env_vars(framework):
            if var not in ("DECIMAL_API_KEY", "DECIMALAI_API_KEY"):
                run_env[var] = STUB_API_KEY
        # Both spellings, because the two provider bindings read different ones:
        # the `openai` SDK takes OPENAI_BASE_URL, langchain_openai takes
        # OPENAI_API_BASE. Setting one would silently send the other framework's
        # run to api.openai.com with a stub key.
        run_env["OPENAI_BASE_URL"] = model.base_url
        run_env["OPENAI_API_BASE"] = model.base_url
        capture.run_command = [sys.executable, str(_RUNNER), str(out_path)]
        run = _run(capture.run_command, workdir, run_env)
        capture.run_returncode = run.returncode
        capture.run_stdout = _tail(run.stdout)
        capture.run_stderr = _tail(run.stderr)
        capture.run_phase.requests = probe.since(cursor)

        capture.model_requests = len(model.requests)
        capture.model_context = model.shown_to_the_model()
        return capture
    finally:
        model.stop()
        probe.stop()
        # Kept on disk when the journey did NOT complete — same rule
        # ``isolation.run_driver_in_child`` uses for an unreadable capture. The
        # generated file is the whole artefact of a failed cell, and the failure
        # message names its path; deleting it would send the reader to a path
        # that no longer exists.
        if capture.file_written and capture.run_returncode == 0:
            shutil.rmtree(workdir, ignore_errors=True)


@dataclass
class _Completed:
    returncode: int
    stdout: str
    stderr: str


def _run(cmd: Sequence[str], workdir: str, env: Dict[str, str]) -> _Completed:
    """One subprocess, with a timeout that reports instead of hanging."""
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=_timeout_seconds(),
        )
        return _Completed(proc.returncode, proc.stdout or "", proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        return _Completed(
            -9,
            _tail(exc.stdout),
            _tail(exc.stderr)
            + f"\n[conformance] killed after {_timeout_seconds():.0f}s",
        )


def run_journeys(
    pairs: Sequence[Tuple[str, str]],
) -> Tuple[Dict[str, JourneyCapture], Dict[str, str]]:
    """Walk each ``(driver, framework)`` journey, in parallel. Keyed by driver.

    A journey that raises lands in ``failures`` and is turned into a hard error
    at lookup time — never a skip, for the reason ``isolation`` gives: a cell
    that was not graded must not reach the exit code as a success.
    """
    captured: Dict[str, JourneyCapture] = {}
    failures: Dict[str, str] = {}
    if not pairs:
        return captured, failures
    with ThreadPoolExecutor(max_workers=jobs(len(pairs))) as pool:
        futures = {
            driver: pool.submit(run_journey, driver, framework)
            for driver, framework in pairs
        }
    for driver, _ in pairs:
        try:
            captured[driver] = futures[driver].result()
        except Exception as exc:  # noqa: BLE001 — reported, never swallowed
            failures[driver] = f"{driver}: the journey could not be run: {exc!r}"
    return captured, failures


#: Re-exported so callers do not have to know the recording type lives in probe.
__all__ = [
    "JOURNEY_AGENT_NAME",
    "JOURNEY_ANSWER_SENTINEL",
    "JOURNEY_PROMPT_SENTINEL",
    "JOURNEY_SYSTEM_PROMPT",
    "JourneyCapture",
    "JourneyModel",
    "Recorded",
    "SCAFFOLD_KEYS",
    "journey_framework",
    "journey_na_ledger",
    "journey_requirements",
    "missing_requirements",
    "run_journey",
    "run_journeys",
]
