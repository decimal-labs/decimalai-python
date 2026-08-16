#!/usr/bin/env python3
"""Post-publish smoke: run the DOCUMENTED quickstart against the PyPI artifact.

Every "the docs teach an API the wheel doesn't have" / "the wheel predates its
tag" defect shares one root: nothing ties a user-facing claim to the artifact a
user can actually ``pip install``. The release gate's clean-room step is
``import decimalai`` plus ``--help``, which both survive a quickstart that has
stopped producing a single trace.

This script closes that. It is deliberately built on two rules:

1. **The artifact under test is the wheel from PyPI, never this checkout.**
   ``import decimalai`` resolving to the repo would make every assertion below
   vacuous, so it is checked first and hard-fails. (That is not theoretical:
   running ``python -c "import decimalai"`` with the repo root as cwd picks the
   checkout even inside a venv that has the released wheel installed.)

2. **A claim is proved on the wire, not by stdout and not by "it didn't
   raise".** ``decimalai init`` prints "✓ Test trace sent successfully" from
   inside a ``try`` whose body only *queues* the trace — the message is printed
   whether or not anything lands. So the assertion here is that the ingest
   probe RECEIVED the trace and ACCEPTED it under the backend's own validator.

**No secrets, no backend, no provider key.** Both wires are local HTTP servers:

* ingest → ``tests/conformance/probe.py``, the same real ``http.server`` the
  conformance tier uses. Its ``validate_trace_payload`` is a port of the
  backend's trace-ingest validator, fingerprinted against the platform source,
  so "accepted" here means "production would have stored this row" — not
  "some bytes reached a socket";
* the model → a stub OpenAI Chat Completions service on 127.0.0.1 that the real
  ``openai`` client is pointed at with ``base_url=``. Every line of the real
  provider SDK still runs; only the inference is stubbed. A smoke gated on
  secrets silently no-ops, which is the exact failure mode this file exists to
  end.

What it runs, in the order the docs present it:

* ``decimalai --version`` / ``--help``           docs quickstart §1 checkpoint
* ``decimalai init``                             docs quickstart §2
* the "No framework" snippet (``@decimalai.trace`` + ``SkillRouter`` +
  ``log_llm_call``)                              docs quickstart §4

Usage::

    python scripts/pypi_smoke.py --expect-version 0.10.2

Exit status is 0 only if every check passed. Run it from ANY directory except
the repo root (rule 1); the workflow runs it from a temp dir.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import threading
import types
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

# The agent names the two documented snippets use. Not ours to choose — they are
# what the CLI and the docs page respectively put on the wire, and asserting the
# literal string is what makes "the trace we got is the trace the docs promised".
INIT_AGENT = "decimalai-init-test"          # decimalai/cli/main.py::init
DOC_AGENT = "support-agent"                 # docs quickstart §4, "No framework"
DOC_MODEL = "gpt-4o"                        # same snippet
DOC_QUESTION = "How do I reset my password?"

# One skill for the probe's router to offer, so the snippet's
# build_prompt_fragment() has something real to return.
SMOKE_SKILL = {
    "name": "password-reset-runbook",
    "description": "How to walk a customer through a password reset.",
    "body": "# Password reset\n\nSend the reset link, then confirm the mailbox.",
}


class SmokeFailure(AssertionError):
    """A user-facing claim the released artifact does not keep."""


# ── the stub model ────────────────────────────────────────────────────────────


class StubOpenAI:
    """A local HTTP server answering OpenAI Chat Completions.

    A stub *service*, not a monkeypatched client: the real ``openai`` package
    builds the request, opens the socket, parses the response and constructs its
    real response models. Pointing a client at it needs no key and no network.
    """

    #: The openai client refuses to construct without *a* key. Nothing reads it.
    api_key = "sk-" + "stub" * 3

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.requests: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "StubOpenAI":
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # noqa: A003 - quiet
                pass

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except Exception:
                    payload = {}
                with stub._lock:
                    stub.requests.append(payload)
                body = json.dumps({
                    "id": "chatcmpl-" + uuid.uuid4().hex[:20],
                    "object": "chat.completion",
                    "created": 0,
                    "model": payload.get("model", DOC_MODEL),
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": stub.reply},
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 7,
                        "total_tokens": 18,
                    },
                }).encode("utf-8")
                self.send_response(200)
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
        assert self._server is not None, "stub model not started"
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"


# ── loading the conformance probe without importing the checkout ──────────────


def load_probe_module(repo_root: Path) -> types.ModuleType:
    """Import ``tests/conformance/probe.py`` by path.

    By path, and NOT by putting ``repo_root`` on ``sys.path``: the probe is the
    one thing we want from the checkout, and adding the repo to the path would
    also shadow the installed ``decimalai`` with the source tree — the exact
    substitution this smoke exists to rule out. ``probe.py`` imports nothing but
    the standard library, so loading it in isolation is honest.
    """
    path = repo_root / "tests" / "conformance" / "probe.py"
    if not path.is_file():
        raise SmokeFailure(
            f"conformance probe not found at {path}. Pass --repo-root pointing at "
            "a decimalai-python checkout (the smoke needs the probe, not the SDK, "
            "from source)."
        )
    spec = importlib.util.spec_from_file_location("decimalai_smoke_probe", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise SmokeFailure(f"could not load a module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves annotations through
    # sys.modules[cls.__module__], and blows up on a module that is not there yet.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ── checks ────────────────────────────────────────────────────────────────────


class Smoke:
    def __init__(self, expect_version: str, repo_root: Path, require_manifest: bool):
        self.expect_version = expect_version
        self.repo_root = repo_root.resolve()
        self.require_manifest = require_manifest
        self.failures: List[str] = []
        self.passed: List[str] = []

    # -- reporting -------------------------------------------------------

    def ok(self, label: str, detail: str = "") -> None:
        self.passed.append(label)
        print(f"  PASS  {label}" + (f" — {detail}" if detail else ""), flush=True)

    def fail(self, label: str, detail: str) -> None:
        self.failures.append(f"{label}: {detail}")
        print(f"  FAIL  {label} — {detail}", flush=True)

    def check(self, label: str, condition: bool, detail: str) -> bool:
        if condition:
            self.ok(label)
        else:
            self.fail(label, detail)
        return condition

    # -- phase 1: is this really the released artifact? ------------------

    def phase_identity(self) -> None:
        print("\n[1/3] The artifact under test", flush=True)
        import importlib.metadata as md

        import decimalai

        origin = Path(decimalai.__file__).resolve()
        # Hard-fail, not a soft check: every later assertion is vacuous if the
        # import came from the checkout, so there is nothing to collect.
        if self.repo_root in origin.parents:
            raise SmokeFailure(
                f"`import decimalai` resolved to the CHECKOUT ({origin}), not to an "
                "installed wheel. The smoke would then be testing source that was "
                "never published. Run it from a directory outside the repo, with "
                "the repo NOT on sys.path/PYTHONPATH."
            )
        self.ok("installed, not the checkout", str(origin))

        self.check(
            "decimalai.__version__ matches the tag",
            decimalai.__version__ == self.expect_version,
            f"expected {self.expect_version}, wheel reports {decimalai.__version__}",
        )
        dist_version = md.version("decimalai")
        self.check(
            "dist metadata matches the tag",
            dist_version == self.expect_version,
            f"expected {self.expect_version}, dist-info says {dist_version}",
        )
        # Same value from the two places that can disagree: a wheel built from a
        # commit that never got its __version__ bump ships a dist-info version
        # the module contradicts, and only one of them is what users print.
        self.check(
            "__version__ agrees with dist metadata",
            decimalai.__version__ == dist_version,
            f"module {decimalai.__version__} vs dist-info {dist_version}",
        )

    def phase_cli(self, workdir: Path) -> None:
        """docs quickstart §1 checkpoint: `decimalai --version` prints a version."""
        print("\n[2/3] The CLI the quickstart's first checkpoint runs", flush=True)
        proc = self._run_cli(["--version"], workdir)
        self.check(
            "`decimalai --version` exits 0",
            proc.returncode == 0,
            f"exit {proc.returncode}; stderr: {proc.stderr.strip()[:400]}",
        )
        self.check(
            "`decimalai --version` prints the released version",
            self.expect_version in proc.stdout,
            f"expected {self.expect_version!r} in {proc.stdout.strip()!r}",
        )
        proc = self._run_cli(["--help"], workdir)
        self.check(
            "`decimalai --help` exits 0",
            proc.returncode == 0,
            f"exit {proc.returncode}; stderr: {proc.stderr.strip()[:400]}",
        )
        # The docs put "demo" on the page as the thing a too-old Python silently
        # loses, so its presence in --help is itself a documented claim.
        self.check(
            "`decimalai --help` lists the documented `demo` command",
            "demo" in proc.stdout,
            "no `demo` command in --help output",
        )

    def _run_cli(
        self, args: List[str], workdir: Path, env: Optional[Dict[str, str]] = None
    ) -> "subprocess.CompletedProcess[str]":
        """Run the installed console script, never `python -m` from a checkout."""
        return subprocess.run(
            ["decimalai", *args],
            cwd=str(workdir),
            env={**os.environ, **(env or {})},
            capture_output=True,
            text=True,
            timeout=180,
        )

    # -- phase 3: the documented quickstart, end to end ------------------

    def phase_quickstart(self, probe_mod: types.ModuleType, workdir: Path) -> None:
        print("\n[3/3] The documented quickstart, against a real wire", flush=True)
        probe = probe_mod.Probe(require_manifest_on_ingest=self.require_manifest).start()
        stub = StubOpenAI(reply="Open Settings → Security and choose Reset password.").start()
        try:
            self._quickstart_step2(probe, workdir)
            self._quickstart_step4(probe, stub)
        finally:
            stub.stop()
            probe.stop()

    def _accepted_traces(self, probe: Any, cursor: int, agent: str) -> List[Dict[str, Any]]:
        """Trace payloads the probe accepted for ``agent`` since ``cursor``.

        "Accepted" is the load-bearing word: the probe runs the backend's own
        validator, so a payload that left the SDK but would 400 in production is
        NOT counted. That is the difference between "the code did not raise" and
        "a trace was really produced".
        """
        out: List[Dict[str, Any]] = []
        for rec in probe.since(cursor):
            if rec.method != "POST":
                continue
            if rec.path not in ("/api/v1/traces", "/api/v1/traces/batch"):
                continue
            if not rec.accepted:
                continue
            bodies = rec.body if isinstance(rec.body, list) else [rec.body]
            out.extend(
                b for b in bodies
                if isinstance(b, dict) and b.get("agent_name") == agent
            )
        return out

    def _rejections(self, probe: Any, cursor: int) -> List[str]:
        """Why the backend would have refused whatever the SDK did send."""
        reasons: List[str] = []
        for rec in probe.since(cursor):
            if rec.method == "POST" and rec.path.startswith("/api/v1/traces"):
                if not rec.accepted:
                    reasons.extend(rec.errors or [f"HTTP {rec.status}"])
        return reasons

    def _quickstart_step2(self, probe: Any, workdir: Path) -> None:
        """docs quickstart §2 — `decimalai init` sends your first trace."""
        cursor = probe.mark()
        env = {
            "DECIMAL_API_KEY": probe.api_key,
            "DECIMAL_BASE_URL": probe.base_url,
            # The CLI's closing links resolve through this; pinning it keeps the
            # smoke off the public internet entirely.
            "DECIMAL_APP_URL": "http://127.0.0.1:3000",
        }
        proc = self._run_cli(["init"], workdir, env=env)
        out = proc.stdout + proc.stderr
        self.check(
            "`decimalai init` exits 0",
            proc.returncode == 0,
            f"exit {proc.returncode}; output: {out.strip()[:600]}",
        )
        # The three checkmarks the docs page shows as its checkpoint.
        self.check(
            "`decimalai init` prints the API-key check",
            "✓ API key:" in out,
            f"missing '✓ API key:' in: {out.strip()[:400]}",
        )
        self.check(
            "`decimalai init` prints the connectivity check",
            "✓ Connected" in out,
            f"missing '✓ Connected' in: {out.strip()[:400]}",
        )
        self.check(
            "`decimalai init` prints the test-trace check",
            "✓ Test trace sent successfully" in out,
            f"missing '✓ Test trace sent successfully' in: {out.strip()[:400]}",
        )
        # …and now the part stdout cannot prove. The CLI prints that line from a
        # `try` whose body only queues the trace, so this is the check that would
        # actually catch a quickstart that has stopped producing traces.
        traces = self._accepted_traces(probe, cursor, INIT_AGENT)
        if not self.check(
            "the init trace REACHED the ingest wire and was accepted",
            bool(traces),
            "the probe accepted no trace for agent "
            f"{INIT_AGENT!r}"
            + (
                "; the backend validator rejected: " + "; ".join(self._rejections(probe, cursor))
                if self._rejections(probe, cursor)
                else " and nothing was POSTed at all"
            ),
        ):
            return
        self.ok(
            f"init trace payload ({len(traces)} accepted)",
            f"agent_name={traces[0].get('agent_name')!r} id={traces[0].get('id')!r}",
        )

    def _quickstart_step4(self, probe: Any, stub: StubOpenAI) -> None:
        """docs quickstart §4, "No framework" tab — the snippet users copy.

        Kept as close to the published text as a hermetic run allows: the only
        edits are the two ``base_url=`` arguments that point the SDK at the probe
        and the real ``openai`` client at the stub model. The decorator, the
        router call, ``log_llm_call`` and the message assembly are verbatim.
        """
        from openai import OpenAI

        import decimalai
        from decimalai.skill_router import SkillRouter

        probe.skills = [dict(SMOKE_SKILL)]
        os.environ["DECIMAL_API_KEY"] = probe.api_key
        os.environ["DECIMAL_BASE_URL"] = probe.base_url
        cursor = probe.mark()

        decimalai.init(api_key=probe.api_key, base_url=probe.base_url)
        client = OpenAI(base_url=stub.base_url, api_key=stub.api_key)
        router = SkillRouter(
            api_key=probe.api_key, base_url=probe.base_url,
            strategy="auto", inject_body=True,
        )

        @decimalai.trace(agent_name=DOC_AGENT)
        def answer(question: str) -> str:
            skills, _ = router.build_prompt_fragment(query=question)
            messages = [
                {"role": "system", "content": "You are a support agent.\n\n" + skills},
                {"role": "user", "content": question},
            ]
            resp = client.chat.completions.create(model=DOC_MODEL, messages=messages)
            decimalai.log_llm_call(
                model=DOC_MODEL,
                input=messages,
                output={"content": resp.choices[0].message.content},
            )
            return resp.choices[0].message.content

        reply = answer(DOC_QUESTION)
        decimalai.flush()

        self.check(
            "the documented snippet returns the model's answer",
            bool(reply and reply.strip()),
            f"snippet returned {reply!r}",
        )
        self.check(
            "the real openai client reached the model",
            bool(stub.requests),
            "the stub model was never called",
        )

        # The skills claim on the same page: build_prompt_fragment swallows
        # errors and returns ("", None), so a dead skills rail looks exactly like
        # a healthy one from inside the snippet. Only the assembled prompt tells.
        sent = stub.requests[0] if stub.requests else {}
        system = ""
        for msg in sent.get("messages") or []:
            if msg.get("role") == "system":
                system = msg.get("content") or ""
        self.check(
            "the routed skill actually landed in the prompt",
            SMOKE_SKILL["name"] in system,
            "build_prompt_fragment contributed nothing to the system message "
            f"(got {system[:200]!r}) — the rail returns ('', None) on failure, "
            "so the snippet cannot notice this itself",
        )

        traces = self._accepted_traces(probe, cursor, DOC_AGENT)
        rejected = self._rejections(probe, cursor)
        if not self.check(
            "the snippet's trace REACHED the ingest wire and was accepted",
            bool(traces),
            f"the probe accepted no trace for agent {DOC_AGENT!r}"
            + ("; rejected: " + "; ".join(rejected) if rejected else " and nothing was POSTed"),
        ):
            return

        trace = traces[0]
        calls = [c for c in (trace.get("llm_calls") or []) if isinstance(c, dict)]
        logged = [
            c for c in calls
            # model_name is the wire field; `model` is what the snippet passes.
            # Accept either so a rename shows up as a rename, not as a silence.
            if DOC_MODEL in (c.get("model_name"), c.get("model"))
        ]
        # A trace that lands empty is the shape of a "traces still arrive" green
        # that a user would call broken, so check the payload carries what the
        # snippet logged — not merely that a row appeared.
        if self.check(
            "the trace carries the logged LLM call",
            bool(logged),
            f"no llm_call naming {DOC_MODEL!r}; got {json.dumps(calls)[:400]}",
        ):
            call = logged[0]
            output = call.get("output")
            self.check(
                "log_llm_call's output survived onto the wire",
                isinstance(output, dict) and reply in str(output.get("content") or ""),
                f"llm_call.output={json.dumps(output)[:300]} does not carry {reply!r}",
            )
            self.check(
                "the assembled prompt survived onto the wire",
                DOC_QUESTION in json.dumps(call.get("rendered_input"), ensure_ascii=False),
                "llm_call.rendered_input does not carry the question the snippet asked",
            )
        self.check(
            "the trace summarises the run the snippet performed",
            trace.get("user_input_preview") == DOC_QUESTION
            and reply in (trace.get("final_output_preview") or ""),
            f"user_input_preview={trace.get('user_input_preview')!r}, "
            f"final_output_preview={trace.get('final_output_preview')!r}",
        )
        # The page's own sentence about the router: "build_prompt_fragment stamps
        # the routing decision and the offered skill names onto the active trace
        # automatically — no extra logging calls."
        self.check(
            "the routing decision is stamped on the trace, as documented",
            bool(trace.get("routing_id")),
            "trace carries no routing_id — the documented automatic stamp did not happen",
        )
        self.check(
            "the offered skill names are stamped on the trace, as documented",
            SMOKE_SKILL["name"] in (trace.get("skills_offered_in_prompt") or []),
            f"skills_offered_in_prompt={trace.get('skills_offered_in_prompt')!r}",
        )

        status = decimalai.export_status()
        failed = status.get("failed") if isinstance(status, dict) else None
        self.check(
            "the SDK reports no export failures",
            not failed,
            f"export_status()={status!r}, last_send_error={decimalai.last_send_error()!r}",
        )
        self.ok(
            "snippet trace payload",
            f"agent_name={trace.get('agent_name')!r} llm_calls={len(calls)}",
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--expect-version", required=True,
        help="The version the release tag claims. Every version surface must match it.",
    )
    parser.add_argument(
        "--repo-root", default=str(Path(__file__).resolve().parents[1]),
        help="Checkout to take the conformance probe from (NOT the SDK under test).",
    )
    parser.add_argument(
        "--require-manifest", dest="require_manifest",
        action="store_true", default=True,
        help="Run the probe in the backend's production posture "
             "(require_manifest_on_ingest=True). Default.",
    )
    parser.add_argument(
        "--no-require-manifest", dest="require_manifest", action="store_false",
        help="Relax the manifest requirement (a self-hosted backend may set this).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    print(
        f"decimalai post-publish smoke — expecting {args.expect_version}\n"
        f"  probe from : {repo_root}\n"
        f"  cwd        : {Path.cwd()}\n"
        f"  python     : {sys.executable}",
        flush=True,
    )

    smoke = Smoke(args.expect_version, repo_root, args.require_manifest)
    try:
        probe_mod = load_probe_module(repo_root)
        smoke.phase_identity()
        workdir = Path.cwd()
        smoke.phase_cli(workdir)
        smoke.phase_quickstart(probe_mod, workdir)
    except SmokeFailure as exc:
        print(f"\nABORTED — {exc}", flush=True)
        return 2

    print(f"\n{len(smoke.passed)} checks passed, {len(smoke.failures)} failed", flush=True)
    if smoke.failures:
        print("\nThe released artifact does not keep these documented claims:", flush=True)
        for line in smoke.failures:
            print(f"  - {line}", flush=True)
        return 1
    print("The documented quickstart works, from PyPI, end to end.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
