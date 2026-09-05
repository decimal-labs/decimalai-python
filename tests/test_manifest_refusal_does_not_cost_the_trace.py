"""A refused manifest registration must not cost the trace.

Measured on production: 1,315 `POST /api/v1/traces` answered 400 in the 48 h to
2026-09-04T09:33Z, every one from `decimalai-sdk/0.13.1`, every one
"Trace validation failed: manifest_id '...' does not exist". `b974726` (in
0.13.1) stopped a refusal being CACHED for the life of the process, but the
trace in hand when the refusal happened still shipped the snapshot's LOCAL id,
and `require_manifest_on_ingest` rejects that -- so every refused registration
still cost exactly one trace.

The window it has to bridge is a Cloud Run revision with no available instance:
it answers 429 in ~0 ms, with no Retry-After, for tens of seconds. The
registration ladder in `langchain._register_snapshot` runs on the CALLER's
thread at root-run end, so it is deliberately 0.6 s of its own sleeping and
cannot bridge one.

Hermetic. The stub below is the production contract in miniature: `POST
/api/v1/manifests` aborts at admission for the first 30 (virtual) seconds and
then works, and `POST /api/v1/traces` 400s on any manifest_id it never stored.
`time.sleep` is replaced by a virtual clock, so the 30 s window costs no real
time and the assertion is on the SERVER's counters, not on timing.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

import pytest

import decimalai
import decimalai._config as cfg
import decimalai.langchain as lc
from decimalai.schema.manifest import ManifestTracker

# The stub's clock. Every `time.sleep` in the process advances it instead of
# waiting, so a 30 s outage runs in milliseconds and stays deterministic.
_CLOCK = [0.0]
# A list, not a bare float: the handler reads it at request time and two of
# the tests move the window without restarting the server.
_WINDOW = [30.0]


class _Prod(BaseHTTPRequestHandler):
    """Cloud Run + `require_manifest_on_ingest`, in miniature."""

    stored: set = set()
    rejected: list = []
    accepted: list = []
    aborted: list = []

    def log_message(self, *a):  # noqa: A003 - silence the stdlib access log
        pass

    def _reply(self, code, body):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"null")

        if self.path.startswith("/api/v1/manifests"):
            if _CLOCK[0] < _WINDOW[0]:
                # Verbatim shape of the production abort: 429, no Retry-After.
                type(self).aborted.append(_CLOCK[0])
                return self._reply(429, {
                    "detail": "The request was aborted because there was no "
                              "available instance."})
            type(self).stored.add("mf-real")
            return self._reply(200, {"manifest_id": "mf-real", "status": "created"})

        if self.path.startswith("/api/v1/traces"):
            mid = (payload or {}).get("manifest_id")
            if mid not in type(self).stored:
                type(self).rejected.append(mid)
                return self._reply(400, {
                    "detail": "Trace validation failed: manifest_id "
                              f"'{mid}' does not exist"})
            type(self).accepted.append(payload.get("id"))
            return self._reply(200, {"status": "ok", "trace_id": payload.get("id")})

        return self._reply(404, {"detail": "not found"})

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/v1/manifests"):
            return self._reply(200, {"manifests": []})
        return self._reply(404, {"detail": "not found"})


@pytest.fixture
def prod(monkeypatch):
    _CLOCK[0] = 0.0
    _WINDOW[0] = 30.0
    _Prod.stored = set()
    _Prod.rejected = []
    _Prod.accepted = []
    _Prod.aborted = []

    def _virtual_sleep(seconds):
        _CLOCK[0] += float(seconds or 0.0)

    monkeypatch.setattr("time.sleep", _virtual_sleep)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Prod)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"

    decimalai.init(api_key="dai_sk_test", base_url=base, verify=False)

    monkeypatch.setattr(lc, "_manifest_id", None)
    monkeypatch.setattr(lc, "_manifest_tracker", ManifestTracker())
    monkeypatch.setattr(lc, "_manifest_ids", {})
    monkeypatch.setattr(lc, "_manifest_hashes", {})
    monkeypatch.setattr(lc, "_unregistered_agents", set())
    monkeypatch.setattr(lc, "_manifest_adoption_probed", set())
    monkeypatch.setattr(lc, "_explicit_manifest_config", None)
    for name in ("_pending_snapshots", "_last_registration_error"):
        if hasattr(lc, name):
            monkeypatch.setattr(lc, name, {})

    yield _Prod

    srv.server_close()
    cfg._config = None
    cfg._client = None


def _one_run(agent="support-bot"):
    """One root run with a model to declare -- the shape that registers."""
    state = lc._RunState(uuid4())
    state.agent_hint = agent
    state.seen_model = {"provider": "google", "model": "gemini-3.6-flash"}
    return state


def test_a_restart_window_costs_no_trace_and_produces_no_400(prod):
    """THE bug. The whole run happens inside a 30 s admission-abort window.

    RED before the fix: `_register_snapshot`'s 0.6 s ladder is exhausted well
    inside the window, `_auto_send` posts the trace under the snapshot's local
    id, and the stub answers 400 exactly as production does -- `rejected` is
    non-empty and `accepted` is empty.
    """
    handler = lc.CallbackHandler(auto_send=False, agent_name="support-bot")
    handler._auto_send(_one_run())
    cfg._sender.flush(timeout=60.0)

    assert prod.aborted, "the stub never actually refused a registration"
    assert prod.rejected == [], (
        "the SDK posted a trace under a manifest_id the platform never stored; "
        f"the platform answered 400 for {prod.rejected}"
    )
    assert len(prod.accepted) == 1, "the trace was lost instead of held"
    assert lc._manifest_ids["support-bot"] == "mf-real"
    assert "support-bot" not in lc._unregistered_agents


def test_a_registration_that_never_recovers_still_never_posts_a_bad_id(prod):
    """The other end of the ladder: an outage longer than the SDK will wait.

    The trace is lost either way -- but it must not be spent on a request whose
    400 is a certainty, and the failure must land on the sender so
    `export_status()` names it.
    """
    _WINDOW[0] = 1e9
    handler = lc.CallbackHandler(auto_send=False, agent_name="support-bot")
    handler._auto_send(_one_run())
    cfg._sender.flush(timeout=60.0)
    assert prod.rejected == [], "a doomed id was posted anyway"
    assert prod.accepted == []
    status = decimalai.export_status()
    assert status.last_manifest_error, "the registration failure is not observable"


def test_a_healthy_platform_is_untouched(prod):
    """Guard against over-correcting: with no outage the send path is the old one."""
    _WINDOW[0] = 0.0
    handler = lc.CallbackHandler(auto_send=False, agent_name="support-bot")
    handler._auto_send(_one_run())
    cfg._sender.flush(timeout=30.0)
    assert prod.aborted == []
    assert prod.rejected == []
    assert len(prod.accepted) == 1


# ── the generic tracer takes the same route ────────────────────────
#
# `@decimalai.trace` / `start_trace` is the adapter the DecimalAI fleet runs on
# EVERY session (fleet/workloads/session.py:3927), langchain or not, so it is
# the larger half of the production 400s. Its failure branch stamps the
# snapshot's synthetic id and rolls the tracker back -- the next trace
# re-registers, but the trace in hand is still spent on a certain 400.


@pytest.fixture
def generic_state(monkeypatch):
    import decimalai.generic as g

    monkeypatch.setattr(g, "_manifest_id", None)
    monkeypatch.setattr(g, "_manifest_tracker", ManifestTracker())
    for name in ("_pending_snapshot", "_last_registration_error"):
        if hasattr(g, name):
            monkeypatch.setattr(g, name, None)
    return g


def test_the_generic_tracer_also_holds_the_trace(prod, generic_state):
    with decimalai.start_trace(agent_name="native-bot") as t:
        t.log_llm_call(
            model="gemini-3.6-flash",
            input=[{"role": "user", "content": "hi"}],
            output={"role": "assistant", "content": "hello"},
        )
    cfg._sender.flush(timeout=60.0)

    assert prod.aborted, "the stub never actually refused a registration"
    assert prod.rejected == [], (
        "the generic tracer posted a trace under a synthetic manifest_id; "
        f"the platform answered 400 for {prod.rejected}"
    )
    assert len(prod.accepted) == 1, "the trace was lost instead of held"
    assert generic_state._manifest_id == "mf-real"
