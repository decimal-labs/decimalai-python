"""A failed manifest registration on the OTel path must not cost the trace.

Seen on production 2026-11-17 17:54Z: requests to api.decimal.ai timed out for
~70 minutes. The prober's `agent-framework-openai` probe (decimalai 0.29.0)
logged "could not register ... manifest from OTel spans (ReadTimeout ...)", and
the trace that followed was answered 400 "Trace validation failed: manifest_id
'...' does not exist" and never ingested.

`_ManifestRegistry.manifest_id_for` already forgot the failed hash so the NEXT
trace re-registers, but it stamped the trace in hand with the snapshot's own
client-side uuid, which the platform has never stored. Sending no manifest_id
instead is no better: `require_manifest_on_ingest` answers that 400 "'manifest_id'
is required". So every failed registration cost exactly one trace. LangChain,
the generic tracer and the four other adapters stopped this in September by
HOLDING the trace on the background sender and re-registering there
(`_config.submit_trace_pending_manifest`, see
`test_manifest_refusal_does_not_cost_the_trace.py`); the OTel exporter was the
one adapter left out.

Hermetic, and through the real thing on both ends: a real OpenTelemetry
`TracerProvider` feeds `DecimalSpanExporter`, and a real `DecimalAIClient` talks
HTTP to the stub below, which is the production contract in miniature. For the
first 30 (virtual) seconds `POST /api/v1/manifests` either never answers (the
client's read timeout fires: the 2026-11-17 failure) or is aborted at admission
(429, no Retry-After: a Cloud Run revision with no instance), and then it works.
`POST /api/v1/traces` 400s on any manifest_id it never stored, and on none.
`time.sleep` is a virtual clock, so the window costs no real time; only the
stalled requests' 0.2 s read timeouts do.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

import decimalai
import decimalai._config as cfg
from decimalai._client import _MAX_RETRIES
from decimalai.otel import DecimalSpanExporter

_CLOCK = [0.0]
_WINDOW = [30.0]
# "stall": a registration is never answered, so the client's read timeout fires.
# "refuse": a 429 abort. "unauthorized": a 401, which no amount of waiting fixes.
_MODE = ["stall"]
# The client's read timeout in these tests, and how long a stalled request is
# left hanging (comfortably longer, so the timeout is what ends it).
_CLIENT_TIMEOUT_S = 0.2
_STALL_S = 1.0

# The caller's own attempt (the client's ladder), then one request per step of
# the sender's ladder. A timeout is not retried by the client, so it sends fewer.
_REGISTRATIONS_PER_HELD_TRACE = (_MAX_RETRIES + 1) + len(cfg._MANIFEST_RETRY_BACKOFFS_S)


class _Prod(BaseHTTPRequestHandler):
    """api.decimal.ai + `require_manifest_on_ingest`, in miniature."""

    stored: set = set()
    registrations: list = []
    failed: list = []
    rejected: list = []
    accepted: list = []

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
            type(self).registrations.append(_CLOCK[0])
            if _CLOCK[0] < _WINDOW[0]:
                type(self).failed.append(_CLOCK[0])
                if _MODE[0] == "stall":
                    # Never answered: the request was lost on the way. A real
                    # wait, not time.sleep, which is the virtual clock here.
                    threading.Event().wait(_STALL_S)
                    return None
                if _MODE[0] == "unauthorized":
                    return self._reply(401, {"detail": "Invalid API key"})
                return self._reply(429, {
                    "detail": "The request was aborted because there was no "
                              "available instance."})
            type(self).stored.add("mf-real")
            return self._reply(200, {"manifest_id": "mf-real", "status": "created"})

        if self.path.startswith("/api/v1/traces"):
            mid = (payload or {}).get("manifest_id")
            if mid not in type(self).stored:
                type(self).rejected.append(mid)
                detail = (
                    f"Trace validation failed: manifest_id '{mid}' does not exist"
                    if mid else "Trace validation failed: 'manifest_id' is required."
                )
                return self._reply(400, {"detail": detail})
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
    _MODE[0] = "stall"
    _Prod.stored = set()
    _Prod.registrations = []
    _Prod.failed = []
    _Prod.rejected = []
    _Prod.accepted = []

    def _virtual_sleep(seconds):
        _CLOCK[0] += float(seconds or 0.0)

    monkeypatch.setattr("time.sleep", _virtual_sleep)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Prod)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    decimalai.init(
        api_key="dai_sk_test", base_url=f"http://127.0.0.1:{srv.server_port}", verify=False
    )
    cfg._client._http.timeout = httpx.Timeout(_CLIENT_TIMEOUT_S)
    # A previous test's failure must not satisfy this one's assertion on it.
    monkeypatch.setattr(cfg._sender, "_last_manifest_error", None)

    yield _Prod

    srv.shutdown()
    srv.server_close()
    cfg._config = None
    cfg._client = None


def _tracer():
    """A real provider feeding a fresh exporter, as `otel.instrument()` wires it."""
    exporter = DecimalSpanExporter(agent_name="support-bot")
    provider = TracerProvider()
    # Simple, so a run's spans are exported the moment its root span ends.
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("agent_framework")


def _one_run(tracer):
    """One agent run that made a model call, so it has a model to declare."""
    with tracer.start_as_current_span("invoke_agent support-bot", attributes={
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": "support-bot",
    }):
        with tracer.start_as_current_span("chat gpt-5.4-mini", attributes={
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "openai",
            "gen_ai.request.model": "gpt-5.4-mini",
        }):
            pass


@pytest.mark.parametrize("mode", ["stall", "refuse"])
def test_a_failed_registration_costs_no_trace_and_produces_no_400(prod, mode):
    """THE bug. The run ends inside the 30 s window, so its registration fails.

    RED before the fix: the trace was posted at once under the snapshot's local
    id and the stub answered 400 exactly as production does, so `rejected` held
    that id and `accepted` was empty.
    """
    _MODE[0] = mode
    tracer = _tracer()
    _one_run(tracer)
    cfg._sender.flush(timeout=60.0)

    assert prod.failed, "the stub never actually failed a registration"
    assert prod.rejected == [], (
        "the OTel exporter posted a trace under a manifest_id the platform never "
        f"stored; the platform answered 400 for {prod.rejected}"
    )
    assert len(prod.accepted) == 1, "the trace was lost instead of held"

    # The background registration counts for the agent's later traces too: the
    # next run ships at once under the stored id, with no registration of its own.
    before = len(prod.registrations)
    _one_run(tracer)
    cfg._sender.flush(timeout=60.0)
    assert len(prod.accepted) == 2 and prod.rejected == []
    assert len(prod.registrations) == before, "an unchanged manifest was registered again"


@pytest.mark.parametrize("mode", ["stall", "refuse"])
def test_a_registration_that_never_recovers_never_posts_a_bad_id(prod, mode):
    """An outage longer than the SDK will wait. The trace is lost either way, but
    it must not be spent on a request whose 400 is a certainty, one trace must
    send a bounded number of registrations, and the failure must be observable."""
    _MODE[0] = mode
    _WINDOW[0] = 1e9
    _one_run(_tracer())
    cfg._sender.flush(timeout=60.0)

    assert prod.rejected == [], "a doomed manifest_id was posted anyway"
    assert prod.accepted == []
    assert 1 < len(prod.registrations) <= _REGISTRATIONS_PER_HELD_TRACE, (
        f"one held trace sent {len(prod.registrations)} registrations; the bound "
        f"is {_REGISTRATIONS_PER_HELD_TRACE}"
    )
    status = decimalai.export_status()
    assert status.last_manifest_error, "the registration failure is not observable"
    assert status.last_error and "NOT sent" in status.last_error


def test_a_credential_refusal_is_not_retried(prod):
    """401 will not become a different answer by waiting: one registration, no
    trace posted."""
    _MODE[0] = "unauthorized"
    _WINDOW[0] = 1e9
    _one_run(_tracer())
    cfg._sender.flush(timeout=60.0)

    assert prod.rejected == [] and prod.accepted == []
    assert len(prod.registrations) == 1


def test_a_healthy_platform_is_untouched(prod):
    """Guard against over-correcting: with no outage the trace goes straight out,
    with one registration and no waiting."""
    _WINDOW[0] = 0.0
    _one_run(_tracer())
    cfg._sender.flush(timeout=30.0)

    assert prod.failed == [] and prod.rejected == []
    assert len(prod.registrations) == 1
    assert len(prod.accepted) == 1
    assert _CLOCK[0] == 0.0, "the trace was held although its registration succeeded"
