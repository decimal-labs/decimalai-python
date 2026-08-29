"""The wire probe — a real HTTP server that behaves like the ingest backend.

Every adapter under conformance talks to THIS over real HTTP (httpx → socket →
``http.server``). Nothing is mocked, monkeypatched, or transport-swapped: if the
adapter does not actually make the request, the probe does not see it, and the
contract fails. That is the whole point — the repo already has 626 adapter tests
that drive a fake and therefore only ever proved "it called our fake".

Two jobs:

1. **Record.** Every request (method, path, query, headers-of-interest, parsed
   body) lands in ``Probe.requests`` in arrival order. ``Probe.mark()`` returns a
   cursor so the harness can slice one driver phase's traffic out of the log.

2. **Reject exactly what the backend rejects.** ``validate_trace_payload`` is a
   port of ``the platform's trace-ingest validator``
   (plus the manifest-existence / trace-id checks that ``ingest_trace`` does
   right after it). "The backend 400s this" is the defect class the mock-driven
   suite keeps missing — a trace that leaves the SDK but never lands is
   indistinguishable from a trace that landed, unless something on the far end
   applies the real rules.

Keeping the port honest: ``BACKEND_VALIDATOR_SHA256`` fingerprints the backend
functions this file mirrors. ``test_conformance.py`` re-derives it whenever the
platform repo is present on disk and fails when it drifts, so a backend rule
change cannot silently make this probe more permissive than production.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, unquote, urlparse

# ── Ported constants (backend trust-boundary allowlists) ──────────────────────

TRACE_STATUS_ALLOWED = frozenset({"success", "error", "degraded"})
TRACE_SOURCE_TYPE_ALLOWED = frozenset({
    "production", "test", "evaluation", "sdk", "manual", "synthetic",
    "development", "sandbox", "distillation", "snapshot", "file", "url",
    "eval_replay", "sample", "demo",
})
TRACE_TIMESTAMP_MAX_PAST_DAYS = 365 * 5
TRACE_TIMESTAMP_MAX_FUTURE_DAYS = 1

# sha256 over the backend source of `validate_element_shapes` + `_validate_payload`
# + the four allowlist/bound constants they read (see
# test_conformance.py::test_backend_validator_has_not_drifted for the exact
# recipe). Recorded 2026-08-15 against
# the platform's trace-ingest validator.
#
# Scope caveat, stated plainly: the manifest-exists / trace-id-shape /
# duplicate-id rules ported at the bottom of validate_trace_payload live inside
# the 200-line `ingest_trace`, so they are NOT covered by this fingerprint —
# hashing that whole function would churn on every unrelated edit. Re-read them
# by hand when the guard fires.
BACKEND_VALIDATOR_SHA256 = "a319c8d1d8c8add38441f7eca043e75c74d79d5cfd07f5a7dc2b8ddf054af6b1"


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return None


def _short_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _is_uuid(value: Any) -> bool:
    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def validate_trace_payload(
    payload: Any,
    *,
    require_manifest: bool = True,
    known_manifest_ids: Optional[set] = None,
    known_trace_ids: Optional[set] = None,
) -> List[str]:
    """Return the list of reasons the backend would reject this trace.

    Empty list == the backend would accept it. Port of
    ``trace_service._validate_payload`` + ``validate_element_shapes`` + the
    manifest-existence / trace-id-shape / duplicate-id checks at the top of
    ``trace_service.ingest_trace``.
    """
    errors: List[str] = []

    if not isinstance(payload, dict):
        return [f"trace payload must be an object, got {type(payload).__name__}"]

    # validate_element_shapes (backend: 422)
    for field_name in ("spans", "llm_calls"):
        value = payload.get(field_name)
        if value is None:
            continue
        if not isinstance(value, list):
            errors.append(
                f"'{field_name}' must be a list of objects, got {type(value).__name__}"
            )
            continue
        for i, element in enumerate(value):
            if not isinstance(element, dict):
                errors.append(
                    f"{field_name}[{i}] must be an object, got {type(element).__name__}"
                )
    if errors:
        return errors

    if require_manifest and not payload.get("manifest_id"):
        errors.append(
            "'manifest_id' is required. Register your agent's manifest first — "
            "POST /api/v1/manifests"
        )

    agent_name = payload.get("agent_name")
    if not agent_name:
        errors.append("'agent_name' is required")
    elif not isinstance(agent_name, str):
        errors.append("'agent_name' must be a string")
    elif agent_name.strip() == "":
        errors.append("'agent_name' must not be empty or whitespace-only")
    elif len(agent_name) > 255:
        errors.append(f"'agent_name' must be ≤255 characters (got {len(agent_name)})")
    elif "\x00" in agent_name:
        errors.append("'agent_name' must not contain null bytes")

    status_val = payload.get("status")
    if status_val is not None and (
        not isinstance(status_val, str) or status_val not in TRACE_STATUS_ALLOWED
    ):
        errors.append(
            f"'status' must be one of {sorted(TRACE_STATUS_ALLOWED)} (got {status_val!r})"
        )

    source_type_val = payload.get("source_type")
    if source_type_val is not None and (
        not isinstance(source_type_val, str)
        or source_type_val not in TRACE_SOURCE_TYPE_ALLOWED
    ):
        errors.append(
            f"'source_type' must be one of {sorted(TRACE_SOURCE_TYPE_ALLOWED)} "
            f"(got {source_type_val!r})"
        )

    started_raw = payload.get("started_at")
    ended_raw = payload.get("ended_at")
    try:
        started_dt = _parse_dt(started_raw) if started_raw else None
    except (ValueError, TypeError):
        started_dt = None
    try:
        ended_dt = _parse_dt(ended_raw) if ended_raw else None
    except (ValueError, TypeError):
        ended_dt = None

    if started_dt and ended_dt and ended_dt < started_dt:
        errors.append(
            f"'ended_at' ({ended_raw}) must not precede 'started_at' ({started_raw})"
        )

    now = datetime.now(timezone.utc)
    past_floor = now - timedelta(days=TRACE_TIMESTAMP_MAX_PAST_DAYS)
    future_ceiling = now + timedelta(days=TRACE_TIMESTAMP_MAX_FUTURE_DAYS)
    for field_name, raw, parsed in (
        ("started_at", started_raw, started_dt),
        ("ended_at", ended_raw, ended_dt),
    ):
        if parsed is None:
            continue
        parsed_utc = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        if parsed_utc < past_floor:
            errors.append(
                f"'{field_name}' ({raw}) is more than "
                f"{TRACE_TIMESTAMP_MAX_PAST_DAYS} days in the past"
            )
        elif parsed_utc > future_ceiling:
            errors.append(
                f"'{field_name}' ({raw}) is more than "
                f"{TRACE_TIMESTAMP_MAX_FUTURE_DAYS} day(s) in the future"
            )

    if "eval_score" in payload:
        errors.append(
            "'eval_score' is not accepted on trace ingest — pass per-evaluator "
            "scores via 'eval_scores' instead"
        )

    manifest_id_val = payload.get("manifest_id")
    if manifest_id_val:
        if not isinstance(manifest_id_val, str):
            errors.append(
                f"'manifest_id' must be a string (got {type(manifest_id_val).__name__})"
            )
        elif not _is_uuid(manifest_id_val):
            errors.append(
                f"'manifest_id' is not a valid UUID (got {manifest_id_val!r}) — "
                f"expected 8-4-4-4-12 hex format"
            )

    llm_calls = payload.get("llm_calls")
    if llm_calls and isinstance(llm_calls, list):
        for i, call in enumerate(llm_calls):
            if not call.get("model_name"):
                errors.append(f"llm_calls[{i}]: 'model_name' is required")

    for i, span in enumerate(payload.get("spans", []) or []):
        if not span.get("name"):
            errors.append(f"spans[{i}]: 'name' is required")
        if not span.get("span_type"):
            errors.append(f"spans[{i}]: 'span_type' is required")
        if not span.get("started_at"):
            errors.append(f"spans[{i}]: 'started_at' is required")
        if not span.get("ended_at"):
            errors.append(f"spans[{i}]: 'ended_at' is required")

    # ── ingest_trace()'s own checks, which run right after _validate_payload ──
    if require_manifest and manifest_id_val and known_manifest_ids is not None:
        if manifest_id_val not in known_manifest_ids:
            errors.append(f"manifest_id '{manifest_id_val}' does not exist")

    trace_id = payload.get("id")
    if trace_id and not _is_uuid(trace_id):
        errors.append(
            f"'id' is not a valid UUID (got {trace_id!r}) — expected 8-4-4-4-12 hex format"
        )
    if trace_id and known_trace_ids is not None and str(trace_id) in known_trace_ids:
        errors.append(f"Trace '{trace_id}' already exists")

    return errors


# ── Recording ────────────────────────────────────────────────────────────────


@dataclass
class Recorded:
    """One HTTP request that reached the probe."""

    seq: int
    method: str
    path: str
    query: Dict[str, List[str]]
    body: Any
    status: int
    response: Any = None
    errors: List[str] = field(default_factory=list)
    thread: str = ""

    @property
    def accepted(self) -> bool:
        return 200 <= self.status < 300


class Probe:
    """A real HTTP server standing in for the DecimalAI ingest API."""

    api_key = "dai_sk_conformance_probe_key"

    def __init__(self, *, require_manifest_on_ingest: bool = True) -> None:
        self.require_manifest = require_manifest_on_ingest
        self.requests: List[Recorded] = []
        # manifest_id -> {"agent_name", "manifest_hash", "version_label", "seq"}
        self.manifests: Dict[str, Dict[str, Any]] = {}
        self.trace_ids: set = set()
        # Skills the probe's router offers. Populated by the harness so the
        # skills rail is exercisable with no platform and no provider key.
        self.skills: List[Dict[str, Any]] = []
        # routing_id -> the query text the routing decision was made FOR.
        # This is the provenance that makes a leaked routing decision
        # deterministically detectable: if run A's trace carries a routing_id
        # the probe minted for run B's query, the rail crossed runs.
        self.routing_queries: Dict[str, Optional[str]] = {}
        # Agents this workspace holds — the platform state `decimalai init`
        # resolves a name against, reads a prompt from and lists skills for.
        # Populated by the JOURNEY tier (`journey.py`) via `register_agent`;
        # empty for every driver run, which is why the adapter matrix sees no
        # change. agent_name -> the row below.
        self.agents: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ────────────────────────────────────────────

    def start(self) -> "Probe":
        probe = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # noqa: A003 - silence stderr
                pass

            def _read_body(self) -> Any:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                if not raw:
                    return None
                try:
                    return json.loads(raw.decode("utf-8"))
                except Exception:
                    return raw.decode("utf-8", "replace")

            def _respond(self, status: int, payload: Any) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _handle(self, method: str) -> None:
                parsed = urlparse(self.path)
                body = self._read_body()
                status, payload, errors = probe.route(
                    method, parsed.path, parse_qs(parsed.query), body
                )
                probe._record(
                    method, parsed.path, parse_qs(parsed.query), body, status,
                    payload, errors,
                )
                self._respond(status, payload)

            def do_GET(self) -> None:  # noqa: N802
                self._handle("GET")

            def do_POST(self) -> None:  # noqa: N802
                self._handle("POST")

            def do_PUT(self) -> None:  # noqa: N802
                self._handle("PUT")

            def do_DELETE(self) -> None:  # noqa: N802
                self._handle("DELETE")

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

    def __enter__(self) -> "Probe":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    @property
    def base_url(self) -> str:
        assert self._server is not None, "probe not started"
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    # ── recording / slicing ──────────────────────────────────

    def _record(
        self,
        method: str,
        path: str,
        query: Dict[str, List[str]],
        body: Any,
        status: int,
        response: Any,
        errors: List[str],
    ) -> None:
        with self._lock:
            self.requests.append(
                Recorded(
                    seq=len(self.requests),
                    method=method,
                    path=path,
                    query=query,
                    body=body,
                    status=status,
                    response=response,
                    errors=errors,
                    thread=threading.current_thread().name,
                )
            )

    def mark(self) -> int:
        """A cursor into the request log — pass to :meth:`since`."""
        with self._lock:
            return len(self.requests)

    def since(self, cursor: int) -> List[Recorded]:
        with self._lock:
            return list(self.requests[cursor:])

    # ── routing ──────────────────────────────────────────────

    def route(
        self, method: str, path: str, query: Dict[str, List[str]], body: Any
    ) -> Tuple[int, Any, List[str]]:
        """Dispatch one request. Returns ``(status, response_json, errors)``."""
        if method == "GET" and path == "/api/v1/auth/verify":
            return 200, {
                "status": "ok",
                "api_key_valid": True,
                "scope": "org",
                "project_id": None,
                "workspace_id": None,
                "permissions": {},
                "require_manifest_on_ingest": self.require_manifest,
                "message": "conformance probe",
            }, []

        # ── agents (the surface `decimalai init` resolves against) ──
        #
        # Three routes, because `decimalai init <name>` makes exactly three
        # calls and the generated file makes a fourth. Modelled from the
        # backend's own handlers rather than invented — see
        # the platform's agent handlers: `list_agents`,
        # `list_agent_skills` (:2563) and `get_agent_prompt` (:3399, whose
        # response shape is `_GET_PROMPT_EXAMPLE` at :3382).
        #
        # Deliberately NO pagination on the list. The route returns the full
        # set when `limit` is unset, and `init` never sends one on purpose —
        # its comment says a `limit` "activates a pagination path that
        # truncates", whose ordering drops the never-traced UI-created agent
        # the command exists to find. A probe that paginated anyway would make
        # this tier greener than production.
        if method == "GET" and path == "/api/v1/agents":
            with self._lock:
                rows = [self._agent_row(a) for a in self.agents.values()]
            return 200, {"agents": rows, "total": len(rows)}, []

        m = re.fullmatch(r"/api/v1/agents/([^/]+)/skills", path)
        if method == "GET" and m:
            return self._agent_skills(unquote(m.group(1)))

        m = re.fullmatch(r"/api/v1/agents/([^/]+)/prompt", path)
        if method == "GET" and m:
            # No `If-None-Match` / 304 arm, and the omission is deliberate
            # rather than a shortcut: the handler does not receive headers, and
            # neither caller in this journey sends one — `load_agent()` never
            # does (a cache is what would break the no-redeploy property it is
            # sold on) and `init` does not either. Modelling a conditional
            # nobody sends would be untested probe code standing between the
            # SDK and its answer.
            return self._agent_prompt(unquote(m.group(1)), query)

        if method == "POST" and path == "/api/v1/traces":
            return self._ingest_one(body)

        if method == "POST" and path == "/api/v1/traces/batch":
            if not isinstance(body, list):
                return 400, {"detail": "batch body must be a list"}, ["batch body must be a list"]
            trace_ids, all_errors = [], []
            for item in body:
                status, payload, errs = self._ingest_one(item)
                all_errors.extend(errs)
                if status == 200:
                    trace_ids.append(payload.get("id"))
            return (
                200 if not all_errors else 400,
                {"count": len(trace_ids), "trace_ids": trace_ids, "errors": all_errors},
                all_errors,
            )

        if method == "POST" and path == "/api/v1/manifests":
            return self._register_manifest(body)

        if method == "GET" and path == "/api/v1/manifests":
            agent = (query.get("agent_name") or [None])[0]
            with self._lock:
                rows = [
                    {
                        "id": mid,
                        "agent_name": m["agent_name"],
                        "manifest_hash": m["manifest_hash"],
                        "version_label": m["version_label"],
                        "status": m["status"],
                        "components_count": m["components"],
                    }
                    for mid, m in self.manifests.items()
                    if agent is None or m["agent_name"] == agent
                ]
            return 200, {"manifests": rows, "total": len(rows), "limit": 20, "offset": 0}, []

        m = re.fullmatch(r"/api/v1/manifests/([^/]+)", path)
        if method == "GET" and m:
            with self._lock:
                row = self.manifests.get(m.group(1))
            if row is None:
                return 404, {"detail": "manifest not found"}, []
            return 200, {"id": m.group(1), **row}, []

        # ── skills rail (hermetic SkillRouter) ───────────────
        if method == "POST" and path == "/api/v1/skills/route":
            return self._route_skills(body)

        if method == "GET" and path == "/api/v1/skills/menu":
            fragment, routing_id = self._skill_fragment()
            return 200, {
                "skills": self.skills,
                "prompt_fragment": fragment,
                "routing_id": routing_id,
                "strategy": "menu",
            }, []

        m = re.fullmatch(r"/api/v1/skills/([^/]+)/body", path)
        if method == "GET" and m:
            skill = self._skill(m.group(1))
            if skill is None:
                return 404, {"detail": "skill not found"}, []
            return 200, {"body": skill.get("body", "")}, []

        # The disk-sync surface. Modelled because an adapter that mirrors
        # platform skills to disk reaches these, and what it writes is exactly
        # what C11 grades — a 404 here would hide the side effect.
        if method == "GET" and path == "/api/v1/skills/hashes":
            return 200, {
                "hashes": {
                    s["name"]: {"hash": _short_hash(s.get("body", ""))}
                    for s in self.skills
                }
            }, []

        if method == "GET" and path == "/api/v1/skills":
            return 200, {"skills": [self._skill_row(s) for s in self.skills]}, []

        if method == "POST" and path == "/api/v1/skills/sync":
            sent = (body or {}).get("skills") or [] if isinstance(body, dict) else []
            return 200, {
                "created": 0, "updated": 0, "unchanged": len(sent),
                "pulled": 0, "failures": 0,
            }, []

        m = re.fullmatch(r"/api/v1/skills/([^/]+)", path)
        if method == "GET" and m:
            skill = self._skill(m.group(1))
            if skill is None:
                return 404, {"detail": "skill not found"}, []
            return 200, self._skill_row(skill), []

        # Anything else: record it and answer the way an unknown route does.
        # A driver that depends on an endpoint the probe does not model shows
        # up as a 404 in the request log rather than as a mystery hang.
        return 404, {"detail": f"conformance probe has no route for {method} {path}"}, []

    # ── handlers ─────────────────────────────────────────────

    def _ingest_one(self, body: Any) -> Tuple[int, Any, List[str]]:
        with self._lock:
            known_manifests = set(self.manifests)
            known_traces = set(self.trace_ids)
        errors = validate_trace_payload(
            body,
            require_manifest=self.require_manifest,
            known_manifest_ids=known_manifests,
            known_trace_ids=known_traces,
        )
        if errors:
            return 400, {
                "detail": "Trace validation failed: " + "; ".join(errors),
                "error_code": "TRACE_VALIDATION_FAILED",
            }, errors
        trace_id = str(body.get("id") or uuid.uuid4())
        with self._lock:
            self.trace_ids.add(trace_id)
        return 200, {
            "status": "ok",
            "id": trace_id,
            "trace_id": trace_id,
            "agent_name": body.get("agent_name"),
            "spans": len(body.get("spans") or []),
            "llm_calls": len(body.get("llm_calls") or []),
        }, []

    def _register_manifest(self, body: Any) -> Tuple[int, Any, List[str]]:
        if not isinstance(body, dict):
            return 400, {"detail": "manifest body must be an object"}, ["bad manifest body"]
        agent_name = body.get("agent_name")
        manifest_hash = body.get("manifest_hash", "")
        if not agent_name:
            return 400, {"detail": "'agent_name' is required"}, ["manifest agent_name required"]
        with self._lock:
            # Agent-scoped hash dedup, exactly like manifest_service:
            # same hash + same agent → idempotent return of the existing row.
            for mid, row in self.manifests.items():
                if row["agent_name"] == agent_name and row["manifest_hash"] == manifest_hash:
                    return 200, {
                        "status": "ok",
                        "manifest_id": mid,
                        "manifest_hash": manifest_hash,
                        "version_label": row["version_label"],
                        "is_new": False,
                        "components": row["components"],
                    }, []
            # New version for this agent — supersede the prior active one.
            prior = [r for r in self.manifests.values() if r["agent_name"] == agent_name]
            for r in prior:
                r["status"] = "superseded"
            version = f"v{len(prior) + 1}"
            mid = str(uuid.uuid4())
            self.manifests[mid] = {
                "agent_name": agent_name,
                "manifest_hash": manifest_hash,
                "version_label": version,
                "status": "active",
                "components": len(body.get("components") or []),
            }
        return 200, {
            "status": "ok",
            "manifest_id": mid,
            "manifest_hash": manifest_hash,
            "version_label": version,
            "is_new": True,
            "components": len(body.get("components") or []),
        }, []

    # ── agents ───────────────────────────────────────────────

    def register_agent(
        self,
        agent_name: str,
        *,
        system_prompt: Optional[str] = None,
        skills: Sequence[Dict[str, Any]] = (),
    ) -> Dict[str, Any]:
        """Put an agent in this workspace, as the dashboard would.

        The journey this backs is the one a user walks: an agent already EXISTS
        on the platform, with a prompt somebody typed and skills somebody
        attached, and `decimalai init` turns that into a file. So the fixture
        has to be platform state, not a constructor argument the CLI is handed
        — the whole point of the tier is that the prompt and the skill bodies
        travel over the wire.

        The skills are also added to the ROUTER's offer set (`self.skills`), so
        the rail at run time offers exactly what this agent holds. One agent per
        journey run makes that exact; the router is not agent-scoped here for
        the same reason it is not in the driver tier — it is a fixture, and a
        per-agent index would be probe machinery nothing grades.
        """
        prompt = system_prompt
        with self._lock:
            row = {
                "agent_name": agent_name,
                "agent_id": str(uuid.uuid5(uuid.NAMESPACE_URL, "agent:" + agent_name)),
                "system_prompt": prompt,
                "version_number": 1 if prompt is not None else None,
                "version_id": (
                    str(uuid.uuid5(uuid.NAMESPACE_URL, "prompt:" + agent_name))
                    if prompt is not None else None
                ),
                "content_hash": _short_hash(prompt) if prompt is not None else None,
                "label": "created with the agent" if prompt is not None else None,
                "skills": [dict(s) for s in skills],
            }
            self.agents[agent_name] = row
            offered = {s.get("name") for s in self.skills}
            for skill in skills:
                if skill.get("name") not in offered:
                    self.skills.append(dict(skill))
        return row

    @staticmethod
    def _agent_row(agent: Dict[str, Any]) -> Dict[str, Any]:
        """One row of `GET /api/v1/agents` — the backend's `list_agents` shape.

        `trace_count: 0` is the honest value AND the interesting one: a
        UI-created agent that has never been traced is exactly the agent
        `decimalai init` exists to serve, and the one a truncating pagination
        path would drop.
        """
        return {
            "agent_name": agent["agent_name"],
            "trace_count": 0,
            "last_trace_at": None,
            "unevaluated_count": 0,
            "is_subagent": False,
            "is_demo": False,
            "latest_manifest": None,
            "compat_counts": None,
            "compat_scope": None,
        }

    def _agent_skills(self, agent_name: str) -> Tuple[int, Any, List[str]]:
        """`GET /api/v1/agents/{name}/skills` — subscriptions, never bodies.

        The backend says so in as many words ("The skill's full metadata
        (description, body, etc.) is not returned here"), and that is the
        load-bearing property for this tier: the scaffold CANNOT learn a skill
        body from this route, so a body that reaches the model at run time got
        there through the rail. A probe that helpfully included bodies would
        make the journey's sentinel clause unfalsifiable.
        """
        with self._lock:
            agent = self.agents.get(agent_name)
        if agent is None:
            return 404, {"detail": f"Agent '{agent_name}' not found"}, []
        rows = [
            {
                "subscription_id": str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"sub:{agent_name}:{s['name']}")
                ),
                "skill_id": str(
                    uuid.uuid5(uuid.NAMESPACE_URL, "skill:" + s["name"])
                ),
                "skill_name": s["name"],
                "description": s.get("description", ""),
                "category": None,
                "pinned_version_id": None,
                "pinned_version_number": None,
                "scope": "agent",
                "created_at": None,
            }
            for s in agent["skills"]
        ]
        return 200, {"agent_name": agent_name, "skills": rows, "total": len(rows)}, []

    def _agent_prompt(
        self, agent_name: str, query: Dict[str, List[str]]
    ) -> Tuple[int, Any, List[str]]:
        """`GET /api/v1/agents/{name}/prompt` — the route `load_agent()` reads.

        Shape copied from the backend's `_GET_PROMPT_EXAMPLE`. The one field
        that must be present even when there is no prompt is `system_prompt`
        itself: `AgentConfig._from_payload` checks for the KEY and refuses a
        payload without it rather than reading `.get()` and calling a
        wrong-shaped 200 "no prompt set".
        """
        with self._lock:
            agent = self.agents.get(agent_name)
        if agent is None:
            return 404, {
                "detail": f"No agent named '{agent_name}' in this workspace."
            }, []
        requested = (query.get("version") or [None])[0]
        if requested is not None and str(requested) != str(agent["version_number"]):
            return 404, {
                "detail": f"No version {requested} of '{agent_name}'s system prompt."
            }, []
        return 200, {
            "agent_name": agent["agent_name"],
            "agent_id": agent["agent_id"],
            "resolved_from": None,
            "system_prompt": agent["system_prompt"],
            "version_id": agent["version_id"],
            "version_number": agent["version_number"],
            "content_hash": agent["content_hash"],
            "label": agent["label"],
            "provenance": "ui",
            "created_at": None,
            "updated_at": None,
            "version_mode": "latest",
            "pinned_version_number": None,
        }, []

    def _skill(self, name: str) -> Optional[Dict[str, Any]]:
        for s in self.skills:
            if s.get("name") == name:
                return s
        return None

    @staticmethod
    def _skill_row(skill: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, "skill:" + skill["name"])),
            "name": skill["name"],
            "description": skill.get("description", ""),
            "body_markdown": skill.get("body", ""),
            "latest_version": {"version_number": 1},
        }

    def _skill_fragment(self, query: Optional[str] = None) -> Tuple[str, Optional[str]]:
        if not self.skills:
            return "", None
        routing_id = "rt_" + uuid.uuid4().hex[:24]
        with self._lock:
            self.routing_queries[routing_id] = query
        lines = ["Available skills:"]
        for s in self.skills:
            lines.append(f"- {s['name']}: {s.get('description', '')}")
        return "\n".join(lines), routing_id

    def _route_skills(self, body: Any) -> Tuple[int, Any, List[str]]:
        query = body.get("query") if isinstance(body, dict) else None
        fragment, routing_id = self._skill_fragment(query)
        return 200, {
            "skills": self.skills,
            "prompt_fragment": fragment,
            "routing_id": routing_id,
            "strategy": "semantic",
        }, []

    # ── views the contract reads ─────────────────────────────

    def manifest_owner(self, manifest_id: Optional[str]) -> Optional[str]:
        """Which agent a manifest id belongs to, or None if never registered."""
        if not manifest_id:
            return None
        with self._lock:
            row = self.manifests.get(str(manifest_id))
        return row["agent_name"] if row else None
