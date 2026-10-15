"""A stand-in for the registry's install routes: resolve, bundle and their anonymous twins.

The skills engine (``decimalai.skillpkg``) installs registry skills through four routes:

==========================================================  =================================
route                                                        caller
==========================================================  =================================
``POST /api/v1/install/resolve``                             an API key; org-scoped
``GET  /api/v1/install/bundles/{version_id}``                an API key; org-scoped
``POST /api/v1/registry/install/resolve``                    anonymous; public skills only
``GET  /api/v1/registry/install/bundles/{version_id}``       anonymous; public skills only
``GET  /api/v1/registry/skills/{id_or_slug}/bundle``         anonymous convenience (``pull``)
==========================================================  =================================

:class:`FakeRegistry` answers them in process with the same JSON shapes, the same status
codes and the same per-item error codes, and renders every byte with
``skillevaluation.skillfmt``, the module the platform renders with. A test therefore sees
the folder the platform would serve, not an invented one.

It also answers ``GET /api/v1/skills/{name}/body``, the delivery route older clients used
for disk exports, so a test can prove the engine never falls back to it. ``old=True`` makes
it a registry that predates the install routes: those answer the framework's own 404 while
``/body`` keeps working.

Use it in process through an ``httpx`` transport, or on a real socket::

    registry = FakeRegistry()
    skill = registry.add_skill("acme/refund-policy", org=OTHER_ORG)
    client = RegistryClient(api_key=OWNER_KEY, base_url=BASE_URL,
                            transport=registry.transport())

    with registry.serve() as base_url:      # http://127.0.0.1:<port>, for subprocesses
        ...

Every request is recorded in :attr:`FakeRegistry.requests` (path as sent and as decoded,
query, headers, JSON body, the client's timeout). :meth:`FakeRegistry.fault` queues a
failure for the next request to one route: a status code, a crafted response, a stall
(``"timeout"``) or a dropped connection (``"connect"``). :attr:`FakeRegistry.tamper` edits
a bundle response after its hashes were computed, to serve bytes that differ from what
the response declares; :attr:`FakeRegistry.tamper_resolve` edits a resolve response.

Simplifications, each one deliberate:

- Blocked and withheld skills are refused to every caller, owners included.
- A skill's "latest" is its newest version for its owner and for organizations with a Use
  link, and :attr:`FakeSkill.public_number` (default: also the newest) for everyone else.
- Ref resolution follows the documented order: a UUID is a skill id, then an exact
  url_slug, then a name (owned, then a Use target, then public; two public skills with the
  name are ``ambiguous``), then an install name.
"""
from __future__ import annotations

import contextlib
import difflib
import hashlib
import http.server
import json
import re
import threading
import time
import urllib.parse
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, Callable, Deque, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Union

import httpx
from skillevaluation import skillfmt

BASE_URL = "https://registry.test"
APP_URL = "https://app.decimal.test"

OWNER_ORG = "org-owner"
OWNER_KEY = "dai_sk_owner"
OTHER_ORG = "org-other"
OTHER_KEY = "dai_sk_other"

MAX_RESOLVE_ITEMS = 100

AUTH_RESOLVE = "/api/v1/install/resolve"
AUTH_BUNDLES = "/api/v1/install/bundles/"
ANON_RESOLVE = "/api/v1/registry/install/resolve"
ANON_BUNDLES = "/api/v1/registry/install/bundles/"
REGISTRY_SKILLS = "/api/v1/registry/skills/"
SKILLS = "/api/v1/skills/"

#: Route names, for :meth:`FakeRegistry.fault` and :meth:`FakeRegistry.calls`.
ROUTES = ("resolve", "bundle", "registry_bundle", "body", "other")

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_REGISTRY_BUNDLE_RE = re.compile(r"^/api/v1/registry/skills/(?P<ref>.+)/bundle$")
_BODY_RE = re.compile(r"^/api/v1/skills/(?P<name>.+)/body$")

FaultAction = Union[int, str, httpx.Response, Callable[[httpx.Request], httpx.Response]]


def fake_id(n: int) -> str:
    """A deterministic UUID-shaped id, so tests can write expected names down."""
    return f"00000000-0000-4000-8000-{n:012x}"


# ── State ───────────────────────────────────────────────────────────────────────────


@dataclass
class FakeVersion:
    """One version of a skill: a body and its files. ``safe=False`` is a version marked unsafe."""

    version_id: str
    number: int
    body: str
    files: List[Dict[str, Any]] = field(default_factory=list)
    safe: bool = True
    #: The description recorded with this version; an older version renders with it.
    description: Optional[str] = None


@dataclass
class FakeSkill:
    """One registry skill. Tests edit the attributes directly (``skill.description = ...``)."""

    skill_id: str
    name: str
    url_slug: str
    org_id: str
    public: bool
    description: str
    versions: List[FakeVersion] = field(default_factory=list)
    license: Optional[str] = None
    compatibility: Optional[str] = None
    allowed_tools: Optional[Sequence[str]] = None
    invocation_mode: str = "model"
    category: Optional[str] = None
    band: str = "clean"          # clean | caution | unreviewed | blocked
    withheld: bool = False
    imported: bool = False       # imported from GitHub: never the authored variant
    forked: bool = False         # a fork: never the authored variant
    users: Set[str] = field(default_factory=set)   # orgs with a Use link
    #: The version others get for "latest"; None means the newest.
    public_number: Optional[int] = None

    def version(self, number: int) -> FakeVersion:
        for v in self.versions:
            if v.number == number:
                return v
        raise KeyError(number)

    @property
    def newest(self) -> FakeVersion:
        return max(self.versions, key=lambda v: v.number)


@dataclass
class FakeRequest:
    """One request as the registry saw it."""

    method: str
    route: str
    path: str        # decoded, as the server routes it
    raw_path: str    # as it travelled, percent-encoding intact, without the query
    query: Dict[str, str]
    headers: Dict[str, str]   # names lowercased
    json: Any
    timeout: Optional[Dict[str, Optional[float]]]


@dataclass(frozen=True)
class Fetch:
    """One bundle served for disk (what the platform counts in its fetch table)."""

    skill_id: str
    org_id: str      # "" for an anonymous fetch
    channel: Optional[str]
    purpose: Optional[str]


class _ItemError(Exception):
    def __init__(self, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.extra = extra


class _Stall(Exception):
    """The registry does not answer in time."""


class _Drop(Exception):
    """The connection fails."""


# ── The registry ─────────────────────────────────────────────────────────────────────


class FakeRegistry:
    """The install routes over an in-memory registry. See the module docstring."""

    def __init__(
        self,
        *,
        base_url: str = BASE_URL,
        keys: Optional[Mapping[str, str]] = None,
        old: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        #: API key -> organization.
        self.keys: Dict[str, str] = dict(keys or {OWNER_KEY: OWNER_ORG, OTHER_KEY: OTHER_ORG})
        #: True: a registry from before the install routes (they 404; ``/body`` answers).
        self.old = old
        self.skills: List[FakeSkill] = []
        self.requests: List[FakeRequest] = []
        self.fetches: List[Fetch] = []
        #: Called with each bundle response (a dict) after its hashes are computed.
        self.tamper: Optional[Callable[[Dict[str, Any]], None]] = None
        #: Called with each resolve response (a dict) before it is sent.
        self.tamper_resolve: Optional[Callable[[Dict[str, Any]], None]] = None
        #: How long a stalled request holds the socket in :meth:`serve` mode.
        self.stall_seconds = 3.0
        self._faults: Dict[str, Deque[FaultAction]] = {}
        #: version id -> render hash -> the fields that rendering was made from.
        self._renders: Dict[str, Dict[str, skillfmt.RenderFields]] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    # ── building the registry ─────────────────────────────────────────

    def _new_id(self) -> str:
        n = self._next_id
        self._next_id += 1
        return fake_id(n)

    def add_skill(
        self,
        name: str,
        *,
        org: str = OTHER_ORG,
        public: bool = True,
        body: Optional[str] = None,
        description: Optional[str] = None,
        files: Sequence[Mapping[str, Any]] = (),
        url_slug: Optional[str] = None,
        skill_id: Optional[str] = None,
        versions: int = 1,
        **attrs: Any,
    ) -> FakeSkill:
        """Add a skill with ``versions`` versions (each a different body, the same files).

        ``attrs`` sets any other :class:`FakeSkill` field (``band``, ``withheld``,
        ``allowed_tools``, ``invocation_mode``, ``imported``, ...).
        """
        skill = FakeSkill(
            skill_id=skill_id or self._new_id(),
            name=name,
            url_slug=url_slug or skillfmt.install_name_base(name) or self._new_id(),
            org_id=org,
            public=public,
            description=description if description is not None else f"Use when the task involves {name}.",
            **attrs,
        )
        self.skills.append(skill)
        for _ in range(versions):
            self.add_version(skill, body, files=files)
        return skill

    def add_version(
        self,
        skill: FakeSkill,
        body: Optional[str] = None,
        *,
        files: Optional[Sequence[Mapping[str, Any]]] = None,
        safe: bool = True,
    ) -> FakeVersion:
        """A new version, numbered after the newest, recording the live description."""
        number = max((v.number for v in skill.versions), default=0) + 1
        if files is None:
            files = skill.versions[-1].files if skill.versions else ()
        version = FakeVersion(
            version_id=self._new_id(),
            number=number,
            body=body if body is not None else f"# {skill.name}\n\nStep {number}: follow the procedure.\n",
            files=[dict(f) for f in files],
            safe=safe,
            description=skill.description,
        )
        skill.versions.append(version)
        return version

    def use(self, skill: FakeSkill, org: str) -> None:
        """Give ``org`` a Use link to ``skill``."""
        skill.users.add(org)

    def forget_renders(self, version: FakeVersion) -> None:
        """Drop every render snapshot of a version (as a database restore would)."""
        self._renders.pop(version.version_id, None)

    def fault(self, route: str, *actions: FaultAction) -> None:
        """Answer the next requests to ``route`` with ``actions``, one per request.

        An action is a status code (a framework-style error body), an ``httpx.Response``
        sent as it is, ``"timeout"`` (no answer in time), ``"connect"`` (the connection
        fails), or a callable that takes the request and returns a response.
        """
        if route not in ROUTES and route != "any":
            raise ValueError(f"unknown route {route!r}; one of {ROUTES + ('any',)}")
        self._faults.setdefault(route, deque()).extend(actions)

    # ── what tests read back ──────────────────────────────────────────

    def calls(self, route: Optional[str] = None) -> List[FakeRequest]:
        return [r for r in self.requests if route is None or r.route == route]

    def snapshot(
        self, skill: FakeSkill, version: Optional[FakeVersion] = None,
    ) -> Tuple[str, skillfmt.RenderFields]:
        """The current render snapshot of a version: created on first use, never changed."""
        version = version or skill.newest
        fields = self._fields(skill, version)
        render_hash = skillfmt.render_hash(fields)
        with self._lock:
            known = self._renders.setdefault(version.version_id, {})
            fields = known.setdefault(render_hash, fields)
        return render_hash, fields

    def render(
        self,
        skill: FakeSkill,
        version: Optional[FakeVersion] = None,
        *,
        install_name: Optional[str] = None,
        variant: str = "foreign",
        render_hash: Optional[str] = None,
    ) -> skillfmt.Bundle:
        """The bundle the registry serves for this key: the bytes a test expects."""
        version = version or skill.newest
        if render_hash is None:
            render_hash, fields = self.snapshot(skill, version)
        else:
            fields = self._renders[version.version_id][render_hash]
        return skillfmt.render_bundle(
            fields, version.files,
            install_name=install_name or skillfmt.install_name_for(skill.name, skill.skill_id),
            variant=variant,
        )

    # ── transports ────────────────────────────────────────────────────

    def handle(self, request: httpx.Request) -> httpx.Response:
        """The ``httpx.MockTransport`` handler."""
        try:
            return self._dispatch(request)
        except _Stall:
            raise httpx.ReadTimeout("fake registry: no answer in time", request=request)
        except _Drop:
            raise httpx.ConnectError("fake registry: connection refused", request=request)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    @contextlib.contextmanager
    def serve(self) -> Iterator[str]:
        """Serve on ``127.0.0.1`` and yield the base URL; for subprocesses and real sockets."""
        registry = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _any(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                content = self.rfile.read(length) if length else b""
                request = httpx.Request(
                    self.command, f"http://{self.headers.get('Host')}{self.path}",
                    headers=list(self.headers.items()), content=content,
                )
                try:
                    response = registry._dispatch(request)
                except _Stall:
                    time.sleep(registry.stall_seconds)
                    response = _framework(503)
                except _Drop:
                    self.close_connection = True
                    return
                body = response.content
                try:
                    self.send_response(response.status_code)
                    for name, value in response.headers.items():
                        if name.lower() not in ("content-length", "transfer-encoding", "connection"):
                            self.send_header(name, value)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):  # the client gave up
                    self.close_connection = True

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _any

            def log_message(self, *args: Any) -> None:
                pass

        class Server(http.server.ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request: Any, client_address: Any) -> None:
                # A client that gave up (a timeout test) resets the connection; that is
                # the scenario, not a failure of the fake.
                import sys

                if not isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
                    super().handle_error(request, client_address)

        server = Server(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    # ── dispatch ──────────────────────────────────────────────────────

    def _dispatch(self, request: httpx.Request) -> httpx.Response:
        raw = request.url.raw_path.decode("ascii")
        raw_path = raw.split("?", 1)[0]
        path = urllib.parse.unquote(raw_path)
        route = _route_of(path)
        try:
            body = json.loads(request.content) if request.content else None
        except ValueError:
            body = None
        self.requests.append(FakeRequest(
            method=request.method,
            route=route,
            path=path,
            raw_path=raw_path,
            query=dict(urllib.parse.parse_qsl(request.url.query.decode("ascii"), keep_blank_values=True)),
            headers={k.lower(): v for k, v in request.headers.items()},
            json=body,
            timeout=request.extensions.get("timeout"),
        ))
        action = self._next_fault(route)
        if action is not None:
            return self._act(action, request)
        if self.old and route in ("resolve", "bundle", "registry_bundle"):
            return _framework(404)

        method = request.method.upper()
        query = self.requests[-1].query
        if path in (AUTH_RESOLVE, ANON_RESOLVE):
            if method != "POST":
                return _framework(405, accepted=["POST"])
            org = None
            if path == AUTH_RESOLVE:
                org = self._org(request)
                if org is None:
                    return _unauthorized()
            return self._resolve(body, org)
        if path.startswith(AUTH_BUNDLES) or path.startswith(ANON_BUNDLES):
            if method != "GET":
                return _framework(405, accepted=["GET"])
            authenticated = path.startswith(AUTH_BUNDLES)
            org = None
            if authenticated:
                org = self._org(request)
                if org is None:
                    return _unauthorized()
            version_id = path[len(AUTH_BUNDLES if authenticated else ANON_BUNDLES):]
            return self._bundle(version_id, query, org, anonymous=not authenticated)
        match = _REGISTRY_BUNDLE_RE.match(path)
        if match:
            if method != "GET":
                return _framework(405, accepted=["GET"])
            return self._registry_bundle(match.group("ref"), query)
        match = _BODY_RE.match(path)
        if match and method == "GET":
            return self._body(match.group("name"))
        return _framework(404)

    def _next_fault(self, route: str) -> Optional[FaultAction]:
        for key in (route, "any"):
            queue = self._faults.get(key)
            if queue:
                return queue.popleft()
        return None

    @staticmethod
    def _act(action: FaultAction, request: httpx.Request) -> httpx.Response:
        if action == "timeout":
            raise _Stall()
        if action == "connect":
            raise _Drop()
        if isinstance(action, httpx.Response):
            return action
        if isinstance(action, int):
            return _framework(action)
        if callable(action):
            return action(request)
        raise ValueError(f"unknown fault action {action!r}")

    def _org(self, request: httpx.Request) -> Optional[str]:
        auth = request.headers.get("authorization") or ""
        scheme, _, key = auth.partition(" ")
        if scheme.lower() != "bearer":
            return None
        return self.keys.get(key.strip())

    # ── access ────────────────────────────────────────────────────────

    @staticmethod
    def _visible(skill: FakeSkill, org: Optional[str]) -> bool:
        if org is None:
            return skill.public
        return skill.org_id == org or org in skill.users or skill.public

    @staticmethod
    def _variant(skill: FakeSkill, org: Optional[str]) -> str:
        authored = org is not None and skill.org_id == org and not skill.forked and not skill.imported
        return "authored" if authored else "foreign"

    @staticmethod
    def _relationship(skill: FakeSkill, org: Optional[str]) -> str:
        if org is not None and skill.org_id == org:
            return "owned"
        if org is not None and org in skill.users:
            return "used"
        return "public"

    def _pointer(self, skill: FakeSkill, org: Optional[str]) -> FakeVersion:
        if self._relationship(skill, org) == "public" and skill.public_number is not None:
            return skill.version(skill.public_number)
        return skill.newest

    def _fields(self, skill: FakeSkill, version: FakeVersion) -> skillfmt.RenderFields:
        is_pointer = version.number == skill.newest.number or version.number == skill.public_number
        description = skill.description if is_pointer else (version.description or skill.description)
        source = f"{APP_URL}/s/{skill.url_slug}@{version.number}/SKILL.md" if skill.public else None
        return skillfmt.RenderFields(
            skill_id=skill.skill_id,
            version_id=version.version_id,
            version=version.number,
            registry_name=skill.name,
            body=version.body,
            description=description,
            license=skill.license,
            compatibility=skill.compatibility,
            allowed_tools=list(skill.allowed_tools) if skill.allowed_tools is not None else None,
            invocation_mode=skill.invocation_mode,
            category=skill.category,
            source=source,
        )

    def _refusal_for(self, skill: FakeSkill, version: Optional[FakeVersion]) -> Optional[str]:
        if skill.band == "blocked":
            return "blocked"
        if skill.withheld:
            return "withheld"
        if version is not None and not version.safe:
            return "unsafe"
        return None

    # ── resolve ───────────────────────────────────────────────────────

    def _resolve(self, body: Any, org: Optional[str]) -> httpx.Response:
        if not isinstance(body, dict) or not isinstance(body.get("items"), list):
            return _validation_error("items", "a list of items is required")
        items = body["items"]
        if len(items) > MAX_RESOLVE_ITEMS:
            return _refusal(422, "too_many_items",
                            f"at most {MAX_RESOLVE_ITEMS} items per request, got {len(items)}")
        taken = body.get("taken") or {}
        if not isinstance(taken, dict):
            return _validation_error("taken", "a map of install name to skill id or null")
        taken = dict(taken)
        data = {"items": [self._resolve_item(item, org, taken) for item in items]}
        if self.tamper_resolve is not None:
            self.tamper_resolve(data)
        return _json(200, data, cache="private, no-store" if org else None)

    def _resolve_item(self, item: Any, org: Optional[str], taken: Dict[str, Optional[str]]) -> Dict[str, Any]:
        if not isinstance(item, dict):
            return {"ok": False, "error": {"code": "not_found", "message": "an item must be an object"}}
        ref, skill_id = item.get("ref"), item.get("skill_id")
        label = ref if ref is not None else skill_id
        try:
            skill = self._find(ref, skill_id, org)
            refusal = self._refusal_for(skill, None)
            if refusal:
                raise _ItemError(refusal, f"{skill.name} is {refusal}")
            version, notes = self._choose_version(skill, item, org)
            try:
                install_name = skillfmt.choose_install_name(skill.name, skill.skill_id, taken)
            except skillfmt.InstallNameCollision as exc:
                raise _ItemError("install_name_collision", str(exc), install_name=exc.install_name)
            variant = self._variant(skill, org)
            render_hash, fields = self.snapshot(skill, version)
            try:
                bundle = skillfmt.render_bundle(fields, version.files, install_name=install_name,
                                                variant=variant)
            except skillfmt.RenderError as exc:
                raise _ItemError(exc.code, str(exc))
        except _ItemError as exc:
            return {"ref": label, "ok": False,
                    "error": {"code": exc.code, "message": exc.message, **exc.extra}}
        taken[install_name] = skill.skill_id
        pointer = self._pointer(skill, org)
        base, disambiguated = skillfmt.install_name_forms(skill.name, skill.skill_id)
        return {
            "ref": label,
            "ok": True,
            "skill_id": skill.skill_id,
            "registry_name": skill.name,
            "url_slug": skill.url_slug,
            "install_name": install_name,
            "install_name_base": base,
            "install_name_disambiguated": disambiguated,
            "version_id": version.version_id,
            "version": version.number,
            "latest": {"version_id": pointer.version_id, "version": pointer.number},
            "render": {"format": skillfmt.FORMAT, "hash": render_hash, "variant": variant},
            "body_sha256": bundle.body_sha256,
            "description_sha256": bundle.description_sha256,
            "invocation": bundle.invocation,
            "relationship": self._relationship(skill, org),
            "files": [{"path": f.path, "sha256": f.sha256, "mode": f.mode} for f in bundle.files],
            "unavailable": [{"path": u.path, "reason": u.reason} for u in bundle.unavailable],
            "policy_notes": notes + list(bundle.policy_notes),
            "safety": {"band": skill.band, "withheld": skill.withheld},
        }

    def _find(self, ref: Any, skill_id: Any, org: Optional[str]) -> FakeSkill:
        visible = [s for s in self.skills if self._visible(s, org)]
        if skill_id is not None:
            for s in visible:
                if s.skill_id == skill_id:
                    return s
            raise _ItemError("not_found", f"no skill with id {skill_id}")
        if not isinstance(ref, str) or not ref:
            raise _ItemError("not_found", "an item needs a ref or a skill_id")
        if _UUID_RE.match(ref):
            for s in visible:
                if s.skill_id == ref:
                    return s
            raise _ItemError("not_found", f"no skill with id {ref}")
        for s in visible:
            if s.url_slug == ref:
                return s
        named = [s for s in visible if s.name == ref]
        for tier in (
            [s for s in named if org is not None and s.org_id == org],
            [s for s in named if org is not None and org in s.users],
        ):
            if tier:
                return tier[0]
        public = [s for s in named if s.public]
        if len(public) == 1:
            return public[0]
        if len(public) > 1:
            raise _ItemError("ambiguous", f"{len(public)} public skills are named {ref}",
                             near=sorted(s.url_slug for s in public))
        by_install_name = [s for s in visible
                           if skillfmt.install_name_for(s.name, s.skill_id) == ref]
        if len(by_install_name) == 1:
            return by_install_name[0]
        if len(by_install_name) > 1:
            raise _ItemError("ambiguous", f"{len(by_install_name)} skills install as {ref}",
                             near=sorted(s.url_slug for s in by_install_name))
        candidates = sorted({s.name for s in visible} | {s.url_slug for s in visible})
        raise _ItemError("not_found", f"no skill matches {ref}",
                         near=difflib.get_close_matches(ref, candidates, n=3))

    def _choose_version(
        self, skill: FakeSkill, item: Mapping[str, Any], org: Optional[str],
    ) -> Tuple[FakeVersion, List[str]]:
        version_id = item.get("version_id")
        spec = item.get("version", "latest")
        if version_id is not None:
            matches = [v for v in skill.versions if v.version_id == version_id]
            if not matches:
                raise _ItemError("not_found", f"{version_id} is not a version of {skill.name}")
            version = matches[0]
        elif spec == "latest":
            pointer = self._pointer(skill, org)
            safe = [v for v in skill.versions if v.number <= pointer.number and v.safe]
            if not safe:
                raise _ItemError("version_unsafe", f"no version of {skill.name} is marked safe")
            version = max(safe, key=lambda v: v.number)
            if version is not pointer:
                return version, [f"version {pointer.number} is marked unsafe; installed version "
                                 f"{version.number}, the newest safe one"]
            return version, []
        elif isinstance(spec, int) and not isinstance(spec, bool):
            try:
                version = skill.version(spec)
            except KeyError:
                raise _ItemError("not_found", f"{skill.name} has no version {spec}")
        else:
            raise _ItemError("not_found", f"version must be 'latest' or a number, not {spec!r}")
        if not version.safe:
            raise _ItemError("version_unsafe", f"version {version.number} of {skill.name} is marked unsafe")
        return version, []

    # ── bundles ───────────────────────────────────────────────────────

    def _by_version_id(self, version_id: str) -> Optional[Tuple[FakeSkill, FakeVersion]]:
        for s in self.skills:
            for v in s.versions:
                if v.version_id == version_id:
                    return s, v
        return None

    def _bundle(
        self, version_id: str, query: Mapping[str, str], org: Optional[str], *, anonymous: bool,
    ) -> httpx.Response:
        found = self._by_version_id(version_id)
        if found is None:
            return _refusal(404, "not_found", f"no version {version_id}")
        skill, version = found
        if not self._visible(skill, org):
            return _refusal(404, "no_access", f"no version {version_id}")
        refusal = self._refusal_for(skill, version)
        if refusal:
            return _refusal(410, "revoked", f"version {version.number} of {skill.name} is {refusal}",
                            reason=refusal)
        install_name = query.get("install_name")
        if not install_name:
            return _validation_error("install_name", "field required")
        if install_name not in skillfmt.install_name_forms(skill.name, skill.skill_id):
            return _refusal(422, "invalid_install_name",
                            f"{install_name} is not an install name of {skill.name}")
        try:
            fmt = int(query.get("format") or skillfmt.FORMAT)
        except ValueError:
            return _validation_error("format", "an integer is required")
        if fmt not in skillfmt.SUPPORTED_FORMATS:
            return _refusal(422, "unsupported_format", f"format {fmt} is not supported")
        natural = self._variant(skill, org)
        variant = query.get("variant") or natural
        if variant not in skillfmt.VARIANTS:
            return _validation_error("variant", f"one of {skillfmt.VARIANTS}")
        if variant == "authored" and natural != "authored":
            return _refusal(403, "not_author", "the authored variant is served only to its authors")
        render_hash = query.get("render_hash")
        if render_hash:
            fields = self._renders.get(version.version_id, {}).get(render_hash)
            if fields is None:
                return _refusal(409, "render_gone", f"render {render_hash[:12]} no longer exists")
        else:
            render_hash, fields = self.snapshot(skill, version)
        try:
            bundle = skillfmt.render_bundle(fields, version.files, install_name=install_name,
                                            variant=variant, format=fmt)
        except skillfmt.RenderError as exc:
            return _refusal(422, exc.code, str(exc))
        data = bundle.to_dict()
        if self.tamper is not None:
            self.tamper(data)
        self.fetches.append(Fetch(skill.skill_id, org or "", query.get("channel"), query.get("purpose")))
        return _json(200, data, cache="public, max-age=300" if anonymous else "private, no-store")

    def _registry_bundle(self, ref: str, query: Mapping[str, str]) -> httpx.Response:
        public = [s for s in self.skills if s.public]
        skill = next((s for s in public if s.skill_id == ref), None) \
            or next((s for s in public if s.url_slug == ref), None) \
            or next((s for s in public if s.name == ref), None)
        if skill is None:
            return _refusal(404, "not_found", f"no public skill {ref}")
        try:
            version = skill.version(int(query["version"])) if query.get("version") \
                else self._pointer(skill, None)
        except (KeyError, ValueError):
            return _refusal(404, "not_found", f"{skill.name} has no version {query.get('version')}")
        refusal = self._refusal_for(skill, version)
        if refusal:
            return _refusal(410, "revoked", f"version {version.number} of {skill.name} is {refusal}",
                            reason=refusal)
        render_hash, fields = self.snapshot(skill, version)
        bundle = skillfmt.render_bundle(fields, version.files,
                                        install_name=skillfmt.install_name_for(skill.name, skill.skill_id),
                                        variant="foreign")
        data = bundle.to_dict()
        if self.tamper is not None:
            self.tamper(data)
        self.fetches.append(Fetch(skill.skill_id, "", query.get("channel"), query.get("purpose")))
        return _json(200, data, cache="public, max-age=300")

    def _body(self, name: str) -> httpx.Response:
        skill = next((s for s in self.skills if s.name == name), None)
        if skill is None:
            return _refusal(404, "not_found", f"no skill named {name}")
        body = skill.newest.body
        return _json(200, {"name": skill.name, "body": body, "skill_id": skill.skill_id,
                           "description": skill.description,
                           "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest()})


# ── responses, shaped like the platform's ────────────────────────────────────────────


def _route_of(path: str) -> str:
    if path in (AUTH_RESOLVE, ANON_RESOLVE):
        return "resolve"
    if path.startswith(AUTH_BUNDLES) or path.startswith(ANON_BUNDLES):
        return "bundle"
    if _REGISTRY_BUNDLE_RE.match(path):
        return "registry_bundle"
    if _BODY_RE.match(path):
        return "body"
    return "other"


def _json(status: int, data: Any, *, cache: Optional[str] = None,
          headers: Optional[Mapping[str, str]] = None) -> httpx.Response:
    out = {"Content-Type": "application/json", "X-Request-ID": "req-fake"}
    if cache:
        out["Cache-Control"] = cache
    out.update(headers or {})
    # ASCII-escaped, as the platform's encoder writes it; it also lets a test serve text
    # that is not valid UTF-8 (a lone surrogate), as a broken server could.
    return httpx.Response(status, content=json.dumps(data).encode("ascii"), headers=out)


def _framework(status: int, *, accepted: Optional[List[str]] = None) -> httpx.Response:
    """What the platform answers for a path no route matches, or any bare status.

    The platform's error envelope keeps the framework's own ``detail`` text ("Not Found",
    "Method Not Allowed") and adds ``code`` and ``request_id``.
    """
    body: Dict[str, Any] = {"detail": HTTPStatus(status).phrase, "code": f"HTTP_{status}",
                            "request_id": "req-fake"}
    headers = {}
    if accepted:
        body["accepted_methods"] = accepted
        headers["Allow"] = ", ".join(accepted)
    return _json(status, body, headers=headers)


def _refusal(status: int, code: str, message: str, **extra: Any) -> httpx.Response:
    """A refusal raised by an install route's handler: a structured ``detail``."""
    detail = {"code": code, "message": message, **extra}
    return _json(status, {"detail": detail, "code": f"HTTP_{status}", "request_id": "req-fake"})


def _validation_error(field_name: str, message: str) -> httpx.Response:
    return _json(422, {"detail": "Request validation failed", "code": "validation_error",
                       "request_id": "req-fake",
                       "details": {"errors": [{"field": field_name, "message": message}]}})


def _unauthorized() -> httpx.Response:
    return _json(401, {"detail": "Invalid or missing API key", "code": "HTTP_401",
                       "request_id": "req-fake"})


__all__ = [
    "ANON_BUNDLES", "ANON_RESOLVE", "APP_URL", "AUTH_BUNDLES", "AUTH_RESOLVE", "BASE_URL",
    "FakeRegistry", "FakeRequest", "FakeSkill", "FakeVersion", "Fetch", "MAX_RESOLVE_ITEMS",
    "OTHER_KEY", "OTHER_ORG", "OWNER_KEY", "OWNER_ORG", "ROUTES", "fake_id",
]

