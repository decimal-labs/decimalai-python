"""The registry client of the skills engine: resolve references, fetch bundles, check bytes.

The engine reaches the registry through two calls. ``resolve`` turns references into a
skill, a version, a rendering and the install name this project should use;
``bundle`` serves the exact files of one rendering. Each has an anonymous twin that
serves public skills only, used when no API key is set:

====================  ===========================================  ==========================================
call                  with an API key                              anonymous
====================  ===========================================  ==========================================
resolve               ``POST /api/v1/install/resolve``             ``POST /api/v1/registry/install/resolve``
bundle                ``GET /api/v1/install/bundles/{version_id}`` ``GET /api/v1/registry/install/bundles/{version_id}``
loose copy (pull)     --                                           ``GET /api/v1/registry/skills/{ref}/bundle``
====================  ===========================================  ==========================================

What every call holds to:

- **Cold, and unrouted.** Installing is a command someone ran, not an agent's turn, so it
  runs on the 30 s budget (:data:`DEFAULT_TIMEOUT`), never on the in-turn budget, and its
  failures never reach the circuit breakers or ``decimalai.routing_status()`` that agents
  depend on. No request carries a ``routing_id``: it made no routing decision.
- **No fallback.** A registry that answers the install routes with the framework's own
  404 (or 405) predates them, and the call raises :class:`RegistryTooOldError`. The client
  never retries through ``/api/v1/skills/{name}/body``: that route counts a delivery to a
  model, and its bytes are not the rendered folder.
- **Typed refusals.** Each per-item error code of ``resolve`` becomes its own error class,
  and every failure is a :class:`~decimalai.skillpkg.errors.SkillpkgError` with the
  documented exit code: 7 for anything the registry will not serve, 8 for the network
  (a timeout, a refused connection, a 5xx, admission that never clears), 6 for bytes
  that cannot be trusted.
- **Bytes are checked on receipt.** Each file's sha256 must equal the one the registry
  declares next to it and, when the caller passes them, the hashes and exec bits it
  expects (the lock's, for ``install --frozen``). Every path must be safe to write. A
  bundle that fails any check raises before the caller writes anything.

A reference that travels in a URL path is percent-encoded, with ``/`` kept literal for
namespaced registry names, as the delivery routes have always sent them. References in a
``resolve`` body travel as JSON, unchanged.

Admission aborts (429 without a spent quota, 502, 503, 504) are retried on a short
ladder, honouring ``Retry-After``; a timeout is retried once, because the first request
to a cold registry instance can take most of the budget and leaves it warm.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import httpx
from skillevaluation import skillfmt

from .._client import _parse_retry_after
from .._config import sdk_headers, sdk_user_agent
from .errors import (
    AccessDeniedError,
    BundleContractError,
    IntegrityError,
    LockedVersionRevokedError,
    NetworkError,
    RegistryTooOldError,
    RenderGoneError,
    SkillpkgError,
    UnavailableError,
    UsageError,
)

__all__ = [
    "CHANNELS",
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT",
    "MAX_RESOLVE_ITEMS",
    "PURPOSES",
    "Bundle",
    "BundleFile",
    "FileEntry",
    "RegistryClient",
    "RenderKey",
    "ResolveRequest",
    "Resolved",
    "Safety",
    "Unavailable",
    "VersionRef",
]

DEFAULT_BASE_URL = "https://api.decimal.ai"

#: Seconds per request: the cold budget every CLI action runs on. The first request to a
#: registry instance that is still starting can take most of it.
DEFAULT_TIMEOUT = 30.0

#: Items per ``resolve`` request; a longer list is sent in batches.
MAX_RESOLVE_ITEMS = 100

#: Who is fetching, as the registry's disk-fetch counter records it.
CHANNELS = ("sdk", "cli", "plugin", "npx", "web", "raw", "mcp")
#: Why it is fetching.
PURPOSES = ("add", "install", "frozen", "update", "pull", "migrate")

_RESOLVE = "/api/v1/install/resolve"
_BUNDLES = "/api/v1/install/bundles/"
_ANON_RESOLVE = "/api/v1/registry/install/resolve"
_ANON_BUNDLES = "/api/v1/registry/install/bundles/"
_REGISTRY_SKILLS = "/api/v1/registry/skills/"

# Admission aborts: the request never reached a healthy instance, so the same request
# can succeed a moment later. A 500 is not one: the registry ran and failed.
_ADMISSION_STATUSES = frozenset({429, 502, 503, 504})
_RETRY_DELAYS_S = (0.5, 1.5, 4.0)
_MAX_RETRY_WAIT_S = 30.0
_TIMEOUT_RETRIES = 1

# The ``detail`` the registry's error envelope keeps for a path that matches no route.
_FRAMEWORK_DETAILS = frozenset({"Not Found", "Method Not Allowed"})

_MODES = frozenset({0o644, 0o755})
_MODE_RE = re.compile(r"^0?[0-7]{3}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DIGITS_RE = re.compile(r"^[0-9]+$")
_UPGRADE_FIX = "run `pip install -U decimalai`, then retry"


# ── What a caller asks for ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResolveRequest:
    """One item of a ``resolve`` call: a reference, or a skill id, at a version.

    ``ref`` is a registry name, url_slug, install name or skill id. ``version`` is
    ``"latest"`` or a version number (an ``int``, or the digits a manifest records).
    ``version_id`` pins one exact version of ``skill_id`` (or of ``ref``); ``version`` is
    then not sent.

    Raises:
        UsageError: the item names no skill, or the version is neither ``"latest"`` nor
            a positive number.
    """

    ref: Optional[str] = None
    version: Union[int, str] = "latest"
    skill_id: Optional[str] = None
    version_id: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("ref", "skill_id", "version_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise UsageError(f"{name} must be a non-empty string, not {value!r}")
        if self.ref is None and self.skill_id is None:
            raise UsageError("a skill to resolve needs a reference or a skill id")
        object.__setattr__(self, "version", _version_spec(self.version))

    @property
    def label(self) -> str:
        """How errors name this item: the reference as given, else the skill id."""
        return self.ref if self.ref is not None else str(self.skill_id)

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.ref is not None:
            out["ref"] = self.ref
        if self.skill_id is not None:
            out["skill_id"] = self.skill_id
        if self.version_id is not None:
            out["version_id"] = self.version_id
        else:
            out["version"] = self.version
        return out


def _version_spec(value: Any) -> Union[int, str]:
    if value == "latest":
        return "latest"
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    if isinstance(value, str) and _DIGITS_RE.match(value) and int(value) >= 1:
        return int(value)
    raise UsageError(f"a version is `latest` or a number from 1, not {value!r}")


# ── What the registry answers ────────────────────────────────────────────────────


@dataclass(frozen=True)
class RenderKey:
    """One rendering of a version: the renderer's format, the snapshot hash, the variant."""

    format: int
    hash: str
    variant: str  # "authored" or "foreign"

    def to_dict(self) -> Dict[str, Any]:
        return {"format": self.format, "hash": self.hash, "variant": self.variant}


@dataclass(frozen=True)
class FileEntry:
    """A file a rendering contains, as ``resolve`` declares it (no content)."""

    path: str
    sha256: str
    mode: int

    @property
    def executable(self) -> bool:
        return self.mode == 0o755


@dataclass(frozen=True)
class Unavailable:
    """A file of the skill that is listed but not served, and why (binary, nested SKILL.md)."""

    path: str
    reason: str


@dataclass(frozen=True)
class Safety:
    band: str          # clean, caution or unreviewed (blocked skills are refused)
    withheld: bool


@dataclass(frozen=True)
class VersionRef:
    version_id: str
    version: int


@dataclass(frozen=True)
class Resolved:
    """A reference resolved to one rendering of one version, under a chosen install name."""

    ref: str
    skill_id: str
    registry_name: str
    install_name: str
    version_id: str
    version: int
    render: RenderKey
    body_sha256: str
    description_sha256: str
    files: Tuple[FileEntry, ...]
    url_slug: Optional[str] = None
    install_name_base: Optional[str] = None
    install_name_disambiguated: Optional[str] = None
    latest: Optional[VersionRef] = None
    invocation: str = "model"
    relationship: Optional[str] = None   # owned, used or public
    unavailable: Tuple[Unavailable, ...] = ()
    policy_notes: Tuple[str, ...] = ()
    safety: Safety = Safety("unreviewed", False)
    #: The item exactly as the registry sent it, for fields this client does not model.
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def file_hashes(self) -> Dict[str, str]:
        return {f.path: f.sha256 for f in self.files}

    @property
    def executable(self) -> Tuple[str, ...]:
        return tuple(sorted(f.path for f in self.files if f.executable))


@dataclass(frozen=True)
class BundleFile:
    """One file to write: its path under the skill folder, exact bytes, sha256 and mode."""

    path: str
    sha256: str
    mode: int
    content: bytes

    @property
    def executable(self) -> bool:
        return self.mode == 0o755


@dataclass(frozen=True)
class Bundle:
    """The files of one rendering, every one checked against its declared sha256."""

    version_id: str
    install_name: str
    render: RenderKey
    files: Tuple[BundleFile, ...]
    unavailable: Tuple[Unavailable, ...] = ()
    policy_notes: Tuple[str, ...] = ()
    skill_id: Optional[str] = None
    body_sha256: Optional[str] = None
    description_sha256: Optional[str] = None
    invocation: Optional[str] = None

    @property
    def file_hashes(self) -> Dict[str, str]:
        return {f.path: f.sha256 for f in self.files}

    @property
    def executable(self) -> Tuple[str, ...]:
        return tuple(sorted(f.path for f in self.files if f.executable))

    @property
    def skill_md(self) -> bytes:
        return next(f.content for f in self.files if f.path == "SKILL.md")


# ── The client ───────────────────────────────────────────────────────────────────


class RegistryClient:
    """The install routes of one registry, with one API key or anonymously.

    Args:
        api_key: the key to send; ``None`` or empty means anonymous (public skills only).
            :meth:`from_env` reads ``DECIMAL_API_KEY`` instead.
        base_url: the registry, ``https://api.decimal.ai`` by default.
        channel: who is fetching, for the registry's counters (one of :data:`CHANNELS`).
        timeout: seconds per request.
        transport: an ``httpx`` transport to send through (tests).
        sleep: how to wait between retries (tests).

    Raises:
        UsageError: an unknown channel, a timeout that is not positive seconds, an API key
            with control characters, or a base URL that is not a URL.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        channel: str = "sdk",
        timeout: float = DEFAULT_TIMEOUT,
        transport: Optional[httpx.BaseTransport] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        key = (api_key or "").strip() or None
        if key is not None and any(ord(c) < 0x20 or ord(c) == 0x7F for c in key):
            raise UsageError("the API key contains non-printable characters",
                             fix="check `DECIMAL_API_KEY` for a stray newline or tab")
        _check_choice("channel", channel, CHANNELS)
        if (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
                or not math.isfinite(timeout) or timeout <= 0):
            raise UsageError(f"timeout must be a positive number of seconds, not {timeout!r}")
        self.base_url = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
        self.channel = channel
        self.timeout = float(timeout)
        self._api_key = key
        self._sleep = sleep
        try:
            self._http = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(self.timeout),
                                      transport=transport)
        except (httpx.InvalidURL, TypeError, ValueError) as exc:
            raise UsageError(f"the registry URL {self.base_url!r} is not a URL ({exc})",
                             fix="check `--base-url` and `DECIMAL_BASE_URL`") from exc

    @classmethod
    def from_env(
        cls,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        anonymous: bool = False,
        **kwargs: Any,
    ) -> "RegistryClient":
        """A client configured the way ``decimalai.init()`` is.

        The key is ``api_key``, else ``DECIMAL_API_KEY``, else ``DECIMALAI_API_KEY``;
        ``anonymous=True`` sends none even when one is set. The registry is ``base_url``,
        else ``DECIMAL_BASE_URL``, else ``DECIMALAI_BASE_URL``, else the default.
        """
        from .. import _api_key_from_env  # the package's own resolution, alias included

        key = None if anonymous else ((api_key or "").strip() or _api_key_from_env() or None)
        url = (base_url
               or os.environ.get("DECIMAL_BASE_URL", "").strip()
               or os.environ.get("DECIMALAI_BASE_URL", "").strip()
               or DEFAULT_BASE_URL)
        return cls(api_key=key, base_url=url, **kwargs)

    @property
    def anonymous(self) -> bool:
        """True when no API key is sent: only public skills, always the foreign variant."""
        return self._api_key is None

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "RegistryClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        who = "anonymous" if self.anonymous else "with an API key"
        return f"<RegistryClient {self.base_url} {who}>"

    # ── resolve ───────────────────────────────────────────────────────

    def resolve(
        self,
        items: Iterable[Union[ResolveRequest, str, Mapping[str, Any]]],
        *,
        taken: Optional[Mapping[str, Optional[str]]] = None,
        purpose: str,
    ) -> List[Union[Resolved, UnavailableError]]:
        """Resolve each item; one result per item, in order.

        A result is a :class:`Resolved`, or the item's refusal as a typed
        :class:`~decimalai.skillpkg.errors.UnavailableError` (``SkillNotFoundError``,
        ``AmbiguousRefError``, ``SkillWithheldError``, ...), returned, not raised: one
        refused item never hides the others.

        ``taken`` maps each install name this project already uses to its skill id, or to
        ``None`` for a folder the engine does not manage; the registry picks names around
        them. More than :data:`MAX_RESOLVE_ITEMS` items are sent in batches, and each
        batch's chosen names are added to ``taken`` for the next.

        Raises:
            UsageError: a malformed item, ``taken`` or ``purpose``.
            RegistryTooOldError: the registry has no resolve route.
            AccessDeniedError: the registry rejected the API key.
            NetworkError: no answer (a timeout, a refused connection, a 5xx).
            BundleContractError: an answer this client cannot use.
        """
        _check_choice("purpose", purpose, PURPOSES)
        requests = [_as_request(item) for item in items]
        names = _check_taken(taken)
        path = _ANON_RESOLVE if self.anonymous else _RESOLVE
        out: List[Union[Resolved, UnavailableError]] = []
        for start in range(0, len(requests), MAX_RESOLVE_ITEMS):
            batch = requests[start:start + MAX_RESOLVE_ITEMS]
            body = {"items": [r.to_json() for r in batch], "taken": dict(names),
                    "channel": self.channel, "purpose": purpose}
            skill = batch[0].label if len(batch) == 1 else None
            resp = self._send("POST", path, json=body, skill=skill)
            if resp.status_code != 200:
                self._refuse(resp, "POST", path, skill=skill, route_is_resolve=True)
            answers = _json_object(resp, "resolve", skill).get("items")
            if not isinstance(answers, list) or len(answers) != len(batch):
                raise BundleContractError(
                    f"the registry answered {len(batch)} resolve item(s) with "
                    f"{len(answers) if isinstance(answers, list) else 'no'} result(s)",
                    skill=skill, fix=_UPGRADE_FIX)
            for request, answer in zip(batch, answers):
                result = _item_result(request, answer)
                out.append(result)
                if isinstance(result, Resolved):
                    names[result.install_name] = result.skill_id
        return out

    def resolve_one(
        self,
        item: Union[ResolveRequest, str, Mapping[str, Any]],
        *,
        taken: Optional[Mapping[str, Optional[str]]] = None,
        purpose: str,
    ) -> Resolved:
        """Resolve one item, raising its refusal instead of returning it."""
        (result,) = self.resolve([item], taken=taken, purpose=purpose)
        if isinstance(result, SkillpkgError):
            raise result
        return result

    # ── bundles ───────────────────────────────────────────────────────

    def fetch_bundle(
        self,
        version_id: str,
        *,
        install_name: str,
        purpose: str,
        render_hash: Optional[str] = None,
        format: Optional[int] = None,
        variant: Optional[str] = None,
        expect_files: Optional[Mapping[str, str]] = None,
        expect_executable: Optional[Iterable[str]] = None,
        expected_from: str = "the lock",
        skill: Optional[str] = None,
        version: Optional[int] = None,
    ) -> Bundle:
        """The files of one rendering, checked; nothing is written.

        ``render_hash``, ``format`` and ``variant`` name the rendering (the lock records
        all three); left out, the registry serves its current one. ``expect_files``
        (path -> sha256) and ``expect_executable`` are what the served files must equal,
        usually the lock's; ``expected_from`` names that source in the error. ``skill``
        and ``version`` are how errors name the skill (a registry name, and the version
        number).

        Raises:
            IntegrityError: a file's bytes differ from its declared sha256, or the files
                differ from ``expect_files`` / ``expect_executable`` (exit 6).
            BundleContractError: the bundle is not one this client may write: an unsafe
                path, a mode or encoding it does not know, a rendering other than the one
                asked for (exit 6).
            LockedVersionRevokedError: the version has since been blocked, withheld or
                marked unsafe (410).
            RenderGoneError: the named rendering no longer exists (409).
            AccessDeniedError: no access, or the authored variant without authorship.
            RegistryTooOldError: the registry has no bundle route.
            NetworkError: no answer.
            UsageError: a malformed argument.
        """
        _check_choice("purpose", purpose, PURPOSES)
        if not isinstance(version_id, str) or not version_id.strip():
            raise UsageError(f"version_id must be a non-empty string, not {version_id!r}")
        if not skillfmt.is_valid_install_name(install_name):
            raise UsageError(f"{install_name!r} is not an install name", skill=skill)
        if variant is not None:
            _check_choice("variant", variant, skillfmt.VARIANTS)
        if format is not None and not _is_supported_format(format):
            raise BundleContractError(
                f"format {format!r} is not one this decimalai reads (it reads format(s) "
                f"{_formats()})", skill=skill or install_name, fix=_UPGRADE_FIX)
        name = skill or install_name
        if variant == "authored" and self.anonymous:
            raise AccessDeniedError(
                "this skill was installed with its authors' API key (the authored variant), "
                "and no key is set", skill=name)
        params: Dict[str, Any] = {"install_name": install_name}
        for key, value in (("render_hash", render_hash), ("format", format), ("variant", variant)):
            if value is not None:
                params[key] = value
        params.update(channel=self.channel, purpose=purpose)
        path = (_ANON_BUNDLES if self.anonymous else _BUNDLES) + _segment(version_id)
        resp = self._send("GET", path, params=params, skill=name)
        if resp.status_code != 200:
            self._refuse(resp, "GET", path, skill=name, version=version)
        bundle = _parse_bundle(_json_object(resp, "bundle", name), skill=name,
                               version_id=version_id, install_name=install_name,
                               render_hash=render_hash, format=format, variant=variant)
        _check_expected(bundle, expect_files, expect_executable, expected_from, skill=name)
        return bundle

    def fetch_resolved(self, resolved: Resolved, *, purpose: str) -> Bundle:
        """The bundle of a :class:`Resolved`: exactly its rendering, checked against its files."""
        bundle = self.fetch_bundle(
            resolved.version_id, install_name=resolved.install_name, purpose=purpose,
            render_hash=resolved.render.hash, format=resolved.render.format,
            variant=resolved.render.variant, expect_files=resolved.file_hashes,
            expect_executable=resolved.executable, expected_from="the resolve answer",
            skill=resolved.registry_name, version=resolved.version,
        )
        if bundle.skill_id is not None and bundle.skill_id != resolved.skill_id:
            raise BundleContractError(
                f"the bundle is of skill {bundle.skill_id}, not {resolved.skill_id}",
                skill=resolved.registry_name, fix=_UPGRADE_FIX)
        return bundle

    def fetch_registry_bundle(
        self, ref: str, *, version: Optional[Union[int, str]] = None, purpose: str = "pull",
    ) -> Bundle:
        """A public skill's current rendering under its project-free install name.

        For a loose copy (``skills pull``): always anonymous, even when this client holds
        a key, and always the foreign variant. ``ref`` is a skill id, url_slug or name.

        Raises the same errors as :meth:`fetch_bundle`, except the expectations.
        """
        _check_choice("purpose", purpose, PURPOSES)
        if not isinstance(ref, str) or not ref.strip():
            raise UsageError(f"a reference must be a non-empty string, not {ref!r}")
        params: Dict[str, Any] = {}
        if version is not None and version != "latest":
            params["version"] = _version_spec(version)
        params.update(channel=self.channel, purpose=purpose)
        path = f"{_REGISTRY_SKILLS}{_path_text(ref)}/bundle"
        resp = self._send("GET", path, params=params, skill=ref, anonymous=True)
        if resp.status_code != 200:
            self._refuse(resp, "GET", path, skill=ref)
        return _parse_bundle(_json_object(resp, "bundle", ref), skill=ref, variant="foreign")

    # ── transport ─────────────────────────────────────────────────────

    def _headers(self, anonymous: bool) -> Dict[str, str]:
        if anonymous:
            headers = {"User-Agent": sdk_user_agent("skills")}
        else:
            headers = sdk_headers(self._api_key or "", "skills")
        headers["Accept"] = "application/json"
        return headers

    def _send(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json: Any = None,
        skill: Optional[str] = None,
        anonymous: Optional[bool] = None,
    ) -> httpx.Response:
        """One request, retried past admission aborts and one timeout; any answer returns."""
        headers = self._headers(self.anonymous if anonymous is None else anonymous)
        timeouts_left = _TIMEOUT_RETRIES
        retries = 0
        waited = 0.0
        while True:
            try:
                resp = self._http.request(method, path, params=params, json=json, headers=headers)
            except httpx.TimeoutException as exc:
                if timeouts_left > 0:
                    timeouts_left -= 1
                    continue
                raise NetworkError(
                    f"the registry at {self.base_url} did not answer within "
                    f"{self.timeout:g}s ({method} {path})",
                    skill=skill, details={"route": f"{method} {path}"}) from exc
            except (httpx.HTTPError, httpx.InvalidURL) as exc:
                raise NetworkError(
                    f"could not reach the registry at {self.base_url} ({type(exc).__name__}: {exc})",
                    skill=skill, details={"route": f"{method} {path}"}) from exc
            if (resp.status_code in _ADMISSION_STATUSES
                    and not resp.headers.get("X-Quota-Exceeded")
                    and retries < len(_RETRY_DELAYS_S)):
                delay = max(_parse_retry_after(resp.headers.get("Retry-After")),
                            _RETRY_DELAYS_S[retries])
                if waited + delay <= _MAX_RETRY_WAIT_S:
                    retries += 1
                    waited += delay
                    self._sleep(delay)
                    continue
            return resp

    def _refuse(
        self,
        resp: httpx.Response,
        method: str,
        path: str,
        *,
        skill: Optional[str],
        version: Optional[int] = None,
        route_is_resolve: bool = False,
    ) -> None:
        """Raise the error for a response that is not a 200."""
        status = resp.status_code
        ref = _Refusal.of(resp)
        details = {k: v for k, v in (
            ("status", status), ("route", f"{method} {path}"), ("registry_code", ref.code),
            ("request_id", ref.request_id),
        ) if v is not None}
        said = f": {ref.message}" if ref.message else ""

        # The resolve route never refuses a whole batch with a 404, so any 404 there means
        # the route is absent. Elsewhere, only the framework's own 404 does: a handler's
        # 404 is an answer ("no access").
        if status in (404, 405) and (route_is_resolve or ref.framework):
            raise RegistryTooOldError(
                f"the registry at {self.base_url} has no install routes (HTTP {status} on "
                f"{method} {path}); this decimalai needs a newer registry",
                skill=skill, details=details)
        if status == 401:
            raise AccessDeniedError(f"the registry rejected the API key (HTTP 401){said}",
                                    skill=skill, details=details)
        if status == 403:
            raise AccessDeniedError(ref.message or "the API key may not fetch this (HTTP 403)",
                                    skill=skill, details=details)
        if status == 404:
            if ref.code in (None, AccessDeniedError.code):
                raise AccessDeniedError(
                    ref.message or "the registry has no such version that this API key can see",
                    skill=skill, details=details)
            raise UnavailableError.for_code(ref.code, ref.message or f"HTTP 404 ({ref.code})",
                                            skill=skill, details=details)
        if status == 410:
            reason = ref.reason or (ref.code if ref.code not in (None, "revoked") else "revoked")
            details["reason"] = reason
            what = f"version {version}" if version is not None else "this version"
            raise LockedVersionRevokedError(f"{what} is no longer available ({reason})",
                                            skill=skill, details=details)
        if status == 409 and ref.code in (None, RenderGoneError.code):
            raise RenderGoneError(ref.message or "the locked rendering no longer exists (HTTP 409)",
                                  skill=skill, details=details)
        if status == 429:
            quota = resp.headers.get("X-Quota-Exceeded")
            if quota:
                raise NetworkError(f"the registry's {quota} quota is spent (HTTP 429){said}",
                                   skill=skill, details=details,
                                   fix="wait for the quota to reset, or raise the plan's limit")
            raise NetworkError(f"the registry is still busy after retries (HTTP 429){said}",
                               skill=skill, details=details)
        if status >= 500:
            raise NetworkError(f"the registry failed (HTTP {status}){said}",
                               skill=skill, details=details)
        raise UnavailableError.for_code(
            ref.code or f"http_{status}", ref.message or f"the registry refused it (HTTP {status})",
            skill=skill, details=details)


# ── Parsing what the registry sends ──────────────────────────────────────────────


@dataclass(frozen=True)
class _Refusal:
    """The parts of an error response the client reads, whichever envelope carried them."""

    code: Optional[str]
    message: Optional[str]
    reason: Optional[str]
    request_id: Optional[str]
    framework: bool   # the framework's own "no such route"

    @classmethod
    def of(cls, resp: httpx.Response) -> "_Refusal":
        try:
            body = resp.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            # No JSON at all: a proxy's or load balancer's page, never an install route's.
            return cls(None, None, None, resp.headers.get("X-Request-ID"), True)
        detail = body.get("detail")
        code = message = reason = None
        if isinstance(detail, dict):
            code = _text(detail.get("code")) or _text(detail.get("error"))
            message = _text(detail.get("message")) or _text(detail.get("detail"))
            reason = _text(detail.get("reason"))
        elif isinstance(detail, str):
            message = detail
        top = _text(body.get("code"))
        if code is None and top is not None and not top.startswith("HTTP_"):
            code = top
        reason = reason or _text(body.get("reason"))
        message = message or _text(body.get("message"))
        framework = (code is None and reason is None
                     and (detail is None or (isinstance(detail, str) and detail in _FRAMEWORK_DETAILS)))
        request_id = _text(body.get("request_id")) or resp.headers.get("X-Request-ID")
        return cls(code, message, reason, request_id, framework)


def _text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _json_object(resp: httpx.Response, what: str, skill: Optional[str]) -> Dict[str, Any]:
    try:
        data = resp.json()
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise BundleContractError(f"the registry's {what} answer is not a JSON object",
                                  skill=skill, fix=_UPGRADE_FIX)
    return data


def _item_result(request: ResolveRequest, answer: Any) -> Union[Resolved, UnavailableError]:
    skill = request.label
    if isinstance(answer, dict) and answer.get("ok") is True:
        return _parse_resolved(request, answer)
    error = answer.get("error") if isinstance(answer, dict) else None
    if (isinstance(answer, dict) and answer.get("ok") is False and isinstance(error, dict)
            and _text(error.get("code"))):
        code = error["code"]
        extra = {k: v for k, v in error.items() if k not in ("code", "message")}
        return UnavailableError.for_code(
            code, _text(error.get("message")) or f"the registry refused it ({code})",
            skill=skill, details=extra or None)
    raise BundleContractError("the registry's resolve answer has an item that is neither a "
                              "result nor an error", skill=skill, fix=_UPGRADE_FIX)


class _Fields:
    """Typed reads of one answer object; a missing or mistyped field is a contract error."""

    def __init__(self, data: Mapping[str, Any], what: str, skill: Optional[str]) -> None:
        self.data, self.what, self.skill = data, what, skill

    def fail(self, message: str) -> BundleContractError:
        return BundleContractError(f"the registry's {self.what} {message}", skill=self.skill,
                                   fix=_UPGRADE_FIX)

    def string(self, key: str, *, required: bool = True) -> Optional[str]:
        value = self.data.get(key)
        if value is None and not required:
            return None
        if not isinstance(value, str) or not value:
            raise self.fail(f"has no `{key}`" if value is None else f"has a bad `{key}`: {value!r}")
        return value

    def positive(self, key: str) -> int:
        value = self.data.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise self.fail(f"has no `{key}`" if value is None else f"has a bad `{key}`: {value!r}")
        return value

    def array(self, key: str, *, required: bool = True) -> List[Any]:
        value = self.data.get(key)
        if value is None and not required:
            return []
        if not isinstance(value, list):
            raise self.fail(f"has no `{key}` list" if value is None else f"has a bad `{key}`: {value!r}")
        return value

    def install_name(self) -> str:
        value = self.data.get("install_name")
        if not skillfmt.is_valid_install_name(value):
            raise self.fail("has no `install_name`" if value is None
                            else f"names a folder no loader accepts: {value!r}")
        return value

    def render(self) -> RenderKey:
        raw = self.data.get("render")
        if not isinstance(raw, dict):
            raise self.fail("has no `render`")
        render = _Fields(raw, f"{self.what} render", self.skill)
        fmt = raw.get("format")
        if not _is_supported_format(fmt):
            raise BundleContractError(
                f"the registry rendered format {fmt!r}; this decimalai reads format(s) {_formats()}",
                skill=self.skill, fix=_UPGRADE_FIX)
        variant = render.string("variant")
        if variant not in skillfmt.VARIANTS:
            raise self.fail(f"has an unknown variant {variant!r}")
        return RenderKey(format=fmt, hash=render.string("hash") or "", variant=variant)

    def unavailable(self) -> Tuple[Unavailable, ...]:
        out = []
        for entry in self.array("unavailable", required=False):
            row = _Fields(entry if isinstance(entry, dict) else {}, f"{self.what} unavailable entry",
                          self.skill)
            out.append(Unavailable(row.string("path") or "", row.string("reason") or ""))
        return tuple(out)

    def notes(self) -> Tuple[str, ...]:
        notes = self.array("policy_notes", required=False)
        if not all(isinstance(n, str) for n in notes):
            raise self.fail(f"has a bad `policy_notes`: {notes!r}")
        return tuple(notes)


def _parse_resolved(request: ResolveRequest, data: Mapping[str, Any]) -> Resolved:
    skill = request.label
    f = _Fields(data, f"resolve answer for {skill}", skill)
    entries = []
    for raw in f.array("files"):
        row = _Fields(raw if isinstance(raw, dict) else {}, f"resolve answer for {skill}", skill)
        entries.append(FileEntry(path=row.string("path") or "", sha256=_sha256_of(row, "sha256"),
                                 mode=_mode_of(row)))
    _check_paths([e.path for e in entries], skill)
    latest = None
    if isinstance(data.get("latest"), dict):
        row = _Fields(data["latest"], f"resolve answer for {skill} latest", skill)
        latest = VersionRef(row.string("version_id") or "", row.positive("version"))
    raw_safety = data.get("safety") if isinstance(data.get("safety"), dict) else {}
    return Resolved(
        ref=skill,
        skill_id=f.string("skill_id") or "",
        registry_name=f.string("registry_name") or "",
        install_name=f.install_name(),
        version_id=f.string("version_id") or "",
        version=f.positive("version"),
        render=f.render(),
        body_sha256=f.string("body_sha256") or "",
        description_sha256=f.string("description_sha256") or "",
        files=tuple(entries),
        url_slug=_text(data.get("url_slug")),
        install_name_base=_text(data.get("install_name_base")),
        install_name_disambiguated=_text(data.get("install_name_disambiguated")),
        latest=latest,
        invocation=_text(data.get("invocation")) or "model",
        relationship=_text(data.get("relationship")),
        unavailable=f.unavailable(),
        policy_notes=f.notes(),
        safety=Safety(band=_text(raw_safety.get("band")) or "unreviewed",
                      withheld=raw_safety.get("withheld") is True),
        raw=dict(data),
    )


def _parse_bundle(
    data: Mapping[str, Any],
    *,
    skill: str,
    version_id: Optional[str] = None,
    install_name: Optional[str] = None,
    render_hash: Optional[str] = None,
    format: Optional[int] = None,
    variant: Optional[str] = None,
) -> Bundle:
    f = _Fields(data, "bundle", skill)
    got_version_id = f.string("version_id") or ""
    got_install_name = f.install_name()
    render = f.render()
    for asked, got, what in (
        (version_id, got_version_id, "version_id"),
        (install_name, got_install_name, "install_name"),
        (render_hash, render.hash, "render hash"),
        (format, render.format, "format"),
        (variant, render.variant, "variant"),
    ):
        if asked is not None and got != asked:
            raise f.fail(f"is not the one asked for: {what} {got!r}, asked for {asked!r}")

    rows = []
    for raw in f.array("files"):
        raw = raw if isinstance(raw, dict) else {}
        row = _Fields(raw, "bundle", skill)
        path = row.string("path") or ""
        declared = _sha256_of(row, "sha256")
        mode = _mode_of(row)
        if raw.get("encoding") != "utf-8":
            raise f.fail(f"serves {path!r} as {raw.get('encoding')!r}; this decimalai reads utf-8")
        text = raw.get("content")
        if not isinstance(text, str):
            raise f.fail(f"serves {path!r} without text content")
        try:
            content = text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise f.fail(f"serves {path!r} as text that is not valid UTF-8 ({exc.reason})")
        rows.append((path, declared, mode, content))
    _check_paths([path for path, _, _, _ in rows], skill)

    files = []
    for path, declared, mode, content in rows:
        received = hashlib.sha256(content).hexdigest()
        if received != declared:
            raise IntegrityError(
                f"{path}: the bytes received hash to {received[:12]}..., the registry "
                f"declared {declared[:12]}...",
                skill=skill, path=path,
                details={"declared": declared, "received": received, "size": len(content)})
        files.append(BundleFile(path=path, sha256=received, mode=mode, content=content))

    return Bundle(
        version_id=got_version_id,
        install_name=got_install_name,
        render=render,
        files=tuple(files),
        unavailable=f.unavailable(),
        policy_notes=f.notes(),
        skill_id=_text(data.get("skill_id")),
        body_sha256=_text(data.get("body_sha256")),
        description_sha256=_text(data.get("description_sha256")),
        invocation=_text(data.get("invocation")),
    )


def _sha256_of(row: _Fields, key: str) -> str:
    value = row.string(key) or ""
    if not _SHA256_RE.match(value.lower()):
        raise row.fail(f"declares {key} {value!r}, which is not a sha256")
    return value.lower()


def _mode_of(row: _Fields) -> int:
    value = row.data.get("mode")
    if isinstance(value, int) and not isinstance(value, bool):
        mode = value
    elif isinstance(value, str) and _MODE_RE.match(value):
        mode = int(value, 8)
    else:
        mode = -1
    if mode not in _MODES:
        raise row.fail(f"gives {row.data.get('path')!r} mode {value!r}; a file is 0644 or 0755")
    return mode


def _check_paths(paths: Sequence[str], skill: str) -> None:
    """Every path is safe to write under the skill folder, and lands on its own file."""
    def fail(message: str) -> BundleContractError:
        return BundleContractError(f"the registry served {message}", skill=skill, fix=_UPGRADE_FIX)

    count = sum(1 for p in paths if p == "SKILL.md")
    if count != 1:
        raise fail("no SKILL.md" if count == 0 else "SKILL.md twice")
    seen: Dict[str, str] = {}
    for path in paths:
        if path == "SKILL.md":
            continue
        problems = skillfmt.validate_attachment_path(path)
        if problems:
            raise fail(f"a file it may not: {problems[0].message}")
        key = path.casefold()
        if key in seen:
            raise fail(f"{seen[key]!r} and {path!r}, one file on a case-insensitive disk")
        seen[key] = path
    for key, path in seen.items():
        parts = key.split("/")
        for i in range(1, len(parts)):
            if "/".join(parts[:i]) in seen:
                raise fail(f"{seen['/'.join(parts[:i])]!r} as a file and as the folder of {path!r}")


def _check_expected(
    bundle: Bundle,
    expect_files: Optional[Mapping[str, str]],
    expect_executable: Optional[Iterable[str]],
    expected_from: str,
    *,
    skill: str,
) -> None:
    if expect_files is not None:
        expected = {path: str(digest).lower() for path, digest in expect_files.items()}
        got = bundle.file_hashes
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        differ = sorted(p for p in set(expected) & set(got) if expected[p] != got[p])
        if missing or extra or differ:
            parts = [f"{label} {', '.join(paths)}" for label, paths in (
                ("changed:", differ), ("missing:", missing), ("added:", extra)) if paths]
            raise IntegrityError(
                f"the served files differ from {expected_from} ({'; '.join(parts)})",
                skill=skill, details={"differ": differ, "missing": missing, "extra": extra,
                                      "expected_from": expected_from})
    if expect_executable is not None:
        expected_x = set(expect_executable)
        got_x = set(bundle.executable)
        if expected_x != got_x:
            raise IntegrityError(
                f"the served exec bits differ from {expected_from} (executable: "
                f"{', '.join(sorted(got_x)) or 'none'}; expected: {', '.join(sorted(expected_x)) or 'none'})",
                skill=skill, details={"executable": sorted(got_x), "expected": sorted(expected_x),
                                      "expected_from": expected_from})


# ── small helpers ────────────────────────────────────────────────────────────────


def _as_request(item: Any) -> ResolveRequest:
    if isinstance(item, ResolveRequest):
        return item
    if isinstance(item, str):
        return ResolveRequest(ref=item)
    if isinstance(item, Mapping):
        unknown = set(item) - {"ref", "version", "skill_id", "version_id"}
        if unknown:
            raise UsageError(f"unknown resolve item keys: {sorted(unknown)}")
        return ResolveRequest(**dict(item))
    raise UsageError(f"a resolve item is a reference or a ResolveRequest, not {item!r}")


def _check_taken(taken: Optional[Mapping[str, Optional[str]]]) -> Dict[str, Optional[str]]:
    if taken is None:
        return {}
    if not isinstance(taken, Mapping):
        raise UsageError(f"taken maps install names to skill ids, not {taken!r}")
    out: Dict[str, Optional[str]] = {}
    for name, skill_id in taken.items():
        if not isinstance(name, str) or not (skill_id is None or isinstance(skill_id, str)):
            raise UsageError(f"taken maps install names to a skill id or None, not "
                             f"{name!r}: {skill_id!r}")
        out[name] = skill_id
    return out


def _check_choice(what: str, value: Any, allowed: Sequence[str]) -> None:
    if value not in allowed:
        raise UsageError(f"{what} must be one of {', '.join(allowed)}, not {value!r}")


def _segment(value: str) -> str:
    """One path segment: every reserved character encoded, ``/`` included."""
    return urllib.parse.quote(value, safe="")


def _path_text(ref: str) -> str:
    """A reference as path text: ``#``, ``?``, ``%``, spaces and non-ASCII encoded.

    ``/`` stays literal. Namespaced registry names (``owner/name``) have always travelled
    that way to the delivery routes, and it is the form proven to pass the proxies in
    front of the registry; an encoded ``%2F`` is decoded before routing anyway.
    """
    return urllib.parse.quote(ref, safe="/")


def _is_supported_format(value: Any) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)
            and value in skillfmt.SUPPORTED_FORMATS)


def _formats() -> str:
    return ", ".join(str(f) for f in sorted(skillfmt.SUPPORTED_FORMATS))
