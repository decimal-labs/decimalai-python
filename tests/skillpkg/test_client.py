"""The skills engine's registry client: resolve a reference, fetch the bundle, trust nothing.

``decimalai skills add | install | update`` reach the registry through exactly two calls
(and their anonymous twins for public skills): ``resolve`` turns references into a chosen
version, render and install name; ``bundle`` serves that rendering's bytes. Everything the
CLI promises about the network rests on this client, so this file pins:

- the request shapes, authenticated and anonymous;
- that the install rail is cold: no ``routing_id``, the 30 s budget, never counted as a
  failed agent turn by ``routing_status()``;
- that references are URL-encoded where they travel in a path;
- that every per-item refusal is a typed error with the documented exit code (7);
- that a registry without these routes is ``RegistryTooOldError`` and the client never
  falls back to ``/body``;
- that a network failure or timeout is exit 8;
- that bytes are checked on receipt: a file whose sha256 differs from its declaration, or
  from the lock, is exit 6, and so is a bundle that breaks the on-disk contract.

The registry is ``tests/fakes/fake_registry.py``, which renders with the same format
module the platform uses.
"""
from __future__ import annotations

import hashlib
import json
import socket
import time

import httpx
import pytest

import decimalai.skill_router as sr
from decimalai.skill_router import _COLD_PATH_TIMEOUT, _HotPathBreakers, _hot_path_read_budget, routing_status
from decimalai.skillpkg.client import (
    DEFAULT_TIMEOUT,
    FORMATS,
    MAX_ANONYMOUS_RESOLVE_ITEMS,
    MAX_RESOLVE_ITEMS,
    Bundle,
    RegistryClient,
    ResolveRequest,
    Resolved,
)
from decimalai.skillpkg.errors import (
    AccessDeniedError,
    AmbiguousRefError,
    BundleContractError,
    ExitCode,
    InstallNameCollisionError,
    IntegrityError,
    LockedVersionRevokedError,
    NetworkError,
    RegistryTooOldError,
    RenderGoneError,
    RenderInvalidError,
    SkillBlockedError,
    SkillNotFoundError,
    SkillWithheldError,
    UnavailableError,
    UsageError,
    VersionUnsafeError,
)
from tests.fakes.fake_registry import (
    ANON_BUNDLES,
    ANON_RESOLVE,
    AUTH_BUNDLES,
    AUTH_RESOLVE,
    BASE_URL,
    OTHER_KEY,
    OTHER_ORG,
    OWNER_KEY,
    OWNER_ORG,
    REGISTRY_SKILLS,
    FakeRegistry,
)
from skillevaluation import skillfmt

REFUND_FILES = (
    {"path": "scripts/compute_refund.py", "content": "print('refund')\n", "executable": True},
    {"path": "references/fees.md", "content": "# Fees\n\nA 5% restocking fee.\n"},
    {"path": "assets/logo.png", "content": None},  # binary: listed, never served
)


@pytest.fixture
def registry() -> FakeRegistry:
    return FakeRegistry()


@pytest.fixture
def sleeps() -> list:
    return []


def make_client(registry: FakeRegistry, sleeps: list, *, key=OWNER_KEY, **kwargs) -> RegistryClient:
    kwargs.setdefault("channel", "cli")
    return RegistryClient(api_key=key, base_url=BASE_URL, transport=registry.transport(),
                          sleep=sleeps.append, **kwargs)


def refund(registry: FakeRegistry, **kwargs):
    return registry.add_skill("acme/refund-policy", org=OTHER_ORG, files=REFUND_FILES, **kwargs)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def respond(status: int, body=None, headers=None, text=None) -> httpx.Response:
    if text is not None:
        return httpx.Response(status, text=text, headers=headers or {})
    return httpx.Response(status, json=body, headers=headers or {})


def add_file(data: dict, path: str, content: str = "x\n", mode: str = "0644") -> None:
    """A tamper hook body: one more file whose declared hash matches its bytes."""
    data["files"].append({"path": path, "sha256": sha(content), "mode": mode,
                          "encoding": "utf-8", "content": content})


# ── request shapes ───────────────────────────────────────────────────────────────


class TestResolveRequests:
    def test_the_authenticated_resolve_posts_the_documented_body(self, registry, sleeps):
        refund(registry)
        own = registry.add_skill("pdf-tools", org=OWNER_ORG, public=False, versions=3)
        client = make_client(registry, sleeps)

        out = client.resolve(
            [ResolveRequest(ref="acme/refund-policy"),
             ResolveRequest(ref="pdf-tools", version=3),
             ResolveRequest(skill_id=own.skill_id, version_id=own.version(2).version_id)],
            taken={"my-own-skill": None}, purpose="add",
        )

        (call,) = registry.calls("resolve")
        assert (call.method, call.path) == ("POST", AUTH_RESOLVE)
        assert call.headers["authorization"] == f"Bearer {OWNER_KEY}"
        assert call.json == {
            "items": [
                {"ref": "acme/refund-policy", "version": "latest"},
                {"ref": "pdf-tools", "version": 3},
                {"skill_id": own.skill_id, "version_id": own.version(2).version_id},
            ],
            "taken": {"my-own-skill": None},
            "channel": "cli",
            "purpose": "add",
            "formats": [1],
        }
        assert all(isinstance(r, Resolved) for r in out)
        assert [r.version for r in out] == [1, 3, 2]

    def test_the_anonymous_resolve_uses_the_registry_twin_and_sends_no_key(self, registry, sleeps):
        refund(registry)
        client = make_client(registry, sleeps, key=None)
        assert client.anonymous

        (result,) = client.resolve([ResolveRequest(ref="acme/refund-policy")], purpose="pull")

        (call,) = registry.calls("resolve")
        assert call.path == ANON_RESOLVE
        assert "authorization" not in call.headers
        assert call.json["purpose"] == "pull" and call.json["channel"] == "cli"
        assert result.render.variant == "foreign"

    def test_a_resolved_item_carries_every_documented_field(self, registry, sleeps):
        skill = refund(registry)
        r = make_client(registry, sleeps).resolve_one(
            ResolveRequest(ref="acme/refund-policy"), purpose="add")
        bundle = registry.render(skill, install_name="acme-refund-policy", variant="foreign")

        assert r.ref == "acme/refund-policy"
        assert r.skill_id == skill.skill_id
        assert r.registry_name == "acme/refund-policy"
        assert r.url_slug == "acme-refund-policy"
        assert r.install_name == "acme-refund-policy"
        assert (r.install_name_base, r.install_name_disambiguated) == skillfmt.install_name_forms(
            skill.name, skill.skill_id)
        assert (r.version_id, r.version) == (skill.newest.version_id, 1)
        assert (r.latest.version_id, r.latest.version) == (skill.newest.version_id, 1)
        assert (r.render.format, r.render.hash, r.render.variant) == (1, bundle.render_hash, "foreign")
        assert (r.body_sha256, r.description_sha256) == (bundle.body_sha256, bundle.description_sha256)
        assert r.invocation == "model" and r.relationship == "public"
        assert r.file_hashes == {f.path: f.sha256 for f in bundle.files}
        assert r.executable == ("scripts/compute_refund.py",)
        assert [(u.path, u.reason) for u in r.unavailable] == [
            (u.path, u.reason) for u in bundle.unavailable]
        assert r.policy_notes == bundle.policy_notes
        assert (r.safety.band, r.safety.withheld) == ("clean", False)

    def test_requests_identify_the_sdk(self, registry, sleeps):
        refund(registry)
        make_client(registry, sleeps).resolve([ResolveRequest(ref="acme/refund-policy")], purpose="add")
        (call,) = registry.calls("resolve")
        assert call.headers["user-agent"].startswith("decimalai-sdk/")

    @pytest.mark.parametrize("given,sent", [("3", 3), (3, 3), ("latest", "latest")])
    def test_a_manifest_version_string_is_sent_as_a_number(self, given, sent):
        assert ResolveRequest(ref="pdf-tools", version=given).to_json() == {
            "ref": "pdf-tools", "version": sent}

    def test_a_plain_string_is_a_ref_at_latest(self, registry, sleeps):
        refund(registry)
        (r,) = make_client(registry, sleeps).resolve(["acme/refund-policy"], purpose="add")
        assert isinstance(r, Resolved)
        assert registry.calls("resolve")[0].json["items"] == [
            {"ref": "acme/refund-policy", "version": "latest"}]

    @pytest.mark.parametrize("kwargs", [
        {},
        {"ref": ""},
        {"ref": "   "},
        {"ref": "x", "version": 0},
        {"ref": "x", "version": -1},
        {"ref": "x", "version": "v3"},
        {"ref": "x", "version": "03x"},
        {"ref": "x", "version": True},
        {"ref": "x", "version": 1.5},
        {"ref": 3},
        {"version_id": "9b2e"},
        {"skill_id": ""},
    ])
    def test_a_malformed_item_is_a_usage_error(self, kwargs):
        with pytest.raises(UsageError) as excinfo:
            ResolveRequest(**kwargs)
        assert excinfo.value.exit_code == ExitCode.USAGE

    def test_unknown_purposes_channels_and_taken_shapes_are_usage_errors(self, registry, sleeps):
        client = make_client(registry, sleeps)
        with pytest.raises(UsageError):
            client.resolve(["acme"], purpose="instal")
        with pytest.raises(UsageError):
            client.resolve(["acme"], taken={"x": 3}, purpose="add")
        with pytest.raises(UsageError):
            client.resolve([{"ref": "acme", "versoin": 3}], purpose="add")
        with pytest.raises(UsageError):
            make_client(registry, sleeps, channel="clii")
        assert registry.requests == []

    def test_no_items_make_no_request(self, registry, sleeps):
        assert make_client(registry, sleeps).resolve([], purpose="install") == []
        assert registry.requests == []

    def test_more_than_one_batch_is_split_and_earlier_names_are_taken(self, registry, sleeps):
        # The 1st and the 101st skill have the same base install name. The registry only
        # knows the names it is told about, so the second batch must carry the first
        # batch's choices, or both skills would be handed one folder.
        skills = [registry.add_skill("acme/policy")]
        skills += [registry.add_skill(f"team/skill-{i}") for i in range(MAX_RESOLVE_ITEMS - 1)]
        skills.append(registry.add_skill("acme-policy", url_slug="acme-policy-2"))
        client = make_client(registry, sleeps)

        out = client.resolve([ResolveRequest(skill_id=s.skill_id) for s in skills],
                             taken={"hand-made": None}, purpose="install")

        first, second = registry.calls("resolve")
        assert len(first.json["items"]) == MAX_RESOLVE_ITEMS == 100
        assert len(second.json["items"]) == 1
        assert first.json["taken"] == {"hand-made": None}
        assert second.json["taken"] == {
            "hand-made": None, **{r.install_name: r.skill_id for r in out[:MAX_RESOLVE_ITEMS]}}
        assert [r.skill_id for r in out] == [s.skill_id for s in skills]
        assert out[0].install_name == "acme-policy"
        assert out[-1].install_name == skillfmt.install_name_disambiguated("acme-policy", skills[-1].skill_id)
        assert len({r.install_name for r in out}) == len(out)

    def test_anonymous_resolves_go_in_batches_of_20_and_carry_earlier_names(
            self, registry, sleeps):
        # The 1st and the 21st skill share a base install name.
        skills = [registry.add_skill("acme/policy")]
        skills += [registry.add_skill(f"team/skill-{i}")
                   for i in range(MAX_ANONYMOUS_RESOLVE_ITEMS - 1)]
        skills.append(registry.add_skill("acme-policy", url_slug="acme-policy-2"))
        skills += [registry.add_skill(f"more/skill-{i}") for i in range(20)]
        out = make_client(registry, sleeps, key=None).resolve(
            [ResolveRequest(skill_id=s.skill_id) for s in skills], purpose="install")

        calls = registry.calls("resolve")
        assert MAX_ANONYMOUS_RESOLVE_ITEMS == 20
        assert [len(c.json["items"]) for c in calls] == [20, 20, 1]
        assert all(c.path == ANON_RESOLVE for c in calls)
        assert calls[1].json["taken"] == {r.install_name: r.skill_id for r in out[:20]}
        assert [r.skill_id for r in out] == [s.skill_id for s in skills]
        assert len({r.install_name for r in out}) == len(out)

    def test_a_lower_cap_the_registry_names_is_followed(self, registry, sleeps):
        skills = [registry.add_skill(f"team/skill-{i}") for i in range(12)]
        registry.fault("resolve", respond(422, {"detail": {
            "code": "too_many_items", "message": "at most 5 items", "max_items": 5}}))
        out = make_client(registry, sleeps).resolve(
            [ResolveRequest(skill_id=s.skill_id) for s in skills], purpose="install")
        assert [len(c.json["items"]) for c in registry.calls("resolve")] == [12, 5, 5, 2]
        assert [r.skill_id for r in out] == [s.skill_id for s in skills]

    def test_a_cap_refusal_without_a_smaller_cap_is_not_retried(self, registry, sleeps):
        registry.add_skill("team/a")
        registry.fault("resolve", respond(422, {"detail": {
            "code": "too_many_items", "message": "no", "max_items": 50}}))
        with pytest.raises(UnavailableError) as excinfo:
            make_client(registry, sleeps).resolve(["team/a"], purpose="install")
        assert excinfo.value.details["registry_code"] == "too_many_items"
        assert len(registry.calls("resolve")) == 1

    @pytest.mark.parametrize("key,cap", [(None, 20), (OWNER_KEY, 100)])
    def test_the_fake_enforces_the_caps(self, registry, key, cap):
        path = ANON_RESOLVE if key is None else AUTH_RESOLVE
        headers = {} if key is None else {"Authorization": f"Bearer {key}"}
        with httpx.Client(base_url=BASE_URL, transport=registry.transport()) as http:
            over = http.post(path, headers=headers,
                             json={"items": [{"ref": "x"}] * (cap + 1), "formats": [1]})
            at = http.post(path, headers=headers,
                           json={"items": [{"ref": "x"}] * cap, "formats": [1]})
        assert over.status_code == 422
        assert over.json()["detail"]["code"] == "too_many_items"
        assert over.json()["detail"]["max_items"] == cap
        assert at.status_code == 200

    def test_resolve_one_returns_the_item_or_raises_its_error(self, registry, sleeps):
        refund(registry)
        client = make_client(registry, sleeps)
        assert client.resolve_one("acme/refund-policy", purpose="add").install_name == "acme-refund-policy"
        with pytest.raises(SkillNotFoundError) as excinfo:
            client.resolve_one("acme/refund-polcy", purpose="add")
        assert excinfo.value.skill == "acme/refund-polcy"
        assert "acme/refund-policy" in excinfo.value.details["near"]


class TestBundleRequests:
    def test_the_authenticated_bundle_get_sends_the_documented_query(self, registry, sleeps):
        refund(registry)
        client = make_client(registry, sleeps)
        r = client.resolve_one("acme/refund-policy", purpose="add")

        bundle = client.fetch_bundle(r.version_id, install_name=r.install_name,
                                     render_hash=r.render.hash, format=1, variant="foreign",
                                     purpose="add")

        (call,) = registry.calls("bundle")
        assert (call.method, call.path) == ("GET", AUTH_BUNDLES + r.version_id)
        assert call.query == {
            "install_name": "acme-refund-policy", "render_hash": r.render.hash, "format": "1",
            "variant": "foreign", "formats": "1", "channel": "cli", "purpose": "add",
        }
        assert call.headers["authorization"] == f"Bearer {OWNER_KEY}"
        assert call.json is None
        assert isinstance(bundle, Bundle) and bundle.version_id == r.version_id

    def test_unset_parameters_are_left_to_the_registry(self, registry, sleeps):
        skill = refund(registry)
        make_client(registry, sleeps).fetch_bundle(
            skill.newest.version_id, install_name="acme-refund-policy", purpose="frozen")
        (call,) = registry.calls("bundle")
        assert call.query == {"install_name": "acme-refund-policy", "formats": "1",
                              "channel": "cli", "purpose": "frozen"}

    def test_the_anonymous_bundle_uses_the_registry_twin(self, registry, sleeps):
        skill = refund(registry)
        make_client(registry, sleeps, key=None).fetch_bundle(
            skill.newest.version_id, install_name="acme-refund-policy", variant="foreign",
            purpose="frozen")
        (call,) = registry.calls("bundle")
        assert call.path == ANON_BUNDLES + skill.newest.version_id
        assert "authorization" not in call.headers

    def test_an_anonymous_client_never_asks_for_the_authored_variant(self, registry, sleeps):
        skill = registry.add_skill("pdf-tools", org=OWNER_ORG)
        with pytest.raises(AccessDeniedError) as excinfo:
            make_client(registry, sleeps, key=None).fetch_bundle(
                skill.newest.version_id, install_name="pdf-tools", variant="authored",
                purpose="frozen", skill="pdf-tools")
        assert excinfo.value.exit_code == ExitCode.UNAVAILABLE
        assert "DECIMAL_API_KEY" in str(excinfo.value)
        assert registry.requests == []

    def test_fetch_resolved_asks_for_exactly_what_resolve_chose(self, registry, sleeps):
        registry.add_skill("pdf-tools", org=OWNER_ORG, files=REFUND_FILES[:1])
        client = make_client(registry, sleeps)
        r = client.resolve_one("pdf-tools", purpose="add")
        assert r.render.variant == "authored" and r.relationship == "owned"

        bundle = client.fetch_resolved(r, purpose="add")

        (call,) = registry.calls("bundle")
        assert call.query == {
            "install_name": r.install_name, "render_hash": r.render.hash,
            "format": str(r.render.format), "variant": "authored", "formats": "1",
            "channel": "cli", "purpose": "add",
        }
        assert bundle.file_hashes == r.file_hashes
        assert bundle.executable == ("scripts/compute_refund.py",)

    def test_the_registry_bundle_route_is_anonymous_even_with_a_key(self, registry, sleeps):
        refund(registry)
        bundle = make_client(registry, sleeps).fetch_registry_bundle(
            "acme-refund-policy", version=1, purpose="pull")
        (call,) = registry.calls("registry_bundle")
        assert call.path == REGISTRY_SKILLS + "acme-refund-policy/bundle"
        assert "authorization" not in call.headers
        assert call.query == {"version": "1", "formats": "1", "channel": "cli", "purpose": "pull"}
        assert bundle.install_name == "acme-refund-policy"
        assert bundle.render.variant == "foreign"

    @pytest.mark.parametrize("kwargs", [
        {"install_name": "../escape"},
        {"install_name": "Acme"},
        {"variant": "mine"},
        {"purpose": "steal"},
    ])
    def test_bad_arguments_are_refused_before_any_request(self, registry, sleeps, kwargs):
        args = {"install_name": "acme-refund-policy", "purpose": "frozen", **kwargs}
        with pytest.raises(UsageError):
            make_client(registry, sleeps).fetch_bundle("9b2e", **args)
        assert registry.requests == []

    def test_a_format_this_client_cannot_read_is_refused_before_any_request(self, registry, sleeps):
        with pytest.raises(BundleContractError) as excinfo:
            make_client(registry, sleeps).fetch_bundle(
                "9b2e", install_name="acme-refund-policy", format=2, purpose="frozen")
        assert "pip install -U decimalai" in str(excinfo.value)
        assert registry.requests == []


# ── the install rail is cold and carries no routing decision ─────────────────────


def _keys(value) -> set:
    """Every mapping key anywhere in a JSON value."""
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in _keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in _keys(v)}
    return set()


class TestTheInstallRailIsColdAndUnrouted:
    """Disk export was pinned cold (`hot=False`) and free of `routing_id`; installs keep both.

    An install is a CLI action, not an agent turn: a slow registry must not fail it at the
    5 s hot budget, its failures must not open the breakers agents depend on, and it must
    not claim a routing decision it never made.
    """

    def _every_call(self, registry, sleeps):
        refund(registry)
        for client in (make_client(registry, sleeps), make_client(registry, sleeps, key=None)):
            r = client.resolve_one("acme/refund-policy", purpose="add")
            client.fetch_resolved(r, purpose="add")
            client.fetch_bundle(r.version_id, install_name=r.install_name, purpose="frozen")
            client.fetch_registry_bundle("acme-refund-policy", purpose="pull")
        return registry.requests

    def test_no_request_carries_a_routing_id_or_a_hot_flag(self, registry, sleeps):
        calls = self._every_call(registry, sleeps)
        assert len(calls) == 8
        for call in calls:
            names = set(call.query) | _keys(call.json) | set(call.headers)
            assert not {n for n in names if "routing" in n.lower() or n.lower() == "hot"}, call

    def test_every_request_runs_on_the_cold_budget(self, registry, sleeps):
        assert DEFAULT_TIMEOUT == _COLD_PATH_TIMEOUT == 30.0
        assert DEFAULT_TIMEOUT > _hot_path_read_budget()
        for call in self._every_call(registry, sleeps):
            assert call.timeout == {"connect": 30.0, "read": 30.0, "write": 30.0, "pool": 30.0}

    def test_failures_never_reach_the_routing_breakers(self, registry, sleeps, monkeypatch):
        monkeypatch.setattr(sr, "_hot_path_breakers", _HotPathBreakers(threshold=1))
        refund(registry)
        client = make_client(registry, sleeps)
        registry.fault("resolve", "timeout", "timeout", 503, 503, 503, 503, 500)
        for _ in range(3):
            with pytest.raises(NetworkError):
                client.resolve_one("acme/refund-policy", purpose="add")
        with pytest.raises(AccessDeniedError):
            make_client(registry, sleeps, key="dai_sk_revoked").resolve_one("acme", purpose="add")

        status = routing_status()
        assert status.healthy is True
        assert (status.timeouts, status.http_errors, status.consecutive_failures) == (0, 0, 0)
        assert status.breaker_open is False

    def test_it_calls_only_the_install_routes(self, registry, sleeps):
        calls = self._every_call(registry, sleeps)
        assert {c.route for c in calls} == {"resolve", "bundle", "registry_bundle"}
        assert not [c for c in calls if c.path.endswith("/body")
                    or c.path in ("/api/v1/skills/route", "/api/v1/skills/menu")]


# ── references in a path are encoded ─────────────────────────────────────────────


class TestRefsAreUrlEncoded:
    @pytest.mark.parametrize("ref,raw", [
        ("weird#name", "weird%23name"),
        ("what?now", "what%3Fnow"),
        ("has space", "has%20space"),
        ("pct%41", "pct%2541"),
        ("a&b=c", "a%26b%3Dc"),
        ("日本語スキル", "%E6%97%A5%E6%9C%AC%E8%AA%9E%E3%82%B9%E3%82%AD%E3%83%AB"),
        # A namespaced name keeps its slash literal, as the delivery routes always have:
        # it is the form proven to pass the proxies in front of the registry.
        ("acme/refund-policy", "acme/refund-policy"),
    ])
    def test_a_ref_reaches_the_registry_as_exactly_itself(self, registry, sleeps, ref, raw):
        skill = registry.add_skill(ref, url_slug="unrelated-slug")
        bundle = make_client(registry, sleeps, key=None).fetch_registry_bundle(ref, purpose="pull")
        (call,) = registry.calls("registry_bundle")
        assert call.raw_path == f"{REGISTRY_SKILLS}{raw}/bundle"
        assert call.path == f"{REGISTRY_SKILLS}{ref}/bundle"
        assert bundle.skill_id == skill.skill_id

    def test_a_version_id_is_one_encoded_path_segment(self, registry, sleeps):
        with pytest.raises(SkillNotFoundError):
            make_client(registry, sleeps).fetch_bundle(
                "../../admin?x=1", install_name="acme-refund-policy", purpose="frozen")
        (call,) = registry.calls()
        assert call.raw_path == AUTH_BUNDLES + "..%2F..%2Fadmin%3Fx%3D1"
        assert call.query == {"install_name": "acme-refund-policy", "formats": "1",
                              "channel": "cli", "purpose": "frozen"}

    def test_refs_in_a_resolve_body_travel_verbatim(self, registry, sleeps):
        registry.add_skill("weird#name")
        (r,) = make_client(registry, sleeps).resolve(["weird#name"], purpose="add")
        assert registry.calls("resolve")[0].json["items"][0]["ref"] == "weird#name"
        assert r.registry_name == "weird#name"

    def test_an_encoded_ref_survives_a_real_socket(self, registry):
        skill = registry.add_skill("weird#name?x=1 y")
        with registry.serve() as base_url:
            with RegistryClient(api_key=None, base_url=base_url) as client:
                bundle = client.fetch_registry_bundle("weird#name?x=1 y", purpose="pull")
        (call,) = registry.calls("registry_bundle")
        assert call.raw_path == f"{REGISTRY_SKILLS}weird%23name%3Fx%3D1%20y/bundle"
        assert bundle.skill_id == skill.skill_id
        assert bundle.skill_md == registry.render(skill).files[0].content.encode("utf-8")


# ── per-item refusals ────────────────────────────────────────────────────────────


def _scenario(registry: FakeRegistry, code: str):
    """(item, taken) that makes the registry refuse the item with ``code``."""
    if code == "not_found":
        return ResolveRequest(ref="nope"), None
    if code == "ambiguous":
        registry.add_skill("shared/name", org="org-a", url_slug="shared-name-a")
        registry.add_skill("shared/name", org="org-b", url_slug="shared-name-b")
        return ResolveRequest(ref="shared/name"), None
    if code == "withheld":
        registry.add_skill("acme/held", withheld=True)
        return ResolveRequest(ref="acme/held"), None
    if code == "blocked":
        registry.add_skill("acme/bad", band="blocked")
        return ResolveRequest(ref="acme/bad"), None
    if code == "version_unsafe":
        skill = registry.add_skill("acme/risky", versions=2)
        skill.version(2).safe = False
        return ResolveRequest(ref="acme/risky", version=2), None
    if code == "install_name_collision":
        skill = registry.add_skill("acme/crowded")
        base, alt = skillfmt.install_name_forms(skill.name, skill.skill_id)
        return ResolveRequest(ref="acme/crowded"), {base: None, alt: None}
    if code == "render_invalid":
        registry.add_skill("acme/blank", description="")
        return ResolveRequest(ref="acme/blank"), None
    raise AssertionError(code)


PER_ITEM = [
    ("not_found", SkillNotFoundError),
    ("ambiguous", AmbiguousRefError),
    ("withheld", SkillWithheldError),
    ("blocked", SkillBlockedError),
    ("version_unsafe", VersionUnsafeError),
    ("install_name_collision", InstallNameCollisionError),
    ("render_invalid", RenderInvalidError),
]


class TestPerItemErrors:
    @pytest.mark.parametrize("code,cls", PER_ITEM)
    def test_each_code_is_its_typed_error_and_exits_7(self, registry, sleeps, code, cls):
        item, taken = _scenario(registry, code)
        (err,) = make_client(registry, sleeps).resolve([item], taken=taken, purpose="add")
        assert type(err) is cls
        assert err.code == code
        assert err.exit_code == ExitCode.UNAVAILABLE == 7
        assert err.skill == item.ref
        assert err.message, "the registry's own words reach the user"
        assert item.ref in str(err)

    def test_refusals_and_successes_come_back_in_request_order(self, registry, sleeps):
        refund(registry)
        registry.add_skill("acme/bad", band="blocked")
        out = make_client(registry, sleeps).resolve(
            ["acme/refund-policy", "acme/bad", "nope", "acme/refund-policy"], purpose="install")
        assert [type(x) for x in out] == [Resolved, SkillBlockedError, SkillNotFoundError, Resolved]
        assert len(registry.calls("resolve")) == 1

    def test_the_registrys_details_reach_the_error(self, registry, sleeps):
        item, _ = _scenario(registry, "ambiguous")
        (err,) = make_client(registry, sleeps).resolve([item], purpose="add")
        assert err.details["near"] == ["shared-name-a", "shared-name-b"]

    def test_the_latest_of_a_skill_with_an_unsafe_newest_steps_down_and_says_so(self, registry, sleeps):
        skill = registry.add_skill("acme/risky", versions=3)
        skill.version(3).safe = False
        r = make_client(registry, sleeps).resolve_one("acme/risky", purpose="add")
        assert r.version == 2
        assert any("unsafe" in note for note in r.policy_notes)

    def test_a_code_from_a_newer_registry_still_exits_7(self, registry, sleeps):
        registry.fault("resolve", respond(200, {"items": [
            {"ref": "acme", "ok": False, "error": {"code": "quarantined", "message": "held for review"}}]}))
        (err,) = make_client(registry, sleeps).resolve(["acme"], purpose="add")
        assert type(err) is UnavailableError and err.exit_code == 7
        assert err.details["registry_code"] == "quarantined"
        assert err.message == "held for review"

    @pytest.mark.parametrize("answer", [
        {"items": []},
        {"items": [{"ok": True}, {"ok": True}]},
        {"items": "nope"},
        {"results": []},
        {"items": [{"ok": False}]},
        {"items": [{"ok": False, "error": {"message": "no code"}}]},
        {"items": ["acme"]},
    ])
    def test_an_answer_that_does_not_match_the_items_is_a_contract_error(self, registry, sleeps, answer):
        registry.fault("resolve", respond(200, answer))
        with pytest.raises(BundleContractError) as excinfo:
            make_client(registry, sleeps).resolve(["acme"], purpose="add")
        assert excinfo.value.exit_code == ExitCode.INTEGRITY

    @pytest.mark.parametrize("drop", ["skill_id", "registry_name", "install_name", "version_id",
                                      "version", "render", "body_sha256", "description_sha256",
                                      "files"])
    def test_an_ok_item_without_a_field_the_lock_needs_is_a_contract_error(self, registry, sleeps, drop):
        refund(registry)
        registry.tamper_resolve = lambda data: data["items"][0].pop(drop)
        with pytest.raises(BundleContractError) as excinfo:
            make_client(registry, sleeps).resolve(["acme/refund-policy"], purpose="add")
        assert drop in str(excinfo.value)

    @pytest.mark.parametrize("install_name", ["../evil", "a/b", "Acme", "", "yes", 3])
    def test_a_resolved_install_name_that_is_not_a_folder_name_is_refused(self, registry, sleeps, install_name):
        refund(registry)
        registry.tamper_resolve = lambda data: data["items"][0].__setitem__("install_name", install_name)
        with pytest.raises(BundleContractError):
            make_client(registry, sleeps).resolve(["acme/refund-policy"], purpose="add")

    @pytest.mark.parametrize("path", ["../escape.md", "notes/x.md", "references/SKILL.md"])
    def test_a_resolved_file_list_with_an_unsafe_path_is_refused(self, registry, sleeps, path):
        refund(registry)
        registry.tamper_resolve = lambda data: data["items"][0]["files"].append(
            {"path": path, "sha256": "0" * 64, "mode": "0644"})
        with pytest.raises(BundleContractError):
            make_client(registry, sleeps).resolve(["acme/refund-policy"], purpose="add")


# ── a registry without the install routes ───────────────────────────────────────


class TestARegistryWithoutTheInstallRoutes:
    def test_resolve_is_registry_too_old_and_never_falls_back_to_body(self, sleeps):
        registry = FakeRegistry(old=True)
        refund(registry)
        with pytest.raises(RegistryTooOldError) as excinfo:
            make_client(registry, sleeps).resolve(["acme/refund-policy"], purpose="add")
        assert excinfo.value.exit_code == ExitCode.UNAVAILABLE
        assert BASE_URL in str(excinfo.value)
        assert [c.route for c in registry.requests] == ["resolve"]
        # The old route still answers here, so a fallback would have "worked".
        old = httpx.Client(base_url=BASE_URL, transport=registry.transport())
        assert old.get("/api/v1/skills/acme/refund-policy/body").status_code == 200

    @pytest.mark.parametrize("key", [OWNER_KEY, None])
    def test_a_bundle_fetch_is_registry_too_old(self, sleeps, key):
        registry = FakeRegistry(old=True)
        skill = refund(registry)
        with pytest.raises(RegistryTooOldError):
            make_client(registry, sleeps, key=key).fetch_bundle(
                skill.newest.version_id, install_name="acme-refund-policy", purpose="frozen")
        assert [c.route for c in registry.requests] == ["bundle"]

    def test_the_registry_bundle_route_is_registry_too_old(self, sleeps):
        registry = FakeRegistry(old=True)
        refund(registry)
        with pytest.raises(RegistryTooOldError):
            make_client(registry, sleeps).fetch_registry_bundle("acme-refund-policy", purpose="pull")
        assert [c.route for c in registry.requests] == ["registry_bundle"]

    @pytest.mark.parametrize("route", ["resolve", "bundle"])
    @pytest.mark.parametrize("answer", [
        respond(405, {"detail": "Method Not Allowed", "code": "HTTP_405"}),
        respond(404, text="<html><body>404 Not Found</body></html>"),
        respond(404, {"detail": "Not Found"}),
        respond(404, text=""),
    ])
    def test_a_framework_or_proxy_404_is_registry_too_old(self, registry, sleeps, route, answer):
        registry.fault(route, answer)
        client = make_client(registry, sleeps)
        with pytest.raises(RegistryTooOldError):
            if route == "resolve":
                client.resolve(["acme"], purpose="add")
            else:
                client.fetch_bundle("9b2e", install_name="acme", purpose="frozen")
        assert len(registry.requests) == 1
        assert sleeps == []

    def test_a_404_from_the_bundle_handler_is_an_answer_not_an_old_registry(self, registry, sleeps):
        private = registry.add_skill("their/secret", org=OTHER_ORG, public=False)
        with pytest.raises(AccessDeniedError) as excinfo:
            make_client(registry, sleeps).fetch_bundle(
                private.newest.version_id, install_name="their-secret", purpose="frozen",
                skill="their/secret")
        assert type(excinfo.value) is AccessDeniedError
        assert excinfo.value.skill == "their/secret"

    def test_a_404_with_the_handlers_own_words_is_an_answer(self, registry, sleeps):
        registry.fault("bundle", respond(404, {"detail": "Version not found", "code": "HTTP_404"}))
        with pytest.raises(AccessDeniedError):
            make_client(registry, sleeps).fetch_bundle("9b2e", install_name="acme", purpose="frozen")

    @pytest.mark.parametrize("status,cls", [(404, AccessDeniedError), (422, UnavailableError)])
    def test_a_bare_framework_validation_body_is_still_a_typed_error(self, registry, sleeps, status, cls):
        # FastAPI's own shape, without the registry's envelope: `detail` is a list.
        registry.fault("bundle", respond(status, {"detail": [{"loc": ["query"], "msg": "bad"}]}))
        with pytest.raises(cls) as excinfo:
            make_client(registry, sleeps).fetch_bundle("9b2e", install_name="acme", purpose="frozen")
        assert excinfo.value.exit_code == 7

    def test_a_deleted_version_is_not_found(self, registry, sleeps):
        with pytest.raises(SkillNotFoundError):
            make_client(registry, sleeps).fetch_bundle(
                "00000000-0000-4000-8000-0000000000ff", install_name="acme", purpose="frozen")

    def test_a_missing_public_skill_on_the_registry_bundle_route_is_not_found(self, registry, sleeps):
        with pytest.raises(SkillNotFoundError) as excinfo:
            make_client(registry, sleeps).fetch_registry_bundle("nope", purpose="pull")
        assert excinfo.value.skill == "nope"


# ── the network ──────────────────────────────────────────────────────────────────


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestNetworkFailures:
    def test_a_timeout_is_exit_8(self, registry, sleeps):
        registry.fault("resolve", "timeout", "timeout")
        with pytest.raises(NetworkError) as excinfo:
            make_client(registry, sleeps).resolve(["acme"], purpose="add")
        assert excinfo.value.exit_code == ExitCode.NETWORK == 8
        assert "30s" in str(excinfo.value)
        assert len(registry.calls("resolve")) == 2, "one retry: the first try warms a cold instance"

    def test_a_bundle_timeout_is_exit_8(self, registry, sleeps):
        skill = refund(registry)
        registry.fault("bundle", "timeout", "timeout")
        with pytest.raises(NetworkError) as excinfo:
            make_client(registry, sleeps).fetch_bundle(
                skill.newest.version_id, install_name="acme-refund-policy", purpose="frozen",
                skill="acme/refund-policy")
        assert excinfo.value.exit_code == 8
        assert excinfo.value.skill == "acme/refund-policy"

    def test_one_timeout_is_retried(self, registry, sleeps):
        refund(registry)
        registry.fault("resolve", "timeout")
        r = make_client(registry, sleeps).resolve_one("acme/refund-policy", purpose="add")
        assert r.install_name == "acme-refund-policy"
        assert len(registry.calls("resolve")) == 2

    def test_a_timeout_on_a_real_socket_is_exit_8(self, registry):
        registry.stall_seconds = 1.0
        registry.fault("resolve", "timeout", "timeout")
        with registry.serve() as base_url:
            client = RegistryClient(api_key=OWNER_KEY, base_url=base_url, timeout=0.2)
            started = time.monotonic()
            with pytest.raises(NetworkError) as excinfo:
                client.resolve(["acme"], purpose="add")
            elapsed = time.monotonic() - started
            client.close()
        assert excinfo.value.exit_code == 8
        assert elapsed < 1.0, "the client gave up at its own budget, not the server's pace"

    def test_a_refused_connection_is_exit_8(self):
        base_url = f"http://127.0.0.1:{_closed_port()}"
        with pytest.raises(NetworkError) as excinfo:
            RegistryClient(api_key=None, base_url=base_url).resolve(["acme"], purpose="add")
        assert excinfo.value.exit_code == 8
        assert base_url in str(excinfo.value)

    def test_a_dropped_connection_is_exit_8(self, registry, sleeps):
        registry.fault("bundle", "connect")
        with pytest.raises(NetworkError):
            make_client(registry, sleeps).fetch_bundle("9b2e", install_name="acme", purpose="frozen")
        assert len(registry.requests) == 1

    def test_admission_aborts_are_retried(self, registry, sleeps):
        refund(registry)
        registry.fault("resolve", 503, 429, 502)
        r = make_client(registry, sleeps).resolve_one("acme/refund-policy", purpose="add")
        assert r.install_name == "acme-refund-policy"
        assert sleeps == [0.5, 1.5, 4.0]

    def test_retry_after_is_honoured(self, registry, sleeps):
        refund(registry)
        registry.fault("resolve", respond(429, {"detail": "slow down"}, {"Retry-After": "2"}))
        make_client(registry, sleeps).resolve_one("acme/refund-policy", purpose="add")
        assert sleeps == [2.0]

    def test_admission_that_never_clears_is_exit_8(self, registry, sleeps):
        registry.fault("resolve", 503, 503, 503, 503)
        with pytest.raises(NetworkError) as excinfo:
            make_client(registry, sleeps).resolve(["acme"], purpose="add")
        assert "503" in str(excinfo.value)
        assert len(registry.requests) == 4
        assert sleeps == [0.5, 1.5, 4.0]

    def test_a_retry_after_beyond_the_budget_is_not_waited_for(self, registry, sleeps):
        registry.fault("resolve", respond(429, {}, {"Retry-After": "120"}))
        with pytest.raises(NetworkError):
            make_client(registry, sleeps).resolve(["acme"], purpose="add")
        assert sleeps == [] and len(registry.requests) == 1

    def test_a_spent_quota_is_not_retried(self, registry, sleeps):
        registry.fault("resolve", respond(429, {"detail": {"resets_in_seconds": 3600}},
                                          {"X-Quota-Exceeded": "skill_fetches"}))
        with pytest.raises(NetworkError) as excinfo:
            make_client(registry, sleeps).resolve(["acme"], purpose="add")
        assert "skill_fetches" in str(excinfo.value)
        assert sleeps == [] and len(registry.requests) == 1

    def test_a_500_is_exit_8_and_not_retried(self, registry, sleeps):
        registry.fault("bundle", 500)
        with pytest.raises(NetworkError) as excinfo:
            make_client(registry, sleeps).fetch_bundle("9b2e", install_name="acme", purpose="frozen")
        assert "500" in str(excinfo.value)
        assert len(registry.requests) == 1

    @pytest.mark.parametrize("key", ["dai_sk_revoked", OWNER_KEY])
    def test_a_rejected_key_is_exit_7_not_8(self, registry, sleeps, key):
        if key == OWNER_KEY:
            registry.fault("resolve", respond(403, {"detail": "this key cannot install skills"}))
        with pytest.raises(AccessDeniedError) as excinfo:
            make_client(registry, sleeps, key=key).resolve(["acme"], purpose="add")
        assert excinfo.value.exit_code == 7
        assert len(registry.requests) == 1


# ── bytes are checked on receipt ─────────────────────────────────────────────────


class TestReceivedBytesMatchTheirHashes:
    def test_a_good_bundle_is_exactly_the_registrys_bytes(self, registry, sleeps):
        skill = refund(registry)
        client = make_client(registry, sleeps)
        bundle = client.fetch_resolved(client.resolve_one("acme/refund-policy", purpose="add"),
                                       purpose="add")
        expected = registry.render(skill, install_name="acme-refund-policy")

        assert [f.path for f in bundle.files] == [f.path for f in expected.files]
        for got, want in zip(bundle.files, expected.files):
            assert got.content == want.content.encode("utf-8")
            assert got.sha256 == hashlib.sha256(got.content).hexdigest() == want.sha256
            assert got.mode == int(want.mode, 8)
        assert bundle.executable == ("scripts/compute_refund.py",)
        assert [u.path for u in bundle.unavailable] == ["assets/logo.png"]
        assert bundle.skill_md == expected.skill_md.encode("utf-8")
        assert (bundle.render.format, bundle.render.hash, bundle.render.variant) == (
            1, expected.render_hash, "foreign")
        assert bundle.skill_id == skill.skill_id

    @pytest.mark.parametrize("index,path", [(0, "SKILL.md"), (1, "references/fees.md"),
                                            (2, "scripts/compute_refund.py")])
    def test_a_file_whose_bytes_differ_from_its_declared_hash_is_exit_6(self, registry, sleeps, index, path):
        refund(registry)
        client = make_client(registry, sleeps)
        r = client.resolve_one("acme/refund-policy", purpose="add")
        registry.tamper = lambda data: data["files"][index].__setitem__("content", "rm -rf ~\n")

        with pytest.raises(IntegrityError) as excinfo:
            client.fetch_resolved(r, purpose="add")

        err = excinfo.value
        assert type(err) is IntegrityError
        assert err.exit_code == ExitCode.INTEGRITY == 6
        assert err.path == path and err.skill == "acme/refund-policy"
        assert err.details["received"] == sha("rm -rf ~\n")
        assert err.details["declared"] == r.file_hashes[path]

    def test_a_declared_hash_in_upper_case_is_the_same_hash(self, registry, sleeps):
        skill = refund(registry)
        registry.tamper = lambda data: [f.__setitem__("sha256", f["sha256"].upper()) for f in data["files"]]
        bundle = make_client(registry, sleeps).fetch_bundle(
            skill.newest.version_id, install_name="acme-refund-policy", purpose="frozen")
        assert all(f.sha256 == f.sha256.lower() for f in bundle.files)

    def test_bytes_that_differ_from_the_lock_are_exit_6(self, registry, sleeps):
        skill = refund(registry)
        client = make_client(registry, sleeps)
        good = client.fetch_bundle(skill.newest.version_id, install_name="acme-refund-policy",
                                   purpose="add")
        locked = {**good.file_hashes, "references/fees.md": "0" * 64}

        with pytest.raises(IntegrityError) as excinfo:
            client.fetch_bundle(skill.newest.version_id, install_name="acme-refund-policy",
                                purpose="frozen", expect_files=locked,
                                expect_executable=good.executable, skill="acme/refund-policy")

        assert excinfo.value.exit_code == 6
        assert "the lock" in str(excinfo.value)
        assert excinfo.value.details["differ"] == ["references/fees.md"]

    @pytest.mark.parametrize("change,key", [
        (lambda files: files.pop("references/fees.md"), "extra"),
        (lambda files: files.__setitem__("references/new.md", "1" * 64), "missing"),
    ])
    def test_a_file_set_that_differs_from_the_lock_is_exit_6(self, registry, sleeps, change, key):
        skill = refund(registry)
        client = make_client(registry, sleeps)
        good = client.fetch_bundle(skill.newest.version_id, install_name="acme-refund-policy",
                                   purpose="add")
        locked = dict(good.file_hashes)
        change(locked)
        with pytest.raises(IntegrityError) as excinfo:
            client.fetch_bundle(skill.newest.version_id, install_name="acme-refund-policy",
                                purpose="frozen", expect_files=locked)
        assert excinfo.value.details[key]

    def test_an_exec_bit_that_differs_from_the_lock_is_exit_6(self, registry, sleeps):
        skill = refund(registry)
        client = make_client(registry, sleeps)
        good = client.fetch_bundle(skill.newest.version_id, install_name="acme-refund-policy",
                                   purpose="add")
        with pytest.raises(IntegrityError) as excinfo:
            client.fetch_bundle(skill.newest.version_id, install_name="acme-refund-policy",
                                purpose="frozen", expect_files=good.file_hashes,
                                expect_executable=["scripts/compute_refund.py", "references/fees.md"])
        assert excinfo.value.exit_code == 6

    def test_a_bundle_that_disagrees_with_its_resolve_is_exit_6(self, registry, sleeps):
        # Self-consistent bytes (the declared hash matches), but not what resolve declared.
        refund(registry)
        client = make_client(registry, sleeps)
        r = client.resolve_one("acme/refund-policy", purpose="add")

        def swap(data):
            data["files"][1].update(content="different\n", sha256=sha("different\n"))
        registry.tamper = swap

        with pytest.raises(IntegrityError) as excinfo:
            client.fetch_resolved(r, purpose="add")
        assert "resolve" in str(excinfo.value)

    def test_a_frozen_fetch_gets_the_locked_bytes_after_an_upstream_edit(self, registry, sleeps):
        skill = refund(registry)
        client = make_client(registry, sleeps)
        locked = client.resolve_one("acme/refund-policy", purpose="add")
        installed = client.fetch_resolved(locked, purpose="add")

        skill.description = "Apply the new refund rules."
        moved = client.resolve_one("acme/refund-policy", purpose="update")
        assert moved.version_id == locked.version_id and moved.render.hash != locked.render.hash

        frozen = client.fetch_bundle(
            locked.version_id, install_name=locked.install_name, render_hash=locked.render.hash,
            format=locked.render.format, variant=locked.render.variant,
            expect_files=installed.file_hashes, expect_executable=installed.executable,
            purpose="frozen")
        assert frozen.files == installed.files
        assert b"new refund rules" not in frozen.skill_md
        assert b"new refund rules" in client.fetch_resolved(moved, purpose="update").skill_md


class TestTheBundleContract:
    """A bundle the client could not write safely is refused whole (exit 6)."""

    def _fetch(self, registry, sleeps, **kwargs):
        skill = refund(registry)
        return make_client(registry, sleeps).fetch_bundle(
            skill.newest.version_id, install_name="acme-refund-policy", purpose="frozen", **kwargs)

    @pytest.mark.parametrize("path", [
        "../escape.md", "/etc/passwd", "scripts/../../escape.py", "notes/x.md", "x.md",
        "scripts\\x.py", "references/SKILL.md", "references/skill.md/x.md", "skill.md",
        "scripts/", "scripts//x.py", "./scripts/x.py", "references/a\x00.md", "",
    ])
    def test_an_unsafe_path_is_refused(self, registry, sleeps, path):
        registry.tamper = lambda data: add_file(data, path)
        with pytest.raises(BundleContractError) as excinfo:
            self._fetch(registry, sleeps)
        assert excinfo.value.exit_code == ExitCode.INTEGRITY

    @pytest.mark.parametrize("edit", [
        pytest.param(lambda data: data["files"].pop(0), id="no SKILL.md"),
        pytest.param(lambda data: add_file(data, "SKILL.md"), id="two SKILL.md"),
        pytest.param(lambda data: add_file(data, "References/fees.md"), id="case-insensitive twin"),
        pytest.param(lambda data: add_file(data, "references/fees.md"), id="exact twin"),
        pytest.param(lambda data: add_file(data, "references/fees.md/x.md"), id="file is a folder"),
        pytest.param(lambda data: add_file(data, "scripts/x.sh", mode="0777"), id="mode 0777"),
        pytest.param(lambda data: add_file(data, "scripts/x.sh", mode="rwx"), id="mode rwx"),
        pytest.param(lambda data: data["files"][1].__setitem__("encoding", "base64"), id="base64"),
        pytest.param(lambda data: data["files"][1].__setitem__("content", None), id="no content"),
        pytest.param(lambda data: data["files"][1].__setitem__("content", "\ud800"), id="lone surrogate"),
        pytest.param(lambda data: data["files"][1].__setitem__("sha256", "abc"), id="short hash"),
        pytest.param(lambda data: data.__setitem__("files", "SKILL.md"), id="files not a list"),
        pytest.param(lambda data: data["render"].__setitem__("format", 2), id="format 2"),
        pytest.param(lambda data: data["render"].__setitem__("variant", "mine"), id="variant"),
        pytest.param(lambda data: data.__setitem__("install_name", "other-name"), id="install name"),
        pytest.param(lambda data: data.__setitem__("version_id", "another"), id="version id"),
        pytest.param(lambda data: data.__setitem__("unavailable", [{"reason": "x"}]), id="unavailable"),
        pytest.param(lambda data: data.__setitem__("policy_notes", [3]), id="policy notes"),
    ])
    def test_a_bundle_that_breaks_the_contract_is_refused(self, registry, sleeps, edit):
        registry.tamper = edit
        with pytest.raises(BundleContractError) as excinfo:
            self._fetch(registry, sleeps)
        assert excinfo.value.exit_code == 6

    @pytest.mark.parametrize("field", ["render_hash", "variant"])
    def test_a_bundle_must_be_the_rendering_asked_for(self, registry, sleeps, field):
        skill = refund(registry)
        real_hash, _ = registry.snapshot(skill)
        asked = {"render_hash": real_hash, "variant": "foreign"}[field]
        served = {"render_hash": ("hash", "e" * 64), "variant": ("variant", "authored")}[field]
        registry.tamper = lambda data: data["render"].__setitem__(*served)
        with pytest.raises(BundleContractError):
            make_client(registry, sleeps).fetch_bundle(
                skill.newest.version_id, install_name="acme-refund-policy", purpose="frozen",
                **{field: asked})

    def test_an_unsupported_format_names_the_upgrade(self, registry, sleeps):
        registry.tamper = lambda data: data["render"].__setitem__("format", 2)
        with pytest.raises(BundleContractError) as excinfo:
            self._fetch(registry, sleeps)
        assert "pip install -U decimalai" in str(excinfo.value)

    @pytest.mark.parametrize("mode,expected", [("0644", 0o644), ("644", 0o644), ("0755", 0o755),
                                               (0o755, 0o755)])
    def test_the_two_modes_are_read_in_their_spellings(self, registry, sleeps, mode, expected):
        registry.tamper = lambda data: data["files"][-1].__setitem__("mode", mode)
        bundle = self._fetch(registry, sleeps)
        assert bundle.files[-1].mode == expected

    @pytest.mark.parametrize("answer", [respond(200, text="<html>captive portal</html>"),
                                        respond(200, ["SKILL.md"])])
    def test_an_answer_that_is_not_a_json_object_is_refused(self, registry, sleeps, answer):
        registry.fault("bundle", answer)
        with pytest.raises(BundleContractError):
            self._fetch(registry, sleeps)

    def test_a_nested_skill_md_the_registry_lists_as_unavailable_is_fine(self, registry, sleeps):
        skill = registry.add_skill("acme/nested", files=[
            {"path": "references/SKILL.md", "content": "# another skill\n"}])
        bundle = make_client(registry, sleeps).fetch_bundle(
            skill.newest.version_id, install_name="acme-nested", purpose="add")
        assert [f.path for f in bundle.files] == ["SKILL.md"]
        assert [u.path for u in bundle.unavailable] == ["references/SKILL.md"]


class TestBundleRefusals:
    @pytest.mark.parametrize("body,reason", [
        ({"detail": {"code": "revoked", "reason": "blocked", "message": "version 1 is blocked"},
          "code": "HTTP_410"}, "blocked"),
        ({"detail": {"reason": "withheld"}, "code": "HTTP_410"}, "withheld"),
        ({"reason": "unsafe"}, "unsafe"),
        ({"detail": "gone", "code": "HTTP_410"}, "revoked"),
    ])
    def test_a_410_in_any_shape_is_a_revoked_version(self, registry, sleeps, body, reason):
        registry.fault("bundle", respond(410, body))
        with pytest.raises(LockedVersionRevokedError) as excinfo:
            make_client(registry, sleeps).fetch_bundle(
                "9b2e", install_name="acme", purpose="frozen", skill="acme/refund-policy", version=3)
        err = excinfo.value
        assert err.exit_code == 7
        assert err.message == f"version 3 is no longer available ({reason})"
        assert err.details["reason"] == reason

    def test_a_version_blocked_after_it_was_locked(self, registry, sleeps):
        skill = refund(registry)
        client = make_client(registry, sleeps)
        locked = client.resolve_one("acme/refund-policy", purpose="add")
        skill.band = "blocked"
        with pytest.raises(LockedVersionRevokedError) as excinfo:
            client.fetch_bundle(locked.version_id, install_name=locked.install_name,
                                render_hash=locked.render.hash, purpose="frozen",
                                skill=locked.registry_name, version=locked.version)
        text = str(excinfo.value)
        assert text.startswith("acme/refund-policy: version 1 is no longer available (blocked)")
        assert "decimalai skills update acme/refund-policy" in text
        assert "decimalai skills remove acme/refund-policy" in text

    def test_a_render_that_is_gone_is_render_gone(self, registry, sleeps):
        skill = refund(registry)
        client = make_client(registry, sleeps)
        locked = client.resolve_one("acme/refund-policy", purpose="add")
        registry.forget_renders(skill.newest)
        with pytest.raises(RenderGoneError) as excinfo:
            client.fetch_bundle(locked.version_id, install_name=locked.install_name,
                                render_hash=locked.render.hash, purpose="frozen",
                                skill=locked.registry_name)
        assert excinfo.value.exit_code == 7
        assert "decimalai skills update acme/refund-policy" in str(excinfo.value)

    def test_the_authored_variant_for_someone_else_is_access_denied(self, registry, sleeps):
        skill = refund(registry)
        with pytest.raises(AccessDeniedError) as excinfo:
            make_client(registry, sleeps).fetch_bundle(
                skill.newest.version_id, install_name="acme-refund-policy", variant="authored",
                purpose="frozen")
        assert excinfo.value.exit_code == 7

    def test_any_valid_install_name_is_served(self, registry, sleeps):
        # A locked folder keeps its name after an upstream rename; the name only sets the
        # `name:` line of the client's own folder.
        skill = refund(registry)
        bundle = make_client(registry, sleeps).fetch_bundle(
            skill.newest.version_id, install_name="refunds-kept-name", purpose="frozen")
        assert bundle.install_name == "refunds-kept-name"
        fm, _ = skillfmt.parse_skill_md_text(bundle.skill_md.decode("utf-8"))
        assert fm["name"] == "refunds-kept-name"

    def test_the_registry_refuses_a_name_no_loader_accepts(self, registry):
        skill = refund(registry)
        with httpx.Client(base_url=BASE_URL, transport=registry.transport()) as http:
            resp = http.get(AUTH_BUNDLES + skill.newest.version_id,
                            params={"install_name": "Not A Name"},
                            headers={"Authorization": f"Bearer {OWNER_KEY}"})
        assert resp.status_code == 422
        assert resp.json()["detail"]["code"] == "invalid_install_name"

    def test_an_install_name_refusal_is_exit_7(self, registry, sleeps):
        skill = refund(registry)
        registry.fault("bundle", respond(422, {"detail": {
            "code": "invalid_install_name", "message": "not an install name"}}))
        with pytest.raises(UnavailableError) as excinfo:
            make_client(registry, sleeps).fetch_bundle(
                skill.newest.version_id, install_name="acme-refund-policy", purpose="frozen")
        assert excinfo.value.exit_code == 7
        assert excinfo.value.details["registry_code"] == "invalid_install_name"

    def test_the_registry_bundle_route_refuses_a_blocked_skill(self, registry, sleeps):
        refund(registry, band="blocked")
        with pytest.raises(LockedVersionRevokedError):
            make_client(registry, sleeps).fetch_registry_bundle("acme-refund-policy", purpose="pull")


# ── format negotiation ───────────────────────────────────────────────────────────


class TestFormatNegotiation:
    """The client says which formats it reads; it never receives another one."""

    def test_the_formats_this_client_reads_are_one_constant(self):
        assert FORMATS == (1,)
        assert set(FORMATS) == set(skillfmt.SUPPORTED_FORMATS)

    def test_resolve_and_bundle_send_the_formats(self, registry, sleeps):
        refund(registry)
        client = make_client(registry, sleeps)
        r = client.resolve_one("acme/refund-policy", purpose="add")
        client.fetch_resolved(r, purpose="add")
        assert registry.calls("resolve")[0].json["formats"] == [1]
        assert registry.calls("bundle")[0].query["formats"] == "1"

    def test_a_per_item_format_unsupported_names_the_upgrade(self, registry, sleeps):
        refund(registry)
        registry.fault("resolve", respond(200, {"items": [{"ref": "acme/refund-policy", "ok": False,
            "error": {"code": "format_unsupported",
                      "message": "this version renders in format 2 only"}}]}))
        (err,) = make_client(registry, sleeps).resolve(["acme/refund-policy"], purpose="add")
        assert isinstance(err, RenderInvalidError)
        assert err.exit_code == ExitCode.UNAVAILABLE
        assert err.details["registry_code"] == "format_unsupported"
        assert err.details["formats"] == [1]
        assert "pip install -U decimalai" in err.fix
        assert "format 2 only" in err.message

    def test_a_bundle_refused_for_its_format_names_the_upgrade(self, registry, sleeps):
        skill = refund(registry)
        registry.fault("bundle", respond(422, {"detail": {
            "code": "format_unsupported", "message": "no format this client reads"}}))
        with pytest.raises(RenderInvalidError) as excinfo:
            make_client(registry, sleeps).fetch_bundle(
                skill.newest.version_id, install_name="acme-refund-policy", purpose="frozen",
                skill="acme/refund-policy")
        assert excinfo.value.exit_code == ExitCode.UNAVAILABLE
        assert excinfo.value.details["registry_code"] == "format_unsupported"
        assert "pip install -U decimalai" in excinfo.value.fix

    def test_no_shared_format_refuses_the_whole_resolve_as_exit_7(
            self, registry, sleeps, monkeypatch):
        refund(registry)
        monkeypatch.setattr("decimalai.skillpkg.client.FORMATS", (7,))
        with pytest.raises(RenderInvalidError) as excinfo:
            make_client(registry, sleeps).resolve(["acme/refund-policy", "other"],
                                                  purpose="add")
        err = excinfo.value
        assert err.exit_code == ExitCode.UNAVAILABLE == 7
        assert err.details["registry_code"] == "format_unsupported"
        assert err.details["supported"] == [1]
        assert "pip install -U decimalai" in err.fix
        assert registry.calls("resolve")[0].json["formats"] == [7]

    def test_the_fake_answers_one_request_level_refusal_and_never_echoes(self, registry):
        refund(registry)
        with httpx.Client(base_url=BASE_URL, transport=registry.transport()) as http:
            resp = http.post(ANON_RESOLVE, json={"items": [{"ref": "acme/refund-policy"}],
                                                 "formats": [7, 9]})
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert detail["code"] == "format_unsupported" and detail["supported"] == [1]
        assert "7" not in json.dumps(resp.json()) and "9" not in detail["message"]

    @pytest.mark.parametrize("route", ["resolve", "bundle"])
    def test_more_than_8_formats_is_refused(self, registry, route):
        skill = refund(registry)
        with httpx.Client(base_url=BASE_URL, transport=registry.transport()) as http:
            if route == "resolve":
                resp = http.post(ANON_RESOLVE, json={"items": [{"ref": "acme/refund-policy"}],
                                                     "formats": [1] * 9})
            else:
                resp = http.get(ANON_BUNDLES + skill.newest.version_id,
                                params={"install_name": "acme-refund-policy",
                                        "formats": ["1"] * 9})
        assert resp.status_code == 422
        assert resp.json()["detail"]["code"] == "too_many_formats"

    @pytest.mark.parametrize("value", ["1x", "\u0661", "-1", "", " 1"])
    def test_bundle_formats_must_be_ascii_digits(self, registry, value):
        skill = refund(registry)
        with httpx.Client(base_url=BASE_URL, transport=registry.transport()) as http:
            resp = http.get(ANON_BUNDLES + skill.newest.version_id,
                            params={"install_name": "acme-refund-policy", "formats": [value]})
        assert resp.status_code == 422
        assert resp.json()["detail"]["code"] == "invalid_formats"

    @pytest.mark.parametrize("render_hash", ["not-a-hash", "A" * 64, "0" * 63])
    def test_a_malformed_render_hash_is_render_gone(self, registry, sleeps, render_hash):
        skill = refund(registry)
        with pytest.raises(RenderGoneError) as excinfo:
            make_client(registry, sleeps).fetch_bundle(
                skill.newest.version_id, install_name="acme-refund-policy",
                render_hash=render_hash, purpose="frozen")
        assert excinfo.value.exit_code == 7
        assert registry.calls("bundle")[0].query["render_hash"] == render_hash

    def test_the_fake_bundle_route_refuses_formats_without_one_it_renders(self, registry):
        skill = refund(registry)
        with httpx.Client(base_url=BASE_URL, transport=registry.transport()) as http:
            resp = http.get(AUTH_BUNDLES + skill.newest.version_id,
                            params={"install_name": "acme-refund-policy", "formats": [7, 8]},
                            headers={"Authorization": f"Bearer {OWNER_KEY}"})
            ok = http.get(AUTH_BUNDLES + skill.newest.version_id,
                          params={"install_name": "acme-refund-policy", "formats": [7, 1]},
                          headers={"Authorization": f"Bearer {OWNER_KEY}"})
        assert resp.status_code == 422
        assert resp.json()["detail"]["code"] == "format_unsupported"
        assert ok.status_code == 200 and ok.json()["render"]["format"] == 1


class TestBadRef:
    """A reference the registry cannot even read is the caller's mistake: exit 2."""

    @pytest.mark.parametrize("bad,why", [
        ("bad\x00ref", "control character at position 4"),
        ("tab\tref", "control character at position 4"),
        ("lone\udc80surrogate", "unpaired surrogate at position 5"),
        ("x" * 513, "513 characters"),
    ])
    def test_a_bad_ref_is_a_usage_error_naming_its_position(self, registry, sleeps, bad, why):
        refund(registry)
        out = make_client(registry, sleeps).resolve(["acme/refund-policy", bad],
                                                    purpose="add")
        assert isinstance(out[0], Resolved)
        err = out[1]
        assert isinstance(err, UsageError)
        assert err.exit_code == ExitCode.USAGE == 2
        assert err.details["registry_code"] == "bad_ref"
        assert "reference #2" in err.fix and why in err.fix
        assert str(err).isprintable(), "the message never prints the raw characters"
        assert registry.calls("resolve")[0].json["items"][1]["ref"] == bad

    def test_a_lone_surrogate_reaches_the_registry_instead_of_crashing(self, registry, sleeps):
        (err,) = make_client(registry, sleeps).resolve(["a\udc80b"], purpose="add")
        assert isinstance(err, UsageError)
        (call,) = registry.calls("resolve")
        assert call.json["items"][0]["ref"] == "a\udc80b"

    def test_resolve_one_raises_it(self, registry, sleeps):
        with pytest.raises(UsageError) as excinfo:
            make_client(registry, sleeps).resolve_one("bad\x01", purpose="add")
        assert "reference #1" in excinfo.value.fix

    def test_the_position_counts_across_batches(self, registry, sleeps):
        refs = [f"team/skill-{i}" for i in range(22)] + ["bad\x00"]
        for ref in refs[:-1]:
            registry.add_skill(ref)
        out = make_client(registry, sleeps, key=None).resolve(refs, purpose="install")
        assert "reference #23" in out[-1].fix


class TestOrgId:
    def test_the_resolve_answer_names_the_callers_organization(self, registry, sleeps):
        refund(registry)
        r = make_client(registry, sleeps).resolve_one("acme/refund-policy", purpose="add")
        assert r.org_id == OWNER_ORG

    def test_anonymous_has_no_organization(self, registry, sleeps):
        refund(registry)
        r = make_client(registry, sleeps, key=None).resolve_one("acme/refund-policy",
                                                                purpose="add")
        assert r.org_id is None

    def test_an_answer_without_one_is_none(self, registry, sleeps):
        refund(registry)
        registry.tamper_resolve = lambda data: data["items"][0].pop("org_id")
        r = make_client(registry, sleeps).resolve_one("acme/refund-policy", purpose="add")
        assert r.org_id is None


# ── construction ─────────────────────────────────────────────────────────────────


class TestConstruction:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for name in ("DECIMAL_API_KEY", "DECIMALAI_API_KEY", "DECIMAL_BASE_URL", "DECIMALAI_BASE_URL"):
            monkeypatch.delenv(name, raising=False)

    def test_the_key_and_the_url_come_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("DECIMAL_API_KEY", OWNER_KEY)
        monkeypatch.setenv("DECIMAL_BASE_URL", "https://example.test/")
        client = RegistryClient.from_env()
        assert not client.anonymous
        assert client.base_url == "https://example.test"

    def test_the_alias_key_variable_is_read(self, monkeypatch):
        monkeypatch.setenv("DECIMALAI_API_KEY", OTHER_KEY)
        assert not RegistryClient.from_env().anonymous

    def test_no_key_is_anonymous_against_the_default_registry(self):
        client = RegistryClient.from_env()
        assert client.anonymous
        assert client.base_url == "https://api.decimal.ai"

    def test_anonymous_ignores_a_key_in_the_environment(self, monkeypatch, registry):
        monkeypatch.setenv("DECIMAL_API_KEY", OWNER_KEY)
        refund(registry)
        client = RegistryClient.from_env(anonymous=True, base_url=BASE_URL,
                                         transport=registry.transport())
        assert client.anonymous
        client.resolve(["acme/refund-policy"], purpose="pull")
        assert "authorization" not in registry.requests[0].headers

    def test_explicit_arguments_win(self, monkeypatch, registry):
        monkeypatch.setenv("DECIMAL_API_KEY", OTHER_KEY)
        monkeypatch.setenv("DECIMAL_BASE_URL", "https://elsewhere.test")
        refund(registry)
        client = RegistryClient.from_env(api_key=OWNER_KEY, base_url=BASE_URL,
                                         transport=registry.transport())
        client.resolve(["acme/refund-policy"], purpose="add")
        assert registry.requests[0].headers["authorization"] == f"Bearer {OWNER_KEY}"

    @pytest.mark.parametrize("key", ["dai_sk_a\nb", "dai_sk_\x00"])
    def test_a_key_with_control_characters_is_a_usage_error(self, key):
        with pytest.raises(UsageError):
            RegistryClient(api_key=key)

    @pytest.mark.parametrize("timeout", [0, -1, "30"])
    def test_a_timeout_must_be_positive_seconds(self, timeout):
        with pytest.raises(UsageError):
            RegistryClient(api_key=None, timeout=timeout)

    def test_the_client_is_a_context_manager(self, registry):
        with RegistryClient(api_key=None, base_url=BASE_URL, transport=registry.transport()) as c:
            assert isinstance(c, RegistryClient)

    def test_the_key_never_appears_in_the_repr(self):
        assert OWNER_KEY not in repr(RegistryClient(api_key=OWNER_KEY))
