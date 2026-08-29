"""The skill VERSION reaches the trace, not just the skill name.

The platform returns ``content_hash`` alongside every body precisely so that no
client has to recompute it, ``TraceSkillActivation.skill_hash`` exists to hold
it, and ``log_skill_activation`` has always accepted ``hash=``. None of that was
connected: the Router captured the hash into ``_loaded_hashes`` and every rail
between there and the trace was name-only, so EVERY activation the SDK reported
arrived with ``skill_hash: null``. A measured lift could be attributed to a
skill but never to the version of it the model actually read — which is the join
the whole versioning product rests on.

WHAT THIS FILE PINS, and why each half is separate:

  * the Router captures the hash of the body IT served, race-free (the
    thread-local handoff), and hands it back through ``consume_loaded_hashes``
    with the same scoped/unscoped shape ``consume_loaded_names`` has;
  * each adapter stamps it onto the trace as an ``active_skills`` entry.

THE INVARIANT THAT MAKES THE ADAPTER HALF SAFE, asserted directly below and
worth stating because it is what bounds the blast radius: every name written to
``active_skills`` is already in ``skills_loaded_by_agent``. The backend unions
those two fields into one activation set and dedupes by name with the
``active_skills`` entry winning (``trace_service._record_skill_activations``),
so the change swaps a bare string for a dict of the same name. It cannot add,
remove or double-count an activation. The single observable difference is that
``skill_hash`` stops being null.

Network is mocked throughout by patching ``get_skill_body_record`` — the method
that actually talks to the backend — rather than ``get_skill_body``. That is
deliberate: ``get_skill_body`` is the seam the hash travels through, so stubbing
it would skip the code under test entirely.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from decimalai import skill_router as sr
from decimalai.skill_router import SkillRouter

BODY = "The restocking fee is 23.5%."
HASH = "sha256:" + "a" * 64
OTHER_HASH = "sha256:" + "b" * 64


@pytest.fixture(autouse=True)
def _fresh_body_budget():
    """The body budget is a module-level ContextVar shared across this thread."""
    sr._body_budget_ctx.set(None)
    yield
    sr._body_budget_ctx.set(None)


def _router(**kw) -> SkillRouter:
    return SkillRouter(api_key="dai_sk_test", base_url="http://localhost:8000", **kw)


def _record(body: str = BODY, content_hash: Optional[str] = HASH) -> Dict[str, Any]:
    """The shape ``GET /api/v1/skills/{name}/body`` returns."""
    rec: Dict[str, Any] = {"body": body, "version": 3}
    if content_hash is not None:
        rec["content_hash"] = content_hash
    return rec


# ── 1. the Router captures and hands back the hash ───────────────


class TestRouterCapturesTheHash:
    def test_a_load_carries_the_hash_the_platform_returned(self):
        router = _router()
        with patch.object(router, "get_skill_body_record", return_value=_record()):
            out = router.load_skill("restocking-policy")

        assert out.startswith("## Skill: restocking-policy")
        assert router.consume_loaded_names() == ["restocking-policy"]
        assert router.consume_loaded_hashes() == {"restocking-policy": HASH}

    def test_the_hash_rail_drains_like_the_name_rail(self):
        """Read-and-clear, so one load cannot be reported by two traces."""
        router = _router()
        with patch.object(router, "get_skill_body_record", return_value=_record()):
            router.load_skill("restocking-policy")

        assert router.consume_loaded_hashes() == {"restocking-policy": HASH}
        assert router.consume_loaded_hashes() == {}

    def test_a_backend_that_sends_no_hash_reports_none_not_a_guess(self):
        """The SDK must never compute a hash itself. An older backend, or a
        body served from a path that carries no digest, means version-unknown —
        which is a gap, and gaps are fine. Inventing sha256(body) here would
        manufacture a digest the platform never minted."""
        router = _router()
        with patch.object(
            router, "get_skill_body_record", return_value=_record(content_hash=None),
        ):
            router.load_skill("restocking-policy")

        assert router.consume_loaded_names() == ["restocking-policy"]
        assert router.consume_loaded_hashes() == {"restocking-policy": None}

    def test_a_stubbed_body_fetch_reports_no_hash(self):
        """`get_skill_body` is public and widely stubbed. A stub that returns
        only text leaves no hash, and the load must still be RECORDED — the
        name rail is what tells the platform a body reached the model, and it
        cannot depend on a digest being available."""
        router = _router()
        with patch.object(router, "get_skill_body", return_value=BODY):
            router.load_skill("restocking-policy")

        assert router.consume_loaded_names() == ["restocking-policy"]
        assert router.consume_loaded_hashes() == {"restocking-policy": None}

    def test_a_not_found_leaves_no_hash_behind(self):
        """A miss must not leave the PREVIOUS body's hash on the thread for the
        next load to pick up — that would attribute one skill's version to
        another skill entirely."""
        router = _router()
        with patch.object(router, "get_skill_body_record", return_value=_record()):
            router.load_skill("restocking-policy")
        router.consume_loaded_hashes()

        with patch.object(router, "get_skill_body_record", return_value=None):
            out = router.load_skill("ghost")
        assert "no skill named" in out
        assert router.consume_loaded_hashes() == {}

    def test_two_versions_of_one_name_claim_neither(self):
        """A rail can name one version. Two different digests for one name means
        the rail cannot say which the model read, and NULL ("no known hash") is
        the honest answer — a wrong hash resolves to a real, wrong version row,
        which is strictly worse than the gap it replaces.

        Two RUNS, because that is the only way the conflict actually arises: a
        repeat load inside one run is deduped by the turn budget and never
        re-fetches, while two concurrent runs both write the shared unscoped
        window. This is the case the unscoped rail is already documented as
        being lossy about for NAMES; the rule keeps it from also inventing a
        version for a name it got right.
        """
        router = _router(max_loaded_bodies=5)
        with patch.object(
            router, "get_skill_body_record",
            side_effect=[_record(content_hash=HASH), _record(content_hash=OTHER_HASH)],
        ):
            router.load_skill("restocking-policy", scope="run-a")
            router.load_skill("restocking-policy", scope="run-b")

        # Each run's OWN rail still knows exactly what it read...
        assert router.consume_loaded_hashes(scope="run-a") == {
            "restocking-policy": HASH
        }
        assert router.consume_loaded_hashes(scope="run-b") == {
            "restocking-policy": OTHER_HASH
        }
        # ...but the shared window, which cannot tell them apart, claims neither.
        assert router.consume_loaded_hashes() == {"restocking-policy": None}

    def test_a_repeat_load_of_the_same_version_keeps_its_hash(self):
        """The degrade above must fire on a CONFLICT, not on any repeat."""
        router = _router(max_loaded_bodies=5)
        with patch.object(router, "get_skill_body_record", return_value=_record()):
            router.load_skill("restocking-policy")
            router.load_skill("restocking-policy")

        assert router.consume_loaded_hashes() == {"restocking-policy": HASH}


class TestScopedHashRail:
    def test_two_runs_do_not_see_each_others_hashes(self):
        router = _router()
        with patch.object(
            router, "get_skill_body_record",
            side_effect=lambda n, *a, **kw: _record(
                content_hash=HASH if n == "alpha" else OTHER_HASH
            ),
        ):
            router.load_skill("alpha", scope="run-a")
            router.load_skill("beta", scope="run-b")

        assert router.consume_loaded_hashes(scope="run-a") == {"alpha": HASH}
        assert router.consume_loaded_hashes(scope="run-b") == {"beta": OTHER_HASH}

    def test_a_scoped_drain_is_consumed_once(self):
        router = _router()
        with patch.object(router, "get_skill_body_record", return_value=_record()):
            router.load_skill("alpha", scope="run-a")

        assert router.consume_loaded_hashes(scope="run-a") == {"alpha": HASH}
        assert router.consume_loaded_hashes(scope="run-a") == {}

    def test_the_names_and_hashes_of_one_scope_agree(self):
        """The two rails are separate stores written in one call. If they ever
        fall out of step, a trace reports a load whose hash belongs to a
        different name."""
        router = _router(max_loaded_bodies=5)
        with patch.object(router, "get_skill_body_record", return_value=_record()):
            router.load_skill("alpha", scope="run-a")
            router.load_skill("beta", scope="run-a")

        names = router.consume_loaded_names(scope="run-a")
        hashes = router.consume_loaded_hashes(scope="run-a")
        assert sorted(names) == sorted(hashes) == ["alpha", "beta"]


class TestTheRaceTheThreadLocalCloses:
    def test_concurrent_loads_of_one_name_at_two_versions_do_not_swap(self):
        """THE reason the hash rides a thread-local instead of a re-read.

        `_loaded_hashes` is a persistent last-seen map keyed by skill name. Two
        threads loading the same skill at different versions both write it, so a
        `load_skill` that re-read that map after fetching would take whichever
        version the OTHER thread stored last — reporting a version this run
        never read. The thread-local is written and read on one thread between
        two synchronous calls, so it cannot be crossed.

        The window is genuinely narrow (write, then read, a few statements
        later), so it is FORCED rather than hoped for: the barrier below sits
        between the real `get_skill_body` — which performs the shared write —
        and `load_skill` reading the hash back, so both writes are guaranteed to
        have landed before either read happens. Without the barrier this test
        passes against the racy implementation and proves nothing; it was
        written that way first and had to be fixed.
        """
        router = _router(max_loaded_bodies=5)
        real_get_body = router.get_skill_body
        gate = threading.Barrier(2)
        seen: Dict[str, Optional[str]] = {}
        errors: List[BaseException] = []

        def _both_have_written(*a, **kw):
            out = real_get_body(*a, **kw)
            gate.wait(timeout=10)   # every thread's shared write has now landed
            return out

        def _fetch(name, *a, **kw):
            digest = HASH if threading.current_thread().name == "t1" else OTHER_HASH
            return _record(content_hash=digest)

        def _run(scope: str):
            try:
                router.load_skill("restocking-policy", scope=scope)
                seen[scope] = router.consume_loaded_hashes(scope=scope).get(
                    "restocking-policy"
                )
            except BaseException as exc:      # noqa: BLE001 — reported below
                errors.append(exc)

        # Patched ONCE, on this thread, around both workers. Patching inside
        # each thread is what an earlier version did and it made the test flaky:
        # `patch.object` saves the attribute at enter and restores it at exit,
        # so the second thread saved the FIRST thread's patch as the original
        # and the exits unwound each other mid-run.
        with patch.object(router, "get_skill_body_record", side_effect=_fetch), \
             patch.object(router, "get_skill_body", _both_have_written):
            t1 = threading.Thread(target=_run, args=("run-1",), name="t1")
            t2 = threading.Thread(target=_run, args=("run-2",), name="t2")
            t1.start(); t2.start(); t1.join(20); t2.join(20)

        assert not errors, f"a worker raised: {errors!r}"
        assert seen == {"run-1": HASH, "run-2": OTHER_HASH}, (
            "a run reported a version another thread had loaded — the hash was "
            f"read from shared state, not from this call: {seen}"
        )


# ── 2. the adapters stamp it onto the trace ──────────────────────


def _activation_map(trace: Any) -> Dict[str, Optional[str]]:
    """``name -> hash`` as the backend will read ``active_skills``."""
    out: Dict[str, Optional[str]] = {}
    for entry in trace.active_skills or []:
        if isinstance(entry, dict) and entry.get("name"):
            out[entry["name"]] = entry.get("hash")
        elif isinstance(entry, str):
            out[entry] = None
    return out


def _assert_activation_set_unchanged(trace: Any) -> None:
    """THE bounding invariant — see this module's docstring.

    Every name the hash stamp adds must already be reported as loaded, so the
    set of skills the backend records as activated is byte-for-byte what it was
    before the stamp existed.
    """
    added = set(_activation_map(trace))
    loaded = set(trace.skills_loaded_by_agent or [])
    assert added <= loaded, (
        "active_skills names a skill that is not in skills_loaded_by_agent — "
        "the hash stamp has started ADDING activations instead of annotating "
        f"them: {sorted(added - loaded)}"
    )


class TestOpenAIAgentsStampsTheVersion:
    @pytest.fixture(autouse=True)
    def _reset_sdk(self, monkeypatch):
        import decimalai._config as cfg
        import decimalai.openai_agents as oa
        from decimalai._config import DecimalConfig

        prev_config, prev_client = cfg._config, cfg._client
        cfg._config = DecimalConfig(
            api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True,
        )
        cfg._client = MagicMock()
        cfg._client.register_manifest.return_value = {
            "manifest_id": "test-manifest-id", "status": "active",
        }
        oa._manifest_id = None
        monkeypatch.setattr(oa, "_skill_router_singleton", None)
        yield
        cfg._config, cfg._client = prev_config, prev_client

    def _run_one_trace(self):
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="test-agent")
        trace_id = f"trace_{uuid4().hex[:16]}"
        mock_trace = MagicMock(trace_id=trace_id, name="wf")
        processor.on_trace_start(mock_trace)
        processor.on_trace_end(mock_trace)

        import decimalai._config as cfg
        from decimalai._config import _sender
        _sender.flush()
        return cfg._client.ingest_trace.call_args[0][0]

    def test_a_loaded_body_reports_its_version(self):
        import decimalai.openai_agents as oa

        router = _router()
        oa._skill_router_singleton = router
        with patch.object(router, "get_skill_body_record", return_value=_record()):
            router.load_skill("restocking-policy")

        trace = self._run_one_trace()
        assert trace.skills_loaded_by_agent == ["restocking-policy"]
        assert _activation_map(trace) == {"restocking-policy": HASH}
        _assert_activation_set_unchanged(trace)

    def test_a_load_with_no_known_version_adds_no_entry(self):
        """A dict carrying only a name says nothing the plain string in
        `skills_loaded_by_agent` does not already say."""
        import decimalai.openai_agents as oa

        router = _router()
        oa._skill_router_singleton = router
        with patch.object(router, "get_skill_body", return_value=BODY):
            router.load_skill("restocking-policy")

        trace = self._run_one_trace()
        assert trace.skills_loaded_by_agent == ["restocking-policy"]
        assert trace.active_skills == []
        _assert_activation_set_unchanged(trace)

    def test_a_trace_with_no_loads_stamps_nothing(self):
        trace = self._run_one_trace()
        assert trace.active_skills == []
        assert trace.skills_loaded_by_agent == []


class TestLangchainStampsTheVersion:
    def _trace(self, monkeypatch, router):
        import decimalai.langchain as lc
        from decimalai.langchain import CallbackHandler

        monkeypatch.setattr(lc, "_skill_router_singleton", router)
        handler = CallbackHandler(agent_name="lc-agent")
        handler._trace_started_at = datetime.now(timezone.utc)
        with patch("decimalai._config._config", None):
            return handler.build_trace()

    def test_a_loaded_body_reports_its_version(self, monkeypatch):
        router = _router()
        with patch.object(router, "get_skill_body_record", return_value=_record()):
            router.load_skill("restocking-policy")

        trace = self._trace(monkeypatch, router)
        assert trace.skills_loaded_by_agent == ["restocking-policy"]
        assert _activation_map(trace) == {"restocking-policy": HASH}
        _assert_activation_set_unchanged(trace)

    def test_a_load_with_no_known_version_adds_no_entry(self, monkeypatch):
        router = _router()
        with patch.object(router, "get_skill_body", return_value=BODY):
            router.load_skill("restocking-policy")

        trace = self._trace(monkeypatch, router)
        assert trace.skills_loaded_by_agent == ["restocking-policy"]
        assert trace.active_skills == []
        _assert_activation_set_unchanged(trace)


class TestOtelRailCarriesTheVersion:
    """The rail `pydantic_ai` reports through (`otel.record_skill_rail`).

    `current_run_key` is stubbed rather than standing up a real TracerProvider:
    what is under test is the rail's STORAGE of the hashes, and an OTel pipeline
    would only decide which integer key they are filed under.
    """

    @pytest.fixture
    def rail(self, monkeypatch):
        from decimalai import otel as otel_mod

        otel_mod._reset_skill_rails()
        monkeypatch.setattr(otel_mod, "current_run_key", lambda: 4242)
        yield otel_mod
        otel_mod._reset_skill_rails()

    def test_the_rail_stores_the_hashes_it_is_given(self, rail):
        assert rail.record_skill_rail(
            loaded=["restocking-policy"],
            loaded_hashes={"restocking-policy": HASH},
        ) is True
        stored = rail._skill_rails[4242]
        assert stored["loaded"] == ["restocking-policy"]
        assert stored["loaded_hashes"] == {"restocking-policy": HASH}

    def test_a_caller_that_passes_no_hashes_is_unchanged(self, rail):
        """Every pre-existing caller. The load is still recorded; it simply has
        no known version, which is what it had before this parameter existed."""
        rail.record_skill_rail(loaded=["restocking-policy"])
        stored = rail._skill_rails[4242]
        assert stored["loaded"] == ["restocking-policy"]
        assert stored["loaded_hashes"] == {"restocking-policy": None}

    def test_two_versions_of_one_name_claim_neither(self, rail):
        rail.record_skill_rail(loaded=["a"], loaded_hashes={"a": HASH})
        rail.record_skill_rail(loaded=["a"], loaded_hashes={"a": OTHER_HASH})
        assert rail._skill_rails[4242]["loaded_hashes"] == {"a": None}

    def test_a_hash_for_a_name_that_was_not_loaded_is_ignored(self, rail):
        """`loaded` is the claim; `loaded_hashes` only annotates it. A digest for
        a name the rail is not recording must not smuggle that name onto the
        trace."""
        rail.record_skill_rail(loaded=["a"], loaded_hashes={"a": HASH, "b": OTHER_HASH})
        stored = rail._skill_rails[4242]
        assert stored["loaded"] == ["a"]
        assert stored["loaded_hashes"] == {"a": HASH}


class TestPydanticAIPassesTheHashes:
    def test_the_handler_forwards_what_it_drained(self, monkeypatch):
        """`pydantic_ai` drains names and hashes from the same scope in one
        breath and hands both to the otel rail."""
        import decimalai.pydantic_ai as pa

        router = _router()
        monkeypatch.setattr(pa, "_get_skill_router", lambda: router)
        monkeypatch.setattr(pa, "_scope", lambda: "run-a")

        seen: List[Dict[str, Any]] = []
        import decimalai.otel as otel_mod
        monkeypatch.setattr(
            otel_mod, "record_skill_rail",
            lambda **kw: seen.append(kw) or True,
        )

        with patch.object(router, "get_skill_body_record", return_value=_record()):
            out = pa._handle_load_skill("restocking-policy")

        assert out.startswith("## Skill: restocking-policy")
        assert seen and seen[-1]["loaded"] == ["restocking-policy"]
        assert seen[-1]["loaded_hashes"] == {"restocking-policy": HASH}


class TestOlderRoutersStillWork:
    """A router object that predates the hash rail — an older installed SDK, or
    a caller's stand-in — must not break the drain. Both adapters answer `{}`
    and report exactly what they reported before."""

    class _NoHashRouter:
        def consume_routing_id(self, **kw): return None
        def consume_offered_names(self, **kw): return []
        def consume_delivered_names(self, **kw): return []
        def consume_loaded_names(self, **kw): return []
        # deliberately no consume_loaded_hashes

    def test_openai_agents_drain_tolerates_it(self, monkeypatch):
        import decimalai.openai_agents as oa

        monkeypatch.setattr(oa, "_skill_router_singleton", self._NoHashRouter())
        assert oa._drain_router_loaded_hashes("trace_1") == {}

    def test_langchain_drain_tolerates_it(self, monkeypatch):
        import decimalai.langchain as lc

        monkeypatch.setattr(lc, "_skill_router_singleton", self._NoHashRouter())
        assert lc._drain_router_loaded_hashes({"s1"}, include_unscoped=True) == {}

    def test_the_release_path_still_frees_every_scope(self, monkeypatch):
        """A router with scoped NAMES but no hash rail must still have all of
        its scopes released.

        The hash release was first written inside the same `try` as the four
        name rails, whose `except` returns. On a router like this one that
        aborted the LOOP after the first scope — releasing one and leaking every
        other, a worse leak than the one the line was added to close. It now
        sits in its own `try`.
        """
        import decimalai.langchain as lc

        released: List[str] = []

        class _ScopedNamesNoHashes:
            def consume_routing_id(self, scope=None): released.append(f"rid:{scope}")
            def consume_offered_names(self, scope=None): return []
            def consume_delivered_names(self, scope=None): return []
            def consume_loaded_names(self, scope=None): return []
            def consume_loaded_hashes(self, scope=None):
                raise AttributeError("no hash rail on this router")

        monkeypatch.setattr(lc, "_skill_router_singleton", _ScopedNamesNoHashes())
        state = lc._RunState()
        state.rail_scopes.update({"s1", "s2", "s3"})

        lc._discard_scoped_router_rails(state)

        assert sorted(released) == ["rid:s1", "rid:s2", "rid:s3"], (
            "the hash release aborted the loop and leaked the remaining scopes: "
            f"{released}"
        )


class TestNativeTracePathCarriesTheVersion:
    """The fourth surface: `@decimalai.trace`, where `load_skill` reports through
    `generic.log_skill_loaded` rather than through an adapter rail.

    It had the same null-hash gap as the adapters and for the same reason —
    `log_skill_loaded` took only a name — so a native-path trace could say WHICH
    skill the model read but never WHICH VERSION.
    """

    def test_a_load_inside_a_native_trace_reports_its_version(self, monkeypatch):
        import decimalai
        import decimalai._config as cfg
        from decimalai._config import DecimalConfig

        prev_config, prev_client = cfg._config, cfg._client
        cfg._config = DecimalConfig(
            api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True,
        )
        cfg._client = MagicMock()
        cfg._client.register_manifest.return_value = {
            "manifest_id": "m", "status": "active",
        }
        try:
            router = _router()
            with decimalai.start_trace(agent_name="native-agent") as ctx:
                with patch.object(
                    router, "get_skill_body_record", return_value=_record(),
                ):
                    router.load_skill("restocking-policy")
                trace = ctx.build_trace()
        finally:
            cfg._config, cfg._client = prev_config, prev_client

        assert trace.skills_loaded_by_agent == ["restocking-policy"]
        assert _activation_map(trace) == {"restocking-policy": HASH}
        _assert_activation_set_unchanged(trace)

    def test_an_explicit_activation_outranks_the_rail(self):
        """`log_skill_activation` is the caller SAYING which version influenced
        the output. A body load observed afterwards must not overwrite it."""
        from decimalai.generic import TraceContext

        ctx = TraceContext(agent_name="a")
        ctx.log_skill_activation(name="restocking-policy", hash=OTHER_HASH)
        ctx.log_skill_loaded(name="restocking-policy", hash=HASH)

        assert ctx._active_skills["restocking-policy"] == OTHER_HASH

    def test_the_old_one_argument_call_still_works(self):
        """Every existing caller, including a user's. The load is recorded; it
        simply has no known version."""
        from decimalai.generic import TraceContext

        ctx = TraceContext(agent_name="a")
        ctx.log_skill_loaded(name="restocking-policy")

        assert ctx._skills_loaded_by_agent == {"restocking-policy"}
        assert ctx._active_skills == {}
