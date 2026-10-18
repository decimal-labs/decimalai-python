"""The native-load observer (``decimalai._native``): which version a framework's own skill
loader delivered, decided from the delivered bytes and never from the file on disk.

The resolution order is the design's D2: the delivered stamp when its hash is the delivered
body's, the lock by body hash, every body hash this process has seen (the mid-session
update), the skill alone with no version (unverified), and a skill DecimalAI did not install.
"""
from __future__ import annotations

import logging
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
from skillevaluation import skillfmt

import decimalai._config as cfg
from decimalai import _native
from decimalai._config import DecimalConfig
from decimalai._native import (
    BY_LOCK, BY_SEEN, BY_STAMP, UNMANAGED, UNVERIFIED, Activation, Load, NativeLoads,
    record, resolve,
)
from decimalai.generic import TraceContext, _set_current_trace
from decimalai.skillpkg import identity
from decimalai.skillpkg.identity import ManagedIndex, ManagedSkill

SKILL_ID = str(uuid.UUID(int=31))
OTHER_SKILL_ID = str(uuid.UUID(int=32))
V1 = str(uuid.UUID(int=3101))
V2 = str(uuid.UUID(int=3102))
BODY_V1 = "# Refunds\n\nREFUND-SENTINEL-V1: store credit only.\n"
BODY_V2 = "# Refunds\n\nREFUND-SENTINEL-V2: cash back within 30 days.\n"
H1 = skillfmt.body_sha256(BODY_V1)
H2 = skillfmt.body_sha256(BODY_V2)


def _skill(version_id: str = V1, body_sha: str = H1, *, install_name: str = "acme-refund-policy",
           registry_name: str = "acme/refund-policy",
           skill_id: str = SKILL_ID) -> ManagedSkill:
    return ManagedSkill(install_name=install_name, registry_name=registry_name,
                        skill_id=skill_id, version_id=version_id, body_sha256=body_sha,
                        path=f"/project/.agents/skills/{install_name}")


def _index(*skills: ManagedSkill) -> ManagedIndex:
    return ManagedIndex(skills, plugin_name="decimalai-skills", directory="/project")


def _skill_md(body: str, *, version_id: str = V1, body_sha: Optional[str] = None,
              skill_id: str = SKILL_ID, name: str = "acme-refund-policy",
              registry_name: str = "acme/refund-policy") -> str:
    digest = body_sha if body_sha is not None else skillfmt.body_sha256(body)
    return (
        "---\n"
        f"name: {name}\n"
        "description: \"Refund rules.\"\n"
        "metadata:\n"
        f"  decimalai-skill-id: \"{skill_id}\"\n"
        f"  decimalai-version-id: \"{version_id}\"\n"
        f"  decimalai-body-sha256: \"{digest}\"\n"
        f"  decimalai-name: \"{registry_name}\"\n"
        "---\n\n" + body
    )


@pytest.fixture(autouse=True)
def _fresh(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    identity.clear_cache()
    _native._reset_seen()
    _native._reset_warnings()
    monkeypatch.setattr(cfg, "_config", DecimalConfig(api_key="dai_sk_test",
                                                      base_url="http://wire.test", enabled=True))
    yield
    _set_current_trace(None)
    identity.clear_cache()
    _native._reset_seen()


class TestResolve:
    def test_a_stamp_whose_hash_is_the_delivered_body_names_the_version(self):
        got = resolve(Load(framework="adk", name="acme-refund-policy",
                           text=_skill_md(BODY_V1)), index=_index())
        assert got == Activation("acme/refund-policy", SKILL_ID, V1, H1, BY_STAMP)

    def test_the_stamp_wins_over_the_lock_after_an_update(self):
        # The lock already says v2; the loader still delivered its cached v1 copy.
        got = resolve(Load(name="acme-refund-policy", text=_skill_md(BODY_V1)),
                      index=_index(_skill(V2, H2)))
        assert (got.version_id, got.how) == (V1, BY_STAMP)

    def test_a_stamp_the_body_does_not_match_is_not_believed(self):
        edited = BODY_V1 + "\nA local edit.\n"
        got = resolve(Load(name="acme-refund-policy", text=_skill_md(edited, body_sha=H1)),
                      index=_index())
        assert got == Activation("acme/refund-policy", SKILL_ID, None,
                                 skillfmt.body_sha256(edited), UNVERIFIED)

    def test_a_body_without_frontmatter_is_found_in_the_lock_by_its_hash(self):
        # Claude Code and the Pydantic AI harness deliver the body only.
        got = resolve(Load(name="decimalai-skills:acme-refund-policy", text=BODY_V1),
                      index=_index(_skill()))
        assert got == Activation("acme/refund-policy", SKILL_ID, V1, H1, BY_LOCK)

    def test_a_body_handed_over_apart_from_its_metadata(self):
        meta = {"decimalai-skill-id": SKILL_ID, "decimalai-version-id": V1,
                "decimalai-body-sha256": H1, "decimalai-name": "acme/refund-policy"}
        got = resolve(Load(name="acme-refund-policy", body=BODY_V1, metadata=meta),
                      index=_index())
        assert (got.version_id, got.how) == (V1, BY_STAMP)

    def test_mid_session_update_names_the_version_delivered_not_the_one_on_disk(self):
        # Run 1: the lock holds v1. Run 2: `skills update` moved the lock to v2, but the
        # loader (Microsoft Agent Framework caches by default) delivers v1, with no stamp.
        assert resolve(Load(name="acme-refund-policy", text=BODY_V1),
                       index=_index(_skill(V1, H1))).version_id == V1
        got = resolve(Load(name="acme-refund-policy", text=BODY_V1),
                      index=_index(_skill(V2, H2)))
        assert got == Activation("acme/refund-policy", SKILL_ID, V1, H1, BY_SEEN)
        # And the loader that re-reads gets the new version credited.
        got = resolve(Load(name="acme-refund-policy", text=BODY_V2),
                      index=_index(_skill(V2, H2)))
        assert (got.version_id, got.how) == (V2, BY_LOCK)

    def test_an_unknown_body_of_a_known_skill_is_unverified(self):
        got = resolve(Load(name="acme-refund-policy", text="# Something else entirely\n"),
                      index=_index(_skill()))
        assert (got.name, got.skill_id, got.version_id, got.how) == (
            "acme/refund-policy", SKILL_ID, None, UNVERIFIED)
        assert got.body_sha256 == skillfmt.body_sha256("# Something else entirely\n")

    def test_a_body_two_installed_skills_share_names_neither_version(self):
        twin = _skill(install_name="beta-refunds", registry_name="beta/refunds",
                      skill_id=OTHER_SKILL_ID, version_id=V2)
        got = resolve(Load(name="beta-refunds", text=BODY_V1), index=_index(_skill(), twin))
        assert (got.skill_id, got.version_id, got.how) == (OTHER_SKILL_ID, None, UNVERIFIED)

    def test_a_skill_decimalai_did_not_install(self):
        got = resolve(Load(name="team-handbook", text="# Our handbook\n"), index=_index(_skill()))
        assert got == Activation("team-handbook", None, None,
                                 skillfmt.body_sha256("# Our handbook\n"), UNMANAGED)

    def test_a_path_names_the_folder(self):
        got = resolve(Load(path="/project/.agents/skills/acme-refund-policy/SKILL.md",
                           text="# drifted\n"), index=_index(_skill()))
        assert (got.skill_id, got.how) == (SKILL_ID, UNVERIFIED)

    def test_a_hashed_install_name_is_reported_as_itself(self):
        name = skillfmt.install_name_disambiguated("acme-refund-policy", SKILL_ID)
        got = resolve(Load(name=name, text=_skill_md(BODY_V1, name=name)), index=_index())
        assert got.name == name

    def test_no_body_and_no_name_is_nothing(self):
        assert resolve(Load(framework="x"), index=_index(_skill())) is None

    def test_broken_frontmatter_does_not_raise(self):
        got = resolve(Load(name="acme-refund-policy", text="---\nname: [unclosed\n---\nbody\n"),
                      index=_index(_skill()))
        assert got.how == UNVERIFIED

    def test_it_never_reads_the_folder(self, tmp_path, monkeypatch):
        opened: List[str] = []
        real_open = open
        monkeypatch.setattr("builtins.open",
                            lambda f, *a, **k: opened.append(str(f)) or real_open(f, *a, **k))
        resolve(Load(name="acme-refund-policy", text=BODY_V1), index=_index(_skill()))
        assert not [f for f in opened if f.endswith("SKILL.md")]


class TestEntry:
    def test_only_the_keys_that_have_a_value(self):
        assert Activation("a", None, None, None, UNMANAGED).entry() == {"name": "a"}
        assert Activation("a/b", SKILL_ID, None, H1, UNVERIFIED).entry() == {
            "name": "a/b", "skill_id": SKILL_ID, "body_sha256": H1}
        assert Activation("a/b", SKILL_ID, V1, H1, BY_LOCK).entry() == {
            "name": "a/b", "skill_id": SKILL_ID, "version_id": V1, "body_sha256": H1}


def _trace(**fields: Any) -> SimpleNamespace:
    base: Dict[str, Any] = dict(active_skills=[], skills_loaded_by_agent=[],
                                skills_delivered=[], skills_offered_in_prompt=[])
    base.update(fields)
    return SimpleNamespace(**base)


class TestNativeLoads:
    def test_a_repeat_load_is_counted_once(self):
        loads = NativeLoads()
        act = Activation("acme/refund-policy", SKILL_ID, V1, H1, BY_STAMP)
        assert loads.add(act) is True
        assert loads.add(act) is False
        assert loads.activations == [act]

    def test_the_first_version_a_run_loaded_is_kept(self):
        loads = NativeLoads()
        loads.add(Activation("acme/refund-policy", SKILL_ID, V1, H1, BY_STAMP))
        loads.add(Activation("acme/refund-policy", SKILL_ID, V2, H2, BY_LOCK))
        assert [a.version_id for a in loads.activations] == [V1]

    def test_apply_writes_the_ids_and_the_name_lists(self):
        loads = NativeLoads()
        loads.listed(["acme/refund-policy", "acme-returns-desk"])
        loads.add(Activation("acme/refund-policy", SKILL_ID, V1, H1, BY_STAMP))
        trace = _trace(active_skills=[{"name": "acme/refund-policy", "hash": "sha256:" + H1}],
                       skills_loaded_by_agent=["other"])
        loads.apply(trace)
        assert trace.active_skills == [{"name": "acme/refund-policy", "skill_id": SKILL_ID,
                                        "version_id": V1, "body_sha256": H1}]
        assert trace.skills_loaded_by_agent == ["acme/refund-policy", "other"]
        assert trace.skills_delivered == ["acme/refund-policy"]
        assert trace.skills_offered_in_prompt == ["acme-returns-desk", "acme/refund-policy"]

    def test_a_load_is_not_evidence_the_name_was_in_the_prompt(self):
        loads = NativeLoads()
        loads.add(Activation("acme/refund-policy", SKILL_ID, V1, H1, BY_STAMP))
        trace = _trace()
        loads.apply(trace)
        assert trace.skills_offered_in_prompt == []

    def test_an_entry_the_caller_declared_with_an_id_is_kept(self):
        declared = {"name": "acme/refund-policy", "skill_id": SKILL_ID, "version_id": V2}
        loads = NativeLoads()
        loads.add(Activation("acme/refund-policy", SKILL_ID, V1, H1, BY_STAMP))
        trace = _trace(active_skills=[declared])
        loads.apply(trace)
        assert trace.active_skills == [declared]

    def test_an_empty_collector_changes_nothing(self):
        trace = _trace(skills_offered_in_prompt=["x"])
        NativeLoads().apply(trace)
        assert trace.skills_offered_in_prompt == ["x"]


class TestRecord:
    def test_an_adapter_collector_gets_the_load(self):
        loads = NativeLoads()
        got = record(Load(name="acme-refund-policy", text=_skill_md(BODY_V1)), loads=loads,
                     index=_index())
        assert loads.activations == [got]

    def test_a_generic_trace_carries_the_load_onto_the_wire(self):
        ctx = TraceContext(agent_name="native-test")
        _set_current_trace(ctx)
        record(Load(name="acme-refund-policy", text=_skill_md(BODY_V1)), index=_index())
        record(Load(name="acme-refund-policy", text=_skill_md(BODY_V1)), index=_index())
        trace = ctx.build_trace()
        assert trace.active_skills == [{"name": "acme/refund-policy", "skill_id": SKILL_ID,
                                        "version_id": V1, "body_sha256": H1}]
        assert trace.skills_loaded_by_agent == ["acme/refund-policy"]
        assert trace.skills_offered_in_prompt == []

    def test_the_live_otel_run_gets_the_load(self):
        trace_api = pytest.importorskip("opentelemetry.trace")
        from opentelemetry.sdk.trace import TracerProvider

        from decimalai import otel

        otel._reset_skill_rails()
        tracer = TracerProvider().get_tracer("native-test")
        with tracer.start_as_current_span("run") as span:
            record(Load(framework="strands", name="acme-refund-policy",
                        text=_skill_md(BODY_V1)), index=_index())
            tid = span.get_span_context().trace_id
        loads = otel._pop_native_rail(tid)
        assert [a.version_id for a in loads.activations] == [V1]
        assert trace_api  # imported for the skip only

    def test_with_nowhere_to_record_it_warns_once(self, caplog):
        caplog.set_level(logging.WARNING, logger="decimalai.native")
        for _ in range(3):
            got = record(Load(framework="strands", name="acme-refund-policy",
                              text=_skill_md(BODY_V1)), index=_index())
        assert got.version_id == V1
        warnings = [r for r in caplog.records if "no trace is open" in r.getMessage()]
        assert len(warnings) == 1 and "strands" in warnings[0].getMessage()

    def test_a_resolver_failure_never_reaches_the_agent(self, monkeypatch):
        monkeypatch.setattr(_native, "resolve", lambda *a, **k: 1 / 0)
        assert record(Load(name="x", text="y"), loads=NativeLoads()) is None
