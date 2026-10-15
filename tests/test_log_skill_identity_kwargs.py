"""``log_skill_activation`` and ``log_skill_loaded`` take the registry identity.

For a framework with no adapter that reads skill folders itself (deepagents, Microsoft
Agent Framework, Strands, a pydantic-ai harness), the manual calls are the only way to say
which skill ran. They gain keyword-only ``skill_id``, ``version_id`` and ``body_sha256``,
the values a managed folder's stamp carries and ``discover_skills`` returns, and those
reach the trace's ``active_skills`` entry and the bytes posted for it.
"""
from __future__ import annotations

import inspect
import json
import logging
import uuid
from typing import Any, Dict, List

import httpx
import pytest

import decimalai
import decimalai._config as cfg
from decimalai import generic
from decimalai._client import DecimalAIClient
from decimalai._config import DecimalConfig
from decimalai.generic import TraceContext
from decimalai.skillpkg import identity

SKILL_ID = str(uuid.UUID(int=11))
VERSION_ID = str(uuid.UUID(int=1101))
BODY_SHA = "d" * 64
HASH = "sha256:" + "a" * 64


def _active(ctx: TraceContext) -> Dict[str, Dict[str, Any]]:
    return {e["name"]: e for e in ctx.build_trace().active_skills}


@pytest.fixture(autouse=True)
def _config(tmp_path, monkeypatch):
    # No lock anywhere above: what reaches the wire is exactly what was logged.
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    identity.clear_cache()
    cfg._config = DecimalConfig(api_key="dai_sk_test", base_url="http://wire.test",
                                enabled=True)
    yield
    identity.clear_cache()


class TestSignatures:
    @pytest.mark.parametrize("fn", [
        TraceContext.log_skill_activation, TraceContext.log_skill_loaded,
        decimalai.log_skill_activation, decimalai.log_skill_loaded,
    ])
    def test_the_identity_kwargs_are_keyword_only(self, fn):
        params = inspect.signature(fn).parameters
        for key in ("skill_id", "version_id", "body_sha256"):
            assert params[key].kind is inspect.Parameter.KEYWORD_ONLY, key
            assert params[key].default is None, key

    def test_positional_arguments_are_refused(self):
        ctx = TraceContext(agent_name="a")
        with pytest.raises(TypeError):
            ctx.log_skill_activation("x", None, SKILL_ID)  # type: ignore[misc]


class TestTheEntry:
    def test_activation_carries_the_identity(self):
        ctx = TraceContext(agent_name="a")
        ctx.log_skill_activation(name="acme-refund-policy", hash=HASH, skill_id=SKILL_ID,
                                 version_id=VERSION_ID, body_sha256=BODY_SHA)
        assert _active(ctx) == {"acme-refund-policy": {
            "name": "acme-refund-policy", "hash": HASH, "skill_id": SKILL_ID,
            "version_id": VERSION_ID, "body_sha256": BODY_SHA,
        }}

    def test_only_what_was_given_is_sent(self):
        ctx = TraceContext(agent_name="a")
        ctx.log_skill_activation(name="edited", skill_id=SKILL_ID, body_sha256=BODY_SHA)
        assert _active(ctx)["edited"] == {"name": "edited", "skill_id": SKILL_ID,
                                          "body_sha256": BODY_SHA}

    def test_a_uuid_is_sent_as_its_string(self):
        ctx = TraceContext(agent_name="a")
        ctx.log_skill_activation(name="x", skill_id=uuid.UUID(SKILL_ID),  # type: ignore[arg-type]
                                 version_id=uuid.UUID(VERSION_ID))  # type: ignore[arg-type]
        assert _active(ctx)["x"]["skill_id"] == SKILL_ID
        assert _active(ctx)["x"]["version_id"] == VERSION_ID

    @pytest.mark.parametrize("bad", ["", "   ", 7, ["id"]])
    def test_an_unusable_value_is_not_sent_and_says_so(self, bad, caplog):
        # The platform reads only non-empty strings; anything else would be dropped
        # there without a word.
        caplog.set_level(logging.WARNING, logger="decimalai")
        ctx = TraceContext(agent_name="a")
        ctx.log_skill_activation(name="x", skill_id=bad, version_id=VERSION_ID)
        assert _active(ctx)["x"] == {"name": "x", "version_id": VERSION_ID}
        assert any("skill_id" in r.getMessage() for r in caplog.records)

    def test_without_the_kwargs_the_entry_is_unchanged(self):
        ctx = TraceContext(agent_name="a")
        ctx.log_skill_activation(name="code-review", hash=HASH)
        ctx.log_skill_activation(name="plain")
        assert ctx.build_trace().active_skills == [
            {"name": "code-review", "hash": HASH}, {"name": "plain"}]

    def test_a_later_activation_replaces_the_identity_like_the_hash(self):
        ctx = TraceContext(agent_name="a")
        ctx.log_skill_activation(name="x", hash=HASH, skill_id=SKILL_ID)
        ctx.log_skill_activation(name="x", version_id=VERSION_ID)
        assert _active(ctx)["x"] == {"name": "x", "version_id": VERSION_ID}


class TestLoaded:
    def test_a_load_with_ids_is_an_activation_with_them(self):
        ctx = TraceContext(agent_name="a")
        ctx.log_skill_loaded(name="acme-refund-policy", skill_id=SKILL_ID,
                             version_id=VERSION_ID, body_sha256=BODY_SHA)
        trace = ctx.build_trace()
        assert trace.active_skills == [{"name": "acme-refund-policy", "skill_id": SKILL_ID,
                                        "version_id": VERSION_ID, "body_sha256": BODY_SHA}]
        # Still a list of names: the loaded rung never carries the identity.
        assert trace.skills_loaded_by_agent == ["acme-refund-policy"]
        assert trace.skills_delivered == ["acme-refund-policy"]
        assert trace.skills_offered_in_prompt == ["acme-refund-policy"]

    def test_a_load_never_overwrites_the_callers_declaration(self):
        ctx = TraceContext(agent_name="a")
        ctx.log_skill_activation(name="x", hash=HASH)
        ctx.log_skill_loaded(name="x", hash="sha256:" + "b" * 64, skill_id=SKILL_ID,
                             version_id=VERSION_ID)
        assert _active(ctx)["x"] == {"name": "x", "hash": HASH}

    def test_a_bare_load_is_still_not_an_activation(self):
        ctx = TraceContext(agent_name="a")
        ctx.log_skill_loaded(name="x")
        assert ctx.build_trace().active_skills == []
        assert ctx._active_skills == {}


# ── on the wire ──────────────────────────────────────────────────────────────


class _Wire:
    def __init__(self) -> None:
        self.requests: List[httpx.Request] = []
        self.client = DecimalAIClient(api_key="dai_sk_test", base_url="http://wire.test")
        self.client._http = httpx.Client(base_url="http://wire.test",
                                         transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/api/v1/manifests":
            return httpx.Response(200, json={"manifest_id": "m-1", "status": "created"})
        return httpx.Response(200, json={"id": "t", "spans": 0, "llm_calls": 0})

    def traces(self) -> List[Dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.url.path == "/api/v1/traces"]


class TestTheKwargsReachThePayload:
    @pytest.fixture
    def wire(self, monkeypatch) -> _Wire:
        w = _Wire()
        monkeypatch.setattr(cfg, "_client", w.client)
        monkeypatch.setattr(generic, "_manifest_id", None)
        monkeypatch.setattr(generic, "_pending_snapshot", None)
        return w

    def test_through_the_module_level_calls(self, wire):
        with decimalai.start_trace(agent_name="strands-agent"):
            decimalai.log_skill_activation(name="acme-refund-policy", skill_id=SKILL_ID,
                                           version_id=VERSION_ID, body_sha256=BODY_SHA)
            decimalai.log_skill_loaded(name="pdf-tools", skill_id=str(uuid.UUID(int=12)),
                                       body_sha256="e" * 64)
        [body] = wire.traces()
        assert sorted(body["active_skills"], key=lambda e: e["name"]) == [
            {"name": "acme-refund-policy", "skill_id": SKILL_ID, "version_id": VERSION_ID,
             "body_sha256": BODY_SHA},
            {"name": "pdf-tools", "skill_id": str(uuid.UUID(int=12)), "body_sha256": "e" * 64},
        ]
        assert body["skills_loaded_by_agent"] == ["pdf-tools"]
