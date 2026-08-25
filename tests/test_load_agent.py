"""`decimalai.load_agent()` — the agent's system prompt, read at run time.

THE ONE INVARIANT THIS MODULE EXISTS FOR: `config.system_prompt is None` means
the agent exists and has no prompt set, and it means nothing else. It is a real
state (the prompt is optional at creation), which is exactly why every OTHER
way of ending up with an empty prompt has to raise instead — a 404, a 5xx, a
timeout, an uninitialized SDK, a backend that answers 200 with a body this SDK
cannot read. An agent that starts with no instructions at all looks identical
to a working one until someone reads its output closely, so the failure has to
happen at the line that loads the prompt, at startup, where it is still cheap.

Every test below is one path to an empty prompt, asserting it raises.
"""

from __future__ import annotations

import dataclasses
import logging

import httpx
import pytest

import decimalai
from decimalai import AgentConfig
from decimalai._client import AgentNotFoundError, DecimalAIClient, DecimalAPIError
from decimalai._config import DecimalConfigError

AGENT = "refund-bot"
PROMPT = "Never issue a refund over $500."
BASE_URL = "http://backend.test"


def _payload(**overrides):
    """The real shape of `GET /api/v1/agents/{name}/prompt` (agents.py:3176)."""
    body = {
        "agent_name": AGENT,
        "agent_id": "5f2c1a90-0f1e-4b6c-9a11-2b3c4d5e6f70",
        "resolved_from": None,
        "system_prompt": PROMPT,
        "version_id": "9c1f0b22-3344-4d55-8e66-77f8899aabbc",
        "version_number": 3,
        "content_hash": "b7f4c1",
        "label": "tightened refund wording",
        "provenance": "ui",
        "created_at": "2026-08-25T10:14:02.113Z",
        "updated_at": "2026-08-25T10:14:02.113Z",
        "version_mode": "latest",
        "pinned_version_number": None,
    }
    body.update(overrides)
    return body


def _client(handler, monkeypatch=None):
    """A real DecimalAIClient whose transport is a MockTransport.

    Real client, not a stub: the URL quoting, the 304 branch and the 404 branch
    are the code under test, and a MagicMock would answer for all three.
    """
    client = DecimalAIClient(api_key="dai_sk_x", base_url=BASE_URL)
    client._http = httpx.Client(
        base_url=BASE_URL, transport=httpx.MockTransport(handler),
    )
    if monkeypatch is not None:
        # What `init()` sets, which is what `_get_client()` reads.
        monkeypatch.setattr("decimalai._config._client", client)
    return client


def _always(status=200, json_body=None, text=None, record=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record.append(request)
        if text is not None:
            return httpx.Response(status, text=text, request=request)
        return httpx.Response(status, json=json_body, request=request)
    return handler


# ── the two real states ──────────────────────────────────────────────


class TestReadsTheRealStates:
    def test_a_configured_prompt_comes_back_whole(self, monkeypatch):
        _client(_always(json_body=_payload()), monkeypatch)
        config = decimalai.load_agent(AGENT)

        assert config.system_prompt == PROMPT
        assert config.agent_name == AGENT
        assert config.version_number == 3
        assert config.content_hash == "b7f4c1"
        assert config.label == "tightened refund wording"
        assert config.version_mode == "latest"
        assert config.pinned_version_number is None

    def test_no_prompt_set_is_none_and_is_not_an_error(self, monkeypatch):
        """The other real state. Every version-scoped field is null with it, so
        nobody reads a stale version number next to an absent prompt."""
        _client(_always(json_body=_payload(
            system_prompt=None, version_id=None, version_number=None,
            content_hash=None, label=None, provenance=None,
        )), monkeypatch)
        config = decimalai.load_agent(AGENT)

        assert config.system_prompt is None
        assert config.version_number is None
        assert config.agent_name == AGENT

    def test_a_pinned_agent_reports_the_pin(self, monkeypatch):
        _client(_always(json_body=_payload(
            version_mode="pinned", pinned_version_number=2, version_number=2,
        )), monkeypatch)
        config = decimalai.load_agent(AGENT)
        assert (config.version_mode, config.pinned_version_number) == ("pinned", 2)

    def test_a_renamed_agent_answers_to_its_old_name(self, monkeypatch):
        """A deployed file keeps sending the name it was generated with —
        there is no server→client push — so the server canonicalizes, and the
        config says which name was asked for."""
        _client(_always(json_body=_payload(
            agent_name="refunds", resolved_from=AGENT,
        )), monkeypatch)
        config = decimalai.load_agent(AGENT)
        assert config.agent_name == "refunds"
        assert config.resolved_from == AGENT

    def test_a_rename_is_warned_about_because_traces_do_not_follow_it(
        self, monkeypatch, caplog
    ):
        """The asymmetry that makes this worth a warning: the prompt and the
        skills resolve through `canonical_agent_name`, and trace ingest does
        not call it at all. So after a rename the agent looks healthy and is
        quietly split in two — prompt and skills on the new name, runs on the
        old one, nothing erroring. This is the only place that sees both."""
        _client(_always(json_body=_payload(
            agent_name="refunds", resolved_from=AGENT,
        )), monkeypatch)
        monkeypatch.setattr("decimalai._agent._WARNED_RENAMES", set())

        with caplog.at_level(logging.WARNING, logger="decimalai"):
            config = decimalai.load_agent(AGENT)

        assert len(caplog.records) == 1
        msg = caplog.records[0].getMessage()
        assert AGENT in msg and "refunds" in msg
        # Names the exact edit that fixes it, not just the fact.
        assert "instrument(agent_name=" in msg
        # And warning is ALL it does — the caller's name binding stays theirs.
        assert config.agent_name == "refunds"

    def test_the_rename_warning_fires_once_per_process(self, monkeypatch, caplog):
        """A warning that repeats on every call is a warning people filter —
        the same reason `_warn_on_near_miss_agent_name` fires only on a near
        miss."""
        _client(_always(json_body=_payload(
            agent_name="refunds", resolved_from=AGENT,
        )), monkeypatch)
        monkeypatch.setattr("decimalai._agent._WARNED_RENAMES", set())

        with caplog.at_level(logging.WARNING, logger="decimalai"):
            decimalai.load_agent(AGENT)
            decimalai.load_agent(AGENT)

        assert len(caplog.records) == 1

    def test_an_agent_that_was_not_renamed_warns_about_nothing(
        self, monkeypatch, caplog
    ):
        _client(_always(json_body=_payload()), monkeypatch)
        monkeypatch.setattr("decimalai._agent._WARNED_RENAMES", set())

        with caplog.at_level(logging.WARNING, logger="decimalai"):
            decimalai.load_agent(AGENT)

        assert caplog.records == []


# ── every other route to an empty prompt raises ──────────────────────


class TestFailsClosed:
    def test_a_missing_agent_raises_rather_than_reporting_no_prompt(self, monkeypatch):
        _client(_always(404, json_body={
            "detail": f"Agent '{AGENT}' not found — no traces or manifests in this org",
        }), monkeypatch)
        with pytest.raises(AgentNotFoundError) as exc:
            decimalai.load_agent(AGENT)

        msg = str(exc.value)
        assert AGENT in msg
        assert "not found" in msg
        # The fixable failure, so it names the fix.
        assert "dashboard" in msg and "workspace" in msg
        assert exc.value.agent_name == AGENT
        # Still catchable the old way.
        assert isinstance(exc.value, httpx.HTTPStatusError)

    def test_a_backend_with_no_prompt_route_says_so(self, monkeypatch):
        """FastAPI answers an unmatched route with exactly {"detail": "Not
        Found"} — same status as a missing agent, opposite fix. Guessing wrong
        sends someone to rename an agent that was never the problem."""
        _client(_always(404, json_body={"detail": "Not Found"}), monkeypatch)
        with pytest.raises(AgentNotFoundError) as exc:
            decimalai.load_agent(AGENT)
        assert "matched no route" in str(exc.value)

    def test_a_5xx_raises(self, monkeypatch):
        _client(_always(500, json_body={"detail": "boom"}), monkeypatch)
        with pytest.raises(DecimalAPIError):
            decimalai.load_agent(AGENT)

    def test_a_bad_key_raises(self, monkeypatch):
        _client(_always(401, json_body={"detail": "Invalid API key"}), monkeypatch)
        with pytest.raises(DecimalAPIError) as exc:
            decimalai.load_agent(AGENT)
        assert "Invalid API key" in str(exc.value)

    def test_an_unresolvable_pin_raises_rather_than_falling_back(self, monkeypatch):
        """The 409 exists because falling back to `latest` under a pinned label
        is the incident the backend's CHECK constraint was added for."""
        _client(_always(409, json_body={"detail": {
            "error": "prompt_version_unresolvable", "message": "pinned version is gone",
        }}), monkeypatch)
        with pytest.raises(DecimalAPIError) as exc:
            decimalai.load_agent(AGENT)
        assert "prompt_version_unresolvable" in str(exc.value)

    def test_a_network_failure_raises(self, monkeypatch):
        def handler(request):
            raise httpx.ConnectError("connection refused", request=request)
        _client(handler, monkeypatch)
        with pytest.raises(httpx.ConnectError):
            decimalai.load_agent(AGENT)

    def test_a_timeout_raises(self, monkeypatch):
        def handler(request):
            raise httpx.ReadTimeout("timed out", request=request)
        _client(handler, monkeypatch)
        with pytest.raises(httpx.ReadTimeout):
            decimalai.load_agent(AGENT)

    def test_an_uninitialized_sdk_raises(self, monkeypatch):
        monkeypatch.setattr("decimalai._config._client", None)
        with pytest.raises(DecimalConfigError) as exc:
            decimalai.load_agent(AGENT)
        assert "init" in str(exc.value)

    def test_an_empty_name_is_refused_before_any_request(self, monkeypatch):
        seen = []
        _client(_always(json_body=_payload(), record=seen), monkeypatch)
        for name in ("", "   ", None):
            with pytest.raises(ValueError):
                decimalai.load_agent(name)  # type: ignore[arg-type]
        assert seen == []


class TestRefusesAResponseItCannotRead:
    """A 200 whose body is not a prompt payload. `.get("system_prompt")` would
    turn every one of these into a plausible-looking empty prompt — the same
    shape as the `?? 'free'` default that told paying customers they were on
    the free plan."""

    @pytest.mark.parametrize("body", [
        {},                                        # an older backend
        {"agent_name": AGENT},                     # the field renamed or dropped
        {"detail": "Not Found"},                   # an error body served as 200
        [],                                        # a list where an object belongs
        "OK",                                      # a proxy's health check
    ])
    def test_a_wrong_shaped_200_raises(self, body, monkeypatch):
        _client(_always(json_body=body), monkeypatch)
        with pytest.raises(ValueError) as exc:
            decimalai.load_agent(AGENT)
        assert "system_prompt" in str(exc.value)
        assert "Refusing to treat this as 'no prompt set'" in str(exc.value)

    def test_a_login_page_raises_and_is_not_quoted_back(self, monkeypatch):
        """A captive portal / SSO redirect is the realistic version of this.
        The diagnosis names the shape; it never pastes the body into a log,
        because a body we could not read may hold anything."""
        html = "<html><body>Sign in to continue</body></html>"
        _client(_always(text=html), monkeypatch)
        with pytest.raises(Exception) as exc:  # noqa: PT011 — JSON or ValueError
            decimalai.load_agent(AGENT)
        assert "Sign in to continue" not in str(exc.value)

    def test_a_non_string_prompt_raises(self, monkeypatch):
        _client(_always(json_body=_payload(system_prompt={"text": PROMPT})),
                monkeypatch)
        with pytest.raises(ValueError) as exc:
            decimalai.load_agent(AGENT)
        assert "not a string" in str(exc.value)


# ── the shape of what comes back ─────────────────────────────────────


class TestAgentConfigShape:
    def test_it_is_frozen(self, monkeypatch):
        _client(_always(json_body=_payload()), monkeypatch)
        config = decimalai.load_agent(AGENT)
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.system_prompt = "something else"  # type: ignore[misc]

    def test_a_missing_field_fails_at_the_attribute(self, monkeypatch):
        """A dataclass, not a dict: against an older backend the failure is an
        AttributeError on the line that reads it, not a KeyError days later
        inside a request handler."""
        _client(_always(json_body=_payload()), monkeypatch)
        config = decimalai.load_agent(AGENT)
        with pytest.raises(AttributeError):
            config.systen_prompt  # noqa: B018 — the typo is the test

    def test_repr_reports_the_prompt_without_printing_it(self, monkeypatch):
        """A prompt runs to 100,000 characters and lands in every traceback."""
        _client(_always(json_body=_payload()), monkeypatch)
        text = repr(decimalai.load_agent(AGENT))
        assert PROMPT not in text
        assert f"<{len(PROMPT)} chars>" in text
        assert AGENT in text

    def test_repr_still_distinguishes_no_prompt(self, monkeypatch):
        _client(_always(json_body=_payload(system_prompt=None)), monkeypatch)
        assert "system_prompt=None" in repr(decimalai.load_agent(AGENT))

    def test_it_carries_no_model(self):
        """The platform stores no per-agent model, so AgentConfig does not
        pretend to. A field that is always None is a gap wearing a real value's
        costume, and `model=config.model` would silently build every agent on
        the framework's default."""
        assert "model" not in {f.name for f in dataclasses.fields(AgentConfig)}


# ── the request itself ───────────────────────────────────────────────


class TestTheRequest:
    def test_it_asks_the_prompt_route_directly(self, monkeypatch):
        """NOT a scan of `GET /api/v1/agents`: passing a limit there activates
        a truncating pagination path whose ordering puts manifest-only agents
        last, so the never-traced UI-created agent this feature exists for is
        the first one dropped."""
        seen = []
        _client(_always(json_body=_payload(), record=seen), monkeypatch)
        decimalai.load_agent(AGENT)

        assert len(seen) == 1
        assert seen[0].url.path == f"/api/v1/agents/{AGENT}/prompt"
        assert seen[0].url.params.get("version") is None

    def test_a_name_needing_escaping_still_addresses_the_right_agent(
        self, monkeypatch
    ):
        seen = []
        _client(_always(json_body=_payload(), record=seen), monkeypatch)
        decimalai.load_agent("[Demo] support/agent")
        # raw_path, not path: `path` hands back the decoded form, which would
        # look identical whether or not the slash was escaped. The slash is the
        # one that matters — unescaped it addresses a different route entirely.
        assert seen[0].url.raw_path == (
            b"/api/v1/agents/%5BDemo%5D%20support%2Fagent/prompt"
        )

    def test_version_reads_one_historical_version(self, monkeypatch):
        seen = []
        _client(_always(json_body=_payload(version_number=2), record=seen),
                monkeypatch)
        assert decimalai.load_agent(AGENT, version=2).version_number == 2
        assert seen[0].url.params["version"] == "2"

    def test_load_agent_never_sends_a_conditional_request(self, monkeypatch):
        """The route supports ETag / If-None-Match, and `load_agent()`
        deliberately does not use it. It runs once per process, so there is
        nothing cached to revalidate — and a cache is precisely what would
        break the no-redeploy property the call is sold on."""
        seen = []
        _client(_always(json_body=_payload(), record=seen), monkeypatch)
        decimalai.load_agent(AGENT)
        assert "if-none-match" not in seen[0].headers

    def test_the_client_can_still_poll_with_an_etag(self):
        """For anyone building a refresh loop: 304 comes back as None —
        "unchanged" — which is a different answer from a payload whose
        system_prompt is None, "no prompt set"."""
        seen = []
        client = _client(_always(304, json_body=None, record=seen))
        assert client.get_agent_prompt(AGENT, if_none_match="b7f4c1") is None
        assert seen[0].headers["if-none-match"] == "b7f4c1"
