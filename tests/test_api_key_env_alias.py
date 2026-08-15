"""`init()` accepts DECIMALAI_API_KEY as an alias for DECIMAL_API_KEY.

The CLI has taken both spellings since it shipped
(``envvar=["DECIMAL_API_KEY", "DECIMALAI_API_KEY"]`` in ``cli/main.py``), and
the install snippet on the marketing site exports the alias. The library took
only the primary, so a copy-paste developer got
``DecimalConfigError: No API key provided`` with a usable key sitting in the
environment — the SDK disagreeing with its own CLI about the name of its own
variable.

What is pinned here: the primary still wins when both are set, the alias works
on its own, and neither set still raises an error that names the primary (the
one variable the docs and the error message should keep agreeing on).
"""
from __future__ import annotations

import pytest

import decimalai
import decimalai._config as _cfg
from decimalai._config import DecimalConfigError

PRIMARY = "DECIMAL_API_KEY"
ALIAS = "DECIMALAI_API_KEY"


@pytest.fixture(autouse=True)
def _clean_key_env(monkeypatch):
    """Neither spelling leaks in from the developer's own shell."""
    monkeypatch.delenv(PRIMARY, raising=False)
    monkeypatch.delenv(ALIAS, raising=False)
    yield


def _init_and_read_key() -> str:
    decimalai.init(verify=False)
    return _cfg._config.api_key


def test_primary_wins_when_both_are_set(monkeypatch):
    monkeypatch.setenv(PRIMARY, "dai_sk_primary")
    monkeypatch.setenv(ALIAS, "dai_sk_alias")
    assert _init_and_read_key() == "dai_sk_primary"


def test_alias_alone_is_accepted(monkeypatch):
    monkeypatch.setenv(ALIAS, "dai_sk_alias")
    assert _init_and_read_key() == "dai_sk_alias"


def test_primary_alone_still_works(monkeypatch):
    monkeypatch.setenv(PRIMARY, "dai_sk_primary")
    assert _init_and_read_key() == "dai_sk_primary"


def test_neither_set_still_raises_naming_the_primary():
    with pytest.raises(DecimalConfigError) as exc:
        decimalai.init(verify=False)
    message = str(exc.value)
    assert "No API key provided" in message
    assert PRIMARY in message


def test_explicit_argument_beats_both_env_vars(monkeypatch):
    monkeypatch.setenv(PRIMARY, "dai_sk_primary")
    monkeypatch.setenv(ALIAS, "dai_sk_alias")
    decimalai.init(api_key="dai_sk_explicit", verify=False)
    assert _cfg._config.api_key == "dai_sk_explicit"


def test_blank_primary_falls_through_to_alias(monkeypatch):
    # An exported-but-empty primary is not a key; it must not shadow a real
    # alias. `os.environ.get(PRIMARY, "")` is truthy for "   ", which is how a
    # first-attempt fix would swallow this case.
    monkeypatch.setenv(PRIMARY, "   ")
    monkeypatch.setenv(ALIAS, "dai_sk_alias")
    assert _init_and_read_key() == "dai_sk_alias"


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, ""),
        ({PRIMARY: "a"}, "a"),
        ({ALIAS: "b"}, "b"),
        ({PRIMARY: "a", ALIAS: "b"}, "a"),
        ({PRIMARY: "  a  "}, "a"),
    ],
)
def test_resolver_helper(monkeypatch, env, expected):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert decimalai._api_key_from_env() == expected


def test_auto_init_sees_the_alias(monkeypatch):
    """The bare auto-init path resolves the alias too.

    Without this, exporting only the alias meant `import decimalai` warned
    that the key was "missing" (DECIMAL_AUTO_TRACE path) or silently skipped
    the bare init — while holding a usable key.
    """
    monkeypatch.setenv(ALIAS, "dai_sk_alias")
    called = {}

    def _fake_init(**kwargs):
        called["kwargs"] = kwargs

    monkeypatch.setattr(decimalai, "init", _fake_init)
    decimalai._auto_init_from_env()
    assert "kwargs" in called, "bare auto-init did not run with only the alias set"
