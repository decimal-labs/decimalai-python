"""`DECIMALAI_SKILLS=0`: the environment opt-out for the init() skill loader.

Phase 1 made `init()`'s framework flags (`langchain`, `openai_agents`, `adk`,
`crewai`) install the skill loader as well as tracing, with `init(...,
skills=False)` as the way back to tracing only. `DECIMAL_AUTO_TRACE=<framework>`
calls those same flags at import, so it started delivering skills too: every
chat-model call routes the turn's message through the platform and the prompt
gains a skills menu. On that path nobody writes an `init()` call, so a code
argument is no opt-out at all.

`DECIMALAI_SKILLS` is the environment default of `init(skills=...)`, the way
`DECIMALAI_LOAD_SKILL_TOOL` is for `load_skill_tool` and
`DECIMALAI_INJECT_SKILL_BODY` for `inject_skill_body`: an explicit argument
wins, then the variable, then on. `0`/`false`/`no`/`off` turn the loader off.
A value it does not recognise also turns it off, with a warning: the variable
exists to opt out, so a mistyped opt-out must not route every message.

Offline: adapters' process-wide side effects are stubbed in-process, and the
end-to-end check runs the import-time auto-init in a child interpreter.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

import decimalai
import decimalai._config as cfg

KEY = "dai_sk_test"
URL = "http://localhost:9"
ENV = "DECIMALAI_SKILLS"
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Put the SDK's config and client back, and start with the variable unset."""
    monkeypatch.setattr(cfg, "_config", cfg._config)
    monkeypatch.setattr(cfg, "_client", cfg._client)
    monkeypatch.setattr(decimalai, "_global_client", decimalai._global_client)
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv("DECIMAL_AUTO_TRACE", raising=False)
    yield


def _init(**kwargs: Any) -> None:
    kwargs.setdefault("api_key", KEY)
    kwargs.setdefault("base_url", URL)
    kwargs.setdefault("verify", False)
    decimalai.init(**kwargs)


def _keep(monkeypatch, module: Any, *names: str) -> None:
    for name in names:
        monkeypatch.setattr(module, name, getattr(module, name))


@pytest.fixture
def langchain_adapter(monkeypatch):
    """The real LangChain `instrument()`, minus its process-wide side effects.

    Returns the adapter module and a list that gets one entry per skill-loader
    install. Everything that decides WHETHER to install it is the real code.
    """
    pytest.importorskip("langchain_core")
    import langchain_core.tracers.context as lc_ctx

    import decimalai.langchain as lc
    import decimalai.skills as skills_mod

    _keep(
        monkeypatch, lc,
        "_installed", "_skill_loader_installed", "_install_agent_name",
        "_explicit_manifest_config", "_evals", "_builtin_evals_enabled",
        "_skill_body_top_k", "_priority_skills", "_runtime_policy",
    )
    monkeypatch.setattr(lc, "_installed", False)
    monkeypatch.setattr(lc_ctx, "register_configure_hook", lambda *a, **k: None)
    # This machine's own ~/.claude/skills must not be uploaded by a unit test.
    monkeypatch.setattr(skills_mod, "discover_skills", lambda *a, **k: [])
    loader: List[int] = []
    monkeypatch.setattr(lc, "_install_skill_loader", lambda *a, **k: loader.append(1))
    return lc, loader


def _record_instrument(monkeypatch, module_name: str) -> List[Dict[str, Any]]:
    """Replace `decimalai.<module>.instrument` with a recorder of its kwargs.

    `init()` imports `instrument` from the adapter module at call time, so the
    recorder is what the flag form calls.
    """
    mod = pytest.importorskip(f"decimalai.{module_name}")
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(mod, "instrument", lambda **kw: calls.append(kw))
    return calls


@pytest.fixture
def crewai_loader(monkeypatch):
    """`init(crewai=True)` with the exporter and activation stubbed: returns
    one entry per CrewAI skill-loader install."""
    import decimalai.otel as otel

    monkeypatch.setattr(otel, "instrument", lambda **k: MagicMock(name="provider"))
    monkeypatch.setattr(
        decimalai, "_activate_crewai_instrumentation", lambda *a, **k: None,
    )
    calls: List[int] = []
    monkeypatch.setattr(
        decimalai, "_install_crewai_skill_loader", lambda *a, **k: calls.append(1),
    )
    return calls


def _auto_trace(monkeypatch, framework: str, skills_env: Any = None) -> None:
    monkeypatch.setenv("DECIMAL_API_KEY", KEY)
    monkeypatch.setenv("DECIMAL_BASE_URL", URL)
    monkeypatch.setenv("DECIMAL_AUTO_TRACE", framework)
    if skills_env is not None:
        monkeypatch.setenv(ENV, skills_env)
    decimalai._auto_init_from_env()


# ── 1. DECIMAL_AUTO_TRACE can be kept to tracing only ───────────────────


class TestAutoTraceOptOut:
    def test_langchain_auto_trace_with_skills_off_is_tracing_only(
        self, langchain_adapter, monkeypatch,
    ):
        lc, loader = langchain_adapter
        _auto_trace(monkeypatch, "langchain", "0")
        assert lc._installed is True, "the opt-out must still install tracing"
        assert loader == [], (
            "DECIMAL_AUTO_TRACE=langchain with DECIMALAI_SKILLS=0 installed the "
            "skill loader: a zero-code deployment has no way to stay tracing-only"
        )

    def test_langchain_auto_trace_still_delivers_skills_by_default(
        self, langchain_adapter, monkeypatch,
    ):
        """The opt-out is an opt-out: unset, the env path is the flag form."""
        lc, loader = langchain_adapter
        _auto_trace(monkeypatch, "langchain")
        assert lc._installed is True
        assert loader == [1]

    @pytest.mark.parametrize("framework, module", [
        ("openai-agents", "openai_agents"),
        ("adk", "adk"),
    ])
    @pytest.mark.parametrize("skills_env, expected", [("0", False), (None, True)])
    def test_other_loader_frameworks_follow_the_variable(
        self, monkeypatch, framework, module, skills_env, expected,
    ):
        calls = _record_instrument(monkeypatch, module)
        _auto_trace(monkeypatch, framework, skills_env)
        assert len(calls) == 1, f"DECIMAL_AUTO_TRACE={framework} installed nothing"
        assert calls[0]["enable_skill_loader"] is expected

    @pytest.mark.parametrize("skills_env, expected", [("0", []), (None, [1])])
    def test_crewai_follows_the_variable(
        self, monkeypatch, crewai_loader, skills_env, expected,
    ):
        _auto_trace(monkeypatch, "crewai", skills_env)
        assert crewai_loader == expected


# ── 2. the variable is init(skills=...)'s default, in code too ──────────


class TestEnvironmentDefault:
    @pytest.mark.parametrize("raw", ["0", "false", "no", "off", " OFF ", "False"])
    def test_an_off_value_keeps_the_flag_to_tracing(self, monkeypatch, raw):
        calls = _record_instrument(monkeypatch, "langchain")
        monkeypatch.setenv(ENV, raw)
        _init(langchain=True, agent_name="support")
        assert calls == [{"agent_name": "support", "enable_skill_loader": False}]

    @pytest.mark.parametrize("raw", ["1", "true", "yes", "on", " On ", ""])
    def test_an_on_value_or_an_empty_one_delivers_skills(self, monkeypatch, raw):
        calls = _record_instrument(monkeypatch, "langchain")
        monkeypatch.setenv(ENV, raw)
        _init(langchain=True, agent_name="support")
        assert calls == [{"agent_name": "support", "enable_skill_loader": True}]

    def test_unset_delivers_skills(self, monkeypatch):
        calls = _record_instrument(monkeypatch, "langchain")
        _init(langchain=True, agent_name="support")
        assert calls == [{"agent_name": "support", "enable_skill_loader": True}]

    def test_an_explicit_true_beats_the_variable(self, monkeypatch):
        """explicit parameter → environment variable → default, like every
        other init() knob."""
        calls = _record_instrument(monkeypatch, "langchain")
        monkeypatch.setenv(ENV, "0")
        _init(langchain=True, skills=True)
        assert calls[0]["enable_skill_loader"] is True

    def test_an_explicit_false_beats_the_variable(self, monkeypatch):
        calls = _record_instrument(monkeypatch, "langchain")
        monkeypatch.setenv(ENV, "1")
        _init(langchain=True, skills=False)
        assert calls[0]["enable_skill_loader"] is False

    @pytest.mark.parametrize("bad", [None, "0", 0, [{"name": "x"}]])
    def test_skills_still_takes_only_a_bool(self, monkeypatch, bad):
        """The variable supplies the default; it does not widen what the
        argument accepts. `skills=None` or `skills="0"` must not read as
        "use the environment"."""
        monkeypatch.setenv(ENV, "0")
        with pytest.raises(TypeError, match="skills"):
            _init(langchain=True, skills=bad)

    def test_the_variable_is_not_read_without_a_loader_flag(
        self, monkeypatch, caplog,
    ):
        """Bare init() and the provider flags install no loader, so the
        variable means nothing there and is not even parsed."""
        monkeypatch.setenv(ENV, "nonsense")
        providers = pytest.importorskip("decimalai.providers")
        monkeypatch.setattr(providers, "instrument", lambda **k: None)
        with caplog.at_level(logging.WARNING, logger="decimalai"):
            _init()
            _init(anthropic=True)
        assert ENV not in caplog.text

    def test_it_does_not_reach_an_explicit_instrument_call(
        self, langchain_adapter, monkeypatch,
    ):
        """It is init()'s default, not a kill switch for code that asked for
        the loader by name."""
        lc, loader = langchain_adapter
        monkeypatch.setenv(ENV, "0")
        _init()
        lc.instrument(agent_name="support", enable_skill_loader=True)
        assert loader == [1]


# ── 3. an unrecognised value is an opt-out, said out loud ───────────────


class TestUnrecognisedValue:
    @pytest.mark.parametrize("raw", ["disabled", "none", "2", "of"])
    def test_it_turns_the_loader_off_and_warns(self, monkeypatch, caplog, raw):
        calls = _record_instrument(monkeypatch, "langchain")
        monkeypatch.setenv(ENV, raw)
        with caplog.at_level(logging.WARNING, logger="decimalai"):
            _init(langchain=True)
        assert calls[0]["enable_skill_loader"] is False
        warned = [r for r in caplog.records if ENV in r.getMessage()]
        assert len(warned) == 1, caplog.text
        message = warned[0].getMessage()
        assert repr(raw) in message
        assert "0" in message and "1" in message, (
            "the warning must say which values the variable accepts"
        )


# ── 4. end to end: the import-time auto-init in a fresh interpreter ─────


@pytest.mark.parametrize("skills_env, expected", [
    ("0", "installed=True loader=False"),
    (None, "installed=True loader=True"),
])
def test_import_time_auto_trace_honours_the_variable(tmp_path, skills_env, expected):
    """The reviewer's repro: `DECIMAL_AUTO_TRACE=langchain`, then `import
    decimalai`, in a process nothing else has configured."""
    pytest.importorskip("langchain_core")
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("DECIMAL_", "DECIMALAI_"))
    }
    env.update({
        "DECIMAL_AUTO_TRACE": "langchain",
        "DECIMAL_API_KEY": KEY,
        # Connection refused at once: nothing leaves the machine.
        "DECIMAL_BASE_URL": "http://127.0.0.1:9",
        "DECIMALAI_SKIP_VERIFY": "1",
        "DECIMALAI_HANDLE_SIGTERM": "0",
        # Tracing-only LangChain mirrors local SKILL.md files; give it none.
        "HOME": str(tmp_path),
        "PYTHONPATH": str(REPO_ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    if skills_env is not None:
        env[ENV] = skills_env
    code = (
        "import decimalai, decimalai.langchain as lc\n"
        "print(f'installed={lc._installed} loader={lc._skill_loader_installed}')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == expected, proc.stderr
