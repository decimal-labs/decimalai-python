"""`DECIMALAI_SKILLS=0`: the environment opt-out for the init() skill loader.

Phase 1 made `init()`'s framework flags install the skill loader as well as
tracing, with `init(..., skills=False)` as the way back to tracing only.
`DECIMAL_AUTO_TRACE=<framework>` calls those same flags at import, so it started
delivering skills too. On that path nobody writes an `init()` call, so a code
argument is no opt-out at all. Since 0.25 only `crewai` has a loader;
`langchain` traces only, and for it `skills=False` (or the variable) says the
app traces only, so a run without an `agent_config()` middleware is not warned
about (`decimalai.langchain._say_runs_get_no_skills`).

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
        "_installed", "_install_agent_name", "_say_runs_get_no_skills",
        "_explicit_manifest_config", "_evals", "_builtin_evals_enabled",
    )
    monkeypatch.setattr(lc, "_installed", False)
    monkeypatch.setattr(lc, "_say_runs_get_no_skills", False)
    monkeypatch.setattr(lc_ctx, "register_configure_hook", lambda *a, **k: None)
    # This machine's own ~/.claude/skills must not be uploaded by a unit test.
    monkeypatch.setattr(skills_mod, "discover_skills", lambda *a, **k: [])
    return lc


def _langchain_flag(monkeypatch, **init_kwargs: Any) -> tuple:
    """`(instrument() calls, _say_runs_get_no_skills)` after one `init(langchain=True, ...)`."""
    import decimalai.langchain as lc

    calls = _record_instrument(monkeypatch, "langchain")
    monkeypatch.setattr(lc, "_say_runs_get_no_skills", False)
    _init(langchain=True, **init_kwargs)
    return calls, lc._say_runs_get_no_skills


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
    def test_langchain_auto_trace_with_skills_off_traces_and_warns_of_nothing(
        self, langchain_adapter, monkeypatch,
    ):
        lc = langchain_adapter
        _auto_trace(monkeypatch, "langchain", "0")
        assert lc._installed is True, "the opt-out must still install tracing"
        assert lc._say_runs_get_no_skills is False

    def test_langchain_auto_trace_unset_traces_and_says_runs_get_no_skills(
        self, langchain_adapter, monkeypatch,
    ):
        """Unset, the env path is the flag form: tracing, and since 0.25 no skills, which the
        first run without an agent_config() middleware says."""
        lc = langchain_adapter
        _auto_trace(monkeypatch, "langchain")
        assert lc._installed is True
        assert lc._say_runs_get_no_skills is True

    @pytest.mark.parametrize("skills_env", ["0", None])
    def test_openai_agents_auto_trace_installs_tracing_either_way(self, monkeypatch, skills_env):
        """Since 0.18 the OpenAI Agents flag has no skill loader to turn off:
        skills come from `agent_config()` in the developer's own code."""
        calls = _record_instrument(monkeypatch, "openai_agents")
        _auto_trace(monkeypatch, "openai-agents", skills_env)
        assert len(calls) == 1, "DECIMAL_AUTO_TRACE=openai-agents installed nothing"
        assert "enable_skill_loader" not in calls[0]

    @pytest.mark.parametrize("skills_env, expected", [("0", []), (None, [1])])
    def test_crewai_follows_the_variable(
        self, monkeypatch, crewai_loader, skills_env, expected,
    ):
        _auto_trace(monkeypatch, "crewai", skills_env)
        assert crewai_loader == expected


# ── 2. the variable is init(skills=...)'s default, in code too ──────────


class TestEnvironmentDefault:
    @pytest.mark.parametrize("raw", ["0", "false", "no", "off", " OFF ", "False"])
    def test_an_off_value_says_the_app_traces_only(self, monkeypatch, raw):
        monkeypatch.setenv(ENV, raw)
        calls, warns = _langchain_flag(monkeypatch, agent_name="support")
        assert calls == [{"agent_name": "support"}] and warns is False

    @pytest.mark.parametrize("raw", ["1", "true", "yes", "on", " On ", ""])
    def test_an_on_value_or_an_empty_one_expects_skills(self, monkeypatch, raw):
        monkeypatch.setenv(ENV, raw)
        calls, warns = _langchain_flag(monkeypatch, agent_name="support")
        assert calls == [{"agent_name": "support"}] and warns is True

    def test_unset_expects_skills(self, monkeypatch):
        calls, warns = _langchain_flag(monkeypatch, agent_name="support")
        assert calls == [{"agent_name": "support"}] and warns is True

    def test_an_explicit_true_beats_the_variable(self, monkeypatch):
        """explicit parameter → environment variable → default, like every
        other init() knob."""
        monkeypatch.setenv(ENV, "0")
        _calls, warns = _langchain_flag(monkeypatch, skills=True)
        assert warns is True

    def test_an_explicit_false_beats_the_variable(self, monkeypatch):
        monkeypatch.setenv(ENV, "1")
        _calls, warns = _langchain_flag(monkeypatch, skills=False)
        assert warns is False

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



# ── 3. an unrecognised value is an opt-out, said out loud ───────────────


class TestUnrecognisedValue:
    @pytest.mark.parametrize("raw", ["disabled", "none", "2", "of"])
    def test_it_turns_the_loader_off_and_warns(self, monkeypatch, caplog, raw):
        monkeypatch.setenv(ENV, raw)
        with caplog.at_level(logging.WARNING, logger="decimalai"):
            _calls, warns = _langchain_flag(monkeypatch)
        assert warns is False
        warned = [r for r in caplog.records if ENV in r.getMessage()]
        assert len(warned) == 1, caplog.text
        message = warned[0].getMessage()
        assert repr(raw) in message
        assert "0" in message and "1" in message, (
            "the warning must say which values the variable accepts"
        )


# ── 4. end to end: the import-time auto-init in a fresh interpreter ─────


@pytest.mark.parametrize("skills_env, expected", [
    ("0", "installed=True warns=False"),
    (None, "installed=True warns=True"),
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
        "print(f'installed={lc._installed} warns={lc._say_runs_get_no_skills}')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == expected, proc.stderr
