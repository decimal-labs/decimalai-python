"""Top-level test conftest for the DecimalAI Python SDK.

Four cross-cutting concerns are handled here so individual test modules
don't have to:

1. **Init-time verify probe is bypassed** via the ``DECIMALAI_SKIP_VERIFY``
   env var, set at ``pytest_configure``. Unit tests don't have a real
   backend (or use fake keys against a real one); the probe would either
   DNS-fail or 401, neither of which is what the unit tests are exercising.
   Integration tests under ``tests/integration/`` can re-enable verify by
   unsetting this var.

2. **Module-instance stability for ``decimalai`` + submodules.**
   ``test_auto_init_logging.py`` legitimately reimports ``decimalai`` (and
   sometimes ``decimalai._config``) to test the auto-init-on-import path.
   But other test modules import names like ``from decimalai._config
   import _is_manifest_only`` at module load — those bindings stay attached
   to the *original* module instance. After a reimport, ``init()`` mutates
   the NEW ``decimalai._config`` global while the test's bound name still
   reads the OLD one, and the assertion fails with no obvious cause.

   The ``_restore_decimalai_modules`` autouse fixture below snapshots the
   live ``decimalai*`` entries in ``sys.modules`` before each test and
   restores them after. A test that reimports the package gets to see its
   reimported view during the test body; the next test sees the original
   instances again. The fix is invisible to tests — no per-test changes.

3. **LangChain global instrumentation is torn down after each test.**
   ``decimalai.langchain.instrument()`` publishes a handler into a module
   ContextVar and hands that var to langchain-core's
   ``register_configure_hook``. Both are process-global and the hook list has
   no unregister, so the ContextVar is the only half a test can put back —
   and the adapter offers no uninstall to do it with. Any test that installs
   (directly, or indirectly through a ``DECIMAL_AUTO_TRACE=langchain``
   module-load auto-init) therefore leaves a live handler behind for the rest
   of the session, and every LATER test that runs a real chain with its own
   per-call handler is traced TWICE — it asserts one trace and gets two.
   The ``_clear_langchain_global_handler`` autouse fixture below is that
   missing teardown.
"""
from __future__ import annotations

import os
import sys
import weakref

import pytest


def pytest_configure(config):
    """Set the skip-verify env var as early as possible.

    Runs before any test module is imported, so even tests that import
    decimalai at module-level get the right behavior.
    """
    os.environ.setdefault("DECIMALAI_SKIP_VERIFY", "1")
    _check_test_environment()
    _check_openai2_environment()


# ── 5. Two test environments, and the openai 2 one runs its own slice ─────────
#
# No environment can install every framework the SDK supports: openai-agents and
# pydantic-ai need openai>=3, while crewai, litellm and llama-index-llms-openai pin
# openai<3. tests/conformance/environments.py defines the two by the extras that
# build them (TEST_ENV_EXTRAS):
#
#   openai3  `.[dev,conformance-tests]`   the whole unit suite
#   openai2  `.[conformance-tests-openai2]`  the files below, which need crewai/litellm
#
# A plain `.[dev]` install is neither: it lacks google-adk, pydantic-ai-slim and the
# OpenInference instrumentors, so the unit cases that need those skip in it (it does
# hold anthropic, which langchain-anthropic in `[dev]` requires, so the Anthropic
# cases run there), and the CrewAI and LiteLLM cases skip in any openai 3 environment. Setting DECIMAL_TEST_ENV is the claim that
# this IS one of the two, and the session refuses to start when the claim is false
# (_check_test_environment) — so CI, the publish gate and `bin/wt land`, which all
# set it, cannot pass by skipping, and a venv left below the SDK's floors says so
# instead of failing on tests unrelated to the change being gated. Unset, a run is
# leniently "openai3": a contributor's `.[dev]` venv keeps its skips.
#
#     DECIMAL_TEST_ENV=openai2 .venv-openai2/bin/python -m pytest tests/ -q
#
# collects ONLY the files below (the rest of the unit suite imports openai 3
# frameworks that environment cannot hold). bin/wt land runs it whenever
# `.venv-openai2` exists (the landing gate's suite definition). tests/conformance
# is never filtered here: it reads DECIMAL_TEST_ENV itself.
OPENAI2_UNIT_TESTS = (
    "test_crewai_*.py",          # crewai + its OpenInference instrumentor
    "unit/test_llm_evals.py",    # litellm, through decimalai.evals
    "unit/test_prebuilt_evals.py",
    "test_llamaindex_*.py",      # llama-index-core beside llama-index-llms-openai
    "test_version_floors_phase1.py",  # the extras admit what THIS environment runs
    # The native skill loaders: CrewAI and Strands are graded here, the other six in
    # openai3 (tests/conformance/environments.py:LOADER_ENVS).
    "native_loaders/test_*.py",
    # Native mode (Phase 3): the CrewAI and Strands hooks and their native-column cells
    # run here; the other six frameworks' cells skip as graded in openai3.
    "native/test_*.py",
)

#: Importable in the openai 2 environment, or the slice above would pass by
#: skipping the very cases it exists to run. Found, not imported: find_spec loads
#: only a dotted name's parent packages, so crewai itself is not imported at startup.
OPENAI2_REQUIRED_MODULES = (
    "crewai",
    "openinference.instrumentation.crewai",
    "litellm",
    "llama_index.core",
    "llama_index.llms.openai",
    "strands",                    # the Strands native loader (tests/native_loaders)
)

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _test_env() -> str:
    """DECIMAL_TEST_ENV, validated against the environments the suite defines."""
    if not (os.environ.get("DECIMAL_TEST_ENV") or "").strip():
        return "openai3"  # the default; no import, so a plain run is unchanged
    from tests.conformance.environments import current_env

    try:
        return current_env()
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from None


def _check_test_environment() -> None:
    """Refuse a run that names an environment this interpreter is not.

    Every distribution the environment's extras require must be installed, at or
    above its floor. Only a NAMED run is held to it (see section 5 above).
    """
    if not (os.environ.get("DECIMAL_TEST_ENV") or "").strip():
        return
    env = _test_env()
    from tests.conformance import environments

    problems = environments.environment_problems(env)
    if not problems:
        return
    repo = os.path.dirname(_TESTS_DIR)
    also = ("Add `,live-llm-tests` to the extras if this venv also runs the live "
            "release gate." if env == "openai3" else "")
    if environments.is_venv(sys.prefix, sys.base_prefix):
        venv = os.path.realpath(sys.prefix)
        checkout = os.path.dirname(venv)
        if not os.path.isfile(os.path.join(checkout, "pyproject.toml")):
            checkout = repo
        fix = (f"Rebuild this venv ({venv}) from its extras, exactly:\n  "
               + environments.rebuild_commands(env, venv, checkout) + "\n" + also)
    else:
        # Not a venv: Homebrew's or the system's python, a conda env, a CI runner's
        # own interpreter. Its prefix is a Python installation, so the fix must
        # never name it. Build a new venv in the checkout and leave this one alone.
        target = os.path.join(repo, environments.VENV_DIRS[env])
        fix = (f"This interpreter is not a virtualenv ({sys.prefix}), so do not "
               "rebuild it in place. Build the environment as a new venv in the "
               "checkout, then run the tests with that venv's python:\n  "
               + environments.rebuild_commands(env, None, repo) + "\n"
               f"  DECIMAL_TEST_ENV={env} {target}/bin/python -m pytest tests/ -q\n"
               + also)
    raise pytest.UsageError(
        f"DECIMAL_TEST_ENV={env} names the environment built by "
        f"`{environments.TEST_ENVS[env]}`, and this interpreter ({sys.executable}) is "
        f"not it:\n  - " + "\n  - ".join(problems) + "\n\nA run that names its "
        "environment must not pass by skipping what that environment exists to run, "
        "or grade a framework below the floor the SDK supports (it carries no code "
        "for older releases, so those fail on tests unrelated to any change). "
        + fix
    )


_ENV_NOTE: list = []   # computed once per session: (note, holds another world's dists)


def _environment_note() -> tuple:
    """One line naming the environment, and another world's distributions if it holds any."""
    if not _ENV_NOTE:
        env = _test_env()
        from tests.conformance import environments

        extra = environments.strays(env)
        note = f"test environment: {env} ({environments.TEST_ENVS[env]})"
        if extra:
            note += (
                f" ALSO holds {', '.join(extra)}, which only the other environment's "
                "extras install. This run grades more than CI's and the gate's will; "
                "rebuild the venv from its extras for counts that match."
            )
        _ENV_NOTE.append((note, bool(extra)))
    return _ENV_NOTE[0]


def pytest_report_header(config):
    if not (os.environ.get("DECIMAL_TEST_ENV") or "").strip():
        return None
    return _environment_note()[0]


def pytest_terminal_summary(terminalreporter):
    """Repeat a stray-distribution note at the end, where `-q` (the gates' mode) shows it."""
    if not (os.environ.get("DECIMAL_TEST_ENV") or "").strip():
        return
    note, strays = _environment_note()
    if strays:
        terminalreporter.write_line(note, yellow=True)


def _check_openai2_environment() -> None:
    if _test_env() != "openai2":
        return
    import importlib.util

    missing = []
    for mod in OPENAI2_REQUIRED_MODULES:
        try:
            found = importlib.util.find_spec(mod) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(mod)
    if missing:
        raise pytest.UsageError(
            f"DECIMAL_TEST_ENV=openai2, but {', '.join(missing)} cannot be imported "
            f"here, so the openai 2 slice would pass by skipping. Build this "
            f'environment with: pip install -e ".[conformance-tests-openai2]"'
        )


def pytest_ignore_collect(collection_path, config):
    """Under DECIMAL_TEST_ENV=openai2, collect only OPENAI2_UNIT_TESTS (+ conformance)."""
    if _test_env() != "openai2":
        return None
    import fnmatch

    path = str(collection_path)
    rel = os.path.relpath(path, _TESTS_DIR).replace(os.sep, "/")
    if rel.startswith("..") or rel == "." or rel.split("/", 1)[0] == "conformance":
        return None
    if os.path.isdir(path):
        return None   # descend; files are decided one by one
    if not os.path.basename(rel).startswith("test_"):
        return None
    return None if any(fnmatch.fnmatch(rel, pat) for pat in OPENAI2_UNIT_TESTS) else True


@pytest.fixture(autouse=True)
def _restore_decimalai_modules():
    """Snapshot ``decimalai*`` modules in sys.modules; restore after test.

    Prevents test-isolation breakage when one test deletes
    ``sys.modules["decimalai"]`` (or ``decimalai._config``) and re-imports
    — subsequent tests with module-level ``from decimalai._config import X``
    bindings would otherwise read stale globals from an orphaned module
    instance. See module docstring for the failure mode this catches.
    """
    snapshot = {
        name: mod
        for name, mod in sys.modules.items()
        if name == "decimalai" or name.startswith("decimalai.")
    }
    try:
        yield
    finally:
        # Drop any modules the test added that weren't in the snapshot
        # (e.g. test imported decimalai.langchain for the first time —
        # safe to keep, but consistency matters more than cache).
        current = [
            name
            for name in list(sys.modules)
            if name == "decimalai" or name.startswith("decimalai.")
        ]
        for name in current:
            if name not in snapshot:
                # Newly imported during the test — leave it alone; not
                # an isolation hazard, and re-importing it costs cycles.
                continue
        # Restore the original module instances so module-level bindings
        # in OTHER test files keep pointing at live globals.
        for name, mod in snapshot.items():
            sys.modules[name] = mod


@pytest.fixture(autouse=True)
def _restore_sdk_config():
    """Put the SDK's global config back after each test.

    ``decimalai.init()`` writes process-global state (``_config._config``,
    ``_config._client``, ``decimalai._global_client``) and many tests call it
    without undoing it. A test that ran ``init(enabled=False)`` used to leave
    the whole session DISABLED: harmless while only the trace senders read the
    flag, but every adapter's skill router now honours the kill switch too, so
    a leaked disabled config switched skills off in whichever test ran next.
    """
    cfg = sys.modules.get("decimalai._config")
    pkg = sys.modules.get("decimalai")
    saved = (
        getattr(cfg, "_config", None), getattr(cfg, "_client", None),
        getattr(pkg, "_global_client", None),
    )
    try:
        yield
    finally:
        cfg = sys.modules.get("decimalai._config") if cfg is None else cfg
        if cfg is not None:
            cfg._config, cfg._client = saved[0], saved[1]
        if pkg is not None:
            pkg._global_client = saved[2]


@pytest.fixture(autouse=True)
def _clear_langchain_global_handler():
    """Drop the process-global LangChain handler an install leaves behind.

    See concern 3 in the module docstring for the double-tracing failure this
    prevents. Only reaches for the adapter when a test already imported it, so
    it never drags langchain-core into a run that didn't ask for it.
    """
    try:
        yield
    finally:
        lc_mod = sys.modules.get("decimalai.langchain")
        if lc_mod is not None:
            lc_mod._decimal_callback_var.set(None)


# ── 4. Span exports are drained between tests ──────────────────────────────────
#
# The SDK exports through a `BatchSpanProcessor`, whose whole job is to hand spans
# to a worker thread and return. Nothing tied that worker to a test boundary, so a
# span opened in one test could be exported while the NEXT test was running — and
# land in that test's mock.
#
# That is not hypothetical. Three intermittent CI failures on 2026-08-22..23 had
# this shape, each in a different file, each passing on rerun and unreproducible
# locally:
#
#   test_skill_rail_per_run.py    TestAnthropicClaimsOnlyWhatItInjected
#   test_pydantic_ai_run_scope.py TestRunScopeInstall
#   test_manifest_only_mode.py    assert captured["agent_name"] == "support-agent"
#                                 AssertionError: 'doomed' == 'support-agent'
#
# `doomed` is the agent name from test_llamaindex_run_identity.py:431 — a different
# FILE. Demonstrated directly: three spans opened in one function were exported by
# the batch worker while the next function was running, every time.
#
# Note what this is NOT. `_synchronous_sender` below already makes the *send*
# inline, and `agent_run` already resets its ContextVar in a `finally` — both were
# checked and both are correct. The gap is one layer up: the send is only reached
# when the batch processor decides to export, and that decision was on its own
# timer. Fixing the send half and leaving the export half async is why the flake
# survived a fixture that looks like it should have caught it.
#
# Draining after every test closes it: a test's spans are exported before the next
# one starts, so nothing can arrive late. `force_flush` is exactly this operation.
# Providers built BY tests are covered too — six test files construct their own
# `TracerProvider`, three of them the files that flaked — so `__init__` is wrapped
# to record every instance weakly rather than relying on a hand-maintained list a
# new test would silently fall out of.
_LIVE_PROVIDERS: "weakref.WeakSet" = weakref.WeakSet()


def pytest_sessionstart(session):
    """Record every TracerProvider built during the session, weakly."""
    try:
        from opentelemetry.sdk.trace import TracerProvider
    except ImportError:  # pragma: no cover - OTel is a core dep
        return
    if getattr(TracerProvider, "_decimalai_tracked", False):
        return
    original_init = TracerProvider.__init__

    def _tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            _LIVE_PROVIDERS.add(self)
        except TypeError:  # pragma: no cover - unhashable provider
            pass

    TracerProvider.__init__ = _tracking_init
    TracerProvider._decimalai_tracked = True


@pytest.fixture(autouse=True)
def _drain_span_exports():
    """Flush every live TracerProvider after each test, so no export lands late."""
    yield
    providers = list(_LIVE_PROVIDERS)
    try:
        from opentelemetry import trace as trace_api

        providers.append(trace_api.get_tracer_provider())
    except Exception:  # pragma: no cover - no OTel, nothing to drain
        pass
    try:
        from decimalai import providers as _dp

        if _dp._last_provider is not None:
            providers.append(_dp._last_provider)
    except Exception:  # pragma: no cover - SDK not importable here
        pass
    for provider in providers:
        flush = getattr(provider, "force_flush", None)
        if not callable(flush):
            continue  # ProxyTracerProvider and friends have nothing to drain
        try:
            # Short on purpose: this runs after EVERY test, and a provider that
            # cannot drain in a second is a finding, not something to wait out.
            flush(timeout_millis=1000)
        except Exception:  # noqa: BLE001 - a shut-down provider raises; not this
            pass            # fixture's business, and never a reason to fail a test


@pytest.fixture(autouse=True)
def _synchronous_sender(request, monkeypatch):
    """Run the background trace-sender INLINE for unit tests.

    The SDK ships traces on a background thread (``_config._sender.submit``); the
    handler tests then wait on it with a 5s ``flush()`` timeout before asserting
    the trace reached the (mock) client. Under heavy CPU load — e.g. right after
    the live suite has hammered the machine — that worker thread can be starved
    and miss the window, so ``ingest_trace`` hasn't been called when the assert
    runs: an intermittent "red only when the box is busy" flake. (Unit tests do
    NOT hit a real backend; the client is always mocked or HTTP-stubbed — so the
    only thing async dispatch buys here is that flake.)

    Patching the *global sender instance's* submit to run inline removes the
    thread and the timeout race, so unit tests are deterministic regardless of
    load. We patch the instance — not ``BackgroundSender`` itself — so the
    sender's own tests (which build their own ``BackgroundSender()`` to assert
    async behavior) and ``test_silent_drop_visibility`` (drives ``flush()`` /
    ``_pending`` directly) are untouched. Live/integration tests exercise the
    REAL async sender against a real backend, so they opt out.
    """
    if request.node.get_closest_marker("integration") or request.node.get_closest_marker("live_llm"):
        yield
        return

    from decimalai import _config

    sender = _config._sender

    def _sync_submit(fn, *args, **kwargs):
        # Mirror BackgroundSender.submit's bookkeeping, minus the executor: run
        # the send inline, record the outcome, and never propagate — the async
        # path records the failure inside the worker thread and the submit caller
        # never sees it, so neither should we.
        try:
            result = fn(*args, **kwargs)
            sender._record_success()
            return result
        except Exception as exc:  # noqa: BLE001 — match async: record + swallow
            sender._record_failure(exc, _config._extract_trace_id(args))
            return None

    monkeypatch.setattr(sender, "submit", _sync_submit)
    yield



@pytest.fixture(autouse=True)
def _clear_prompt_cache():
    """`load_agent()` keeps the last good config per process so it can serve it
    during a platform outage. That is process-local state, so it has to be reset
    between tests or one case's fetch answers the next case's assertion."""
    from decimalai._agent import _reset_prompt_cache

    _reset_prompt_cache()
    yield
    _reset_prompt_cache()
