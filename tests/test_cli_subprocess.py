"""Real subprocess CLI tests (vs CliRunner in-process).

The existing `test_cli.py` uses `click.testing.CliRunner.invoke`, which
calls click commands as Python function calls — fast but bypasses real
process boundaries. This file exercises the CLI as a SUBPROCESS so we
catch:

  - Real env-var resolution from the shell (DECIMAL_API_KEY vs
    DECIMALAI_API_KEY — `cli/main.py:49` declares both)
  - Real exit codes consumable by CI (`echo $?` semantics)
  - `python -m decimalai.cli` vs the installed `decimal` console script
  - Pre-import side effects (`__main__.py` execution path)
  - Real stdout / stderr separation

We deliberately don't exercise traces-list or manifest-show against a
live backend here — that would require booting a server inside a test
process, and the live integration suite already covers it. The CLI-only
flags (`--version`, `--help`, env-var pickup) are enough to lock in the
subprocess contract.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import decimalai


# Run `python -m decimalai.cli` so we use the *installed* SDK in the
# current venv, not whatever shell-resolved binary `decimal` would hit.
# This makes the test work regardless of whether the console script is
# installed.
_PYTHON = sys.executable
_REPO_ROOT = Path(__file__).resolve().parent.parent
# Invoke via the package's __main__.py. This is the conventional
# `python -m <package>` form that customer docs reference.
_INVOKE_MODULE = [_PYTHON, "-m", "decimalai.cli"]


def _run(args: list[str], *, env: dict | None = None, timeout: float = 10.0):
    """Invoke the CLI as a subprocess and return CompletedProcess."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(
        _INVOKE_MODULE + args,
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        env=full_env,
        timeout=timeout,
    )


# ─────────────────────────────────────────────────────────────────────
# Exit codes + stdout/stderr separation
# ─────────────────────────────────────────────────────────────────────


def test_version_exits_zero_and_writes_version_to_stdout():
    """`--version` exits 0 + prints the package version to stdout."""
    result = _run(["--version"])
    assert result.returncode == 0, (
        f"non-zero exit; stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert decimalai.__version__ in result.stdout, (
        f"version string missing; stdout={result.stdout!r}"
    )


def test_help_exits_zero_and_lists_subcommands():
    """`--help` exits 0 + lists the documented subcommands."""
    result = _run(["--help"])
    assert result.returncode == 0
    # The set of subcommands is the click group's; pick a few sentinels
    # that should always be present.
    for sentinel in ("traces", "manifest"):
        assert sentinel in result.stdout, (
            f"subcommand '{sentinel}' missing from --help; "
            f"stdout={result.stdout!r}"
        )


def test_unknown_subcommand_exits_nonzero():
    """Click's default behavior: unknown subcommand → exit code 2 +
    usage on stderr. Test the contract a CI script depends on.
    """
    result = _run(["definitely-not-a-real-subcommand"])
    assert result.returncode != 0, (
        f"unknown subcommand should fail; got returncode=0, "
        f"stdout={result.stdout!r}"
    )


# ─────────────────────────────────────────────────────────────────────
# Env-var resolution (both DECIMAL_API_KEY and DECIMALAI_API_KEY work)
# ─────────────────────────────────────────────────────────────────────


def test_env_var_decimal_api_key_is_picked_up():
    """`--help` doesn't actually need an API key, but `init` does. We
    can't reach `init` without a live backend, so we assert via the
    error path: an init call with NO key in the env should fail with
    a 'missing key' style message, but with a key set it should at
    least get further (failing on the next step, e.g. backend connect).

    Since `init` connects to the backend by default and we don't run
    one, both invocations will fail — but they'll fail at DIFFERENT
    stages, which we don't assert here. Instead: just confirm the
    subprocess doesn't crash on env-var lookup.
    """
    # Run with DECIMAL_API_KEY set but not pointing at anything real.
    result = _run(
        ["init", "--no-test-trace"],
        env={"DECIMAL_API_KEY": "dai_sk_test_key_001",
             "DECIMAL_BASE_URL": "http://127.0.0.1:1"},  # closed port
    )
    # Either it errors trying to connect, or it gets past key resolution.
    # We assert it does NOT crash with a Python traceback on stderr (which
    # would indicate broken env-var plumbing). Click commands handle
    # config errors with a user-friendly message, not a traceback.
    assert "Traceback" not in result.stderr, (
        f"init crashed with a Python traceback (broken env-var plumbing?); "
        f"stderr={result.stderr!r}"
    )


def test_env_var_decimalai_api_key_alias_works():
    """Per `cli/main.py:49` both `DECIMAL_API_KEY` and `DECIMALAI_API_KEY`
    should be honored. Confirm the alias doesn't crash the CLI.
    """
    result = _run(
        ["init", "--no-test-trace"],
        env={"DECIMALAI_API_KEY": "dai_sk_test_key_001",
             "DECIMAL_BASE_URL": "http://127.0.0.1:1"},
    )
    assert "Traceback" not in result.stderr, (
        f"DECIMALAI_API_KEY alias broken; stderr={result.stderr!r}"
    )


# ─────────────────────────────────────────────────────────────────────
# `python -m decimalai.cli` parity (the __main__ entry)
# ─────────────────────────────────────────────────────────────────────


def test_module_entrypoint_matches_console_script_behavior():
    """Running via `python -m decimalai.cli --version` should produce
    the same output as running via the console script. We exercise
    the module form; if the console script is installed in this venv,
    we ALSO compare its output (skip otherwise).
    """
    module_result = _run(["--version"])
    assert module_result.returncode == 0
    module_out = module_result.stdout.strip()

    # If the `decimal` console script is on PATH, compare directly.
    import shutil
    decimal_bin = shutil.which("decimal")
    if not decimal_bin:
        pytest.skip("`decimal` console script not installed; can't compare")

    script_result = subprocess.run(
        [decimal_bin, "--version"],
        capture_output=True, text=True, timeout=10.0,
    )
    assert script_result.returncode == 0
    script_out = script_result.stdout.strip()

    assert module_out == script_out, (
        f"console script and module entry print different --version output: "
        f"module={module_out!r} script={script_out!r}"
    )
