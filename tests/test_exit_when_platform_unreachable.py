"""When DecimalAI cannot be reached, the agent's process still exits promptly.

Two things a tutorial reader saw with the platform down (0.23, 2026-10-21):

1. "Failed to register manifest" at WARNING with a full traceback, for a
   condition the SDK handles (the trace is held and retried in the
   background). One line says that.
2. The process took 31-62 s to exit after the answer had printed. The sender
   ran on a ThreadPoolExecutor, whose non-daemon worker the interpreter joins
   at shutdown after it has run EVERY queued item, and a held trace waits out
   a 31-62 s re-registration ladder. Exit now waits only while sends make
   progress, and gives up after `_EXIT_STALL_TIMEOUT_S` without one.

The answer itself never waited on any of this (the send is in the
background); `test_the_answer_does_not_wait_for_the_sender` keeps it so.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]

PRELUDE = """
import sys, time
import decimalai._config as cfg
from decimalai._config import DecimalConfig
from types import SimpleNamespace
cfg._config = DecimalConfig(api_key="dai_sk_test", base_url="http://127.0.0.1:9", enabled=True)
"""


def _run(script: str, timeout: float = 30.0) -> tuple[subprocess.CompletedProcess, float]:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(ROOT)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
    ))
    env.pop("DECIMAL_API_KEY", None)
    env.pop("DECIMALAI_API_KEY", None)
    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(PRELUDE) + textwrap.dedent(script)],
        capture_output=True, text=True, timeout=timeout, env=env, cwd=str(ROOT),
    )
    return proc, time.monotonic() - start


def test_the_exit_wait_is_a_few_seconds():
    from decimalai import _config

    assert 1.0 <= _config._EXIT_STALL_TIMEOUT_S <= 5.0


def test_a_held_trace_does_not_hold_the_process_open():
    """The platform refuses the manifest; the trace is held for the 31-62 s
    re-registration ladder. Exit waits about `_EXIT_STALL_TIMEOUT_S`, says in one
    line that the trace was not sent, and posts nothing under a local id."""
    proc, elapsed = _run("""
        class Client:
            def ingest_trace(self, trace):
                print("POSTED", flush=True)
            def close(self):
                pass
        trace = SimpleNamespace(id="trace-held", agent_name="order-support", manifest_id=None)
        cfg.submit_trace_pending_manifest(Client(), trace, reregister=lambda: None)
        print("ANSWER", flush=True)
    """)
    assert proc.returncode == 0, proc.stderr
    assert "ANSWER" in proc.stdout and "POSTED" not in proc.stdout
    assert elapsed < 15, f"exit took {elapsed:.1f}s"
    gave_up = [line for line in proc.stderr.splitlines() if "not sent" in line]
    assert len(gave_up) == 1, proc.stderr
    assert "1 trace" in gave_up[0]
    assert "Traceback" not in proc.stderr, proc.stderr


def test_a_stalled_send_releases_exit_after_the_stall_timeout():
    """A platform that accepts the connection and never answers."""
    proc, elapsed = _run("""
        cfg._EXIT_STALL_TIMEOUT_S = 1.0
        cfg._sender.submit(time.sleep, 60)
        cfg._sender.submit(print, "SECOND", flush=True)
        print("ANSWER", flush=True)
    """)
    assert proc.returncode == 0, proc.stderr
    assert "SECOND" not in proc.stdout
    assert elapsed < 10, f"exit took {elapsed:.1f}s"
    assert "2 trace" in proc.stderr, proc.stderr


def test_a_healthy_backlog_is_all_sent_at_exit():
    """The cap is on a send that makes no progress, not on the whole drain: a
    script that queued many traces still delivers every one of them."""
    proc, _ = _run("""
        cfg._EXIT_STALL_TIMEOUT_S = 1.0
        def send(i):
            time.sleep(0.25)
            print("SENT", i, flush=True)
        for i in range(10):          # 2.5 s of sends, against a 1 s stall timeout
            cfg._sender.submit(send, i)
        print("ANSWER", flush=True)
    """)
    assert proc.returncode == 0, proc.stderr
    assert [line for line in proc.stdout.splitlines() if line.startswith("SENT")] == [
        f"SENT {i}" for i in range(10)
    ]
    assert "not sent" not in proc.stderr


def test_the_answer_does_not_wait_for_the_sender():
    from decimalai._config import BackgroundSender

    sender = BackgroundSender()
    started = time.monotonic()
    sender.submit(time.sleep, 0.5)
    assert time.monotonic() - started < 0.1
    sender.flush(timeout=5)


def test_a_failed_manifest_registration_is_one_warning_line(monkeypatch, caplog):
    import decimalai._config as cfg
    import decimalai.openai_agents as oa
    from decimalai._config import DecimalConfig
    from decimalai.schema.manifest import extract_from_config

    cfg._config = DecimalConfig(api_key="dai_sk_test", base_url="http://127.0.0.1:9", enabled=True)
    client = MagicMock()
    client.register_manifest.side_effect = ConnectionError("[Errno 61] Connection refused")
    monkeypatch.setattr(cfg, "_client", client)
    monkeypatch.setattr(oa, "_manifest_ids", {})
    monkeypatch.setattr(oa, "_manifest_hashes", {})
    monkeypatch.setattr(oa, "_pending_manifests", {})

    snapshot = extract_from_config(agent_name="order-support", tools=[{"name": "lookup_order"}])
    with caplog.at_level(logging.DEBUG, logger="decimalai"):
        assert oa._register_snapshot("order-support", snapshot) is None

    loud = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(loud) == 1, [r.getMessage() for r in loud]
    (record,) = loud
    assert record.exc_info is None
    message = record.getMessage()
    assert "\n" not in message
    assert "order-support" in message and "Connection refused" in message
    assert "background" in message  # says what happens to the trace
    # The traceback is still there for whoever turns on DEBUG.
    assert any(r.exc_info for r in caplog.records if r.levelno == logging.DEBUG)
    assert "order-support" in oa._pending_manifests


def test_a_send_that_raises_its_own_timeout_is_a_failure_not_a_stall():
    """`concurrent.futures.TimeoutError` is the builtin `TimeoutError` on 3.11+,
    and a socket timeout raises it from inside the send. That send failed; the
    work queued behind it is still sent."""
    from decimalai._config import BackgroundSender

    sender = BackgroundSender()
    done = []

    def times_out():
        raise TimeoutError("timed out")

    sender.submit(times_out)
    sender.submit(lambda: done.append("next"))
    assert sender.flush(5.0, give_up_on_stall=True) == 0
    assert done == ["next"]
    assert sender.status().failed == 1


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_forked_child_still_sends():
    """A child made by os.fork() inherits the sender but not its thread."""
    proc, _ = _run("""
        import os
        cfg._sender.submit(print, "PARENT", flush=True)
        cfg._sender.flush(5)
        pid = os.fork()
        if pid == 0:
            cfg._sender.submit(print, "CHILD", flush=True)
            cfg._sender.flush(5)
            os._exit(0)
        os.waitpid(pid, 0)
    """)
    assert proc.returncode == 0, proc.stderr
    assert "PARENT" in proc.stdout and "CHILD" in proc.stdout, proc.stdout
