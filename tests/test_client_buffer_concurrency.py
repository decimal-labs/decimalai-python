"""`DecimalAIClient.buffer_trace` / `flush` under concurrent writers.

Two defects lived here, and both are data loss rather than a crash — which is
why nothing caught them:

  1. NO LOCK. `_trace_buffer` was a plain list. Two threads crossing
     `_AUTO_FLUSH_THRESHOLD` together each called `flush()` on the SAME list, so
     one batch went over the wire more than once.
  2. CLEAR-AFTER-SEND. `flush()` posted `self._trace_buffer` and cleared it
     afterwards, so every trace appended during the in-flight request was
     destroyed unsent.

Measured before the fix over 8 threads x 20 traces: ~15% of traces never
reached the transport and the ones that did were written ~5x.

WHICH HALF IS PROVEN. Reverting `flush()` to clear-after-send turns six of the
tests below red, so the SWAP is what carries the no-loss / no-duplication
properties. Removing `_buffer_lock` from `buffer_trace` turns NOTHING red — the
only thing it uniquely guards is the `del` / `+=` pair in the overflow trim, and
CPython's GIL would not split that even under 64 threads at a 1-microsecond
switch interval. The lock is kept because the pair is unsafe by the language's
rules rather than by CPython's accident, and a free-threaded build removes the
accident — but it is defensive, and no test here should be read as proving it.

No shipped adapter calls `buffer_trace` — they all use `_config._sender.submit`
— so this is a LATENT defect. It is still a defect: the class documents
`buffer_trace` as a batching entry point, and a documented API that silently
loses data under concurrency is broken whether or not we are its only caller.

The transport is stubbed at `ingest_traces_batch`, the one method that talks to
the network. Everything above it — the lock, the swap, the threshold, the
cooldown, the cap — is the real code.
"""

from __future__ import annotations

import threading
import time
from typing import Any, List
from unittest.mock import patch
from uuid import uuid4

import httpx
import pytest

from decimalai import _client as client_mod
from decimalai._client import DecimalAIClient, DecimalAPIError
from decimalai.schema.trace import RunTrace

THRESHOLD = client_mod._AUTO_FLUSH_THRESHOLD


def _client() -> DecimalAIClient:
    return DecimalAIClient(api_key="dai_sk_test", base_url="http://localhost:8000")


def _trace(tag: str) -> RunTrace:
    return RunTrace(id=uuid4(), agent_name=tag, status="success")


def _api_error(status: int) -> DecimalAPIError:
    """A REAL DecimalAPIError, built the way `_raise_for_status` builds one —
    so `status_code` comes from the response rather than from a hand-set
    attribute that could drift from the class."""
    request = httpx.Request("POST", "http://localhost:8000/api/v1/traces/batch")
    response = httpx.Response(status, json={"detail": "nope"}, request=request)
    return DecimalAPIError(response)


class _Recorder:
    """Stands in for the HTTP batch call, and records what it was handed.

    ``delay`` widens the in-flight window so a concurrent ``buffer_trace``
    lands during the request rather than by luck — the exact interval the
    clear-after-send bug destroyed.
    """

    def __init__(self, delay: float = 0.0, raises: BaseException | None = None):
        self.delay = delay
        self.raises = raises
        self.batches: List[List[RunTrace]] = []
        self.lock = threading.Lock()
        self.in_flight = threading.Event()

    def __call__(self, traces: List[RunTrace]) -> Any:
        # Copy: the whole question is whether the caller mutates the list we
        # were handed, and holding a reference would hide exactly that.
        with self.lock:
            self.batches.append(list(traces))
        self.in_flight.set()
        if self.delay:
            time.sleep(self.delay)
        if self.raises:
            raise self.raises
        return {"ingested": len(traces)}

    @property
    def sent(self) -> List[str]:
        return [t.agent_name for b in self.batches for t in b]


class TestNoTraceIsLostOrDuplicated:
    def test_concurrent_writers_send_every_trace_exactly_once(self):
        """The end-to-end property, and the one that was measurably false.

        8 threads x 20 traces. Every trace must reach the transport, and none
        may reach it twice. Both halves are needed: the swap alone would still
        allow a double send, and the lock alone would still lose the traces
        appended during a request.
        """
        c = _client()
        rec = _Recorder(delay=0.002)
        threads, per_thread = 8, 20
        expected = {f"t{i}-{j}" for i in range(threads) for j in range(per_thread)}

        with patch.object(c, "ingest_traces_batch", rec):
            def _writer(i: int) -> None:
                for j in range(per_thread):
                    c.buffer_trace(_trace(f"t{i}-{j}"))

            ts = [threading.Thread(target=_writer, args=(i,)) for i in range(threads)]
            for t in ts:
                t.start()
            for t in ts:
                t.join(30)
            c.flush()   # drain whatever sits below the threshold

        sent = rec.sent
        assert set(sent) == expected, (
            f"{len(expected - set(sent))} trace(s) never reached the transport"
        )
        dupes = {n for n in sent if sent.count(n) > 1}
        assert not dupes, (
            f"{len(dupes)} trace(s) were sent more than once — concurrent "
            f"flushes posted the same list: {sorted(dupes)[:5]}"
        )
        assert len(sent) == len(expected)

    def test_a_trace_buffered_during_the_send_survives(self):
        """The clear-after-send half, isolated and deterministic.

        One flush is held open while another thread buffers a trace. The old
        code cleared the buffer after the post returned, taking the newcomer
        with it; the swap means the newcomer was never in the posted list.
        """
        c = _client()
        rec = _Recorder(delay=0.25)
        c._trace_buffer = [_trace("old")]

        with patch.object(c, "ingest_traces_batch", rec):
            flusher = threading.Thread(target=c.flush)
            flusher.start()
            assert rec.in_flight.wait(timeout=5), "the send never started"
            c.buffer_trace(_trace("arrived-mid-flight"))
            flusher.join(10)

            assert rec.sent == ["old"]
            assert [t.agent_name for t in c._trace_buffer] == ["arrived-mid-flight"], (
                "the trace buffered during the in-flight request was destroyed "
                "by the post-send clear"
            )
            c.flush()

        assert rec.sent == ["old", "arrived-mid-flight"]

    def test_two_flushes_at_once_post_one_batch(self):
        """The double-send half, isolated. The second caller must find an empty
        buffer and return, not post the same list again."""
        c = _client()
        rec = _Recorder(delay=0.25)
        c._trace_buffer = [_trace("a"), _trace("b")]

        with patch.object(c, "ingest_traces_batch", rec):
            ts = [threading.Thread(target=c.flush) for _ in range(4)]
            for t in ts:
                t.start()
            for t in ts:
                t.join(10)

        assert len(rec.batches) == 1, (
            f"the same batch was posted {len(rec.batches)} times"
        )
        assert rec.sent == ["a", "b"]


class TestFailurePathsStillHoldTheirPromises:
    def test_a_5xx_preserves_the_batch_for_a_later_flush(self):
        c = _client()
        rec = _Recorder(raises=_api_error(503))
        c._trace_buffer = [_trace("a")]

        with patch.object(c, "ingest_traces_batch", rec):
            c.flush()

        assert [t.agent_name for t in c._trace_buffer] == ["a"], (
            "a 5xx must put the batch back — it is not a verdict on the payload"
        )
        assert c._flush_cooldown_until > time.monotonic()

    def test_a_preserved_batch_goes_back_in_front_of_newer_traces(self):
        """Order matters: `_preserve_buffer_after_failed_flush` trims to the cap
        by dropping the OLDEST, so a batch put back at the wrong end would make
        the SDK discard the wrong traces during an outage."""
        c = _client()
        rec = _Recorder(delay=0.25, raises=httpx.ConnectError("refused"))
        c._trace_buffer = [_trace("older")]

        with patch.object(c, "ingest_traces_batch", rec):
            flusher = threading.Thread(target=c.flush)
            flusher.start()
            assert rec.in_flight.wait(timeout=5)
            c.buffer_trace(_trace("newer"))
            flusher.join(10)

        assert [t.agent_name for t in c._trace_buffer] == ["older", "newer"]

    def test_a_4xx_drops_only_the_bytes_the_server_judged(self):
        """A terminal rejection is a verdict on the POSTED batch. Traces that
        arrived during that request were never shown to the server, and the
        pre-swap code destroyed them anyway."""
        c = _client()
        rec = _Recorder(delay=0.25, raises=_api_error(422))
        c._trace_buffer = [_trace("rejected")]

        with patch.object(c, "ingest_traces_batch", rec):
            flusher = threading.Thread(target=c.flush)
            flusher.start()
            assert rec.in_flight.wait(timeout=5)
            c.buffer_trace(_trace("never-shown-to-the-server"))
            flusher.join(10)

        assert [t.agent_name for t in c._trace_buffer] == [
            "never-shown-to-the-server"
        ], "an untried trace was dropped by another batch's terminal failure"


class TestTheBufferStaysBounded:
    def test_the_cap_holds_while_the_cooldown_suppresses_the_flush(self):
        """The cap is the promise that an outage cannot grow the buffer without
        limit. Moving the trim under the lock must not weaken it."""
        c = _client()
        c._trace_buffer = [_trace(f"seed-{i}") for i in range(THRESHOLD)]
        c._flush_cooldown_until = time.monotonic() + 60

        for i in range(25):
            c.buffer_trace(_trace(f"extra-{i}"))

        assert len(c._trace_buffer) == THRESHOLD
        assert c._dropped_while_buffer_full == 25
        # Newest kept, oldest dropped.
        assert c._trace_buffer[-1].agent_name == "extra-24"

    def test_concurrent_overflow_holds_the_cap_and_the_count(self):
        """The cap and the drop count agree after concurrent overflow.

        HONEST SCOPE, because this one is easy to over-claim: removing
        `_buffer_lock` from `buffer_trace` does NOT make this test fail. It was
        tried — 64 threads, `sys.setswitchinterval(1e-6)`, 60 trials — and
        CPython's GIL never split the `del` / `+=` pair. So this asserts an
        INVARIANT (the cap holds, nothing is over-counted), not the lock.

        The lock stays anyway: the read-modify-write it guards is genuinely
        unsafe by the language's rules rather than by CPython's accident, and a
        free-threaded build removes the accident. What IS mutation-proven is the
        other half — the swap in `flush()`; see this module's docstring.
        """
        c = _client()
        c._trace_buffer = [_trace(f"seed-{i}") for i in range(THRESHOLD)]
        c._flush_cooldown_until = time.monotonic() + 60

        ts = [
            threading.Thread(target=c.buffer_trace, args=(_trace(f"x{i}"),))
            for i in range(16)
        ]
        for t in ts:
            t.start()
        for t in ts:
            t.join(10)

        assert len(c._trace_buffer) == THRESHOLD
        assert c._dropped_while_buffer_full == 16


class TestDeadFlushConfigIsGone:
    """`_flush_interval_seconds` was documented-looking config that turned
    nothing: no code read it, and there is no periodic flush for it to control.
    `BackgroundSender.submit` dispatches each trace to the executor
    immediately, so the send path holds no queue a timer could drain.

    Pinned as a test because the failure mode is a reader, not a runtime: a
    field named like a knob invites someone to reason about trace latency from
    a number that means nothing.
    """

    def test_no_dead_flush_interval_field(self):
        from decimalai._config import DecimalConfig

        names = {f.name for f in DecimalConfig.__dataclass_fields__.values()}
        assert "_flush_interval_seconds" not in names
        assert "_max_batch_size" not in names

    def test_nothing_in_the_package_references_them(self):
        import pathlib

        root = pathlib.Path(client_mod.__file__).parent
        hits = [
            str(p)
            for p in root.rglob("*.py")
            if "_flush_interval_seconds" in p.read_text()
            or "_max_batch_size" in p.read_text()
        ]
        # _config.py keeps a comment explaining the deletion; nothing else may
        # mention them at all.
        assert [h for h in hits if not h.endswith("_config.py")] == [], hits

    def test_submit_dispatches_immediately(self):
        """The claim the deletion rests on: there is no queue to drain. If a
        future change starts batching in the sender, this goes red and the
        deletion has to be revisited rather than silently outlived."""
        from decimalai._config import BackgroundSender

        sender = BackgroundSender()
        started = threading.Event()
        try:
            sender.submit(started.set)
            assert started.wait(timeout=5), (
                "submit() did not start the work — the sender now holds a queue, "
                "and a time-based flush may genuinely be needed"
            )
        finally:
            sender.flush(timeout=5)


@pytest.fixture(autouse=True)
def _quiet(caplog):
    """These tests deliberately drive failure paths, which log at WARNING."""
    caplog.set_level("CRITICAL")
    yield
