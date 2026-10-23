"""The conformance suite's stub servers hold a whole concurrent burst of connections.

``openai-C9`` went red intermittently on 2026-10-23 (8 runs in 10 at a load
average of about 12, green on an idle machine). The cause was not the adapter:
the stub model service was a plain ``ThreadingHTTPServer``, whose listen backlog
is ``socketserver``'s default of five. The concurrency phase connects
``harness.CONCURRENCY`` lanes at once; when the accept thread was not scheduled in
time, the sixth connection overflowed the queue, macOS reset it, and that lane's
first model call failed (``ConnectError: [Errno 54] Connection reset by peer``).
The SDK filed an honest errored trace, and C9 read it as a lane that lost its own
prompt.

These tests run in the DEFAULT suite, like ``test_conformance_contract_*.py``. The
first reproduces the race without relying on machine load: it never runs the accept
loop at all, which is the slowest an accept thread can ever be, and connects the
burst. The second holds every stub server under ``tests/conformance`` to the class
that passes it, so a new stub cannot bring the five-slot default back. The third
keeps that class where ``scripts/pypi_smoke.py`` can still load it: the smoke
imports ``probe.py`` by path, alone, so ``probe.py`` must import nothing but the
standard library.
"""

from __future__ import annotations

import ast
import importlib.util
import socket
import sys
from contextlib import ExitStack, closing
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import List

from tests.conformance.harness import CONCURRENCY
from tests.conformance.probe import StubHTTPServer

CONFORMANCE_DIR = Path(__file__).resolve().parent / "conformance"

#: ``http.server`` / ``socketserver`` classes that listen with the five-slot default.
_STDLIB_SERVERS = frozenset({
    "HTTPServer", "ThreadingHTTPServer", "TCPServer", "ThreadingTCPServer",
    "ForkingTCPServer", "ForkingHTTPServer",
})


def test_a_concurrent_burst_waits_for_an_accept_loop_that_has_not_run() -> None:
    """Twice the concurrency phase's lanes connect while nothing accepts: none is refused.

    Every handshake has to wait in the listen queue, exactly as it does when the
    accept thread is starved. Linux drops an overflowing connect rather than
    resetting it, so the timeout turns that into a failure here too.
    """
    burst = 2 * CONCURRENCY
    server = StubHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)  # never served
    try:
        refused: List[str] = []
        with ExitStack() as held:
            for i in range(burst):
                sock = held.enter_context(closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)))
                sock.settimeout(2.0)
                try:
                    sock.connect(server.server_address)
                except OSError as exc:
                    refused.append(f"connection {i + 1}: {exc!r}")
        assert not refused, (
            f"{len(refused)} of {burst} connections failed while the accept loop was "
            f"stalled (listen backlog {server.request_queue_size}); a concurrent lane "
            f"would see its model call fail: {refused[:3]}"
        )
    finally:
        server.server_close()


def test_every_conformance_stub_server_listens_with_the_deep_backlog() -> None:
    """No module under ``tests/conformance`` constructs a stdlib server directly."""
    offenders: List[str] = []
    for path in sorted(CONFORMANCE_DIR.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name in _STDLIB_SERVERS:
                offenders.append(f"{path.relative_to(CONFORMANCE_DIR)}:{node.lineno} {name}(...)")
    assert not offenders, (
        "these stub servers listen with socketserver's backlog of 5, which a concurrent "
        "phase overflows on a busy machine; build them with "
        "tests.conformance.probe.StubHTTPServer instead: " + "; ".join(offenders)
    )


def test_the_probe_still_loads_by_path_with_its_deep_backlog() -> None:
    """``probe.py`` loaded alone, as ``scripts/pypi_smoke.py`` loads it, has the server.

    A relative import in ``probe.py`` passes every test that imports it as
    ``tests.conformance.probe`` and breaks only the release smoke.
    """
    name = "conformance_probe_by_path"
    spec = importlib.util.spec_from_file_location(name, CONFORMANCE_DIR / "probe.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # @dataclass resolves annotations through sys.modules
    try:
        spec.loader.exec_module(module)
        assert module.StubHTTPServer.request_queue_size >= 2 * CONCURRENCY
    finally:
        sys.modules.pop(name, None)
