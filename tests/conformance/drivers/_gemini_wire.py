"""A real HTTP server speaking the Gemini API's ``generateContent`` — the hermetic model.

The google-genai SDK's twin of ``_openai_wire.OpenAIWire``: the real ``google.genai``
client builds its real request, opens a real socket (``HttpOptions(base_url=...)``), and
the OpenInference instrumentor sees exactly what it would in production. Only the
inference behind the endpoint is a stub.

Lanes are keyed by ``ctx.prompt_sentinel``, longest match first, and the turn to answer is
derived from the conversation the caller sent (the number of ``functionResponse`` parts in
it), never from server state, so one server answers N concurrent lanes.

Everything it says comes from the shared ``stub_script`` / ``StubTurn``. NO ASSERTIONS —
same rule as a driver.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..probe import StubHTTPServer
from . import Ctx, StubTurn, stub_script

#: google-genai requires *a* key to build a client. Nothing on the far end reads it.
STUB_API_KEY = "stub-" + "gemini" * 2

_MODEL_IN_PATH = re.compile(r"/models/([^/:]+):generateContent")


@dataclass
class _Lane:
    ctx: Ctx
    script: Tuple[StubTurn, ...]
    fail: bool


class GeminiWire:
    """A local server that answers ``POST /v1beta/models/{model}:generateContent``."""

    def __init__(self) -> None:
        self._lanes: Dict[str, _Lane] = {}
        self._lock = threading.RLock()
        self._server: Optional[StubHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "GeminiWire":
        wire = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # noqa: A003 - silence stderr
                pass

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                status, payload = wire.answer(self.path, raw)
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = StubHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("conformance Gemini wire stub was not started")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/"

    def client(self) -> Any:
        """A ``google.genai.Client`` pointed here."""
        from google import genai
        from google.genai import types

        return genai.Client(api_key=STUB_API_KEY, http_options=types.HttpOptions(base_url=self.base_url))

    def register(self, ctx: Ctx, *, script: Optional[Sequence[StubTurn]] = None,
                 fail: bool = False) -> None:
        """Teach the server one lane. Default script is the shared one."""
        turns = tuple(script) if script is not None else tuple(stub_script(ctx))
        with self._lock:
            self._lanes[ctx.prompt_sentinel] = _Lane(ctx=ctx, script=turns, fail=fail)

    def answer(self, path: str, raw: bytes) -> Tuple[int, Any]:
        text = raw.decode("utf-8", "replace")
        try:
            body = json.loads(text) if text else {}
        except ValueError:
            return 400, _error("conformance stub: request body was not JSON")
        lane = self._lane_for(text)
        if lane is None:
            return 400, _error("conformance stub: no registered lane appears in this request")
        if lane.fail:
            return 400, _error("conformance stub: this lane is scripted to fail on purpose")
        match = _MODEL_IN_PATH.search(path)
        if match is None:
            return 404, _error(f"conformance stub has no route for POST {path}")
        turn = lane.script[min(_turns_taken(body), len(lane.script) - 1)]
        return 200, _payload(match.group(1), turn)

    def _lane_for(self, text: str) -> Optional[_Lane]:
        with self._lock:
            hits = [lane for sentinel, lane in self._lanes.items() if sentinel in text]
        return max(hits, key=lambda lane: len(lane.ctx.prompt_sentinel)) if hits else None


def _error(message: str) -> Dict[str, Any]:
    """A Gemini-shaped error. 400 is deliberate: the SDK does not retry it."""
    return {"error": {"code": 400, "message": message, "status": "INVALID_ARGUMENT"}}


def _turns_taken(body: Any) -> int:
    """How many tool turns the conversation has had: its ``functionResponse`` parts."""
    count = 0
    for content in (body or {}).get("contents") or []:
        for part in (content or {}).get("parts") or []:
            if isinstance(part, dict) and "functionResponse" in part:
                count += 1
    return count


def _payload(model: str, turn: StubTurn) -> Dict[str, Any]:
    """One ``GenerateContentResponse``: a function call for a tool turn, else the text."""
    if turn.tool_call:
        name, args = turn.tool_call
        parts: List[Dict[str, Any]] = [{"functionCall": {
            "name": name, "args": args, "id": "fc_" + uuid.uuid4().hex[:12]}}]
    else:
        parts = [{"text": turn.content}]
    return {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": {
            "promptTokenCount": turn.input_tokens,
            "candidatesTokenCount": turn.output_tokens,
            "totalTokenCount": turn.input_tokens + turn.output_tokens,
        },
        "modelVersion": model,
        "responseId": "resp-" + uuid.uuid4().hex[:12],
    }
