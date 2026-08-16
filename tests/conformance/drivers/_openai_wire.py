"""A real HTTP server speaking the OpenAI wire format — the hermetic model.

Shared by every driver whose framework reaches its model through the ``openai``
SDK: the OpenAI Agents SDK (Responses API) and Pydantic AI (Chat Completions
API). It is the *model* half of the hermetic tier, exactly as ``probe.py`` is
the *ingest* half.

This is a stub **model**, not a mock **transport**. The framework builds its
real model class, the real ``openai`` client opens a real socket, and the real
span/instrumentation machinery runs — the only thing that is not real is the
inference behind the endpoint. That distinction is load-bearing here: an
openai-agents ``response`` span and every OpenInference ``llm.*`` attribute are
produced *by the provider SDK*, so a monkeypatched client would delete precisely
the code the adapters read, and the run would prove nothing.

Lanes are keyed by ``ctx.prompt_sentinel``. Which turn to answer is derived from
the conversation the caller sent — the number of tool results already in it —
never from server-side state, so one server answers eight concurrent lanes with
no cursor to corrupt.

Everything it says comes from the shared ``stub_script`` / ``StubTurn``: same
tool call, same completion text, same token counts as every other framework's
stub. NO ASSERTIONS — same rule as a driver.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import Ctx, StubTurn, stub_script

#: The ``openai`` client requires *a* key. This is not one — nothing on the far
#: end reads it, and the hermetic tier must run with no provider key at all.
# Not a credential: the stub server ignores it. Kept token-shaped only because
# the openai client refuses to construct without one.
STUB_API_KEY = "sk-" + "stub" * 3


@dataclass
class _Lane:
    """One registered run: whose sentinels, what to say, whether to fail."""

    ctx: Ctx
    script: Tuple[StubTurn, ...]
    fail: bool


class OpenAIWire:
    """A local server that answers the two OpenAI endpoints frameworks call."""

    def __init__(self) -> None:
        self._lanes: Dict[str, _Lane] = {}
        self._lock = threading.RLock()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ────────────────────────────────────────────

    def start(self) -> "OpenAIWire":
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

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
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
            raise RuntimeError("conformance OpenAI wire stub was not started")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def async_client(self) -> Any:
        """An ``AsyncOpenAI`` pointed here, with retries off.

        ``max_retries=0`` keeps the error phase honest: a retried 400 would
        turn one intended failure into three requests and blur what the
        adapter recorded.
        """
        from openai import AsyncOpenAI

        return AsyncOpenAI(base_url=self.base_url, api_key=STUB_API_KEY, max_retries=0)

    # ── registration ─────────────────────────────────────────

    def register(
        self,
        ctx: Ctx,
        *,
        script: Optional[Sequence[StubTurn]] = None,
        fail: bool = False,
    ) -> None:
        """Teach the server one lane. Default script is the shared one."""
        turns = tuple(script) if script is not None else tuple(stub_script(ctx))
        with self._lock:
            self._lanes[ctx.prompt_sentinel] = _Lane(ctx=ctx, script=turns, fail=fail)

    # ── answering ────────────────────────────────────────────

    def answer(self, path: str, raw: bytes) -> Tuple[int, Any]:
        text = raw.decode("utf-8", "replace")
        try:
            body = json.loads(text) if text else {}
        except ValueError:
            return 400, _error("conformance stub: request body was not JSON")

        lane = self._lane_for(text)
        if lane is None:
            return 400, _error(
                "conformance stub: no registered lane appears in this request — the "
                "driver ran a prompt it never registered a sentinel for"
            )
        if lane.fail:
            return 400, _error(
                "conformance stub: this lane is scripted to fail on purpose"
            )

        turn = lane.script[min(_turns_taken(body), len(lane.script) - 1)]
        model = body.get("model") or "conformance-stub"
        if path.endswith("/responses"):
            return 200, _responses_payload(model, turn, body)
        if path.endswith("/chat/completions"):
            return 200, _chat_payload(model, turn)
        return 404, _error(f"conformance stub has no route for POST {path}")

    def _lane_for(self, text: str) -> Optional[_Lane]:
        """The lane whose sentinel is in this request — longest match wins.

        Longest match matters: a derived lane's sentinel *contains* the base
        one (``…-lane3`` ends with the base sentinel's text), so a shortest- or
        first-match rule would answer every concurrent lane as lane zero and
        manufacture the cross-contamination C9 exists to detect.
        """
        with self._lock:
            hits = [lane for sentinel, lane in self._lanes.items() if sentinel in text]
        if not hits:
            return None
        return max(hits, key=lambda lane: len(lane.ctx.prompt_sentinel))


# ── payload shaping ──────────────────────────────────────────────────────────


def _error(message: str) -> Dict[str, Any]:
    """An OpenAI-shaped error. 400 is deliberate: it is not retried."""
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": "conformance_stub",
            "param": None,
        }
    }


def _turns_taken(body: Any) -> int:
    """How many model turns this conversation has already consumed.

    Counted as tool RESULTS present in the request — one per completed turn of
    the shared script. Deriving the index from the caller's own conversation is
    what makes the server stateless, and therefore safe for N lanes at once.
    """
    if not isinstance(body, dict):
        return 0
    count = 0
    for item in body.get("input") or []:  # Responses API
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            count += 1
    for message in body.get("messages") or []:  # Chat Completions API
        if isinstance(message, dict) and message.get("role") == "tool":
            count += 1
    return count


def _responses_payload(
    model: str, turn: StubTurn, body: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """One ``Response`` — the API ``Agent(model=...)`` uses by default.

    ``instructions`` is echoed back from the request, which is what the real
    Responses API does: ``Response.instructions`` is a field of the response
    object ("a system (or developer) message inserted into the model's
    context"), not something the client fills in. Omitting it here would have
    made the stub LESS faithful than the endpoint it stands in for, and would
    have left an adapter no way to observe the instructions at all — the SDK's
    ``ResponseSpanData`` carries ``(response, input, usage)``, so the request
    body is not otherwise recoverable.

    Note what this echo does NOT do: it cannot manufacture a passing grade. It
    returns whatever the caller actually serialised into the HTTP body, so an
    adapter that never got its skills menu into ``instructions`` gets a string
    without it back.
    """
    if turn.tool_call:
        name, args = turn.tool_call
        output: List[Dict[str, Any]] = [{
            "id": "fc_" + uuid.uuid4().hex,
            "type": "function_call",
            "call_id": "call_" + uuid.uuid4().hex[:16],
            "name": name,
            "arguments": json.dumps(args),
            "status": "completed",
        }]
    else:
        output = [{
            "id": "msg_" + uuid.uuid4().hex,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": turn.content, "annotations": []}
            ],
        }]
    in_tokens, out_tokens = turn.input_tokens, turn.output_tokens
    instructions = (body or {}).get("instructions")
    return {
        "id": "resp_" + uuid.uuid4().hex,
        "object": "response",
        "created_at": time.time(),
        "model": model,
        "status": "completed",
        "instructions": instructions if isinstance(instructions, str) else None,
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": in_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": out_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": in_tokens + out_tokens,
        },
    }


def _chat_payload(model: str, turn: StubTurn) -> Dict[str, Any]:
    """One ``chat.completion`` — the API Pydantic AI's OpenAI model uses."""
    if turn.tool_call:
        name, args = turn.tool_call
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": turn.content or None,
            "tool_calls": [{
                "id": "call_" + uuid.uuid4().hex[:16],
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }],
        }
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": turn.content}
        finish_reason = "stop"
    in_tokens, out_tokens = turn.input_tokens, turn.output_tokens
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": in_tokens,
            "completion_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
        },
    }
