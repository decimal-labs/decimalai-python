"""A local Anthropic Messages API (JSON and streaming SSE) on 127.0.0.1.

The Claude Code CLI is the Claude loader, so the check runs the real bundled CLI
and points it here with ``ANTHROPIC_BASE_URL``. Nothing reaches a real model.

Requests are recorded (auth headers reduced to "dummy" or a redaction marker) and
answered by a scripted agent: a list of step functions, one per assistant tool-use
turn already in the conversation. A request that carries no ``Skill`` tool is a
side call (title generation, a classifier) and gets a plain "ok".
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, List, Optional

Step = Callable[[dict], list]   # request body -> content blocks


def text_of(obj: Any) -> str:
    """Every string inside a request-body fragment, newline-joined."""
    out: List[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    return "\n".join(out)


def tool_turns(body: dict) -> int:
    n = 0
    for m in body.get("messages", []):
        if m.get("role") == "assistant" and isinstance(m.get("content"), list):
            if any(b.get("type") == "tool_use" for b in m["content"]):
                n += 1
    return n


def last_base_dir(body: dict) -> Optional[str]:
    """The most recent "Base directory for this skill:" the model was shown."""
    found = re.findall(r"Base directory for this skill:\s*(\S+)", text_of(body.get("messages", [])))
    return found[-1] if found else None


def tool_use(name: str, inp: dict) -> dict:
    return {"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:20], "name": name, "input": inp}


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


class Stub:
    def __init__(self, steps: List[Step], final_text: str = "DONE") -> None:
        self.steps = steps
        self.final_text = final_text
        self.requests: List[dict] = []
        self._lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:
                pass

            def _read(self) -> dict:
                n = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(n) if n else b""
                try:
                    return json.loads(raw or b"{}")
                except ValueError:
                    return {"_raw": raw.decode("latin-1")[:2000]}

            def _headers(self) -> dict:
                h = {}
                for k, v in self.headers.items():
                    kl = k.lower()
                    if kl in ("x-api-key", "authorization"):
                        h[kl] = "dummy" if v.strip() in ("dummy", "Bearer dummy") else "<REDACTED>"
                return h

            def _json(self, code: int, obj: Any) -> None:
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                stub._record({"method": "GET", "path": self.path, "headers": self._headers()})
                if self.path.startswith("/v1/models"):
                    self._json(200, {"data": [], "has_more": False})
                else:
                    self._json(404, {"type": "error",
                                     "error": {"type": "not_found_error", "message": "stub"}})

            def do_HEAD(self) -> None:
                self.send_response(200)
                self.send_header("content-length", "0")
                self.end_headers()

            def do_POST(self) -> None:
                body = self._read()
                stub._record({"method": "POST", "path": self.path, "headers": self._headers(),
                              "body": body})
                path = self.path.split("?")[0]
                if path.endswith("/count_tokens"):
                    return self._json(200, {"input_tokens": 1000})
                if not path.endswith("/v1/messages"):
                    return self._json(404, {"type": "error",
                                            "error": {"type": "not_found_error", "message": "stub"}})
                blocks, stop = stub.decide(body)
                msg = {
                    "id": "msg_" + uuid.uuid4().hex[:20], "type": "message", "role": "assistant",
                    "model": body.get("model", "stub-model"), "content": blocks,
                    "stop_reason": stop, "stop_sequence": None,
                    "usage": {"input_tokens": 100, "output_tokens": 20,
                              "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
                }
                if body.get("stream"):
                    self._sse(msg)
                else:
                    self._json(200, msg)

            def _sse(self, msg: dict) -> None:
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("cache-control", "no-cache")
                self.send_header("connection", "close")
                self.end_headers()

                def ev(name: str, data: dict) -> None:
                    self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())

                start = dict(msg, content=[], stop_reason=None)
                start["usage"] = dict(msg["usage"], output_tokens=1)
                ev("message_start", {"type": "message_start", "message": start})
                for i, b in enumerate(msg["content"]):
                    if b["type"] == "text":
                        ev("content_block_start", {"type": "content_block_start", "index": i,
                                                   "content_block": {"type": "text", "text": ""}})
                        ev("content_block_delta", {"type": "content_block_delta", "index": i,
                                                   "delta": {"type": "text_delta", "text": b["text"]}})
                    else:
                        ev("content_block_start", {"type": "content_block_start", "index": i,
                                                   "content_block": {"type": "tool_use", "id": b["id"],
                                                                     "name": b["name"], "input": {}}})
                        ev("content_block_delta", {"type": "content_block_delta", "index": i,
                                                   "delta": {"type": "input_json_delta",
                                                             "partial_json": json.dumps(b["input"])}})
                    ev("content_block_stop", {"type": "content_block_stop", "index": i})
                ev("message_delta", {"type": "message_delta",
                                     "delta": {"stop_reason": msg["stop_reason"], "stop_sequence": None},
                                     "usage": {"output_tokens": 20}})
                ev("message_stop", {"type": "message_stop"})
                self.wfile.flush()
                self.close_connection = True

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def _record(self, rec: dict) -> None:
        with self._lock:
            self.requests.append(rec)

    @staticmethod
    def is_main_loop(body: dict) -> bool:
        return any(t.get("name") == "Skill" for t in body.get("tools", []) or [])

    def main_loop_requests(self) -> List[dict]:
        with self._lock:
            return [r["body"] for r in self.requests
                    if r.get("method") == "POST" and isinstance(r.get("body"), dict)
                    and self.is_main_loop(r["body"])]

    def decide(self, body: dict):
        if not self.is_main_loop(body):
            return [text_block("ok")], "end_turn"
        i = tool_turns(body)
        if i < len(self.steps):
            blocks = self.steps[i](body)
            if blocks:
                stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
                return blocks, stop
        return [text_block(self.final_text)], "end_turn"

    def __enter__(self) -> "Stub":
        self.thread.start()
        return self

    def __exit__(self, *a: Any) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"
