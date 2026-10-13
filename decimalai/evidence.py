"""Opt-in, durable metadata evidence. No model work or business-state mutations."""

from __future__ import annotations

import atexit
import hashlib
import importlib.metadata
import itertools
import json
import marshal
import random
import sqlite3
import sys
import threading
import time
import types
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator
from uuid import uuid4

if TYPE_CHECKING:
    from ._client import DecimalAIClient
    from .schema.trace import RunTrace

import httpx

from ._evidence_contract import EvidenceEventInput, canonical, digest

MAX_BYTES = 256 * 1024 * 1024
_BOOT = str(uuid4())
_START = time.monotonic()
_local = threading.RLock()
_bindings: dict[str, "EvidenceClient"] = {}
_shutdown_registered = False


def target_id(base_url: str, api_key: str) -> str:
    # Local partition only; never exported. A new credential cannot upload another partition.
    return hashlib.sha256((base_url.rstrip("/") + "\0" + api_key).encode()).hexdigest()


class EvidenceSpool:
    def __init__(
        self, root: str | Path, target: str, *, path: Path | None = None
    ) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.is_symlink():
            raise ValueError("evidence root must not be a symlink")
        self.path = path or self.root / (str(uuid4()) + ".sqlite3")
        if self.path.is_symlink():
            raise ValueError("evidence file must not be a symlink")
        self.target = target
        self.failures = 0
        self.lock = threading.RLock()
        with self.db() as db:
            db.executescript(
                "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);"
                "CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY,event_id TEXT UNIQUE NOT NULL,hash TEXT NOT NULL,"
                'body TEXT NOT NULL,status TEXT NOT NULL DEFAULT "pending",reason TEXT);'
            )
            for key, value in [
                ("stream_id", str(uuid4())),
                ("boot_id", _BOOT),
                ("target", target),
            ]:
                db.execute("INSERT OR IGNORE INTO meta VALUES (?,?)", (key, value))
            db.execute(
                'INSERT OR IGNORE INTO meta VALUES ("next_seq",?)',
                (
                    str(
                        db.execute(
                            "SELECT COALESCE(MAX(seq),0)+1 FROM events"
                        ).fetchone()[0]
                    ),
                ),
            )
            db.execute('INSERT OR IGNORE INTO meta VALUES ("sealed_seq","0")')
            db.execute('INSERT OR IGNORE INTO meta VALUES ("sealed_failures","0")')
            saved = dict(db.execute("SELECT key,value FROM meta"))
            if saved["target"] != target:
                raise ValueError("evidence credential partition mismatch")
            self.stream_id, self.boot_id = saved["stream_id"], saved["boot_id"]
        self.path.chmod(0o600)

    @contextmanager
    def db(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=0.05)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def append(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        operation_id: str | None = None,
        correlation: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        try:
            size = sum(
                p.stat().st_size for p in self.root.glob("*.sqlite3*") if p.is_file()
            )
            if size >= MAX_BYTES:
                raise ValueError("spool full")
            with self.lock, self.db() as db:
                db.execute("BEGIN IMMEDIATE")
                if db.execute('SELECT value FROM meta WHERE key="closed"').fetchone():
                    raise ValueError("closed evidence stream")
                seq = int(
                    db.execute(
                        'SELECT value FROM meta WHERE key="next_seq"'
                    ).fetchone()[0]
                )
                event = EvidenceEventInput(
                    event_id=uuid4(),
                    seq=seq,
                    boot_id=self.boot_id,
                    kind=kind,
                    occurred_at=datetime.now(timezone.utc),
                    boot_elapsed_ms=int((time.monotonic() - _START) * 1000),
                    operation_id=operation_id,
                    correlation=correlation or {},
                    payload=payload,
                ).model_dump(mode="json")
                body = canonical(event)
                if len(body) > 8192:
                    raise ValueError("event oversized")
                db.execute(
                    "INSERT INTO events(seq,event_id,hash,body) VALUES (?,?,?,?)",
                    (seq, event["event_id"], digest(event), body.decode()),
                )
                db.execute(
                    'UPDATE meta SET value=? WHERE key="next_seq"', (str(seq + 1),)
                )
            return event
        except Exception:
            self.failures += 1
            return None

    def pending(self) -> list[dict[str, Any]]:
        with self.lock, self.db() as db:
            result, size = [], 0
            for row in db.execute(
                'SELECT body FROM events WHERE status="pending" ORDER BY seq LIMIT 100'
            ):
                event = json.loads(row[0])
                n = len(row[0].encode())
                if size + n > 240 * 1024:
                    break
                result.append(event)
                size += n
        return result

    def acknowledge(self, response: dict[str, Any], sent: list[dict[str, Any]]) -> bool:
        # Only the sent identities and exact canonical hashes can be marked accepted.
        if response.get("durable") is not True or not isinstance(
            response.get("items"), list
        ):
            return False
        expected = {e["event_id"]: e for e in sent}
        changes = []
        seen = set()
        for item in response["items"]:
            eid = item.get("event_id")
            original = expected.get(eid)
            if not original or eid in seen or item.get("seq") != original["seq"]:
                return False
            seen.add(eid)
            if item.get("status") in ("accepted", "duplicate"):
                if item.get("event_hash") != digest(original) or not item.get(
                    "receipt_id"
                ):
                    return False
                changes.append(("acked", None, eid))
            elif item.get("status") == "rejected":
                if item.get("reason") not in (
                    "identity_conflict",
                    "receipt_expired",
                    "unsupported_version",
                ):
                    return False
                changes.append(("quarantined", item["reason"], eid))
            else:
                return False
        with self.lock, self.db() as db:
            db.executemany(
                'UPDATE events SET status=?,reason=? WHERE event_id=? AND status="pending"',
                changes,
            )
        return seen == set(expected)

    def seal(
        self, first: int | None = None, *, operation_id: str | None = None
    ) -> dict[str, Any] | None:
        with self.lock:
            with self.db() as db:
                saved = dict(db.execute("SELECT key,value FROM meta"))
                last = int(saved["next_seq"]) - 1
                first = first or int(saved["sealed_seq"]) + 1
            if last < first or last - first >= 1000:
                return None
            event = self.append(
                "window.sealed",
                {
                    "ranges": [
                        {"stream_id": self.stream_id, "first": first, "last": last}
                    ],
                    "observer_failures": self.failures,
                    "before_observer_failures": int(saved["sealed_failures"]),
                    "after_observer_failures": self.failures,
                },
                operation_id=operation_id,
            )
            if event:
                with self.db() as db:
                    db.execute(
                        'UPDATE meta SET value=? WHERE key="sealed_seq"',
                        (str(event["seq"]),),
                    )
                    db.execute(
                        'UPDATE meta SET value=? WHERE key="sealed_failures"',
                        (str(self.failures),),
                    )
            return event

    def compact(self) -> None:
        # Sequence allocation is independent of row retention. Keep rejects and unacked rows.
        with self.lock, self.db() as db:
            db.execute('DELETE FROM events WHERE status="acked"')
        # Do not VACUUM on the business thread or discard a live WAL.

    def close_stream(self) -> None:
        with self.lock:
            with self.db() as db:
                final = (
                    int(
                        db.execute(
                            'SELECT value FROM meta WHERE key="next_seq"'
                        ).fetchone()[0]
                    )
                    + 1
                )
            event = self.append("stream.closed", {"final_seq": final})
            seal = self.seal()
            if event and seal and seal["seq"] == final:
                with self.db() as db:
                    db.execute('INSERT OR REPLACE INTO meta VALUES ("closed","1")')

    def prune_closed(self) -> bool:
        """Consume only closed, completely acknowledged transport files."""
        with self.lock, self.db() as db:
            if not db.execute('SELECT value FROM meta WHERE key="closed"').fetchone():
                return False
            if db.execute("SELECT count(*) FROM events").fetchone()[0]:
                return False
        for suffix in ("-wal", "-shm", ""):
            Path(str(self.path) + suffix).unlink(missing_ok=True)
        return True

    def health(self) -> dict[str, Any]:
        with self.db() as db:
            states = dict(
                db.execute("SELECT status,count(*) FROM events GROUP BY status")
            )
        return {
            "spool_pending": states.get("pending", 0),
            "quarantined": states.get("quarantined", 0),
            "observer_failures": self.failures,
            "spool_bytes": self.path.stat().st_size,
            "stream_id": self.stream_id,
            "boot_id": self.boot_id,
        }


class EvidenceClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        root: str | Path,
        *,
        background: bool = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key, self.base_url, self.root = (
            api_key,
            base_url.rstrip("/"),
            Path(root),
        )
        self.target = target_id(base_url, api_key)
        self.spool = EvidenceSpool(root, self.target)
        self.http = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": "Bearer " + api_key},
            timeout=5.0,
            transport=transport,
        )
        self.ready, self.next_try, self.last_status = False, 0.0, "pending"
        self.stop = threading.Event()
        self.thread = None
        self._shard_offset = 0
        versions = {"python": ".".join(map(str, sys.version_info[:3]))}
        for key, package in [
            ("sdk", "decimalai"),
            ("adk", "google-adk"),
            ("genai", "google-genai"),
            ("pydantic", "pydantic"),
        ]:
            try:
                versions[key] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        self.emit("runtime.boot", {"versions": versions, "scope_complete": False})
        self.capture_modules()
        if background:
            self.thread = threading.Thread(
                target=self._loop, name="decimalai-evidence", daemon=True
            )
            self.thread.start()

    def emit(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        operation_id: str | None = None,
        correlation: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        return self.spool.append(
            kind, payload, operation_id=operation_id, correlation=correlation
        )

    def _loop(self) -> None:
        while not self.stop.wait(2):
            self.upload_once()

    def capture_modules(self) -> None:
        """Fingerprint resident code objects; never claim a loader or Git witness."""

        def normalized(code: types.CodeType) -> types.CodeType:
            return code.replace(
                co_filename="",
                co_consts=tuple(
                    normalized(c) if isinstance(c, types.CodeType) else c
                    for c in code.co_consts
                ),
            )

        for name in (
            "decimalai.evidence",
            "decimalai._client",
            "decimalai.adk",
            "fleet.api",
            "fleet.skill_coverage",
            "fleet.run_all",
            "fleet.server",
            "fleet.scheduler",
        ):
            module = sys.modules.get(name)
            if module is None:
                continue
            code = []
            for key, obj in sorted(vars(module).items()):
                if isinstance(obj, types.FunctionType) and obj.__module__ == name:
                    code.append((key, obj.__code__))
                elif isinstance(obj, type) and obj.__module__ == name:
                    for method, fn in sorted(vars(obj).items()):
                        if isinstance(fn, (classmethod, staticmethod)):
                            fn = fn.__func__
                        if isinstance(fn, types.FunctionType):
                            code.append((key + "." + method, fn.__code__))
            if code:
                self.emit(
                    "runtime.module",
                    {
                        "module": name,
                        "capture_class": "loaded_code_snapshot",
                        "artifact_hash": hashlib.sha256(
                            b"".join(
                                key.encode() + marshal.dumps(normalized(c))
                                for key, c in code
                            )
                        ).hexdigest(),
                    },
                )

    def _shards(self) -> Iterator[EvidenceSpool]:
        # Upload only matching authenticated partitions, including retained shards from an old process.
        paths = sorted(self.root.glob("*.sqlite3"))
        if not paths:
            return
        start = self._shard_offset % len(paths)
        for offset, path in enumerate(
            itertools.islice(itertools.cycle(paths), start, start + min(32, len(paths)))
        ):
            self._shard_offset = start + offset + 1
            try:
                db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.05)
                try:
                    saved = dict(db.execute("SELECT key,value FROM meta"))
                finally:
                    db.close()
                if saved.get("target") == self.target:
                    yield EvidenceSpool(self.root, self.target, path=path)
            except (OSError, sqlite3.Error, ValueError):
                continue

    def upload_once(self) -> bool:
        if time.monotonic() < self.next_try:
            return False
        try:
            if not self.ready:
                response = self.http.get("/api/v1/evidence/capabilities")
                caps = response.json() if response.status_code == 200 else {}
                if not (
                    caps.get("enabled") is True
                    and caps.get("schema_version") == 1
                    and caps.get("acknowledgement") == "per_event_after_commit"
                ):
                    self.last_status = "capability_pending"
                    self.next_try = time.monotonic() + 300
                    return False
                self.ready = True
            for spool in self._shards():
                events = spool.pending()
                if not events:
                    spool.prune_closed()
                    continue
                batch_id = str(uuid4())
                response = self.http.post(
                    "/api/v1/evidence/events/batch",
                    json={
                        "schema_version": 1,
                        "batch_id": batch_id,
                        "stream_id": spool.stream_id,
                        "events": events,
                    },
                )
                if response.status_code != 200:
                    self.last_status = "upload_pending"
                    try:
                        delay = max(
                            10,
                            min(3600, float(response.headers.get("Retry-After", "30"))),
                        )
                    except ValueError:
                        delay = 30
                    self.next_try = (
                        time.monotonic() + delay + random.uniform(0, delay / 10)
                    )
                    return False
                result = response.json()
                if result.get("batch_id") != batch_id or not spool.acknowledge(
                    result, events
                ):
                    self.last_status = "acknowledgement_invalid"
                    self.next_try = time.monotonic() + 30
                    return False
                self.last_status = "uploaded"
                self.next_try = time.monotonic() + 2
                spool.compact()
                spool.prune_closed()
                return True
            return True
        except Exception:
            self.last_status = "transport_pending"
            self.next_try = time.monotonic() + random.uniform(30, 33)
            return False

    def close(self) -> None:
        self.stop.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=6)
        self.emit("runtime.stopping", {"scope_complete": False})
        self.spool.close_stream()
        self.http.close()


def enable(
    api_key: str,
    base_url: str = "https://api.decimal.ai",
    directory: str | Path | None = None,
    *,
    background: bool = True,
) -> EvidenceClient:
    global _shutdown_registered
    root = (
        Path(directory or Path.home() / ".decimalai" / "evidence")
        .expanduser()
        .absolute()
    )
    key = target_id(base_url, api_key)
    with _local:
        if key not in _bindings:
            _bindings[key] = EvidenceClient(
                api_key, base_url, root, background=background
            )
        if not _shutdown_registered:
            atexit.register(close_all)
            _shutdown_registered = True
        return _bindings[key]


def observer(api_key: str, base_url: str) -> EvidenceClient | None:
    return _bindings.get(target_id(base_url, api_key))


def observe_trace(client: "DecimalAIClient", trace: "RunTrace") -> None:
    binding = observer(client.api_key, client.base_url)
    if binding is None:
        return
    # Explicit adapter-delivery witnesses only; no prompt, response or loaded menu is copied.
    witnesses = [
        {
            "skill_id": x.get("skill_id"),
            "version_id": x.get("version_id"),
            "skill_hash": x.get("hash"),
        }
        for x in trace.skills_delivered_versions
        if x.get("skill_id") and x.get("version_id") and x.get("hash")
    ]
    binding.emit(
        "adapter.invocation",
        {
            "trace_id": str(trace.id),
            "parent_trace_id": trace.parent_trace_id,
            "framework": str(
                (trace.session_metadata or {}).get("framework", "unknown")
            ),
            "model_call_ids": [str(x.id) for x in trace.llm_calls][:100],
            "witnesses": witnesses[:100],
            "witnesses_complete": len(witnesses) == len(trace.skills_delivered_versions)
            and len(witnesses) <= 100
            and len(trace.llm_calls) <= 100,
        },
        operation_id=str(trace.id),
    )
    binding.spool.seal(operation_id=str(trace.id))


def close_all() -> None:
    with _local:
        for client in _bindings.values():
            try:
                client.close()
            except Exception:
                pass  # Never disrupt the process's ordinary exit.
        _bindings.clear()
