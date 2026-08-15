"""Runs a driver's phases and captures what reached the wire.

No assertions live here — this is the instrument, ``contract.py`` is the spec.
The harness's only opinions are operational: one temp cwd per phase (so C11 can
diff it), one probe cursor per phase (so each phase's traffic is separable), and
a flush between phases (so a background sender can't smear one phase's traces
into the next one's slice).

Deliberately NOT reset between phases: the adapters' module-level globals. A
process that runs two agents back to back is the situation where sticky
process-global state bites, so the phases run in one process, in order, exactly
as a user's would.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .drivers import Ctx, Driver
from .probe import Probe, Recorded

#: Lanes used by the concurrency phase.
CONCURRENCY = 8

#: How long to wait for the background sender to be done after flush().
SETTLE_SECONDS = 0.75


@dataclass
class Phase:
    """One driver phase: what it was asked to do, and what reached the wire."""

    name: str
    ctxs: List[Ctx]
    ran: bool = True
    na_reason: Optional[str] = None
    requests: List[Recorded] = field(default_factory=list)
    exception: Optional[BaseException] = None
    logs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    new_paths: List[str] = field(default_factory=list)
    # Public SDK export surface, measured across this phase only.
    export_sent_delta: int = 0
    export_failed_delta: int = 0
    export_last_error: Optional[str] = None

    # ── views ────────────────────────────────────────────────

    def _trace_posts(self) -> List[Recorded]:
        return [
            r for r in self.requests
            if r.method == "POST" and r.path in ("/api/v1/traces", "/api/v1/traces/batch")
        ]

    @property
    def traces(self) -> List[Dict[str, Any]]:
        """Trace payloads the probe ACCEPTED — i.e. that would have landed."""
        out: List[Dict[str, Any]] = []
        for r in self._trace_posts():
            bodies = r.body if isinstance(r.body, list) else [r.body]
            if r.accepted:
                out.extend(b for b in bodies if isinstance(b, dict))
        return out

    @property
    def rejected(self) -> List[Recorded]:
        """Trace POSTs the probe refused, with the backend's own reasons."""
        return [r for r in self._trace_posts() if not r.accepted]

    @property
    def attempted(self) -> List[Dict[str, Any]]:
        """Every trace payload that left the SDK, accepted or not."""
        out: List[Dict[str, Any]] = []
        for r in self._trace_posts():
            bodies = r.body if isinstance(r.body, list) else [r.body]
            out.extend(b for b in bodies if isinstance(b, dict))
        return out

    @property
    def manifest_posts(self) -> List[Recorded]:
        return [
            r for r in self.requests
            if r.method == "POST" and r.path == "/api/v1/manifests"
        ]


@dataclass
class Observation:
    """Everything one driver produced, across every phase. The contract's input."""

    driver: Driver
    probe: Probe
    ctx: Ctx
    phases: Dict[str, Phase]

    def phase(self, name: str) -> Phase:
        return self.phases[name]

    @property
    def all_attempted(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for p in self.phases.values():
            out.extend(p.attempted)
        return out

    @property
    def all_rejected(self) -> List[Recorded]:
        out: List[Recorded] = []
        for p in self.phases.values():
            out.extend(p.rejected)
        return out


# ── capture plumbing ─────────────────────────────────────────────────────────


class _LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(f"{record.name}: {record.getMessage()}")
        except Exception:  # pragma: no cover - never break the run being observed
            pass


@contextmanager
def _observe(workdir: str) -> Iterator[Dict[str, Any]]:
    """Capture logs, warnings and cwd side effects around one phase."""
    captured: Dict[str, Any] = {"logs": [], "warnings": [], "new_paths": []}
    handler = _LogCapture()
    root = logging.getLogger()
    root.addHandler(handler)
    prev_cwd = os.getcwd()
    os.chdir(workdir)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield captured
        captured["warnings"] = [str(w.message) for w in caught]
    finally:
        os.chdir(prev_cwd)
        root.removeHandler(handler)
        captured["logs"] = handler.records
        captured["new_paths"] = sorted(
            str(p.relative_to(workdir)) for p in Path(workdir).rglob("*")
        )


def _flush_sdk() -> None:
    import decimalai

    try:
        decimalai.flush()
    except Exception:  # pragma: no cover - flush must never mask a phase result
        pass
    time.sleep(SETTLE_SECONDS)
    try:
        decimalai.flush()
    except Exception:  # pragma: no cover
        pass


def _run_phase(
    name: str,
    fn: Any,
    ctxs: List[Ctx],
    probe: Probe,
    *,
    fanout: bool,
    na_reason: Optional[str] = None,
) -> Phase:
    if fn is None or na_reason is not None:
        return Phase(
            name=name, ctxs=ctxs, ran=False,
            na_reason=na_reason or f"the {name} driver hook is not implemented",
        )

    import decimalai

    workdir = tempfile.mkdtemp(prefix=f"conformance-{name}-")
    cursor = probe.mark()
    before = decimalai.export_status()
    exc: Optional[BaseException] = None
    with _observe(workdir) as cap:
        try:
            fn(ctxs if fanout else ctxs[0])
        except BaseException as e:  # noqa: BLE001 - a failing run is a phase result
            exc = e
        _flush_sdk()
    after = decimalai.export_status()
    return Phase(
        name=name,
        ctxs=ctxs,
        requests=probe.since(cursor),
        exception=exc,
        logs=cap["logs"],
        warnings=cap["warnings"],
        new_paths=cap["new_paths"],
        export_sent_delta=after.sent - before.sent,
        export_failed_delta=after.failed - before.failed,
        export_last_error=str(after.last_error) if after.last_error else None,
    )


def observe(driver: Driver, probe: Probe, *, skills: Sequence[Dict[str, Any]] = ()) -> Observation:
    """Run every phase of ``driver`` against ``probe`` and return the capture."""
    import decimalai

    caps = driver.capabilities
    token = uuid.uuid4().hex[:8]
    workdir = tempfile.mkdtemp(prefix="conformance-base-")
    ctx = Ctx(
        base_url=probe.base_url,
        api_key=probe.api_key,
        agent_name=f"conformance-{driver.name}-{token}",
        prompt_sentinel=f"SENTINEL-PROMPT-{token}",
        reply_sentinel=f"SENTINEL-REPLY-{token}",
        tool_name="conformance_lookup",
        tool_sentinel=f"SENTINEL-TOOL-{token}",
        workdir=workdir,
        skills=tuple(skills),
    )

    probe.skills = [dict(s) for s in skills] if caps.has_skills_rail else []

    prev_env = {k: os.environ.get(k) for k in ("DECIMAL_API_KEY", "DECIMAL_BASE_URL")}
    os.environ["DECIMAL_API_KEY"] = probe.api_key
    os.environ["DECIMAL_BASE_URL"] = probe.base_url
    try:
        decimalai.init(api_key=probe.api_key, base_url=probe.base_url, verify=False)

        phases: Dict[str, Phase] = {}
        phases["main"] = _run_phase("main", driver.run, [ctx], probe, fanout=False)
        phases["repeat"] = _run_phase("repeat", driver.run, [ctx], probe, fanout=False)
        # A SECOND, differently-named agent in the same process. Costs a driver
        # nothing (it is `run` again with a derived ctx) and is the only way to
        # see process-global agent identity — the "every trace after the first
        # keeps shipping the first agent's name" defect.
        phases["second_agent"] = _run_phase(
            "second_agent", driver.run, [ctx.derive(99)], probe, fanout=False,
        )
        phases["degenerate"] = _run_phase(
            "degenerate", driver.run_degenerate, [ctx], probe, fanout=False,
            na_reason=caps.na_reason("C7b"),
        )
        phases["error"] = _run_phase(
            "error", driver.run_error, [ctx], probe, fanout=False,
            na_reason=caps.na_reason("C10"),
        )
        phases["concurrent"] = _run_phase(
            "concurrent",
            driver.run_concurrent,
            [ctx.derive(i) for i in range(CONCURRENCY)],
            probe,
            fanout=True,
            na_reason=caps.na_reason("C9"),
        )
        # LAST on purpose: on several adapters turning the rail on is an
        # irreversible process-wide monkey-patch, so it must not colour the
        # phases above.
        # Lanes share the agent name here (rename=False): the rail is a
        # process-wide mode on most adapters, so one agent is all the API offers.
        # The lanes still differ by sentinel, which is what makes a leaked
        # routing decision or a bled prompt visible.
        phases["skills"] = _run_phase(
            "skills",
            driver.run_skills,
            [ctx.derive(i, rename=False) for i in range(CONCURRENCY)],
            probe,
            fanout=True,
            na_reason=caps.na_reason("C8"),
        )
    finally:
        for k, v in prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    return Observation(driver=driver, probe=probe, ctx=ctx, phases=phases)
