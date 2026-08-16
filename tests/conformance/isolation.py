"""One driver, one process — and the wire that carries its capture home.

Why this exists
---------------
Every adapter under conformance installs PROCESS-GLOBAL instrumentation: module
ContextVars, langchain-core's global configure-hook list, an OTel global
``TracerProvider`` that can only be set once, monkeypatched ``__init__``s that
cannot be undone. Running every driver's phases in ONE process therefore lets
whichever driver goes first decide what the ones after it are allowed to
observe. That does not merely add noise: it produced FALSE results in both
directions — ``langchain:C1`` ("emits at all") red on a framework that emits
fine, a known-red item green under one ``-k`` selection and red under another,
and one driver's ``manifest_id`` on another driver's traces, failing C2 on the
adapter that was not at fault. A false failure on a healthy framework is worse
than having no gate: the next person to see it switches the job off.

So the phases run in a child process, one per driver, and only the CAPTURE comes
back. The parent imports no framework and runs no driver code, so there is
nothing left for a driver to contaminate.

What crosses the boundary
-------------------------
Everything ``contract.py`` reads, and nothing else:

* every ``Phase`` — its ctxs, the probe's recorded requests (bodies and
  responses verbatim), logs, warnings, new paths, export deltas;
* the probe state the contract queries — ``manifests`` (C6's
  ``manifest_owner``), ``skills`` (C8), ``routing_queries`` (C8's provenance
  check);
* the ``Ctx`` the run was built from.

It travels as JSON, because the payload is already JSON: every recorded body
came out of ``json.loads`` on the probe's socket, so it round-trips exactly.
``dump_payload`` proves that on every run rather than assuming it — see
:func:`dump_payload`.

The probe stays in the CHILD, which is the half that is easy to get wrong: the
SDK's HTTP calls happen in the child, so a probe listening in the parent would
record nothing at all and every driver would look like it emits nothing — the
very defect this file exists to stop faking.

Deliberately NOT changed: the phases inside one driver still run in one process,
in order, with adapter globals left sticky between them. "Two agents back to
back in one process" is a defect class this suite exists to catch (C6, C7), and
it is a property of ONE driver's run — not something another framework should be
able to inject.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .drivers import Ctx, Driver
from .harness import Observation, Phase
from .probe import Probe, Recorded

#: Bumped when the payload shape changes. A child from a stale checkout then
#: fails loudly instead of being graded on fields the parent misreads.
WIRE_VERSION = 1

#: Skills the probe's router offers when a driver declares a skills rail.
#: Lives here rather than in ``conftest.py`` because the CHILD needs it and a
#: child should not have to import a pytest plugin module to run a driver.
CONFORMANCE_SKILLS: Tuple[Dict[str, str], ...] = (
    {
        "name": "conformance-skill-alpha",
        "description": "Alpha skill offered by the conformance probe.",
        "body": "# Alpha\n\nAlpha guidance for the conformance run.",
    },
    {
        "name": "conformance-skill-beta",
        "description": "Beta skill offered by the conformance probe.",
        "body": "# Beta\n\nBeta guidance for the conformance run.",
    },
)

#: How long one driver's seven phases may take before the child is killed.
DEFAULT_TIMEOUT_SECONDS = 900
TIMEOUT_ENV = "DECIMAL_CONFORMANCE_DRIVER_TIMEOUT"

#: How many driver children to run at once. Each child is a separate process
#: with its own probe on its own ephemeral port, so they cannot interfere; the
#: cap exists only so a laptop is not asked to schedule every framework's
#: thread pools at once, which would make the timing-sensitive items (C8, C9)
#: flaky — trading one untrustworthy result for another.
JOBS_ENV = "DECIMAL_CONFORMANCE_JOBS"
DEFAULT_JOBS = 4


class DriverProcessError(RuntimeError):
    """A driver's child process did not return a capture.

    Raised from the parent's ``observations[driver]`` lookup, so every contract
    item for that driver ERRORS and names the driver. Never a skip: a driver
    that crashed was not graded, and a suite that reports ungraded items as
    passed or skipped is exactly the failure mode this tier exists to remove.
    """


class RemoteDriverException(Exception):
    """Stands in for an exception raised inside a child's phase.

    The contract reads ``phase.exception`` for one thing — whether the phase
    failed loudly (C12) — so the text is enough. Carrying the real object would
    mean pickling arbitrary framework exception types across the boundary, which
    fails on the ones that do not round-trip and would take the run down with
    it.
    """


# ── encode ───────────────────────────────────────────────────────────────────


def _encode_ctx(ctx: Ctx) -> Dict[str, Any]:
    out = {f.name: getattr(ctx, f.name) for f in fields(Ctx)}
    # The one non-JSON member: a tuple of mappings.
    out["skills"] = [dict(s) for s in ctx.skills]
    return out


def _encode_recorded(rec: Recorded) -> Dict[str, Any]:
    # Every field is already JSON: body and response came out of json.loads on
    # the probe's socket (or are plain strings), query is str->list[str].
    return {f.name: getattr(rec, f.name) for f in fields(Recorded)}


def _describe_exception(exc: Optional[BaseException]) -> Optional[str]:
    if exc is None:
        return None
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= 2000 else text[:1997] + "..."


def _encode_phase(phase: Phase) -> Dict[str, Any]:
    out = {f.name: getattr(phase, f.name) for f in fields(Phase)}
    out["ctxs"] = [_encode_ctx(c) for c in phase.ctxs]
    out["requests"] = [_encode_recorded(r) for r in phase.requests]
    out["exception"] = _describe_exception(phase.exception)
    out["logs"] = list(phase.logs)
    out["warnings"] = list(phase.warnings)
    out["new_paths"] = list(phase.new_paths)
    return out


def _encode_probe(probe: Probe) -> Dict[str, Any]:
    # Only what the contract asks the probe: manifest ownership (C6), the skills
    # the router offered (C8), and which query each routing_id was minted for
    # (C8's provenance check). `requests` is deliberately absent — every request
    # is already carried, sliced per phase, and duplicating the log would double
    # a payload that is megabytes of rendered prompts.
    return {
        "require_manifest": probe.require_manifest,
        "manifests": probe.manifests,
        "skills": [dict(s) for s in probe.skills],
        "routing_queries": dict(probe.routing_queries),
        "trace_ids": sorted(probe.trace_ids),
    }


def encode_observation(obs: Observation) -> Dict[str, Any]:
    return {
        "wire_version": WIRE_VERSION,
        "driver": obs.driver.name,
        "ctx": _encode_ctx(obs.ctx),
        "phases": {name: _encode_phase(p) for name, p in obs.phases.items()},
        "probe": _encode_probe(obs.probe),
    }


def dump_payload(payload: Dict[str, Any]) -> str:
    """Serialise, and PROVE the round-trip is lossless before shipping it.

    Two things JSON silently changes: a tuple becomes a list, and NaN/Infinity
    becomes a non-standard literal. Either would reach the contract as a
    plausible-looking value and be graded, so both are turned into a loud child
    crash here instead. Costs a few milliseconds on a payload that took a
    minute to produce.
    """
    text = json.dumps(payload, allow_nan=False)
    if json.loads(text) != payload:
        raise TypeError(
            "the conformance observation does not survive a JSON round-trip — "
            "some field is not the plain JSON it looks like (a tuple, a set, or "
            "a custom object). Fix the encoder; grading a mangled capture is "
            "worse than not grading."
        )
    return text


# ── decode ───────────────────────────────────────────────────────────────────


def _decode_ctx(data: Dict[str, Any]) -> Ctx:
    kwargs = dict(data)
    kwargs["skills"] = tuple(dict(s) for s in data.get("skills") or ())
    return Ctx(**kwargs)


def _decode_phase(data: Dict[str, Any]) -> Phase:
    kwargs = dict(data)
    kwargs["ctxs"] = [_decode_ctx(c) for c in data["ctxs"]]
    kwargs["requests"] = [Recorded(**r) for r in data["requests"]]
    kwargs["exception"] = (
        RemoteDriverException(data["exception"]) if data.get("exception") else None
    )
    return Phase(**kwargs)


def _decode_probe(data: Dict[str, Any]) -> Probe:
    # A real Probe, never started: the contract only queries its state, and a
    # stand-in class would drift from the thing the child actually ran.
    probe = Probe(require_manifest_on_ingest=data["require_manifest"])
    probe.manifests = data["manifests"]
    probe.skills = data["skills"]
    probe.routing_queries = data["routing_queries"]
    probe.trace_ids = set(data["trace_ids"])
    return probe


def decode_observation(payload: Dict[str, Any], driver: Driver) -> Observation:
    version = payload.get("wire_version")
    if version != WIRE_VERSION:
        raise DriverProcessError(
            f"{driver.name}: child returned wire_version {version!r}, this parent "
            f"speaks {WIRE_VERSION}"
        )
    if payload.get("driver") != driver.name:
        raise DriverProcessError(
            f"{driver.name}: child returned a capture for {payload.get('driver')!r}"
        )
    return Observation(
        driver=driver,
        probe=_decode_probe(payload["probe"]),
        ctx=_decode_ctx(payload["ctx"]),
        phases={name: _decode_phase(p) for name, p in payload["phases"].items()},
    )


# ── run ──────────────────────────────────────────────────────────────────────


_HERE = Path(__file__).resolve().parent
_TESTS_DIR = _HERE.parent
_REPO_ROOT = _TESTS_DIR.parent
_CHILD = _HERE / "_child.py"


def _child_env() -> Dict[str, str]:
    """Inherit the caller's environment, plus the paths the child imports from.

    ``sys.executable`` + an inherited ``PYTHONPATH`` is what makes this work
    from a scratch venv (repo not installed, PYTHONPATH points at the checkout)
    and in CI (repo installed) without either knowing about the other.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    parts = [str(_REPO_ROOT)] + ([existing] if existing else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _timeout_seconds() -> float:
    raw = os.environ.get(TIMEOUT_ENV)
    try:
        return float(raw) if raw else float(DEFAULT_TIMEOUT_SECONDS)
    except ValueError:
        return float(DEFAULT_TIMEOUT_SECONDS)


def _tail(text: Any, limit: int = 2000) -> str:
    """The last of a child's output — enough to see what killed it."""
    if isinstance(text, bytes):  # TimeoutExpired can hand back either
        text = text.decode("utf-8", "replace")
    text = (text or "").strip()
    return text if len(text) <= limit else "...\n" + text[-limit:]


def run_driver_in_child(driver: Driver) -> Observation:
    """Run every phase of ``driver`` in a fresh process and return the capture."""
    workdir = tempfile.mkdtemp(prefix=f"conformance-child-{driver.name}-")
    out_path = Path(workdir) / "observation.json"
    cmd = [sys.executable, str(_CHILD), driver.name, str(out_path)]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            env=_child_env(),
            capture_output=True,
            text=True,
            timeout=_timeout_seconds(),
        )
    except subprocess.TimeoutExpired as exc:
        raise DriverProcessError(
            f"{driver.name}: the driver process did not finish within "
            f"{_timeout_seconds():.0f}s and was killed — nothing was graded for this "
            f"framework.\n"
            f"stdout: {_tail(exc.stdout)}\n"
            f"stderr: {_tail(exc.stderr)}"
        ) from exc

    if proc.returncode != 0 or not out_path.exists():
        raise DriverProcessError(
            f"{driver.name}: the driver process exited {proc.returncode} "
            f"{'without writing a capture' if not out_path.exists() else ''} — "
            f"nothing was graded for this framework.\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout: {_tail(proc.stdout)}\n"
            f"stderr: {_tail(proc.stderr)}"
        )
    try:
        payload = json.loads(out_path.read_text())
    except (OSError, ValueError) as exc:
        raise DriverProcessError(
            f"{driver.name}: the driver process wrote a capture this parent cannot "
            f"read ({exc}) — nothing was graded for this framework.\n"
            f"stderr: {_tail(proc.stderr)}"
        ) from exc
    observation = decode_observation(payload, driver)
    # Only once it is safely in memory. A capture that could NOT be read stays on
    # disk, because that is the one time somebody needs to look at it.
    shutil.rmtree(workdir, ignore_errors=True)
    return observation


def jobs(count: int) -> int:
    """How many driver children to run at once."""
    raw = os.environ.get(JOBS_ENV)
    if raw:
        try:
            return max(1, min(int(raw), max(1, count)))
        except ValueError:
            pass
    cpus = os.cpu_count() or 1
    return max(1, min(DEFAULT_JOBS, max(1, cpus // 2), max(1, count)))


def run_drivers(
    drivers: Sequence[Driver],
) -> Tuple[Dict[str, Observation], Dict[str, str]]:
    """Run each driver in its own process. Returns ``(observations, failures)``.

    A crashed child lands in ``failures`` keyed by driver name; the caller turns
    that into a hard error at lookup time. It is never silently dropped.
    """
    observed: Dict[str, Observation] = {}
    failures: Dict[str, str] = {}
    if not drivers:
        return observed, failures

    order: List[str] = [d.name for d in drivers]
    with ThreadPoolExecutor(max_workers=jobs(len(drivers))) as pool:
        futures = {d.name: pool.submit(run_driver_in_child, d) for d in drivers}
    for name in order:
        try:
            observed[name] = futures[name].result()
        except DriverProcessError as exc:
            failures[name] = str(exc)
        except Exception as exc:  # noqa: BLE001 - a harness bug is still that driver's error
            failures[name] = f"{name}: harness could not run the driver process: {exc!r}"
    return observed, failures
