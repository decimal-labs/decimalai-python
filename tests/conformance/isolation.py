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

from .delivery import DEFAULT as DELIVERY_DEFAULT
from .delivery import MODE_ENV, mode_env
from .drivers import Ctx, Driver
from .harness import Observation, Phase
from .probe import Probe, Recorded

#: Every environment variable any delivery mode sets — stripped from a child's
#: environment before the mode is applied, so an inherited value cannot decide
#: what a cell means.
_DELIVERY_ENV_KEYS = frozenset(k for env in MODE_ENV.values() for k in env)

#: Bumped when the payload shape changes. A child from a stale checkout then
#: fails loudly instead of being graded on fields the parent misreads.
#: 2 — ``Ctx`` gained ``delivery_mode`` (the body-channel axis).
WIRE_VERSION = 2

#: Skills the probe's router offers when a driver declares a skills rail.
#: Lives here rather than in ``conftest.py`` because the CHILD needs it and a
#: child should not have to import a pytest plugin module to run a driver.
#:
#: Each body carries a SENTINEL line: a fact that appears nowhere else, that a
#: menu row cannot contain (menu rows are name + description), and that no model
#: has priors for. C14 asserts the sentinel reached the model, which is the only
#: way to tell "the skill was offered" from "the skill was readable".
#:
#: Deliberately shaped as a checkable fact rather than a random token so ONE
#: fixture can serve both tiers: the hermetic tier asserts the sentence is in the
#: prompt, and a live tier asserts the number is in the model's answer. Two
#: fixtures would drift.
CONFORMANCE_SKILLS: Tuple[Dict[str, str], ...] = (
    {
        "name": "conformance-skill-alpha",
        "description": "Alpha skill offered by the conformance probe.",
        "body": (
            "# Alpha\n\n"
            "SENTINEL-SKILLBODY-ALPHA: opened boxes carry a 23.5% restocking fee.\n"
            "Only a delivered body carries this line."
        ),
    },
    {
        "name": "conformance-skill-beta",
        "description": "Beta skill offered by the conformance probe.",
        "body": (
            "# Beta\n\n"
            "SENTINEL-SKILLBODY-BETA: expedited returns close within 4.75 business days.\n"
            "Only a delivered body carries this line."
        ),
    },
)

#: The per-skill sentinel, derived from the body so the two cannot drift.
#: `_body_signature`'s longest-line heuristic is fine for today's fixture but is
#: not safe in general — a real skill whose longest body line restates its
#: description would let a MENU ROW satisfy the very clause that exists to catch
#: an undelivered body.
SENTINEL_PREFIX = "SENTINEL-SKILLBODY-"


def body_sentinel(body: str) -> str:
    """The sentinel line of a conformance skill body, or "" if it has none."""
    for line in (body or "").splitlines():
        stripped = line.strip()
        if stripped.startswith(SENTINEL_PREFIX):
            return stripped
    return ""

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


def _child_env(delivery_mode: str = DELIVERY_DEFAULT) -> Dict[str, str]:
    """Inherit the caller's environment, plus the paths the child imports from.

    ``sys.executable`` + an inherited ``PYTHONPATH`` is what makes this work
    from a scratch venv (repo not installed, PYTHONPATH points at the checkout)
    and in CI (repo installed) without either knowing about the other.

    The delivery mode is applied HERE rather than inside the child, so the
    variables are in place before the interpreter starts. ``DecimalConfig`` reads
    them in ``default_factory``, i.e. once, at construction — and each adapter
    then freezes the answer into a module-level ``SkillRouter`` singleton. Set
    them any later and the child would be graded on the mode it did not run.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    parts = [str(_REPO_ROOT)] + ([existing] if existing else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    # Strip first, then apply. `default` must mean "whatever the ADAPTER resolves
    # to", and an inherited DECIMALAI_INJECT_SKILL_BODY in the developer's shell
    # would silently redefine the whole per-driver matrix — the same class of
    # "it passes if you run it this way" the registry-order comment in
    # conftest.py refuses.
    for key in _DELIVERY_ENV_KEYS:
        env.pop(key, None)
    env.update(mode_env(delivery_mode))
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


def run_driver_in_child(
    driver: Driver, delivery_mode: str = DELIVERY_DEFAULT
) -> Observation:
    """Run ``driver`` in a fresh process and return the capture.

    ``delivery_mode`` picks which body channel the run is held to. Anything but
    ``default`` runs the SKILLS PHASE ONLY: the other six phases do not touch the
    rail, so paying for them per mode would triple the suite's runtime to grade
    nothing new. The unrun phases come back ``ran=False`` with a reason, never as
    empty phases that ran.
    """
    label = driver.name if delivery_mode == DELIVERY_DEFAULT else (
        f"{driver.name}[{delivery_mode}]"
    )
    workdir = tempfile.mkdtemp(prefix=f"conformance-child-{driver.name}-")
    out_path = Path(workdir) / "observation.json"
    cmd = [sys.executable, str(_CHILD), driver.name, str(out_path), delivery_mode]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            env=_child_env(delivery_mode),
            capture_output=True,
            text=True,
            timeout=_timeout_seconds(),
        )
    except subprocess.TimeoutExpired as exc:
        raise DriverProcessError(
            f"{label}: the driver process did not finish within "
            f"{_timeout_seconds():.0f}s and was killed — nothing was graded for this "
            f"framework.\n"
            f"stdout: {_tail(exc.stdout)}\n"
            f"stderr: {_tail(exc.stderr)}"
        ) from exc

    if proc.returncode != 0 or not out_path.exists():
        raise DriverProcessError(
            f"{label}: the driver process exited {proc.returncode} "
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
            f"{label}: the driver process wrote a capture this parent cannot "
            f"read ({exc}) — nothing was graded for this framework.\n"
            f"stderr: {_tail(proc.stderr)}"
        ) from exc
    observation = decode_observation(payload, driver)
    got = observation.ctx.delivery_mode
    if got != delivery_mode:
        # The one failure the capture itself can hide: a child that ran the
        # adapter's default and came back looking like a mode. Grading that
        # would report a channel nobody switched on.
        raise DriverProcessError(
            f"{label}: the child returned a capture stamped delivery_mode={got!r} "
            f"but was asked for {delivery_mode!r} — nothing was graded for this cell."
        )
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


def run_jobs(
    jobs_: Sequence[Tuple[Driver, str]],
) -> Tuple[Dict[Tuple[str, str], Observation], Dict[Tuple[str, str], str]]:
    """Run each ``(driver, delivery_mode)`` in its own process.

    Returns ``(observations, failures)``, both keyed by ``(driver name, mode)``.
    A crashed child lands in ``failures``; the caller turns that into a hard
    error at lookup time. It is never silently dropped.
    """
    observed: Dict[Tuple[str, str], Observation] = {}
    failures: Dict[Tuple[str, str], str] = {}
    if not jobs_:
        return observed, failures

    order: List[Tuple[str, str]] = [(d.name, m) for d, m in jobs_]
    with ThreadPoolExecutor(max_workers=jobs(len(jobs_))) as pool:
        futures = {
            (d.name, m): pool.submit(run_driver_in_child, d, m) for d, m in jobs_
        }
    for key in order:
        try:
            observed[key] = futures[key].result()
        except DriverProcessError as exc:
            failures[key] = str(exc)
        except Exception as exc:  # noqa: BLE001 - a harness bug is still that driver's error
            failures[key] = (
                f"{key[0]}[{key[1]}]: harness could not run the driver process: {exc!r}"
            )
    return observed, failures


def run_drivers(
    drivers: Sequence[Driver],
) -> Tuple[Dict[str, Observation], Dict[str, str]]:
    """The per-driver matrix's jobs: every driver, in its adapter's own default
    delivery mode. Keyed by driver name, as the contract matrix has always been.
    """
    observed, failures = run_jobs([(d, DELIVERY_DEFAULT) for d in drivers])
    return (
        {name: obs for (name, _), obs in observed.items()},
        {name: why for (name, _), why in failures.items()},
    )
