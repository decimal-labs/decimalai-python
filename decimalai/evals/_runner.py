"""`decimalai.evaluate()`: run a set of cases through your agent, check each run, compare setups.

::

    served = decimalai.evaluate(agent, "cases.jsonl", evals=CHECKS)
    draft = decimalai.evaluate(agent, "cases.jsonl", evals=CHECKS, prompt_version=4)
    print(draft.compare(served))

Each case is one ordinary run of the agent, made an evaluation run (`decimalai._evaluation`): it can
use a saved prompt version other than the served one, or leave skills out, without changing what
production serves. Its trace goes to DecimalAI marked ``source_type="evaluation"``, kept out of the
agent's own numbers, with the checks' results attached (``RunTrace.eval_scores``): the run's trace
is held until its checks have run, then sent once.

A check is an ``@eval`` function of one :class:`~decimalai.evals.TraceData`, which carries the case
row as ``trace.case``. It returns True/False, a score, an ``EvalResult``, or None when the case is
not its concern. A check that raises is reported as an error on that case, never as a skip.
Sampling (``@eval(sampling_rate=...)``) is for production traffic; evaluate() runs every check on
every case.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import DecimalEval, EvalResult, TraceData, trace_to_trace_data

#: The note under every comparison: eight hand-written cases, one run each, are a smoke test.
_NOISE_NOTE = "{n} cases, one run each: a smoke test, not a measurement. Models vary run to run."


@dataclass
class CaseResult:
    """One case's run and what each check made of it.

    Attributes:
        id: The case's ``id``, or ``case <line>`` when it has none.
        input: The case's ``input``.
        output: The run's final answer (``RunResult.final_output`` as text); None when it errored.
        trace_id: The run's DecimalAI trace; None when the run produced none.
        trace_url: Its page in the dashboard.
        results: Check name -> its result, or None when the check skipped this case.
        check_errors: Check name -> the exception a check raised on this case.
        error: Why the run itself failed, or None.
        prompt_version: The saved prompt version the run used, read off its trace.
    """

    id: str
    input: Any
    output: Optional[str] = None
    trace_id: Optional[str] = None
    trace_url: Optional[str] = None
    results: Dict[str, Optional[EvalResult]] = field(default_factory=dict)
    check_errors: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    prompt_version: Optional[int] = None

    def outcome(self, check: str) -> str:
        """``"pass"``, ``"fail"``, ``"skip"`` (not this case's concern) or ``"error"``."""
        if self.error is not None or check in self.check_errors:
            return "error"
        result = self.results.get(check)
        if result is None:
            return "skip"
        return "pass" if result.passed else "fail"


class EvalReport:
    """What :func:`evaluate` found: every case, and each check's counts.

    Attributes:
        agent: The DecimalAI agent evaluated.
        label: This setup's name in reports (``"served"``, ``"prompt v4"``, ...).
        excluded_skills: The skills its runs left out.
        checks: The checks' names, in the order given.
        cases: One :class:`CaseResult` per case, in the cases' order.
    """

    def __init__(self, *, agent: str, label: str, excluded_skills: Sequence[str],
                 checks: Sequence[str], cases: Sequence[CaseResult]) -> None:
        self.agent = agent
        self.label = label
        self.excluded_skills = list(excluded_skills)
        self.checks = list(checks)
        self.cases = list(cases)

    def __repr__(self) -> str:
        return f"EvalReport(agent={self.agent!r}, label={self.label!r}, cases={len(self.cases)})"

    @property
    def prompt_version(self) -> Optional[int]:
        """The saved prompt version the runs used, read off their traces; None when they differ or none says."""
        versions = {case.prompt_version for case in self.cases if case.prompt_version is not None}
        return versions.pop() if len(versions) == 1 else None

    @property
    def summary(self) -> Dict[str, Dict[str, int]]:
        """Check name -> ``{"passed", "failed", "skipped", "errors"}`` over the cases."""
        out: Dict[str, Dict[str, int]] = {}
        for check in self.checks:
            counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
            for case in self.cases:
                outcome = case.outcome(check)
                counts[{"pass": "passed", "fail": "failed", "skip": "skipped", "error": "errors"}[outcome]] += 1
            out[check] = counts
        return out

    @property
    def errors(self) -> List[CaseResult]:
        """The cases whose run failed."""
        return [case for case in self.cases if case.error is not None]

    def _heading(self) -> str:
        if self.label == "served" and self.prompt_version is not None:
            return f"served (prompt v{self.prompt_version})"
        return self.label

    def __str__(self) -> str:
        width = max([len(c) for c in self.checks] + [24]) + 2
        lines = [f"{self.agent} · {self._heading()} · {len(self.cases)} cases"]
        for check, counts in self.summary.items():
            lines.append(f"  {check.ljust(width)}{_score(counts)}")
        failed = [(case, check) for case in self.cases for check in self.checks
                  if case.outcome(check) == "fail"]
        if failed:
            lines += ["", "Failed checks"]
            ids = max(len(case.id) for case, _check in failed) + 3
            names = max(len(check) for _case, check in failed) + 3
            for case, check in failed:
                reason = (case.results[check].reason or "").strip()
                lines.append(f"  {case.id.ljust(ids)}{check.ljust(names)}".rstrip()
                             + (f"   {reason}" if reason else "")
                             + (f"   {case.trace_url}" if case.trace_url else ""))
        broken = [(case, check, err) for case in self.cases for check, err in case.check_errors.items()]
        if broken:
            lines += ["", "Checks that raised"]
            lines += [f"  {case.id}   {check}   {err}" for case, check, err in broken]
        if self.errors:
            lines += ["", "Cases that did not run"]
            lines += [f"  {case.id}   {case.error}" for case in self.errors]
        return "\n".join(lines)

    def compare(self, baseline: "EvalReport") -> "Comparison":
        """This setup beside ``baseline``, case by case (cases are matched by ``id``)."""
        return Comparison(baseline, self)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready: the summary and every case."""
        return {
            "agent": self.agent, "label": self.label, "prompt_version": self.prompt_version,
            "excluded_skills": self.excluded_skills, "summary": self.summary,
            "cases": [{
                "id": c.id, "input": c.input, "output": c.output, "trace_id": c.trace_id,
                "trace_url": c.trace_url, "error": c.error, "prompt_version": c.prompt_version,
                "check_errors": dict(c.check_errors),
                "results": {name: (dataclasses.asdict(r) if r is not None else None) for name, r in c.results.items()},
            } for c in self.cases],
        }


class Comparison:
    """Two reports of the same cases side by side: each check's counts, and the cases that changed.

    It reports counts. It does not call a difference an effect: a handful of cases, run once
    each, move between runs of the same setup too.
    """

    def __init__(self, baseline: EvalReport, candidate: EvalReport) -> None:
        self.baseline = baseline
        self.candidate = candidate

    @property
    def changed(self) -> List[Tuple[str, str, str, str, Optional[str]]]:
        """``(case id, check, baseline outcome, candidate outcome, candidate trace url)`` that differ."""
        before = {case.id: case for case in self.baseline.cases}
        out = []
        for case in self.candidate.cases:
            old = before.get(case.id)
            if old is None:
                continue
            for check in self.candidate.checks:
                if check in self.baseline.checks and old.outcome(check) != case.outcome(check):
                    out.append((case.id, check, old.outcome(check), case.outcome(check), case.trace_url))
        return out

    def __str__(self) -> str:
        a, b = self.baseline, self.candidate
        checks = [c for c in b.checks if c in a.checks]
        width = max([len(c) for c in checks] + [24]) + 2
        left = max(len(a._heading()), 8) + 3
        shared = {c.id for c in a.cases} & {c.id for c in b.cases}
        lines = [f"{b.agent} · {len(shared)} cases".ljust(width + 2) + a._heading().ljust(left) + b._heading()]
        sa, sb = a.summary, b.summary
        for check in checks:
            delta = ""
            scored_a = sa[check]["passed"] + sa[check]["failed"]
            scored_b = sb[check]["passed"] + sb[check]["failed"]
            moved = sb[check]["passed"] - sa[check]["passed"]
            if scored_a == scored_b and moved:
                delta = f"     {'+' if moved > 0 else '−'}{abs(moved)}"
            lines.append(f"  {check.ljust(width)}{_score(sa[check]).ljust(left)}{_score(sb[check])}{delta}")
        if self.changed:
            lines += ["", "Cases that changed"]
            ids = max(len(row[0]) for row in self.changed) + 3
            names = max(len(row[1]) for row in self.changed) + 3
            for case_id, check, old, new, url in self.changed:
                lines.append(f"  {case_id.ljust(ids)}{check.ljust(names)}{old} → {new}" + (f"   {url}" if url else ""))
        only = ({c.id for c in a.cases} | {c.id for c in b.cases}) - shared
        if only:
            lines += ["", f"{len(only)} case(s) are in only one report and are not compared: {', '.join(sorted(only))}"]
        lines += ["", _NOISE_NOTE.format(n=len(shared))]
        return "\n".join(lines)


def _score(counts: Dict[str, int]) -> str:
    """``passed/scored``, with errors said when there are any."""
    text = f"{counts['passed']}/{counts['passed'] + counts['failed']}"
    return f"{text} ({counts['errors']} errored)" if counts["errors"] else text


# ── cases and checks ─────────────────────────────────────────────────────────


def _load_cases(cases: Any) -> List[Dict[str, Any]]:
    """Case rows from a .jsonl path, a ``decimalai.Dataset`` or a list of dicts, each with an id."""
    from ..dataset import Dataset

    lines: Optional[List[int]] = None
    if isinstance(cases, (str, Path)):
        rows, lines = _read_jsonl(Path(cases))
    elif isinstance(cases, Dataset):
        rows = list(cases.rows)
    elif isinstance(cases, Iterable) and not isinstance(cases, (bytes, dict)):
        rows = list(cases)
    else:
        raise TypeError("cases takes a path to a .jsonl file, a decimalai.Dataset, or a list of dicts")
    if not rows:
        raise ValueError("there are no cases to run")
    out: List[Dict[str, Any]] = []
    seen = set()
    for i, row in enumerate(rows):
        where = f"line {lines[i]}" if lines else f"case {i + 1}"
        if not isinstance(row, dict):
            raise ValueError(f"{where}: a case is a JSON object, got {type(row).__name__}")
        if "input" not in row or row["input"] in (None, ""):
            raise ValueError(f"{where}: a case needs an \"input\" (the message to send the agent)")
        case_id = str(row.get("id") or where.replace("line", "case"))
        if case_id in seen:
            raise ValueError(f"{where}: two cases are named {case_id!r}; give each its own id")
        seen.add(case_id)
        out.append({**row, "id": case_id})
    return out


def _read_jsonl(path: Path) -> Tuple[List[Any], List[int]]:
    """Rows of a .jsonl file and their line numbers; blank lines and ``#`` comments skipped."""
    rows, lines = [], []
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            lines.append(lineno)
    return rows, lines


def _checks(evals: Any) -> List[DecimalEval]:
    """The checks, each a ``DecimalEval``; a plain function of one TraceData is wrapped."""
    if evals is None:
        raise TypeError("evaluate() needs evals=[...]: the @eval checks to run on each case")
    if callable(evals):
        evals = [evals]
    out = [ev if isinstance(ev, DecimalEval) else DecimalEval(ev) for ev in evals]
    names = [ev.name for ev in out]
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        raise ValueError(f"two checks are named {', '.join(twice)}; give each its own @eval(name=...)")
    return out


async def _run_check(check: DecimalEval, trace: TraceData) -> Tuple[Dict[str, Optional[EvalResult]], Optional[str]]:
    """One check on one run: its results by name, or the exception it raised."""
    try:
        raw = check.fn(trace)
        if inspect.isawaitable(raw):
            raw = await raw
    except Exception as exc:  # noqa: BLE001 - reported on the case, never a skip
        return {}, f"{type(exc).__name__}: {exc}"
    coerced = check._coerce_result(raw)
    if isinstance(coerced, dict):
        return dict(coerced), None
    return {check.name: coerced}, None


# ── running ──────────────────────────────────────────────────────────────────


def _target(agent: Any) -> Tuple[Any, Callable[..., Any]]:
    """``(agent_config, run one case)`` for the agent given, or a TypeError saying what fits."""
    module = type(agent).__module__ or ""
    if module == "agents" or module.startswith("agents."):
        from ..openai_agents import _evaluation_target

        return _evaluation_target(agent)
    raise TypeError(
        "evaluate() runs an OpenAI Agents SDK Agent whose instructions come from "
        "decimalai.openai_agents.agent_config(); got a " + type(agent).__name__)


def _cause(exc: BaseException, kind: type) -> Optional[BaseException]:
    """``exc`` or the first exception it was raised from that is a ``kind``."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, kind):
            return exc
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return None


async def evaluate_async(
    agent: Any,
    cases: Any,
    evals: Any,
    *,
    prompt_version: Optional[int] = None,
    exclude_skills: Optional[Sequence[str]] = None,
    context: Optional[Callable[[Dict[str, Any]], Any]] = None,
    label: Optional[str] = None,
    concurrency: int = 4,
) -> EvalReport:
    """:func:`evaluate` for code already on an event loop (a notebook, an async app)."""
    from .._evaluation import EvaluationNotApplied, Override, Trial, block, trace_url

    config, run_case = _target(agent)
    override = Override.make(config.name, prompt_version, exclude_skills)
    rows = _load_cases(cases)
    checks = _checks(evals)
    if not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("concurrency takes a whole number of cases to run at once, 1 or more")
    gate = asyncio.Semaphore(concurrency)
    from ..openai_agents import _submit_trace

    #: Check name -> the result names it produced (one, or a dict result's several), in order.
    produced: Dict[str, List[str]] = {}

    async def one(row: Dict[str, Any]) -> Tuple[CaseResult, Optional[BaseException]]:
        held: Dict[str, Any] = {}

        def hold(trace: Any) -> bool:
            held["trace"] = trace
            return True

        result = CaseResult(id=row["id"], input=row["input"])
        refused: Optional[BaseException] = None
        async with gate:
            trial = Trial(override, hold=hold)
            try:
                with block(trial):
                    run_context = context(row) if context is not None else row.get("context")
                    outcome = await run_case(row["input"], run_context)
                final = getattr(outcome, "final_output", outcome)
                result.output = final if isinstance(final, str) or final is None else str(final)
            except Exception as exc:  # noqa: BLE001 - one case's failure is that case's
                refused = _cause(exc, EvaluationNotApplied)
                shown = refused or exc
                result.error = f"{type(shown).__name__}: {shown}"
            trace = held.get("trace")
            if trace is None:
                if result.error is None:
                    result.error = ("the run sent no trace, so it could not be checked: OpenAI Agents "
                                    "tracing is off (OPENAI_AGENTS_DISABLE_TRACING or "
                                    "RunConfig(tracing_disabled=True))")
                return result, refused
            try:
                result.trace_id = str(trace.id)
                result.trace_url = trace_url(result.trace_id)
                result.prompt_version = _prompt_version_of(trace, config.name)
                if result.error is None:
                    data = dataclasses.replace(trace_to_trace_data(trace), case=dict(row))
                    if not data.output and result.output:
                        # A model that records no response (a custom Model): the run's own answer.
                        data = dataclasses.replace(data, output=result.output)
                    scores: List[Dict[str, Any]] = []
                    for check in checks:
                        named, error = await _run_check(check, data)
                        if error is not None:
                            result.check_errors[check.name] = error
                            continue
                        result.results.update(named)
                        seen = produced.setdefault(check.name, [])
                        seen.extend(name for name in named if name not in seen)
                        for name, value in named.items():
                            scores.extend(_score_dicts(check, name, value))
                    trace.eval_scores = scores
            except Exception as exc:  # noqa: BLE001 - the trace still goes out, the case says why
                result.error = f"could not check the run's trace: {type(exc).__name__}: {exc}"
            finally:
                _submit_trace(trace)
        return result, refused

    outcomes = await asyncio.gather(*(one(row) for row in rows))
    results = [result for result, _refused in outcomes]
    refusals = [refused for _result, refused in outcomes if refused is not None]
    if refusals and len(refusals) == len(rows):
        # Nothing ran on the setup asked for (an unknown version or skill, an older backend):
        # a report of N identical errors would only hide that.
        raise refusals[0]
    names = [name for check in checks for name in (produced.get(check.name) or [check.name])]
    return EvalReport(agent=config.name, label=label or override.label,
                      excluded_skills=override.exclude_skills, checks=names, cases=results)


def _score_dicts(check: DecimalEval, name: str, value: Optional[EvalResult]) -> List[Dict[str, Any]]:
    """One named result as the score dicts the trace carries."""
    if value is None:
        return []
    scores = check._score_dicts({name: value})
    for score in scores:
        score["source"] = "evaluate"
    return scores


def _prompt_version_of(trace: Any, agent: str) -> Optional[int]:
    """The saved prompt version ``agent``'s run used, from the prompt record on its trace's spans."""
    for span in getattr(trace, "spans", None) or []:
        attributes = getattr(span, "attributes", None) or {}
        if attributes.get("decimal.agent") in (None, agent) and isinstance(attributes.get("decimal.prompt.version"), int):
            return attributes["decimal.prompt.version"]
    return None


def _default_loop() -> asyncio.AbstractEventLoop:
    """This thread's default event loop, made (and left open) when it has none.

    The one ``Runner.run_sync`` runs on, chosen the same way: the OpenAI client the Agents SDK
    shares keeps connections bound to the loop that opened them, so a script that calls
    ``evaluate()`` and ``Runner.run_sync()`` must give both the same loop. A fresh
    ``asyncio.run`` per call leaves the next run's connections on a closed loop.
    """
    policy = asyncio.get_event_loop_policy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            loop = policy.get_event_loop()
        except RuntimeError:
            loop = policy.new_event_loop()
            policy.set_event_loop(loop)
    if loop.is_closed():
        loop = policy.new_event_loop()
        policy.set_event_loop(loop)
    return loop


def evaluate(
    agent: Any,
    cases: Any,
    evals: Any,
    *,
    prompt_version: Optional[int] = None,
    exclude_skills: Optional[Sequence[str]] = None,
    context: Optional[Callable[[Dict[str, Any]], Any]] = None,
    label: Optional[str] = None,
    concurrency: int = 4,
) -> EvalReport:
    """Run each case through ``agent``, check each run, and report.

    ::

        served = decimalai.evaluate(agent, "cases.jsonl", evals=[looked_up_the_order])
        draft = decimalai.evaluate(agent, "cases.jsonl", evals=[looked_up_the_order], prompt_version=4)
        print(draft.compare(served))

    Each case is one run of the agent, traced to DecimalAI with the checks' results attached
    and kept out of the agent's own numbers. ``prompt_version`` and ``exclude_skills`` change
    what these runs get, not what production serves.

    Args:
        agent: An OpenAI Agents SDK ``Agent`` whose instructions come from
            ``decimalai.openai_agents.agent_config()``.
        cases: A path to a ``.jsonl`` file (one JSON object per line), a
            ``decimalai.Dataset``, or a list of dicts. Each case needs an ``input`` (the
            message); ``id`` names it in reports; ``context`` is passed to
            ``Runner.run(context=...)``. Every other field is yours: checks read the row
            as ``trace.case``.
        evals: The ``@eval`` checks to run on each case's trace.
        prompt_version: Run every case on this saved prompt version instead of the served one.
        exclude_skills: Leave these skills out of every case: not listed, not loadable.
        context: ``case -> object`` for ``Runner.run(context=...)``, for a typed context; by
            default the case's ``context`` value as is.
        label: This setup's name in reports. Default: ``"served"``, ``"prompt v4"``,
            ``"without refund-policy"``.
        concurrency: How many cases run at once.

    Returns:
        An :class:`~decimalai.evals.EvalReport`; ``print()`` it, or ``.compare(other)``.

    Raises:
        decimalai.EvaluationNotApplied: no case could run on the setup asked for (an unknown
            version or skill name, no API key, a DecimalAI from before evaluation runs). A
            single case that could not is reported on that case instead.
        RuntimeError: called inside a running event loop: ``await evaluate_async(...)`` there.
        TypeError, ValueError: ``agent``, ``cases`` or ``evals`` are not what this takes.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("evaluate() can't run inside a running event loop (a notebook, an async "
                           "app): use `await decimalai.evaluate_async(...)` there")
    return _default_loop().run_until_complete(evaluate_async(
        agent, cases, evals, prompt_version=prompt_version, exclude_skills=exclude_skills,
        context=context, label=label, concurrency=concurrency,
    ))
