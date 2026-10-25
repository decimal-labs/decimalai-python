"""Evaluation runs: one run's own prompt version and skill list, leaving what the agent serves alone.

``support.override(prompt_version=4, exclude_skills=["refund-policy"])`` (a block) and
``decimalai.evaluate()`` (which runs every case in one) make the runs inside them evaluation runs:

- each routes first, asking DecimalAI for its own setup. The answer carries the saved prompt
  version's text, leaves the excluded skills off the list, and the platform refuses the run's
  ``load_skill`` calls for them;
- the platform must echo what was asked. A run it cannot confirm raises
  :class:`EvaluationNotApplied` instead of running on the served setup: an evaluation that quietly
  ran on something else would report on the wrong prompt;
- its answer never becomes the agent's local copy of its skill list, so ordinary runs are untouched;
- its trace is marked ``source_type="evaluation"`` and kept out of the agent's own numbers, and the
  :class:`Trial` the block yields has its trace id.

Every binding asks :func:`active` for the trial covering its agent (``AgentBinding._begin_run``).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple


class EvaluationNotApplied(RuntimeError):
    """DecimalAI could not give an evaluation run the setup it asked for, so the run did not start."""


@dataclass(frozen=True)
class Override:
    """What one evaluation run asks for: a saved prompt version, skills to leave out, or neither.

    Neither is an evaluation run on the served setup: ``evaluate()``'s baseline.
    """

    agent: str
    prompt_version: Optional[int] = None
    exclude_skills: Tuple[str, ...] = ()

    @classmethod
    def make(cls, agent: str, prompt_version: Any = None, exclude_skills: Any = None) -> "Override":
        """Validated: a positive version number, and skill names as a list (one string is one name)."""
        if prompt_version is not None and (
                isinstance(prompt_version, bool) or not isinstance(prompt_version, int) or prompt_version < 1):
            raise ValueError(f"prompt_version takes a saved prompt version number, got {prompt_version!r}")
        names: List[str] = []
        raw: Iterable[Any] = [exclude_skills] if isinstance(exclude_skills, str) else (exclude_skills or ())
        for name in raw:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"exclude_skills takes skill names, got {name!r}")
            if name.strip() not in names:
                names.append(name.strip())
        return cls(agent=agent, prompt_version=prompt_version, exclude_skills=tuple(names))

    def request(self) -> Dict[str, Any]:
        """The fields a routing call sends (`SkillRouter.build_prompt_parts(evaluation=...)`)."""
        return {"prompt_version": self.prompt_version, "exclude_skills": list(self.exclude_skills)}

    @property
    def label(self) -> str:
        """``"served"``, ``"prompt v4"``, ``"without refund-policy"`` or ``"prompt v4 without a, b"``."""
        parts = []
        if self.prompt_version is not None:
            parts.append(f"prompt v{self.prompt_version}")
        if self.exclude_skills:
            parts.append("without " + ", ".join(self.exclude_skills))
        return " ".join(parts) or "served"

    def check(self, echo: Any, run_prompt: Any) -> None:
        """Raise unless the platform's answer is this override's (see the module docstring)."""
        if not isinstance(echo, dict):
            raise EvaluationNotApplied(
                f"DecimalAI did not confirm this evaluation run of {self.agent!r} ({self.label}): the "
                "backend predates evaluation runs, so the run would have used the served prompt and "
                "skills. Run it against a DecimalAI that supports them (decimalai 0.26+).")
        echoed = (echo.get("prompt_version"), sorted(echo.get("exclude_skills") or []))
        if echoed != (self.prompt_version, sorted(self.exclude_skills)):
            raise EvaluationNotApplied(
                f"DecimalAI answered this evaluation run of {self.agent!r} for another setup "
                f"(prompt_version={echoed[0]!r}, exclude_skills={echoed[1]!r}) than it asked for "
                f"({self.label}).")
        if self.prompt_version is not None and not (
                isinstance(run_prompt, dict) and isinstance(run_prompt.get("content"), str)):
            raise EvaluationNotApplied(
                f"DecimalAI confirmed prompt version {self.prompt_version} of {self.agent!r} but sent "
                "no text for it.")


class Trial:
    """The evaluation runs one ``override(...)`` block made, as their traces were sent.

    Attributes:
        override: What the runs asked for.
    """

    def __init__(self, override: Override, *, hold: Optional[Callable[[Any], bool]] = None) -> None:
        self.override = override
        self._hold = hold
        self._lock = Lock()
        self._trace_ids: List[str] = []

    def __repr__(self) -> str:
        return f"Trial(agent={self.override.agent!r}, {self.override.label}, traces={len(self._trace_ids)})"

    @property
    def trace_ids(self) -> List[str]:
        """Every run's DecimalAI trace id, in the order the runs ended."""
        with self._lock:
            return list(self._trace_ids)

    @property
    def trace_id(self) -> Optional[str]:
        """The last run's DecimalAI trace id; None before a run in the block has ended."""
        with self._lock:
            return self._trace_ids[-1] if self._trace_ids else None

    @property
    def trace_url(self) -> Optional[str]:
        """The last run's trace in the DecimalAI dashboard; None before a run has ended."""
        trace_id = self.trace_id
        return trace_url(trace_id) if trace_id else None

    def _take(self, trace: Any) -> bool:
        """A run of this block finished: note its trace. True: the trial holds it (evaluate() sends it)."""
        self._note(str(trace.id))
        return bool(self._hold(trace)) if self._hold is not None else False

    def _note(self, trace_id: str) -> None:
        """A run of this block will be sent as ``trace_id``."""
        with self._lock:
            self._trace_ids.append(trace_id)


#: The trials whose block the current context is inside, innermost last.
_ACTIVE: "ContextVar[Tuple[Trial, ...]]" = ContextVar("decimalai_evaluation_trials", default=())


def active(agent_name: Optional[str]) -> Optional[Trial]:
    """The innermost trial covering ``agent_name`` here, or None: an ordinary run."""
    for trial in reversed(_ACTIVE.get()):
        if trial.override.agent == agent_name:
            return trial
    return None


@contextmanager
def block(trial: Trial) -> Iterator[Trial]:
    """Runs started in this context (async tasks it starts included) are ``trial``'s."""
    token = _ACTIVE.set(_ACTIVE.get() + (trial,))
    try:
        yield trial
    finally:
        _ACTIVE.reset(token)


def trace_url(trace_id: str) -> str:
    """A trace's page in the DecimalAI dashboard, for the backend this process sends to."""
    base = "https://api.decimal.ai"
    try:
        from ._config import _config

        if _config is not None and _config.base_url:
            base = _config.base_url
    except Exception:  # noqa: BLE001 - a link is never worth an error
        pass
    return f"{dashboard_url(base)}/traces/{trace_id}"


def dashboard_url(base_url: str) -> str:
    """The dashboard for an API base URL.

    ``DECIMAL_APP_URL`` wins when set; a localhost backend maps to the conventional :3000
    frontend (it serves no UI itself); a hosted backend's ``api.`` becomes ``app.``.
    """
    explicit = os.environ.get("DECIMAL_APP_URL") or os.environ.get("DECIMALAI_APP_URL")
    if explicit:
        return explicit.rstrip("/")
    if "localhost" in base_url or "127.0.0.1" in base_url:
        return "http://localhost:3000"
    return base_url.replace("api.", "app.").replace("/api", "").rstrip("/")
