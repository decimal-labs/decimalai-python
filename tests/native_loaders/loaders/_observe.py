"""Observation helpers shared by the loader modules (child side; stdlib + PyYAML).

Nothing here grades anything. These functions collect what a loader did — what it
logged, what file it pointed at, whether a bit is set — in one shape, so
``contract.py`` reads every loader the same way.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import stat
import warnings
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .. import oracle
from ..source import skills_dir

SCENARIO = oracle.SCENARIO


@dataclass
class Ctx:
    """What a loader child is handed."""

    project: Path
    home: Path
    #: install name -> the lock-v2-shaped entry the folder is graded against.
    expect: Dict[str, dict]
    plugin_name: str = oracle.PLUGIN_NAME
    scenario: oracle.Scenario = field(default_factory=lambda: SCENARIO)

    @property
    def names(self) -> List[str]:
        return sorted(self.expect)

    def by_variant(self, variant: str) -> List[str]:
        return sorted(n for n, e in self.expect.items()
                      if (e.get("render") or {}).get("variant") == variant)

    def skills_dir(self, **kw: Any) -> Path:
        # As the documented snippets call it: from the working directory, which the
        # parent makes the project root.
        return skills_dir(**kw)


def new_observation(loader: str, *dists: str) -> Dict[str, Any]:
    return {
        "loader": loader,
        "versions": versions(*dists),
        "errors": {},
        "listed": None,
        "offered": None,
        "channel": [],
        "body_text": None,
        "body": None,
        "resource_text": None,
        "resource_exact": None,
        "script_stdout": None,
        "stamp": {"via": None, "path": None, "metadata": None},
        "user_only": {"offered": None, "refusal": None},
        "exec_bit": {"path": None, "source": None, "copy": None},
        "claude": None,
        "proofs": {},
        "notes": {},
    }


def versions(*dists: str) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for d in dists:
        try:
            out[d] = metadata.version(d)
        except metadata.PackageNotFoundError:
            out[d] = None
    return out


def describe(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= 600 else text[:597] + "..."


def attempt(obs: Dict[str, Any], stage: str, fn: Callable[[], Any], default: Any = None) -> Any:
    """Run one stage; an exception is recorded under ``errors[stage]``, never raised."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - the observation is the point
        obs["errors"][stage] = describe(exc)
        return default


class Channel(logging.Handler):
    """WARNING-and-above records from a loader's own loggers."""

    def __init__(self, prefixes: Iterable[str], level: int = logging.WARNING) -> None:
        super().__init__(level)
        self.prefixes = tuple(prefixes)
        self.lines: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(self.prefixes):
            try:
                msg = record.getMessage()
            except Exception:  # noqa: BLE001
                msg = str(record.msg)
            self.lines.append(f"{record.levelname} {record.name}: {msg}"[:400])


def watch_loggers(*prefixes: str) -> Channel:
    """Collect WARNING+ records from the loggers under ``prefixes`` (the loader's channel)."""
    handler = Channel(prefixes)
    logging.getLogger().addHandler(handler)
    for p in prefixes:
        lg = logging.getLogger(p)
        if lg.level == logging.NOTSET or lg.level > logging.WARNING:
            lg.setLevel(logging.WARNING)
        if not lg.propagate:
            lg.addHandler(handler)
    return handler


class WarningsLog:
    """``warnings.warn`` calls whose category is one of ``categories``."""

    def __init__(self, *categories: type) -> None:
        self.categories = categories or (UserWarning,)
        self.lines: List[str] = []
        warnings.simplefilter("always")
        self._orig = warnings.showwarning
        warnings.showwarning = self._show  # type: ignore[assignment]

    def _show(self, message, category, filename, lineno, file=None, line=None) -> None:
        if issubclass(category, self.categories):
            self.lines.append(f"{category.__name__}: {message}"[:400])


def split_frontmatter(text: str) -> Tuple[Optional[str], Optional[str]]:
    """``(frontmatter, body)`` of a SKILL.md text, or ``(None, None)``."""
    if not text.startswith("---\n"):
        return None, None
    end = text.find("\n---\n", 3)
    if end < 0:
        return None, None
    return text[4:end + 1], text[end + 5:]


def stamp_of(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """The ``metadata`` map a SKILL.md text declares (values as YAML typed them)."""
    if not text:
        return None
    import yaml

    fm, _ = split_frontmatter(text)
    if fm is None:
        return None
    try:
        data = yaml.safe_load(fm) or {}
    except yaml.YAMLError:
        return None
    md = data.get("metadata") if isinstance(data, dict) else None
    return md if isinstance(md, dict) else {}


def body_of(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    return split_frontmatter(text)[1]


def read_text(path: Optional[os.PathLike]) -> Optional[str]:
    if not path:
        return None
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def executable(path: Optional[os.PathLike]) -> Optional[bool]:
    if not path:
        return None
    try:
        return bool(os.stat(path).st_mode & stat.S_IXUSR)
    except OSError:
        return None


_LINE_NO = re.compile(r"^ *\d+(?:\t|→)", re.M)


def without_line_numbers(text: Optional[str]) -> Optional[str]:
    """Undo a file viewer's ``<n>\\t`` / ``<n>→`` line prefixes."""
    if text is None:
        return None
    return _LINE_NO.sub("", text)


def sha256(text: Optional[str]) -> Optional[str]:
    return None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()


def unique(seq: Iterable[str]) -> List[str]:
    return sorted(set(seq))
