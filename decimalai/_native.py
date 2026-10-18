"""Seeing a framework's own skill loader load a skill, and saying which version it was.

Eight frameworks read ``SKILL.md`` folders with a loader of their own (Claude Code, ADK's
``SkillToolset``, CrewAI's ``Agent(skills=...)``, the OpenAI sandbox's ``Skills``, Deep
Agents, Microsoft Agent Framework, the Pydantic AI harness, Strands). When one of them hands
a skill to the model, the hook DecimalAI keeps in that framework passes what it saw to
:func:`record`, and this module turns it into an activation carrying the registry identity
of the bytes the model received.

**Identity comes from the delivered bytes, never from re-reading the file.** A loader that
caches keeps serving the old version after ``decimalai skills update`` wrote the new one, so
the folder on disk is the wrong witness. The order, first hit wins:

1. the delivered copy's stamp (``metadata.decimalai-*``), when its body hash is the delivered
   body's;
2. the lock, by the delivered body's hash;
3. every body hash this process has seen in a lock or a verified stamp, which attributes the
   old version after the lock moved on;
4. the skill alone, when it is known by stamp, name or path but the body is none of its known
   versions: ``skill_id`` and ``body_sha256``, no ``version_id``. The platform resolves the
   version from those if it can; if it cannot, the activation stays unverified;
5. a skill DecimalAI did not install: its name and body hash, which the platform resolves as
   it resolves the router rail's ``{name, hash}``.

One run reports one activation per skill: a repeat load is counted once, and if a run loads
two versions of one skill the first is reported. :class:`NativeLoads` collects a run's loads
and writes them onto that run's :class:`~decimalai.schema.trace.RunTrace`.
"""

from __future__ import annotations

import logging
import os
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from skillevaluation import skillfmt

from .skillpkg.identity import ManagedIndex, ManagedSkill, _norm_hash, current_index

logger = logging.getLogger("decimalai.native")

#: The stamp keys this module reads, under the frontmatter's ``metadata:``.
STAMP_SKILL_ID = "decimalai-skill-id"
STAMP_VERSION_ID = "decimalai-version-id"
STAMP_BODY_SHA256 = "decimalai-body-sha256"
STAMP_NAME = "decimalai-name"

#: How an activation was identified (:attr:`Activation.how`).
BY_STAMP = "stamp"
BY_LOCK = "lock"
BY_SEEN = "seen"
UNVERIFIED = "unverified"
UNMANAGED = "unmanaged"

#: Body hashes remembered across lock changes. A process installs a few hundred skills at
#: most; the cap only keeps a pathological one bounded.
_SEEN_MAX = 4096


@dataclass(frozen=True)
class Load:
    """One skill load, as a framework hook saw it.

    Give whatever the loader exposes; the more, the better the identity.

    Attributes:
        framework: the loader's short name, for logs (``"adk"``, ``"strands"``).
        name: the name the loader reported: an install name, a Claude Code plugin skill
            name (``<plugin>:<install name>``), or a path.
        text: the ``SKILL.md`` text the model received, frontmatter and all, or just its
            body when the loader strips the frontmatter.
        body: the body alone, when the loader hands it over apart from its metadata. Wins
            over ``text``.
        metadata: the frontmatter's ``metadata`` map as the loader parsed it. Wins over the
            metadata in ``text``.
        path: the skill's folder or its ``SKILL.md``, when the loader says where it read it.
    """

    framework: str = ""
    name: Optional[str] = None
    text: Optional[str] = None
    body: Optional[str] = None
    metadata: Optional[Mapping[str, Any]] = None
    path: Optional[str] = None


@dataclass(frozen=True)
class Activation:
    """Which skill, and which version of it, one load delivered.

    Attributes:
        name: the name the trace reports it by (:attr:`ManagedSkill.wire_name` for a
            DecimalAI-installed skill; the loader's name otherwise).
        skill_id, version_id: the registry identity; ``version_id`` is None when the
            delivered body is not a version this process can name.
        body_sha256: :func:`skillfmt.body_sha256` of the delivered body, or None when the
            hook could not see it.
        how: one of :data:`BY_STAMP`, :data:`BY_LOCK`, :data:`BY_SEEN`, :data:`UNVERIFIED`,
            :data:`UNMANAGED`.
    """

    name: str
    skill_id: Optional[str]
    version_id: Optional[str]
    body_sha256: Optional[str]
    how: str

    @property
    def verified(self) -> bool:
        """Whether the version is known to be the one the model received."""
        return self.version_id is not None

    def entry(self) -> Dict[str, Any]:
        """The ``active_skills`` entry: only the keys that have a value."""
        out: Dict[str, Any] = {"name": self.name}
        if self.skill_id:
            out["skill_id"] = self.skill_id
        if self.version_id:
            out["version_id"] = self.version_id
        if self.body_sha256:
            out["body_sha256"] = self.body_sha256
        return out


# ── what was delivered ───────────────────────────────────────────────────────


def _delivered(load: Load) -> Tuple[Optional[str], Optional[Mapping[str, Any]]]:
    """``(body, metadata)`` of what the model received."""
    metadata = load.metadata if isinstance(load.metadata, Mapping) else None
    if isinstance(load.body, str):
        return load.body, metadata
    if not isinstance(load.text, str):
        return None, metadata
    try:
        frontmatter, body = skillfmt.parse_skill_md_text(load.text)
    except Exception:  # noqa: BLE001 - a loader may hand over a body with no frontmatter
        parts = skillfmt.split_frontmatter(load.text)
        return (parts[1] if parts is not None else load.text), metadata
    if metadata is None and isinstance(frontmatter.get("metadata"), Mapping):
        metadata = frontmatter["metadata"]
    return body, metadata


@dataclass(frozen=True)
class _Stamp:
    skill_id: str
    version_id: str
    body_sha256: str
    registry_name: Optional[str]


def _stamp(metadata: Optional[Mapping[str, Any]]) -> Optional[_Stamp]:
    if not metadata:
        return None
    values = [metadata.get(k) for k in (STAMP_SKILL_ID, STAMP_VERSION_ID)]
    if not all(isinstance(v, str) and v.strip() for v in values):
        return None
    digest = _norm_hash(metadata.get(STAMP_BODY_SHA256))
    if digest is None:
        return None
    name = metadata.get(STAMP_NAME)
    return _Stamp(skill_id=values[0].strip(), version_id=values[1].strip(), body_sha256=digest,
                  registry_name=name.strip() if isinstance(name, str) and name.strip() else None)


def _install_name(load: Load) -> Optional[str]:
    """The install name the loader's report implies: the name without a plugin prefix, or
    the folder of the reported path."""
    candidates: List[str] = []
    if isinstance(load.name, str) and load.name:
        name = load.name.rstrip("/\\")
        if "/" in name or os.sep in name:
            candidates.append(_folder_name(name))
        else:
            candidates.append(name.rsplit(":", 1)[-1])
    if isinstance(load.path, str) and load.path:
        candidates.append(_folder_name(load.path))
    for candidate in candidates:
        if skillfmt.is_valid_install_name(candidate):
            return candidate
    return None


def _folder_name(path: str) -> str:
    text = path.rstrip("/\\")
    if os.path.basename(text).lower() == "skill.md":
        text = os.path.dirname(text)
    return os.path.basename(text)


# ── what this process has seen ───────────────────────────────────────────────


class _Seen:
    """Every ``(skill, version)`` a body hash has named in this process.

    Fed from every lock index :func:`resolve` reads and from every stamp it verifies, and
    never pruned when the lock changes: that is the point. A body that two skills share, or
    that two versions of one skill share, names no version here.
    """

    def __init__(self, cap: int = _SEEN_MAX) -> None:
        self._cap = cap
        self._by_hash: "OrderedDict[str, Set[Tuple[str, str, str]]]" = OrderedDict()
        self._indexes: "weakref.WeakSet[ManagedIndex]" = weakref.WeakSet()
        self._lock = threading.Lock()

    def remember(self, digest: str, skill_id: str, version_id: str, wire_name: str) -> None:
        with self._lock:
            bucket = self._by_hash.setdefault(digest, set())
            bucket.add((skill_id, version_id, wire_name))
            self._by_hash.move_to_end(digest)
            while len(self._by_hash) > self._cap:
                self._by_hash.popitem(last=False)

    def remember_index(self, index: ManagedIndex) -> None:
        with self._lock:
            if index in self._indexes:
                return
            self._indexes.add(index)
        for skill in index:
            self.remember(skill.body_sha256, skill.skill_id, skill.version_id, skill.wire_name)

    def lookup(self, digest: str) -> Tuple[Optional[Tuple[str, str, str]], Optional[str]]:
        """``(the one (skill_id, version_id, wire_name), None)`` when the hash names exactly
        one version; ``(None, skill_id)`` when it names several versions of one skill;
        ``(None, None)`` otherwise."""
        with self._lock:
            bucket = set(self._by_hash.get(digest, ()))
        if len(bucket) == 1:
            return next(iter(bucket)), None
        skill_ids = {skill_id for skill_id, _, _ in bucket}
        return None, (skill_ids.pop() if len(skill_ids) == 1 else None)

    def clear(self) -> None:
        with self._lock:
            self._by_hash.clear()
            self._indexes = weakref.WeakSet()


_seen = _Seen()


def _reset_seen() -> None:
    """Forget every remembered body hash. Test seam only."""
    _seen.clear()


# ── resolution ───────────────────────────────────────────────────────────────


def _wire_name(skill_id: str, registry_name: Optional[str], install_name: Optional[str],
               index: ManagedIndex) -> Optional[str]:
    """The trace name for a stamped skill, as :attr:`ManagedSkill.wire_name` would give it."""
    locked = index.by_skill_id(skill_id)
    if locked is not None:
        return locked.wire_name
    if install_name is None:
        return registry_name
    probe = ManagedSkill(install_name=install_name, registry_name=registry_name or install_name,
                         skill_id=skill_id, version_id="", body_sha256="", path="")
    return probe.wire_name


def resolve(load: Load, *, index: Optional[ManagedIndex] = None) -> Optional[Activation]:
    """The activation one load amounts to, or None when nothing names a skill.

    Never raises for what the loader handed over, and never reads the skill's folder.
    """
    if index is None:
        index = current_index()
    _seen.remember_index(index)
    body, metadata = _delivered(load)
    digest = skillfmt.body_sha256(body) if body is not None else None
    stamp = _stamp(metadata)
    install_name = _install_name(load)

    if stamp is not None and digest is not None and stamp.body_sha256 == digest:
        name = _wire_name(stamp.skill_id, stamp.registry_name, install_name, index)
        if name:
            _seen.remember(digest, stamp.skill_id, stamp.version_id, name)
            return Activation(name, stamp.skill_id, stamp.version_id, digest, BY_STAMP)

    if digest is not None:
        locked = index.by_hash(digest)
        if locked is not None:
            return Activation(locked.wire_name, locked.skill_id, locked.version_id, digest,
                              BY_LOCK)
        one, skill_id = _seen.lookup(digest)
        if one is not None:
            return Activation(one[2], one[0], one[1], digest, BY_SEEN)
    else:
        skill_id = None

    known = _known_skill(load, stamp, skill_id, install_name, index)
    if known is not None:
        known_id, known_name = known
        return Activation(known_name, known_id, None, digest, UNVERIFIED)

    name = install_name or (load.name if isinstance(load.name, str) and load.name else None)
    if name is None:
        return None
    return Activation(name, None, None, digest, UNMANAGED)


def _known_skill(load: Load, stamp: Optional[_Stamp], seen_skill_id: Optional[str],
                 install_name: Optional[str],
                 index: ManagedIndex) -> Optional[Tuple[str, str]]:
    """``(skill_id, trace name)`` of the skill the load was, when anything says which."""
    if stamp is not None:
        name = _wire_name(stamp.skill_id, stamp.registry_name, install_name, index)
        if name:
            return stamp.skill_id, name
    for reported in (load.name, load.path):
        locked = index.get(reported) if isinstance(reported, str) else None
        if locked is not None:
            return locked.skill_id, locked.wire_name
    if install_name is not None:
        locked = index.by_install_name(install_name)
        if locked is not None:
            return locked.skill_id, locked.wire_name
    if seen_skill_id is not None:
        locked = index.by_skill_id(seen_skill_id)
        if locked is not None:
            return locked.skill_id, locked.wire_name
    return None


# ── one run's loads ──────────────────────────────────────────────────────────


class NativeLoads:
    """The native loads of one run, each skill once, the first load winning.

    Thread-safe: a framework may run tools on worker threads or in other tasks. The carrier
    that builds the run's trace owns one and calls :meth:`apply` on the finished trace.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._activations: "OrderedDict[str, Activation]" = OrderedDict()
        self._listed: List[str] = []

    def add(self, activation: Activation) -> bool:
        """Record one load. False when the skill was already recorded in this run."""
        with self._lock:
            first = self._activations.get(activation.name)
            if first is None:
                self._activations[activation.name] = activation
                return True
        if first.body_sha256 != activation.body_sha256:
            logger.debug(
                "native load: %s was loaded again in this run with another body "
                "(%s, then %s); the run keeps the first", activation.name,
                first.version_id or first.body_sha256, activation.version_id or
                activation.body_sha256)
        return False

    def listed(self, names: Iterable[str]) -> None:
        """Record names the loader put in the model's skill listing."""
        with self._lock:
            for name in names:
                if isinstance(name, str) and name and name not in self._listed:
                    self._listed.append(name)

    @property
    def activations(self) -> List[Activation]:
        with self._lock:
            return list(self._activations.values())

    @property
    def listed_names(self) -> List[str]:
        with self._lock:
            return list(self._listed)

    def __bool__(self) -> bool:
        with self._lock:
            return bool(self._activations or self._listed)

    def apply(self, trace: Any) -> None:
        """Write this run's loads onto its finished trace.

        * ``active_skills`` gains each activation's entry. An entry of the same name that
          carries no registry id is replaced (it said less about the same load); one that
          carries an id is the caller's declaration and is kept.
        * ``skills_loaded_by_agent`` and ``skills_delivered`` gain each activated name: the
          body reached the model.
        * ``skills_offered_in_prompt`` gains the listed names only. A load arrives through a
          tool call, not the prompt, so it is not evidence the name was in the prompt.
        """
        activations = self.activations
        listed = self.listed_names
        if not activations and not listed:
            return
        active = [e for e in list(getattr(trace, "active_skills", None) or [])]
        for activation in activations:
            kept = []
            declared = False
            for entry in active:
                name = entry.get("name") if isinstance(entry, dict) else entry
                if name != activation.name:
                    kept.append(entry)
                elif isinstance(entry, dict) and (entry.get("skill_id") or entry.get("version_id")):
                    kept.append(entry)
                    declared = True
            active = kept if declared else kept + [activation.entry()]
        trace.active_skills = active
        names = [a.name for a in activations]
        trace.skills_loaded_by_agent = _union(getattr(trace, "skills_loaded_by_agent", None), names)
        trace.skills_delivered = _union(getattr(trace, "skills_delivered", None), names)
        if listed:
            trace.skills_offered_in_prompt = _union(
                getattr(trace, "skills_offered_in_prompt", None), listed)


def _union(existing: Optional[Iterable[str]], more: Iterable[str]) -> List[str]:
    out = sorted({n for n in (existing or []) if isinstance(n, str)} | set(more))
    return out


# ── carriers ─────────────────────────────────────────────────────────────────

_warned_no_carrier: Set[str] = set()
_warned_lock = threading.Lock()


def record(load: Load, *, loads: Optional[NativeLoads] = None,
           index: Optional[ManagedIndex] = None) -> Optional[Activation]:
    """Resolve one load and record it on the run it belongs to.

    ``loads`` is the run's collector when the caller (an adapter) owns one. Without it the
    load goes to the run that is live here: a ``decimalai.start_trace()`` trace, else the
    OpenTelemetry run the exporter is assembling. With neither, it is dropped and a warning
    says so once per framework. Never raises.
    """
    try:
        activation = resolve(load, index=index)
    except Exception:  # noqa: BLE001 - observing must never break the agent's run
        logger.debug("native load: could not resolve %r", load.name, exc_info=True)
        return None
    if activation is None:
        return None
    if loads is not None:
        loads.add(activation)
        return activation
    try:
        if _record_on_generic_trace(activation) or _record_on_otel_run(activation):
            return activation
    except Exception:  # noqa: BLE001
        logger.debug("native load: could not record %s", activation.name, exc_info=True)
        return activation
    framework = load.framework or "a native loader"
    with _warned_lock:
        first = framework not in _warned_no_carrier
        _warned_no_carrier.add(framework)
    if first:
        logger.warning(
            "decimalai: %s loaded the skill %s, but no trace is open to record it on, so "
            "the activation is dropped. Run the agent inside decimalai.start_trace(), or "
            "turn on its DecimalAI instrumentation.", framework, activation.name)
    return activation


def _record_on_generic_trace(activation: Activation) -> bool:
    from .generic import _get_current_trace

    ctx = _get_current_trace()
    if ctx is None:
        return False
    ctx._native_loads().add(activation)
    return True


def _record_on_otel_run(activation: Activation) -> bool:
    try:
        from .otel import native_loads_for_current_run
    except ImportError:  # the otel extra is not installed
        return False
    loads = native_loads_for_current_run()
    if loads is None:
        return False
    loads.add(activation)
    return True


def _reset_warnings() -> None:
    """Forget which frameworks were warned about. Test seam only."""
    with _warned_lock:
        _warned_no_carrier.clear()
