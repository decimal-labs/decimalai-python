"""Identity on the wire: what a native loader reports, as the registry's skill ids.

A skill ``decimalai skills`` installed sits in ``.agents/skills/<install-name>/``, and an
agent's loader reports it by what it sees there:

* the install name, ``acme-refund-policy``;
* the name Claude Code gives a plugin skill, ``decimalai-skills:acme-refund-policy``, where
  the prefix is the ``name`` in ``.agents/.claude-plugin/plugin.json``;
* the folder's path, or its ``SKILL.md``'s.

None of those is the registry name (``acme/refund-policy``), which is what the platform
looks skills up by, so without a translation those activations resolve to no skill.

:class:`ManagedIndex` maps each of those spellings, the locked body hash and the skill id
to what the lock recorded: the registry name, skill id, version id and body hash.
:func:`translate_payload` rewrites a trace payload with it. ``DecimalAIClient`` calls it on
every trace the SDK built, just before the POST, and nowhere else; a payload a caller built
(``ingest_raw_trace``) is sent as given.

Reading never fails because of a lock: no lock, a version 1 lock, a corrupt one or an entry
that cannot be used all mean "nothing to translate" for that skill.
"""
from __future__ import annotations

import json
import os
import re
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union

from skillevaluation import skillfmt

from . import lock as _lock
from .project import Project

__all__ = [
    "NAME_LISTS",
    "ManagedIndex",
    "ManagedSkill",
    "clear_cache",
    "current_index",
    "translate_payload",
]

PathLike = Union[str, "os.PathLike[str]"]

#: The trace fields that name skills. ``active_skills`` holds entries (dicts, or names);
#: the other three are lists of names and stay lists of names.
NAME_LISTS = ("active_skills", "skills_delivered", "skills_offered_in_prompt",
              "skills_loaded_by_agent")
_STRING_LISTS = NAME_LISTS[1:]

#: Where an activation entry carries its hash, in the order the platform reads them.
_HASH_KEYS = ("body_sha256", "hash", "content_hash")
#: An entry carrying either already says which skill it is.
_ID_KEYS = ("skill_id", "version_id")

_HEX64 = re.compile(r"[0-9a-f]{64}")
#: A SKILL.md larger than this is not read to hash it (the sender thread must not stall
#: on a file nobody installed).
_MAX_SKILL_MD = 8 * 1024 * 1024
#: Projects whose index is kept; a process nearly always runs in one.
_CACHE_SIZE = 8


@dataclass(frozen=True)
class ManagedSkill:
    """One locked skill, as the lock recorded it.

    Attributes:
        install_name: the folder's name, and the frontmatter ``name`` a loader reports.
        registry_name: the skill's name in the registry (``acme/refund-policy``).
        skill_id, version_id: the registry ids of the installed version.
        body_sha256: :func:`skillevaluation.skillfmt.body_sha256` of the installed body.
        path: the folder, absolute.
    """

    install_name: str
    registry_name: str
    skill_id: str
    version_id: str
    body_sha256: str
    path: str

    @property
    def hashed(self) -> bool:
        """Whether the install name is the disambiguated form (``pdf-tools-3f9a1c``).

        A skill gets it when its base name was taken in the project, possibly by the org's
        own skill of that name, or when its name has no usable base form. Read from the
        suffix (6 hex of the skill id's sha256), not from the registry name: a locked
        folder keeps its name when the registry renames the skill.
        """
        suffix = skillfmt.install_name_disambiguated("x", self.skill_id)[1:]  # "-<6 hex>"
        return self.install_name.endswith(suffix)

    @property
    def wire_name(self) -> str:
        """The name a trace reports this skill by.

        The registry name, so the platform's name lookups find it, except for a hashed
        install name, which is reported as itself: the registry name may also be the org's
        own skill's, which the platform keeps on that name, and the skill id says which
        skill this one is.
        """
        return self.install_name if self.hashed else self.registry_name


def _norm_hash(value: Any) -> Optional[str]:
    """64 lowercase hex from ``<hex>`` or ``sha256:<hex>``, in any case; else None."""
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if text.startswith("sha256:"):
        text = text[len("sha256:"):]
    return text if _HEX64.fullmatch(text) else None


def _norm_path(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _looks_like_path(name: str) -> bool:
    return "/" in name or os.sep in name or bool(os.altsep and os.altsep in name)


def _unique(pairs: Iterable[Tuple[str, ManagedSkill]]) -> Dict[str, ManagedSkill]:
    """``key -> skill`` for the keys exactly one skill has: a shared key names neither."""
    seen: Dict[str, Optional[ManagedSkill]] = {}
    for key, skill in pairs:
        seen[key] = skill if key not in seen else None
    return {k: v for k, v in seen.items() if v is not None}


class ManagedIndex:
    """The project's locked skills, by every name a loader may report them by.

    Build one with :meth:`load`, or get the working directory's, cached until the lock
    changes, with :func:`current_index`. An index with no skills is falsy.
    """

    def __init__(self, skills: Iterable[ManagedSkill] = (), *,
                 plugin_name: Optional[str] = None,
                 directory: Optional[PathLike] = None) -> None:
        self._skills: Tuple[ManagedSkill, ...] = tuple(
            sorted(skills, key=lambda s: s.install_name))
        #: The ``name`` in ``plugin.json``, which prefixes a Claude Code plugin skill.
        self.plugin_name = plugin_name
        #: The project directory, which relative paths are read against.
        self.directory = Path(directory) if directory is not None else None
        self._by_name = {s.install_name: s for s in self._skills}
        self._by_plugin_name = (
            {f"{plugin_name}:{s.install_name}": s for s in self._skills} if plugin_name else {})
        self._by_path = {_norm_path(s.path): s for s in self._skills}
        self._by_hash = _unique((s.body_sha256, s) for s in self._skills)
        self._by_id = _unique((s.skill_id, s) for s in self._skills)
        # install name -> (SKILL.md stat, body hash): the body as it is on disk now.
        self._observed: Dict[str, Tuple[Tuple[int, ...], Optional[str]]] = {}
        # install name -> (SKILL.md stat, (body, frontmatter) while it is the locked body).
        self._texts: Dict[str, Tuple[Tuple[int, ...],
                                     Optional[Tuple[str, Dict[str, Any]]]]] = {}
        self._observed_lock = threading.Lock()

    @classmethod
    def load(cls, project: Project) -> "ManagedIndex":
        """Read the project's lock and ``plugin.json``. Never raises for what they hold."""
        try:
            entries = _lock.locked_entries(project)
        except Exception:  # noqa: BLE001 - a symlinked lock, an unreadable file
            entries = None
        skills = [s for s in (_skill(project, name, entry)
                              for name, entry in sorted((entries or {}).items()))
                  if s is not None]
        return cls(skills, plugin_name=_plugin_name(project), directory=project.directory)

    # ── lookups ──────────────────────────────────────────────────────────────

    def get(self, name: Any) -> Optional[ManagedSkill]:
        """The skill a loader reported as ``name``: its install name, its plugin skill name
        or its folder's (or ``SKILL.md``'s) path. Names are compared exactly; the registry
        name is not a key, because a registry name in a trace may be another skill's."""
        if not isinstance(name, str) or not name:
            return None
        return (self._by_name.get(name) or self._by_plugin_name.get(name)
                or self.by_path(name))

    def by_install_name(self, name: str) -> Optional[ManagedSkill]:
        return self._by_name.get(name)

    def by_plugin_name(self, name: str) -> Optional[ManagedSkill]:
        """By ``<plugin name>:<install name>``, as Claude Code names a plugin skill."""
        return self._by_plugin_name.get(name)

    def by_path(self, path: str) -> Optional[ManagedSkill]:
        """By the folder's path or its ``SKILL.md``'s: absolute, or relative to the project
        or to the working directory. A path through a symlink is the folder it reaches."""
        if not isinstance(path, str) or not _looks_like_path(path):
            return None
        for candidate in self._path_candidates(path):
            skill = self._by_path.get(candidate)
            if skill is not None:
                return skill
        return None

    def by_hash(self, digest: str) -> Optional[ManagedSkill]:
        """By the locked body hash (``<hex>`` or ``sha256:<hex>``). None when two installed
        skills share the body, since the hash cannot say which one ran."""
        key = _norm_hash(digest)
        return self._by_hash.get(key) if key else None

    def by_skill_id(self, skill_id: str) -> Optional[ManagedSkill]:
        return self._by_id.get(skill_id) if isinstance(skill_id, str) else None

    def _path_candidates(self, path: str) -> List[str]:
        text = os.path.expanduser(path).rstrip("/\\") or path
        if os.path.basename(text).lower() == "skill.md":
            text = os.path.dirname(text)
        if os.path.isabs(text):
            return [_norm_path(text)]
        bases = [str(self.directory)] if self.directory is not None else []
        try:
            bases.append(os.getcwd())
        except OSError:  # the working directory was deleted
            pass
        return [_norm_path(os.path.join(base, text)) for base in bases]

    # ── the folder as it is now ──────────────────────────────────────────────

    def observed_body_sha256(self, skill: ManagedSkill) -> Optional[str]:
        """``body_sha256`` of the body in the skill's ``SKILL.md`` as it is on disk now.

        None when there is no regular ``SKILL.md`` to read: then no loader read this skill
        from here. Recomputed only when the file's size or times change, so a local edit is
        seen without the lock changing.
        """
        path = os.path.join(skill.path, "SKILL.md")
        try:
            st = os.stat(path)
        except OSError:
            return None
        if not stat.S_ISREG(st.st_mode) or st.st_size > _MAX_SKILL_MD:
            return None
        key = (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        with self._observed_lock:
            cached = self._observed.get(skill.install_name)
        if cached is not None and cached[0] == key:
            return cached[1]
        digest = _body_sha256_of(path)
        with self._observed_lock:
            self._observed[skill.install_name] = (key, digest)
        return digest

    def locked_text(self, skill: ManagedSkill) -> Optional[Tuple[str, Dict[str, Any]]]:
        """``(body, frontmatter)`` of the skill's ``SKILL.md`` as it is on disk now, when its
        body is still the locked version's; None when it is not (a local edit, a newer
        version), or there is no regular ``SKILL.md`` to read.

        For a loader that rewrites a body before delivering it: the locked text is what it
        rewrote. A frontmatter that does not parse reads as empty. Read again only when the
        file's size or times change.
        """
        path = os.path.join(skill.path, "SKILL.md")
        try:
            st = os.stat(path)
        except OSError:
            return None
        if not stat.S_ISREG(st.st_mode) or st.st_size > _MAX_SKILL_MD:
            return None
        key = (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        with self._observed_lock:
            cached = self._texts.get(skill.install_name)
        if cached is not None and cached[0] == key:
            return cached[1]
        text = _locked_text_of(path, skill.body_sha256)
        with self._observed_lock:
            self._texts[skill.install_name] = (key, text)
        return text

    # ── container ────────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[ManagedSkill]:
        return iter(self._skills)

    def __len__(self) -> int:
        return len(self._skills)

    def __bool__(self) -> bool:
        return bool(self._skills)

    def __repr__(self) -> str:
        return f"ManagedIndex({[s.install_name for s in self._skills]!r})"


def _skill(project: Project, name: str, entry: Mapping[str, Any]) -> Optional[ManagedSkill]:
    """The identity of one lock entry, or None when the entry cannot be trusted for it."""
    if not skillfmt.is_valid_install_name(name):
        return None
    values = [entry.get(k) for k in ("name", "skill_id", "version_id")]
    if not all(isinstance(v, str) and v for v in values):
        return None
    body = entry.get("body_sha256")
    if not isinstance(body, str) or not _HEX64.fullmatch(body):
        return None
    registry_name, skill_id, version_id = values
    return ManagedSkill(install_name=name, registry_name=registry_name, skill_id=skill_id,
                        version_id=version_id, body_sha256=body,
                        path=str(project.skill_dir(name)))


def _plugin_name(project: Project) -> Optional[str]:
    """The ``name`` in ``.agents/.claude-plugin/plugin.json``, or None."""
    try:
        data = json.loads(project.plugin_json_path.read_bytes().decode("utf-8"))
    except (OSError, ValueError):
        return None
    name = data.get("name") if isinstance(data, dict) else None
    if not isinstance(name, str) or not name.strip() or ":" in name:
        return None
    return name


def _body_sha256_of(path: str) -> Optional[str]:
    """``body_sha256`` of a SKILL.md's body (all of it when there is no frontmatter)."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read(_MAX_SKILL_MD + 1)
    except OSError:
        return None
    if len(raw) > _MAX_SKILL_MD:
        return None
    text = raw.decode("utf-8", errors="replace")
    parts = skillfmt.split_frontmatter(text)
    return skillfmt.body_sha256(parts[1] if parts is not None else text)


def _locked_text_of(path: str, locked_sha256: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """``(body, frontmatter)`` of a SKILL.md whose body hashes to ``locked_sha256``."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read(_MAX_SKILL_MD + 1)
    except OSError:
        return None
    if len(raw) > _MAX_SKILL_MD:
        return None
    text = raw.decode("utf-8", errors="replace")
    parts = skillfmt.split_frontmatter(text)
    body = parts[1] if parts is not None else text
    if skillfmt.body_sha256(body) != locked_sha256:
        return None
    try:
        frontmatter, _ = skillfmt.parse_skill_md_text(text)
    except Exception:  # noqa: BLE001 - no frontmatter, or one that does not parse
        frontmatter = {}
    return body, frontmatter


# ── the working directory's index, cached ────────────────────────────────────

_cache: Dict[str, Tuple[Tuple[Any, ...], ManagedIndex]] = {}
_cache_lock = threading.Lock()


def _stat_key(path: Path) -> Optional[Tuple[int, ...]]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_ino, st.st_size, st.st_mtime_ns)


def current_index(start: Optional[PathLike] = None) -> ManagedIndex:
    """The index of the project ``start`` (default: the working directory) belongs to.

    The project is found as ``decimalai skills`` finds it
    (:meth:`decimalai.skillpkg.project.Project.find`). The index is kept until the lock
    file (or ``plugin.json``) changes, so a trace costs a few ``stat`` calls. Never raises:
    with no lock, or one that cannot be read, the index is empty.
    """
    try:
        project = Project.find(start)
    except Exception:  # noqa: BLE001 - a deleted working directory, no home
        return ManagedIndex()
    lock_key = _stat_key(project.lock_path)
    if lock_key is None:
        return ManagedIndex()
    stamp = (lock_key, _stat_key(project.plugin_json_path))
    cache_key = str(project.directory)
    with _cache_lock:
        hit = _cache.get(cache_key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    index = ManagedIndex.load(project)
    with _cache_lock:
        _cache.pop(cache_key, None)
        _cache[cache_key] = (stamp, index)
        while len(_cache) > _CACHE_SIZE:
            _cache.pop(next(iter(_cache)))
    return index


def clear_cache() -> None:
    """Forget every cached index."""
    with _cache_lock:
        _cache.clear()


# ── the translation ──────────────────────────────────────────────────────────


def translate_payload(payload: Any, *, index: Optional[ManagedIndex] = None) -> Any:
    """``payload`` (a dumped ``RunTrace``) with the skills a loader reported by a managed
    folder's install name, plugin skill name or path given their registry identity.

    * In all four name lists (:data:`NAME_LISTS`) such a name becomes the skill's
      :attr:`~ManagedSkill.wire_name`: the registry name, so the platform's name lookups,
      the ``delivered`` count and the router join keep working, even on an older platform.
      A hashed install name stays itself. The three plain lists stay lists of names, with
      a name that two spellings became listed once.
    * In ``active_skills`` a name becomes ``{name, skill_id, version_id, body_sha256}``,
      and a dict entry gains those keys and keeps its own.
    * An entry that already carries a ``skill_id`` or ``version_id``, or a hash that is
      neither the locked body's nor the folder's current body's, is left exactly as it was:
      it is a delivery of another skill that shares the name (or the caller said which).
      Its name is then left as it is in every list, and in every other ``active_skills``
      entry: an adapter lists an activated name as loaded too, and the platform keys each
      report by its name, so a name translated in one place only would credit the locked
      skill with another body's run and count two activations.
    * A folder whose body was edited after install is credited to its skill, never to the
      locked version: the entry carries the hash of the body on disk, ``modified: true``
      and no ``version_id``.
    * A name whose folder has no ``SKILL.md`` is not translated.

    The payload is not changed; a translated copy is returned, or ``payload`` itself when
    there is nothing to translate.

    Args:
        index: the index to use (default: :func:`current_index` of the working directory).
    """
    if not isinstance(payload, dict):
        return payload
    if index is None:
        index = current_index()
    if not index:
        return payload
    out = dict(payload)
    held: frozenset = frozenset()
    active = payload.get("active_skills")
    if isinstance(active, list):
        decided = [_translate_entry(index, entry) for entry in active]
        # The names of dict entries the guard left alone, so they agree everywhere.
        held = frozenset(
            entry["name"] for entry, result in zip(active, decided)
            if result is entry and isinstance(entry, dict) and isinstance(entry.get("name"), str))
        out["active_skills"] = [entry if _entry_name(entry) in held else result
                                for entry, result in zip(active, decided)]
    for key in _STRING_LISTS:
        names = payload.get(key)
        if isinstance(names, list):
            out[key] = _translate_names(index, names, held)
    return out


def _entry_name(entry: Any) -> Optional[str]:
    if isinstance(entry, str):
        return entry
    name = entry.get("name") if isinstance(entry, dict) else None
    return name if isinstance(name, str) else None


def _present(index: ManagedIndex, name: str) -> Tuple[Optional[ManagedSkill], Optional[str]]:
    """``(skill, body hash on disk)`` for a reported name, or ``(None, None)``."""
    skill = index.get(name)
    if skill is None:
        return None, None
    observed = index.observed_body_sha256(skill)
    return (skill, observed) if observed is not None else (None, None)


def _translate_names(index: ManagedIndex, names: List[Any],
                     held: frozenset = frozenset()) -> List[Any]:
    """The names translated, except those in ``held``; each name listed once."""
    out: List[Any] = []
    seen = set()
    for item in names:
        if isinstance(item, str):
            if item not in held:
                skill, _observed = _present(index, item)
                if skill is not None:
                    item = skill.wire_name
            if item in seen:
                continue
            seen.add(item)
        out.append(item)
    return out


def _reported_hash(entry: Mapping[str, Any]) -> Tuple[bool, Optional[str]]:
    """``(carries a hash, that hash normalised)``, reading the keys as the platform does:
    the first non-empty one. A hash in an unknown spelling is carried but None."""
    for key in _HASH_KEYS:
        value = entry.get(key)
        if isinstance(value, str) and value:
            return True, _norm_hash(value)
    return False, None


def _translate_entry(index: ManagedIndex, entry: Any) -> Any:
    if isinstance(entry, str):
        skill, observed = _present(index, entry)
        if skill is None or observed is None:
            return entry
        return _identified({"name": entry}, skill, observed)
    if not isinstance(entry, dict):
        return entry
    name = entry.get("name")
    if not isinstance(name, str) or any(entry.get(k) for k in _ID_KEYS):
        return entry
    skill, observed = _present(index, name)
    if skill is None or observed is None:
        return entry
    carried, reported = _reported_hash(entry)
    if not carried:
        body = observed  # what is on disk is what was read
    elif reported == skill.body_sha256:
        body = skill.body_sha256  # the locked bytes, read here or elsewhere
    elif reported == observed:
        body = observed
    else:
        return entry
    return _identified(entry, skill, body)


def _identified(entry: Mapping[str, Any], skill: ManagedSkill, body: str) -> Dict[str, Any]:
    out = dict(entry)
    out["name"] = skill.wire_name
    out["skill_id"] = skill.skill_id
    out["body_sha256"] = body
    if body == skill.body_sha256:
        out["version_id"] = skill.version_id
    else:
        out["modified"] = True
    return out
