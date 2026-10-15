"""``.decimal/skills.toml``: the skills a project wants, by registry name.

::

    # Managed by `decimalai skills`. Commit this file. Hand edits are fine;
    # run `decimalai skills install` afterwards.
    root = ".agents/skills"

    [skills]
    "acme/refund-policy" = "latest"
    "pdf-tools" = "3"
    "anthropic/pdf" = { version = "latest", id = "5eed…" }   # when a public name is ambiguous

Keys are canonical registry names. A value is ``"latest"`` or a version number written
as a string; there are no ranges. The inline-table form adds the skill's id.

People edit this file, so reading is strict and writing is gentle. A value the engine
cannot use is a :class:`~decimalai.skillpkg.errors.ManifestError` that names its line.
A save rewrites only the ``[skills]`` lines whose entry changed, adds new entries after
the last one and deletes removed ones; every other byte (comments, blank lines, key
spelling, other tables, line endings) is kept. Every edited text is parsed again before
it is written, and refused if it does not read back as intended.

TOML is read with :mod:`tomllib` (Python 3.11+) or ``tomli`` (3.10).
"""
from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Union

from .errors import ManifestError, SkillsNotInstalledError, UsageError
from .project import ROOT, Project

try:
    import tomllib as _toml
except ModuleNotFoundError:  # Python 3.10
    import tomli as _toml  # type: ignore[no-redef]

__all__ = [
    "HEADER",
    "Manifest",
    "ManifestEntry",
    "load",
    "loads",
    "new",
    "parse_spec",
    "save",
]

#: The comment a new manifest starts with.
HEADER = (
    "# Managed by `decimalai skills`. Commit this file. Hand edits are fine;\n"
    "# run `decimalai skills install` afterwards.\n"
)

_SOURCE = ".decimal/skills.toml"
_VERSION_RE = re.compile(r"[1-9][0-9]*")
_ENTRY_KEYS = ("version", "id")
_BOM = "\ufeff"


@dataclass(frozen=True)
class ManifestEntry:
    """One wanted skill: its registry ``name``, a ``spec`` and, when needed, its ``id``."""

    name: str
    spec: str = "latest"
    id: Optional[str] = None

    @property
    def latest(self) -> bool:
        return self.spec == "latest"

    @property
    def version(self) -> Optional[int]:
        """The pinned version number, or None for ``latest``."""
        return None if self.latest else int(self.spec)

    @property
    def pin(self) -> str:
        """``latest`` or ``exact``, as the lock records it."""
        return "latest" if self.latest else "exact"


def parse_spec(value: Union[str, int]) -> str:
    """``"latest"`` or a positive version number, as the manifest writes it.

    Accepts ``"latest"``, ``"3"`` and ``3``; refuses everything else (``"03"``,
    ``"v3"``, ``"^3"``, ``0``) with :class:`~decimalai.skillpkg.errors.UsageError`.
    """
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return str(value)
    if isinstance(value, str) and (value == "latest" or _VERSION_RE.fullmatch(value)):
        return value
    raise UsageError(
        f"{value!r} is not a version: use `latest` or a positive integer such as `3`")


# ── the statement scanner (for line-preserving edits) ───────────────────────


@dataclass(frozen=True)
class _Stmt:
    """One TOML statement of the original text: its lines and what it defines."""

    start: int                   # first line index (0-based)
    end: int                     # one past the last line index
    kind: str                    # "blank" (comments, blank lines), "header" or "kv"
    table: Tuple[str, ...]       # the table it sits in (a header: the table it opens)
    key: Tuple[str, ...] = ()    # a kv statement's key path, relative to its table
    array: bool = False          # a header of an array of tables

    @property
    def path(self) -> Tuple[str, ...]:
        return self.table + self.key

    @property
    def line(self) -> int:
        return self.start + 1


def _lines(text: str) -> List[str]:
    """Physical lines, each with its ending. TOML breaks lines at LF only."""
    return re.findall(r"[^\n]*\n|[^\n]+$", text)


def _parses(text: str) -> Optional[dict]:
    try:
        return _toml.loads(text)
    except _toml.TOMLDecodeError:
        return None


def _walk(value: Any) -> Tuple[Tuple[str, ...], Any]:
    """Follow single-key tables down to the leaf: ``{"a": {"b": 0}}`` -> ``(("a", "b"), 0)``."""
    path: Tuple[str, ...] = ()
    while isinstance(value, dict) and len(value) == 1:
        key, value = next(iter(value.items()))
        path += (key,)
    return path, value


def _split_key(stmt: str) -> Optional[Tuple[int, Tuple[str, ...]]]:
    """``(index of the '=', key path)`` of a key/value statement."""
    for i, ch in enumerate(stmt):
        if ch == "=":
            probe = _parses(stmt[:i] + "= 0\n")
            if probe is not None:
                path, leaf = _walk(probe)
                if leaf == 0 and path:
                    return i, path
    return None


def _scan(lines: List[str]) -> List[_Stmt]:
    """Group the lines of a valid TOML text into statements.

    Each statement is the shortest run of lines that parses on its own; the text as a
    whole is known to parse, so every run ends. Headers set the table that the
    following key/value statements belong to.
    """
    out: List[_Stmt] = []
    table: Tuple[str, ...] = ()
    start = 0
    buf = ""
    for i, line in enumerate(lines):
        buf += line
        parsed = _parses(buf)
        if parsed is None and i + 1 < len(lines):
            continue
        if parsed is None or not parsed:
            out.append(_Stmt(start, i + 1, "blank", table))
        elif buf.lstrip().startswith("["):
            path, leaf = _walk(parsed)
            table = path
            out.append(_Stmt(start, i + 1, "header", table, array=isinstance(leaf, list)))
        else:
            split = _split_key(buf)
            out.append(_Stmt(start, i + 1, "kv", table, key=split[1] if split else ()))
        start, buf = i + 1, ""
    return out


def _comment_start(line: str) -> Optional[int]:
    """Where a statement line's trailing comment starts, if it has one."""
    body = line.rstrip("\r\n")
    whole = _parses(body + "\n")
    for i, ch in enumerate(body):
        if ch == "#" and _parses(body[:i] + "\n") == whole:
            return i
    return None


# ── quoting ──────────────────────────────────────────────────────────────────


def _quote(s: str) -> str:
    """A TOML basic string that reads back as ``s``."""
    out = ['"']
    for ch in s:
        code = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif code < 0x20 or code == 0x7F:
            out.append(f"\\u{code:04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _value(entry: ManifestEntry) -> str:
    if entry.id is None:
        return _quote(entry.spec)
    return f"{{ version = {_quote(entry.spec)}, id = {_quote(entry.id)} }}"


def _entry_line(entry: ManifestEntry, newline: str) -> str:
    return f"{_quote(entry.name)} = {_value(entry)}{newline}"


# ── the manifest ─────────────────────────────────────────────────────────────


class Manifest:
    """The parsed manifest, plus what a line-preserving save needs of the original text.

    ``entries`` is a read-only view in file order; change it with :meth:`set` and
    :meth:`remove`, then write it with :func:`save`.
    """

    def __init__(self, *, root: str = ROOT, entries: Optional[Mapping[str, ManifestEntry]] = None,
                 text: Optional[str] = None, source: str = _SOURCE) -> None:
        self.root = root
        self.source = source
        self._entries: Dict[str, ManifestEntry] = dict(entries or {})
        self._original: Dict[str, ManifestEntry] = dict(self._entries)
        self._bom = bool(text) and text.startswith(_BOM)  # type: ignore[union-attr]
        self._text = text[1:] if self._bom else text      # type: ignore[index]

    @property
    def entries(self) -> Mapping[str, ManifestEntry]:
        return MappingProxyType(self._entries)

    def __getitem__(self, name: str) -> ManifestEntry:
        return self._entries[name]

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, name: str) -> Optional[ManifestEntry]:
        return self._entries.get(name)

    def set(self, name: str, spec: Union[str, int] = "latest", *,
            id: Optional[str] = None) -> ManifestEntry:
        """Add or replace the entry for ``name``.

        Raises:
            UsageError: a name TOML cannot hold, a bad spec, or an empty id.
        """
        if not isinstance(name, str) or not name:
            raise UsageError("a skill name must be a non-empty string")
        try:
            name.encode("utf-8")
        except UnicodeEncodeError:
            raise UsageError(f"{name!r} cannot be written to `{self.source}`") from None
        if id is not None and (not isinstance(id, str) or not id):
            raise UsageError("a skill id must be a non-empty string", skill=name)
        entry = ManifestEntry(name, parse_spec(spec), id)
        self._entries[name] = entry
        return entry

    def remove(self, name: str) -> bool:
        """Remove the entry for ``name``; False when there was none."""
        return self._entries.pop(name, None) is not None

    # ── writing ──────────────────────────────────────────────────────────────

    def dumps(self) -> str:
        """The file text for the current entries.

        Raises:
            ManifestError: an entry that changed is written in a form this editor does
                not rewrite (its own ``[skills."name"]`` table, or a dotted key outside
                ``[skills]``); the error names its line.
        """
        if self._text is None:
            text = HEADER + f"root = {_quote(ROOT)}\n\n[skills]\n" + "".join(
                _entry_line(e, "\n") for e in self._entries.values())
        else:
            text = self._edit(self._text)
        back = loads((_BOM if self._bom else "") + text, source=self.source)
        if back.root != self.root or dict(back.entries) != self._entries:
            raise ManifestError(  # pragma: no cover - a bug in the editor, never user data
                "an edit did not read back as written; nothing was saved", path=self.source)
        return (_BOM if self._bom else "") + text

    def _edit(self, text: str) -> str:
        lines = _lines(text)
        newline = "\r\n" if "\r\n" in text else "\n"
        stmts = _scan(lines)
        located: Dict[str, Optional[_Stmt]] = {}
        for st in stmts:
            if len(st.path) == 2 and st.path[0] == "skills" and st.kind in ("kv", "header"):
                editable = st.kind == "kv" and st.table == ("skills",)
                located[st.path[1]] = st if editable else None
        skills_table = [st for st in stmts if st.kind == "header" and st.table == ("skills",)
                        and not st.array]
        anchor = max((st.end for st in stmts if st.kind == "kv" and st.table == ("skills",)),
                     default=skills_table[0].end if skills_table else None)

        replace: Dict[int, Tuple[int, List[str]]] = {}  # start -> (end, new lines)
        for name, old in self._original.items():
            new = self._entries.get(name)
            if new == old:
                continue
            loc = located.get(name)
            if loc is None:
                where = next((s.line for s in stmts if s.path[:2] == ("skills", name)), None)
                raise ManifestError(
                    f"`{self.source}` line {where}: {name!r} is written in a form this "
                    "editor does not rewrite; write it as `\"<name>\" = \"<version>\"` "
                    "under `[skills]`",
                    skill=name, path=self.source, details={"line": where})
            replace[loc.start] = (loc.end, [] if new is None else
                                  [self._rewrite(lines[loc.start:loc.end], new, newline)])

        added = [_entry_line(e, newline) for n, e in self._entries.items()
                 if n not in self._original]
        out: List[str] = []
        i = 0
        while i < len(lines):
            if i in replace:
                end, new_lines = replace[i]
                out += new_lines
                i = end
            else:
                out.append(lines[i])
                i += 1
            if anchor is not None and i == anchor and added:
                if out and not out[-1].endswith("\n"):
                    out[-1] += newline
                out += added
                added = []
        if added:
            if out and not out[-1].endswith("\n"):
                out[-1] += newline
            if out and out[-1].strip():
                out.append(newline)
            out.append(f"[skills]{newline}")
            out += added
        return "".join(out)

    @staticmethod
    def _rewrite(stmt_lines: List[str], entry: ManifestEntry, newline: str) -> str:
        """The statement's first line with its value replaced; key spelling and comment kept."""
        stmt = "".join(stmt_lines)
        split = _split_key(stmt)
        key_text = stmt[:split[0]].rstrip() if split else _quote(entry.name)
        indent = key_text[:len(key_text) - len(key_text.lstrip())]
        line = f"{indent}{key_text.strip()} = {_value(entry)}"
        if len(stmt_lines) == 1:
            original = stmt_lines[0]
            c = _comment_start(original)
            if c is not None:
                body = original.rstrip("\r\n")
                gap = re.search(r"\s*$", body[:c]).group()  # type: ignore[union-attr]
                line += gap + body[c:]
            ending = original[len(original.rstrip("\r\n")):]
            return line + (ending or newline)
        return line + newline


def new() -> Manifest:
    """An empty manifest, written as a fresh file."""
    return Manifest()


# ── reading ──────────────────────────────────────────────────────────────────


def _line_of(stmts: List[_Stmt], path: Tuple[str, ...]) -> Optional[int]:
    """The line that defines ``path`` (or the nearest enclosing definition)."""
    for depth in range(len(path), 0, -1):
        for st in stmts:
            if st.kind in ("kv", "header") and st.path == path[:depth]:
                return st.line
    return None


def _error_line(exc: Exception) -> Optional[int]:
    line = getattr(exc, "lineno", None)
    if isinstance(line, int):
        return line
    m = re.search(r"at line (\d+)", str(exc))
    return int(m.group(1)) if m else None


def loads(text: str, *, source: str = _SOURCE) -> Manifest:
    """Parse manifest text. ``source`` names the file in error messages.

    Raises:
        ManifestError: invalid TOML, an unsupported ``root``, or an entry that is not
            ``"latest"``, a positive version string or ``{ version = ..., id = ... }``.
    """
    body = text[1:] if text.startswith(_BOM) else text
    try:
        data = _toml.loads(body)
    except _toml.TOMLDecodeError as exc:
        line = _error_line(exc)
        raise ManifestError(f"`{source}` is not valid TOML: {exc}", path=source,
                            details={"line": line} if line else None) from None
    stmts = _scan(_lines(body))

    def fail(path: Tuple[str, ...], what: str, *, skill: Optional[str] = None) -> ManifestError:
        line = _line_of(stmts, path)
        where = f"`{source}` line {line}" if line else f"`{source}`"
        return ManifestError(f"{where}: {what}", skill=skill, path=source,
                             details={"line": line} if line else None)

    root = data.get("root", ROOT)
    if not isinstance(root, str) or posixpath.normpath(root) != ROOT:
        raise fail(("root",), f"root = {root!r} is not supported: the managed root is "
                              f"`{ROOT}`, the folder Claude's plugin and every loader read")
    skills = data.get("skills", {})
    if not isinstance(skills, dict):
        raise fail(("skills",), "`skills` must be a table: `[skills]` followed by "
                                "`\"<name>\" = \"<version>\"` lines")
    entries: Dict[str, ManifestEntry] = {}
    ids: Dict[str, str] = {}
    for name, value in skills.items():
        path = ("skills", name)
        if not name:
            raise fail(path, "a skill name must not be empty")
        if isinstance(value, dict):
            unknown = sorted(set(value) - set(_ENTRY_KEYS))
            if unknown:
                raise fail(path, f"{name!r}: unknown key {unknown[0]!r}; an entry takes "
                                 "`version` and `id`", skill=name)
            if "version" not in value:
                raise fail(path, f"{name!r}: the table form needs a `version`", skill=name)
            spec, sid = value["version"], value.get("id")
            if "id" in value and (not isinstance(sid, str) or not sid):
                raise fail(path, f"{name!r}: id must be a non-empty string", skill=name)
        else:
            spec, sid = value, None
        if not isinstance(spec, str):
            raise fail(path, f"{name!r} = {spec!r}: the version must be a string, "
                             '"latest" or a positive integer such as "3"', skill=name)
        if spec != "latest" and not _VERSION_RE.fullmatch(spec):
            raise fail(path, f"{name!r} = {spec!r}: the version must be \"latest\" or a "
                             'positive integer such as "3"', skill=name)
        if sid is not None:
            if sid in ids:
                raise fail(path, f"{name!r} has the id of {ids[sid]!r}; each skill is "
                                 "listed once", skill=name)
            ids[sid] = name
        entries[name] = ManifestEntry(name, spec, sid)
    return Manifest(root=ROOT, entries=entries, text=text, source=source)


def load(project: Project, *, missing_ok: bool = False) -> Manifest:
    """Read ``.decimal/skills.toml``.

    Args:
        missing_ok: return an empty manifest (written as a fresh file) when there is
            none, instead of raising.

    Raises:
        SkillsNotInstalledError: there is no manifest, and ``missing_ok`` is false.
        ManifestError: it cannot be used (see :func:`loads`); also when it is not UTF-8.
        IllegalEntryError: ``.decimal`` or the manifest is a symlink.
    """
    path = project.manifest_path
    project.refuse_symlinks(path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        if missing_ok:
            return new()
        raise SkillsNotInstalledError(f"`{_SOURCE}` does not exist in {project.directory}",
                                      path=path) from None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError(f"`{_SOURCE}` is not UTF-8 ({exc.reason} at byte {exc.start})",
                            path=_SOURCE) from None
    return loads(text)


def save(project: Project, manifest: Manifest) -> bool:
    """Write the manifest if its text changed. Returns whether it wrote."""
    data = manifest.dumps().encode("utf-8")
    path = project.manifest_path
    project.refuse_symlinks(path)
    try:
        if path.read_bytes() == data:
            return False
    except FileNotFoundError:
        pass
    project.write_file(path, data)
    return True
