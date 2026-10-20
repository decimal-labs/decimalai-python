"""What Claude Code does to a skill's body before the model reads it.

The Claude Agent SDK hook (:mod:`decimalai.claude_agent_sdk`) sees the body the CLI handed
the model and names its version by hashing it against the lock. That fails whenever the CLI
rewrote the body first, and it rewrites it on every load: it fills the skill's argument
placeholders (``$0``, ``$1``, ..., ``$ARGUMENTS``) with the ``Skill`` call's ``args``, and its
path and session variables with their values. A body that quotes a $12 fee, loaded with a
13-word ``args``, reaches the model with ``$12`` replaced by the 13th word, and its hash is no
version's.

:func:`renderings` redoes that rewrite on a stored body, so the hook can check that the
delivered body is exactly what the locked version becomes. What it reproduces, measured
against the bundled CLI (claude-agent-sdk 0.2.166, Claude Code 2.1.299) with the local
Messages stub and one claude-haiku-4-5 run:

* **The arguments, on every load.** A call without ``args`` is filled with the empty string,
  so ``$ARGUMENTS`` disappears and ``\\$1`` loses its backslash then too. In this order:
  ``\\$`` before a digit, ``ARGUMENTS`` or an argument name becomes a plain ``$``; each name
  the frontmatter's ``arguments`` declares (``$city``), longest first, becomes its word, or
  nothing when there are too few words; ``$ARGUMENTS[N]`` and ``$N`` (0-based, not followed
  by a letter, digit or ``_``) become word N, or stay as written when there is no word N;
  ``$ARGUMENTS`` becomes the whole ``args``. A body none of that changed gets
  ``"\\n\\nARGUMENTS: <args>"`` appended when ``args`` is not empty. In every filled-in value
  a ``!`` that starts a word gets a backslash, so an argument cannot start a shell command.
* **Then the variables:** ``${CLAUDE_PLUGIN_ROOT}``, ``${CLAUDE_PROJECT_DIR}`` (the working
  directory, symlinks resolved), ``${CLAUDE_SKILL_DIR}`` (the folder the delivery's header
  names) and ``${CLAUDE_SESSION_ID}``.

Not reproduced, so a body that uses one is never matched: ``${CLAUDE_PLUGIN_DATA}``,
``${CLAUDE_EFFORT}`` and inline shell (``!`cmd```), whose output only the CLI saw.

The words come from the CLI's own bash parser: it reads ``args`` as a command line, and the
words of its first command are the arguments. :func:`words` follows that parser as far as it
was measured (``tests/fixtures/claude_skill_args_2_1_299.json``); on an ``args`` it reads
differently the result is only a wrong candidate. That costs coverage, never correctness: the
hook names a version only when a rendering is the delivered body byte for byte, so a split
this module gets wrong leaves the activation without a version, and never gives it a wrong
one.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

#: What the CLI's JavaScript regular expressions mean by ``\\s`` (and ``trim()`` strips).
_JS_SPACE_CHARS = ("\t\n\v\f\r   " + "".join(chr(c) for c in range(0x2000, 0x200B))
                   + "    　﻿")
_JS_SPACE = re.compile("[" + re.escape(_JS_SPACE_CHARS) + "]+")
_WORD_CHAR = "[A-Za-z0-9_]"  # JavaScript's \w: ASCII only

#: The CLI's own markers: a ``$`` no later pattern may read, and the brackets around a
#: filled-in value. Either one in a body or a value becomes U+FFFD first.
_DOLLAR = "￿"
_MARK = "￾"
_REPLACED = "�"

_BANG = re.compile("(?:^|(?<=[" + re.escape(_JS_SPACE_CHARS) + "]))!")

#: An ``args`` longer than this, in UTF-16 code units, is not parsed, only split.
_MAX_PARSED = 10_000

#: The variables only the CLI knows the value of.
_OPAQUE = re.compile(r"\$\{(?:CLAUDE_PLUGIN_DATA|CLAUDE_EFFORT)\}")
#: Inline shell, which the CLI runs and replaces with its output.
_SHELL = re.compile("```!|(?:^|(?<=[" + re.escape(_JS_SPACE_CHARS) + "]))!`[^`]+`")


def _clean(text: str) -> str:
    return text.replace(_DOLLAR, _REPLACED).replace(_MARK, _REPLACED)


def _escape_bang(text: str) -> str:
    """A filled-in value as the CLI writes it: no ``!`` starts a word or touches a backtick."""
    text = text.replace("`!", "` !").replace("!`", "! `")
    return _BANG.sub(lambda _: "\\!", text)


def appended(args: str) -> str:
    """The text the CLI appends to a body none of whose placeholders it filled:
    ``"ARGUMENTS: <args>"``, after a blank line."""
    return "ARGUMENTS: " + _escape_bang(_clean(args))


def argument_names(frontmatter: Optional[Mapping[str, Any]]) -> List[str]:
    """The names a skill's frontmatter ``arguments`` gives its words (``$city`` is word 0
    when the list starts with ``city``): a list, or one string split on whitespace. A blank
    name or one made of digits is no name."""
    raw = frontmatter.get("arguments") if isinstance(frontmatter, Mapping) else None
    if isinstance(raw, str):
        raw = _JS_SPACE.split(raw)
    if not isinstance(raw, list):
        return []
    return [n for n in raw if isinstance(n, str) and n.strip(_JS_SPACE_CHARS)
            and not re.fullmatch("[0-9]+", n)]


def fill(body: str, args: str, words: Sequence[str], names: Sequence[str] = ()) -> str:
    """``body`` with its argument placeholders filled from ``args`` split into ``words``."""
    text = _clean(body)
    filled = False

    def value(word: Optional[str]) -> str:
        return _MARK + _escape_bang(_clean(word or "")).replace("$", _DOLLAR) + _MARK

    def word(i: int) -> Optional[str]:
        return words[i] if i < len(words) else None

    named = sorted(((n, i) for i, n in enumerate(names) if n), key=lambda p: -len(p[0]))
    guard = "|".join(["[0-9]", "ARGUMENTS"]
                     + [re.escape(n) + r"(?![\[A-Za-z0-9_])" for n, _ in named])
    text = re.sub(r"(?<!\\)\\\$(?=" + guard + ")", lambda _: _DOLLAR, text)
    for name, i in named:
        text, n = re.subn(r"\$" + re.escape(name) + r"(?![\[A-Za-z0-9_])",
                          lambda _, i=i: value(word(i)), text)
        filled = filled or n > 0

    def by_index(m: "re.Match[str]") -> str:
        nonlocal filled
        w = word(int(m.group(1)))
        if w is None:
            return _DOLLAR + m.group(0)[1:]
        filled = True
        return value(w)

    def positional(m: "re.Match[str]") -> str:
        nonlocal filled
        w = word(int(m.group(1)))
        if w is None:
            return m.group(0)
        filled = True
        return value(w)

    text = re.sub(r"\$ARGUMENTS\[([0-9]+)\]", by_index, text)
    text = re.sub(r"\$([0-9]+)(?!" + _WORD_CHAR + ")", positional, text)
    if "$ARGUMENTS" in text:
        text = text.replace("$ARGUMENTS", value(args))
        filled = True
    if not filled and args:
        text += "\n\n" + appended(args)
    return text.replace(_DOLLAR, "$").replace(_MARK, "")


def render(body: str, args: str, words: Sequence[str], *, names: Sequence[str] = (),
           skill_dir: Optional[str] = None, plugin_root: Optional[str] = None,
           project_dir: Optional[str] = None,
           session_id: Optional[str] = None) -> Optional[str]:
    """What the CLI hands the model for ``body`` (the header aside), or None when the body
    uses something only the CLI knows (see the module docstring)."""
    text = fill(body, args, words, names)
    for var, val in (("${CLAUDE_PLUGIN_ROOT}", plugin_root),
                     ("${CLAUDE_PROJECT_DIR}", project_dir)):
        if var in text:
            if val is None:
                return None
            text = text.replace(var, val)
    if _OPAQUE.search(text):
        return None
    for var, val in (("${CLAUDE_SKILL_DIR}", skill_dir), ("${CLAUDE_SESSION_ID}", session_id)):
        if var in text:
            if val is None:
                return None
            text = text.replace(var, val)
    if _OPAQUE.search(text) or _SHELL.search(text):
        return None
    return text


def renderings(body: str, args: str, *, names: Sequence[str] = (),
               skill_dir: Optional[str] = None, plugin_root: Optional[str] = None,
               project_dirs: Iterable[Optional[str]] = (None,),
               session_id: Optional[str] = None) -> Iterator[str]:
    """Every rendering of ``body`` the CLI may have delivered for ``args``: one per way of
    splitting ``args`` (:func:`candidate_words`) and per candidate project directory."""
    seen = set()
    for ws in candidate_words(args):
        for project_dir in project_dirs:
            out = render(body, args, ws, names=names, skill_dir=skill_dir,
                         plugin_root=plugin_root, project_dir=project_dir,
                         session_id=session_id)
            if out is not None and out not in seen:
                seen.add(out)
                yield out


# ── the words ────────────────────────────────────────────────────────────────


def candidate_words(args: str) -> List[List[str]]:
    """:func:`words`, then a plain split on whitespace when that is a different list."""
    first = words(args)
    out = [first]
    plain = _split(args)
    if plain != first:
        out.append(plain)
    return out


def words(args: str) -> List[str]:
    """The words the CLI fills ``$0``, ``$1``, ... with: its parser's words of the first
    command in ``args``, or, when that finds none, ``args`` split on whitespace."""
    if not args or not args.strip(_JS_SPACE_CHARS):
        return []
    parsed = _first_command(args) if len(args.encode("utf-16-le")) // 2 <= _MAX_PARSED else []
    return parsed if parsed else _split(args)


def _split(args: str) -> List[str]:
    return [w for w in _JS_SPACE.split(args) if w]


#: Unquoted, these are operators rather than word characters.
_OPERATORS = set(";&|<>()\n")
_BLANK = set(" \t\r\v\f")
#: At the start of a command these are syntax, not a word, and the command starts after them.
_SKIPPED_KEYWORDS = {"!", "if", "then", "else", "elif", "fi", "do", "done", "while", "until",
                     "esac", "{", "}"}
#: ``for x`` and ``select x`` are skipped together with their variable.
_LOOP_KEYWORDS = {"for", "select"}
#: Words the parser does not read as a plain command; it then finds no words.
_NO_COMMAND = {"case", "function", "[[", "unset", "unsetenv"}
#: A declaration is its keyword alone.
_DECLARATIONS = {"export", "declare", "typeset", "readonly", "local"}
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]*\])?\+?=")

#: Part kinds: plain text, a quoted string, an expansion ($x, ${x}, $((x)), $'x'), and a
#: command substitution ($(x), `x`).
_TEXT, _QUOTED, _EXPANSION, _SUBSTITUTION = "text", "quoted", "expansion", "substitution"
_Part = Tuple[str, str]
#: A word (its parts) or an operator (its character), with where it starts in ``args``.
_WORD, _OP = "word", "op"
_Token = Tuple[str, Any, int]


def _plain(token: _Token) -> Optional[str]:
    """The text of a word that is one unquoted part, else None."""
    kind, parts, _ = token
    if kind == _WORD and len(parts) == 1 and parts[0][0] == _TEXT:
        return parts[0][1]
    return None


def _first_command(args: str) -> List[str]:
    """The arguments the CLI's bash parser makes of ``args``, as far as it was measured."""
    try:
        tokens = _lex(args)
    except _Unsupported:
        return []
    out: List[str] = []
    command = True  # nothing of the command yet: the next word is its name
    skipped = False  # a keyword was skipped, so a separator may follow it
    i = 0
    while i < len(tokens):
        kind, value, start = tokens[i]
        i += 1
        if kind == _OP:
            if command:
                if value == "\n" or value == "(" or (skipped and value in ";&"):
                    continue  # a blank line, a subshell, `for x; do`
                return []
            if value == "(":
                # Mid-command, a parenthesized group is dropped and the command goes on.
                depth = 1
                while i < len(tokens) and depth:
                    if tokens[i][0] == _OP:
                        depth += {"(": 1, ")": -1}.get(tokens[i][1], 0)
                    i += 1
                if depth:
                    break
                continue
            break
        parts = value
        plain = _plain(tokens[i - 1])
        if command:
            if plain in _SKIPPED_KEYWORDS:
                skipped = True
                continue
            if plain in _LOOP_KEYWORDS:
                if i + 1 < len(tokens) and _plain(tokens[i + 1]) == "in":
                    return []  # `for x in ...` with no `do`: no command
                skipped = True
                i += 1
                continue
            if plain in _NO_COMMAND:
                return []
            if plain in _DECLARATIONS:
                return [plain]
            if parts[0][0] == _TEXT and _ASSIGNMENT.match(parts[0][1]):
                continue
            if plain is not None and plain.startswith("["):
                # A test, `[ x ]` or `[x]`, skipped as far as its closing bracket.
                closing = plain
                while not closing.endswith("]"):
                    if i >= len(tokens) or tokens[i][0] != _WORD:
                        return []
                    closing = "".join(text for _, text in tokens[i][1])
                    i += 1
                continue
            if parts[0][0] == _SUBSTITUTION and parts[0][1].startswith("`"):
                # Measured, not explained: a command that starts with a backtick is one
                # word, the rest of ``args`` after that backtick.
                return [args[start + 1:]]
            command = False
            if any(k == _SUBSTITUTION for k, _ in parts) and len(parts) > 1:
                out.append("".join(text for _, text in parts))
            else:
                out.append("".join(_unquote(part) for part in parts))
            continue
        if any(k == _SUBSTITUTION for k, _ in parts):
            break
        if len(parts) == 1 and parts[0][0] == _EXPANSION:
            continue
        out.append("".join(_unquote(part) for part in parts))
    return out


def _unquote(part: _Part) -> str:
    """A word part as the parser's word list holds it: plain text without its backslashes,
    a string without its quotes when both are there, anything else as written."""
    kind, text = part
    if kind == _TEXT:
        return re.sub("\\\\([^\n\r  ])", r"\1", text)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    return text


class _Unsupported(Exception):
    """An ``args`` this emulation does not follow the parser on."""


def _lex(src: str) -> List[_Token]:
    """``src`` as words and operators. A comment is dropped up to its line's end."""
    out: List[_Token] = []
    parts: List[_Part] = []
    text: List[str] = []
    start = 0
    i, n = 0, len(src)

    def begin() -> None:
        nonlocal start
        if not parts and not text:
            start = i

    def flush_text() -> None:
        if text:
            parts.append((_TEXT, "".join(text)))
            text.clear()

    def end_word() -> None:
        flush_text()
        if parts:
            out.append((_WORD, list(parts), start))
            parts.clear()

    while i < n:
        c = src[i]
        if c == "\\":
            if i + 1 >= n:  # a trailing backslash is dropped
                i += 1
                continue
            if src[i + 1] == "\n":  # a line continuation separates words
                end_word()
                i += 2
                continue
            begin()
            text.append(src[i:i + 2])
            i += 2
            continue
        if c in _BLANK:
            end_word()
            i += 1
            continue
        if c in _OPERATORS:
            if c in "<>" and not parts and text and "".join(text).isdigit():
                text.clear()  # `2>`: the digits are the redirection's, not a word
            end_word()
            out.append((_OP, c, i))
            i += 1
            continue
        if c == "#" and not parts and not text:
            j = src.find("\n", i)
            i = n if j == -1 else j
            continue
        begin()
        if c == "'":
            j = src.find("'", i + 1)
            j = n if j == -1 else j + 1
            flush_text()
            parts.append((_QUOTED, src[i:j]))
            i = j
            continue
        if c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            j = n if j >= n else j + 1
            flush_text()
            parts.append((_QUOTED, src[i:j]))
            i = j
            continue
        if c == "`":
            j = src.find("`", i + 1)
            if j == -1:  # an unclosed backtick is a plain character
                text.append(c)
                i += 1
                continue
            flush_text()
            parts.append((_SUBSTITUTION, src[i:j + 1]))
            i = j + 1
            continue
        if c == "$":
            j = _expansion_end(src, i)
            if j is None:
                lone = i + 1 >= n or src[i + 1] in _BLANK or src[i + 1] in _OPERATORS
                if lone and not parts and not text:  # a `$` by itself is no word
                    i += 1
                    continue
                text.append(c)
                i += 1
                continue
            flush_text()
            kind = (_SUBSTITUTION if src.startswith("$(", i) and not src.startswith("$((", i)
                    else _EXPANSION)
            parts.append((kind, src[i:j]))
            i = j
            continue
        text.append(c)
        i += 1
    end_word()
    return out


def _expansion_end(src: str, i: int) -> Optional[int]:
    """Where the expansion starting at ``src[i] == "$"`` ends, or None when it is no
    expansion (a ``$`` before a space, or before punctuation)."""
    n = len(src)
    if i + 1 >= n:
        return None
    c = src[i + 1]
    if c in "({":
        close = ")" if c == "(" else "}"
        depth, j = 0, i + 1
        while j < n:
            if src[j] == c:
                depth += 1
            elif src[j] == close:
                depth -= 1
                if depth == 0:
                    return j + 1
            j += 1
        raise _Unsupported("unclosed expansion")
    if c in "'\"":
        j = src.find(c, i + 2)
        if j == -1:
            raise _Unsupported("unclosed quote")
        return j + 1
    if c.isascii() and (c.isalpha() or c == "_"):
        j = i + 2
        while j < n and src[j].isascii() and (src[j].isalnum() or src[j] == "_"):
            j += 1
        return j
    if c.isascii() and c.isdigit():
        return i + 2
    if c in "?$@*#-!":
        return i + 2
    return None
