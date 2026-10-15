"""The oracle: the project a correct ``decimalai skills install`` would leave on disk.

Until the real producer exists (``decimalai skills add`` + ``install --frozen``), the
suite grades the eight loaders against this hand-written, independent rendering of
the on-disk contract, format 1. It is deliberately small and obvious, so a reader
can check every byte against the contract by eye:

* ``.agents/skills/<install-name>/SKILL.md`` — frontmatter keys in the fixed order
  ``name, description, license?, compatibility?, allowed-tools?,
  disable-model-invocation?, metadata``; every value a YAML double-quoted scalar
  except ``name`` (plain, the name rules guarantee it needs no quoting) and
  ``disable-model-invocation`` (the bare ``true``); ``metadata`` a FLAT map of
  quoted strings, fixed keys first; then the closing fence, one ``\\n``, the
  stripped body and exactly one ``\\n``. UTF-8, LF only.
* attachments under ``references/`` and ``scripts/``, scripts mode 0755, the rest
  0644. Never a second ``SKILL.md`` anywhere in the tree.
* ``.agents/.claude-plugin/plugin.json`` — the Claude Code plugin root. Named
  ``decimalai-skills``: ``decimalai`` is the marketplace plugin's name, and two
  plugins with one name replace each other in Claude Code.
* ``.decimal/skills.lock`` (lock v2) and ``.decimal/skills.toml`` (the manifest).

Three skills, chosen to hit every rule a loader has been seen to enforce:

``acme-refund-policy``
    registry name ``acme/refund-policy`` (namespaced, so the install name differs
    from it); model-invocable; ``references/`` and an executable ``scripts/``;
    authored by the org, so a bare ``Bash`` becomes one scoped entry per script.
``acme-returns-desk``
    user-only (``disable-model-invocation: true``); no attachments, no tools.
``<57 chars>-<6 hex>``
    a registry name longer than 64 characters, disambiguated with six hex of
    ``sha256(skill_id)``; a public GitHub import, so it is ``foreign`` and its
    stored ``Bash`` and ``WebFetch`` are dropped (only ``Read``, ``Grep`` and
    ``Glob`` survive for a skill the org did not author).

Every body carries its own sentinel line, so a loader that hands the model the
wrong skill's body cannot satisfy the body item.

    python -m tests.native_loaders.oracle <project-dir>      # write it by hand
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

FORMAT = 1
ROOT = ".agents/skills"
PLUGIN_NAME = "decimalai-skills"
APP_URL = "https://app.decimal.ai"
API_URL = "https://api.decimal.ai"
ORG_ID = str(uuid.UUID(int=0x0A6E0000))
#: Excluded from ``lock_hash``, so two machines installing one lock agree; fixed
#: here so the oracle's bytes are the same on every run.
INSTALLED_AT = "2026-10-14T00:00:00Z"

FEES = (
    "# Restocking fees\n\n| category | fee |\n|---|---|\n"
    "| electronics | 15% |\n| apparel | 0% |\n"
)
SCRIPT = (
    "#!/usr/bin/env python3\n"
    '"""compute_refund.py <amount> <days>"""\n'
    "import sys\n"
    "amount, days = float(sys.argv[1]), int(sys.argv[2])\n"
    "print('REFUND full refund: %.2f' % amount if days <= 30 else\n"
    "      'REFUND store credit: %.2f' % amount if days <= 60 else 'REFUND none')\n"
)


@dataclass(frozen=True)
class Scenario:
    """What the scripted model does, and what each step must produce.

    Shared by the loader modules (which script the model) and ``contract.py``
    (which grades the result), so the two cannot drift.
    """

    target: str = "acme-refund-policy"
    user_only: str = "acme-returns-desk"
    resource: str = "references/fees.md"
    resource_marker: str = "electronics"
    script: str = "scripts/compute_refund.py"
    script_args: Tuple[str, ...] = ("100", "45")
    script_out: str = "REFUND store credit: 100.00"


SCENARIO = Scenario()


@dataclass(frozen=True)
class SkillRow:
    """One skill as the registry stores it: the inputs the renderer reads."""

    name: str                      # the registry name
    title: str
    sentinel: str
    skill_id: str
    version_id: str
    version: int
    description: str
    license: str
    compatibility: str
    allowed_tools: Tuple[str, ...]  # the stored list
    invocation_mode: str            # model | user | any
    variant: str                    # authored | foreign
    category: str
    public: bool
    pin: str                        # latest | exact
    files: Tuple[Tuple[str, str, bool], ...] = field(default=())  # (path, text, executable)

    @property
    def stored_body(self) -> str:
        # Leading and trailing blank lines on purpose: the renderer writes
        # stored_body.strip(), and body_sha256 is defined over the stripped text.
        return (
            f"\n# {self.title}\n\nSentinel: {self.sentinel}\n\n"
            "- Returned within 30 days: full refund.\n- 31-60 days: store credit.\n"
            "- After 60 days: none.\n\n"
            "For fees read references/fees.md. To compute a refund, run "
            "scripts/compute_refund.py <amount> <days>.\n\n"
        )


def _row(i: int, **kw) -> SkillRow:
    return SkillRow(
        skill_id=str(uuid.UUID(int=0x5EED0000 + i)),
        version_id=str(uuid.UUID(int=0xF00D0000 + i)),
        version=3,
        license="Apache-2.0",
        compatibility="Requires python3",
        category="support",
        description=(
            f"Apply {kw['title']} rules when a customer asks for a refund, a return or "
            "store credit."
        ),
        **kw,
    )


ATTACHMENTS = (("references/fees.md", FEES, False), ("scripts/compute_refund.py", SCRIPT, True))

ROWS: Tuple[SkillRow, ...] = (
    _row(0, name="acme/refund-policy", title="Acme refund policy",
         sentinel="SENTINEL-NATIVE-REFUND-7f3a", allowed_tools=("Read", "Bash"),
         invocation_mode="model", variant="authored", public=False, pin="latest",
         files=ATTACHMENTS),
    _row(1, name="acme/returns-desk", title="Acme returns desk",
         sentinel="SENTINEL-NATIVE-DESK-2b9c", allowed_tools=(),
         invocation_mode="user", variant="authored", public=False, pin="latest"),
    _row(2, name="some-github-owner/an-extremely-long-imported-skill-name-that-exceeds-the-limit",
         title="Imported refund policy", sentinel="SENTINEL-NATIVE-IMPORT-c41d",
         allowed_tools=("Read", "Bash", "WebFetch"), invocation_mode="model",
         variant="foreign", public=True, pin="exact", files=ATTACHMENTS),
)


# ── the rules, written out ───────────────────────────────────────────────────

NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
RESERVED = frozenset({"install", "search", "check", PLUGIN_NAME})


def install_name_base(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9]+", "-", s.lower())
    return re.sub(r"-{2,}", "-", s).strip("-")


def install_name_for(name: str, skill_id: str) -> str:
    base = install_name_base(name)
    if base and len(base) <= 64 and base not in RESERVED:
        return base
    stem = (base or "skill")[:57].rstrip("-")
    return f"{stem}-{hashlib.sha256(skill_id.encode()).hexdigest()[:6]}"


_ESCAPED = {0x85, 0x2028, 0x2029, 0xFEFF}


def yaml_quote(s: str) -> str:
    """A YAML double-quoted scalar that PyYAML reads back equal to ``s``."""
    out = []
    for ch in s:
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif o < 0x20 or o == 0x7F or o in _ESCAPED:
            out.append("\\u%04x" % o)
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


_INTERPRETERS = {".py": "python3", ".sh": "bash", ".js": "node"}


def allowed_tools_for(row: SkillRow) -> str:
    """The allowed-tools policy: never a list, never a bare ``Bash``."""
    scripts = [p for p, _, ex in row.files if ex and p.startswith("scripts/")]
    out: List[str] = []
    for tok in row.allowed_tools:
        if row.variant == "foreign":
            if tok in ("Read", "Grep", "Glob"):
                out.append(tok)
        elif tok == "Bash":
            for p in scripts:
                interp = _INTERPRETERS.get(Path(p).suffix)
                cmd = f"{interp} " if interp else ""
                out.append(f"Bash({cmd}${{CLAUDE_SKILL_DIR}}/{p} *)")
        else:
            out.append(tok)
    return " ".join(out)


def _description(row: SkillRow) -> str:
    return re.sub(r"\s*\n\s*", " ", row.description.strip())


def metadata_for(row: SkillRow, install_name: str) -> Dict[str, str]:
    meta = {
        "decimalai-skill-id": row.skill_id,
        "decimalai-version-id": row.version_id,
        "decimalai-version": str(row.version),
        "decimalai-body-sha256": body_sha256(row),
        "decimalai-name": row.name,
        "decimalai-format": str(FORMAT),
        "decimalai-invocation": row.invocation_mode,
    }
    optional = {"decimalai-category": row.category}
    if row.public:
        optional["decimalai-source"] = f"{APP_URL}/s/{install_name}@{row.version}/SKILL.md"
    meta.update(sorted((k, v) for k, v in optional.items() if v))
    return meta


def sentinel_of(install_name: str) -> str:
    """The sentinel line of the oracle skill installed under ``install_name``."""
    for row in ROWS:
        if install_name_for(row.name, row.skill_id) == install_name:
            return row.sentinel
    raise KeyError(f"no oracle skill installs as {install_name!r}")


def body_sha256(row: SkillRow) -> str:
    return hashlib.sha256(row.stored_body.strip().encode("utf-8")).hexdigest()


def render_skill_md(row: SkillRow, install_name: str) -> bytes:
    lines = ["---", f"name: {install_name}", f"description: {yaml_quote(_description(row))}"]
    if row.license:
        lines.append(f"license: {yaml_quote(row.license)}")
    if row.compatibility:
        lines.append(f"compatibility: {yaml_quote(row.compatibility)}")
    tools = allowed_tools_for(row)
    if tools:
        lines.append(f"allowed-tools: {yaml_quote(tools)}")
    if row.invocation_mode == "user":
        lines.append("disable-model-invocation: true")
    lines.append("metadata:")
    lines += [f"  {k}: {yaml_quote(v)}" for k, v in metadata_for(row, install_name).items()]
    lines.append("---")
    return ("\n".join(lines) + "\n" + row.stored_body.strip() + "\n").encode("utf-8")


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def render_hash(row: SkillRow) -> str:
    return hashlib.sha256(canonical_json({
        "allowed_tools": list(row.allowed_tools),
        "compatibility": row.compatibility,
        "description": _description(row),
        "invocation_mode": row.invocation_mode,
        "license": row.license,
    })).hexdigest()


def lock_hash(root: str, skills: Dict[str, dict]) -> str:
    """``sha256(canonical_json({"root", "skills" minus installed_at}))``."""
    bare = {k: {f: v for f, v in e.items() if f != "installed_at"} for k, e in skills.items()}
    return hashlib.sha256(canonical_json({"root": root, "skills": bare})).hexdigest()


# ── the project ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Project:
    files: Dict[str, bytes]        # project-relative path -> exact bytes
    executable: Tuple[str, ...]    # project-relative paths written 0755
    lock: dict


def build(rows: Tuple[SkillRow, ...] = ROWS) -> Project:
    files: Dict[str, bytes] = {}
    executable: List[str] = []
    skills: Dict[str, dict] = {}
    for row in rows:
        name = install_name_for(row.name, row.skill_id)
        folder = f"{ROOT}/{name}"
        written: Dict[str, bytes] = {"SKILL.md": render_skill_md(row, name)}
        for rel, text, _ in row.files:
            if Path(rel).name.lower() == "skill.md":
                raise ValueError(f"{row.name}: an attachment may never be a SKILL.md ({rel})")
            written[rel] = text.encode("utf-8")
        for rel, data in written.items():
            files[f"{folder}/{rel}"] = data
        exe = sorted(rel for rel, _, ex in row.files if ex)
        executable += [f"{folder}/{rel}" for rel in exe]
        skills[name] = {
            "name": row.name,
            "skill_id": row.skill_id,
            "version_id": row.version_id,
            "version": row.version,
            "render": {"format": FORMAT, "hash": render_hash(row), "variant": row.variant},
            "body_sha256": body_sha256(row),
            "description_sha256": hashlib.sha256(_description(row).encode("utf-8")).hexdigest(),
            "path": folder,
            "files": {rel: hashlib.sha256(data).hexdigest() for rel, data in sorted(written.items())},
            "executable": exe,
            "unavailable": [],
            "source": {"base_url": API_URL, "org": ORG_ID},
            "pin": row.pin,
            "invocation": "user" if row.invocation_mode == "user" else "model",
            "installed_at": INSTALLED_AT,
        }
    lock = {"version": 2, "root": ROOT, "lock_hash": lock_hash(ROOT, skills), "skills": skills}
    files[".decimal/skills.lock"] = (json.dumps(lock, indent=2, sort_keys=True) + "\n").encode()
    plugin = {
        "name": PLUGIN_NAME,
        "version": f"lock-{lock['lock_hash'][:12]}",
        "description": "Skills installed by `decimalai skills`. Managed: edit through the CLI.",
        "author": {"name": "decimalai"},
    }
    files[".agents/.claude-plugin/plugin.json"] = (
        json.dumps(plugin, separators=(",", ":")) + "\n").encode()
    manifest = [
        "# Managed by `decimalai skills`. Commit this file. Hand edits are fine;",
        "# run `decimalai skills install` afterwards.",
        f'root = "{ROOT}"',
        "",
        "[skills]",
    ] + [f'"{r.name}" = "{"latest" if r.pin == "latest" else r.version}"' for r in rows]
    files[".decimal/skills.toml"] = ("\n".join(manifest) + "\n").encode()
    return Project(files=files, executable=tuple(sorted(executable)), lock=lock)


def write(root: Path, rows: Tuple[SkillRow, ...] = ROWS) -> dict:
    """Write the oracle project into ``root`` (which must not hold one yet)."""
    project = build(rows)
    root = Path(root).resolve()
    for rel, data in project.files.items():
        path = root / rel
        if path.exists():
            raise FileExistsError(f"the oracle never overwrites: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o755 if rel in project.executable else 0o644)
    return project.lock


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m tests.native_loaders.oracle <project-dir>")
    print(json.dumps(sorted(write(Path(sys.argv[1]))["skills"]), indent=2))
