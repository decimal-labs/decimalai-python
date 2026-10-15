"""Rewrite the ``decimalai skills verify`` defect fixtures (provenance, not a test).

    python tests/skillpkg/fixtures/defects/make_defects.py      # from the checkout root

Each directory beside this file is a project. ``.agents`` and ``.decimal`` are stored as
``agents`` and ``decimal``: this repository's ``.gitignore`` excludes the dot names at
every depth. ``test_verify.py`` copies a project to a temporary directory, restores the
names, sets every file's mode from the lock, and makes the symlinks git would not keep.

``managed/`` holds one skill folder per folder-level defect. Each managed folder is
locked with the bytes it holds, so it reports its own defect and no other; the
exceptions are the folders whose defect IS a difference from the lock (an edited or an
extra file). Beside them sit folders the lock does not list: hand-authored skills, an
orphan that carries a stamp, and a skill nested one level too deep.

Every other project breaks one project-level rule: ``plugin.json``, ``lock_hash``, an
entry an older decimalai merged into the lock, the manifest, an interrupted run, and a
version 1 lock. ``clean/`` breaks none, and ``no-lock/`` has hand-authored skills in
``.agents/skills`` and no lock at all.

The test regenerates this tree and compares it byte for byte with the committed copy,
so a fixture file that goes missing, or is edited by hand, fails there.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

from skillevaluation import skillfmt

from decimalai.skillpkg import lock as L
from decimalai.skillpkg import manifest as M

HERE = Path(__file__).resolve().parent

#: The stored names of ``.agents`` and ``.decimal``.
AGENTS, DECIMAL = "agents", "decimal"

PROJECTS = (
    "clean",
    "managed",
    "plugin-json-stale",
    "lock-hash-edited",
    "foreign-entry",
    "manifest-disagrees",
    "interrupted",
    "lock-v1",
    "no-lock",
)

INSTALLED_AT = "2026-10-15T00:00:00Z"
BASE_URL = "https://api.example.test"
DESCRIPTION = "Apply Acme refund rules when a customer asks for a refund, a return or store credit."
BODY = (
    "# Acme refund policy\n\n"
    "- Returned within 30 days: full refund.\n"
    "- 31-60 days: store credit.\n\n"
    "Fees are in references/fees.md. To compute a refund, run "
    "scripts/compute_refund.py <amount> <days>.\n"
)
FEES = "# Restocking fees\n\n| category | fee |\n|---|---|\n| electronics | 15% |\n"
SCRIPT = (
    "#!/usr/bin/env python3\n"
    "import sys\n"
    "amount, days = float(sys.argv[1]), int(sys.argv[2])\n"
    "print(amount if days <= 30 else 0.0)\n"
)
TOOLS = 'allowed-tools: "Read"\n'  # what a skill without scripts is rendered with
REFERENCES = (skillfmt.Attachment("references/fees.md", FEES),)
SCRIPTS = (skillfmt.Attachment("scripts/compute_refund.py", SCRIPT, executable=True),)


def _id(n: int) -> str:
    return str(uuid.UUID(int=n))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _edit(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"{old!r} is not in the rendered SKILL.md; the renderer changed")
    return text.replace(old, new, 1)


def registry_name(install_name: str) -> str:
    return "acme/refund-policy" if install_name == "acme-refund-policy" else f"acme/{install_name}"


def render(name: str, n: int, attachments: Iterable[skillfmt.Attachment] = ()) -> skillfmt.Bundle:
    """Format 1 bytes of the skill :func:`registry_name` ``(name)``, installed as ``name``."""
    attachments = tuple(attachments)
    scripted = any(a.executable for a in attachments)
    fields = skillfmt.RenderFields(
        skill_id=_id(0x5EED0000 + n), version_id=_id(0xF00D0000 + n), version=3,
        registry_name=registry_name(name), body=BODY, description=DESCRIPTION,
        license="Apache-2.0", compatibility="Requires python3",
        allowed_tools=["Read", "Bash"] if scripted else ["Read"],
        invocation_mode="model", category="support")
    return skillfmt.render_bundle(fields, attachments, install_name=name, variant="authored")


def hand_written(name: str, extra: str = "") -> str:
    """A SKILL.md as a person writes one: no stamp, plain YAML."""
    return (
        f"---\nname: {name}\n"
        "description: Summarize a support ticket in three bullet points.\n"
        f"{extra}---\n"
        "# Summarize a ticket\n\n"
        "Read the ticket, then write three bullets: the problem, what was tried, "
        "what happens next.\n"
    )


def with_companions(name: str) -> Dict[str, str]:
    """A hand-authored skill with the files authors keep beside SKILL.md: most are not
    under scripts/, references/, templates/ or assets/, and none is a loader's business."""
    return {
        "SKILL.md": hand_written(name),
        "reference.md": "# Reference\n\nThe long version of the rules.\n",
        "LICENSE.txt": "Apache-2.0\n",
        "eval.yaml": "cases: []\n",
        "docs/guide.md": "# Guide\n",
        "references/forms.md": "# Forms\n",
    }


class _Project:
    def __init__(self, name: str) -> None:
        self.name = name
        self.files: Dict[str, bytes] = {}
        self.entries: Dict[str, dict] = {}

    def put(self, rel: str, data) -> None:
        self.files[rel] = data if isinstance(data, bytes) else data.encode("utf-8")

    def folder(self, install_name: str, files: Mapping[str, str]) -> None:
        for rel, text in files.items():
            self.put(f"{AGENTS}/skills/{install_name}/{rel}", text)

    def install(self, bundle: skillfmt.Bundle, *, skill_md: Optional[str] = None,
                disk: Optional[Mapping[str, str]] = None,
                body_sha256: Optional[str] = None) -> None:
        """Write ``bundle``'s folder and lock it.

        ``skill_md`` replaces the rendered SKILL.md, and the lock records it as written.
        ``disk`` files are written over (or beside) the locked ones, and the lock never
        learns of them. ``body_sha256`` replaces the lock's.
        """
        written = {f.path: f.content for f in bundle.files}
        if skill_md is not None:
            written["SKILL.md"] = skill_md
        self.folder(bundle.install_name, {**written, **(disk or {})})
        self.entries[bundle.install_name] = L.make_entry(
            bundle.install_name, name=registry_name(bundle.install_name), skill_id=bundle.skill_id,
            version_id=bundle.version_id, version=3,
            render={"format": bundle.format, "hash": bundle.render_hash,
                    "variant": bundle.variant},
            body_sha256=body_sha256 or bundle.body_sha256,
            description_sha256=bundle.description_sha256,
            files={path: _sha(text) for path, text in written.items()},
            executable=[f.path for f in bundle.files if f.mode == "0755"],
            source={"base_url": BASE_URL, "org": None}, pin="latest",
            invocation=bundle.invocation, installed_at=INSTALLED_AT)

    @property
    def lock(self) -> L.Lock:
        return L.Lock(skills=dict(sorted(self.entries.items())))

    def manifest_text(self) -> str:
        manifest = M.new()
        for entry in self.lock.skills.values():
            manifest.set(entry["name"])
        return manifest.dumps()

    def finish(self, *, lock: Optional[bytes] = None, plugin: Optional[bytes] = None,
               manifest: Optional[str] = None) -> Dict[str, bytes]:
        self.put(f"{DECIMAL}/skills.lock", L.dumps(self.lock) if lock is None else lock)
        self.put(f"{AGENTS}/.claude-plugin/plugin.json",
                 L.plugin_json_bytes(self.lock.lock_hash) if plugin is None else plugin)
        self.put(f"{DECIMAL}/skills.toml", self.manifest_text() if manifest is None else manifest)
        return self.out()

    def out(self) -> Dict[str, bytes]:
        return {f"{self.name}/{rel}": data for rel, data in sorted(self.files.items())}


def _lock_bytes(data: dict) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _one_skill(project: str) -> _Project:
    p = _Project(project)
    p.install(render("acme-refund-policy", 0, REFERENCES + SCRIPTS))
    return p


# ── the projects ─────────────────────────────────────────────────────────────


def clean() -> Dict[str, bytes]:
    return _one_skill("clean").finish()


def managed() -> Dict[str, bytes]:
    p = _Project("managed")
    counter = iter(range(1, 1000))

    def bundle(name: str, attachments: Iterable[skillfmt.Attachment] = ()) -> skillfmt.Bundle:
        return render(name, next(counter), attachments)

    p.install(bundle("acme-refund-policy", REFERENCES + SCRIPTS))

    b = bundle("f1-duplicate-key")
    p.install(b, skill_md=_edit(b.skill_md, "license:", "name: f1-duplicate-key\nlicense:"))
    b = bundle("f2-unknown-key")
    p.install(b, skill_md=_edit(b.skill_md, "metadata:\n", 'category: "support"\nmetadata:\n'))
    b = bundle("f3-namespaced-name")
    p.install(b, skill_md=_edit(b.skill_md, "name: f3-namespaced-name\n",
                                "name: acme/f3-namespaced-name\n"))
    b = bundle("f4-empty-description")
    p.install(b, skill_md=_edit(b.skill_md, f"description: {skillfmt.yaml_quote(DESCRIPTION)}\n",
                                'description: ""\n'))
    b = bundle("f5-compatibility-list")
    p.install(b, skill_md=_edit(b.skill_md, 'compatibility: "Requires python3"\n',
                                'compatibility:\n  - "python>=3.10"\n'))
    b = bundle("f6a-allowed-tools-list")
    p.install(b, skill_md=_edit(b.skill_md, TOOLS, "allowed-tools:\n  - Read\n  - Grep\n"))
    b = bundle("f6b-bare-bash")
    p.install(b, skill_md=_edit(b.skill_md, TOOLS, 'allowed-tools: "Read Bash"\n'))
    b = bundle("f7-disable-model-invocation-false")
    p.install(b, skill_md=_edit(b.skill_md, "metadata:\n",
                                "disable-model-invocation: false\nmetadata:\n"))
    b = bundle("f8-nested-metadata")
    p.install(b, skill_md=_edit(b.skill_md, "metadata:\n",
                                'metadata:\n  decimalai:\n    version: "3"\n'))
    b = bundle("f9-license-list")
    p.install(b, skill_md=_edit(b.skill_md, 'license: "Apache-2.0"\n',
                                "license:\n  - Apache-2.0\n"))
    # An empty body whose stamp and lock agree with it: only the empty body is wrong.
    b = bundle("f10-empty-body")
    empty = skillfmt.body_sha256("")
    head = b.skill_md[: b.skill_md.index("\n---\n") + len("\n---\n")]
    p.install(b, skill_md=_edit(head, b.body_sha256, empty), body_sha256=empty)
    b = bundle("f11-stamp-differs")
    p.install(b, skill_md=_edit(b.skill_md, b.version_id, _id(0xBAD00011)))
    b = bundle("f12-body-differs")
    p.install(b, skill_md=b.skill_md + "Always quote the fee table.\n")
    b = bundle("f13-two-trailing-newlines")
    p.install(b, skill_md=b.skill_md + "\n")
    p.install(bundle("f14-top-level-file"),
              disk={"notes.md": "Ask for the receipt number first.\n"})
    p.install(bundle("f15-nested-skill-md", REFERENCES), disk={
        "references/SKILL.md": '---\nname: references\ndescription: "A skill inside a skill."\n'
                               "---\nbody\n"})

    # The tests turn these into the defect: git keeps neither symlinks nor modes reliably.
    p.install(bundle("v1-symlinked-folder"))
    p.install(bundle("v1-symlinked-resource", REFERENCES))
    p.install(bundle("v4-exec-bit", SCRIPTS))

    p.install(bundle("v3-edited-attachment", REFERENCES),
              disk={"references/fees.md": FEES + "| apparel | 5% |\n"})

    # Folders the lock does not list.
    p.folder("v2-nested-folder", {"refund-policy/SKILL.md": hand_written("refund-policy")})
    orphan = bundle("v8-stamped-orphan")
    p.folder(orphan.install_name, {f.path: f.content for f in orphan.files})
    p.folder("hand-authored-clean", {"SKILL.md": hand_written("hand-authored-clean")})
    p.folder("hand-authored-unknown-key", {
        "SKILL.md": hand_written("hand-authored-unknown-key", "category: support\n")})
    p.folder("hand-authored-namespaced-name", {
        "SKILL.md": hand_written("acme/hand-authored-namespaced-name")})
    p.folder("hand-authored-companion-files", with_companions("hand-authored-companion-files"))
    return p.finish()


def plugin_json_stale() -> Dict[str, bytes]:
    return _one_skill("plugin-json-stale").finish(plugin=L.plugin_json_bytes("0" * 64))


def lock_hash_edited() -> Dict[str, bytes]:
    # A hand edit of a field nothing else checks, without recomputing lock_hash.
    p = _one_skill("lock-hash-edited")
    data = p.lock.to_dict()
    data["skills"]["acme-refund-policy"]["source"]["base_url"] = "https://registry.example.test"
    return p.finish(lock=_lock_bytes(data))


def foreign_entry() -> Dict[str, bytes]:
    # What SDK 0.13.9 does to a version 2 lock it finds: it merges a version 1 entry in,
    # keyed by registry name, and leaves lock_hash alone.
    p = _one_skill("foreign-entry")
    data = p.lock.to_dict()
    data["skills"]["acme/returns-desk"] = {
        "version": 3,
        "content_hash": _sha("# Acme returns desk\n"),
        "body_hash": _sha("# Acme returns desk"),
        "source": "platform",
        "skill_id": _id(0x5EED00FF),
        "installed_at": INSTALLED_AT,
        "agents": ["universal"],
        "scope": "project",
    }
    p.folder("acme-returns-desk", {"SKILL.md": hand_written("acme-returns-desk")})
    return p.finish(lock=_lock_bytes(data))


def manifest_disagrees() -> Dict[str, bytes]:
    # The manifest pins a version the lock does not hold, and wants a skill never installed.
    p = _one_skill("manifest-disagrees")
    manifest = M.new()
    manifest.set("acme/refund-policy", "4")
    manifest.set("acme/returns-desk", "latest")
    return p.finish(manifest=manifest.dumps())


def interrupted() -> Dict[str, bytes]:
    # A run that died between writing the lock and deleting its journal.
    p = _one_skill("interrupted")
    entry = p.entries["acme-refund-policy"]
    journal = {"txid": "20261015T000000Z-0a0b0c0d", "name": "acme-refund-policy",
               "old_entry": None, "new_entry": entry, "old_present": False}
    p.put(f"{DECIMAL}/skills.lock.journal", _lock_bytes(journal))
    return p.finish()


def lock_v1() -> Dict[str, bytes]:
    # A version 1 lock, as SDK 0.13.9 writes it: keyed by registry name, no ids of the
    # version, no file hashes.
    p = _Project("lock-v1")
    p.folder("acme-refund-policy", {"SKILL.md": hand_written("acme-refund-policy")})
    p.put(f"{DECIMAL}/skills.lock", _lock_bytes({
        "version": 1,
        "skills": {"acme/refund-policy": {
            "version": 3, "content_hash": _sha(BODY), "body_hash": _sha(BODY.strip()),
            "source": "platform", "skill_id": _id(0x5EED0000), "installed_at": INSTALLED_AT,
            "agents": ["universal"], "scope": "project"}},
        "updated_at": INSTALLED_AT,
    }))
    return p.out()


def no_lock() -> Dict[str, bytes]:
    # Skills a person put in .agents/skills before (or without) `decimalai skills add`: two
    # hand-authored ones and a loose copy of a registry skill, which carries its stamp.
    p = _Project("no-lock")
    p.folder("hand-authored-clean", {"SKILL.md": hand_written("hand-authored-clean"),
                                     "reference.md": "# Reference\n"})
    p.folder("hand-authored-unknown-key", {
        "SKILL.md": hand_written("hand-authored-unknown-key", "category: support\n")})
    pulled = render("pulled-skill", 0x40, REFERENCES)
    p.folder(pulled.install_name, {f.path: f.content for f in pulled.files})
    return p.out()


def build() -> Dict[str, bytes]:
    """Every fixture file: path relative to this directory -> exact bytes."""
    out: Dict[str, bytes] = {}
    for make in (clean, managed, plugin_json_stale, lock_hash_edited, foreign_entry,
                 manifest_disagrees, interrupted, lock_v1, no_lock):
        out.update(make())
    assert {path.split("/")[0] for path in out} == set(PROJECTS)
    return dict(sorted(out.items()))


def write(dest: Path) -> None:
    for rel, data in build().items():
        path = dest / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def main() -> int:
    for project in PROJECTS:
        shutil.rmtree(HERE / project, ignore_errors=True)
    write(HERE)
    print(f"wrote {len(PROJECTS)} projects under {HERE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
