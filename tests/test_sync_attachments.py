"""`skills sync` carries the bundle (CUJ 27).

The sync endpoint has accepted an optional ``attachments`` map since 2026-09-05; until
2026-09-13 neither sync path in this SDK sent one, so a `references/` file an author wrote
never reached the platform, while `skills pull` happily delivered bundles that only the
platform's own disk loader had been able to create.
"""

from __future__ import annotations

from decimalai import skills as sk
from decimalai.skills import collect_bundle_attachments, sync_to_platform


def _skill(tmp_path, name="bundled"):
    sdir = tmp_path / name
    sdir.mkdir()
    (sdir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: a skill with a bundle\n---\n\n"
        "## When to use\n\nUse this when the bundle must travel with the body. " * 4
    )
    return sdir


def test_collects_text_files_one_level_under_the_four_directories(tmp_path):
    sdir = _skill(tmp_path)
    (sdir / "references").mkdir()
    (sdir / "references" / "icd10.md").write_text("# codes\n")
    (sdir / "scripts").mkdir()
    (sdir / "scripts" / "check.py").write_text("print('ok')\n")
    (sdir / "assets").mkdir()
    (sdir / "assets" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff\xfe")   # binary: skipped
    (sdir / "references" / "deep").mkdir()
    (sdir / "references" / "deep" / "x.md").write_text("too deep\n")                 # nested: skipped
    (sdir / "notes.txt").write_text("not in an attachment directory\n")                # root: skipped

    got = collect_bundle_attachments(str(sdir))
    assert got == {"references/icd10.md": "# codes\n", "scripts/check.py": "print('ok')\n"}
    assert list(got) == sorted(got), "payload order must be stable"


def test_oversized_files_are_skipped_and_the_count_is_capped(tmp_path, monkeypatch):
    sdir = _skill(tmp_path)
    (sdir / "references").mkdir()
    (sdir / "references" / "big.txt").write_text("x" * (sk.MAX_ATTACHMENT_BYTES + 1))
    for i in range(5):
        (sdir / "references" / f"f{i}.md").write_text(f"file {i}\n")
    monkeypatch.setattr(sk, "MAX_ATTACHMENTS", 3)
    got = collect_bundle_attachments(str(sdir))
    assert "references/big.txt" not in got
    assert len(got) == 3


def test_no_bundle_means_no_key_and_an_empty_dir_means_none(tmp_path):
    sdir = _skill(tmp_path)
    assert collect_bundle_attachments(str(sdir)) == {}
    assert collect_bundle_attachments("") == {}


def test_sync_to_platform_sends_the_bundle_only_when_there_is_one(tmp_path, monkeypatch):
    with_bundle = _skill(tmp_path, "with-bundle")
    (with_bundle / "references").mkdir()
    (with_bundle / "references" / "table.csv").write_text("a,b\n1,2\n")
    _skill(tmp_path, "bare")

    captured = {}

    class FakeRouter:
        def __init__(self, **kw):
            pass

        def sync_skills(self, skills, **kwargs):
            captured["skills"] = {s["name"]: s for s in skills}
            return {"created": 2, "updated": 0, "unchanged": 0}

    monkeypatch.setattr("decimalai.skill_router.SkillRouter", FakeRouter)
    sync_to_platform("dai_sk_test", search_paths=[str(tmp_path)])

    assert captured["skills"]["with-bundle"]["attachments"] == {"references/table.csv": "a,b\n1,2\n"}
    assert "attachments" not in captured["skills"]["bare"], (
        "an absent key means 'leave attachments alone' on the server; an empty map must not be sent"
    )
