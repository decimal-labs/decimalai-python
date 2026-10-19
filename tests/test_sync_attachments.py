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
    monkeypatch.chdir(tmp_path)  # the sync stamps the working directory's install identity
    sync_to_platform("dai_sk_test", search_paths=[str(tmp_path)])

    assert captured["skills"]["with-bundle"]["attachments"] == {"references/table.csv": "a,b\n1,2\n"}
    assert "attachments" not in captured["skills"]["bare"], (
        "an absent key means 'leave attachments alone' on the server; an empty map must not be sent"
    )


def test_sync_to_platform_skips_a_registry_install_and_its_bundle(tmp_path, monkeypatch):
    """A folder that `decimalai skills add` installed carries the registry's stamp
    (``metadata.decimalai-skill-id``). Syncing it would mint the org a private copy of
    someone else's skill, bundle included, so it is not sent. Its unmanaged neighbour is."""
    _skill(tmp_path, "mine")
    installed = tmp_path / "acme-refund-policy"
    installed.mkdir()
    (installed / "SKILL.md").write_text(
        "---\nname: acme-refund-policy\ndescription: \"Refund rules.\"\nmetadata:\n"
        "  decimalai-skill-id: \"6f1c0000-0000-0000-0000-000000000001\"\n"
        "  decimalai-version-id: \"9b2e0000-0000-0000-0000-000000000003\"\n"
        "  decimalai-version: \"3\"\n"
        f"  decimalai-body-sha256: \"{'0' * 64}\"\n"
        "  decimalai-name: \"acme/refund-policy\"\n"
        "  decimalai-format: \"1\"\n"
        "  decimalai-invocation: \"model\"\n"
        "---\n# Acme refund policy\nRefunds within thirty days of delivery.\n"
    )
    (installed / "references").mkdir()
    (installed / "references" / "fees.md").write_text("# fees\n")

    captured = {}

    class FakeRouter:
        def __init__(self, **kw):
            pass

        def sync_skills(self, skills, **kwargs):
            captured["skills"] = {s["name"]: s for s in skills}
            return {"created": 1, "updated": 0, "unchanged": 0}

    monkeypatch.setattr("decimalai.skill_router.SkillRouter", FakeRouter)
    monkeypatch.chdir(tmp_path)  # the sync stamps the working directory's install identity
    sync_to_platform("dai_sk_test", search_paths=[str(tmp_path)])

    assert set(captured["skills"]) == {"mine"}


def test_sync_to_platform_sends_nothing_when_every_folder_is_a_registry_install(
    tmp_path, monkeypatch
):
    installed = tmp_path / "acme-refund-policy"
    installed.mkdir()
    (installed / "SKILL.md").write_text(
        "---\nname: acme-refund-policy\ndescription: d\nmetadata:\n"
        "  decimalai-skill-id: \"6f1c\"\n---\n# Acme\nRefunds within thirty days.\n"
    )

    class FakeRouter:
        def __init__(self, **kw):
            raise AssertionError("no request is made when there is nothing to sync")

    monkeypatch.setattr("decimalai.skill_router.SkillRouter", FakeRouter)
    assert sync_to_platform("dai_sk_test", search_paths=[str(tmp_path)]) == {
        "created": 0, "updated": 0, "unchanged": 0,
    }


def test_cli_sync_sends_a_category_only_when_it_is_one_string(tmp_path, monkeypatch):
    """The sync endpoint takes ``category`` as a string and rejects the WHOLE batch (422)
    when one skill sends a list. Read as YAML, ``category: [devops, ci]`` is a list, so it
    is sent as no category; the frontmatter still carries the list as written."""
    from unittest.mock import MagicMock, patch

    from click.testing import CliRunner

    from decimalai.cli.main import cli

    monkeypatch.delenv("DECIMALAI_INSTALL_LABEL", raising=False)
    for name, category in (("listed", "[devops, ci]"), ("single", "devops")):
        sdir = tmp_path / "skills" / name
        sdir.mkdir(parents=True)
        (sdir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: d\ncategory: {category}\n---\n\n# {name}\nBody.\n",
            encoding="utf-8",
        )
    client = MagicMock()
    client._http.post.return_value.json.return_value = {"actions": []}
    client._http.post.return_value.status_code = 200
    monkeypatch.chdir(tmp_path)
    with patch("decimalai.cli.main._make_client", return_value=client):
        result = CliRunner().invoke(cli, [
            "skills", "sync", "--api-key", "k", "--base-url", "http://localhost:8000",
        ])
    assert result.exit_code == 0, result.output
    sent = {s["name"]: s for s in client._http.post.call_args.kwargs["json"]["skills"]}
    assert sent["listed"]["category"] is None
    assert sent["listed"]["frontmatter"]["category"] == ["devops", "ci"]
    assert sent["single"]["category"] == "devops"
