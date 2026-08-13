"""Tests for per-install sync divergence: `_install` identity + `skills status`/`sync`.

Three surfaces, one feature — reporting drift without bidirectional sync:

  * ``decimalai._install`` persists a *per-checkout* install identity in
    ``.decimal/install.json`` so the platform can attribute drift to one
    workspace. It must be stable across runs, kept out of git, and never
    written during a dry-run preview.
  * ``decimalai skills status`` hashes local SKILL.md bodies (the SAME body
    hash `skills sync` uses, so a freshly-synced skill reads in_sync) and
    asks ``POST /skills/installs/report`` how each compares to this install's
    baseline, rendering drift-first.
  * ``decimalai skills sync`` stamps that same install identity onto
    ``POST /skills/sync`` (so the backend can record this install's synced
    baseline) and writes ``pulled`` bodies back to disk — the *reconcile*
    half of the loop — preserving local frontmatter and honouring
    ``--no-apply-pulls``.

Network is mocked at ``_make_client`` — these lock command wiring (endpoint,
posted body, render order, disk effects), not backend status math (covered
platform-side).
"""

import hashlib
import json
import os
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from decimalai import _install
from decimalai.cli.main import cli


# ── _install identity ──────────────────────────────────────


class TestInstallIdentity:
    def test_stable_id_across_calls(self, tmp_path):
        # The id is the install's stable handle — a second call must not mint
        # a new one, or every run would look like a different workspace.
        a = _install.get_install_identity(str(tmp_path))
        b = _install.get_install_identity(str(tmp_path))
        assert a["install_id"] and a["install_id"] == b["install_id"]
        assert (tmp_path / ".decimal" / "install.json").is_file()

    def test_writes_gitignore_excluding_install_json(self, tmp_path):
        # Per-checkout identity must never be committed, so creating it drops a
        # .gitignore that excludes the file.
        _install.get_install_identity(str(tmp_path))
        gi = tmp_path / ".decimal" / ".gitignore"
        assert gi.is_file()
        assert "install.json" in gi.read_text()

    def test_create_false_writes_nothing(self, tmp_path):
        # --dry-run path: a preview returns an in-memory identity but must leave
        # no footprint on disk.
        ident = _install.get_install_identity(str(tmp_path), create=False)
        assert ident["install_id"]
        assert not (tmp_path / ".decimal").exists()

    def test_corrupt_file_is_regenerated(self, tmp_path):
        # A truncated/garbage file shouldn't wedge the CLI — regenerate and
        # rewrite valid JSON in place.
        state = tmp_path / ".decimal"
        state.mkdir()
        bad = state / "install.json"
        bad.write_text("{ this is not json", encoding="utf-8")

        ident = _install.get_install_identity(str(tmp_path))

        assert ident.get("install_id")
        on_disk = json.loads(bad.read_text(encoding="utf-8"))
        assert on_disk["install_id"] == ident["install_id"]

    def test_existing_label_is_not_overwritten(self, tmp_path):
        # A pre-recorded label wins; passing a new label must not clobber it.
        first = _install.get_install_identity(str(tmp_path), label="ci-box")
        assert first["install_label"] == "ci-box"
        again = _install.get_install_identity(str(tmp_path), label="other")
        assert again["install_label"] == "ci-box"
        assert again["install_id"] == first["install_id"]

    def test_find_project_root_prefers_existing_decimal_dir(self, tmp_path):
        # A run from a subdirectory must share the repo's install_id, so the
        # root resolver prefers an ancestor that already has .decimal/.
        root = tmp_path / "repo"
        (root / ".decimal").mkdir(parents=True)
        sub = root / "pkg" / "nested"
        sub.mkdir(parents=True)
        assert _install.find_project_root(str(sub)) == str(root)


# ── skills status ──────────────────────────────────────────


def _mock_response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


# name in frontmatter (wins over the "alpha-dir" parent folder).
SKILL_ALPHA = """---
name: alpha
description: first
---

# Alpha

Alpha body.
"""

# no name in frontmatter → falls back to the parent directory name ("bravo").
SKILL_BRAVO = """---
description: second, unnamed
---

# Bravo

Bravo body.
"""

ALPHA_HASH = hashlib.sha256(b"# Alpha\n\nAlpha body.").hexdigest()
BRAVO_HASH = hashlib.sha256(b"# Bravo\n\nBravo body.").hexdigest()

# What the backend hands back as the newer remote body for a `pulled` action.
ALPHA_REMOTE_BODY = "# Alpha\n\nAlpha body UPDATED from remote."


def _seed(name_dir, content):
    os.makedirs(f"skills/{name_dir}", exist_ok=True)
    with open(f"skills/{name_dir}/SKILL.md", "w", encoding="utf-8") as f:
        f.write(content)


class TestSkillsStatusCommand:
    def test_status_registered_under_skills(self):
        result = CliRunner().invoke(cli, ["skills", "--help"])
        assert result.exit_code == 0
        assert "status" in result.output

    def test_reports_hashes_and_renders_drift_first(self):
        client = MagicMock()
        # Server replies in_sync-first on purpose — the client must re-sort so
        # drift can't hide below a wall of in_sync rows.
        client._http.post.return_value = _mock_response({
            "skills": [
                {"name": "bravo", "status": "in_sync"},
                {"name": "alpha", "status": "conflict"},
            ]
        })
        runner = CliRunner()
        with runner.isolated_filesystem():
            _seed("alpha-dir", SKILL_ALPHA)
            _seed("bravo", SKILL_BRAVO)
            with patch("decimalai.cli.main._make_client", return_value=client):
                result = runner.invoke(cli, [
                    "skills", "status",
                    "--api-key", "k", "--base-url", "http://localhost:8000",
                ])

        assert result.exit_code == 0, result.output

        # Reported to the installs endpoint, stamped with this checkout's id.
        path = client._http.post.call_args[0][0]
        body = client._http.post.call_args.kwargs["json"]
        assert path == "/api/v1/skills/installs/report"
        assert body["install_id"]

        # Hash is SHA-256 of the frontmatter-stripped body (byte-identical to
        # what `skills sync` sends), and the frontmatter name beats the folder.
        posted = {s["name"]: s["content_hash"] for s in body["skills"]}
        assert posted == {"alpha": ALPHA_HASH, "bravo": BRAVO_HASH}

        # conflict (drift) sorts above in_sync, and the reconcile hint shows.
        assert result.output.index("alpha") < result.output.index("bravo")
        assert "conflict" in result.output and "in_sync" in result.output
        assert "decimalai skills sync" in result.output

    def test_all_in_sync_omits_reconcile_hint(self):
        client = MagicMock()
        client._http.post.return_value = _mock_response({
            "skills": [{"name": "alpha", "status": "in_sync"}]
        })
        runner = CliRunner()
        with runner.isolated_filesystem():
            _seed("alpha", SKILL_ALPHA)
            with patch("decimalai.cli.main._make_client", return_value=client):
                result = runner.invoke(cli, [
                    "skills", "status",
                    "--api-key", "k", "--base-url", "http://localhost:8000",
                ])
        assert result.exit_code == 0, result.output
        assert "in_sync=1" in result.output
        assert "decimalai skills sync" not in result.output

    def test_no_skill_files_short_circuits_before_network(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            os.makedirs("skills")  # exists for click, but holds no SKILL.md
            with patch("decimalai.cli.main._make_client") as mk:
                result = runner.invoke(cli, [
                    "skills", "status",
                    "--api-key", "k", "--base-url", "http://localhost:8000",
                ])
                mk.assert_not_called()
        assert result.exit_code == 0, result.output
        assert "No SKILL.md files found" in result.output


# ── skills sync (the install-stamp + pull-reconcile half) ─────


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class TestSkillsSyncInstallWiring:
    def test_sync_stamps_install_identity_into_post(self):
        client = MagicMock()
        client._http.post.return_value = _mock_response(
            {"actions": [{"action": "no_change", "name": "alpha"}]}
        )
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Pre-record this checkout's identity with an explicit (opt-in)
            # label so the assertion is exact — otherwise there is no label at
            # all and the id would be freshly minted inside the call. The
            # no-label default is covered in
            # tests/test_no_machine_identity_in_payloads.py.
            ident = _install.get_install_identity(".", label="ci-box")
            _seed("alpha-dir", SKILL_ALPHA)
            with patch("decimalai.cli.main._make_client", return_value=client):
                result = runner.invoke(cli, [
                    "skills", "sync",
                    "--api-key", "k", "--base-url", "http://localhost:8000",
                ])

        assert result.exit_code == 0, result.output

        # Exactly one POST, to the sync endpoint…
        assert client._http.post.call_count == 1
        path = client._http.post.call_args[0][0]
        body = client._http.post.call_args.kwargs["json"]
        assert path == "/api/v1/skills/sync"

        # …carrying THIS install's id + label (so the backend can record a
        # per-install synced baseline) plus the always-on reconcile knobs.
        assert body["install_id"] == ident["install_id"]
        assert body["install_label"] == "ci-box"
        assert body["conflict_policy"] == "newer_wins"
        assert body["response_mode"] == "diff"

        # …and the discovered skill, hashed on the frontmatter-stripped body
        # byte-identically to `skills status` (so a fresh sync reads in_sync).
        posted = {s["name"]: s["content_hash"] for s in body["skills"]}
        assert posted == {"alpha": ALPHA_HASH}

    def test_pulled_action_overwrites_local_body_preserving_frontmatter(self):
        client = MagicMock()
        client._http.post.return_value = _mock_response({
            "actions": [{
                "action": "pulled",
                "name": "alpha",
                "body_markdown": ALPHA_REMOTE_BODY,
                "version_number": 7,
            }]
        })
        runner = CliRunner()
        with runner.isolated_filesystem():
            _seed("alpha-dir", SKILL_ALPHA)
            with patch("decimalai.cli.main._make_client", return_value=client):
                result = runner.invoke(cli, [
                    "skills", "sync",
                    "--api-key", "k", "--base-url", "http://localhost:8000",
                ])
            assert result.exit_code == 0, result.output
            on_disk = _read("skills/alpha-dir/SKILL.md")

        # The newer remote body is reconciled onto disk by default…
        assert "Alpha body UPDATED from remote." in on_disk
        # …and only the body is pulled — local frontmatter survives intact.
        assert "name: alpha" in on_disk

    def test_no_apply_pulls_leaves_local_file_untouched(self):
        client = MagicMock()
        client._http.post.return_value = _mock_response({
            "actions": [{
                "action": "pulled",
                "name": "alpha",
                "body_markdown": ALPHA_REMOTE_BODY,
                "version_number": 7,
            }]
        })
        runner = CliRunner()
        with runner.isolated_filesystem():
            _seed("alpha-dir", SKILL_ALPHA)
            with patch("decimalai.cli.main._make_client", return_value=client):
                result = runner.invoke(cli, [
                    "skills", "sync", "--no-apply-pulls",
                    "--api-key", "k", "--base-url", "http://localhost:8000",
                ])
            assert result.exit_code == 0, result.output
            on_disk = _read("skills/alpha-dir/SKILL.md")

        # Opt-out is honoured: disk is exactly as seeded, remote body NOT written.
        assert "Alpha body." in on_disk
        assert "UPDATED from remote" not in on_disk
        assert "skipping disk writes" in result.output

    def test_sync_422_surfaces_per_field_validation_errors(self):
        """A backend 422 (e.g. an agentskills.io name-rule violation) must
        print the actionable per-field message, not just '422'."""
        resp = MagicMock()
        resp.status_code = 422
        resp.json.return_value = {
            "error": "validation_error",
            "details": {
                "errors": [{
                    "field": "body.skills.0.name",
                    "message": "Skill name 'Emoji 🚀' does not match agentskills.io spec",
                }],
            },
        }
        client = MagicMock()
        client._http.post.return_value = resp
        runner = CliRunner()
        with runner.isolated_filesystem():
            _seed("alpha-dir", SKILL_ALPHA)
            with patch("decimalai.cli.main._make_client", return_value=client):
                result = runner.invoke(cli, [
                    "skills", "sync",
                    "--api-key", "k", "--base-url", "http://localhost:8000",
                ])
        assert result.exit_code == 1
        assert "Sync rejected by validation" in result.output
        assert "body.skills.0.name" in result.output
        assert "agentskills.io spec" in result.output
        assert "Traceback" not in result.output
