"""Keyless starter download validates all bytes before writing a new project."""

import hashlib
import io
import json
import zipfile
from unittest.mock import patch

import httpx
import pytest
from click.testing import CliRunner

from decimalai.cli.main import cli
from decimalai.cli.starter import extract_starter


def bundle(extra=None):
    snapshot = {"schema_version": 1, "runtime_version": "support-stdlib-v1", "name": "support-agent", "skills": [{"id": "s1"}], "system_prompt": "Reviewed prompt"}
    digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    snapshot["snapshot_hash"] = digest
    files = {"agent.py": b"# Standalone runtime\n", "snapshot.json": json.dumps(snapshot).encode(), "skills/reply/SKILL.md": b"Use the invoice identifier.\n", **(extra or {})}
    manifest = {"schema_version": 1, "snapshot_hash": digest, "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    files["bundle-manifest.json"] = json.dumps(manifest).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return output.getvalue(), digest


def test_anonymous_starter_command_never_sends_decimal_key(monkeypatch, tmp_path):
    monkeypatch.setenv("DECIMAL_API_KEY", "workspace-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    content, digest = bundle()
    captured = []

    def handler(request):
        captured.append(request)
        assert "authorization" not in request.headers
        assert b"workspace-secret" not in request.content and b"provider-secret" not in request.content
        if request.method == "GET":
            return httpx.Response(200, json={"starter_selection_revision": "a" * 64, "starter_prompt": "Reviewed prompt", "starter_skills": [{"id": "s1"}], "starter_skills_missing": []})
        assert json.loads(request.content) == {"pack_revision": "a" * 64, "name": "support-agent", "system_prompt": "Reviewed prompt", "selected_skill_ids": ["s1"]}
        return httpx.Response(200, content=content, headers={"X-Decimal-Snapshot-Hash": digest})

    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    with patch("decimalai._client.DecimalAIClient", side_effect=AssertionError("No authenticated client")):
        result = CliRunner().invoke(cli, ["starter", "support", "--project", str(tmp_path / "new")])
    assert result.exit_code == 0, result.output
    assert len(captured) == 2
    assert (tmp_path / "new" / "agent.py").is_file()
    assert "No model run has been performed" in result.output


@pytest.mark.parametrize("path", ["../outside", "/tmp/outside", "skills/../../outside", "skills\\outside", "C:outside"])
def test_unsafe_archive_never_writes_anything(tmp_path, path):
    content, digest = bundle({path: b"payload"})
    with pytest.raises(ValueError, match="unsafe"):
        extract_starter(content, tmp_path / "new", digest)
    assert not (tmp_path / "new").exists()
    assert not (tmp_path / "outside").exists()


def test_tampered_hash_and_wrong_snapshot_are_refused(tmp_path):
    content, digest = bundle()
    with pytest.raises(ValueError, match="content hash"):
        extract_starter(content, tmp_path / "wrong", "0" * 64)
    rewritten = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(content)) as source, zipfile.ZipFile(rewritten, "w") as target:
        for name in source.namelist():
            target.writestr(name, b"tampered" if name == "agent.py" else source.read(name))
    with pytest.raises(ValueError, match="file hash"):
        extract_starter(rewritten.getvalue(), tmp_path / "tampered", digest)
    assert not (tmp_path / "tampered").exists()


def test_does_not_replace_existing_directory_or_symlink(tmp_path):
    content, digest = bundle()
    project = tmp_path / "exists"
    project.mkdir()
    (project / "keep").write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        extract_starter(content, project, digest)
    assert (project / "keep").read_text() == "keep"
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "absent", target_is_directory=True)
    with pytest.raises(ValueError, match="already exists"):
        extract_starter(content, link, digest)


def test_symlink_zip_member_is_refused(tmp_path):
    content, digest = bundle()
    rewritten = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(content)) as source, zipfile.ZipFile(rewritten, "w") as target:
        for name in source.namelist():
            info = zipfile.ZipInfo(name)
            if name == "agent.py":
                info.external_attr = 0o120777 << 16
            target.writestr(info, source.read(name))
    with pytest.raises(ValueError, match="regular files"):
        extract_starter(rewritten.getvalue(), tmp_path / "new", digest)


def test_stale_review_does_not_retry_with_new_selection(monkeypatch, tmp_path):
    captured = []

    def handler(request):
        captured.append(request)
        return httpx.Response(200, json={"starter_selection_revision": "b" * 64, "starter_prompt": "New prompt", "starter_skills": [{"id": "s2"}]})

    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    result = CliRunner().invoke(cli, ["starter", "support", "--project", str(tmp_path / "new"), "--pack-revision", "a" * 64])
    assert result.exit_code == 1
    assert len(captured) == 1
    assert not (tmp_path / "new").exists()
