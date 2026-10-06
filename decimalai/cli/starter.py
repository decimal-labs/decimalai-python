"""Public starter download: no authenticated client and safe verified extraction."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import click

MAX_ZIP_BYTES = 2_000_000
MAX_EXPANDED_BYTES = 1_000_000
MAX_FILES = 512


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_archive_name(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value or path.is_absolute() or "\\" in value or ":" in value
        or any(ord(c) < 32 for c in value)
        or any(p in ("", ".", "..") for p in value.split("/"))
    ):
        raise ValueError("Starter archive contains an unsafe file path.")
    return value


def extract_starter(content: bytes, project: Path, expected_snapshot_hash: str) -> dict:
    """Validate the entire ZIP first, then create a new project atomically."""
    if project.exists() or project.is_symlink():
        raise ValueError(f"{project} already exists; choose a new project directory.")
    if len(content) > MAX_ZIP_BYTES:
        raise ValueError("Starter archive is too large.")
    files = {}
    seen = set()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        entries = archive.infolist()
        if len(entries) > MAX_FILES or sum(entry.file_size for entry in entries) > MAX_EXPANDED_BYTES:
            raise ValueError("Starter archive exceeds the file/expanded-size limit.")
        for entry in entries:
            name = _safe_archive_name(entry.filename)
            mode = entry.external_attr >> 16
            if entry.is_dir() or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                raise ValueError("Starter archive must contain regular files only.")
            if name.casefold() in seen:
                raise ValueError("Starter archive contains conflicting file paths.")
            seen.add(name.casefold())
            files[name] = archive.read(entry)
    if "bundle-manifest.json" not in files or "snapshot.json" not in files or "agent.py" not in files:
        raise ValueError("Starter archive is missing its manifest, snapshot, or runtime.")
    manifest = json.loads(files["bundle-manifest.json"])
    snapshot = json.loads(files["snapshot.json"])
    hashes = manifest.get("files")
    if not isinstance(hashes, dict) or set(hashes) != set(files) - {"bundle-manifest.json"}:
        raise ValueError("Starter archive file list does not match its manifest.")
    if any(_digest(files[name]) != digest for name, digest in hashes.items()):
        raise ValueError("Starter file hash verification failed.")
    actual = _digest(json.dumps(
        {k: v for k, v in snapshot.items() if k != "snapshot_hash"},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8"))
    if not expected_snapshot_hash or actual != expected_snapshot_hash or snapshot.get("snapshot_hash") != actual or manifest.get("snapshot_hash") != actual:
        raise ValueError("Starter snapshot does not match the server's content hash.")
    if snapshot.get("schema_version") != 1 or snapshot.get("runtime_version") != "support-stdlib-v1":
        raise ValueError("Unsupported starter snapshot/runtime version. Update your SDK.")
    if not isinstance(snapshot.get("name"), str) or not snapshot["name"].strip() or not isinstance(snapshot.get("skills"), list) or not 1 <= len(snapshot["skills"]) <= 8:
        raise ValueError("Starter snapshot is missing its name or selected skills.")
    project.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".starter-", dir=project.parent))
    try:
        for name, data in files.items():
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        # os.rename refuses an existing nonempty project. Check lexists too:
        # existing symlinks and empty directories should never be replaced.
        if os.path.lexists(project):
            raise ValueError(f"{project} already exists; choose a new project directory.")
        os.rename(staging, project)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return snapshot


@click.command("starter")
@click.argument("pack", type=click.Choice(["support"]))
@click.option("--project", "project_dir", type=click.Path(path_type=Path), required=True, help="New directory for the runnable project.")
@click.option("--name", default="support-agent", show_default=True, help="Local agent name.")
@click.option("--prompt-file", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Use your reviewed system prompt file.")
@click.option("--skill-id", "skill_ids", multiple=True, help="Select specific starter skill IDs (default: all available starters).")
@click.option("--pack-revision", help="Require the exact revision you reviewed; a changed revision fails.")
@click.option("--base-url", envvar=["DECIMAL_BASE_URL", "DECIMALAI_BASE_URL"], default="https://api.decimal.ai", show_envvar=True, help="Public registry API URL.")
def starter(pack, project_dir, name, prompt_file, skill_ids, pack_revision, base_url):
    """Create a real local Support agent without a DecimalAI account.

    Download only; no model call is made. Run agent.py with your OpenAI key.
    The project requires Python 3.10+ and no pip dependencies or telemetry.
    """
    import httpx

    project = project_dir.expanduser().absolute()
    if os.path.lexists(project):
        raise click.ClickException(f"{project} already exists; choose a new project directory.")
    url = f"{base_url.rstrip('/')}/api/v1/registry/packs/{pack}"
    try:
        # Explicitly no auth/client API key, cookies, or environment credentials.
        with httpx.Client(timeout=30, follow_redirects=False, trust_env=False) as http:
            response = http.get(url)
            response.raise_for_status()
            detail = response.json()
            current_revision = detail["starter_selection_revision"]
            if pack_revision and pack_revision != current_revision:
                raise ValueError("The pack changed since your reviewed revision. Review it again before downloading.")
            selected = list(skill_ids) or [row["id"] for row in detail["starter_skills"]]
            if not selected:
                raise ValueError("This pack currently has no available starter skills.")
            if detail.get("starter_skills_missing") and not skill_ids:
                raise ValueError("Some starter skills are unavailable. Review the pack and explicitly select --skill-id values.")
            prompt = prompt_file.read_text(encoding="utf-8") if prompt_file else detail["starter_prompt"]
            with http.stream("POST", url + "/starter", json={
                "pack_revision": pack_revision or current_revision, "name": name,
                "system_prompt": prompt, "selected_skill_ids": selected,
            }) as downloaded:
                if downloaded.status_code >= 400:
                    downloaded.read()
                downloaded.raise_for_status()
                chunks = []
                size = 0
                for chunk in downloaded.iter_bytes():
                    size += len(chunk)
                    if size > MAX_ZIP_BYTES:
                        raise ValueError("Starter archive is too large.")
                    chunks.append(chunk)
                snapshot = extract_starter(b"".join(chunks), project, downloaded.headers.get("X-Decimal-Snapshot-Hash", ""))
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 409:
            message = "The reviewed pack or a required skill file changed. Review your selection and retry; no project was created."
        else:
            message = f"Public starter download returned HTTP {exc.response.status_code}; no project was created."
        raise click.ClickException(message) from None
    except httpx.HTTPError:
        raise click.ClickException("Public registry connection failed; no project was created. Retry the download.") from None
    except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile):
        # Never print reflected server text (which could contain user prompt).
        raise click.ClickException("Starter validation failed or selection is incomplete. Review the pack, choose a new project path, and retry.") from None
    click.echo(f"Created {snapshot['name']} in {project} with {len(snapshot['skills'])} frozen public skills.")
    click.echo("No model run has been performed. Next:")
    click.echo(f"  cd {project}")
    click.echo("  python3 agent.py --check-setup")
    click.echo("  export OPENAI_API_KEY='your-provider-key'")
    click.echo("  python3 agent.py --checks  # two tasks + two model-judge calls; provider charges apply")
