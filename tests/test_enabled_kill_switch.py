"""``init(enabled=False)`` must be a real kill switch.

The regression this locks down: ``init(enabled=False, langchain=True)`` used to
skip only the client, then fall through to the framework auto-install block
anyway. ``decimalai.langchain.instrument()`` is not a passive no-op — it calls
the DecimalAI API (``GET /api/v1/skills/hashes``, then the per-skill body
routes) and writes SKILL.md files into the caller's working tree. So a user who
explicitly turned the SDK off still got network traffic and new files on disk.

Both tests drive the REAL adapter against a fake httpx transport in a temp cwd,
so they observe the two effects a user actually cares about:

* ``test_disabled_*`` — nothing attempted, nothing written (fails without the
  gate: the old code records ``GET .../skills/hashes`` and writes a SKILL.md).
* ``test_enabled_*`` — the control. It proves the gate is conditional, not a
  blanket off-switch: with ``enabled=True`` the same call still syncs and still
  writes.

The control asks for disk mirroring explicitly (``skill_authority="harness"``).
It did not have to when this file was written, because writing to the caller's
working tree was the unconditional default — the very default that turned out to
be the bug in ``tests/test_disk_mirror_consent.py``: instrumenting an agent
created ``.agents/skills/`` in whatever directory the user ran from, and with no
disk-loading runtime in the process nothing read those files back. Consent is
now required, so the control states it. What this file tests is unchanged:
``enabled`` gates the adapter, and nothing else about it moved.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

import httpx
import pytest


# The fake platform serves one skill that is NOT on local disk, so the pull
# path classifies it as "missing" and exports it — i.e. a real file write.
_SKILL_NAME = "kill-switch-demo-skill"
_SKILL_BODY = {
    "skill_id": "skl_kill_switch_demo",
    "name": _SKILL_NAME,
    "description": "Demo skill served by the fake platform.",
    "body": "# Kill Switch Demo\n\nThis body should only ever reach disk when enabled.\n",
}


class _FakePlatform:
    """Records every HTTP attempt and answers the skill-pull endpoints."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def __call__(self, method: str, url: Any, **kwargs: Any) -> httpx.Response:
        url = str(url)
        self.calls.append(f"{method} {url}")
        if "/skills/hashes" in url:
            return httpx.Response(
                200, json={"hashes": {_SKILL_NAME: {"hash": "deadbeefcafe00"}}}
            )
        if url.endswith("/body"):
            return httpx.Response(200, json=_SKILL_BODY)
        return httpx.Response(200, json={})


@pytest.fixture
def platform(monkeypatch) -> _FakePlatform:
    """Intercept every httpx entry point the SDK can reach."""
    fake = _FakePlatform()
    monkeypatch.setattr(httpx, "request", fake)
    monkeypatch.setattr(
        httpx.Client, "request",
        lambda self, method, url, **kw: fake(method, url, **kw),
    )
    monkeypatch.setattr(
        httpx.Client, "send",
        lambda self, request, **kw: fake(request.method, request.url),
    )
    return fake


@pytest.fixture
def fresh_sdk(monkeypatch, tmp_path):
    """A clean SDK + LangChain adapter, rooted in an empty temp project.

    ``instrument()`` is idempotent via a module global, and it registers a
    process-global LangChain hook — both are saved and restored so this file
    can't leak state into the rest of the suite.
    """
    import decimalai._config as cfg
    import decimalai.langchain as dlc
    import langchain_core.tracers.context as lc_ctx

    monkeypatch.chdir(tmp_path)
    # Don't let a real ~/.claude/skills checkout leak into the run.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()

    monkeypatch.setattr(cfg, "_config", None, raising=False)
    monkeypatch.setattr(cfg, "_client", None, raising=False)
    monkeypatch.setattr(dlc, "_installed", False)
    # Keep the real adapter's skill sync/pull intact — only neutralize the
    # global callback registration, which has no bearing on this behaviour.
    monkeypatch.setattr(lc_ctx, "register_configure_hook", lambda *a, **k: None)
    return tmp_path


def _files_under(root: Path) -> List[str]:
    return sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and not str(p.relative_to(root)).startswith("home/")
    )


def test_disabled_langchain_makes_no_http_call_and_writes_no_files(
    platform, fresh_sdk
):
    """enabled=False + langchain=True: no adapter, no API call, no disk write."""
    import decimalai
    import decimalai._config as cfg
    import decimalai.langchain as dlc

    decimalai.init(
        api_key="dai_sk_test",
        base_url="http://localhost:9",
        enabled=False,
        langchain=True,
    )

    assert platform.calls == [], (
        "init(enabled=False) attempted HTTP calls: " + repr(platform.calls)
    )
    assert _files_under(fresh_sdk) == [], (
        "init(enabled=False) wrote files into the project: "
        + repr(_files_under(fresh_sdk))
    )
    assert dlc._installed is False, "the LangChain adapter was installed anyway"
    assert cfg._client is None


def test_enabled_langchain_still_syncs_and_writes(platform, fresh_sdk):
    """The control: enabled=True behaviour is unchanged by the gate."""
    import decimalai

    decimalai.init(
        api_key="dai_sk_test",
        base_url="http://localhost:9",
        enabled=True,
        langchain=True,
        # Opt in to disk mirroring: writing is no longer implicit, so without
        # this the control would prove nothing about `enabled`.
        skill_authority="harness",
    )

    assert any("/skills/hashes" in c for c in platform.calls), (
        "enabled=True no longer pulls skills from the platform: "
        + repr(platform.calls)
    )
    written = _files_under(fresh_sdk)
    assert any(p.endswith("SKILL.md") for p in written), (
        "enabled=True no longer writes skills to disk: " + repr(written)
    )
