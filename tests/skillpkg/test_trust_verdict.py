"""One trust verdict, shared: ``client.trust_verdict(resolved)``.

``add``/``install`` (the gate in ``ops``) and ``migrate`` decide the same refusals for a
resolved skill. They used to carry two copies of the rule, kept equal by a parity test;
now both call the one pure function, which reads only the resolve answer. The parity
test in ``test_migrate_v1.py`` grades that function against the gate's behaviour.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from decimalai.skillpkg import client, migration, ops
from decimalai.skillpkg import errors as E
from decimalai.skillpkg.client import RenderKey, Resolved, Safety


def _resolved(band: str = "clean", withheld: bool = False,
              relationship: str = "public") -> Resolved:
    return Resolved(
        ref="acme/x", skill_id="sid", registry_name="acme/x", install_name="acme-x",
        version_id="vid", version=3, render=RenderKey(1, "0" * 64, "foreign"),
        body_sha256="0" * 64, description_sha256="0" * 64, files=(),
        relationship=relationship, safety=Safety(band, withheld))


@pytest.mark.parametrize("band,withheld,relationship,verdict", [
    ("clean", False, "public", (None, None)),
    ("caution", False, "public", (None, "caution")),
    ("unreviewed", False, None, (None, "unreviewed")),
    ("blocked", False, "owned", ("blocked", None)),
    ("clean", True, "public", ("withheld", None)),
    ("clean", True, "used", (None, "withheld pending review")),
    ("caution", True, "owned", (None, "withheld pending review")),
])
def test_the_verdict(band, withheld, relationship, verdict):
    assert client.trust_verdict(_resolved(band, withheld, relationship)) == verdict


def test_it_is_exported_and_the_migration_has_no_copy_of_its_own():
    assert "trust_verdict" in client.__all__
    assert migration.trust_verdict is client.trust_verdict
    assert not hasattr(migration, "_trust")


def test_the_gate_asks_the_shared_verdict(monkeypatch):
    """The gate refuses whatever the verdict refuses, even for a clean skill."""
    run = SimpleNamespace(interactive=False, confirm=None)
    monkeypatch.setattr(client, "trust_verdict", lambda resolved: ("blocked", None))
    with pytest.raises(E.SkillBlockedError):
        ops._gate(run, _resolved("clean"), yes=True)
    monkeypatch.setattr(client, "trust_verdict", lambda resolved: ("withheld", None))
    with pytest.raises(E.SkillWithheldError):
        ops._gate(run, _resolved("clean"), yes=True)
    monkeypatch.setattr(client, "trust_verdict", lambda resolved: (None, "caution"))
    with pytest.raises(E.ConfirmationRequiredError):
        ops._gate(run, _resolved("clean"), yes=False)
    monkeypatch.setattr(client, "trust_verdict", lambda resolved: (None, None))
    ops._gate(run, _resolved("unreviewed"), yes=False)   # nothing to ask about


def test_owned_skills_are_never_asked_about(monkeypatch):
    run = SimpleNamespace(interactive=False, confirm=None)
    ops._gate(run, _resolved("caution", relationship="owned"), yes=False)
    ops._gate(run, _resolved("clean", withheld=True, relationship="owned"), yes=False)
