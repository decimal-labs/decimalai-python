"""Nothing that identifies the developer's machine may reach the API.

Two collection sites used to leak it and both are locked here:

  * ``decimalai._install._default_label()`` defaulted to
    ``socket.gethostname()``, so every ``POST /skills/sync`` and
    ``POST /skills/installs/report`` carried the machine name as
    ``install_label``. A label is now strictly opt-in
    (``DECIMALAI_INSTALL_LABEL``), and when there is none the key is
    omitted from the request body entirely — the anonymous ``install_id``
    is what per-install drift attribution actually needs.
  * ``@eval``'s ``source_location_extra`` shipped ``abs_path``
    (``os.path.abspath`` of the user's source file) and
    ``cwd_at_decoration`` (``os.getcwd()``) to
    ``POST /api/v1/evaluators/register``. Only the non-identifying
    ``lineno`` survives.

Assertions are on the *serialized payload*, not just the accessor, so a
future re-add anywhere along the path (identity dict → sync body → wire)
fails here.
"""

import json
import os
import socket
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from decimalai import _install
from decimalai._client import DecimalAIClient
from decimalai.cli.main import cli
from decimalai.evals import DecimalEval, eval as decimal_eval
from decimalai.skill_router import SkillRouter


# ── install identity: no hostname ──────────────────────────


class TestInstallLabelIsNotMachineIdentity:
    def test_default_label_is_none_without_opt_in(self, monkeypatch):
        # The whole point: no env var → no label at all (it used to be the
        # hostname).
        monkeypatch.delenv("DECIMALAI_INSTALL_LABEL", raising=False)
        assert _install._default_label() is None

    def test_default_label_is_never_the_hostname(self, monkeypatch):
        monkeypatch.delenv("DECIMALAI_INSTALL_LABEL", raising=False)
        host = socket.gethostname()
        assert host  # the value we must not be sending
        assert _install._default_label() != host

    def test_env_var_opts_in_and_is_stripped(self, monkeypatch):
        monkeypatch.setenv("DECIMALAI_INSTALL_LABEL", "  ci-runner-7  ")
        assert _install._default_label() == "ci-runner-7"

    def test_whitespace_only_env_var_is_no_label(self, monkeypatch):
        # An accidentally-blank export must not become an empty-string label.
        monkeypatch.setenv("DECIMALAI_INSTALL_LABEL", "   ")
        assert _install._default_label() is None

    def test_identity_on_disk_carries_no_hostname(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DECIMALAI_INSTALL_LABEL", raising=False)
        ident = _install.get_install_identity(str(tmp_path))

        assert ident["install_id"]  # the anonymous handle still exists
        assert ident["install_label"] is None

        on_disk = (tmp_path / ".decimal" / "install.json").read_text(encoding="utf-8")
        assert socket.gethostname() not in on_disk

    def test_explicit_label_still_wins(self, tmp_path, monkeypatch):
        # Opting in is unaffected — a caller-supplied label is recorded as-is.
        monkeypatch.delenv("DECIMALAI_INSTALL_LABEL", raising=False)
        ident = _install.get_install_identity(str(tmp_path), label="ci-box")
        assert ident["install_label"] == "ci-box"


class TestSyncPayloadOmitsLabel:
    def test_sync_skills_omits_install_label_when_none(self):
        router = SkillRouter(api_key="k", base_url="http://localhost:8000")
        with patch.object(router, "_request", return_value={}) as req:
            router.sync_skills(
                [{"name": "alpha", "body_markdown": "x"}],
                install_id="11111111-2222-3333-4444-555555555555",
                install_label=None,
            )
        body = req.call_args.kwargs["json"]
        assert body["install_id"]
        # Omitted, not null: the backend must not record an empty label row.
        assert "install_label" not in body

    def test_cli_sync_posts_no_label_and_no_hostname(self, monkeypatch):
        monkeypatch.delenv("DECIMALAI_INSTALL_LABEL", raising=False)
        client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {"actions": []}
        resp.raise_for_status = MagicMock()
        resp.status_code = 200
        client._http.post.return_value = resp

        runner = CliRunner()
        with runner.isolated_filesystem():
            os.makedirs("skills/alpha-dir")
            with open("skills/alpha-dir/SKILL.md", "w", encoding="utf-8") as f:
                f.write("---\nname: alpha\ndescription: first\n---\n\n# Alpha\n")
            with patch("decimalai.cli.main._make_client", return_value=client):
                result = runner.invoke(cli, [
                    "skills", "sync",
                    "--api-key", "k", "--base-url", "http://localhost:8000",
                ])

        assert result.exit_code == 0, result.output
        body = client._http.post.call_args.kwargs["json"]
        assert body["install_id"]
        assert "install_label" not in body
        assert socket.gethostname() not in json.dumps(body)

    def test_cli_status_posts_no_label_and_no_hostname(self, monkeypatch):
        monkeypatch.delenv("DECIMALAI_INSTALL_LABEL", raising=False)
        client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {"skills": []}
        resp.raise_for_status = MagicMock()
        client._http.post.return_value = resp

        runner = CliRunner()
        with runner.isolated_filesystem():
            os.makedirs("skills/alpha-dir")
            with open("skills/alpha-dir/SKILL.md", "w", encoding="utf-8") as f:
                f.write("---\nname: alpha\ndescription: first\n---\n\n# Alpha\n")
            with patch("decimalai.cli.main._make_client", return_value=client):
                result = runner.invoke(cli, [
                    "skills", "status",
                    "--api-key", "k", "--base-url", "http://localhost:8000",
                ])

        assert result.exit_code == 0, result.output
        path = client._http.post.call_args[0][0]
        body = client._http.post.call_args.kwargs["json"]
        assert path == "/api/v1/skills/installs/report"
        assert "install_label" not in body
        assert socket.gethostname() not in json.dumps(body)


# ── @eval registration: no abs path, no cwd ────────────────


@decimal_eval
def _sample_eval(trace):
    """Sample evaluator used to inspect the registration payload."""
    return True


class TestEvalRegistrationHasNoFilesystemIdentity:
    def test_source_location_extra_is_lineno_only(self):
        extra = _sample_eval.source_location_extra
        assert extra is not None
        assert set(extra) == {"lineno"}
        assert isinstance(extra["lineno"], int)

    def test_registration_dict_has_no_abs_path_or_cwd(self):
        payload = _sample_eval.to_registration_dict(agent_name="agent")
        blob = json.dumps(payload)

        assert "abs_path" not in blob
        assert "cwd_at_decoration" not in blob
        # The two concrete values that used to ship.
        assert os.path.abspath(__file__) not in blob
        assert os.getcwd() not in blob
        # And nothing else absolute sneaks in via source_location.
        assert not str(payload["source_location"] or "").startswith(os.sep)
        assert payload["source_location_extra"] == {
            "lineno": _sample_eval.source_location_extra["lineno"]
        }

    def test_registered_wire_body_carries_no_machine_paths(self):
        client = DecimalAIClient(api_key="k", base_url="http://localhost:8000")
        try:
            resp = MagicMock()
            resp.json.return_value = {"registered": [], "total": 0}
            with patch.object(
                client, "_request_with_retry", return_value=resp
            ) as req:
                client.register_evals([_sample_eval], agent_name="agent")
            blob = json.dumps(req.call_args.kwargs["json"])
        finally:
            client.close()

        assert os.path.abspath(__file__) not in blob
        assert os.getcwd() not in blob
        assert "cwd_at_decoration" not in blob

    def test_lineno_survives_for_a_locally_defined_eval(self):
        # The useful, non-identifying half must still be captured.
        def scorer(trace):
            return True

        ev = DecimalEval(scorer)
        assert ev.source_location_extra["lineno"] > 0


class TestUpgradePathStripsRecordedHostname:
    """A file written by an older SDK already holds the machine hostname.

    Removing the default is not enough on its own: get_install_identity()
    returns a pre-existing file verbatim, so without this the leak survives
    the upgrade forever on every machine that ever ran the old version.
    """

    def _seed(self, root, label):
        import json as _json
        import os as _os

        d = _os.path.join(root, ".decimal")
        _os.makedirs(d, exist_ok=True)
        with open(_os.path.join(d, "install.json"), "w", encoding="utf-8") as fh:
            _json.dump(
                {
                    "install_id": "11111111-2222-3333-4444-555555555555",
                    "install_label": label,
                    "created_at": "2026-01-01T00:00:00+00:00",
                },
                fh,
            )
        return _os.path.join(d, "install.json")

    def test_recorded_hostname_is_dropped_on_read(self, tmp_path):
        from decimalai import _install

        host = _install._hostname()
        assert host, "test needs a resolvable hostname"
        path = self._seed(str(tmp_path), host)

        ident = _install.get_install_identity(str(tmp_path))

        assert ident["install_label"] is None
        assert ident["install_id"] == "11111111-2222-3333-4444-555555555555"
        # and it is persisted, so the next run cannot resurrect it
        import json as _json

        assert _json.load(open(path))["install_label"] is None

    def test_an_explicit_unrelated_label_is_preserved(self, tmp_path):
        from decimalai import _install

        self._seed(str(tmp_path), "ci-box")
        ident = _install.get_install_identity(str(tmp_path))
        assert ident["install_label"] == "ci-box"

    def test_dry_run_does_not_rewrite_the_file(self, tmp_path):
        import json as _json

        from decimalai import _install

        host = _install._hostname()
        path = self._seed(str(tmp_path), host)

        ident = _install.get_install_identity(str(tmp_path), create=False)

        assert ident["install_label"] is None
        # create=False must leave no footprint, even to fix the label
        assert _json.load(open(path))["install_label"] == host
