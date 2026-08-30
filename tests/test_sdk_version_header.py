"""Every SDK request must identify which SDK version made it.

The incident this file exists to prevent
----------------------------------------
An investigation could not answer "what SDK version produced this trace?" for
any of 285,660 production traces. Only ONE request in the whole SDK identified
itself — the one-off ``init()`` auth-verify probe. Trace ingest went out over an
``httpx.Client`` that set no ``User-Agent``, so httpx supplied its own default
and every ingested trace arrived stamped ``python-httpx/<x.y.z>``: the
TRANSPORT's version, saying nothing about the SDK.

The measured cost: the synthetic-user fleet ran 0.8.0 (published 2026-07-05)
against production for six weeks while everyone believed it was current, and it
was detectable only by grepping Cloud Run logs for that single startup probe::

    gcloud logging read '... httpRequest.userAgent:"decimalai-sdk"' --freshness=14d
    -> 1000/1000 results: decimalai-sdk/0.8.0 (init-verify)

That in turn explained a separate mystery — ``skills_loaded_by_agent`` was null
on all 285,660 traces because load-recording only shipped in 0.10.2.

So the tests below assert three separable things, and it matters that they are
separate:

1. the shared builder produces a version, and that version is READ FROM
   ``decimalai.__version__`` rather than copied into a literal that will rot;
2. every real request path routes through that builder — ingest especially,
   since ingest is the path that was blind;
3. the User-Agent carries nothing that identifies a user or a machine.
"""

from __future__ import annotations

import re
import urllib.request

import pytest

import decimalai
from decimalai._client import DecimalAIClient
from decimalai._config import DecimalConfig, sdk_headers, sdk_user_agent
from decimalai.skill_router import SkillRouter

# `decimalai-sdk/<version>`, the token a log query greps for.
_UA_RE = re.compile(r"^decimalai-sdk/(\S+) \((.*)\)$")


def _version_from(ua: str) -> str:
    m = _UA_RE.match(ua)
    assert m, f"User-Agent is not in `product/version (comment)` form: {ua!r}"
    return m.group(1)


# ── The builder ────────────────────────────────────────────────


class TestSdkUserAgent:
    def test_reports_the_package_version(self):
        assert _version_from(sdk_user_agent()) == decimalai.__version__

    def test_version_is_read_not_hardcoded(self, monkeypatch):
        """The whole point: a literal here would rot silently.

        A hardcoded string passes the test above on the day it is written and
        lies forever after. Repointing ``decimalai.__version__`` and requiring
        the User-Agent to follow proves the value is genuinely derived.
        """
        monkeypatch.setattr(decimalai, "__version__", "99.99.99-test")
        assert _version_from(sdk_user_agent()) == "99.99.99-test"
        assert "99.99.99-test" in sdk_headers("dai_sk_test")["User-Agent"]

    def test_grep_token_is_stable(self):
        """`httpRequest.userAgent:"decimalai-sdk"` is the query that found the
        0.8.0 fleet. Renaming the product token silently breaks it."""
        assert sdk_user_agent().startswith("decimalai-sdk/")

    def test_carries_python_version_and_platform(self):
        import sys

        comment = _UA_RE.match(sdk_user_agent()).group(2)
        py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        assert f"python/{py}" in comment
        assert sys.platform in comment

    def test_sends_nothing_user_identifying(self):
        """Cheap triage data only — never anything that fingerprints a user.

        Hostname, username and filesystem paths answer no triage question and
        would turn a diagnostic header into a privacy problem, so they must not
        leak in via a future "just add a bit more context" change.
        """
        import getpass
        import os
        import socket

        ua = sdk_user_agent("init-verify")
        for secret in (
            socket.gethostname(),
            getpass.getuser(),
            os.getcwd(),
            os.path.expanduser("~"),
        ):
            if secret:
                assert secret not in ua, f"{secret!r} leaked into User-Agent {ua!r}"

    def test_context_is_appended_not_duplicated(self):
        """`init-verify` must compose with the shared string, not restate it."""
        ua = sdk_user_agent("init-verify")
        assert ua.count("decimalai-sdk/") == 1
        assert _version_from(ua) == decimalai.__version__
        assert "init-verify" in ua
        # ...and the plain form must NOT claim to be a startup probe.
        assert "init-verify" not in sdk_user_agent()

    def test_init_verify_substring_is_backwards_compatible(self):
        """Before this change the probe sent `decimalai-sdk/0.8.0 (init-verify)`.

        Log filters already written against the `init-verify` substring must
        keep matching the new, longer comment.
        """
        assert "init-verify" in sdk_user_agent("init-verify")


class TestSdkHeaders:
    def test_shared_builder_carries_all_three(self):
        h = sdk_headers("dai_sk_test")
        assert h["Authorization"] == "Bearer dai_sk_test"
        assert h["Content-Type"] == "application/json"
        assert _version_from(h["User-Agent"]) == decimalai.__version__

    def test_no_second_unread_header(self):
        """Deliberately User-Agent ONLY.

        A custom `X-Decimal-SDK-Version` is invisible to Cloud Run's
        `httpRequest.userAgent` field and would be dead weight until the
        backend is changed to read it — which is exactly what happened to
        `X-Decimal-Project` (sent for versions, silently discarded on arrival).
        Both headers would ship in the same release, so a custom one reaches no
        client the User-Agent doesn't already reach; its only gain is parse
        convenience over `decimalai-sdk/(\\S+)`. If this assertion is ever
        removed, it should be because the BACKEND now reads the new header.
        """
        assert set(sdk_headers("dai_sk_test")) == {
            "Authorization",
            "Content-Type",
            "User-Agent",
        }


# ── Every request path ─────────────────────────────────────────


class TestEveryRequestPathIdentifiesItself:
    def test_config_api_headers(self):
        ua = DecimalConfig(api_key="dai_sk_test").api_headers["User-Agent"]
        assert _version_from(ua) == decimalai.__version__

    def test_ingest_path(self):
        """THE regression test for the incident.

        `DecimalAIClient._http` is the client every trace POST and manifest
        registration goes out on. Before the fix its User-Agent was
        `python-httpx/<x.y.z>`.
        """
        client = DecimalAIClient(api_key="dai_sk_test")
        ua = client._http.headers["user-agent"]
        assert not ua.startswith("python-httpx/"), (
            "trace ingest is advertising the TRANSPORT's version again — this is "
            "the exact blindness that left 285,660 traces unattributable"
        )
        assert _version_from(ua) == decimalai.__version__

    def test_skill_router_path(self):
        ua = SkillRouter(api_key="dai_sk_test")._headers()["User-Agent"]
        assert _version_from(ua) == decimalai.__version__

    def test_init_verify_path(self, monkeypatch):
        """The startup probe still identifies itself AND stays distinguishable.

        Asserts on the request actually handed to urllib, not on a helper, so
        that rewiring `_verify_backend_at_init` to skip the shared builder is
        caught here.
        """
        seen = {}

        class _Resp:
            def read(self):
                return b'{"scope": "workspace", "require_manifest_on_ingest": false}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def _fake_urlopen(req, timeout=None):
            seen["headers"] = dict(req.headers)
            seen["url"] = req.full_url
            return _Resp()

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
        decimalai._verify_backend_at_init(
            base_url="https://api.example.invalid",
            api_key="dai_sk_test",
            timeout=5.0,
        )

        # urllib title-cases header keys on the Request object.
        ua = seen["headers"]["User-agent"]
        assert _version_from(ua) == decimalai.__version__
        assert "init-verify" in ua, "a startup probe must stay distinguishable from ingest"
        assert seen["headers"]["Authorization"] == "Bearer dai_sk_test"
        # Bodyless GET: the probe must not have grown a Content-Type.
        assert "Content-type" not in seen["headers"]


# ── Drift guards ───────────────────────────────────────────────


class TestNoVersionDrift:
    def test_version_string_has_exactly_one_source_in_the_package(self):
        """No module may hardcode a version literal that the header would outrun.

        ``__init__.__version__`` is the single in-package source (pinned to
        pyproject by ``TestVersion.test_version_matches_pyproject``). A second
        copy anywhere else is a value that keeps reporting the old version
        after a bump — which is the shape of this whole incident.

        Deliberately AST-based rather than a grep over lines:

        * a grep flags prose. Comments and docstrings that show an example
          User-Agent are documentation, not a runtime source of truth, and
          failing on them would train people to delete the explanation.
        * a substring grep also false-positives — ``0.10.3`` matches inside
          ``llamaindex.py``'s ``pre-0.10.30``, an unrelated framework version.

        Only real, non-docstring string constants count.
        """
        import ast
        from pathlib import Path

        pkg = Path(decimalai.__file__).parent
        current = decimalai.__version__
        offenders = []

        for py in sorted(pkg.rglob("*.py")):
            tree = ast.parse(py.read_text(), filename=str(py))
            # Collect docstring nodes so they can be skipped.
            docstrings = set()
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    body = getattr(node, "body", None)
                    if (
                        body
                        and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)
                    ):
                        docstrings.add(id(body[0].value))

            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                if id(node) in docstrings:
                    continue
                if current not in node.value:
                    continue
                # The one legitimate site: `__version__ = "<current>"`.
                if py.name == "__init__.py" and node.value == current:
                    continue
                # ...and one illegitimate match, which is ANOTHER package's version.
                #
                # The docstring above already names this failure class — `0.10.3`
                # matching inside `pre-0.10.30` — and going AST-based fixed it only
                # for docstrings. It recurred on the 0.12.0 bump in a real literal:
                # llamaindex.py's install hint says `llama-index-core>=0.12.0`,
                # which has nothing to do with our version and does not go stale
                # when we bump. A version preceded by a comparison operator or a
                # hyphen belongs to whatever package name sits in front of it.
                #
                # Only skip when EVERY occurrence is qualified that way, so a
                # literal that mentions both a dependency pin and a bare copy of
                # our own version still fails.
                starts = []
                at = node.value.find(current)
                while at != -1:
                    starts.append(at)
                    at = node.value.find(current, at + 1)
                if all(
                    node.value[:s].rstrip().endswith((">=", "<=", "==", "~=", "!=", ">", "<", "-"))
                    for s in starts
                ):
                    continue
                offenders.append(f"{py.relative_to(pkg)}:{node.lineno}: {node.value!r}")

        assert not offenders, (
            "these string literals hardcode the current version and will "
            "silently report a stale one after the next bump:\n" + "\n".join(offenders)
        )


@pytest.mark.parametrize("context", [None, "init-verify"])
def test_user_agent_is_a_single_header_safe_line(context):
    """Header values must not contain CR/LF (request-splitting) or stray parens."""
    ua = sdk_user_agent(context)
    assert "\n" not in ua and "\r" not in ua
    assert ua.count("(") == 1 and ua.count(")") == 1
    assert ua.isascii()
