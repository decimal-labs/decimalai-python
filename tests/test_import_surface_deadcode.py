"""Lock in: the wheel's import surface carries no empty/never-read leftovers.

Two dead surfaces shipped to users and are now removed:

1. ``decimalai.integrations.langgraph`` and ``decimalai.integrations.openai``
   were ZERO-BYTE modules. They were importable from the published wheel, so
   ``from decimalai.integrations import langgraph`` succeeded and then did
   nothing — a working import that promises an adapter it does not have.
   LangGraph is covered by the LangChain callback handler and OpenAI by
   ``decimalai.openai_agents``, so nothing was lost by deleting them.

2. ``replay.run(pairwise_scoring=...)`` was accepted and documented as
   changing how replays are scored, but the parameter was never read — the
   backend's scoring was identical either way. A caller who passed it got
   silence, not the behaviour the docstring described.

No backend and no framework deps — these are surface assertions only.
"""

import importlib.util
import inspect
from pathlib import Path

import decimalai.integrations as integrations
from decimalai import replay


def _shipped_module_names():
    """Module basenames actually present in the integrations package."""
    pkg_dir = Path(integrations.__file__).parent
    return {p.stem for p in pkg_dir.glob("*.py") if p.stem != "__init__"}


class TestNoEmptyIntegrationModules:
    def test_langgraph_and_openai_modules_are_gone(self):
        for name in ("langgraph", "openai"):
            dotted = f"decimalai.integrations.{name}"
            assert importlib.util.find_spec(dotted) is None, (
                f"{dotted} is importable again. If a real adapter was added, "
                f"give it a docstring and a test; if it is another empty "
                f"placeholder, delete it — an importable no-op module is a "
                f"promise the wheel cannot keep."
            )

    def test_no_shipped_integration_module_is_empty(self):
        pkg_dir = Path(integrations.__file__).parent
        empty = [
            p.name
            for p in pkg_dir.glob("*.py")
            if not p.read_text().strip()
        ]
        assert empty == [], f"Zero-byte modules in the import surface: {empty}"


class TestIntegrationsDocstringMatchesDisk:
    """The package docstring names the modules it ships — keep it honest.

    It previously claimed the package "keeps the ``_lc_compat`` helper
    module" while shipping five others, two of them empty.
    """

    def test_every_shipped_module_is_named_in_the_docstring(self):
        doc = integrations.__doc__ or ""
        missing = sorted(n for n in _shipped_module_names() if n not in doc)
        assert missing == [], (
            f"integrations/__init__.py docstring does not mention {missing}. "
            f"Update the docstring when adding or removing a module."
        )

    def test_docstring_claims_no_module_that_is_absent(self):
        doc = integrations.__doc__ or ""
        shipped = _shipped_module_names()
        for gone in ("langgraph", "openai_adapter"):
            assert gone not in doc or gone in shipped, (
                f"docstring advertises `{gone}`, which is not on disk."
            )


class TestReplayRunHasNoUnreadParameter:
    def test_pairwise_scoring_is_not_a_parameter(self):
        params = inspect.signature(replay.run).parameters
        assert "pairwise_scoring" not in params, (
            "replay.run() accepts `pairwise_scoring` again. It was never "
            "read — reintroduce it only together with the code that acts on it."
        )

    def test_pairwise_scoring_is_not_documented(self):
        assert "pairwise_scoring" not in (replay.run.__doc__ or "")

    def test_passing_pairwise_scoring_is_a_loud_error(self):
        """Better a TypeError than silently ignoring a scoring request."""
        try:
            replay.run(
                agent_fn=lambda s: s,
                agent_name="nope",
                pairwise_scoring=True,
            )
        except TypeError as exc:
            assert "pairwise_scoring" in str(exc)
        else:  # pragma: no cover - only reached if the param comes back
            raise AssertionError("replay.run() silently accepted pairwise_scoring")
