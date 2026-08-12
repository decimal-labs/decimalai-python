"""Lock in: minimal trace always gets a manifest registered.

`_maybe_register_manifest()` used to early-return if no tools/models/
skills/subagents had been observed — leaving the simplest possible trace
without a manifest_id, which the backend rejects with a 400.

This test does NOT run the backend — it just confirms the SDK's
behavior: `_maybe_register_manifest()` produces a manifest even from
an "empty" trace (only agent_name set).
"""

from unittest.mock import MagicMock, patch

import pytest


def test_minimal_trace_registers_manifest():
    """A trace with only agent_name (no tools/models/skills/subagents)
    must STILL trigger a manifest registration.

    This is the simplest SDK path: `init` + `with start_trace()`.
    It used to early-return; it must always register.
    """
    from decimalai.generic import TraceContext

    ctx = TraceContext(
        agent_name="hello-world",
        session_id=None,
        auto_send=False,
    )
    # No tools, no models, no skills, no subagents — minimal Hello World.
    assert not ctx._seen_tools
    assert not ctx._seen_models
    assert not ctx._skills_registry
    assert not ctx.subagents

    # Patch the config so we don't hit a real backend; verify
    # register_manifest gets called regardless.
    with patch("decimalai._config._is_enabled", return_value=True), \
         patch("decimalai._config._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.register_manifest.return_value = {"manifest_id": "test-mid-123"}
        mock_get_client.return_value = mock_client

        ctx._maybe_register_manifest()

        # The fix: register_manifest WAS called even with no components.
        # Pre-fix this assertion would fail because the early-return
        # at the (now-removed) `if not tools and not models...: return`
        # bailed out before register_manifest could fire.
        assert mock_client.register_manifest.called, (
            "register_manifest must be called even for an empty manifest "
            "— the Hello-World SDK trace would otherwise have no manifest_id "
            "and the backend would 400-reject it."
        )


def test_empty_manifest_no_components_no_early_return_in_source():
    """Locked-in static check: the early-return that suppressed
    Hello-World manifests must not creep back in.

    Greps `generic.py` for the exact early-return pattern that was
    deleted. If a developer re-adds it, this fails.
    """
    from pathlib import Path
    src = (
        Path(__file__).resolve().parent.parent / "decimalai" / "generic.py"
    ).read_text()
    bad_pattern = "if not tools and not models and not self._skills_registry and not subagents:\n            return"
    assert bad_pattern not in src, (
        "The early-return guard in _maybe_register_manifest was removed "
        "deliberately. It must not be re-added — traces with no observed "
        "components would fail to register a manifest and the backend "
        "would 400-reject them."
    )
