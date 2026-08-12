"""Tests for skill auto-loading during install() / initialization.

Verifies the gap: install() → discover_skills() → skills_registry wired into
processor/handler/exporter → auto-detected during agent runs.

Covers:
  1. Generic tracer: start_trace() auto-discovers SKILL.md files
  2. OpenAI Agents: install() auto-discovers and wires to processor
  3. LangChain: install() auto-discovers and stores in _explicit_manifest_config
  4. OTEL: install() auto-discovers and passes to exporter
  5. End-to-end: auto-discovered skills are used for detection during runs
"""

import os
import sys
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call
from uuid import uuid4

import pytest


# ── Fixtures ──────────────────────────────────────────────

SAMPLE_SKILL_MD = """---
name: code-review
description: Reviews code for security and style.
metadata:
  version: "1.0"
---
# Code Review

## Instructions
1. Check for common security vulnerabilities
2. Review code style and naming conventions
"""

SECOND_SKILL_MD = """---
name: test-generator
description: Generates unit tests.
---
# Test Generator

Write comprehensive unit tests for the provided code.
"""

MOCK_DISCOVERED_SKILLS = [
    {"name": "code-review", "hash": "sha256:abc123", "description": "Reviews code"},
    {"name": "test-generator", "hash": "sha256:def456", "description": "Generates tests"},
]


@pytest.fixture
def skill_dir(tmp_path):
    """Create a temp directory with SKILL.md files."""
    base = tmp_path / ".claude" / "skills"

    cr = base / "code-review"
    cr.mkdir(parents=True)
    (cr / "SKILL.md").write_text(SAMPLE_SKILL_MD)

    tg = base / "test-generator"
    tg.mkdir(parents=True)
    (tg / "SKILL.md").write_text(SECOND_SKILL_MD)

    return str(base)


@pytest.fixture(autouse=True)
def _reset_sdk():
    """Reset global SDK state before each test."""
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    cfg._config = DecimalConfig(
        api_key="dai_sk_test",
        base_url="http://localhost:8000",
        enabled=True,
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {
        "manifest_id": "test-manifest-id",
        "status": "active",
    }

    # Reset module-level state to prevent leaks between tests
    try:
        import decimalai.openai_agents as oai
        oai._manifest_id = None
    except Exception:
        pass

    try:
        import decimalai.langchain as lc
        lc._installed = False
        lc._manifest_id = None
        lc._explicit_manifest_config = None
        lc._install_agent_name = None
    except Exception:
        pass

    try:
        import decimalai.otel as otel_mod
        if hasattr(otel_mod, '_manifest_id'):
            otel_mod._manifest_id = None
    except Exception:
        pass

    yield


# ── Generic Tracer ────────────────────────────────────────


class TestGenericAutoLoading:
    """start_trace() auto-discovers skills from SKILL.md files."""

    def test_start_trace_auto_discovers_skills(self, skill_dir):
        """start_trace(skill_dirs=[...]) calls discover_skills and wires registry."""
        from decimalai.generic import start_trace

        with start_trace(
            agent_name="test-agent",
            skill_dirs=[skill_dir],
            auto_send=False,
        ) as ctx:
            # Skills registry should be populated from SKILL.md files
            assert ctx._skills_registry is not None
            assert len(ctx._skills_registry) == 2
            names = {s["name"] for s in ctx._skills_registry}
            assert names == {"code-review", "test-generator"}

    def test_start_trace_no_skill_dirs_still_tries_discovery(self):
        """start_trace() without skill_dirs calls discover_skills(None)."""
        with patch("decimalai.skills.discover_skills") as mock_discover:
            mock_discover.return_value = [
                {"name": "auto-found", "hash": "sha256:auto123"},
            ]
            from decimalai.generic import start_trace

            with start_trace(
                agent_name="test-agent",
                auto_send=False,
            ) as ctx:
                mock_discover.assert_called_once_with(None)
                assert ctx._skills_registry is not None
                assert len(ctx._skills_registry) == 1
                assert ctx._skills_registry[0]["name"] == "auto-found"

    def test_start_trace_explicit_skills_override_discovery(self, skill_dir):
        """Explicit skills= param skips auto-discovery entirely."""
        explicit_skills = [{"name": "explicit-skill", "hash": "sha256:explicit"}]

        with patch("decimalai.skills.discover_skills") as mock_discover:
            from decimalai.generic import start_trace

            with start_trace(
                agent_name="test-agent",
                skills=explicit_skills,
                auto_send=False,
            ) as ctx:
                # discover_skills should NOT be called when explicit skills are provided
                mock_discover.assert_not_called()
                assert ctx._skills_registry == explicit_skills

    def test_start_trace_discovery_failure_is_silent(self):
        """If discover_skills raises, start_trace still works (skills=None)."""
        with patch("decimalai.skills.discover_skills", side_effect=RuntimeError("boom")):
            from decimalai.generic import start_trace

            with start_trace(
                agent_name="test-agent",
                auto_send=False,
            ) as ctx:
                # Should not crash — just no skills registry
                assert ctx._skills_registry is None

    def test_auto_discovered_skills_used_for_detection(self, skill_dir):
        """Skills from auto-discovery are used for detection during the trace."""
        from decimalai.generic import start_trace

        with start_trace(
            agent_name="test-agent",
            skill_dirs=[skill_dir],
            auto_send=False,
        ) as ctx:
            # Simulate an LLM call that references a discovered skill
            ctx.log_llm_call(
                model="gpt-4o",
                input=[
                    {"role": "system", "content": "## Skill: code-review\nReview this code."},
                    {"role": "user", "content": "Check my PR."},
                ],
                output={"content": "LGTM"},
            )

            # Run auto-detection (normally called in _send)
            ctx._auto_detect_skills()

            # Should have detected the auto-discovered skill
            assert "code-review" in ctx._active_skills
            # Hash should come from the registry (SKILL.md content hash)
            assert ctx._active_skills["code-review"] is not None
            assert ctx._active_skills["code-review"].startswith("sha256:")


# ── OpenAI Agents ─────────────────────────────────────────


class TestOpenAIAgentsAutoLoading:
    """install() auto-discovers skills and wires to DecimalTracingProcessor."""

    def test_install_auto_discovers_skills(self, skill_dir):
        """install(skill_dirs=[...]) discovers skills and stores on processor."""
        mock_add = MagicMock()

        with patch.dict("sys.modules", {
            "agents": MagicMock(),
            "agents.tracing": MagicMock(
                add_trace_processor=mock_add,
                set_trace_processors=MagicMock(),
            ),
        }):
            from decimalai.openai_agents import install
            install(skill_dirs=[skill_dir])

            processor = mock_add.call_args[0][0]
            assert len(processor._skills_registry) == 2
            names = {s["name"] for s in processor._skills_registry}
            assert names == {"code-review", "test-generator"}

    def test_install_no_skills_calls_discover(self):
        """install() without skills= triggers discover_skills()."""
        mock_add = MagicMock()

        with patch.dict("sys.modules", {
            "agents": MagicMock(),
            "agents.tracing": MagicMock(
                add_trace_processor=mock_add,
                set_trace_processors=MagicMock(),
            ),
        }):
            with patch("decimalai.skills.discover_skills") as mock_discover:
                mock_discover.return_value = [
                    {"name": "discovered-skill", "hash": "sha256:disc123"},
                ]
                from decimalai.openai_agents import install
                install()

                mock_discover.assert_called_once()

    def test_install_explicit_skills_skip_discovery(self):
        """install(skills=[...]) skips auto-discovery."""
        mock_add = MagicMock()
        explicit = [{"name": "explicit-only", "hash": "sha256:exp"}]

        with patch.dict("sys.modules", {
            "agents": MagicMock(),
            "agents.tracing": MagicMock(
                add_trace_processor=mock_add,
                set_trace_processors=MagicMock(),
            ),
        }):
            with patch("decimalai.skills.discover_skills") as mock_discover:
                from decimalai.openai_agents import install
                install(skills=explicit)

                mock_discover.assert_not_called()
                processor = mock_add.call_args[0][0]
                assert len(processor._skills_registry) == 1
                assert processor._skills_registry[0]["name"] == "explicit-only"


# ── LangChain ─────────────────────────────────────────────


class TestLangChainAutoLoading:
    """install() auto-discovers skills and stores in _explicit_manifest_config."""

    def test_install_with_skill_dirs_discovers(self, skill_dir):
        """install(skill_dirs=[...]) auto-discovers skills."""
        with patch.dict("sys.modules", {
            "langchain_core": MagicMock(),
            "langchain_core.tracers": MagicMock(),
            "langchain_core.tracers.context": MagicMock(
                register_configure_hook=MagicMock(),
            ),
        }):
            import decimalai.langchain as lc
            lc._installed = False
            lc._explicit_manifest_config = None

            from decimalai.langchain import install
            install(skill_dirs=[skill_dir])

            assert lc._explicit_manifest_config is not None
            skills = lc._explicit_manifest_config.get("skills")
            assert skills is not None
            assert len(skills) == 2
            names = {s["name"] for s in skills}
            assert names == {"code-review", "test-generator"}

    def test_install_with_tools_and_no_skills_auto_discovers(self):
        """install(tools=[...]) without skills= still triggers skill discovery."""
        with patch.dict("sys.modules", {
            "langchain_core": MagicMock(),
            "langchain_core.tracers": MagicMock(),
            "langchain_core.tracers.context": MagicMock(
                register_configure_hook=MagicMock(),
            ),
        }):
            import decimalai.langchain as lc
            lc._installed = False
            lc._explicit_manifest_config = None

            with patch("decimalai.skills.discover_skills") as mock_discover:
                mock_discover.return_value = [
                    {"name": "auto-skill", "hash": "sha256:auto"},
                ]
                from decimalai.langchain import install
                install(tools=[{"name": "search"}])

                mock_discover.assert_called_once()
                assert lc._explicit_manifest_config["skills"] is not None
                assert lc._explicit_manifest_config["skills"][0]["name"] == "auto-skill"

    def test_install_explicit_skills_skip_discovery(self):
        """install(skills=[...]) does not call discover_skills."""
        with patch.dict("sys.modules", {
            "langchain_core": MagicMock(),
            "langchain_core.tracers": MagicMock(),
            "langchain_core.tracers.context": MagicMock(
                register_configure_hook=MagicMock(),
            ),
        }):
            import decimalai.langchain as lc
            lc._installed = False
            lc._explicit_manifest_config = None

            explicit = [{"name": "my-skill", "hash": "sha256:mine"}]

            with patch("decimalai.skills.discover_skills") as mock_discover:
                from decimalai.langchain import install
                install(skills=explicit)

                mock_discover.assert_not_called()
                assert lc._explicit_manifest_config["skills"] == explicit


# ── OTEL ──────────────────────────────────────────────────


class TestOTELAutoLoading:
    """install() auto-discovers skills and passes to DecimalSpanExporter."""

    def test_install_auto_discovers_skills(self, skill_dir):
        """install(skill_dirs=[...]) discovers skills and passes to exporter."""
        mock_provider_cls = MagicMock()
        mock_batch_cls = MagicMock()
        mock_trace_api = MagicMock()

        with patch.dict("sys.modules", {
            "opentelemetry": MagicMock(trace=mock_trace_api),
            "opentelemetry.trace": mock_trace_api,
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.resources": MagicMock(
                Resource=MagicMock(), SERVICE_NAME="service.name"
            ),
            "opentelemetry.sdk.trace": MagicMock(
                TracerProvider=mock_provider_cls
            ),
            "opentelemetry.sdk.trace.export": MagicMock(
                BatchSpanProcessor=mock_batch_cls
            ),
        }):
            from decimalai.otel import install
            install(skill_dirs=[skill_dir])

            # The BatchSpanProcessor was called with a DecimalSpanExporter
            exporter = mock_batch_cls.call_args[0][0]
            assert exporter._skills is not None
            assert len(exporter._skills) == 2
            names = {s["name"] for s in exporter._skills}
            assert names == {"code-review", "test-generator"}

    def test_install_no_skills_calls_discover(self):
        """install() without skills= triggers discover_skills()."""
        mock_provider_cls = MagicMock()
        mock_batch_cls = MagicMock()
        mock_trace_api = MagicMock()

        with patch.dict("sys.modules", {
            "opentelemetry": MagicMock(trace=mock_trace_api),
            "opentelemetry.trace": mock_trace_api,
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.resources": MagicMock(
                Resource=MagicMock(), SERVICE_NAME="service.name"
            ),
            "opentelemetry.sdk.trace": MagicMock(
                TracerProvider=mock_provider_cls
            ),
            "opentelemetry.sdk.trace.export": MagicMock(
                BatchSpanProcessor=mock_batch_cls
            ),
        }):
            with patch("decimalai.skills.discover_skills") as mock_discover:
                mock_discover.return_value = [
                    {"name": "otel-skill", "hash": "sha256:otel123"},
                ]
                from decimalai.otel import install
                install()

                mock_discover.assert_called_once()

    def test_install_explicit_skills_skip_discovery(self):
        """install(skills=[...]) skips auto-discovery."""
        mock_provider_cls = MagicMock()
        mock_batch_cls = MagicMock()
        mock_trace_api = MagicMock()
        explicit = [{"name": "explicit-otel", "hash": "sha256:expotel"}]

        with patch.dict("sys.modules", {
            "opentelemetry": MagicMock(trace=mock_trace_api),
            "opentelemetry.trace": mock_trace_api,
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.resources": MagicMock(
                Resource=MagicMock(), SERVICE_NAME="service.name"
            ),
            "opentelemetry.sdk.trace": MagicMock(
                TracerProvider=mock_provider_cls
            ),
            "opentelemetry.sdk.trace.export": MagicMock(
                BatchSpanProcessor=mock_batch_cls
            ),
        }):
            with patch("decimalai.skills.discover_skills") as mock_discover:
                from decimalai.otel import install
                install(skills=explicit)

                mock_discover.assert_not_called()
                exporter = mock_batch_cls.call_args[0][0]
                assert exporter._skills == explicit


# ── End-to-End: Auto-discovered → Detection → Trace ──────


class TestAutoLoadE2E:
    """Full pipeline: SKILL.md files → discover → detect → RunTrace.active_skills."""

    def test_generic_e2e_auto_load_to_trace(self, skill_dir):
        """Auto-discovered skills are detected in LLM calls and appear on the trace."""
        from decimalai.generic import start_trace

        with start_trace(
            agent_name="e2e-agent",
            skill_dirs=[skill_dir],
            auto_send=False,
        ) as ctx:
            # Simulate LLM call referencing the auto-discovered skill
            ctx.log_llm_call(
                model="gpt-4o",
                input=[
                    {"role": "system", "content": "## Skill: code-review\nPerform a security audit."},
                    {"role": "user", "content": "Review this code."},
                ],
                output={"content": "Found 2 issues."},
            )

            # Trigger auto-detection
            ctx._auto_detect_skills()

        with patch("decimalai._config._get_config") as mock_get_config:
            mock_cfg = MagicMock()
            mock_cfg.project = "test"
            mock_get_config.return_value = mock_cfg

            trace = ctx.build_trace()

        assert len(trace.active_skills) >= 1
        skill_names = {s["name"] for s in trace.active_skills}
        assert "code-review" in skill_names

        # Verify hash came from the SKILL.md file (not None)
        cr_skill = next(s for s in trace.active_skills if s["name"] == "code-review")
        assert "hash" in cr_skill
        assert cr_skill["hash"].startswith("sha256:")

    def test_generic_e2e_serialization_preserves_auto_skills(self, skill_dir):
        """Auto-discovered skills survive model_dump() → JSON serialization."""
        from decimalai.generic import start_trace
        from decimalai.schema.trace import RunTrace

        with start_trace(
            agent_name="serial-agent",
            skill_dirs=[skill_dir],
            auto_send=False,
        ) as ctx:
            ctx.log_llm_call(
                model="gpt-4o",
                input=[
                    {"role": "system", "content": "## Skill: test-generator\nWrite tests."},
                    {"role": "user", "content": "Add tests for utils.py."},
                ],
                output={"content": "Here are 5 tests."},
            )
            ctx._auto_detect_skills()

        with patch("decimalai._config._get_config") as mock_get_config:
            mock_cfg = MagicMock()
            mock_cfg.project = "test"
            mock_get_config.return_value = mock_cfg

            trace = ctx.build_trace()

        # Serialize → deserialize roundtrip
        data = trace.model_dump(mode="json")
        assert "active_skills" in data
        assert any(s["name"] == "test-generator" for s in data["active_skills"])

        restored = RunTrace.model_validate(data)
        assert any(s["name"] == "test-generator" for s in restored.active_skills)

    def test_oai_processor_uses_auto_discovered_skills_for_detection(self, skill_dir):
        """DecimalTracingProcessor with auto-discovered skills detects them in LLM calls."""
        from decimalai.skills import discover_skills
        from decimalai.openai_agents import DecimalTracingProcessor, _TraceAccumulator
        from decimalai.schema.trace import LlmCallRecord
        from decimalai.schema.common import Status

        # Simulate what install(skill_dirs=[...]) does
        registry = discover_skills([skill_dir], include_global=False)
        processor = DecimalTracingProcessor(
            agent_name="test-agent",
            skills_registry=registry,
        )

        # Create a trace accumulator with an LLM call referencing a skill
        acc = _TraceAccumulator(trace_id="test-123", trace_name="test")
        acc.llm_calls.append(LlmCallRecord(
            model_name="gpt-4o",
            rendered_input=[
                {"role": "system", "content": "## Skill: code-review\nReview code."},
                {"role": "user", "content": "Check this."},
            ],
            status=Status.SUCCESS,
            started_at=datetime.now(timezone.utc),
        ))

        # Run detection
        processor._detect_skills(acc)

        assert "code-review" in acc.active_skills
        assert acc.active_skills["code-review"] is not None
        assert acc.active_skills["code-review"].startswith("sha256:")


class TestPullMissingTargetsDiskRuntimes:
    """install(agent_name=<trace name>) must not hand a non-runtime name
    to pull_missing — 'acme-support-agent' is a trace label, not a disk
    runtime, and pre-fix every such install ERROR-logged 'Unknown agent'.
    Only real AGENT_PATHS keys pass through; everything else falls back to
    'universal'."""

    def _synchronous_sender(self):
        sender = MagicMock()
        sender.submit.side_effect = lambda fn, *a, **kw: fn(*a, **kw)
        return sender

    def _mock_router(self):
        router = MagicMock()
        router.pull_missing.return_value = {"pulled": 0, "updated": 0, "skipped": 0}
        return router

    def _run_openai_install(self, agent_name):
        router = self._mock_router()
        with patch.dict("sys.modules", {
            "agents": MagicMock(),
            "agents.tracing": MagicMock(
                add_trace_processor=MagicMock(),
                set_trace_processors=MagicMock(),
            ),
        }):
            with patch("decimalai.skills.discover_skills", return_value=[]), \
                    patch("decimalai.skill_router.SkillRouter", MagicMock(return_value=router)), \
                    patch("decimalai._config._sender", self._synchronous_sender()):
                from decimalai.openai_agents import install
                install(agent_name=agent_name, disk_sync=True)
        return router

    def _run_langchain_install(self, agent_name):
        router = self._mock_router()
        with patch.dict("sys.modules", {
            "langchain_core": MagicMock(),
            "langchain_core.tracers": MagicMock(),
            "langchain_core.tracers.context": MagicMock(
                register_configure_hook=MagicMock(),
            ),
        }):
            import decimalai.langchain as lc
            lc._installed = False
            lc._explicit_manifest_config = None

            with patch("decimalai.skills.discover_skills", return_value=[]), \
                    patch("decimalai.skill_router.SkillRouter", MagicMock(return_value=router)), \
                    patch("decimalai._config._sender", self._synchronous_sender()):
                from decimalai.langchain import install
                install(agent_name=agent_name, disk_sync=True)
        return router

    def test_openai_agents_trace_name_falls_back_to_universal(self):
        router = self._run_openai_install("acme-support-agent")
        router.pull_missing.assert_called_once()
        assert router.pull_missing.call_args.kwargs["agents"] == ["universal"]

    def test_openai_agents_runtime_key_passes_through(self):
        router = self._run_openai_install("claude-code")
        router.pull_missing.assert_called_once()
        assert router.pull_missing.call_args.kwargs["agents"] == ["claude-code"]

    def test_langchain_trace_name_falls_back_to_universal(self):
        router = self._run_langchain_install("acme-support-agent")
        router.pull_missing.assert_called_once()
        assert router.pull_missing.call_args.kwargs["agents"] == ["universal"]

    def test_langchain_runtime_key_passes_through(self):
        router = self._run_langchain_install("claude-code")
        router.pull_missing.assert_called_once()
        assert router.pull_missing.call_args.kwargs["agents"] == ["claude-code"]

    def test_openai_agents_none_agent_name_uses_universal(self):
        router = self._run_openai_install(None)
        router.pull_missing.assert_called_once()
        assert router.pull_missing.call_args.kwargs["agents"] == ["universal"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
