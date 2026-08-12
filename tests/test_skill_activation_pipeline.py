"""Tests for skill activation pipeline — verifies skills flow through all frameworks.

Tests the end-to-end pipeline:
  SDK collects skills → RunTrace.active_skills populated → serialized correctly

Covers: Generic tracer, OpenAI Agents, LangChain, OTEL.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

from decimalai.schema.trace import RunTrace, TraceSpan, LlmCallRecord
from decimalai.schema.common import SpanType, Status
from decimalai.skills import detect_skill_activations, _fuzzy_body_match


# ── Fixtures ──────────────────────────────────────────────


SKILL_REGISTRY = [
    {"name": "code-review", "hash": "sha256:abc123", "description": "Code review skill"},
    {"name": "test-generator", "hash": "sha256:def456", "description": "Test generation"},
    {
        "name": "security-audit",
        "hash": "sha256:ghi789",
        "description": "Security auditing",
        "body": (
            "You are a security auditor.\n"
            "Check for SQL injection vulnerabilities.\n"
            "Check for XSS vulnerabilities.\n"
            "Check for CSRF vulnerabilities.\n"
            "Verify proper input validation.\n"
            "Report all findings with severity levels."
        ),
    },
]

SYSTEM_PROMPT_WITH_SKILL = [
    {"role": "system", "content": "## Skill: code-review\nYou are a code reviewer."},
    {"role": "user", "content": "Review this PR."},
]

SYSTEM_PROMPT_WITH_TWO_SKILLS = [
    {"role": "system", "content": "## Skill: code-review\n## Skill: test-generator\nHelp me."},
    {"role": "user", "content": "Review and test."},
]

SYSTEM_PROMPT_WITHOUT_SKILL = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello."},
]


# ── detect_skill_activations tests ───────────────────────


class TestDetectSkillActivations:
    """Tests for the detect_skill_activations function."""

    def test_name_pattern_match(self):
        """Skills referenced by name in system prompt are detected."""
        activated = detect_skill_activations(SYSTEM_PROMPT_WITH_SKILL, SKILL_REGISTRY)
        assert "code-review" in activated
        assert "test-generator" not in activated

    def test_multiple_skills_detected(self):
        """Multiple skills in same prompt are all detected."""
        activated = detect_skill_activations(SYSTEM_PROMPT_WITH_TWO_SKILLS, SKILL_REGISTRY)
        assert "code-review" in activated
        assert "test-generator" in activated

    def test_no_match_returns_empty(self):
        """Prompt without skill references returns empty list."""
        activated = detect_skill_activations(SYSTEM_PROMPT_WITHOUT_SKILL, SKILL_REGISTRY)
        assert activated == []

    def test_empty_registry_returns_empty(self):
        """Empty registry always returns empty."""
        activated = detect_skill_activations(SYSTEM_PROMPT_WITH_SKILL, [])
        assert activated == []

    def test_empty_input_returns_empty(self):
        """Empty input always returns empty."""
        activated = detect_skill_activations(None, SKILL_REGISTRY)
        assert activated == []

    def test_fuzzy_match_disabled(self):
        """When fuzzy_match=False, only name patterns are used."""
        prompt = [{"role": "system", "content": "Check for SQL injection vulnerabilities.\nCheck for XSS vulnerabilities.\nCheck for CSRF vulnerabilities.\nVerify proper input validation.\nReport all findings with severity levels."}]
        activated = detect_skill_activations(prompt, SKILL_REGISTRY, fuzzy_match=False)
        assert "security-audit" not in activated

    def test_fuzzy_match_enabled(self):
        """With fuzzy_match=True, body content matching catches injected skill bodies."""
        # Prompt contains most of the security-audit skill body but no header
        prompt = [{"role": "system", "content": "You are a security auditor.\nCheck for SQL injection vulnerabilities.\nCheck for XSS vulnerabilities.\nCheck for CSRF vulnerabilities.\nVerify proper input validation.\nReport all findings with severity levels."}]
        activated = detect_skill_activations(prompt, SKILL_REGISTRY, fuzzy_match=True)
        assert "security-audit" in activated

    def test_fuzzy_threshold_configurable(self):
        """Higher fuzzy thresholds require more body lines to match."""
        # Only 2 of 5 significant lines match — below 0.6 threshold
        prompt = [{"role": "system", "content": "Check for SQL injection vulnerabilities.\nCheck for XSS vulnerabilities.\nDo other stuff."}]
        activated = detect_skill_activations(
            prompt, SKILL_REGISTRY, fuzzy_match=True, fuzzy_threshold=0.6
        )
        assert "security-audit" not in activated

        # But passes with a lower threshold
        activated_low = detect_skill_activations(
            prompt, SKILL_REGISTRY, fuzzy_match=True, fuzzy_threshold=0.3
        )
        assert "security-audit" in activated_low


class TestFuzzyBodyMatch:
    """Tests for the _fuzzy_body_match helper."""

    def test_exact_match(self):
        """100% body overlap passes any threshold."""
        body = "Line one long enough.\nLine two long enough.\nLine three long enough."
        assert _fuzzy_body_match(body, body, 1.0) is True

    def test_partial_match_above_threshold(self):
        """Partial overlap above threshold passes."""
        body = "First significant line here.\nSecond significant line here.\nThird significant line here."
        prompt = "First significant line here.\nSecond significant line here.\nSomething else entirely."
        assert _fuzzy_body_match(body, prompt, 0.6) is True

    def test_partial_match_below_threshold(self):
        """Partial overlap below threshold fails."""
        body = "First significant line here.\nSecond significant line here.\nThird significant line here."
        prompt = "First significant line here.\nSomething else entirely.\nAnother random line."
        assert _fuzzy_body_match(body, prompt, 0.6) is False

    def test_empty_body(self):
        """Empty body always returns False."""
        assert _fuzzy_body_match("", "some prompt", 0.5) is False

    def test_empty_prompt(self):
        """Empty prompt always returns False."""
        assert _fuzzy_body_match("some body content", "", 0.5) is False

    def test_short_lines_skipped(self):
        """Lines <= 10 chars are not considered."""
        body = "# Header\n- bullet\nThis is a significant line in the body."
        prompt = "This is a significant line in the body."
        # Only 1 significant line, and it matches → 100%
        assert _fuzzy_body_match(body, prompt, 0.5) is True


# ── Generic Tracer tests ─────────────────────────────────


class TestGenericTracerSkills:
    """Tests for skill activation in the generic tracer."""

    def test_explicit_log_skill_activation(self):
        """log_skill_activation() populates build_trace().active_skills."""
        from decimalai.generic import TraceContext

        ctx = TraceContext(agent_name="test-agent")
        ctx.log_skill_activation(name="code-review", hash="sha256:abc")
        ctx.log_skill_activation(name="test-gen")

        # Mock the config module's _get_config
        with patch("decimalai._config._get_config") as mock_get_config:
            mock_cfg = MagicMock()
            mock_cfg.project = "test"
            mock_get_config.return_value = mock_cfg

            trace = ctx.build_trace()

        assert len(trace.active_skills) == 2
        names = {s["name"] for s in trace.active_skills}
        assert names == {"code-review", "test-gen"}

        # Verify hash is included when provided
        cr = next(s for s in trace.active_skills if s["name"] == "code-review")
        assert cr["hash"] == "sha256:abc"

        # Verify hash is absent when not provided
        tg = next(s for s in trace.active_skills if s["name"] == "test-gen")
        assert "hash" not in tg

    def test_auto_detect_from_prompt(self):
        """Skills are auto-detected from LLM calls when skills_registry is set."""
        from decimalai.generic import TraceContext

        ctx = TraceContext(agent_name="test-agent")
        ctx._skills_registry = SKILL_REGISTRY

        # Simulate an LLM call with a system prompt referencing code-review
        ctx.log_llm_call(
            model="gpt-4o",
            input=SYSTEM_PROMPT_WITH_SKILL,
            output={"content": "LGTM"},
        )

        # Run auto-detection
        ctx._auto_detect_skills()

        assert "code-review" in ctx._active_skills
        assert ctx._active_skills["code-review"] == "sha256:abc123"

    def test_explicit_plus_auto_merge(self):
        """Explicit skills are preserved; auto-detected skills are merged without overwriting."""
        from decimalai.generic import TraceContext

        ctx = TraceContext(agent_name="test-agent")
        ctx._skills_registry = SKILL_REGISTRY

        # Explicitly log with a custom hash
        ctx.log_skill_activation(name="code-review", hash="sha256:custom")

        # LLM call also references code-review
        ctx.log_llm_call(
            model="gpt-4o",
            input=SYSTEM_PROMPT_WITH_SKILL,
            output={"content": "review"},
        )
        ctx._auto_detect_skills()

        # Explicit hash should be preserved (not overwritten by registry hash)
        assert ctx._active_skills["code-review"] == "sha256:custom"

    def test_no_skills_no_noise(self):
        """Without skills installed, active_skills is empty."""
        from decimalai.generic import TraceContext

        ctx = TraceContext(agent_name="test-agent")

        with patch("decimalai._config._get_config") as mock_get_config:
            mock_cfg = MagicMock()
            mock_cfg.project = "test"
            mock_get_config.return_value = mock_cfg

            trace = ctx.build_trace()

        assert trace.active_skills == []

    def test_serialization_roundtrip(self):
        """active_skills survives model_dump() → JSON → model_validate()."""
        from decimalai.generic import TraceContext

        ctx = TraceContext(agent_name="test-agent")
        ctx.log_skill_activation(name="code-review", hash="sha256:abc")

        with patch("decimalai._config._get_config") as mock_get_config:
            mock_cfg = MagicMock()
            mock_cfg.project = "test"
            mock_get_config.return_value = mock_cfg

            trace = ctx.build_trace()

        # Serialize
        data = trace.model_dump(mode="json")
        assert "active_skills" in data
        assert len(data["active_skills"]) == 1
        assert data["active_skills"][0]["name"] == "code-review"

        # Deserialize
        restored = RunTrace.model_validate(data)
        assert len(restored.active_skills) == 1
        assert restored.active_skills[0]["name"] == "code-review"


# ── activation ladder: skills_delivered rung ─────────────


class TestSkillsDeliveredLadder:
    """'delivered' = the full skill body reached the model — a rung between
    offered (menu row) and activated. It implies offered, NEVER activation."""

    def _build(self, ctx):
        with patch("decimalai._config._get_config") as mock_get_config:
            mock_cfg = MagicMock()
            mock_cfg.project = "test"
            mock_get_config.return_value = mock_cfg
            return ctx.build_trace()

    def test_log_skill_delivered_implies_offered_not_activation(self):
        from decimalai.generic import TraceContext

        ctx = TraceContext(agent_name="test-agent")
        ctx.log_skill_delivered(names=["code-review"])
        trace = self._build(ctx)

        assert trace.skills_delivered == ["code-review"]
        assert trace.skills_offered_in_prompt == ["code-review"]
        assert trace.skills_loaded_by_agent == []
        assert trace.active_skills == []  # no fake activation

    def test_log_skill_loaded_counts_as_delivered(self):
        """A load_skill body serve counts as delivered, and is additionally
        marked activated server-side."""
        from decimalai.generic import TraceContext

        ctx = TraceContext(agent_name="test-agent")
        ctx.log_skill_loaded(name="pdf-extract")
        trace = self._build(ctx)

        assert trace.skills_loaded_by_agent == ["pdf-extract"]
        assert trace.skills_delivered == ["pdf-extract"]
        assert trace.skills_offered_in_prompt == ["pdf-extract"]

    def test_delivered_serialization_roundtrip(self):
        from decimalai.generic import TraceContext

        ctx = TraceContext(agent_name="test-agent")
        ctx.log_skill_delivered(names=["b-skill", "a-skill"])
        trace = self._build(ctx)

        data = trace.model_dump(mode="json")
        assert data["skills_delivered"] == ["a-skill", "b-skill"]  # sorted
        restored = RunTrace.model_validate(data)
        assert restored.skills_delivered == ["a-skill", "b-skill"]

    def test_module_level_log_skill_delivered_requires_trace(self):
        import decimalai
        from decimalai._config import DecimalConfigError

        with pytest.raises(DecimalConfigError):
            decimalai.log_skill_delivered(names=["orphan"])

    def test_openai_agents_accumulator_and_drain(self):
        """The oai accumulator carries skills_delivered; the contextvar
        drain path mirrors the offered rail."""
        import decimalai.openai_agents as oa
        from decimalai.openai_agents import _TraceAccumulator

        acc = _TraceAccumulator(trace_id="t-1", trace_name="t")
        assert acc.skills_delivered == set()

        oa._skills_delivered_ctx.set(None)
        oa._add_skills_delivered(["s1", "s2"])
        assert oa._consume_skills_delivered() == ["s1", "s2"]
        assert oa._consume_skills_delivered() == []  # drained

    def test_langchain_build_trace_includes_delivered(self):
        from decimalai.langchain import CallbackHandler
        import decimalai.langchain as lc_mod

        handler = CallbackHandler(agent_name="test")
        handler.log_skill_delivered(names=["code-review"])
        handler._trace_started_at = datetime.now(timezone.utc)

        lc_mod._skills_delivered_ctx.set(None)
        with patch("decimalai._config._config") as mock_cfg:
            mock_cfg.project = "test"
            old_mid = getattr(lc_mod, "_manifest_id", None)
            lc_mod._manifest_id = None
            try:
                trace = handler.build_trace()
            finally:
                lc_mod._manifest_id = old_mid

        assert trace.skills_delivered == ["code-review"]
        assert trace.skills_offered_in_prompt == ["code-review"]  # implied
        assert trace.active_skills == []

    def test_langchain_build_trace_drains_delivered_ctx(self):
        from decimalai.langchain import CallbackHandler
        import decimalai.langchain as lc_mod

        handler = CallbackHandler(agent_name="test")
        handler._trace_started_at = datetime.now(timezone.utc)

        lc_mod._skills_delivered_ctx.set(None)
        lc_mod._add_skills_delivered(["injected-skill"])
        with patch("decimalai._config._config") as mock_cfg:
            mock_cfg.project = "test"
            old_mid = getattr(lc_mod, "_manifest_id", None)
            lc_mod._manifest_id = None
            try:
                trace = handler.build_trace()
            finally:
                lc_mod._manifest_id = old_mid

        assert trace.skills_delivered == ["injected-skill"]
        assert lc_mod._consume_skills_delivered() == []  # drained by build


# ── OpenAI Agents Integration tests ──────────────────────


class TestOpenAIAgentsSkills:
    """Tests for skill activation in the OpenAI Agents integration."""

    def test_accumulator_tracks_skills(self):
        """_TraceAccumulator has active_skills dict initialized."""
        from decimalai.openai_agents import _TraceAccumulator
        acc = _TraceAccumulator(trace_id="test-123", trace_name="test")
        assert acc.active_skills == {}

    def test_skills_registry_stored_on_processor(self):
        """skills_registry passed to DecimalTracingProcessor is stored."""
        from decimalai.openai_agents import DecimalTracingProcessor
        proc = DecimalTracingProcessor(
            agent_name="test",
            skills_registry=SKILL_REGISTRY,
        )
        assert len(proc._skills_registry) == 3


# ── LangChain Integration tests ──────────────────────────


class TestLangChainSkills:
    """Tests for skill activation in the LangChain integration."""

    def test_reset_state_includes_active_skills(self):
        """_reset_state() initializes _active_skills as empty dict."""
        from decimalai.langchain import CallbackHandler
        handler = CallbackHandler(agent_name="test")
        assert hasattr(handler, "_active_skills")
        assert handler._active_skills == {}

    def test_build_trace_includes_active_skills(self):
        """build_trace() populates RunTrace.active_skills from handler state."""
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="test")
        handler._active_skills = {"code-review": "sha256:abc", "test-gen": None}
        handler._trace_started_at = datetime.now(timezone.utc)

        with patch("decimalai._config._config") as mock_cfg:
            mock_cfg.project = "test"

            import decimalai.langchain as lc_mod
            old_mid = getattr(lc_mod, '_manifest_id', None)
            lc_mod._manifest_id = None

            try:
                trace = handler.build_trace()
            finally:
                lc_mod._manifest_id = old_mid

        assert len(trace.active_skills) == 2
        names = {s["name"] for s in trace.active_skills}
        assert names == {"code-review", "test-gen"}


# ── OTEL Integration tests ───────────────────────────────


class TestOTELSkills:
    """Tests for skill activation in the OTEL integration."""

    def test_skills_stored_on_exporter(self):
        """Skills passed to DecimalSpanExporter are stored."""
        from decimalai.otel import DecimalSpanExporter
        exporter = DecimalSpanExporter(skills=SKILL_REGISTRY)
        assert len(exporter._skills) == 3

    def test_otel_span_active_skills_preserved(self):
        """OTEL spans with decimal.active_skills attributes are preserved in trace."""
        from decimalai.otel import DecimalSpanExporter

        exporter = DecimalSpanExporter(agent_name="test-agent")

        # Create mock OTEL span with active_skills attribute
        mock_span = MagicMock()
        mock_span.context.trace_id = 12345
        mock_span.context.span_id = 67890
        mock_span.parent = None  # root span
        mock_span.name = "agent-run"
        mock_span.start_time = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1e9)
        mock_span.end_time = int(datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc).timestamp() * 1e9)
        mock_span.status = MagicMock()
        mock_span.status.status_code = "OK"
        mock_span.resource = MagicMock()
        mock_span.resource.attributes = {"service.name": "test-agent"}

        # Set attributes including active_skills
        mock_span.attributes = {
            "gen_ai.request.model": "gpt-4o",
            "decimal.active_skills": ["code-review", "test-generator"],
        }

        with patch("decimalai._config._is_enabled", return_value=True), \
             patch("decimalai._config._config") as mock_cfg:
            mock_cfg.project = "test"

            result = exporter._assemble_trace([mock_span])

        assert result is not None
        trace, _, _, _ = result
        assert len(trace.active_skills) == 2
        skill_names = {s["name"] for s in trace.active_skills}
        assert skill_names == {"code-review", "test-generator"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
