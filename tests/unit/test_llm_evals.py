"""Tests for LLM-as-Judge evaluators.

Covers:
1. Prompt construction for all 5 evaluators
2. Response parsing and EvalResult coercion
3. Error handling (LLM failures)
4. Integration: run_evals() with mixed deterministic + LLM evals (mocked)
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from decimalai.evals import (
    TraceData,
    EvalResult,
    run_evals,
    ToolCallView,
)
from decimalai.evals.llm_evaluators import (
    LlmEval,
    Relevance,
    Factuality,
    Faithfulness,
    Toxicity,
    Conciseness,
    _response_to_result,
)
from decimalai.evals.prebuilt import json_valid


def _make_trace(
    output: str = "The answer is 42.",
    input: str = "What is the answer?",
    tool_calls=None,
    context=None,
    metadata=None,
) -> TraceData:
    """Helper to create a TraceData for testing."""
    return TraceData(
        id="test-trace-llm-001",
        input=input,
        output=output,
        status="success",
        tool_calls=tool_calls or [],
        context=context,
        metadata=metadata or {},
    )


def _mock_llm_response(score: float, passed: bool, reason: str):
    """Return a dict matching _call_llm's return format."""
    return {"score": score, "passed": passed, "reason": reason}


# ── Prompt Construction ──────────────────────────────────────────


class TestPromptConstruction:
    """Verify that each evaluator builds prompts with the correct fields."""

    def test_relevance_prompt_contains_input_and_output(self):
        ev = Relevance()
        trace = _make_trace(input="How do I cook pasta?", output="Boil water first.")
        prompt = ev._build_prompt(trace)
        assert "How do I cook pasta?" in prompt
        assert "Boil water first." in prompt
        assert "relevant" in prompt.lower()

    def test_factuality_prompt_contains_context(self):
        ev = Factuality()
        trace = _make_trace(
            input="What is the capital?",
            output="Paris is the capital.",
            context="France's capital is Paris.",
        )
        prompt = ev._build_prompt(trace)
        assert "France's capital is Paris." in prompt
        assert "Paris is the capital." in prompt
        assert "factual" in prompt.lower()

    def test_factuality_uses_metadata_context(self):
        ev = Factuality()
        trace = _make_trace(
            output="Tokyo is in Japan.",
            metadata={"context": "Japan's capital city is Tokyo."},
        )
        prompt = ev._build_prompt(trace)
        assert "Japan's capital city is Tokyo." in prompt

    def test_faithfulness_prompt_contains_tool_results(self):
        ev = Faithfulness()
        trace = _make_trace(
            output="The weather is sunny.",
            tool_calls=[
                ToolCallView(name="get_weather", args={"city": "LA"}, result="sunny, 72°F"),
            ],
        )
        prompt = ev._build_prompt(trace)
        assert "get_weather" in prompt
        assert "sunny, 72°F" in prompt
        assert "faithful" in prompt.lower()

    def test_toxicity_prompt_focus_on_output(self):
        ev = Toxicity()
        trace = _make_trace(output="This is a helpful response.")
        prompt = ev._build_prompt(trace)
        assert "This is a helpful response." in prompt
        assert "safe" in prompt.lower()

    def test_conciseness_prompt_contains_both(self):
        ev = Conciseness()
        trace = _make_trace(
            input="What is 2+2?",
            output="Well, you see, when we consider the mathematical operation of addition...",
        )
        prompt = ev._build_prompt(trace)
        assert "What is 2+2?" in prompt
        assert "addition" in prompt
        assert "concise" in prompt.lower()


# ── Response Parsing ─────────────────────────────────────────────


class TestResponseParsing:
    def test_valid_response(self):
        resp = {"score": 0.85, "passed": True, "reason": "Good output"}
        result = _response_to_result(resp)
        assert result.score == 0.85
        assert result.passed is True
        assert result.reason == "Good output"

    def test_score_clamped_high(self):
        resp = {"score": 1.5, "passed": True, "reason": "Over max"}
        result = _response_to_result(resp)
        assert result.score == 1.0

    def test_score_clamped_low(self):
        resp = {"score": -0.5, "passed": False, "reason": "Under min"}
        result = _response_to_result(resp)
        assert result.score == 0.0

    def test_missing_passed_derives_from_score(self):
        resp = {"score": 0.3}
        result = _response_to_result(resp)
        assert result.passed is False  # 0.3 < 0.5

    def test_missing_passed_passing_score(self):
        resp = {"score": 0.7}
        result = _response_to_result(resp)
        assert result.passed is True  # 0.7 >= 0.5

    def test_empty_response(self):
        result = _response_to_result({})
        assert result.score == 0.5
        assert result.passed is True


# ── _call_llm robustness (the litellm call itself) ───────────────


class TestCallLlmRobustness:
    """`_call_llm` must survive a None/empty completion body. Reasoning models
    (Gemini 2.5/3.x, o-series) can spend the whole token budget on internal
    thinking and return ``message.content=None``; calling ``.strip()`` on that
    used to crash with AttributeError. These pin the guard + max_tokens fix."""

    def test_none_content_returns_empty_response(self):
        pytest.importorskip("litellm")
        from decimalai.evals.llm_evaluators import _call_llm

        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content=None))]
        with patch("litellm.completion", return_value=resp):
            out = _call_llm("rate this", "gemini/gemini-2.5-pro")

        assert out["score"] == 0.5
        assert out["reason"] == "Evaluator returned an empty response"

    def test_valid_json_content_parses(self):
        pytest.importorskip("litellm")
        from decimalai.evals.llm_evaluators import _call_llm

        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(
            content='{"score": 0.9, "passed": true, "reason": "good"}'))]
        with patch("litellm.completion", return_value=resp):
            out = _call_llm("rate this", "gemini/gemini-2.5-pro")

        assert out["score"] == 0.9
        assert out["passed"] is True


# ── Mocked LLM Calls ────────────────────────────────────────────


class TestMockedLlmEval:
    """Test evaluators with _call_llm mocked at the function level."""

    @patch("decimalai.evals.llm_evaluators._call_llm")
    def test_relevance_returns_score(self, mock_call):
        mock_call.return_value = _mock_llm_response(0.9, True, "Addresses the question")

        ev = Relevance()
        trace = _make_trace()
        result = ev(trace)

        assert result is not None
        assert result.score == 0.9
        assert result.passed is True
        mock_call.assert_called_once()

        # Verify the model was passed (default = current budget OpenAI model)
        call_args = mock_call.call_args
        assert call_args[0][1] == "gpt-5.4-mini"  # model arg

    @patch("decimalai.evals.llm_evaluators._call_llm")
    def test_toxicity_safe_output(self, mock_call):
        mock_call.return_value = _mock_llm_response(1.0, True, "Content is safe")

        ev = Toxicity()
        trace = _make_trace(output="Have a great day!")
        result = ev(trace)

        assert result.score == 1.0
        assert result.passed is True

    @patch("decimalai.evals.llm_evaluators._call_llm")
    def test_custom_model(self, mock_call):
        mock_call.return_value = _mock_llm_response(0.8, True, "OK")

        ev = Relevance(model="claude-3-5-haiku-20241022")
        trace = _make_trace()
        ev(trace)

        call_args = mock_call.call_args
        assert call_args[0][1] == "claude-3-5-haiku-20241022"

    @patch("decimalai.evals.llm_evaluators._call_llm")
    def test_llm_error_graceful(self, mock_call):
        """When _call_llm returns a fallback, eval should not crash."""
        mock_call.return_value = {"score": 0.5, "passed": True, "reason": "Eval error"}

        ev = Relevance()
        trace = _make_trace()
        result = ev(trace)

        assert result is not None
        assert result.score == 0.5

    @patch("decimalai.evals.llm_evaluators._call_llm")
    def test_factuality_with_context(self, mock_call):
        mock_call.return_value = _mock_llm_response(0.95, True, "Grounded in context")

        ev = Factuality()
        trace = _make_trace(
            output="Paris is the capital of France.",
            context="France's capital city is Paris.",
        )
        result = ev(trace)

        assert result.score == 0.95
        # Verify the prompt included the context
        prompt_arg = mock_call.call_args[0][0]
        assert "France's capital city is Paris." in prompt_arg

    @patch("decimalai.evals.llm_evaluators._call_llm")
    def test_faithfulness_with_tools(self, mock_call):
        mock_call.return_value = _mock_llm_response(0.88, True, "Faithful to tools")

        ev = Faithfulness()
        trace = _make_trace(
            output="Weather is sunny in LA",
            tool_calls=[ToolCallView(name="weather_api", args={}, result="sunny, 72°F")],
        )
        result = ev(trace)

        assert result.score == 0.88
        prompt_arg = mock_call.call_args[0][0]
        assert "weather_api" in prompt_arg


# ── Integration: run_evals with mixed eval types ──────────────────


class TestIntegration:
    """Test run_evals() with both deterministic and LLM evals together."""

    @patch("decimalai.evals.llm_evaluators._call_llm")
    def test_mixed_evals(self, mock_call):
        """Run deterministic + LLM evals in a single run_evals() call."""
        mock_call.return_value = _mock_llm_response(0.85, True, "Relevant")

        trace = _make_trace(output='{"result": "success"}')

        # Mix deterministic (json_valid) + LLM (Relevance)
        scores = run_evals(trace, [json_valid, Relevance()])

        assert len(scores) == 2

        # json_valid should pass (output is valid JSON)
        json_score = next(s for s in scores if s["name"] == "json_valid")
        assert json_score["passed"] is True
        assert json_score["score"] == 1.0

        # Relevance should pass (mocked)
        relevance_score = next(s for s in scores if s["name"] == "relevance")
        assert relevance_score["passed"] is True
        assert relevance_score["score"] == 0.85

    @patch("decimalai.evals.llm_evaluators._call_llm")
    def test_full_eval_suite(self, mock_call):
        """Run all 5 LLM evaluators together."""
        mock_call.return_value = _mock_llm_response(0.75, True, "Acceptable")

        trace = _make_trace(
            input="Summarize this document",
            output="The document discusses AI safety protocols.",
            tool_calls=[ToolCallView(name="search", args={}, result="AI safety doc")],
            context="A document about AI safety protocols and guidelines.",
        )

        evals = [Relevance(), Factuality(), Faithfulness(), Toxicity(), Conciseness()]
        scores = run_evals(trace, evals)

        assert len(scores) == 5
        eval_names = {s["name"] for s in scores}
        assert eval_names == {"relevance", "factuality", "faithfulness", "toxicity", "conciseness"}

        for s in scores:
            assert s["passed"] is True
            assert s["score"] == 0.75
