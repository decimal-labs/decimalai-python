"""LangChain compatibility tests.

Tests the DecimalAI callback handler against REAL LangChain objects
to catch API incompatibilities when langchain-core updates.

These tests require langchain-core to be installed and will skip
gracefully if it is not available.

Run against different versions in CI:
    pip install langchain-core==0.2.38 && pytest tests/test_langchain_compat.py
    pip install langchain-core==0.3.29 && pytest tests/test_langchain_compat.py
"""

from __future__ import annotations

import sys
from uuid import uuid4

import pytest

# Skip entire module if langchain-core not installed
try:
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.outputs import ChatGeneration, LLMResult

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

pytestmark = pytest.mark.skipif(
    not HAS_LANGCHAIN, reason="langchain-core not installed"
)

# Import SDK after the skip check
if HAS_LANGCHAIN:
    sys.path.insert(0, ".")
    from decimalai.langchain import CallbackHandler
    from decimalai.integrations._lc_compat import (
        detect_langchain_version,
        extract_message_content,
        extract_token_usage,
        extract_tool_calls,
        has_tool_calls,
        normalize_role,
    )


# ── Version Detection ──────────────────────────────────────────


class TestVersionDetection:
    """Verify that version detection works with installed langchain-core."""

    def test_detects_installed_version(self):
        version = detect_langchain_version()
        assert version is not None
        assert len(version) >= 2  # (major, minor) at minimum
        print(f"Detected langchain-core: {'.'.join(str(v) for v in version)}")


# ── Message Role Normalization ─────────────────────────────────


class TestRoleNormalization:
    """Verify role extraction from real LangChain message objects."""

    def test_human_message_role(self):
        msg = HumanMessage(content="hello")
        assert normalize_role(msg) == "user"

    def test_ai_message_role(self):
        msg = AIMessage(content="hi there")
        assert normalize_role(msg) == "assistant"

    def test_system_message_role(self):
        msg = SystemMessage(content="You are helpful.")
        assert normalize_role(msg) == "system"

    def test_tool_message_role(self):
        msg = ToolMessage(content="result", tool_call_id="tc_1")
        assert normalize_role(msg) == "tool"

    def test_dict_message_role(self):
        msg = {"role": "user", "content": "test"}
        assert normalize_role(msg) == "user"


# ── Content Extraction ─────────────────────────────────────────


class TestContentExtraction:
    """Verify content extraction from real LangChain messages."""

    def test_simple_string_content(self):
        msg = HumanMessage(content="What is 2+2?")
        assert extract_message_content(msg) == "What is 2+2?"

    def test_ai_message_content(self):
        msg = AIMessage(content="The answer is 4.")
        assert extract_message_content(msg) == "The answer is 4."

    def test_empty_content(self):
        msg = AIMessage(content="")
        assert extract_message_content(msg) == ""

    def test_multimodal_list_content(self):
        """In 0.3+, content can be a list for multimodal messages."""
        msg = HumanMessage(content=[
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
        ])
        extracted = extract_message_content(msg)
        assert "What is this?" in extracted


# ── Tool Call Extraction ───────────────────────────────────────


class TestToolCallExtraction:
    """Verify tool call extraction from real AIMessage objects."""

    def test_ai_message_with_tool_calls(self):
        """AIMessage.tool_calls is the primary format in 0.2+."""
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "get_weather", "args": {"city": "SF"}, "id": "tc_1"},
            ],
        )
        calls = extract_tool_calls(msg)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["args"]["city"] == "SF"
        assert calls[0]["id"] == "tc_1"

    def test_ai_message_without_tool_calls(self):
        msg = AIMessage(content="Just a normal response.")
        calls = extract_tool_calls(msg)
        assert len(calls) == 0

    def test_has_tool_calls_true(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "calc", "args": {}, "id": "tc_2"}],
        )
        assert has_tool_calls(msg) is True

    def test_has_tool_calls_false(self):
        msg = AIMessage(content="No tools here.")
        assert has_tool_calls(msg) is False

    def test_multiple_tool_calls(self):
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "search", "args": {"q": "weather"}, "id": "tc_1"},
                {"name": "calculate", "args": {"expr": "2+2"}, "id": "tc_2"},
            ],
        )
        calls = extract_tool_calls(msg)
        assert len(calls) == 2
        names = {c["name"] for c in calls}
        assert names == {"search", "calculate"}

    def test_legacy_additional_kwargs_tool_calls(self):
        """Pre-0.2 format: tool_calls in additional_kwargs."""
        msg = AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_stock",
                            "arguments": '{"ticker": "AAPL"}',
                        },
                    }
                ]
            },
        )
        calls = extract_tool_calls(msg)
        assert len(calls) >= 1
        assert calls[0]["name"] == "get_stock"


# ── Token Usage Extraction ─────────────────────────────────────


class TestTokenUsage:
    """Verify token usage extraction from LLMResult objects."""

    def test_standard_llm_output_token_usage(self):
        """Classic format: response.llm_output["token_usage"]."""
        result = LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="hello"))]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 20,
                }
            },
        )
        input_tok, output_tok = extract_token_usage(result)
        assert input_tok == 15
        assert output_tok == 20

    def test_no_token_usage(self):
        """When llm_output has no token_usage."""
        result = LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="hello"))]],
            llm_output={},
        )
        input_tok, output_tok = extract_token_usage(result)
        assert input_tok is None
        assert output_tok is None

    def test_usage_metadata_on_message(self):
        """0.3+ format: usage_metadata on AIMessage."""
        msg = AIMessage(
            content="hello",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 25,
                "total_tokens": 35,
            },
        )
        result = LLMResult(
            generations=[[ChatGeneration(message=msg)]],
        )
        input_tok, output_tok = extract_token_usage(result)
        assert input_tok == 10
        assert output_tok == 25


# ── Full Callback Handler Round-Trip ───────────────────────────


class TestCallbackRoundTrip:
    """End-to-end: real LangChain objects → callback handler → valid trace."""

    def test_simple_chat(self):
        """Simple chat: HumanMessage → AIMessage → valid trace."""
        handler = CallbackHandler(agent_name="compat-test-agent")
        chain_id = uuid4()
        llm_id = uuid4()

        # Chain start
        handler.on_chain_start(
            serialized={"name": "TestChain"},
            inputs={"input": "What is 2+2?"},
            run_id=chain_id,
        )

        # Chat model with real messages
        handler.on_chat_model_start(
            serialized={"name": "ChatOpenAI"},
            messages=[[HumanMessage(content="What is 2+2?")]],
            run_id=llm_id,
            parent_run_id=chain_id,
            invocation_params={"model_name": "gpt-4o", "temperature": 0.0},
        )

        # LLM response with real LLMResult
        result = LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="4"))]],
            llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        )
        handler.on_llm_end(response=result, run_id=llm_id)

        # Chain end
        handler.on_chain_end(
            outputs={"output": "4"},
            run_id=chain_id,
        )

        # Build and validate
        trace = handler.get_completed_trace()
        assert trace.agent_name == "compat-test-agent"
        assert len(trace.spans) >= 1
        assert len(trace.llm_calls) == 1

        llm_call = trace.llm_calls[0]
        assert llm_call.model_name == "gpt-4o"
        assert llm_call.input_tokens == 10
        assert llm_call.output_tokens == 5
        assert llm_call.output is not None
        assert "4" in llm_call.output.get("content", "")

        # Check that role was normalized
        assert llm_call.rendered_input is not None
        assert llm_call.rendered_input[0]["role"] == "user"  # "human" → "user"

    def test_chat_with_tool_calls(self):
        """Chat with tool calls: AIMessage with tool_calls → correct trace."""
        handler = CallbackHandler(agent_name="tool-compat-agent")
        chain_id = uuid4()
        llm_id = uuid4()
        tool_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "ToolChain"},
            inputs={"input": "What's the weather?"},
            run_id=chain_id,
        )

        # LLM call with real messages
        handler.on_chat_model_start(
            serialized={"name": "ChatOpenAI"},
            messages=[[HumanMessage(content="What's the weather?")]],
            run_id=llm_id,
            parent_run_id=chain_id,
            invocation_params={"model": "gpt-4o-mini"},
        )

        # LLM responds with tool calls
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "get_weather", "args": {"city": "SF"}, "id": "tc_1"},
            ],
        )
        result = LLMResult(
            generations=[[ChatGeneration(message=ai_msg)]],
            llm_output={"token_usage": {"prompt_tokens": 15, "completion_tokens": 8}},
        )
        handler.on_llm_end(response=result, run_id=llm_id)

        # Tool execution
        handler.on_tool_start(
            serialized={"name": "get_weather"},
            input_str='{"city": "SF"}',
            run_id=tool_id,
            parent_run_id=chain_id,
        )
        handler.on_tool_end(output="Sunny, 72°F", run_id=tool_id)

        handler.on_chain_end(
            outputs={"output": "It's sunny in SF."},
            run_id=chain_id,
        )

        trace = handler.get_completed_trace()
        assert len(trace.llm_calls) >= 1
        assert trace.llm_calls[0].model_name == "gpt-4o-mini"

        # Verify tool span was created
        tool_spans = [s for s in trace.spans if s.name == "get_weather"]
        assert len(tool_spans) == 1
        assert tool_spans[0].output_preview == "Sunny, 72°F"

    def test_multi_turn_with_system_message(self):
        """Multi-turn chat with system, human, and AI messages."""
        handler = CallbackHandler(agent_name="multi-turn-agent")
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "MultiTurnChain"},
            inputs={"input": "Tell me about Python."},
            run_id=chain_id,
        )

        handler.on_chat_model_start(
            serialized={"name": "ChatGemini"},
            messages=[[
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Tell me about Python."),
                AIMessage(content="Python is great!"),
                HumanMessage(content="What about its GIL?"),
            ]],
            run_id=llm_id,
            parent_run_id=chain_id,
            invocation_params={"model_name": "gemini-2.0-flash"},
        )

        result = LLMResult(
            generations=[[ChatGeneration(
                message=AIMessage(content="The GIL is being removed in 3.13+.")
            )]],
        )
        handler.on_llm_end(response=result, run_id=llm_id)
        handler.on_chain_end(
            outputs={"output": "The GIL is being removed."},
            run_id=chain_id,
        )

        trace = handler.get_completed_trace()
        llm_call = trace.llm_calls[0]

        # Verify all roles are normalized
        roles = [m["role"] for m in llm_call.rendered_input]
        assert roles == ["system", "user", "assistant", "user"]

    def test_error_handling(self):
        """Verify LLM errors are captured correctly."""
        handler = CallbackHandler(agent_name="error-agent")
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "ErrorChain"},
            inputs={"input": "test"},
            run_id=chain_id,
        )
        handler.on_chat_model_start(
            serialized={"name": "ChatOpenAI"},
            messages=[[HumanMessage(content="test")]],
            run_id=llm_id,
            parent_run_id=chain_id,
        )
        handler.on_llm_error(
            error=RuntimeError("Rate limit exceeded"),
            run_id=llm_id,
        )
        handler.on_chain_end(
            outputs={"output": "Error occurred"},
            run_id=chain_id,
        )

        trace = handler.get_completed_trace()
        assert len(trace.llm_calls) == 1
        assert trace.llm_calls[0].status.value == "error"
        assert "Rate limit" in str(trace.llm_calls[0].output)

    def test_tool_message_in_conversation(self):
        """ToolMessage is correctly handled in conversation history."""
        handler = CallbackHandler(agent_name="tool-msg-agent")
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "ToolMsgChain"},
            inputs={"input": "test"},
            run_id=chain_id,
        )

        handler.on_chat_model_start(
            serialized={"name": "ChatOpenAI"},
            messages=[[
                HumanMessage(content="Get weather"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "weather", "args": {}, "id": "tc_1"}],
                ),
                ToolMessage(content="Sunny 72F", tool_call_id="tc_1"),
            ]],
            run_id=llm_id,
            parent_run_id=chain_id,
            invocation_params={"model_name": "gpt-4o"},
        )

        result = LLMResult(
            generations=[[ChatGeneration(
                message=AIMessage(content="It's sunny and 72°F.")
            )]],
        )
        handler.on_llm_end(response=result, run_id=llm_id)
        handler.on_chain_end(
            outputs={"output": "It's sunny."},
            run_id=chain_id,
        )

        trace = handler.get_completed_trace()
        llm_call = trace.llm_calls[0]

        # Check roles: user, assistant, tool - verify ToolMessage is handled
        roles = [m["role"] for m in llm_call.rendered_input]
        assert roles == ["user", "assistant", "tool"]
        assert llm_call.rendered_input[2]["content"] == "Sunny 72F"
