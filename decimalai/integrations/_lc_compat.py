"""LangChain version compatibility helpers.

Abstracts version-specific differences in langchain-core so the
callback handler works across 0.2.x → 0.3.x → 1.x.

Known 1.5.x difference (handled in langchain.py, not here): child steps
reuse the root run_id — an LCEL sequence's ChatPromptTemplate and chat
model callbacks all arrive with run_id == parent_run_id == the root's
run_id, so run_id identity no longer distinguishes the outermost run.
Only parent_run_id=None marks the true outermost callback.

Strategy:
    - Detect installed version at import time
    - Provide helper functions that handle known API differences
    - Use hasattr / getattr fallback chains for unknown differences
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("decimalai.langchain.compat")

# ── Version Detection ──────────────────────────────────────────

_LANGCHAIN_VERSION: Optional[Tuple[int, ...]] = None


def detect_langchain_version() -> Optional[Tuple[int, ...]]:
    """Detect the installed langchain-core version.

    Returns:
        Tuple of (major, minor, patch) or None if not installed.
    """
    global _LANGCHAIN_VERSION
    if _LANGCHAIN_VERSION is not None:
        return _LANGCHAIN_VERSION

    try:
        from importlib.metadata import version as pkg_version
        v = pkg_version("langchain-core")
        parts = v.split(".")
        _LANGCHAIN_VERSION = tuple(int(p) for p in parts[:3])
        logger.debug("Detected langchain-core version: %s", v)
        return _LANGCHAIN_VERSION
    except Exception:
        return None


def is_langchain_installed() -> bool:
    """Check if langchain-core is installed."""
    return detect_langchain_version() is not None


def langchain_version_gte(major: int, minor: int, patch: int = 0) -> bool:
    """Check if installed langchain-core >= the given version."""
    v = detect_langchain_version()
    if v is None:
        return False
    return v >= (major, minor, patch)


# ── Message Helpers ────────────────────────────────────────────

def normalize_role(message: Any) -> str:
    """Extract a normalized role string from a LangChain message.

    Handles differences across versions:
        - 0.2.x: msg.type = "human" | "ai" | "system" | "tool" | "function"
        - 0.3.x: same, but also includes "generic" for ChatMessage
        - Some providers return custom types

    Returns:
        Normalized role: "user", "assistant", "system", "tool", or "unknown"
    """
    # Try .type attribute (standard for all LangChain message classes)
    role = getattr(message, "type", None)
    if role is None:
        # Fallback: dict-style messages
        if isinstance(message, dict):
            role = message.get("role") or message.get("type", "unknown")
        else:
            role = "unknown"

    # Normalize LangChain-specific names to standard roles
    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
        "function": "tool",  # deprecated in 0.2+
        "generic": "unknown",
        "chat": "unknown",
    }
    return role_map.get(str(role), str(role))


def extract_message_content(message: Any) -> str:
    """Extract text content from a LangChain message.

    Handles:
        - Simple string content
        - List content (multimodal messages in 0.3+)
        - Dict messages
    """
    content = getattr(message, "content", None)

    if content is None and isinstance(message, dict):
        content = message.get("content", "")

    if content is None:
        return str(message)

    # In 0.3+, content can be a list of dicts for multimodal messages
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        return " ".join(text_parts) if text_parts else str(content)

    return str(content)


# ── Tool Call Extraction ───────────────────────────────────────

def extract_tool_calls(message: Any) -> List[Dict[str, Any]]:
    """Extract tool calls from an LLM response message.

    Handles:
        - 0.2.x: AIMessage.tool_calls (list of ToolCall dicts)
        - 0.2.x: AIMessage.additional_kwargs["tool_calls"] (older format)
        - 0.3.x: AIMessage.tool_calls (standardized ToolCall objects)
        - Invalid/malformed tool calls via invalid_tool_calls attr

    Returns:
        List of dicts with keys: name, args, id (if available)
    """
    tool_calls = []

    # Primary: .tool_calls attribute (0.2+)
    raw_calls = getattr(message, "tool_calls", None)
    if raw_calls:
        for tc in raw_calls:
            if isinstance(tc, dict):
                tool_calls.append({
                    "name": tc.get("name", "unknown"),
                    "args": tc.get("args", {}),
                    "id": tc.get("id"),
                })
            elif hasattr(tc, "name"):
                # ToolCall object (Pydantic model in some versions)
                tool_calls.append({
                    "name": getattr(tc, "name", "unknown"),
                    "args": getattr(tc, "args", {}),
                    "id": getattr(tc, "id", None),
                })

    # Fallback: additional_kwargs (pre-0.2 and some providers)
    if not tool_calls:
        additional = getattr(message, "additional_kwargs", {})
        if isinstance(additional, dict):
            legacy_calls = additional.get("tool_calls", [])
            for tc in legacy_calls:
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    tool_calls.append({
                        "name": func.get("name", tc.get("name", "unknown")),
                        "args": _parse_args(func.get("arguments", "{}")),
                        "id": tc.get("id"),
                    })

    return tool_calls


def has_tool_calls(message: Any) -> bool:
    """Check if a message contains tool calls."""
    return len(extract_tool_calls(message)) > 0


# ── Token Usage Extraction ─────────────────────────────────────

def extract_token_usage(response: Any) -> Tuple[Optional[int], Optional[int]]:
    """Extract token usage from an LLM response.

    Handles:
        - 0.2.x: response.llm_output["token_usage"]
        - 0.3.x: response.llm_output["token_usage"] (same, but may be missing)
        - 0.3.x: AIMessage.usage_metadata (new in 0.3)
        - Provider-specific: different key names

    Returns:
        Tuple of (input_tokens, output_tokens), either may be None.
    """
    input_tokens = None
    output_tokens = None

    # Method 1: llm_output (works in 0.2 and 0.3)
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        usage = llm_output.get("token_usage", {})
        if isinstance(usage, dict):
            # OpenAI-style
            input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
            output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")

    # Method 2: usage_metadata on the generation's message (0.3+)
    if input_tokens is None:
        gen = _get_first_generation(response)
        if gen:
            msg = getattr(gen, "message", None)
            if msg:
                usage_meta = getattr(msg, "usage_metadata", None)
                if usage_meta:
                    if isinstance(usage_meta, dict):
                        input_tokens = usage_meta.get("input_tokens")
                        output_tokens = usage_meta.get("output_tokens")
                    elif hasattr(usage_meta, "input_tokens"):
                        input_tokens = getattr(usage_meta, "input_tokens", None)
                        output_tokens = getattr(usage_meta, "output_tokens", None)

    # Method 3: response_metadata on the message (some 0.3 providers)
    if input_tokens is None:
        gen = _get_first_generation(response)
        if gen:
            msg = getattr(gen, "message", None)
            if msg:
                resp_meta = getattr(msg, "response_metadata", {})
                if isinstance(resp_meta, dict):
                    usage = resp_meta.get("usage", resp_meta.get("token_usage", {}))
                    if isinstance(usage, dict):
                        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
                        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")

    return (input_tokens, output_tokens)


# ── Model Name Extraction ──────────────────────────────────────

def extract_model_name(invocation_params: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extract model name from invocation params.

    Different providers use different keys:
        - OpenAI: "model_name" or "model"
        - Google: "model"
        - Anthropic: "model"
        - Together: "model_name"
    """
    if not invocation_params:
        return None

    return (
        invocation_params.get("model_name")
        or invocation_params.get("model")
        or invocation_params.get("model_id")
    )


def extract_provider(invocation_params: Optional[Dict[str, Any]], serialized: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Extract provider name from invocation params and serialized data.

    Falls back through multiple possible sources.
    """
    if invocation_params:
        provider = invocation_params.get("_type")
        if provider:
            return str(provider)

    if serialized:
        # LangChain serialized often has id like ["langchain", "chat_models", "openai", "ChatOpenAI"]
        id_list = serialized.get("id", [])
        if len(id_list) >= 3:
            return str(id_list[-2])  # e.g., "openai"

        name = serialized.get("name", "")
        if name:
            # Extract provider hint from class names like "ChatOpenAI", "ChatGoogleGenerativeAI"
            name_lower = str(name).lower()
            for provider in ["openai", "anthropic", "google", "cohere", "together", "mistral"]:
                if provider in name_lower:
                    return provider

    return None


# ── Output Extraction ──────────────────────────────────────────

def extract_output_text(response: Any) -> Optional[str]:
    """Extract the text output from an LLM response.

    Handles:
        - generation.text (all versions)
        - generation.message.content (all versions)
    """
    gen = _get_first_generation(response)
    if gen is None:
        return None

    # Primary: .text attribute
    text = getattr(gen, "text", None)
    if text:
        return str(text)

    # Fallback: .message.content
    msg = getattr(gen, "message", None)
    if msg:
        content = extract_message_content(msg)
        if content:
            return content

    return str(gen)


def extract_output_dict(response: Any) -> Optional[Dict[str, Any]]:
    """Extract a structured output dict from an LLM response."""
    text = extract_output_text(response)
    if text is None:
        return None
    return {"content": text}


# ── Internal Helpers ───────────────────────────────────────────

def _get_first_generation(response: Any) -> Any:
    """Safely get the first generation from a response."""
    gens = getattr(response, "generations", None)
    if gens and len(gens) > 0 and len(gens[0]) > 0:
        return gens[0][0]
    return None


def _parse_args(args: Any) -> Dict[str, Any]:
    """Parse tool call arguments from various formats."""
    import json

    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            return json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return {"raw": args}
    return {}
