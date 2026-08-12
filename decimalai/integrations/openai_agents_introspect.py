"""OpenAI Agents SDK introspection — extract manifest data without invoking the agent.

Parallel to `langchain_introspect.py`. The OpenAI Agents SDK (`agents`
package, sometimes called `openai-agents-sdk`) exposes an `Agent` class
with explicit `tools`, `instructions`, `model`, and `handoffs` fields,
which makes structural extraction cleaner than LangChain (no walk
through Runnable wrappers).

Used by `decimalai.flush_manifest_for_ci(chain=...)`. The function
returns the same `(tools, prompts, models)` shape so the CI flush path
treats both SDKs identically.

Design notes
------------
- Pure attribute access. No agent invocation, no API calls.
- Handles handoffs (sub-agents) as DELEGATE references in the manifest.
- For unsupported tool types (ComputerTool, FileSearchTool, etc.) we
  extract name + a small descriptor instead of full schemas.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("decimalai.openai_agents.introspect")


def introspect_agent(agent: Any) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, str],
    Dict[str, Dict[str, Any]],
]:
    """Pull (tools, prompts, models) from an OpenAI Agents SDK `Agent`.

    Args:
        agent: An `agents.Agent` instance (or anything duck-typing the same
            shape: `name`, `tools`, `instructions`, `model` attributes).

    Returns:
        (tools, prompts, models) — same shape as `flush_manifest_for_ci`
        expects. Empty dicts/lists where extraction fails so the caller
        can supplement explicitly.
    """
    tools = _extract_tools(agent)
    prompts = _extract_prompts(agent)
    models = _extract_models(agent)
    return tools, prompts, models


# ─────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────


def _extract_tools(agent: Any) -> List[Dict[str, Any]]:
    """Return `[{name, schema}, ...]` for every tool on the agent.

    The Agents SDK uses several tool types (FunctionTool, FileSearchTool,
    ComputerTool, CodeInterpreterTool, etc.). FunctionTool exposes a
    `params_json_schema` directly. Built-in tools generally just have a
    `name` and a type-specific config dict that's good enough for the
    manifest diff to compare against.
    """
    raw_tools = getattr(agent, "tools", None) or []
    if not isinstance(raw_tools, (list, tuple)):
        return []

    out: List[Dict[str, Any]] = []
    seen_names = set()
    for tool in raw_tools:
        name = getattr(tool, "name", None)
        if not name or not isinstance(name, str):
            continue
        if name in seen_names:
            continue
        seen_names.add(name)
        out.append({
            "name": name,
            "schema": _tool_schema(tool),
        })
    return out


def _tool_schema(tool: Any) -> Dict[str, Any]:
    """Best-effort JSON schema for an Agents SDK tool's args.

    FunctionTool exposes `params_json_schema` (a dict). Built-in tools
    don't have args schemas — we return a small descriptor with the
    tool's class name so the diff can still detect type changes.
    """
    # FunctionTool / FunctionToolBase
    schema = getattr(tool, "params_json_schema", None)
    if isinstance(schema, dict):
        return schema

    # Built-in tools (FileSearchTool, ComputerTool, ...) — emit the class
    # name + description so manifest diffs can spot a tool type change.
    descriptor: Dict[str, Any] = {"tool_type": type(tool).__name__}
    description = getattr(tool, "description", None)
    if isinstance(description, str) and description:
        descriptor["description"] = description
    return descriptor


# ─────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────


def _extract_prompts(agent: Any) -> Dict[str, str]:
    """Return `{section_name: text}` for prompts on the agent.

    `Agent.instructions` is the canonical system prompt. It may be either
    a plain string or a callable (dynamic prompt). For callables we
    record the function's name + docstring as a best-effort hint —
    structural diff can still detect that the callable changed.
    """
    instructions = getattr(agent, "instructions", None)
    if instructions is None:
        return {}

    if isinstance(instructions, str):
        return {"system": instructions}

    # Callable / DynamicPromptFunction — record its identity so a diff
    # detects code-level changes.
    if callable(instructions):
        fn_name = getattr(instructions, "__name__", "dynamic")
        doc = (getattr(instructions, "__doc__", None) or "").strip()
        return {"system_dynamic": f"<dynamic:{fn_name}> {doc}".strip()}

    return {}


# ─────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────


def _extract_models(agent: Any) -> Dict[str, Dict[str, Any]]:
    """Return `{key: {provider, model, ...}}` for the agent's LLM.

    `Agent.model` is either a string (e.g. "gpt-4o"), a Model instance,
    or None. We extract whatever we can; missing fields come back empty.
    """
    model = getattr(agent, "model", None)
    if model is None:
        return {}

    model_name = _model_name(model)
    if not model_name:
        return {}

    # Try to pull temperature/max_tokens from agent.model_settings
    settings = getattr(agent, "model_settings", None)
    temperature = getattr(settings, "temperature", None)
    max_tokens = getattr(settings, "max_tokens", None)

    return {
        "default": {
            "provider": _infer_provider(model, model_name),
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    }


def _model_name(model: Any) -> str:
    """Pull the model name from a string-or-Model value."""
    if isinstance(model, str):
        return model
    # Model instances expose `.model` or `.name`
    name = getattr(model, "model", None) or getattr(model, "name", None)
    return name if isinstance(name, str) else ""


def _infer_provider(model: Any, name: str) -> str:
    """Best-effort provider hint. The Agents SDK is OpenAI-first but
    supports other providers via the `Model` interface.
    """
    cls_name = type(model).__name__.lower() if not isinstance(model, str) else ""
    haystack = (cls_name + " " + name).lower()
    for hint, provider in (
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        ("claude", "anthropic"),
        ("gemini", "google"),
        ("vertex", "google"),
        ("google", "google"),
        ("mistral", "mistral"),
        ("cohere", "cohere"),
        ("groq", "groq"),
        ("ollama", "ollama"),
    ):
        if hint in haystack:
            return provider
    # OpenAI Agents SDK defaults to OpenAI when given a bare string
    return "openai" if isinstance(model, str) else "unknown"
