"""LangChain agent introspection — extract manifest data without invoking the agent.

The runtime LangChain callback (decimalai/langchain.py) captures manifests as
a side effect of agent EXECUTION. That works for production tracing but not
for CI manifest extraction — in CI we instantiate the agent but don't run it
(no API keys, no side effects). This module fills that gap by introspecting
the agent object STATICALLY after instantiation.

Used by `decimalai.flush_manifest_for_ci(chain=...)`. See
`tests/test_langchain_introspect.py` for the supported agent shapes.

Design notes
------------
- We support the common shapes: `create_react_agent` results, simple chains
  with `.tools` + `.llm` + `.prompt`, AgentExecutors wrapping the above.
- For unsupported shapes the introspection returns whatever it can find +
  empty dicts for the rest. The caller can fall back to explicit args.
- We intentionally do not invoke any chain methods. Pure attribute access.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("decimalai.langchain.introspect")


def introspect_chain(chain: Any) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, Dict[str, Any]]]:
    """Pull (tools, prompts, models) from a LangChain chain/agent object.

    Args:
        chain: a LangChain Runnable / AgentExecutor / chain. Common shapes:
            - Result of `create_react_agent(llm, tools, prompt)` (a Runnable)
            - AgentExecutor wrapping an agent
            - Simple Chain with `.tools`, `.prompt`, `.llm`

    Returns:
        (tools, prompts, models) — same shape as `flush_manifest_for_ci` expects.
        Any field we can't find comes back empty; the caller can fill in
        explicitly or accept the partial extraction.

    **langgraph `create_react_agent` limitation**: tools ARE extracted
    (via explicit `nodes['tools'].bound.tools_by_name` handling), but
    model + prompt are closure-captured inside `nodes['agent'].bound`
    (a `RunnableCallable`) with no public attribute to read them from.
    For langgraph users, pass `models=` and `prompts=` explicitly to
    `flush_manifest_for_ci(..., chain=...)` — the merge logic in that
    function uses explicit args first, introspection second, so the
    explicit values fill the gap. See `tests/test_langgraph_model_prompt_extraction.py`
    for the contract.
    """
    tools = _extract_tools(chain)
    prompts = _extract_prompts(chain)
    models = _extract_models(chain)
    return tools, prompts, models


# ─────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────


def _extract_tools(chain: Any) -> List[Dict[str, Any]]:
    """Return a list of {name, schema} dicts for every tool on the chain.

    Looks at chain.tools, chain.agent.tools, chain.bound.tools (Runnables
    sometimes wrap things in `.bound`), and chain.steps[-1].tools.

    For a langgraph `CompiledStateGraph` (the output of `create_react_agent`),
    tools live at `nodes['tools'].bound.tools_by_name` — a different shape
    that the generic walker can't reach, so it gets an explicit branch below.
    """
    # langgraph CompiledStateGraph path. Detected via the `nodes` dict
    # rather than by isinstance to avoid an import dependency on langgraph
    # in environments where it isn't installed.
    compiled_tools = _extract_tools_from_compiled_graph(chain)
    if compiled_tools:
        return compiled_tools

    candidates = _collect_attr(chain, "tools")
    if not candidates:
        return []

    out: List[Dict[str, Any]] = []
    seen_names = set()
    for tool in candidates:
        name = getattr(tool, "name", None)
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        out.append({"name": name, "schema": _tool_schema(tool)})
    return out


def _extract_tools_from_compiled_graph(chain: Any) -> List[Dict[str, Any]]:
    """Pull tools from a langgraph `CompiledStateGraph` (create_react_agent shape).

    The compiled graph stores tools at `nodes['tools'].bound.tools_by_name`
    (a `dict[str, BaseTool]`). Returns [] for any other shape so the caller
    falls through to the generic walker.

    Detected by duck-typing (presence of `.nodes` with a `'tools'` key
    whose `.bound` has `.tools_by_name`). No langgraph import required.
    """
    nodes = getattr(chain, "nodes", None)
    if not isinstance(nodes, dict):
        return []
    tools_node = nodes.get("tools")
    if tools_node is None:
        return []
    inner = getattr(tools_node, "bound", None)
    if inner is None:
        return []
    tools_by_name = getattr(inner, "tools_by_name", None)
    if not isinstance(tools_by_name, dict):
        return []

    out: List[Dict[str, Any]] = []
    seen_names = set()
    for name, tool in tools_by_name.items():
        actual_name = getattr(tool, "name", None) or name
        if not actual_name or actual_name in seen_names:
            continue
        seen_names.add(actual_name)
        out.append({"name": actual_name, "schema": _tool_schema(tool)})
    return out


def _tool_schema(tool: Any) -> Dict[str, Any]:
    """Best-effort JSON schema for a LangChain tool's args."""
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is None:
        return {}
    # Pydantic v2
    if hasattr(args_schema, "model_json_schema"):
        try:
            return args_schema.model_json_schema()
        except Exception:
            pass
    # Pydantic v1 fallback
    if hasattr(args_schema, "schema"):
        try:
            return args_schema.schema()
        except Exception:
            pass
    return {}


# ─────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────


def _extract_prompts(chain: Any) -> Dict[str, str]:
    """Return {section_name: text} for prompts on the chain.

    Prompts can come from chain.prompt (BasePromptTemplate) or be embedded
    in chain.messages / chain.input_variables for chat templates.
    """
    prompt_obj = _walk_for_attr(chain, "prompt") or _walk_for_attr(chain, "messages")
    if prompt_obj is None:
        return {}

    # ChatPromptTemplate has .messages — a list of message templates
    messages = getattr(prompt_obj, "messages", None)
    if messages:
        parts: List[str] = []
        for m in messages:
            content = _message_template_to_text(m)
            if content:
                parts.append(content)
        if parts:
            return {"system": "\n\n".join(parts)}

    # PromptTemplate has .template directly
    template = getattr(prompt_obj, "template", None)
    if isinstance(template, str):
        return {"system": template}

    return {}


def _message_template_to_text(message: Any) -> Optional[str]:
    """Render a single message template as text. Best effort."""
    prompt = getattr(message, "prompt", None)
    if prompt is not None:
        text = getattr(prompt, "template", None)
        if isinstance(text, str):
            return text
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    return None


# ─────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────


def _extract_models(chain: Any) -> Dict[str, Dict[str, Any]]:
    """Return {key: {provider, model, ...}} for LLMs on the chain."""
    llm = _walk_for_attr(chain, "llm") or _walk_for_attr(chain, "model")
    if llm is None:
        return {}

    model_name = (
        getattr(llm, "model_name", None)
        or getattr(llm, "model", None)
        or getattr(llm, "deployment_name", None)
    )
    if not model_name:
        return {}

    return {
        "default": {
            "provider": _infer_provider(llm),
            "model": model_name,
            "temperature": getattr(llm, "temperature", None),
            "max_tokens": getattr(llm, "max_tokens", None),
        }
    }


def _infer_provider(llm: Any) -> str:
    """Guess the provider from the LLM class name. ChatOpenAI → openai, etc."""
    class_name = type(llm).__name__.lower()
    for hint, provider in (
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        ("vertex", "google"),
        ("google", "google"),
        ("gemini", "google"),
        ("mistral", "mistral"),
        ("cohere", "cohere"),
        ("groq", "groq"),
        ("ollama", "ollama"),
    ):
        if hint in class_name:
            return provider
    return "unknown"


# ─────────────────────────────────────────────────────────────────────
# Walk helpers — LangChain wraps things in lots of nesting (Runnables,
# AgentExecutors, RunnableBindings). Both `_collect_attr` and
# `_walk_for_attr` try a small fixed set of common paths and stop on
# first hit, rather than doing infinite recursion.
# ─────────────────────────────────────────────────────────────────────


# A few common paths where LangChain wraps the real object.
# Ordered most-likely-first.
_NESTING_ATTRS = ("agent", "bound", "first", "last", "runnable", "llm_chain")


def _walk_for_attr(chain: Any, attr: str, max_depth: int = 4) -> Any:
    """Return chain.<attr> or chain.<nested>.<attr>, walking common LangChain wrappers."""
    if chain is None:
        return None
    value = getattr(chain, attr, None)
    if value is not None:
        return value
    if max_depth <= 0:
        return None
    for nest in _NESTING_ATTRS:
        nested = getattr(chain, nest, None)
        if nested is None:
            continue
        result = _walk_for_attr(nested, attr, max_depth - 1)
        if result is not None:
            return result
    # Also walk chain.steps if present (RunnableSequence)
    steps = getattr(chain, "steps", None)
    if isinstance(steps, (list, tuple)):
        for step in steps:
            result = _walk_for_attr(step, attr, max_depth - 1)
            if result is not None:
                return result
    return None


def _collect_attr(chain: Any, attr: str, max_depth: int = 4) -> List[Any]:
    """Like _walk_for_attr but expects a list value and returns it flat."""
    value = _walk_for_attr(chain, attr, max_depth)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]
