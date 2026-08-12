"""Decorators for manifest-aware instrumentation.

The ``@decimalai.tool`` decorator registers a function's signature as a
tool in the manifest without changing the function's behavior.

Example::

    import decimalai

    @decimalai.tool
    def search(query: str, limit: int = 10) -> list[dict]:
        \"\"\"Search the knowledge base for relevant documents.\"\"\"
        ...

    # The tool's JSON schema is auto-generated from type hints:
    # {
    #   "name": "search",
    #   "description": "Search the knowledge base for relevant documents.",
    #   "schema": {
    #     "type": "object",
    #     "properties": {
    #       "query": {"type": "string"},
    #       "limit": {"type": "integer", "default": 10}
    #     },
    #     "required": ["query"]
    #   }
    # }
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, get_type_hints

logger = logging.getLogger("decimalai")

# Module-level registry of decorated tools — consumed by the generic
# tracer's _maybe_register_manifest() to merge with auto-detected tools.
_registered_tools: Dict[str, Dict[str, Any]] = {}


# ── Python type → JSON Schema type mapping ──────────────────

_TYPE_MAP: Dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    bytes: "string",
}


def _python_type_to_json_schema(py_type: Any) -> Dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema type descriptor.

    Handles:
    - Basic types (str, int, float, bool, list, dict)
    - Optional[T] (nullable)
    - list[T] (array with items)
    - Pydantic BaseModel subclasses (full JSON schema via model_json_schema)
    - Falls back to {} for unrecognized types
    """
    if py_type is None or py_type is inspect.Parameter.empty:
        return {}

    # Handle Optional[T] (Union[T, None])
    origin = getattr(py_type, "__origin__", None)
    args = getattr(py_type, "__args__", None)

    if origin is type(None):
        return {"type": "null"}

    # Optional[T] = Union[T, None]
    if origin is getattr(__builtins__, "Union", None) or str(origin) == "typing.Union":
        if args and type(None) in args:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                schema = _python_type_to_json_schema(non_none[0])
                schema["nullable"] = True
                return schema

    # list[T] → array with items
    if origin is list:
        schema: Dict[str, Any] = {"type": "array"}
        if args and len(args) == 1:
            schema["items"] = _python_type_to_json_schema(args[0])
        return schema

    # dict[K, V] → object
    if origin is dict:
        return {"type": "object"}

    # Check if it's a Pydantic BaseModel
    try:
        if hasattr(py_type, "model_json_schema"):
            return py_type.model_json_schema()
    except Exception:
        pass

    # Basic type lookup
    if isinstance(py_type, type) and py_type in _TYPE_MAP:
        return {"type": _TYPE_MAP[py_type]}

    # Fallback
    return {}


def _function_to_tool_schema(fn: Callable) -> Dict[str, Any]:
    """Generate a tool schema dict from a function's signature and docstring.

    Returns:
        Dict with keys: name, description, schema (JSON Schema object).
    """
    name = fn.__name__
    doc = inspect.getdoc(fn) or ""
    description = doc.split("\n")[0].strip() if doc else ""

    sig = inspect.signature(fn)

    # Try to get type hints (may fail in some edge cases)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        param_schema: Dict[str, Any] = {}

        # Get type from hints
        if param_name in hints:
            param_schema = _python_type_to_json_schema(hints[param_name])

        # Add default value
        if param.default is not inspect.Parameter.empty:
            param_schema["default"] = param.default
        else:
            required.append(param_name)

        properties[param_name] = param_schema

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return {
        "name": name,
        "description": description,
        "schema": schema,
    }


def tool(fn: Optional[Callable] = None, *, name: Optional[str] = None) -> Any:
    """Register a function as a DecimalAI tool for manifest tracking.

    This decorator is **transparent** — it does NOT change the function's
    behavior. It only registers the function's schema in a module-level
    registry that is read by the generic tracer when building manifests.

    Can be used with or without arguments::

        @decimalai.tool
        def search(query: str) -> list:
            ...

        @decimalai.tool(name="custom_search")
        def search(query: str) -> list:
            ...

    Args:
        fn: The function to register (when used without arguments).
        name: Override the tool name (defaults to function name).

    Returns:
        The original function, unmodified.
    """
    def decorator(func: Callable) -> Callable:
        tool_schema = _function_to_tool_schema(func)
        tool_name = name or tool_schema["name"]
        tool_schema["name"] = tool_name
        _registered_tools[tool_name] = tool_schema

        logger.debug(
            "Registered tool '%s' with %d properties",
            tool_name,
            len(tool_schema.get("schema", {}).get("properties", {})),
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        # Attach schema to function for introspection
        wrapper._decimal_tool_schema = tool_schema  # type: ignore[attr-defined]
        return wrapper

    # Handle both @decimalai.tool and @decimalai.tool(name="...")
    if fn is not None:
        return decorator(fn)
    return decorator


def get_registered_tools() -> List[Dict[str, Any]]:
    """Return all tools registered via ``@decimalai.tool``.

    Used by the generic tracer to merge with auto-detected tools
    when building the manifest snapshot.
    """
    return list(_registered_tools.values())
