"""Tests for the @decimalai.tool decorator and schema extraction."""

from typing import Dict, List, Optional

import pytest


class TestToolDecorator:
    """Test @decimalai.tool decorator."""

    def setup_method(self):
        """Clear the registry before each test."""
        from decimalai.decorators import _registered_tools
        _registered_tools.clear()

    def test_basic_registration(self):
        """Decorated function is registered in the global registry."""
        from decimalai.decorators import tool, get_registered_tools

        @tool
        def search(query: str) -> list:
            """Search for documents."""
            return []

        tools = get_registered_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "search"
        assert tools[0]["description"] == "Search for documents."

    def test_schema_from_type_hints(self):
        """Type hints are converted to JSON Schema."""
        from decimalai.decorators import tool, get_registered_tools

        @tool
        def calc(x: int, y: float, label: str) -> dict:
            """Calculate something."""
            return {}

        schema = get_registered_tools()[0]["schema"]
        assert schema["type"] == "object"
        assert schema["properties"]["x"] == {"type": "integer"}
        assert schema["properties"]["y"] == {"type": "number"}
        assert schema["properties"]["label"] == {"type": "string"}
        assert schema["required"] == ["x", "y", "label"]

    def test_default_values(self):
        """Default values are captured and params become optional."""
        from decimalai.decorators import tool, get_registered_tools

        @tool
        def search(query: str, limit: int = 10, fuzzy: bool = False) -> list:
            """Search."""
            return []

        schema = get_registered_tools()[0]["schema"]
        assert schema["required"] == ["query"]
        assert schema["properties"]["limit"] == {"type": "integer", "default": 10}
        assert schema["properties"]["fuzzy"] == {"type": "boolean", "default": False}

    def test_custom_name(self):
        """Custom tool name overrides function name."""
        from decimalai.decorators import tool, get_registered_tools

        @tool(name="custom_search")
        def search(query: str) -> list:
            return []

        tools = get_registered_tools()
        assert tools[0]["name"] == "custom_search"

    def test_function_behavior_unchanged(self):
        """Decorator does not affect function behavior."""
        from decimalai.decorators import tool

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert add(2, 3) == 5
        assert add(a=10, b=20) == 30

    def test_list_type_hint(self):
        """list[T] is converted to array with items."""
        from decimalai.decorators import tool, get_registered_tools

        @tool
        def process(items: List[str]) -> List[dict]:
            return []

        schema = get_registered_tools()[0]["schema"]
        assert schema["properties"]["items"]["type"] == "array"
        assert schema["properties"]["items"]["items"] == {"type": "string"}

    def test_dict_type_hint(self):
        """dict is converted to object type."""
        from decimalai.decorators import tool, get_registered_tools

        @tool
        def update(data: Dict[str, int]) -> dict:
            return {}

        schema = get_registered_tools()[0]["schema"]
        assert schema["properties"]["data"]["type"] == "object"

    def test_no_type_hints(self):
        """Functions without type hints still register (with empty schema)."""
        from decimalai.decorators import tool, get_registered_tools

        @tool
        def mystery(x, y):
            return x + y

        tools = get_registered_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "mystery"
        # Properties exist but have no type info
        assert "x" in tools[0]["schema"]["properties"]

    def test_multiple_tools(self):
        """Multiple tools can be registered."""
        from decimalai.decorators import tool, get_registered_tools

        @tool
        def search(q: str) -> list:
            return []

        @tool
        def calculate(expr: str) -> float:
            return 0.0

        @tool
        def fetch_url(url: str, timeout: int = 30) -> str:
            return ""

        tools = get_registered_tools()
        assert len(tools) == 3
        names = {t["name"] for t in tools}
        assert names == {"search", "calculate", "fetch_url"}

    def test_schema_attached_to_function(self):
        """Schema is attached to the function as _decimal_tool_schema."""
        from decimalai.decorators import tool

        @tool
        def search(query: str) -> list:
            """Search."""
            return []

        assert hasattr(search, "_decimal_tool_schema")
        assert search._decimal_tool_schema["name"] == "search"

    def test_docstring_first_line_only(self):
        """Only the first line of the docstring is used as description."""
        from decimalai.decorators import tool, get_registered_tools

        @tool
        def search(query: str) -> list:
            """Search for relevant documents.

            This function searches the knowledge base using semantic
            similarity and returns the top results.

            Args:
                query: The search query string.
            """
            return []

        assert get_registered_tools()[0]["description"] == "Search for relevant documents."


class TestFunctionToToolSchema:
    """Test the _function_to_tool_schema helper directly."""

    def test_self_and_cls_excluded(self):
        """self and cls parameters are excluded from schema."""
        from decimalai.decorators import _function_to_tool_schema

        class MyClass:
            def method(self, query: str) -> list:
                """A method."""
                return []

            @classmethod
            def class_method(cls, query: str) -> list:
                """A class method."""
                return []

        schema = _function_to_tool_schema(MyClass.method)
        assert "self" not in schema["schema"]["properties"]
        assert "query" in schema["schema"]["properties"]
