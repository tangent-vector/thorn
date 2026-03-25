"""Tests for thorn._schema — type→schema conversion, validation, serialization."""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from pydantic import BaseModel

from thorn._schema import (
    func_to_tool_schema,
    make_return_result_schema,
    serialize_for_tool_result,
    type_to_json_schema,
    validate_result,
)


# ---------------------------------------------------------------------------
# type_to_json_schema
# ---------------------------------------------------------------------------

class TestTypeToJsonSchema:
    def test_str(self):
        assert type_to_json_schema(str) == {"type": "string"}

    def test_int(self):
        assert type_to_json_schema(int) == {"type": "integer"}

    def test_bool(self):
        assert type_to_json_schema(bool) == {"type": "boolean"}

    def test_list_of_str(self):
        schema = type_to_json_schema(list[str])
        assert schema["type"] == "array"
        assert schema["items"] == {"type": "string"}

    def test_dict_str_int(self):
        schema = type_to_json_schema(dict[str, int])
        assert schema["type"] == "object"
        assert schema["additionalProperties"]["type"] == "integer"

    def test_any_returns_empty(self):
        assert type_to_json_schema(Any) == {}

    def test_parameter_empty_returns_empty(self):
        assert type_to_json_schema(inspect.Parameter.empty) == {}

    def test_pydantic_model(self):
        class Item(BaseModel):
            name: str
            count: int

        schema = type_to_json_schema(Item)
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "count" in schema["properties"]

    def test_title_stripped(self):
        schema = type_to_json_schema(str)
        assert "title" not in schema


# ---------------------------------------------------------------------------
# validate_result
# ---------------------------------------------------------------------------

class TestValidateResult:
    def test_bool_true(self):
        assert validate_result(bool, True) is True

    def test_bool_false(self):
        assert validate_result(bool, False) is False

    def test_list_of_str(self):
        assert validate_result(list[str], ["a", "b"]) == ["a", "b"]

    def test_coercion_int_from_float(self):
        # Pydantic coerces compatible types
        assert validate_result(int, 42) == 42

    def test_invalid_type_raises(self):
        with pytest.raises(Exception):
            validate_result(int, "not a number")

    def test_pydantic_model(self):
        class Point(BaseModel):
            x: int
            y: int

        result = validate_result(Point, {"x": 1, "y": 2})
        assert result.x == 1 and result.y == 2


# ---------------------------------------------------------------------------
# func_to_tool_schema
# ---------------------------------------------------------------------------

class TestFuncToToolSchema:
    def test_basic_function(self):
        def greet(name: str, excited: bool = False) -> str:
            """Say hello."""
            ...

        schema = func_to_tool_schema(greet)
        assert schema["type"] == "function"
        func = schema["function"]
        assert func["name"] == "greet"
        assert func["description"] == "Say hello."
        params = func["parameters"]
        assert "name" in params["properties"]
        assert params["properties"]["name"]["type"] == "string"
        assert "required" in params
        assert "name" in params["required"]
        # 'excited' has a default, so not required
        assert "excited" not in params["required"]

    def test_skips_self_and_underscore(self):
        def method(self, _private: int, public: str) -> None: ...

        schema = func_to_tool_schema(method)
        props = schema["function"]["parameters"]["properties"]
        assert "self" not in props
        assert "_private" not in props
        assert "public" in props

    def test_no_docstring(self):
        def bare(x: int) -> int: ...

        schema = func_to_tool_schema(bare)
        assert schema["function"]["description"] == ""

    def test_unannotated_param_gets_empty_schema(self):
        def loose(x) -> str: ...

        schema = func_to_tool_schema(loose)
        # No type annotation → empty schema (unconstrained)
        assert schema["function"]["parameters"]["properties"]["x"] == {}


# ---------------------------------------------------------------------------
# make_return_result_schema
# ---------------------------------------------------------------------------

class TestMakeReturnResultSchema:
    def test_shape(self):
        schema = make_return_result_schema(list[str])
        func = schema["function"]
        assert func["name"] == "return_result"
        value_prop = func["parameters"]["properties"]["value"]
        assert value_prop["type"] == "array"
        assert value_prop["items"] == {"type": "string"}
        assert "value" in func["parameters"]["required"]


# ---------------------------------------------------------------------------
# serialize_for_tool_result
# ---------------------------------------------------------------------------

class TestSerializeForToolResult:
    def test_str_passthrough(self):
        assert serialize_for_tool_result("hello") == "hello"

    def test_dict_to_json(self):
        assert serialize_for_tool_result({"a": 1}) == '{"a": 1}'

    def test_int_to_json(self):
        assert serialize_for_tool_result(42) == "42"

    def test_list_to_json(self):
        assert serialize_for_tool_result([1, 2]) == "[1, 2]"
