"""Type-annotation introspection and JSON-schema generation.

Bridges Python function signatures to OpenAI-style tool schemas and
provides Pydantic-based validation for structured agent output.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable, get_type_hints

from pydantic import TypeAdapter


def type_to_json_schema(t: Any) -> dict[str, Any]:
    """Convert an arbitrary Python type annotation to a JSON Schema dict.

    Uses Pydantic's ``TypeAdapter`` so that complex types (``list[str]``,
    ``dict[str, int]``, Pydantic models, ``Literal``, unions, etc.) are
    all handled automatically.
    """
    if t is inspect.Parameter.empty or t is Any:
        return {}
    adapter = TypeAdapter(t)
    schema = adapter.json_schema()
    # Strip Pydantic metadata keys that the OpenAI API ignores.
    schema.pop("title", None)
    return schema


def validate_result(result_type: type, raw: Any) -> Any:
    """Validate and coerce *raw* (typically parsed JSON) into *result_type*.

    Returns the validated Python value.  Raises ``ValidationError`` on
    type mismatches.
    """
    adapter = TypeAdapter(result_type)
    return adapter.validate_python(raw)


def func_to_tool_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Build an OpenAI-style tool schema from a Python function.

    Introspects the function's name, docstring, and typed parameters to
    produce a dict of the form::

        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": { JSON Schema object }
            }
        }

    Parameters named ``self``, ``cls``, or starting with ``_`` are
    excluded.  Parameter descriptions are not populated here (we'd need
    structured docstring parsing for that) but the types are faithfully
    converted.
    """
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls") or name.startswith("_"):
            continue

        annotation = hints.get(name, inspect.Parameter.empty)
        prop_schema = type_to_json_schema(annotation)
        properties[name] = prop_schema

        if param.default is inspect.Parameter.empty:
            required.append(name)

    params_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        params_schema["required"] = required

    description = inspect.getdoc(fn) or ""

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": params_schema,
        },
    }


def make_return_result_schema(result_type: type) -> dict[str, Any]:
    """Build a tool schema for the synthetic ``return_result`` tool.

    The tool has a single ``value`` parameter whose schema is derived
    from *result_type*.
    """
    value_schema = type_to_json_schema(result_type)

    return {
        "type": "function",
        "function": {
            "name": "return_result",
            "description": (
                "Return your final result.  The 'value' parameter must "
                "conform to the specified schema."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value": value_schema,
                },
                "required": ["value"],
            },
        },
    }


RAISE_ERROR_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "raise_error",
        "description": (
            "Signal that you cannot fulfil the request.  Provide a "
            "clear description of the problem."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "A clear description of the problem.",
                },
            },
            "required": ["message"],
        },
    },
}


def serialize_for_tool_result(value: Any) -> str:
    """Serialize a Python value to a string suitable for a tool result."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except TypeError:
        return str(value)
