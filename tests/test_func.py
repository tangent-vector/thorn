"""Tests for thorn.core._func — prompt, @skill, @tool, wrap_function, _prepare_tools."""

from __future__ import annotations

import pytest
from pydantic.dataclasses import dataclass as pydantic_dataclass

from thorn.core._executor import ToolVenue
from thorn.core._func import _prepare_tools, prompt, skill, tool, wrap_function
from thorn.core._history import FileReadCallNode
from thorn.core._loop import _WrappedTool
from thorn.core._provider import FinishChunk, MockProvider, ToolCallChunk


@pydantic_dataclass
class _TestEdit:
    old: str
    new: str


# ---------------------------------------------------------------------------
# wrap_function
# ---------------------------------------------------------------------------

class TestWrapFunction:
    def test_sync_function(self):
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        tool = wrap_function(multiply, venue=ToolVenue.SANDBOX)
        assert tool.schema["function"]["name"] == "multiply"
        assert "a" in tool.schema["function"]["parameters"]["properties"]

    async def test_sync_execute(self):
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        tool = wrap_function(multiply, venue=ToolVenue.SANDBOX)
        result = await tool.execute(a=3, b=7)
        assert result == "21"

    async def test_async_execute(self):
        async def fetch(url: str) -> str:
            """Fetch a URL."""
            return f"data from {url}"

        tool = wrap_function(fetch, venue=ToolVenue.SANDBOX)
        result = await tool.execute(url="http://example.com")
        assert result == "data from http://example.com"

    async def test_dict_return_serialized_as_json(self):
        def info() -> dict:
            """Get info."""
            return {"status": "ok", "count": 3}

        tool = wrap_function(info, venue=ToolVenue.SANDBOX)
        result = await tool.execute()
        import json
        assert json.loads(result) == {"status": "ok", "count": 3}

    async def test_nested_dataclass_coerced_from_dict(self):
        """Raw dicts (from JSON) are coerced to the annotated dataclass type."""
        def apply_edits(edits: list[_TestEdit]) -> str:
            """Apply edits."""
            return ", ".join(f"{e.old}->{e.new}" for e in edits)

        tool = wrap_function(apply_edits, venue=ToolVenue.SANDBOX)
        result = await tool.execute(
            edits=[{"old": "foo", "new": "bar"}, {"old": "a", "new": "b"}],
        )
        assert result == "foo->bar, a->b"


# ---------------------------------------------------------------------------
# _prepare_tools
# ---------------------------------------------------------------------------

class TestPrepareTools:
    def test_empty_list(self):
        assert _prepare_tools([]) == []

    def test_none(self):
        assert _prepare_tools(None) == []

    def test_known_builtin_callable_auto_wrapped(self):
        from thorn.core._tools import read_file

        result = _prepare_tools([read_file])
        assert len(result) == 1
        assert isinstance(result[0], _WrappedTool)
        assert result[0].schema["function"]["name"] == "read_file"

    def test_unknown_callable_rejected(self):
        def greet(name: str) -> str:
            """Greet someone."""
            return f"hi {name}"

        with pytest.raises(TypeError, match="not a registered Thorn tool"):
            _prepare_tools([greet])

    def test_unknown_callable_accepted_when_pre_wrapped(self):
        def greet(name: str) -> str:
            """Greet someone."""
            return f"hi {name}"

        wrapped = wrap_function(greet, venue=ToolVenue.SANDBOX)
        result = _prepare_tools([wrapped])
        assert result == [wrapped]

    def test_wrapped_tool_passthrough(self):
        def noop() -> str:
            """Nothing."""
            return ""

        already = wrap_function(noop, venue=ToolVenue.SANDBOX)
        result = _prepare_tools([already])
        assert result[0] is already

    def test_mixed_input(self):
        def fn_a() -> str:
            """A."""
            return "a"

        fn_b_wrapped = wrap_function(fn_a, venue=ToolVenue.SANDBOX)
        result = _prepare_tools([wrap_function(fn_a, venue=ToolVenue.SANDBOX), fn_b_wrapped])
        assert len(result) == 2

    def test_non_callable_raises(self):
        with pytest.raises(TypeError, match="Expected a callable"):
            _prepare_tools([42])

    def test_nested_list_flattened(self):
        def fn_a() -> str:
            """A."""
            return "a"

        def fn_b() -> str:
            """B."""
            return "b"

        def fn_c() -> str:
            """C."""
            return "c"

        result = _prepare_tools(
            [[wrap_function(fn_a, venue=ToolVenue.SANDBOX), wrap_function(fn_b, venue=ToolVenue.SANDBOX)], wrap_function(fn_c, venue=ToolVenue.SANDBOX)],
        )
        assert len(result) == 3
        names = [t.schema["function"]["name"] for t in result]
        assert names == ["fn_a", "fn_b", "fn_c"]

    def test_deeply_nested_flattened(self):
        def fn_a() -> str:
            """A."""
            return "a"

        result = _prepare_tools([[[wrap_function(fn_a, venue=ToolVenue.SANDBOX)]]])
        assert len(result) == 1
        assert result[0].schema["function"]["name"] == "fn_a"

    def test_empty_sublists_ignored(self):
        def fn_a() -> str:
            """A."""
            return "a"

        result = _prepare_tools([[], wrap_function(fn_a, venue=ToolVenue.SANDBOX), []])
        assert len(result) == 1

    def test_tuple_flattened(self):
        def fn_a() -> str:
            """A."""
            return "a"

        def fn_b() -> str:
            """B."""
            return "b"

        result = _prepare_tools([(wrap_function(fn_a, venue=ToolVenue.SANDBOX), wrap_function(fn_b, venue=ToolVenue.SANDBOX))])
        assert len(result) == 2

    def test_nested_non_callable_raises(self):
        with pytest.raises(TypeError, match="Expected a callable"):
            _prepare_tools([[42]])

    def test_mixed_nesting_with_wrapped(self):
        def fn_a() -> str:
            """A."""
            return "a"

        def fn_b() -> str:
            """B."""
            return "b"

        wrapped_b = wrap_function(fn_b, venue=ToolVenue.SANDBOX)
        result = _prepare_tools([[wrap_function(fn_a, venue=ToolVenue.SANDBOX)], wrapped_b])
        assert len(result) == 2
        assert result[1] is wrapped_b


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------

class TestToolDecorator:
    def test_decorator_sets_marker(self):
        @tool(venue=ToolVenue.SANDBOX)
        def my_tool() -> str:
            """Do something."""
            return "done"

        assert my_tool._thorn_tool is True

    def test_decorator_stamps_venue(self):
        @tool(venue=ToolVenue.IN_PROCESS)
        def my_tool() -> str:
            """Do something."""
            return "done"

        assert my_tool._thorn_venue is ToolVenue.IN_PROCESS

    def test_decorator_no_call_node_class(self):
        @tool(venue=ToolVenue.SANDBOX)
        def my_tool() -> str:
            """Do something."""
            return "done"

        assert not hasattr(my_tool, "_thorn_call_node_class")

    def test_parameterized_sets_marker_and_class(self):
        @tool(venue=ToolVenue.SANDBOX, call_node_class=FileReadCallNode)
        def my_reader(path: str) -> str:
            """Read something."""
            return "content"

        assert my_reader._thorn_tool is True
        assert my_reader._thorn_call_node_class is FileReadCallNode

    def test_bare_decorator_rejected(self):
        # @tool (no parens) is no longer accepted: every tool author
        # must pick a venue, with no silent default.
        with pytest.raises(TypeError):
            @tool
            def my_tool() -> str:
                """No venue."""
                return "x"

    def test_decorator_without_venue_rejected(self):
        # @tool() without a venue keyword is also rejected.
        with pytest.raises(TypeError):
            @tool()
            def my_tool() -> str:
                """No venue."""
                return "x"

    def test_preserves_function_identity(self):
        @tool(venue=ToolVenue.SANDBOX, call_node_class=FileReadCallNode)
        def my_fn() -> str:
            """Test."""
            return "x"

        assert my_fn() == "x"
        assert my_fn.__name__ == "my_fn"


# ---------------------------------------------------------------------------
# wrap_function — call_node_class threading
# ---------------------------------------------------------------------------

class TestWrapFunctionCallNodeClass:
    def test_reads_call_node_class_attribute(self):
        def my_reader(path: str) -> str:
            """Read a file."""
            return "content"
        my_reader._thorn_call_node_class = FileReadCallNode  # type: ignore[attr-defined]

        wrapped = wrap_function(my_reader, venue=ToolVenue.SANDBOX)
        assert wrapped.call_node_class is FileReadCallNode

    def test_no_attribute_defaults_to_none(self):
        def plain(x: int) -> int:
            """Double."""
            return x * 2

        wrapped = wrap_function(plain, venue=ToolVenue.SANDBOX)
        assert wrapped.call_node_class is None

    def test_prepare_tools_preserves_call_node_class(self):
        def my_reader(path: str) -> str:
            """Read a file."""
            return "content"
        my_reader._thorn_call_node_class = FileReadCallNode  # type: ignore[attr-defined]

        result = _prepare_tools([wrap_function(my_reader, venue=ToolVenue.SANDBOX)])
        assert len(result) == 1
        assert result[0].call_node_class is FileReadCallNode

    def test_prepare_tools_passthrough_preserves_class(self):
        def my_reader(path: str) -> str:
            """Read a file."""
            return "content"
        my_reader._thorn_call_node_class = FileReadCallNode  # type: ignore[attr-defined]

        already_wrapped = wrap_function(my_reader, venue=ToolVenue.SANDBOX)
        result = _prepare_tools([already_wrapped])
        assert result[0] is already_wrapped
        assert result[0].call_node_class is FileReadCallNode


# ---------------------------------------------------------------------------
# prompt (text mode)
# ---------------------------------------------------------------------------

class TestPromptTextMode:
    async def test_returns_text(self, ctx):
        result = await prompt("say hello")
        assert result == "[mock] say hello"

    async def test_with_system_prompt(self, ctx):
        # System prompt doesn't affect MockProvider output, but we verify
        # it doesn't crash and the call completes.
        result = await prompt("say hello", system="You are friendly.")
        assert isinstance(result, str)

    async def test_with_tools(self, ctx):
        def helper() -> str:
            """Help."""
            return "helped"

        result = await prompt("use helper", tools=[wrap_function(helper, venue=ToolVenue.SANDBOX)])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# prompt (structured mode)
# ---------------------------------------------------------------------------

class TestPromptStructuredMode:
    async def test_bool(self):
        provider = MockProvider(canned_responses=[[
            ToolCallChunk(call_id="c1", name="return_result", arguments='{"value": false}'),
            FinishChunk(reason="stop"),
        ]])
        from thorn.core._context import ExecutionContext, reset_context, set_context
        ctx = ExecutionContext(provider=provider)
        token = set_context(ctx)
        try:
            result = await prompt[bool]("is it running?")
            assert result is False
        finally:
            reset_context(token)

    async def test_list_of_str(self):
        provider = MockProvider(canned_responses=[[
            ToolCallChunk(
                call_id="c1", name="return_result",
                arguments='{"value": ["x", "y", "z"]}',
            ),
            FinishChunk(reason="stop"),
        ]])
        from thorn.core._context import ExecutionContext, reset_context, set_context
        ctx = ExecutionContext(provider=provider)
        token = set_context(ctx)
        try:
            result = await prompt[list[str]]("list them")
            assert result == ["x", "y", "z"]
        finally:
            reset_context(token)


# ---------------------------------------------------------------------------
# @skill
# ---------------------------------------------------------------------------

class TestSkillDecorator:
    async def test_bare_skill_formats_docstring(self):
        """The docstring template is formatted with the arguments."""
        @skill
        async def greet(name: str) -> str:
            """Say hello to {name}."""

        provider = MockProvider()
        from thorn.core._context import ExecutionContext, reset_context, set_context
        ctx = ExecutionContext(provider=provider)
        token = set_context(ctx)
        try:
            result = await greet("Alice")
            # MockProvider echoes the user prompt, which is the formatted docstring
            assert "Say hello to Alice" in result
        finally:
            reset_context(token)

    async def test_skill_with_return_type(self):
        """Skill with a non-str return type uses structured mode."""
        @skill
        async def count_items(category: str) -> int:
            """How many items are in the {category} category?"""

        provider = MockProvider(canned_responses=[[
            ToolCallChunk(
                call_id="c1", name="return_result", arguments='{"value": 42}',
            ),
            FinishChunk(reason="stop"),
        ]])
        from thorn.core._context import ExecutionContext, reset_context, set_context
        ctx = ExecutionContext(provider=provider)
        token = set_context(ctx)
        try:
            result = await count_items("books")
            assert result == 42
        finally:
            reset_context(token)

    async def test_skill_with_tools_config(self):
        """@skill(tools=[...]) passes tools through without error."""
        def helper(x: int) -> int:
            """Double."""
            return x * 2

        @skill(tools=[wrap_function(helper, venue=ToolVenue.SANDBOX)])
        async def compute(n: int) -> str:
            """Compute something with {n}."""

        provider = MockProvider()
        from thorn.core._context import ExecutionContext, reset_context, set_context
        ctx = ExecutionContext(provider=provider)
        token = set_context(ctx)
        try:
            result = await compute(5)
            assert isinstance(result, str)
        finally:
            reset_context(token)

    def test_skill_preserves_function_name(self):
        @skill
        async def my_func(x: str) -> str:
            """Do {x}."""

        assert my_func.__name__ == "my_func"

    def test_skill_sets_metadata(self):
        @skill
        async def my_func(x: str) -> str:
            """Do {x}."""

        assert my_func._thorn_skill is True
        assert my_func._thorn_return_type is str


# ---------------------------------------------------------------------------
# prompt with role=
# ---------------------------------------------------------------------------

class TestPromptWithRole:
    async def test_text_mode_with_role_class(self, ctx):
        from thorn.core._agent import Agent

        class Helper(Agent):
            system_prompts = ["You are helpful."]

        result = await prompt("say hello", role=Helper)
        assert isinstance(result, str)

    async def test_text_mode_with_role_instance(self, ctx):
        from thorn.core._agent import Agent

        class Helper(Agent):
            system_prompts = ["Working on {module}."]

        agent = Helper(module="parser")
        result = await prompt("say hello", role=agent)
        assert isinstance(result, str)

    async def test_structured_mode_with_role(self):
        from thorn.core._agent import Agent
        from thorn.core._context import ExecutionContext, reset_context, set_context

        class Helper(Agent):
            system_prompts = ["You count things."]

        provider = MockProvider(canned_responses=[[
            ToolCallChunk(
                call_id="c1", name="return_result",
                arguments='{"value": ["a", "b"]}',
            ),
            FinishChunk(reason="stop"),
        ]])
        ctx = ExecutionContext(provider=provider)
        token = set_context(ctx)
        try:
            result = await prompt[list[str]]("list items", role=Helper)
            assert result == ["a", "b"]
        finally:
            reset_context(token)

    async def test_role_with_extra_tools(self, ctx):
        from thorn.core._agent import Agent

        async def extra() -> str:
            """Extra tool."""
            return "extra"

        class Helper(Agent):
            system_prompts = ["You are helpful."]

        result = await prompt("do it", role=Helper, tools=[wrap_function(extra, venue=ToolVenue.SANDBOX)])
        assert isinstance(result, str)

    async def test_role_with_extra_system(self, ctx):
        from thorn.core._agent import Agent

        class Helper(Agent):
            system_prompts = ["Base prompt."]

        result = await prompt("do it", role=Helper, system="Extra instruction.")
        assert isinstance(result, str)
