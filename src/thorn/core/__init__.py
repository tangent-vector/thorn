"""thorn.core -- Agent primitives and core infrastructure.

This subpackage contains the foundational types, agent loop, provider
abstraction, history management, tool system, and all other primitives
that make up the core of Thorn.

The public API surface is re-exported by the top-level ``thorn`` package
for backward compatibility, so ``from thorn import Agent`` and
``from thorn.core import Agent`` both work.
"""

from __future__ import annotations

from thorn.core._context import (
    AskUserHandler,
    ConsoleEventSink,
    EventSink,
    ExecutionContext,
    NullEventSink,
    Scope,
    UsageTracker,
    Verbosity,
    get_context,
    reset_context,
    set_context,
)
from thorn.core._agent import Agent
from thorn.core._context_injection import DirectorySeed, FileSeed, SearchSeed, SeedContent
from thorn.core._discovery import discover_tools
from thorn.core._history import CollapseState, CompactionResult, HistoryTree
from thorn.core._module import ModulePath
from thorn.core._validation import ValidationRule
from thorn.core._validation_tracker import ValidationStatus, ValidationTracker
from thorn.core._file_access import FileAccessLevel, FileAccessPolicy, FileAccessRule
from thorn.core._func import prompt, skill, tool, wrap_function
from thorn.core._retry import bound_retries
from thorn.core._loop import _WrappedTool, run_agent_loop
from thorn.core._provider import LLMProvider, MockProvider, UsageChunk, load_provider_from_env
from thorn.core._tools import (
    ALL_BUILTIN_TOOLS,
    FILE_READING,
    FILE_WRITING,
    FileEdit,
    ask_user,
    create_file,
    delete_file,
    edit_file,
    find_files,
    list_directory,
    move_file,
    read_file,
    search_files,
    write_file,
)
from thorn.core.errors import (
    AgentFailureError,
    LoopLimitError,
    ProviderError,
    RateLimitError,
    SkillError,
    ThornError,
)
from thorn.core._trace import CompositeEventSink, JsonLinesSink

try:
    from thorn.core._mcp import MCPServerConfig, MCPToolSource, load_mcp_configs, serve_tools
except ImportError:
    pass

__all__ = [
    # Agent
    "Agent",
    # Context
    "AskUserHandler",
    "ConsoleEventSink",
    "EventSink",
    "ExecutionContext",
    "NullEventSink",
    "Scope",
    "UsageTracker",
    "Verbosity",
    "get_context",
    "reset_context",
    "set_context",
    # Context injection
    "DirectorySeed",
    "FileSeed",
    "SearchSeed",
    "SeedContent",
    # Discovery
    "discover_tools",
    # History
    "CollapseState",
    "CompactionResult",
    "HistoryTree",
    # Module
    "ModulePath",
    # Validation
    "ValidationRule",
    "ValidationStatus",
    "ValidationTracker",
    # File access
    "FileAccessLevel",
    "FileAccessPolicy",
    "FileAccessRule",
    # Function decorators
    "prompt",
    "skill",
    "tool",
    "wrap_function",
    # Retry
    "bound_retries",
    # Loop
    "_WrappedTool",
    "run_agent_loop",
    # Provider
    "LLMProvider",
    "MockProvider",
    "UsageChunk",
    "load_provider_from_env",
    # Built-in tools
    "ALL_BUILTIN_TOOLS",
    "FILE_READING",
    "FILE_WRITING",
    "FileEdit",
    "ask_user",
    "create_file",
    "delete_file",
    "edit_file",
    "find_files",
    "list_directory",
    "move_file",
    "read_file",
    "search_files",
    "write_file",
    # Errors
    "AgentFailureError",
    "LoopLimitError",
    "ProviderError",
    "RateLimitError",
    "SkillError",
    "ThornError",
    # Tracing
    "CompositeEventSink",
    "JsonLinesSink",
]
